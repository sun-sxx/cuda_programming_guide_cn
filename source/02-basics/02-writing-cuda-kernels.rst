.. _writing-cuda-kernels:

2.2. 编写 CUDA SIMT Kernels
============================

CUDA C++ kernels 在很大程度上可以像传统 CPU 代码那样编写。然而，GPU 有一些独特的特性可以用来提高性能。
此外，了解 GPU 上的线程如何调度、如何访问内存以及如何执行，可以帮助开发者编写能够最大化利用计算资源的 kernels。

.. _writing-cuda-kernels-basics-of-simt:

2.2.1. SIMT 基础
-----------------

从开发者的角度来看，CUDA thread 是并行的基本单位。
:ref:`warps-and-simt` 描述了 GPU 执行的基本 SIMT 模型，:ref:`simt-execution-model` 提供了 SIMT 模型的更多细节。
SIMT 模型允许每个线程维护自己的状态和控制流。从功能角度来看，每个线程可以执行独立的代码路径。
然而，通过优化 kernel 代码，减少同一 warp 中的线程执行不同代码路径的情况，可以显著的提升性能。

.. _writing-cuda-kernels-thread-hierarchy-review:

2.2.2. 线程层次结构
--------------------

Threads 被组织成 thread blocks，然后被组织成 grid。Grids 可以是 1、2 或 3 维的，grid 的大小可以在 kernel 内使用 ``gridDim`` 内置变量查询。
Thread blocks 也可以是 1、2 或 3 维的。Thread block 的大小可以在 kernel 内使用 ``blockDim`` 内置变量查询。
Thread block 的索引可以使用 ``blockIdx`` 内置变量查询。在 thread block 内，thread 的索引使用 ``threadIdx`` 内置变量获取。
这些内置变量用于为每个线程计算唯一的全局线程索引，从而使每个线程能够从全局内存加载/存储特定数据，并根据需要执行特定的代码路径。

- ``gridDim.[x|y|z]`` ：分别表示 grid 在 ``x``、``y`` 和 ``z`` 维度上的大小。这些值在 kernel 启动时设置。
- ``blockDim.[x|y|z]`` ：分别表示 block 在 ``x``、``y`` 和 ``z`` 维度上的大小。这些值在 kernel 启动时设置。
- ``blockIdx.[x|y|z]`` ：分别表示 block 在 ``x``、``y`` 和 ``z`` 维度上的索引。这些值根据正在执行的 block 而变化。
- ``threadIdx.[x|y|z]`` ：分别表示 thread 在 ``x``、``y`` 和 ``z`` 维度上的索引。这些值根据正在执行的 thread 而变化。

使用多维 thread blocks 和 grids 只是为了方便，并不影响性能。
Block 的 threads 以可预测的方式线性化：第一个索引 ``x`` 移动最快，其次是 ``y`` ，然后是 ``z`` 。
这意味着在线程索引的线性化中， ``threadIdx.x`` 的连续值表示连续的 threads， ``threadIdx.y`` 的步长为 ``blockDim.x`` ， ``threadIdx.z`` 的步长为 ``blockDim.x * blockDim.y`` 。
这影响 threads 如何分配给 warps，详见 :ref:`hardware-multithreading` 。

:numref:`fig-writing-cuda-kernels-thread-hierarchy-review-grid-of-thread-blocks` 展示了一个带有 1D thread blocks 的 2D grid 的简单示例。

.. _fig-writing-cuda-kernels-thread-hierarchy-review-grid-of-thread-blocks:
.. figure:: /_static/images/grid-of-thread-blocks.png
   :alt: Grid of Thread Blocks
   :width: 454px

   Grid of Thread Blocks

.. _writing-cuda-kernels-gpu-device-memory-spaces:

2.2.3. GPU 设备内存空间
------------------------

CUDA 设备有几个内存空间，可以被 kernels 中的 CUDA threads 访问。:numref:`tbl:writing-cuda-kernels-memory-types-scopes-lifetimes` 总结了常见内存类型、其线程范围和生命期。
以下章节详细解释每种内存类型。

.. _tbl:writing-cuda-kernels-memory-types-scopes-lifetimes:

.. table:: 内存类型、范围和生命期

   ============ ============ ============ ============
   内存类型     范围         生命期       位置
   ============ ============ ============ ============
   Global       Grid         Application  Device
   Constant     Grid         Application  Device
   Shared       Block        Kernel       SM
   Local        Thread       Kernel       Device
   Register     Thread       Kernel       SM
   ============ ============ ============ ============

.. _writing-cuda-kernels-global-memory:

2.2.3.1. 全局内存
^^^^^^^^^^^^^^^^^

Global memory（也称为 device memory）是存储可被 kernel 中所有 threads 访问的数据的主要内存空间。
它类似于 CPU 系统中的 RAM。运行在 GPU 上的 kernels 可以直接访问全局内存，就像运行在 CPU 上的代码可以访问系统内存一样。

全局内存是持久的。也就是说，在全局内存中进行的分配和存储在其中的数据会持续存在，直到分配被释放或应用程序终止。 ``cudaDeviceReset`` 也会释放所有分配。

全局内存通过 CUDA API 调用（如 ``cudaMalloc`` 和 ``cudaMallocManaged`` ）进行分配。
可以使用 CUDA runtime API 调用（如 ``cudaMemcpy`` ）将数据从 CPU 内存复制到全局内存。使用 CUDA APIs 进行的全局内存分配使用 ``cudaFree`` 释放。

在 kernel 启动之前，全局内存通过 CUDA API 调用进行分配和初始化。
在 kernel 执行期间，CUDA threads 可以从全局内存读取数据，CUDA threads 执行的操作结果可以写回全局内存。
一旦 kernel 完成执行，它写入全局内存的结果可以被复制回 host 或被 GPU 上的其他 kernels 使用。

由于全局内存可被 grid 中的所有 threads 访问，必须注意避免 threads 之间的数据竞争。
由于从 host 启动的 CUDA kernels 的返回类型是 ``void`` ，kernel 计算的数值结果返回给 host 的唯一方法是将其写入全局内存。

下面的 ``vecAdd`` kernel 是一个使用全局内存的简单示例，其中三个数组 ``A``、 ``B`` 和 ``C`` 位于全局内存中，正在被这个向量加法 kernel 访问。

.. code-block:: cuda

   __global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
   {
       int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
       if(workIndex < vectorLength)
       {
           C[workIndex] = A[workIndex] + B[workIndex];
       }
   }

.. _writing-cuda-kernels-shared-memory:

2.2.3.2. 共享内存
^^^^^^^^^^^^^^^^^

Shared memory 是一个可被 thread block 中所有 threads 访问的内存空间。
它物理上位于每个 SM 上，并与 L1 cache 使用相同的物理资源，即统一数据缓存。
共享内存中的数据在整个 kernel 执行期间持续存在。
共享内存可以被视为 kernel 执行期间使用的用户管理的暂存区。
虽然与全局内存相比大小较小，但由于共享内存位于每个 SM 上，其带宽更高，延迟比访问全局内存更低。

由于共享内存可被 thread block 中所有 threads 访问，必须注意避免同一 thread block 中 threads 之间的数据竞争。
同一 thread block 中 threads 之间的同步可以使用 ``__syncthreads()`` 函数实现。
此函数阻塞 thread block 中的所有 threads，直到所有 threads 都到达对 ``__syncthreads()`` 的调用。

.. code-block:: cuda

   // assuming blockDim.x is 128
   __global__ void example_syncthreads(int* input_data, int* output_data) {
       __shared__ int shared_data[128];
       // Every thread writes to a distinct element of 'shared_data':
       shared_data[threadIdx.x] = input_data[threadIdx.x];

       // All threads synchronize, guaranteeing all writes to 'shared_data' are ordered 
       // before any thread is unblocked from '__syncthreads()':
       __syncthreads();

       // A single thread safely reads 'shared_data':
       if (threadIdx.x == 0) {
           int sum = 0;
           for (int i = 0; i < blockDim.x; ++i) {
               sum += shared_data[i];
           }
           output_data[blockIdx.x] = sum;
       }
   }

共享内存的大小因使用的 GPU 架构而异。
由于共享内存和 L1 cache 共享相同的物理空间，使用共享内存会减少 kernel 可用的 L1 cache 大小。
此外，如果 kernel 不使用共享内存，整个物理空间将被 L1 cache 利用。
CUDA runtime API 提供了使用 ``cudaGetDeviceProperties`` 函数查询每个 SM 和每个 thread block 的共享内存大小的功能，
通过查看 ``cudaDeviceProp.sharedMemPerMultiprocessor`` 和 ``cudaDeviceProp.sharedMemPerBlock`` 设备属性。

CUDA runtime API 提供 ``cudaFuncSetCacheConfig`` 函数来告诉 runtime 是为共享内存分配更多空间，还是为 L1 cache 分配更多空间。
此函数向 runtime 指定偏好，但不保证会被采纳。
Runtime 可以根据可用资源和 kernel 的需求自由做出决定。

共享内存可以静态和动态分配。

.. _writing-cuda-kernels-static-allocation-shared-memory:

2.2.3.2.1. 静态分配共享内存
"""""""""""""""""""""""""""""""

要静态分配共享内存，程序员必须使用 ``__shared__`` 说明符在 kernel 内声明变量。
变量将在共享内存中分配，并在 kernel 执行期间持续存在。以这种方式声明的共享内存大小必须在编译时指定。
例如，以下代码片段（位于 kernel 主体中）声明了一个类型为 ``float`` 、包含 1024 个元素的共享内存数组。

.. code-block:: c++

   __shared__ float sharedArray[1024];

在此声明之后，thread block 中的所有 threads 都可以访问此共享内存数组。
必须注意避免同一 thread block 中 threads 之间的数据竞争，通常使用 ``__syncthreads()`` 。

.. _writing-cuda-kernels-dynamic-allocation-shared-memory:

2.2.3.2.2. 动态分配共享内存
"""""""""""""""""""""""""""""""

要动态分配共享内存，程序员可以在三重尖括号表示法的 kernel 启动中将每个 thread block 所需的共享内存字节数指定为第三个（可选）参数，
如 ``functionName<<<grid, block, sharedMemoryBytes>>>()`` 。

然后，在 kernel 内部，程序员可以使用 ``extern __shared__`` 说明符声明一个将在 kernel 启动时动态分配的变量。

.. code-block:: c++

   extern __shared__ float sharedArray[];

需要注意的是，如果想要多个动态分配的共享内存数组，必须使用指针算术手动划分单个 ``extern __shared__`` 。
例如，如果想要以下等效内容：

.. code-block:: c++

   short array0[128];
   float array1[64];
   int   array2[256];

在动态分配的共享内存中，可以按以下方式声明和初始化数组：

.. code-block:: c++

   extern __shared__ float array[];

   short* array0 = (short*)array;
   float* array1 = (float*)&array0[128];
   int*   array2 =   (int*)&array1[64];

注意指针需要对齐到它们指向的类型，因此以下代码（例如）不起作用，因为 ``array1`` 未对齐到 4 字节。

.. code-block:: c++

   extern __shared__ float array[];
   short* array0 = (short*)array;
   float* array1 = (float*)&array0[127];

.. _writing-cuda-kernels-registers:

2.2.3.3. 寄存器
^^^^^^^^^^^^^^^^^^^^

寄存器位于 SM 上，具有线程局部范围。寄存器使用由编译器管理，寄存器用于 kernel 执行期间的线程局部存储。
每个 SM 的寄存器数量和每个 thread block 的寄存器数量可以通过 GPU 的 ``regsPerMultiprocessor`` 和 ``regsPerBlock`` 设备属性查询。

NVCC 允许开发者通过 ``-maxrregcount`` 选项 `指定 kernel 使用的最大寄存器数量 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#maxrregcount-amount-maxrregcount>`_。
使用此选项减少 kernel 可使用的寄存器数量, 从而使更多 thread blocks 在 SM 上并发调度，但也增加了寄存器溢出的可能。

.. _writing-cuda-kernels-local-memory:

2.2.3.4. 局部内存
^^^^^^^^^^^^^^^^^^^^^^

Local memory 是类似于寄存器的线程局部存储，由 NVCC 管理，但局部内存的物理位置在全局内存空间中。
「local」标签指的是其逻辑范围，而非物理位置。局部内存用于 kernel 执行期间的线程局部存储。
编译器可能放置在局部内存中的自动变量包括：

- 无法确定使用常量索引的数组
- 会消耗过多寄存器空间的大型结构或数组
- 如果 kernel 使用的寄存器超过可用数量，即寄存器溢出的任何变量

由于局部内存空间位于设备内存中，局部内存访问具有与全局内存访问相同的延迟和带宽，并受 :ref:`writing-cuda-kernels-coalesced-global-memory-access` 中描述的相同内存合并要求约束。
然而，局部内存的组织方式使得连续的 32 位字由连续的线程 ID 访问。
因此，只要 warp 中的所有 threads 访问相同的相对地址（例如数组变量中的相同索引或结构变量中的相同成员），访问就是完全合并的。

.. _writing-cuda-kernels-constant-memory:

2.2.3.5. 常量内存
^^^^^^^^^^^^^^^^^

Constant memory 具有 grid 范围，在应用程序生命期内可访问。常量内存位于设备上，对 kernel 是只读的。
因此，它必须在 host 上使用 ``__constant__`` 说明符声明和初始化，位于任何函数之外。

``__constant__`` 内存空间说明符声明的变量：

- 位于常量内存空间中
- 具有其创建所在 CUDA context 的生命期
- 每个设备有一个不同的对象
- 可从 grid 内的所有 threads 和通过 runtime 库（ ``cudaGetSymbolAddress()`` / ``cudaGetSymbolSize()`` / ``cudaMemcpyToSymbol()`` / ``cudaMemcpyFromSymbol()`` ）从 host 访问

常量内存的总量可以通过 ``totalConstMem`` 设备属性元素查询。

常量内存对于每个线程以只读方式使用的小量数据很有用。常量内存相对于其他内存较小，通常每个设备 64KB。

以下是声明和使用常量内存的示例代码片段。

.. code-block:: c++

   // In your .cu file
   __constant__ float coeffs[4];

   __global__ void compute(float *out) {
       int idx = threadIdx.x;
       out[idx] = coeffs[0] * idx + coeffs[1];
   }

   // In your host code
   float h_coeffs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
   cudaMemcpyToSymbol(coeffs, h_coeffs, sizeof(h_coeffs));
   compute<<<1, 10>>>(device_out);

.. _writing-cuda-kernels-caches:

2.2.3.6. 缓存
^^^^^^^^^^^^^

GPU 设备具有多级缓存结构，包括 L2 和 L1 caches。

L2 cache 位于设备上，在所有 SMs 之间共享。
L2 cache 的大小可以通过 ``cudaGetDeviceProperties`` 函数的 ``l2CacheSize`` 设备属性元素查询。

如上文 :ref:`writing-cuda-kernels-shared-memory` 所述，L1 cache 物理上位于每个 SM 上，是与共享内存使用的相同物理空间。
如果 kernel 不使用共享内存，整个物理空间将被 L1 cache 利用。

L2 和 L1 caches 可以通过函数控制，允许开发者指定各种缓存行为。
这些函数的详细信息见 :ref:`configuring-l1-shared-memory-balance` 、 :ref:`l2-cache-control` 和 :ref:`low-level-load-store-functions` 。

如果不使用这些提示，编译器和 runtime 将尽最大努力高效利用 caches。

.. _writing-cuda-kernels-texture-surface:

2.2.3.7. 纹理和表面内存
^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   一些较旧的 CUDA 代码可能使用纹理内存，因为在较旧的 NVIDIA GPUs 中，在某些场景下这样做可以提供性能优势。
   在所有当前支持的 GPUs 上，这些场景可以使用直接加载和存储指令处理，使用纹理和表面内存指令不再提供任何性能优势。

GPU 可能有专用指令用于从图像加载数据以用作 3D 渲染中的纹理。
CUDA 在 `texture object API <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html>`_ 
和 `surface object API <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__SURFACE__OBJECT.html>`_ 中公开了这些指令和使用它们的机制。

纹理和表面内存在本指南中不再进一步讨论，因为在任何当前支持的 NVIDIA GPU 上的 CUDA 中使用它们没有任何优势。
CUDA 开发者可以放心地忽略这些 APIs。
对于仍在使用它们的现有代码库，这些 APIs 的解释仍可在旧版 `CUDA C++ Programming Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory>`_ 中找到。

.. _writing-cuda-kernels-distributed-shared-memory:

2.2.3.8. 分布式共享内存
^^^^^^^^^^^^^^^^^^^^^^^

:ref:`thread-block-clusters` 在计算能力 9.0 中引入，并由 :ref:`cooperative-groups` 提供支持，
使得集群内的线程能够访问该集群所拥有的线程块的共享内存。
这种分区的共享内存称为 *Distributed Shared Memory* ，相应的地址空间称为分布式共享内存地址空间。
属于同一 cluster 的 threads 可以在分布式地址空间中读取、写入或执行原子操作，无论地址属于本 thread block 还是其他 thread block。
无论 kernel 是否使用分布式共享内存，共享内存的规格（静态或动态）仍然是按每个线程块来计算的。
分布式共享内存的大小等于 cluster 中 thread blocks 数量乘以单个 thread block 的共享内存大小。

访问分布式共享内存中的数据需要所有 thread blocks 存在。
用户可以使用 :ref:`class-cluster-group` 中的 ``cluster.sync()`` 来确保所有 thread blocks 已开始执行。
用户还需要确保所有分布式共享内存操作在 thread block 退出之前完成，
例如，如果远程 thread block 正在尝试读取给定 thread block 的共享内存，程序需要确保远程 thread block 读取的共享内存在其退出之前完成。

让我们看一个简单的直方图计算以及如何使用 thread block cluster 在 GPU 上优化它。
计算直方图的标准方法是在每个 thread block 的共享内存中执行计算，然后执行全局内存原子操作。这种方法的限制是共享内存容量。
一旦直方图 bins 不再适合共享内存，用户需要直接计算直方图，因此在全局内存中进行原子操作。
有了分布式共享内存，CUDA 提供了一个中间步骤，根据直方图 bins 大小，可以在共享内存、分布式共享内存或直接在全局内存中计算直方图。

下面的 CUDA kernel 示例展示了如何根据直方图 bins 数量在共享内存或分布式共享内存中计算直方图。

.. code-block:: c++

   #include <cooperative_groups.h>

   // Distributed Shared memory histogram kernel
   __global__ void clusterHist_kernel(int *bins, 
                                      const int nbins, 
                                      const int bins_per_block, 
                                      const int *__restrict__ input,
                                      size_t array_size)
   {
     extern __shared__ int smem[];
     namespace cg = cooperative_groups;
     int tid = cg::this_grid().thread_rank();

     // Cluster initialization, size and calculating local bin offsets.
     cg::cluster_group cluster = cg::this_cluster();
     unsigned int clusterBlockRank = cluster.block_rank();
     int cluster_size = cluster.dim_blocks().x;

     for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
     {
       smem[i] = 0; //Initialize shared memory histogram to zeros
     }

     // cluster synchronization ensures that shared memory is initialized to zero in
     // all thread blocks in the cluster. It also ensures that all thread blocks
     // have started executing and they exist concurrently.
     cluster.sync();

     for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
     {
       int ldata = input[i];

       //Find the right histogram bin.
       int binid = ldata;
       if (ldata < 0)
         binid = 0;
       else if (ldata >= nbins)
         binid = nbins - 1;

       //Find destination block rank and offset for computing
       //distributed shared memory histogram
       int dst_block_rank = (int)(binid / bins_per_block);
       int dst_offset = binid % bins_per_block;

       //Pointer to target block shared memory
       int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

       //Perform atomic update of the histogram bin
       atomicAdd(dst_smem + dst_offset, 1);
     }

     // cluster synchronization is required to ensure all distributed shared
     // memory operations are completed and no thread block exits while
     // other thread blocks are still accessing distributed shared memory
     cluster.sync();

     // Perform global memory histogram, using the local distributed memory histogram
     int *lbins = bins + cluster.block_rank() * bins_per_block;
     for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
     {
       atomicAdd(&lbins[i], smem[i]);
     }
   }

上述 kernel 可以在运行时根据所需的分布式共享内存量以 cluster 大小启动。
如果直方图足够小以至于只适合一个 block 的共享内存，用户可以以 cluster 大小 1 启动 kernel。
下面的代码片段展示了如何根据共享内存需求动态启动 cluster kernel。

.. code-block:: c++

   // Launch via extensible launch
   {
     cudaLaunchConfig_t config = {0};
     config.gridDim = array_size / threads_per_block;
     config.blockDim = threads_per_block;

     // cluster_size depends on the histogram size.
     // cluster_size == 1 implies no distributed shared memory, just thread block local shared memory
     int cluster_size = 2; // size 2 is an example here
     int nbins_per_block = nbins / cluster_size;

     // dynamic shared memory size is per block.
     // Distributed shared memory size =  cluster_size * nbins_per_block * sizeof(int)
     config.dynamicSmemBytes = nbins_per_block * sizeof(int);

     CUDA_CHECK(::cudaFuncSetAttribute((void *)clusterHist_kernel, 
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       config.dynamicSmemBytes));

     cudaLaunchAttribute attribute[1];
     attribute[0].id = cudaLaunchAttributeClusterDimension;
     attribute[0].val.clusterDim.x = cluster_size;
     attribute[0].val.clusterDim.y = 1;
     attribute[0].val.clusterDim.z = 1;

     config.numAttrs = 1;
     config.attrs = attribute;

     cudaLaunchKernelEx(&config, clusterHist_kernel, bins, nbins, nbins_per_block, input, array_size);
   }

.. _writing-cuda-kernels-memory-performance:

2.2.4. 内存性能
---------------

确保正确的内存使用是在 CUDA kernels 中实现高性能的关键。本节讨论在 CUDA kernels 中实现高内存吞吐量的一些通用原则和示例。

.. _writing-cuda-kernels-coalesced-global-memory-access:

2.2.4.1. 合并全局内存访问
^^^^^^^^^^^^^^^^^^^^^^^^^

全局内存通过 32 字节内存事务访问。
当 CUDA thread 请求从全局内存读取一个字的数据时，相关的 warp 将该 warp 中所有 threads 的内存请求合并为满足请求所需的内存事务数量，具体取决于每个线程访问的字大小以及内存在线程间的分布。
例如，如果一个线程请求一个 4 字节的字，warp 对全局内存生成的实际内存事务总共将是 32 字节。
为了最有效地使用内存系统，warp 应该使用单个内存事务中获取的所有内存。
也就是说，如果一个线程从全局内存请求一个 4 字节的字，事务大小为 32 字节，如果该 warp 中的其他线程可以使用该 32 字节请求中的其他 4 字节字的数据，这将导致内存系统的最高效使用。

作为一个简单的示例，如果 warp 中的连续 threads 请求内存中连续的 4 字节字，那么 warp 将总共请求 128 字节的内存，这 128 字节将在四个 32 字节内存事务中获取。
这导致内存系统 100% 的利用率。
也就是说，100% 的内存流量被 warp 使用。:numref:`fig-writing-cuda-kernels-128-byte-coalesced-access` 展示了这个完美合并内存访问的示例。

.. _fig-writing-cuda-kernels-128-byte-coalesced-access:

.. figure:: /_static/images/perfect_coalescing_32byte_segments.png
   :alt: Coalesced memory access

   合并内存访问

相反，病理上最坏的情况是连续 threads 访问在内存中彼此相距 32 字节或更多的数据元素。
在这种情况下，warp 将被迫为每个线程发出一个 32 字节内存事务，内存流量总字节数将是 32 字节 * 32 threads/warp = 1024 字节。
然而，使用的内存量将只有 128 字节（warp 中每个线程 4 字节），因此内存利用率将只有 128 / 1024 = 12.5%。
这是内存系统的非常低效的使用。:numref:`fig-writing-cuda-kernels-128-byte-no-coalesced-access` 展示了这个非合并内存访问的示例。

.. _fig-writing-cuda-kernels-128-byte-no-coalesced-access:

.. figure:: /_static/images/no_coalescing_32byte_segments.png
   :alt: Uncoalesced memory access

   非合并内存访问

实现合并内存访问最直接的方法是让连续 threads 访问内存中的连续元素。
例如，对于以 1D thread blocks 启动的 kernel，以下 ``VecAdd`` kernel 将实现合并内存访问。
注意 thread ``workIndex`` 如何访问三个数组，连续 threads（由 ``workIndex`` 的连续值表示）访问数组中的连续元素。

.. code-block:: cuda

   __global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
   {
       int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
       if(workIndex < vectorLength)
       {
           C[workIndex] = A[workIndex] + B[workIndex];
       }
   }

并不要求连续 threads 访问连续的内存元素才能实现合并内存访问，这只是实现合并的典型方式。
只要 warp 中的所有 threads 以某种线性或排列方式访问来自相同 32 字节内存段的数据元素，就会发生合并内存访问。
换句话说，实现合并内存访问的最佳方法是最大化使用字节与传输字节的比率。

.. note::

   确保全局内存访问的正确合并是编写高性能 CUDA kernels 最重要的性能考虑因素之一。
   应用程序必须尽可能高效地使用内存系统。

.. _writing-cuda-kernels-matrix-transpose-example-global-memory:

2.2.4.1.1. 矩阵转置示例（使用全局内存）
"""""""""""""""""""""""""""""""""""""""""""

作为一个简单的示例，考虑一个就地矩阵转置 kernel，它将大小为 N x N 的 32 位 float 方阵从矩阵 ``a`` 转置到矩阵 ``c`` 。
此示例使用 2D grid，并假设启动大小为 32 x 32 threads 的 2D thread blocks，即 ``blockDim.x = 32`` 和 ``blockDim.y = 32`` ，
因此每个 2D thread block 将操作矩阵的一个 32 x 32 tile。
每个 thread 操作矩阵的一个唯一元素，因此不需要显式的线程同步。:numref:`fig-writing-cuda-kernels-figure-global-transpose` 展示了这个矩阵转置操作。
kernel 源代码紧随图后。

.. _fig-writing-cuda-kernels-figure-global-transpose:

.. figure:: /_static/images/global_transpose.png
   :alt: Matrix Transpose using Global memory

   使用全局内存的矩阵转置

   每个矩阵顶部和左侧的标签是 2D thread block 索引，也可以视为 tile 索引，其中每个小方块表示将由 2D thread block 操作的矩阵 tile。
   在此示例中，tile 大小为 32 x 32 元素，因此每个小方块表示矩阵的一个 32 x 32 tile。绿色阴影方块显示了转置操作前后示例 tile 的位置。

.. code-block:: c++

   /* macro to index a 1D memory array with 2D indices in row-major order */
   /* ld is the leading dimension, i.e. the number of columns in the matrix     */

   #define INDX( row, col, ld ) ( ( (row) * (ld) ) + (col) )

   /* CUDA kernel for naive matrix transpose */

   __global__ void naive_cuda_transpose(int m, float *a, float *c )
   {
       int myCol = blockDim.x * blockIdx.x + threadIdx.x;
       int myRow = blockDim.y * blockIdx.y + threadIdx.y;

       if( myRow < m && myCol < m )
       {
           c[INDX( myCol, myRow, m )] = a[INDX( myRow, myCol, m )];
       } /* end if */
       return;
   } /* end naive_cuda_transpose */

要确定此 kernel 是否实现合并内存访问，需要确定连续 threads 是否访问内存中的连续元素。
在 2D thread block 中， ``x`` 索引移动最快，因此 ``threadIdx.x`` 的连续值应该访问内存中的连续元素。
``threadIdx.x`` 出现在 ``myCol`` 中，可以观察到当 ``myCol`` 是 ``INDX`` 宏的第二个参数时，连续 threads 正在读取 ``a`` 的连续值，因此对 ``a`` 的读取是完美合并的。

然而，对 ``c`` 的写入没有合并，因为 ``threadIdx.x`` 的连续值（再次查看 ``myCol`` ）正在向 ``c`` 写入彼此相距 ``ld``（前导维度）个元素的元素。
这是因为现在 ``myCol`` 是 ``INDX`` 宏的第一个参数，随着 ``INDX`` 的第一个参数增加 1，内存位置变化 ``ld`` 。
当 ``ld`` 大于 32 时（当矩阵大小大于 32 时会发生这种情况），这等效于 :numref:`fig-writing-cuda-kernels-128-byte-no-coalesced-access` 中显示的病理情况。

为了缓解这些非合并写入，可以使用共享内存，这将在下一节中描述。

.. _writing-cuda-kernels-shared-memory-access-patterns:

2.2.4.2. 共享内存访问模式
^^^^^^^^^^^^^^^^^^^^^^^^^

共享内存有 32 个 banks，组织方式是连续的 32 位字映射到连续的 banks。每个 bank 的带宽为每个时钟周期 32 位。

当同一 warp 中的多个 threads 尝试访问同一 bank 中的不同元素时，会发生 bank conflict。
在这种情况下，对该 bank 中数据的访问将被序列化，直到请求它的所有 threads 都获得了该 bank 中的数据。这种访问的序列化会导致性能损失。

这种情况的两个例外发生在同一 warp 中的多个 threads 正在访问（读取或写入）相同的共享内存位置时。
对于读取访问，该字被广播给请求的 threads。对于写入访问，每个共享内存地址只被其中一个线程写入（哪个线程执行写入是未定义的）。

:numref:`fig-writing-cuda-kernels-shared-memory-5-x-examples-of-strided-shared-memory-accesses` 展示了一些步幅访问的示例。bank 内的红色框表示共享内存中的唯一位置。

.. _fig-writing-cuda-kernels-shared-memory-5-x-examples-of-strided-shared-memory-accesses:

.. figure:: /_static/images/examples-of-strided-shared-memory-accesses.png
   :alt: Strided Shared Memory Accesses in 32 bit bank size mode.

   32 位 bank 大小模式下的步幅共享内存访问

   左
      步幅为一个 32 位字的线性寻址（无 bank conflict）。
   中
      步幅为两个 32 位字的线性寻址（双路 bank conflict）。
   右
      步幅为三个 32 位字的线性寻址（无 bank conflict）。

:numref:`fig-writing-cuda-kernels-shared-memory-5-x-examples-of-irregular-shared-memory-accesses` 展示了一些涉及广播机制的内存读取访问示例。
bank 内的红色框表示共享内存中的唯一位置。如果多个箭头指向同一位置，数据将被广播给所有请求它的 threads。

.. _fig-writing-cuda-kernels-shared-memory-5-x-examples-of-irregular-shared-memory-accesses:

.. figure:: /_static/images/examples-of-irregular-shared-memory-accesses.png
   :alt: Irregular Shared Memory Accesses.

   不规则共享内存访问

   左
      通过随机排列的无冲突访问。
   中
      无冲突访问，因为 threads 3、4、6、7 和 9 访问 bank 5 内的同一字。
   右
      无冲突广播访问（threads 访问 bank 内的同一字）。

.. note::

   避免 bank conflicts 是编写使用共享内存的高性能 CUDA kernels 的重要性能考虑因素。

.. _writing-cuda-kernels-matrix-transpose-example-shared-memory:

2.2.4.2.1. 矩阵转置示例（使用共享内存）
"""""""""""""""""""""""""""""""""""""""""""

在之前的示例 :ref:`writing-cuda-kernels-matrix-transpose-example-global-memory` 中，
展示了一个简单的矩阵转置实现，它在功能上是正确的，但没有针对全局内存的高效使用进行优化，
因为 ``c`` 矩阵的写入没有正确合并。在此示例中，共享内存将被视为用户管理的缓存，
用于暂存全局内存的加载和存储，从而实现读取和写入的合并全局内存访问。

.. tab-set::

   .. tab-item:: Example

      .. code-block:: cuda

         /* definitions of thread block size in X and Y directions */

         #define THREADS_PER_BLOCK_X 32
         #define THREADS_PER_BLOCK_Y 32

         /* macro to index a 1D memory array with 2D indices in row-major order */
         /* ld is the leading dimension, i.e. the number of columns in the matrix */

         #define INDX( row, col, ld ) ( ( (row) * (ld) ) + (col) )

         /* CUDA kernel for shared memory matrix transpose */

         __global__ void smem_cuda_transpose(int m, float *a, float *c )
         {

            /* declare a statically allocated shared memory array */

            __shared__ float smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y];

            /* determine my row tile and column tile index */

            const int tileCol = blockDim.x * blockIdx.x;
            const int tileRow = blockDim.y * blockIdx.y;

            /* read from global memory into shared memory array */
            smemArray[threadIdx.x][threadIdx.y] = a[INDX(tileRow + threadIdx.y, tileCol + threadIdx.x, m)];

            /* synchronize the threads in the thread block */
            __syncthreads();

            /* write the result from shared memory to global memory */
            c[INDX(tileCol + threadIdx.y, tileRow + threadIdx.x, m)] = smemArray[threadIdx.y][threadIdx.x];
            return;

         } /* end smem_cuda_transpose */

   .. tab-item:: Example with array checks

      .. code-block:: cuda

         /* definitions of thread block size in X and Y directions */

         #define THREADS_PER_BLOCK_X 32
         #define THREADS_PER_BLOCK_Y 32

         /* macro to index a 1D memory array with 2D indices in column-major order */
         /* ld is the leading dimension, i.e. the number of rows in the matrix     */

         #define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

         /* CUDA kernel for shared memory matrix transpose */

         __global__ void smem_cuda_transpose(int m, float *a, float *c)
         {

            /* declare a statically allocated shared memory array */

            __shared__ float smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y];

            /* determine my row and column indices for the error checking code */

            const int myRow = blockDim.x * blockIdx.x + threadIdx.x;
            const int myCol = blockDim.y * blockIdx.y + threadIdx.y;

            /* determine my row tile and column tile index */

            const int tileX = blockDim.x * blockIdx.x;
            const int tileY = blockDim.y * blockIdx.y;

            if(myRow < m && myCol < m)
            {
               /* read from global memory into shared memory array */
               smemArray[threadIdx.x][threadIdx.y] = a[INDX(tileX + threadIdx.x, tileY + threadIdx.y, m)];
            } /* end if */

            /* synchronize the threads in the thread block */
            __syncthreads();

            if( myRow < m && myCol < m )
            {
               /* write the result from shared memory to global memory */
               c[INDX(tileY + threadIdx.x, tileX + threadIdx.y, m)] = smemArray[threadIdx.y][threadIdx.x];
            } /* end if */
            return;

         } /* end smem_cuda_transpose */


此示例中说明的基本性能优化是确保在访问全局内存时，内存访问正确合并。
在执行复制之前，每个线程计算其 ``tileRow`` 和 ``tileCol`` 索引。
这些是要操作的具体 tile 的索引，这些 tile 索引基于正在执行的 thread block。
同一 thread block 中的每个线程具有相同的 ``tileRow`` 和 ``tileCol`` 值，因此可以将其视为此特定 thread block 将操作的 tile 的起始位置。

然后 kernel 继续每个 thread block 使用以下语句将矩阵的一个 32 x 32 tile 从全局内存复制到共享内存。
由于 warp 的大小为 32 threads，此复制操作将由 32 warps 执行，warps 之间没有保证的顺序。

.. code-block:: c++

   smemArray[threadIdx.x][threadIdx.y] = a[INDX( tileRow + threadIdx.y, tileCol + threadIdx.x, m )];

注意，由于 ``threadIdx.x`` 出现在 ``INDX`` 的第二个参数中，连续 threads 正在访问内存中的连续元素，因此对 ``a`` 的读取是完美合并的。

kernel 中的下一步是调用 ``__syncthreads()`` 函数。
这确保 thread block 中的所有 threads 在继续之前已完成对先前代码的执行，因此在下一步之前，对共享内存中 ``a`` 的写入已完成。
这至关重要，因为下一步将涉及 threads 从共享内存读取。如果没有 ``__syncthreads()`` 调用，在某些 warps 继续推进代码之前，不能保证 thread block 中的所有 warps 都已完成对共享内存中 ``a`` 的读取。

在 kernel 的这一点，对于每个 thread block， ``smemArray`` 拥有矩阵的一个 32 x 32 tile，按与原始矩阵相同的顺序排列。
为确保 tile 内的元素正确转置，当它们读取 ``smemArray`` 时， ``threadIdx.x`` 和 ``threadIdx.y`` 被交换。
为确保整体 tile 放置在 ``c`` 中的正确位置，当它们写入 ``c`` 时， ``tileRow`` 和 ``tileCol`` 索引也被交换。
为确保正确合并， ``threadIdx.x`` 用于 ``INDX`` 的第二个参数，如下面的语句所示。

.. code-block:: c++

   c[INDX( tileCol + threadIdx.y, tileRow + threadIdx.x, m )] = smemArray[threadIdx.y][threadIdx.x];

此 kernel 展示了共享内存的两个常见用途。

- 共享内存用于暂存来自全局内存的数据，以确保对全局内存的读取和写入都正确合并。
- 共享内存用于允许同一 thread block 中的 threads 相互共享数据。

.. _writing-cuda-kernels-shared-memory-bank-conflicts:

2.2.4.2.2. 共享内存 Bank Conflicts
"""""""""""""""""""""""""""""""""""""""""""

在 :ref:`writing-cuda-kernels-shared-memory-access-patterns` 中，描述了共享内存的 bank 结构。
在之前的矩阵转置示例中，实现了对全局内存的正确合并内存访问，但没有考虑是否存在共享内存 bank conflicts。
考虑以下 2D 共享内存声明：

.. code-block:: c++

   __shared__ float smemArray[32][32];

由于一个 warp 是 32 threads，同一 warp 中的每个 thread 将具有固定的 ``threadIdx.y`` 值，并且 ``0 <= threadIdx.x < 32`` 。

:numref:`fig-writing-cuda-kernels-figure-bank-conflicts-shared-mem` 的左面板展示了 warp 中的 threads 访问 ``smemArray`` 一列中数据的情况。
Warp 0 正在访问内存位置 ``smemArray[0][0]`` 到 ``smemArray[31][0]`` 。
在 C++ 多维数组排序中，最后一个索引移动最快，因此 warp 0 中的连续 threads 正在访问相距 32 个元素的内存位置。
如图所示，颜色表示 banks，warp 0 对整个列的这种访问导致 32 路 bank conflict。

:numref:`fig-writing-cuda-kernels-figure-bank-conflicts-shared-mem` 的右面板展示了 warp 中的 threads 访问 ``smemArray`` 一行中数据的情况。
Warp 0 正在访问内存位置 ``smemArray[0][0]`` 到 ``smemArray[0][31]`` 。
在这种情况下，warp 0 中的连续 threads 正在访问相邻的内存位置。
如图所示，颜色表示 banks，warp 0 对整个行的这种访问没有 bank conflicts。
理想情况是 warp 中的每个 thread 访问具有不同颜色的共享内存位置。

.. _fig-writing-cuda-kernels-figure-bank-conflicts-shared-mem:

.. figure:: /_static/images/bank-conflicts-shared-mem.png
   :alt: Bank Structure in Shared Memory

   32 x 32 共享内存数组中的 Bank 结构

   框中的数字表示 warp 索引。颜色表示与该共享内存位置关联的 bank。

回到 :ref:`writing-cuda-kernels-matrix-transpose-example-shared-memory` 中的示例，可以检查共享内存的使用以确定是否存在 bank conflicts。
共享内存的第一次使用是将数据从全局内存存储到共享内存时：

.. code-block:: c++

   smemArray[threadIdx.x][threadIdx.y] = a[INDX( tileRow + threadIdx.y, tileCol + threadIdx.x, m )];

由于 C++ 数组按行优先顺序存储，同一 warp 中的连续 threads（由 ``threadIdx.x`` 的连续值表示）将以 32 个元素的步幅访问 ``smemArray`` ，因为 ``threadIdx.x`` 是数组的第一个索引。
这导致 32 路 bank conflict，并由 :numref:`fig-writing-cuda-kernels-figure-bank-conflicts-shared-mem` 的左面板展示。

共享内存的第二次使用是将数据从共享内存写回全局内存时：

.. code-block:: c++

   c[INDX( tileCol + threadIdx.y, tileRow + threadIdx.x, m )] = smemArray[threadIdx.y][threadIdx.x];

在这种情况下，由于 ``threadIdx.x`` 是 ``smemArray`` 数组的第二个索引，同一 warp 中的连续 threads 将以 1 个元素的步幅访问 ``smemArray`` 。
这不会导致 bank conflicts，并由 :numref:`fig-writing-cuda-kernels-figure-bank-conflicts-shared-mem` 的右面板展示。

:ref:`writing-cuda-kernels-matrix-transpose-example-shared-memory` 中展示的矩阵转置 kernel 有一次共享内存访问没有 bank conflicts，
有一次访问有 32 路 bank conflict。避免 bank conflicts 的常见修复方法是通过在数组的列维度上加一来填充共享内存：

.. code-block:: c++

   __shared__ float smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y+1];

这个对 ``smemArray`` 声明的微小调整将消除 bank conflicts。
为了说明这一点，考虑 :numref:`fig-writing-cuda-kernels-figure-no-bank-conflicts-shared-mem`，其中共享内存数组声明为 32 x 33 大小。
可以看到，无论同一 warp 中的 threads 是访问整个列还是整个行的共享内存数组，bank conflicts 都已被消除，即同一 warp 中的 threads 访问具有不同颜色的位置。

.. _fig-writing-cuda-kernels-figure-no-bank-conflicts-shared-mem:

.. figure:: /_static/images/no-bank-conflicts-shared-mem.png
   :alt: Bank Structure in Shared Memory

   32 x 33 共享内存数组中的 Bank 结构

   框中的数字表示 warp 索引。颜色表示与该共享内存位置关联的 bank。

.. _writing-cuda-kernels-atomics:

2.2.5. 原子操作
---------------

高性能 CUDA kernels 依赖于尽可能多地表达算法并行性。GPU kernel 执行的异步特性要求 threads 尽可能独立运行。
并不总是能够实现 threads 的完全独立，正如我们在 :ref:`writing-cuda-kernels-shared-memory` 中看到的，存在一种机制让同一 thread block 中的 threads 交换数据和同步。

在整个 grid 级别上，没有机制来同步 grid 中的所有 threads。
然而，有一种机制通过使用原子函数提供对全局内存位置的同步访问。
原子函数允许 thread 获取对全局内存位置的锁，并对该位置执行读-改-写操作。
当持有锁时，没有其他 thread 可以访问同一位置。
CUDA 提供了与 C++ 标准库原子行为相同的原子操作，即 ``cuda::std::atomic`` 和 ``cuda::std::atomic_ref`` 。
CUDA 还提供扩展的 C++ 原子操作 ``cuda::atomic`` 和 ``cuda::atomic_ref`` ，允许用户指定原子操作的 :ref:`thread-scopes`。
原子函数的详细信息见 :ref:`atomic-functions`。

以下是使用 ``cuda::atomic_ref`` 执行设备级原子加法的示例用法，其中 ``array`` 是一个 float 数组， ``result`` 是指向全局内存中某个位置的 float 指针，该位置是存储数组和的位置。

.. code-block:: c++

   __global__ void sumReduction(int n, float *array, float *result) {
      ...
      tid = threadIdx.x + blockIdx.x * blockDim.x;

      cuda::atomic_ref<float, cuda::thread_scope_device> result_ref(result);
      result_ref.fetch_add(array[tid]);
      ...
   }

原子函数应该谨慎使用，因为它们强制执行可能影响性能的线程同步。

.. _writing-cuda-kernels-cooperative-groups:

2.2.6. 协作组
-------------

:ref:`cooperative-groups` 是 CUDA C++ 中提供的软件工具，允许应用程序定义可以相互同步的线程组，即使该线程组跨越多个 thread blocks、单个 GPU 上的多个 grids，甚至跨多个 GPUs。
一般来说，传统 CUDA 编程模型允许线程块或线程块集群内的线程进行高效同步，但它并没有提供机制来指定比线程块或集群更小的线程组。
同样，传统 CUDA 编程模型也没有提供能够跨线程块进行同步的机制或保证。

协作组通过软件提供这两种能力。
协作组允许应用程序创建跨越 thread blocks 和 clusters 边界的线程组，尽管这样做会有一些语义限制和性能影响，这些在 :ref:`cooperative-groups` 中有详细描述。

.. _writing-cuda-kernels-kernel-launch-and-occupancy:

2.2.7. Kernel 启动和占用率
---------------------------

当 CUDA kernel 启动时，CUDA threads 根据启动时指定的执行配置被分组为 thread blocks 和 grid。一旦 kernel 启动，调度器将 thread blocks 分配给 SMs。
调度器将哪些 thread blocks 调度到哪些 SMs 上执行的详细信息无法被应用程序控制或查询，调度器也不做出任何排序保证，因此程序不能依赖特定的调度顺序或方案进行正确执行。

可以在 SM 上调度的 blocks 数量取决于给定 thread block 所需的硬件资源以及 SM 上可用的硬件资源。当 kernel 首次启动时，调度器开始将 thread blocks 分配给 SMs。
只要 SM 上还有未被其他线程块占用的充足硬件资源，调度器就会继续向 SM 分配线程块。
如果在某个时刻，没有任何一个 SM 有能力接纳新的线程块，调度器就会等待，直到 SM 完成之前分配的线程块。
一旦发生这种情况（即 SM 完成工作），SM 就可以接受更多任务，调度器随后会将新的线程块分配给它们。
这个过程会一直持续，直到所有线程块都被调度并执行完毕。

``cudaGetDeviceProperties`` 函数允许应用程序通过 `设备属性 <https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp>`_ 查询每个 SM 的限制。
注意有每个 SM 和每个 thread block 的限制。

- ``maxBlocksPerMultiProcessor`` ：每个 SM 的最大驻留 blocks 数量。
- ``sharedMemPerMultiprocessor`` ：每个 SM 可用的共享内存量（字节）。
- ``regsPerMultiprocessor`` ：每个 SM 可用的 32 位寄存器数量。
- ``maxThreadsPerMultiProcessor`` ：每个 SM 的最大驻留 threads 数量。
- ``sharedMemPerBlock`` ：thread block 可分配的最大共享内存量（字节）。
- ``regsPerBlock`` ：thread block 可分配的最大 32 位寄存器数量。
- ``maxThreadsPerBlock`` ：每个 thread block 的最大 threads 数量。

CUDA kernel 的占用率是活动 warps 数量与 SM 支持的最大活动 warps 数量的比率。
通常，尽可能提高占用率是一个好的做法，这可以隐藏延迟并提高性能。

要计算占用率，需要知道 SM 的资源限制（刚才描述的），还需要知道所讨论的 CUDA kernel 需要什么资源。
要在每个 kernel 基础上确定资源使用情况，
在程序编译期间可以使用 ``nvcc`` 的 `--resource-usage <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#resource-usage-res-usage>`_ ，
它将显示 kernel 所需的寄存器数量和共享内存。

为了说明这一点，请看一个计算能力为 10.0 的设备，其设备属性如 :numref:`tbl:writing-cuda-kernels-sm-resource-example` 所列。

.. _tbl:writing-cuda-kernels-sm-resource-example:

.. table:: SM 资源示例

   =============================================== ============
   资源                                            值
   =============================================== ============
   ``maxBlocksPerMultiProcessor``                  32
   ``sharedMemPerMultiprocessor``                  233472
   ``regsPerMultiprocessor``                       65536
   ``maxThreadsPerMultiProcessor``                 2048
   ``sharedMemPerBlock``                           49152
   ``regsPerBlock``                                65536
   ``maxThreadsPerBlock``                          1024
   =============================================== ============

如果 kernel 以 ``testKernel<<<512, 768>>>()`` 启动，即每个 block 768 threads，每个 SM 一次只能执行 2 个 thread blocks。
调度器无法在每个 SM 上分配超过 2 个 thread blocks，因为 ``maxThreadsPerMultiProcessor`` 是 2048。所以占用率将是 (768 * 2) / 2048，即 75%。

如果 kernel 以 ``testKernel<<<512, 32>>>()`` 启动，即每个 block 32 threads，每个 SM 不会达到 ``maxThreadsPerMultiProcessor`` 的限制，
但由于 ``maxBlocksPerMultiProcessor`` 是 32，调度器只能向每个 SM 分配 32 个 thread blocks。由于 block 中的 threads 数量为 32，
SM 上驻留的总 threads 数量将是 32 blocks * 32 threads per block，即总共 1024 threads。
由于计算能力 10.0 SM 的每个 SM 最大驻留 threads 数量为 2048，这种情况下的占用率是 1024 / 2048，即 50%。

同样的分析可以用共享内存进行。
例如，如果 kernel 使用 100KB 共享内存，调度器只能向每个 SM 分配 2 个 thread blocks，
因为该 SM 上的第三个 thread block 将需要另外 100KB 共享内存，总共 300KB，这超过了每个 SM 可用的 233472 字节。

每个线程块的线程数以及每个线程块的共享内存使用量是由程序员显式控制的，可以对其进行调整以达到理想的占用率。
程序员对寄存器使用量的控制能力有限，因为编译器和运行时会尝试自动优化寄存器的使用。
不过，程序员可以通过 ``nvcc`` 的 `--maxrregcount <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#maxrregcount-amount-maxrregcount>`_ 
指定每个 thread block 的最大寄存器数量。
如果 kernel 需要的寄存器超过此指定数量， kernel 很可能会发生寄存器溢出到本地内存的情况，这将影响 kernel 的性能。
在某些情况下，即使发生溢出，限制寄存器允许调度更多 thread blocks，这反过来增加占用率，并可能导致整体性能的提升。