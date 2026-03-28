.. _intro-to-cuda-c:
.. _intro-applications-cpp:

2.1. CUDA C++ 简介
==================

本章通过 C++ 示例介绍 CUDA 编程模型的一些基本概念。

本编程指南主要关注 CUDA runtime API。CUDA runtime API 是在 C++ 中使用 CUDA 最常用的方式，它建立在较低级别的 CUDA driver API 之上。

:ref:`CUDA 工具包和 NVIDIA 驱动程序<cuda-toolkit-and-nvidia-driver>` 讨论了这两种 API 之间的区别， :ref:`CUDA Driver API<driver-api>` 讨论了混合使用这两种 API 编写代码的方法。

本指南假设已安装 CUDA Toolkit 和 NVIDIA Driver，并且存在支持的 NVIDIA GPU。有关安装必要 CUDA 组件的说明，请参阅 `CUDA 快速入门指南 <https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html>`__。

.. _compilation-with-nvcc:

2.1.1. 使用 NVCC 编译
----------------------

用 C++ 编写的 GPU 代码使用 NVIDIA Cuda 编译器 ``nvcc`` 进行编译。 ``nvcc`` 是一个编译器驱动程序，它简化了编译 C++ 或 PTX 代码的过程：它提供简单且熟悉的命令行选项，并通过调用实现不同编译阶段的工具集合来执行这些选项。

本指南将展示可以在任何安装了 CUDA Toolkit 的 Linux 系统、Windows 命令行或 PowerShell，以及安装了 CUDA Toolkit 的 Windows Subsystem for Linux 上使用的 ``nvcc`` 命令行。
本指南的 :doc:`nvcc` 章节涵盖了 ``nvcc`` 的常见用例，完整文档由 `nvcc 用户手册 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>`__ 提供。

.. _kernels:

2.1.2. Kernel
--------------

正如 :ref:`programming-model` 的介绍中所述，在 GPU 上执行且可以从主机调用的函数称为 kernel。Kernel 被编写为由许多并行线程同时运行。

.. _intro-cpp-specifying-kernels:

2.1.2.1. 指定 Kernel
^^^^^^^^^^^^^^^^^^^^

Kernel 的代码使用 ``__global__`` 声明说明符指定。这向编译器表明该函数将被编译为 GPU 代码，并允许从 kernel 启动中调用它。Kernel 启动是一个启动 kernel 运行的操作，通常从 CPU 发起。Kernel 是返回类型为 ``void`` 的函数。

.. code-block:: cuda

   // Kernel definition
   __global__ void vecAdd(float* A, float* B, float* C)
   {

   }

.. _intro-cpp-launching-kernels:

2.1.2.2. 启动 Kernel
^^^^^^^^^^^^^^^^^^^^

并行执行 kernel 的线程数量作为 kernel 启动的一部分指定。这称为执行配置。同一 kernel 的不同调用可以使用不同的执行配置，例如不同数量的线程或线程块。

从 CPU 代码启动 kernel 有两种方式：:ref:`intro-cpp-launching-kernels-triple-chevron` 和 ``cudaLaunchKernelEx`` 。三重尖括号表示法是最常用的 kernel 启动方式，在此介绍。使用 ``cudaLaunchKernelEx`` 启动 kernel 的示例在 :numref:`Section 3.1.1` 中详细展示和讨论。

.. _intro-cpp-launching-kernels-triple-chevron:

2.1.2.2.1. 三重尖括号表示法
""""""""""""""""""""""""""""

三重尖括号表示法是一种用于启动 kernel 的 :ref:`execution-configuration`。它之所以称为三重尖括号，是因为它使用三个尖括号字符来封装 kernel 启动的执行配置，即 ``<<< >>>`` 。执行配置参数在尖括号内以逗号分隔的列表形式指定，类似于函数调用的参数。下面显示了 ``vecAdd`` kernel 启动的语法。

.. code-block:: cuda

    __global__ void vecAdd(float* A, float* B, float* C)
    {

    }

   int main()
   {
       ...
       // Kernel invocation
       vecAdd<<<1, 256>>>(A, B, C);
       ...
   }

三重尖括号表示法的前两个参数分别是 grid 维度和 thread block 维度。当使用一维 thread block 或 grid 时，可以使用整数来指定维度。

上面的代码启动一个包含 256 个线程的单个 thread block。每个线程将执行完全相同的 kernel 代码。在 :ref:`intro-cpp-thread-indexing` 中，我们将展示每个线程如何使用其在 thread block 和 grid 中的索引来更改它操作的数据。

每个 block 中的线程数量有限制，因为 block 的所有线程都驻留在同一个流式多处理器 (SM) 上，并且必须共享 SM 的资源。在当前的 GPU 上，一个 thread block 最多可以包含 1024 个线程。如果资源允许，可以在一个 SM 上同时调度多个 thread block。

Kernel 启动相对于主机线程是异步的。也就是说，kernel 将被设置为在 GPU 上执行，但主机代码不会等待 kernel 在 GPU 上完成（甚至开始）执行后才继续。必须使用某种形式的 GPU 和 CPU 之间的同步来确定 kernel 已完成。最基本的版本是完全同步整个 GPU，如 :ref:`intro-synchronizing-the-gpu` 所示。更复杂的同步方法在 :doc:`asynchronous-execution` 中介绍。

当使用 2 维或 3 维 grid 或 thread block 时，CUDA 类型 ``dim3`` 用作 grid 和 thread block 维度参数。下面的代码片段显示了一个使用 16×16 grid 的 thread block 启动 ``MatAdd`` kernel，每个 thread block 为 8×8。

.. code-block:: cuda

   int main()
   {
       ...
       dim3 grid(16,16);
       dim3 block(8,8);
       MatAdd<<<grid, block>>>(A, B, C);
       ...
   }

.. _intro-cpp-thread-indexing:

2.1.2.3. 线程和 Grid 索引进内函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 kernel 代码中，CUDA 提供了内建函数来访问执行配置的参数以及线程或 block 的索引。

- ``threadIdx`` 给出线程在其 thread block 内的索引。Thread block 中的每个线程都有不同的索引。

- ``blockDim`` 给出 thread block 的维度，这是在 kernel 启动的执行配置中指定的。

- ``blockIdx`` 给出 thread block 在 grid 内的索引。每个 thread block 都有不同的索引。

- ``gridDim`` 给出 grid 的维度，这是在 kernel 启动时的执行配置中指定的。

这些内建函数中的每一个都是一个 3 分量向量，具有 ``.x``、``.y`` 和 ``.z`` 成员。启动配置未指定的维度默认为 1。

``threadIdx`` 和 ``blockIdx`` 从零开始索引。也就是说， ``threadIdx.x`` 的值将从 0 到 ``blockDim.x-1``（含）。 ``.y`` 和 ``.z`` 在各自维度上的操作相同。

类似地， ``blockIdx.x`` 的值将从 0 到 ``gridDim.x-1``（含）， ``.y`` 和 ``.z`` 维度也分别相同。

这些允许单个线程识别它应该执行什么工作。回到 ``vecAdd`` kernel，该 kernel 接受三个参数，每个都是浮点向量。Kernel 对 ``A`` 和 ``B`` 执行逐元素加法，并将结果存储在 ``C`` 中。Kernel 被并行化，使得每个线程执行一次加法。它计算哪个元素由其线程和 grid 索引决定。

.. code-block:: cuda

   __global__ void vecAdd(float* A, float* B, float* C)
   {
      // calculate which element this thread is responsible for computing
      int workIndex = threadIdx.x + blockDim.x * blockIdx.x

      // Perform computation
      C[workIndex] = A[workIndex] + B[workIndex];
   }

   int main()
   {
       ...
       // A, B, and C are vectors of 1024 elements
       vecAdd<<<4, 256>>>(A, B, C);
       ...
   }

在此示例中，使用 4 个 thread block，每个 block 有 256 个线程来添加 1024 个元素的向量。在第一个 thread block 中， ``blockIdx.x`` 将为零，因此每个线程的 workIndex 将只是其 ``threadIdx.x`` 。在第二个 thread block 中， ``blockIdx.x`` 将为 1，因此 ``blockDim.x * blockIdx.x`` 将与 ``blockDim.x`` 相同，在这种情况下为 256。第二个 thread block 中每个线程的 ``workIndex`` 将是其 ``threadIdx.x + 256`` 。在第三个 thread block 中， ``workIndex`` 将是 ``threadIdx.x + 512`` 。

这种 ``workIndex`` 的计算对于一维并行化非常常见。扩展到二维或三维通常在每个维度上遵循相同的模式。

.. _intro-cpp-bounds-checking:

2.1.2.3.1. 边界检查
"""""""""""""""""""

上面给出的示例假设向量的长度是 thread block 大小的倍数，在这种情况下为 256 个线程。为了使 kernel 能够处理任何向量长度，我们可以添加检查以确保内存访问不会超出数组的边界，如下所示，然后启动一个会有一些非活动线程的 thread block。

.. code-block:: cuda

   __global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
   {
        // calculate which element this thread is responsible for computing
        int workIndex = threadIdx.x + blockDim.x * blockIdx.x

        if(workIndex < vectorLength)
        {
            // Perform computation
            C[workIndex] = A[workIndex] + B[workIndex];
        }
   }

使用上面的 kernel 代码，可以启动比所需更多的线程，而不会导致对数组的越界访问。当 ``workIndex`` 超过 ``vectorLength`` 时，线程退出并且不做任何工作。启动一个 block 中有不工作线程的额外线程不会产生很大的开销成本，但是应该避免启动没有线程工作的 thread block。这个 kernel 现在可以处理不是 block 大小倍数的向量长度。

所需的 thread block 数量可以计算为所需线程数量（在这种情况下为向量长度）除以每个 block 的线程数的上限。也就是说，所需线程数除以每个 block 的线程数的整数除法，向上取整。下面给出了一种将其表示为单个整数除法的常用方法。通过在整数除法之前加上 ``threads - 1`` ，这就像一个向上取整函数，仅当向量长度不能被每个 block 的线程数整除时才添加另一个 thread block。

.. code-block:: cuda

   // vectorLength is an integer storing number of elements in the vector
   int threads = 256;
   int blocks = (vectorLength + threads-1)/threads;
   vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);

`CUDA Core Compute Library (CCCL) <https://nvidia.github.io/cccl/>`__ 提供了一个方便的实用程序 ``cuda::ceil_div`` ，用于执行此向上取整除法以计算 kernel 启动所需的 block 数量。此实用程序可通过包含头文件 ``<cuda/cmath>`` 来使用。

.. code-block:: cuda

   // vectorLength is an integer storing number of elements in the vector
   int threads = 256;
   int blocks = cuda::ceil_div(vectorLength, threads);
   vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);

这里选择每个 block 256 个线程是任意的，但这通常是开始的一个好值。

.. _intro-cpp-allocating-memory:

2.1.3. GPU 计算中的内存
-----------------------

为了使用上面显示的 ``vecAdd`` kernel，数组 ``A``、``B`` 和 ``C`` 必须位于 GPU 可访问的内存中。有几种不同的方法可以做到这一点，这里将介绍其中两种。其他方法将在后面的 :ref:`memory-unified-memory` 章节中介绍。GPU 上运行的代码可用的内存空间在 :ref:`programming-model-memory` 中介绍，并在 :ref:`writing-cuda-kernels-gpu-device-memory-spaces` 中详细介绍。

.. _intro-cpp-unified-memory:

2.1.3.1. 统一内存
^^^^^^^^^^^^^^^^^

统一内存是 CUDA runtime 的一个功能，它让 NVIDIA Driver 管理主机和设备之间的数据移动。内存使用 ``cudaMallocManaged`` API 分配，或者通过使用 ``__managed__`` 说明符声明变量。NVIDIA Driver 将确保无论 GPU 还是 CPU 尝试访问内存，内存都是可访问的。

下面的代码显示了一个完整的函数来启动 ``vecAdd`` kernel，该 kernel 对将在 GPU 上使用的输入和输出向量使用统一内存。 ``cudaMallocManaged`` 分配可以从 CPU 或 GPU 访问的缓冲区。这些缓冲区使用 ``cudaFree`` 释放。

.. code-block:: cuda

   void unifiedMemExample(int vectorLength)
   {
       // Pointers to memory vectors
       float* A = nullptr;
       float* B = nullptr;
       float* C = nullptr;
       float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

       // Use unified memory to allocate buffers
       cudaMallocManaged(&A, vectorLength*sizeof(float));
       cudaMallocManaged(&B, vectorLength*sizeof(float));
       cudaMallocManaged(&C, vectorLength*sizeof(float));

       // Initialize vectors on the host
       initArray(A, vectorLength);
       initArray(B, vectorLength);

       // Launch the kernel. Unified memory will make sure A, B, and C are
       // accessible to the GPU
       int threads = 256;
       int blocks = cuda::ceil_div(vectorLength, threads);
       vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
       // Wait for the kernel to complete execution
       cudaDeviceSynchronize();

       // Perform computation serially on CPU for comparison
       serialVecAdd(A, B, comparisonResult, vectorLength);

       // Confirm that CPU and GPU got the same answer
       if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
       {
           printf("Unified Memory: CPU and GPU answers match\n");
       }
       else
       {
           printf("Unified Memory: Error - CPU and GPU answers do not match\n");
       }

       // Clean Up
       cudaFree(A);
       cudaFree(B);
       cudaFree(C);
       free(comparisonResult);

   }

统一内存在 CUDA 支持的所有操作系统和 GPU 上都受支持，尽管底层机制和性能可能因系统架构而异。:ref:`memory-unified-memory` 提供了更多详细信息。在某些 Linux 系统上（例如具有 :ref:`memory-unified-address-translation-services` 或 :ref:`memory-heterogeneous-memory-management` 的系统），所有系统内存都会自动成为统一内存，不需要使用 ``cudaMallocManaged`` 或 ``__managed__`` 说明符。

.. _intro-cpp-explicit-memory-management:

2.1.3.2. 显式内存管理
^^^^^^^^^^^^^^^^^^^^^

显式管理内存分配和内存空间之间的数据迁移可以帮助提高应用程序性能，尽管这会使代码更加冗长。下面的代码使用 ``cudaMalloc`` 在 GPU 上显式分配内存。GPU 上的内存使用与前面统一内存示例中相同的 ``cudaFree`` API 释放。

.. code-block:: cuda

   void explicitMemExample(int vectorLength)
   {
       // Pointers for host memory
       float* A = nullptr;
       float* B = nullptr;
       float* C = nullptr;
       float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
       
       // Pointers for device memory
       float* devA = nullptr;
       float* devB = nullptr;
       float* devC = nullptr;

       //Allocate Host Memory using cudaMallocHost API. This is best practice
       // when buffers will be used for copies between CPU and GPU memory
       cudaMallocHost(&A, vectorLength*sizeof(float));
       cudaMallocHost(&B, vectorLength*sizeof(float));
       cudaMallocHost(&C, vectorLength*sizeof(float));

       // Initialize vectors on the host
       initArray(A, vectorLength);
       initArray(B, vectorLength);

       // start-allocate-and-copy
       // Allocate memory on the GPU
       cudaMalloc(&devA, vectorLength*sizeof(float));
       cudaMalloc(&devB, vectorLength*sizeof(float));
       cudaMalloc(&devC, vectorLength*sizeof(float));

       // Copy data to the GPU
       cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
       cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
       cudaMemset(devC, 0, vectorLength*sizeof(float));
       // end-allocate-and-copy

       // Launch the kernel
       int threads = 256;
       int blocks = cuda::ceil_div(vectorLength, threads);
       vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
       // wait for kernel execution to complete
       cudaDeviceSynchronize();

       // Copy results back to host
       cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);

       // Perform computation serially on CPU for comparison
       serialVecAdd(A, B, comparisonResult, vectorLength);

       // Confirm that CPU and GPU got the same answer
       if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
       {
           printf("Explicit Memory: CPU and GPU answers match\n");
       }
       else
       {
           printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
       }

       // clean up
       cudaFree(devA);
       cudaFree(devB);
       cudaFree(devC);
       cudaFreeHost(A);
       cudaFreeHost(B);
       cudaFreeHost(C);
       free(comparisonResult);
   }


CUDA API ``cudaMemcpy`` 用于将数据从驻留在 CPU 上的缓冲区复制到驻留在 GPU 上的缓冲区。
除了目标指针、源指针和大小（以字节为单位）之外， ``cudaMemcpy`` 的最后一个参数是 ``cudaMemcpyKind_t`` 。
它可以具有诸如 ``cudaMemcpyHostToDevice``（用于从 CPU 到 GPU 的复制）、 ``cudaMemcpyDeviceToHost``（用于从 GPU 到 CPU 的复制）或 ``cudaMemcpyDeviceToDevice``（用于 GPU 内部或 GPU 之间的复制）等值。

在此示例中， ``cudaMemcpyDefault`` 作为最后一个参数传递给 ``cudaMemcpy`` 。这使 CUDA 使用源指针和目标指针的值来确定要执行的复制类型。

``cudaMemcpy`` API 是同步的。也就是说，它在复制完成之前不会返回。异步复制在 :ref:`async-execution-memory-transfers` 中介绍。

代码使用 ``cudaMallocHost`` 在 CPU 上分配内存。这在主机上分配 :ref:`memory-page-locked-host-memory` ，这可以提高复制性能，并且对于 :ref:`async-execution-memory-transfers` 是必需的。一般来说，对于将用于与 GPU 进行数据传输的 CPU 缓冲区，使用页锁定内存是良好的实践。如果在某些系统上锁定了太多主机内存，性能可能会下降。最佳实践是仅锁定用于向 GPU 发送数据或从 GPU 接收数据的缓冲区。

2.1.3.3. 内存管理和应用程序性能
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

从上面的示例可以看出，显式内存管理更加冗长，需要程序员指定主机和设备之间的复制。这是显式内存管理的优点和缺点：它提供了更多控制何时在主机和设备之间复制数据、内存驻留在哪里以及确切分配什么内存的机会。显式内存管理可以提供控制内存传输并将其与其他计算重叠的性能机会。

当使用统一内存时，有 CUDA API（将在 :ref:`memory-mem-advise-prefetch` 中介绍）向管理内存的 NVIDIA driver 提供提示，这可以在使用统一内存时实现使用显式内存管理的一些性能优势。

.. _intro-synchronizing-the-gpu:

2.1.4. 同步 CPU 和 GPU
-----------------------

如 :ref:`intro-cpp-launching-kernels` 中所述，kernel 启动相对于调用它们的 CPU 线程是异步的。这意味着 CPU 线程的控制流将在 kernel 完成之前继续执行，甚至可能在 kernel 启动之前。为了保证 kernel 在主机代码继续之前已完成执行，需要某种同步机制。

同步 GPU 和主机线程的最简单方法是使用 ``cudaDeviceSynchronize`` ，它会阻塞主机线程直到 GPU 上所有先前发出的工作都已完成。在本章的示例中，这是足够的，因为 GPU 上只执行单个操作。在较大的应用程序中，可能有多个 :ref:`cuda-streams` 在 GPU 上执行工作， ``cudaDeviceSynchronize`` 将等待所有 stream 中的工作完成。在这些应用程序中，建议使用 :ref:`async-execution-stream-synchronization` API 仅与特定 stream 同步或使用 :ref:`cuda-events`。这些将在 :doc:`asynchronous-execution` 章节中详细介绍。

.. _intro-cuda-cpp-all-together:

2.1.5. 完整示例
---------------

以下清单显示了本章介绍的简单向量加法 kernel 的完整代码，以及所有主机代码和用于检查验证获得的答案是否正确的实用函数。这些示例默认使用 1024 的向量长度，但也接受可执行文件的命令行参数来指定不同的向量长度。

.. tab-set::

   .. tab-item:: 统一内存

      .. code-block:: cuda

         #include <cuda_runtime_api.h>
         #include <memory.h>
         #include <cstdlib>
         #include <ctime>
         #include <stdio.h>
         #include <cuda/cmath>

         __global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
         {
             int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
             if(workIndex < vectorLength)
             {
                 C[workIndex] = A[workIndex] + B[workIndex];
             }
         }

         void initArray(float* A, int length)
         {
              std::srand(std::time({}));
             for(int i=0; i<length; i++)
             {
                 A[i] = rand() / (float)RAND_MAX;
             }
         }

         void serialVecAdd(float* A, float* B, float* C,  int length)
         {
             for(int i=0; i<length; i++)
             {
                 C[i] = A[i] + B[i];
             }
         }

         bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
         {
             for(int i=0; i<length; i++)
             {
                 if(fabs(A[i] - B[i]) > epsilon)
                 {
                     printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
                     return false;
                 }
             }
             return true;
         }

         //unified-memory-begin
         void unifiedMemExample(int vectorLength)
         {
             // Pointers to memory vectors
             float* A = nullptr;
             float* B = nullptr;
             float* C = nullptr;
             float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

             // Use unified memory to allocate buffers
             cudaMallocManaged(&A, vectorLength*sizeof(float));
             cudaMallocManaged(&B, vectorLength*sizeof(float));
             cudaMallocManaged(&C, vectorLength*sizeof(float));

             // Initialize vectors on the host
             initArray(A, vectorLength);
             initArray(B, vectorLength);

             // Launch the kernel. Unified memory will make sure A, B, and C are
             // accessible to the GPU
             int threads = 256;
             int blocks = cuda::ceil_div(vectorLength, threads);
             vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
             // Wait for the kernel to complete execution
             cudaDeviceSynchronize();

             // Perform computation serially on CPU for comparison
             serialVecAdd(A, B, comparisonResult, vectorLength);

             // Confirm that CPU and GPU got the same answer
             if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
             {
                 printf("Unified Memory: CPU and GPU answers match\n");
             }
             else
             {
                 printf("Unified Memory: Error - CPU and GPU answers do not match\n");
             }

             // Clean Up
             cudaFree(A);
             cudaFree(B);
             cudaFree(C);
             free(comparisonResult);

         }
         //unified-memory-end


         int main(int argc, char** argv)
         {
             int vectorLength = 1024;
             if(argc >= 2)
             {
                 vectorLength = std::atoi(argv[1]);
             }
             unifiedMemExample(vectorLength);		
             return 0;
         }

   .. tab-item:: 显式内存管理

      .. code-block:: cuda

         #include <cuda_runtime_api.h>
         #include <memory.h>
         #include <cstdlib>
         #include <ctime>
         #include <stdio.h>
         #include <cuda/cmath>

         __global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
         {
             int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
             if(workIndex < vectorLength)
             {
                 C[workIndex] = A[workIndex] + B[workIndex];
             }
         }

         void initArray(float* A, int length)
         {
              std::srand(std::time({}));
             for(int i=0; i<length; i++)
             {
                 A[i] = rand() / (float)RAND_MAX;
             }
         }

         void serialVecAdd(float* A, float* B, float* C,  int length)
         {
             for(int i=0; i<length; i++)
             {
                 C[i] = A[i] + B[i];
             }
         }

         bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
         {
             for(int i=0; i<length; i++)
             {
                 if(fabs(A[i] - B[i]) > epsilon)
                 {
                     printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
                     return false;
                 }
             }
             return true;
         }

         //explicit-memory-begin
         void explicitMemExample(int vectorLength)
         {
             // Pointers for host memory
             float* A = nullptr;
             float* B = nullptr;
             float* C = nullptr;
             float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
             
             // Pointers for device memory
             float* devA = nullptr;
             float* devB = nullptr;
             float* devC = nullptr;

             //Allocate Host Memory using cudaMallocHost API. This is best practice
             // when buffers will be used for copies between CPU and GPU memory
             cudaMallocHost(&A, vectorLength*sizeof(float));
             cudaMallocHost(&B, vectorLength*sizeof(float));
             cudaMallocHost(&C, vectorLength*sizeof(float));

             // Initialize vectors on the host
             initArray(A, vectorLength);
             initArray(B, vectorLength);

             // start-allocate-and-copy
             // Allocate memory on the GPU
             cudaMalloc(&devA, vectorLength*sizeof(float));
             cudaMalloc(&devB, vectorLength*sizeof(float));
             cudaMalloc(&devC, vectorLength*sizeof(float));

             // Copy data to the GPU
             cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
             cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
             cudaMemset(devC, 0, vectorLength*sizeof(float));
             // end-allocate-and-copy

             // Launch the kernel
             int threads = 256;
             int blocks = cuda::ceil_div(vectorLength, threads);
             vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
             // wait for kernel execution to complete
             cudaDeviceSynchronize();

             // Copy results back to host
             cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);

             // Perform computation serially on CPU for comparison
             serialVecAdd(A, B, comparisonResult, vectorLength);

             // Confirm that CPU and GPU got the same answer
             if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
             {
                 printf("Explicit Memory: CPU and GPU answers match\n");
             }
             else
             {
                 printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
             }

             // clean up
             cudaFree(devA);
             cudaFree(devB);
             cudaFree(devC);
             cudaFreeHost(A);
             cudaFreeHost(B);
             cudaFreeHost(C);
             free(comparisonResult);
         }
         //explicit-memory-end


         int main(int argc, char** argv)
         {
             int vectorLength = 1024;
             if(argc >= 2)
             {
                 vectorLength = std::atoi(argv[1]);
             }
             explicitMemExample(vectorLength);		
             return 0;
         }

可以使用 ``nvcc`` 构建和运行这些示例，如下所示：

.. code-block:: bash

   $ nvcc vecAdd_unifiedMemory.cu -o vecAdd_unifiedMemory
   $ ./vecAdd_unifiedMemory
   Unified Memory: CPU and GPU answers match
   $ ./vecAdd_unifiedMemory 4096
   Unified Memory: CPU and GPU answers match

.. code-block:: bash

   $ nvcc vecAdd_explicitMemory.cu -o vecAdd_explicitMemory
   $ ./vecAdd_explicitMemory
   Explicit Memory: CPU and GPU answers match
   $ ./vecAdd_explicitMemory 4096
   Explicit Memory: CPU and GPU answers match

在这些示例中，所有线程都在做独立的工作，不需要相互协调或同步。通常，线程需要与其他线程合作和通信来完成它们的工作。Block 内的线程可以通过 :ref:`writing-cuda-kernels-shared-memory` 共享数据，并同步以协调内存访问。

Block 级别同步的最基本机制是 ``__syncthreads()`` 内建函数，它充当一个屏障，block 中的所有线程必须在此等待，然后才能继续执行任何线程。:ref:`writing-cuda-kernels-shared-memory` 给出了使用共享内存的示例。

为了高效协作，共享内存被期望是靠近每个处理器核心的低延迟内存（很像 L1 缓存），而 ``__syncthreads()`` 被期望是轻量级的。 ``__syncthreads()`` 仅同步单个 thread block 内的线程。CUDA 编程模型不支持 block 之间的同步。:ref:`cooperative-groups` 提供了设置单个 thread block 以外的同步域的机制。

当同步保持在 thread block 内时，通常可以获得最佳性能。Thread block 仍然可以使用 :ref:`writing-cuda-kernels-atomics` 处理共同的结果，这将在接下来的章节中介绍。

:numref:`Section 3.2.4` 介绍了 CUDA 同步原语，它们提供了非常细粒度的控制，以最大化性能和资源利用率。

.. _intro-cpp-runtime-initialization:

2.1.6. Runtime 初始化
----------------------

CUDA runtime 为系统中的每个设备创建一个 :ref:`driver-api-context`。此 context 是此设备的主要 context，在需要在此设备上激活 context 的第一个 runtime 函数时初始化。此 context 在应用程序的所有主机线程之间共享。作为 context 创建的一部分，设备代码会根据需要 :ref:`cuda-platform-just-in-time-compiled` 并加载到设备内存中。这一切都是透明发生的。CUDA runtime 创建的主要 context 可以从 driver API 访问以实现互操作性，如 :ref:`driver-api-interop-with-runtime` 中所述。

从 CUDA 12.0 开始， ``cudaInitDevice`` 和 ``cudaSetDevice`` 调用会初始化 runtime 以及与指定设备关联的主要 :ref:`driver-api-context`。如果在这些调用之前发生 runtime API 请求，runtime 将隐式使用设备 0 并根据需要自行初始化来处理这些请求。在对 runtime 函数调用进行计时以及解释第一次调用 runtime 的错误代码时，这很重要。在 CUDA 12.0 之前， ``cudaSetDevice`` 不会初始化 runtime。

``cudaDeviceReset`` 销毁当前设备的主要 context。如果在主要 context 被销毁后调用 CUDA runtime API，将为该设备创建一个新的主要 context。

.. note::

   CUDA 接口使用在主机程序启动期间初始化并在主机程序终止期间销毁的全局状态。在程序启动或终止期间（main 之后）使用任何这些接口（隐式或显式）将导致未定义的行为。

   从 CUDA 12.0 开始， ``cudaSetDevice`` 在更改主机线程的当前设备后，会显式初始化 runtime（如果尚未初始化）。以前版本的 CUDA 将新设备上的 runtime 初始化延迟到 ``cudaSetDevice`` 之后的第一个 runtime 调用。因此，检查 ``cudaSetDevice`` 的返回值以获取初始化错误非常重要。

   参考手册的错误处理和版本管理部分中的 runtime 函数不会初始化 runtime。

.. _intro-cpp-error-checking:

2.1.7. CUDA 中的错误检查
-------------------------

每个 CUDA API 返回一个枚举类型的值 ``cudaError_t`` 。在示例代码中，这些错误通常不被检查。在生产应用程序中，检查和管理每个 CUDA API 调用的返回值是最佳实践。当没有错误时，返回的值是 ``cudaSuccess`` 。许多应用程序选择实现一个实用宏，如下所示

.. code-block:: cuda

   #define CUDA_CHECK(expr_to_check) do {            \
       cudaError_t result  = expr_to_check;          \
       if(result != cudaSuccess)                     \
       {                                             \
           fprintf(stderr,                           \
                   "CUDA Runtime Error: %s:%i:%d = %s\n", \
                   __FILE__,                         \
                   __LINE__,                         \
                   result,\
                   cudaGetErrorString(result));      \
       }                                             \
   } while(0)

此宏使用 ``cudaGetErrorString`` API，该 API 返回描述特定 ``cudaError_t`` 值含义的可读字符串。使用上面的宏，应用程序将在 ``CUDA_CHECK(expression)`` 宏内调用 CUDA runtime API，如下所示：

.. code-block:: cuda

       CUDA_CHECK(cudaMalloc(&devA, vectorLength*sizeof(float)));
       CUDA_CHECK(cudaMalloc(&devB, vectorLength*sizeof(float)));
       CUDA_CHECK(cudaMalloc(&devC, vectorLength*sizeof(float)));

如果这些调用中的任何一个检测到错误，将使用此宏将其打印到 ``stderr`` 。此宏对于较小的项目很常见，但可以适应较大应用程序中的日志系统或其他错误处理机制。

.. note::

   请注意，任何 CUDA API 调用返回的错误状态也可能指示先前发出的异步操作的错误。:ref:`asynchronous-execution-error-handling` 更详细地介绍了这一点。

2.1.7.1. 错误状态
^^^^^^^^^^^^^^^^^

CUDA runtime 为每个主机线程维护一个 ``cudaError_t`` 状态。该值默认为 ``cudaSuccess`` ，并在发生错误时被覆盖。 ``cudaGetLastError`` 返回当前错误状态，然后将其重置为 ``cudaSuccess`` 。或者， ``cudaPeekLastError`` 返回错误状态而不重置它。

使用 :ref:`intro-cpp-launching-kernels-triple-chevron` 的 kernel 启动不返回 ``cudaError_t`` 。最佳做法是在 kernel 启动后立即检查错误状态，以检测 kernel 启动中的即时错误或 kernel 启动之前的 :ref:`intro-cpp-error-checking-asynchronous`。在 kernel 启动后立即检查错误状态时返回 ``cudaSuccess`` 值并不意味着 kernel 已成功执行甚至开始执行。它仅验证传递给 runtime 的 kernel 启动参数和执行配置没有触发任何错误，并且错误状态不是 kernel 启动之前的先前或异步错误。

.. _intro-cpp-error-checking-asynchronous:

2.1.7.2. 异步错误
^^^^^^^^^^^^^^^^^^^

CUDA kernel 启动和许多 runtime API 是异步的。异步 CUDA runtime API 将在 :doc:`asynchronous-execution` 中详细讨论。CUDA 错误状态在发生错误时被设置和覆盖。这意味着在异步操作执行期间发生的错误只会在下次检查错误状态时报告。如前所述，这可能是对 ``cudaGetLastError``、``cudaPeekLastError`` 的调用，也可能是任何返回 ``cudaError_t`` 的 CUDA API。

当 CUDA runtime API 函数返回错误时，错误状态不会被清除。这意味着来自异步错误（如 kernel 的无效内存访问）的错误代码将由每个 CUDA runtime API 返回，直到通过调用 ``cudaGetLastError`` 清除错误状态。

.. code-block:: cuda

       vecAdd<<<blocks, threads>>>(devA, devB, devC);
       // check error state after kernel launch
       CUDA_CHECK(cudaGetLastError());
       // wait for kernel execution to complete
       // The CUDA_CHECK will report errors that occurred during execution of the kernel
       CUDA_CHECK(cudaDeviceSynchronize());
       

.. note::

   ``cudaError_t`` 值 ``cudaErrorNotReady`` 可能由 ``cudaStreamQuery`` 和 ``cudaEventQuery`` 返回，不被视为错误，也不会由 ``cudaPeekAtLastError`` 或 ``cudaGetLastError`` 报告。

2.1.7.3. ``CUDA_LOG_FILE``
^^^^^^^^^^^^^^^^^^^^^^^^^^

识别 CUDA 错误的另一种好方法是使用 ``CUDA_LOG_FILE`` 环境变量。当设置此环境变量时，CUDA driver 将遇到的错误消息写入环境变量中指定路径的文件。例如，考虑以下错误的 CUDA 代码，它尝试启动一个大于任何架构支持的最大值的 thread block。

.. code-block:: cuda

   __global__ void k()
   { }

   int main()
   {
           k<<<8192, 4096>>>(); // Invalid block size
           CUDA_CHECK(cudaGetLastError());
           return 0;
   }

构建并运行此代码，kernel 启动后的检查使用 :numref:`Section 2.1.7` 中说明的宏检测并报告错误。

.. code-block:: bash

   $ nvcc errorLogIllustration.cu -o errlog
   $ ./errlog
   CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument

然而，当应用程序在设置了 ``CUDA_LOG_FILE`` 的情况下运行时，该文件包含有关错误的更多信息。

.. code-block:: bash

   $ env CUDA_LOG_FILE=cudaLog.txt ./errlog
   CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument
   $ cat cudaLog.txt
   [12:46:23.854][137216133754880][CUDA][E] One or more of block dimensions of (4096,1,1) exceeds corresponding maximum value of (1024,1024,64)
   [12:46:23.854][137216133754880][CUDA][E] Returning 1 (CUDA_ERROR_INVALID_VALUE) from cuLaunchKernel

将 ``CUDA_LOG_FILE`` 设置为 ``stdout`` 或 ``stderr`` 将分别打印到标准输出和标准错误。使用 ``CUDA_LOG_FILE`` 环境变量，即使应用程序没有对 CUDA 返回值实现正确的错误检查，也可以捕获和识别 CUDA 错误。这种方法对于调试非常强大，但仅靠环境变量不允许应用程序在运行时处理和恢复 CUDA 错误。CUDA 的 :ref:`error-log-management` 功能还允许向 driver 注册一个回调函数，该函数将在检测到错误时被调用。这可用于在运行时捕获和处理错误，也可以将 CUDA 错误日志无缝集成到应用程序现有的日志系统中。

:numref:`Section 4.8` 显示了 CUDA 错误日志管理功能的更多示例。错误日志管理和 ``CUDA_LOG_FILE`` 需要 NVIDIA Driver 版本 r570 及更高版本。

.. _intro-device-host-functions:

2.1.8. Device 和 Host 函数
----------------------------

``__global__`` 说明符用于指示 kernel 的入口点。也就是说，一个将在 GPU 上并行调用的函数。大多数情况下，kernel 从主机启动，但也可以使用 :ref:`cuda-dynamic-parallelism` 从另一个 kernel 内部启动 kernel。

说明符 ``__device__`` 指示函数应该为 GPU 编译，并且可以从其他 ``__device__`` 或 ``__global__`` 函数调用。函数，包括类成员函数、函数对象和 lambda，可以同时指定为 ``__device__`` 和 ``__host__`` ，如下例所示。

.. _intro-device-managed-variables:

2.1.9. 变量说明符
-----------------

:ref:`memory-space-specifiers` 可以用于静态变量声明以控制放置。

- ``__device__`` 指定变量存储在 :ref:`writing-cuda-kernels-global-memory` 中

- ``__constant__`` 指定变量存储在 :ref:`writing-cuda-kernels-constant-memory` 中

- ``__managed__`` 指定变量存储为 :ref:`memory-unified-memory`

- ``__shared__`` 指定变量存储在 :ref:`writing-cuda-kernels-shared-memory` 中

当在 ``__device__`` 或 ``__global__`` 函数内部声明没有说明符的变量时，它会在可能的情况下分配给寄存器，在必要时分配给 :ref:`writing-cuda-kernels-local-memory`。在 ``__device__`` 或 ``__global__`` 函数外部声明没有说明符的任何变量都将分配在系统内存中。

.. _intro-detecting-device-compilation:

2.1.9.1. 检测 Device 编译
^^^^^^^^^^^^^^^^^^^^^^^^^^

当函数使用 ``__host__`` ``__device__`` 指定时，编译器被指示为该函数生成 GPU 和 CPU 代码。在此类函数中，可能需要使用预处理器指定仅用于函数的 GPU 或 CPU 副本的代码。检查 ``__CUDA_ARCH_`` 是否已定义是执行此操作的最常用方法，如下例所示。

2.1.10. Thread Block Cluster
------------------------------

从计算能力 9.0 开始，CUDA 编程模型包含一个可选的层次结构级别，称为 thread block cluster，由 thread block 组成。类似于 thread block 中的线程保证在流式多处理器上共同调度，cluster 中的 thread block 也保证在 GPU 的 GPU 处理集群 (GPC) 上共同调度。

与 thread block 类似，cluster 也组织成一维、二维或三维的 thread block cluster grid，如 :numref:`Figure 5` 所示。

Cluster 中的 thread block 数量可以由用户定义，CUDA 支持的可移植 cluster 大小最多为 8 个 thread block。请注意，对于太小无法支持 8 个多处理器的 GPU 硬件或 MIG 配置，最大 cluster 大小将相应减少。识别这些较小的配置以及支持超过 8 个 thread block cluster 大小的较大配置是特定于架构的，可以使用 ``cudaOccupancyMaxPotentialClusterSize`` API 查询。

Cluster 中的所有 thread block 保证在单个 GPU 处理集群 (GPC) 上同时调度执行，并允许 cluster 中的 thread block 使用 :ref:`cooperative-groups` API ``cluster.sync()`` 执行硬件支持的同步。Cluster 组还提供成员函数，使用 ``num_threads()`` 和 ``num_blocks()`` API 分别以线程数或 block 数查询 cluster 组大小。可以使用 ``dim_threads()`` 和 ``dim_blocks()`` API 分别查询线程或 block 在 cluster 组中的排名。

属于 cluster 的 thread block 可以访问 *分布式共享内存*，即 cluster 中所有 thread block 的共享内存的组合。Cluster 中的 thread block 能够对分布式共享内存中的任何地址执行读取、写入和原子操作。:ref:`writing-cuda-kernels-distributed-shared-memory` 给出了在分布式共享内存中执行直方图的示例。

.. note::

   在使用 cluster 支持启动的 kernel 中，gridDim 变量仍然表示以 thread block 数为单位的 grid 大小，以保持兼容性。可以使用 :ref:`cooperative-groups` API 找到 block 在 cluster 中的排名。

.. _intro-cpp-launching-cluster-triple-chevron:

2.1.10.1. 使用三重尖括号表示法启动 Cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

可以使用 ``__cluster_dims__(X,Y,Z)`` 的编译时 kernel 属性或使用 CUDA kernel 启动 API ``cudaLaunchKernelEx`` 在 kernel 中启用 thread block cluster。下面的示例显示如何使用编译时 kernel 属性启动 cluster。使用 kernel 属性的 cluster 大小在编译时固定，然后可以使用经典的 ``<<< , >>>`` 启动 kernel。如果 kernel 使用编译时 cluster 大小，则在启动 kernel 时无法修改 cluster 大小。

.. code-block:: cuda

   // Kernel definition
   // Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
   __global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
   {

   }

   int main()
   {
       float *input, *output;
       // Kernel invocation with compile time cluster size
       dim3 threadsPerBlock(16, 16);
       dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

       // The grid dimension is not affected by cluster launch, and is still enumerated
       // using number of blocks.
       // The grid dimension must be a multiple of cluster size.
       cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
   }