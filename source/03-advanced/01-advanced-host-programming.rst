.. _advanced-host-programming:

3.1. 高级 CUDA API 和特性
==========================

本节将介绍更高级的 CUDA API 和特性的使用。
这些主题涵盖的技术或特性通常不需要修改 CUDA kernel，但仍可以从主机端影响应用级行为，包括 GPU 工作执行、性能以及 CPU 端性能。

.. _cudalaunchkernelex:

3.1.1. cudaLaunchKernelEx
--------------------------

当 CUDA 早期版本引入 :ref:`三重尖括号表示法 <intro-cpp-launching-kernels-triple-chevron>` 时，kernel 的 :ref:`执行配置 <kernel-configuration>` 只有四个可编程参数：

- 线程块维度
- grid 维度
- 动态共享内存（可选，未指定时为 0）
- 流（未指定时使用默认流）

一些 CUDA 特性可以从 kernel 启动时提供的额外属性和提示中受益。 
``cudaLaunchKernelEx`` 允许程序通过 ``cudaLaunchConfig_t`` 结构体设置上述执行配置参数。
此外， ``cudaLaunchConfig_t`` 结构体还允许程序传入零个或多个 ``cudaLaunchAttributes`` 来控制或建议 kernel 启动的其他参数。
例如，本章后面讨论的 ``cudaLaunchAttributePreferredSharedMemoryCarveout`` （参见 :ref:`配置 L1/共享内存平衡 <configuring-l1-shared-memory-balance>` ）
就是使用 ``cudaLaunchKernelEx`` 指定的。
本章后面讨论的 ``cudaLaunchAttributeClusterDimension`` 属性用于指定 kernel 启动所需的 cluster 大小。

支持的属性完整列表及其含义请参阅 `CUDA Runtime API 参考文档 <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gfc5ed48085f05863b1aeebb14934b056>`_ 。

.. _launching-clusters:

3.1.2. 启动 Cluster
--------------------

:ref:`线程块 cluster <thread-block-clusters>` 在前面的章节中介绍过，
是计算能力 9.0 及更高版本中提供的可选的线程块组织级别，它使应用程序能够保证 cluster 中的线程块在单个 GPC 上同时执行。
这使得比单个 SM 容量更大的线程组能够交换数据并相互同步。

:ref:`intro-cpp-launching-cluster-triple-chevron` 展示了如何使用三重尖括号表示法指定和启动使用 cluster 的 kernel。
在该节中，使用 ``__cluster_dims__`` 注解来指定用于启动 kernel 的 cluster 维度。使用三重尖括号表示法时，cluster 的大小是隐式确定的。

.. _launching-with-clusters-using-cudalaunchkernelex:

3.1.2.1. 使用 cudaLaunchKernelEx 启动 Cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

与 :ref:`使用三重尖括号表示法启动 cluster kernel <intro-cpp-launching-cluster-triple-chevron>` 不同，线程块 cluster 的大小可以在每次启动时配置。
下面的代码示例展示了如何使用 ``cudaLaunchKernelEx`` 启动 cluster kernel。

.. code-block:: cuda
   :caption: 使用 cudaLaunchKernelEx 启动 cluster kernel

   // Kernel definition
   // No compile time attribute attached to the kernel
   __global__ void cluster_kernel(float *input, float* output)
   {

   }

   int main()
   {
       float *input, *output;
       dim3 threadsPerBlock(16, 16);
       dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

       // Kernel invocation with runtime cluster size
       {
           cudaLaunchConfig_t config = {0};
           // The grid dimension is not affected by cluster launch, and is still enumerated
           // using number of blocks.
           // The grid dimension should be a multiple of cluster size.
           config.gridDim = numBlocks;
           config.blockDim = threadsPerBlock;

           cudaLaunchAttribute attribute[1];
           attribute[0].id = cudaLaunchAttributeClusterDimension;
           attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
           attribute[0].val.clusterDim.y = 1;
           attribute[0].val.clusterDim.z = 1;
           config.attrs = attribute;
           config.numAttrs = 1;

           cudaLaunchKernelEx(&config, cluster_kernel, input, output);
       }
   }

有两种与线程块 cluster 相关的 ``cudaLaunchAttribute`` 类型： ``cudaLaunchAttributeClusterDimension`` 和 ``cudaLaunchAttributePreferredClusterDimension`` 。

属性 ID ``cudaLaunchAttributeClusterDimension`` 指定执行 cluster 所需的维度。此属性的值 ``clusterDim`` 是一个三维值。
grid 的相应维度（x、y 和 z）必须能被指定 cluster 维度的相应维度整除。
设置此属性类似于在编译时在 kernel 定义上使用 ``__cluster_dims__`` 属性，
如 :ref:`使用三重尖括号表示法启动 cluster <intro-cpp-launching-cluster-triple-chevron>` 中所示，
但可以在运行时为同一个 kernel 的不同启动更改此设置。

在计算能力为 10.0 及更高的 GPU 上，另一个属性 ID ``cudaLaunchAttributePreferredClusterDimension`` 允许应用程序额外指定 cluster 的首选维度。
首选维度必须是 kernel 上的 ``__cluster_dims__`` 属性或 ``cudaLaunchKernelEx`` 的 ``cudaLaunchAttributeClusterDimension`` 属性指定的最小 cluster 维度的整数倍。
也就是说，除了首选 cluster 维度外，还必须指定最小 cluster 维度。grid 的相应维度（x、y 和 z）必须能被指定首选 cluster 维度的相应维度整除。

所有线程块都将在至少为最小 cluster 维度的 cluster 中执行。
在可能的情况下，将使用首选维度的 cluster，但不保证所有 cluster 都以首选维度执行。
所有线程块都将在最小或首选 cluster 维度的 cluster 中执行。
使用首选 cluster 维度的 kernel 必须编写为能在最小或首选 cluster 维度下正确运行。

.. _blocks-as-clusters:

3.1.2.2. 作为 Cluster 的 Block
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当使用 ``__cluster_dims__`` 注解定义 kernel 时，grid 中 cluster 的数量是隐式的，可以从 grid 大小除以指定的 cluster 大小计算得出。

.. code-block:: cuda

   __cluster_dims__((2, 2, 2)) __global__ void foo();

   // 8x8x8 clusters each with 2x2x2 thread blocks.
   foo<<<dim3(16, 16, 16), dim3(1024, 1, 1)>>>();

在上面的示例中，kernel 作为 16x16x16 线程块的 grid 启动，这意味着使用 8x8x8 cluster 的 grid。

kernel 也可以使用 ``__block_size__`` 注解，它在定义 kernel 时同时指定所需的 block 大小和 cluster 大小。
使用此注解时，三重尖括号启动变为以 cluster 而非线程块为单位的 grid 维度，如下所示。

.. code-block:: cuda

   // Implementation detail of how many threads per block and blocks per cluster
   // is handled as an attribute of the kernel.
   __block_size__((1024, 1, 1), (2, 2, 2)) __global__ void foo();

   // 8x8x8 clusters.
   foo<<<dim3(8, 8, 8)>>>();

``__block_size__`` 需要两个字段，每个字段都是包含 3 个元素的元组。第一个元组表示 block 维度，第二个表示 cluster 大小。
如果未传递第二个元组，则假设为 ``(1,1,1)`` 。
要指定流，必须在 ``<<<>>>`` 中传递 ``1`` 和 ``0`` 作为第二个和第三个参数，最后是流。
传递其他值将导致未定义行为。

请注意，同时指定 ``__block_size__`` 的第二个元组和 ``__cluster_dims__`` 是非法的。
在空的 ``__cluster_dims__`` 情况下使用 ``__block_size__`` 也是非法的。
当指定了 ``__block_size__`` 的第二个元组时，意味着启用了"Blocks as Clusters"模式，编译器会将 ``<<<>>>`` 内的第一个参数识别为 cluster 数量而非线程块数量。

.. _more-on-streams-and-events:

3.1.3. 关于流和事件的更多内容
------------------------------

:ref:`cuda-streams` 介绍了 CUDA 流的基础知识。
默认情况下，在给定 CUDA 流上提交的操作是串行化的：一个操作在前一个操作完成之前无法开始执行。
唯一的例外是最近添加的 :ref:`programmatic-dependent-launch-details` 特性。
拥有多个 CUDA 流是实现并发执行的一种方式；另一种方式是使用 :ref:`cuda-graphs-details`。
这两种方法也可以结合使用。

在特定情况下，不同 CUDA 流上提交的工作可能会并发执行，例如，如果没有事件依赖关系，没有隐式同步，有足够的资源等。

如果在不同的 CUDA 流中的独立操作之间提交了任何对 NULL 流的 CUDA 操作，这些独立操作就无法并发运行，除非这些流是非阻塞 CUDA 流。
这些是使用带有 ``cudaStreamNonBlocking`` 标志的 ``cudaStreamCreateWithFlags()`` runtime API 创建的流。
为了提高 GPU 工作并发执行的潜力，建议用户创建非阻塞 CUDA 流。

还建议用户选择足以满足其问题的最小范围的同步选项。
例如，如果要求 CPU 等待（阻塞）特定 CUDA 流上的所有工作完成，那么对该流使用 ``cudaStreamSynchronize()`` 比 ``cudaDeviceSynchronize()`` 更可取，
因为后者会不必要地等待设备上所有 CUDA 流的 GPU 工作完成。
如果要求 CPU 在不阻塞的情况下等待，那么使用 ``cudaStreamQuery()`` 并在轮询循环中检查其返回值可能更可取。

使用 :ref:`CUDA 事件 <cuda-events>` 也可以实现类似的同步效果。
例如，通过在该流上记录一个事件并调用 ``cudaEventSynchronize()`` 以阻塞方式等待该事件捕获的工作完成。
同样，这比使用 ``cudaDeviceSynchronize()`` 更可取、更集中。调用 ``cudaEventQuery()`` 并检查其返回值（例如在轮询循环中）将是一种非阻塞的替代方案。

如果此操作发生在应用程序的关键路径中，选择显式同步方法就特别重要。:numref:`table-streams-event-sync-summary` 提供了与主机同步的各种选项的高级摘要。

.. _table-streams-event-sync-summary:
.. list-table:: 与主机同步选项摘要
   :header-rows: 1
   :stub-columns: 1
   
   * - 
     - 等待特定流
     - 等待特定事件
     - 等待设备上的所有内容
   * - 非阻塞（轮询）
     - ``cudaStreamQuery()``
     - ``cudaEventQuery()``
     - N/A
   * - 阻塞
     - ``cudaStreamSynchronize()``
     - ``cudaEventSynchronize()``
     - ``cudaDeviceSynchronize()``

对于 CUDA 流之间的同步（即表达依赖关系），建议使用非计时 CUDA 事件，如 :ref:`CUDA 事件 <cuda-events>` 中所述。
用户可以调用 ``cudaStreamWaitEvent()`` 强制特定流上未来提交的操作等待先前记录的事件（例如，在另一个流上）完成。
请注意，对于任何等待或查询事件的 CUDA API，用户有责任确保已调用 ``cudaEventRecord`` API，因为未记录的事件将始终返回成功。

CUDA 事件默认携带计时信息，因为它们可以在 ``cudaEventElapsedTime()`` API 调用中使用。
然而，仅用于表达流间依赖关系的 CUDA 事件不需要计时信息。
对于这种情况，建议创建禁用计时信息的事件以提高性能。
这可以通过使用带有 ``cudaEventDisableTiming`` 标志的 ``cudaEventCreateWithFlags()`` API 来实现。

.. _stream-priorities:

3.1.3.1. 流优先级
^^^^^^^^^^^^^^^^^

可以使用 ``cudaStreamCreateWithPriority()`` 在创建时指定流的相对优先级。
可以使用 ``cudaDeviceGetStreamPriorityRange()`` 函数获取允许的优先级范围，按 [最高优先级，最低优先级] 排序。
在运行时，GPU 调度器利用流优先级确定任务执行顺序，但这些优先级作为提示而非保证。
当选择要启动的工作时，较高优先级流中的待处理任务优先于较低优先级流中的任务。较高优先级的任务不会抢占已运行的较低优先级任务。
GPU 在任务执行期间不会重新评估工作队列，增加流的优先级不会中断正在进行的工作。
流优先级影响任务执行而不强制严格排序，因此用户可以利用流优先级影响任务执行，而不依赖严格的排序保证。

以下代码示例获取当前设备允许的优先级范围，并创建具有最高和最低可用优先级的两个非阻塞 CUDA 流。

.. code-block:: cuda
   :caption: 流优先级示例

   // get the range of stream priorities for this device
   int leastPriority, greatestPriority;
   cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

   // create streams with highest and lowest available priorities
   cudaStream_t st_high, st_low;
   cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, greatestPriority));
   cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, leastPriority);

.. _explicit-synchronization:

3.1.3.2. 显式同步
^^^^^^^^^^^^^^^^^

如前所述，有多种方法可以实现流与其他流的同步。以下是不同粒度级别的常用方法：

- ``cudaDeviceSynchronize()`` 等待所有主机线程的所有流中的所有先前命令完成。
- ``cudaStreamSynchronize()`` 接受一个流作为参数，等待给定流中的所有先前命令完成。
  它可用于将主机与特定流同步，允许其他流在设备上继续执行。
- ``cudaStreamWaitEvent()`` 接受一个流和一个事件作为参数（参见 :ref:`CUDA 事件 <cuda-events>` 了解事件的描述），
  使调用 ``cudaStreamWaitEvent()`` 后添加到给定流的所有命令延迟执行，直到给定事件完成。
- ``cudaStreamQuery()`` 为应用程序提供一种了解流中所有先前命令是否已完成的方法。

.. _implicit-synchronization:

3.1.3.3. 隐式同步
^^^^^^^^^^^^^^^^^

如果主机线程在两个不同流的命令之间执行以下任一操作，则这两个命令无法并发运行：

- 页锁定主机内存分配
- 设备内存分配
- 设备内存设置
- 同一设备内存的两个地址之间的内存复制
- 对 NULL 流的任何 CUDA 命令
- L1/共享内存配置之间的切换

需要依赖检查的操作包括与被检查的启动在同一流中的任何其他命令，以及对该流的任何 ``cudaStreamQuery()`` 调用。
因此，应用程序应遵循以下准则以提高 kernel 并发执行的潜力：

- 所有独立操作应在依赖操作之前发出，
- 任何类型的同步应尽可能延迟。

.. _programmatic-dependent-kernel-launch:

3.1.4. Programmatic Dependent Kernel Launch
----------------------------------------------

正如我们前面讨论的，CUDA 流的语义是 kernel 按顺序执行。
这样，如果我们有两个连续的 kernel，其中第二个 kernel 依赖于第一个 kernel 的结果，程序员可以确信当第二个 kernel 开始执行时，依赖数据将可用。
然而，可能存在这样的情况：第一个 kernel 已经将后续 kernel 依赖的数据写入全局内存，但它还有更多工作要做。
同样，依赖的第二个 kernel 在需要第一个 kernel 的数据之前可能有一些独立的工作。
在这种情况下，可以部分重叠两个 kernel 的执行（假设硬件资源可用）。
这种重叠也可以重叠第二个 kernel 的启动开销。除了硬件资源的可用性，可实现的重叠程度取决于 kernel 的具体结构，例如：

- 第一个 kernel 在其执行的什么时候完成第二个 kernel 依赖的工作？
- 第二个 kernel 在其执行的什么时候开始处理来自第一个 kernel 的数据？

由于这在很大程度上取决于具体的 kernel，因此很难完全自动化，因此 CUDA 提供了一种机制，允许应用程序开发者指定两个 kernel 之间的同步点。
这通过一种称为程序化依赖 Kernel 启动（ Programmatic Dependent Kernel Launch, PDL ）的技术完成。下图描述了这种情况。

.. _fig-pdl:
.. figure:: /_static/images/pdl.png
   :alt: 程序化依赖 Kernel 启动
   :align: center

   程序化依赖 Kernel 启动

PDL 有三个主要组件：

1. 第一个 kernel（所谓的 *主 kernel*）需要调用一个特殊函数来指示它已完成后续依赖 kernel（也称为 *从 kernel*）所需的所有工作。
   这通过调用函数 ``cudaTriggerProgrammaticLaunchCompletion()`` 完成。

2. 相应地，依赖的从 kernel 需要指示它已到达其工作中独立于主 kernel 的部分，并且现在正在等待主 kernel 完成它依赖的工作。
   这通过函数 ``cudaGridDependencySynchronize()`` 完成。

3. 第二个 kernel 需要使用特殊属性 ``cudaLaunchAttributeProgrammaticStreamSerialization`` 启动，其 ``programmaticStreamSerializationAllowed`` 字段设置为 `1` 。

以下代码片段展示了如何实现这一点。

.. _lst:pdl-example:
.. code-block:: cuda
   :caption: 程序化依赖 Kernel 启动示例（两个 Kernel）
   :linenos:

   __global__ void primary_kernel() {
       // Initial work that should finish before starting secondary kernel

       // Trigger the secondary kernel
       cudaTriggerProgrammaticLaunchCompletion();

       // Work that can coincide with the secondary kernel
   }

   __global__ void secondary_kernel()
   {
       // Initialization, Independent work, etc.

       // Will block until all primary kernels the secondary kernel is dependent on have
       // completed and flushed results to global memory
       cudaGridDependencySynchronize();

       // Dependent work
   }

   // Launch the secondary kernel with the special attribute

   // Set Up the attribute
   cudaLaunchAttribute attribute[1];
   attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
   attribute[0].val.programmaticStreamSerializationAllowed = 1;

   // Set the attribute in a kernel launch configuration
    cudaLaunchConfig_t config = {0};

   // Base launch configuration
   config.gridDim = grid_dim;
   config.blockDim = block_dim;
   config.dynamicSmemBytes= 0;
   config.stream = stream;

   // Add special attribute for PDL
   config.attrs = attribute;
   config.numAttrs = 1;

   // Launch primary kernel
   primary_kernel<<<grid_dim, block_dim, 0, stream>>>();

   // Launch secondary (dependent) kernel using the configuration with
   // the attribute
   cudaLaunchKernelEx(&config, secondary_kernel);

.. _batched-memory-transfers:

3.1.5. 批量内存传输
-------------------

CUDA 开发中的一个常见模式是使用批处理技术。批处理是指我们将多个（通常是小的）任务组合成一个（通常是大的）操作。
批处理的组件不一定都必须相同，尽管它们通常相同。这种思想的一个例子是 cuBLAS 提供的批量矩阵乘法操作。

通常，与 CUDA Graphs 和 PDL 一样，批处理的目的是减少单独调度各个批处理任务的开销。
在内存传输方面，启动内存传输可能会产生一些 CPU 和驱动程序开销。此外，当前形式的常规 ``cudaMemcpyAsync()`` 函数不一定为驱动程序提供足够的信息来优化传输，
例如，关于源和目标的提示。在 Tegra 平台上，可以选择使用 SM 或复制引擎（ Copy Engines, CE ）来执行传输。
目前选择哪一个由驱动程序中的启发式方法指定。
这可能很重要，因为使用 SM 可能会导致更快的传输，但会占用一些可用的计算能力。
另一方面，使用 CE 可能会导致较慢的传输，但整体应用程序性能更高，因为它让 SM 可以自由执行其他工作。

这些考虑促成了 ``cudaMemcpyBatchAsync()`` 函数（及其相关函数 ``cudaMemcpyBatch3DAsync()`` ）的设计。
这些函数允许优化批量内存传输。除了源和目标指针列表外，API 还使用内存复制属性来指定排序期望，以及源和目标位置的提示，以及是否希望将传输与计算重叠（目前仅在带有 CE 的 Tegra 平台上支持）。

让我们首先考虑从页锁定主机内存到页锁定设备内存的简单批量传输的最简单情况。

.. _lst:batched-memory-transfer-homogeneous:
.. code-block:: cuda
   :caption: 从页锁定主机内存到页锁定设备内存的同构批量内存传输示例
   :linenos:

   std::vector<void *> srcs(batch_size);
   std::vector<void *> dsts(batch_size);
   std::vector<void *> sizes(batch_size);

   // Allocate the source and destination buffers
   // initialize with the stream number
   for (size_t i = 0; i < batch_size; i++) {
       cudaMallocHost(&srcs[i], sizes[i]);
       cudaMalloc(&dsts[i], sizes[i]);
       cudaMemsetAsync(srcs[i], sizes[i], stream);
   }

   // Setup attributes for this batch of copies
   cudaMemcpyAttributes attrs = {};
   attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;

   // All copies in the batch have same copy attributes.
   size_t attrsIdxs = 0;  // Index of the attributes

   // Launch the batched memory transfer
   cudaMemcpyBatchAsync(&dsts[0], &srcs[0], &sizes[0], batch_size,
       &attrs, &attrsIdxs, 1 /*numAttrs*/, nullptr /*failIdx*/, stream);

``cudaMemcpyBatchAsync()`` 函数的前几个参数看起来很直观。它们由包含源指针、目标指针以及传输大小的数组组成。
每个数组必须有 ``batch_size`` 个元素。新信息来自属性。该函数需要一个指向属性数组的指针，以及相应的属性索引数组。
原则上，也可以传递一个 ``size_t`` 数组，在这个数组中可以记录失败传输的索引，但是在这里传递 ``nullptr`` 是安全的，在这种情况下，失败的索引将不会被记录。

关于属性，在这个例子中传输是同构的。所以我们只使用一个属性，它将应用于所有传输。
这由 attrIndex 参数控制。原则上，这可以是一个数组。
数组的第 *i* 个元素包含属性数组的第 *i* 个元素适用的第一个传输的索引。
在这种情况下，attrIndex 被视为单个元素数组，值为 '0' 意味着 ``attribute[0]`` 将应用于索引 0 及以上的所有传输，换句话说，所有传输。

最后，我们注意到我们将 ``srcAccessOrder`` 属性设置为 ``cudaMemcpySrcAccessOrderStream`` 。
这意味着源数据将按常规流顺序访问。换句话说，memcpy 将阻塞直到处理来自这些源指针和目标指针中任何一个的数据的先前 kernel 完成。

在下一个示例中，我们将考虑异构批量传输的更复杂情况。

.. _lst:batched-memory-transfer-heterogeneous:
.. code-block:: cuda
   :caption: 使用一些临时主机内存到页锁定设备内存的异构批量内存传输示例
   :linenos:

   std::vector<void *> srcs(batch_size);
   std::vector<void *> dsts(batch_size);
   std::vector<void *> sizes(batch_size);

   // Allocate the src and dst buffers
   for (size_t i = 0; i < batch_size - 10; i++) {
       cudaMallocHost(&srcs[i], sizes[i]);
       cudaMalloc(&dsts[i], sizes[i]);
   }

   int buffer[10];

   for (size_t i = batch_size - 10; i < batch_size; i++) {
       srcs[i] = &buffer[10 - (batch_size - i];
       cudaMalloc(&dsts[i], sizes[i]);
   }

   // Setup attributes for this batch of copies
   cudaMemcpyAttributes attrs[2] = {};
   attrs[0].srcAccessOrder = cudaMemcpySrcAccessOrderStream;
   attrs[1].srcAccessOrder = cudaMemcpySrcAccessOrderDuringApiCall;

   size_t attrsIdxs[2];
   attrsIdxs[0] = 0;
   attrsIdxs[1] = batch_size - 10;

   // Launch the batched memory transfer
   cudaMemcpyBatchAsync(&dsts[0], &srcs[0], &sizes[0], batch_size,
       &attrs, &attrsIdxs, 2 /*numAttrs*/, nullptr /*failIdx*/, stream);

这里我们有两种传输： ``batch_size-10`` 次从页锁定主机内存到页锁定设备内存的传输，以及 10 次从主机数组到页锁定设备内存的传输。
此外，buffer 数组不仅在主机上，而且只存在于当前作用域中——它的地址是所谓的 *临时指针* 。
此指针在 API 调用完成后可能无效（它是异步的）。
要使用此类临时指针执行复制，属性中的 ``srcAccessOrder`` 必须设置为 ``cudaMemcpySrcAccessOrderDuringApiCall`` 。

我们现在有两个属性，第一个应用于索引从 0 开始且小于 ``batch_size-10`` 的所有传输。
第二个应用于索引从 ``batch_size-10`` 开始且小于 ``batch_size`` 的所有传输。

如果我们不是从栈上分配 buffer 数组，而是使用 malloc 从堆上分配它，数据就不再是临时的。它将有效直到指针被显式释放。
在这种情况下，如何暂存复制的最佳选择取决于系统是否有硬件管理内存或通过地址转换对主机内存的连贯 GPU 访问，这种情况下最好使用流排序，或者如果没有，那么立即暂存传输最有意义。
在这种情况下，应该为属性的 ``srcAccessOrder`` 使用值 ``cudaMemcpyAccessOrderAny`` 。

``cudaMemcpyBatchAsync`` 函数还允许程序员提供关于源和目标位置的提示。
这通过设置 ``cudaMemcpyAttributes`` 结构体的 ``srcLocation`` 和 ``dstLocation`` 字段来完成。
``srcLocation`` 和 ``dstLocation`` 字段都是 ``cudaMemLocation`` 类型，这是一个包含位置类型和位置 ID 的结构体。
这与在使用 ``cudaMemPrefetchAsync()`` 时可用于向运行时提供预取提示的 ``cudaMemLocation`` 结构体相同。
我们在下面的代码示例中说明如何为从设备到主机特定 NUMA 节点的传输设置提示：

.. _lst:batched-memory-location-hints:
.. code-block:: cuda
   :caption: 设置源和目标位置提示示例
   :linenos:

   // Allocate the source and destination buffers
   std::vector<void *> srcs(batch_size);
   std::vector<void *> dsts(batch_size);
   std::vector<void *> sizes(batch_size);

   // cudaMemLocation structures we will use tp provide location hints
   // Device device_id
   cudaMemLocation srcLoc = {cudaMemLocationTypeDevice, dev_id};

   // Host with numa Node numa_id
   cudaMemLocation dstLoc = {cudaMemLocationTypeHostNuma, numa_id};

   // Allocate the src and dst buffers
   for (size_t i = 0; i < batch_size; i++) {
       cudaMallocManaged(&srcs[i], sizes[i]);
       cudaMallocManaged(&dsts[i], sizes[i]);

       cudaMemPrefetchAsync(srcs[i], sizes[i], srcLoc, 0, stream);
       cudaMemPrefetchAsync(dsts[i], sizes[i], dstLoc, 0, stream);
       cudaMemsetAsync(srcs[i], sizes[i], stream);
   }

   // Setup attributes for this batch of copies
   cudaMemcpyAttributes attrs = {};

   // These are managed memory pointers so Stream Order is appropriate
   attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;

   // Now we can specify the location hints here.
   attrs.srcLocHint = srcLoc;
   attrs.dstlocHint = dstLoc;

   // All copies in the batch have same copy attributes.
   size_t attrsIdxs = 0;

   // Launch the batched memory transfer
   cudaMemcpyBatchAsync(&dsts[0], &srcs[0], &sizes[0], batch_size,
       &attrs, &attrsIdxs, 1 /*numAttrs*/, nullptr /*failIdx*/, stream);

最后要介绍的是提示我们是否希望使用 SM 还是 CE 进行传输的标志。此字段是 ``cudaMemcpyAttributesflags::flags`` ，可能的值是：

- ``cudaMemcpyFlagDefault`` – 默认行为
- ``cudaMemcpyFlagPreferOverlapWithCompute`` – 这提示系统应该优先使用 CE 进行传输，将传输与计算重叠。
  此标志在非 Tegra 平台上被忽略

总之，关于 ``cudaMemcpyBatchAsync`` 的要点如下：

- ``cudaMemcpyBatchAsync`` 函数（及其 3D 变体）允许程序员指定一批内存传输，从而分摊传输设置开销。
- 除了源和目标指针以及传输大小外，该函数还可以接受一个或多个内存复制属性，提供关于正在传输的内存类型的信息、源指针的相应流排序行为、关于源和目标位置的提示，
  以及关于是否优先将传输与计算重叠（如果可能）或是否使用 SM 进行传输的提示。
- 基于上述信息，运行时可以尝试最大限度地优化传输。

.. _environment-variables:

3.1.6. 环境变量
----------------

CUDA 提供了各种环境变量（参见 :ref:`第 5.2 节 <environment-variables-details>` ），这些变量可以影响执行和性能。
如果未显式设置它们，CUDA 会为它们使用合理的默认值，但在特定情况下可能需要特殊处理，例如，用于调试目的或获得改进的性能。

例如，增加环境变量 ``CUDA_DEVICE_MAX_CONNECTIONS`` 的值可能是必要的，可以减少不同 CUDA 流上的独立任务由于假依赖而被串行化的可能性。
当使用相同的底层资源时，可能会引入此类虚假依赖。
建议从使用默认值开始，当出现性能问题时才探索此环境变量的影响。 例如，无法归因于其他因素（如缺少可用的 SM 资源） 的不同 CUDA 流之间独立工作的意外串行化。
值得注意的是，在 MPS 情况下，此环境变量具有不同的（较低的）默认值。

类似地，对于延迟敏感的应用程序，将 ``CUDA_MODULE_LOADING`` 环境变量设置为 ``EAGER`` 可能更可取，
以便将模块加载的所有开销移到应用程序初始化阶段，而不是其关键阶段。
当前的默认模式是延迟模块加载。
在此默认模式下，可以通过在应用程序初始化阶段添加各种 kernel 的"预热"调用来实现类似于立即模块加载的效果，以强制模块加载更早发生。

有关各种 CUDA 环境变量的更多信息，请参阅 :ref:`CUDA 环境变量 <environment-variables-details>`。
请在启动应用程序 **之前** 设置环境变量；尝试在应用程序内设置它们可能无效。