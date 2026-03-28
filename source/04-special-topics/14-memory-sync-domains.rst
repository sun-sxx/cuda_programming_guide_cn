.. _memory-sync-domains:

内存同步域
====================

.. _memory-fence-interference:

内存栅栏干扰
---------------------------

某些 CUDA 应用程序可能会因为内存栅栏/刷新操作等待了超过 CUDA 内存一致性模型所需的事务而出现性能下降。

.. code-block:: c++

   __managed__ int x = 0;
   __device__  cuda::atomic<int, cuda::thread_scope_device> a(0);
   __managed__ cuda::atomic<int, cuda::thread_scope_system> b(0);

+----------------------------------+----------------------------------+----------------------------------+
| Thread 1 (SM)                    | Thread 2 (SM)                    | Thread 3 (CPU)                   |
+==================================+==================================+==================================+
| .. code-block:: c++              | .. code-block:: c++              | .. code-block:: c++              |
|                                  |                                  |                                  |
|    x = 1;                        |    while (a != 1) ;              |    while (b != 1) ;              |
|    a = 1;                        |    assert(x == 1);               |    assert(x == 1);               |
|                                  |    b = 1;                        |                                  |
+----------------------------------+----------------------------------+----------------------------------+

考虑上面的示例。CUDA 内存一致性模型保证断言条件将为真，因此线程 1 对 ``x`` 的写入必须在线程 2 对 ``b`` 的写入之前对线程 3 可见。

由 ``a`` 的释放和获取提供的内存排序仅足以使 ``x`` 对线程 2 可见，而不是线程 3，因为这是一个设备作用域操作。因此，由 ``b`` 的释放和获取提供的系统作用域排序不仅需要确保从线程 2 本身发出的写入对线程 3 可见，还需要确保来自其他线程且对线程 2 可见的写入也对线程 3 可见。这被称为累积性（cumulativity）。由于 GPU 在执行时无法知道哪些写入在源级别上被保证可见，哪些仅因时机巧合而可见，因此它必须对进行中的内存操作采取保守的广泛覆盖。

这有时会导致干扰：因为 GPU 正在等待它在源级别上不需要等待的内存操作，栅栏/刷新可能比必要的时间更长。

请注意，栅栏可能在代码中显式出现，如示例中的内联函数或原子操作，也可能是隐式的，用于在任务边界实现 \*同步关系\*（synchronizes-with）。

一个常见的例子是当一个内核在本地 GPU 内存中执行计算，而一个并行内核（例如来自 NCCL）正在与对等节点进行通信时。完成时，本地内核将隐式刷新其写入，以满足与下游工作的任何 \*同步关系\*。这可能会不必要地等待通信内核发出的较慢的 NVLink 或 PCIe 写入，无论是完全等待还是部分等待。

.. _isolating-traffic-with-domains:

使用域隔离流量
-----------------------------

从计算能力 9.0（Hopper 架构）GPU 和 CUDA 12.0 开始，内存同步域功能提供了一种缓解这种干扰的方法。作为代码显式协助的交换，GPU 可以减少栅栏操作的覆盖范围。每个内核启动都被分配一个域 ID。写入和栅栏都带有该 ID 标记，栅栏只会排序与栅栏域匹配的写入。在并发计算与通信的示例中，通信内核可以放置在不同的域中。

使用域时，代码必须遵守以下规则：**同一 GPU 上不同域之间的排序或同步需要系统作用域栅栏**。在域内，设备作用域栅栏仍然足够。这对于累积性是必要的，因为一个内核的写入不会被另一个域中的内核发出的栅栏所包含。本质上，累积性是通过确保跨域流量提前刷新到系统作用域来满足的。

请注意，这修改了 ``thread_scope_device`` 的定义。然而，由于内核将默认使用下面描述的域 0，因此保持了向后兼容性。

.. _using-domains-in-cuda:

在 CUDA 中使用域
-------------------------

域可以通过新的启动属性 ``cudaLaunchAttributeMemSyncDomain`` 和 ``cudaLaunchAttributeMemSyncDomainMap`` 访问。前者在逻辑域 ``cudaLaunchMemSyncDomainDefault`` 和 ``cudaLaunchMemSyncDomainRemote`` 之间进行选择，后者提供从逻辑域到物理域的映射。远程域旨在用于执行远程内存访问的内核，以便将其内存流量与本地内核隔离。但请注意，选择特定域并不影响内核可以合法执行的内存访问。

域数量可以通过设备属性 ``cudaDevAttrMemSyncDomainCount`` 查询。计算能力 9.0（Hopper）的设备有 4 个域。为了便于代码移植，域功能可以在所有设备上使用，CUDA 会在计算能力 9.0 之前的设备上报告数量为 1。

使用逻辑域可以简化应用程序组合。堆栈底层的单个内核启动（例如来自 NCCL）可以选择语义逻辑域，而无需考虑周围的应用程序架构。更高层级可以使用映射来引导逻辑域。如果未设置，逻辑域的默认值是默认域，默认映射是将默认域映射到 0，远程域映射到 1（在具有超过 1 个域的 GPU 上）。特定库可能会在 CUDA 12.0 及更高版本中使用远程域标记启动；例如，NCCL 2.16 将这样做。这共同为常见应用程序提供了开箱即用的有益使用模式，无需在其他组件、框架或应用程序级别进行代码更改。另一种使用模式，例如在使用 NVSHMEM 的应用程序中或没有明确内核类型分离的情况下，可以对并行流进行分区。流 A 可以将两个逻辑域都映射到物理域 0，流 B 映射到 1，以此类推。

.. code-block:: c++

   // Example of launching a kernel with the remote logical domain
   cudaLaunchAttribute domainAttr;
   domainAttr.id = cudaLaunchAttrMemSyncDomain;
   domainAttr.val = cudaLaunchMemSyncDomainRemote;
   cudaLaunchConfig_t config;
   // Fill out other config fields
   config.attrs = &domainAttr;
   config.numAttrs = 1;
   cudaLaunchKernelEx(&config, myKernel, kernelArg1, kernelArg2...);

.. code-block:: c++

   // Example of setting a mapping for a stream
   // (This mapping is the default for streams starting on compute capability 9.0 (Hopper) or later if not
   // explicitly set, and provided for illustration)
   cudaLaunchAttributeValue mapAttr;
   mapAttr.memSyncDomainMap.default_ = 0;
   mapAttr.memSyncDomainMap.remote = 1;
   cudaStreamSetAttribute(stream, cudaLaunchAttributeMemSyncDomainMap, &mapAttr);

.. code-block:: c++

   // Example of mapping different streams to different physical domains, ignoring
   // logical domain settings
   cudaLaunchAttributeValue mapAttr;
   mapAttr.memSyncDomainMap.default_ = 0;
   mapAttr.memSyncDomainMap.remote = 0;
   cudaStreamSetAttribute(streamA, cudaLaunchAttributeMemSyncDomainMap, &mapAttr);
   mapAttr.memSyncDomainMap.default_ = 1;
   mapAttr.memSyncDomainMap.remote = 1;
   cudaStreamSetAttribute(streamB, cudaLaunchAttributeMemSyncDomainMap, &mapAttr);

与其他启动属性一样，这些属性在 CUDA 流、使用 ``cudaLaunchKernelEx`` 的单个启动以及 CUDA 图中的内核节点上统一公开。典型的用法是在流级别设置映射，在启动级别设置逻辑域（或包围一段流使用区域），如上所述。

两个属性在流捕获期间都会被复制到图节点。图从节点本身获取两个属性，这本质上是指定物理域的间接方式。在图启动到的流上设置的与域相关的属性不会用于图的执行。
