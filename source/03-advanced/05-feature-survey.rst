.. _tour-of-features:

3.5. CUDA 功能概览
==================

本编程指南的第 1-3 章介绍了 CUDA 与 GPU 编程，涵盖了基础概念并辅以简单的代码示例。
本指南第 4 部分在介绍具体的 CUDA 特性时，假定读者已经掌握了前 1-3 章所涵盖的概念。

CUDA 包含众多特性，用于解决不同类型的问题，但并非所有特性都适用于每一种应用场景。
本章旨在逐一介绍这些特性，阐述其用途以及期望解决的问题。
根据问题类型，我们对这些特性进行了粗略的分类。
需要注意的是，某些特性（如 CUDA Graphs）可能会同时归属于多个类别。

我们将在 :ref:`part4` 更详细地介绍这些 CUDA 功能。

.. _improving-kernel-performance:

3.5.1. 提升 Kernel 性能
------------------------

本节概述的各项特性，旨在帮助开发者最大限度地提升 kernel 性能。

.. _feature-survey-asynchronous-barriers:

3.5.1.1. 异步屏障
^^^^^^^^^^^^^^^^^

异步屏障（Asynchronous barriers）在 :ref:`asynchronous-barriers` 中有所介绍，它允许对线程间的同步进行更细粒度的控制。
异步屏障将屏障的到达（arrival）与等待（wait）这两个阶段分离开来。
这使得应用程序在等待其他线程到达的同时，能够继续执行不依赖于该屏障的其他工作。
此外，还可以针对不同的 :ref:`线程作用域 <thread-scopes>` 来指定异步屏障。
关于异步屏障的完整详细信息，请参阅 :ref:`async-barriers-details` 。

.. _asynchronous-data-copies-and-the-tensor-memory-accelerator-tma:

3.5.1.2. 异步数据拷贝和 Tensor Memory Accelerator (TMA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 kernel 代码中，异步数据拷贝（asynchronous data copies）是指在执行计算任务的同时，能够在共享内存与 GPU 显存（DRAM）之间进行数据搬运。
请注意，在这里不要将其与 CPU 和 GPU 之间的异步内存拷贝相混淆。
该特性利用了前文提到的异步屏障机制。关于异步拷贝的具体使用方法，将在 :ref:`async-copies-details` 中进行详细讲解。

.. _feature-survey-pipelines:

3.5.1.3. 流水线
^^^^^^^^^^^^^^^

流水线（Pipelines）是一种用于暂存任务以及协调多缓冲生产-消费模式的机制，通常被用来实现计算与 :ref:`异步数据拷贝 <async-copies-details>` 的重叠执行。
关于在 CUDA 中使用流水线的详细说明与示例，请参阅 :ref:`pipelines-details` 。

.. _work-stealing-with-cluster-launch-control:

3.5.1.4. 使用 Cluster Launch Control 的工作窃取
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

工作窃取（Work stealing）是一种在负载不均的情况下维持资源利用率的技巧，它允许已经完成自身任务的线程 `窃取` 其他线程的任务。
集群启动控制（Cluster launch control）是计算能力 10.0 (Blackwell) 架构引入的一项特性，
它赋予了 kernel 直接控制正在执行中的块调度（in-flight block scheduling）的能力，使 kernel 能够实时适应不均匀的负载。
具体而言，一个线程块可以取消另一个尚未启动的线程块或集群的启动，接管其索引，并立即开始执行被窃取的工作。
这种工作窃取机制能够在数据分布不规则或运行时发生变化的情况下，保持 SMs 处于忙碌状态并减少空闲时间——它在不完全依赖硬件调度器的前提下，实现了更细粒度的负载均衡。

关于该特性的详细说明，请参阅 :ref:`cluster-launch-control` 。

.. _improving-latencies:

3.5.2. 改善延迟
---------------

本节概述的各项特性有一个共同的主题，即旨在降低某种类型的延迟，尽管它们各自针对的延迟类型有所不同。
总体而言，这些特性主要关注 kernel 启动级别或更高层级的延迟。
而在 kernel 执行过程中访问 GPU 内存所产生的延迟，并不在本节的讨论范围之内。

.. _green-contexts:

3.5.2.1. Green Contexts
^^^^^^^^^^^^^^^^^^^^^^^

Green 上下文（Green contexts），也称为执行上下文（execution contexts），是 CUDA 的一项特性。
它允许应用创建特定的 :ref:`CUDA 上下文 <driver-api-context>`，从而使其任务仅在 GPU 的部分 SM 上执行。
在默认情况下，kernel 启动时的线程块会被调度到 GPU 内任何能够满足该 kernel 资源需求的 SM 上。
有许多因素会影响哪些 SM 能够执行某个线程块，包括但不限于：共享内存的使用量、寄存器的使用量、集群（clusters）的使用情况，以及线程块中的总线程数。

执行上下文允许将 kernel 下发到一个专门创建的上下文中，从而限制可执行该 kernel 的 SM 的数量。
重要的是，当应用创建一个使用了特定 SM 集合的 Green 上下文后，GPU 上的其他上下文将不能把线程块调度到已分配给该 Green 上下文的 SMs 上。
这也包括主上下文（primary context），即 CUDA 运行时默认使用的上下文。
这种机制使得这些被隔离出来的 SM 能够被保留下来，专门用于处理高优先级或对延迟敏感的工作负载。

Green 上下文自 CUDA 13.1 及更高版本的 CUDA runtime 起正式提供支持， 在 :ref:`green-contexts-details` 给出完整细节。

.. _stream-ordered-memory-allocation:

3.5.2.2. 流序内存分配
^^^^^^^^^^^^^^^^^^^^^

流排序内存分配器（Stream-ordered memory allocator）允许程序将 GPU 内存的分配与释放操作编排到特定的 CUDA 流中。
与 ``cudaMalloc`` 和 ``cudaFree`` 立即执行不同， ``cudaMallocAsync`` 和 ``cudaFreeAsync`` 会将内存分配或释放操作插入到指定的 :ref:`CUDA 流 <cuda-streams>` 中。
关于这些 API 的所有细节将在 :ref:`stream-ordered-memory-allocation-details` 中进行详细介绍。

.. _tour-cuda-graphs:

3.5.2.3. CUDA Graphs
^^^^^^^^^^^^^^^^^^^^

CUDA 图允许应用程序预先指定一系列 CUDA 操作（如启动 kernel 或内存拷贝）及其相互之间的依赖关系，从而使这些操作能够在 GPU 上得到高效执行。
利用 :ref:`CUDA 流 <cuda-streams>` 也能实现类似的效果，事实上，
创建图的一种机制就叫做 :ref:`流捕获（stream capture） <cuda-graphs-stream-capture>`，它能够将流中的操作录制下来并转化为一个 CUDA 图。
此外，开发者也可以直接使用 :ref:`CUDA 图 API <cuda-graphs-graph-api>` 来手动构建图。

一旦创建了图，就可以对其进行多次实例化和执行。
这对于需要重复执行的工作非常有用。图在降低调用 CUDA 操作时产生的 CPU 启动开销方面具有一定的性能优势；此外，当整个工作负载被提前完整指定时，它还能启用一些专属的底层优化。

:ref:`cuda-graphs` 将描述并演示了如何使用 CUDA Graphs。

.. _programmatic-dependent-launch:

3.5.2.4. 程序化依赖启动
^^^^^^^^^^^^^^^^^^^^^^^
程序化依赖启动（Programmatic dependent launch）是 CUDA 的一项特性。
它允许一个依赖型 kernel （即需要依赖前序 kernel 输出结果的 kernel），在其所依赖的主 kernel 尚未完成时就开始执行。
该依赖型 kernel 可以提前执行一些初始化代码和不相关的任务，直到真正需要主 kernel 的数据时才在此处阻塞等待。
当主 kernel 准备好所需数据后，会发出信号通知，从而解除阻塞，让依赖型 kernel 继续执行。
这种机制实现了多个 kernel 之间的部分重叠执行，不仅有助于保持较高的 GPU 利用率，还能最大限度地缩短关键数据路径上的延迟。

更多详细内容，请参阅 :ref:`programmatic-dependent-launch-details` 。

.. _lazy-loading:

3.5.2.5. 延迟加载
^^^^^^^^^^^^^^^^^

延迟加载（Lazy loading）是一项允许开发者控制 JIT 编译器在应用程序启动时如何运行的特性。
如果包含大量需要从 PTX 即时编译为 cubin 的应用程序在启动时进行全量编译，可能会导致漫长的启动时间。
系统的默认行为是：模块只有在被实际调用时才会进行编译。
不过，开发者可以通过设置 :ref:`环境变量 <module-loading>` 来更改这一默认行为，具体细节请参阅 :ref:`lazy-loading-details` 。

.. _functionality-features:

3.5.3. 功能性特性
-----------------

本节介绍的特性有一个共同点：它们旨在赋予程序额外的能力或功能。

.. _extended-gpu-memory:

3.5.3.1. 扩展 GPU 内存
^^^^^^^^^^^^^^^^^^^^^^

扩展 GPU 内存（Extended GPU memory, EGM）是 NVLink-C2C 连接系统中提供的一项特性，它允许从 GPU 内部高效地访问系统内的所有内存。
关于 EGM 的详细内容，请参阅 :ref:`extended-gpu-memory-details` 。

.. _dynamic-parallelism:

3.5.3.2. 动态并行
^^^^^^^^^^^^^^^^^

CUDA 应用程序通常是从运行在 CPU 上的代码来启动 kernel 。
不过，开发者也可以在 GPU 上运行的 kernel 内部创建新的 kernel 调用。
这项特性被称为 CUDA 动态并行（CUDA dynamic parallelism）。

关于如何从运行在 GPU 上的代码中启动新的 GPU kernel， 将在 :ref:`cuda-dynamic-parallelism` 详细介绍。

.. _cuda-interoperability:

3.5.4. CUDA 互操作性
--------------------

.. _cuda-interoperability-with-other-apis:

3.5.4.1. CUDA 与其他 API 的互操作性
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

除了 CUDA 之外，还有其他在 GPU 上运行代码的机制。
GPU 最初是为了加速计算机图形应用而构建的，这类应用使用自己的一套 API（如 Direct3D 和 Vulkan）。
应用程序可能希望在使用某种图形 API 进行 3D 渲染的同时，利用 CUDA 执行计算任务。
为此，CUDA 提供了相应的机制，用于在 CUDA 上下文与 3D API 使用的 GPU 上下文之间交换存储在 GPU 上的数据。
例如，应用程序可以使用 CUDA 进行物理模拟，然后使用 3D API 将结果进行可视化呈现。
实现这一过程的方法是，让某些缓冲区（buffers）能够被 CUDA 和图形 API 同时读取和/或写入。

相同机制也被用于与通信共享缓冲区。这使得在多节点环境中，能够实现快速的、GPU 到 GPU 的直接通信。

:ref:`graphics-interop-details` 介绍了 CUDA 如何与其他 GPU API 进行互操作，以及如何在 CUDA 和其他 API 之间共享数据，并针对多种不同的 API 提供了具体的示例


.. _interprocess-communication:

3.5.4.2. 进程间通信
^^^^^^^^^^^^^^^^^^^

对于超大规模的计算任务，通常会联合使用多个 GPU，以便在处理特定问题时能够调用更多的显存和计算资源。
在单个系统内（在集群计算术语中称为 **节点** ），可以在同一个主机进程中使用多个 GPU。 这在 :ref:`multi-gpu-introduction` 有过介绍。

此外，使用跨越单台或多台计算机的独立主机进程也是非常常见的做法。
当多个进程协同工作时，它们之间的通信被称为进程间通信（interprocess communication, IPC）。
CUDA 进程间通信（CUDA IPC）提供了在不同进程之间共享 GPU 缓冲区的机制。
:ref:`interprocess-communication-details` 将详细解释并演示如何使用 CUDA IPC 在不同的主机进程之间进行协调与通信。


.. _fine-grained-control:

3.5.5. 细粒度控制
------------------

.. _virtual-memory-management:

3.5.5.1. 虚拟内存管理
^^^^^^^^^^^^^^^^^^^^^

正如 :ref:`memory-unified-virtual-address-space` 所述，系统中的所有 GPU 以及 CPU 内存共享一个统一的虚拟地址空间。
大多数应用程序可以直接使用 CUDA 提供的默认内存管理机制，而无需更改其行为。
然而，对于有更高需求的开发者， :ref:`CUDA 驱动 API <driver-api>` 提供了针对该虚拟内存空间布局的高级且精细的控制手段。
这种高级控制主要适用于需要跨系统或在系统内部的多 GPU 之间共享缓冲区时，用于控制这些缓冲区的行为。

:ref:`virtual-memory-management-details` 详细讲解 CUDA 驱动 API 所提供的各项控制手段，解释了它们的工作原理，以及开发者在何种情况下会发现这些功能大有裨益。

.. _driver-entry-point-access:

3.5.5.2. 驱动入口点访问
^^^^^^^^^^^^^^^^^^^^^^^^

驱动入口点访问（Driver entry point access）是指从 CUDA 11.3 开始提供的一种能力，它允许开发者检索指向 CUDA 驱动 API 和 CUDA 运行时 API 的函数指针。
此外，它还允许开发者获取特定驱动函数的变体版本的函数指针，并能够调用比当前安装的 CUDA 工具包更新的驱动程序中所包含的驱动函数。
关于驱动入口点访问的详细内容，请参阅 :ref:`driver-entry-point-access` 。

.. _error-log-management:

3.5.5.3. 错误日志管理
^^^^^^^^^^^^^^^^^^^^^

错误日志管理（Error log management）提供了一系列实用工具，用于处理和记录来自 CUDA API 的错误。
只需设置一个环境变量 ``CUDA_LOG_FILE`` ，即可将捕获到的 CUDA 错误直接输出到标准错误流（stderr）、标准输出流（stdout）或写入文件中。
此外，错误日志管理还允许应用程序注册一个回调函数，当 CUDA 遇到错误时便会触发该回调。
关于错误日志管理的更多详细信息，请参阅 :ref:`error-log-management-details` 。