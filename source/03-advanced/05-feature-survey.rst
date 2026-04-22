.. _tour-of-features:

3.5. CUDA 功能概览
==================

本编程指南的第 1-3 章介绍了 CUDA 和 GPU 编程，涵盖了基础概念和简单的代码示例。本指南第 4 部分中描述特定 CUDA 功能的章节假设读者已了解第 1-3 章中涵盖的概念。

CUDA 拥有许多适用于不同问题的功能。并非所有功能都适用于每个用例。本章旨在介绍每个功能，描述其预期用途以及它可能帮助解决的问题。功能按其旨在解决的问题类型粗略分类。某些功能（如 CUDA Graphs）可能适合多个类别。

:numref:`Section 4` 更详细地介绍了这些 CUDA 功能。

.. _improving-kernel-performance:

3.5.1. 提升 Kernel 性能
------------------------

本节概述的功能旨在帮助 kernel 开发者最大化其 kernel 的性能。

.. _feature-survey-asynchronous-barriers:

3.5.1.1. 异步屏障
^^^^^^^^^^^^^^^^^

:ref:`异步屏障 <asynchronous-barriers>` 在 :numref:`Section 3.2.4.2` 中介绍，允许对线程间的同步进行更精细的控制。异步屏障将屏障的到达和等待分开。这使应用程序能够在等待其他线程到达时执行不依赖于屏障的工作。可以为不同的 :ref:`线程作用域 <advanced-kernels-thread-scopes>` 指定异步屏障。:numref:`Section 4.9` 详细介绍了异步屏障。

.. _asynchronous-data-copies-and-the-tensor-memory-accelerator-tma:

3.5.1.2. 异步数据拷贝和 Tensor Memory Accelerator (TMA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA kernel 代码中的 :ref:`异步数据拷贝 <async-copies>` 指的是在共享内存和 GPU DRAM 之间移动数据的同时仍在进行计算的能力。这不应与 CPU 和 GPU 之间的异步内存拷贝混淆。此功能使用异步屏障。:numref:`Section 4.11` 详细介绍了异步拷贝的使用。

.. _feature-survey-pipelines:

3.5.1.3. 流水线
^^^^^^^^^^^^^^^

:ref:`流水线 <pipelines>` 是一种用于分阶段工作和协调多缓冲区生产者-消费者模式的机制，通常用于将计算与 :ref:`异步数据拷贝 <async-copies>` 重叠。:numref:`Section 4.10` 提供了在 CUDA 中使用流水线的详细信息和示例。

.. _work-stealing-with-cluster-launch-control:

3.5.1.4. 使用 Cluster Launch Control 的工作窃取
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

工作窃取是一种在负载不均时保持利用率的技术，已完成工作的工作线程可以从其他工作线程"窃取"任务。Cluster launch control 是计算能力 10.0 (Blackwell) 中引入的功能，使 kernel 能够直接控制正在执行的 block 调度，从而可以实时适应不均的负载。线程块可以取消尚未启动的其他线程块或 cluster 的启动，声明其索引，并立即开始执行窃取的工作。这种工作窃取流程使 SM 保持忙碌，减少了不规则数据或运行时变化下的空闲时间——在不单独依赖硬件调度器的情况下提供更细粒度的负载均衡。

:numref:`Section 4.12` 提供了如何使用此功能的详细信息。

.. _improving-latencies:

3.5.2. 改善延迟
---------------

本节概述的功能有一个共同主题，即旨在减少某种类型的延迟，尽管不同功能解决的延迟类型不同。总的来说，它们专注于 kernel 启动级别或更高层的延迟。kernel 内的 GPU 内存访问延迟不在此考虑范围内。

.. _green-contexts:

3.5.2.1. Green Contexts
^^^^^^^^^^^^^^^^^^^^^^^

:ref:`Green contexts <green-contexts>`，也称为 *执行上下文*，是 CUDA 功能的名称，它使程序能够创建仅在 GPU 的 SM 子集上执行工作的 :ref:`CUDA 上下文 <driver-api-context>`。默认情况下，kernel 启动的线程块被分派到 GPU 内可以满足 kernel 资源要求的任何 SM。有许多因素会影响哪些 SM 可以执行线程块，包括但不限于：共享内存使用、寄存器使用、cluster 的使用以及线程块中的线程总数。

执行上下文允许将 kernel 启动到专门创建的上下文中，该上下文进一步限制了可用于执行 kernel 的 SM 数量。重要的是，当程序创建使用某些 SM 集的 green context 时，GPU 上的其他上下文不会将线程块调度到分配给 green context 的 SM。这包括主上下文，即 CUDA runtime 使用的默认上下文。这允许为高优先级或延迟敏感的工作负载保留这些 SM。

:numref:`Section 4.6` 提供了 green contexts 使用的完整详细信息。Green contexts 在 CUDA 13.1 及更高版本的 CUDA runtime 中可用。

.. _stream-ordered-memory-allocation:

3.5.2.2. 流序内存分配
^^^^^^^^^^^^^^^^^^^^^

:ref:`流序内存分配器 <stream-ordered-memory-allocator>` 允许程序将 GPU 内存的分配和释放顺序安排到 :ref:`CUDA 流 <cuda-streams>` 中。与立即执行的 ``cudaMalloc`` 和 ``cudaFree`` 不同， ``cudaMallocAsync`` 和 ``cudaFreeAsync`` 将内存分配或释放操作插入到 CUDA 流中。:numref:`Section 4.3` 涵盖了这些 API 的所有详细信息。

.. _cuda-graphs:

3.5.2.3. CUDA Graphs
^^^^^^^^^^^^^^^^^^^^

:ref:`CUDA Graphs <cuda-graphs>` 使应用程序能够指定一系列 CUDA 操作（如 kernel 启动或内存拷贝）以及这些操作之间的依赖关系，以便它们可以在 GPU 上高效执行。类似的行为可以通过使用 :ref:`CUDA 流 <cuda-streams>` 来实现，实际上创建 graph 的机制之一称为 :ref:`流捕获 <cuda-graphs-creating-a-graph-using-stream-capture>`，它使流上的操作能够记录到 CUDA graph 中。Graphs 也可以使用 :ref:`CUDA Graphs API <cuda-graphs-creating-a-graph-using-graph-apis>` 创建。

一旦创建了 graph，它就可以被实例化并多次执行。这对于指定将要重复的工作负载很有用。Graphs 提供了一些性能优势，减少了与调用 CUDA 操作相关的 CPU 启动成本，并使仅在预先指定整个工作负载时才可用的优化成为可能。

:numref:`Section 4.2` 描述并演示了如何使用 CUDA Graphs。

.. _programmatic-dependent-launch:

3.5.2.4. 程序化依赖启动
^^^^^^^^^^^^^^^^^^^^^^^

:ref:`程序化依赖启动 <programmatic-dependent-launch-and-synchronization>` 是一种 CUDA 功能，允许依赖 kernel（即依赖于前一个 kernel 输出的 kernel）在其所依赖的主 kernel 完成之前开始执行。依赖 kernel 可以执行设置代码和无关工作，直到它需要来自主 kernel 的数据并在那里阻塞。主 kernel 可以在依赖 kernel 所需的数据准备好时发出信号，这将释放依赖 kernel 继续执行。这使 kernel 之间能够有一些重叠，可以帮助保持高 GPU 利用率，同时最小化关键数据路径的延迟。:numref:`Section 4.5` 介绍了程序化依赖启动。

.. _lazy-loading-feature:

3.5.2.5. 延迟加载
^^^^^^^^^^^^^^^^^

:ref:`延迟加载 <lazy-loading-feature>` 是一种允许控制 JIT 编译器在应用程序启动时如何操作的功能。有许多 kernel 需要从 PTX JIT 编译为 cubin 的应用程序，如果在应用程序启动期间所有 kernel 都进行 JIT 编译，可能会经历较长的启动时间。默认行为是模块在需要之前不会被编译。这可以通过使用 :ref:`环境变量 <cuda-environment-variables>` 来更改，如 :numref:`Section 4.7` 中详细介绍的那样。

.. _functionality-features:

3.5.3. 功能性特性
-----------------

这里描述的功能有一个共同特点，即它们旨在启用额外的能力或功能。

.. _extended-gpu-memory:

3.5.3.1. 扩展 GPU 内存
^^^^^^^^^^^^^^^^^^^^^^

:ref:`扩展 GPU 内存 <extended-gpu-memory>` 是 NVLink-C2C 连接系统中可用的功能，使 GPU 能够高效访问系统内的所有内存。:numref:`Section 4.17` 详细介绍了 EGM。

.. _dynamic-parallelism:

3.5.3.2. 动态并行
^^^^^^^^^^^^^^^^^

CUDA 应用程序最常见的是从 CPU 上运行的代码启动 kernel。也可以从 GPU 上运行的 kernel 创建新的 kernel 调用。此功能称为 :ref:`CUDA 动态并行 <cuda-dynamic-parallelism>`。:numref:`Section 4.18` 介绍了从 GPU 上运行的代码创建新 GPU kernel 启动的详细信息。

.. _cuda-interoperability:

3.5.4. CUDA 互操作性
--------------------

.. _cuda-interoperability-with-other-apis:

3.5.4.1. CUDA 与其他 API 的互操作性
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

除了 CUDA 之外，还有其他在 GPU 上运行代码的机制。GPU 最初构建用于加速的应用程序——计算机图形——使用自己的一组 API，如 Direct3D 和 Vulkan。应用程序可能希望使用其中一个图形 API 进行 3D 渲染，同时在 CUDA 中执行计算。CUDA 提供了在 CUDA 上下文和 3D API 使用的 GPU 上下文之间交换存储在 GPU 上的数据的机制。例如，应用程序可以使用 CUDA 执行模拟，然后使用 3D API 创建结果的可视化。这是通过使某些缓冲区可从 CUDA 和图形 API 同时读取和/或写入来实现的。

允许与图形 API 共享缓冲区的相同机制也用于与通信机制共享缓冲区，这可以实现多节点环境中的快速、直接 GPU 到 GPU 通信。

:numref:`Section 4.19` 描述了 CUDA 如何与其他 GPU API 互操作以及如何在 CUDA 和其他 API 之间共享数据，为多种不同的 API 提供了具体示例。

.. _interprocess-communication:

3.5.4.2. 进程间通信
^^^^^^^^^^^^^^^^^^^

对于非常大的计算，通常一起使用多个 GPU 来利用更多内存和更多协同处理问题的计算资源。在单个系统内，或集群计算术语中的节点内，多个 GPU 可以在单个主机进程中使用。这在 :numref:`Section 3.4` 中有描述。

也常见使用跨越单台计算机或多台计算机的独立主机进程。当多个进程协同工作时，它们之间的通信称为进程间通信。CUDA 进程间通信 (CUDA IPC) 提供了在不同进程之间共享 GPU 缓冲区的机制。:numref:`Section 4.15` 解释并演示了如何使用 CUDA IPC 在不同主机进程之间进行协调和通信。

.. _fine-grained-control:

3.5.5. 细粒度控制
------------------

.. _virtual-memory-management:

3.5.5.1. 虚拟内存管理
^^^^^^^^^^^^^^^^^^^^^

如 :numref:`Section 2.4.1` 所述，系统中的所有 GPU 以及 CPU 内存共享一个统一的虚拟地址空间。大多数应用程序可以使用 CUDA 提供的默认内存管理，而无需更改其行为。然而，:ref:`CUDA 驱动 API <driver-api>` 为需要的人提供了对这个虚拟内存空间布局的高级和详细控制。这主要适用于控制在 GPU 之间共享缓冲区的行为，无论是在系统内还是跨多个系统。

:numref:`Section 4.16` 介绍了 CUDA 驱动 API 提供的控制、它们如何工作以及开发者何时可能发现它们有用。

.. _driver-entry-point-access-feature:

3.5.5.2. 驱动入口点访问
^^^^^^^^^^^^^^^^^^^^^^^

:ref:`驱动入口点访问 <driver-entry-point-access-feature>` 指的是从 CUDA 11.3 开始，能够检索 CUDA Driver 和 CUDA Runtime API 的函数指针。它还允许开发者检索特定驱动函数变体的函数指针，并访问比 CUDA toolkit 中可用驱动更新的驱动中的驱动函数。:numref:`Section 4.20` 介绍了驱动入口点访问。

.. _error-log-management:

3.5.5.3. 错误日志管理
^^^^^^^^^^^^^^^^^^^^^

:ref:`错误日志管理 <error-log-management>` 提供了处理和记录 CUDA API 错误的工具。设置单个环境变量 ``CUDA_LOG_FILE`` 可以将 CUDA 错误直接捕获到 stderr、stdout 或文件中。错误日志管理还使应用程序能够注册在 CUDA 遇到错误时触发的回调。:numref:`Section 4.8` 提供了有关错误日志管理的更多详细信息。