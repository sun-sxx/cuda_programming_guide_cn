.. _cuda-platform:

1.3. CUDA 平台
===============

NVIDIA CUDA 平台由许多软件和硬件部分以及许多为在异构系统上实现计算而开发的重要技术组成。本章介绍应用程序开发者需要了解的 CUDA 平台的一些基本概念和组件。本章与 :doc:`programming-model` 一样，不特定于任何编程语言，但适用于使用 CUDA 平台的所有内容。

.. _compute-capability-and-streaming-multiprocessor-versions:

1.3.1. 计算能力和流式多处理器版本
---------------------------------

每个 NVIDIA GPU 都有一个 *计算能力* (CC) 编号，指示该 GPU 支持哪些功能并指定该 GPU 的某些硬件参数。这些规范记录在 :ref:`sec:compute-capabilities` 附录中。所有 NVIDIA GPU 及其计算能力的列表维护在 `CUDA GPU 计算能力页面 <https://developer.nvidia.com/cuda-gpus>`_ 上。

计算能力表示为主版本号和次版本号，格式为 X.Y，其中 X 是主版本号，Y 是次版本号。例如，CC 12.0 的主版本为 12，次版本为 0。计算能力直接对应于 SM 的版本号。例如，CC 12.0 的 GPU 中的 SM 具有 SM 版本 sm_120。此版本用于标记二进制文件。

:ref:`sec:compute-capabilities-querying` 展示了如何查询和确定系统中 GPU 的计算能力。

.. _cuda-toolkit-and-nvidia-driver:

1.3.2. CUDA 工具包和 NVIDIA 驱动程序
-------------------------------------

*NVIDIA 驱动程序* 可以被视为 GPU 的操作系统。NVIDIA 驱动程序是必须安装在主机系统操作系统上的软件组件，对于所有 GPU 使用（包括显示和图形功能）都是必需的。NVIDIA 驱动程序是 CUDA 平台的基础。除了 CUDA，NVIDIA 驱动程序还提供所有其他使用 GPU 的方法，例如 Vulkan 和 Direct3D。NVIDIA 驱动程序有版本号，如 r580。

*CUDA 工具包* 是一组库、头文件和工具，用于编写、构建和分析利用 GPU 计算的软件。CUDA 工具包是与 NVIDIA 驱动程序分开的软件产品。

*CUDA 运行时* 是 CUDA 工具包提供的库中的一个特例。CUDA 运行时提供 API 和一些语言扩展来处理常见任务，如分配内存、在 GPU 和其他 GPU 或 CPU 之间复制数据以及启动内核。CUDA 运行时的 API 组件称为 CUDA 运行时 API。

`CUDA 兼容性 <https://docs.nvidia.com/deploy/cuda-compatibility/index.html>`_ 文档提供了不同 GPU、NVIDIA 驱动程序和 CUDA 工具包版本之间兼容性的完整详细信息。

.. _cuda-runtime-api-and-cuda-driver-api:

1.3.2.1. CUDA 运行时 API 和 CUDA 驱动程序 API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA 运行时 API 实现在称为 *CUDA 驱动程序 API* 的低级 API 之上，该 API 是 NVIDIA 驱动程序公开的 API。本指南重点介绍 CUDA 运行时 API 公开的 API。如果需要，可以仅使用驱动程序 API 实现所有相同的功能。某些功能仅在使用驱动程序 API 时可用。应用程序可以使用任一 API 或两者互操作。:doc:`../03-advanced/driver-api` 部分介绍了运行时和驱动程序 API 之间的互操作。

CUDA 运行时 API 函数的完整 API 参考可以在 `CUDA 运行时 API 文档 <https://docs.nvidia.com/cuda/cuda-runtime-api/index.html>`_ 中找到。

CUDA 驱动程序 API 的完整 API 参考可以在 `CUDA 驱动程序 API 文档 <https://docs.nvidia.com/cuda/cuda-driver-api/index.html>`_ 中找到。

.. _parallel-thread-execution-ptx:

1.3.3. 并行线程执行 (PTX)
-------------------------

CUDA 平台的一个基本但有时不可见的层是 *并行线程执行* (PTX) 虚拟指令集架构 (ISA)。PTX 是 NVIDIA GPU 的高级汇编语言。PTX 提供了真实 GPU 硬件物理 ISA 上的抽象层。像其他平台一样，应用程序可以直接用这种汇编语言编写，尽管这样做可能会给软件开发增加不必要的复杂性和困难。

特定领域语言和高级语言的编译器可以生成 PTX 代码作为中间表示 (IR)，然后使用 NVIDIA 的离线或即时 (JIT) 编译工具生成可执行的二进制 GPU 代码。这使得 CUDA 平台可以从 NVIDIA 提供的工具（如 :doc:`../02-basics/nvcc` ）支持的语言以外的语言编程。

由于 GPU 能力随时间变化和增长，PTX 虚拟 ISA 规范有版本控制。PTX 版本与 SM 版本一样，对应于计算能力。例如，支持计算能力 8.0 所有功能的 PTX 称为 compute_80。

有关 PTX 的完整文档可以在 `PTX ISA <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>`_ 中找到。

.. _cubins-and-fatbins:

1.3.4. Cubin 和 Fatbin
-----------------------

CUDA 应用程序和库通常用 C++ 等高级语言编写。该高级语言编译为 PTX，然后 PTX 编译为物理 GPU 的真实二进制，称为 *CUDA 二进制*，简称 *cubin*。cubin 具有特定 SM 版本的特定二进制格式，如 sm_120。

使用 GPU 计算的可执行文件和库二进制文件包含 CPU 和 GPU 代码。GPU 代码存储在称为 *fatbin* 的容器内。Fatbin 可以包含多个不同目标的 cubin 和 PTX。例如，应用程序可以为多个不同的 GPU 架构（即不同的 SM 版本）构建二进制文件。当运行应用程序时，其 GPU 代码加载到特定 GPU 上，并使用 fatbin 中最适合该 GPU 的二进制文件。

.. _fig:fatbin-graphic:
.. figure:: /_static/images/fatbin.png
   :alt: 可执行文件或库中的 Fatbin 容器可以包含多个 GPU 代码版本
   :width: 80%

   可执行文件或库的二进制文件包含 CPU 二进制代码和 GPU 代码的 fatbin 容器。fatbin 可以同时包含 cubin GPU 二进制代码和 PTX 虚拟 ISA 代码。PTX 代码可以为未来目标进行 JIT 编译。

Fatbin 还可以包含一个或多个 PTX 版本的 GPU 代码，其用途在 :ref:`sec:cuda-platform-ptx-compatibility` 中描述。:numref:`fig:fatbin-graphic` 显示了一个包含多个 cubin 版本 GPU 代码以及一个 PTX 代码版本的应用程序或库二进制文件的示例。

.. _binary-compatibility:

1.3.4.1. 二进制兼容性
^^^^^^^^^^^^^^^^^^^^^^

NVIDIA GPU 在某些情况下保证二进制兼容性。具体来说，在计算能力的主版本内，计算能力次版本大于或等于 cubin 目标版本的 GPU 可以加载和执行该 cubin。例如，如果应用程序包含为计算能力 8.6 编译的 cubin 代码，则该 cubin 可以在计算能力 8.6 或 8.9 的 GPU 上加载和执行。但是，它不能在计算能力 8.0 的 GPU 上加载，因为 GPU 的 CC 次版本 0 低于代码的次版本 6。

NVIDIA GPU 在主计算能力版本之间不具有二进制兼容性。也就是说，为计算能力 8.6 编译的 cubin 代码将不会在计算能力 9.0 的 GPU 上加载。

在讨论二进制代码时，二进制代码通常被称为具有版本号，如上例中的 sm_86。这与说二进制是为计算能力 8.6 构建的相同。这种简写经常使用，因为这是开发者向 NVIDIA CUDA 编译器 :doc:`../02-basics/nvcc` 指定此二进制构建目标的方式。

.. note::

   二进制兼容性仅对由 NVIDIA 工具（如 ``nvcc`` ）创建的二进制文件有保证。不支持手动编辑或生成 NVIDIA GPU 的二进制代码。如果以任何方式修改二进制文件，兼容性承诺将失效。

.. _ptx-compatibility:

1.3.4.2. PTX 兼容性
^^^^^^^^^^^^^^^^^^^

GPU 代码可以以二进制或 PTX 形式存储在可执行文件中，这在 :ref:`sec:cuda-platform-cubins-fatbins` 中介绍。当应用程序存储 GPU 代码的 PTX 版本时，该 PTX 可以在应用程序运行时为任何等于或高于 PTX 代码计算能力的计算能力进行 JIT 编译。例如，如果应用程序包含 compute_80 的 PTX，则该 PTX 代码可以在应用程序运行时 JIT 编译为更高的 SM 版本，如 sm_120。这实现了与未来 GPU 的前向兼容性，而无需重新构建应用程序或库。

.. _just-in-time-compilation:

1.3.4.3. 即时编译
^^^^^^^^^^^^^^^^^

应用程序在运行时加载的 PTX 代码由设备驱动程序编译为二进制代码。这称为即时 (JIT) 编译。即时编译增加了应用程序加载时间，但允许应用程序受益于每个新设备驱动程序带来的任何新编译器改进。它还允许应用程序在编译应用程序时不存在的设备上运行。

当设备驱动程序为应用程序即时编译 PTX 代码时，它会自动缓存生成的二进制代码的副本，以避免在后续调用应用程序时重复编译。缓存（称为计算缓存）在设备驱动程序升级时自动失效，以便应用程序可以受益于内置在设备驱动程序中的新即时编译器的改进。

自 CUDA 最早版本以来，PTX 在运行时如何以及何时进行 JIT 编译已经放宽，允许更灵活地决定何时以及是否 JIT 编译部分或全部内核。:doc:`../04-special-topics/lazy-loading` 部分描述了可用选项以及如何控制 JIT 行为。还有一些控制即时编译行为的环境变量，如 :doc:`../05-appendices/environment-variables` 中所述。

作为使用 ``nvcc`` 编译 CUDA C++ 设备代码的替代方法，可以使用 NVRTC 在运行时将 CUDA C++ 设备代码编译为 PTX。NVRTC 是 CUDA C++ 的运行时编译库；更多信息可以在 NVRTC 用户指南中找到。