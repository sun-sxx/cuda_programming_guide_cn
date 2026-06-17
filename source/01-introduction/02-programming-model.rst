.. _programming-model:

1.2. 编程模型
=============

本章从高层次介绍 CUDA 编程模型，不涉及任何语言。这里介绍的术语和概念适用于任何支持 CUDA 的编程语言。
后面的章节将在 C++ 中说明这些概念。

.. _heterogeneous-systems:

1.2.1. 异构系统
---------------

CUDA 编程模型假设一个异构计算系统，即包括 GPU 和 CPU 的系统。
CPU 及其直接连接的内存分别称为 *主机* 和 *主机内存*。
GPU 及其直接连接的内存分别称为 *设备* 和 *设备内存*。
在某些片上系统 (SoC) 中，这些可能是单个封装的一部分。
在更大的系统中，可能有多个 CPU 或 GPU。

CUDA 应用程序在 GPU 上执行部分代码，但应用程序总是从 CPU 上开始执行。
在 CPU 上运行的代码称为主机代码，可以使用 CUDA API 在主机内存和设备内存之间复制数据、启动在 GPU 上执行的代码，以及等待数据复制或 GPU 代码完成。
CPU 和 GPU 可以同时执行代码，通常通过最大化 CPU 和 GPU 的利用率来获得最佳性能。

应用程序在 GPU 上执行的代码称为 *设备代码*，为在 GPU 上执行而调用的函数由于历史原因称为 *核函数* (kernel)。启动核函数运行的操作称为 *launching* 核函数。
核函数启动可以看作是在 GPU 上并行启动许多线程执行核函数代码。
GPU 线程的操作类似于 CPU 上的线程，尽管存在一些对正确性和性能都很重要的差异，这些将在后面的章节中介绍（参见 :ref:`independent-thread-scheduling` ）。

.. _gpu-hardware-model:

1.2.2. GPU 硬件模型
-------------------

像任何编程模型一样，CUDA 依赖于底层硬件的概念模型。
出于 CUDA 编程的目的，GPU 可以被视为 *流式多处理器* (*Streaming Multiprocessors*, SM) 的集合，这些 SM 组织成称为 *图形处理集群* (*Graphics Processing Clusters*, GPC) 的组。
每个 SM 包含本地寄存器文件、统一数据缓存和执行计算的多个功能单元。统一数据缓存为 *共享内存* 和 L1 缓存提供物理资源。
统一数据缓存到 L1 和共享内存的分配可以在运行时配置。
不同类型内存的大小和 SM 内功能单元的数量可能因 GPU 架构而异。

.. note::

   GPU 的实际硬件布局或其物理执行编程模型的方式可能会有所不同。这些差异不影响使用 CUDA 编程模型编写的软件的正确性。

.. _fig-gpu-cpu-system-diagram:
.. figure:: /_static/images/gpu-cpu-system-diagram.png
   :alt: CUDA 编程模型视图下的 CPU 和 GPU 组件及连接
   :width: 80%
   :align: center

   CUDA 编程模型视图下的 CPU 和 GPU 组件及连接。GPU 有许多流式多处理器 (SM)，每个 SM 包含许多功能单元。图形处理集群 (GPC) 是 SM 的集合。
   GPU 是连接到 GPU 内存的一组 GPC。CPU 通常有几个核心和一个内存控制器，连接到系统内存。CPU 和 GPU 通过 PCIe 或 NVLINK 等互连连接。

.. _thread-blocks-and-grids:

1.2.2.1. 线程块和网格
^^^^^^^^^^^^^^^^^^^^^^

当应用程序启动核函数时，它会使用许多线程，通常是数百万个线程。这些线程被组织成块。顾名思义——称为 *线程块* (thread block)。
线程块被组织成 *网格* (grid)。网格中的所有线程块具有相同的大小和维度。
:numref:`fig-grid-of-thread-blocks` 显示了线程块网格的示意图。

.. _fig-grid-of-thread-blocks:
.. figure:: /_static/images/grid-of-thread-blocks.png
   :alt: 线程块网格
   :width: 80%
   :align: center

   线程块网格。每个箭头代表一个线程（箭头数量不代表实际线程数量）。

线程块和网格可以是一维、二维或三维的。这些维度可以简化单个线程到工作单元或数据项的映射。

当启动核函数时，使用特定的 *执行配置* (execution configuration)，指定网格和线程块维度。执行配置还可以包括可选参数，如集群大小、流和 SM 配置设置，这些将在后面的章节中介绍。

使用内置变量(built-in variables)，执行核函数的每个线程可以确定其在线程块中的位置以及其所在线程块在网格中的位置。
线程还可以使用这些内置变量确定启动核函数的线程块和网格的维度。
这为每个线程在运行核函数的所有线程中提供了唯一标识。此标识通常用于确定线程负责什么数据或操作。

线程块的所有线程在单个 SM 中执行。这允许线程块内的线程高效地相互通信和同步。线程块内的所有线程都可以访问片上共享内存，该内存可用于在线程块的线程之间交换信息。

网格可能由数百万个线程块组成，而执行网格的 GPU 可能只有几十或几百个 SM。
线程块的所有线程由单个 SM 执行，在大多数情况下 [#fn-non-completion]_ 在该 SM 上运行至完成。
线程块之间没有调度顺序保证，因此线程块不能依赖其他线程块的结果，因为它们可能无法在该线程块完成之前被调度。
:numref:`fig-thread-block-scheduling` 显示了网格中的线程块如何分配给 SM 的示例。

.. _fig-thread-block-scheduling:
.. figure:: /_static/images/thread-block-scheduling.png
   :alt: 线程块在 SM 上调度
   :width: 80%
   :align: center

   每个 SM 有一个或多个活动线程块。在此示例中，每个 SM 同时调度了三个线程块。对于网格中的线程块分配给 SM 的顺序没有保证。

CUDA 编程模型使任意大小的网格能够在任何大小的 GPU 上运行，无论它只有一个 SM 还是数千个 SM。
为了实现这一点，CUDA 编程模型（除一些例外）要求不同线程块中的线程之间不存在数据依赖性。
也就是说，线程不应依赖同一网格中不同线程块的线程的结果或与之同步。
线程块内的所有线程在同一时间在同一 SM 上运行。
网格中的不同线程块在可用 SM 之间调度，可以以任何顺序执行。
简而言之，CUDA 编程模型要求能够以任何顺序、并行或串行执行线程块。

.. _thread-block-clusters:

1.2.2.1.1. 线程块集群
""""""""""""""""""""""

除了线程块，计算能力 (compute capability) 9.0 及更高的 GPU 还有一个可选的分组级别，称为 *集群*。
集群是一组线程块，像线程块和网格一样，可以以一维、二维或三维布局。
:numref:`fig-thread-block-clusters` 说明了一个组织成集群的线程块网格。
指定集群不会更改网格维度或线程块在网格中的索引。

.. _fig-thread-block-clusters:
.. figure:: /_static/images/grid-of-clusters.png
   :alt: 线程块网格中的集群
   :width: 80%
   :align: center

   当指定集群时，线程块在网格中的位置不变，但在包含集群中也有一个位置。

指定集群将相邻的线程块分组为集群，并在集群级别提供一些额外的同步和通信机会。具体来说，集群中的所有线程块在单个 GPC 中执行。
:numref:`fig-thread-block-scheduling-with-clusters` 显示了指定集群时线程块如何调度到 GPC 中的 SM。
由于线程块同时调度并在单个 GPC 内，因此不同块但在同一集群内的线程可以使用 :ref:`writing-cuda-kernels-cooperative-groups` 提供的软件接口相互通信和同步。
集群中的线程可以访问集群中所有块的共享内存，这称为 :ref:`writing-cuda-kernels-distributed-shared-memory`。集群的最大大小取决于硬件，因设备而异。

:numref:`fig-thread-block-scheduling-with-clusters` 说明集群内的线程块如何在 GPC 内的 SM 上同时调度。集群内的线程块在网格内始终彼此相邻。

.. _fig-thread-block-scheduling-with-clusters:
.. figure:: /_static/images/thread-block-scheduling-with-clusters.png
   :alt: 线程块在集群中的 GPC 上调度
   :width: 80%
   :align: center

   当指定集群时，集群中的线程块按其集群形状在网格中排列。集群的线程块在单个 GPC 的 SM 上同时调度。

.. _warps-and-simt:

1.2.2.2. 线程束和 SIMT
^^^^^^^^^^^^^^^^^^^^^^^

在线程块内，线程被组织成 32 个线程的组，称为 *线程束* (warp)。线程束以 *单指令多线程* (Single-Instruction Multiple-Threads, SIMT) 范式执行核函数代码。
在 SIMT 中，线程束中的所有线程执行相同的核函数代码，但每个线程可以遵循代码中的不同分支。
也就是说，虽然程序的所有线程执行相同的代码，但线程不需要遵循相同的执行路径。

当线程由线程束执行时，它们被分配一个线程束通道。
线程束通道编号为 0 到 31，线程块中的线程以可预测的方式分配给线程束，详见 :ref:`hardware-multithreading`。

线程束中的所有线程同时执行相同的指令。
如果线程束中的某些线程在执行中遵循控制流分支，而其他线程不遵循，则不遵循分支的线程将被屏蔽，而遵循分支的线程被执行。
例如，如果条件仅对线程束中一半的线程为真，则线程束的另一半将被屏蔽，而活动线程执行这些指令。
这种情况如 :numref:`fig-active-warp-lanes` 所示。
当线程束中的不同线程遵循不同的代码路径时，这有时称为线程束分化 (warp divergence)。
因此，当线程束内的线程遵循相同的控制流路径时，GPU 的利用率最大化。

.. _fig-active-warp-lanes:
.. figure:: /_static/images/active-warp-lanes.png
   :alt: 不活动时屏蔽线程束通道
   :width: 80%
   :align: center

   在此示例中，只有具有偶数线程索引的线程执行 if 语句体，其他线程在执行体时被屏蔽。

在 SIMT 模型中，线程束中的所有线程同步推进核函数执行。
硬件执行可能有所不同。有关此区别重要性的更多信息，请参见 :ref:`independent-thread-scheduling` 部分。
不建议利用线程束执行如何实际映射到真实硬件的知识。
CUDA 编程模型和 SIMT 表示线程束中的所有线程一起推进代码。
只要遵循编程模型，硬件可以以对程序透明的方式优化屏蔽通道。
如果程序违反此模型，则可能导致未定义行为，在不同的 GPU 硬件中可能不同。

虽然编写 CUDA 代码时不需要考虑线程束，但了解线程束执行模型有助于理解 :ref:`writing-cuda-kernels-coalesced-global-memory-access` 和 :ref:`writing-cuda-kernels-shared-memory-access-patterns` 等概念。
一些高级编程技术使用线程块内的线程束专门化来限制线程分化并最大化利用率。此优化和其他优化利用了线程在执行时被分组为线程束的知识。

线程束执行的一个含义是，最好将线程块指定为具有总数为 32 的倍数的线程。
使用任何数量的线程都是合法的，但当总数不是 32 的倍数时，线程块的最后一个线程束将有一些在整个执行过程中未使用的通道。
这可能导致该线程束的功能单元利用率和内存访问欠佳。

.. topic:: SIMT 与 SIMD 的比较

   SIMT 通常与单指令多数据 (Single Instruction Multiple Data, SIMD) 并行性进行比较，但存在一些重要差异。
   在 SIMD 中，执行遵循单个控制流路径，而在 SIMT 中，每个线程允许遵循自己的控制流路径。
   因此，SIMT 没有像 SIMD 那样的固定数据宽度。
   有关 SIMT 的更详细讨论，请参见 :ref:`simt-execution-model`。

.. _programming-model-tile:

1.2.2.3. CUDA 中的 Tile 编程
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

除了前面章节描述的 SIMT 模型外，CUDA 还支持 Tile 编程模型。
在 Tile 编程中，程序员在整个线程块的级别编写代码，描述对称为 **tile** 的多维数据集合的操作。
编译器将这些操作映射到线程块的各个线程。

Tile kernel 在线程块网格上启动，如 :ref:`thread-blocks-and-grids` 部分所述。
每个 block 执行 tile kernel，并可以查询其在网格中的位置以确定其负责的数据部分。
程序员仅指定网格维度；每个 block 的线程数由编译器根据 kernel 中的 tile 操作确定（:numref:`fig-tile-programming-abstraction`）。

.. _fig-tile-programming-abstraction:
.. figure:: /_static/images/tile-simt.png
   :alt: SIMT 和 Tile 编程模型中程序员的视角
   :width: 80%
   :align: center

   SIMT 和 Tile 编程模型中程序员的视角。在 SIMT 中，程序员编写每线程代码并控制每个线程如何访问数据。
   在 Tile 编程中，程序员编写每 block 代码，对 tile 进行操作；编译器将操作映射到 block 的线程。

在 tile kernel 中，block 执行单个控制流。
程序员指定对 tile 的操作，编译器将工作分发到 block 的各个线程。
支持标准的控制流构造（如条件语句和循环），但由于 block 遵循单个控制流，因此不存在线程束分化的概念。
标量操作（如计算索引或循环边界）由 block 的单个线程执行。
Tile 操作（如逐元素相加两个 tile）由 block 的所有线程并行执行。

重要的是不要将 block（执行单元）与 tile（数据单元）混淆。
单个 block 可以创建和操作许多不同形状和数据类型的 tile。


.. _tile-arrays-and-tiles:

1.2.2.3.1. 数组和 Tile
"""""""""""""""""""""""

Tile kernel 使用两种类型的数据： **数组** 和 **tile**。
数组（或全局数组）是存储在设备内存中的多维元素容器。
数组是可变的：其内容可以通过 kernel 中的存储操作修改。数组具有形状和数据类型。

Tile 是仅存在于 tile 代码中且仅限于单个 block 的多维值集合。
Tile 是不可变的：对 tile 的每次操作都会生成新的 tile，而不是修改现有的 tile。
与数组不同，tile 不一定在内存中有表示——编译器决定 tile 数据的存储方式，可以使用寄存器、共享内存或 SM 的其他资源。
Tile 的每个维度必须是 2 的幂，并且必须在编译时已知（即其值必须在 kernel 执行之前确定，而不是在执行期间计算）。
Tile 不能作为 kernel 参数传递；它们完全在 tile 代码中创建和使用。


.. _tile-space-and-data-movement:

1.2.2.3.2. Tile 空间和数据移动
"""""""""""""""""""""""""""""""""

数据通过加载和存储操作在数组和 tile 之间移动。
这些操作使用一个称为 **tile 空间** 的概念，它是将数组概念性地分区为大小相等、不重叠的 tile 的结果。
例如，考虑形状为 (M, N) 的二维数组。
如果加载操作指定 tile 形状为 (tm, tn)，则数组被概念性地划分为 :math:`\lceil M/t_m \rceil` 行和 :math:`\lceil N/t_n \rceil` 列的 tile。
此 tile 空间中的索引（如 (i, j)）标识要加载的 tile。
加载返回形状为 (tm, tn) 的 tile，包含数组中对应的元素。
当 tile 超出数组边界时（例如在数组维度不是 tile 维度的精确倍数时的边缘处），由加载指定如何处理越界元素，例如用零填充（ :numref:`fig-tile-space-data-movement` ）。

.. _fig-tile-space-data-movement:
.. figure:: /_static/images/tile-data-movement.png
   :alt: Tile 空间和数据移动
   :width: 80%
   :align: center

   Tile 空间和数据移动。
   形状为 (M, N) 的二维数组被概念性地分区为形状为 (tm, tn) 的 tile 网格。
   tile 空间索引 (i, j) 处的加载返回对应的 tile。
   在数组边界处，落在数组外的元素可以用零填充。
   存储将 tile 写回到给定 tile 空间索引处的数组。

存储操作执行相反的操作：给定一个 tile 和 tile 空间中的索引，它将 tile 的元素写入数组的对应区域。
落在数组边界外的任何写入都被静默丢弃。
Tile 程序还支持 gather 和 scatter 操作，可以从数组中的任意位置加载或存储。


.. _tile-operations:

1.2.2.3.3. Tile 上的操作
"""""""""""""""""""""""""

Tile 程序提供一组作用于 tile 的内建操作，包括逐元素算术、矩阵乘法、沿一个或多个轴的归约（如求和和最大值）、形状操作（如 reshape 和转置）以及类型转换。当两个不同形状的 tile 在一个操作中组合时，较小的 tile 会自动扩展以匹配较大的 tile，然后再应用操作。


.. _tile-relationship-to-simt:

1.2.2.3.4. 与 SIMT 编程的关系
"""""""""""""""""""""""""""""""

Tile 编程和 SIMT 编程在 CUDA 中共存。应用程序可以同时包含 SIMT 和 tile kernel，两种类型的 kernel 都可以在设备内存中对相同数据进行操作。编程模型的选择是每个 kernel 的决策。Tile 编程不替代 SIMT 编程。SIMT 提供对单个线程的细粒度控制，这对于某些算法和优化技术仍然是必要的。Tile 编程提供更高层次的抽象，可以简化 kernel 开发。由于线程级决策留给编译器，相同的 tile kernel 可以在不同的 GPU 架构上运行而无需更改源代码。两种模型都构建在前面章节描述的相同底层硬件（SM、线程块和网格）之上。两种模型也使用相同的设备内存空间，这些将在下一节中介绍。


.. _gpu-memory:

1.2.3. GPU 内存
---------------

在现代计算系统中，高效利用内存与最大化使用执行计算的功能单元同样重要。
异构系统有多个内存空间，GPU 除了缓存外还包含各种类型的可编程片上内存。以下章节更详细地介绍这些内存空间。

.. _dram-memory-in-heterogeneous-systems:

1.2.3.1. 异构系统中的 DRAM 内存
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GPU 和 CPU 都有直接连接的 DRAM 芯片。在有多个 GPU 的系统中，每个 GPU 都有自己的内存。
从设备代码的角度来看，连接到 GPU 的 DRAM 称为 *全局内存*，因为它可以被 GPU 中的所有 SM 访问。
此术语并不意味着它一定可以在系统内的任何地方访问。连接到 CPU 的 DRAM 称为 *系统内存* 或 *主机内存*。

像 CPU 一样，GPU 使用虚拟内存寻址。在所有当前支持的系统上，CPU 和 GPU 使用统一的虚拟内存空间。
这意味着系统中每个 GPU 的虚拟内存地址范围是唯一的，并且与 CPU 和系统中每个其他 GPU 不同。
对于给定的虚拟内存地址，可以确定该地址是在 GPU 内存还是系统内存中，以及在多 GPU 系统中，哪个 GPU 内存包含该地址。

有 CUDA API 用于分配 GPU 内存、CPU 内存，以及在 CPU 和 GPU 之间的分配复制、GPU 内部或 GPU 之间复制。可以根据需要显式控制数据的局部性。
下面讨论的 :ref:`unified-memory` 允许 CUDA 运行时或系统硬件自动处理内存放置。

.. _on-chip-memory-in-gpus:

1.2.3.2. GPU 中的片上内存
^^^^^^^^^^^^^^^^^^^^^^^^^

除了全局内存，每个 GPU 都有一些片上内存。每个 SM 都有自己的寄存器文件和共享内存。这些内存是 SM 的一部分，可以从 SM 内执行的线程极快地访问，但不能被其他 SM 中运行的线程访问。

寄存器文件存储线程局部变量，通常由编译器分配。共享内存可以由线程块或集群内的所有线程访问。共享内存可用于在线程块或集群的线程之间交换数据。

SM 中的寄存器文件和统一数据缓存大小有限。
SM 的寄存器文件大小、统一数据缓存以及统一数据缓存如何在 L1 和共享内存之间分配可以在 :ref:`compute-capabilities-table-memory-information-per-compute-capability` 中找到。
寄存器文件、共享内存空间和 L1 缓存在线程块中的所有线程之间共享。

要将线程块调度到 SM，每个线程所需的寄存器总数乘以线程块中的线程数必须小于或等于 SM 中的可用寄存器。
如果线程块所需的寄存器数超过 SM 中寄存器文件的大小，则核函数无法启动，必须减少线程块中的线程数。

共享内存分配在线程块级别完成。也就是说，与每个线程的寄存器分配不同，共享内存的分配对整个线程块是通用的。

.. _caches:

1.2.3.2.1. 缓存
"""""""""""""""

除了可编程内存，GPU 还有 L1 和 L2 缓存。每个 SM 都有一个 L1 缓存，它是统一数据缓存的一部分。
更大的 L2 缓存由 GPU 内的所有 SM 共享。这可以在 :numref:`fig-gpu-cpu-system-diagram` 中的 GPU 框图中看到。
每个 SM 还有一个单独的 :ref:`writing-cuda-kernels-constant-memory`，用于缓存全局内存中已声明在核函数生命周期内保持常量的值。
编译器也可以将核函数参数放入常量内存。这可以通过允许核函数参数在 SM 中单独缓存（与 L1 数据缓存分开）来提高核函数性能。

.. _unified-memory:

1.2.3.3. 统一内存
^^^^^^^^^^^^^^^^^

当应用程序在 GPU 或 CPU 上显式分配内存时，该内存只能被在该设备上运行的代码访问。
也就是说，CPU 内存只能从 CPU 代码访问，GPU 内存只能从在 GPU 上运行的核函数访问 [#fn-mapped-memory-system-access]_。
用于在 CPU 和 GPU 之间复制内存的 CUDA API 用于在正确的时间将数据显式复制到正确的内存。

一个名为 *统一内存* (unified memory) 的 CUDA 功能允许应用程序进行可以从 CPU 或 GPU 访问的内存分配。
CUDA 运行时或底层硬件在需要时启用访问或将数据重定位到正确的位置。
即使使用统一内存，最佳性能也是通过将内存迁移保持在最低限度并尽可能从直接连接到内存所在位置的处理器访问数据来实现的。

系统的硬件特性决定了内存空间之间如何实现数据访问和交换。:ref:`memory-unified-memory` 部分介绍了不同类别的统一内存系统。
:ref:`um-details` 部分包含有关在所有情况下使用统一内存及其行为的更多详细信息。

.. rubric:: 脚注

.. [#fn-non-completion] 在使用 :ref:`cuda-dynamic-parallelism` 等功能的某些情况下，线程块可能会被挂起到内存。
   这意味着 SM 的状态存储到 GPU 内存的系统管理区域，SM 被释放以执行其他线程块。这类似于 CPU 上的上下文切换。这并不常见。

.. [#fn-mapped-memory-system-access] 此规则的一个例外是 :ref:`memory-mapped-memory`，它是分配了使 GPU 可以直接访问的属性的 CPU 内存。
   但是，映射访问通过 PCIe 或 NVLINK 连接进行。
   GPU 无法隐藏更高的延迟和更低的带宽，因此映射内存不是统一内存或将数据放置在适当内存空间的高性能替代方案。