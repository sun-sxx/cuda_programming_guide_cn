.. _writing-tile-kernels:

2.4. 编写 Tile Kernels
========================

CUDA Tile（瓦片、分块） 提供了一种与前面章节所讲的 SIMT 模型截然不同的 GPU kernel 编写方式。
Tile 编程允许程序员以不同的方式来描述并行性，并将最底层的并行细节交给编译器和内置操作去处理。
通过这种方式，Tile 提供了一种更简单的方式来使用 NVIDIA GPU 的最新性能特性，例如 :ref:`张量内存加速器 TMA 单元<async-copies-tma-multi-dim>` 和张量核心（Tensor Cores）。

- CUDA Tile 编程在 Python 中通过 cuTile Python 包 ``cuda.tile`` 提供。

- CUDA Tile C++ 从 CUDA Toolkit 13.3 版本开始提供。

围绕 Tile kernel 的应用代码（比如分配设备内存、在主机和设备之间传输数据，以及制定 kernel 的启动顺序）与前几章为 SIMT 核函数所描述的内容完全相同。
Tile kernel 操作的是通过标准 CUDA API 分配的全局内存，其计算结果也以同样的方式拷贝回主机。
唯一发生改变的，是程序员在 kernel 内部所编写的代码。

在 SIMT kernel 中，程序员是以单个线程为单位来思考的：需要计算全局线程索引，加载该线程对应的元素，对其执行操作，然后存储结果。
而在 Tile kernel 中，程序员则是以整个线程块（block）为单位来思考的：加载包含许多元素的图块（tile），对整个图块执行操作，存储结果。
编译器会负责将 tile 操作映射到每个块的硬件线程上，而这正是 SIMT 程序员需要显式处理的问题。

本章将完全聚焦于这一差异：即如何编写 kernel 的入口点以及其中的图块（tile）操作。
每一种模式都会分别用 CuTile Python (``cuda.tile``) 和 CUDA Tile C++ (``cuda::tiles``) 来进行演示。
这两者共享同一个编译器后端（CUDA Tile IR），因此也拥有完全相同的执行语义。

按照惯例，tile API 在两种语言中都别名为 ``ct``。

- Python 中：``import cuda.tile as ct``

- C++ 中：``namespace ct = cuda::tiles``

在 Python 中，tile API 位于模块 ``cuda.tiles`` 中，按上述方式导入。

在 C++ 中，tile API 位于 ``cuda::tiles`` 命名空间中，由 ``cuda_tile.h`` 头文件暴露。

.. code-block:: c++

   #include "cuda_tile.h"
   namespace ct = cuda::tiles;

下面代码片段中的 ``ct.`` / ``ct::`` 前缀指的是所阅读语言中的 tile API。


.. _writing-tile-kernels-kernel-and-function-declarations:

2.4.1. Kernel 和函数声明
--------------------------

Tile kernel 是 GPU 的入口点，它在启动网格中会针对每一个线程块执行一次。
而 Tile 函数可以由 Tile kernel 或其他 Tile 函数调用，但它本身并不是一个入口点。
与 SIMT kernel 一样，Tile kernel 也不能直接从主机代码中调用；它们必须通过 :ref:`启动 <writing-tile-kernels-launching-kernels>` 的方式来运行。

在 CUDA Tile C++ 中：

- ``__tile_global__`` 是 ``__global__`` 在 Tile 编程中的对应物，它用来标记一个 Tile kernel 入口。
- ``__tile__`` 是 ``__device__`` 在 Tile 编程中的对应物，它表示该函数应当被编译为 GPU 代码，并且可以被其他的 ``__tile__`` 或 ``__tile_global__`` 函数调用。

数组和标量参数的传递方式与 SIMT kernel 完全相同。
Tile 代码和 SIMT 代码可以共存。
单个 ``.cu`` 文件里可以同时定义 ``__tile_global__`` 和 ``__global__`` 两种核函数，同一主机程序可以同时启动它们。

.. note::

   目前，``__tile__`` 函数不能被 ``__global__`` 或 ``__device__`` 函数调用。
   同样，``__device__`` 函数不能被 ``__tile_global__`` 或 ``__tile__`` 函数调用。
   此限制可能在 CUDA 未来版本的中取消。

在 cuTile Python 中：

- ``@ct.kernel`` 装饰器标记函数为 tile kernel 入口点
- ``@ct.function`` 装饰器标记该函数是一个 tile 函数，可被 tile kernel 或其他 tile 函数调用。

实际上，任何从 tile kernel 内部调用的函数都会自动被编译为 Tile 代码，因此 ``@ct.function`` 装饰器是可选的。
数组参数可以接受任何支持 DLPack 或 CUDA Array Interface 的设备端数组，例如 PyTorch 张量和 CuPy 数组。
标量参数则是直接传递的。

.. tab-set::

   .. tab-item:: C++

      .. code-block:: cuda

         #include "cuda_tile.h"

         // Tile kernel entry point. Cannot be called directly; must be launched.
         __tile_global__ void my_kernel(float* a, float* b, float* c) {
             ...
         }

         // Tile function. Callable from tile kernels and tile functions.
         __tile__ float helper(float x, float y) {
             return x + y;
         }

   .. tab-item:: Python

      .. code-block:: python

         import cuda.tile as ct

         # Tile kernel entry point. Cannot be called directly; must be launched.
         @ct.kernel
         def my_kernel(a, b, c):
             ...

         # Tile function. Callable from tile kernels and tile functions.
         # @ct.function is optional, any function called from tile code
         # is automatically compiled as tile code.
         @ct.function
         def helper(x, y):
             return x + y


.. _writing-tile-kernels-launching-kernels:

2.4.2. 启动 Kernels
---------------------

Tile kernel 在由 tile block 组成的 grid 上启动，就像 SIMT kernel 在 thread block 组成的 grid 上启动一样。
程序员指定 grid 形状，最多三个维度。从程序员的角度来看，每个 tile block 由单个逻辑线程执行。Tile block 内的并行性由编译器管理。

在 C++ 中，Tile kernel 复用了 SIMT 编程中大家非常熟悉的三尖括号 ``<<<...>>>`` 启动语法。
第一个尖括号内的参数是网格形状（即 Tile block 的数量）。
第二个参数在 SIMT 中代表每个块的线程数量；但对于 Tile kenrel，编译器会在内部自动确定线程数，因此第二个参数 **必须固定为** 1。
Tile kernel 本质上也是一个普通的 CUDA kernel，
所以它也可以通过现有的 runtime API（如 ``cudaLaunchKernel`` 和 ``cudaLaunchKernelEx`` ）来启动，并且同样采用“网格形状 + 1”的配置方式。
当需要将 Tile kernel 集成到那些已经使用这些 API 的现有代码库时，这一点会非常有用。

在 Python 中，``ct.launch`` 接受四个位置参数：CUDA 流、指定每个维度 tile block 数量的 grid 元组、kernel 对象和 kernel 参数元组。

.. tab-set::

   .. tab-item:: C++

      .. code-block:: cuda

         my_kernel<<<dim3(num_blocks_x, num_blocks_y), 1>>>(a, b, c);  // second arg must be 1

   .. tab-item:: Python

      .. code-block:: python

         import torch

         stream = torch.cuda.current_stream()     # CUDA stream object
         grid = (num_blocks_x, num_blocks_y, 1)   # tile-block grid (x, y, z)
         ct.launch(stream, grid, my_kernel, (a, b, c))


2.4.2.1. Grid 大小设置方法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

一种常见的模式是启动足够多的 block 来覆盖整个数组，包括在一个或多个维度上可能超出数组大小的最后一个 block。

.. tab-set::

   .. tab-item:: C++

      .. code-block:: cuda

         int num_blocks = (N + tile_size - 1) / tile_size;   // ceil division -> covers partial tail
         kernel<<<num_blocks, 1>>>(in, out, N);

   .. tab-item:: Python

      .. code-block:: python

         import math

         grid = (math.ceil(N / TILE),)   # ceil division -> covers partial tail
         ct.launch(stream, grid, my_kernel, (arr_in, arr_out, TILE))

数组大小不能被 tile 大小整除时的处理方式在 :ref:`writing-tile-kernels-loading-and-storing-tiles` 的中讨论。


.. _writing-tile-kernels-querying-block-position:

2.4.3. 查询 Block 位置
------------------------

每个 block 需要知道它在 grid 中的位置，以便确定要处理的数据部分。
在 SIMT 中，程序员结合 ``blockIdx`` 和 ``threadIdx`` 来计算全局线程索引。
在 tile 代码中，只需要 block 索引。 Block 内的线程级索引由编译器处理 。

在 C++ 中， ``ct::bid()`` 返回 ``uint3`` 类型的 block 索引（含所有三个维度） 。
``ct::num_blocks()`` 返回 ``dim3`` 类型 grid 大小（由 kernel 启动参数确定， 包含每个维度的 block 数）。
通过 ``.x``、 ``.y`` 、 ``.z`` 访问各个分量。

在 Python 中，``ct.bid(axis)`` 返回当前 block 沿给定轴（0、1 或 2）的索引，作为 ``int32`` 标量。``ct.num_blocks(axis)`` 返回沿该轴的总 block 数——对于边界检查和循环计数很有用。

.. tab-set::

   .. tab-item:: C++

      .. code-block:: c++

         #include "cuda_tile.h"

         __tile_global__ void my_kernel(float* a, float* b, float* c) {
             namespace ct = cuda::tiles;
             int bid_x = ct::bid().x;          // block index along .x
             int bid_y = ct::bid().y;          // block index along .y
             int num_x = ct::num_blocks().x;   // total blocks along .x
         }

   .. tab-item:: Python

      .. code-block:: python

         @ct.kernel
         def my_kernel(a, b, c):
             bid_x = ct.bid(0)          # block index along axis 0
             bid_y = ct.bid(1)          # block index along axis 1
             num_x = ct.num_blocks(0)   # total blocks along axis 0


.. _writing-tile-kernels-creating-tiles:

2.4.4. 创建 Tiles
-------------------

在确定了 block 的标识之后，接下来的问题就是：Tile kernel 到底是在操作什么？
答案就是 tile（瓦片）：它是一个固定大小的多维标量元素数组，其形状和元素类型在编译时就已经确定。
tile 的每个维度都必须是 2 的幂。
Tile 具有值语义，这意味着复制一个 tile 会连同它的元素一起复制，并且这两个副本是完全独立的。
尽管如此，复制操作的开销依然很小，因为编译器能够控制 tile 在硬件内部是如何被表示的。
程序员不需要为 tile 手动分配或释放内存。

在实际应用中，tile 要么通过从数组中加载数据来创建（即 :ref:`Tile-Space 加载与存储<writing-tile-kernels-tile-space-loads-and-stores>` ），
要么使用生成指定模式填充 tile 的工厂函数来创建。

在 C++ 中，tile 类型是显式的： ``ct::tile<T, ct::shape<dims...>>`` ，其中 ``T`` 是元素类型，
``ct::shape<dims...>`` 将维度编码为模板参数（整数值是沿每个轴的编译时大小）。
例如， ``ct::tile<float, ct::shape<8>>`` 是包含 8 个 float 的 1 维 tile，
``ct::tile<float, ct::shape<4, 4>>`` 是 4×4 的 float tile。
由于形状是类型的一部分，它始终在编译时已知。

工厂函数将完整的 tile 类型（下面的 ``Tile``）作为模板参数：

- ``ct::zeros<Tile>()`` 和 ``ct::ones<Tile>()`` — 填充 0 或 1 的 tile。

- ``ct::full<Tile>(val)`` — 每个元素值为 ``val`` 的 tile。

- ``ct::iota<Tile>()`` — 包含 ``(0, 1, ..., N-1)`` 的 tile，其中 ``N`` 是 tile 的大小。

C++ 示例在本章中使用 ``using`` 别名（例如 ``using f32x4x4 = ct::tile<float, ct::shape<4, 4>>``）来保持 tile 类型在调用点的可读性。

在 Python 中，传递给 tile 工厂函数的 ``shape`` 元组和 ``dtype`` 参数都必须是编译期的常量值。
Python 的字面量（比如 ``(64, 64)`` 和 ``ct.float32`` ）自然就满足要求。
此外，也可以通过使用带有 ``Constant`` 注解的核函数参数来提供这些值，如下方 :ref:`Python Constant[T] <writing-tile-kernels-compile-time-constants-python>` 所示。
最终生成的 tile 会暴露出 ``.shape`` 、 ``.dtype`` 和 ``.ndim``，以反映其编译期的特征。

工厂函数为：

- ``ct.zeros(shape, dtype)`` 和 ``ct.ones(shape, dtype)`` — 填充零或一的 tile。

- ``ct.full(shape, fill_value, dtype)`` — 具有任意常量值的 tile。

- ``ct.arange(size, dtype=...)`` — 包含 ``[0, 1, ..., size-1]`` 的 1 维 tile。

.. tab-set::

   .. tab-item:: C++

      .. code-block:: c++

         #include "cuda_tile.h"

         __tile__ void factories() {
             namespace ct = cuda::tiles;

             using i32x8   = ct::tile<int,   ct::shape<8>>;      // 1-D: 8 ints
             using f32x4x4 = ct::tile<float, ct::shape<4, 4>>;   // 2-D: 4x4 floats

             auto z      = ct::zeros<f32x4x4>();       // all zeros
             auto o      = ct::ones<f32x4x4>();        // all ones
             auto filled = ct::full<f32x4x4>(3.14f);   // all 3.14
             auto seq    = ct::iota<i32x8>();          // {0, 1, 2, 3, 4, 5, 6, 7}
         }

   .. tab-item:: Python

      .. code-block:: python

         import cuda.tile as ct

         @ct.function
         def factories():
             zeros  = ct.zeros((64, 64), dtype=ct.float32)            # 64x64 tile of 0.0
             ones   = ct.ones((128,), dtype=ct.float16)               # 128-element tile of 1.0
             filled = ct.full((32, 32), 3.14, dtype=ct.float32)       # 32x32 tile of 3.14
             seq    = ct.arange(8, dtype=ct.int32)                    # [0, 1, 2, 3, 4, 5, 6, 7]


.. _writing-tile-kernels-compile-time-constants:

2.4.5. 编译时常量
-------------------

编译器会为每一种 tile 形状、数据类型以及其他结构参数的组合，生成专门优化的机器码。
因此，所有会影响最终生成代码的参数值，都必须在编译时就已经确定。
也就是说，tile 的形状和数据类型必须在编译时已知。
之前 :ref:`创建 Tile<writing-tile-kernels-creating-tiles>` 时，我们正是使用了字面量来指定 tile 的形状和数据类型的。
比如前面示例 Python 里的 ``ct.zeros((64, 64), dtype=ct.float32)`` 和 C++ 里的 ``ct::tile<int, ct::shape<8>>`` 。

Shape 也可以通过 kernel 的接口传递进来，但前提是这些值在编译时就必须已知，具体方法将在接下来的章节中展示。


.. _writing-tile-kernels-compile-time-constants-python:

2.4.5.1. Python Constant[T]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 Tile kernel 参数上使用 ``ct.Constant[T]`` 提示，会将其标记为 **常量嵌入** （constant-embedded）。
这意味着在 kernel 内部，对该参数的每一次使用，其行为都等同于直接把字面量写在了那里。
类型参数是可选的；如果不带类型参数直接使用 ``ct.Constant`` ，则会嵌入任意类型的常量。
``ct.Constant`` 最常用于整数即 ``ct.Constant[int]`` ，作用于那些决定 tile 形状和循环边界的参数上。

.. code-block:: python

   import cuda.tile as ct

   @ct.kernel
   def my_kernel(TILE: ct.Constant[int]):
       # TILE is constant-embedded: wherever TILE appears, the compiler sees its
       # literal value (e.g., 128) and generates specialized code. Here TILE drives
       # the shape of a factory-built tile.
       zeros = ct.zeros((TILE,), dtype=ct.float32)


2.4.5.2. C++ integral_constant 和 _ic 字面量
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 CUDA Tile C++ 中，编译期的值是通过 ``ct::integral_constant`` 来表达的。
这是一种特殊的类型，它的数值被直接编码在了类型本身之中。
而来自 ``ct::literals`` 命名空间的 ``_ic`` 字面量则提供了一种简洁的简写方式：比如 ``0_ic`` 就能直接生成一个 ``ct::integral_constant<0>`` 的值。

那些接收编译期值的 API，既支持非类型模板参数（non-type template parameter, NTTP）的形式，也支持 ``_ic`` 字面量的形式。
举个例子， ``ct::cat`` 的作用是将两个 tile 沿着指定的维度进行拼接，而这个维度必须在编译期就确定下来。
下面这两行代码调用 ``ct::cat`` 时使用的是同一个编译期轴；它们的唯一区别，仅仅在于这个编译期值写在了哪里：

.. code-block:: c++

   #include "cuda_tile.h"

   __tile__ void concat_demo() {
       namespace ct = cuda::tiles;
       using namespace ct::literals;

       using T = ct::tile<int, ct::shape<4, 8>>;
       T lhs = ct::full<T>(0);
       T rhs = ct::full<T>(1);

       auto a = ct::cat<0>(lhs, rhs);     // NTTP form
       auto b = ct::cat(lhs, rhs, 0_ic);  // _ic form
   }

``_ic`` 字面量还有一个经常出现的场合。
``ct::extents`` 和 ``ct::shape`` 这两个类，各自有 NTTP 形式（比如 ``ct::extents<std::uint32_t, 4, 8>`` ），以及一种大括号 ``{}`` 的形式。
与 NTTP 形式不同的是，大括号形式可以接收运行时的值。
因此，当有一个或多个维度只有在程序启动时才能确定时，就需要使用这种大括号形式。
其中编译期确定的维度用 ``_ic`` 字面量表示，而运行时才确定的维度则直接使用普通的变量。
Tile 空间 API 如 ``ct::tensor_span`` 和 ``ct::partition_view`` （见 :ref:`Tile 空间加载和存储 <writing-tile-kernels-tile-space-loads-and-stores>` ）正是使用这种形式来封装此类数组的：

.. code-block:: c++

   auto shape2d = ct::extents{8_ic, length};  // 8 is compile-time; length is runtime

在任何需要传入 **值形式** API参数的地方（比如 ``ct::cat`` 的维度参数，或者 ``extents`` 和 ``shape`` 的某个组成部分），
只要该位置要求的是一个编译期常量， ``_ic`` 字面量就是统一且便捷的简写方式。

.. _writing-tile-kernels-loading-and-storing-tiles:

2.4.6. 加载和存储 Tiles
-------------------------

正如 :ref:`tile-arrays-and-tiles` 介绍的那样，CUDA Tile 编程模型中有两种关键的内存对象：tile 和 array 。
Array 是位于全局内存中的一个多维元素容器，对于 tile kernel 的所有线程块都是可见的。
Tile 同样也是一个多维元素容器，但仅服务于（is local to） CUDA tile 代码的单个 block（即每个 block 处理一个 tile）。
通常情况下，tile 是某个 array 中元素的子集。
本节将讨论如何将数据从 array 加载到 tile 中以便在 tile kernel 中使用，以及如何将 tile 中的数据存回 array。

后续章节介绍两种加载和存储 tile 的方法：

- :ref:`Tile-space 加载和存储 <writing-tile-kernels-tile-space-loads-and-stores>` 主要讲解了如何使用 Tile 空间索引（tile-space indices）加载和存储数据。
  这会用到视图对象（view objects），视图对象提供可预测的映射模式，明确 array 中的元素是如何对应到各个 Tile 上的。

- :ref:`Gather 和 Scatter <writing-tile-kernels-gather-and-scatter>` 介绍使用索引或指针的分块来指示加载或存储时 array 中哪个元素作为 tile 元素的源或目标。

**性能说明**： 在支持 TMA 的硬件上， Tile-space 加载可以由编译器自动优化，这比逐元素 gather 快得多。
（对于 C++ 端，另请参见 :ref:`C++ 性能提示 <writing-tile-kernels-cpp-perf-tips>`。）

程序员必须决定在加载数据时，那些越界的元素应该被赋予什么值。
而在写入数据时，越界操作会被静默丢弃——这在 Python 中是默认行为，在 C++ 中使用带掩码的（存储）接口时也是如此。

.. _writing-tile-kernels-tile-space-loads-and-stores:

2.4.6.1. Tile-space 加载和存储
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在使用 tile-space 加载时，系统会创建一个视图对象，用来指定如何将一个数组划分成网格状的、大小等同于 tile 的区域。
这种映射关系被称为 **tile-space** （tile 空间坐标系）。
Ttile kernel 可以通过 tile 空间索引，每次加载或存储其中一个区域。

Tile-space 加载理念的核心在于 array 的分块视图（tiled view），它规定了 array 中的元素如何映射到指定大小的 tile 上。
图 19 所示的分区视图，这是一种特殊的分块空间（坐标系），其中的 tile 具有指定的大小、彼此互不重叠，且 tile 之间没有任何间隙。

.. _fig-cutile-tile-space-indexing:
.. figure:: /_static/images/cutile-tile-space-indexing.png
   :alt: 将一个 10×16 的数组划分为 2×4 的分块后的分块空间索引
   :width: 80%

   图 19：分区视图的分块空间索引。
   一个形状为 ``(10, 16)`` 的二维数组被划分为大小为 ``(2, 4)`` 的分块，从而生成一个形状为 ``(5, 4)`` 的分块网格。
   每个单元格展示了其在分块空间中的索引 ``(i, j)`` 。
   图中高亮显示的区域位于分块空间索引 ``(1, 2)`` 处，它覆盖了从元素索引 ``(2, 8)`` 到 ``(3, 11)`` 的数据范围。

当数组的维度无法被分块（Tile）大小整除时，跨越数组边界的一个或多个维度上的分块将会被部分填充。
程序员可以指定加载这些分块时的处理行为，相关内容将在 :ref:`writing-tile-kernels-boundary-handling-tile-space-loads-and-stores` 节中介绍。

.. note::

   此处的示例和描述使用分区视图来说明 tile 空间加载和存储，因为这是 CUDA Tile 代码中支持的第一种视图类型。预计后续版本的 CUDA Tile 将添加其他视图类型。


2.4.6.1.1. 分区视图加载和存储
""""""""""""""""""""""""""""""""

结构化 tile 空间加载是在全局内存和 tile 之间移动数据的首选方式。
Kernel 必须首先构建定义 tile 空间的视图对象，然后通过其 tile 空间索引一次加载或存储一个 tile。

在 C++ 中，分区视图分两步构建：

- ``ct::tensor_span`` — 将原始指针与 ``ct::extents`` 配对，为指针提供多维结构。

- ``ct::partition_view`` — 将 span 划分为固定大小 tile 的 grid，并提供 ``.load(idx...)`` / ``.store(tile, idx...)`` 方法，在 tile 空间坐标中操作。

在 Python 中，``Array.tiled_view(tile_shape)`` 返回一个 ``TiledView``，将数组分区为给定形状的 tile。
视图提供 ``.load(index)`` / ``.store(index, tile)`` 方法，接受 tile 空间索引，直接镜像 C++ ``partition_view``。

.. note::

   本章中的 C++ 示例代码使用 ``__restrict__`` 注解指针参数，并在 kernel 入口附近调用 ``ct::assume_aligned(ptr, 16_ic)`` 。
   这是一个编译性能提示，即向编译器担保数组 arr 在内存中已经按照 16 字节对齐，从而允许编译器生成更高效的底层指令。
   这将在 :ref:`writing-tile-kernels-cpp-perf-tips` 中进一步介绍。
   数字字面量上的 ``_ic`` 后缀（例如 ``128_ic``、 ``8_ic``）将它们标记为编译时常量，见 :ref:`编译时常量 <writing-tile-kernels-compile-time-constants>` 。

.. tab-set::

   .. tab-item:: C++

      .. code-block:: c++

         __tile_global__ void vec_add(float* __restrict__ a, 
                                      float* __restrict__ b, 
                                      float* __restrict__ out) {
             namespace ct = cuda::tiles;
             using namespace ct::literals;

             a   = ct::assume_aligned(a,   16_ic);
             b   = ct::assume_aligned(b,   16_ic);
             out = ct::assume_aligned(out, 16_ic);

             // Step 1: attach a shape to each raw pointer. 
             // 128_ic marks 128 as a compile-time constant.
             auto aSpan = ct::tensor_span{a,   ct::extents{128_ic}};
             auto bSpan = ct::tensor_span{b,   ct::extents{128_ic}};
             auto oSpan = ct::tensor_span{out, ct::extents{128_ic}};

             // Step 2: partition each span into a tile space of fixed 8-element tiles.
             auto aView = ct::partition_view{aSpan, ct::shape{8_ic}};
             auto bView = ct::partition_view{bSpan, ct::shape{8_ic}};
             auto oView = ct::partition_view{oSpan, ct::shape{8_ic}};

             int  bx    = ct::bid().x;       // this block's tile-space index along .x
             auto aTile = aView.load(bx);    // pick the bx-th tile of a
             auto bTile = bView.load(bx);
             oView.store(aTile + bTile, bx); // write the tile back at the bx-th position of out
         }

   .. tab-item:: Python

      .. code-block:: python

         @ct.kernel
         def vec_add(a, b, c, TILE: ct.Constant[int]):
             a_view = a.tiled_view((TILE,))
             b_view = b.tiled_view((TILE,))
             c_view = c.tiled_view((TILE,))

             bid = ct.bid(0)
             a_tile = a_view.load((bid,))
             b_tile = b_view.load((bid,))
             c_view.store((bid,), a_tile + b_tile)


2.4.6.1.2. Python 单次调用加载和存储
""""""""""""""""""""""""""""""""""""""

Python 还额外提供单次调用形式，在每次加载和存储时内联提供 tile 形状，无需显式视图对象。
``ct.load(array, index, shape)`` 在给定 tile 空间索引处读取给定形状的 tile。``ct.store(array, index, tile)`` 是对应的写入。

``ct.load``/``ct.store`` 和 ``Array.tiled_view`` 都表达相同的 tile 空间访问模式。
区别在于 tile 形状的位置。使用 ``Array.tiled_view`` 时，tile 形状一次性绑定到视图对象。
使用 ``ct.load``/``ct.store`` 时，tile 形状在每次调用时内联提供。
当同一分区在多个加载和存储中重用时，首选使用 ``tiled_view``。当单次一次性加载更简洁时，使用 ``ct.load``/``ct.store``。

.. code-block:: python

   @ct.kernel
   def vec_add(a, b, c, TILE: ct.Constant[int]):
       bid = ct.bid(0)                                   # this block's tile-space index along axis 0
       a_tile = ct.load(a, index=(bid,), shape=(TILE,))  # (index, shape) = pick the bid-th TILE-sized region of a
       b_tile = ct.load(b, index=(bid,), shape=(TILE,))
       ct.store(c, index=(bid,), tile=a_tile + b_tile)   # write the tile back to the bid-th region of c


.. _writing-tile-kernels-boundary-handling-tile-space-loads-and-stores:

2.4.6.1.3. Tile 空间边界处理
""""""""""""""""""""""""""""""

在 C++ 中，``partition_view`` 提供非掩码和掩码接口：

- ``.load(idx...)`` ， ``.store(tile, idx...)`` 假设 tile 完全在边界内。部分越界访问是未定义行为。

- ``.load_masked(idx...)`` ， ``.store_masked(tile, idx...)`` 安全处理部分边缘 tile。

  - ``.load_masked()`` 默认用零填充越界位置；可以指定填充模式（如 ``float`` 类型填充 ``NaN`` ）。

  - ``.store_masked()`` 静默丢弃越界写入。

当数组大小能被 tile 大小完美整除时，优先推荐使用无掩码接口。
而当必须处理边界条件时，即使对于数据完全填满的分块，也可以使用带掩码接口。

这也是本指南中第一个数组维度为运行时值（runtime value）的 C++ 示例。
``ct::extents{N}`` 接受一个运行时维度，而 ``ct::extents`` 支持编译时值（ ``_ic`` ）和运行时值的任意组合。
因此， ``span`` 和 ``partition view`` 可以包装那些只有在启动内核时才能确定大小的数组。

在 Python 中， ``ct.load`` 接受一个 ``padding_mode`` （填充模式）的参数，用于控制越界元素将被赋予什么值。两种常用的模式如下：

- ``PaddingMode.ZERO`` — 越界元素填充为零。

- ``PaddingMode.UNDETERMINED`` （默认）— 越界元素的值由底层实现自行决定。当程序员明确知道分块完全在合法边界内时，使用此模式是合适的。

对于存储，``ct.store`` 始终静默丢弃对越界位置的写入，不需要 ``padding_mode`` 参数。
相同的规则适用于 ``tiled_view``，它在视图创建时固定其 ``padding_mode``。

.. tab-set::

   .. tab-item:: C++

      .. code-block:: c++

         __tile_global__ void edge_safe(float* __restrict__ in, float* __restrict__ out, int N) {
             namespace ct = cuda::tiles;
             using namespace ct::literals;

             in  = ct::assume_aligned(in,  16_ic);
             out = ct::assume_aligned(out, 16_ic);

             // ct::extents{N} uses a runtime dimension; 128_ic stays compile-time.
             auto inView  = ct::partition_view{ct::tensor_span{in,  ct::extents{N}}, ct::shape{128_ic}};
             auto outView = ct::partition_view{ct::tensor_span{out, ct::extents{N}}, ct::shape{128_ic}};

             int  bx   = ct::bid().x;
             auto tile = inView.load_masked(bx);    // masked load: OOB lanes default to 0
             outView.store_masked(tile, bx);        // masked store: OOB writes silently discarded
         }

   .. tab-item:: Python

      .. code-block:: python

         @ct.kernel
         def edge_safe(arr_in, arr_out, TILE: ct.Constant[int]):
             bid = ct.bid(0)
             tile = ct.load(arr_in, index=(bid,), shape=(TILE,),
                            padding_mode=ct.PaddingMode.ZERO)  # OOB lanes of a partial edge tile become 0
             ct.store(arr_out, index=(bid,), tile=tile)        # OOB writes are silently discarded

在 C++ kernel 内部， ``.load_masked()`` 和 ``.store_masked()`` 用于处理位于边缘的不完整分块（partial edge tile）。
而在 Python kernel 代码中，加载操作时的 ``PaddingMode.ZERO`` 可确保对边缘不完整分块进行零填充；
同时， ``ct.store`` 会自动静默丢弃超出数组边界的写入操作。
如需了解完整的填充模式、掩码选项及填充值设定，请参阅各语言对应的 API 参考文档（
`CUDA Tile C++ 视图填充机制 <https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/constant_wrappers_and_flags.html#view-padding>`_ ，
以及 `cuTile Python 的填充模式 <https://docs.nvidia.com/cuda/cutile-python/data.html#padding-modes>`_ ）。

对完全位于数组外部的分块进行加载或存储，其行为是未定义的（undefined）。
这里所讨论的边界处理机制，仅适用于在一个或多个维度上部分越界的分块。

.. _writing-tile-kernels-gather-and-scatter:

2.4.6.2. Gather 和 Scatter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`Tile 空间加载和存储 <writing-tile-kernels-tile-space-loads-and-stores>` 中 tile 的加载依赖于分块视图，它定义了 array 的一种规则且按块对齐的划分方式。
然而，当访问模式是不规则的或依赖于数据的（例如查表操作或数据重排）时， Gather 和 Scatter 操作则允许通过任意的索引或地址，
从 array 的非均匀、非连续元素中加载 tile，或将 tile 存储到这些位置。

Gather 和 scatter 操作在 C++ 和 Python 中略有不同：

- 在 Python 中，通过向 ``ct.gather()`` / ``ct.scatter()`` 传递整数索引分块（tiles of integer index）来实现该操作，并且系统内置了边界检查机制。

- C++ 通过向 ``ct::load()`` / ``ct::store()`` 传递指针分块（tiles of pointers ）。
  提供掩码版本 ``ct::load_masked()`` 和 ``ct::store_masked()``， 接受布尔掩码块 （tile of boolean mask）来处理数组边界处的 tile。

在 C++ 中，Gather 和 Scatter 操作是通过构建一个指针 Tile（每个元素对应一个指针），并将该指针 Tile 传递给 ``ct::load()`` 或 ``ct::store()`` 来实现的。
将一个标量指针与整数 tile 进行算术运算时，系统会按元素逐个计算，从而生成一个指针 tile。
这是在 C++ 中构造 Gather/Scatter 索引 Tile 的标准惯用写法。

在 Python 中， ``ct.gather`` 加载索引 tile 中每个索引处的元素。
边界检查默认开启：越界索引返回填充值（默认为零，可通过 ``padding_value=`` 配置），可以使用 ``check_bounds=False`` 禁用。
``ct.scatter`` 每个索引存储一个值；越界写入被静默丢弃。

.. tab-set::

   .. tab-item:: C++

      .. code-block:: c++

         __tile_global__ void vec_add_gather(int* __restrict__ a, 
                                             int* __restrict__ b, 
                                             int* __restrict__ out) {
             namespace ct = cuda::tiles;
             using namespace ct::literals;
             using i32x8 = ct::tile<int, ct::shape<8>>;

             a   = ct::assume_aligned(a,   16_ic);
             b   = ct::assume_aligned(b,   16_ic);
             out = ct::assume_aligned(out, 16_ic);

             int bx       = ct::bid().x;
             auto offsets = 8 * bx + ct::iota<i32x8>();  // element-level offsets, one per lane

             // scalar pointer + int tile = tile of pointers (one pointer per offset).
             auto aPtrs = a + offsets;
             auto bPtrs = b + offsets;

             auto aTile = ct::load(aPtrs);                // gather: one load per pointer
             auto bTile = ct::load(bPtrs);
             ct::store(out + offsets, aTile + bTile);     // scatter: one store per pointer
         }

   .. tab-item:: Python

      .. code-block:: python

         @ct.kernel
         def vec_add_gather(a, b, c, TILE: ct.Constant[int]):
             bid = ct.bid(0)
             indices = bid * TILE + ct.arange(TILE, dtype=ct.int32)  # one element index per lane

             a_tile = ct.gather(a, indices)                          # load a[indices[i]] per lane
             b_tile = ct.gather(b, indices)
             ct.scatter(c, indices, a_tile + b_tile)                 # store one value per index into c

.. note::

   Tile （分块）本身就是一个多维数组，相对于 array 来说， 它是一个大的多维数组的一个分块。 
   C++ 中使用的 **指针 Tile** ， 就是一个多维数组，每个元素都是一个指针，指向 arry 中的某个元素。
   Python 中使用 **索引 Tile** ，也是一个多维数组，每个元素存放一个索引，索引 arry 中某个元素的位置。
   在这个章节的代码示例中， 输入数据是一维数组。所以在取 bid 的时候，都是取的 0 轴。

.. _writing-tile-kernels-boundary-handling-gather-and-scatter:

2.4.6.2.1. Gather 和 Scatter 边界处理
""""""""""""""""""""""""""""""""""""""""

:ref:`writing-tile-kernels-gather-and-scatter` 中介绍的 gather/scatter 操作的边界处理遵循不同的规则。

在 Python 中， ``ct.gather`` 和 ``ct.scatter`` 默认是边界安全的。
越界读取会返回一个填充值（默认为零），而越界写入则会被静默丢弃。
如果你能证明所有索引都在有效范围内，可以选择禁用边界检查；但这样做会使越界访问变成未定义行为。
关于可选的掩码和填充值控制参数，
请参阅 API 参考文档（ `CUDA Tile C++ 加载操作 <https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/memory_operations.html#load-operations>`_ ，
`cuTile Python 加载/存储操作 <https://docs.nvidia.com/cuda/cutile-python/operations.html#load-store>`_ ）。

在 C++ 中，边界检查并不是自动进行的。
程序员需要自己构建一个布尔掩码（例如，通过将偏移量与数组长度进行比较），然后将其传递给 ``ct::load_masked`` 或 ``ct::store_masked`` 函数。

.. code-block:: c++

   __tile_global__ void gather_safe(int* __restrict__ arr, int* __restrict__ out, int N) {
       namespace ct = cuda::tiles;
       using namespace ct::literals;
       using i32x8 = ct::tile<int, ct::shape<8>>;

       arr = ct::assume_aligned(arr, 16_ic);
       out = ct::assume_aligned(out, 16_ic);

       int bx       = ct::bid().x;
       auto offsets = 8 * bx + ct::iota<i32x8>();   // element-level offsets, one per lane
       auto mask    = offsets < N;  // boolean tile: true where the offset is in-bounds

       auto ptrs = arr + offsets;                   // tile of pointers, one per offset
       auto tile = ct::load_masked(ptrs, mask, 0);  // masked lanes get the pad value 0
       ct::store_masked(out + offsets, tile, mask); // masked lanes are skipped on the store
   }


.. _writing-tile-kernels-control-flow:

2.4.7. 控制流
---------------

从程序员的角度来看，tile kernel 在每个线程块中只遵循一条单一的控制流路径。
条件和循环边界中的标量值驱动着控制流，而循环体内的 tile 操作则由编译器自动分配到各个硬件线程上。

并非所有的控制流结构都能得到支持。
例如，在 tile 代码中不允许从循环内部直接返回。
完整的限制列表，请参阅各语言的 API 参考文档
（ `CUDA Tile C++ 通用原则 <https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/general_principles.html>`_ 、
`cuTile Python 控制流 <https://docs.nvidia.com/cuda/cutile-python/execution.html#control-flow>`_ ）。

2.4.7.1. 循环
~~~~~~~~~~~~~~

一种常见的模式是遍历数组中的各个 tile ，依次对它们进行处理

在 C++ 中， ``ct::irange`` 是一个前向范围，它表示一个 ``[下界,上界)`` 的递增整数序列，步长可选。
使用 ``ct::irange`` 能为编译器提供关于迭代边界的结构化信息，从而有助于更好地优化生成的代码。
要使该优化生效，循环变量必须通过基于 ``ct::irange`` 的范围 ``for(range-for)`` 表达式来绑定。

在 Python 中，内建的 ``range()``、 ``for`` 、 ``while`` 和嵌套循环都在 tile 代码中受支持。

步长参数必须严格为正；不支持负步长范围。

以下是一个单 block kernel 对 1D 数组的所有 tile 求和的示例：

.. tab-set::

   .. tab-item:: C++

      .. code-block:: c++

         __tile_global__ void tile_sum(float* __restrict__ arr, 
                                       float* __restrict__ out, 
                                       int num_tiles) {
             namespace ct = cuda::tiles;
             using namespace ct::literals;
             using f32x8 = ct::tile<float, ct::shape<8>>;

             arr = ct::assume_aligned(arr, 16_ic);
             out = ct::assume_aligned(out, 16_ic);

             auto inView  = ct::partition_view{ct::tensor_span{arr, ct::extents{8 * num_tiles}},
                                              ct::shape{8_ic}};
             auto outView = ct::partition_view{ct::tensor_span{out, ct::extents{8_ic}},
                                              ct::shape{8_ic}};

             auto acc = ct::full<f32x8>(0.0f);
             // range-for over ct::irange gives the compiler structured iteration bounds.
             for (auto k : ct::irange(0, num_tiles)) {
                 auto tile = inView.load(k);
                 acc = acc + tile;  // accumulate the k-th tile into acc
             }
             outView.store(acc, 0);  // write the final result as the 0-th tile of out
         }

   .. tab-item:: Python

      .. code-block:: python

         @ct.kernel
         def tile_sum(arr, out, TILE: ct.Constant[int], N_TILES: ct.Constant[int]):
             # Intended grid: (1,) -- a single block sums all tiles of arr.
             acc = ct.zeros((TILE,), dtype=ct.float32)
             for k in range(N_TILES):  # range() works natively in tile code
                 tile = ct.load(arr, index=(k,), shape=(TILE,))
                 acc = acc + tile  # accumulate the k-th tile into acc
             ct.store(out, index=(0,), tile=acc)  # write the final result as the 0-th tile of out


2.4.7.2. 条件语句
~~~~~~~~~~~~~~~~~~

标准的 ``if/else`` 条件语句可以正常工作。
由于每个线程块都遵循单一的控制流路径，因此传统编程中关于 :ref:`warp 内的分支发散 <warps-and-simt>` 的顾虑在 tile kernel 中不再适用。

.. tab-set::

   .. tab-item:: C++

      .. code-block:: c++

         __tile_global__ void conditional_load(float* __restrict__ arr, 
                                               float* __restrict__ out, 
                                               int N) {
             namespace ct = cuda::tiles;
             using namespace ct::literals;
             using f32x8 = ct::tile<float, ct::shape<8>>;

             arr = ct::assume_aligned(arr, 16_ic);
             out = ct::assume_aligned(out, 16_ic);

             auto inView  = ct::partition_view{ct::tensor_span{arr, ct::extents{N}}, ct::shape{8_ic}};
             auto outView = ct::partition_view{ct::tensor_span{out, ct::extents{N}}, ct::shape{8_ic}};

             int bx   = ct::bid().x;
             int nb_x = ct::num_blocks().x;

             auto tile = ct::full<f32x8>(0.0f);  // default for the last-block branch
             // Scalar condition -> one control-flow path per block; no divergence to reason about.
             if (bx < nb_x - 1) {
                 tile = inView.load(bx);  // all blocks except the last
             }
             outView.store_masked(tile, bx);  // masked to handle a potentially partial final tile
         }

   .. tab-item:: Python

      .. code-block:: python

         @ct.kernel
         def conditional_load(arr, out, TILE: ct.Constant[int]):
             bid = ct.bid(0)
             # Scalar condition -> one control-flow path per block; no divergence to reason about.
             if bid < ct.num_blocks(0) - 1:
                 tile = ct.load(arr, index=(bid,), shape=(TILE,))  # all blocks except the last
             else:
                 tile = ct.zeros((TILE,), dtype=ct.float32)  # last block: emit zeros
             ct.store(out, index=(bid,), tile=tile)


.. _writing-tile-kernels-elementwise-arithmetic-and-broadcasting:

2.4.8. 逐元素算术运算与广播机制
-------------------------------

Tile 支持标准的逐元素算术运算。
当两个操作数的形状兼容但不同时，在执行运算之前，较小的那个会被广播以匹配较大的形状。

2.4.8.1. 广播
~~~~~~~~~~~~~~

广播遵循 NumPy 的语义规则：标量会在整个 tile 中被复制；长度为 1 的单例维度会被拉伸，以匹配另一个操作数对应的维度；
对于低秩（rank）操作数，其缺失的前导维度会被视为单例维度，从而使其与高秩操作数的尾部维度对齐。
如果两个对应的维度都不是单例维度且大小不相等，则该运算是不合法的。

下面的示例在单次加法运算中同时演示了单例维度拉伸和秩提升：一个秩为 2、形状为 ``8x2`` 的 tile 首先被提升为秩-3，即 ``1x8x2`` ；
然后与另一个形状为 ``4x1x2`` 的三维瓦片进行广播，最终得到共同的形状 ``4x8x2`` 。

.. tab-set::

   .. tab-item:: C++

      .. code-block:: c++

         auto x = ct::iota<ct::tile<int, ct::shape<8, 2>>>();  // 8x2 (rank 2)
         auto y = ct::iota<ct::tile<int, ct::shape<4, 1, 2>>>();  // 4x1x2 (rank 3)
         auto z = x + y;  // x promoted to 1x8x2, then stretched to 4x8x2

   .. tab-item:: Python

      .. code-block:: python

         x = ct.full((8, 2),    3, dtype=ct.int32)   # 8x2   (rank 2)
         y = ct.full((4, 1, 2), 5, dtype=ct.int32)   # 4x1x2 (rank 3)
         z = x + y                                   # x promoted to 1x8x2, then broadcasts to 4x8x2


2.4.8.2. 算术运算符
~~~~~~~~~~~~~~~~~~~~

所有支持的算术运算符都会对 tile 进行逐元素（element-wise）运算，并生成一个具有广播后形状的 tile 。
当标量与瓦片结合时，该标量会被广播到每一个元素上。
当操作数的数据类型不同时，系统会优先保留信息量更大的那种类型（类型自动提升）：

- **Tile 与 tile 运算**：结果是具有更高精度或范围的类型的 tile。例如：

  - ``int + float`` -> ``float``
  - ``int16 + int32`` -> ``int32``

- **标量与 tile 运算**：当标量的类型可以使用 tile 的元素类型中时表达（例如整数字面量 ``2`` 与 ``int`` 类型的 tile， 或 ``2.0f`` 与 ``float`` 类型的 tile 运算），
  操作结果为 tile 的元素类型。当标量必须缩窄以适应 tile 的元素类型时（例如字面量 ``2.5`` 与 ``int`` 类型的 tile 运算），两种语言有所不同：

  - Python 将结果提升为可以容纳两者的类型
  - C++ 判定表达式非法

下面的代码片段说明了不同的标量-tile 情况：

.. tab-set::

   .. tab-item:: C++

      .. code-block:: c++

         using i32x8 = ct::tile<int, ct::shape<8>>;
         i32x8 x = ct::full<i32x8>(3);

         x + 2;       // OK - int literal matches int tile element type
         x + 2.5;     // ill-formed - 2.5 would narrow to int

   .. tab-item:: Python

      .. code-block:: python

         x = ct.full((8,), 3, dtype=ct.int32)

         x + 2          # int32 - int literal matches int32 tile dtype
         x + 2.5        # float32 - result promoted to hold both

在实践中，尽量使用与 Tile 元素类型相同的标量字面量；如果需要使用不同的精度，则应进行显式转换。
当操作数是已加载的 tile 时，在 kernel 内部同样适用这些规则：

.. tab-set::

   .. tab-item:: C++

      .. code-block:: c++

         __tile_global__ void elementwise(float* __restrict__ a, 
                                          float* __restrict__ b, 
                                          float* __restrict__ out, 
                                          int N) {
             namespace ct = cuda::tiles;
             using namespace ct::literals;

             a   = ct::assume_aligned(a,   16_ic);
             b   = ct::assume_aligned(b,   16_ic);
             out = ct::assume_aligned(out, 16_ic);

             auto aView = ct::partition_view{ct::tensor_span{a,   ct::extents{N}}, ct::shape{8_ic}};
             auto bView = ct::partition_view{ct::tensor_span{b,   ct::extents{N}}, ct::shape{8_ic}};
             auto cView = ct::partition_view{ct::tensor_span{out, ct::extents{N}}, ct::shape{8_ic}};

             int  bx = ct::bid().x;
             auto x  = aView.load(bx);
             auto y  = bView.load(bx);
             // 2.0f matches the float tiles' element type, so no narrowing conversion is required.
             // The scalar is broadcast across every element; + then runs elementwise.
             auto z  = 2.0f * x + y;
             cView.store(z, bx);
         }

   .. tab-item:: Python

      .. code-block:: python

         @ct.kernel
         def elementwise(a, b, c, TILE: ct.Constant[int]):
             bid = ct.bid(0)
             x = ct.load(a, index=(bid,), shape=(TILE,))
             y = ct.load(b, index=(bid,), shape=(TILE,))
             # 2.0 is a loosely typed float constant; with float tiles, the result stays float.
             # Scalars are broadcast across every element of the tile, then + runs elementwise.
             z = 2.0 * x + y
             ct.store(c, index=(bid,), tile=z)

当你需要对舍入模式（rounding mode）或非规格化数处理（subnormal handling）进行显式控制时，
CUDA Tile API 提供了相应的 :ref:`数学函数<writing-tile-kernels-mathematical-functions>` （如 ``ct.add`` 、 ``ct::add`` ），允许你将这些选项作为参数传入。

.. _writing-tile-kernels-tile-primitives:

2.4.9. Tile 原语
------------------

工厂函数（:ref:`创建 Tiles <writing-tile-kernels-creating-tiles>`）、
加载和存储（:ref:`Tile 空间加载和存储 <writing-tile-kernels-tile-space-loads-and-stores>`）
以及逐元素数学运算（:ref:`逐元素算术和广播 <writing-tile-kernels-elementwise-arithmetic-and-broadcasting>`）都是 *tile 原语*，即属于语言的操作。
程序员以 tile 为粒度来编写这些代码，随后由编译器将其映射到硬件上执行（包括在可用的情况下调用 Tensor Cores）。
本节将介绍 CUDA Tile 中提供的其他原语。

.. _writing-tile-kernels-matrix-multiply:

2.4.9.1. 矩阵乘法
~~~~~~~~~~~~~~~~~~~

两个 tile 矩阵乘法是实现数组间矩阵乘法的基石操作。
CUDA Tile 提供了两种 tile 间的矩阵乘法形式：纯矩阵乘法（matmul，即 ``a @ b`` ）和矩阵乘加运算（mma，即 ``a @ b + acc`` ）。
在 mma 中，累加器（accumulator）负责将部分乘积从一个 ``K-tile`` 传递到下一个 ``K-tile``。
这在分块矩阵乘法的内层循环中非常有用。
matmul 和 mma 均支持二维矩阵乘法、三维批量乘法，以及允许操作数与累加器之间混合使用不同的数据类型（精度）。
关于秩（rank）和数据类型的约束条件，请参阅相关操作的 API 参考文档（ `CUDA Tile C++ matrix multiplication <https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/matrix_multiplication.html>`_ , 
`cuTile Python matmul <https://docs.nvidia.com/cuda/cutile-python/operations.html#matmul>`_ ）。

一个常见的编程模式是（如下面的 kernel 示例）：无论输入数据的精度如何，都使用 FP32 进行累加运算，并在最终写入时再将其转换为目标输出的元素类型。
在 Python 中，这表现为 ``ct.mma(a, b, acc)`` ，使用 FP32 类型的 ``acc`` 。
在 C++ 中，则是 ``ct::mma(a, b, acc)`` ，并显式指定 ``acc`` 为 FP32 类型。
K 维度的循环会执行 ``ceil(K / tk)`` 次，以确保完整覆盖矩阵 A 的右边缘和矩阵 B 的下边缘。
对于不完整的 K-tile，会在加载时进行零填充处理（Python 中使用 ``PaddingMode.ZERO`` ，C++ 中使用 ``.load_masked()`` ）；
而对于输出端（C 侧）处于 M/N 边缘的不完整 tile ，则通过存储时的越界丢弃机制来处理（Python 中使用 ``ct.store`` ，C++ 中使用 ``.store_masked()``）。

.. tab-set::

   .. tab-item:: C++

      .. code-block:: c++

         __tile_global__ void gemm(const __half* __restrict__ A, 
                                   const __half* __restrict__ B, 
                                   float* __restrict__ C,
                                   std::size_t M, std::size_t K, std::size_t N) {
             namespace ct = cuda::tiles;
             using namespace ct::literals;
             using f32_acc = ct::tile<float, ct::shape<32, 32>>;

             A = ct::assume_aligned(A, 16_ic);
             B = ct::assume_aligned(B, 16_ic);
             C = ct::assume_aligned(C, 16_ic);

             constexpr auto tm = 32_ic;
             constexpr auto tn = 32_ic;
             constexpr auto tk = 16_ic;

             auto aView = ct::partition_view{ct::tensor_span{A, ct::extents{M, K}}, ct::shape{tm, tk}};
             auto bView = ct::partition_view{ct::tensor_span{B, ct::extents{K, N}}, ct::shape{tk, tn}};
             auto cView = ct::partition_view{ct::tensor_span{C, ct::extents{M, N}}, ct::shape{tm, tn}};

             auto [bx, by, bz] = ct::bid();
             auto acc = ct::full<f32_acc>(0.0f);                 // FP32 accumulator

             std::size_t num_k = (K + tk - 1) / tk;
             for (auto k : ct::irange(std::size_t{0}, num_k)) {
                 acc = ct::mma(aView.load_masked(bx, k),         // zero-pad partial K-tile
                               bView.load_masked(k, by),
                               acc);                             // acc += a @ b
             }
             cView.store_masked(acc, bx, by);                    // drop OOB edge lanes
         }

   .. tab-item:: Python

      .. code-block:: python

         @ct.kernel
         def gemm(A, B, C,
                  tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
             bx, by = ct.bid(0), ct.bid(1)
             num_k  = ct.num_tiles(A, axis=1, shape=(tm, tk))    # number of K-tiles

             acc = ct.full((tm, tn), 0, dtype=ct.float32)        # FP32 accumulator
             for k in range(num_k):
                 a = ct.load(A, index=(bx, k), shape=(tm, tk),
                             padding_mode=ct.PaddingMode.ZERO)   # zero-pad partial K-tile
                 b = ct.load(B, index=(k, by), shape=(tk, tn),
                             padding_mode=ct.PaddingMode.ZERO)
                 acc = ct.mma(a, b, acc)                         # acc += a @ b

             ct.store(C, index=(bx, by), tile=acc.astype(C.dtype))  # cast + store


.. _writing-tile-kernels-reductions-and-scans:

2.4.9.2. 归约和扫描
~~~~~~~~~~~~~~~~~~~~~

归约（Reduction）是一种将数据块压缩为单个标量或一行标量的工具。
例如计算 ``Softmax`` 的分母、层归一化（Layer Norm）的均值和方差，或是注意力机制评分中的最大值，这些过程都涉及归约操作。

首先需要牢记的一个关键点是运算结果的形状（shape）。
在 Python 中，默认会丢弃被归约的维度（如果需要保留该维度使其长度为 1，需传入 ``keepdims=True`` 参数）；
而在 C++ 中，系统始终会保留该维度，从而维持数据块（tile）原有的秩（rank/阶数）。
下面这两段代码都是沿着第 1 轴对一个 2x4 的数据块进行归约，其输出形状的差异是它们最直观的区别。

.. tab-set::

   .. tab-item:: C++

      .. code-block:: c++

         using namespace ct::literals;
         using i32x2x4 = ct::tile<int, ct::shape<2, 4>>;

         auto x = ct::iota<i32x2x4>();                         // [[0,1,2,3],[4,5,6,7]]
         auto row_sums = ct::sum(x, 1_ic);                     // shape (2, 1) - axis kept
         // row_sums == [[6], [22]]

   .. tab-item:: Python

      .. code-block:: python

         x   = ct.arange(8, dtype=ct.int32).reshape((2, 4))    # [[0,1,2,3],[4,5,6,7]]
         s   = ct.sum(x, axis=1)                               # shape (2,)    - axis dropped
         s_k = ct.sum(x, axis=1, keepdims=True)                # shape (2, 1)  - axis kept
         # s == [6, 22];  s_k == [[6], [22]]

前缀扫描则是与之对应的”逐步累积“操作，它会沿着指定的轴生成一个累积结果。
例如，前缀和（ ``cumsum`` ）生成的输出维度与输入完全相同，其中给定索引位置的值，等于沿指定轴从起始位置到该索引位置（包含该索引本身）所有元素的总和。
关于各语言中支持的完整操作集，请参阅 API 参考文档（ 
`CUDA Tile C++ reductions and scans <https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/reductions_and_scans.html>`_ ,
`cuTile Python reductions and scans <https://docs.nvidia.com/cuda/cutile-python/operations.html#scan>`_ ）。

2.4.9.3. 转置和重排
~~~~~~~~~~~~~~~~~~~~~

转置（Transpose）和重排（Permutation）可以在不改变底层数据的情况下，重新排列数据块的轴：其中 ``transpose`` 专门用于交换前两个轴，而 ``permute`` 则执行任意顺序的重排。
当数据块的逻辑布局需要在不同操作之间进行切换时，它们就会派上用场。
例如：具象化矩阵（materializing，注解：改变数据的物理内存布局，为硬件计算准备）乘法操作数的转置、在注意力机制模块中交换行与列，或者在执行广播（broadcast）操作前对齐各个轴。

在 Python 中，对秩为 2 的数据块调用 ``ct.transpose(x)`` 会直接交换它的两个轴；而对于更高阶的数据块，则需要显式传入 ``axis0`` 和 ``axis1`` 参数。
此外， ``ct.permute(x, axes)`` 接收一个包含各轴索引的元组作为参数。
在 C++ 中， ``ct::transpose(x)`` 仅互换前两个维度（其余尾部维度保持不变），而 ``ct::permute(x, map)`` 则接收一个用于描述新维度的顺序的 ``ct::dimension_map`` 对象。

.. tab-set::

   .. tab-item:: C++

      .. code-block:: c++

         using namespace ct::literals;
         using t2d = ct::tile<int, ct::shape<2, 4>>;
         using t3d = ct::tile<int, ct::shape<2, 2, 2>>;

         auto tx = ct::iota<t2d>();
         auto ty = ct::transpose(tx);                                     // shape (4, 2)

         auto tz = ct::iota<t3d>();
         auto tw = ct::permute(tz, ct::dimension_map{2_ic, 0_ic, 1_ic});  // axes (0,1,2) -> (2,0,1)

   .. tab-item:: Python

      .. code-block:: python

         tx = ct.arange(8, dtype=ct.int32).reshape((2, 4))
         ty = ct.transpose(tx)                                            # shape (4, 2)

         tz = ct.arange(8, dtype=ct.int32).reshape((2, 2, 2))
         tw = ct.permute(tz, (2, 0, 1))                                   # axes (0,1,2) -> (2,0,1)


2.4.9.4. 逐元素选择
~~~~~~~~~~~~~~~~~~~~

逐元素（Element-wise）选择是条件判断在数据块层面的表现形式：给定一个布尔型数据块和两个操作数数据块，系统会根据对应的布尔值，从这两个操作数中挑选其一作为每个输出元素的值。
在此过程中，条件数据块的形状会被广播以匹配操作数的形状；
同时，两个操作数的数据类型必须兼容（关于各语言的具体规则，请参阅 API 参考文档：
`CUDA Tile C++ select <https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/tile_operations.html#cuda-tiles-select>`_,
`cuTile Python selection <https://docs.nvidia.com/cuda/cutile-python/operations.html#selection>`_ ）。
在 Python 中，该操作表示为 ``ct.where(cond, x, y)`` ；而在 C++ 中则写为 ``ct::select(cond, lhs, rhs)`` 。

.. tab-set::

   .. tab-item:: C++

      .. code-block:: c++

         using namespace ct::literals;
         auto cond = ct::iota<ct::tile<int, ct::shape<4>>>() < 2;   // {T, T, F, F}
         auto t    = ct::full<ct::tile<float, ct::shape<4>>>( 1.0f);
         auto f    = ct::full<ct::tile<float, ct::shape<4>>>(-1.0f);
         auto r    = ct::select(cond, t, f);                        // {1, 1, -1, -1}

   .. tab-item:: Python

      .. code-block:: python

         cond    = ct.arange(4, dtype=ct.int32) < 2        # [T, T, F, F]
         x_true  = ct.full((4,),  1.0, dtype=ct.float32)
         x_false = ct.full((4,), -1.0, dtype=ct.float32)
         result  = ct.where(cond, x_true, x_false)         # [1, 1, -1, -1]

.. note::

   类似 C/C++ 中的三元操作。
   ``result  = ct.where(cond, x_true, x_false)`` 等价与  
   ``result[i][j] = cond[i][j] ? x_true[i][j] : x_false[i][j]``


.. _writing-tile-kernels-mathematical-functions:

2.4.9.5. 数学函数
~~~~~~~~~~~~~~~~~~

常见的逐元素数学运算可作为 ``ct`` 命名空间中的函数在 tile 代码中使用：

- ``add``、``sub``、``mul``

- ``truediv``、``floordiv``、``cdiv``

- ``mod``

- ``pow``

- ``exp``、``exp2``、``log``、``log2``

- ``sqrt``、``rsqrt``

- ``sin``、``cos``、``tan``

- ``sinh``、``cosh``、``tanh``

- ``minimum``、``maximum``

- ``negative``

- ``floor``、``ceil``

每个函数将其操作按元素应用于输入 tile 并返回相同形状的 tile。这些操作也适用于 tile 代码中的标量。

有关确切细节和支持的逐元素操作的完整列表，请参阅 API 参考：

- `cuTile Python 数学运算 <https://docs.nvidia.com/cuda/cutile-python/operations.html#math>`_

- `CUDA Tile C++ 数学运算 <https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/math_operations.html>`_


2.4.10. 原子内存操作
----------------------

在 tile 代码中有两种情况需要使用内存原子操作：

- 在 *跨 block 竞争* 中，每个 block 生成部分结果并使用原子操作将其与其他 block 的部分结果合并到全局内存位置中。

- 在 *block 内竞争* 中，tile 的多个元素被写入内存中的同一位置。

tile 上的原子操作对 tile 的 *每个元素* 执行一次原子更新。逐元素操作是原子的，但整个调用不是。逐元素原子操作的顺序是未指定的。

在 Python 中，原子操作通过数组的索引寻址目标，使用与 ``ct.gather`` 和 ``ct.scatter`` 相同的约定。可选参数控制边界检查、内存顺序和线程范围。默认值（边界检查开启、``ACQ_REL``、设备范围）允许普通调用仅传递数组、索引和更新值。``TiledView`` 还暴露相同的原子操作作为实例方法（例如 ``TiledView.atomic_add(index, update)``）；这些通过 tile 空间索引寻址目标，不返回值，并降低为 PTX 中的原子归约。当不需要先前值时，``TiledView`` 形式因更好的性能而被首选。

在 C++ 中，原子操作接受指针和对应的值：单个位置的原始指针和标量，或指针 tile 和值 tile。内存顺序是调用点的编译时类型标签，例如 ``ct::memory_order_relaxed_t{}``。线程范围是相同形式的类型标签，省略时默认为系统级可见性。


2.4.10.1. 跨 Block 竞争
~~~~~~~~~~~~~~~~~~~~~~~~~

在下面的代码示例中，跨 block 竞争发生是因为不同的 block 正在写入同一内存位置 ``out``。没有原子操作，并行运行的 block 将导致错误答案。在此示例中，使用设备线程范围（C++ 中为 ``ct::thread_scope_device_t{}``，Python 中线程范围默认为设备级范围），因为内存操作的结果必须对设备上运行的所有 block 可见。Python kernel 使用 ``TiledView.atomic_add``，因为每个 block 的部分和被累加到 ``out[0]`` 中并立即丢弃。

.. tab-set::

   .. tab-item:: C++

      .. code-block:: c++

         __tile_global__ void block_sum(int* __restrict__ arr, int* __restrict__ out, std::size_t N) {
             namespace ct = cuda::tiles;
             using namespace ct::literals;
             constexpr auto TILE = 16_ic;

             arr = ct::assume_aligned(arr, 16_ic);
             out = ct::assume_aligned(out, 16_ic);

             auto aView = ct::partition_view{ct::tensor_span{arr, ct::extents{N}},
                                             ct::shape{TILE}};
             int bid = ct::bid().x;
             auto tile    = aView.load_masked(bid);        // partial final tile -> OOB lanes default to 0
             auto partial = ct::sum(tile, 0_ic);           // reduce to a 1-element tile

             ct::atomic_add(out, (int)partial,             // accumulate the scalar into out[0]
                            ct::memory_order_relaxed_t{},  // single-location accumulator -> relaxed suffices
                            ct::thread_scope_device_t{});  // visible across the device
         }

   .. tab-item:: Python

      .. code-block:: python

         @ct.kernel
         def block_sum(arr, out, TILE: ct.Constant[int]):
             bid = ct.bid(0)
             # partial final tile -> OOB lanes default to 0
             tile    = ct.load(arr, index=(bid,), shape=(TILE,),
                               padding_mode=ct.PaddingMode.ZERO)
             partial = ct.sum(tile)                               # reduce to a scalar
             out.tiled_view((1,)).atomic_add((0,), partial)       # atomically accumulate into out[0]


2.4.10.2. Block 内竞争
~~~~~~~~~~~~~~~~~~~~~~~

在下面的代码片段中，block 内竞争发生是因为 tile 的所有值都被原子地添加到内存中的单个位置。

在此示例中，``ptrs`` tile 中的每个元素都指向内存中的同一位置 ``slot``。由 ``ct::iota<i32x16>()`` 创建的 tile 的每个元素被原子地添加到存储在该内存位置的值中。从 tile 到单个内存地址的多个原子操作的执行顺序是未指定的。使用 block 线程范围 ``ct::thread_scope_block_t{}`` 来指定原子操作的结果仅需在此 thread block 内可见。

.. code-block:: c++

   using i32x16 = ct::tile<int, ct::shape<16>>;

   int* slot = /* pointer to the contended location */;

   // 16 lanes all aim at the same address. Add is commutative, so the
   // unspecified ordering doesn't affect this sum; block scope suffices
   // since contention stays within one block.
   auto ptrs = ct::full<ct::tile<int*, ct::shape<16>>>(slot);
   ct::atomic_add(ptrs, ct::iota<i32x16>(),
               ct::memory_order_relaxed_t{},
               ct::thread_scope_block_t{});

.. note::

   这仅用于说明目的。要在 block 内将 tile 求和为标量，:ref:`第 2.4.9.2 节 <writing-tile-kernels-reductions-and-scans>` 中展示的 tile 归约操作是首选方法。


2.4.10.3. 支持的原子操作
~~~~~~~~~~~~~~~~~~~~~~~~~

Tile 代码支持多种原子内存操作，它们在写入值与内存中已有值的组合方式上有所不同：

- ``atomic_and`` — 在传递的值和内存中的值之间执行逐元素原子按位 AND

- ``atomic_or`` — 在传递的值和内存中的值之间执行逐元素原子按位 OR

- ``atomic_xor`` — 在传递的值和内存中的值之间执行逐元素原子按位 XOR

- ``atomic_max`` — 在传递的值和内存中的值之间执行逐元素比较，并将较大的值存储到内存中

- ``atomic_min`` — 在传递的值和内存中的值之间执行逐元素比较，并将较小的值存储到内存中

- ``atomic_add`` — 将传递的值加到内存中的值上，并将结果存储到内存中

- ``atomic_xchng`` — 将传递的值写入内存，并返回写入前内存中的值

- ``atomic_cas`` — 在内存中的值和作为参数传递的期望值之间执行逐元素比较。如果匹配，内存中的值被替换为期望值

有关所有支持的原子内存操作的完整文档，请参阅 `CUDA Tile C++ API 参考 <https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/memory_operations.html>`_ 或 `cuTile Python API 参考 <https://docs.nvidia.com/cuda/cutile-python/operations.html#atomic>`_ 的内存操作部分。


.. _writing-tile-kernels-optimization-hints:

2.4.11. 优化提示
------------------

优化提示是附加到源构造（tile kernel 函数、加载/存储调用站点等）的元数据，用于指导编译器的代码生成。提示不改变程序的语义：kernel 有无提示都编译和运行相同，因此可以自由添加、删除或调整而不影响正确性。编译器也可能忽略任何提示。

提示共享两个一般属性：

- **提示是按构造的。** 提示适用于它所附加的特定 kernel 函数或特定调用表达式，而非周围代码。

- **提示可以按架构指定。** 每个提示可以为不同的 GPU 架构设置为不同的值，或设置为适用于所有目标的单个值。

两种语言以不同方式暴露提示：

- C++ 使用放置在相关声明或语句上的 C++ 属性。

- Python 使用 kernel 装饰器和各个内存操作调用站点上的关键字参数。

提示种类集以及每个提示实际控制的内容在两种语言之间共享，记录在 :ref:`提示种类 <writing-tile-kernels-optimization-hints-kinds>` 部分中。


2.4.11.1. C++ — ``cutile::hint`` 属性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 C++ 中，提示通过 C++ 属性 ``cutile::hint`` 表达：

.. code-block:: text

   [[ cutile::hint(arch, kind1=value1, kind2=value2, ...) ]]

第一个参数是目标架构，使用与 ``__CUDA_ARCH__`` 宏相同的约定编码为整数（例如 ``900`` 表示 ``sm_90``，``1000`` 表示 ``sm_100``）。特殊值 ``0`` 表示 *架构无关* 提示，适用于每个目标架构。每个剩余参数是指定提示种类及其值的 ``kind=value`` 对。

``cutile::hint`` 属性适用于它前面的构造：

- 对于 tile kernel 函数，将属性放在函数声明上。

- 对于内存操作如 ``ct::load``、``ct::store`` 和 ``ct::partition_view`` 加载/存储，将其放在包含调用的表达式语句上。

下面的 kernel 同时说明了两种放置方式：一个为 ``sm_90`` 和 ``sm_100`` 设置不同 ``num_cta_in_cga`` 的 kernel 级提示，以及将特定加载标记为带宽密集型的表达式语句提示。

.. code-block:: c++

   [[ cutile::hint(900,  num_cta_in_cga=4),    // sm_90:  prefer 4 CTAs per cluster
      cutile::hint(1000, num_cta_in_cga=8) ]]  // sm_100: prefer 8 CTAs per cluster
   __tile_global__ void optimization_hints(float* __restrict__ in,
                                           float* __restrict__ out) {
       namespace ct = cuda::tiles;
       using namespace ct::literals;

       in  = ct::assume_aligned(in,  16_ic);
       out = ct::assume_aligned(out, 16_ic);

       auto inSpan  = ct::tensor_span{in,  ct::extents{128_ic}};
       auto outSpan = ct::tensor_span{out, ct::extents{128_ic}};
       auto inView  = ct::partition_view{inSpan,  ct::shape{8_ic}};
       auto outView = ct::partition_view{outSpan, ct::shape{8_ic}};

       int bx = ct::bid().x;

       // Expression-statement hint: tag this particular load as bandwidth-heavy.
       ct::tile<float, ct::shape<8>> tile;
       [[ cutile::hint(0, latency=8) ]]
       tile = inView.load(bx);

       outView.store(tile, bx);
   }

当同一种类的多个提示适用于同一构造时，架构特定的提示覆盖架构无关的提示。


2.4.11.2. Python — 装饰器参数和调用站点关键字
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python 以两种方式暴露提示：

- **Kernel 级提示** 是 ``@ct.kernel(...)`` 装饰器的关键字参数。编译的 kernel 对象还有 ``.replace_hints(**hints)`` 方法，返回具有覆盖提示的新 kernel；新 kernel 有自己的 JIT 缓存，使 ``replace_hints`` 成为自动调优循环的自然构建块。

- **每次调用提示** 是内存操作调用站点上的关键字参数：``ct.load`` / ``ct.store``、``TiledView.load`` / ``TiledView.store`` 和 ``ct.gather`` / ``ct.scatter``。

对于每架构值，将值包装在 ``cuda.tile.ByTarget(*, default=..., sm_XXX=..., sm_YYY=...)`` 中。架构键必须是 ``"sm_<major><minor>"`` 形式的字符串（例如 ``"sm_100"`` 或 ``"sm_120"``）。普通（非 ``ByTarget``）值适用于每个目标——它是 C++ 架构无关提示 ``arch=0`` 的 Python 等价物。


.. _writing-tile-kernels-optimization-hints-kinds:

2.4.11.3. 提示种类
~~~~~~~~~~~~~~~~~~~~

以下提示在两种语言之间共享。在每个提示中，**C++ 名称** 和 **Python 名称** 条目是同一底层提示的不同拼写；其余部分相同：提示适用的位置、其值、其含义。


2.4.11.3.1. 每个 cluster 的 CTA 数
""""""""""""""""""""""""""""""""""""

- **C++ 名称：**  ``num_cta_in_cga`` （kernel 属性）。

- **Python 名称：**  ``num_ctas`` （``@ct.kernel`` 装饰器参数）。

- **允许值：** ``1``、``2``、``4``、``8``、``16``。在 ``sm_80`` 上，仅 ``1`` 适用。

- **含义：** 编译器在启动 kernel 时应优先为每个协作组数组 (CGA) 分配的协作线程数组 (CTA) 数量。


2.4.11.3.2. 占用率
""""""""""""""""""""

- **C++ 名称：**  ``occupancy`` （kernel 属性）。

- **Python 名称：**  ``occupancy`` （``@ct.kernel`` 装饰器参数）。

- **允许值：** ``[1, 32]`` 闭范围内的任何整数。

- **含义：** 每个流式多处理器 (SM) 的目标活动 CTA 数量。编译器将该值视为建议，并将在代码生成期间尝试遵循它。


2.4.11.3.3. 内存访问延迟
""""""""""""""""""""""""""

- **C++ 名称：**  ``latency`` （包含调用的表达式语句上的属性）。

- **Python 名称：**  ``latency`` （调用站点上的关键字参数）。

- **适用于：** tile 空间加载和存储（C++ 中为 ``ct::partition_view``；Python 中为 ``Array.tiled_view`` 和 ``ct.load`` / ``ct.store``）以及 gather/scatter（C++ 中为带指针 tile 的 ``ct::load`` / ``ct::store``；Python 中为 ``ct.gather`` / ``ct.scatter``）。

- **允许值：** ``[1, 10]`` 闭范围内的任何整数，其中 ``1`` 表示轻量 DRAM 流量，``10`` 表示重流量。较大的值通常导致编译器调度更大的预取深度。


2.4.11.3.4. 允许 TMA
""""""""""""""""""""""

- **C++ 名称：**  ``allow_tma`` （包含调用的表达式语句上的属性）。

- **Python 名称：**  ``allow_tma`` （调用站点上的关键字参数）。

- **适用于：** 仅限 tile 空间加载和存储（C++ 中为 ``ct::partition_view``；Python 中为 ``Array.tiled_view`` 和 ``ct.load`` / ``ct.store``）。Gather 和 scatter 操作不接受此提示。

- **允许值：** ``true`` / ``false`` (C++) 或 ``True`` / ``False`` (Python)。默认允许 TMA；将提示设置为 ``false``/``False`` 指示编译器不要将此特定加载或存储降低为支持 TMA 的硬件上的 TMA。


.. _writing-tile-kernels-cpp-perf-tips:

2.4.12. C++ 性能提示
----------------------

本指南中的 C++ kernel 都使用相同的一些注解和惯用法。本节解释它们的作用以及为什么重要。


2.4.12.1. 对内存中的数组使用 ``__restrict__`` 指针
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``__restrict__`` 关键字告诉编译器，通过指针访问的内存区域在指针的生命周期内将仅通过该指针访问。参见 :ref:`第 5.4.1.4 节 <restrict-pointers>`。

在 Tile C++ 中，对符合这些条件的内存中的数组使用 ``__restrict__`` 关键字标记其指针对于良好的内存操作性能至关重要。

为了理解原因，考虑一个使用非 ``__restrict__`` 指针数组的逐元素复制：

.. code-block:: c++

   __tile_global__ void tile_elementwise_copy(float* out, float const* in) {
       namespace ct = cuda::tiles;

       using f32x64 = ct::tile<float, ct::shape<64>>;
       using i32x64 = ct::tile<int, ct::shape<64>>;

       auto inPtrs  = in  + 64 * ct::bid().x + ct::iota<i32x64>();
       auto outPtrs = out + 64 * ct::bid().x + ct::iota<i32x64>();

       auto data = ct::load(inPtrs);   // (1)
       ct::store(outPtrs, data);       // (2)
   }

编译器如何并行化 tile 操作通常可以在 CUDA Tile 程序中忽略。但是，我们将在此处考虑它以理解为什么使用非重叠数组能使编译器生成性能更好的代码。

考虑编译器如何并行化 ``load`` 和 ``store`` tile 操作。如果输入和输出数组不重叠，``load`` 可以并行化为一组独立的内存读取操作。类似地，``store`` 可以并行化为多个内存写入操作，每个操作仅依赖于为其写入数据元素的加载操作。

然而，如果输入和输出数组可能重叠，则编译器必须确保整个 tile 的所有内存加载操作在发出任何内存存储操作之前已完成，以确保正确的程序语义。否则，存储操作可能在加载操作读取元素之前执行并覆盖该元素，导致不正确的程序执行。这限制了编译器交错读写的能力，因为所有读取必须在任何写入可以发出之前完成。

简而言之，当编译器无法保证非重叠数组时，它必须生成更保守的代码。这就是为什么使用非重叠数组并使用 ``__restrict__`` 关键字将其通知编译器有助于实现最佳性能。

当内存区域可以被另一个指针访问时，将指针标记为 ``__restrict__`` 将导致未定义行为。


2.4.12.2. 将数组指针标记为 16 字节对齐
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用 ``ct::assume_aligned`` 将数组指针标记为 16 字节对齐：

.. code-block:: c++

   __tile_global__ void foo(float* __restrict__ in) {
       namespace ct = cuda::tiles;
       using namespace ct::literals;

       in = ct::assume_aligned(in, 16_ic);

       ct::tensor_span t{in, ct::extents{256_ic, 256_ic}};
       ct::partition_view{t, ct::shape{4_ic, 4_ic}};

       // ...
   }

此对齐保证是 ``ct::partition_view`` 使用张量内存加速器 (TMA) 所必需的。使用此技术时必须在运行时提供 16 字节对齐的指针，否则行为是未定义的。

由 CUDA 内存分配器（如 ``cudaMalloc``）返回的指针保证至少 16 字节对齐。


2.4.12.3. 优先使用 ``ct::partition_view`` 进行内存访问
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

对于结构化内存访问，优先使用 ``ct::partition_view`` 而非 gather 和 scatter 形式的 ``ct::load`` 和 ``ct::store``。基于视图的形式可以在支持的硬件上降低为张量内存加速器 (TMA)，这比逐元素 gather 快得多。有关 gather/scatter 上下文，请参见 :ref:`Gather 和 Scatter <writing-tile-kernels-gather-and-scatter>`。


2.4.12.4. 对有界循环使用 ``ct::irange``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当迭代固定范围时，使用 ``ct::irange`` 而非普通 ``for`` 循环。结构化形式允许编译器应用诸如流水线和向量化等优化，这些优化在循环边界和步长是不透明整数表达式时不可用（参见 :ref:`控制流 <writing-tile-kernels-control-flow>`）：

.. code-block:: c++

   for (auto idx : ct::irange(lowerBound, upperBound, step)) {
       // ...
   }
