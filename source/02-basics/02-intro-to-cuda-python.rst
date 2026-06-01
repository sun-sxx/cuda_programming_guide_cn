.. _intro-to-cuda-python:

2.2. CUDA Python 简介
========================

本章介绍 Python 中的 CUDA kernel 编程。
CUDA Python 生态系统包含大量且不断发展的工具和库。
本章首先介绍其中一些组件，然后使用其中几个来说明在 Python 中编写和执行 GPU 代码的方法。

在 Python 中利用 GPU 计算的方式有很多，其中许多方式不需要显式编写 GPU kernel。
`CUDA Python 生态系统 <#intro-cuda-python-ecosystem>`_ 的一些组件提供了在 GPU 上执行操作的函数，开发者无需进行任何特定的 GPU 控制或编码。
`NVIDIA 加速计算中心 <https://github.com/NVIDIA/accelerated-computing-hub>`_ 提供了 `加速 Python 用户指南 <https://github.com/NVIDIA/accelerated-computing-hub/tree/main/Accelerated_Python_User_Guide/notebooks>`_，介绍和讨论了许多支持 GPU 加速计算的库和工具。
该资源对于希望尽可能快速、轻松地使用 GPU 而不希望编写 GPU 代码的用户来说是一个很好的起点。

另一方面，本章侧重于直接控制 GPU 并在 Python 中编写在 GPU 上执行的 kernel。
本章重点介绍 Python 中的 :ref:`CUDA 单指令多线程 (SIMT) <warps-and-simt>` 编程。


.. _intro-cuda-python-ecosystem:

2.2.1. CUDA Python 生态系统
----------------------------

CUDA Python 是一个在 Python 中使用 GPU 计算的工具和库的生态系统。
以下列表介绍了 CUDA Python 的主要组件，并非所有组件都是本章内容所必需的。
此列表改编自 `CUDA Python GitHub 仓库 <https://github.com/NVIDIA/cuda-python>`_ 中的完整列表。

**主要组件** — 用于控制 GPU 以及运行由库形式提供的 GPU 代码。

- ``cuda.core`` — 用于内存和设备管理等 CUDA 控制操作的 Pythonic 接口。它为 Python 提供了与 CUDA Runtime 为 C++ 所提供的相同功能。

- ``cuda.compute`` — 一个 Python 模块，提供了由 CUDA 核心计算库（ `CCCL <https://nvidia.github.io/cccl/unstable/python/compute.html>`_ ）所支持的 GPU 加速函数。

- ``CuPy`` — 一个 Python 库，提供了 NumPy 常用函数的 GPU 加速版本，同时也提供了一个 GPU 加速版的 ``ndarray`` 数据容器。

**Kernel 开发组件**

- ``cuda.lang`` — 一种 Python DSL，使用 Python 语言子集，在 SIMT 编程模型下编写 CUDA 核函数和设备端函数。

- ``cuda.coop`` — 一个 Python 模块，提供了 CUDA 核心计算库（CCCL）中可由设备端调用的原语（需使用 cuda.lang）。

- ``cuda.tile`` — 一种 Python DSL，用于在 Tile 编程模式下编写 CUDA 核函数和设备端函数。

**其他组件**

- ``cuda.pathfinder`` — 一个用于定位当前 Python 环境中已安装的 CUDA 组件的实用工具。

- ``cuda.bindings`` — CUDA 各类库和工具的底层 Python 绑定，包括 CUDA Driver API、CUDA Runtime API、 NVRTC 、 NVVM 等。
  ``cuda.bindings`` 通过其 CUDA 驱动和 CUDA 运行时组件，提供了与 ``cuda.core`` 相同的功能。
  不过， ``cuda.bindings`` 是以 C 语言 API 的 Python 封装形式来提供这些功能的，而不是原生的 Pythonic 接口。


2.2.1.1. 在 Python 中使用 CUDA 库
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA C++ 拥有极其丰富的库生态系统，开发者无需编写 kernel 或底层 GPU 代码，就能轻松实现 GPU 加速。
回想 2006 年 CUDA C++ 刚问世的时候，相关的库寥寥无几，开发者们基本上只能靠自己手写 GPU 核函数。
但从那以后， `大量库 <https://developer.nvidia.com/cuda/cuda-x-libraries>`_ 如雨后春笋般涌现出来，
让现在的开发者在 C++ 环境下而几乎不需要（甚至完全不需要）再去写繁琐的 GPU 代码， 就能充分利用 GPU 的计算能力。

CUDA Python 生态系统从另一个方向演进：在开发者能够直接用 Python 的语法和语义来编写自定义核函数之前，像 CuPy 这样的 Python 库就已经率先为开发者们提供了各种计算任务和算法的 GPU 加速实现。
这其中许多库提供了用 CUDA C++ 实现的 GPU 代码的 Python 绑定。

在当下的 CUDA 时代，如果现有的 GPU 加速库能够满足你的需求，那么强烈建议优先使用它们。
毕竟，这些库里的很多实现都是经过 GPU 计算专家精心调优的。
当然，如果遇到没有现成库可用，或者现有库无法满足需求的情况，你依然可以在 Python 里直接编写 GPU 核函数和设备端函数，就像在 CUDA C++ 做的中那样。


2.2.1.2. 本章范围
~~~~~~~~~~~~~~~~~~

虽然开发者应该尽可能优先使用现成的库，但本章的其余部分将为大家介绍如何使用 Python 编写自定义 GPU 代码。
本章的内容安排与 :ref:`intro-to-cuda-c` 中 C++ 的方式完全一致，将从如何定义一个 GPU 核函数开始讲起，
然后介绍如何使用 cuPy 提供的 GPU 加速版 ``ndarray`` 在显存中分配内存，并实现 CPU 与 GPU 之间的数据传输。

2.2.1.3. 环境设置
~~~~~~~~~~~~~~~~~~

总的来说，大多数 CUDA Python 生态系统组件都可以在 PyPI 上找到，并且可以通过 ``pip`` 命令或任何流行的 Python 包管理器来安装。
所有这些包都要求系统上已经安装了最新版本的 NVIDIA Driver。
不过，在编写或运行 CUDA Python 程序时，通常并不需要额外安装完整的 CUDA Toolkit。

有关在不同平台上安装和配置 CUDA Python 的信息，请参阅 `NVIDIA 开发者专区中 CUDA Python 专题 <https://developer.nvidia.com/how-to-cuda-python>`_ 。

2.2.1.4. 运行 CUDA Python 应用程序
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA Python 应用程序，无论是使用 CUDA 加速库还是包含用户编写的 GPU 代码，都以与传统 Python 应用程序相同的方式运行。
在本节中，示例始终通过从命令行调用 ``python3`` 来运行，如下所示，执行名为 ``cuda-python-app.py`` 的程序。

CUDA Python 应用程序，无论是使用 CUDA 加速库还是包含用户自己编写的 GPU 代码，运行方式都和普通的 Python 应用程序完全一样。
在本节中，所有的示例都会通过命令行来运行，就像下面这样调用 ``python3`` 来执行一个名为 ``cuda-python-app.py`` 的程序

.. code-block:: bash

   $ python3 cuda-python-app.py


.. _simt-kernels-in-python:

2.2.2. Python 中的 SIMT Kernels
--------------------------------

正如在 :ref:`CUDA 编程模型 <programming-model>` 中提到的那样，那些能够在 GPU 上执行、并且可以由主机调用的函数，被称为核函数（kernels）。
CUDA 提供了两种不同的编程模型：:ref:`SIMT <warps-and-simt>` 和 :ref:`CUDA Tile <programming-model-tile>` 。
其中，SIMT kernel 的编写目的是为了让大量并行的线程能够同时运行。
这一概念在 CUDA Python 和 CUDA C++ 中是完全一致的。
本章将使用 SIMT 核函数来为大家介绍 CUDA Python。


2.2.2.1. 指定 Kernels
~~~~~~~~~~~~~~~~~~~~~~

在 CUDA Python 中定义 kernel 之前，必须先导入 ``numba.cuda`` 包。通常可以按照下面所示的方式来操作

.. code-block:: python

   from numba import cuda

这就导入了 ``numba.cuda`` 包，并允许我们使用该包所提供的 ``cuda`` 命名空间下的各种组件。

要在 CUDA Python 中将一个函数指定为 kernel ，只需在函数定义的上一行加上装饰器 ``@cuda.jit`` ，如下所示。

.. code-block:: python

   from numba import cuda

   @cuda.jit
   def function(input_array, output_array):
       ...

这样，当 kernel 首次启动时，系统就会对它进行即时编译（JIT），以适配当前正在使用的 GPU。
如果当前未指定 GPU（本节中的示例就是这种情况），程序将会使用 CUDA 的默认设备。

.. _intro-cuda-python-launching-kernels:

2.2.2.2. 启动 Kernels
~~~~~~~~~~~~~~~~~~~~~~

执行 kernel 的线程数量是在启动 kernel 时指定的，这也被称为执行配置。
每一次调用核函数都可以拥有不同的执行配置，比如使用不同的块大小或不同数量的线程块。


2.2.2.2.1. Kernel 启动
"""""""""""""""""""""""

启动 kernel 时，需要将执行配置放在方括号 ``[ ]`` 中，紧跟在核函数名之后、函数参数之前。
参数的顺序与 :ref:`第 2.1.2.2.1 节 <intro-cpp-launching-kernels-triple-chevron>` 节中介绍的 C++ 三尖括号表示法完全一致，具体如下：

.. code-block:: python3

   kernel_name[number_of_thread_blocks, threads_per_block](arguments, ...)

下面的代码片段展示了如何在 Python 源文件中定义并调用一个核函数。

.. code-block:: python

   from numba import cuda

   @cuda.jit
   def my_kernel(input, output):
       ...

   ## launch the kernel
   my_kernel[num_thread_blocks, threads_per_block](in_array, out_array)


每个 block 中的线程数量是有限制的，因为 block 内的所有线程都驻留在同一个 SM 上，并且必须共享该 SM 的资源。
在当前的 GPU 上，一个 block 最多可以包含 1024 个线程。
如果资源允许，多个 block 可以被同时调度到同一个 SM 上运行。

2.2.2.2.2. 多维 Grid 和 Thread Blocks
""""""""""""""""""""""""""""""""""""""

在 CUDA 中，线程块以及由线程块组成的网格（grid）都可以是 1维、2维或 3维的。
当网格或线程块是一维的时候，在指定 kernel 执行配置时，直接使用一个整数就可以了。
而当线程块或网格是 2维或 3维时，则需要使用二维或三维的元组（tuple）。
如下所示，这是一个 2D 网格和线程块的启动示例，其中 ``gridX`` 和 ``gridY`` 分别代表网格在 x 和 y 方向上的维度，
而 ``blockX`` 和 ``blockY`` 则代表网格内每个线程块在 x 和 y 方向上的维度。

.. code-block:: python

   from numba import cuda

   @cuda.jit
   def function(input, output):
       ...

   ## launch the kernel
   function[(gridX, gridY), (blockX, blockY)](in_array, out_array)


.. _intro-cuda-python-thread-indexing:

2.2.2.3. 线程和 Grid 索引内建函数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`thread-blocks-and-grids` 绍了线程块和网格， :ref:`intro-cuda-python-launching-kernels` 节展示了如何为 kernel 指定网格和线程块的大小。
在 kernel 内部，每个线程都可以访问执行配置的参数，以及该线程在线程块内的索引和在网格中的线程块索引。

在 kernel 中，可以通过访问以下变量来确定一个线程的身份：

- ``cuda.threadIdx.[xyz]`` 给出线程在其 thread block 内的索引。Thread block 中的每个线程都有不同的索引。

- ``cuda.blockDim.[xyz]`` 给出 thread block 的维度，这是在 kernel 启动的执行配置中指定的。

- ``cuda.blockIdx.[xyz]`` 给出 thread block 在 grid 内的索引。每个 thread block 都有不同的索引。

- ``cuda.gridDim.[xyz]`` 给出 grid 的维度，这是在 kernel 启动时的执行配置中指定的。

这些变量中的每一个都是一个 3 分量向量，具有 ``.x``、``.y`` 和 ``.z`` 成员。
如果在 kernel 启动时的执行配置中未指定某个维度，则维度默认值为 1，索引默认值为 0。

``cuda.threadIdx`` 和 ``cuda.blockIdx`` 从零开始索引。
也就是说， ``cuda.threadIdx.x`` 的值将从 0 到 ``cuda.blockDim.x - 1`` （含）。 ``.y`` 和 ``.z`` 在各自维度上的范围相同。

下面展示了一个简单的向量加法 kernel 的代码，该 kernel 将两个向量逐元素相加。此函数接受三个数组 ``A``、``B`` 和 ``C``，实现逐元素向量加法 ``C = A + B``。

.. code-block:: python

   # C = A + B vector addition
   @cuda.jit
   def vecadd(A, B, C):
       idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
       C[idx] = A[idx] + B[idx]

kernel 首先计算线程在 grid 中的唯一索引。
此 kernel 假设使用 1 维 thread block 在 1 维 grid 中启动。
``idx`` 变量是从 0 到 ``N-1`` 的唯一索引，其中 N 是 grid 中的总线程数，即 ``N = cuda.gridDim.x * cuda.blockDim.x`` 。

上面代码块中计算线程索引的模式非常常见，Numba 为此操作提供了简写语法： ``cuda.grid(n)`` ，其中 ``n`` 是维度数。
在上面的示例中，以下行

.. code-block:: python

   idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

可以替换为更简单的

.. code-block:: python

   idx = cuda.grid(1)

此 kernel 的一个值得注意的方面是它不检查对 ``A``、 ``B`` 或 ``C`` 的越界访问。
在本章中，我们假设这些是由 cuPy 创建的 ``ndarray``，将在 :ref:`intro-cuda-python-ndarray` 中介绍。
使用 cuPy ``ndarray`` 时，边界检查由数组类型隐式实现。


2.2.3. GPU 计算中的内存
-------------------------

.. note::
   
   cuPy 等 Python 包通过直接调用 CUDA C++ API（例如 :ref:`intro-cpp-explicit-memory-management` 节中介绍的那些接口）来执行 GPU 内存管理。
   目前有多种 Python 包都提供了用于控制 GPU 内存分配的封装和实用工具。
   在本指南中，我们将只介绍 cuPy。
   大多数包的概念都是相似的，并且除了特别说明的情况外，它们的行为方式通常也与对应的 C++ 版本非常接近。

正如 :ref:`gpu-memory` 所介绍的，GPU 拥有自己独立的 DRAM。
在 kernel 访问数据数组之前，这些数据通常必须已经存放在 GPU 的 DRAM 中。
在 Python 中，控制数据在内存中的位置——也就是在 CPU 和 GPU 之间搬运数据——是程序员的责任。
这与 :ref:`intro-cpp-explicit-memory-management` 中介绍的 C++ 显式内存管理的情况是完全一样的。

2.2.3.1. 在 GPU 上实例化数组
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuPy 提供了一系列函数，用于在 GPU 上创建指定类型和维度的 ``ndarray`` 对象，同时也支持在 CPU 和 GPU 之间拷贝数据。
CuPy 中的许多函数，其调用方式与 NumPy 中创建 ``ndarray`` 的函数非常相似。
下面展示了一些如何使用 CuPy 在 GPU 显存中创建并填充数组的示例。

.. code-block:: python

   import cupy as cp
   import numpy as np

   ## create a matrix of zeros on the GPU
   ## when a datatype is not specified, float32 is used by default
   A_device = cp.zeros((1024, 1024))

   ## create an array of 2^20 random doubles on the GPU
   B_device = cp.rand.random((2**20), dtype=np.double)

   ## create an array of zeroes with the same shape and datatype as an existing array
   C_device = cp.zeros_like(A)


2.2.3.2. 在主机和 GPU 内存之间复制数组
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuPy 还可以用于将数据从位于 CPU 内存中的 Numpy ``ndarray`` 复制到位于 GPU 内存中的 CuPy 数组。

.. code-block:: python

   import cupy as cp
   import numpy as np

   ## Create an array in host memory
   A_host = np.zeros((1024, 1024))
   ## Copy the array to the GPU
   A_device = cp.array(A_host)

   ## Create an array in GPU memory
   B_device = cp.rand.random((1024, 1024))
   ## copy the array to host memory
   B_host = cp.asnumpy(B_device)


.. _intro-cuda-python-ndarray:

2.2.3.3. ndarray 对象类型
~~~~~~~~~~~~~~~~~~~~~~~~~~

上一节展示的 ``ndarray`` 对象，要么存在于主机内存中，要么存在于 GPU 显存中，但绝不会同时存在于两者之中。
如果将驻留在主机上的数组作为参数传递给 CUDA kernel ，会导致报错。
同样地，如果将驻留在 GPU 显存中的数组传递给普通的 Python 函数（即非核函数），也会导致报错。
CuPy 不会在 CPU 和 GPU 之间隐式地拷贝数据，因为这种操作开销很大，而且过度的数据搬运会严重拖慢程序性能。
因此，CuPy 要求程序员必须明确、有意识地去控制数据在 CPU 和 GPU 之间的拷贝时机。

在 GPU kernel 中使用 ``ndarray`` 类型的一个优势是，该数组自身就携带了其各维度的边界信息。
正如 :ref:`intro-cuda-python-thread-indexing` 所示，边界检查会自动完成。
因此，当所需的线程总数略小于执行块或网格的总规模时，内核代码无需再手动去检查是否发生了越界访问。

2.2.4. 同步 CPU 和 GPU
------------------------

与 C++ 一样，CUDA Python 中的 kernel 启动对于主机线程来说也是异步的。
也就是说，在启动 kernel 之后，主机代码会立刻继续在 CPU 上往下执行，而不会去保证该 kernel 已完成甚至是否已开始执行。
为了确保 GPU kernel 确实已经全部执行完成，主机线程必须与 GPU 进行某种形式的同步操作。

最简单的方法是同步整个 GPU 。
这种设备级的同步是由 CUDA 驱动程序提供的，在 cuPy 和 numba.cuda 中 都是通过 ``synchronize()`` 这个方法暴露给 Python。

.. code-block:: python

   import cupy as cp
   from numba import cuda

   ...

   ## Wait on host thread for all pending GPU work to complete
   ## this uses the interface provided by cupy
   cp.synchronize()

   ## Wait on host thread for all pending GPU work to complete
   ## this uses the interface provided by numba.cuda
   cuda.synchronize()

设备级同步会让主机线程一直等待，直到 GPU 上之前发出的所有任务都全部执行完毕。
如果想要进行更细粒度的同步，可以使用 CUDA 流 ，具体方法在 :ref:`asynchronous-execution` 中有详细描述。
在 Python 中，当使用流时，推荐的做法是使用 `cuda.core` 来创建 CUDA 流，并且只在需要的时候对特定的流进行同步。

2.2.5. 完整示例
----------------

下面这段源代码展示了那个无处不在、用来做并行向量加法的经典 GPU kernel ，这里呈现的是它的 Python 形式。

.. code-block:: python

   import numpy as np
   from numba import cuda
   import cupy as cp

   ## Defines a CUDA kernel to perform C = A + B vector addition
   @cuda.jit
   def vecadd(A, B, C):
       work_index = cuda.grid(1)
       C[work_index] = A[work_index] + B[work_index]

   # note that vector size is not a power of 2 nor a multiple of the block_size defined below
   vector_size = 2**24 + 11

   device = cp.cuda.Device()
   ## Create device arrays of uniform random float32 values as input, and an array of zeros
   ## as the result vector
   a = cp.random.uniform(-1, 1, vector_size)
   b = cp.random.uniform(-1, 1, vector_size)
   c = cp.zeros_like(a)

   block_size = 256
   grid_size = int(np.ceil(vector_size/block_size))
   vecadd[grid_size, block_size](a, b, c)

   ## synchronize the CPU thread and the GPU to ensure that the kernel has completed
   ## this is included to illustrate good practices, 
   ## even though the copy below would implicitly wait for the kernel to complete
   device.synchronize()

   ## Copy all 3 arrays to the CPU as ndarrays
   a_np = cp.asnumpy(a)
   b_np = cp.asnumpy(b)
   c_np = cp.asnumpy(c)

   ## Perform the copy on the CPU to verify the answer
   expected = a_np + b_np

   ## Test that the answer is correct, within floating point epsilon
   np.testing.assert_array_almost_equal(c_np, expected)

   ## The assert will print diagnostics and abort
   ## so this only prints if the assertion passes
   print("Test succeeded")

在此示例中， ``A`` 和 ``B`` 输入数组由 CuPy 在 GPU 上创建并初始化为随机值。
它们在代码末尾被复制到 CPU，以便 CPU 也可以执行向量加法并验证 CPU 和 GPU 的答案是否匹配。


2.2.6. CUDA Python 中的错误检查
---------------------------------

任何涉及 GPU 的操作，从内存分配、数据拷贝到 kernel 启动，都有可能引发错误。
正如第 :ref:`intro-cpp-error-checking` 节针对 C++ 所阐述的那样，确保在与 GPU 交互的过程中没有发生错误，是公认的最佳实践。

在 Python 中，CUDA 的错误会抛出异常，如果这些异常没有被捕获，就会导致程序直接终止。
我们可以使用标准的 Python 语法来捕获这些异常。
下面的例子展示了与上文相同的向量加法代码，但故意加入了一个错误：将每个线程块的线程数设置为了 2048，这超出了目前任何 GPU 所能支持的上限。
这将导致 kernel 启动失败并抛出一个异常，而这段代码正好会将该异常捕获。

.. code-block:: python

   import numpy as np
   from numba import cuda
   import cupy as cp

   ## Defines a CUDA kernel to perform C = A + B vector addition
   @cuda.jit
   def vecadd(A, B, C):
       work_index = cuda.grid(1)
       C[work_index] = A[work_index] + B[work_index]

   try:
       vector_size = 2**24 + 11

       device = cp.cuda.Device()
       a = cp.random.uniform(-1, 1, vector_size)
       b = cp.random.uniform(-1, 1, vector_size)
       c = cp.zeros_like(a)

       ## this block size is too large for any current GPUs
       block_size = 2048
       grid_size = int(np.ceil(vector_size/block_size))
       # Error: launching kernel with invalid block size
       vecadd[grid_size, block_size](a, b, c)

       device.synchronize()
       print("Test did not encounter any errors")

   except Exception as e:
       print(f"Exception occurred: {e}")

运行这段代码后，错误就会被成功捕获，并像下面这样显示出来：

.. code-block:: bash

   $ python3 vecadd_error.py
   Exception occurred: CUDA_ERROR_INVALID_VALUE: This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.

程序因为成功捕获了异常，所以正常退出了。如果这段代码在没有 ``try:`` 和 ``except:`` 的情况下运行，它就会非正常退出，并且会在控制台打印出一长串回溯信息（traceback），其中应该也会显示出同样的错误。
