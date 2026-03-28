.. _cooperative-groups:

4.4. Cooperative Groups
========================

.. _cg-introduction:

4.4.1. 简介
-----------

Cooperative Groups 是 CUDA 编程模型的扩展，用于组织协作线程组。Cooperative Groups 允许开发者控制线程协作的粒度，帮助他们表达更丰富、更高效的并行分解。Cooperative Groups 还提供了常见并行原语的实现，如 scan 和 parallel reduce。

历史上，CUDA 编程模型仅提供了一种简单的构造来同步协作线程：跨越线程块中所有线程的屏障，通过 ``__syncthreads()`` 内置函数实现。为了表达更广泛的并行交互模式，许多面向性能的程序员不得不编写自己的临时且不安全的原语来同步单个 warp 内的线程，或在单个 GPU 上运行的线程块集合之间进行同步。虽然所实现的性能提升通常很有价值，但这导致了越来越多的脆弱代码，这些代码在跨 GPU 代际编写、调优和维护方面成本高昂。Cooperative Groups 提供了一种安全且面向未来的机制来编写高性能代码。

完整的 Cooperative Groups API 可在 :doc:`../05-appendices/device-callable-apis` 中的 Cooperative Groups API 章节找到。

.. _cg-handle-member-functions:

4.4.2. Cooperative Group 句柄与成员函数
----------------------------------------

Cooperative Groups 通过 Cooperative Group 句柄进行管理。Cooperative Group 句柄允许参与的线程了解它们在组中的位置、组大小和其他组信息。下表展示了部分组成员函数。

.. _tbl:cg-member-functions:

.. list-table:: 选择性成员函数
   :header-rows: 1
   :widths: 30 70

   * - 访问器
     - 返回值
   * - ``thread_rank()``
     - 调用线程的排名。
   * - ``num_threads()``
     - 组中的线程总数。
   * - ``thread_index()``
     - 线程在启动块内的三维索引。
   * - ``dim_threads()``
     - 启动块的三维尺寸（以线程为单位）。

完整的成员函数列表可在 :doc:`../05-appendices/device-callable-apis` 中的 Cooperative Groups API 章节找到。

.. _cg-default-behavior:

4.4.3. 默认行为 / 无组执行
---------------------------

基于内核启动配置，表示 grid 和线程块的组会被隐式创建。这些「隐式」组为开发者提供了一个起点，可以显式分解为更细粒度的组。可以使用以下方法访问隐式组：

.. _tbl:cg-implicit-groups:

.. list-table:: CUDA 运行时隐式创建的 Cooperative Groups
   :header-rows: 1
   :widths: 40 60

   * - 访问器
     - 组作用域
   * - ``this_thread_block()``
     - 返回包含当前线程块中所有线程的组句柄。
   * - ``this_grid()``
     - 返回包含 grid 中所有线程的组句柄。
   * - ``coalesced_threads()`` [1]_
     - 返回 warp 中当前活动线程组的句柄。
   * - ``this_cluster()`` [2]_
     - 返回当前 cluster 中线程组的句柄。

.. [1] ``coalesced_threads()`` 操作符返回该时间点的活动线程集合，不保证返回哪些线程（只要是活动的）或它们在执行过程中保持合并状态。

.. [2] 当启动非 cluster grid 时， ``this_cluster()`` 假设一个 1x1x1 的 cluster。需要计算能力 9.0 或更高版本。

更多信息可在 :doc:`../05-appendices/device-callable-apis` 中的 Cooperative Groups API 章节找到。

.. _cg-create-early:

4.4.3.1. 尽早创建隐式组句柄
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

为了获得最佳性能，建议您提前（尽可能早，在任何分支发生之前）为隐式组创建句柄，并在整个内核中使用该句柄。

.. _cg-pass-by-reference:

4.4.3.2. 仅通过引用传递组句柄
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

建议在将组句柄传递给函数时通过引用传递。组句柄必须在声明时初始化，因为没有默认构造函数。不鼓励复制构造组句柄。

.. _cg-creating-groups:

4.4.4. 创建 Cooperative Groups
-------------------------------

组是通过将父组划分为子组来创建的。当组被划分时，会创建一个组句柄来管理生成的子组。开发者可以使用以下划分操作：

.. _tbl:cg-partition-operations:

.. list-table:: Cooperative Group 划分操作
   :header-rows: 1
   :widths: 25 75

   * - 划分类型
     - 描述
   * - tiled_partition
     - 将父组划分为一系列固定大小的子组，按一维行优先格式排列。
   * - stride_partition
     - 将父组划分为等大小的子组，线程以轮询方式分配到子组。
   * - labeled_partition
     - 基于条件标签将父组划分为一维子组，标签可以是任何整数类型。
   * - binary_partition
     - labeled partition 的特殊形式，标签只能是「0」或「1」。

以下示例展示了如何创建 tiled partition：

.. code-block:: cuda

   namespace cg = cooperative_groups;
   // 获取当前线程的 cooperative group
   cg::thread_block my_group = cg::this_thread_block();

   // 将 cooperative group 划分为大小为 8 的 tile
   cg::thread_block_tile<8> my_subgroup = cg::tiled_partition<8>(cta);

   // 作为 my_subgroup 执行工作

最佳划分策略取决于上下文。更多信息可在 :doc:`../05-appendices/device-callable-apis` 中的 Cooperative Groups API 章节找到。

.. _cg-creation-hazards:

4.4.4.1. 避免组创建危险
~~~~~~~~~~~~~~~~~~~~~~~~

划分组是一个集合操作，组中的所有线程都必须参与。如果组是在并非所有线程都能到达的条件分支中创建的，这可能导致死锁或数据损坏。

.. _cg-synchronization:

4.4.5. 同步
------------

在引入 Cooperative Groups 之前，CUDA 编程模型只允许在内核完成边界处进行线程块之间的同步。Cooperative Groups 允许开发者以不同的粒度同步协作线程组。

.. _cg-sync:

4.4.5.1. Sync
~~~~~~~~~~~~~

您可以通过调用集合 ``sync()`` 函数来同步组。与 ``__syncthreads()`` 类似， ``sync()`` 函数提供以下保证：

- 组中线程在同步点之前的所有内存访问（如读取和写入）对组中所有线程在同步点之后都是可见的。
- 组中的所有线程都必须到达同步点，然后任何线程才能继续执行。

以下示例展示了与 ``__syncthreads()`` 等效的 ``cooperative_groups::sync()`` ：

.. code-block:: cuda

   namespace cg = cooperative_groups;

   cg::thread_block my_group = cg::this_thread_block();

   // 同步块中的线程
   cg::sync(my_group);

Cooperative Groups 可用于同步整个 grid。从 CUDA 13 开始，Cooperative Groups 不再能用于多设备同步。详情请参阅 :ref:`sec:cg-large-scale-groups` 章节。

有关同步的更多信息可在 :doc:`../05-appendices/device-callable-apis` 中的 Cooperative Groups API 章节找到。

.. _cg-barriers:

4.4.5.2. Barriers
~~~~~~~~~~~~~~~~~

Cooperative Groups 提供了类似于 ``cuda::barrier`` 的屏障 API，可用于更高级的同步。Cooperative Groups 屏障 API 与 ``cuda::barrier`` 在几个关键方面有所不同：

- Cooperative Groups 屏障会自动初始化
- 组中的所有线程在每个阶段必须到达并等待屏障一次
- ``barrier_arrive`` 返回一个 ``arrival_token`` 对象，该对象必须传递给相应的 ``barrier_wait`` ，在那里它被消耗且不能再次使用

程序员在使用 Cooperative Groups 屏障时必须注意避免危险：

- 在调用 ``barrier_arrive`` 之后、调用 ``barrier_wait`` 之前，组不能使用任何集合操作
- ``barrier_wait`` 仅保证组中的所有线程都已调用 ``barrier_arrive`` 。 ``barrier_wait`` 并不保证所有线程都已调用 ``barrier_wait``

.. code-block:: cuda

   namespace cg = cooperative_groups;

   cg::thread_block my_group = this_block();

   auto token = cluster.barrier_arrive();

   // 可选：执行一些本地处理以隐藏同步延迟
   local_processing(block);

   // 确保集群中的所有其他块都在运行并初始化了共享数据，然后再访问 dsmem
   cluster.barrier_wait(std::move(token));

.. _cg-collective-operations:

4.4.6. 集合操作
----------------

Cooperative Groups 包含一组可由线程组执行的集合操作。这些操作需要指定组中的所有线程参与才能完成操作。

组中的所有线程必须为每个集合调用的相应参数传递相同的值，除非在 :doc:`../05-appendices/device-callable-apis` 中的 Cooperative Groups API 章节中明确允许不同的值。否则，调用的行为是未定义的。

.. _cg-reduce:

4.4.6.1. Reduce
~~~~~~~~~~~~~~~

``reduce`` 函数用于对指定组中每个线程提供的数据执行并行归约。必须通过提供下表中列出的操作符之一来指定归约类型。

.. _tbl:cg-reduction-operators:

.. list-table:: Cooperative Groups 归约操作符
   :header-rows: 1
   :widths: 25 75

   * - 操作符
     - 返回值
   * - plus
     - 组中所有值的总和
   * - less
     - 最小值
   * - greater
     - 最大值
   * - bit_and
     - 按位与归约
   * - bit_or
     - 按位或归约
   * - bit_xor
     - 按位异或归约

当硬件加速可用时，归约会使用硬件加速（需要计算能力 8.0 或更高版本）。对于硬件加速不可用的旧硬件，提供软件回退。只有 4B 类型由硬件加速。

有关归约的更多信息可在 :doc:`../05-appendices/device-callable-apis` 中的 Cooperative Groups API 章节找到。

以下示例展示了如何使用 ``cooperative_groups::reduce()`` 执行块级求和归约：

.. code-block:: cuda

   namespace cg = cooperative_groups;

   cg::thread_block my_group = cg::this_thread_block();

   int val = data[threadIdx.x];

   int sum = cg::reduce(cta, val, cg::plus<int>());

   // 存储归约结果
   if (my_group.thread_rank() == 0) {
      result[blockIdx.x] = sum;
   }

.. _cg-scans:

4.4.6.2. Scans
~~~~~~~~~~~~~~~

Cooperative Groups 包含 ``inclusive_scan`` 和 ``exclusive_scan`` 的实现，可用于任意组大小。这些函数对指定组中每个命名线程提供的数据执行 scan 操作。

程序员可以选择指定归约操作符，如上 :ref:`tbl:cg-reduction-operators` 中所列。

.. code-block:: cuda

   namespace cg = cooperative_groups;

   cg::thread_block my_group = cg::this_thread_block();

   int val = data[my_group.thread_rank()];

   int exclusive_sum = cg::exclusive_scan(my_group, val, cg::plus<int>());

   result[my_group.thread_rank()] = exclusive_sum;

有关 scan 的更多信息可在 :doc:`../05-appendices/device-callable-apis` 中的 Cooperative Groups Scan API 章节找到。

.. _cg-invoke-one:

4.4.6.3. Invoke One
~~~~~~~~~~~~~~~~~~~~

Cooperative Groups 提供了 ``invoke_one`` 函数，用于单个线程必须代表组执行串行部分工作的情况：

- ``invoke_one`` 从调用组中选择一个任意线程，并使用该线程以提供的参数调用提供的可调用函数
- ``invoke_one_broadcast`` 与 ``invoke_one`` 相同，只是调用的结果也会广播给组中的所有线程

线程选择机制不保证是确定性的。

以下示例展示了基本的 ``invoke_one`` 用法：

.. code-block:: cuda

   namespace cg = cooperative_groups;
   cg::thread_block my_group = cg::this_thread_block();

   // 确保线程块中只有一个线程打印消息
   cg::invoke_one(my_group, [&]() {
      printf("Hello from one thread in the block!");
   });

   // 同步以确保所有线程等待直到消息打印完成
   cg::sync(my_group);

在可调用函数内部不允许与调用组进行通信或同步。允许与调用组之外的线程进行通信。

.. _cg-async-data-movement:

4.4.7. 异步数据移动
--------------------

CUDA 中的 Cooperative Groups ``memcpy_async`` 功能提供了一种在全局内存和共享内存之间执行异步内存复制的方法。 ``memcpy_async`` 对于优化内存传输和将计算与数据传输重叠以提高性能特别有用。

``memcpy_async`` 函数用于启动从全局内存到共享内存的异步加载。 ``memcpy_async`` 旨在像「预取」一样使用，在需要数据之前加载数据。

``wait`` 函数强制组中的所有线程等待，直到异步内存传输完成。在共享内存中访问数据之前，组中的所有线程都必须调用 ``wait`` 。

以下示例展示了如何使用 ``memcpy_async`` 和 ``wait`` 来预取数据：

.. code-block:: cuda

   namespace cg = cooperative_groups;

   cg::thread_group my_group = cg::this_thread_block();

   __shared__ int shared_data[];

   // 执行从全局内存到共享内存的异步复制
   cg::memcpy_async(my_group, shared_data + my_group.rank(), input + my_group.rank(), sizeof(int));

   // 在此隐藏延迟。不能使用 shared_data

   // 等待异步复制完成
   cg::wait(my_group);

   // 预取的数据现在可用

有关更多信息，请参阅 :doc:`../05-appendices/device-callable-apis` 中的 Cooperative Groups API 章节。

.. _cg-memcpy-async-alignment:

4.4.7.1. Memcpy Async 对齐要求
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

仅当源是全局内存且目标是共享内存，且两者都至少是 4 字节对齐时， ``memcpy_async`` 才是异步的。为了获得最佳性能：建议共享内存和全局内存都对齐 16 字节。

.. _cg-large-scale-groups:

4.4.8. 大规模组
----------------

Cooperative Groups 允许创建跨越整个 grid 的大组。前面描述的所有 Cooperative Group 功能都可用于这些大组，但有一个值得注意的例外：同步整个 grid 需要使用 ``cudaLaunchCooperativeKernel`` 运行时启动 API。

从 CUDA 13 开始，Cooperative Groups 的多设备启动 API 和相关引用已被移除。

.. _cg-cooperative-kernel:

4.4.8.1. 何时使用 ``cudaLaunchCooperativeKernel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cudaLaunchCooperativeKernel`` 是一个 CUDA 运行时 API 函数，用于启动使用 cooperative groups 的单设备内核，专门设计用于执行需要块间同步的内核。此函数确保内核中的所有线程可以在整个 grid 范围内同步和协作，这对于仅允许在单个线程块内同步的传统 CUDA 内核来说是不可能的。 ``cudaLaunchCooperativeKernel`` 确保内核启动是原子的，即如果 API 调用成功，则指定数量的线程块将在指定设备上启动。

最佳实践是首先通过查询设备属性 ``cudaDevAttrCooperativeLaunch`` 来确保设备支持 cooperative launch：

.. code-block:: cuda

   int dev = 0;
   int supportsCoopLaunch = 0;
   cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);

如果设备 0 支持该属性，这会将 ``supportsCoopLaunch`` 设置为 1。仅支持计算能力 6.0 及更高版本的设备。此外，您需要在以下环境中运行：

- 不带 MPS 的 Linux 平台
- 带 MPS 的 Linux 平台，且设备计算能力为 7.0 或更高版本
- 最新的 Windows 平台
