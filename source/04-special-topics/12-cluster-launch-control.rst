.. _cluster-launch-control:

4.12. 使用 Cluster Launch Control 的工作窃取
==============================================

处理可变数据和计算规模的问题是开发 CUDA 应用程序时的关键问题。传统上，CUDA 开发者使用两种主要方法来确定启动的 kernel 线程块数量：*固定每个线程块的工作量* 和 *固定线程块数量*。这两种方法各有优缺点。

**固定每个线程块的工作量**：在这种方法中，线程块数量由问题规模决定，而每个线程块完成的工作量保持不变。

这种方法的主要优点：

- *SM 之间的负载均衡*

  当线程块运行时间存在变化，和/或当线程块数量远大于 GPU 可以同时执行的数量时（导致低尾部效应），这种方法允许 GPU 调度器在某些 SM 上运行比其他 SM 更多的线程块。

- *抢占*

  GPU 调度器可以开始执行 :doc:`更高优先级的 kernel <../02-basics/asynchronous-execution>`，即使它是在较低优先级的 kernel 已经开始执行后才启动的，方法是在较低优先级的 kernel 的线程块完成时调度其线程块。然后，一旦高优先级的 kernel 完成执行，它就可以恢复执行较低优先级的 kernel。

**固定线程块数量**：在这种方法中，通常实现为 block-stride 或 grid-stride 循环，线程块数量不依赖于问题规模。相反，每个线程块完成的工作量是问题规模的函数。通常，线程块数量基于执行 kernel 的 GPU 上的 SM 数量和所需的占用率。

这种方法的主要优点：

- *减少线程块开销*

  这种方法不仅减少了分摊的线程块启动延迟，还最小化了与所有线程块的共享操作相关的计算开销。这些开销可能比启动延迟开销高得多。

  例如，在卷积 kernel 中，用于计算卷积系数（与线程块索引无关）的序言由于固定的线程块数量可以被计算更少的次数，从而减少了冗余计算。

**Cluster Launch Control** 是 NVIDIA Blackwell GPU 架构（计算能力 10.0）中引入的一项功能，旨在结合前两种方法的优点。它通过允许开发者取消线程块或线程块集群，为他们提供了对线程块调度的更多控制。这种机制实现了工作窃取。工作窃取是并行计算中的一种动态负载均衡技术，其中空闲的处理器主动从忙碌处理器的工作队列中"窃取"任务，而不是等待分配工作。

.. _fig:cluster-launch-control:

.. figure:: ../_static/images/cluster_launch_control.png
   :alt: Cluster Launch Control Flow
   :align: center

   图 51 Cluster Launch Control 流程

使用 cluster launch control，线程块尝试取消尚未开始执行的另一个线程块的启动。如果取消请求成功，它会使用另一个线程块的索引来执行任务，从而"窃取"其工作。如果没有更多可用的线程块索引或由于其他原因（例如调度了更高优先级的 kernel），取消将失败。在后一种情况下，如果线程块在取消失败后退出，调度器可以开始执行更高优先级的 kernel，之后它将继续调度当前 kernel 的剩余线程块以执行。上图 :numref:`fig:cluster-launch-control` 展示了此过程的执行流程。

下表总结了三种方法的优缺点：

.. list-table:: 线程块调度方法比较
   :header-rows: 1
   :widths: auto

   * - 
     - **固定每个线程块的工作量**
     - **固定线程块数量**
     - **Cluster Launch Control**
   * - 减少开销
     - ✗
     - ✓
     - ✓
   * - 抢占
     - ✓
     - ✗
     - ✓
   * - 负载均衡
     - ✓
     - ✗
     - ✓

.. raw:: latex

   \newpage

4.12.1. API 详情
----------------

通过 cluster launch control API 取消线程块是异步完成的，并使用共享内存屏障进行同步，遵循与 :doc:`异步数据复制 <../03-advanced/advanced-kernel-programming>` 类似的编程模式。

该 API 通过 `libcu++ <https://nvidia.github.io/cccl/libcudacxx/ptx_api.html>`_ 提供，提供：

- 一个请求指令，将编码的取消结果写入 ``__shared__`` 变量。

- 解码指令，提取成功/失败状态和被取消的线程块索引。

注意，cluster launch control 操作被建模为异步代理操作（参见 :ref:`sec:async-thread-proxy` ）。

.. _thread-block-cancellation:

4.12.1.1. 线程块取消
~~~~~~~~~~~~~~~~~~~~

使用 Cluster Launch Control 的首选方式是从单个线程进行，即一次一个请求。

取消过程涉及五个步骤：

- **设置阶段** （步骤 1-2）：声明并初始化取消结果和同步变量。

- **工作窃取循环** （步骤 3-5）：重复执行以请求、同步和处理取消结果。

1. 声明线程块取消的变量：

   .. code-block:: cuda
      :linenos:

      __shared__ uint4 result;  // 请求结果。
      __shared__ uint64_t bar;  // 同步屏障。
      int phase = 0;            // 同步屏障阶段。

2. 使用单个到达计数初始化共享内存屏障：

   .. code-block:: cuda
      :linenos:

      if (cg::thread_block::thread_rank() == 0)
          ptx::mbarrier_init(&bar, 1);
      __syncthreads();

3. 由单个线程提交异步取消请求并设置事务计数：

   .. code-block:: cuda
      :linenos:

      if (cg::thread_block::thread_rank() == 0) {
          cg::invoke_one(cg::coalesced_threads(), [&](){ptx::clusterlaunchcontrol_try_cancel(&result, &bar);});
          ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
      }

   .. note::

      由于线程块取消是一个统一指令，建议在 :ref:`sec:cooperative-groups-invoke-one` 线程选择器内提交它。这允许编译器优化掉剥离循环。

4. 同步（完成）异步取消请求：

   .. code-block:: cuda
      :linenos:

      while (!ptx::mbarrier_try_wait_parity(&bar, phase))
      {}
      phase ^= 1;

5. 获取取消状态和被取消的线程块索引：

   .. code-block:: cuda
      :linenos:

      bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
      if (success) {
          // 对于 1D/2D 线程块不需要全部三个：
          int bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x(result);
          int by = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_y(result);
          int bz = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_z(result);
      }

6. 确保异步代理和通用 `代理 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#proxies>`_ 之间共享内存操作的可见性，并防止工作窃取循环迭代之间的数据竞争。

.. _constraints-thread-block-cancellation:

4.12.1.2. 线程块取消的约束
~~~~~~~~~~~~~~~~~~~~~~~~~~

这些约束与失败的取消请求有关：

- 在**观察**到先前失败的请求后提交另一个取消请求是*未定义行为*。

  在下面的两个代码示例中，假设第一个取消请求失败，只有第一个示例表现出未定义行为。第二个示例是正确的，因为取消请求之间没有观察：

  **无效代码：**

  .. code-block:: cuda
     :linenos:

     // 第一个请求：
     ptx::clusterlaunchcontrol_try_cancel(&result0, &bar0);

     // 第一个请求查询：
     [同步 bar0 的代码放在这里。]
     bool success0 = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result0);
     assert(!success0);  // 观察到失败；第二个取消将是无效的。

     // 第二个请求 - 下一行是未定义行为：
     ptx::clusterlaunchcontrol_try_cancel(&result1, &bar1);

  **有效代码：**

  .. code-block:: cuda
     :linenos:

     // 第一个请求：
     ptx::clusterlaunchcontrol_try_cancel(&result0, &bar0);

     // 第二个请求：
     ptx::clusterlaunchcontrol_try_cancel(&result1, &bar1);

     // 第一个请求查询：
     [同步 bar0 的代码放在这里。]
     bool success0 = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result0);
     assert(!success0);  // 观察到失败；第二个取消是有效的。

- 获取失败取消请求的线程块索引是未定义行为。

- 不建议从多个线程提交取消请求。这会导致多个线程块被取消，需要小心处理，例如：

  - 每个提交线程必须提供唯一的 ``__shared__`` 结果指针以避免数据竞争。

  - 如果使用相同的屏障进行同步，则必须相应调整到达计数和事务计数。

4.12.2. 示例：向量-标量乘法
---------------------------

在以下小节中，我们通过向量-标量乘法 kernel 演示使用 cluster launch control 的工作窃取。我们展示了同一问题的两个变体：一个使用线程块，一个使用线程块集群。

.. _use-case-thread-blocks:

4.12.2.1. 用例：线程块
~~~~~~~~~~~~~~~~~~~~~~

下面的三个 kernel 演示了向量-标量乘法 :math:`\overline{v} := \alpha \overline{v}` 的 *固定每个线程块的工作量*、*固定线程块数量* 和 *Cluster Launch Control* 方法。

- 固定每个线程块的工作量：

  .. code-block:: cuda
     :linenos:

     __global__
     void kernel_fixed_work (float* data, int n)
     {
         // 序言：
         float alpha = compute_scalar();

         // 计算：
         int i = blockIdx.x * blockDim.x + threadIdx.x;
         if (i < n)
             data[i] *= alpha;
     }

     // 启动：kernel_fixed_work<<<1024, (n + 1023) / 1024>>>(data, n);

- 固定线程块数量：

  .. code-block:: cuda
     :linenos:

     __global__
     void kernel_fixed_blocks (float* data, int n)
     {
         // 序言：
         float alpha = compute_scalar();

         // 计算：
         int i = blockIdx.x * blockDim.x + threadIdx.x;
         while (i < n) {
             data[i] *= alpha;
             i += gridDim.x * blockDim.x;
         }
     }

     // 启动：kernel_fixed_blocks<<<1024, SM_COUNT>>>(data, n);

- Cluster Launch Control：

  .. code-block:: cuda
     :linenos:

     #include <cooperative_groups.h>
     #include <cuda/ptx>

     namespace cg = cooperative_groups;
     namespace ptx = cuda::ptx;

     __global__
     void kernel_cluster_launch_control (float* data, int n)
     {
         // Cluster launch control 初始化：
         __shared__ uint4 result;
         __shared__ uint64_t bar;
         int phase = 0;

         if (cg::thread_block::thread_rank() == 0)
             ptx::mbarrier_init(&bar, 1);

         // 序言：
         float alpha = compute_scalar();  // 此代码片段中未显示设备函数。

         // 工作窃取循环：
         int bx = blockIdx.x;  // 假设是一维 x 轴线程块。

         while (true) {
             // 保护结果在下一次迭代中被覆盖，
             // （也确保在第一次迭代时屏障初始化）：
             __syncthreads();

             // 取消请求：
             if (cg::thread_block::thread_rank() == 0) {
                 // 在异步代理中获取结果的写入：
                 ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);

                 cg::invoke_one(cg::coalesced_threads(), [&](){ptx::clusterlaunchcontrol_try_cancel(&result, &bar);});
                 ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
             }

             // 计算：
             int i = bx * blockDim.x + threadIdx.x;
             if (i < n)
                 data[i] *= alpha;

             // 取消请求同步：
             while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase))
             {}
             phase ^= 1;

             // 取消请求解码：
             bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
             if (!success)
                 break;

             bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);

             // 将结果的读取释放到异步代理：
             ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
         }
     }

     // 启动：kernel_cluster_launch_control<<<1024, (n + 1023) / 1024>>>(data, n);

.. _use-case-thread-block-clusters:

4.12.2.2. 用例：线程块集群
~~~~~~~~~~~~~~~~~~~~~~~~~~

在 :ref:`sec:thread-block-clusters` 的情况下，线程块取消步骤与非集群设置相同，只是略有调整。与非集群情况一样，不建议从**集群内**的多个线程提交取消请求，因为这将尝试取消多个集群。

- 取消由单个集群线程提交。

- 每个集群的线程块的共享内存结果将接收相同的（编码的）被取消线程块索引值（即，结果值被多播）。所有线程块接收的结果对应于集群内的本地块索引 ``{0, 0, 0}`` 。因此，集群内的线程块需要添加本地块索引。

- 同步由每个集群的线程块使用本地 ``__shared__`` 内存屏障执行。屏障操作必须使用 ``ptx::scope_cluster`` 作用域执行。

- 集群情况下的取消需要所有线程块都存在。用户可以使用 :ref:`sec:cg-api-sync-function` API 中的 ``cg::cluster_group::sync()`` 来保证所有线程块都在运行。

下面的 kernel 演示了使用线程块集群的 cluster launch control 方法。

.. code-block:: cuda
   :linenos:

   #include <cooperative_groups.h>
   #include <cuda/ptx>

   namespace cg = cooperative_groups;
   namespace ptx = cuda::ptx;

   __global__ __cluster_dims__(2, 1, 1)
   void kernel_cluster_launch_control (float* data, int n)
   {
       // Cluster launch control 初始化：
       __shared__ uint4 result;
       __shared__ uint64_t bar;
       int phase = 0;

       if (cg::thread_block::thread_rank() == 0) {
           ptx::mbarrier_init(&bar, 1);
           ptx::fence_mbarrier_init(ptx::sem_release, ptx::scope_cluster);  // CGA 级别栅栏。
       }

       // 序言：
       float alpha = compute_scalar();  // 此代码片段中未显示设备函数。

       // 工作窃取循环：
       int bx = blockIdx.x;  // 假设是一维 x 轴线程块。

       while (true) {
           // 保护结果在下一次迭代中被覆盖，
           // （也确保所有线程块在第一次迭代时已启动）：
           cg::cluster_group::sync();

           // 由单个集群线程取消请求：
           if (cg::cluster_group::thread_rank() == 0) {
               // 在异步代理中获取结果的写入：
               ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);

               cg::invoke_one(cg::coalesced_threads(), [&](){ptx::clusterlaunchcontrol_try_cancel_multicast(&result, &bar);});
           }

           // 每个线程块跟踪的取消完成：
           if (cg::thread_block::thread_rank() == 0)
               ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cluster, ptx::space_shared, &bar, sizeof(uint4));

           // 计算：
           int i = bx * blockDim.x + threadIdx.x;
           if (i < n)
               data[i] *= alpha;

           // 取消请求同步：
           while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cluster, &bar, phase))
           {}
           phase ^= 1;

           // 取消请求解码：
           bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
           if (!success)
               break;

           bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);
           bx += cg::cluster_group::block_index().x;  // 添加本地偏移。

           // 将结果的读取释放到异步代理：
           ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
       }
   }

   // 启动：kernel_cluster_launch_control<<<1024, (n + 1023) / 1024>>>(data, n);