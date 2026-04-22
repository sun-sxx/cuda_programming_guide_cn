.. _pipelines-details:

4.10. Pipelines
================

Pipelines 在 :ref:`advanced-kernels-advanced-sync-primitives` 中介绍，是一种用于阶段化工作和协调多缓冲区生产者-消费者模式的机制，通常用于将计算与 :ref:`advanced-kernels-async-copies` 重叠执行。

本节详细介绍如何主要通过 ``cuda::pipeline`` API 使用 pipelines（在适用的地方会指向原语 API）。

.. _pipelines-init:

初始化
------

``cuda::pipeline`` 可以在不同的线程作用域创建。对于 ``cuda::thread_scope_thread`` 以外的作用域，需要一个 ``cuda::pipeline_shared_state<scope, count>`` 对象来协调参与的线程。该状态封装了有限的资源，允许 pipeline 处理最多 ``count`` 个并发阶段。

.. code-block:: c++

   // Create a pipeline at thread scope
   constexpr auto scope = cuda::thread_scope_thread;
   cuda::pipeline<scope> pipeline = cuda::make_pipeline();

.. code-block:: c++

   // Create a pipeline at block scope
   constexpr auto scope = cuda::thread_scope_block;
   constexpr auto stages_count = 2;
   __shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
   auto pipeline = cuda::make_pipeline(group, &shared_state);

Pipelines 可以是统一（unified）或分区（partitioned）的。在统一 pipeline 中，所有参与的线程既是生产者也是消费者。在分区 pipeline 中，每个参与的线程要么是生产者要么是消费者，其角色在 pipeline 对象的生命周期内不能改变。线程局部 pipeline 不能被分区。要创建分区 pipeline，我们需要向 ``cuda::make_pipeline()`` 提供生产者数量或线程角色。

.. code-block:: c++

   // Create a partitioned pipeline at block scope where only thread 0 is a producer
   constexpr auto scope = cuda::thread_scope_block;
   constexpr auto stages_count = 2;
   __shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
   auto thread_role = (group.thread_rank() == 0) ? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;
   auto pipeline = cuda::make_pipeline(group, &shared_state, thread_role);

为了支持分区，共享的 ``cuda::pipeline`` 会产生额外的开销，包括每个阶段使用一组共享内存屏障进行同步。即使 pipeline 是统一的并且可以使用 ``__syncthreads()`` ，这些开销也会产生。因此，在可能的情况下，最好使用线程局部 pipeline 来避免这些开销。

.. _pipelines-submit:

提交工作
--------

将工作提交到 pipeline 阶段涉及：

- 使用 ``pipeline.producer_acquire()`` 从一组生产者线程中集体获取（acquire）pipeline 头部（head）。
- 向 pipeline 头部提交异步操作，例如 ``memcpy_async`` 。
- 使用 ``pipeline.producer_commit()`` 集体提交（commit）（推进）pipeline 头部。

如果所有资源都在使用中， ``pipeline.producer_acquire()`` 会阻塞生产者线程，直到消费者线程释放下一个 pipeline 阶段的资源。

.. _pipelines-consume:

消费工作
--------

从先前提交的阶段消费工作涉及：

- 使用 ``pipeline.consumer_wait()`` 从一组消费者线程中集体等待阶段完成，例如等待尾部（最旧的）阶段。
- 使用 ``pipeline.consumer_release()`` 集体释放（release）该阶段。

对于 ``cuda::pipeline<cuda:thread_scope_thread>`` ，还可以使用 ``cuda::pipeline_consumer_wait_prior<N>()`` 友元函数等待除最后 N 个阶段外的所有阶段完成，类似于原语 API 中的 ``__pipeline_wait_prior(N)`` 。

.. _pipelines-entanglement:

Warp 纠缠
---------

Pipeline 机制在同一个 warp 中的 CUDA 线程之间共享。这种共享导致提交的操作序列在一个 warp 内被纠缠，在某些情况下可能会影响性能。

**提交（Commit）**\ 。提交操作会被合并，使得 pipeline 的序列对于所有调用提交操作的收敛线程只递增一次，并且它们提交的操作会被批处理在一起。如果 warp 完全收敛，序列递增一，所有提交的操作将被批处理在 pipeline 的同一阶段；如果 warp 完全发散，序列递增 32，所有提交的操作将被分散到不同的阶段。

- 设 *PB* 为 warp 共享 pipeline 的实际操作序列。

  ``PB = {BP0, BP1, BP2, …, BPL}``

- 设 *TB* 为线程的感知操作序列，就好像序列只由此线程调用提交操作而递增。

  ``TB = {BT0, BT1, BT2, …, BTL}``

  ``pipeline::producer_commit()`` 的返回值来自线程的感知批处理序列。

- 线程感知序列中的索引总是对齐到实际 warp 共享序列中相等或更大的索引。只有当所有提交操作都由完全收敛的线程调用时，序列才相等。

  ``BTn ≡ BPm`` 其中 ``n <= m``

例如，当一个 warp 完全发散时：

- warp 共享 pipeline 的实际序列将是： ``PB = {0, 1, 2, 3, ..., 31}`` （ ``PL=31`` ）。
- 该 warp 每个线程的感知序列将是：

  - Thread 0: ``TB = {0}`` （ ``TL=0`` ）
  - Thread 1: ``TB = {0}`` （ ``TL=0`` ）
  - ``…``
  - Thread 31: ``TB = {0}`` （ ``TL=0`` ）

**等待（Wait）**\ 。CUDA 线程调用 ``pipeline::consumer_wait()`` 或 ``pipeline_consumer_wait_prior<N>()`` 来等待感知序列 ``TB`` 中的批处理完成。注意 ``pipeline::consumer_wait()`` 等价于 ``pipeline_consumer_wait_prior<N>()`` ，其中 ``N = PL`` 。

**等待优先（wait prior）**\ 变体等待实际序列中至少到并包括 ``PL-N`` 的批处理。由于 ``TL <= PL`` ，等待到并包括 ``PL-N`` 的批处理包括等待批处理 ``TL-N`` 。因此，当 ``TL < PL`` 时，线程会意外地等待额外的、更近的批处理。在上面的极端完全发散 warp 示例中，每个线程可能等待所有 32 个批处理。

.. note::

   建议提交操作由收敛的线程调用，以避免过度等待，通过保持线程的感知批处理序列与实际序列对齐来实现。

   当这些操作之前的代码使线程发散时，应该在调用提交操作之前通过 ``__syncwarp`` 重新收敛 warp。

.. _pipelines-early-exit:

提前退出
--------

当参与 pipeline 的线程必须提前退出时，该线程必须在退出前使用 ``cuda::pipeline::quit()`` 显式退出参与。剩余的参与线程可以正常进行后续操作。

.. _pipelines-tracking:

跟踪异步内存操作
----------------

以下示例演示了如何使用 pipeline 跟踪复制操作，通过异步内存复制集体将数据从全局内存复制到共享内存。每个线程使用自己的 pipeline 独立提交内存复制，然后等待它们完成并消费数据。关于异步数据复制的更多详细信息，请参阅 :numref:`Section 3.2.5`。

.. raw:: html

   <details>
   <summary><b>CUDA C++ <code>cuda::pipeline</code></b></summary>

.. code-block:: cuda

   #include <cuda/pipeline>

   __global__ void example_kernel(const float *in)
   {
       constexpr int block_size = 128;
       __shared__ __align__(sizeof(float)) float buffer[4 * block_size];

       // Create a unified pipeline per thread
       cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

       // First stage of memory copies
       pipeline.producer_acquire();
       // Every thread fetches one element of the first block
       cuda::memcpy_async(buffer, in, sizeof(float), pipeline);
       pipeline.producer_commit();

       // Second stage of memory copies
       pipeline.producer_acquire();
       // Every thread fetches one element of the second and third block
       cuda::memcpy_async(buffer + block_size, in + block_size, sizeof(float), pipeline);
       cuda::memcpy_async(buffer + 2 * block_size, in + 2 * block_size, sizeof(float), pipeline);
       pipeline.producer_commit();

       // Third stage of memory copies
       pipeline.producer_acquire();
       // Every thread fetches one element of the last block
       cuda::memcpy_async(buffer + 3 * block_size, in + 3 * block_size, sizeof(float), pipeline);
       pipeline.producer_commit();

       // Wait for the oldest stage (waits for first stage)
       pipeline.consumer_wait();
       pipeline.consumer_release();

       // __syncthreads();
       // Use data from the first stage

       // Wait for the oldest stage (waits for second stage)
       pipeline.consumer_wait();
       pipeline.consumer_release();

       // __syncthreads();
       // Use data from the second stage

       // Wait for the oldest stage (waits for third stage)
       pipeline.consumer_wait();
       pipeline.consumer_release();

       // __syncthreads();
       // Use data from the third stage
   }

.. raw:: html

   </details>
   <details>
   <summary><b>CUDA C primitives</b></summary>

.. code-block:: cuda

   #include <cuda_pipeline.h>

   __global__ void example_kernel(const float *in)
   {
       constexpr int block_size = 128;
       __shared__ __align__(sizeof(float)) float buffer[4 * block_size];

       // First batch of memory copies
       // Every thread fetches one element of the first block
       __pipeline_memcpy_async(buffer, in, sizeof(float));
       __pipeline_commit();

       // Second batch of memory copies
       // Every thread fetches one element of the second and third block
       __pipeline_memcpy_async(buffer + block_size, in + block_size, sizeof(float));
       __pipeline_memcpy_async(buffer + 2 * block_size, in + 2 * block_size, sizeof(float));
       __pipeline_commit();

       // Third batch of memory copies
       // Every thread fetches one element of the last block
       __pipeline_memcpy_async(buffer + 3 * block_size, in + 3 * block_size, sizeof(float));
       __pipeline_commit();

       // Wait for all except the last two batches of memory copies (waits for first batch)
       __pipeline_wait_prior(2);

       // __syncthreads();
       // Use data from the first batch

       // Wait for all except the last batch of memory copies (waits for second batch)
       __pipeline_wait_prior(1);

       // __syncthreads();
       // Use data from the second batch

       // Wait for all batches of memory copies (waits for third batch)
       __pipeline_wait_prior(0);

       // __syncthreads();
       // Use data from the last batch
   }

.. raw:: html

   </details>

.. _pipelines-producer-consumer:

使用 Pipelines 的生产者-消费者模式
----------------------------------

在 :numref:`Section 4.9.7` 中，我们展示了如何使用 :ref:`asynchronous-barriers` 对线程块进行空间分区以实现生产者-消费者模式。使用 ``cuda::pipeline`` ，可以通过单个分区 pipeline 简化这一过程，每个数据缓冲区使用一个阶段，而不是每个缓冲区使用两个异步屏障。

.. code-block:: cuda

   #include <cuda/pipeline>
   #include <cooperative_groups.h>

   #pragma nv_diag_suppress static_var_with_dynamic_init

   using pipeline = cuda::pipeline<cuda::thread_scope_block>;

   __device__ void produce(pipeline &pipe, int num_stages, int stage, int num_batches, int batch, float *buffer, int buffer_len, float *in, int N)
   {
     if (batch < num_batches)
     {
       pipe.producer_acquire();
       /* copy data from in(batch) to buffer(stage) using asynchronous memory copies */
       pipe.producer_commit();
     }
   }

   __device__ void consume(pipeline &pipe, int num_stages, int stage, int num_batches, int batch, float *buffer, int buffer_len, float *out, int N)
   {
     pipe.consumer_wait();
     /* consume buffer(stage) and update out(batch) */
     pipe.consumer_release();
   }

   __global__ void producer_consumer_pattern(float *in, float *out, int N, int buffer_len)
   {
     auto block = cooperative_groups::this_thread_block();

     /* Shared memory buffer declared below is of size 2 * buffer_len
        so that we can alternatively work between two buffers.
        buffer_0 = buffer and buffer_1 = buffer + buffer_len */
     __shared__ extern float buffer[];

     const int num_batches = N / buffer_len;

     // Create a partitioned pipeline with 2 stages where half the threads are producers and the other half are consumers.
     constexpr auto scope = cuda::thread_scope_block;
     constexpr int num_stages = 2;
     cuda::std::size_t producer_count = block.size() / 2;
     __shared__ cuda::pipeline_shared_state<scope, num_stages> shared_state;
     pipeline pipe = cuda::make_pipeline(block, &shared_state, producer_count);

     // Fill the pipeline
     if (block.thread_rank() < producer_count)
     {
       for (int s = 0; s < num_stages; ++s)
       {
         produce(pipe, num_stages, s, num_batches, s, buffer, buffer_len, in, N);
       }
     }

     // Process the batches
     int stage = 0;
     for (size_t b = 0; b < num_batches; ++b)
     {
       if (block.thread_rank() < producer_count)
       {
         // Prefetch the next batch
         produce(pipe, num_stages, stage, num_batches, b + num_stages, buffer, buffer_len, in, N);
       }
       else
       {
         // Consume the oldest batch
         consume(pipe, num_stages, stage, num_batches, b, buffer, buffer_len, out, N);
       }
       stage = (stage + 1) % num_stages;
     }
   }

在这个示例中，我们使用线程块中一半的线程作为生产者，另一半作为消费者。首先，我们需要创建一个 ``cuda::pipeline`` 对象。由于我们希望一些线程是生产者而另一些是消费者，我们需要使用 ``cuda::thread_scope_block`` 的分区 pipeline。分区 pipeline 需要一个 ``cuda::pipeline_shared_state`` 来协调参与的线程。我们为线程块作用域的 2 阶段 pipeline 初始化状态，然后调用 ``cuda::make_pipeline()`` 。接下来，生产者线程通过提交从 ``in`` 到 ``buffer`` 的异步复制来填充 pipeline。此时所有数据复制都在进行中。最后，在主循环中，我们遍历所有数据批次，根据线程是生产者还是消费者，我们要么为未来的批次提交另一个异步复制，要么消费当前批次。
