.. _async-copies-details:

4.11. 异步数据拷贝
==================

基于 :doc:`../03-advanced/02-advanced-kernel-programming` 的第 3.2.5 节，本节为 GPU 内存层次结构内的异步数据移动提供详细指导和示例。它涵盖了用于元素级拷贝的 LDGSTS、用于批量（一维和多维）传输的张量内存加速器 (TMA)、用于寄存器到分布式共享内存拷贝的 STAS，并展示了这些机制如何与 :doc:`09-async-barriers` 和 :doc:`10-pipelines` 集成。

.. _using-ldgsts:

4.11.1. 使用 LDGSTS
-------------------

许多 CUDA 应用程序需要在 Global 内存和共享内存之间频繁移动数据。通常，这涉及复制较小的数据元素或执行不规则的内存访问模式。LDGSTS（CC 8.0+，参见 `PTX 文档 <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy>`_）的主要目标是为较小的元素级数据传输提供从 Global 内存到共享内存的有效异步数据传输机制，同时通过重叠执行更好地利用计算资源。

**维度**。LDGSTS 支持复制 4、8 或 16 字节。复制 4 或 8 字节始终以所谓的 L1 ACCESS 模式进行，此时数据也缓存在 L1 中，而复制 16 字节启用 L1 BYPASS 模式，此时 L1 不会被污染。

**源和目标**。LDGSTS 异步拷贝操作支持的唯一方向是从 Global 内存到共享内存。指针需要根据复制的数据大小对齐到 4、8 或 16 字节。当共享内存和 Global 内存的对齐都是 128 字节时，可获得最佳性能。

**异步性**。使用 LDGSTS 的数据传输是 `异步的 <../03-advanced/02-advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features>`_，并建模为异步线程操作（参见 `异步线程和异步代理 <../03-advanced/02-advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features-async-thread-proxy>`_）。这允许发起线程继续计算，而硬件异步复制数据。*数据传输是否实际异步执行取决于硬件实现，未来可能会发生变化*。

LDGSTS 必须在操作完成时提供信号。LDGSTS 可以使用 `共享内存屏障 <../03-advanced/02-advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers>`_ 或 `管道 <../03-advanced/02-advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-pipelines>`_ 作为提供完成信号的机制。默认情况下，每个线程只等待自己的 LDGSTS 拷贝。因此，如果您使用 LDGSTS 预取一些将与其他线程共享的数据，则在同步 LDGSTS 完成机制后需要 ``__syncthreads()``。

.. list-table:: 使用 LDGSTS 的异步拷贝的可能源和目标内存空间及完成机制。空白单元格表示不支持的源 - 目标对。
   :widths: 20 20 15 15 15 15
   :header-rows: 1

   * - 方向
     - 异步拷贝 (LDGSTS, CC 8.0+)
     - 源
     - 目标
     - 完成机制
     - API
   * - Global → Shared
     - ✓
     - Global
     - shared::cta
     - 共享内存屏障、管道
     - ``cuda::memcpy_async``、``cooperative_groups::memcpy_async``、``__pipeline_memcpy_async``
   * - Global → Cluster Shared
     - ✓
     - Global
     - shared::cluster
     - shared::cta
     - -

在接下来的章节中，我们将通过示例演示如何使用 LDGSTS，并解释不同 API 之间的差异。

.. _async-copies-batching-loads:

4.11.1.1. 条件代码中的批量加载
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在这个模板示例中，线程块的第一个 warp 负责集体加载中心和左右 halo 所需的所有数据。使用同步拷贝时，由于代码的条件性质，编译器可能会选择生成一系列从 Global 加载 (LDG) 到共享存储 (STS) 的指令，而不是 3 个 LDG 后跟 3 个 STS，这将是加载数据以隐藏 Global 内存延迟的最佳方式。

.. code-block:: cuda

   __global__ void stencil_kernel(const float *left, const float *center, const float *right)
   {
       // 左 halo（8 个元素）- 中心（32 个元素）- 右 halo（8 个元素）
       __shared__ float buffer[8 + 32 + 8];
       const int tid = threadIdx.x;

       if (tid < 8) {
           buffer[tid] = left[tid]; // 左 halo
       } else if (tid >= 32 - 8) {
           buffer[tid + 16] = right[tid]; // 右 halo
       }
       if (tid < 32) {
         buffer[tid + 8] = center[tid]; // 中心
       }
       __syncthreads();

       // 计算模板
   }

为了确保以最佳方式加载数据，我们可以用异步内存拷贝替换同步内存拷贝，直接从 Global 内存加载数据到共享内存。这不仅通过直接将数据复制到共享内存来减少寄存器使用，还确保所有来自 Global 内存的加载都在进行中。

使用 CUDA C++ ``cuda::memcpy_async``:

.. code-block:: cuda

   #include <cooperative_groups.h>
   #include <cuda/barrier>

   __global__ void stencil_kernel(const float *left, const float *center, const float *right)
   {
       auto block = cooperative_groups::this_thread_block();
       auto thread = cooperative_groups::this_thread();
       using barrier_t = cuda::barrier<cuda::thread_scope_block>;
       __shared__ barrier_t barrier;
       __shared__ float buffer[8 + 32 + 8];
       
       // 初始化同步对象
       if (block.thread_rank() == 0) {
           init(&barrier, block.size());
       }
       __syncthreads();

       // 版本 1：在各个线程中发出拷贝
       if (tid < 8) {
           cuda::memcpy_async(buffer + tid, left + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier); // 左 halo
           // 或 cuda::memcpy_async(thread, buffer + tid, left + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier);
       } else if (tid >= 32 - 8) {
           cuda::memcpy_async(buffer + tid + 16, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier); // 右 halo
           // 或 cuda::memcpy_async(thread, buffer + tid + 16, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier);
       }
       if (tid < 32) {
           cuda::memcpy_async(buffer + 40, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier); // 中心
           // 或 cuda::memcpy_async(thread, buffer + 40, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier);
       }
       
       // 版本 2：跨所有线程集体发出拷贝
       cuda::memcpy_async(block, buffer, left, cuda::aligned_size_t<4>(8 * sizeof(float)), barrier); // 左 halo
       cuda::memcpy_async(block, buffer + 8, center, cuda::aligned_size_t<4>(32 * sizeof(float)), barrier); // 中心
       cuda::memcpy_async(block, buffer + 40, right, cuda::aligned_size_t<4>(8 * sizeof(float)), barrier); // 右 halo
       
       // 等待所有拷贝完成
       barrier.arrive_and_wait();
       __syncthreads();

       // 计算模板      
   }

使用 ``cooperative_groups::memcpy_async``:

.. code-block:: cuda

   #include <cooperative_groups.h>
   #include <cooperative_groups/memcpy_async.h>

   namespace cg = cooperative_groups;

   __global__ void stencil_kernel(const float *left, const float *center, const float *right)
   {
       cg::thread_block block = cg::this_thread_block();
       // 左 halo（8 个元素）- 中心（32 个元素）- 右 halo（8 个元素）
       __shared__ float buffer[8 + 32 + 8];

       // 跨所有线程集体发出拷贝
       cg::memcpy_async(block, buffer, left, 8 * sizeof(float)); // 左 halo
       cg::memcpy_async(block, buffer + 8, center, 32 * sizeof(float)); // 中心
       cg::memcpy_async(block, buffer + 40, right, 8 * sizeof(float)); // 右 halo
       cg::wait(block); // 等待所有拷贝完成
       __syncthreads();

       // 计算模板
   }

使用 CUDA C 原始 API:

.. code-block:: cuda

   #include <cuda_pipeline.h>

   __global__ void stencil_kernel(const float *left, const float *center, const float *right)
   {
       // 左 halo（8 个元素）- 中心（32 个元素）- 右 halo（8 个元素）
       __shared__ float buffer[8 + 32 + 8];
       const int tid = threadIdx.x;

       if (tid < 8) {
           __pipeline_memcpy_async(buffer + tid, left + tid, sizeof(float)); // 左 halo
       } else if (tid >= 32 - 8) {
           __pipeline_memcpy_async(buffer + tid + 16, right + tid, sizeof(float)); // 右 halo
       }
       if (tid < 32) {
           __pipeline_memcpy_async(buffer + tid + 8, center + tid, sizeof(float)); // 中心
       }
       __pipeline_commit();
       __pipeline_wait_prior(0);
       __syncthreads();

       // 计算模板
   }

``cuda::memcpy_async`` 用于 ``cuda::barrier`` 的重载允许使用 `异步屏障 <../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers>`_ 同步异步数据传输。此重载执行拷贝操作，就像由绑定到屏障的另一个线程执行一样，通过增加当前相位的预期计数，并在拷贝操作完成时减少它，使得 ``barrier`` 的相位只有在屏障参与的所有线程都已到达且所有绑定到屏障当前相位的 ``memcpy_async`` 都完成后才会推进。我们使用块级 ``barrier``，块中的所有线程都参与，并使用 ``arrive_and_wait`` 合并屏障的到达和等待，因为在相位之间我们不执行任何工作。

请注意，我们可以使用线程级拷贝（版本 1）或集体拷贝（版本 2）来达到相同的结果。在版本 2 中，API 将自动处理底层拷贝的完成方式。在这两个版本中，我们使用 ``cuda::aligned_size_t<4>()`` 告知编译器数据按 4 字节对齐且拷贝的数据大小是 4 的倍数，以启用 LDGSTS。请注意，为了与 ``cuda::barrier`` 互操作，这里使用来自 ``cuda/barrier`` 头文件的 ``cuda::memcpy_async``。

`cooperative_groups::memcpy_async <../05-appendices/device-callable-apis.html#cg-api-async-memcpy>`_ 实现在块的所有线程中集体协调内存传输，但使用 ``cg::wait(block)`` 而不是显式屏障操作来同步完成。

基于低级原始 API 的实现使用 ``__pipeline_memcpy_async()`` 启动元素级内存传输，``__pipeline_commit()`` 提交一批拷贝，``__pipeline_wait_prior(0)`` 等待管道中的所有操作完成。与高级 API 相比，这提供了最直接的控制，但代码更冗长。它还确保底层将使用 LDGSTS，而高级 API 不保证这一点。

.. note::

   ``cooperative_groups::memcpy_async`` API 在此示例中效率较低，因为它会在启动时自动立即提交每个拷贝操作，从而阻止了其他 API 能够实现的在单个提交操作之前批处理多个拷贝的优化。

.. _async-copies-prefetching:

4.11.1.2. 预取数据
^^^^^^^^^^^^^^^^^^

在此示例中，我们将演示如何使用异步数据拷贝从 Global 内存预取数据到共享内存。在迭代拷贝和计算模式中，这允许用当前迭代的计算隐藏未来迭代的数据传输延迟，可能增加飞行中的字节数。

使用 CUDA C++ ``cuda::memcpy_async``:

.. code-block:: cuda

   #include <cooperative_groups.h>
   #include <cuda/pipeline>

   template <size_t num_stages = 2 /* 具有 num_stages 阶段的管道 */>
   __global__ void prefetch_kernel(int* global_out, int const* global_in, size_t size, size_t batch_size) {
       auto grid = cooperative_groups::this_grid();
       auto block = cooperative_groups::this_thread_block();
       auto thread = cooperative_groups::this_thread();
       assert(size == batch_size * grid.size()); // 假设输入大小适合 batch_size * grid_size

       extern __shared__ int shared[]; // num_stages * block.size() * sizeof(int) 字节
       size_t shared_offset[num_stages];
       for (int s = 0; s < num_stages; ++s) shared_offset[s] = s * block.size();

       cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

       auto block_batch = [&](size_t batch) -> int {
           return block.group_index().x * block.size() + grid.size() * batch;
       };

       // 用前 num_stages 个批次填充管道
       for (int s = 0; s < num_stages; ++s) {
           pipeline.producer_acquire();
           cuda::memcpy_async(shared + shared_offset[s] + tid, global_in + block_batch(s) + tid, 
                              cuda::aligned_size_t<4>(sizeof(int)), pipeline);
           pipeline.producer_commit();
       }

       int stage = 0;

       // compute_batch: 下一个要处理的批次
       // fetch_batch:   下一个要从 Global 内存获取的批次
       for (size_t compute_batch = 0, fetch_batch = num_stages; compute_batch < batch_size; 
            ++compute_batch, ++fetch_batch) {
           // 等待第一个请求的阶段完成
           constexpr size_t pending_batches = num_stages - 1;
           cuda::pipeline_consumer_wait_prior<pending_batches>(pipeline);
           __syncthreads(); // 如果每个线程处理它拷贝的数据则不需要

           // 在当前批次上计算
           compute(global_out + block_batch(compute_batch) + tid, shared + shared_offset[stage] + tid);
           
           // 释放当前阶段
           pipeline.consumer_release();
           __syncthreads(); // 如果每个线程处理它拷贝的数据则不需要

           // 加载未来阶段，领先当前计算批次 num_stages
           pipeline.producer_acquire();
           if (fetch_batch < batch_size) {
               cuda::memcpy_async(shared + shared_offset[stage] + tid, 
                                  global_in + block_batch(fetch_batch) + tid, 
                                  cuda::aligned_size_t<4>(sizeof(int)), pipeline);
           }
           pipeline.producer_commit();
           stage = (stage + 1) % num_stages;
       }
   }

使用 CUDA C++ ``cooperative_groups::memcpy_async``:

.. code-block:: cuda

   #include <cooperative_groups.h>
   #include <cooperative_groups/memcpy_async.h>

   namespace cg = cooperative_groups;

   template <size_t num_stages = 2 /* 具有 num_stages 阶段的管道 */>
   __global__ void prefetch_kernel(int* global_out, int const* global_in, size_t size, size_t batch_size) {
       auto grid = cooperative_groups::this_grid();
       auto block = cooperative_groups::this_thread_block();
       assert(size == batch_size * grid.size()); // 假设输入大小适合 batch_size * grid_size

       extern __shared__ int shared[]; // num_stages * block.size() * sizeof(int) 字节
       size_t shared_offset[num_stages];
       for (int s = 0; s < num_stages; ++s) shared_offset[s] = s * block.size();

       auto block_batch = [&](size_t batch) -> int {
           return block.group_index().x * block.size() + grid.size() * batch;
       };

       // 用前 num_stages 个批次填充管道
       for (int s = 0; s < num_stages; ++s) {
           size_t block_batch_idx = block_batch(s);
           cg::memcpy_async(block, shared + shared_offset[s], global_in + block_batch_idx, 
                            cuda::aligned_size_t<4>(sizeof(int)));
       }

       int stage = 0;

       // compute_batch: 下一个要处理的批次
       // fetch_batch:   下一个要从 Global 内存获取的批次
       for (size_t compute_batch = 0, fetch_batch = num_stages; compute_batch < batch_size; 
            ++compute_batch, ++fetch_batch) {
           // 等待第一个请求的阶段完成
           size_t pending_batches = (fetch_batch < batch_size - num_stages) ? num_stages - 1 : batch_size - fetch_batch - 1;
           cg::wait_prior(pending_batches);
           __syncthreads(); // 如果每个线程处理它拷贝的数据则不需要

           // 在当前批次上计算
           compute(global_out + block_batch(compute_batch) + tid, shared + shared_offset[stage] + tid);
           
           __syncthreads(); // 如果每个线程处理它拷贝的数据则不需要

           // 加载未来阶段，领先当前计算批次 num_stages
           size_t fetch_batch_idx = block_batch(fetch_batch);
           if (fetch_batch < batch_size) {
               cg::memcpy_async(block, shared + shared_offset[stage], global_in + block_batch(fetch_batch), 
                                cuda::aligned_size_t<4>(sizeof(int)) * block.size());
           }
           stage = (stage + 1) % num_stages;
       }
   }

使用 CUDA C 原始 API:

.. code-block:: cuda

   #include <cooperative_groups.h>
   #include <cuda_awbarrier_primitives.h>

   template <size_t num_stages = 2 /* 具有 num_stages 阶段的管道 */>
   __global__ void prefetch_kernel(int* global_out, int const* global_in, size_t size, size_t batch_size) {
       auto grid = cooperative_groups::this_grid();
       auto block = cooperative_groups::this_thread_block();
       assert(size == batch_size * grid.size()); // 假设输入大小适合 batch_size * grid_size

       extern __shared__ int shared[]; // num_stages * block.size() * sizeof(int) 字节
       size_t shared_offset[num_stages];
       for (int s = 0; s < num_stages; ++s) shared_offset[s] = s * block.size();

       auto block_batch = [&](size_t batch) -> int {
           return block.group_index().x * block.size() + grid.size() * batch;
       };

       // 用前 num_stages 个批次填充管道
       for (int s = 0; s < num_stages; ++s) {
           __pipeline_memcpy_async(shared + shared_offset[s] + tid, global_in + block_batch(s) + tid, 
                                   cuda::aligned_size_t<4>(sizeof(int)));
           __pipeline_commit();
       }

       // compute_batch: 下一个要处理的批次
       // fetch_batch:   下一个要从 Global 内存获取的批次
       for (size_t compute_batch = 0, fetch_batch = num_stages; compute_batch < batch_size; 
            ++compute_batch, ++fetch_batch) {
           // 等待第一个请求的阶段完成
           constexpr size_t pending_batches = num_stages - 1;
           __pipeline_wait_prior<pending_batches>();
           __syncthreads(); // 如果每个线程处理它拷贝的数据则不需要

           // 在当前批次上计算
           compute(global_out + block_batch(compute_batch) + tid, shared + shared_offset[stage] + tid);
           
           __syncthreads(); // 如果每个线程处理它拷贝的数据则不需要

           // 加载未来阶段，领先当前计算批次 num_stages
           if (fetch_batch < batch_size) {
               __pipeline_memcpy_async(shared + shared_offset[stage] + tid, 
                                       global_in + block_batch(fetch_batch) + tid, 
                                       cuda::aligned_size_t<4>(sizeof(int)));
           }
           __pipeline_commit();
           stage = (stage + 1) % num_stages;
       }
   }

``cuda::memcpy_async`` 实现演示了使用 ``cuda::pipeline``（参见 `管道 <../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-pipelines>`_）和 ``cuda::memcpy_async`` 的多阶段数据预取。它：

- 初始化一个线程本地的管道
- 通过调度 ``num_stages`` 个 ``memcpy_async`` 操作启动管道
- 循环所有批次：阻塞所有线程等待当前批次完成，然后对当前批次执行计算，最后调度下一个 ``memcpy_async``（如果有的话）

``cooperative_groups::memcpy_async`` 实现演示了使用 ``cooperative_groups::memcpy_async`` 的多阶段数据预取。与前一个实现的主要区别是，我们不使用管道对象，而是依赖 ``cooperative_groups::memcpy_async`` 在底层分阶段调度内存传输。

CUDA C 原始 API 实现以与第一个非常相似的方式演示了使用低级原始 API 的多阶段数据预取。

此示例中实现高效代码生成的一个重要细节是保持 ``num_stages`` 个批次在管道中，即使没有更多批次要获取。这是通过提交到管道来完成的，即使没有更多批次要获取（``pipeline.producer_commit()`` 或 ``__pipeline_commit()``）。请注意，这对于 cooperative groups API 是不可能的，因为我们无法访问内部管道。

.. _async-copies-producer-consumer:

4.11.1.3. 通过 Warp 特化的生产者 - 消费者模式
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在此示例中，我们将演示如何实现生产者 - 消费者模式，其中单个 warp 专门化作为生产者，执行从 Global 到共享内存的异步数据拷贝，而剩余的 warp 从共享内存消费数据并执行计算。为了启用生产者和消费者线程之间的并发性，我们在共享内存中使用双缓冲。当消费者 warp 处理一个缓冲区中的数据时，生产者 warp 异步获取下一批数据到另一个缓冲区。

使用 CUDA C++ ``cuda::memcpy_async``:

.. code-block:: cuda

   #include <cooperative_groups.h>
   #include <cuda/pipeline>

   #pragma nv_diag_suppress static_var_with_dynamic_init

   using pipeline = cuda::pipeline<cuda::thread_scope_block>;

   __device__ void produce(pipeline &pipe, int num_stages, int stage, int num_batches, int batch, 
                          float *buffer, int buffer_len, float *in, int N)
   {
     if (batch < num_batches)
     {
       pipe.producer_acquire();
       /* 使用异步内存拷贝将数据从 in(batch) 拷贝到 buffer(stage) */
       cuda::memcpy_async(buffer + stage * buffer_len + threadIdx.x, in + batch * buffer_len + threadIdx.x, 
                          cuda::aligned_size_t<4>(sizeof(float)), pipe);
       pipe.producer_commit();
     }
   }

   __device__ void consume(pipeline &pipe, int num_stages, int stage, int num_batches, int batch, 
                          float *buffer, int buffer_len, float *out, int N)
   {
     pipe.consumer_wait();
     /* 消费 buffer(stage) 并更新 out(batch) */
     pipe.consumer_release();
   }

   __global__ void producer_consumer_pattern(float *in, float *out, int N, int buffer_len)
   {
     auto block = cooperative_groups::this_thread_block();
     constexpr int warpSize = 32;

     /* 下面声明的共享内存缓冲区大小为 2 * buffer_len
        这样我们就可以在两个缓冲区之间交替工作
        buffer_0 = buffer 且 buffer_1 = buffer + buffer_len */
     __shared__ extern float buffer[];

     const int num_batches = N / buffer_len;

     // 创建一个具有 2 个阶段的分区管道，第一个 warp 是生产者，其他 warp 是消费者
     constexpr auto scope = cuda::thread_scope_block;
     constexpr int num_stages = 2;
     cuda::std::size_t producer_count = warpSize;
     __shared__ cuda::pipeline_shared_state<scope, num_stages> shared_state;
     pipeline pipe = cuda::make_pipeline(block, &shared_state, producer_count);

     // 生产者填充管道
     if (block.thread_rank() < producer_count)
       for (int s = 0; s < num_stages; ++s)
         produce(pipe, num_stages, s, num_batches, s, buffer, buffer_len, in, N);

     // 处理批次
     int stage = 0;
     for (size_t b = 0; b < num_batches; ++b)
     {
       if (block.thread_rank() < producer_count)
       {
         // 生产者预取下一批次
         produce(pipe, num_stages, stage, num_batches, b + num_stages, buffer, buffer_len, in, N);
       }
       else
       {
         // 消费者消费最旧的批次
         consume(pipe, num_stages, stage, num_batches, b, buffer, buffer_len, out, N);
       }
       stage = (stage + 1) % num_stages;
     }
   }

使用 CUDA C 原始 API:

.. code-block:: cuda

   #include <cooperative_groups.h>
   #include <cuda_awbarrier_primitives.h>

   __device__ void produce(__mbarrier_t ready[], __mbarrier_t filled[], float *buffer, int buffer_len, float *in, int N)
   {
     for (int i = 0; i < N / buffer_len; ++i)
     {
       __mbarrier_token_t token = __mbarrier_arrive(&ready[i % 2]); /* 等待 buffer_(i%2) 准备好被填充 */
       while(!__mbarrier_try_wait(&ready[i % 2], token, 1000)) {}
       /* 生产，即填充 buffer_(i%2) */
       __pipeline_memcpy_async(buffer + i * buffer_len + threadIdx.x, in + i * buffer_len + threadIdx.x, 
                               cuda::aligned_size_t<4>(sizeof(float)));
       __pipeline_arrive_on(filled[i % 2]);
       __mbarrier_arrive(filled[i % 2]);  /* buffer_(i%2) 已填充 */
     }
   }

   __device__ void consume(__mbarrier_t ready[], __mbarrier_t filled[], float *buffer, int buffer_len, float *out, int N)
   {
     __mbarrier_arrive(&ready[0]); /* buffer_0 准备好初始填充 */
     __mbarrier_arrive(&ready[1]); /* buffer_1 准备好初始填充 */
     for (int i = 0; i < N / buffer_len; ++i)
     {
       __mbarrier_token_t token = __mbarrier_arrive(&filled[i % 2]);
       while(!__mbarrier_try_wait(&filled[i % 2], token, 1000)) {}
       /* 消费 buffer_(i%2) */
       __mbarrier_arrive(&ready[i % 2]); /* buffer_(i%2) 准备好重新填充 */
     }
   }

   __global__ void producer_consumer_pattern(int N, float *in, float *out, int buffer_len)
   {
     /* 下面声明的共享内存缓冲区大小为 2 * buffer_len
        这样我们就可以在两个缓冲区之间交替工作
        buffer_0 = buffer 且 buffer_1 = buffer + buffer_len */
     __shared__ extern float buffer[];

     /* bar[0] 和 bar[1] 跟踪缓冲区 buffer_0 和 buffer_1 是否准备好被填充
        而 bar[2] 和 bar[3] 跟踪缓冲区 buffer_0 和 buffer_1 是否已填充 */
     __shared__ __mbarrier_t bar[4];

     // 初始化屏障
     auto block = cooperative_groups::this_thread_block();
     if (block.thread_rank() < 4)
       __mbarrier_init(bar + block.thread_rank(), block.size());
     __syncthreads();

     if (block.thread_rank() < warpSize)
       produce(bar, bar + 2, buffer, buffer_len, in, N);
     else
       consume(bar, bar + 2, buffer, buffer_len, out, N);
   }

``cuda::memcpy_async`` 实现演示了使用 ``cuda::memcpy_async`` 和具有 2 个阶段的 ``cuda::pipeline`` 的最高抽象级别 API。它使用分区管道（参见 `管道 <../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-pipelines>`_），其中第一个 warp 作为生产者，其余 warp 作为消费者。生产者最初填充两个管道阶段。然后在主处理循环中，当消费者处理当前批次时，生产者为未来批次获取数据，保持稳定的工作流。

基于原始 API 的 CUDA C 实现结合 ``__pipeline_memcpy_async()`` 与 `共享内存屏障 <../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers>`_ 作为完成机制来协调异步内存传输。``__pipeline_arrive_on()`` 函数将内存拷贝与屏障关联。它将屏障到达计数增加一，当它之前的所有异步操作完成时，到达计数会自动减少一，因此对到达计数的净效应为零。因此，我们还需要使用 ``__mbarrier_arrive()`` 显式等待屏障。

.. _using-tma:

4.11.2. 使用张量内存加速器 (TMA)
--------------------------------

许多应用程序需要将大量数据传输到 Global 内存和从 Global 内存传输。通常，数据在 Global 内存中布局为具有非顺序数据访问模式的多维数组。为了减少 Global 内存访问，在用于计算之前，此类数组的子图块被复制到共享内存。加载和存储涉及容易出错且重复的地址计算。为了卸载这些计算，计算能力 9.0（Hopper）及更高版本（参见 `PTX 文档 <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk>`_）具有 *张量内存加速器* (TMA)。TMA 的主要目标是为多维数组提供从 Global 内存到共享内存的有效数据传输机制。

**命名**。张量内存加速器 (TMA) 是用于指代本节中描述的功能的广泛术语。为了前向兼容性和减少与 PTX ISA 的差异，本节中的文本将 TMA 操作称为 *批量异步拷贝* 或 *批量张量异步拷贝*，具体取决于使用的拷贝类型。术语"批量"用于将这些操作与上一节中描述的异步内存操作进行对比。

**维度**。TMA 支持复制一维和多维数组（最多 5 维）。一维连续数组的批量异步拷贝编程模型与多维数组的批量张量异步拷贝编程模型不同。要执行多维数组的批量张量异步拷贝，硬件需要 `张量映射 <https://docs.nvidia.com/cuda/cuda-driver-api/structCUtensorMap.html#structCUtensorMap>`_。此对象描述多维数组在 Global 和共享内存中的布局。张量映射通常在主机上使用 `cuTensorMapEncode API <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY>`_ 创建，然后作为用 ``__grid_constant__`` 注解的 ``const`` 内核参数从主机传输到设备（参见 `\_\_grid\_constant\_\_ 参数 <../05-appendices/cpp-language-extensions.html#grid-constant>`_）。张量映射作为用 ``__grid_constant__`` 注解的 ``const`` 内核参数从主机传输到设备，可在设备上用于在共享和 Global 内存之间拷贝数据图块。相比之下，执行连续一维数组的批量异步拷贝不需要张量映射：它可以使用指针和大小参数在设备上执行。

**源和目标**。TMA 操作的源和目标地址可以在共享或 Global 内存中。操作可以从 Global 读取到共享内存，从共享写入到 Global 内存，也可以从共享内存复制到同一集群中另一个块的 `分布式共享内存 <../02-basics/02-writing-cuda-kernels.html#writing-cuda-kernels-distributed-shared-memory>`_。此外，在集群中时，批量异步张量操作可以指定为 *多播*。在这种情况下，数据可以从 Global 内存传输到集群中多个块的共享内存。多播功能针对 ``sm_90a`` 目标架构进行了优化，在其他目标上可能 `性能显著降低 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor>`_。因此，建议使用 `计算架构 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list>`_ ``sm_90a`` 使用它。

**异步性**。使用 TMA 的数据传输是 `异步的 <../03-advanced/02-advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features>`_，并建模为异步代理操作（参见 `异步线程和异步代理 <../03-advanced/02-advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features-async-thread-proxy>`_）。这允许发起线程继续计算，而硬件异步复制数据。*数据传输是否实际异步执行取决于硬件实现，未来可能会发生变化*。批量异步操作可以使用多种 `完成机制 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms>`_ 来发出完成信号。当操作从 Global 读取到共享内存时，块中的任何线程都可以通过等待 `共享内存屏障 <../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers>`_ 等待数据在共享内存中可读。当批量异步操作将数据从共享内存写入到 Global 或分布式共享内存时，只有发起线程可以等待操作完成。这是通过使用 *批量异步组* 完成机制完成的。下表描述了完成机制。

.. list-table:: 使用 TMA 的异步拷贝的可能源和目标内存空间及完成机制。空白单元格表示不支持的源 - 目标对。
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - 方向
     - 异步拷贝 (TMA, CC 9.0+)
     - 源
     - 目标
     - 完成机制
   * - Global → Shared
     - ✓
     - Global
     - shared::cta
     - 共享内存屏障
   * - Global → Shared
     - ✓
     - Global
     - shared::cluster
     - 批量异步组
   * - Global → Cluster Shared
     - ✓
     - Global
     - shared::cluster
     - 共享内存屏障（多播）
   * - Shared → Cluster Shared
     - ✓
     - shared::cta
     - shared::cluster
     - 共享内存屏障
   * - Shared → Shared
     - ✓
     - shared::cta
     - shared::cta
     - -

.. _async-copies-tma-one-dim:

4.11.2.1. 使用 TMA 传输一维数组
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

下表总结了使用批量异步 TMA 的可能源和目标内存空间及完成机制以及暴露它的 API。

.. list-table:: 使用批量异步 TMA 的异步拷贝的可能源和目标内存空间及完成机制。空白单元格表示不支持的源 - 目标对。
   :widths: 15 15 15 15 20 20
   :header-rows: 1

   * - 方向
     - 批量异步拷贝 (TMA, CC9.0+)
     - 源
     - 目标
     - 完成机制
     - API
   * - Global → Shared
     - ✓
     - Global
     - shared::cta
     - 共享内存屏障
     - ``cuda::memcpy_async``、``cuda::device::memcpy_async_tx``、``cuda::ptx::cp_async_bulk``
   * - Global → Shared
     - ✓
     - Global
     - shared::cluster
     - 批量异步组
     - ``cuda::ptx::cp_async_bulk``
   * - Global → Cluster Shared
     - ✓
     - Global
     - shared::cluster
     - 共享内存屏障
     - ``cuda::ptx::cp_async_bulk``
   * - Shared → Cluster Shared
     - ✓
     - shared::cta
     - shared::cluster
     - 共享内存屏障
     - ``cuda::ptx::cp_async_bulk``
   * - Shared → Shared
     - ✓
     - shared::cta
     - shared::cta
     - -
     - -

.. note::

   建议由块中的单个线程发起 TMA 操作。虽然使用 ``if (threadIdx.x == 0)`` 看起来可能足够，但编译器无法验证确实只有一个线程发起拷贝，并可能为所有活动线程插入剥离循环，这会导致 warp 序列化和性能降低。为了防止这种情况，我们定义 ``is_elected()`` 辅助函数，使用 ``cuda::ptx::elect_sync`` 从 warp 0（编译器已知的）选择一个线程执行拷贝，允许编译器生成更高效的代码。或者，可以使用 `cooperative_groups::invoke_one <cooperative-groups.html#cooperative-groups-invoke-one>`_ 实现相同的效果。

批量异步指令对其源和目标地址有特定的对齐要求。更多信息请见下表。

.. list-table:: 一维批量异步操作的对齐要求
   :widths: 40 60
   :header-rows: 1

   * - 地址/大小
     - 对齐要求
   * - Global 内存地址
     - 必须 16 字节对齐
   * - 共享内存地址
     - 必须 16 字节对齐
   * - 共享内存屏障地址
     - 必须 8 字节对齐（由 ``cuda::barrier`` 保证）
   * - 传输大小
     - 必须是 16 字节的倍数

.. _async-copies-tma-one-dim-staging:

4.11.2.1.1. 使用 TMA 预取数据
"""""""""""""""""""""""""""""

在此示例中，我们将演示如何使用 TMA 从 Global 内存预取数据到共享内存。
在迭代拷贝和计算模式中，这允许用当前迭代的计算隐藏未来迭代的数据传输延迟，可能增加飞行中的字节数。

使用 CUDA C++ ``cuda::device::memcpy_async_tx``:

.. code-block:: cuda

   #include <cooperative_groups.h>
   #include <cuda/barrier>
   #include <cuda/ptx>

   namespace ptx = cuda::ptx;
   namespace cg = cooperative_groups;

   __device__ inline bool is_elected()
   {
       unsigned int tid = threadIdx.x;
       unsigned int warp_id = tid / 32;
       unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0);
       return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF));
   }

   template <int block_size, int num_stages>
   __global__ void prefetch_kernel(int* global_out, int const* global_in, size_t size, size_t batch_size) {
       auto grid = cg::this_grid();
       auto block = cg::this_thread_block();
       const int tid = threadIdx.x;
       assert(size == batch_size * grid.size());

       __shared__ int shared[num_stages * block_size];
       size_t shared_offset[num_stages];
       for (int s = 0; s < num_stages; ++s) shared_offset[s] = s * block.size();

       auto block_batch = [&](size_t batch) -> int {
           return block.group_index().x * block.size() + grid.size() * batch;
       };

       #pragma nv_diag_suppress static_var_with_dynamic_init
       __shared__ cuda::barrier<cuda::thread_scope_block> bar[num_stages];
       if (tid == 0) {
           #pragma unroll num_stages
           for (int i = 0; i < num_stages; i++) {
               init(&bar[i], 1);
           }
       }
       __syncthreads();

       if (is_elected()) {
           size_t num_bytes = block_size * sizeof(int);
           #pragma unroll num_stages
           for (int s = 0; s < num_stages; ++s) {
               cuda::device::memcpy_async_tx(&shared[shared_offset[s]], &global_in[block_batch(s)], 
                                             cuda::aligned_size_t<16>(num_bytes), bar[s]);
               (void)cuda::device::barrier_arrive_tx(bar[s], 1, num_bytes);
           }
       }

       int stage = 0;
       uint32_t parity = 0;
       for (size_t compute_batch = 0, fetch_batch = num_stages; compute_batch < batch_size; 
            ++compute_batch, ++fetch_batch) {
           while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, 
                                                 cuda::device::barrier_native_handle(bar[stage]), parity)) {}

           compute(global_out + block_batch(compute_batch) + tid, shared + shared_offset[stage] + tid);
           __syncthreads();

           if (is_elected() && fetch_batch < batch_size) {
               size_t num_bytes = block_size * sizeof(int);
               cuda::device::memcpy_async_tx(&shared[shared_offset[stage]], &global_in[block_batch(fetch_batch)], 
                                             cuda::aligned_size_t<16>(num_bytes), bar[stage]);
               (void)cuda::device::barrier_arrive_tx(bar[stage], 1, num_bytes);
           }

           stage++;
           if (stage == num_stages) {
               stage = 0;
               parity ^= 1;
           }
       }
   }

.. _async-copies-tma-multi-dim:

4.11.2.2. 使用 TMA 传输多维数组
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: 使用批量张量异步 TMA 的异步拷贝的可能源和目标内存空间及完成机制。空白单元格表示不支持的源 - 目标对。
   :widths: 15 15 15 15 20 20
   :header-rows: 1

   * - 方向
     - 批量张量异步拷贝 (TMA, CC9.0+)
     - 源
     - 目标
     - 完成机制
     - API
   * - Global → Shared
     - ✓
     - Global
     - shared::cta
     - 共享内存屏障
     - ``cuda::ptx::cp_async_bulk_tensor``
   * - Global → Cluster Shared
     - ✓
     - Global
     - shared::cluster
     - 共享内存屏障
     - ``cuda::ptx::cp_async_bulk_tensor``
   * - Shared → Cluster Shared
     - ✓
     - shared::cta
     - shared::cluster
     - 共享内存屏障
     - ``cuda::ptx::cp_async_bulk_tensor``
   * - Shared → Global
     - ✓
     - shared::cta
     - Global
     - 批量异步组
     - ``cuda::ptx::cp_async_bulk_tensor``

**Driver API - 创建张量映射**:

.. code-block:: cuda

   #include <cudaTypedefs.h>

   PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
     cudaDriverEntryPointQueryResult driver_status;
     void* cuTensorMapEncodeTiled_ptr = nullptr;
     CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, 
                                                  cudaEnableDefault, &driver_status));
     assert(driver_status == cudaDriverEntryPointSuccess);
     return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
   }

   CUtensorMap tensor_map{};
   constexpr uint32_t rank = 2;
   uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
   uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
   uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
   uint32_t elem_stride[rank] = {1, 1};

   auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
   CUresult res = cuTensorMapEncodeTiled(
     &tensor_map,
     CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
     rank,
     tensor_ptr,
     size,
     stride,
     box_size,
     elem_stride,
     CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
     CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
     CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
     CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
   );

**主机到设备传输 - 三种方式**:

1. ``__grid_constant__`` 参数（推荐）:

.. code-block:: cuda

   __global__ void kernel(const __grid_constant__ CUtensorMap tensor_map) { }

2. ``__constant__`` 内存:

.. code-block:: cuda

   __constant__ CUtensorMap global_tensor_map;
   cudaMemcpyToSymbol(global_tensor_map, &local_tensor_map, sizeof(CUtensorMap));

3. **Global 内存**:

.. code-block:: cuda

   __device__ CUtensorMap global_tensor_map;
   ptx::fence_proxy_tensormap_generic(ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes);

**多维 TMA 使用示例**:

.. code-block:: cuda

   #include <cuda.h>
   #include <cuda/barrier>

   using barrier = cuda::barrier<cuda::thread_scope_block>;
   namespace ptx = cuda::ptx;

   __global__ void kernel(const __grid_constant__ CUtensorMap tensor_map, int x, int y) {
     __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

     #pragma nv_diag_suppress static_var_with_dynamic_init
     __shared__ barrier bar;

     if (threadIdx.x == 0) {
       init(&bar, blockDim.x);
     }
     __syncthreads();

     barrier::arrival_token token;
     if (is_elected()) {
       int32_t tensor_coords[2] = { x, y };
       ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, &smem_buffer, &tensor_map, 
                                 tensor_coords, cuda::device::barrier_native_handle(bar));
       token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
     } else {
       token = bar.arrive();
     }
     bar.wait(std::move(token));

     smem_buffer[0][threadIdx.x] += threadIdx.x;

     ptx::fence_proxy_async(ptx::space_shared);
     __syncthreads();

     if (is_elected()) {
       int32_t tensor_coords[2] = { x, y };
       ptx::cp_async_bulk_tensor(ptx::space_global, ptx::space_shared, &tensor_map, tensor_coords, &smem_buffer);
       ptx::cp_async_bulk_commit_group();
       ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
     }

     if (threadIdx.x == 0) {
       (&bar)->~barrier();
     }
   }

**负索引和越界**: 当从 Global 内存读取的部分超出边界时，对应的共享内存区域会被零填充。写入时，左上角索引不能为负。

.. list-table:: 多维批量张量异步拷贝操作的对齐要求
   :widths: 40 60
   :header-rows: 1

   * - 地址/大小
     - 对齐要求
   * - Global 内存地址
     - 必须 16 字节对齐
   * - Global 内存大小
     - 必须大于或等于一
   * - Global 内存跨步
     - 必须是 16 字节的倍数
   * - 共享内存地址
     - 必须 128 字节对齐
   * - 共享内存屏障地址
     - 必须 8 字节对齐
   * - 传输大小
     - 必须是 16 字节的倍数

.. _async-copies-tma-swizzling:

4.11.2.2.5. 共享内存 Bank 交错
""""""""""""""""""""""""""""""""

TMA 引擎可以按"swizzle 模式"对数据进行洗牌，以减少共享内存银行冲突。这通过在加载或存储数据时修改共享内存地址来实现。

.. note::

   交错仅在计算能力 9 及更高版本上受支持。交错功能仅适用于使用 ``cuda::ptx::cp_async_bulk_tensor`` 指令的多维数组。

**示例：矩阵转置**

.. code-block:: cuda

   __global__ void kernel_tma(const __grid_constant__ CUtensorMap tensor_map) {
      __shared__ alignas(1024) int4 smem_buffer[8][8];
      __shared__ alignas(1024) int4 smem_buffer_tr[8][8];

      #pragma nv_diag_suppress static_var_with_dynamic_init
      __shared__ barrier bar;

      if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
      }
      __syncthreads();

      barrier::arrival_token token;
      if (is_elected()) {
        int32_t tensor_coords[2] = { 0, 0 };
        ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, &smem_buffer, &tensor_map, 
                                  tensor_coords, cuda::device::barrier_native_handle(bar));
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
      } else {
        token = bar.arrive();
      }
      bar.wait(std::move(token));

      for(int sidx_j = threadIdx.x; sidx_j < 8; sidx_j += blockDim.x) {
         for(int sidx_i = 0; sidx_i < 8; ++sidx_i) {
            const int swiz_j_idx = (sidx_i % 8) ^ sidx_j;
            const int swiz_i_idx_tr = (sidx_j % 8) ^ sidx_i;
            smem_buffer_tr[sidx_j][swiz_i_idx_tr] = smem_buffer[sidx_i][swiz_j_idx];
         }
      }

      ptx::fence_proxy_async(ptx::space_shared);
      __syncthreads();

      if (is_elected()) {
          int32_t tensor_coords[2] = { x, y };
          ptx::cp_async_bulk_tensor(ptx::space_global, ptx::space_shared, &tensor_map, tensor_coords, &smem_buffer_tr);
         ptx::cp_async_bulk_commit_group();
         ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
      }

      if (threadIdx.x == 0) {
        (&bar)->~barrier();
      }
   }

.. list-table:: 计算能力 9 的不同交错模式的要求和属性
   :widths: 20 15 20 15 15 15
   :header-rows: 1

   * - 模式
     - 交错宽度
     - 共享框的内部维度
     - 重复周期
     - 共享内存对齐
     - Global 内存对齐
   * - CU_TENSOR_MAP_SWIZZLE_128B
     - 128 字节
     - <=128 字节
     - 1024 字节
     - 128 字节
     - 128 字节
   * - CU_TENSOR_MAP_SWIZZLE_64B
     - 64 字节
     - <=64 字节
     - 512 字节
     - 128 字节
     - 128 字节
   * - CU_TENSOR_MAP_SWIZZLE_32B
     - 32 字节
     - <=32 字节
     - 256 字节
     - 128 字节
     - 128 字节
   * - CU_TENSOR_MAP_SWIZZLE_NONE
     - 128 字节
     - 16 字节
     - -
     - -
     - -

.. list-table:: 交错模式指针偏移公式和索引关系
   :widths: 25 45 30
   :header-rows: 1

   * - 交错模式
     - 偏移公式
     - 索引关系
   * - CU_TENSOR_MAP_SWIZZLE_128B
     - ``(reinterpret_cast<uintptr_t>(smem_ptr)/128)%8``
     - ``smem[y][x] <-> smem[y][((y+offset)%8)^x]``
   * - CU_TENSOR_MAP_SWIZZLE_64B
     - ``(reinterpret_cast<uintptr_t>(smem_ptr)/128)%4``
     - ``smem[y][x] <-> smem[y][((y+offset)%4)^x]``
   * - CU_TENSOR_MAP_SWIZZLE_32B
     - ``(reinterpret_cast<uintptr_t>(smem_ptr)/128)%2``
     - ``smem[y][x] <-> smem[y][((y+offset)%2)^x]``

.. _using-stas:

4.11.3. 使用 STAS
-----------------

使用线程块集群的 CUDA 应用程序可能需要在集群内的线程块之间移动小数据元素。STAS 指令（CC 9.0+）支持从寄存器直接异步数据拷贝到分布式共享内存。STAS 仅通过较低级别的 ``cuda::ptx::st_async`` API 公开。

**维度**。STAS 支持复制 4、8 或 16 字节。

**源和目标**。STAS 异步拷贝操作支持的唯一方向是从寄存器到分布式共享内存。目标指针需要根据复制的数据大小对齐到 4、8 或 16 字节。

**异步性**。使用 STAS 的数据传输是异步的，并建模为异步线程操作。STAS 操作可用于发出完成信号的完成机制是共享内存屏障。

**STAS 生产者 - 消费者示例（集群环）**:

.. code-block:: cuda

   #include <cooperative_groups.h>
   #include <cuda/barrier>
   #include <cuda/ptx>

   __global__ __cluster_dims__(8, 1, 1) void producer_consumer_kernel() 
   {
       using namespace cooperative_groups;
       using namespace cuda::device;
       using namespace cuda::ptx;
       using barrier_t = cuda::barrier<cuda::thread_scope_block>;

       auto cluster = this_cluster();

       #pragma nv_diag_suppress static_var_with_dynamic_init
       __shared__ int buffer[BLOCK_SIZE];
       __shared__ barrier_t filled;
       __shared__ barrier_t ready;
       
       if (threadIdx.x == 0) {
           init(&filled, 1);
           init(&ready, BLOCK_SIZE);
       }
       
       cluster.sync();
       
       int rk = cluster.block_rank();
       int rk_next = (rk + 1) % 8;
       int rk_prev = (rk + 7) % 8;
         
       auto buffer_next = cluster.map_shared_rank(buffer, rk_next);
       auto bar_next = cluster.map_shared_rank(barrier_native_handle(filled), rk_next);
       auto bar_prev = cluster.map_shared_rank(barrier_native_handle(ready), rk_prev);
       
       int phase = 0;
       for (int it = 0; it < 1000; ++it) {
           st_async(&buffer_next[threadIdx.x], rk, bar_next);
           
           if (threadIdx.x == 0) {
               mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared, 
                                         barrier_native_handle(filled), sizeof(buffer));
           }

           while (!mbarrier_try_wait_parity(barrier_native_handle(filled), phase, 1000)) {}
           
           int r = buffer[threadIdx.x];
           
           mbarrier_arrive(sem_release, scope_cluster, space_cluster, bar_prev);
           
           while (!mbarrier_try_wait_parity(barrier_native_handle(ready), phase, 1000)) {}
           phase ^= 1;
       }
   }

.. note::

   有关异步数据拷贝的更多详细信息，请参考 `CUDA 官方文档 <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html>`_。
