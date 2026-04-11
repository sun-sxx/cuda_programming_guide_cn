.. _async-copies-details:

4.11. 异步数据拷贝
==================

基于 :doc:`../03-advanced/02-advanced-kernel-programming` 的第 3.2.5 节，本节为 GPU 内存层次结构内的异步数据移动提供详细指导和示例。它涵盖了用于元素级拷贝的 LDGSTS、用于批量（一维和多维）传输的张量内存加速器 (TMA) 以及用于寄存器到分布式共享内存拷贝的 STAS，并展示了这些机制如何与 :doc:`09-async-barriers` 和 :doc:`10-pipelines` 集成。

.. _using-ldgsts:

4.11.1. 使用 LDGSTS
-------------------

许多 CUDA 应用程序需要在 Global 内存和共享内存之间频繁移动数据。通常，这涉及复制较小的数据元素或执行不规则的内存访问模式。LDGSTS（CC 8.0+，参见 `PTX 文档 <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy>`_）的主要目标是为较小的元素级数据传输提供从 Global 内存到共享内存的有效异步数据传输机制，同时通过重叠执行更好地利用计算资源。

**维度**。LDGSTS 支持复制 4、8 或 16 字节。复制 4 或 8 字节始终以所谓的 L1 ACCESS 模式进行，此时数据也缓存在 L1 中，而复制 16 字节启用 L1 BYPASS 模式，此时 L1 不会被污染。

**源和目标**。LDGSTS 异步拷贝操作支持的唯一方向是从 Global 内存到共享内存。指针需要根据复制的数据大小对齐到 4、8 或 16 字节。当共享内存和 Global 内存的对齐都是 128 字节时，可获得最佳性能。

**异步性**。使用 LDGSTS 的数据传输是 `异步的 <../03-advanced/02-advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features>`_，并建模为异步线程操作（参见 `异步线程和异步代理 <../03-advanced/02-advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features-async-thread-proxy>`_）。这允许发起线程继续计算，而硬件异步复制数据。*数据传输是否实际异步执行取决于硬件实现，未来可能会发生变化*。

LDGSTS 必须在操作完成时提供信号。LDGSTS 可以使用 `共享内存屏障 <../03-advanced/02-advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers>`_ 或 `管道 <../03-advanced/02-advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-pipelines>`_ 作为提供完成信号的机制。默认情况下，每个线程只等待自己的 LDGSTS 拷贝。因此，如果您使用 LDGSTS 预取一些将与其他线程共享的数据，则在同步 LDGSTS 完成机制后需要 ``__syncthreads()``。

.. list-table:: 使用 LDGSTS 的异步拷贝的可能源和目标内存空间及完成机制。空白单元格表示不支持的源-目标对。
   :widths: 25 25 25 25
   :header-rows: 1

   * - 方向
     - 异步拷贝 (LDGSTS, CC 8.0+)
     - 源
     - 目标
     - 完成机制
   * - Global → Shared
     - ✓
     - Global
     - shared::cta
     - 共享内存屏障、管道
   * - Global → Cluster Shared
     - ✓
     - Global
     - shared::cluster
     - 共享内存屏障

.. _async-copies-batching-loads:

4.11.1.1. 条件代码中的批量加载
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
           cuda::memcpy_async(buffer + tid, left + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier);
       } else if (tid >= 32 - 8) {
           cuda::memcpy_async(buffer + tid + 16, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier);
       }
       if (tid < 32) {
           cuda::memcpy_async(buffer + 40, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier);
       }
       
       // 版本 2：跨所有线程集体发出拷贝
       cuda::memcpy_async(block, buffer, left, cuda::aligned_size_t<4>(8 * sizeof(float)), barrier);
       cuda::memcpy_async(block, buffer + 8, center, cuda::aligned_size_t<4>(32 * sizeof(float)), barrier);
       cuda::memcpy_async(block, buffer + 40, right, cuda::aligned_size_t<4>(8 * sizeof(float)), barrier);
       
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
       cg::memcpy_async(block, buffer, left, 8 * sizeof(float));
       cg::memcpy_async(block, buffer + 8, center, 32 * sizeof(float));
       cg::memcpy_async(block, buffer + 40, right, 8 * sizeof(float));
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
           __pipeline_memcpy_async(buffer + tid, left + tid, sizeof(float));
       } else if (tid >= 32 - 8) {
           __pipeline_memcpy_async(buffer + tid + 16, right + tid, sizeof(float));
       }
       if (tid < 32) {
           __pipeline_memcpy_async(buffer + tid + 8, center + tid, sizeof(float));
       }
       __pipeline_commit();
       __pipeline_wait_prior(0);
       __syncthreads();

       // 计算模板
   }

.. _async-copies-prefetching:

4.11.1.2. 预取数据
^^^^^^^^^^^^^^^^^^

在此示例中，我们将演示如何使用异步数据拷贝从 Global 内存预取数据到共享内存。在迭代拷贝和计算模式中，这允许用当前迭代的计算隐藏未来迭代的数据传输延迟，可能增加飞行中的字节数。

.. code-block:: cuda

   #include <cooperative_groups.h>
   #include <cuda/pipeline>

   template <size_t num_stages = 2 /* 具有 num_stages 阶段的管道 */>
   __global__ void prefetch_kernel(int* global_out, int const* global_in, size_t size, size_t batch_size) {
       auto grid = cooperative_groups::this_grid();
       auto block = cooperative_groups::this_thread_block();
       auto thread = cooperative_groups::this_thread();
       assert(size == batch_size * grid.size());

       extern __shared__ int shared[];
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

       for (size_t compute_batch = 0, fetch_batch = num_stages; compute_batch < batch_size; 
            ++compute_batch, ++fetch_batch) {
           // 等待第一个请求的阶段完成
           constexpr size_t pending_batches = num_stages - 1;
           cuda::pipeline_consumer_wait_prior<pending_batches>(pipeline);
           __syncthreads();

           // 在当前批次上计算
           compute(global_out + block_batch(compute_batch) + tid, shared + shared_offset[stage] + tid);
           
           // 释放当前阶段
           pipeline.consumer_release();
           __syncthreads();

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

.. _async-copies-producer-consumer:

4.11.1.3. 通过 Warp 特化的生产者 - 消费者模式
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在此示例中，我们将演示如何实现生产者 - 消费者模式，其中单个 warp 专门化作为生产者，执行从 Global 到共享内存的异步数据拷贝，而剩余的 warp 从共享内存消费数据并执行计算。为了启用生产者和消费者线程之间的并发性，我们在共享内存中使用双缓冲。当消费者 warp 处理一个缓冲区中的数据时，生产者 warp 异步获取下一批数据到另一个缓冲区。

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
       cuda::memcpy_async(buffer + stage * buffer_len + threadIdx.x, in + batch * buffer_len + threadIdx.x, 
                          cuda::aligned_size_t<4>(sizeof(float)), pipe);
       pipe.producer_commit();
     }
   }

   __device__ void consume(pipeline &pipe, int num_stages, int stage, int num_batches, int batch, 
                          float *buffer, int buffer_len, float *out, int N)
   {
     pipe.consumer_wait();
     pipe.consumer_release();
   }

   __global__ void producer_consumer_pattern(float *in, float *out, int N, int buffer_len)
   {
     auto block = cooperative_groups::this_thread_block();
     constexpr int warpSize = 32;

     __shared__ extern float buffer[];

     const int num_batches = N / buffer_len;

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
         produce(pipe, num_stages, stage, num_batches, b + num_stages, buffer, buffer_len, in, N);
       }
       else
       {
         consume(pipe, num_stages, stage, num_batches, b, buffer, buffer_len, out, N);
       }
       stage = (stage + 1) % num_stages;
     }
   }

.. _using-tma:

4.11.2. 使用张量内存加速器 (TMA)
--------------------------------

许多应用程序需要将大量数据传输到 Global 内存和从 Global 内存传输。通常，数据在 Global 内存中布局为具有非顺序数据访问模式的多维数组。为了减少 Global 内存访问，在用于计算之前，此类数组的子图块被复制到共享内存。加载和存储涉及容易出错且重复的地址计算。为了卸载这些计算，计算能力 9.0（Hopper）及更高版本（参见 `PTX 文档 <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk>`_）具有 *张量内存加速器* (TMA)。TMA 的主要目标是为多维数组提供从 Global 内存到共享内存的有效数据传输机制。

**命名**。张量内存加速器 (TMA) 是用于指代本节中描述的功能的广泛术语。为了前向兼容性和减少与 PTX ISA 的差异，本节中的文本将 TMA 操作称为 *批量异步拷贝* 或 *批量张量异步拷贝*，具体取决于使用的拷贝类型。术语"批量"用于将这些操作与上一节中描述的异步内存操作进行对比。

**维度**。TMA 支持复制一维和多维数组（最多 5 维）。一维连续数组的批量异步拷贝编程模型与多维数组的批量张量异步拷贝编程模型不同。要执行多维数组的批量张量异步拷贝，硬件需要 `张量映射 <https://docs.nvidia.com/cuda/cuda-driver-api/structCUtensorMap.html#structCUtensorMap>`_。此对象描述多维数组在 Global 和共享内存中的布局。

**源和目标**。TMA 操作的源和目标地址可以在共享或 Global 内存中。操作可以从 Global 读取到共享内存，从共享写入到 Global 内存，也可以从共享内存复制到同一集群中另一个块的 `分布式共享内存 <../02-basics/02-writing-cuda-kernels.html#writing-cuda-kernels-distributed-shared-memory>`_。此外，在集群中时，批量异步张量操作可以指定为 *多播*。

**异步性**。使用 TMA 的数据传输是 `异步的 <../03-advanced/02-advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features>`_，并建模为异步代理操作。

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
   * - Shared → Global
     - ✓
     - shared::cta
     - Global
     - 批量异步组
   * - Shared → Cluster Shared
     - ✓
     - shared::cta
     - shared::cluster
     - 共享内存屏障


.. _async-copies-tma-one-dim:

4.11.2.1. 使用 TMA 传输一维数组
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

以下示例演示了如何使用批量异步拷贝。该示例对一维数组进行读 - 修改 - 写操作。

.. code-block:: cuda

   #include <cuda/barrier>
   #include <cuda/ptx>

   using barrier = cuda::barrier<cuda::thread_scope_block>;
   namespace ptx = cuda::ptx;

   static constexpr size_t buf_len = 1024;

   __device__ inline bool is_elected()
   {
       unsigned int tid = threadIdx.x;
       unsigned int warp_id = tid / 32;
       unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0);
       return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF));
   }

   __global__ void add_one_kernel(int* data, size_t offset)
   {
     __shared__ alignas(16) int smem_data[buf_len];

     #pragma nv_diag_suppress static_var_with_dynamic_init
     __shared__ barrier bar;
     if (threadIdx.x == 0) {
       init(&bar, blockDim.x);
     }
     __syncthreads();

     // 从单个线程发起 TMA 传输，从 Global 拷贝到共享内存
     if (is_elected()) {
       cuda::memcpy_async(
           smem_data, data + offset, 
           cuda::aligned_size_t<16>(sizeof(smem_data)),
           bar);
     }
     
     // 所有线程到达屏障
     barrier::arrival_token token = bar.arrive();
     
     // 等待数据到达
     bar.wait(std::move(token));

     // 计算 saxpy 并写回共享内存
     for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
       smem_data[i] += 1;
     }

     // 等待共享内存写入对 TMA 引擎可见
     ptx::fence_proxy_async(ptx::space_shared);
     __syncthreads();

     // 发起 TMA 传输，从共享内存拷贝到 Global 内存
     if (is_elected()) {
       ptx::cp_async_bulk(
           ptx::space_global, ptx::space_shared,
           data + offset, smem_data, sizeof(smem_data));
       ptx::cp_async_bulk_commit_group();
       ptx::cp_async_bulk_wait_group_read<ptx::n32_t<0>>();
     }
   }

**屏障初始化**。屏障用参与块的线程数初始化。因此，只有当所有线程都到达此屏障时，屏障才会翻转。

**TMA 读取**。批量异步拷贝指令指示硬件将大量数据复制到共享内存，并在完成读取后更新共享内存屏障的 `事务计数 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-tracking-async-operations>`_。通常，发出尽可能少的具有尽可能大批量的拷贝会产生最佳性能。

**屏障等待**。使用令牌通过 ``bar.wait()`` 等待屏障翻转。使用屏障的显式相位跟踪可能更高效（参见 `显式相位跟踪 <09-async-barriers.html#asynchronous-barriers-explicit-phase>`_）。

**SMEM 写入和同步**。缓冲区值的增量读取和写入共享内存。为了使写入对后续批量异步拷贝可见，使用 ``cuda::ptx::fence_proxy_async`` 函数。

**TMA 写入和同步**。从共享到 Global 内存的写入再次由单个线程发起。写入的完成不由共享内存屏障跟踪。而是使用线程局部机制。

.. note::

   建议由块中的单个线程发起 TMA 操作。虽然使用 ``if (threadIdx.x == 0)`` 看起来可能足够，但编译器无法验证确实只有一个线程发起拷贝，并可能为所有活动线程插入剥离循环，这会导致 warp 序列化和性能降低。

**对齐要求**：

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

.. _async-copies-tma-prefetching:

4.11.2.2. 使用 TMA 预取数据
^^^^^^^^^^^^^^^^^^^^^^^^^^^

在此示例中，我们将演示如何使用 TMA 从 Global 内存预取数据到共享内存。

.. code-block:: cuda

   #include <cuda/barrier>
   #include <cuda/ptx>

   template<int num_stages = 2>
   __global__ void tma_prefetch_kernel(float* output, const float* input, size_t N) {
       extern __shared__ float smem[];
       
       constexpr size_t stage_size = (N + blockDim.x - 1) / blockDim.x;
       __shared__ cuda::barrier<cuda::thread_scope_block> bar;
       
       if (threadIdx.x == 0) {
           init(&bar, blockDim.x);
       }
       __syncthreads();
       
       size_t offset = threadIdx.x * stage_size;
       
       // 填充管道
       for (int stage = 0; stage < num_stages; ++stage) {
           size_t stage_offset = offset + stage * stage_size * blockDim.x;
           if (stage_offset < N && is_elected()) {
               size_t copy_size = min(stage_size * sizeof(float), (N - stage_offset) * sizeof(float));
               cuda::memcpy_async(
                   smem + stage * stage_size * blockDim.x + threadIdx.x,
                   input + stage_offset,
                   cuda::aligned_size_t<16>(copy_size),
                   bar);
           }
       }
       
       for (int stage = 0; stage < num_stages; ++stage) {
           bar.arrive_and_wait();
           
           // 计算
           for (size_t i = 0; i < stage_size && (offset + i) < N; ++i) {
               output[offset + i] = smem[stage * stage_size * blockDim.x + threadIdx.x] * 2.0f;
           }
           
           __syncthreads();
           
           // 预取下一阶段
           int next_stage = (stage + num_stages) % num_stages;
           size_t next_offset = offset + next_stage * stage_size * blockDim.x;
           if (next_offset < N && is_elected()) {
               size_t copy_size = min(stage_size * sizeof(float), (N - next_offset) * sizeof(float));
               cuda::memcpy_async(
                   smem + next_stage * stage_size * blockDim.x + threadIdx.x,
                   input + next_offset,
                   cuda::aligned_size_t<16>(copy_size),
                   bar);
           }
       }
   }

.. note::

   有关异步数据拷贝的更多详细信息，请参考 `CUDA 官方文档 <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html>`_。
