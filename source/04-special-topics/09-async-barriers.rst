.. _async-barriers-details:

4.9. 异步屏障
=============

异步屏障在 :doc:`../../03-advanced/02-advanced-kernel-programming` 的「高级同步原语」一节中介绍，它扩展了 CUDA 同步功能，超越了 `__syncthreads()` 和 `__syncwarp()`，实现了细粒度、非阻塞协调以及更好的通信与计算重叠。

本节详细介绍如何主要通过 `cuda::barrier` API 使用异步屏障（在适用时也会指向 `cuda::ptx` 和原语 API）。

.. _async-barriers-initialization:

4.9.1. 初始化
-------------

初始化必须在线程开始参与屏障之前完成。

.. tab-set::

   .. tab-item:: CUDA C++ `cuda::barrier`

      .. code-block:: cpp

         #include <cuda/barrier>
         #include <cooperative_groups.h>

         __global__ void init_barrier()
         {
           __shared__ cuda::barrier<cuda::thread_scope_block> bar;
           auto block = cooperative_groups::this_thread_block();

           if (block.thread_rank() == 0)
           {
             // A single thread initializes the total expected arrival count.
             init(&bar, block.size());
           }
           block.sync();
         }

   .. tab-item:: CUDA C++ `cuda::ptx`

      .. code-block:: cpp

         #include <cuda/ptx>
         #include <cooperative_groups.h>

         __global__ void init_barrier()
         {
           __shared__ uint64_t bar;
           auto block = cooperative_groups::this_thread_block();

           if (block.thread_rank() == 0)
           {
             // A single thread initializes the total expected arrival count.
             cuda::ptx::mbarrier_init(&bar, block.size());
           }
           block.sync();
         }

   .. tab-item:: CUDA C 原语

      .. code-block:: cpp

         #include <cuda_awbarrier_primitives.h>
         #include <cooperative_groups.h>

         __global__ void init_barrier()
         {
           __shared__ uint64_t bar;
           auto block = cooperative_groups::this_thread_block();

           if (block.thread_rank() == 0)
           {
             // A single thread initializes the total expected arrival count.
             __mbarrier_init(&bar, block.size());
           }
           block.sync();
         }

在任何线程参与屏障之前，必须使用 `cuda::barrier::init()` 友元函数初始化屏障。这必须发生在线程到达屏障之前。这带来了一个引导挑战：线程必须在参与屏障之前同步，但线程创建屏障正是为了同步。在此示例中，将参与屏障的线程属于一个协作组，并使用 `block.sync()` 来引导初始化。由于整个线程块都参与屏障，也可以使用 `__syncthreads()`。

`init()` 的第二个参数是 *预期到达计数* （expected arrival count），即在参与线程被其 `bar.wait(std::move(token))` 调用解除阻塞之前，参与线程调用 `bar.arrive()` 的次数。在此示例及前面的示例中，屏障使用线程块中的线程数进行初始化，即 `cooperative_groups::this_thread_block().size()` ，以便线程块内的所有线程都能参与屏障。

异步屏障可以灵活地指定线程 *如何* 参与（分离 arrive/wait）以及 *哪些* 线程参与。相比之下， `this_thread_block.sync()` 或 `__syncthreads()` 适用于整个线程块，而 `__syncwarp(mask)` 适用于 warp 的指定子集。尽管如此，如果用户的意图是同步完整的线程块或完整的 warp，我们建议分别使用 `__syncthreads()` 和 `__syncwarp()` 以获得更好的性能。

.. _async-barriers-phase:

4.9.2. 屏障的阶段：到达、倒计时、完成和重置
-------------------------------------------

异步屏障在参与线程调用 `bar.arrive()` 时从预期到达计数倒数到零。当倒计时达到零时，屏障在当前阶段完成。当最后一次 `bar.arrive()` 调用导致倒计时达到零时，倒计时会自动且原子地重置。重置将倒计时设置为预期到达计数，并将屏障移至下一阶段。

从 `token=bar.arrive()` 返回的 `cuda::barrier::arrival_token` 类的 `token` 对象与屏障的当前阶段相关联。调用 `bar.wait(std::move(token))` 会在屏障处于当前阶段时阻塞调用线程，即当与 token 关联的阶段与屏障的阶段匹配时。如果在调用 `bar.wait(std::move(token))` 之前阶段已推进（因为倒计时达到零），则线程不会阻塞；如果在线程阻塞于 `bar.wait(std::move(token))` 期间阶段推进，则线程被解除阻塞。

**了解重置何时可能发生或不发生至关重要，尤其是在非平凡的 arrive/wait 同步模式中。**

- 线程对 `token=bar.arrive()` 和 `bar.wait(std::move(token))` 的调用必须按顺序进行，使 `token=bar.arrive()` 在屏障的当前阶段发生，而 `bar.wait(std::move(token))` 在同一阶段或下一阶段发生。

- 线程对 `bar.arrive()` 的调用必须在屏障计数器非零时发生。屏障初始化后，如果线程对 `bar.arrive()` 的调用导致倒计时达到零，则在屏障可以重新用于后续的 `bar.arrive()` 调用之前，必须先调用 `bar.wait(std::move(token))`。

- `bar.wait()` 只能使用当前阶段或紧邻的前一阶段的 `token` 对象调用。对于 `token` 对象的任何其他值，行为未定义。

对于简单的 arrive/wait 同步模式，遵守这些使用规则很简单。

.. _async-barriers-warp-entanglement:

4.9.2.1. Warp 纠缠
~~~~~~~~~~~~~~~~~~

Warp 分歧会影响 arrive 操作更新屏障的次数。如果调用的 warp 完全收敛，则屏障更新一次。如果调用的 warp 完全发散，则对屏障应用 32 次单独更新。

.. note::

   建议由收敛的线程使用 `arrive-on(bar)` 调用，以尽量减少对屏障对象的更新。当这些操作之前的代码使线程分歧时，应在调用 arrive-on 操作之前通过 `__syncwarp` 重新收敛 warp。

.. _async-barriers-explicit-phase-tracking:

4.9.3. 显式阶段跟踪
-------------------

异步屏障根据用于同步线程和内存操作的次数可以有多个阶段。我们可以通过 `cuda::ptx` 和原语 API 提供的 `mbarrier_try_wait_parity()` 系列函数直接跟踪阶段，而不是使用 token 来跟踪屏障阶段翻转。

在最简单的形式中，`cuda::ptx::mbarrier_try_wait_parity(uint64_t* bar, const uint32_t& phaseParity)` 函数等待具有特定奇偶性的阶段。`phaseParity` 操作数是屏障对象当前阶段或紧邻前一阶段的整数奇偶性。偶数阶段的整数奇偶性为 0，奇数阶段的整数奇偶性为 1。初始化屏障时，其阶段的奇偶性为 0。因此 `phaseParity` 的有效值为 0 和 1。显式阶段跟踪在跟踪 :ref:`异步内存操作 <sec:async-copies>` 时很有用，因为它只允许单个线程到达屏障并设置事务计数，而其他线程只等待基于奇偶性的阶段翻转。这比让所有线程都到达屏障并使用 token 更高效。此功能仅适用于线程块和集群作用域的共享内存屏障。

.. tab-set::

   .. tab-item:: CUDA C++ `cuda::barrier`

      .. code-block:: cpp

         #include <cuda/ptx>
         #include <cooperative_groups.h>

         __device__ void compute(float *data, int iteration);

         __global__ void split_arrive_wait(int iteration_count, float *data)
         {
           using barrier_t = cuda::barrier<cuda::thread_scope_block>;
           __shared__ barrier_t bar;
           int parity = 0; // Initial phase parity is 0.
           auto block = cooperative_groups::this_thread_block();

           if (block.thread_rank() == 0)
           {
             // Initialize barrier with expected arrival count.
             init(&bar, block.size());
           }
           block.sync();

           for (int i = 0; i < iteration_count; ++i)
           {
             /* code before arrive */

             // This thread arrives. Arrival does not block a thread.
             // Get a handle to the native barrier to use with cuda::ptx API.
             (void)cuda::ptx::mbarrier_arrive(cuda::device::barrier_native_handle(bar));

             compute(data, i);

             // Wait for all threads participating in the barrier to complete mbarrier_arrive().
             // Get a handle to the native barrier to use with cuda::ptx API.
             while (!cuda::ptx::mbarrier_try_wait_parity(cuda::device::barrier_native_handle(bar), parity)) {}
             // Flip parity.
             parity ^= 1;

             /* code after wait */
           }
         }

   .. tab-item:: CUDA C++ `cuda::ptx`

      .. code-block:: cpp

         #include <cuda/ptx>
         #include <cooperative_groups.h>

         __device__ void compute(float *data, int iteration);

         __global__ void split_arrive_wait(int iteration_count, float *data)
         {
           __shared__ uint64_t bar;
           int parity = 0; // Initial phase parity is 0.
           auto block = cooperative_groups::this_thread_block();

           if (block.thread_rank() == 0)
           {
             // Initialize barrier with expected arrival count.
             cuda::ptx::mbarrier_init(&bar, block.size());
           }
           block.sync();

           for (int i = 0; i < iteration_count; ++i)
           {
             /* code before arrive */

             // This thread arrives. Arrival does not block a thread.
             (void)cuda::ptx::mbarrier_arrive(&bar);

             compute(data, i);

             // Wait for all threads participating in the barrier to complete mbarrier_arrive().
             while (!cuda::ptx::mbarrier_try_wait_parity(&bar, parity)) {}
             // Flip parity.
             parity ^= 1;

             /* code after wait */
           }
         }

   .. tab-item:: CUDA C 原语

      .. code-block:: cpp

         #include <cuda_awbarrier_primitives.h>
         #include <cooperative_groups.h>

         __device__ void compute(float *data, int iteration);

         __global__ void split_arrive_wait(int iteration_count, float *data)
         {
           __shared__ __mbarrier_t bar;
           bool parity = false; // Initial phase parity is false.
           auto block = cooperative_groups::this_thread_block();

           if (block.thread_rank() == 0)
           {
             // Initialize barrier with expected arrival count.
             __mbarrier_init(&bar, block.size());
           }
           block.sync();

           for (int i = 0; i < iteration_count; ++i)
           {
             /* code before arrive */

             // This thread arrives. Arrival does not block a thread.
             (void)__mbarrier_arrive(&bar);

             compute(data, i);

             // Wait for all threads participating in the barrier to complete __mbarrier_arrive().
             while(!__mbarrier_try_wait_parity(&bar, parity, 1000)) {}
             parity ^= 1;

             /* code after wait */
           }
         }

.. _async-barriers-early-exit:

4.9.4. 提前退出
---------------

当参与一系列同步的线程必须提前退出该序列时，该线程必须在退出前显式退出参与。剩余的参与线程可以正常进行后续的 arrive 和 wait 操作。

.. tab-set::

   .. tab-item:: CUDA C++ `cuda::barrier`

      .. code-block:: cpp

         #include <cuda/barrier>
         #include <cooperative_groups.h>

         __device__ bool condition_check();

         __global__ void early_exit_kernel(int N)
         {
           __shared__ cuda::barrier<cuda::thread_scope_block> bar;
           auto block = cooperative_groups::this_thread_block();

           if (block.thread_rank() == 0)
           {
             init(&bar, block.size());
           }
           block.sync();

           for (int i = 0; i < N; ++i)
           {
             if (condition_check())
             {
               bar.arrive_and_drop();
               return;
             }
             // Other threads can proceed normally.
             auto token = bar.arrive();

             /* code between arrive and wait */

             // Wait for all threads to arrive.
             bar.wait(std::move(token));

             /* code after wait */
           }
         }

   .. tab-item:: CUDA C 原语

      .. code-block:: cpp

         #include <cuda_awbarrier_primitives.h>
         #include <cooperative_groups.h>

         __device__ bool condition_check();

         __global__ void early_exit_kernel(int N)
         {
           __shared__ __mbarrier_t bar;
           auto block = cooperative_groups::this_thread_block();

           if (block.thread_rank() == 0)
           {
             __mbarrier_init(&bar, block.size());
           }
           block.sync();

           for (int i = 0; i < N; ++i)
           {
             if (condition_check())
             {
               __mbarrier_token_t token = __mbarrier_arrive_and_drop(&bar);
               return;
             }
             // Other threads can proceed normally.
             __mbarrier_token_t token = __mbarrier_arrive(&bar);

             /* code between arrive and wait */

             // Wait for all threads to arrive.
             while (!__mbarrier_try_wait(&bar, token, 1000)) {}

             /* code after wait */
           }
         }

`bar.arrive_and_drop()` 操作到达屏障以履行参与线程在 **当前** 阶段到达的义务，然后递减 **下一** 阶段的预期到达计数，使该线程不再被期望到达屏障。

.. _async-barriers-completion-function:

4.9.5. 完成函数
---------------

`cuda::barrier` API 支持可选的完成函数。`cuda::barrier<Scope, CompletionFunction>` 的 `CompletionFunction` 在每个阶段执行一次，在最后一个线程 *到达* 之后、任何线程从 `wait` 解除阻塞之前。在阶段期间到达 `barrier` 的线程执行的内存操作对执行 `CompletionFunction` 的线程可见，并且 `CompletionFunction` 内执行的所有内存操作在从 `wait` 解除阻塞后对所有在 `barrier` 等待的线程可见。

.. tab-set::

   .. tab-item:: CUDA C++ `cuda::barrier`

      .. code-block:: cpp

         #include <cuda/barrier>
         #include <cooperative_groups.h>
         #include <functional>
         namespace cg = cooperative_groups;

         __device__ int divergent_compute(int *, int);
         __device__ int independent_computation(int *, int);

         __global__ void psum(int *data, int n, int *acc)
         {
           auto block = cg::this_thread_block();

           constexpr int BlockSize = 128;
           __shared__ int smem[BlockSize];
           assert(BlockSize == block.size());
           assert(n % BlockSize == 0);

           auto completion_fn = [&]
           {
             int sum = 0;
             for (int i = 0; i < BlockSize; ++i)
             {
               sum += smem[i];
             }
             *acc += sum;
           };

           /* Barrier storage.
              Note: the barrier is not default-constructible because
                    completion_fn is not default-constructible due
                    to the capture. */
           using completion_fn_t = decltype(completion_fn);
           using barrier_t = cuda::barrier<cuda::thread_scope_block,
                                           completion_fn_t>;
           __shared__ std::aligned_storage<sizeof(barrier_t),
                                           alignof(barrier_t)>
               bar_storage;

           // Initialize barrier.
           barrier_t *bar = (barrier_t *)&bar_storage;
           if (block.thread_rank() == 0)
           {
             assert(*acc == 0);
             assert(blockDim.x == blockDim.y == blockDim.y == 1);
             new (bar) barrier_t{block.size(), completion_fn};
             /* equivalent to: init(bar, block.size(), completion_fn); */
           }
           block.sync();

           // Main loop.
           for (int i = 0; i < n; i += block.size())
           {
             smem[block.thread_rank()] = data[i] + *acc;
             auto token = bar->arrive();
             // We can do independent computation here.
             bar->wait(std::move(token));
             // Shared-memory is safe to re-use in the next iteration
             // since all threads are done with it, including the one
             // that did the reduction.
           }
         }

.. _async-barriers-tracking-async-mem-ops:

4.9.6. 跟踪异步内存操作
-----------------------

异步屏障可用于跟踪 :ref:`异步内存拷贝 <sec:async-copies>`。当异步拷贝操作绑定到屏障时，该拷贝操作在启动时自动递增当前屏障阶段的预期计数，并在完成时递减它。此机制确保屏障的 `wait()` 操作将阻塞，直到所有关联的异步内存拷贝完成，提供了一种方便的方式来同步多个并发内存操作。

从计算能力 9.0 开始，具有线程块或集群作用域的共享内存中的异步屏障可以 **显式** 跟踪异步内存操作。我们将这些屏障称为 *异步事务屏障*（asynchronous transaction barriers）。除了预期到达计数外，屏障对象还可以接受 **事务计数**（transaction count），可用于跟踪异步事务的完成情况。事务计数跟踪尚未完成的异步事务数量，以异步内存操作指定的单位（通常是字节）。当前阶段要跟踪的事务计数可以在到达时通过 `cuda::device::barrier_arrive_tx()` 设置，或直接通过 `cuda::device::barrier_expect_tx()` 设置。当屏障使用事务计数时，它会在线程执行 wait 操作时阻塞，直到所有生产者线程执行了 arrive **并且** 所有事务计数的总和达到预期值。

.. tab-set::

   .. tab-item:: CUDA C++ `cuda::barrier`

      .. code-block:: cpp

         #include <cuda/barrier>
         #include <cooperative_groups.h>

         __global__ void track_kernel()
         {
           __shared__ cuda::barrier<cuda::thread_scope_block> bar;
           auto block = cooperative_groups::this_thread_block();

           if (block.thread_rank() == 0)
           {
             init(&bar, block.size());
           }
           block.sync();

           auto token = cuda::device::barrier_arrive_tx(bar, 1, 0);

           bar.wait(cuda::std::move(token));
         }

   .. tab-item:: CUDA C++ `cuda::ptx`

      .. code-block:: cpp

         #include <cuda/ptx>
         #include <cooperative_groups.h>

         __global__ void track_kernel()
         {
           __shared__ uint64_t bar;
           auto block = cooperative_groups::this_thread_thread_block();

           if (block.thread_rank() == 0)
           {
             cuda::ptx::mbarrier_init(&bar, block.size());
           }
           block.sync();

           uint64_t token = cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release, cuda::ptx::scope_cluster, cuda::ptx::space_shared, &bar, 1, 0);

           while (!cuda::ptx::mbarrier_try_wait(&bar, token)) {}
         }

在此示例中，`cuda::device::barrier_arrive_tx()` 操作构造一个与当前阶段的阶段同步点相关联的到达 token 对象。然后，将到达计数减 1 并将预期事务计数加 0。由于事务计数更新为 0，屏障不跟踪任何事务。后续关于 :doc:`11-async-copies` 的部分包含跟踪异步内存操作的示例。

.. _async-barriers-producer-consumer:

4.9.7. 使用屏障的生产者-消费者模式
----------------------------------

线程块可以进行空间分区，以允许不同的线程执行独立的操作。这通常通过将线程块内不同 warp 的线程分配给特定任务来完成。这种技术称为 *warp 特化* （warp specialization）。

本节展示生产者-消费者模式中空间分区的一个示例，其中一个线程子集产生数据，另一个（不相交的）线程子集同时消费这些数据。生产者-消费者空间分区模式需要两个单向同步来管理生产者和消费者之间的数据缓冲区。

.. list-table::
   :header-rows: 1

   * - 生产者
     - 消费者
   * - 等待缓冲区准备好被填充
     - 发出缓冲区准备好被填充的信号
   * - 产生数据并填充缓冲区
     - 发出缓冲区已填充的信号
   * - 等待缓冲区被填充
     - 消费已填充缓冲区中的数据

生产者线程等待消费者线程发出缓冲区准备好被填充的信号；但是，消费者线程不等待此信号。消费者线程等待生产者线程发出缓冲区已填充的信号；但是，生产者线程不等待此信号。对于完整的生产者/消费者并发，此模式具有（至少）双缓冲，其中每个缓冲区需要两个屏障。

.. tab-set::

   .. tab-item:: CUDA C++ `cuda::barrier`

      .. code-block:: cpp

         #include <cuda/barrier>

         using barrier_t = cuda::barrier<cuda::thread_scope_block>;

         __device__ void produce(barrier_t ready[], barrier_t filled[], float *buffer, int buffer_len, float *in, int N)
         {
           for (int i = 0; i < N / buffer_len; ++i)
           {
             ready[i % 2].arrive_and_wait(); /* wait for buffer_(i%2) to be ready to be filled */
             /* produce, i.e., fill in, buffer_(i%2)  */
             barrier_t::arrival_token token = filled[i % 2].arrive(); /* buffer_(i%2) is filled */
           }
         }

         __device__ void consume(barrier_t ready[], barrier_t filled[], float *buffer, int buffer_len, float *out, int N)
         {
           barrier_t::arrival_token token1 = ready[0].arrive(); /* buffer_0 is ready for initial fill */
           barrier_t::arrival_token token2 = ready[1].arrive(); /* buffer_1 is ready for initial fill */
           for (int i = 0; i < N / buffer_len; ++i)
           {
             filled[i % 2].arrive_and_wait(); /* wait for buffer_(i%2) to be filled */
             /* consume buffer_(i%2) */
             barrier_t::arrival_token token3 = ready[i % 2].arrive(); /* buffer_(i%2) is ready to be re-filled */
           }
         }

         __global__ void producer_consumer_pattern(int N, float *in, float *out, int buffer_len)
         {
           constexpr int warpSize = 32;

           /* Shared memory buffer declared below is of size 2 * buffer_len
              so that we can alternatively work between two buffers.
              buffer_0 = buffer and buffer_1 = buffer + buffer_len */
           __shared__ extern float buffer[];

           /* bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
              while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively */
           #pragma nv_diag_suppress static_var_with_dynamic_init
           __shared__ barrier_t bar[4];

           if (threadIdx.x < 4)
           {
             init(bar + threadIdx.x, blockDim.x);
           }
           __syncthreads();

           if (threadIdx.x < warpSize)
           { produce(bar, bar + 2, buffer, buffer_len, in, N); }
           else
           { consume(bar, bar + 2, buffer, buffer_len, out, N); }
         }

   .. tab-item:: CUDA C++ `cuda::ptx`

      .. code-block:: cpp

         #include <cuda/ptx>

         __device__ void produce(barrier ready[], barrier filled[], float *buffer, int buffer_len, float *in, int N)
         {
           for (int i = 0; i < N / buffer_len; ++i)
           {
             uint64_t token1 = cuda::ptx::mbarrier_arrive(ready[i % 2]);
             while(!cuda::ptx::mbarrier_try_wait(&ready[i % 2], token1)) {} /* wait for buffer_(i%2) to be ready to be filled */
             /* produce, i.e., fill in, buffer_(i%2)  */
             uint64_t token2 = cuda::ptx::mbarrier_arrive(&filled[i % 2]); /* buffer_(i%2) is filled */
           }
         }

         __device__ void consume(barrier ready[], barrier filled[], float *buffer, buffer_len, float *out, int N)
         {
           uint64_t token1 = cuda::ptx::mbarrier_arrive(&ready[0]); /* buffer_0 is ready for initial fill */
           uint64_t token2 = cuda::ptx::mbarrier_arrive(&ready[1]); /* buffer_1 is ready for initial fill */
           for (int i = 0; i < N / buffer_len; ++i)
           {
             uint64_t token3 = cuda::ptx::mbarrier_arrive(&filled[i % 2]);
             while(!cuda::ptx::mbarrier_try_wait(&filled[i % 2], token3)) {} /* wait for buffer_(i%2) to be filled */
             /* consume buffer_(i%2) */
             uint64_t token4 = cuda::ptx::mbarrier_arrive(&ready[i % 2]); /* buffer_(i%2) is ready to be re-filled */
           }
         }

         __global__ void producer_consumer_pattern(int N, float *in, float *out, int buffer_len)
         {
           constexpr int warpSize = 32;

           /* Shared memory buffer declared below is of size 2 * buffer_len
              so that we can alternatively work between two buffers.
              buffer_0 = buffer and buffer_1 = buffer + buffer_len */
           __shared__ extern float buffer[];

           /* bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
              while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively */
           #pragma nv_diag_suppress static_var_with_dynamic_init
           __shared__ uint64_t bar[4];

           if (threadIdx.x < 4)
           {
             cuda::ptx::mbarrier_init(bar + block.thread_rank(), block.size());
           }
           __syncthreads();

           if (threadIdx.x < warpSize)
           {  produce(bar, bar + 2, buffer, buffer_len, in, N); }
           else
           {  consume(bar, bar + 2, buffer, buffer_len, out, N); }
         }

   .. tab-item:: CUDA C 原语

      .. code-block:: cpp

         #include <cuda_awbarrier_primitives.h>

         __device__ void produce(__mbarrier_t ready[], __mbarrier_t filled[], float *buffer, int buffer_len, float *in, int N)
         {
           for (int i = 0; i < N / buffer_len; ++i)
           {
             __mbarrier_token_t token1 = __mbarrier_arrive(&ready[i % 2]); /* wait for buffer_(i%2) to be ready to be filled */
             while(!__mbarrier_try_wait(&ready[i % 2], token1, 1000)) {}
             /* produce, i.e., fill in, buffer_(i%2)  */
             __mbarrier_token_t token2 = __mbarrier_arrive(filled[i % 2]);  /* buffer_(i%2) is filled */
           }
         }

         __device__ void consume(__mbarrier_t ready[], __mbarrier_t filled[], float *buffer, int buffer_len, float *out, int N)
         {
           __mbarrier_token_t token1 = __mbarrier_arrive(&ready[0]); /* buffer_0 is ready for initial fill */
           __mbarrier_token_t token2 = __mbarrier_arrive(&ready[1]); /* buffer_1 is ready for initial fill */
           for (int i = 0; i < N / buffer_len; ++i)
           {
             __mbarrier_token_t token3 = __mbarrier_arrive(&filled[i % 2]);
             while(!__mbarrier_try_wait(&filled[i % 2], token3, 1000)) {}
             /* consume buffer_(i%2) */
             __mbarrier_token_t token4 = __mbarrier_arrive(&ready[i % 2]); /* buffer_(i%2) is ready to be re-filled */
           }
         }

         __global__ void producer_consumer_pattern(int N, float *in, float *out, int buffer_len)
         {
           constexpr int warpSize = 32;

           /* Shared memory buffer declared below is of size 2 * buffer_len
              so that we can alternatively work between two buffers.
              buffer_0 = buffer and buffer_1 = buffer + buffer_len */
           __shared__ extern float buffer[];

           /* bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
              while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively */
           #pragma nv_diag_suppress static_var_with_dynamic_init
           __shared__ __mbarrier_t bar[4];

           if (threadIdx.x < 4)
           {
             __mbarrier_init(bar + threadIdx.x, blockDim.x);
           }
           __syncthreads();

           if (threadIdx.x < warpSize)
           { produce(bar, bar + 2, buffer, buffer_len, in, N); }
           else
           { consume(bar, bar + 2, buffer, buffer_len, out, N); }
         }

在此示例中，第一个 warp 特化为生产者，其余 warp 特化为消费者。所有生产者和消费者线程参与四个屏障中的每一个（调用 `bar.arrive()` 或 `bar.arrive_and_wait()` ），因此预期到达计数等于 `block.size()`。

生产者线程等待消费者线程发出共享内存缓冲区可以被填充的信号。为了等待屏障，生产者线程必须先到达 `ready[i%2].arrive()` 以获取 token，然后使用该 token 执行 `ready[i%2].wait(token)`。为简化起见，`ready[i%2].arrive_and_wait()` 合并了这些操作。

.. code-block:: cpp

   bar.arrive_and_wait();
   /* is equivalent to */
   bar.wait(bar.arrive());

生产者线程计算并填充就绪缓冲区，然后通过到达 filled 屏障 `filled[i%2].arrive()` 发出缓冲区已填充的信号。生产者线程此时不等待，而是等待下一迭代的缓冲区（双缓冲）准备好被填充。

消费者线程首先发出两个缓冲区都准备好被填充的信号。消费者线程此时不等待，而是等待此迭代的缓冲区被填充 `filled[i%2].arrive_and_wait()`。消费者线程消费缓冲区后，发出缓冲区准备好再次被填充的信号 `ready[i%2].arrive()`，然后等待下一迭代的缓冲区被填充。