.. _cuda-cpp-execution-model:

第 5.8 节 CUDA C++ 执行模型
============================

CUDA C++ 旨在为所有设备执行线程提供 `parallel forward progress [intro.progress.9] <https://eel.is/c++draft/intro.progress#9>`__，从而促进使用 CUDA C++ 并行化现有的 C++ 应用程序。

.. dropdown:: [intro.progress]

   - `[intro.progress.7] <https://eel.is/c++draft/intro.progress#7>`__：对于提供 `concurrent forward progress guarantees <https://eel.is/c++draft/intro.progress#def:concurrent_forward_progress_guarantees>`__ 的执行线程，实现确保该线程只要未终止就会最终取得进展。

     [注 5：这适用于无论其他执行线程（如果有）是否已经或正在取得进展。最终满足此要求意味着这将在未指定但有限的时间内发生。——尾注]

   - `[intro.progress.9] <https://eel.is/c++draft/intro.progress#9>`__：对于提供 `parallel forward progress guarantees <https://eel.is/c++draft/intro.progress#9>`__ 的执行线程，如果该线程尚未执行任何执行步骤，则实现不需要确保该线程最终会取得进展；一旦该线程执行了一个步骤，它就提供 `concurrent forward progress guarantees <https://eel.is/c++draft/intro.progress#def:concurrent_forward_progress_guarantees>`__。

     [注 6：这没有规定何时启动此执行线程的要求，这通常由创建此执行线程的实体指定。例如，一个提供并发前进保证的执行线程以任意顺序从一组任务中逐个执行任务，满足这些任务的并行前进要求。——尾注]

CUDA C++ 编程语言是 C++ 编程语言的扩展。本节记录了对当前 `ISO International Standard ISO/IEC 14882 – Programming Language C++ <https://eel.is/c++draft/>`__ 草案的 `[intro.progress] <https://eel.is/c++draft/intro.progress>`__ 部分的修改和扩展。修改的部分会明确标出，其差异以**粗体**显示。所有其他部分为新增内容。

.. _cuda-cpp-execution-model-host-threads:

第 5.8.1 节 主机线程
--------------------

由主机实现创建的执行线程用于执行 `main <https://en.cppreference.com/w/cpp/language/main_function>`__、`std::thread <https://en.cppreference.com/w/cpp/thread/thread>`__ 和 `std::jthread <https://en.cppreference.com/w/cpp/thread/jthread>`__ 所提供的前进保证是主机实现 `[intro.progress] <https://eel.is/c++draft/intro.progress>`__ 的实现定义行为。通用主机实现应提供并发前进保证。

如果主机实现提供 `concurrent forward progress [intro.progress.7] <https://eel.is/c++draft/intro.progress#7>`__，则 CUDA C++ 为设备线程提供 `parallel forward progress [intro.progress.9] <https://eel.is/c++draft/intro.progress#9>`__。

.. _cuda-cpp-execution-model-device-threads:

第 5.8.2 节 设备线程
--------------------

一旦设备线程取得进展：

- 如果它是 `Cooperative Grid <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g504b94170f83285c71031be6d5d15f73>`__ 的一部分，则其 grid 中的所有设备线程应最终取得进展。

- 否则，其 :ref:`thread-block-cluster` 中的所有设备线程应最终取得进展。

  [注：其他线程块集群中的线程不保证最终取得进展。——尾注]

  [注：这意味着其线程块内的所有设备线程应最终取得进展。——尾注]

对 `[intro.progress.1] <https://eel.is/c++draft/intro.progress#1>`__ 进行如下修改（修改以**粗体**显示）：

实现可以假设任何 **主机** 线程最终将执行以下操作之一：

   1. 终止，

   2. 调用函数 `std::this_thread::yield <https://en.cppreference.com/w/cpp/thread/yield>`__（`[thread.thread.this] <http://eel.is/c++draft/thread.thread.this>`__），

   3. 调用库 I/O 函数，

   4. 通过 volatile 泛左值进行访问，

   5. 执行同步操作或原子操作，或

   6. 继续执行平凡无限循环（`[stmt.iter.general] <http://eel.is/c++draft/stmt.iter.general>`__）。

**实现可以假设任何设备线程最终将执行以下操作之一：**

   1. **终止**，

   2. **调用库 I/O 函数**，

   3. **通过 volatile 泛左值进行访问，除非指定对象具有自动存储期，或**

   4. **执行同步操作或原子读取操作，除非指定对象具有自动存储期。**

   [注：设备线程相对于主机线程的一些当前限制是我们已知的实现缺陷，我们可能会随着时间的推移修复这些缺陷。示例包括设备线程最终仅对自动存储期对象执行 volatile 或原子操作所产生的未定义行为。但是，设备线程相对于主机线程的其他限制是有意的选择。它们使性能优化成为可能，如果设备线程严格遵循 C++ 标准，则无法实现这些优化。例如，为最终仅执行原子写入或栅栏的程序提供前进保证会降低整体性能，而实际收益很小。——尾注]

.. dropdown:: 主机线程和设备线程前进保证差异示例（由于对 [intro.progress.1] 的修改）

   以下示例使用 "host.threads.<id>" 和 "device.threads.<id>" 分别指代上述主机和设备线程的实现假设的条款子项。

   .. code-block:: cuda
      :linenos:
      :caption: 示例：Execution.Model.Device.0

      // Outcome: grid eventually terminates per device.threads.4 because the atomic object does not have automatic storage duration.
      __global__ void ex0(cuda::atomic_ref<int, cuda::thread_scope_device> atom) {
          if (threadIdx.x == 0) {
              while(atom.load(cuda::memory_order_relaxed) == 0);
          } else if (threadIdx.x == 1) {
              atom.store(1, cuda::memory_order_relaxed);
          }
      }

   .. code-block:: cuda
      :linenos:
      :caption: 示例：Execution.Model.Device.1

      // Allowed outcome: No thread makes progress because device threads don't support host.threads.2.
      __global__ void ex1() {
          while(true) cuda::std::this_thread::yield();
      }

   .. code-block:: cuda
      :linenos:
      :caption: 示例：Execution.Model.Device.2

      // Allowed outcome: No thread makes progress because device threads don't support host.threads.4
      // for objects with automatic storage duration (see exception in device.threads.3).
      __global__ void ex2() {
          volatile bool True = true;
          while(True);
      }

   .. code-block:: cuda
      :linenos:
      :caption: 示例：Execution.Model.Device.3

      // Allowed outcome: No thread makes progress because device threads don't support host.threads.5
      // for objects with automatic storage duration (see exception in device.threads.4).
      __global__ void ex3() {
          cuda::atomic<bool, cuda::thread_scope_thread> True = true;
          while(True.load());
      }

   .. code-block:: cuda
      :linenos:
      :caption: 示例：Execution.Model.Device.4

      // Allowed outcome: No thread makes progress because device threads don't support host.thread.6.
      __global void ex4() {
          while(true) { /* empty */ }
      }

.. _cuda-cpp-execution-model-cuda-apis:

第 5.8.3 节 CUDA API
--------------------

CUDA API 调用应最终返回或确保至少一个设备线程取得进展。

CUDA 查询函数（例如 `cudaStreamQuery <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435>`__、`cudaEventQuery <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2bf738909b4a059023537eaa29d8a5b7>`__ 等）不应在没有设备线程取得进展的情况下持续返回 ``cudaErrorNotReady`` 。

[注：设备线程不需要与 API 调用"相关"，例如，在一个流或进程上操作的 API 可以确保另一个流或进程上的设备线程取得进展。——尾注]

[注：测试程序是否符合 CUDA API 前进保证的一种简单但不充分的方法是使用以下环境变量运行程序： ``CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_LAUNCH_BLOCKING=1`` ，然后检查程序是否仍然终止。如果不终止，程序就有错误。此方法不充分，因为它不能捕获所有前进保证错误，但确实能捕获许多此类错误。——尾注]

.. dropdown:: CUDA API 前进保证示例

   .. code-block:: cuda
      :linenos:
      :caption: 示例：Execution.Model.API.1

      // Outcome: if no other device threads (e.g., from other processes) are making progress,
      // this program terminates and returns cudaSuccess.
      // Rationale: CUDA guarantees that if the device is empty:
      // - `cudaDeviceSynchronize` eventually ensures that at least one device-thread makes progress, which implies that eventually `hello_world` grid and one of its device-threads start.
      // - All thread-block threads eventually start (due to "if a device thread makes progress, all other threads in its thread-block cluster eventually make progress").
      // - Once all threads in thread-block arrive at `__syncthreads` barrier, all waiting threads are unblocked.
      // - Therefore all device threads eventually exit the `hello_world`` grid.
      // - And `cudaDeviceSynchronize`` eventually unblocks.
      __global__ void hello_world() { __syncthreads(); }
      int main() {
          hello_world<<<1,2>>>();
          return (int)cudaDeviceSynchronize();
      }

   .. code-block:: cuda
      :linenos:
      :caption: 示例：Execution.Model.API.2

      // Allowed outcome: eventually, no thread makes progress.
      // Rationale: the `cudaDeviceSynchronize` API below is only called if a device thread eventually makes progress and sets the flag.
      // However, CUDA only guarantees that `producer` device thread eventually starts if the synchronization API is called.
      // Therefore, the host thread may never be unblocked from the flag spin-loop.
      cuda::atomic<int, cuda::thread_scope_system> flag = 0;
      __global__ void producer() { flag.store(1); }
      int main() {
          cudaHostRegister(&flag, sizeof(flag));
          producer<<<1,1>>>();
          while (flag.load() == 0);
          return cudaDeviceSynchronize();
      }

   .. code-block:: cuda
      :linenos:
      :caption: 示例：Execution.Model.API.3

      // Allowed outcome: eventually, no thread makes progress.
      // Rationale: same as Example.Model.API.2, with the addition that a single CUDA query API call does not guarantee
      // the device thread eventually starts, only repeated CUDA query API calls do (see Execution.Model.API.4).
      cuda::atomic<int, cuda::thread_scope_system> flag = 0;
      __global__ void producer() { flag.store(1); }
      int main() {
          cudaHostRegister(&flag, sizeof(flag));
          producer<<<1,1>>>();
          (void)cudaStreamQuery(0);
          while (flag.load() == 0);
          return cudaDeviceSynchronize();
      }

   .. code-block:: cuda
      :linenos:
      :caption: 示例：Execution.Model.API.4

      // Outcome: terminates.
      // Rationale: same as Execution.Model.API.3, but this example repeatedly calls
      // a CUDA query API in within the flag spin-loop, which guarantees that the device thread
      // eventually makes progress.
      cuda::atomic<int, cuda::thread_scope_system> flag = 0;
      __global__ void producer() { flag.store(1); }
      int main() {
          cudaHostRegister(&flag, sizeof(flag));
          producer<<<1,1>>>();
          while (flag.load() == 0) {
              (void)cudaStreamQuery(0);
          }
          return cudaDeviceSynchronize();
      }

.. _cuda-cpp-execution-model-dependencies:

第 5.8.3.1 节 依赖关系
^^^^^^^^^^^^^^^^^^^^^^

设备线程在其所有依赖项完成之前不应启动。

[注：阻止设备线程开始取得进展的依赖关系可以通过多种方式创建，例如通过 :ref:`cuda-streams`。这些依赖关系可能包括对 :ref:`cuda-events` 和 :ref:`kernels` 等完成的依赖。——尾注]

.. dropdown:: 依赖关系导致的 CUDA API 前进保证示例

   .. code-block:: cuda
      :linenos:
      :caption: 示例：Execution.Model.Stream.0

      // Allowed outcome: eventually, no thread makes progress.
      // Rationale: while CUDA guarantees that one device thread makes progress, since there
      // is no dependency between `first` and `second`, it does not guarantee which thread,
      // and therefore it could always pick the device thread from `second`, which then never
      // unblocks from the spin-loop.
      // That is, `second` may starve `first`.
      cuda::atomic<int, cuda::thread_scope_system> flag = 0;
      __global__ void first() { flag.store(1, cuda::memory_order_relaxed); }
      __global__ void second() { while(flag.load(cuda::memory_order_relaxed) == 0) {} }
      int main() {
          cudaHostRegister(&flag, sizeof(flag));
          cudaStream_t s0, s1;
          cudaStreamCreate(&s0);
          cudaStreamCreate(&s1);
          first<<<1,1,0,s0>>>();
          second<<<1,1,0,s1>>>();
          return cudaDeviceSynchronize();
      }

   .. code-block:: cuda
      :linenos:
      :caption: 示例：Execution.Model.Stream.1

      // Outcome: terminates.
      // Rationale: same as Execution.Model.Stream.0, but this example has a stream dependency
      // between first and second, which requires CUDA to run the grids in order.
      cuda::atomic<int, cuda::thread_scope_system> flag = 0;
      __global__ void first() { flag.store(1, cuda::memory_order_relaxed); }
      __global__ void second() { while(flag.load(cuda::memory_order_relaxed) == 0) {} }
      int main() {
          cudaHostRegister(&flag, sizeof(flag));
          cudaStream_t s0;
          cudaStreamCreate(&s0);
          first<<<1,1,0,s0>>>();
          second<<<1,1,0,s0>>>();
          return cudaDeviceSynchronize();
      }