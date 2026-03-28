.. _cuda-cplusplus-memory-model:

CUDA C++ 内存模型
==================

标准 C++ 呈现的观点是线程同步的成本是统一且较低的。

CUDA C++ 则不同：线程同步的成本随着线程之间的距离增加而增长。在线程块内的线程之间成本较低，但在多 GPU 和 CPU 上运行的任意线程之间成本较高。

为了应对并非总是较低的非统一线程同步成本，CUDA C++ 在 ``cuda::`` 命名空间中通过**线程作用域 (thread scopes)** 扩展了标准 C++ 内存模型和并发设施，同时默认保留标准 C++ 的语法和语义。

.. _cuda-cplusplus-memory-model-thread-scopes:

线程作用域
----------

**线程作用域** 指定可以使用同步原语（如 :ref:`cuda::atomic <cuda-atomic>` 或 :ref:`cuda::barrier <cuda-barrier>` ）相互同步的线程类型。

.. code-block:: cuda

   namespace cuda {

   enum thread_scope {
     thread_scope_system,
     thread_scope_device,
     thread_scope_block,
     thread_scope_thread
   };

   }  // namespace cuda

作用域关系
~~~~~~~~~~

每个程序线程通过一个或多个线程作用域关系与其他程序线程相关联：

- 系统中的每个线程通过 *系统* 线程作用域与系统中的每个其他线程相关联： ``cuda::thread_scope_system`` 。
- 每个 GPU 线程通过 *设备* 线程作用域与同一 CUDA 设备内且同一 :ref:`内存同步域 <memory-synchronization-domains>` 中的每个其他 GPU 线程相关联： ``cuda::thread_scope_device`` 。
- 每个 GPU 线程通过 *线程块* 线程作用域与同一 CUDA 线程块中的每个其他 GPU 线程相关联： ``cuda::thread_scope_block`` 。
- 每个线程通过 *线程* 线程作用域与其自身相关联： ``cuda::thread_scope_thread`` 。

同步原语
--------

``std::`` 和 ``cuda::std::`` 命名空间中的类型在以 ``cuda::thread_scope_system`` 作用域实例化时，其行为与 ``cuda::`` 命名空间中的相应类型相同。

原子性
------

原子操作在其指定的作用域上是原子的，如果：

- 它指定的作用域不是 ``cuda::thread_scope_system`` ，**或者**
- 作用域是 ``cuda::thread_scope_system`` **并且**：
  
  - 它影响 :ref:`系统分配内存 <um-details-intro>` 中的对象，且 `pageableMemoryAccess`_ 为 ``1`` [0]，**或者**
  - 它影响 :ref:`托管内存 <um-details-intro>` 中的对象，且 `concurrentManagedAccess`_ 为 ``1`` ，**或者**
  - 它影响 :ref:`映射内存 <memory-mapped-memory>` 中的对象，且 `hostNativeAtomicSupported`_ 为 ``1`` ，**或者**
  - 它是在 :ref:`映射内存 <memory-mapped-memory>` 上影响大小为 ``1``、``2``、``4``、``8`` 或 ``16`` 字节的自然对齐对象的加载或存储操作 [1]，**或者**
  - 它影响 GPU 内存中的对象，只有 GPU 线程访问它，**并且**
    
    - 在每个访问的 ``srcDev`` 和对象所在的 GPU ``dstDev`` 之间，`cudaDeviceGetP2PAttribute`_ ``(&val, cudaDevP2PAttrNativeAtomicSupported, srcDev, dstDev)`` 为 ``1`` ，**或者**
    - 只有来自单个 GPU 的 GPU 线程并发访问它。

.. note::

   - [0] 如果 `PageableMemoryAccessUsesHostPagetables`_ 为 ``0`` ，则对内存映射文件或 ``hugetlbfs`` 分配的原子操作不是原子的。
   - [1] 如果 `hostNativeAtomicSupported`_ 为 ``0`` ，则对 :ref:`系统分配内存 <um-details-intro>` 或 :ref:`映射内存 <memory-mapped-memory>` 中自然对齐的 16 字节宽对象进行的系统作用域原子加载或存储操作需要系统支持。NVIDIA 尚未发现任何缺乏此支持的系统，且没有可用的 CUDA API 查询来检测此类系统。

有关 :ref:`系统分配内存 <um-details-intro>`、:ref:`托管内存 <um-details-intro>`、:ref:`映射内存 <memory-mapped-memory>`、CPU 内存和 GPU 内存的更多信息，请参阅本指南的相关章节。

.. _pageableMemoryAccess: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg49e2f8c2c0bd6fe264f2fc970912e5cddc80992427a92713e699953a6d249d6f
.. _concurrentManagedAccess: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b
.. _hostNativeAtomicSupported: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f
.. _cudaDeviceGetP2PAttribute: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g2f597e2acceab33f60bd61c41fea0c1b
.. _PageableMemoryAccessUsesHostPagetables: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg49e2f8c2c0bd6fe264f2fc970912e5cdc228cf8983c97d0e035da72a71494eaa

数据竞争
--------

修改 ISO/IEC IS 14882（C++ 标准）的 `intro.races 第 21 段`_ 如下：

   程序的执行包含数据竞争，如果它包含两个可能并发的冲突操作，其中至少有一个**在包含执行另一个操作的线程的作用域上**不是原子的，并且两者之间没有发生先于关系，除了下面描述的信号处理程序的特殊情况。任何此类数据竞争都会导致未定义行为。[...]

修改 ISO/IEC IS 14882（C++ 标准）的 `thread.barrier.class 第 4 段`_ 如下：

   4. ``barrier`` 的成员函数的并发调用，除了其析构函数外， **就像它们是原子操作一样** 不会引入数据竞争。[...]

修改 ISO/IEC IS 14882（C++ 标准）的 `thread.latch.class 第 2 段`_ 如下：

   2. ``latch`` 的成员函数的并发调用，除了其析构函数外， **就像它们是原子操作一样** 不会引入数据竞争。[...]

修改 ISO/IEC IS 14882（C++ 标准）的 `thread.sema.cnt 第 3 段`_ 如下：

   3. ``counting_semaphore`` 的成员函数的并发调用，除了其析构函数外， **就像它们是原子操作一样** 不会引入数据竞争。

修改 ISO/IEC IS 14882（C++ 标准）的 `thread.stoptoken.intro 第 5 段`_ 如下：

   对函数 ``request_stop``、``stop_requested`` 和 ``stop_possible`` 的调用**就像它们是原子操作一样**不会引入数据竞争。[...]

修改 ISO/IEC IS 14882（C++ 标准）的 `atomics.fences 第 2 到 4 段`_ 如下：

   如果存在原子操作 X 和 Y，两者都对某个原子对象 M 进行操作，使得 A 在 X 之前排序，X 修改 M，Y 在 B 之前排序，并且 Y 读取由 X 写入的值或由 X 如果是释放操作则将引导的假设释放序列中的任何副作用写入的值，**并且每个操作（A、B、X 和 Y）指定的作用域包含执行每个其他操作的线程**，则释放栅栏 A 与获取栅栏 B 同步。

   如果存在原子操作 X，使得 A 在 X 之前排序，X 修改 M，并且 B 读取由 X 写入的值或由 X 如果是释放操作则将引导的假设释放序列中的任何副作用写入的值，**并且每个操作（A、B 和 X）指定的作用域包含执行每个其他操作的线程**，则释放栅栏 A 与对原子对象 M 执行获取操作的原子操作 B 同步。

   如果在 M 上存在某个原子操作 X，使得 X 在 B 之前排序并读取由 A 写入的值或由 A 引导的释放序列中的任何副作用写入的值，**并且每个操作（A、B 和 X）指定的作用域包含执行每个其他操作的线程**，则对原子对象 M 的释放操作原子操作 A 与获取栅栏 B 同步。

.. _intro.races 第 21 段: https://eel.is/c++draft/intro.races#21
.. _thread.barrier.class 第 4 段: https://eel.is/c++draft/thread.barrier.class#4
.. _thread.latch.class 第 2 段: https://eel.is/c++draft/thread.latch.class#2
.. _thread.sema.cnt 第 3 段: https://eel.is/c++draft/thread.sema.cnt#3
.. _thread.stoptoken.intro 第 5 段: https://eel.is/c++draft/thread#stoptoken.intro-5
.. _atomics.fences 第 2 到 4 段: https://eel.is/c++draft/atomics.fences#2

.. _cuda-cplusplus-memory-model-message-passing:

示例：消息传递
--------------

以下示例通过标志 ``f`` 将线程块 ``0`` 中的某个线程存储到 ``x`` 变量的消息传递给线程块 ``1`` 中的某个线程：

.. list-table::
   :widths: 100

   * - **初始状态**
   * - .. code-block:: cpp

        int x = 0, f = 0;
   * - **线程 0 线程块 0**
   * - .. code-block:: cpp

        x = 42;
        cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);
        flag.store(1, memory_order_release);
   * - **线程 0 线程块 1**
   * - .. code-block:: cpp

        cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);
        while(flag.load(memory_order_acquire) != 1);
        assert(x == 42);

在以下对前一示例的变体中，两个线程在没有同步的情况下并发访问 ``f`` 对象，这会导致**数据竞争**，并表现出**未定义行为**：

.. list-table::
   :widths: 100

   * - **初始状态**
   * - .. code-block:: cpp

        int x = 0, f = 0;
   * - **线程 0 线程块 0**
   * - .. code-block:: cpp

        x = 42;
        cuda::atomic_ref<int, cuda::thread_scope_block> flag(f);
        flag.store(1, memory_order_release); // UB: 数据竞争
   * - **线程 0 线程块 1**
   * - .. code-block:: cpp

        cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);
        while(flag.load(memory_order_acquire) != 1); // UB: 数据竞争
        assert(x == 42);

虽然对 ``f`` 的内存操作——存储和加载——是原子的，但存储操作的作用域是「线程块作用域」。由于存储由线程块 0 的线程 0 执行，它只包含线程块 0 的所有其他线程。然而，执行加载的线程在线程块 1 中，即它不在线程块 0 中执行的操作所包含的作用域中，导致存储和加载不是「原子的」，从而引入数据竞争。

更多示例请参阅 `PTX 内存一致性模型 litmus 测试`_。

.. _PTX 内存一致性模型 litmus 测试: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#axioms
.. _cuda-atomic: https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html
.. _cuda-barrier: https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/barrier.html