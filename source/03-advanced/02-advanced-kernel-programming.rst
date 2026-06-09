.. _advanced-kernel-programming:

3.2. 高级核函数编程
=======================

本章首先深入探讨 NVIDIA GPU 的硬件模型，然后介绍 CUDA 核函数代码中一些旨在提高核函数性能的高级特性。
本章将介绍与线程作用域、异步执行以及相关同步原语的一些概念。
这些概念性讨论为核函数中可用的一些高级性能特性提供了必要的基础。

其中一些特性的详细描述将在本编程指南下一部分中专门介绍。

- 本章介绍的 :ref:`高级同步原语<advanced-synchronization-primitives>` 在 :ref:`async-barriers-details` 和 :ref:`pipelines-details` 中有完整介绍。
- 本章介绍的 :ref:`异步数据拷贝<asynchronous-data-copies>` , 包括张量内存加速器（ tensor memory accelerator, TMA ）在 :ref:`async-copies-details` 中有完整介绍。

.. _using-ptx:

3.2.1. 使用 PTX
-------------------

并行线程执行（Parallel Thread Execution，PTX）是 CUDA 用来抽象硬件指令集 （instruction set architecture， ISA） 的虚拟机指令集架构，在 :ref:`parallel-thread-execution-ptx` 中已介绍。
直接用 PTX 编写代码是一种非常高级的优化技术，大多数开发者并不需要，应该作为最后的手段。
然而，在某些情况下，直接编写 PTX 所启用的细粒度控制可以在特定应用中实现性能改进。
这些情况通常出现在应用中对性能极其敏感的部分，每一分性能改进都有显著收益。
所有可用的 PTX 指令都在 `PTX ISA 文档 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>`__ 中。

``cuda::ptx`` **命名空间**

在代码中直接使用 PTX 的一种方法是使用 `libcu++ <https://nvidia.github.io/cccl/libcudacxx/>`__ 中的 ``cuda::ptx`` 命名空间。
该命名空间提供了直接映射到 PTX 指令的 C++ 函数，简化了它们在 C++ 应用程序中的使用。
更多信息，请参阅 `cuda::ptx 命名空间 <https://nvidia.github.io/cccl/libcudacxx/ptx_api.html>`__ 文档。

**内联 PTX**

另一种在代码中包含 PTX 的方法是使用内联 PTX。
该方法在相应的 `文档 <https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html>`__ 中有详细描述。
这与在 CPU 上编写汇编代码非常相似。

.. _hardware-implementation:

3.2.2. 硬件实现
-------------------

流式多处理器或 SM（参见 :ref:`gpu-hardware-model` ）设计用于并发执行数百个线程。
为了管理如此大量的线程，它采用了一种称为单指令多线程（Single-Instruction, Multiple-Thread，SIMT）的独特并行计算模型，
该模型在 :ref:`simt-execution-model` 中描述。
指令采用流水线方式，利用单线程内的指令级并行性，以及通过同时硬件多线程实现的广泛线程级并行性，详见 :ref:`hardware-multithreading`。
与 CPU 核心不同，SM 按顺序发射指令，不进行分支预测或投机执行。

:ref:`simt-execution-model` 和 :ref:`hardware-multithreading` 描述了 SM 中所有设备共有的架构特性。
:ref:`compute-capabilities` 提供了不同计算能力设备的详细信息。

NVIDIA GPU 架构使用小端表示法。

.. _simt-execution-model:

3.2.2.1. SIMT 执行模型
^^^^^^^^^^^^^^^^^^^^^^^

每个 SM 创建、管理、调度和执行称为 *warp* 的 32 个并行线程组。
组成 warp 的各个线程从相同的程序地址一起开始，但它们有自己的指令地址计数器和寄存器状态，因此可以自由地分支和独立执行。
*warp* 一词源于编织，这是第一个并行线程技术。
*half-warp* 是 warp 的前半部分或后半部分。
*quarter-warp* 是 warp 的第一、第二、第三或第四个四分之一。

一个 warp 一次执行一条公共指令，因此当 warp 的所有 32 个线程在执行路径上一致时，可以实现完全效率。
如果 warp 的线程通过数据相关的条件分支发生分歧，warp 会执行所采用的每条分支路径，禁用不在该路径上的线程。
分支分歧只发生在 warp 内部；不同的 warp 独立执行，无论它们执行的是公共还是不相交的代码路径。

SIMT 架构类似于 SIMD（单指令多数据）向量组织，因为单条指令控制多个处理元素。
一个关键区别是 SIMD 向量组织将 SIMD 宽度暴露给软件，而 SIMT 指令指定单个线程的执行和分支行为。
与 SIMD 向量机相比，SIMT 使程序员能够为独立的标量线程编写线程级并行代码，以及为协调线程编写数据并行代码。
为了正确性，程序员基本上可以忽略 SIMT 行为；然而，只要注意让同一个线程束（warp）中的线程极少出现执行分歧，就能显著提升性能。
在实践中，这类似于缓存行的作用：在设计正确性时可以安全地忽略缓存行大小，但在设计峰值性能时必须在代码结构中考虑它。
另一方面，向量架构要求软件将加载合并为向量并手动管理分歧。

.. _independent-thread-scheduling:

3.2.2.1.1. 独立线程调度
"""""""""""""""""""""""""""""""

在计算能力低于 7.0 的 GPU 上，warp 使用所有 32 个线程共享的单个程序计数器以及指定 warp 活动线程的活动掩码。
因此，来自同一 warp 的线程在分歧区域或不同执行状态时无法相互发信号或交换数据，
需要锁或互斥锁保护的细粒度数据共享的算法可能导致死锁，这取决于竞争线程来自哪个 warp。

在计算能力 7.0 及更高的 GPU 中， **独立线程调度** 允许线程之间完全并发，无论 warp 如何。
通过独立线程调度，GPU 维护每个线程的执行状态，包括程序计数器和调用栈，
并且可以以每个线程的粒度让出执行，要么是为了更好地利用执行资源，要么是为了允许一个线程等待另一个线程产生的数据。
调度优化器确定如何将同一 warp 的活动线程分组为 SIMT 单元。
这保留了先前 NVIDIA GPU 中 SIMT 执行的高吞吐量，但具有更大的灵活性：线程现在可以在子 warp 粒度上分歧和重新汇聚。

独立线程调度可能会破坏依赖先前 GPU 架构隐式 warp 同步行为的代码。
Warp 同步代码假设同一 warp 中的线程在每条指令上都以锁步方式执行，但线程在子 warp 粒度上分歧和重新汇聚的能力使这种假设无效。
这可能导致与预期不同的线程集参与执行代码。
任何为 CC 7.0 之前的 GPU 开发的 warp 同步代码（如无同步的 warp 内归约）都应该重新审视以确保兼容性。
开发者应该使用 ``__syncwarp()`` 显式同步此类代码，以确保在所有 GPU 代际上的正确行为。

.. _simt-architecture-notes:

.. note::

   参与当前指令的 warp 线程称为活动线程，而不在当前指令上的线程是非活动（禁用）线程。
   线程可能因多种原因而变为非活动，包括比 warp 的其他线程更早退出、采用了与 warp 当前执行的分支路径不同的分支路径，
   或者是线程数不是 warp 大小倍数的块的最后线程。
   
   如果 warp 执行的非原子指令从多个线程写入全局或共享内存中的同一位置，则发生的对该位置的序列化写入次数可能因设备的计算能力而异。
   但是，对于所有计算能力，哪个线程执行最终写入是未定义的。
   
   如果 warp 执行的 :ref:`atomic <atomic-functions>` 指令从多个线程读取、修改并写入全局内存中的同一位置，则对该位置的每次读取/修改/写入都会发生，并且它们都被序列化，但它们发生的顺序是未定义的。

.. _hardware-multithreading:

3.2.2.2. 硬件多线程
^^^^^^^^^^^^^^^^^^^^^^^

当 SM 被赋予一个或多个线程块来执行时，它将它们划分为 warp，每个 warp 由warp 调度器调度执行。
块划分为 warp 的方式总是相同的；每个 warp 包含连续递增线程 ID 的线程，第一个 warp 包含线程 0。
:ref:`writing-cuda-kernels-thread-hierarchy-review` 描述了线程 ID 与块中线程索引的关系。

块中 warp 的总数定义如下：

:math:`\text{ceil}\left( \frac{T}{W_{size}}, 1 \right)`

- *T* 是每块的线程数，
- *Wsize* 是 warp 大小，等于 32，
- ceil(x, y) 等于 x 向上舍入到 y 的最近倍数。

.. figure:: /_static/images/warps-in-a-block.png
   :alt: 线程块被划分为 32 个线程的 warp
   :figwidth: 80%

   线程块被划分为 32 个线程的 warp

SM 处理的每个 warp 的执行上下文（程序计数器、寄存器等）在 warp 的整个生命周期内都保持在片上。因此，在 warp 之间切换没有成本。
在每个指令发射周期，warp 调度器选择一个有线程准备好执行其下一条指令的 warp（warp 的 :ref:`活动线程 <simt-architecture-notes>` ），
并将指令发射给这些线程。

每个 SM 拥有一组 32 位寄存器，这些寄存器会被划分给各个 warp（线程束）使用；
同时它还拥有一块 :ref:`共享内存 <writing-cuda-kernels-shared-memory>` ，被划分给各个线程块（thread blocks）使用。
对于一个 kernel ，能有多少个线程块和 warp 驻留在 SM 上并发执行，主要取决于两个因素：一是该 kernel 实际消耗的寄存器和共享内存大小，二是该 SM 本身所拥有的寄存器和共享内存总量。
此外，每个 SM 能驻留的线程块和 warp 数量也有一个上限值。
这些限制条件，以及 SM 可用的寄存器和共享内存容量，都取决于设备的计算能力（Compute Capability），在 :ref:`compute-capabilities` 中有着明确的规格说明。
如果单个 SM 的可用资源连一个线程块的需求都无法满足，那么 kernel 将无法启动。
至于如何计算一个线程块需要的寄存器和共享内存总量，可以通过 :ref:`writing-cuda-kernels-kernel-launch-and-occupancy` 中记录的几种方法来确定。

.. _asynchronous-execution-features:

3.2.2.3. 异步执行特性
^^^^^^^^^^^^^^^^^^^^^^^^^

近几代的 NVIDIA GPU 都加入了异步执行功能，旨在让 GPU 内部的数据搬运、计算以及同步操作能够更大程度地重叠进行。
这些能力使从 GPU 代码调用的某些操作能够与同一线程块中的其他 GPU 代码异步执行。
这种异步执行不应与 :ref:`asynchronous-execution` 中讨论的异步 CUDA API 混淆，后者使 GPU 核函数启动或内存操作能够彼此或与 CPU 异步运行。

计算能力 8.0（NVIDIA Ampere GPU 架构）引入了从全局内存到共享内存的硬件加速异步数据拷贝和异步屏障（参见 `NVIDIA A100 Tensor Core GPU 架构 <https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf>`__）。

计算能力 9.0（NVIDIA Hopper GPU 架构）通过 :ref:`Tensor Memory Accelerator (TMA) <asynchronous-data-copies>` 单元扩展了异步执行特性，
该单元可以在全局内存和共享内存之间传输大数据块和多维张量，
以及异步事务屏障和异步矩阵乘累加操作（详见 `Hopper 架构深入 <https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/>`__ 博客文章）。

CUDA 提供了一些 API，设备端代码（device code）中的线程可以通过调用它们来使用这些功能。
而这个异步编程模型，则定义了异步操作相对于 CUDA 线程的具体行为方式。

所谓异步操作，是指由一个 CUDA 线程发起，但实际上却是异步执行的的操作（就好比是由另一个线程执行的一样）。
为了表述方便，我们把实际执行这个操作的虚拟角色称为“异步线程（async thread）”。
在一个编写规范的程序中，必须有一个或多个 CUDA 线程与该异步操作进行同步。
不过，并不强制要求当初发起这个异步操作的那个 CUDA 线程参与同步。
但是需要明确的是，这个“异步线程”始终是与发起该操作的 CUDA 线程相互关联绑定的。

异步操作使用同步对象来表示其完成，同步对象可以是屏障或流水线。
这些同步对象在 :ref:`advanced-synchronization-primitives` 中详细解释，
在 :ref:`asynchronous-data-copies` 中演示它们在执行异步内存操作中的作用。

.. _async-thread-and-async-proxy:

3.2.2.3.1. 异步线程和异步代理
""""""""""""""""""""""""""""""""""

异步操作访问内存的方式可能与常规操作不同。为了区分这些不同的内存访问方法，CUDA 引入了异步线程（async thread）、通用代理（generic proxy）和异步代理（async proxy）这几个概念。
常规的操作（比如普通的加载和存储指令）是通过“通用代理”来进行的。
而某些异步指令，例如 :ref:`LDGSTS <using-ldgsts>` 和 :ref:`STAS/REDAS <using-stas>` ，被描述为：一个在“通用代理”中运行的异步线程。
另一些异步指令，比如使用 TMA（张量内存加速器）进行的大批量异步拷贝，以及部分 Tensor Core 操作（如 tcgen05.\*, wgmma.mma\_async.\*），则被描述为：一个在“异步代理”中运行的异步线程。

**在通用代理中进行的异步线程操作**。当发起异步操作时，它与一个异步线程相关联，该线程与发起操作的 CUDA 线程不同。
此前针对同一地址发起的常规代理（普通）加载和存储操作，保证会排在异步操作之前执行。
然而，在此之后针对同一地址进行的常规加载和存储操作，并不能保证顺序，在异步线程完成之前可能会引发竞态条件（race condition）。

**在异步代理中进行的异步线程操作**。当发起异步操作时，它与一个异步线程相关联，该线程与发起操作的 CUDA 线程不同。
在发起操作之前和之后针对同一地址的普通加载和存储操作不保证保持其顺序。
此时需要代理栅栏（proxy fence）来在不同的代理之间对它们进行同步，以确保正确的内存顺序。
:ref:`using-tma` 演示了使用代理栅栏确保使用 TMA 执行异步拷贝时的正确性。

有关这些概念的更多详细信息，请参阅 `PTX ISA <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=proxy#proxies>`__ 文档。

.. _thread-scopes:

3.2.3. 线程作用域
---------------------

CUDA 线程构成了一个 :ref:`writing-cuda-kernels-thread-hierarchy-review` ，要编写出既正确又高效的 CUDA kernel ，用好这个层次结构是至关重要的。
在这个层次结构中，内存操作的可见性以及同步的范围可能会有所不同。
为了应对这种非统一性，CUDA 编程模型引入了线程作用域（thread scopes）的概念。
线程作用域定义了哪些线程能够观察到某个线程的读写操作，并指明了哪些线程之间可以使用原子操作（atomic operations）和屏障（barriers）等同步原语来进行相互同步。
每个作用域在内存层次结构中都有一个与之对应的一致性点（point of coherency）。


线程作用域在 `CUDA PTX <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=thread%2520scopes#scope>`__ 中公开，也可作为 `libcu++ <https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes>`__ 库中的扩展使用。下表定义了可用的线程作用域：

.. list-table:: CUDA 线程作用域
   :header-rows: 1
   :widths: 20 20 35 25

   * - C++ 线程作用域
     - PTX 线程作用域
     - 描述
     - 内存一致性点
   * - ``cuda::thread_scope_thread``
     - –
     - 仅本地线程可见
     - –
   * - ``cuda::thread_scope_block``
     - ``.cta``
     - 同一线程块中的其他线程可见
     - L1
   * - 
     - ``.cluster``
     - 同一线程块集群中的其他线程可见
     - L2
   * - ``cuda::thread_scope_device``
     - ``.gpu``
     - 同一 GPU 设备中的其他线程可见
     - L2
   * - ``cuda::thread_scope_system``
     - ``.sys``
     - | 同一系统中的其他线程可见 
       | （CPU、其他 GPU）
     - L2+ 连接的缓存

:ref:`advanced-synchronization-primitives` 和 :ref:`asynchronous-data-copies` 演示了线程作用域的使用。

.. _advanced-synchronization-primitives:

3.2.4. 高级同步原语
-----------------------

本节介绍三类同步原语：

- :ref:`scoped-atomics`，将 C++ 内存顺序与 CUDA 的线程作用域相结合，从而能够安全地在块（block）、簇（cluster）、设备（device）或系统（system）作用域范围内的线程之间进行通信（参见 :ref:`thread-scopes` ）。
- :ref:`asynchronous-barriers`，将同步过程拆分为到达和等待阶段，并且可以用来追踪异步操作的进度。
- :ref:`pipelines` 能够将工作进行流水线化处理，并协调多缓冲区的生产者-消费者模式，通常被用来让计算与 :ref:`asynchronous-data-copies` 相互重叠。


.. _scoped-atomics:

3.2.4.1. 作用域原子操作
^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`atomic-functions` 概述了 CUDA 中可用的各类原子函数。
在本节中，我们将重点聚焦于支持 `C++ 标准原子内存语义 <https://en.cppreference.com/w/cpp/atomic/memory_order.html>`__ 的作用域原子操作（scoped atomics），
这些功能可通过 `libcu++ <https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives.html>`__ 库或编译器内置函数使用。
作用域原子操作提供了在 CUDA 线程层次结构的适当层级上进行高效同步的工具，从而确保了复杂并行算法的正确性与高性能。


.. _thread-scope-and-memory-ordering:

3.2.4.1.1. 线程作用域和内存顺序
"""""""""""""""""""""""""""""""""""""

作用域原子操作结合了两个关键概念：

- **线程作用域**：定义了哪些线程能够观察到该原子操作所产生的效果（参见 :ref:`thread-scopes` ）。
- **内存排序**：定义了相对于其他内存操作的顺序约束（参见 `C++ 标准原子内存语义 <https://en.cppreference.com/w/cpp/atomic/memory_order.html>`__）。

.. tab-set::

   .. tab-item:: CUDA C++ cuda::atomic
    
      .. code-block:: cuda

         #include <cuda/atomic>

         __global__ void block_scoped_counter() {
           // Shared atomic counter visible only within this block
           __shared__ cuda::atomic<int, cuda::thread_scope_block> counter;

           // Initialize counter (only one thread should do this)
           if (threadIdx.x == 0) {
             counter.store(0, cuda::memory_order_relaxed);
           }
           __syncthreads();

           // All threads in block atomically increment
           int old_value = counter.fetch_add(1, cuda::memory_order_relaxed);

           // Use old_value...
         }

   .. tab-item:: 内置原子函数

      .. code-block:: cuda

         __global__ void block_scoped_counter() {
           // Shared counter visible only within this block
           __shared__ int counter;

           // Initialize counter (only one thread should do this)
           if (threadIdx.x == 0) {
             __nv_atomic_store_n(&counter, 0,
                                 __NV_ATOMIC_RELAXED,
                                 __NV_THREAD_SCOPE_BLOCK);
           }
           __syncthreads();

           // All threads in block atomically increment
           int old_value = __nv_atomic_fetch_add(&counter, 1,
                                                 __NV_ATOMIC_RELAXED,
                                                 __NV_THREAD_SCOPE_BLOCK);

           // Use old_value...
         }


这个例子实现了一个块作用域（block-scoped）的原子计数器，它展示了作用域原子操作的基本概念：

- **共享变量**：使用 ``__shared__`` 内存， 让单个计数器在同一个线程块（block）的所有线程之间共享。
- **原子类型声明**： ``cuda::atomic<int, cuda::thread_scope_block>`` 创建了一个具有块级（block-level）可见性的原子整数。
- **单一初始化**：仅由 0 号线程对计数器进行初始化，以防止在设置阶段发生竞态条件。
- **块同步**： ``__syncthreads()`` 确保所有线程在继续执行之前，都能看到（读取到）已经初始化好的计数器。
- **原子递增**：每个线程对计数器进行原子递增操作，并获取递增之前的旧值。

这里选择 ``cuda::memory_order_relaxed`` 是因为我们只需要保证原子性（即不可分割的读取-修改-写入操作），而不需要针对不同内存位置施加额外的顺序约束。
由于这只是一个简单的计数操作，递增发生的先后顺序并不影响程序的正确性。


对于生产者-消费者模式，获取-释放（acquire-release）语义能够确保正确的执行顺序：

.. tab-set::

   .. tab-item:: CUDA C++ cuda::atomic

      .. code-block:: cuda 

         __global__ void producer_consumer() {
           __shared__ int data;
           __shared__ cuda::atomic<bool, cuda::thread_scope_block> ready;

           if (threadIdx.x == 0) {
             // Producer: write data then signal ready
             data = 42;
             ready.store(true, cuda::memory_order_release);  // Release ensures data write is visible
           } else {
             // Consumer: wait for ready signal then read data
             while (!ready.load(cuda::memory_order_acquire)) {  // Acquire ensures data read sees the write
             // spin wait
           }
           int value = data;
           // Process value...
         }

   .. tab-item:: 内置原子函数

      .. code-block:: cuda
         
         __global__ void producer_consumer() {
           __shared__ int data;
           __shared__ bool ready; // Only ready flag needs atomic operations

           if (threadIdx.x == 0) {
             // Producer: write data then signal ready
             data = 42;
             __nv_atomic_store_n(&ready, true,
                                 __NV_ATOMIC_RELEASE,  // Release ensures data write is visible
                                 __NV_THREAD_SCOPE_BLOCK);
           } else {
             // Consumer: wait for ready signal then read data
             while (!__nv_atomic_load_n(&ready,
                                        __NV_ATOMIC_ACQUIRE, // Acquire ensures data read sees the write
                                        __NV_THREAD_SCOPE_BLOCK)) {
               // spin wait
             }
             int value = data;
             // Process value...
           }
         }


.. _performance-considerations:

3.2.4.1.2. 性能考量
"""""""""""""""""""""""""

- 使用尽可能窄的作用域：块作用域原子操作比系统作用域原子操作快得多。
- 优先选择更宽松的内存顺序：只有在为了保证程序逻辑正确、不得不用的时候，才去使用更强的（更严格的）内存顺序。
- 注意内存位置：共享内存上的原子操作，速度要快于全局内存。

.. _asynchronous-barriers:

3.2.4.2. 异步屏障
^^^^^^^^^^^^^^^^^^^^^

异步屏障（asynchronous barrier）与典型的单阶段屏障（ ``__syncthreads()`` ）不同，它将线程通知自己到达屏障的操作（arrival），与等待其他线程到达屏障的操作（wait）分离开来。
这种分离提高了执行效率，因为它允许线程在等待期间去处理一些与屏障无关的额外任务，从而更高效地利用了原本只能干等的时间。
异步屏障可以用来在 CUDA 线程间实现生产者-消费者模式，或者通过让数据拷贝操作在完成时发出信号（arrive on），来启用存储层次结构内的异步数据拷贝。

异步屏障在计算能力 7.0 或更高的设备上可用。
计算能力 8.0 及以上的设备，不仅为共享内存中的异步屏障提供了硬件加速支持，还在同步粒度上实现了显著的飞跃——它支持线程块内任意 CUDA 线程子集同步的硬件加速。
相比之下，之前的架构只能在整 warp（ ``__syncwarp()`` ）或整线程块（ ``__syncthreads()`` ）级别上提供同步加速。

CUDA 编程模型通过 ``cuda::std::barrier`` 提供异步屏障，这是 `libcu++ <https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/barrier.html>`__ 库中一个符合 ISO C++ 标准的屏障实现。
除了实现标准的 `std::barrier <https://en.cppreference.com/w/cpp/thread/barrier.html>`__ 外，该库还提供了一些 CUDA 专属的扩展功能。
例如，允许用户指定屏障的线程作用域以提升性能，并开放了更底层的 `cuda::ptx <https://nvidia.github.io/cccl/libcudacxx/ptx_api.html>`__ API。
``cuda::barrier`` 可以通过使用 ``friend`` 函数 ``cuda::device::barrier_native_handle()`` 获取屏障的原生句柄（native handle），并将其传递给 ``cuda::ptx`` 的相关函数使用，以实现与 ``cuda::ptx`` 互操作。
此外，CUDA 还提供了一套 :ref:`原语 API <memory-barrier-primitives-interface>` 用于共享内存的线程块级别的异步屏障接口。


下表概述了可在不同线程作用域进行同步的异步屏障。

.. list-table:: 异步屏障概述
   :header-rows: 1
   :widths: 15 20 15 15 20 15

   * - 线程作用域
     - 内存位置
     - 到达屏障
     - 等待屏障
     - 硬件加速
     - CUDA API
   * - block
     - 本地共享内存
     - 允许
     - 允许
     - 是（8.0+）
     - ``cuda::barrier`` ``cuda::ptx`` ``primitives``
   * - cluster
     - 本地共享内存
     - 允许
     - 允许
     - 是（9.0+）
     - ``cuda::barrier`` ``cuda::ptx``
   * - cluster
     - 远程共享内存
     - 允许
     - 不允许
     - 是（9.0+）
     - ``cuda::barrier`` ``cuda::ptx``
   * - device
     - 全局内存
     - 允许
     - 允许
     - 否
     - ``cuda::barrier``
   * - system
     - 全局/统一内存
     - 允许
     - 允许
     - 否
     - ``cuda::barrier``

.. _advanced-kernels-sync-time-division:

3.2.4.2.1. 同步的时间分割
"""""""""""""""""""""""""""""""

如果没有异步的 arrive-wait（到达-等待）屏障机制，线程块内的同步通常是通过 ``__syncthreads()`` 来实现的；
而如果使用的是 :ref:`cooperative-groups` （协作组），则是通过 ``block.sync()`` 来达成。

.. code-block:: cuda

   #include <cooperative_groups.h>

   __global__ void simple_sync(int iteration_count) {
       auto block = cooperative_groups::this_thread_block();

       for (int i = 0; i < iteration_count; ++i) {
         /* code before arrive */

         // Wait for all threads to arrive here.
         block.sync();

         /* code after wait */
       }
   }

线程会在同步点（ ``block.sync()`` ）处被阻塞，直到所有线程都抵达该同步点。此外，在同步点之前发生的内存更新操作，能够保证在同步点之后对块内的所有线程可见。

此模式有三个阶段：

- 在同步点 **之前** 的代码所执行的内存更新操作，将会在同步点 **之后** 被读取。
- 同步点。
- 位于同步点 **之后** 的代码，能够访问（可见）在同步点 **之前** 发生的内存更新。

如果改用异步屏障，这种时间拆分（temporally-split ）的同步模式如下所示。

.. tab-set::

   .. tab-item:: CUDA C++ cuda::barrier

      .. code-block:: cuda

         #include <cuda/barrier>
         #include <cooperative_groups.h>

         __device__ void compute(float *data, int iteration);

         __global__ void split_arrive_wait(int iteration_count, float *data)
         {
           using barrier_t = cuda::barrier<cuda::thread_scope_block>;
           __shared__ barrier_t bar;
           auto block = cooperative_groups::this_thread_block();

           // 由块中的第0号线程负责初始化屏障，预期到达的线程数就是整个块的大小
           if (block.thread_rank() == 0)
           {
             // Initialize barrier with expected arrival count.
             init(&bar, block.size());
           }
           block.sync();  // 【传统同步】确保所有线程都看到屏障被正确初始化了

           for (int i = 0; i < iteration_count; ++i)
           {
             /* code before arrive */

             // 阶段二：线程举手签到（Arrival）
             // 调用 arrive() 告诉屏障“我到了”，但它不会阻塞线程！
             // 同时会拿到一个令牌（token），代表当前这一轮的签到凭证
             // This thread arrives. Arrival does not block a thread.
             barrier_t::arrival_token token = bar.arrive();

             compute(data, i);

             // 阶段三：拿着凭证等待其他人（Wait）
             // 只有当所有参与屏障的线程都完成了 arrive()，wait() 才会放行
             // Wait for all threads participating in the barrier to complete bar.arrive().
             bar.wait(std::move(token));

             /* code after wait */
           }
         }

   .. tab-item:: CUDA C++ cuda::ptx
          
      .. code-block:: cuda
         
         #include <cuda/ptx>
         #include <cooperative_groups.h>

         __device__ void compute(float *data, int iteration);

         __global__ void split_arrive_wait(int iteration_count, float *data)
         {
           __shared__ uint64_t bar;
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
             uint64_t token = cuda::ptx::mbarrier_arrive(&bar);

             compute(data, i);

             // Wait for all threads participating in the barrier to complete mbarrier_arrive().
             while(!cuda::ptx::mbarrier_try_wait(&bar, token)) {}

             /* code after wait */
           }
         }

   .. tab-item:: CUDA C 原语

      .. code-block:: cuda

         #include <cuda_awbarrier_primitives.h>
         #include <cooperative_groups.h>

         __device__ void compute(float *data, int iteration);

         __global__ void split_arrive_wait(int iteration_count, float *data)
         {
           __shared__ __mbarrier_t bar;
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
             __mbarrier_token_t token = __mbarrier_arrive(&bar);

             compute(data, i);

             // Wait for all threads participating in the barrier to complete __mbarrier_arrive().
             while(!__mbarrier_try_wait(&bar, token, 1000)) {}

             /* code after wait */
           }
         }

在这种模式下，同步点被拆分成了一个 **到达点** ``bar.arrive()`` 和一个 **等待点** ``bar.wait(std::move(token))`` 。
线程通过第一次调用 ``bar.arrive()`` 开始参与 ``cuda::barrier`` 的同步。
当线程调用 ``bar.wait(std::move(token))`` 时，它会被阻塞，直到所有参与的线程都完成了预期次数的 ``bar.arrive()`` 调用（这个次数就是初始化 ``init()`` 时传入的预期到达计数参数）。
参与同步的线程在调用 ``bar.arrive()`` 之前发生的内存更新，能够保证在其他参与线程调用 ``bar.wait(std::move(token))`` 之后对它们可见。
需要注意的是，调用 ``bar.arrive()`` 并不会阻塞线程，它可以继续去执行其他不依赖于其他线程在 ``bar.arrive()`` 之前所做内存更新的工作。

到达与等待（arrive and wait）模式包含以下五个阶段：

- 位于 arrive 之前的代码： 执行内存更新操作，这些数据将会在 wait 之后被读取。
- 到达点（Arrive point）： 此处带有隐式的内存栅栏（即等同于执行了 ``cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block)`` ）。
- 位于 arrive 和 wait 之间的代码。
- 等待点（Wait point）。
- 位于 wait 之后的代码：此时能够访问（可见）在 arrive 之前所执行的内存更新。

关于如何使用异步屏障的完整指南，请参见 :ref:`async-barriers-details`。

.. _pipelines:

3.2.4.3. 流水线
^^^^^^^^^^^^^^^^^^^

CUDA 编程模型提供了流水线（pipeline）这一同步对象，作为一种协调机制，用于将异步内存拷贝按顺序划分为多个阶段，从而方便实现双缓冲或多缓冲的生产者-消费者模式。
流水线本质上是一个带有首尾两端的双端队列，它按照先进先出（FIFO）的顺序来处理工作。
生产者线程将任务提交到流水线的头部，而消费者线程则从流水线的尾部拉取任务。

流水线通过 `libcu++ <https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/pipeline.html>`__ 库中的 ``cuda::pipeline`` 接口，
以及 :ref:`原语 API <pipeline-primitives-interface>` 对外提供。接下来的表格将分别描述这两种 API 的主要功能。

.. list-table:: cuda::pipeline API
   :header-rows: 1
   :widths: 35 65

   * - ``cuda::pipeline``
     - 描述
   * - ``producer_acquire``
     - 获取流水线内部队列中一个可用的阶段。
   * - ``producer_commit``
     - 提交在当前获取的流水线阶段上，在 ``producer_acquire`` 调用之后发出的异步操作。
   * - ``consumer_wait``
     - 等待流水线中最旧的（即最先进入的）那个阶段里的异步操作执行完毕。
   * - ``consumer_release``
     - 将流水线中最旧的阶段释放回流水线对象中，以便重复利用。 被释放的阶段随后就可以被生产者重新获取了。

.. list-table:: 原语 API
   :header-rows: 1
   :widths: 40 60

   * - 原语 API
     - 描述
   * - ``__pipeline_memcpy_async``
     - 请求将数据从全局内存拷贝到共享内存，并提交异步执行。
   * - ``__pipeline_commit``
     - 提交在当前流水线阶段中、于本次调用之前所发起的异步操作。
   * - ``__pipeline_wait_prior(N)``
     - 等待流水线中除了最后 N 次提交之外的所有异步操作执行完毕。

``cuda::pipeline`` 拥有更丰富的接口且限制更少； 而原语（primitives） 接口仅支持追踪从全局内存到共享内存的异步拷贝，并且对数据的大小和对齐方式有特定的要求。
原语接口所提供的功能，等同于一个带有 ``cuda::thread_scope_thread`` 作用域的 ``cuda::pipeline`` 对象。

关于详细的使用模式和示例，请参见 :ref:`pipelines-details`。

.. _asynchronous-data-copies:

3.2.5. 异步数据拷贝
-----------------------

在不同内存层级之间高效地移动数据，是实现 GPU 高性能计算的基石。
传统的同步内存操作会强制线程在数据传输期间处于空闲等待状态。
虽然 GPU 天生就能通过并行性来掩盖内存延迟（也就是说，当内存操作正在进行时， SM 可以切换去执行另一个 warp），
但即便有了这种并行带来的延迟掩盖，内存延迟依然有可能成为限制内存带宽利用率和计算资源效率的瓶颈。
为了解决这些问题，现代 GPU 提供了硬件加速的异步数据拷贝机制，使得内存传输可以独立进行，而线程则能继续执行其他工作。

异步数据拷贝将内存传输的发起与等待完成这两个步骤解耦开来，从而实现了计算与数据移动的并行重叠。
这样一来，线程就能在原本需要等待内存延迟的空档期里去做有意义的工作（比如执行计算任务），进而显著提升整体的吞吐量和资源利用率。

.. note::

   虽然本节背后的概念和原理与之前 :ref:`asynchronous-execution` 的章节中讨论的类似，但那个章节涵盖的是核函数和内存传输（如 ``cudaMemcpyAsync`` 这类内存传输操作）的异步执行。
   这可以被视为应用程序中不同组件之间的异步。

   而本节所描述的异步，指的是在不阻塞 GPU 线程的情况下，实现数据在 GPU 显存（即全局内存）与 SM 片上内存（如共享内存或张量内存）之间的传输。
   这是单个核函数内的异步性。

为了理解异步拷贝是如何提升性能的，我们可以先来看一个常见的 GPU 计算模式。CUDA 应用程序经常会采用一种拷贝与计算的模式：

- 从全局内存读取数据，
- 将数据存储到共享内存，以及
- 对共享内存数据执行计算，并可能将结果写回全局内存。

这种模式中的拷贝阶段通常表现为 ``shared[local_idx] = global[global_idx]`` 。
编译器会将这种从全局内存到共享内存的拷贝操作展开为两步：先从全局内存读取数据到一个寄存器，然后再将该数据从寄存器写入到共享内存中。

当这种模式出现在迭代算法中时，每个线程块都需要在执行完 ``shared[local_idx] = global[global_idx]`` 赋值后进行同步，以确保在计算阶段开始之前，所有对共享内存的写入操作都已经完成。
此外，线程块在计算阶段结束后还需要再次进行同步，以防止在所有线程完成各自的计算之前，共享内存就被（下一轮的数据）覆盖。
下面的代码片段就展示了这种模式。

.. code-block:: cuda

   #include <cooperative_groups.h>

   __device__ void compute(int* global_out, int const* shared_in) {
     // Computes using all values of current batch from shared memory.
     // Stores this thread's result back to global memory.
   }

   __global__ void without_async_copy(int* global_out, 
                                      int const* global_in, 
                                      size_t size, 
                                      size_t batch_sz) {
     auto grid = cooperative_groups::this_grid();
     auto block = cooperative_groups::this_thread_block();
     assert(size == batch_sz * grid.size()); // Exposition: input size fits batch_sz * grid_size

     extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

     size_t local_idx = block.thread_rank();

     for (size_t batch = 0; batch < batch_sz; ++batch) {
       // Compute the index of the current batch for this block in global memory.
       size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
       size_t global_idx = block_batch_idx + threadIdx.x;
       shared[local_idx] = global_in[global_idx];

       // Wait for all copies to complete.
       block.sync();

       // Compute and write result to global memory.
       compute(global_out + block_batch_idx, shared);

       // Wait for compute using shared memory to finish.
       block.sync();
     }
   }

通过异步数据拷贝，从全局内存到共享内存的数据移动可以异步进行，从而在等待数据到达的过程中，让 SM 得到更高效的利用。

.. code-block:: cuda

   #include <cooperative_groups.h>
   #include <cooperative_groups/memcpy_async.h>

   __device__ void compute(int* global_out, int const* shared_in) {
     // Computes using all values of current batch from shared memory.
     // Stores this thread's result back to global memory.
   }

   __global__ void with_async_copy(int* global_out, 
                                   int const* global_in, 
                                   size_t size, 
                                   size_t batch_sz) {
     auto grid = cooperative_groups::this_grid();
     auto block = cooperative_groups::this_thread_block();
     assert(size == batch_sz * grid.size()); // Exposition: input size fits batch_sz * grid_size

     extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

     size_t local_idx = block.thread_rank();

     for (size_t batch = 0; batch < batch_sz; ++batch) {
       // Compute the index of the current batch for this block in global memory.
       size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;

       // Whole thread-group cooperatively copies whole batch to shared memory.
       cooperative_groups::memcpy_async(block, shared, global_in + block_batch_idx, block.size());

       // Compute on different data while waiting.

       // Wait for all copies to complete.
       cooperative_groups::wait(block);

       // Compute and write result to global memory.
       compute(global_out + block_batch_idx, shared);

       // Wait for compute using shared memory to finish.
       block.sync();
     }
   }

:ref:`cooperative_groups::memcpy_async <cooperative-groups-async-h>` 函数会将 ``block.size()`` 个元素从全局内存拷贝到共享内存中。
这个操作就像是由另一个独立的线程在执行一样，它会在拷贝完成后，与当前线程调用的 :ref:`cooperative_groups::wait <wait-and-wait-prior>` 进行同步。
在拷贝操作完成之前，如果修改全局数据，或者读取、写入共享数据，都会引发数据竞争。

这个例子揭示了所有异步拷贝操作背后的核心概念：它们将内存传输的发起与完成解耦开来，使得线程在数据移动的同时，能够去执行其他工作。
CUDA 编程模型提供了多种 API 来调用这些功能，包括 :ref:`Cooperative Groups <memcpy-async>` 和 `libcu++ <https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html>`__ 库中提供的 ``memcpy_async`` 函数，
以及更底层的 ``cuda::ptx`` 和 primitives（原语）API。
这些 API 拥有相似的语义：它们都会将对象从源地址拷贝到目标地址，其过程就像是由另一个线程在执行一样；当拷贝完成后，可以通过不同的完成机制来进行同步。

现代 GPU 架构提供了多种用于异步数据移动的硬件机制。

- LDGSTS（计算能力 8.0 及以上）支持从全局内存到共享内存的小规模的高效异步传输
- 张量内存加速器（TMA，计算能力 9.0 及以上）进一步扩展了这些功能，针对大规模多维数据传输提供了优化的批量异步拷贝操作。
- STAS 指令（计算能力 9.0 及以上）则能够实现从寄存器到集群内分布式共享内存的小规模的异步传输。

这些机制支持不同的数据路径、传输大小以及对齐要求，开发者可以根据自己具体的数据访问模式来选择最合适的方法。
下面的表格概述了 GPU 内部异步拷贝所支持的数据路径

.. list-table:: 异步拷贝可能的源和目标内存空间（空单元格表示不支持源-目标对）
   :header-rows: 1
   :name: tbl:async-source-dest-state-spaces

   * - 方向
     - 
     - 拷贝机制
     - 
   * - 源
     - 目的
     - 异步拷贝
     - 批量异步拷贝
   * - global
     - global
     - 
     - 
   * - shared::cta
     - global
     - 
     - 支持（TMA, 9.0+）
   * - global
     - shared::cta
     - 支持（LDGSTS, 8.0+）
     - 支持（TMA, 9.0+）
   * - global
     - shared::cluster
     - 
     - 支持（TMA, 9.0+）
   * - shared::cluster
     - shared::cta
     - 
     - 支持（TMA, 9.0+）
   * - shared::cta
     - shared::cta
     - 
     - 
   * - registers
     - shared::cluster
     - 支持（STAS, 9.0+）
     - 

.. note::

  CTA （Cooperative Thread Array） 是 CUDA 官方文档中对线程块（Thread Block）的正式称呼。
  一个 CTA 里的所有线程会被调度到同一个 SM 上协同工作。
  因此， ``shared::cta`` 就是属于当前线程块的共享内存空间。

  ``shared::cluster`` 作用域扩大到了整个线程块集群（Cluster）内所有线程块共享的分布式共享内存（Distributed Shared Memory）。

  （来源： 千问）

:ref:`使用 LDGSTS <using-ldgsts>`、 :ref:`使用张量内存加速器（TMA） <using-tma>` 和 :ref:`使用 STAS <using-stas>` 部分将详细介绍每种机制。

.. _configuring-l1-shared-memory-balance:

3.2.6. 配置 L1/共享内存平衡
--------------------------------

正如在 :ref:`writing-cuda-kernels-caches` 所提到的，SM 上的 L1 缓存和共享内存使用的是同一块物理资源，即统一数据缓存。
在大多数 GPU 上，如果一个 kernel 很少使用或者根本不使用共享内存，那么这块统一数据缓存就可以被配置为提供 GPU 该架构所允许的最大容量的 L1 缓存。

可以针对每个 kernel 单独设置预留用于共享内存的统一数据缓存。
应用程序可以通过在启动 kernel 之前调用 `cudaFuncSetAttribute <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g317e77d2657abf915fd9ed03e75f3eb0>`__ 函数，来设置 ``carveout`` 即期望的共享内存容量（百分比）。

.. code-block:: cuda

   cudaFuncSetAttribute(kernel_name, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);


应用程序可以将 ``carveout`` 设置为该架构支持的最大共享内存容量的整数百分比。
除了整数百分比外，还提供了三个便捷枚举作为 carveout 值。

应用程序可以将其设置为该架构支持的最大共享内存容量的整数百分比。
除了具体的整数百分比之外，CUDA 还提供了三个便捷的枚举值作为 ``carveout`` 的预设选项。

- ``cudaSharedmemCarveoutDefault`` 默认分区。也就是使用 CUDA 编译器或驱动程序为你自动选择的默认分配比例（通常是一个比较均衡的设置）
- ``cudaSharedmemCarveoutMaxL1`` 最大化 L1 缓存。这个选项会尽量把统一数据缓存的大头都划拨给 L1 缓存使用，只保留架构所允许的最小共享内存容量。如果你的程序主要依赖全局内存读取，不怎么需要线程块内部频繁交换数据，选这个准没错。
- ``cudaSharedmemCarveoutMaxShared`` 最大化共享内存。和上面正好相反，这个选项会把尽可能多的空间划拨给共享内存，只保留最小的 L1 缓存。如果你的内核计算非常复杂，且极度依赖共享内存来做高速数据中转（比如做矩阵分块计算时），这个就是你的最佳拍档。

不同架构所支持的最大共享内存容量以及支持的 ``carveout`` 大小各不相同，具体细节请查阅 :ref:`compute-capabilities-table-shared-memory-capacity-per-compute-capability`。

如果设定的整数百分比无法精确对应到某个受支持的共享内存容量，系统就会采用下一个更大的容量。
例如，对于计算能力 12.0 的设备，其最大共享内存容量为 100KB。
如果将 ``carveout`` 设置为 50%，最终得到的共享内存将是 64KB，而不是 50KB。
这是因为计算能力为 12.0 的设备只支持 0、8、16、32、64 和 100 KB 这几种固定的共享内存大小。

传递给 ``cudaFuncSetAttribute`` 的函数必须使用 ``__global__`` 说明符进行声明。
此外，驱动程序会将 ``cudaFuncSetAttribute`` 的设置视为一种提示，如果为了使 kernel 成功执行，必要时驱动可能会选择一个不同的 ``carveout`` 大小（即不是设置值）。

.. note::

   ``cudaFuncSetCacheConfig`` 也允许应用程序调整核函数 L1 缓存和共享内存之间的平衡。
   但是，这个 API 会对 kernel 启动时的共享内存/L1 比例设定硬性要求。
   因此，如果交替启动具有不同共享内存配置的 kernel ，就会因为需要重新配置共享内存比例而导致这些启动操作被 :ref:`串行化 <implicit-synchronization>` 。
   相比之下， ``cudaFuncSetAttribute`` 是更推荐的选择，因为驱动程序可以根据需要自动选择不同的配置，避免 **thrashing** （资源颠簸，即系统花费了大量时间在来回修改配置，切换状态上）。

如果核函数要求每个线程块拥有超过 48 KB 共享内存，这需要 GPU 架构支持（旧架构不支持）。
因此， 它们必须使用 :ref:`动态共享内存 <writing-cuda-kernels-dynamic-allocation-shared-memory>`，而不能使用静态大小的数组。
并且必须显式调用 ``cudaFuncSetAttribute`` 主动授权（及通知驱动，就需要这么大的共享内存），具体如下：

.. code-block:: cuda

   // Device code
   __global__ void MyKernel(...)
   {
     extern __shared__ float buffer[];
     ...
   }

   // Host code
   int maxbytes = 98304; // 96 KB
   cudaFuncSetAttribute(MyKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
   MyKernel <<<gridDim, blockDim, maxbytes>>>(...);

.. note::

  早期的 NVIDIA 显卡架构（比如计算能力低于 7.0 的 Pascal、Maxwell 等老架构），它们的 L1 缓存和共享内存在物理上是严格分开的，单块共享内存的上限就是锁死在 48KB，根本没有办法通过软件指令去扩充。
  只有到了较新的架构（如 Volta sm_70、Ampere sm_80、Hopper sm_90 及以后的 Blackwell 等），硬件才支持将 L1 缓存的空间动态划拨给共享内存使用。
  因此，共享内存大于 48KB 是这些新架构的特性。
  即使新硬件支持，但不是默认开启的，需要使用 ``cudaFuncSetAttribute`` 主动授权（explicit opt-in）。
  否则，即使架构支持，驱动依然会报错。
  
  （来源： 千问）

