.. _advanced-kernel-programming:

3.2. 高级核函数编程
=======================

本章首先深入探讨 NVIDIA GPU 的硬件模型，然后介绍 CUDA 核函数代码中一些旨在提高核函数性能的高级特性。
本章将介绍与线程作用域、异步执行以及相关同步原语相关的一些概念。这些概念性讨论为核函数中可用的一些高级性能特性提供了必要的基础。

本编程指南第四部分专门介绍了其中一些特性的详细描述。

- 本章介绍的 :ref:`高级同步原语<advanced-synchronization-primitives>` 在 :ref:`async-barriers-details` 和 :ref:`pipelines-details` 中有完整介绍。
- 本章介绍的 :ref:`异步数据拷贝<asynchronous-data-copies>` ，包括张量内存加速器 （ tensor memory accelerator, TMA）在 :ref:`async-copies-details` 中有完整介绍。

.. _using-ptx:

3.2.1. 使用 PTX
-------------------

并行线程执行（Parallel Thread Execution，PTX）是 CUDA 用来抽象硬件 ISA 的虚拟机指令集架构（ISA），在 :ref:`parallel-thread-execution-ptx` 中已介绍。
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
指令采用流水线方式，利用单线程内的指令级并行性，以及通过同时硬件多线程实现的广泛线程级并行性，详见 :ref:`hardware-multithreading` 。
与 CPU 核心不同，SM 按顺序发射指令，不进行分支预测或投机执行。

:ref:`simt-execution-model` 和 :ref:`hardware-multithreading` 描述了 SM 中所有设备共有的架构特性。
:ref:`compute-capabilities` 提供了不同计算能力设备的详细信息。

NVIDIA GPU 架构使用小端表示法。

.. _simt-execution-model:

3.2.2.1. SIMT 执行模型
^^^^^^^^^^^^^^^^^^^^^^^

每个 SM 创建、管理、调度和执行称为 *warp* 的 32 个并行线程组。
组成 warp 的各个线程从相同的程序地址一起开始，但它们有自己的指令地址计数器和寄存器状态，因此可以自由地分支和独立执行。
*warp* 一词源于编织，这是第一个并行线程技术。 *half-warp* 是 warp 的前半部分或后半部分。
*quarter-warp* 是 warp 的第一、第二、第三或第四个四分之一。

一个 warp 一次执行一条公共指令，因此当 warp 的所有 32 个线程在执行路径上一致时，可以实现完全效率。
如果 warp 的线程通过数据相关的条件分支发生分歧，warp 会执行所采用的每条分支路径，禁用不在该路径上的线程。
分支分歧只发生在 warp 内部；不同的 warp 独立执行，无论它们执行的是公共还是不相交的代码路径。

SIMT 架构类似于 SIMD（单指令多数据）向量组织，因为单条指令控制多个处理元素。
一个关键区别是 SIMD 向量组织将 SIMD 宽度暴露给软件，而 SIMT 指令指定单个线程的执行和分支行为。
与 SIMD 向量机相比，SIMT 使程序员能够为独立的标量线程编写线程级并行代码，以及为协调线程编写数据并行代码。
为了正确性，程序员基本上可以忽略 SIMT 行为；然而，通过注意代码很少需要 warp 中的线程发生分歧，可以实现显著的性能改进。
在实践中，这类似于缓存行的作用：在设计正确性时可以安全地忽略缓存行大小，但在设计峰值性能时必须在代码结构中考虑它。
另一方面，向量架构要求软件将加载合并为向量并手动管理分歧。

.. _independent-thread-scheduling:

3.2.2.1.1. 独立线程调度
"""""""""""""""""""""""""""""""

在计算能力低于 7.0 的 GPU 上，warp 使用所有 32 个线程共享的单个程序计数器以及指定 warp 活动线程的活动掩码。
因此，来自同一 warp 的线程在分歧区域或不同执行状态时无法相互发信号或交换数据，需要锁或互斥锁保护的细粒度数据共享的算法可能导致死锁，这取决于竞争线程来自哪个 warp。

在计算能力 7.0 及更高的 GPU 中，「独立线程调度」允许线程之间完全并发，无论 warp 如何。
通过独立线程调度，GPU 维护每个线程的执行状态，包括程序计数器和调用栈，并且可以以每个线程的粒度让出执行，要么是为了更好地利用执行资源，要么是为了允许一个线程等待另一个线程产生的数据。
调度优化器确定如何将同一 warp 的活动线程分组为 SIMT 单元。
这保留了先前 NVIDIA GPU 中 SIMT 执行的高吞吐量，但具有更大的灵活性：线程现在可以在子 warp 粒度上分歧和重新汇聚。

独立线程调度可能会破坏依赖先前 GPU 架构隐式 warp 同步行为的代码。
「Warp 同步」代码假设同一 warp 中的线程在每条指令上都以锁步方式执行，但线程在子 warp 粒度上分歧和重新汇聚的能力使这种假设无效。
这可能导致与预期不同的线程集参与执行代码。任何为 CC 7.0 之前的 GPU 开发的 warp 同步代码（如无同步的 warp 内归约）都应该重新审视以确保兼容性。
开发者应该使用 ``__syncwarp()`` 显式同步此类代码，以确保在所有 GPU 代际上的正确行为。

.. _simt-architecture-notes:

.. note::

   参与当前指令的 warp 线程称为「活动」线程，而不在当前指令上的线程是「非活动」（禁用）线程。
   线程可能因多种原因而变为非活动，包括比其 warp 的其他线程更早退出、采用了与 warp 当前执行的分支路径不同的分支路径，或者是线程数不是 warp 大小倍数的块的最后线程。

   如果 warp 执行的非原子指令从多个线程写入全局或共享内存中的同一位置，则发生的对该位置的序列化写入次数可能因设备的计算能力而异。
   但是，对于所有计算能力，哪个线程执行最终写入是未定义的。

   如果 warp 执行的 :ref:`原子<atomic-functions>` 指令从多个线程读取、修改并写入全局内存中的同一位置，
   则对该位置的每次读取/修改/写入都会发生，并且它们都被序列化，但它们发生的顺序是未定义的。

.. _hardware-multithreading:

3.2.2.2. 硬件多线程
^^^^^^^^^^^^^^^^^^^^^^^

当 SM 被赋予一个或多个线程块来执行时，它将它们划分为 warp，每个 warp 由「warp 调度器」调度执行。
块划分为 warp 的方式总是相同的；每个 warp 包含连续递增线程 ID 的线程，
第一个 warp 包含线程 0。:ref:`writing-cuda-kernels-thread-hierarchy-review` 描述了线程 ID 与块中线程索引的关系。

块中 warp 的总数定义如下：

:math:`\text{ceil}\left( \frac{T}{W_{size}}, 1 \right)`

- *T* 是每块的线程数，
- *Wsize* 是 warp 大小，等于 32，
- ceil(x, y) 等于 x 向上舍入到 y 的最近倍数。

.. figure:: /_static/images/warps-in-a-block.png
   :alt: 线程块被划分为 32 个线程的 warp
   :figwidth: 80%

   线程块被划分为 32 个线程的 warp

SM 处理的每个 warp 的执行上下文（程序计数器、寄存器等）在 warp 的整个生命周期内都保持在片上。
因此，在 warp 之间切换没有成本。在每个指令发射周期，warp 调度器选择一个有线程准备好执行其下一条指令的 warp（warp 的 :ref:`活动线程<simt-architecture-notes>` ），并将指令发射给这些线程。

每个 SM 有一组 32 位寄存器，在 warp 之间分区，以及一个在线程块之间分区的 :ref:`共享内存<writing-cuda-kernels-shared-memory>`。
对于给定核函数，可以在 SM 上驻留并发处理的块和 warp 的数量取决于核函数使用的寄存器和共享内存量，以及 SM 上可用的寄存器和共享内存量。
每个 SM 还有最大驻留块数和 warp 数的限制。这些限制以及 SM 上可用的寄存器和共享内存量取决于设备的计算能力，并在 :ref:`compute-capabilities` 中指定。
如果每个 SM 没有足够的资源来处理至少一个块，核函数将无法启动。
为块分配的寄存器和共享内存总量可以通过 :ref:`writing-cuda-kernels-kernel-launch-and-occupancy` 部分中记录的几种方式确定。

.. _asynchronous-execution-features:

3.2.2.3. 异步执行特性
^^^^^^^^^^^^^^^^^^^^^^^^^

最近的 NVIDIA GPU 代际包含了异步执行能力，以允许 GPU 内更多的数据移动、计算和同步重叠。
这些能力使从 GPU 代码调用的某些操作能够与同一线程块中的其他 GPU 代码异步执行。
这种异步执行不应与 :ref:`asynchronous-execution` 中讨论的异步 CUDA API 混淆，后者使 GPU 核函数启动或内存操作能够彼此或与 CPU 异步运行。

计算能力 8.0（NVIDIA Ampere GPU 架构）引入了从全局内存到共享内存的硬件加速异步数据拷贝和异步屏障（参见 `NVIDIA A100 Tensor Core GPU 架构 <https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf>`__）。

计算能力 9.0（NVIDIA Hopper GPU 架构）通过 :ref:`Tensor Memory Accelerator (TMA)<asynchronous-data-copies>` 单元扩展了异步执行特性，
该单元可以在全局内存和共享内存之间传输大数据块和多维张量，
以及异步事务屏障和异步矩阵乘累加操作（详见 `Hopper 架构深入 <https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/>`__ 博客文章）。

CUDA 提供可从设备代码由线程调用的 API 来使用这些特性。异步编程模型定义了异步操作相对于 CUDA 线程的行为。

异步操作是由 CUDA 线程发起但异步执行的操作，就像由另一个线程执行一样，我们将这个线程称为「异步线程」。
在格式良好的程序中，一个或多个 CUDA 线程与异步操作同步。
发起异步操作的 CUDA 线程不一定是同步线程之一。异步线程始终与发起操作的 CUDA 线程相关联。

异步操作使用同步对象来表示其完成，同步对象可以是屏障或流水线。
这些同步对象在 :ref:`advanced-synchronization-primitives` 中详细解释，它们在执行异步内存操作中的作用在 :ref:`asynchronous-data-copies` 中演示。

.. _async-thread-and-async-proxy:

3.2.2.3.1. 异步线程和异步代理
""""""""""""""""""""""""""""""""""

异步操作访问内存的方式可能与常规操作不同。
为了区分这些不同的内存访问方法，CUDA 引入了「异步线程」、「通用代理」和「异步代理」的概念。正常操作（加载和存储）通过通用代理进行。
一些异步指令，如 :ref:`LDGSTS<using-ldgsts>` 和 :ref:`STAS/REDAS <using-stas>` ，使用在通用代理中操作的异步线程建模。
其他异步指令，如使用 TMA 的批量异步拷贝和一些张量核心操作（tcgen05.\*, wgmma.mma\_async.\*），使用在异步代理中操作的异步线程建模。

**在通用代理中操作的异步线程**。当发起异步操作时，它与一个异步线程相关联，该线程与发起操作的 CUDA 线程不同。
同一地址的「先前」通用代理（正常）加载和存储保证在异步操作之前排序。
但是，同一地址的「后续」正常加载和存储不保证保持其顺序，可能在异步线程完成之前导致竞争条件。

**在异步代理中操作的异步线程**。当发起异步操作时，它与一个异步线程相关联，该线程与发起操作的 CUDA 线程不同。
同一地址的「先前和后续」正常加载和存储不保证保持其顺序。
需要代理栅栏来跨不同代理同步它们，以确保正确的内存排序。:ref:`using-tma` 演示了使用代理栅栏确保使用 TMA 执行异步拷贝时的正确性。

有关这些概念的更多详细信息，请参阅 `PTX ISA <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=proxy#proxies>`__ 文档。

.. _thread-scopes:

3.2.3. 线程作用域
---------------------

CUDA 线程形成 :ref:`writing-cuda-kernels-thread-hierarchy-review`，使用此层次结构对于编写正确和高性能的 CUDA 核函数至关重要。
在此层次结构中，内存操作的可见性和同步作用域可能会有所不同。
为了解决这种非统一性，CUDA 编程模型引入了「线程作用域」的概念。
线程作用域定义哪些线程可以观察线程的加载和存储，并指定哪些线程可以使用原子操作和屏障等同步原语彼此同步。
每个作用域在内存层次结构中都有一个关联的一致性点。

线程作用域在 `CUDA PTX <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=thread%2520scopes#scope>`__ 中公开，
也可作为 `libcu++ <https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes>`__ 库中的扩展使用。下表定义了可用的线程作用域：

.. list-table:: 线程作用域
   :header-rows: 1
   :widths: 25 25 25 25

   * - CUDA C++ 线程作用域
     - CUDA PTX 线程作用域
     - 描述
     - 内存层次结构中的一致性点
   * - ``cuda::thread_scope_thread``
     - –
     - 内存操作仅对本地线程可见。
     - –
   * - ``cuda::thread_scope_block``
     - ``.cta``
     - 内存操作对同一线程块中的其他线程可见。
     - L1
   * - ``.cluster``
     - ``.cluster``
     - 内存操作对同一线程块集群中的其他线程可见。
     - L2
   * - ``cuda::thread_scope_device``
     - ``.gpu``
     - 内存操作对同一 GPU 设备中的其他线程可见。
     - L2
   * - ``cuda::thread_scope_system``
     - ``.sys``
     - 内存操作对同一系统中的其他线程可见（CPU、其他 GPU）。
     - L2 + 连接的缓存

:ref:`advanced-synchronization-primitives` 和 :ref:`asynchronous-data-copies` 演示了线程作用域的使用。

.. _advanced-synchronization-primitives:

3.2.4. 高级同步原语
-----------------------

本节介绍三类同步原语：

- :ref:`scoped-atomics`，将 C++ 内存排序与 CUDA 线程作用域配对，以在块、集群、设备或系统作用域安全地跨线程通信（参见 :ref:`thread-scopes` ）。
- :ref:`asynchronous-barriers`，将同步分为到达和等待阶段，可用于跟踪异步操作的进度。
- :ref:`pipelines`，将工作分阶段并协调多缓冲生产者-消费者模式，通常用于将计算与 :ref:`asynchronous-data-copies` 重叠。

.. _scoped-atomics:

3.2.4.1. 作用域原子操作
^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`atomic-functions` 概述了 CUDA 中可用的原子函数。
在本节中，我们将重点介绍支持 `C++ 标准原子内存 <https://en.cppreference.com/w/cpp/atomic/memory_order.html>`__ 语义的「作用域」原子操作，
可通过 `libcu++ <https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives.html>`__ 库或编译器内置函数使用。
作用域原子操作为在 CUDA 线程层次结构的适当级别进行高效同步提供了工具，在复杂并行算法中实现正确性和性能。

.. _thread-scope-and-memory-ordering:

3.2.4.1.1. 线程作用域和内存排序
"""""""""""""""""""""""""""""""""""""

作用域原子操作结合了两个关键概念：

- **线程作用域**：定义哪些线程可以观察原子操作的效果（参见 :ref:`thread-scopes` ）。
- **内存排序**：定义相对于其他内存操作的排序约束（参见 `C++ 标准原子内存语义 <https://en.cppreference.com/w/cpp/atomic/memory_order.html>`__）。

.. code-block:: cuda
   :caption: CUDA C++ cuda::atomic

   #include <cuda/atomic>

   __global__ void block_scoped_counter() {
       // 共享原子计数器仅在此块内可见
       __shared__ cuda::atomic<int, cuda::thread_scope_block> counter;

       // 初始化计数器（只有一个线程应该执行此操作）
       if (threadIdx.x == 0) {
           counter.store(0, cuda::memory_order_relaxed);
       }
       __syncthreads();

       // 块中所有线程原子递增
       int old_value = counter.fetch_add(1, cuda::memory_order_relaxed);

       // 使用 old_value...
   }

.. code-block:: cuda
   :caption: 内置原子函数

   __global__ void block_scoped_counter() {
       // 共享计数器仅在此块内可见
       __shared__ int counter;

       // 初始化计数器（只有一个线程应该执行此操作）
       if (threadIdx.x == 0) {
           __nv_atomic_store_n(&counter, 0,
                               __NV_ATOMIC_RELAXED,
                               __NV_THREAD_SCOPE_BLOCK);
       }
       __syncthreads();

       // 块中所有线程原子递增
       int old_value = __nv_atomic_fetch_add(&counter, 1,
                                             __NV_ATOMIC_RELAXED,
                                             __NV_THREAD_SCOPE_BLOCK);

       // 使用 old_value...
   }

此示例实现了一个「块作用域原子计数器」，演示了作用域原子操作的基本概念：

- **共享变量**：使用 ``__shared__`` 内存在线程块中的所有线程之间共享单个计数器。
- **原子类型声明**： ``cuda::atomic<int, cuda::thread_scope_block>`` 创建一个具有块级可见性的原子整数。
- **单一初始化**：只有线程 0 初始化计数器以防止设置期间的竞争条件。
- **块同步**： ``__syncthreads()`` 确保所有线程在继续之前看到初始化的计数器。
- **原子递增**：每个线程原子地递增计数器并接收先前的值。

这里选择 ``cuda::memory_order_relaxed`` 是因为我们只需要原子性（不可分割的读取-修改-写入），而不需要不同内存位置之间的排序约束。
由于这是一个简单的计数操作，递增的顺序对正确性没有影响。

对于生产者-消费者模式，获取-释放语义确保正确的排序：

.. code-block:: cuda
   :caption: CUDA C++ cuda::atomic

   __global__ void producer_consumer() {
       __shared__ int data;
       __shared__ cuda::atomic<bool, cuda::thread_scope_block> ready;

       if (threadIdx.x == 0) {
           // 生产者：写入数据然后发出就绪信号
           data = 42;
           ready.store(true, cuda::memory_order_release);  // Release 确保数据写入可见
       } else {
           // 消费者：等待就绪信号然后读取数据
           while (!ready.load(cuda::memory_order_acquire)) {  // Acquire 确保数据读取看到写入
               // 自旋等待
           }
           int value = data;
           // 处理 value...
       }
   }

.. code-block:: cuda
   :caption: 内置原子函数

   __global__ void producer_consumer() {
       __shared__ int data;
       __shared__ bool ready; // 只有 ready 标志需要原子操作

       if (threadIdx.x == 0) {
           // 生产者：写入数据然后发出就绪信号
           data = 42;
           __nv_atomic_store_n(&ready, true,
                               __NV_ATOMIC_RELEASE,
                               __NV_THREAD_SCOPE_BLOCK);  // Release 确保数据写入可见
       } else {
           // 消费者：等待就绪信号然后读取数据
           while (!__nv_atomic_load_n(&ready,
                                      __NV_ATOMIC_ACQUIRE,
                                      __NV_THREAD_SCOPE_BLOCK)) {  // Acquire 确保数据读取看到写入
               // 自旋等待
           }
           int value = data;
           // 处理 value...
       }
   }

.. _performance-considerations:

3.2.4.1.2. 性能考虑
"""""""""""""""""""""""""

- 使用尽可能窄的作用域：块作用域原子操作比系统作用域原子操作快得多。
- 优先使用较弱的排序：仅在正确性需要时才使用较强的排序。
- 考虑内存位置：共享内存原子操作比全局内存原子操作更快。

.. _asynchronous-barriers:

3.2.4.2. 异步屏障
^^^^^^^^^^^^^^^^^^^^^

异步屏障与典型的单阶段屏障（ ``__syncthreads()`` ）不同之处在于，线程到达屏障的通知（「到达」）与等待其他线程到达屏障的操作（「等待」）是分离的。
这种分离通过允许线程执行与屏障无关的其他操作来提高执行效率，从而更有效地利用等待时间。
异步屏障可用于实现 CUDA 线程的生产者-消费者模式，或通过让拷贝操作在完成时发出信号（「到达」）屏障来启用内存层次结构内的异步数据拷贝。

异步屏障在计算能力 7.0 或更高的设备上可用。
计算能力 8.0 或更高的设备为共享内存中的异步屏障提供硬件加速，并通过允许块内任何 CUDA 线程子集的硬件加速同步，显著提升了同步粒度。
以前的架构仅在整 warp（ ``__syncwarp()`` ）或整块（ ``__syncthreads()`` ）级别加速同步。

CUDA 编程模型通过 ``cuda::std::barrier`` 提供异步屏障，
这是 `libcu++ <https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/barrier.html>`__ 库中可用的符合 ISO C++ 的屏障。
除了实现 `std::barrier <https://en.cppreference.com/w/cpp/thread/barrier.html>`__ 外，该库还提供 CUDA 特定的扩展来选择屏障的线程作用域以提高性能，
并公开低级 `cuda::ptx <https://nvidia.github.io/cccl/libcudacxx/ptx_api.html>`__ API。
``cuda::barrier`` 可以通过使用 ``friend`` 函数 ``cuda::device::barrier_native_handle()`` 检索屏障的本机句柄并将其传递给 ``cuda::ptx`` 函数来与 ``cuda::ptx`` 互操作。
CUDA 还为线程块作用域下共享内存中的异步屏障提供了 :ref:`原语 API<memory-barrier-primitives-interface>`。

下表概述了可用于在不同线程作用域同步的异步屏障。

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
     - ``cuda::barrier``, ``cuda::ptx``, 原语
   * - cluster
     - 本地共享内存
     - 允许
     - 允许
     - 是（9.0+）
     - ``cuda::barrier``, ``cuda::ptx``
   * - cluster
     - 远程共享内存
     - 允许
     - 不允许
     - 是（9.0+）
     - ``cuda::barrier``, ``cuda::ptx``
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

同步的时间分割
""""""""""""""""""""""""

如果没有异步到达-等待屏障，线程块内的同步使用 ``__syncthreads()`` 或使用 :ref:`cooperative-groups` 时的 ``block.sync()`` 来实现。

.. code-block:: cuda

   #include <cooperative_groups.h>

   __global__ void simple_sync(int iteration_count) {
       auto block = cooperative_groups::this_thread_block();

       for (int i = 0; i < iteration_count; ++i) {
           /* 到达前的代码 */

            // 等待所有线程到达此处。
           block.sync();

           /* 等待后的代码 */
       }
   }

线程在同步点（ ``block.sync()`` ）被阻塞，直到所有线程都已到达同步点。此外，同步点之前发生的内存更新保证在同步点之后对块中的所有线程可见。

此模式有三个阶段：

- 同步**之前**的代码执行将在同步**之后**被读取的内存更新。
- 同步点。
- 同步**之后**的代码，可以看到同步**之前**发生的内存更新。

改用异步屏障时，时间分割的同步模式如下。

.. code-block:: cuda
   :caption: CUDA C++ cuda::barrier

   #include <cuda/barrier>
   #include <cooperative_groups.h>

   __device__ void compute(float *data, int iteration);

   __global__ void split_arrive_wait(int iteration_count, float *data)
   {
     using barrier_t = cuda::barrier<cuda::thread_scope_block>;
     __shared__ barrier_t bar;
     auto block = cooperative_groups::this_thread_block();

     if (block.thread_rank() == 0)
     {
       // 使用预期到达计数初始化屏障。
       init(&bar, block.size());
     }
     block.sync();

     for (int i = 0; i < iteration_count; ++i)
     {
       /* 到达前的代码 */

       // 此线程到达。到达不会阻塞线程。
       barrier_t::arrival_token token = bar.arrive();

       compute(data, i);

       // 等待参与屏障的所有线程完成 bar.arrive()。
       bar.wait(std::move(token));

       /* 等待后的代码 */
     }
   }

.. code-block:: cuda
   :caption: CUDA C++ cuda::ptx

   #include <cuda/ptx>
   #include <cooperative_groups.h>

   __device__ void compute(float *data, int iteration);

   __global__ void split_arrive_wait(int iteration_count, float *data)
   {
     __shared__ uint64_t bar;
     auto block = cooperative_groups::this_thread_block();

     if (block.thread_rank() == 0)
     {
       // 使用预期到达计数初始化屏障。
       cuda::ptx::mbarrier_init(&bar, block.size());
     }
     block.sync();

     for (int i = 0; i < iteration_count; ++i)
     {
       /* 到达前的代码 */

       // 此线程到达。到达不会阻塞线程。
       uint64_t token = cuda::ptx::mbarrier_arrive(&bar);

       compute(data, i);

       // 等待参与屏障的所有线程完成 mbarrier_arrive()。
       while(!cuda::ptx::mbarrier_try_wait(&bar, token)) {}

       /* 等待后的代码 */
     }
   }

.. code-block:: cuda
   :caption: CUDA C 原语

   #include <cuda_awbarrier_primitives.h>
   #include <cooperative_groups.h>

   __device__ void compute(float *data, int iteration);

   __global__ void split_arrive_wait(int iteration_count, float *data)
   {
     __shared__ __mbarrier_t bar;
     auto block = cooperative_groups::this_thread_block();

     if (block.thread_rank() == 0)
     {
       // 使用预期到达计数初始化屏障。
       __mbarrier_init(&bar, block.size());
     }
     block.sync();

     for (int i = 0; i < iteration_count; ++i)
     {
       /* 到达前的代码 */

       // 此线程到达。到达不会阻塞线程。
       __mbarrier_token_t token = __mbarrier_arrive(&bar);

       compute(data, i);

       // 等待参与屏障的所有线程完成 __mbarrier_arrive()。
       while(!__mbarrier_try_wait(&bar, token, 1000)) {}

       /* 等待后的代码 */
     }
   }

在此模式中，同步点分为到达点（ ``bar.arrive()`` ）和等待点（ ``bar.wait(std::move(token))`` ）。
线程通过首次调用 ``bar.arrive()`` 开始参与 ``cuda::barrier`` 。当线程调用 ``bar.wait(std::move(token))`` 时，
它将被阻塞，直到参与线程完成 ``bar.arrive()`` 达到预期次数，即传递给 ``init()`` 的预期到达计数参数。
参与线程调用 ``bar.arrive()`` 之前发生的内存更新保证在调用 ``bar.wait(std::move(token))`` 之后对参与线程可见。
注意，调用 ``bar.arrive()`` 不会阻塞线程，它可以继续执行不依赖于其他参与线程调用 ``bar.arrive()`` 之前发生的内存更新的其他工作。

「到达和等待」模式有五个阶段：

- 到达**之前**的代码执行将在等待**之后**被读取的内存更新。
- 到达点带有隐式内存栅栏（即等效于 ``cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block)`` ）。
- 到达和等待**之间**的代码。
- 等待点。
- 等待**之后**的代码，可以看到到达**之前**执行的更新。

有关如何使用异步屏障的综合指南，请参见 :ref:`async-barriers-details`。

.. _pipelines:

3.2.4.3. 流水线
^^^^^^^^^^^^^^^^^^^

CUDA 编程模型提供流水线同步对象作为协调机制，将异步内存拷贝排序为多个阶段，便于实现双缓冲或多缓冲生产者-消费者模式。
流水线是一个双端队列，具有「头」和「尾」，以先进先出（FIFO）顺序处理工作。生产者线程将工作提交到流水线的头部，而消费者线程从流水线的尾部拉取工作。

流水线通过 `libcu++ <https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/pipeline.html>`__ 库中的 ``cuda::pipeline`` API 公开，
以及通过 :ref:`原语 API<pipeline-primitives-interface>` 公开。下表描述了两个 API 的主要功能。

.. list-table:: cuda::pipeline API
   :header-rows: 1
   :widths: 30 70

   * - ``cuda::pipeline`` API
     - 描述
   * - ``producer_acquire``
     - 获取流水线内部队列中的可用阶段。
   * - ``producer_commit``
     - 提交在当前获取的流水线阶段上，在 ``producer_acquire`` 调用之后发出的异步操作。
   * - ``consumer_wait``
     - 等待流水线最老阶段中异步操作的完成。
   * - ``consumer_release``
     - 将流水线的最老阶段释放给流水线对象以供重用。释放的阶段可以被生产者获取。

.. list-table:: 原语 API
   :header-rows: 1
   :widths: 40 60

   * - 原语 API
     - 描述
   * - ``__pipeline_memcpy_async``
     - 请求将从全局内存到共享内存的内存拷贝提交以进行异步评估。
   * - ``__pipeline_commit``
     - 提交在调用之前在流水线当前阶段上发出的异步操作。
   * - ``__pipeline_wait_prior(N)``
     - 等待除最后 N 次提交之外的所有流水线异步操作完成。

``cuda::pipeline`` API 具有更丰富的接口和更少的限制，而原语 API 仅支持跟踪具有特定大小和对齐要求的从全局内存到共享内存的异步拷贝。
原语 API 提供与具有 ``cuda::thread_scope_thread`` 的 ``cuda::pipeline`` 对象等效的功能。

有关详细的使用模式和示例，请参见 :ref:`pipelines-details`。

.. _asynchronous-data-copies:

3.2.5. 异步数据拷贝
-----------------------

内存层次结构内的高效数据移动是实现 GPU 计算高性能的基础。传统的同步内存操作强制线程在数据传输期间空闲等待。
GPU 本质上通过并行性隐藏内存延迟。也就是说，SM 切换执行另一个 warp，同时内存操作完成。
即使通过并行性实现了这种延迟隐藏，内存延迟仍然可能成为内存带宽利用率和计算资源效率的瓶颈。
为了解决这些瓶颈，现代 GPU 架构提供硬件加速的异步数据拷贝机制，允许内存传输独立进行，同时线程继续执行其他工作。

异步数据拷贝通过将内存传输的发起与等待其完成解耦，实现计算与数据移动的重叠。
这样，线程可以在内存延迟期间执行有用的工作，从而提高整体吞吐量和资源利用率。

.. note::

   虽然本节讨论的概念和原理与之前关于 :ref:`asynchronous-execution` 的章节中讨论的类似，但该章节涵盖的是核函数和内存传输（如 ``cudaMemcpyAsync`` 调用的传输）的异步执行。
   这可以被视为应用程序不同组件之间的异步性。

   本节描述的异步性是指启用 GPU 的 DRAM（即全局内存）与片上 SM 内存（如共享内存或张量内存）之间的数据传输，而不阻塞 GPU 线程。
   这是单个核函数启动执行内的异步性。

为了理解异步拷贝如何提高性能，检查一个常见的 GPU 计算模式很有帮助。CUDA 应用程序通常采用「拷贝和计算」模式：

- 从全局内存获取数据，
- 将数据存储到共享内存，以及
- 对共享内存数据执行计算，并可能将结果写回全局内存。

此模式的「拷贝」阶段通常表示为 ``shared[local_idx] = global[global_idx]`` 。
这种全局到共享内存的拷贝被编译器扩展为从全局内存读取到寄存器，然后从寄存器写入共享内存。

当此模式出现在迭代算法中时，每个线程块需要在 ``shared[local_idx] = global[global_idx]`` 赋值后同步，
以确保在计算阶段开始之前所有对共享内存的写入都已完成。
线程块还需要在计算阶段后再次同步，以防止在所有线程完成计算之前覆盖共享内存。此模式在以下代码片段中演示。

.. code-block:: cuda

   #include <cooperative_groups.h>

   __device__ void compute(int* global_out, int const* shared_in) {
       // 使用当前批次的所有共享内存值进行计算。
       // 将此线程的结果存回全局内存。
   }

   __global__ void without_async_copy(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
     auto grid = cooperative_groups::this_grid();
     auto block = cooperative_groups::this_thread_block();
     assert(size == batch_sz * grid.size()); // 说明：输入大小适合 batch_sz * grid_size

     extern __shared__ int shared[]; // block.size() * sizeof(int) 字节

     size_t local_idx = block.thread_rank();

     for (size_t batch = 0; batch < batch_sz; ++batch) {
       // 计算此块在全局内存中当前批次的索引。
       size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
       size_t global_idx = block_batch_idx + threadIdx.x;
       shared[local_idx] = global_in[global_idx];

       // 等待所有拷贝完成。
       block.sync();

       // 计算并将结果写入全局内存。
       compute(global_out + block_batch_idx, shared);

       // 等待使用共享内存的计算完成。
       block.sync();
     }
   }

使用异步数据拷贝，从全局内存到共享内存的数据移动可以异步进行，以便在等待数据到达时更有效地利用 SM。

.. code-block:: cuda

   #include <cooperative_groups.h>
   #include <cooperative_groups/memcpy_async.h>

   __device__ void compute(int* global_out, int const* shared_in) {
       // 使用当前批次的所有共享内存值进行计算。
       // 将此线程的结果存回全局内存。
   }

   __global__ void with_async_copy(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
     auto grid = cooperative_groups::this_grid();
     auto block = cooperative_groups::this_thread_block();
     assert(size == batch_sz * grid.size()); // 说明：输入大小适合 batch_sz * grid_size

     extern __shared__ int shared[]; // block.size() * sizeof(int) 字节

     size_t local_idx = block.thread_rank();

     for (size_t batch = 0; batch < batch_sz; ++batch) {
       // 计算此块在全局内存中当前批次的索引。
       size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;

       // 整个线程组协作将整个批次拷贝到共享内存。
       cooperative_groups::memcpy_async(block, shared, global_in + block_batch_idx, block.size());

       // 在等待时对不同数据进行计算。

       // 等待所有拷贝完成。
       cooperative_groups::wait(block);

       // 计算并将结果写入全局内存。
       compute(global_out + block_batch_idx, shared);

       // 等待使用共享内存的计算完成。
       block.sync();
     }
   }

:ref:`cooperative_groups::memcpy_async<memcpy-async>` 函数从全局内存拷贝 ``block.size()`` 个元素到 ``shared`` 数据。
此操作就像由另一个线程执行一样，该线程在拷贝完成后与当前线程对 :ref:`cooperative_groups::wait<wait-and-wait-prior>` 的调用同步。
在拷贝操作完成之前，修改全局数据或读取或写入共享数据会引入数据竞争。

此示例演示了所有异步拷贝操作背后的基本概念：它们将内存传输的发起与完成解耦，允许线程在数据在后台移动时执行其他工作。
CUDA 编程模型提供了多个 API 来访问这些功能，包括 :ref:`Cooperative Groups<cooperative-groups-async-h>` 和 `libcu++ <https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html>`__ 库中可用的 ``memcpy_async`` 函数，以及低级 ``cuda::ptx`` 和原语 API。
这些 API 共享相似的语义：它们将对象从源拷贝到目的地，就像由另一个线程执行一样，在拷贝完成时可以使用不同的完成机制进行同步。

现代 GPU 架构为异步数据移动提供多种硬件机制。

- LDGSTS（计算能力 8.0+）允许从全局内存到共享内存的高效小规模异步传输。
- 张量内存加速器（TMA，计算能力 9.0+）扩展了这些功能，提供针对大型多维数据传输优化的批量异步拷贝操作。
- STAS 指令（计算能力 9.0+）启用从寄存器到集群内分布式共享内存的小规模异步传输。

这些机制支持不同的数据路径、传输大小和对齐要求，允许开发者为其特定的数据访问模式选择最合适的方法。
下表概述了 GPU 内异步拷贝支持的源和目标内存空间。

.. list-table:: 异步拷贝可能的源和目标内存空间。空单元格表示不支持源-目标对。
   :header-rows: 1
   :name: tbl:async-source-dest-state-spaces
   :widths: 20 20 30 30

   * - 方向
     -
     - 拷贝机制
     -
   * - 源
     - 目标
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
     - 
     - 支持（STAS, 9.0+）

使用 :ref:`LDGSTS<using-ldgsts>` 、:ref:`使用张量内存加速器（TMA）<using-tma>` 和 :ref:`使用 STAS<using-stas>` 部分将详细介绍每种机制。

.. _configuring-l1-shared-memory-balance:

3.2.6. 配置 L1/共享内存平衡
--------------------------------

如 :ref:`writing-cuda-kernels-caches` 所述，SM 上的 L1 和共享内存使用相同的物理资源，称为统一数据缓存。
在大多数架构上，如果核函数使用很少或不使用共享内存，统一数据缓存可以配置为提供架构允许的最大 L1 缓存量。

为共享内存保留的统一数据缓存可以按核函数进行配置。
应用程序可以在启动核函数之前使用 ``cudaFuncSetAttribute`` 函数设置 ``carveout`` （或首选共享内存容量）。
详见 `CUDA Runtime API 文档 <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html>`__ 。

.. code-block:: cuda

   cudaFuncSetAttribute(kernel_name, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);

应用程序可以将 ``carveout`` 设置为该架构支持的最大共享内存容量的整数百分比。除了整数百分比外，还提供了三个便捷枚举作为 carveout 值。

- ``cudaSharedmemCarveoutDefault``
- ``cudaSharedmemCarveoutMaxL1``
- ``cudaSharedmemCarveoutMaxShared``

支持的最大共享内存和支持的 carveout 大小因架构而异；详情请参见 :ref:`compute-capabilities-table-shared-memory-capacity-per-compute-capability`。

如果所选的整数百分比 carveout 不能精确映射到支持的共享内存容量，则使用下一个更大的容量。
例如，对于计算能力 12.0 的设备，其最大共享内存容量为 100KB，将 carveout 设置为 50% 将导致 64KB 的共享内存，而不是 50KB，因为计算能力 12.0 的设备支持的共享内存大小为 0、8、16、32、64 和 100。

传递给 ``cudaFuncSetAttribute`` 的函数必须使用 ``__global__`` 说明符声明。
``cudaFuncSetAttribute`` 被驱动程序解释为提示，如果执行核函数需要，驱动程序可能会选择不同的 carveout 大小。

.. note::

   另一个 CUDA API ``cudaFuncSetCacheConfig`` 也允许应用程序调整核函数的 L1 和共享内存之间的平衡。
   但是，此 API 为核函数启动设置了 L1/共享内存平衡的硬性要求。
   因此，交错具有不同共享内存配置的核函数将导致不必要的 `序列化启动 <advanced-host-programming.html#advanced-host-implicit-synchronization>`__，
   以等待共享内存重新配置。 ``cudaFuncSetAttribute`` 更受推荐，因为驱动程序可能会在需要执行函数或避免抖动时选择不同的配置。

依赖每块超过 48 KB 共享内存分配的核函数是特定于架构的。
因此，它们必须使用 :ref:`writing-cuda-kernels-dynamic-allocation-shared-memory` 而不是静态大小的数组，并且需要使用 ``cudaFuncSetAttribute`` 显式选择加入，如下所示。

.. code-block:: cuda

   // 设备代码
   __global__ void MyKernel(...)
   {
     extern __shared__ float buffer[];
     ...
   }

   // 主机代码
   int maxbytes = 98304; // 96 KB
   cudaFuncSetAttribute(MyKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
   MyKernel <<<gridDim, blockDim, maxbytes>>>(...);