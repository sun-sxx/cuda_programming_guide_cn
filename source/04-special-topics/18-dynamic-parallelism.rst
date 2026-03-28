.. _cuda-dynamic-parallelism:

4.18. CUDA Dynamic Parallelism
==============================

.. _introduction-cuda-dynamic-parallelism:

4.18.1. 简介
------------

.. _cdp-overview:

4.18.1.1. 概述
^^^^^^^^^^^^^^

CUDA Dynamic Parallelism（通常缩写为 CDP）是 CUDA 编程模型的一个特性，它允许在 GPU 上运行的代码创建新的 GPU 工作。也就是说，可以从已经在 GPU 上运行的设备代码中以额外的 kernel 启动的形式向 GPU 添加新的工作。这个特性可以减少在主机和设备之间传输执行控制和数据的需要，因为启动配置决策可以在运行时由设备上执行的线程做出。

数据相关的并行工作可以由 kernel 在运行时生成。在 CUDA 中添加 CDP 之前，某些算法和编程模式需要修改以消除递归、不规则循环结构或其他不适合扁平、单级并行性的结构。使用 CUDA Dynamic Parallelism 可以更自然地表达这些程序结构。

.. note::

   本节记录了 CUDA Dynamic Parallelism 的更新版本，有时称为 CDP2，这是 CUDA 12.0 及更高版本中的默认版本。CDP2 是计算能力 9.0 及更高设备上唯一可用的 CUDA Dynamic Parallelism 版本。开发者仍然可以使用编译器参数 ``-DCUDA_FORCE_CDP1_IF_SUPPORTED`` 为计算能力低于 9.0 的设备选择传统的 CUDA Dynamic Parallelism（CDP1）。CDP1 文档可以在 `CUDA 编程指南的历史版本 <https://developer.nvidia.com/cuda-toolkit-archive>`_ 中找到。CDP1 计划在未来的 CUDA 版本中移除。

.. _execution-environment-cdp2:

4.18.2. 执行环境
----------------

CUDA 中的 Dynamic Parallelism 允许 GPU 线程配置、启动和隐式同步新的 grid。Grid 是 kernel 启动的一个实例，包括线程块的具体形状和线程块的 grid。区分 kernel 函数本身与该 kernel 的特定调用（即 grid）在以下章节中很重要。

.. _parent-and-child-grids-cdp2:

4.18.2.1. 父 Grid 与子 Grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

配置并启动新 grid 的设备线程属于父 grid。通过调用创建的新 grid 称为子 grid。

子 grid 的调用和完成是正确嵌套的，这意味着父 grid 在其线程创建的所有子 grid 完成之前不被视为完成，运行时保证父与子之间的隐式同步。

.. _fig:parent-child-launch-nesting:

.. figure:: /_static/images/parent-child-launch-nesting.png
   :align: center
   :alt: 父子启动嵌套

   父子启动嵌套

.. _scope-of-cuda-primitives-cdp2:

4.18.2.2. CUDA 原语的作用域
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA Dynamic Parallelism 依赖于 :ref:`cuda-device-runtime`，它允许调用一组有限的 API，这些 API 在语法上与 CUDA Runtime API 相似，但在设备代码中可用。设备运行时 API 的行为与其主机对应项相似，但存在一些差异。这些差异在 :ref:`device-runtime-api-reference` 章节的表格中列出。

在主机和设备上，CUDA 运行时都提供了用于启动 kernel 和通过 stream 和 event 跟踪启动之间依赖关系的 API。在设备上，启动的 kernel 和 CUDA 对象对调用 grid 中的所有线程可见。这意味着，例如，一个 stream 可以由一个线程创建，并由同一 grid 中的任何其他线程使用。但是，由设备 API 调用创建的 CUDA 对象（如 stream 和 event）仅在创建它们的 grid 内有效。

.. _streams-and-events-cdp2:

4.18.2.3. Stream 和 Event
^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA *Stream* 和 *Event* 允许控制 kernel 启动之间的依赖关系：启动到同一 stream 中的 kernel 按顺序执行，event 可用于创建 stream 之间的依赖关系。在设备上创建的 stream 和 event 具有完全相同的目的。

在 grid 内创建的 stream 和 event 存在于 grid 作用域内，但在创建它们的 grid 之外使用时具有未定义的行为。如上所述，grid 退出时，grid 启动的所有工作都会被隐式同步；启动到 stream 中的工作也包括在内，所有依赖关系都会被适当解析。对在 grid 作用域之外被修改的 stream 执行操作的行为是未定义的。

在主机上创建的 stream 和 event 在任何 kernel 内使用时具有未定义的行为，就像由父 grid 创建的 stream 和 event 在子 grid 内使用时具有未定义的行为一样。

.. _ordering-and-concurrency-cdp2:

4.18.2.4. 顺序与并发
^^^^^^^^^^^^^^^^^^^^

设备运行时中 kernel 启动的顺序遵循 CUDA Stream 顺序语义。在 grid 内，所有启动到同一 stream 中的 kernel（ :ref:`fire-and-forget-stream` 除外）按顺序执行。当同一 grid 中的多个线程启动到同一 stream 时，stream 内的顺序取决于 grid 内的线程调度，这可以使用同步原语（如 ``__syncthreads()`` ）来控制。

请注意，虽然命名 stream 由 grid 内的所有线程共享，但隐式的 *NULL* stream 仅由线程块内的所有线程共享。如果线程块内的多个线程启动到隐式 stream，则这些启动将按顺序执行。如果不同线程块中的线程启动到隐式 stream，这些启动可能会并发执行。如果希望线程块内的多个线程启动能够并发，则应使用显式命名 stream。

设备运行时在 CUDA 执行模型中不引入新的并发保证。也就是说，不保证设备上任意数量的不同线程块之间能够并发执行。

缺乏并发保证也适用于父 grid 及其子 grid。当父 grid 启动子 grid 时，子 grid 可能在满足 stream 依赖关系且硬件资源可用时开始执行，但不保证在父 grid 到达隐式同步点之前开始执行。

并发可能因设备配置、应用程序工作负载和运行时调度而异。因此，依赖不同线程块之间的任何并发是不安全的。

.. _cdp-memory-coherence-and-consistency:

4.18.3. 内存一致性与连贯性
--------------------------

父 grid 和子 grid 共享相同的全局内存和常量内存存储，但具有不同的本地内存和共享内存。下表显示了哪些内存空间允许父和子通过相同的指针访问。子 grid 永远无法访问父 grid 的本地内存或共享内存，父 grid 也无法访问子 grid 的本地内存或共享内存。

.. _tbl:cdp-memory-scope:

.. list-table:: Dynamic Parallelism：父 Grid 与子 Grid 之间的内存作用域可访问性
   :header-rows: 1
   :widths: 50 50

   * - 内存空间
     - 父/子使用相同指针？
   * - Global Memory
     - 是
   * - Mapped Memory
     - 是
   * - Local Memory
     - 否
   * - Shared Memory
     - 否
   * - Texture Memory
     - 是（只读）

.. _global-memory-cdp2:

4.18.3.1. 全局内存
^^^^^^^^^^^^^^^^^^

父 grid 和子 grid 对全局内存具有一致的访问权限，但在子和父之间有弱一致性保证。在子 grid 执行过程中，只有一个时间点其内存视图与父线程完全一致：在子 grid 被父调用的时间点。

父线程在子 grid 调用之前的所有全局内存操作对子 grid 可见。由于移除了 ``cudaDeviceSynchronize()`` ，不再可能从父 grid 访问子 grid 中线程所做的修改。在父 grid 退出之前访问子 grid 中线程所做的修改的唯一方法是通过启动到 ``cudaStreamTailLaunch`` stream 中的 kernel。

在以下示例中，执行 ``child_launch`` 的子 grid 仅保证看到子 grid 启动之前对 ``data`` 所做的修改。由于父的线程 0 正在执行启动，子将与父线程 0 看到的内存一致。由于第一个 ``__syncthreads()`` 调用，子将看到 ``data[0]=0``、``data[1]=1``、...、``data[255]=255``（如果没有 ``__syncthreads()`` 调用，只有 ``data[0]=0`` 被保证被子看到）。子 grid 仅保证在隐式同步时返回。这意味着子 grid 中线程所做的修改永远不保证对父 grid 可用。要访问 ``child_launch`` 所做的修改，需要将 ``tail_launch`` kernel 启动到 ``cudaStreamTailLaunch`` stream 中。

.. code-block:: c++

   __global__ void tail_launch(int *data) {
      data[threadIdx.x] = data[threadIdx.x]+1;
   }

   __global__ void child_launch(int *data) {
      data[threadIdx.x] = data[threadIdx.x]+1;
   }

   __global__ void parent_launch(int *data) {
      data[threadIdx.x] = threadIdx.x;

      __syncthreads();

      if (threadIdx.x == 0) {
          child_launch<<< 1, 256 >>>(data);
          tail_launch<<< 1, 256, 0, cudaStreamTailLaunch >>>(data);
      }
   }

   void host_launch(int *data) {
       parent_launch<<< 1, 256 >>>(data);
   }

.. _mapped-memory-cdp2:

4.18.3.2. 映射内存
^^^^^^^^^^^^^^^^^^

映射的系统内存与全局内存具有相同的一致性和连贯性保证，并遵循上述详细说明的语义。Kernel 不能分配或释放映射内存，但可以使用从主机程序传入的映射内存指针。

.. _shared-and-local-memory-cdp2:

4.18.3.3. 共享内存和本地内存
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

共享内存和本地内存分别是线程块或线程私有的，在父和子之间不可见或不一致。当引用这些位置之一中的对象超出其所属的作用域时，行为是未定义的，可能导致错误。

如果 NVIDIA 编译器能检测到指向本地内存或共享内存的指针作为参数传递给 kernel 启动，它将尝试发出警告。在运行时，程序员可以使用 ``__isGlobal()`` 内置函数确定指针是否引用全局内存，从而可以安全地传递给子启动。

调用 ``cudaMemcpy*Async()`` 或 ``cudaMemset*Async()`` 可能会在设备上调用新的子 kernel 以保持 stream 语义。因此，将共享内存或本地内存指针传递给这些 API 是非法的，并将返回错误。

.. _local-memory-cdp2:

4.18.3.4. 本地内存
^^^^^^^^^^^^^^^^^^

本地内存是执行线程的私有存储，在该线程之外不可见。启动子 kernel 时将指向本地内存的指针作为启动参数传递是非法的。从子 grid 解引用此类本地内存指针的结果是未定义的。

例如，以下代码是非法的，如果 ``x_array`` 被 ``child_launch`` 访问，则行为未定义：

.. code-block:: c++

   int x_array[10];       // 在父的本地内存中创建 x_array
   child_launch<<< 1, 1 >>>(x_array);

程序员有时很难知道编译器何时将变量放入本地内存。作为一般规则，传递给子 kernel 的所有存储都应该从全局内存堆中显式分配，可以使用 ``cudaMalloc()``、``new()`` 或通过在全局作用域声明 ``__device__`` 存储。例如：

.. code-block:: c++

   // 正确 - "value" 是全局存储
   __device__ int value;
   __device__ void x() {
       value = 5;
       child<<< 1, 1 >>>(&value);
   }

.. code-block:: c++

   // 无效 - "value" 是本地存储
   __device__ void y() {
       int value = 5;
       child<<< 1, 1 >>>(&value);
   }

.. _texture-memory-cdp:

4.18.3.4.1. 纹理内存
~~~~~~~~~~~~~~~~~~~~~

对纹理映射的全局内存区域的写入相对于纹理访问是不一致的。纹理内存的一致性在子 grid 调用时和子 grid 完成时强制执行。这意味着子 kernel 启动之前对内存的写入会反映在子的纹理内存访问中。与上面的全局内存类似，子对内存的写入永远不保证反映在父的纹理内存访问中。在父 grid 退出之前访问子 grid 中线程所做的修改的唯一方法是通过启动到 ``cudaStreamTailLaunch`` stream 中的 kernel。父和子的并发访问可能导致数据不一致。

.. _programming-interface-cdp:

4.18.4. 编程接口
----------------

.. _cdp-basics:

4.18.4.1. 基础
^^^^^^^^^^^^^^

以下示例展示了一个包含 Dynamic Parallelism 的简单「Hello World」程序：

.. code-block:: c++

   #include <stdio.h>

   __global__ void childKernel()
   {
       printf("Hello ");
   }

   __global__ void tailKernel()
   {
       printf("World!\n");
   }

   __global__ void parentKernel()
   {
       // 启动子 kernel
       childKernel<<<1,1>>>();
       if (cudaSuccess != cudaGetLastError()) {
           return;
       }

       // 将 tail kernel 启动到 cudaStreamTailLaunch stream
       // 隐式同步：等待子 kernel 完成
       tailKernel<<<1,1,0,cudaStreamTailLaunch>>>();

   }

   int main(int argc, char *argv[])
   {
       // 启动父 kernel
       parentKernel<<<1,1>>>();
       if (cudaSuccess != cudaGetLastError()) {
           return 1;
       }

       // 等待父 kernel 完成
       if (cudaSuccess != cudaDeviceSynchronize()) {
           return 2;
       }

       return 0;
   }

该程序可以通过以下命令行单步构建：

.. code-block:: text

   $ nvcc -arch=sm_75 -rdc=true hello_world.cu -o hello -lcudadevrt

.. _dynamic-parallelism-cpp-language-structures:

4.18.4.2. CDP 的 C++ 语言接口
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用 CUDA C++ 进行 Dynamic Parallelism 的 CUDA kernel 可用的语言接口和 API 称为 :ref:`cuda-device-runtime`。

在可能的情况下，CUDA Runtime API 的语法和语义被保留，以便于可能在主机或设备环境中运行的例程的代码重用。

与 CUDA C++ 中的所有代码一样，这里概述的 API 和代码是每线程代码。这使每个线程能够就接下来执行什么 kernel 或操作做出独特的、动态的决策。在块内的线程之间没有同步要求来执行任何提供的设备运行时 API，这使得设备运行时 API 函数可以在任意分歧的 kernel 代码中调用而不会死锁。

.. _dynamic-parallelism-device-runtime-kernel-launch:

4.18.4.2.1. 设备端 Kernel 启动
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

可以使用标准 CUDA <<< >>> 语法从设备启动 kernel：

.. code-block:: c++

   kernel_name<<< Dg, Db, Ns, S >>>([kernel arguments]);

- ``Dg`` 类型为 ``dim3`` ，指定 grid 的维度和大小
- ``Db`` 类型为 ``dim3`` ，指定每个线程块的维度和大小
- ``Ns`` 类型为 ``size_t`` ，指定为此调用每个线程块动态分配的共享内存字节数，除静态分配的内存外。 ``Ns`` 是可选参数，默认为 0。
- ``S`` 类型为 ``cudaStream_t`` ，指定与此调用关联的 stream。该 stream 必须在进行调用的同一 grid 中分配。 ``S`` 是可选参数，默认为 NULL stream。

.. _launches-are-asynchronous:

4.18.4.2.1.1. 启动是异步的
...........................

与主机端启动相同，所有设备端 kernel 启动相对于启动线程都是异步的。也就是说， ``<<<>>>`` 启动命令将立即返回，启动线程将继续执行，直到到达隐式启动同步点，例如启动到 ``cudaStreamTailLaunch`` stream 的 kernel（ :ref:`tail-launch-stream` ）。子 grid 可能在启动后任何时间开始执行，但不保证在启动线程到达隐式启动同步点之前开始执行。

与主机端启动类似，启动到单独 stream 中的工作可能并发运行，但不保证实际并发。依赖子 kernel 之间并发的程序不受 CUDA 编程模型支持，将具有未定义的行为。

.. _launch-environment-configuration:

4.18.4.2.1.2. 启动环境配置
...........................

所有全局设备配置设置（例如，从 ``cudaDeviceGetCacheConfig()`` 返回的共享内存和 L1 缓存大小，以及从 ``cudaDeviceGetLimit()`` 返回的设备限制）将从父继承。同样，设备限制（如栈大小）将保持配置状态。

对于主机启动的 kernel，从主机设置的每 kernel 配置将优先于全局设置。当 kernel 从设备启动时，也将使用这些配置。无法从设备重新配置 kernel 的环境。

.. _events-cdp:

4.18.4.2.2. Event
~~~~~~~~~~~~~~~~~

仅支持 CUDA event 的 stream 间同步功能。这意味着支持 ``cudaStreamWaitEvent()`` ，但不支持 ``cudaEventSynchronize()``、``cudaEventElapsedTime()`` 和 ``cudaEventQuery()`` 。由于不支持 ``cudaEventElapsedTime()`` ，必须通过 ``cudaEventCreateWithFlags()`` 创建 cudaEvent，并传递 ``cudaEventDisableTiming`` 标志。

与 stream 一样，event 对象可以在创建它们的 grid 内的所有线程之间共享，但仅限于该 grid，不能传递给其他 kernel。Event 句柄不保证在 grid 之间唯一，因此在未创建它的 grid 内使用 event 句柄将导致未定义的行为。

.. _synchronization-programming-interface:

4.18.4.2.3. 同步
~~~~~~~~~~~~~~~~

如果调用线程打算与从其他线程调用的子 grid 同步，则程序需要执行足够的线程间同步，例如通过 CUDA Event。

由于无法从父线程显式同步子工作，因此无法保证子 grid 中发生的变化对父 grid 中的线程可见。

.. _device-management-programming:

4.18.4.2.4. 设备管理
~~~~~~~~~~~~~~~~~~~~

只有运行 kernel 的设备才能从该 kernel 控制。这意味着 ``cudaSetDevice()`` 等设备 API 不被设备运行时支持。从 GPU 看到的活动设备（从 ``cudaGetDevice()`` 返回）将具有与主机系统看到的相同的设备编号。 ``cudaDeviceGetAttribute()`` 调用可以请求有关另一个设备的信息，因为此 API 允许指定设备 ID 作为调用参数。请注意，设备运行时不提供通用的 ``cudaGetDeviceProperties()`` API - 必须单独查询属性。

.. _programming-guidelines:

4.18.5. 编程指南
----------------

.. _performance:

4.18.5.1. 性能
^^^^^^^^^^^^^^

.. _dynamic-parallelism-enabled-kernel-overhead:

4.18.5.1.1. 启用 Dynamic Parallelism 的 Kernel 开销
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

控制动态启动时活动的系统软件可能会对当时运行的任何 kernel 施加开销，无论它是否调用自己的 kernel 启动。这种开销来自设备运行时的执行跟踪和管理软件，可能导致性能下降。通常，链接到设备运行时库的应用程序会产生这种开销。

.. _implementation-restrictions-and-limitations:

4.18.5.2. 实现限制与局限性
^^^^^^^^^^^^^^^^^^^^^^^^^^

Dynamic Parallelism 保证本文档中描述的所有语义，但是，某些硬件和软件资源是实现相关的，并限制了使用设备运行的程序的范围、性能和其他属性。

.. _runtime:

4.18.5.2.1. 运行时
~~~~~~~~~~~~~~~~~~

.. _memory-footprint:

4.18.5.2.1.1. 内存占用
.......................

设备运行时系统软件保留内存用于各种管理目的，特别是用于跟踪待处理 grid 启动的预留。配置控件可用于减少此预留的大小，以换取某些启动限制。有关详细信息，请参阅下面的 :ref:`device-runtime-configuration-options`。

.. _pending-kernel-launches:

4.18.5.2.1.2. 待处理的 Kernel 启动
..................................

当 kernel 启动时，所有相关的配置和参数数据都会被跟踪，直到 kernel 完成。此数据存储在系统管理的启动池中。

可以通过从主机调用 ``cudaDeviceSetLimit()`` 并指定 ``cudaLimitDevRuntimePendingLaunchCount`` 来配置固定大小的启动池的大小。

.. _compatibility-and-interoperability:

4.18.5.3. 兼容性与互操作性
^^^^^^^^^^^^^^^^^^^^^^^^^^

CDP2 是默认版本。函数可以使用 ``-DCUDA_FORCE_CDP1_IF_SUPPORTED`` 编译，以在计算能力低于 9.0 的设备上选择不使用 CDP2。

.. list-table:: CDP 版本兼容性
   :header-rows: 1
   :widths: 20 40 40

   * - 
     - 使用 CUDA 12.0 及更新版本编译的函数（默认）
     - 使用 CUDA 12.0 之前版本或使用 CUDA 12.0 及更新版本并指定 ``-DCUDA_FORCE_CDP1_IF_SUPPORTED`` 编译的函数
   * - 编译
     - 如果设备代码引用 ``cudaDeviceSynchronize`` 则编译错误。
     - 如果代码引用 ``cudaStreamTailLaunch`` 或 ``cudaStreamFireAndForget`` 则编译错误。如果设备代码引用 ``cudaDeviceSynchronize`` 且代码为 sm_90 或更新版本编译，则编译错误。
   * - 计算能力 < 9.0
     - 使用新接口。
     - 使用传统接口。
   * - 计算能力 9.0 及更高
     - 使用新接口。
     - 使用新接口。如果函数在设备代码中引用 ``cudaDeviceSynchronize`` ，函数加载将返回 ``cudaErrorSymbolNotFound`` 。这种情况可能发生在代码为计算能力低于 9.0 的设备编译，但在计算能力 9.0 或更高的设备上使用 JIT 运行时。

使用 CDP1 和 CDP2 的函数可以在同一上下文中同时加载和运行。CDP1 函数能够使用 CDP1 特定功能（例如 ``cudaDeviceSynchronize`` ），CDP2 函数能够使用 CDP2 特定功能（例如 tail launch 和 fire-and-forget launch）。

使用 CDP1 的函数不能启动使用 CDP2 的函数，反之亦然。如果使用 CDP1 的函数在其调用图中包含使用 CDP2 的函数，或反之，函数加载期间将导致 ``cudaErrorCdpVersionMismatch`` 。

传统 CDP1 的行为不包含在本文档中。有关 CDP1 的信息，请参阅 `CUDA 编程指南的历史版本 <https://developer.nvidia.com/cuda-toolkit-archive>`_。

.. _device-side-launch-from-ptx-cdp2:

4.18.6. 从 PTX 进行设备端启动
-----------------------------

前面章节讨论了使用 :ref:`cuda-device-runtime` 实现 Dynamic Parallelism。Dynamic Parallelism 也可以从 PTX 执行。对于针对 *Parallel Thread Execution* (PTX) 并计划在其语言中支持 *Dynamic Parallelism* 的编程语言和编译器实现者，本节提供了在 PTX 级别支持 kernel 启动的底层细节。

.. _kernel-launch-apis:

4.18.6.1. Kernel 启动 API
^^^^^^^^^^^^^^^^^^^^^^^^^^

设备端 kernel 启动可以使用以下两个可从 PTX 访问的 API 实现： ``cudaLaunchDevice()`` 和 ``cudaGetParameterBuffer()`` 。 ``cudaLaunchDevice()`` 使用通过调用 ``cudaGetParameterBuffer()`` 获取并填充了启动 kernel 参数的参数缓冲区启动指定的 kernel。如果启动的 kernel 不接受任何参数，参数缓冲区可以为 NULL，即无需调用 ``cudaGetParameterBuffer()`` 。

.. _cudalaunchdevice-cdp2:

4.18.6.1.1. cudaLaunchDevice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 PTX 级别， ``cudaLaunchDevice()`` 需要在使用前以以下两种形式之一声明。

.. code-block:: c++

   // 当 .address_size 为 64 时的 PTX 级 cudaLaunchDevice() 声明
   .extern .func(.param .b32 func_retval0) cudaLaunchDevice
   (
     .param .b64 func,
     .param .b64 parameterBuffer,
     .param .align 4 .b8 gridDimension[12],
     .param .align 4 .b8 blockDimension[12],
     .param .b32 sharedMemSize,
     .param .b64 stream
   )
   ;

下面的 CUDA 级声明映射到上述 PTX 级声明之一，可在系统头文件 ``cuda_device_runtime_api.h`` 中找到。该函数在 ``cudadevrt`` 系统库中定义，必须与程序链接才能使用设备端 kernel 启动功能。

.. code-block:: c++

   // CUDA 级 cudaLaunchDevice() 声明
   extern "C" __device__
   cudaError_t cudaLaunchDevice(void *func, void *parameterBuffer,
                                dim3 gridDimension, dim3 blockDimension,
                                unsigned int sharedMemSize,
                                cudaStream_t stream);

第一个参数是指向要启动的 kernel 的指针，第二个参数是保存启动 kernel 实际参数的参数缓冲区。参数缓冲区的布局在下面的 :ref:`parameter-buffer-layout` 中解释。其他参数指定启动配置，即 grid 维度、块维度、共享内存大小和与启动关联的 stream（有关启动配置的详细描述，请参阅 :ref:`execution-configuration` ）。

.. _cudagetparameterbuffer-cdp2:

4.18.6.1.2. cudaGetParameterBuffer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cudaGetParameterBuffer()`` 需要在使用前在 PTX 级别声明。PTX 级声明必须是以下两种形式之一，具体取决于地址大小：

.. code-block:: c++

   // 当 .address_size 为 64 时的 PTX 级 cudaGetParameterBuffer() 声明
   .extern .func(.param .b64 func_retval0) cudaGetParameterBuffer
   (
     .param .b64 alignment,
     .param .b64 size
   )
   ;

``cudaGetParameterBuffer()`` 的以下 CUDA 级声明映射到上述 PTX 级声明：

.. code-block:: c++

   // CUDA 级 cudaGetParameterBuffer() 声明
   extern "C" __device__
   void *cudaGetParameterBuffer(size_t alignment, size_t size);

第一个参数指定参数缓冲区的对齐要求，第二个参数指定字节大小要求。在当前实现中， ``cudaGetParameterBuffer()`` 返回的参数缓冲区始终保证为 64 字节对齐，对齐要求参数被忽略。但是，建议将正确的对齐要求值（即要放入参数缓冲区的任何参数的最大对齐）传递给 ``cudaGetParameterBuffer()`` ，以确保将来的可移植性。

.. _parameter-buffer-layout:

4.18.6.2. 参数缓冲区布局
^^^^^^^^^^^^^^^^^^^^^^^^

参数缓冲区中禁止参数重新排序，放入参数缓冲区的每个单独参数都需要对齐。也就是说，每个参数必须放在参数缓冲区的第 *n* 个字节，其中 *n* 是参数大小的最小倍数，大于前一个参数占用的最后一个字节的偏移量。参数缓冲区的最大大小为 4KB。

有关 CUDA 编译器生成的 PTX 代码的更详细描述，请参阅 PTX-3.5 规范。