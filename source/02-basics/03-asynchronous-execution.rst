.. _asynchronous-execution:

异步执行
===================

什么是异步并发执行？
----------------------------

CUDA 允许多个任务的并发（或重叠）执行，具体包括：

- 主机上的计算
- 设备上的计算
- 从主机到设备的内存传输
- 从设备到主机的内存传输
- 给定设备内存内的内存传输
- 设备之间的内存传输

并发性通过异步接口实现，其中分发函数调用或 kernel 启动会立即返回。异步调用通常在分发的操作完成之前返回，甚至可能在异步操作开始之前返回。然后，应用程序可以在最初分发的操作执行期间自由执行其他任务。当需要最初分发操作的最终结果时，应用程序必须执行某种形式的同步，以确保相关操作已完成。并发执行模式的一个典型示例是将主机和设备内存传输与计算重叠，从而减少或消除其开销。

.. _fig:asynchronous-concurrent-execution-with-cuda-streams:
.. figure:: /_static/images/cuda_streams.png
   :align: center
   :width: 1000px
   :alt: 使用 CUDA 流的异步并发执行

   使用 CUDA 流的异步并发执行

一般来说，异步接口通常提供三种主要方式来与分发的操作同步：

- **阻塞方式** ：应用程序调用一个阻塞函数，等待操作完成
- **非阻塞方式** （或轮询方式）：应用程序调用一个立即返回的函数，并提供有关操作状态的信息
- **回调方式** ：当操作完成时，执行预先注册的函数

虽然编程接口是异步的，但实际并发执行各种操作的能力取决于 CUDA 版本和所用硬件的计算能力——这些细节将在本指南的后续章节中讨论（参见 :ref:`sec:compute-capabilities` ）。

在 :ref:`sec:intro-synchronizing-the-gpu` 中，介绍了 CUDA runtime 函数 ``cudaDeviceSynchronize()`` ，这是一个阻塞调用，等待所有先前发出的工作完成。需要 ``cudaDeviceSynchronize()`` 调用的原因是 kernel 启动是异步的并立即返回。CUDA 为阻塞和非阻塞同步方式都提供了 API，甚至支持使用主机端回调函数。

CUDA 中异步执行的核心 API 组件是 **CUDA 流（CUDA Streams）** 和 **CUDA 事件（CUDA Events）**。
在本节的其余部分，我们将解释如何使用这些元素来表达 CUDA 中的异步执行。

一个相关主题是 **CUDA 图（CUDA Graphs）**，它允许预先定义异步操作图，然后可以以最小的开销重复执行。我们在 :ref:`sec:async-execution-cuda-graphs` 中简要介绍 CUDA 图，更详细的讨论在 :ref:`sec:cuda-graphs` 中。

.. _cuda-streams:

CUDA 流
--------

在最基本的层面上，CUDA 流是一个抽象，允许程序员表达一系列操作。流的工作方式类似于工作队列，程序可以向其中添加操作（如内存复制或 kernel 启动）以按顺序执行。给定流的队列前端的操作被执行，然后出队，允许下一个排队的操作到达前端并被考虑执行。流中操作的执行顺序是顺序的，操作按照它们入队的顺序执行。

应用程序可以同时使用多个流。在这种情况下，runtime 将根据 GPU 资源的状态从有可用工作的流中选择要执行的任务。可以为流分配优先级，这作为给 runtime 的提示来影响调度，但不保证特定的执行顺序。

在流中操作的 API 函数调用和 kernel 启动相对于主机线程是异步的。应用程序可以通过等待流为空来与流同步，也可以在设备级别同步。

CUDA 有一个默认流，没有指定流的操作和 kernel 启动会被排入这个默认流。未指定流的代码示例隐式使用此默认流。默认流有一些特定的语义，在 :ref:`sec:async-execution-blocking-non-blocking-default-stream` 中讨论。

创建和销毁 CUDA 流
~~~~~~~~~~~~~~~~~~

可以使用 ``cudaStreamCreate()`` 函数创建 CUDA 流。函数调用初始化流句柄，该句柄可用于在后续函数调用中识别流。

.. _lst:stream-creation-example:
.. code-block:: c

   cudaStream_t stream;        // Stream handle
   cudaStreamCreate(&stream);  // Create a new stream

   // stream based operations ...

   cudaStreamDestroy(stream);  // Destroy the stream

如果当应用程序调用 ``cudaStreamDestroy()`` 时设备仍在流 ``stream`` 中执行工作，则流将在被销毁之前完成流中的所有工作。

在 CUDA 流中启动 Kernel
~~~~~~~~~~~~~~~~~~~~~~~~

通常用于启动 kernel 的三重尖括号语法也可用于将 kernel 启动到特定流中。流被指定为 kernel 启动的额外参数。在以下示例中，名为 ``kernel`` 的 kernel 被启动到句柄为 ``stream`` 的流中，其类型为 ``cudaStream_t`` ，并假定之前已创建：

.. _lst:kernel-launch-stream:
.. code-block:: c

   kernel<<<grid, block, shared_mem_size, stream>>>(...);

kernel 启动是异步的，函数调用立即返回。假设 kernel 启动成功，kernel 将在流 ``stream`` 中执行，应用程序可以在 kernel 执行期间自由在 CPU 上或其他 GPU 流中执行其他任务。

在 CUDA 流中启动内存传输
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _async-execution-memory-transfers:

要将内存传输启动到流中，可以使用 ``cudaMemcpyAsync()`` 函数。此函数类似于 ``cudaMemcpy()`` 函数，但它需要一个额外的参数来指定用于内存传输的流。下面代码块中的函数调用将 ``size`` 字节从 ``src`` 指向的主机内存复制到 ``dst`` 指向的设备内存，在流 ``stream`` 中执行。

.. _lst:async-memory-transfer:
.. code-block:: c

   // Copy `size` bytes from `src` to `dst` in stream `stream`
   cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);

与其他异步函数调用一样，此函数调用立即返回，而 ``cudaMemcpy()`` 函数会阻塞直到内存传输完成。为了安全地访问传输结果，应用程序必须使用某种形式的同步来确定操作已完成。

其他 CUDA 内存传输函数（如 ``cudaMemcpy2D()`` ）也有异步变体。

.. note::

   为了异步执行涉及 CPU 内存的内存复制，主机缓冲区必须是固定的和页面锁定的。如果使用非固定和页面锁定的主机内存， ``cudaMemcpyAsync()`` 将正常工作，但它会恢复为同步行为，不会与其他工作重叠。这可能会抑制使用异步内存传输的性能优势。建议程序使用 ``cudaMallocHost()`` 来分配将用于向 GPU 发送或从 GPU 接收数据的缓冲区。

流同步
~~~~~~

.. _async-execution-stream-synchronization:

与流同步的最简单方法是等待流为空。可以使用 ``cudaStreamSynchronize()`` 函数或 ``cudaStreamQuery()`` 函数两种方式实现。

``cudaStreamSynchronize()`` 函数将阻塞直到流中的所有工作完成。

.. code-block:: c

   // Wait for the stream to be empty of tasks
   cudaStreamSynchronize(stream);

   // At this point the stream is done
   // and we can access the results of stream operations safely

如果我们不想阻塞，但只需要快速检查流是否为空，可以使用 ``cudaStreamQuery()`` 函数。

.. code-block:: c

   // Have a peek at the stream
   // returns cudaSuccess if the stream is empty
   // returns cudaErrorNotReady if the stream is not empty
   cudaError_t status = cudaStreamQuery(stream);

   switch (status) {
       case cudaSuccess:
           // The stream is empty
           std::cout << "The stream is empty" << std::endl;
           break;
       case cudaErrorNotReady:
           // The stream is not empty
           std::cout << "The stream is not empty" << std::endl;
           break;
       default:
           // An error occurred - we should handle this
           break;
   };

.. _cuda-events:

CUDA 事件
----------

CUDA 事件是一种将标记插入 CUDA 流的机制。它们本质上就像示踪粒子，可用于跟踪流中任务的进度。想象一下将两个 kernel 启动到一个流中。如果没有这样的跟踪事件，我们只能确定流是否为空。如果我们有一个依赖于第一个 kernel 输出的操作，我们将无法安全地启动该操作，直到我们知道流为空，此时两个 kernel 都已完成。

使用 CUDA 事件我们可以做得更好。通过在第一个 kernel 之后、第二个 kernel 之前将事件入队到流中，我们可以等待此事件到达流的前端。然后，我们可以安全地启动依赖操作，知道第一个 kernel 已完成，但第二个 kernel 尚未开始。以这种方式使用 CUDA 事件可以构建操作和流之间的依赖关系图。这个图的类比直接转化为后面关于 :ref:`sec:async-execution-cuda-graphs` 的讨论。

CUDA 流还保留时间信息，可用于对 kernel 启动和内存传输进行计时。

创建和销毁 CUDA 事件
~~~~~~~~~~~~~~~~~~~~

可以使用 ``cudaEventCreate()`` 和 ``cudaEventDestroy()`` 函数创建和销毁 CUDA 事件。

.. code-block:: c

   cudaEvent_t event;

   // Create the event
   cudaEventCreate(&event);

   // do some work involving the event

   // Once the work is done and the event is no longer needed
   // we can destroy the event
   cudaEventDestroy(event);

应用程序负责在不再需要事件时销毁它们。

将事件插入 CUDA 流
~~~~~~~~~~~~~~~~~~

可以使用 ``cudaEventRecord()`` 函数将 CUDA 事件插入流中。

.. code-block:: c

   cudaEvent_t event;
   cudaStream_t stream;

   // Create the event
   cudaEventCreate(&event);

   // Insert the event into the stream
   cudaEventRecord(event, stream);

对 CUDA 流中的操作进行计时
~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA 事件可用于对包括 kernel 在内的各种流操作的执行进行计时。当事件到达流的前端时，它会记录时间戳。通过在流中用两个事件包围 kernel，我们可以获得 kernel 执行持续时间的准确计时，如下面的代码片段所示：

.. code-block:: c

   cudaStream_t stream;
   cudaStreamCreate(&stream);

   cudaEvent_t start;
   cudaEvent_t stop;

   // create the events
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

    // record the start event
   cudaEventRecord(start, stream);

   // launch the kernel
   kernel<<<grid, block, 0, stream>>>(...);

   // record the stop event
   cudaEventRecord(stop, stream);

   // wait for the stream to complete
   // both events will have been triggered
   cudaStreamSynchronize(stream);

   // get the timing
   float elapsedTime;
   cudaEventElapsedTime(&elapsedTime, start, stop);
   std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;

   // clean up
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   cudaStreamDestroy(stream);

检查 CUDA 事件的状态
~~~~~~~~~~~~~~~~~~~~~

与检查流状态的情况类似，我们可以以阻塞或非阻塞方式检查事件的状态。

``cudaEventSynchronize()`` 函数将阻塞直到事件完成。在下面的代码片段中，我们将一个 kernel 启动到流中，然后是一个事件，然后是第二个 kernel。我们可以使用 ``cudaEventSynchronize()`` 函数等待第一个 kernel 之后的事件完成，并在原则上立即启动依赖任务，可能在 ``kernel2`` 完成之前。

.. code-block:: c

   cudaEvent_t event;
   cudaStream_t stream;

   // create the stream
   cudaStreamCreate(&stream);

   // create the event
   cudaEventCreate(&event);

   // launch a kernel into the stream
   kernel<<<grid, block, 0, stream>>>(...);

   // Record the event
   cudaEventRecord(event, stream);

   // launch a kernel into the stream
   kernel2<<<grid, block, 0, stream>>>(...);

   // Wait for the event to complete
   // Kernel 1 will be  guaranteed to have completed
   // and we can launch the dependent task.
   cudaEventSynchronize(event);
   dependentCPUtask();

   // Wait for the stream to be empty
   // Kernel 2 is guaranteed to have completed
   cudaStreamSynchronize(stream);

   // destroy the event
   cudaEventDestroy(event);

   // destroy the stream
   cudaStreamDestroy(stream);

可以使用 ``cudaEventQuery()`` 函数以非阻塞方式检查 CUDA 事件是否完成。在下面的示例中，我们将 2 个 kernel 启动到一个流中。第一个 kernel ``kernel1`` 生成一些我们想要复制到主机的数据，但是我们还有一些 CPU 端工作要做。在下面的代码中，我们将 ``kernel1`` 后跟一个事件（ ``event`` ），然后是 ``kernel2`` 入队到流 ``stream1`` 中。然后我们进入一个 CPU 工作循环，但偶尔查看一下事件是否完成，表明 ``kernel1`` 已完成。如果是，我们将主机到设备的复制启动到流 ``stream2`` 中。这种方法允许 CPU 工作与 GPU kernel 执行和设备到主机复制重叠。

.. code-block:: c

   cudaEvent_t event;
   cudaStream_t stream1;
   cudaStream_t stream2;

   size_t size = LARGE_NUMBER;
   float *d_data;

   // Create some data
   cudaMalloc(&d_data, size);
   float *h_data = (float *)malloc(size);

   // create the streams
   cudaStreamCreate(&stream1);   // Processing stream
   cudaStreamCreate(&stream2);   // Copying stream
   bool copyStarted = false;

   //  create the event
   cudaEventCreate(&event);

   // launch kernel1 into the stream
   kernel1<<<grid, block, 0, stream1>>>(d_data, size);
   // enqueue an event following kernel1
   cudaEventRecord(event, stream1);

   // launch kernel2 into the stream
   kernel2<<<grid, block, 0, stream1>>>();

   // while the kernels are running do some work on the CPU
   // but check if kernel1 has completed because then we will start
   // a device to host copy in stream2
   while ( not allCPUWorkDone() || not copyStarted ) {
       doNextChunkOfCPUWork();

       // peek to see if kernel 1 has completed
       // if so enqueue a non-blocking copy into stream2
       if ( not copyStarted ) {
           if( cudaEventQuery(event) == cudaSuccess ) {
               cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2);
               copyStarted = true;
           }
       }
   }

   // wait for both streams to be done
   cudaStreamSynchronize(stream1);
   cudaStreamSynchronize(stream2);

   // destroy the event
   cudaEventDestroy(event);

   // destroy the streams and free the data
   cudaStreamDestroy(stream1);
   cudaStreamDestroy(stream2);
   cudaFree(d_data);
   free(h_data);

来自流的回调函数
------------------

CUDA 提供了一种从流中在主机上启动函数的机制。目前有两个函数可用于此目的： ``cudaLaunchHostFunc()`` 和 ``cudaStreamAddCallback()`` 。但是， ``cudaStreamAddCallback()`` 即将被弃用，因此应用程序应该使用 ``cudaLaunchHostFunc()`` 。

使用 ``cudaLaunchHostFunc()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cudaLaunchHostFunc()`` 函数的签名如下：

.. code-block:: c

   cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*func)(void *), void *data);

其中：

- ``stream`` ：要启动回调函数的流。
- ``func`` ：要启动的回调函数。
- ``data`` ：传递给回调函数的数据指针。

主机函数本身是一个简单的 C 函数，签名如下：

.. code-block:: c

   void hostFunction(void *data);

其中 ``data`` 参数指向用户定义的数据结构，函数可以解释该结构。使用这样的回调函数时需要注意一些事项。特别是，主机函数不能调用任何 CUDA API。

为了与统一内存一起使用，提供以下执行保证：

- 在函数执行期间，流被视为空闲。因此，例如，函数可以始终使用附加到它入队的流的内存。
- 函数执行的开始具有与在函数之前立即记录在同一流中的事件同步相同的效果。因此，它同步了在函数之前"连接"的流。
- 向任何流添加设备工作不会使流变为活动状态，直到所有先前的主机函数和流回调都已执行。因此，例如，如果工作已通过事件排序在函数调用之后，则函数可能使用全局附加内存，即使工作已添加到另一个流。
- 函数的完成不会导致流变为活动状态，除非如上所述。如果函数后面没有设备工作，流将保持空闲，并且在连续的主机函数或流回调之间没有设备工作时保持空闲。因此，例如，可以通过在流末尾从主机函数发出信号来完成流同步。

使用 ``cudaStreamAddCallback()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   ``cudaStreamAddCallback()`` 函数即将被弃用和移除，此处讨论是为了完整性，因为它可能仍出现在现有代码中。应用程序应该使用或切换到使用 ``cudaLaunchHostFunc()`` 。

``cudaStreamAddCallback()`` 函数的签名如下：

.. code-block:: c

   cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags);

其中：

- ``stream`` ：要启动回调函数的流。
- ``callback`` ：要启动的回调函数。
- ``userData`` ：传递给回调函数的数据指针。
- ``flags`` ：目前，此参数必须为 0 以保持未来兼容性。

``callback`` 函数的签名与我们使用 ``cudaLaunchHostFunc()`` 函数时略有不同。在这种情况下，回调函数是一个 C 函数，签名如下：

.. code-block:: c

   void callbackFunction(cudaStream_t stream, cudaError_t status, void *userData);

其中函数现在接收：

- ``stream`` ：启动回调函数的流句柄。
- ``status`` ：触发回调的流操作的状态。
- ``userData`` ：传递给回调函数的数据指针。

特别是， ``status`` 参数将包含流的当前错误状态，这可能是由先前的操作设置的。与 ``cudaLaunchHostFunc()`` 函数的情况类似，流在主机函数完成之前不会活动并前进到任务，并且不能从回调函数内部调用 CUDA 函数。

异步错误处理
~~~~~~~~~~~~

.. _asynchronous-execution-error-handling:

在 CUDA 流中，错误可能源于流中的任何操作，包括 kernel 启动和内存传输。这些错误可能不会在运行时传播回用户，直到流同步，例如，通过等待事件或调用 ``cudaStreamSynchronize()`` 。有两种方法可以找出流中可能发生的错误。

- 使用函数 ``cudaGetLastError()`` ——此函数返回并清除当前上下文中任何流中遇到的最后一个错误。如果两次调用之间没有发生其他错误，则立即第二次调用 cudaGetLastError() 将返回 ``cudaSuccess`` 。
- 使用函数 ``cudaPeekAtLastError()`` ——此函数返回当前上下文中的最后一个错误，但不清除它。

这两个函数都将错误作为 ``cudaError_t`` 类型的值返回。可以使用函数 ``cudaGetErrorName()`` 和 ``cudaGetErrorString()`` 生成错误的可打印名称。

使用这些函数的示例如下所示：

.. _lst:error-handling-example:
.. code-block:: c
   :caption: 使用 cudaGetLastError() 和 cudaPeekAtLastError() 的示例

   // Some work occurs in streams.
   cudaStreamSynchronize(stream);

   // Look at the last error but do not clear it
   cudaError_t err = cudaPeekAtLastError();
   if (err != cudaSuccess) {
       printf("Error with name: %s\n", cudaGetErrorName(err));
       printf("Error description: %s\n", cudaGetErrorString(err));
   }

   // Look at the last error and clear it
   cudaError_t err2 = cudaGetLastError();
   if (err2 != cudaSuccess) {
       printf("Error with name: %s\n", cudaGetErrorName(err2));
       printf("Error description: %s\n", cudaGetErrorString(err2));
   }

   if (err2 != err) {
       printf("As expected, cudaPeekAtLastError() did not clear the error\n");
   }

   // Check again
   cudaError_t err3 = cudaGetLastError();
   if (err3 == cudaSuccess) {
       printf("As expected, cudaGetLastError() cleared the error\n");
   }

.. tip::

   当错误出现在同步时，特别是在有许多操作的流中，通常很难准确指出错误可能发生在流中的何处。要调试这种情况，一个有用的技巧可能是设置环境变量 ``CUDA_LAUNCH_BLOCKING=1`` 然后运行应用程序。此环境变量的效果是在每次 kernel 启动后同步。这可以帮助追踪哪个 kernel 或传输导致了错误。
   同步可能很昂贵；设置此环境变量时，应用程序可能会明显变慢。

CUDA 流排序
------------

现在我们已经讨论了流、事件和回调函数的基本机制，重要的是要考虑流中异步操作的排序语义。这些语义允许应用程序程序员以安全的方式思考流中操作的排序。有一些特殊情况，这些语义可能会为了性能优化而放宽，例如在*程序化依赖 kernel 启动*场景中，它允许通过使用特殊属性和 kernel 启动机制来实现两个 kernel 的重叠，或者在使用 ``cudaMemcpyBatchAsync()`` 函数批处理内存传输时，runtime 可以并发执行非重叠的批处理复制。我们稍后将讨论这些优化。

最重要的是，CUDA 流是所谓的顺序流。这意味着流中操作的执行顺序与这些操作入队的顺序相同。流中的操作不能跳过其他操作。内存操作（如复制）由 runtime 跟踪，将始终在下一个操作之前完成，以允许依赖 kernel 安全访问正在传输的数据。

阻塞和非阻塞流以及默认流
------------------------

.. _async-execution-blocking-non-blocking-default-stream:

在 CUDA 中有两种类型的流：阻塞和非阻塞。名称可能有点误导，因为阻塞和非阻塞语义仅指流如何与默认流同步。默认情况下，使用 ``cudaStreamCreate()`` 创建的流是阻塞流。要创建非阻塞流，必须使用带有 ``cudaStreamNonBlocking`` 标志的 ``cudaStreamCreateWithFlags()`` 函数：

.. code-block:: c

   cudaStream_t stream;
   cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

非阻塞流可以使用 ``cudaStreamDestroy()`` 以通常的方式销毁。

旧版默认流
~~~~~~~~~~

.. _async-execution-default-stream:

阻塞流和非阻塞流之间的关键区别在于它们如何与**默认流**同步。CUDA 提供了一个旧版默认流（也称为 NULL 流或流 ID 为 0 的流），当在 kernel 启动或阻塞 ``cudaMemcpy()`` 调用中未指定流时使用。此默认流在所有主机线程之间共享，是一个阻塞流。当操作启动到此默认流时，它将与所有其他阻塞流同步，换句话说，它将等待所有其他阻塞流完成才能执行。

.. _lst:legacy-default-stream-example:
.. code-block:: c

   cudaStream_t stream1, stream2;
   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);

   kernel1<<<grid, block, 0, stream1>>>(...);
   kernel2<<<grid, block>>>(...);
   kernel3<<<grid, block, 0, stream2>>>(...);

   cudaDeviceSynchronize();

上面的默认流行为意味着在上面的代码片段中， ``kernel2`` 将等待 ``kernel1`` 完成， ``kernel3`` 将等待 ``kernel2`` 完成，即使原则上所有三个 kernel 都可以并发执行。通过创建非阻塞流，我们可以避免这种同步行为。在下面的代码片段中，我们创建了两个非阻塞流。默认流将不再与这些流同步，原则上所有三个 kernel 都可以并发执行。因此，我们不能假设 kernel 的任何执行顺序，应该执行显式同步（例如使用相当强硬的 ``cudaDeviceSynchronize()`` 调用）以确保 kernel 已完成。

.. _lst:non-blocking-stream-example:
.. code-block:: c

   cudaStream_t stream1, stream2;
   cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

   kernel1<<<grid, block, 0, stream1>>>(...);
   kernel2<<<grid, block>>>(...);
   kernel3<<<grid, block, 0, stream2>>>(...);

   cudaDeviceSynchronize();

每线程默认流
~~~~~~~~~~~~~

从 CUDA-7 开始，CUDA 允许每个主机线程拥有自己独立的默认流，而不是共享的旧版默认流。要启用此行为，必须使用 ``nvcc`` 编译器选项 ``--default-stream per-thread`` 或定义 ``CUDA_API_PER_THREAD_DEFAULT_STREAM`` 预处理器宏。启用此行为后，每个主机线程将拥有自己独立的默认流，该流不会像旧版默认流那样与其他流同步。在这种情况下，:ref:`lst:legacy-default-stream-example` 将表现出与 :ref:`lst:non-blocking-stream-example` 相同的同步行为。

显式同步
--------

有多种方式可以显式地使流彼此同步。

``cudaDeviceSynchronize()`` 等待所有主机线程中所有流的所有先前命令完成。

``cudaStreamSynchronize()`` 接受一个流作为参数，并等待给定流中的所有先前命令完成。它可用于将主机与特定流同步，允许其他流继续在设备上执行。

``cudaStreamWaitEvent()`` 接受一个流和一个事件作为参数（有关事件的描述，请参见 :ref:`sec:cuda-events` ），并使调用 ``cudaStreamWaitEvent()`` 之后添加到给定流的所有命令延迟执行，直到给定事件完成。

``cudaStreamQuery()`` 为应用程序提供了一种了解流中所有先前命令是否已完成的方式。

隐式同步
--------

如果两个来自不同流的操作之间提交了 NULL 流上的任何 CUDA 操作，则这两个操作无法并发运行，除非这些流是非阻塞流（使用 ``cudaStreamNonBlocking`` 标志创建）。

应用程序应遵循以下准则以提高其并发 kernel 执行的潜力：

- 所有独立的操作应在依赖操作之前发出，
- 任何类型的同步应尽可能延迟。

其他和高级主题
--------------

流优先级
~~~~~~~~

.. _async-execution-stream-priorities:

如前所述，开发人员可以为 CUDA 流分配优先级。需要使用 ``cudaStreamCreateWithPriority()`` 函数创建有优先级的流。该函数接受两个参数：流句柄和优先级级别。一般方案是较低的数字对应较高的优先级。可以使用 ``cudaDeviceGetStreamPriorityRange()`` 函数查询给定设备和上下文的给定优先级范围。流的默认优先级为 0。

.. code-block:: c

   int minPriority, maxPriority;

   // Query the priority range for the device
   cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority);

   // Create two streams with different priorities
   // cudaStreamDefault indicates the stream should be created with default flags
   // in other words they will be blocking streams with respect to the legacy default stream
   // One could also use the option `cudaStreamNonBlocking` here to create a non-blocking streams
   cudaStream_t stream1, stream2;
   cudaStreamCreateWithPriority(&stream1, cudaStreamDefault, minPriority);  // Lowest priority
   cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, maxPriority);  // Highest priority

我们应该注意，流的优先级只是给 runtime 的提示，通常主要适用于 kernel 启动，可能不适用于内存传输。流优先级不会抢占已经执行的工作，也不保证任何特定的执行顺序。

使用流捕获介绍 CUDA 图
~~~~~~~~~~~~~~~~~~~~~~~~

.. _async-execution-cuda-graphs:

CUDA 流允许程序按顺序指定一系列操作、kernel 或内存复制。使用多个流和 ``cudaStreamWaitEvent`` 的跨流依赖，应用程序可以指定完整的操作有向无环图（DAG）。某些应用程序可能需要在整个执行过程中多次运行一系列操作或 DAG 操作。

对于这种情况，CUDA 提供了一个称为 CUDA 图的功能。本节介绍 CUDA 图和一种称为*流捕获*的创建机制。:ref:`sec:cuda-graphs` 中提供了 CUDA 图的更详细讨论。捕获或创建图可以帮助减少重复从主机线程调用相同 API 调用链的延迟和 CPU 开销。相反，可以调用一次指定图操作的 API，然后多次执行结果图。

CUDA 图的工作方式如下：

i. 图由应用程序*捕获*。此步骤在第一次执行图时完成一次。图也可以使用 CUDA 图 API 手动组合。
ii. 图被*实例化*。此步骤在捕获图之后完成一次。此步骤可以设置执行图所需的所有各种 runtime 结构，以便尽可能快地启动其组件。
iii. 在剩余的步骤中，预实例化的图被执行所需次数。由于执行图操作所需的所有 runtime 结构已经就位，因此图执行的 CPU 开销被最小化。

.. _lst:cuda-graphs-example:
.. code-block:: c
   :caption: 使用 CUDA 图捕获、实例化和执行简单线性图的阶段（来自 `CUDA Developer Technical Blog <https://developer.nvidia.com/blog/cuda-graphs/>`_，A. Gray, 2019）

   #define N 500000 // tuned such that kernel takes a few microseconds

   // A very lightweight kernel
   __global__ void shortKernel(float * out_d, float * in_d){
       int idx=blockIdx.x*blockDim.x+threadIdx.x;
       if(idx<N) out_d[idx]=1.23*in_d[idx];
   }

   bool graphCreated=false;
   cudaGraph_t graph;
   cudaGraphExec_t instance;

   // The graph will be executed NSTEP times
   for(int istep=0; istep<NSTEP; istep++){
       if(!graphCreated){
           // Capture the graph
           cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

           // Launch NKERNEL kernels
           for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
               shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
           }

           // End the capture
           cudaStreamEndCapture(stream, &graph);

           // Instantiate the graph
           cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
           graphCreated=true;
       }

       // Launch the graph
       cudaGraphLaunch(instance, stream);

       // Synchronize the stream
       cudaStreamSynchronize(stream);
   }

有关 CUDA 图的更多详细信息，请参见 :ref:`sec:cuda-graphs`。

异步执行总结
------------

本节的要点是：

- 异步 API 允许我们表达任务的并发执行，提供了表达各种操作重叠的方式。实际实现的并发性取决于可用的硬件资源和计算能力。
- CUDA 中异步执行的关键抽象是流、事件和回调函数。
- 可以在事件、流和设备级别进行同步。
- 默认流是一个阻塞流，它与所有其他阻塞流同步，但不与非阻塞流同步。
- 可以通过 ``--default-stream per-thread`` 编译器选项或 CUDA_API_PER_THREAD_DEFAULT_STREAM 预处理器宏使用每线程默认流来避免默认流行为。
- 可以创建具有不同优先级的流，这是给 runtime 的提示，可能不适用于内存传输。
- CUDA 提供 API 函数来减少或重叠 kernel 启动和内存传输的开销，例如 CUDA 图、批处理内存传输和程序化依赖 kernel 启动。