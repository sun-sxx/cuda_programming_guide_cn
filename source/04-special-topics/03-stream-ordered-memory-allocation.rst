.. _stream-ordered-memory-allocation-details:

4.3. 流有序内存分配器
======================

.. _soma-introduction:

4.3.1. 简介
-----------

使用 ``cudaMalloc`` 和 ``cudaFree`` 管理内存分配会导致 GPU 在所有正在执行的 CUDA stream 之间进行同步。流有序内存分配器（Stream-Ordered Memory Allocator）使应用程序能够将内存分配和释放与其他在 CUDA stream 中启动的工作（如 kernel 启动和异步拷贝）进行排序。这通过利用流有序语义来重用内存分配，从而改善了应用程序的内存使用。该分配器还允许应用程序控制分配器的内存缓存行为。当设置了适当的释放阈值时，缓存行为允许分配器在应用程序表示愿意接受更大的内存占用时避免昂贵的操作系统调用。该分配器还支持进程之间轻松且安全的分配共享。

流有序内存分配器：

- 减少了对自定义内存管理抽象的需求，并使需要高性能自定义内存管理的应用程序更容易创建此类管理。
- 使多个库能够共享由驱动程序管理的公共内存池。这可以减少过度的内存消耗。
- 允许驱动程序基于其对分配器和其他流管理 API 的了解来执行优化。

.. note::

   自 CUDA 11.3 起，Nsight Compute 和下一代 CUDA 调试器已支持该分配器。

.. _soma-memory-management:

4.3.2. 内存管理
---------------

``cudaMallocAsync`` 和 ``cudaFreeAsync`` 是实现流有序内存管理的 API。 ``cudaMallocAsync`` 返回一个分配，而 ``cudaFreeAsync`` 释放一个分配。这两个 API 都接受 stream 参数来定义分配何时可用以及何时停止可用。这些函数允许内存操作与特定的 CUDA stream 绑定，使其能够在不阻塞 host 或其他 stream 的情况下执行。通过避免 ``cudaMalloc`` 和 ``cudaFree`` 潜在的昂贵同步，可以提高应用程序性能。

这些 API 还可以通过内存池进行进一步的性能优化，内存池管理和重用大块内存以实现更高效的分配和释放。内存池有助于减少开销并防止内存碎片化，从而改善频繁进行内存分配操作场景下的性能。

.. _soma-allocating-memory:

4.3.2.1. 分配内存
~~~~~~~~~~~~~~~~~

``cudaMallocAsync`` 函数触发 GPU 上的异步内存分配，并与特定的 CUDA stream 关联。 ``cudaMallocAsync`` 允许内存分配在不阻碍 host 或其他 stream 的情况下进行，消除了昂贵的同步需求。

.. note::

   ``cudaMallocAsync`` 在确定分配所在位置时会忽略当前设备/上下文。相反， ``cudaMallocAsync`` 根据指定的内存池或提供的 stream 来确定适当的设备。

下面的代码展示了一个基本的使用模式：在内存在同一个 stream 中分配、使用，然后释放。

.. code-block:: cuda
   :caption: 基本的流有序内存分配模式

   void *ptr;
   size_t size = 512;
   cudaMallocAsync(&ptr, size, cudaStreamPerThread);
   // 使用分配进行工作
   kernel<<<..., cudaStreamPerThread>>>(ptr, ...);
   // 可以指定异步释放而无需同步 CPU 和 GPU
   cudaFreeAsync(ptr, cudaStreamPerThread);

.. note::

   当从与进行分配的 stream 不同的 stream 访问分配时，用户必须保证访问发生在分配操作之后，否则行为是未定义的。

.. _soma-freeing-memory:

4.3.2.2. 释放内存
~~~~~~~~~~~~~~~~~

``cudaFreeAsync()`` 以流有序的方式异步释放设备内存，这意味着内存释放被分配给特定的 CUDA stream，并且不会阻塞 host 或其他 stream。

用户必须保证释放操作发生在分配操作和任何对分配的使用之后。在释放操作开始后对分配的任何使用都会导致未定义行为。

应使用事件和/或 stream 同步操作来保证在释放操作开始之前，从其他 stream 对分配的任何访问都已完成，如下例所示。

.. code-block:: cuda
   :caption: 使用事件同步释放内存

   cudaMallocAsync(&ptr, size, stream1);
   cudaEventRecord(event1, stream1);
   // stream2 必须等待分配准备好后才能访问
   cudaStreamWaitEvent(stream2, event1);
   kernel<<<..., stream2>>>(ptr, ...);
   cudaEventRecord(event2, stream2);
   // stream3 必须等待 stream2 完成对分配的访问后才能释放分配
   cudaStreamWaitEvent(stream3, event2);
   cudaFreeAsync(ptr, stream3);

使用 ``cudaMalloc()`` 分配的内存可以用 ``cudaFreeAsync()`` 释放。如上所述，所有对内存的访问必须在释放操作开始之前完成。

.. code-block:: cuda

   cudaMalloc(&ptr, size);
   kernel<<<..., stream>>>(ptr, ...);
   cudaFreeAsync(ptr, stream);

同样，使用 ``cudaMallocAsync`` 分配的内存可以用 ``cudaFree()`` 释放。当通过 ``cudaFree()`` API 释放此类分配时，驱动程序假定对分配的所有访问已完成，并且不执行进一步的同步。用户可以使用 ``cudaStreamQuery`` / ``cudaStreamSynchronize`` / ``cudaEventQuery`` / ``cudaEventSynchronize`` / ``cudaDeviceSynchronize`` 来保证适当的异步工作已完成，并且 GPU 不会尝试访问该分配。

.. code-block:: cuda

   cudaMallocAsync(&ptr, size, stream);
   kernel<<<..., stream>>>(ptr, ...);
   // 需要同步以避免过早释放内存
   cudaStreamSynchronize(stream);
   cudaFree(ptr);

.. _soma-memory-pools:

4.3.3. 内存池
-------------

内存池封装了根据池属性和特性分配和管理的虚拟地址和物理内存资源。内存池的主要方面是它管理的内存类型和位置。

所有对 ``cudaMallocAsync`` 的调用都使用内存池中的资源。如果未指定内存池， ``cudaMallocAsync`` 使用提供的 stream 设备的当前内存池。设备的当前内存池可以通过 ``cudaDeviceSetMempool`` 设置，并通过 ``cudaDeviceGetMempool`` 查询。每个设备都有一个默认内存池，如果未调用 ``cudaDeviceSetMempool`` ，则该默认内存池处于活动状态。

API ``cudaMallocFromPoolAsync`` 和 `cudaMallocAsync 的 C++ 重载 <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1ga31efcffc48981621feddd98d71a0feb>`_ 允许用户指定用于分配的池，而无需将其设置为当前池。API ``cudaDeviceGetDefaultMempool`` 和 ``cudaMemPoolCreate`` 返回内存池的句柄。 ``cudaMemPoolSetAttribute`` 和 ``cudaMemPoolGetAttribute`` 控制内存池的属性。

.. note::

   设备当前的内存池将位于该设备本地。因此，在不指定内存池的情况下进行分配将始终产生位于 stream 设备本地的分配。

.. _soma-default-implicit-pools:

4.3.3.1. 默认/隐式池
~~~~~~~~~~~~~~~~~~~~

可以通过调用 ``cudaDeviceGetDefaultMempool`` 检索设备的默认内存池。从设备默认内存池进行的分配是位于该设备上的不可迁移设备分配。这些分配将始终可以从该设备访问。默认内存池的可访问性可以通过 ``cudaMemPoolSetAccess`` 修改，并通过 ``cudaMemPoolGetAccess`` 查询。由于默认池不需要显式创建，它们有时被称为隐式池。设备的默认内存池不支持 IPC。

.. _soma-explicit-pools:

4.3.3.2. 显式池
~~~~~~~~~~~~~~~

``cudaMemPoolCreate`` 创建一个显式池。这允许应用程序为其分配请求超出默认/隐式池所提供的属性。这些属性包括 IPC 能力、最大池大小、在支持平台上驻留在特定 CPU NUMA 节点上的分配等。

.. code-block:: cuda
   :caption: 创建与设备 0 上隐式池类似的池

   // 创建与设备 0 上隐式池类似的池
   int device = 0;
   cudaMemPoolProps poolProps = { };
   poolProps.allocType = cudaMemAllocationTypePinned;
   poolProps.location.id = device;
   poolProps.location.type = cudaMemLocationTypeDevice;

   cudaMemPoolCreate(&memPool, &poolProps);

以下代码片段展示了在有效的 CPU NUMA 节点上创建支持 IPC 的内存池的示例。

.. code-block:: cuda
   :caption: 创建支持 IPC 共享的 CPU NUMA 节点内存池

   // 创建驻留在 CPU NUMA 节点上并支持 IPC 共享（通过文件描述符）的池
   int cpu_numa_id = 0;
   cudaMemPoolProps poolProps = { };
   poolProps.allocType = cudaMemAllocationTypePinned;
   poolProps.location.id = cpu_numa_id;
   poolProps.location.type = cudaMemLocationTypeHostNuma;
   poolProps.handleType = cudaMemHandleTypePosixFileDescriptor;

   cudaMemPoolCreate(&ipcMemPool, &poolProps);

.. _soma-device-accessibility:

4.3.3.3. 多 GPU 支持的设备可访问性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

与通过虚拟内存管理 API 控制的分配可访问性类似，内存池分配可访问性不遵循 ``cudaDeviceEnablePeerAccess`` 或 ``cuCtxEnablePeerAccess`` 。对于内存池，API ``cudaMemPoolSetAccess`` 修改哪些设备可以访问来自池的分配。默认情况下，分配只能从分配所在的设备访问。此访问权限无法被撤销。要从其他设备启用访问，访问设备必须与内存池的设备具有对等能力。可以使用 ``cudaDeviceCanAccessPeer`` 进行验证。如果未检查对等能力，设置访问可能会失败并返回 ``cudaErrorInvalidDevice`` 。然而，如果尚未从池中进行任何分配，即使设备不具备对等能力， ``cudaMemPoolSetAccess`` 调用也可能成功。在这种情况下，下次从池中分配时将失败。

值得注意的是， ``cudaMemPoolSetAccess`` 影响来自内存池的所有分配，而不仅仅是未来的分配。同样， ``cudaMemPoolGetAccess`` 报告的可访问性也适用于池中的所有分配，而不仅仅是未来的分配。不建议频繁更改给定 GPU 的池可访问性设置。也就是说，一旦池可以从给定 GPU 访问，它应该在池的生命周期内保持可访问。

.. code-block:: cuda
   :caption: cudaMemPoolSetAccess 使用示例

   // cudaMemPoolSetAccess 使用示例
   cudaError_t setAccessOnDevice(cudaMemPool_t memPool, int residentDevice,
                 int accessingDevice) {
       cudaMemAccessDesc accessDesc = {};
       accessDesc.location.type = cudaMemLocationTypeDevice;
       accessDesc.location.id = accessingDevice;
       accessDesc.flags = cudaMemAccessFlagsProtReadWrite;

       int canAccess = 0;
       cudaError_t error = cudaDeviceCanAccessPeer(&canAccess, accessingDevice,
                 residentDevice);
       if (error != cudaSuccess) {
           return error;
       } else if (canAccess == 0) {
           return cudaErrorPeerAccessUnsupported;
       }

       // 使地址可访问
       return cudaMemPoolSetAccess(memPool, &accessDesc, 1);
   }

.. _soma-enabling-memory-pools-for-ipc:

4.3.3.4. 为 IPC 启用内存池
~~~~~~~~~~~~~~~~~~~~~~~~~~~

内存池可以为进程间通信（IPC）启用，以允许进程之间轻松、高效和安全地共享 GPU 内存。CUDA 的 IPC 内存池提供与 CUDA :ref:`sec:virtual-memory-management` 相同的安全优势。

使用内存池在进程之间共享内存有两个步骤：进程首先需要共享对池的访问权限，然后共享该池中的特定分配。第一步建立并强制执行安全性。第二步协调每个进程中使用的虚拟地址以及导入进程中映射需要何时有效。

.. _soma-creating-sharing-ipc-pools:

4.3.3.4.1. 创建和共享 IPC 内存池
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

共享对池的访问涉及使用 ``cudaMemPoolExportToShareableHandle()`` 检索池的操作系统原生句柄，使用操作系统原生 IPC 机制将句柄传输到导入进程，然后使用 ``cudaMemPoolImportFromShareableHandle()`` API 创建导入的内存池。要使 ``cudaMemPoolExportToShareableHandle`` 成功，内存池必须在池属性结构中指定了请求的句柄类型来创建。

请参考 `示例 <https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/streamOrderedAllocationIPC>`_ 了解在进程之间传输操作系统原生句柄的适当 IPC 机制。其余过程可以在以下代码片段中找到。

.. code-block:: cuda
   :caption: 在导出进程中创建和共享 IPC 内存池

   // 在导出进程中
   // 在设备 0 上创建可导出的支持 IPC 的池
   cudaMemPoolProps poolProps = { };
   poolProps.allocType = cudaMemAllocationTypePinned;
   poolProps.location.id = 0;
   poolProps.location.type = cudaMemLocationTypeDevice;

   // 将 handleTypes 设置为非零值将使池可导出（支持 IPC）
   poolProps.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

   cudaMemPoolCreate(&memPool, &poolProps);

   // 基于文件描述符的句柄是整数类型
   int fdHandle = 0;

   // 检索池的操作系统原生句柄
   // 注意这里传递的是句柄内存的指针
   cudaMemPoolExportToShareableHandle(&fdHandle,
                memPool,
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0);

   // 句柄必须使用适当的操作系统特定 API 发送到导入进程

.. code-block:: cuda
   :caption: 在导入进程中导入 IPC 内存池

   // 在导入进程中
   int fdHandle;
   // 需要使用适当的操作系统特定 API 从导出进程检索句柄
   // 从可共享句柄创建导入的池
   // 注意这里句柄是按值传递的
   cudaMemPoolImportFromShareableHandle(&importedMemPool,
             (void*)fdHandle,
             CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
             0);

.. _soma-set-access-importing-process:

4.3.3.4.2. 在导入进程中设置访问权限
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

导入的内存池最初只能从其驻留设备访问。导入的内存池不继承导出进程设置的任何可访问性。导入进程需要使用 ``cudaMemPoolSetAccess`` 从其计划访问内存的任何 GPU 启用访问权限。

如果导入的内存池属于对导入进程不可见的设备，用户必须使用 ``cudaMemPoolSetAccess`` API 从将要使用分配的 GPU 启用访问权限。（参见 :ref:`sec:soma-device-accessibility` ）

.. _soma-creating-sharing-allocations:

4.3.3.4.3. 从导出池创建和共享分配
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

一旦池被共享，导出进程中使用 ``cudaMallocAsync()`` 从池中进行的分配就可以与已导入该池的进程共享。由于池的安全策略是在池级别建立和验证的，操作系统不需要额外的簿记来为特定的池分配提供安全性。换句话说，导入池分配所需的不透明 ``cudaMemPoolPtrExportData`` 可以使用任何机制发送到导入进程。

虽然分配可以在不以任何方式与分配 stream 同步的情况下导出和导入，但导入进程在访问分配时必须遵循与导出进程相同的规则。具体来说，对分配的访问必须在分配 stream 中的分配操作执行之后进行。以下两个代码片段展示了 ``cudaMemPoolExportPointer()`` 和 ``cudaMemPoolImportPointer()`` 共享分配，并使用 IPC 事件来保证分配准备好之前不会在导入进程中访问该分配。

.. code-block:: cuda
   :caption: 在导出进程中准备分配

   // 在导出进程中准备分配
   cudaMemPoolPtrExportData exportData;
   cudaEvent_t readyIpcEvent;
   cudaIpcEventHandle_t readyIpcEventHandle;

   // 用于进程间协调的 IPC 事件
   // cudaEventInterprocess 标志使事件成为 IPC 事件
   // cudaEventDisableTiming 是出于性能原因设置的
   cudaEventCreate(&readyIpcEvent, cudaEventDisableTiming | cudaEventInterprocess);

   // 从导出内存池分配
   cudaMallocAsync(&ptr, size, exportMemPool, stream);

   // 用于共享分配何时准备好的事件
   cudaEventRecord(readyIpcEvent, stream);
   cudaMemPoolExportPointer(&exportData, ptr);
   cudaIpcGetEventHandle(&readyIpcEventHandle, readyIpcEvent);

   // 使用任何机制与导入进程共享 IPC 事件和指针导出数据
   // 这里我们将数据复制到共享内存
   shmem->ptrData = exportData;
   shmem->readyIpcEventHandle = readyIpcEventHandle;
   // 通知消费者数据已准备好

.. code-block:: cuda
   :caption: 导入分配

   // 导入分配
   cudaMemPoolPtrExportData *importData = &shmem->prtData;
   cudaEvent_t readyIpcEvent;
   cudaIpcEventHandle_t *readyIpcEventHandle = &shmem->readyIpcEventHandle;

   // 需要使用任何机制从导出进程检索 IPC 事件句柄和导出数据
   // 这里我们使用共享内存，只需要同步以确保共享内存已填充

   cudaIpcOpenEventHandle(&readyIpcEvent, readyIpcEventHandle);

   // 导入分配。该操作不会阻塞等待分配准备好
   cudaMemPoolImportPointer(&ptr, importedMemPool, importData);

   // 在导入进程中使用分配之前，等待分配 stream 中的先前 stream 操作完成
   cudaStreamWaitEvent(stream, readyIpcEvent);
   kernel<<<..., stream>>>(ptr, ...);

释放分配时，分配必须在导入进程中释放，然后才能在导出进程中释放。以下代码片段演示了使用 CUDA IPC 事件在两个进程中的 ``cudaFreeAsync`` 操作之间提供所需的同步。从导入进程对分配的访问显然受到导入进程端释放操作的限制。值得注意的是， ``cudaFree`` 可用于在两个进程中释放分配，并且可以使用其他 stream 同步 API 代替 CUDA IPC 事件。

.. code-block:: cuda
   :caption: IPC 分配的释放顺序

   // 释放必须在导入进程中先于导出进程进行
   kernel<<<..., stream>>>(ptr, ...);

   // 导入进程中的最后一次访问
   cudaFreeAsync(ptr, stream);

   // 导入进程中释放后不允许访问
   cudaIpcEventRecord(finishedIpcEvent, stream);

   // 导出进程
   // 导出进程需要将其释放与导入进程释放的 stream 顺序协调
   cudaStreamWaitEvent(stream, finishedIpcEvent);
   kernel<<<..., stream>>>(ptrInExportingProcess, ...);

   // 导入进程中的释放不会阻止导出进程使用分配
   cudaFreeAsync(ptrInExportingProcess, stream);

.. _soma-ipc-export-pool-limitations:

4.3.3.4.4. IPC 导出池限制
^^^^^^^^^^^^^^^^^^^^^^^^^^

IPC 池目前不支持将物理块释放回操作系统。因此， ``cudaMemPoolTrimTo`` API 无效， ``cudaMemPoolAttrReleaseThreshold`` 实际上被忽略。此行为由驱动程序控制，而非运行时，可能会在未来的驱动程序更新中更改。

.. _soma-ipc-import-pool-limitations:

4.3.3.4.5. IPC 导入池限制
^^^^^^^^^^^^^^^^^^^^^^^^^^

不允许从导入池进行分配；具体来说，导入池不能设置为当前池，也不能在 ``cudaMallocFromPoolAsync`` API 中使用。因此，分配重用策略属性对这些池没有意义。

IPC 导入池与 IPC 导出池一样，目前不支持将物理块释放回操作系统。

资源使用统计属性查询仅反映导入到进程中的分配和关联的物理内存。

.. _soma-best-practices:

4.3.4. 最佳实践和调优
----------------------

.. _soma-query-support:

4.3.4.1. 查询支持
~~~~~~~~~~~~~~~~~

应用程序可以通过使用设备属性 ``cudaDevAttrMemoryPoolsSupported`` 调用 ``cudaDeviceGetAttribute()`` 来确定设备是否支持流有序内存分配器。详见 `开发者博客 <https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/>`_。

可以使用 ``cudaDevAttrMemoryPoolSupportedHandleTypes`` 设备属性查询 IPC 内存池支持。此属性是在 CUDA 11.3 中添加的，较旧的驱动程序在查询此属性时将返回 ``cudaErrorInvalidValue`` 。

.. code-block:: cuda
   :caption: 查询流有序内存分配器支持

   int driverVersion = 0;
   int deviceSupportsMemoryPools = 0;
   int poolSupportedHandleTypes = 0;
   cudaDriverGetVersion(&driverVersion);
   if (driverVersion >= 11020) {
       cudaDeviceGetAttribute(&deviceSupportsMemoryPools,
                              cudaDevAttrMemoryPoolsSupported, device);
   }
   if (deviceSupportsMemoryPools != 0) {
       // `device` 支持流有序内存分配器
   }

   if (driverVersion >= 11030) {
       cudaDeviceGetAttribute(&poolSupportedHandleTypes,
                 cudaDevAttrMemoryPoolSupportedHandleTypes, device);
   }
   if (poolSupportedHandleTypes & cudaMemHandleTypePosixFileDescriptor) {
      // 指定设备上的池可以使用基于 POSIX 文件描述符的 IPC 创建
   }

在查询之前执行驱动程序版本检查可以避免在尚未定义该属性的驱动程序上遇到 ``cudaErrorInvalidValue`` 错误。可以使用 ``cudaGetLastError`` 来清除错误，而不是避免它。

.. _soma-physical-page-caching:

4.3.4.2. 物理页缓存行为
~~~~~~~~~~~~~~~~~~~~~~~

默认情况下，分配器尝试最小化池拥有的物理内存。为了最小化分配和释放物理内存的操作系统调用，应用程序必须为每个池配置内存占用空间。应用程序可以使用释放阈值属性（ ``cudaMemPoolAttrReleaseThreshold`` ）来执行此操作。

释放阈值是池在尝试将内存释放回操作系统之前应保留的内存量（以字节为单位）。当内存池持有的内存超过释放阈值字节时，分配器将在下次调用 stream、事件或设备同步时尝试将内存释放回操作系统。将释放阈值设置为 ``UINT64_MAX`` 将阻止驱动程序在每次同步后尝试收缩池。

.. code-block:: cuda
   :caption: 设置释放阈值

   cuuint64_t setVal = UINT64_MAX;
   cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);

将 ``cudaMemPoolAttrReleaseThreshold`` 设置得足够高以有效禁用内存池收缩的应用程序可能希望显式收缩内存池的内存占用空间。 ``cudaMemPoolTrimTo`` 允许应用程序这样做。在收缩内存池占用空间时， ``minBytesToKeep`` 参数允许应用程序保留指定数量的内存，例如它预期在执行的后续阶段需要的内存量。

.. code-block:: cuda
   :caption: 使用 cudaMemPoolTrimTo 显式收缩内存池

   cuuint64_t setVal = UINT64_MAX;
   cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);

   // 需要从流有序分配器获取大量内存的应用程序阶段
   for (i=0; i<10; i++) {
       for (j=0; j<10; j++) {
           cudaMallocAsync(&ptrs[j], size[j], stream);
       }
       kernel<<<...,stream>>>(ptrs,...);
       for (j=0; j<10; j++) {
           cudaFreeAsync(ptrs[j], stream);
       }
   }

   // 进程在下一阶段不需要那么多内存
   // 同步以便收缩操作知道分配不再使用
   cudaStreamSynchronize(stream);
   cudaMemPoolTrimTo(mempool, 0);

   // 其他进程/分配机制现在可以使用收缩操作释放的物理内存

.. _soma-resource-usage-statistics:

4.3.4.3. 资源使用统计
~~~~~~~~~~~~~~~~~~~~~

查询池的 ``cudaMemPoolAttrReservedMemCurrent`` 属性报告池当前消耗的总物理 GPU 内存。查询池的 ``cudaMemPoolAttrUsedMemCurrent`` 返回从池中分配且不可重用的所有内存的总大小。

``cudaMemPoolAttr*MemHigh`` 属性是记录自上次重置以来相应 ``cudaMemPoolAttr*MemCurrent`` 属性达到的最大值的水印。可以使用 ``cudaMemPoolSetAttribute`` API 将它们重置为当前值。

.. code-block:: cuda
   :caption: 获取使用统计信息的示例辅助函数

   // 批量获取使用统计信息的示例辅助函数
   struct usageStatistics {
       cuuint64_t reserved;
       cuuint64_t reservedHigh;
       cuuint64_t used;
       cuuint64_t usedHigh;
   };

   void getUsageStatistics(cudaMemoryPool_t memPool, struct usageStatistics *statistics)
   {
       cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, statistics->reserved);
       cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, statistics->reservedHigh);
       cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, statistics->used);
       cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, statistics->usedHigh);
   }

   // 重置水印将使它们采用当前值
   void resetStatistics(cudaMemoryPool_t memPool)
   {
       cuuint64_t value = 0;
       cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, &value);
       cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, &value);
   }

.. _soma-memory-reuse-policies:

4.3.4.4. 内存重用策略
~~~~~~~~~~~~~~~~~~~~~

为了满足分配请求，驱动程序尝试重用先前通过 ``cudaFreeAsync()`` 释放的内存，然后再尝试从操作系统分配更多内存。例如，在一个 stream 中释放的内存可以在同一 stream 的后续分配请求中立即重用。当一个 stream 与 CPU 同步时，该 stream 中先前释放的内存可用于任何 stream 中的分配重用。重用策略可以应用于默认和显式内存池。

流有序分配器有一些可控制的分配策略。池属性 ``cudaMemPoolReuseFollowEventDependencies``、``cudaMemPoolReuseAllowOpportunistic`` 和 ``cudaMemPoolReuseAllowInternalDependencies`` 控制这些策略，详情如下。这些策略可以通过调用 ``cudaMemPoolSetAttribute`` 启用或禁用。升级到较新的 CUDA 驱动程序可能会更改、增强、扩充和/或重新排序重用策略的枚举。

.. _soma-cudaMemPoolReuseFollowEventDependencies:

4.3.4.4.1. cudaMemPoolReuseFollowEventDependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在分配更多物理 GPU 内存之前，分配器检查由 CUDA 事件建立的依赖信息，并尝试从另一个 stream 中释放的内存进行分配。

.. code-block:: cuda

   cudaMallocAsync(&ptr, size, originalStream);
   kernel<<<..., originalStream>>>(ptr, ...);
   cudaFreeAsync(ptr, originalStream);
   cudaEventRecord(event, originalStream);

   // 在另一个 stream 中等待捕获释放的事件
   // 允许分配器在启用 cudaMemPoolReuseFollowEventDependencies 时
   // 重用内存以满足另一个 stream 中的新分配请求
   cudaStreamWaitEvent(otherStream, event);
   cudaMallocAsync(&ptr2, size, otherStream);

.. _soma-cudaMemPoolReuseAllowOpportunistic:

4.3.4.4.2. cudaMemPoolReuseAllowOpportunistic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当启用 ``cudaMemPoolReuseAllowOpportunistic`` 策略时，分配器检查已释放的分配，以查看释放操作的 stream 顺序语义是否已满足，例如 stream 已通过释放操作指示的执行点。禁用此策略时，分配器仍将重用在 stream 与 CPU 同步时变得可用的内存。禁用此策略不会阻止 ``cudaMemPoolReuseFollowEventDependencies`` 的应用。

.. code-block:: cuda

   cudaMallocAsync(&ptr, size, originalStream);
   kernel<<<..., originalStream>>>(ptr, ...);
   cudaFreeAsync(ptr, originalStream);

   // 一段时间后，kernel 完成运行
   wait(10);

   // 当启用 cudaMemPoolReuseAllowOpportunistic 时
   // 此分配请求可以根据 originalStream 的进度
   // 使用先前的分配来满足
   cudaMallocAsync(&ptr2, size, otherStream);

.. _soma-cudaMemPoolReuseAllowInternalDependencies:

4.3.4.4.3. cudaMemPoolReuseAllowInternalDependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果从操作系统分配和映射更多物理内存失败，驱动程序将查找其可用性取决于另一个 stream 待处理进度的内存。如果找到此类内存，驱动程序将在分配 stream 中插入所需的依赖关系并重用该内存。

.. code-block:: cuda

   cudaMallocAsync(&ptr, size, originalStream);
   kernel<<<..., originalStream>>>(ptr, ...);
   cudaFreeAsync(ptr, originalStream);

   // 当启用 cudaMemPoolReuseAllowInternalDependencies 且
   // 驱动程序无法分配更多物理内存时，驱动程序可能会
   // 在分配 stream 中有效地执行 cudaStreamWaitEvent
   // 以确保 'otherStream' 中的未来工作发生在
   // 允许访问原始分配的原始 stream 中的工作之后
   cudaMallocAsync(&ptr2, size, otherStream);

.. _soma-disabling-reuse-policies:

4.3.4.4.4. 禁用重用策略
^^^^^^^^^^^^^^^^^^^^^^^^

虽然可控的重用策略改善了内存重用，但用户可能希望禁用它们。允许机会性重用（如 ``cudaMemPoolReuseAllowOpportunistic`` ）会基于 CPU 和 GPU 执行的交错引入运行间的分配模式差异。内部依赖插入（如 ``cudaMemPoolReuseAllowInternalDependencies`` ）可能会以意想不到且潜在不确定性的方式序列化工作，而用户可能更希望在分配失败时显式同步事件或 stream。

.. _soma-synchronization-api-actions:

4.3.4.5. 同步 API 操作
~~~~~~~~~~~~~~~~~~~~~~

分配器作为 CUDA 驱动程序的一部分带来的优化之一是与同步 API 的集成。当用户请求 CUDA 驱动程序同步时，驱动程序等待异步工作完成。在返回之前，驱动程序将确定同步保证完成的释放。这些分配可用于分配，而不管指定的 stream 或禁用的分配策略。驱动程序还会在此处检查 ``cudaMemPoolAttrReleaseThreshold`` 并释放任何可以释放的多余物理内存。

.. _soma-addendums:

4.3.5. 附录
-----------

.. _soma-cudaMemcpyAsync-sensitivity:

4.3.5.1. cudaMemcpyAsync 当前上下文/设备敏感性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在当前 CUDA 驱动程序中，任何涉及 ``cudaMallocAsync`` 内存的异步 ``memcpy`` 都应使用指定 stream 的上下文作为调用线程的当前上下文。这对于 ``cudaMemcpyPeerAsync`` 不是必需的，因为 API 中指定的设备主上下文被引用而不是当前上下文。

.. _soma-cudaPointerGetAttributes:

4.3.5.2. cudaPointerGetAttributes 查询
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在对分配调用 ``cudaFreeAsync`` 之后对该分配调用 ``cudaPointerGetAttributes`` 会导致未定义行为。具体来说，分配是否仍可从给定 stream 访问并不重要：行为仍然是未定义的。

.. _soma-cudaGraphAddMemsetNode:

4.3.5.3. cudaGraphAddMemsetNode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cudaGraphAddMemsetNode`` 不适用于通过流有序分配器分配的内存。但是，可以对分配进行 stream 捕获的 memset。

.. _soma-pointer-attributes:

4.3.5.4. 指针属性
~~~~~~~~~~~~~~~~~

``cudaPointerGetAttributes`` 查询适用于流有序分配。由于流有序分配不与上下文关联，查询 ``CU_POINTER_ATTRIBUTE_CONTEXT`` 将成功但在 ``*data`` 中返回 NULL。属性 ``CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL`` 可用于确定分配的位置：这在选择用于使用 ``cudaMemcpyPeerAsync`` 进行 p2p 复制的上下文时非常有用。属性 ``CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE`` 是在 CUDA 11.3 中添加的，可用于调试以及在执行 IPC 之前确认分配来自哪个池。

.. _soma-cpu-virtual-memory:

4.3.5.5. CPU 虚拟内存
~~~~~~~~~~~~~~~~~~~~~

使用 CUDA 流有序内存分配器 API 时，避免使用 "ulimit -v" 设置 VRAM 限制，因为不支持此操作。