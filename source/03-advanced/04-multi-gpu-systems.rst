.. _multi-gpu-introduction:

3.4. 多 GPU 系统编程
=====================

多 GPU 编程允许应用程序处理更大规模的问题，实现单张 GPU 无法企及的性能水平。
这是通过利用多 GPU 系统所提供的更大的聚合计算性能、内存容量以及内存带宽来实现的。

CUDA 通过主机端 API、驱动架构以及配套的 GPU 硬件技术，实现了多 GPU 编程:

- 主机线程的 CUDA 上下文管理
- 系统内所有处理器的统一内存寻址
- GPU 之间的点对点（Peer-to-peer）批量内存传输
- 细粒度的 GPU 间点对点加载/存储（load/store）内存访问
- 更高层级的抽象与支持性的系统软件，如 CUDA 进程间通信、使用 `NCCL <https://developer.nvidia.com/nccl>`_ 的并行归约，
  以及通过 NVLink 和/或 GPU-Direct RDMA 进行的通信（配合 `NVSHMEM <https://developer.nvidia.com/nvshmem>`_  和 MPI 等 API 使用）。

最简单的，多 GPU 编程要求应用程序能够同时管理多个活跃的 CUDA 上下文，将数据分发到各个 GPU 上，在 GPU 上启动核函数完成计算任务，并进行通信或收集结果，以便应用程序进行后续处理。
如何实现这些步骤，取决于如何将应用程序的算法、可用的并行度以及现有的代码结构，以最有效的方式映射到合适的多 GPU 编程方法上。
一些最常见的多 GPU 编程方法包括：

- 单个主机线程驱动多个 GPU。
- 多个主机线程，每个线程各自驱动一个 GPU。
- 多个单线程的主机进程，每个进程各自驱动一个 GPU。
- 多个主机进程，每个进程包含多个线程，且各自驱动自己的 GPU。
- 通过 NVLink 互联的多节点集群，其中的 GPU 由运行在集群各节点上、跨越多个操作系统实例的线程和进程来驱动。

GPU 可以通过设备内存之间的数据传输和对等访问互相通信，这涵盖了上述列出的所有多设备工作分配方法。
通过查询并启用 GPU 间的对等内存访问功能，并利用 NVLink 来实现设备间的高带宽传输以及更细粒度的加载/存储操作，系统可以支持高性能、低延迟的 GPU 通信。

CUDA 的统一虚拟寻址（Unified Virtual Addressing）允许同一主机进程内的多个 GPU 之间进行通信，
查询和启用高性能的对等内存访问与传输（例如通过 NVLink）只需极少的额外步骤。

对于由不同主机进程管理的多个 GPU 之间的通信，可以通过使用进程间通信（IPC）和虚拟内存管理（VMM）API 来实现支持。
关于高级 IPC 概念以及节点内 CUDA IPC API 的介绍，请参见 :ref:`interprocess-communication-details` 。
高级虚拟内存管理（VMM）API 同时支持节点内和跨节点的进程间通信，可在 Linux 和 Windows 操作系统上使用，
并且允许开发者对内存缓冲区的 IPC 共享进行分配级别（per-allocation）的粒度控制，具体详情请参阅 :ref:`virtual-memory-management-details` 。

CUDA 本身提供了在一组 GPU（可能还包括 CPU）内部实现集合操作所需的底层 API，但它并不直接提供高级的多 GPU 集合操作 API。
多 GPU 集合操作由更高层次的 CUDA 通信库（如 `NCCL <https://developer.nvidia.com/nccl>`_ 和 `NVSHMEM <https://developer.nvidia.com/nvshmem>`_ ）提供。

.. _multi-gpu-context-execution:

3.4.1. 多设备上下文和执行管理
------------------------------

应用程序若要使用多个 GPU，首先要枚举可用的 GPU 设备；根据硬件属性、CPU 亲和性以及节点间的互联情况，从可用设备中选择，并为应用程序即将使用的每个设备创建 CUDA 上下文。

.. _multi-gpu-device-enumeration:

3.4.1.1. 设备枚举
~~~~~~~~~~~~~~~~~~~~~~~~

以下示例展示了如何查询支持 CUDA 的设备数量、枚举每个设备并查询其属性。

.. code-block:: cuda

   int deviceCount;
   cudaGetDeviceCount(&deviceCount);
   int device;
   for (device = 0; device < deviceCount; ++device) {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device);
       printf("Device %d has compute capability %d.%d.\n",
              device, deviceProp.major, deviceProp.minor);
   }

.. _multi-gpu-device-selection:

3.4.1.2. 设备选择
~~~~~~~~~~~~~~~~~~~~~~~~

主机线程可以随时通过调用 ``cudaSetDevice()`` 函数来设置其 **当前设备** 。
设备内存分配和 kernel 启动均会在 `当前设备` 上执行；同时，流和事件也会与 `当前设备` 关联创建。在主机线程调用 ``cudaSetDevice()`` 之前，默认使用 0 号设备。

以下示例展示了设置当前设备如何影响内存分配和 kernel 执行。

.. code-block:: cuda

   size_t size = 1024 * sizeof(float);
   cudaSetDevice(0);            // Set device 0 as current
   float* p0;
   cudaMalloc(&p0, size);       // Allocate memory on device 0
   MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0

   cudaSetDevice(1);            // Set device 1 as current
   float* p1;
   cudaMalloc(&p1, size);       // Allocate memory on device 1
   MyKernel<<<1000, 128>>>(p1); // Launch kernel on device 1

.. _multi-gpu-stream-and-event-behavior:

3.4.1.3. 多设备系统中流、事件和内存复制行为
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果将 kernel 启动指令发送到一个未与当前设备关联的流上，该操作将会失败。示例如下：

.. code-block:: cuda

   cudaSetDevice(0);               // Set device 0 as current
   cudaStream_t s0;
   cudaStreamCreate(&s0);          // Create stream s0 on device 0
   MyKernel<<<100, 64, 0, s0>>>(); // Launch kernel on device 0 in s0

   cudaSetDevice(1);               // Set device 1 as current
   cudaStream_t s1;
   cudaStreamCreate(&s1);          // Create stream s1 on device 1
   MyKernel<<<100, 64, 0, s1>>>(); // Launch kernel on device 1 in s1

   // This kernel launch will fail, since stream s0 is not associated to device 1:
   MyKernel<<<100, 64, 0, s0>>>(); // Launch kernel on device 1 in s0

即使将内存拷贝指令发送到一个未与当前设备关联的流上，该操作依然能够成功执行。

如果使用不同设备上的事件和流， ``cudaEventRecord()`` 将返回失败。

如果使用了不同设备上的两个事件， ``cudaEventElapsedTime()`` 将失败。

即使使用了不同设备上的事件， ``cudaEventSynchronize()`` 和 ``cudaEventQuery()`` 依然能够成功执行。

即使使用了不同设备上的流和事件， ``cudaStreamWaitEvent()`` 依然能够成功执行。因此，该函数可用于实现多个设备之间的相互同步。

每个设备都有其独立的 :ref:`默认流 <async-execution-blocking-non-blocking-default-stream>` 。
因此，向某个设备的默认流发出的指令，可能与向任何其他设备的默认流发出的指令乱序执行或并发执行。

.. _multi-gpu-peer-memory-access:

3.4.2. 多设备点对点传输和内存访问
-------------------------------------

.. _multi-gpu-peer-to-peer-transfers:

3.4.2.1. 点对点内存传输
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA 能够在设备之间执行内存传输；当支持点对点（peer-to-peer）内存访问时，它会充分利用专用的拷贝引擎和 NVLink 来最大化传输性能。

（如果需要在设备之间执行内存传输）可以使用 ``cudaMemcpy`` ，并指定拷贝类型为 ``cudaMemcpyDeviceToDevice`` 或 ``cudaMemcpyDefault`` 。
否则，必须使用 ``cudaMemcpyPeer()`` 、 ``cudaMemcpyPeerAsync()`` 、 ``cudaMemcpy3DPeer()`` 或 ``cudaMemcpy3DPeerAsync()`` 执行拷贝操作，示例如下。

.. code-block:: cuda

   cudaSetDevice(0);                   // Set device 0 as current
   float* p0;
   size_t size = 1024 * sizeof(float);
   cudaMalloc(&p0, size);              // Allocate memory on device 0

   cudaSetDevice(1);                   // Set device 1 as current
   float* p1;
   cudaMalloc(&p1, size);              // Allocate memory on device 1

   cudaSetDevice(0);                   // Set device 0 as current
   MyKernel<<<1000, 128>>>(p0);        // Launch kernel on device 0

   cudaSetDevice(1);                   // Set device 1 as current
   cudaMemcpyPeer(p1, 1, p0, 0, size); // Copy p0 to p1
   MyKernel<<<1000, 128>>>(p1);        // Launch kernel on device 1

两个不同设备内存之间的复制（在隐式 *NULL* 流中）：

- 不会开始执行，直到先前向这两个设备发出的所有命令都已完成，并且
- 会一直执行到完成，之后向这两个设备发出的任何后续命令（参见 :ref:`asynchronous-execution` ）才能开始运行。

与流的常规行为一致，两个设备内存之间的异步拷贝可以与另一个流中的拷贝或 kernel 并发执行。

如果在两个设备之间启用了即将介绍的 :ref:`点对点内存访问<multi-gpu-peer-to-peer-memory-access>` ， 那么这两个设备之间的点对点内存拷贝就不需要经过主机中转，因此速度更快。

.. _multi-gpu-peer-to-peer-memory-access:

3.4.2.2. 点对点内存访问
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

依赖系统属性（具体而言是 PCIe 和/或 NVLink 的拓扑结构），设备之间能够直接寻址彼此的内存（即在一个设备上运行的 kernel，可以直接解引用指向另一个设备内存的指针）。
如果 ``cudaDeviceCanAccessPeer()`` 对指定的两个设备返回 ``true`` ，则说明它们之间支持点对点内存访问。

必须通过 ``cudaDeviceEnablePeerAccess()`` 来启用两个设备之间的点对点内存访问。
在未配备 NVSwitch 的系统中，每个设备最多支持八个全局（系统级）的点对点连接。

两个设备均使用 :ref:`统一的虚拟地址空间（Unified Virtual Address Space） <memory-unified-virtual-address-space>` ，
因此可以在两个设备上使用同一个指针寻址（访问内存），示例如下：

.. code-block:: cuda

   cudaSetDevice(0);                   // Set device 0 as current
   float* p0;
   size_t size = 1024 * sizeof(float);
   cudaMalloc(&p0, size);              // Allocate memory on device 0
   MyKernel<<<1000, 128>>>(p0);        // Launch kernel on device 0

   cudaSetDevice(1);                   // Set device 1 as current
   cudaDeviceEnablePeerAccess(0, 0);   // Enable peer-to-peer access
                                       // with device 0

   // Launch kernel on device 1
   // This kernel launch can access memory on device 0 at address p0
   MyKernel<<<1000, 128>>>(p0);

.. note::

   使用 ``cudaDeviceEnablePeerAccess()`` 启用点对点内存访问，其作用范围是全局的，涵盖目标设备上所有之前及之后分配的 GPU 内存。
   通过该函数为某个设备开启点对点访问后，会给该设备上的内存分配操作带来额外的运行时开销；这是因为新分配的内存必须立即对当前设备以及其他同样拥有访问权限的对等设备可见。
   这种额外开销会随着对等设备数量的增加而成倍增长。

   一个更具扩展性的替代方案是使用 CUDA 虚拟内存管理 API，仅在需要时显式地申请支持点对点访问的内存。
   通过在内存分配时显式请求点对点访问权限，那些无需对等设备访问的内存分配操作不会受到任何运行时开销的影响；
   同时，支持点对点访问的数据结构也能被正确地限定作用域，从而提升软件的调试效率与可靠性（参见 :ref:`virtual-memory-management` ）。

.. _multi-gpu-peer-to-peer-memory-consistency-synchronization:

3.4.2.3. 点对点内存一致性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在跨多设备的网格中，并发执行的线程必须使用同步操作来保证内存访问的顺序与正确性。
跨设备进行同步的线程运行在 ``thread_scope_system`` `同步作用域 <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#thread-scopes>`_ 内。
同样地，相关的内存操作也属于 ``thread_scope_system`` :ref:`内存同步域 <memory-sync-domains>` 。

CUDA :ref:`atomic-functions` 可以在仅有一个 GPU 访问该对象的情况下，对位于对等设备内存中的对象执行读-改-写操作。
关于点对点原子性的具体要求与限制，请参阅 CUDA 内存模型中关于 `原子性要求 <https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#atomicity>`_ 的讨论部分。

.. _multi-gpu-managed-memory:

3.4.2.4. 多设备托管内存
~~~~~~~~~~~~~~~~~~~~~~~~

在支持点对点访问的多 GPU 系统中，可以使用统一内存（Managed Memory）。
关于多设备并发访问统一内存的详细要求，以及允许 GPU 独占访问统一内存的相关 API，均在 :ref:`Multi-GPU<um-legacy-multi-gpu>` 章节中进行了详细说明。

.. _multi-gpu-host-iommu-acs-vm:

3.4.2.5. 主机 IOMMU 硬件、PCI 访问控制服务和虚拟机
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 Linux 系统下，CUDA 和显示驱动程序不支持在启用了 IOMMU 的裸机环境中进行 PCIe 点对点内存传输。
不过，它们支持通过虚拟机直通（pass through）的方式使用 IOMMU。
当在裸机系统上运行 Linux 时，必须禁用 IOMMU，以防止设备内存发生静默损坏；相反，在为虚拟机配置 PCIe 直通时，则应启用 IOMMU 并使用 VFIO 驱动。

在 Windows 上，上述 IOMMU 限制不存在。

另请参见 `在 64 位平台上分配 DMA 缓冲区 <https://download.nvidia.com/XFree86/Linux-x86_64/510.85.02/README/dma_issues.html>`_。

此外，在支持 IOMMU 的系统上还可能启用 PCI 访问控制服务（Access Control Services, ACS）。
PCI ACS 功能会将所有的 PCI 点对点流量强制重定向至 CPU 的根节点（Root Complex），这会导致整体对分带宽下降，从而造成显著的性能损失。
