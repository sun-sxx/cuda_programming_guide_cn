.. _multi-gpu-introduction:

多 GPU 系统编程
===============

多 GPU 编程允许应用程序利用多 GPU 系统提供的更大聚合算术性能、内存容量和内存带宽，从而处理单 GPU 无法企及的问题规模并实现更高的性能水平。

CUDA 通过主机 API、驱动程序基础设施和支持的 GPU 硬件技术实现多 GPU 编程：

- 主机线程 CUDA 上下文管理
- 系统中所有处理器的统一内存寻址
- GPU 之间的点对点批量内存传输
- 细粒度点对点 GPU 加载/存储内存访问
- 更高级别的抽象和支持系统软件，如 CUDA 进程间通信、使用 `NCCL <https://developer.nvidia.com/nccl>`_ 的并行归约，以及使用 NVLink 和/或 GPU-Direct RDMA 与 `NVSHMEM <https://developer.nvidia.com/nvshmem>`_ 和 MPI 等 API 进行通信

在最基本的层面上，多 GPU 编程要求应用程序同时管理多个活动的 CUDA 上下文，将数据分发到 GPU，在 GPU 上启动 kernel 以完成工作，并通信或收集结果以便应用程序可以对其进行处理。具体的实现方式取决于如何将应用程序的算法、可用的并行性和现有代码结构最有效地映射到合适的多 GPU 编程方法。一些最常见的多 GPU 编程方法包括：

- 单个主机线程驱动多个 GPU
- 多个主机线程，每个线程驱动自己的 GPU
- 多个单线程主机进程，每个进程驱动自己的 GPU
- 多个主机进程包含多个线程，每个线程驱动自己的 GPU
- 多节点 NVLink 连接集群，GPU 由跨集群节点在多个操作系统实例中运行的线程和进程驱动

GPU 可以通过设备内存之间的内存传输和对等访问相互通信，涵盖上述每种多设备工作分发方法。通过查询和启用点对点 GPU 内存访问的使用，并利用 NVLink 实现高带宽传输和设备之间更细粒度的加载/存储操作，支持高性能、低延迟的 GPU 通信。

CUDA 统一虚拟寻址允许同一主机进程内的多个 GPU 之间以最少的额外步骤进行通信，以查询和启用高性能点对点内存访问和传输（例如通过 NVLink）。

由不同主机进程管理的多个 GPU 之间的通信通过使用进程间通信（IPC）和虚拟内存管理（VMM）API 支持。高级 IPC 概念和节点内 CUDA IPC API 的介绍在 :ref:`sec:interprocess-communication` 节中讨论。高级虚拟内存管理（VMM）API 支持节点内和多节点 IPC，可在 Linux 和 Windows 操作系统上使用，并允许对内存缓冲区的 IPC 共享进行按分配粒度的控制，如 :ref:`sec:virtual-memory-management` 中所述。

CUDA 本身提供了在一组 GPU 中实现集合操作所需的 API，可能包括主机，但它本身不提供高级多 GPU 集合 API。多 GPU 集合由更高级别的 CUDA 通信库提供，如 `NCCL <https://developer.nvidia.com/nccl>`_ 和 `NVSHMEM <https://developer.nvidia.com/nvshmem>`_。

.. _multi-gpu-context-execution:

多设备上下文和执行管理
----------------------

应用程序使用多个 GPU 所需的第一步是枚举可用的 GPU 设备，根据其硬件属性、CPU 亲和性和与对等设备的连接性在可用设备中进行适当选择，并为应用程序将使用的每个设备创建 CUDA 上下文。

.. _multi-gpu-device-enumeration:

设备枚举
~~~~~~~~

以下代码示例展示了如何查询支持 CUDA 的设备数量、枚举每个设备并查询其属性。

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

设备选择
~~~~~~~~

主机线程可以随时通过调用 ``cudaSetDevice()`` 设置其当前操作的设备。设备内存分配和 kernel 启动在当前设备上进行；流和事件在与当前设置的设备关联时创建。在主机线程调用 ``cudaSetDevice()`` 之前，当前设备默认为设备 0。

以下代码示例说明了设置当前设备如何影响后续的内存分配和 kernel 执行操作。

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

多设备流、事件和内存复制行为
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果 kernel 启动到与当前设备不关联的流中，则会失败，如下面的代码示例所示。

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

即使内存复制发出到与当前设备不关联的流中，也会成功。

如果输入事件和输入流与不同设备关联， ``cudaEventRecord()`` 将失败。

如果两个输入事件与不同设备关联， ``cudaEventElapsedTime()`` 将失败。

即使输入事件与不同于当前设备的设备关联， ``cudaEventSynchronize()`` 和 ``cudaEventQuery()`` 也会成功。

即使输入流和输入事件与不同设备关联， ``cudaStreamWaitEvent()`` 也会成功。因此， ``cudaStreamWaitEvent()`` 可用于使多个设备相互同步。

每个设备都有自己的 :ref:`默认流 <sec:async-execution-blocking-non-blocking-default-stream>`，因此向一个设备的默认流发出的命令可能会相对于向任何其他设备的默认流发出的命令乱序执行或并发执行。

.. _multi-gpu-peer-memory-access:

多设备点对点传输和内存访问
--------------------------

.. _multi-gpu-peer-to-peer-transfers:

点对点内存传输
~~~~~~~~~~~~~~

CUDA 可以在设备之间执行内存传输，并在可能进行点对点内存访问时利用专用复制引擎和 NVLink 硬件来最大化性能。

``cudaMemcpy`` 可以与复制类型 ``cudaMemcpyDeviceToDevice`` 或 ``cudaMemcpyDefault`` 一起使用。

否则，必须使用 ``cudaMemcpyPeer()``、``cudaMemcpyPeerAsync()``、``cudaMemcpy3DPeer()`` 或 ``cudaMemcpy3DPeerAsync()`` 执行复制，如下面的代码示例所示。

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

- 直到先前向任一设备发出的所有命令都完成后才开始，并且
- 在向任一设备发出的复制之后的任何命令（参见 :ref:`sec:asynchronous-execution` ）可以开始之前运行完成。

与流的正常行为一致，两个设备内存之间的异步复制可以与另一个流中的复制或 kernel 重叠。

如果在两个设备之间启用了点对点访问（例如，如 :ref:`点对点内存访问 <sec:multi-gpu-peer-to-peer-memory-access>` 中所述），则这两个设备之间的点对点内存复制不再需要通过主机进行中转，因此更快。

.. _multi-gpu-peer-to-peer-memory-access:

点对点内存访问
~~~~~~~~~~~~~~

根据系统属性，特别是 PCIe 和/或 NVLink 拓扑，设备能够寻址彼此的内存（即，在一个设备上执行的 kernel 可以解引用指向另一个设备内存的指针）。如果 ``cudaDeviceCanAccessPeer()`` 对指定设备返回 true，则两个设备之间支持点对点内存访问。

必须通过调用 ``cudaDeviceEnablePeerAccess()`` 在两个设备之间启用点对点内存访问，如下面的代码示例所示。在未启用 NVSwitch 的系统上，每个设备最多可以支持系统范围的八个对等连接。

两个设备使用统一的虚拟地址空间（参见 :ref:`统一虚拟地址空间 <sec:memory-unified-virtual-address-space>` ），因此可以使用相同的指针从两个设备寻址内存，如下面的代码示例所示。

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

   使用 ``cudaDeviceEnablePeerAccess()`` 启用对等内存访问会对对等设备上所有先前和后续的 GPU 内存分配全局生效。通过 ``cudaDeviceEnablePeerAccess()`` 启用对设备的对等访问会增加该对等设备上设备内存分配操作的运行时成本，因为需要使分配立即可被当前设备和任何其他也具有访问权限的对等设备访问，增加了随对等设备数量扩展的乘法开销。

   一种更可扩展的替代方案是使用 CUDA 虚拟内存管理 API 在分配时仅按需显式分配可对等访问的内存区域。通过在内存分配期间显式请求对等可访问性，对于不可被对等设备访问的分配，内存分配的运行时成本不受影响，并且可对等访问的数据结构具有正确的作用域，以改善软件调试和可靠性（参见 :ref:`sec:virtual-memory-management` ）。

.. _multi-gpu-peer-to-peer-memory-consistency-synchronization:

点对点内存一致性
~~~~~~~~~~~~~~~~

必须使用同步操作来强制执行跨多个设备分布的网格中并发执行线程对内存访问的顺序和正确性。跨设备同步的线程在 ``thread_scope_system`` `同步作用域 <https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes>`_ 下操作。类似地，内存操作属于 ``thread_scope_system`` `内存同步域 <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-synchronization-domains>`_。

当只有一个 GPU 访问对等设备内存中的对象时，CUDA 原子函数可以在该对象上执行读-修改-写操作。对等原子性的要求和限制在 CUDA 内存模型 `原子性要求 <https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#atomicity>`_ 讨论中描述。

.. _multi-gpu-managed-memory:

多设备托管内存
~~~~~~~~~~~~~~

托管内存可在支持点对点的多 GPU 系统上使用。并发多设备托管内存访问的详细要求以及 GPU 独占访问托管内存的 API 在 :ref:`多 GPU <sec:um-legacy-multi-gpu>` 中描述。

.. _multi-gpu-host-iommu-acs-vm:

主机 IOMMU 硬件、PCI 访问控制服务和虚拟机
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 Linux 上，CUDA 和显示驱动程序不支持启用 IOMMU 的裸机 PCIe 点对点内存传输。但是，CUDA 和显示驱动程序确实支持通过虚拟机直通使用 IOMMU。在裸机系统上运行 Linux 时，必须禁用 IOMMU 以防止设备内存静默损坏。相反，对于虚拟机，应启用 IOMMU 并使用 VFIO 驱动程序进行 PCIe 直通。

在 Windows 上，上述 IOMMU 限制不存在。

另请参见 `在 64 位平台上分配 DMA 缓冲区 <https://download.nvidia.com/XFree86/Linux-x86_64/510.85.02/README/dma_issues.html>`_。

此外，可以在支持 IOMMU 的系统上启用 PCI 访问控制服务（ACS）。PCI ACS 功能将所有 PCI 点对点流量重定向通过 CPU 根复合体，这可能会由于整体二分带宽的减少而导致显著的性能损失。
