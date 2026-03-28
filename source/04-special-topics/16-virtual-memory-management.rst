.. _virtual-memory-management-details:

4.16. 虚拟内存管理
==================

在 CUDA 编程模型中，内存分配调用（如 ``cudaMalloc()`` ）返回 GPU 内存中的内存地址。该地址可用于任何 CUDA API 或在设备内核中使用。开发者可以通过使用 ``cudaEnablePeerAccess`` 启用对等设备访问该内存分配。这样做可以让不同设备上的内核访问相同的数据。然而，所有过去和未来的用户分配也会被映射到目标对等设备。这可能导致用户无意中为将所有 ``cudaMalloc`` 分配映射到对等设备而付出运行时成本。在大多数情况下，应用程序通过仅与另一个设备共享少数分配来进行通信。通常没有必要将所有分配映射到所有设备。此外，将这种方法扩展到多节点设置本身就很困难。

CUDA 提供 *虚拟内存管理* （Virtual Memory Management，VMM）API，让开发者对这个过程有明确的、底层的控制。

虚拟内存分配是由操作系统和内存管理单元（Memory Management Unit，MMU）管理的复杂过程，分两个关键阶段进行。首先，操作系统为程序保留一段连续的虚拟地址范围，而不分配任何物理内存。然后，当程序首次尝试使用该内存时，操作系统提交虚拟地址，根据需要将物理存储分配给虚拟页面。

CUDA 的 VMM API 将类似的概念引入 GPU 内存管理，允许开发者显式保留虚拟地址范围，然后将其映射到物理 GPU 内存。使用 VMM，应用程序可以专门选择某些分配供其他设备访问。

VMM API 让复杂的应用程序能够跨多个 GPU（和 CPU 核心）更高效地管理内存。通过启用手动控制内存保留、映射和访问权限，VMM API 支持高级技术，如细粒度数据共享、零拷贝传输和自定义内存分配器。CUDA VMM API 向用户公开了细粒度的控制，用于在应用程序中管理 GPU 内存。

开发者可以从以下几个方面受益于 VMM API：

- 对虚拟和物理内存管理的细粒度控制，允许将非连续的物理内存块分配和映射到连续的虚拟地址空间。这有助于减少 GPU 内存碎片并提高内存利用率，特别是对于深度神经网络训练等大型工作负载。

- 通过将虚拟地址空间的保留与物理内存分配分离，实现高效的内存分配和释放。开发者可以保留大型虚拟内存区域并按需映射物理内存，而无需昂贵的内存拷贝或重新分配，从而在动态数据结构和可变大小内存分配中实现性能提升。

- 能够动态增长 GPU 内存分配而无需复制和重新分配所有数据，类似于 CPU 内存管理中 ``realloc`` 或 ``std::vector`` 的工作方式。这支持更灵活和高效的 GPU 内存使用模式。

- 通过提供底层 API 来构建复杂的内存分配器和缓存管理系统，提高开发者生产力和应用程序性能，例如在大语言模型中动态管理键值缓存，改善吞吐量和延迟。

- CUDA VMM API 在分布式多 GPU 设置中非常有价值，因为它支持跨多个 GPU 的高效内存共享和访问。通过将虚拟地址与物理内存解耦，API 允许开发者创建统一的虚拟地址空间，其中数据可以动态映射到不同的 GPU。这优化了内存使用并减少了数据传输开销。例如，NVIDIA 的 NCCL 和 NVShmem 库积极使用 VMM。

总之，CUDA VMM API 为开发者提供了超越传统类 malloc 抽象的高级工具，用于精细调整、高效、灵活和可扩展的 GPU 内存管理，这对于高性能和大内存应用程序非常重要。

.. note::

    本节描述的 API 套件需要支持 UVA 的系统。参见 :ref:`Virtual Memory Management APIs <vmm-api-reference>`。

4.16.1. 预备知识
----------------

4.16.1.1. 定义
^^^^^^^^^^^^^^

**Fabric 内存（Fabric Memory）：** Fabric 内存是指可通过高速互连结构（如 NVIDIA 的 NVLink 和 NVSwitch）访问的内存。该结构在多个 GPU 或节点之间提供内存一致性和高带宽容信层，使它们能够高效地共享内存，就像内存连接到统一的结构而不是隔离在各个设备上一样。

CUDA 12.4 及更高版本具有 VMM 分配句柄类型 ``CU_MEM_HANDLE_TYPE_FABRIC`` 。在支持的平台上，如果 NVIDIA IMEX 守护进程正在运行，此分配句柄类型不仅支持使用任何通信机制（如 MPI）进行节点内共享，还支持节点间共享。这允许多节点 NVLink 系统中的 GPU 映射属于同一 NVLink 结构的所有其他 GPU 的内存，即使它们位于不同的节点中。

**内存句柄（Memory Handles）：** 在 VMM 中，句柄是表示物理内存分配的不透明标识符。这些句柄是底层 CUDA VMM API 中管理内存的核心。它们支持对可映射到虚拟地址空间的物理内存对象进行灵活控制。句柄唯一标识物理内存分配。句柄作为内存资源的抽象引用，不暴露直接指针。句柄允许跨进程或设备导出和导入内存等操作，促进内存共享和虚拟化。

**IMEX 通道（IMEX Channels）：** IMEX 代表 *节点间内存交换（internode memory exchange）*，是 NVIDIA GPU 跨不同节点通信解决方案的一部分。IMEX 通道是 GPU 驱动程序功能，在 IMEX 域内的多用户或多节点环境中提供基于用户的内存隔离。IMex 通道作为安全和隔离机制。

IMEX 通道与 fabric 句柄直接相关，必须在多节点 GPU 通信中启用。当 GPU 分配内存并希望使其可被不同节点上的 GPU 访问时，它首先需要导出该内存的句柄。在此导出过程中使用 IMEX 通道生成安全的 fabric 句柄，该句柄只能由具有正确通道访问权限的远程进程导入。

**单播内存访问（Unicast Memory Access）：** 在 VMM API 上下文中，单播内存访问是指特定设备或进程对物理内存到唯一虚拟地址范围的受控、直接映射和访问。单播内存访问意味着特定的 GPU 设备被授予对映射到物理内存分配的保留虚拟地址范围的显式读/写权限，而不是向多个设备广播访问。

**多播内存访问（Multicast Memory Access）：** 在 VMM API 上下文中，多播内存访问是指单个物理内存分配或区域使用多播机制同时映射到多个设备虚拟地址空间的能力。这允许数据以一对多的方式在多个 GPU 之间高效共享，减少冗余数据传输并提高通信效率。NVIDIA 的 CUDA VMM API 支持创建将多个设备的物理内存分配绑定在一起的多播对象。

4.16.1.2. 查询支持
^^^^^^^^^^^^^^^^^^

应用程序在使用功能之前应查询支持情况，因为其可用性可能因 GPU 架构、驱动程序版本和使用的特定软件库而异。以下各节详细介绍如何以编程方式检查必要的支持。

**VMM 支持** 在尝试使用 VMM API 之前，应用程序必须确保它们想要使用的设备支持 CUDA 虚拟内存管理。以下代码示例展示了查询 VMM 支持：

.. code-block:: cpp

    int deviceSupportsVmm;
    CUresult result = cuDeviceGetAttribute(&deviceSupportsVmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device);
    if (deviceSupportsVmm != 0) {
        // `device` 支持 Virtual Memory Management
    }

**Fabric 内存支持：** 在尝试使用 fabric 内存之前，应用程序必须确保它们想要使用的设备支持 fabric 内存。以下代码示例展示了查询 fabric 内存支持：

.. code-block:: cpp

    int deviceSupportsFabricMem;
    CUresult result = cuDeviceGetAttribute(&deviceSupportsFabricMem, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device);
    if (deviceSupportsFabricMem != 0) {
        // `device` 支持 Fabric Memory
    }

除了使用 ``CU_MEM_HANDLE_TYPE_FABRIC`` 作为句柄类型且不需要 OS 原生机制进行进程间通信来交换可共享句柄外，使用 fabric 内存与其他分配句柄类型没有区别。

**IMEX 通道支持** 在 IMEX 域内，IMex 通道支持多用户环境中的安全内存共享。NVIDIA 驱动程序通过创建字符设备 ``nvidia-caps-imex-channels`` 来实现这一点。要使用基于 fabric 句柄的共享，用户应验证两件事：

- 首先，应用程序必须验证该设备是否存在于 /proc/devices 下：

.. code-block:: bash

    # cat /proc/devices | grep nvidia
    195 nvidia
    195 nvidiactl
    234 nvidia-caps-imex-channels
    509 nvidia-nvswitch

nvidia-caps-imex-channels 设备应该有一个主编号（例如 234）。

- 其次，对于两个 CUDA 进程（导出者和导入者）共享内存，它们都必须有权访问同一个 IMex 通道文件。这些文件（如 /dev/nvidia-caps-imex-channels/channel0）是表示各个 IMex 通道的节点。系统管理员必须创建这些文件，例如使用 mknod() 命令。

.. code-block:: bash

    # mknod /dev/nvidia-caps-imex-channels/channelN c <major_number> 0

此命令使用从 /proc/devices 获取的主编号创建 channelN。

.. note::

    默认情况下，如果指定了 NVreg_CreateImexChannel0 模块参数，驱动程序可以创建 channel0。

**多播对象支持：** 在尝试使用多播对象之前，应用程序必须确保它们想要使用的设备支持它们。以下代码示例展示了查询多播对象支持：

.. code-block:: cpp

    int deviceSupportsMultiCast;
    CUresult result = cuDeviceGetAttribute(&deviceSupportsMultiCast, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device);
    if (deviceSupportsMultiCast != 0) {
        // `device` 支持 Multicast Objects
    }

.. _vmm-api-overview:

4.16.2. API 概述
----------------

VMM API 为开发者提供对虚拟内存管理的精细控制。VMM 作为非常底层的 API，需要直接使用 :ref:`CUDA Driver API <sec:driver-api>`。这个多功能 API 可用于单节点和多节点环境。

要有效使用 VMM，开发者必须对内存管理的一些关键概念有扎实的理解：

- 了解操作系统的虚拟内存基础知识，包括它如何处理页面和地址空间
- 了解内存层次结构和硬件特性
- 熟悉进程间通信（IPC）方法，如套接字或消息传递
- 内存访问权限安全的基本知识

.. _fig:vmm-overview-diagram:

.. figure:: /_static/images/vmm-overview-diagram.png
    :align: center
    :alt: VMM 使用概览图

    图 52 VMM 使用概览。此图概述了 VMM 利用所需的步骤序列。该过程首先评估环境设置。基于此评估，用户必须做出关键的初始决定：是使用 fabric 内存句柄还是 OS 特定句柄。根据初始句柄选择，必须采取一系列不同的后续步骤。然而，最终的内存管理操作——特别是映射、保留和设置已分配内存的访问权限——与所选句柄类型相同。

VMM API 工作流涉及一系列内存管理步骤，重点关注在不同设备或进程之间共享内存。首先，开发者必须在源设备上分配物理内存。为了促进共享，VMM API 使用句柄将必要信息传递给目标设备或进程。用户必须导出句柄以进行共享，该句柄可以是 OS 特定句柄或 fabric 特定句柄。OS 特定句柄仅限于单节点上的进程间通信，而 fabric 特定句柄具有更大的通用性，可在单节点和多节点环境中使用。需要注意的是，使用 fabric 特定句柄需要启用 IMex 通道。

一旦导出句柄，必须使用进程间通信协议将其共享给接收进程或进程，方法的选择留给开发者。然后，接收进程使用 VMM API 导入句柄。在句柄成功导出、共享和导入后，源进程和目标进程都必须保留虚拟地址空间，已分配的物理内存将映射到该空间。最后一步是为每个设备设置内存访问权限，确保建立适当的权限。整个过程（包括两种句柄方法）在附图中进一步详细说明。

4.16.3. 单播内存共享
--------------------

共享 GPU 内存可以在具有多个 GPU 的一台机器上或跨机器网络进行。该过程遵循以下步骤：

- **分配和导出：** 一个 GPU 上的 CUDA 程序分配内存并获取其可共享句柄。

- **共享和导入：** 然后使用 IPC、MPI 或 NCCL 等将句柄发送给节点上的其他程序。在接收 GPU 中，CUDA 驱动程序导入句柄，创建必要的内存对象。

- **保留和映射：** 驱动程序创建从程序虚拟地址（VA）到 GPU 物理地址（PA）再到其网络 Fabric 地址（FA）的映射。

- **访问权限：** 为分配设置访问权限。

- **释放内存：** 程序执行结束时释放所有分配。

.. _fig:unicast-memory-sharing:

.. figure:: /_static/images/unicast-memory-sharing.png
    :align: center
    :alt: 单播内存共享示例

    图 53 单播内存共享示例

4.16.3.1. 分配和导出
^^^^^^^^^^^^^^^^^^^^

**分配物理内存** 使用虚拟内存管理 API 进行内存分配的第一步是创建一个物理内存块，为分配提供后备。为了分配物理内存，应用程序必须使用 ``cuMemCreate`` API。此函数创建的分配没有任何设备或主机映射。函数参数 ``CUmemGenericAllocationHandle`` 描述要分配的内存属性，如分配的位置、分配是否将共享到另一个进程（或图形 API），或要分配的内存的物理属性。用户必须确保请求的分配大小与适当的粒度对齐。可以使用 ``cuMemGetAllocationGranularity`` 查询有关分配粒度要求的信息。

**OS 特定句柄 (Linux)**

.. code-block:: cpp

    CUmemGenericAllocationHandle allocatePhysicalMemory(int device, size_t size) {
        CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        prop.requestedHandleType = handleType;

        size_t granularity = 0;
        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

        // Ensure size matches granularity requirements for the allocation
        size_t padded_size = ROUND_UP(size, granularity);

        // Allocate physical memory
        CUmemGenericAllocationHandle allocHandle;
        cuMemCreate(&allocHandle, padded_size, &prop, 0);

        return allocHandle;
    }

**Fabric 句柄**

.. code-block:: cpp

    CUmemGenericAllocationHandle allocatePhysicalMemory(int device, size_t size) {
        CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_FABRIC;
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        prop.requestedHandleType = handleType;

        size_t granularity = 0;
        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

        // Ensure size matches granularity requirements for the allocation
        size_t padded_size = ROUND_UP(size, granularity);

        // Allocate physical memory
        CUmemGenericAllocationHandle allocHandle;
        cuMemCreate(&allocHandle, padded_size, &prop, 0);

        return allocHandle;
    }

.. note::

    由 ``cuMemCreate`` 分配的内存由其返回的 ``CUmemGenericAllocationHandle`` 引用。请注意，此引用不是指针，其内存尚不可访问。

.. note::

    可以使用 ``cuMemGetAllocationPropertiesFromHandle`` 查询分配句柄的属性。

**导出内存句柄** CUDA 虚拟内存管理 API 公开了一种使用句柄进行进程间通信的新机制，以交换有关分配和物理地址空间的必要信息。可以为 OS 特定 IPC 或 fabric 特定 IPC 导出句柄。OS 特定 IPC 句柄只能在单节点设置中使用。Fabric 特定句柄可在单节点或多节点设置中使用。

**OS 特定句柄 (Linux)**

.. code-block:: cpp

    CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CUmemGenericAllocationHandle handle = allocatePhysicalMemory(0, 1<<21);
    int fd;
    cuMemExportToShareableHandle(&fd, handle, handleType, 0);

**Fabric 句柄**

.. code-block:: cpp

    CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_FABRIC;
    CUmemGenericAllocationHandle handle = allocatePhysicalMemory(0, 1<<21);
    CUmemFabricHandle fh;
    cuMemExportToShareableHandle(&fh, handle, handleType, 0);

.. note::

    OS 特定句柄要求所有进程属于同一操作系统。

.. note::

    Fabric 特定句柄需要系统管理员启用 IMex 通道。

`memMapIpcDrv 示例 <https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features/memMapIPCDrv/>`_ 可用作使用 VMM 分配的 IPC 示例。

4.16.3.2. 共享和导入
^^^^^^^^^^^^^^^^^^^^

**共享内存句柄** 一旦导出句柄，必须使用进程间通信协议将其共享给接收进程或进程。开发者可以自由使用任何方法共享句柄。使用的特定 IPC 方法取决于应用程序的设计和环境。常见方法包括 OS 特定进程间套接字和分布式消息传递。使用 OS 特定 IPC 提供高性能传输，但仅限于同一台机器上的进程，不可移植。Fabric 特定 IPC 更简单且更便携。然而，fabric 特定 IPC 需要系统级支持。选择的方法必须安全可靠地将句柄数据传输到目标进程，以便它可以用于导入内存并建立有效的映射。选择 IPC 方法的灵活性允许将 VMM API 集成到各种系统架构中，从单节点应用程序到分布式多节点设置。在以下代码片段中，我们将提供使用套接字编程和 MPI 共享和接收句柄的示例。

**发送：OS 特定 IPC (Linux)**

.. code-block:: cpp

    int ipcSendShareableHandle(int socket, int fd, pid_t process) {
        struct msghdr msg;
        struct iovec iov[1];

        union {
            struct cmsghdr cm;
            char* control;
        } control_un;

        size_t sizeof_control = CMSG_SPACE(sizeof(int)) * sizeof(char);
        control_un.control = (char*) malloc(sizeof_control);

        struct cmsghdr *cmptr;
        ssize_t readResult;
        struct sockaddr_un cliaddr;
        socklen_t len = sizeof(cliaddr);

        // Construct client address to send this Shareable handle to
        memset(&cliaddr, 0, sizeof(cliaddr));
        cliaddr.sun_family = AF_UNIX;
        char temp[20];
        sprintf(temp, "%s%u", "/tmp/", process);
        strcpy(cliaddr.sun_path, temp);
        len = sizeof(cliaddr);

        // Send corresponding shareable handle to the client
        int sendfd = fd;

        msg.msg_control = control_un.control;
        msg.msg_controllen = sizeof_control;

        cmptr = CMSG_FIRSTHDR(&msg);
        cmptr->cmsg_len = CMSG_LEN(sizeof(int));
        cmptr->cmsg_level = SOL_SOCKET;
        cmptr->cmsg_type = SCM_RIGHTS;

        memmove(CMSG_DATA(cmptr), &sendfd, sizeof(sendfd));

        msg.msg_name = (void *)&cliaddr;
        msg.msg_namelen = sizeof(struct sockaddr_un);

        iov[0].iov_base = (void *)"";
        iov[0].iov_len = 1;
        msg.msg_iov = iov;
        msg.msg_iovlen = 1;

        ssize_t sendResult = sendmsg(socket, &msg, 0);
        if (sendResult <= 0) {
            perror("IPC failure: Sending data over socket failed");
            free(control_un.control);
            return -1;
        }

        free(control_un.control);
        return 0;
    }

**发送：OS 特定 IPC (Windows)**

.. code-block:: cpp

    int ipcSendShareableHandle(HANDLE *handle, HANDLE &shareableHandle, PROCESS_INFORMATION process) {
        HANDLE hProcess = OpenProcess(PROCESS_DUP_HANDLE, FALSE, process.dwProcessId);
        HANDLE hDup = INVALID_HANDLE_VALUE;
        DuplicateHandle(GetCurrentProcess(), shareableHandle, hProcess, &hDup, 0, FALSE, DUPLICATE_SAME_ACCESS);
        DWORD cbWritten;
        WriteFile(handle->hMailslot[i], &hDup, (DWORD)sizeof(hDup), &cbWritten, (LPOVERLAPPED)NULL);
        CloseHandle(hProcess);
        return 0;
    }

**发送：Fabric IPC**

.. code-block:: cpp

    MPI_Send(&fh, sizeof(CUmemFabricHandle), MPI_BYTE, 1, 0, MPI_COMM_WORLD);

**接收：OS 特定 IPC (Linux)**

.. code-block:: cpp

    int ipcRecvShareableHandle(int socket, int* fd) {
        struct msghdr msg = {0};
        struct iovec iov[1];
        struct cmsghdr cm;

        // Union to guarantee alignment requirements for control array
        union {
            struct cmsghdr cm;
            // This will not work on QNX as QNX CMSG_SPACE calls __cmsg_alignbytes
            // And __cmsg_alignbytes is a runtime function instead of compile-time macros
            // char control[CMSG_SPACE(sizeof(int))]
            char* control;
        } control_un;

        size_t sizeof_control = CMSG_SPACE(sizeof(int)) * sizeof(char);
        control_un.control = (char*) malloc(sizeof_control);
        struct cmsghdr *cmptr;
        ssize_t n;
        int receivedfd;
        char dummy_buffer[1];
        ssize_t sendResult;
        msg.msg_control = control_un.control;
        msg.msg_controllen = sizeof_control;

        iov[0].iov_base = (void *)dummy_buffer;
        iov[0].iov_len = sizeof(dummy_buffer);

        msg.msg_iov = iov;
        msg.msg_iovlen = 1;
        if ((n = recvmsg(socket, &msg, 0)) <= 0) {
            perror("IPC failure: Receiving data over socket failed");
            free(control_un.control);
            return -1;
        }

        if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) &&
            (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
            if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
            free(control_un.control);
            return -1;
            }

            memmove(&receivedfd, CMSG_DATA(cmptr), sizeof(receivedfd));
            *fd = receivedfd;
        } else {
            free(control_un.control);
            return -1;
        }

        free(control_un.control);
        return 0;
    }

**接收：OS 特定 IPC (Windows)**

.. code-block:: cpp

    int ipcRecvShareableHandle(HANDLE &handle, HANDLE *shareableHandle) {
        DWORD cbRead;
        ReadFile(handle, shareableHandle, (DWORD)sizeof(*shareableHandles), &cbRead, NULL);
        return 0;
    }

**接收：Fabric IPC**

.. code-block:: cpp

    MPI_Recv(&fh, sizeof(CUmemFabricHandle), MPI_BYTE, 1, 0, MPI_COMM_WORLD);

**导入内存句柄** 同样，用户可以为 OS 特定 IPC 或 fabric 特定 IPC 导入句柄。OS 特定 IPC 句柄只能在单节点上使用。Fabric 特定句柄可用于单节点或多节点。

**OS 特定句柄 (Linux)**

.. code-block:: cpp

    CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    cuMemImportFromShareableHandle(handle, (void*) &fd, handleType);

**Fabric 句柄**

.. code-block:: cpp

    CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_FABRIC;
    cuMemImportFromShareableHandle(handle, (void*) &fh, handleType);

4.16.3.3. 保留和映射
^^^^^^^^^^^^^^^^^^^^

**保留虚拟地址范围**

由于在 VMM 中地址和内存的概念是不同的，应用程序必须划分出一个地址范围，该范围可以容纳由 ``cuMemCreate`` 创建的内存分配。保留的地址范围必须至少与用户计划放置在其中的所有物理内存分配的大小之和一样大。

应用程序可以通过向 ``cuMemAddressReserve`` 传递适当的参数来保留虚拟地址范围。获取的地址范围不会有任何设备或主机物理内存与之关联。保留的虚拟地址范围可以映射到属于系统中任何设备的内存块，从而为应用程序提供连续的 VA 范围，由属于不同设备的内存进行后备和映射。应用程序应使用 ``cuMemAddressFree`` 将虚拟地址范围返回给 CUDA。用户必须确保在调用 ``cuMemAddressFree`` 之前整个 VA 范围已取消映射。这些函数在概念上类似于 Linux 上的 ``mmap`` 和 ``munmap`` 或 Windows 上的 ``VirtualAlloc`` 和 ``VirtualFree`` 。以下代码片段说明了该函数的用法：

.. code-block:: cpp

    CUdeviceptr ptr;
    // `ptr` 保存保留的虚拟地址范围的起始地址。
    CUresult result = cuMemAddressReserve(&ptr, size, 0, 0, 0); // alignment = 0 表示默认对齐

**映射内存**

前两节中分配的物理内存和划分出的虚拟地址空间代表了 VMM API 引入的内存和地址区分。要使分配的内存可用，用户必须将内存映射到地址空间。从 ``cuMemAddressReserve`` 获取的地址范围和从 ``cuMemCreate`` 或 ``cuMemImportFromShareableHandle`` 获取的物理分配必须使用 ``cuMemMap`` 相互关联。

用户可以将来自多个设备的分配关联到驻留在连续虚拟地址范围中，只要他们划分了足够的地址空间。要解耦物理分配和地址范围，用户必须使用 ``cuMemUnmap`` 取消映射地址。用户可以多次将内存映射和取消映射到同一地址范围，只要他们确保不尝试在已映射的 VA 范围保留上创建映射。以下代码片段说明了该函数的用法：

.. code-block:: cpp

    CUdeviceptr ptr;
    // `ptr`：先前由 cuMemAddressReserve 保留的地址范围中的地址。
    // `allocHandle`：先前调用 cuMemCreate 获取的 CUmemGenericAllocationHandle。
    CUresult result = cuMemMap(ptr, size, 0, allocHandle, 0);

4.16.3.4. 访问权限
^^^^^^^^^^^^^^^^^^

CUDA 的虚拟内存管理 API 使应用程序能够使用访问控制机制显式保护其 VA 范围。使用 ``cuMemMap`` 将分配映射到地址范围的某个区域并不会使该地址可访问，如果 CUDA 内核访问它会导致程序崩溃。用户必须专门在源设备和访问设备上使用 ``cuMemSetAccess`` 函数选择访问控制。这允许或限制特定设备对映射地址范围的访问。以下代码片段说明了该函数的用法：

.. code-block:: cpp

    void setAccessOnDevice(int device, CUdeviceptr ptr, size_t size) {
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = device;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        // 使地址可访问
        cuMemSetAccess(ptr, size, &accessDesc, 1);
    }

VMM 公开的访问控制机制允许用户明确指定他们希望与系统中其他对等设备共享哪些分配。如前所述， ``cudaEnablePeerAccess`` 强制将使用 ``cudaMalloc`` 进行的所有先前和未来分配映射到目标对等设备。这在许多情况下很方便，因为用户不必担心跟踪每个分配到系统中每个设备的映射状态。但这种方法 `有性能影响 <https://devblogs.nvidia.com/introducing-low-level-gpu-virtual-memory-management/>`_。通过分配粒度的访问控制，VMM 允许以最小的开销进行对等映射。

`vectorAddMMAP 示例 <https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/vectorAddMMAP>`_ 可用作使用虚拟内存管理 API 的示例。

4.16.3.5. 释放内存
^^^^^^^^^^^^^^^^^^

要释放分配的内存和地址空间，源进程和目标进程都应按顺序使用 cuMemUnmap、cuMemRelease 和 cuMemAddressFree 函数。cuMemUnmap 函数取消映射先前从地址范围映射的内存区域，有效地将物理内存与保留的虚拟地址空间分离。接下来，cuMemRelease 释放先前创建的物理内存，将其返回给系统。最后，cuMemAddressFree 释放先前保留的虚拟地址范围，使其可供将来使用。这个特定顺序确保物理内存和虚拟地址空间的完全和干净释放。

.. code-block:: cpp

    cuMemUnmap(ptr, size);
    cuMemRelease(handle);
    cuMemAddressFree(ptr, size);

.. note::

    在 OS 特定情况下，必须使用 fclose 关闭导出的句柄。此步骤不适用于基于 fabric 的情况。

4.16.4. 多播内存共享
--------------------

`多播对象管理 API <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MULTICAST.html#group__CUDA__MULTICAST/>`_ 为应用程序提供了一种创建多播对象的方法，结合上述 :ref:`虚拟内存管理 API <sec:vmm-api-overview>`，允许应用程序在支持 NVLink SHARP 的 NVLink 连接 GPU（通过 NVSwitch 连接）上利用 NVLink SHARP。NVLink SHARP 允许 CUDA 应用程序利用结构内计算来加速通过 NVSwitch 连接的 GPU 之间的广播和归约等操作。为此，多个 NVLink 连接的 GPU 形成多播团队，团队中的每个 GPU 用物理内存支持多播对象。因此，N 个 GPU 的多播团队有 N 个多播对象的物理副本，每个副本对参与的一个 GPU 是本地的。使用多播对象映射的 `multimem PTX 指令 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red/>`_ 与多播对象的所有副本一起工作。

要使用多播对象，应用程序需要：

- 查询多播支持

- 使用 ``cuMulticastCreate`` 创建多播句柄。

- 与控制应参与多播团队的 GPU 的所有进程共享多播句柄。这使用如上所述的 ``cuMemExportToShareableHandle`` 工作。

- 使用 ``cuMulticastAddDevice`` 添加应参与多播团队的所有 GPU。

- 对于每个参与的 GPU，将如上所述使用 ``cuMemCreate`` 分配的物理内存绑定到多播句柄。在绑定任何设备上的内存之前，需要将所有设备添加到多播团队。

- 保留地址范围，映射多播句柄并设置访问权限，如常规单播映射所述。到同一物理内存的单播和多播映射是可能的。参见上面的 :ref:`虚拟别名支持 <sec:virtual-aliasing-support>` 节，了解如何确保到同一物理内存的多个映射之间的一致性。

- 使用带有 `multimem PTX 指令 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red/>`_ 的多播映射。

`Multi GPU Programming Models <https://github.com/NVIDIA/multi-gpu-programming-models/>`_ GitHub 仓库中的 ``multi_node_p2p`` 示例包含使用 fabric 内存（包括多播对象）利用 NVLink SHARP 的完整示例。请注意，此示例面向 NCCL 或 NVSHMEM 等库的开发者。它展示了 NVSHMEM 等高级编程模型如何在（多节点）NVLink 域内内部工作。应用程序开发者通常应使用更高级别的 MPI、NCCL 或 NVSHMEM 接口而不是此 API。

4.16.4.1. 分配多播对象
^^^^^^^^^^^^^^^^^^^^^^^

可以使用 ``cuMulticastCreate`` 创建多播对象：

.. code-block:: cpp

    CUmemGenericAllocationHandle createMCHandle(int numDevices, size_t size) {
        CUmemAllocationProp mcProp = {};
        mcProp.numDevices = numDevices;
        mcProp.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC; // 或在单节点上使用 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

        size_t granularity = 0;
        cuMulticastGetGranularity(&granularity, &mcProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

        // Ensure size matches granularity requirements for the allocation
        size_t padded_size = ROUND_UP(size, granularity);

        mcProp.size = padded_size;

        // Create Multicast Object this has no devices and no physical memory associated yet
        CUmemGenericAllocationHandle mcHandle;
        cuMulticastCreate(&mcHandle, &mcProp);

        return mcHandle;
    }

4.16.4.2. 向多播对象添加设备
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

可以使用 ``cuMulticastAddDevice`` 向多播团队添加设备：

.. code-block:: cpp

    cuMulticastAddDevice(&mcHandle, device);

此步骤需要在任何设备上的内存绑定到多播对象之前，在控制参与多播团队的设备的所有进程上完成。

4.16.4.3. 将内存绑定到多播对象
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在创建多播对象并将所有参与设备添加到多播对象后，需要为每个设备使用 ``cuMemCreate`` 分配的物理内存来支持它：

.. code-block:: cpp

    cuMulticastBindMem(mcHandle, mcOffset, memHandle, memOffset, size, 0 /*flags*/);

.. _use-multicast-mappings:

4.16.4.4. 使用多播映射
^^^^^^^^^^^^^^^^^^^^^^

要在 CUDA C++ 中使用多播映射，需要使用带内联 PTX 的 `multimem PTX 指令 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red/>`_：

.. code-block:: cpp

    __global__ void all_reduce_norm_barrier_kernel(float* l2_norm,
                                                   float* partial_l2_norm_mc,
                                                   unsigned int* arrival_counter_uc, unsigned int* arrival_counter_mc,
                                                   const unsigned int expected_count) {
        assert( 1 == blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z );
        float l2_norm_sum = 0.0;
    #if __CUDA_ARCH__ >= 900

        // atomic reduction to all replicas
        // this can be conceptually thought of as __threadfence_system(); atomicAdd_system(arrival_counter_mc, 1);
        cuda::ptx::multimem_red(cuda::ptx::release_t, cuda::ptx::scope_sys_t, cuda::ptx::op_add_t, arrival_counter_mc, n);

        // Need a fence between Multicast (mc) and Unicast (uc) access to the same memory `arrival_counter_uc` and `arrival_counter_mc`:
        // - fence.proxy instructions establish an ordering between memory accesses that may happen through different proxies
        // - Value .alias of the .proxykind qualifier refers to memory accesses performed using virtually aliased addresses to the same memory location.
        // from https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar
        cuda::ptx::fence_proxy_alias();

        // spin wait with acquire ordering on UC mapping till all peers have arrived in this iteration
        // Note: all ranks need to reach another barrier after this kernel, such that it is not possible for the barrier to be unblocked by an
        // arrival of a rank for the next iteration if some other rank is slow.
        cuda::atomic_ref<unsigned int,cuda::thread_scope_system> ac(arrival_counter_uc);
        while (expected_count > ac.load(cuda::memory_order_acquire));

        // Atomic load reduction from all replicas. It does not provide ordering so it can be relaxed.
        asm volatile ("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : "=f"(l2_norm_sum) : "l"(partial_l2_norm_mc) : "memory");

    #else
        #error "ERROR: multimem instructions require compute capability 9.0 or larger."
    #endif

        *l2_norm = std::sqrt(l2_norm_sum);
    }

.. _advanced-configuration-vmm:

4.16.5. 高级配置
----------------

4.16.5.1. 内存类型
^^^^^^^^^^^^^^^^^^

VMM 还为应用程序提供了一种机制来分配某些设备可能支持的特殊类型内存。使用 ``cuMemCreate`` ，应用程序可以使用 ``CUmemAllocationProp::allocFlags`` 指定内存类型要求，以选择特定的内存功能。应用程序必须确保请求的内存类型受设备支持。

4.16.5.2. 可压缩内存
^^^^^^^^^^^^^^^^^^^^

可压缩内存可用于加速对具有非结构化稀疏性和其他可压缩数据模式的数据的访问。压缩可以节省 DRAM 带宽、L2 读取带宽和 L2 容量，具体取决于数据。想要在支持计算数据压缩的设备上分配可压缩内存的应用程序可以通过将 ``CUmemAllocationProp::allocFlags::compressionType`` 设置为 ``CU_MEM_ALLOCATION_COMP_GENERIC`` 来实现。用户必须使用 ``CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED`` 查询设备是否支持计算数据压缩。以下代码片段说明如何使用 ``cuDeviceGetAttribute`` 查询可压缩内存支持。

.. code-block:: cpp

    int compressionSupported = 0;
    cuDeviceGetAttribute(&compressionSupported, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, device);

在支持计算数据压缩的设备上，用户必须在分配时选择加入，如下所示：

.. code-block:: cpp

    prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

由于各种原因（如硬件资源有限），分配可能没有压缩属性。要验证标志是否生效，用户可以使用 ``cuMemGetAllocationPropertiesFromHandle`` 查询已分配内存的属性。

.. code-block:: cpp

    CUmemAllocationProp allocationProp = {};
    cuMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandle);

    if (allocationProp.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC)
    {
        // 获得了可压缩内存分配
    }

.. _virtual-aliasing-support:

4.16.5.3. 虚拟别名支持
^^^^^^^^^^^^^^^^^^^^^^

虚拟内存管理 API 提供了一种方法，通过使用不同虚拟地址多次调用 ``cuMemMap`` 来创建同一分配的多个虚拟内存映射或"代理"。这称为虚拟别名。除非 PTX ISA 中另有说明，否则对分配的一个代理的写入被认为与同一内存的任何其他代理不一致且不连贯，直到写入设备操作（网格启动、memcpy、memset 等）完成。在写入设备操作之前存在于 GPU 上但在写入设备操作完成后读取的网格也被认为具有不一致和不连贯的代理。

例如，以下片段被认为是未定义的，假设设备指针 A 和 B 是同一内存分配的虚拟别名：

.. code-block:: cpp

    __global__ void foo(char *A, char *B) {
      *A = 0x1;
      printf("%d\n", *B);    // Undefined behavior!  *B can take on either
    // the previous value or some value in-between.
    }

以下是被定义的行为，假设这两个内核按顺序排列（通过流或事件）。

.. code-block:: cpp

    __global__ void foo1(char *A) {
      *A = 0x1;
    }

    __global__ void foo2(char *B) {
      printf("%d\n", *B);    // *B == *A == 0x1 assuming foo2 waits for foo1
    // to complete before launching
    }

    cudaMemcpyAsync(B, input, size, stream1);    // Aliases are allowed at
    // operation boundaries
    foo1<<<1,1,0,stream1>>>(A);                  // allowing foo1 to access A.
    cudaEventRecord(event, stream1);
    cudaStreamWaitEvent(stream2, event);
    foo2<<<1,1,0,stream2>>>(B);
    cudaStreamWaitEvent(stream3, event);
    cudaMemcpyAsync(output, B, size, stream3);  // Both launches of foo2 and
                                                // cudaMemcpy (which both
                                                // read) wait for foo1 (which writes)
                                                // to complete before proceeding

如果在同一内核中需要通过不同"代理"访问同一分配，可以在两次访问之间使用 ``fence.proxy.alias`` 。因此，上面的示例可以通过内联 PTX 汇编使其合法：

.. code-block:: cpp

    __global__ void foo(char *A, char *B) {
      *A = 0x1;
      cuda::ptx::fence_proxy_alias();
      printf("%d\n", *B);    // *B == *A == 0x1
    }

4.16.5.4. IPC 的 OS 特定句柄详情
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用 ``cuMemCreate`` ，用户可以在分配时指示他们已将特定分配指定用于进程间通信或图形互操作目的。应用程序可以通过将 ``CUmemAllocationProp::requestedHandleTypes`` 设置为平台特定字段来实现此目的。在 Windows 上，当 ``CUmemAllocationProp::requestedHandleTypes`` 设置为 ``CU_MEM_HANDLE_TYPE_WIN32`` 时，应用程序还必须在 ``CUmemAllocationProp::win32HandleMetaData`` 中指定 LPSECURITYATTRIBUTES 属性。此安全属性定义导出分配可能传输到其他进程的范围。

用户必须确保在尝试导出使用 ``cuMemCreate`` 分配的内存之前查询请求的句柄类型的支持。以下代码片段以平台特定的方式说明如何查询句柄类型支持。

.. code-block:: cpp

    int deviceSupportsIpcHandle;
    #if defined(__linux__)
        cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
    #else
        cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, device));
    #endif

用户应按如下所示适当设置 ``CUmemAllocationProp::requestedHandleTypes`` ：

.. code-block:: cpp

    #if defined(__linux__)
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    #else
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;
        prop.win32HandleMetaData = // Windows specific LPSECURITYATTRIBUTES attribute.
    #endif