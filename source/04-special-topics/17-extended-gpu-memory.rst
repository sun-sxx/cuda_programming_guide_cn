.. _extended-gpu-memory-details:

扩展 GPU 内存
=============

扩展 GPU 内存（Extended GPU Memory，EGM）功能利用高带宽 NVLink-C2C，使 GPU 能够高效访问所有系统内存，无论是单节点还是多节点系统。EGM 适用于集成 CPU-GPU NVIDIA 系统，允许分配可被系统中任何 GPU 线程访问的物理内存。EGM 确保所有 GPU 都能以 GPU-GPU NVLink 或 NVLink-C2C 的速度访问其资源。

.. figure:: /_static/images/egm-c2c-intro.png
   :align: center
   :alt: EGM 架构图
   :width: 800px

   EGM 架构示意图

在此配置中，内存访问通过本地高带宽 NVLink-C2C 进行。对于远程内存访问，使用 GPU NVLink，在某些情况下使用 NVLink-C2C。通过 EGM，GPU 线程能够通过 NVSwitch fabric 访问所有可用的内存资源，包括 CPU 附加内存和 HBM3。

.. _egm-preliminaries:

基础知识
--------

在深入了解 EGM 功能的 API 变更之前，我们将介绍当前支持的拓扑结构、标识符分配、虚拟内存管理的先决条件以及 EGM 的 CUDA 类型。

.. _egm-platforms:

EGM 平台：系统拓扑
~~~~~~~~~~~~~~~~~~

目前，EGM 可以在以下平台上启用：

**(1) 单节点、单 GPU**：由基于 Arm 的 CPU、CPU 附加内存和一个 GPU 组成。CPU 和 GPU 之间有高带宽 C2C（Chip-to-Chip）互连。

**(2) 单节点、多 GPU**：由基于 ARM 的 CPU 组成，每个 CPU 都有附加内存，多个 GPU 通过基于 NVLink 的网络连接。

**(3) 多节点、多 GPU**：两个或更多单节点系统，每个系统如上述 (1) 或 (2) 所述，通过基于 NVLink 的网络连接。

.. note::

   使用 ``cgroups`` 限制可用设备将阻止通过 EGM 进行路由并导致性能问题。请改用 ``CUDA_VISIBLE_DEVICES`` 。

.. _socket-identifiers:

Socket 标识符：是什么？如何访问？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NUMA（Non-Uniform Memory Access，非统一内存访问）是多处理器计算机系统中使用的内存架构，其中内存被划分为多个节点。每个节点都有自己的处理器和内存。在这样的系统中，NUMA 将系统划分为节点，并为每个节点分配唯一的标识符（ ``numaID`` ）。

EGM 使用由操作系统分配的 NUMA 节点标识符。请注意，此标识符与设备的序号不同，它与最近的主机节点相关联。除了现有方法外，用户可以通过调用 ``cuDeviceGetAttribute`` 并使用 ``CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID`` 属性类型来获取主机节点的标识符（ ``numaID`` ），如下所示：

.. code-block:: c++

   int numaId;
   cuDeviceGetAttribute(&numaId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, deviceOrdinal);

.. _allocators-egm-support:

分配器与 EGM 支持
~~~~~~~~~~~~~~~~~

将系统内存映射为 EGM 不会导致任何性能问题。实际上，访问映射为 EGM 的远程 socket 系统内存会更快。因为通过 EGM，流量保证通过 NVLinks 路由。目前， ``cuMemCreate`` 和 ``cudaMemPoolCreate`` 分配器支持使用适当的位置类型和 NUMA 标识符。

.. _memory-mgmt-extensions:

当前 API 的内存管理扩展
~~~~~~~~~~~~~~~~~~~~~~~

目前，EGM 内存可以使用虚拟内存（ ``cuMemCreate`` ）或流有序内存（ ``cudaMemPoolCreate`` ）分配器进行映射。用户负责分配物理内存并将其映射到所有 socket 上的虚拟内存地址空间。

.. note::

   多节点、多 GPU 平台需要进程间通信。因此，我们建议读者参阅 :ref:`sec:interprocess-communication`。

.. note::

   我们建议读者阅读 CUDA 编程指南的 :ref:`sec:virtual-memory-management` 和 :ref:`sec:stream-ordered-memory-allocator` 以获得更好的理解。

新的 CUDA 属性类型已添加到 API 中，以允许这些方法使用类似 NUMA 的节点标识符来理解分配位置：

.. list-table::
   :header-rows: 1

   * - **CUDA 类型**
     - **用于**
   * - ``CU_MEM_LOCATION_TYPE_HOST_NUMA``
     - ``CUmemAllocationProp`` ，用于 ``cuMemCreate``
   * - ``cudaMemLocationTypeHostNuma``
     - ``cudaMemPoolProps`` ，用于 ``cudaMemPoolCreate``

.. note::

   请参阅 `CUDA Driver API <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html>`__ 和 `CUDA Runtime Data Types <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html>`__ 以了解更多关于 NUMA 相关的 CUDA 类型。

.. _using-egm-interface:

使用 EGM 接口
-------------

.. _egm-single-node-single-gpu:

单节点、单 GPU
~~~~~~~~~~~~~~

任何现有的 CUDA 主机分配器以及系统分配的内存都可以用来受益于高带宽 C2C。对于用户来说，本地访问就是今天的主机分配。

.. note::

   有关内存分配器和页面大小的更多信息，请参阅调优指南。

.. _egm-single-node-multi-gpu:

单节点、多 GPU
~~~~~~~~~~~~~~

在多 GPU 系统中，用户必须提供主机信息用于放置。如前所述，表达该信息的一种自然方式是使用 NUMA 节点 ID，EGM 遵循这种方法。因此，使用 ``cuDeviceGetAttribute`` 函数，用户应该能够获知最近的 NUMA 节点 ID（参见 :ref:`sec:socket-identifiers` ）。然后用户可以使用 VMM（虚拟内存管理）API 或 CUDA 内存池来分配和管理 EGM 内存。

.. _using-vmm-apis:

使用 VMM API
^^^^^^^^^^^^

使用虚拟内存管理 API 分配内存的第一步是创建一个物理内存块，为分配提供后备存储。有关更多详细信息，请参阅 CUDA 编程指南的虚拟内存管理部分。在 EGM 分配中，用户必须显式提供 ``CU_MEM_LOCATION_TYPE_HOST_NUMA`` 作为位置类型，并提供 ``numaID`` 作为位置标识符。此外，在 EGM 中，分配必须与平台的适当粒度对齐。以下代码片段展示了使用 ``cuMemCreate`` 分配物理内存：

.. code-block:: c++

   CUmemAllocationProp prop{};
   prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
   prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
   prop.location.id = numaId;
   size_t granularity = 0;
   cuMemGetAllocationGranularity(&granularity, &prop, MEM_ALLOC_GRANULARITY_MINIMUM);
   size_t padded_size = ROUND_UP(size, granularity);
   CUmemGenericAllocationHandle allocHandle;
   cuMemCreate(&allocHandle, padded_size, &prop, 0);

分配物理内存后，我们需要保留地址空间并将其映射到指针。这些过程没有 EGM 特定的变更：

.. code-block:: c++

   CUdeviceptr dptr;
   cuMemAddressReserve(&dptr, padded_size, 0, 0, 0);
   cuMemMap(dptr, padded_size, 0, allocHandle, 0);

最后，用户必须显式保护映射的虚拟地址范围。否则，访问映射空间将导致崩溃。与内存分配类似，用户必须提供 ``CU_MEM_LOCATION_TYPE_HOST_NUMA`` 作为位置类型，并提供 ``numaId`` 作为位置标识符。以下代码片段为主机节点和 GPU 创建访问描述符，为两者提供对映射内存的读写访问：

.. code-block:: c++

   CUmemAccessDesc accessDesc[2]{{}};
   accessDesc[0].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
   accessDesc[0].location.id = numaId;
   accessDesc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
   accessDesc[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
   accessDesc[1].location.id = currentDev;
   accessDesc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
   cuMemSetAccess(dptr, size, accessDesc, 2);

.. _using-cuda-memory-pool:

使用 CUDA 内存池
^^^^^^^^^^^^^^^^

要定义 EGM，用户可以在节点上创建内存池并授予对等方访问权限。在这种情况下，用户必须显式定义 ``cudaMemLocationTypeHostNuma`` 作为位置类型，并将 ``numaId`` 作为位置标识符。以下代码片段展示了使用 ``cudaMemPoolCreate`` 创建内存池：

.. code-block:: c++

   cudaSetDevice(homeDevice);
   cudaMemPoolProps props{};
   props.allocType = cudaMemAllocationTypePinned;
   props.location.type = cudaMemLocationTypeHostNuma;
   props.location.id = numaId;
   cudaMemPoolCreate(&memPool, &props);

此外，对于直连对等访问，也可以使用现有的对等访问 API ``cudaMemPoolSetAccess`` 。以下代码片段展示了 ``accessingDevice`` 的示例：

.. code-block:: c++

   cudaMemAccessDesc desc{};
   desc.flags = cudaMemAccessFlagsProtReadWrite;
   desc.location.type = cudaMemLocationTypeDevice;
   desc.location.id = accessingDevice;
   cudaMemPoolSetAccess(memPool, &desc, 1);

当内存池创建完成并授予访问权限后，用户可以将创建的内存池设置为 ``residentDevice`` ，并开始使用 ``cudaMallocAsync`` 分配内存：

.. code-block:: c++

   cudaDeviceSetMemPool(residentDevice, memPool);
   cudaMallocAsync(&ptr, size, memPool, stream);

.. note::

   EGM 使用 2MB 页面映射。因此，用户在访问非常大的分配时可能会遇到更多 TLB 未命中。

.. _egm-multi-node-multi-gpu:

多节点、多 GPU
~~~~~~~~~~~~~~

除了内存分配之外，远程对等访问没有 EGM 特定的修改，它遵循 CUDA 进程间（IPC）协议。有关 IPC 的更多详细信息，请参阅 CUDA 编程指南。

用户应该使用 ``cuMemCreate`` 分配内存，并且同样必须显式提供 ``CU_MEM_LOCATION_TYPE_HOST_NUMA`` 作为位置类型，并提供 ``numaID`` 作为位置标识符。此外，应将 ``CU_MEM_HANDLE_TYPE_FABRIC`` 定义为请求的句柄类型。以下代码片段展示了在节点 A 上分配物理内存：

.. code-block:: c++

   CUmemAllocationProp prop{};
   prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
   prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
   prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
   prop.location.id = numaId;
   size_t granularity = 0;
   cuMemGetAllocationGranularity(&granularity, &prop,
                                 MEM_ALLOC_GRANULARITY_MINIMUM);
   size_t padded_size = ROUND_UP(size, granularity);
   size_t page_size = ...;
   assert(padded_size % page_size == 0);
   CUmemGenericAllocationHandle allocHandle;
   cuMemCreate(&allocHandle, padded_size, &prop, 0);

使用 ``cuMemCreate`` 创建分配句柄后，用户可以调用 ``cuMemExportToShareableHandle`` 将该句柄导出到另一个节点（节点 B）：

.. code-block:: c++

   cuMemExportToShareableHandle(&fabricHandle, allocHandle,
                                CU_MEM_HANDLE_TYPE_FABRIC, 0);
   // 此时，fabricHandle 应通过 TCP/IP 发送到节点 B。

在节点 B 上，可以使用 ``cuMemImportFromShareableHandle`` 导入句柄，并将其视为任何其他 fabric 句柄：

.. code-block:: c++

   // 此时，fabricHandle 应通过 TCP/IP 从节点 A 接收。
   CUmemGenericAllocationHandle allocHandle;
   cuMemImportFromShareableHandle(&allocHandle, &fabricHandle,
                                  CU_MEM_HANDLE_TYPE_FABRIC);

当句柄在节点 B 上导入后，用户可以保留地址空间并以常规方式在本地进行映射：

.. code-block:: c++

   size_t granularity = 0;
   cuMemGetAllocationGranularity(&granularity, &prop,
                                 MEM_ALLOC_GRANULARITY_MINIMUM);
   size_t padded_size = ROUND_UP(size, granularity);
   size_t page_size = ...;
   assert(padded_size % page_size == 0);
   CUdeviceptr dptr;
   cuMemAddressReserve(&dptr, padded_size, 0, 0, 0);
   cuMemMap(dptr, padded_size, 0, allocHandle, 0);

作为最后一步，用户应该为节点 B 上的每个本地 GPU 授予适当的访问权限。以下代码片段为八个本地 GPU 授予读写访问权限：

.. code-block:: c++

   // 为所有 8 个本地 GPU 授予对位于节点 A 的导出 EGM 内存的访问权限。
   CUmemAccessDesc accessDesc[8];
   for (int i = 0; i < 8; i++) {
      accessDesc[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      accessDesc[i].location.id = i;
      accessDesc[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
   }
   cuMemSetAccess(dptr, size, accessDesc, 8);