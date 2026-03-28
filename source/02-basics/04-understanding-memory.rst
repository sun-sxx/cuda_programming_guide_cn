.. _unified-system-memory:

2.4. 统一内存和系统内存
========================

异构系统拥有多个可以存储数据的物理内存。主机 CPU 连接有 DRAM，系统中的每个 GPU 都有其自己的 DRAM。当数据驻留在访问它的处理器的内存中时，性能最佳。CUDA 提供了 API 来 :ref:`显式管理内存位置 <sec:intro-cpp-explicit-memory-management>`，但这可能很冗长并使软件设计复杂化。CUDA 提供了一些功能和特性，旨在简化不同物理内存之间的数据分配、放置和迁移。

本章的目的是介绍和解释这些功能，以及它们对应用程序开发人员在功能和性能方面的意义。统一内存有几种不同的表现形式，这取决于操作系统、驱动程序版本和使用的 GPU。本章将展示如何确定适用的统一内存范式以及统一内存功能在每种情况下的行为。后续的 :ref:`统一内存章节 <sec:um-details-intro>` 将更详细地解释统一内存。

本章将定义和解释以下概念：

- :ref:`统一虚拟地址空间 <sec:memory-unified-virtual-address-space>` - CPU 内存和每个 GPU 的内存在单个虚拟地址空间中具有不同的范围

- :ref:`统一内存 <sec:memory-unified-memory>` - 一种 CUDA 功能，支持可在 CPU 和 GPU 之间自动迁移的托管内存

  - :ref:`有限统一内存支持 <sec:memory-limited-unified-memory-support>` - 具有一些限制的统一内存范式
  
  - :ref:`完整统一内存支持 <sec:memory-unified-memory-full>` - 完整支持统一内存功能
  
  - :ref:`具有硬件一致性的完整统一内存 <sec:memory-unified-address-translation-services>` - 使用硬件功能完整支持统一内存
  
  - :ref:`统一内存提示 <sec:memory-mem-advise-prefetch>` - 用于指导特定分配的统一内存行为的 API
  

- :ref:`页锁定主机内存 <sec:memory-page-locked-host-memory>` - 不可分页的系统内存，某些 CUDA 操作需要它

  - :ref:`映射内存 <sec:memory-mapped-memory>` - 一种直接从 kernel 访问主机内存的机制（不同于统一内存）
  

此外，这里还介绍了讨论统一内存和系统内存时使用的以下术语：

- :ref:`异构托管内存 <sec:memory-heterogeneous-memory-management>` (HMM) - Linux 内核的一项功能，为完整统一内存提供软件一致性

- :ref:`地址转换服务 <sec:memory-unified-address-translation-services>` (ATS) - 一种硬件功能，当 GPU 通过 NVLink 芯片到芯片 (C2C) 互连连接到 CPU 时可用，为完整统一内存提供硬件一致性

.. _memory-unified-virtual-address-space:

2.4.1. 统一虚拟地址空间
------------------------

在单个操作系统进程内，所有主机内存和系统中所有 GPU 上的所有全局内存使用单个虚拟地址空间。主机和所有设备上的所有内存分配都位于此虚拟地址空间中。无论使用 CUDA API（如 ``cudaMalloc``、``cudaMallocHost`` ）还是系统分配 API（如 ``new``、``malloc``、``mmap`` ）进行分配，这都是成立的。CPU 和每个 GPU 在统一虚拟地址空间中都有唯一的范围。

这意味着：

- 任何内存的位置（即在 CPU 还是哪个 GPU 的内存中）可以通过 ``cudaPointerGetAttributes()`` 从指针的值来确定

- ``cudaMemcpy*()`` 的 ``cudaMemcpyKind`` 参数可以设置为 ``cudaMemcpyDefault`` ，以自动从指针确定拷贝类型

.. _memory-unified-memory:

2.4.2. 统一内存
---------------

*统一内存* 是一种 CUDA 内存功能，允许在 CPU 或 GPU 上运行的代码访问称为 *托管内存* 的内存分配。统一内存已在 :ref:`CUDA C++ 简介 <sec:intro-cpp-unified-memory>` 中展示。统一内存适用于 CUDA 支持的所有系统。

在某些系统上，托管内存必须显式分配。在 CUDA 中可以通过几种不同的方式显式分配托管内存：

- CUDA API ``cudaMallocManaged``

- CUDA API ``cudaMallocFromPoolAsync`` ，使用 ``allocType`` 设置为 ``cudaMemAllocationTypeManaged`` 创建的池

- 带有 ``__managed__`` 说明符的全局变量（参见 :ref:`内存空间说明符 <sec:memory-space-specifiers>` ）

在具有 :ref:`HMM <sec:memory-heterogeneous-memory-management>` 或 :ref:`ATS <sec:memory-unified-address-translation-services>` 的系统上，所有系统内存都是隐式托管内存，无论其分配方式如何。不需要特殊分配。

.. _unified-memory-paradigms:

2.4.2.1. 统一内存范式
~~~~~~~~~~~~~~~~~~~~~~

统一内存的功能和行为因操作系统、Linux 内核版本、GPU 硬件和 GPU-CPU 互连而异。可以通过使用 ``cudaDeviceGetAttribute`` 查询几个属性来确定可用的统一内存形式：

- ``cudaDevAttrConcurrentManagedAccess`` - 1 表示完整统一内存支持，0 表示有限支持

- ``cudaDevAttrPageableMemoryAccess`` - 1 表示所有系统内存都是完全支持的统一内存，0 表示只有显式分配为托管内存的内存才是完全支持的统一内存

- ``cudaDevAttrPageableMemoryAccessUsesHostPageTables`` - 指示 CPU/GPU 一致性的机制：1 表示硬件，0 表示软件

:numref:`fig:unified-memory-flow-chart` 以图形方式展示了如何确定统一内存范式，后面跟着实现相同逻辑的 :ref:`代码示例 <sec:memory-unified-querying-code>`。

统一内存操作有四种范式：

- :ref:`显式托管内存分配的完整支持 <sec:memory-unified-memory-full>`

- :ref:`具有软件一致性的所有分配的完整支持 <sec:memory-unified-memory-full>`

- :ref:`具有硬件一致性的所有分配的完整支持 <sec:memory-unified-address-translation-services>`

- :ref:`有限统一内存支持 <sec:memory-limited-unified-memory-support>`

当完整支持可用时，它可能需要显式分配，或者所有系统内存可能隐式为统一内存。当所有内存都是隐式统一时，一致性机制可以是软件或硬件。Windows 和某些 Tegra 设备对统一内存的支持有限。

.. figure:: /_static/images/unified-memory-explainer.png
   :align: center
   :alt: 统一内存范式流程图
   :name: fig:unified-memory-flow-chart

   所有当前的 GPU 都使用统一虚拟地址空间并具有可用的统一内存。当 ``cudaDevAttrConcurrentManagedAccess`` 为 1 时，完整统一内存支持可用，否则只有有限支持可用。当完整支持可用时，如果 ``cudaDevAttrPageableMemoryAccess`` 也为 1，则所有系统内存都是统一内存。否则，只有使用 CUDA API（如 ``cudaMallocManaged`` ）分配的内存才是统一内存。当所有系统内存都是统一内存时， ``cudaDevAttrPageableMemoryAccessUsesHostPageTables`` 指示一致性是由硬件提供（值为 1 时）还是软件提供（值为 0 时）。

:numref:`tab:unified-memory-levels` 以表格形式显示了与 :numref:`fig:unified-memory-flow-chart` 相同的信息，并附有指向本章相关部分和本指南后续部分更完整文档的链接。

.. list-table:: 统一内存范式概述
   :name: tab:unified-memory-levels
   :header-rows: 1
   :widths: 35 35 30

   * - 统一内存范式
     - 设备属性
     - 完整文档
   * - :ref:`有限统一内存支持 <sec:memory-limited-unified-memory-support>`
     - ``cudaDevAttrConcurrentManagedAccess`` 为 0
     - :ref:`Windows、WSL 和 Tegra 上的统一内存 <sec:um-legacy-devices>`
   * - :ref:`显式托管内存分配的完整支持 <sec:memory-unified-memory-full>`
     - ``cudaDevAttrPageableMemoryAccess`` 为 0 且 ``cudaDevAttrConcurrentManagedAccess`` 为 1
     - :ref:`仅有 CUDA 托管内存支持的设备上的统一内存 <sec:um-no-pageable-systems>`
   * - :ref:`具有软件一致性的所有分配的完整支持 <sec:memory-unified-memory-full>`
     - ``cudaDevAttrPageableMemoryAccessUsesHostPageTables`` 为 0 且 ``cudaDevAttrPageableMemoryAccess`` 为 1 且 ``cudaDevAttrConcurrentManagedAccess`` 为 1
     - :ref:`具有完整 CUDA 统一内存支持的设备上的统一内存 <sec:um-pageable-systems>`
   * - :ref:`具有硬件一致性的所有分配的完整支持 <sec:memory-unified-address-translation-services>`
     - ``cudaDevAttrPageableMemoryAccessUsesHostPageTables`` 为 1 且 ``cudaDevAttrPageableMemoryAccess`` 为 1 且 ``cudaDevAttrConcurrentManagedAccess`` 为 1
     - :ref:`具有完整 CUDA 统一内存支持的设备上的统一内存 <sec:um-pageable-systems>`

.. _unified-memory-paradigm-code-example:

2.4.2.1.1. 统一内存范式：代码示例
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下代码示例演示了查询设备属性并确定统一内存范式，遵循 :numref:`fig:unified-memory-flow-chart` 的逻辑，适用于系统中的每个 GPU。

.. code-block:: cuda
   :caption: 查询统一内存范式
   :linenos:

   void queryDevices()
   {
       int numDevices = 0;
       cudaGetDeviceCount(&numDevices);
       for(int i=0; i<numDevices; i++)
       {
           cudaSetDevice(i);
           cudaInitDevice(0, 0, 0);
           int deviceId = i;

           int concurrentManagedAccess = -1;     
           cudaDeviceGetAttribute (&concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, deviceId);    
           int pageableMemoryAccess = -1;
           cudaDeviceGetAttribute (&pageableMemoryAccess, cudaDevAttrPageableMemoryAccess, deviceId);
           int pageableMemoryAccessUsesHostPageTables = -1;
           cudaDeviceGetAttribute (&pageableMemoryAccessUsesHostPageTables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, deviceId);

           printf("Device %d has ", deviceId);
           if(concurrentManagedAccess){
               if(pageableMemoryAccess){
                   printf("full unified memory support");
                   if( pageableMemoryAccessUsesHostPageTables)
                       { printf(" with hardware coherency\n");  }
                   else
                       { printf(" with software coherency\n"); }
               }
               else
                   { printf("full unified memory support for CUDA-made managed allocations\n"); }
           }
           else
           {   printf("limited unified memory support: Windows, WSL, or Tegra\n");  }
       }
   }

.. _memory-unified-memory-full:

2.4.2.2. 完整统一内存功能支持
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

大多数 Linux 系统都有完整的统一内存支持。如果设备属性 ``cudaDevAttrPageableMemoryAccess`` 为 1，则所有系统内存，无论是由 CUDA API 还是系统 API 分配，都以完整功能支持作为统一内存运行。这包括使用 ``mmap`` 创建的文件支持的内存分配。

如果 ``cudaDevAttrPageableMemoryAccess`` 为 0，则只有由 CUDA 分配为托管内存的内存才作为统一内存运行。使用系统 API 分配的内存不是托管的，也不一定可以从 GPU kernel 访问。

通常，对于具有完整支持的统一分配：

- 托管内存通常分配在首次访问它的处理器的内存空间中

- 当托管内存被当前驻留处理器以外的处理器使用时，通常会迁移

- 托管内存以内存页（软件一致性）或缓存行（硬件一致性）的粒度进行迁移或访问

- 允许超额订阅：应用程序可以分配比 GPU 物理可用更多的托管内存

分配和迁移行为可能偏离上述情况。程序员可以使用 :ref:`提示和预取 <sec:memory-mem-advise-prefetch>` 来影响这些行为。完整统一内存支持的完整覆盖可以在 :ref:`具有完整 CUDA 统一内存支持的设备上的统一内存 <sec:um-pageable-systems>` 中找到。

.. _memory-unified-address-translation-services:

2.4.2.2.1. 具有硬件一致性的完整统一内存
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 Grace Hopper 和 Grace Blackwell 等硬件上，使用 NVIDIA CPU 且 CPU 和 GPU 之间的互连是 NVLink 芯片到芯片 (C2C) 时，地址转换服务 (ATS) 可用。当 ATS 可用时， ``cudaDevAttrPageableMemoryAccessUsesHostPageTables`` 为 1。

使用 ATS，除了对所有主机分配的完整统一内存支持外：

- GPU 分配（例如 ``cudaMalloc`` ）可以从 CPU 访问（ ``cudaDevAttrDirectManagedMemAccessFromHost`` 将为 1）

- CPU 和 GPU 之间的链路支持原生原子操作（ ``cudaDevAttrHostNativeAtomicSupported`` 将为 1）

- 硬件一致性支持可以提高性能，相比软件一致性

ATS 提供 :ref:`HMM <sec:memory-heterogeneous-memory-management>` 的所有功能。当 ATS 可用时，HMM 自动禁用。关于硬件与软件一致性的进一步讨论可在 :ref:`CPU 和 GPU 页表：硬件一致性与软件一致性 <sec:um-hw-coherency>` 中找到。

.. _memory-heterogeneous-memory-management:

2.4.2.2.2. HMM - 具有软件一致性的完整统一内存
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*异构内存管理* (HMM) 是 Linux 操作系统（具有适当内核版本）上可用的功能，它支持软件一致的 :ref:`完整统一内存支持 <sec:memory-unified-memory-full>`。异构内存管理将 ATS 提供的一些功能和便利性带到 PCIe 连接的 GPU。

在 Linux 上，至少具有 Linux 内核 6.1.24、6.2.11 或 6.3 或更高版本，异构内存管理 (HMM) 可能可用。以下命令可用于查找寻址模式是否为 ``HMM`` 。

.. code-block:: bash
   :caption: 检查 HMM 寻址模式

   $ nvidia-smi -q | grep Addressing
   Addressing Mode : HMM

当 HMM 可用时，支持 :ref:`完整统一内存 <sec:memory-unified-memory-full>`，所有系统分配都是隐式统一内存。如果系统还具有 :ref:`ATS <sec:memory-unified-address-translation-services>`，则 HMM 被禁用并使用 ATS，因为 ATS 提供 HMM 的所有功能以及更多功能。

.. _memory-limited-unified-memory-support:

2.4.2.3. 有限统一内存支持
~~~~~~~~~~~~~~~~~~~~~~~~~

在 Windows（包括适用于 Linux 的 Windows 子系统 (WSL)）和某些 Tegra 系统上，统一内存功能的有限子集可用。在这些系统上，托管内存可用，但在 CPU 和 GPU 之间的迁移行为不同。

- 托管内存首先分配在 CPU 的物理内存中

- 托管内存以比虚拟内存页更大的粒度迁移

- 当 GPU 开始执行时，托管内存迁移到 GPU

- 当 GPU 处于活动状态时，CPU 不能访问托管内存

- 当 GPU 同步时，托管内存迁移回 CPU

- 不允许 GPU 内存超额订阅

- 只有 CUDA 显式分配为托管内存的内存才是统一的

此范式的完整覆盖可以在 :ref:`Windows、WSL 和 Tegra 上的统一内存 <sec:um-legacy-devices>` 中找到。

.. _memory-mem-advise-prefetch:

2.4.2.4. 内存建议和预取
~~~~~~~~~~~~~~~~~~~~~~~

程序员可以向管理统一内存的 NVIDIA 驱动程序提供提示，以帮助其最大化应用程序性能。CUDA API ``cudaMemAdvise`` 允许程序员指定影响分配放置位置以及从其他设备访问时内存是否迁移的分配属性。

``cudaMemPrefetchAsync`` 允许程序员建议开始将特定分配异步迁移到不同位置。一个常见的用途是在 kernel 启动之前开始传输 kernel 将使用的数据。这使数据拷贝可以在其他 GPU kernel 执行时发生。

:ref:`性能提示 <sec:um-perf-hints>` 部分涵盖了可以传递给 ``cudaMemAdvise`` 的不同提示，并展示了使用 ``cudaMemPrefetchAsync`` 的示例。

.. _memory-page-locked-host-memory:

2.4.3. 页锁定主机内存
---------------------

在 :ref:`介绍性代码示例 <sec:intro-cuda-cpp-all-together>` 中，使用 ``cudaMallocHost`` 在 CPU 上分配内存。这在主机上分配 *页锁定* 内存（也称为 *固定* 内存）。通过传统分配机制（如 ``malloc``、``new`` 或 ``mmap`` ）进行的主机分配不是页锁定的，这意味着它们可能被交换到磁盘或被操作系统物理重定位。

页锁定主机内存是 :ref:`CPU 和 GPU 之间异步拷贝 <sec:async-execution-memory-transfers>` 所必需的。页锁定主机内存也提高同步拷贝的性能。页锁定内存可以 :ref:`映射 <sec:memory-mapped-memory>` 到 GPU 以便从 GPU kernel 直接访问。

CUDA 运行时提供 API 来分配页锁定主机内存或页锁定现有分配：

- ``cudaMallocHost`` 分配页锁定主机内存

- ``cudaHostAlloc`` 默认与 ``cudaMallocHost`` 行为相同，但也接受标志来指定其他内存参数

- ``cudaFreeHost`` 释放用 ``cudaMallocHost`` 或 ``cudaHostAlloc`` 分配的内存

- ``cudaHostRegister`` 页锁定 CUDA API 外部分配的现有内存范围，例如使用 ``malloc`` 或 ``mmap`` 分配的内存

``cudaHostRegister`` 使第三方库或开发人员无法控制的其他代码分配的主机内存能够被页锁定，以便可以在异步拷贝或映射中使用。

.. note::

   页锁定主机内存可用于系统中所有 GPU 的异步拷贝和映射内存。

   页锁定主机内存在非 I/O 一致的 Tegra 设备上不被缓存。此外， ``cudaHostRegister()`` 在非 I/O 一致的 Tegra 设备上不受支持。

.. _memory-mapped-memory:

2.4.3.1. 映射内存
~~~~~~~~~~~~~~~~~

在具有 :ref:`HMM <sec:memory-heterogeneous-memory-management>` 或 :ref:`ATS <sec:memory-unified-address-translation-services>` 的系统上，所有主机内存都可以使用主机指针直接从 GPU 访问。当 ATS 或 HMM 不可用时，可以通过将内存 *映射* 到 GPU 的内存空间来使主机分配可被 GPU 访问。映射内存始终是页锁定的。

下面的代码示例将演示以下数组拷贝 kernel 直接对映射的主机内存进行操作。

.. code-block:: cuda
   :caption: 数组拷贝 kernel

   __global__ void copyKernel(float* a, float* b)
   {
           int idx = threadIdx.x + blockDim.x * blockIdx.x;
           a[idx] = b[idx];
   }

虽然映射内存在某些未拷贝到 GPU 的特定数据需要从 kernel 访问的情况下可能有用，但在 kernel 中访问映射内存需要通过 CPU-GPU 互连（PCIe 或 NVLink C2C）进行事务。与访问设备内存相比，这些操作具有更高的延迟和更低的带宽。对于 kernel 的大部分内存需求，映射内存不应被视为 :ref:`统一内存 <sec:memory-unified-memory>` 或 :ref:`显式内存管理 <sec:intro-cpp-explicit-memory-management>` 的高性能替代方案。

.. _cudamallochost-and-cudahostalloc:

2.4.3.1.1. cudaMallocHost 和 cudaHostAlloc
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用 ``cudaHostMalloc`` 或 ``cudaHostAlloc`` 分配的主机内存会自动映射。这些 API 返回的指针可以直接在 kernel 代码中使用来访问主机上的内存。主机内存通过 CPU-GPU 互连访问。

.. code-block:: cuda
   :caption: 使用 cudaMallocHost

   void usingMallocHost() {
     float* a = nullptr;
     float* b = nullptr;
     
     CUDA_CHECK(cudaMallocHost(&a, vLen*sizeof(float)));
     CUDA_CHECK(cudaMallocHost(&b, vLen*sizeof(float)));

     initVector(b, vLen);
     memset(a, 0, vLen*sizeof(float));

     int threads = 256;
     int blocks = vLen/threads;
     copyKernel<<<blocks, threads>>>(a, b);
     CUDA_CHECK(cudaGetLastError());
     CUDA_CHECK(cudaDeviceSynchronize());

     printf("Using cudaMallocHost: ");
     checkAnswer(a,b);
   }

.. code-block:: cuda
   :caption: 使用 cudaHostAlloc

   void usingCudaHostAlloc() {
     float* a = nullptr;
     float* b = nullptr;

     CUDA_CHECK(cudaHostAlloc(&a, vLen*sizeof(float), cudaHostAllocMapped));
     CUDA_CHECK(cudaHostAlloc(&b, vLen*sizeof(float), cudaHostAllocMapped));

     initVector(b, vLen);
     memset(a, 0, vLen*sizeof(float));

     int threads = 256;
     int blocks = vLen/threads;
     copyKernel<<<blocks, threads>>>(a, b);
     CUDA_CHECK(cudaGetLastError());
     CUDA_CHECK(cudaDeviceSynchronize());

     printf("Using cudaAllocHost: ");
     checkAnswer(a, b);
   }

.. _cudahostregister:

2.4.3.1.2. cudaHostRegister
^^^^^^^^^^^^^^^^^^^^^^^^^^^

当 ATS 和 HMM 不可用时，系统分配器进行的分配仍然可以使用 ``cudaHostRegister`` 映射以便从 GPU kernel 直接访问。但是，与使用 CUDA API 创建的内存不同，不能使用主机指针从 kernel 访问该内存。必须使用 ``cudaHostGetDevicePointer()`` 获取设备内存区域中的指针，并且该指针必须用于 kernel 代码中的访问。

.. code-block:: cuda
   :caption: 使用 cudaHostRegister

   void usingRegister() {
     float* a = nullptr;
     float* b = nullptr;
     float* devA = nullptr;
     float* devB = nullptr;

     a = (float*)malloc(vLen*sizeof(float));
     b = (float*)malloc(vLen*sizeof(float));
     CUDA_CHECK(cudaHostRegister(a, vLen*sizeof(float), 0 ));
     CUDA_CHECK(cudaHostRegister(b, vLen*sizeof(float), 0  ));

     CUDA_CHECK(cudaHostGetDevicePointer((void**)&devA, (void*)a, 0));
     CUDA_CHECK(cudaHostGetDevicePointer((void**)&devB, (void*)b, 0));

     initVector(b, vLen);
     memset(a, 0, vLen*sizeof(float));

     int threads = 256;
     int blocks = vLen/threads;
     copyKernel<<<blocks, threads>>>(devA, devB);
     CUDA_CHECK(cudaGetLastError());
     CUDA_CHECK(cudaDeviceSynchronize());

     printf("Using cudaHostRegister: ");
     checkAnswer(a, b);
   }

.. _comparing-unified-memory-and-mapped-memory:

2.4.3.1.3. 比较统一内存和映射内存
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

映射内存使 CPU 内存可从 GPU 访问，但不保证所有类型的访问（例如原子操作）在所有系统上都受支持。统一内存保证支持所有访问类型。

映射内存保留在 CPU 内存中，这意味着所有 GPU 访问必须通过 CPU 和 GPU 之间的连接进行：PCIe 或 NVLink。通过这些链路进行的访问延迟明显高于对 GPU 内存的访问，并且总可用带宽较低。因此，将映射内存用于所有 kernel 内存访问不太可能充分利用 GPU 计算资源。

统一内存最常迁移到访问它的处理器的物理内存中。第一次迁移后，kernel 对同一内存页或缓存行的重复访问可以利用完整的 GPU 内存带宽。

.. note::

   映射内存在以前的文档中也被称为 *零拷贝* 内存。

   在所有 CUDA 应用程序使用 :ref:`统一虚拟地址空间 <sec:memory-unified-virtual-address-space>` 之前，需要额外的 API 来启用内存映射（带 ``cudaDeviceMapHost`` 的 ``cudaSetDeviceFlags`` ）。这些 API 不再需要。

   在映射主机内存上操作的原子函数（参见 :ref:`原子函数 <sec:atomic-functions>` ）从主机或其他 GPU 的角度来看不是原子的。

   CUDA 运行时要求从设备发起的对主机内存的 1 字节、2 字节、4 字节、8 字节和 16 字节自然对齐的加载和存储，从主机和其他设备的角度来看，被保留为单次访问。在某些平台上，对内存的原子操作可能被硬件分解为单独的加载和存储操作。这些组件加载和存储操作对自然对齐访问的保留有相同的要求。CUDA 运行时不支持 PCI Express 总线拓扑，其中 PCI Express 桥接器拆分 8 字节自然对齐的操作，NVIDIA 不知道有任何拓扑会拆分 16 字节自然对齐的操作。

.. _memory-summary:

2.4.4. 总结
-----------

- 在具有异构内存管理 (HMM) 或地址转换服务 (ATS) 的 Linux 平台上，所有系统分配的内存都是托管内存

- 在没有 HMM 或 ATS 的 Linux 平台上，在 Tegra 处理器上，以及在所有 Windows 平台上，托管内存必须使用 CUDA 分配：

  - ``cudaMallocManaged`` 或
  
  - 带有使用 ``allocType=cudaMemAllocationTypeManaged`` 创建的池的 ``cudaMallocFromPoolAsync``
  
  - 带有 ``__managed__`` 说明符的全局变量
  

- 在 Windows 和 Tegra 处理器上，统一内存有限制

- 在具有 ATS 的 NVLINK C2C 连接系统上，使用 ``cudaMalloc`` 分配的设备内存可以直接从 CPU 或其他 GPU 访问