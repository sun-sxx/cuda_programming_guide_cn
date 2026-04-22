.. _um-details:

4.1. 统一内存
============

本节详细解释每种可用统一内存范式的行为和使用方法。:doc:`../02-basics/04-understanding-memory` 展示了如何确定应用哪种统一内存范式，并简要介绍了每种范式。

如前所述，统一内存编程有四种范式：

- 完全支持托管内存分配
- 完全支持所有分配的软件一致性
- 完全支持所有分配的硬件一致性
- 有限的统一内存支持

前三种涉及完全统一内存支持的范式具有非常相似的行为和编程模型，并在 :ref:`um-pageable-systems` 中涵盖，任何差异都会突出显示。

最后一种统一内存支持受限的范式在 :ref:`um-legacy-devices` 中详细讨论。

.. _um-pageable-systems:

4.1.1. 具有完整 CUDA 统一内存支持的设备上的统一内存
--------------------------------------------------------

这些系统包括硬件一致性内存系统，如 NVIDIA Grace Hopper 和启用了异构内存管理 (HMM) 的现代 Linux 系统。HMM 是一种基于软件的内存管理系统，提供与硬件一致性内存系统相同的编程模型。

Linux HMM 需要 Linux 内核版本 6.1.24+、6.2.11+ 或 6.3+，计算能力 7.5 或更高的设备，以及安装了 `Open Kernel Modules <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#nvidia-open-gpu-kernel-modules>`_ 的 CUDA 驱动程序版本 535+。

.. note::

   我们将 CPU 和 GPU 组合页表的系统称为 *硬件一致性* 系统。CPU 和 GPU 具有单独页表的系统称为 *软件一致性* 系统。

硬件一致性系统（如 NVIDIA Grace Hopper）为 CPU 和 GPU 提供逻辑上组合的页表，参见 :ref:`um-hw-coherency` 。以下部分仅适用于硬件一致性系统：

- :ref:`um-perf-hints`

4.1.1.1. 统一内存：深入示例
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

具有完整 CUDA 统一内存支持的系统（参见表 `统一内存范式概述 <../02-basics/04-understanding-memory.html#table-unified-memory-levels>`_）允许设备访问与其交互的主机进程拥有的任何内存。

本节展示了几个高级用例，使用一个简单地将输入字符数组的前 8 个字符打印到标准输出流的核函数：

.. code-block:: cuda

   __global__ void kernel(const char* type, const char* data) {
     static const int n_char = 8;
     printf("%s - first %d characters: '", type, n_char);
     for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
     printf("'\n");
   }

以下展示了如何使用系统分配的内存调用此核函数的各种方式：

.. dropdown:: Malloc

   .. code-block:: c++

      void test_malloc() {
        const char test_string[] = "Hello World";
        char* heap_data = (char*)malloc(sizeof(test_string));
        strncpy(heap_data, test_string, sizeof(test_string));
        kernel<<<1, 1>>>("malloc", heap_data);
        ASSERT(cudaDeviceSynchronize() == cudaSuccess,
          "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
        free(heap_data);
      }

.. dropdown:: Managed

   .. code-block:: c++

      void test_managed() {
        const char test_string[] = "Hello World";
        char* data;
        cudaMallocManaged(&data, sizeof(test_string));
        strncpy(data, test_string, sizeof(test_string));
        kernel<<<1, 1>>>("managed", data);
        ASSERT(cudaDeviceSynchronize() == cudaSuccess,
          "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
        cudaFree(data);
      }

.. dropdown:: 栈变量

   .. code-block:: c++

      void test_stack() {
        const char test_string[] = "Hello World";
        kernel<<<1, 1>>>("stack", test_string);
        ASSERT(cudaDeviceSynchronize() == cudaSuccess,
          "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
      }

.. dropdown:: 文件作用域静态变量

   .. code-block:: c++

      void test_static() {
        static const char test_string[] = "Hello World";
        kernel<<<1, 1>>>("static", test_string);
        ASSERT(cudaDeviceSynchronize() == cudaSuccess,
          "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
      }

.. dropdown:: 全局作用域变量

   .. code-block:: c++

      const char global_string[] = "Hello World";

      void test_global() {
        kernel<<<1, 1>>>("global", global_string);
        ASSERT(cudaDeviceSynchronize() == cudaSuccess,
          "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
      }

.. dropdown:: 全局作用域 extern 变量

   .. code-block:: c++

      // declared in separate file, see below
      extern char* ext_data;

      void test_ext_extern() {
        kernel<<<1, 1>>>("extern", ext_data);
        ASSERT(cudaDeviceSynchronize() == cudaSuccess,
          "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
      }

      /** This may be a non-CUDA file */
      char* ext_data;
      static const char global_string[] = "Hello World";

      void __attribute__ ((constructor)) setup(void) {
        ext_data = (char*)malloc(sizeof(global_string));
        strncpy(ext_data, global_string, sizeof(global_string));
      }

      void __attribute__ ((destructor)) tear_down(void) {
        free(ext_data);
      }

注意，对于 extern 变量，它可以由第三方库声明和管理其内存，而该库完全不与 CUDA 交互。

还需注意，栈变量以及文件作用域和全局作用域的变量只能通过指针被 GPU 访问。在这个具体示例中，这很方便，因为字符数组已经声明为指针：``const char*``。然而，考虑以下具有全局作用域整数的示例：

.. code-block:: c++

   // this variable is declared at global scope
   int global_variable;

   __global__ void kernel_uncompilable() {
     // this causes a compilation error: global (host) variables must not
     // be accessed from __device__ / __global__ code
     printf("%d\n", global_variable);
   }

   // On systems with pageableMemoryAccess set to 1, we can access the address
   // of a global variable. The below kernel takes that address as an argument
   __global__ void kernel(int* global_variable_addr) {
     printf("%d\n", *global_variable_addr);
   }
   int main() {
     kernel<<<1, 1>>>(&global_variable);
     ...
     return 0;
   }

在上面的示例中，我们需要确保将 *指针* 传递给核函数，而不是直接在核函数中访问全局变量。这是因为没有 ``__managed__`` 说明符的全局变量默认声明为仅 ``__host__``，因此大多数编译器目前不允许在设备代码中直接使用这些变量。

4.1.1.1.1. 文件支持的统一内存
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

由于具有完整 CUDA 统一内存支持的系统允许设备访问主机进程拥有的任何内存，它们可以直接访问文件支持的内存。

这里，我们修改了上一节中展示的初始示例，使用文件支持的内存从 GPU 打印字符串，直接从输入文件读取。在以下示例中，内存由物理文件支持，但该示例也适用于内存支持的文件。

.. code-block:: c++

   __global__ void kernel(const char* type, const char* data) {
     static const int n_char = 8;
     printf("%s - first %d characters: '", type, n_char);
     for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
     printf("'\n");
   }

   void test_file_backed() {
     int fd = open(INPUT_FILE_NAME, O_RDONLY);
     ASSERT(fd >= 0, "Invalid file handle");
     struct stat file_stat;
     int status = fstat(fd, &file_stat);
     ASSERT(status >= 0, "Invalid file stats");
     char* mapped = (char*)mmap(0, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
     ASSERT(mapped != MAP_FAILED, "Cannot map file into memory");
     kernel<<<1, 1>>>("file-backed", mapped);
     ASSERT(cudaDeviceSynchronize() == cudaSuccess,
       "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
     ASSERT(munmap(mapped, file_stat.st_size) == 0, "Cannot unmap file");
     ASSERT(close(fd) == 0, "Cannot close file");
   }

注意，在不具有 ``hostNativeAtomicSupported`` 属性的系统上，包括启用了 Linux HMM 的系统，不支持对文件支持的内存进行原子访问。

4.1.1.1.2. 使用统一内存的进程间通信 (IPC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   目前，使用 IPC 与统一内存可能会有显著的性能影响。

许多应用程序更喜欢为每个 GPU 管理一个进程，但仍需要使用统一内存（例如用于超额订阅），并从多个 GPU 访问它。

CUDA IPC（参见 :doc:`15-inter-process-communication`）不支持托管内存：此类内存的句柄不能通过本节讨论的任何机制共享。在具有完整 CUDA 统一内存支持的系统上，系统分配的内存支持 IPC。一旦与其他进程共享了对系统分配内存的访问，就适用相同的编程模型。

有关在 Linux 下创建 IPC 支持的系统分配内存的各种方法的更多信息，请参阅以下参考资料：

- `mmap with MAP_SHARED <https://man7.org/linux/man-pages/man2/mmap.2.html>`_
- `POSIX IPC APIs <https://pubs.opengroup.org/onlinepubs/007904875/functions/shm_open.html>`_
- `Linux memfd_create <https://man7.org/linux/man-pages/man2/memfd_create.2.html>`_

注意，使用此技术无法在不同主机及其设备之间共享内存。

4.1.1.2. 性能调优
~~~~~~~~~~~~~~~~~

为了使用统一内存实现良好的性能，重要的是：

- 理解分页在您的系统上如何工作，以及如何避免不必要的页错误
- 理解各种机制，允许您将数据保持在访问处理器的本地
- 考虑针对系统的内存传输粒度调整您的应用程序

作为一般建议，性能提示（参见 `um-perf-hints`_）可能会提高性能，但不正确地使用它们可能会导致性能比默认行为更差。还需注意，任何提示在主机上都有相关的性能成本，因此有用的提示必须至少提高足够的性能来克服此成本。

4.1.1.2.1. 内存分页和页大小
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为了更好地理解统一内存的性能影响，重要的是理解虚拟寻址、内存页和页大小。本子节尝试定义所有必要的术语并解释为什么分页对性能很重要。

所有当前支持的统一内存系统都使用虚拟地址空间：这意味着应用程序使用的内存地址表示 *虚拟* 位置，该位置可能 *映射* 到内存实际驻留的 *物理* 位置。

所有当前支持的处理器，包括 CPU 和 GPU，都使用内存 *分页*。由于所有系统都使用虚拟地址空间，有两种类型的内存页：

- **虚拟页**：这表示每个进程由操作系统跟踪的固定大小连续虚拟内存块，可以 *映射* 到物理内存中。注意，虚拟页与 *映射* 相关联：例如，单个虚拟地址可能使用不同的页大小映射到物理内存中。
- **物理页**：这表示处理器的主要内存管理单元 (MMU) 支持的固定大小连续内存块，虚拟页可以映射到其中。

目前，所有 x86_64 CPU 使用 4KiB 的默认物理页大小。Arm CPU 支持多种物理页大小——4KiB、16KiB、32KiB 和 64KiB——具体取决于确切的 CPU。最后，NVIDIA GPU 支持多种物理页大小，但更喜欢 2MiB 或更大的物理页。注意，这些大小可能会在未来的硬件中发生变化。

虚拟页的默认页大小通常对应于物理页大小，但应用程序可以使用不同的页大小，只要它们得到操作系统和硬件的支持。通常，支持的虚拟页大小必须是 2 的幂且是物理页大小的倍数。

跟踪虚拟页到物理页映射的逻辑实体将被称为 *页表*，给定虚拟大小的给定虚拟页到物理页的每个映射称为 *页表项 (PTE)*。所有支持的处理器都为页表提供特定的缓存，以加速虚拟地址到物理地址的转换。这些缓存称为 *转换后备缓冲区 (TLB)*。

性能调优有两个重要方面：

- 虚拟页大小的选择
- 系统是否提供 CPU 和 GPU 使用的组合页表，还是每个 CPU 和 GPU 单独的页表

4.1.1.2.1.1. 选择合适的页大小
"""""""""""""""""""""""""""""

一般来说，小页大小导致更少的（虚拟）内存碎片但更多的 TLB 未命中，而大页大小导致更多的内存碎片但更少的 TLB 未命中。此外，内存迁移通常在大页大小下比小页大小更昂贵，因为我们通常迁移完整的内存页。这可能导致使用大页大小的应用程序出现更大的延迟峰值。另见下一节有关页错误的更多详细信息。

性能调优的一个重要方面是，GPU 上的 TLB 未命中通常比 CPU 上昂贵得多。这意味着如果 GPU 线程频繁访问使用足够小页大小映射的统一内存的随机位置，与使用足够大页大小映射的统一内存的相同访问相比，它可能会明显更慢。虽然类似的效果可能发生在 CPU 线程随机访问使用小页大小映射的大内存区域时，但 slowdown 不太明显，这意味着应用程序可能希望用更少的内存碎片来权衡这种 slowdown。

注意，一般来说，应用程序不应该针对给定处理器的物理页大小调整其性能，因为物理页大小会根据硬件而变化。上述建议仅适用于虚拟页大小。

.. _um-hw-coherency:

4.1.1.2.1.2. CPU 和 GPU 页表：硬件一致性与软件一致性
"""""""""""""""""""""""""""""""""""""""""""""""""""""

硬件一致性系统（如 NVIDIA Grace Hopper）为 CPU 和 GPU 提供逻辑上组合的页表。这很重要，因为为了从 GPU 访问系统分配的内存，GPU 使用 CPU 为请求的内存创建的任何页表项。如果该页表项使用 CPU 的默认页大小 4KiB 或 64KiB，对大虚拟内存区域的访问将导致显著的 TLB 未命中，从而显著 slowdown。

另一方面，在 CPU 和 GPU 各有自己逻辑页表的软件一致性系统上，应考虑不同的性能调优方面：为了保证一致性，这些系统通常在处理器访问映射到不同处理器物理内存的内存地址时使用 *页错误*。这样的页错误意味着：

- 需要确保当前拥有处理器（物理页当前驻留的位置）不能再访问此页，通过删除页表项或更新它。
- 需要确保请求访问的处理器可以访问此页，通过创建新页表项或更新现有项，使其变为有效/活动。
- 支持此虚拟页的物理页必须移动/迁移到请求访问的处理器：这可能是一个昂贵的操作，工作量与页大小成正比。

总体而言，与软件一致性系统相比，硬件一致性系统在 CPU 和 GPU 线程频繁并发访问同一内存页的情况下提供显著的性能优势：

- **更少的页错误**：这些系统不需要使用页错误来模拟一致性或迁移内存
- **更少的争用**：这些系统在缓存行粒度上是一致的，而不是页大小粒度，也就是说，当多个处理器在缓存行内有争用时，只交换缓存行，这比最小的页大小小得多，并且当不同处理器访问页内的不同缓存行时，没有争用

这影响以下场景的性能：

- 来自 CPU 和 GPU 并发地对同一地址进行原子更新
- 从 CPU 线程向 GPU 线程发出信号，反之亦然

4.1.1.2.2. 从主机直接访问统一内存
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

某些设备具有硬件支持，用于主机对 GPU 驻留统一内存的相干读取、存储和原子访问。这些设备的 ``cudaDevAttrDirectManagedMemAccessFromHost`` 属性设置为 1。注意，所有硬件一致性系统都为 NVLink 连接的设备设置此属性。在这些系统上，主机无需页错误和数据迁移即可直接访问 GPU 驻留内存。注意，使用 CUDA 托管内存时，需要使用位置类型 ``cudaMemLocationTypeHost`` 的 ``cudaMemAdviseSetAccessedBy`` 提示来启用此直接访问而无需页错误，参见下面的示例。

.. dropdown:: 系统分配器

   .. code-block:: c++

      __global__ void write(int *ret, int a, int b) {
        ret[threadIdx.x] = a + b + threadIdx.x;
      }

      __global__ void append(int *ret, int a, int b) {
        ret[threadIdx.x] += a + b + threadIdx.x;
      }

      void test_malloc() {
        int *ret = (int*)malloc(1000 * sizeof(int));
        // for shared page table systems, the following hint is not necesary
        cudaMemLocation location = {.type = cudaMemLocationTypeHost};
        cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, location);

        write<<< 1, 1000 >>>(ret, 10, 100);            // pages populated in GPU memory
        cudaDeviceSynchronize();
        for(int i = 0; i < 1000; i++)
            printf("%d: A+B = %d\n", i, ret[i]);        // directManagedMemAccessFromHost=1: CPU accesses GPU memory directly without migrations
                                                       // directManagedMemAccessFromHost=0: CPU faults and triggers device-to-host migrations
        append<<< 1, 1000 >>>(ret, 10, 100);            // directManagedMemAccessFromHost=1: GPU accesses GPU memory without migrations
        cudaDeviceSynchronize();                        // directManagedMemAccessFromHost=0: GPU faults and triggers host-to-device migrations
        free(ret);
      }

.. dropdown:: 托管

   .. code-block:: c++

      __global__ void write(int *ret, int a, int b) {
        ret[threadIdx.x] = a + b + threadIdx.x;
      }

      __global__ void append(int *ret, int a, int b) {
        ret[threadIdx.x] += a + b + threadIdx.x;
      }

      void test_managed() {
        int *ret;
        cudaMallocManaged(&ret, 1000 * sizeof(int));
        cudaMemLocation location = {.type = cudaMemLocationTypeHost};
        cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, location);  // set direct access hint

        write<<< 1, 1000 >>>(ret, 10, 100);            // pages populated in GPU memory
        cudaDeviceSynchronize();
        for(int i = 0; i < 1000; i++)
            printf("%d: A+B = %d\n", i, ret[i]);        // directManagedMemAccessFromHost=1: CPU accesses GPU memory directly without migrations
                                                       // directManagedMemAccessFromHost=0: CPU faults and triggers device-to-host migrations
        append<<< 1, 1000 >>>(ret, 10, 100);            // directManagedMemAccessFromHost=1: GPU accesses GPU memory without migrations
        cudaDeviceSynchronize();                        // directManagedMemAccessFromHost=0: GPU faults and triggers host-to-device migrations
        cudaFree(ret);
      }

在 ``write`` 核函数完成后，``ret`` 将在 GPU 内存中创建和初始化。接下来，CPU 将访问 ``ret``，然后 ``append`` 核函数使用相同的 ``ret`` 内存。此代码将根据系统架构和硬件一致性支持显示不同的行为：

- 在 ``directManagedMemAccessFromHost=1`` 的系统上：对托管缓冲区的 CPU 访问不会触发任何迁移；数据将保留在 GPU 内存中，任何后续的 GPU 核函数都可以继续直接访问它，而不会导致错误或迁移
- 在 ``directManagedMemAccessFromHost=0`` 的系统上：对托管缓冲区的 CPU 访问将页错误并启动数据迁移；任何尝试首次访问相同数据的 GPU 核函数将页错误并将页迁移回 GPU 内存

4.1.1.2.3. 主机原生原子操作
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

某些设备，包括硬件一致性系统的 NVLink 连接设备，支持对 CPU 驻留内存的硬件加速原子访问。这意味着对主机内存的原子访问不必用页错误模拟。对于这些设备，``cudaDevAttrHostNativeAtomicSupported`` 属性设置为 1。

4.1.1.2.4. 原子访问和同步原语
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA 统一内存支持主机和设备线程可用的所有原子操作，使所有线程能够通过并发访问同一共享内存位置进行协作。`libcu++ <https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives.html>`_ 库提供了许多针对主机和设备线程之间并发使用调整的异构同步原语，包括 ``cuda::atomic``、``cuda::atomic_ref``、``cuda::barrier``、``cuda::semaphore`` 等。

在软件一致性系统上，不支持从设备对文件支持的主机内存进行原子访问。以下示例代码在硬件一致性系统上有效，但在其他系统上表现出未定义的行为：

.. code-block:: c++

   #include <cuda/atomic>

   #include <cstdio>
   #include <fcntl.h>
   #include <sys/mman.h>

   #define ERR(msg, ...) { fprintf(stderr, msg, ##__VA_ARGS__); return EXIT_FAILURE; }

   __global__ void kernel(int* ptr) {
     cuda::atomic_ref{*ptr}.store(2);
   }

   int main() {
     // this will be closed/deleted by default on exit
     FILE* tmp_file = tmpfile64();
     // need to allocate space in the file, we do this with posix_fallocate here
     int status = posix_fallocate(fileno(tmp_file), 0, 4096);
     if (status != 0) ERR("Failed to allocate space in temp file\n");
     int* ptr = (int*)mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE, fileno(tmp_file), 0);
     if (ptr == MAP_FAILED) ERR("Failed to map temp file\n");

     // initialize the value in our file-backed memory
     *ptr = 1;
     printf("Atom value: %d\n", *ptr);

     // device and host thread access ptr concurrently, using cuda::atomic_ref
     kernel<<<1, 1>>>(ptr);
     while (cuda::atomic_ref{*ptr}.load() != 2);
     // this will always be 2
     printf("Atom value: %d\n", *ptr);

     return EXIT_SUCCESS;
   }

在软件一致性系统上，对统一内存的原子访问可能会产生页错误，这可能导致显著的延迟。注意，对于这些系统上的 CPU 内存的所有 GPU 原子操作并非都是如此：`nvidia-smi -q | grep "Atomic Caps Outbound"` 列出的操作可能会避免页错误。

在硬件一致性系统上，主机和设备之间的原子操作不需要页错误，但仍可能因其他可能导致任何内存访问错误的原因而出错。

4.1.1.2.5. Memcpy()/Memset() 与统一内存的行为
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``cudaMemcpy*()`` 和 ``cudaMemset*()`` 接受任何统一内存指针作为参数。

对于 ``cudaMemcpy*()``，指定为 ``cudaMemcpyKind`` 的方向是性能提示，如果任何参数是统一内存指针，可能会产生更高的性能影响。

因此，建议遵循以下性能建议：

- 当统一内存的物理位置已知时，使用准确的 ``cudaMemcpyKind`` 提示
- 优先使用 ``cudaMemcpyDefault`` 而不是不准确的 ``cudaMemcpyKind`` 提示
- 始终使用已填充（已初始化）的缓冲区：避免使用这些 API 初始化内存
- 如果两个指针都指向系统分配的内存，避免使用 ``cudaMemcpy*()``：改为启动核函数或使用 CPU 内存复制算法（如 ``std::memcpy``）

4.1.1.2.6. 统一内存的内存分配器概述
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

对于具有完整 CUDA 统一内存支持的系统，可以使用各种不同的分配器来分配统一内存。下表显示了所选分配器及其各自功能的概述。注意，本节中的所有信息都可能在未来的 CUDA 版本中发生变化。

.. list-table:: 不同分配器的统一内存支持概述
   :widths: 25 20 15 20 20
   :header-rows: 1

   * - API
     - 放置策略
     - 可访问自
     - 基于访问迁移 [\ [#f2]_ ]
     - 页大小 [\ [#f4]_ ] [\ [#f5]_ ]
   * - ``malloc``、``new``、``mmap``
     - 首次触摸/提示 [\ [#f1]_ ]
     - CPU、GPU
     - 是 [\ [#f3]_ ]
     - 系统或大页大小 [\ [#f6]_ ]
   * - ``cudaMallocManaged``
     - 首次触摸/提示
     - CPU、GPU
     - 是
     - CPU 驻留：系统页大小 GPU 驻留：2MB
   * - ``cudaMalloc``
     - GPU
     - GPU
     - 否
     - GPU 页大小：2MB
   * - ``cudaMallocHost``、``cudaHostAlloc``、``cudaHostRegister``
     - CPU
     - CPU、GPU
     - 否
     - CPU 映射：系统页大小 GPU 映射：2MB
   * - 内存池，位置类型主机：``cuMemCreate``、``cudaMemPoolCreate``
     - CPU
     - CPU、GPU
     - 否
     - CPU 映射：系统页大小 GPU 映射：2MB
   * - 内存池，位置类型设备：``cuMemCreate``、``cudaMemPoolCreate``、``cudaMallocAsync``
     - GPU
     - GPU
     - 否
     - 2MB

.. [#f1] 对于 ``mmap``，文件支持的内存默认放置在 CPU 上，除非通过 ``cudaMemAdviseSetPreferredLocation``（或 ``mbind``，见下面的项目符号）另行指定。
.. [#f2] 此功能可以使用 ``cudaMemAdvise`` 覆盖。即使禁用了基于访问的迁移，如果支持的内存空间已满，内存也可能会迁移。
.. [#f3] 文件支持的内存不会根据访问迁移。
.. [#f4] 默认系统页大小在大多数系统上为 4KiB 或 64KiB，除非明确指定大页大小（例如，使用 ``mmap`` ``MAP_HUGETLB`` / ``MAP_HUGE_SHIFT``）。在这种情况下，支持在系统上配置的任何大页大小。
.. [#f5] GPU 驻留内存的页大小可能会在未来的 CUDA 版本中演变。
.. [#f6] 目前，在将内存迁移到 GPU 或通过首次触摸将其放置在 GPU 上时，可能不会保持大页大小。

表 :ref:`um-perf-hints` 显示了几种可用于分配可同时从多个处理器（包括主机和设备）访问的数据的分配器的语义差异。有关 ``cudaMemPoolCreate`` 的更多详细信息，请参阅 `内存池 <stream-ordered-memory-allocation.html#stream-ordered-memory-pools>`_ 部分，有关 ``cuMemCreate`` 的更多详细信息，请参阅 `虚拟内存管理 <virtual-memory-management.html#virtual-memory-management>`_ 部分。

在硬件一致性系统上，设备内存作为 NUMA 域暴露给系统，可以使用特殊的分配器（如 ``numa_alloc_on_node``）将内存固定到给定的 NUMA 节点（主机或设备）。此内存可从主机和设备访问，且不迁移。类似地，``mbind`` 可用于将内存固定到给定的 NUMA 节点，并可使文件支持的内存首次访问之前放置在给定的 NUMA 节点上。

以下适用于共享内存的分配器：

- 系统分配器（如 ``mmap``）允许使用 ``MAP_SHARED`` 标志在进程之间共享内存。这在 CUDA 中受支持，可用于在同一主机连接的不同设备之间共享内存。但是，目前不支持在多个主机以及多个设备之间共享内存。
- 对于在多个主机上通过网络访问统一内存或其他 CUDA 内存，请咨询所用通信库的文档，例如 `NCCL <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html>`_、`NVSHMEM <https://docs.nvidia.com/nvshmem/api/index.html>`_、`OpenMPI <https://www.open-mpi.org/faq/?category=runcuda>`_、`UCX <https://docs.mellanox.com/category/hpcx>`_ 等。

4.1.1.2.7. 访问计数器迁移
^^^^^^^^^^^^^^^^^^^^^^^^^^

在硬件一致性系统上，访问计数器功能跟踪 GPU 对其他处理器上内存的访问频率。这是为了确保内存页移动到最频繁访问页的处理器的物理内存。它可以指导 CPU 和 GPU 之间以及对等 GPU 之间的迁移，此过程称为访问计数器迁移。

从 CUDA 12.4 开始，访问计数器支持系统分配的内存。注意，文件支持的内存不会根据访问迁移。对于系统分配的内存，可以通过使用 ``cudaMemAdviseSetAccessedBy`` 提示到具有相应设备 ID 的设备来开启访问计数器迁移。如果访问计数器开启，可以使用 ``cudaMemAdviseSetPreferredLocation`` 设置为主机来防止迁移。默认情况下，``cudaMallocManaged`` 基于错误和迁移机制进行迁移。[\ [#f7]_ ]

驱动程序还可以使用访问计数器进行更高效的抖动缓解或内存超额订阅场景。

.. [#f7] 当前系统允许在设置访问设备提示时对托管内存使用访问计数器迁移。这是实现细节，不应依赖未来的兼容性。

4.1.1.2.8. 避免 CPU 频繁写入 GPU 驻留内存
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果主机访问统一内存，缓存未命中可能会在主机和设备之间引入比预期更多的流量。许多 CPU 架构要求所有内存操作通过缓存层次结构，包括写入。如果系统内存驻留在 GPU 上，这意味着 CPU 对此内存的频繁写入可能会导致缓存未命中，从而先将数据从 GPU 传输到 CPU，然后再将实际值写入请求的内存范围。在软件一致性系统上，这可能会引入额外的页错误，而在硬件一致性系统上，它可能会导致 CPU 操作之间的更高延迟。因此，为了与设备共享主机产生的数据，考虑写入 CPU 驻留内存并直接从设备读取值。下面的代码展示了如何使用统一内存实现此目的。

.. dropdown:: 系统分配器

   .. code-block:: c++

      size_t data_size = sizeof(int);
      int* data = (int*)malloc(data_size);
      // ensure that data stays local to the host and avoid faults
      cudaMemLocation location = {.type = cudaMemLocationTypeHost};
      cudaMemAdvise(data, data_size, cudaMemAdviseSetPreferredLocation, location);
      cudaMemAdvise(data, data_size, cudaMemAdviseSetAccessedBy, location);

      // frequent exchanges of small data: if the CPU writes to CPU-resident memory,
      // and GPU directly accesses that data, we can avoid the CPU caches re-loading
      // data if it was evicted in between writes
      for (int i = 0; i < 10; ++i) {
        *data = 42 + i;
        kernel<<<1, 1>>>(data);
        cudaDeviceSynchronize();
        // CPU cache potentially evicted data here
      }
      free(data);

.. dropdown:: 托管

   .. code-block:: c++

      int* data;
      size_t data_size = sizeof(int);
      cudaMallocManaged(&data, data_size);
      // ensure that data stays local to the host and avoid faults
      cudaMemLocation location = {.type = cudaMemLocationTypeHost};
      cudaMemAdvise(data, data_size, cudaMemAdviseSetPreferredLocation, location);
      cudaMemAdvise(data, data_size, cudaMemAdviseSetAccessedBy, location);

      // frequent exchanges of small data: if the CPU writes to CPU-resident memory,
      // and GPU directly accesses that data, we can avoid the CPU caches re-loading
      // data if it was evicted in between writes
      for (int i = 0; i < 10; ++i) {
        *data = 42 + i;
        kernel<<<1, 1>>>(data);
        cudaDeviceSynchronize();
        // CPU cache potentially evicted data here
      }
      cudaFree(data);

4.1.1.2.9. 利用对系统内存的异步访问
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果应用程序需要共享设备工作的结果与主机，有几种可能的选项：

1. 设备将其结果写入 GPU 驻留内存，使用 ``cudaMemcpy*`` 传输结果，主机读取传输的数据
2. 设备直接将其结果写入 CPU 驻留内存，主机读取该数据
3. 设备写入 GPU 驻留内存，主机直接访问该数据

如果可以在设备传输/主机访问结果时安排在设备上进行独立工作，则选项 1 或 3 是首选。如果设备在主机访问结果之前处于饥饿状态，则选项 2 可能是首选。这是因为设备通常可以以比主机读取更高的带宽写入，除非使用许多主机线程来读取数据。

.. dropdown:: 1. 显式复制

   .. code-block:: c++

      void exchange_explicit_copy(cudaStream_t stream) {
        int* data, *host_data;
        size_t n_bytes = sizeof(int) * 16;
        // allocate receiving buffer
        host_data = (int*)malloc(n_bytes);
        // allocate, since we touch on the device first, will be GPU-resident
        cudaMallocManaged(&data, n_bytes);
        kernel<<<1, 16, 0, stream>>>(data);
        // launch independent work on the device
        // other_kernel<<<1024, 256, 0, stream>>>(other_data, ...);
        // transfer to host
        cudaMemcpyAsync(host_data, data, n_bytes, cudaMemcpyDeviceToHost, stream);
        // sync stream to ensure data has been transferred
        cudaStreamSynchronize(stream);
        // read transferred data
        printf("Got values %d - %d from GPU\n", host_data[0], host_data[15]);
        cudaFree(data);
        free(host_data);
      }

.. dropdown:: 2. 设备直接写入

   .. code-block:: c++

      void exchange_device_direct_write(cudaStream_t stream) {
        int* data;
        size_t n_bytes = sizeof(int) * 16;
        // allocate receiving buffer
        cudaMallocManaged(&data, n_bytes);
        // ensure that data is mapped and resident on the host
        cudaMemLocation location = {.type = cudaMemLocationTypeHost};
        cudaMemAdvise(data, n_bytes, cudaMemAdviseSetPreferredLocation, location);
        cudaMemAdvise(data, n_bytes, cudaMemAdviseSetAccessedBy, location);
        kernel<<<1, 16, 0, stream>>>(data);
        // sync stream to ensure data has been transferred
        cudaStreamSynchronize(stream);
        // read transferred data
        printf("Got values %d - %d from GPU\n", data[0], data[15]);
        cudaFree(data);
      }

.. dropdown:: 3. 主机直接读取

   .. code-block:: c++

      void exchange_host_direct_read(cudaStream_t stream) {
        int* data;
        size_t n_bytes = sizeof(int) * 16;
        // allocate receiving buffer
        cudaMallocManaged(&data, n_bytes);
        // ensure that data is mapped and resident on the device
        cudaMemLocation device_loc = {};
        cudaGetDevice(&device_loc.id);
        device_loc.type = cudaMemLocationTypeDevice;
        cudaMemAdvise(data, n_bytes, cudaMemAdviseSetPreferredLocation, device_loc);
        cudaMemAdvise(data, n_bytes, cudaMemAdviseSetAccessedBy, device_loc);
        kernel<<<1, 16, 0, stream>>>(data);
        // launch independent work on the GPU
        // other_kernel<<<1024, 256, 0, stream>>>(other_data, ...);
        // sync stream to ensure data may be accessed (has been written by device)
        cudaStreamSynchronize(stream);
        // read data directly from host
        printf("Got values %d - %d from GPU\n", data[0], data[15]);
        cudaFree(data);
      }

最后，在上面的显式复制示例中，可以使用主机或设备核函数显式执行此传输，而不是使用 ``cudaMemcpy*`` 传输数据。对于连续数据，首选使用 CUDA 复制引擎，因为复制引擎执行的操作可以与主机和设备上的工作重叠。复制引擎可用于 ``cudaMemcpy*`` 和 ``cudaMemPrefetchAsync`` API，但不能保证与 ``cudaMemcpy*`` API 调用一起使用复制引擎。出于同样的原因，对于足够大的数据，显式复制优于直接主机读取：如果主机和设备都执行不饱和各自内存系统的工作，复制引擎可以同时与主机和设备执行的工作并发执行传输。

复制引擎通常用于主机和设备之间的传输以及 NVLink 连接系统内的对等设备之间的传输。由于复制引擎的总数量有限，某些系统的 ``cudaMemcpy*`` 带宽可能低于使用设备显式执行传输的带宽。在这种情况下，如果传输处于应用程序的关键路径中，可能更喜欢使用显式基于设备的传输。

.. _um-no-pageable-systems:

4.1.2. 仅支持 CUDA 托管内存的设备上的统一内存
-------------------------------------------------

对于具有 6.x 或更高计算能力但没有分页内存访问的设备（参见表 `统一内存范式概述 <../02-basics/04-understanding-memory.html#table-unified-memory-levels>`_），CUDA 托管内存完全支持且一致，但 GPU 无法访问系统分配的内存。统一内存的编程模型和性能调优与 :ref:`um-pageable-systems` 节中描述的模型非常相似，但明显的例外是系统分配器不能用于分配内存。因此，以下子节列表不适用：

- 系统分配器
- :ref:`um-hw-coherency`
- 原子操作
- :ref:`um-perf-hints`
- 流量 hints
- 异步访问

.. _um-legacy-devices:

4.1.3. Windows、WSL 和 Tegra 上的统一内存
--------------------------------------------

.. note::

   本节仅针对计算能力低于 6.0 的设备或 Windows 平台，``concurrentManagedAccess`` 属性设置为 0 的设备。

计算能力低于 6.0 的设备或 Windows 平台（``concurrentManagedAccess`` 属性设置为 0 的设备，参见 `统一内存范式概述 <../02-basics/04-understanding-memory.html#table-unified-memory-levels>`_）支持 CUDA 托管内存，但有以下限制：

- **数据迁移和一致性**：不支持托管数据按需细粒度移动到 GPU。每当启动 GPU 核函数时，通常必须将所有托管内存传输到 GPU 内存，以避免内存访问错误。仅支持 CPU 端的页错误。
- **GPU 内存超额订阅**：它们不能分配超过 GPU 内存物理大小的托管内存。
- **一致性和并发性**：无法同时访问托管内存，因为如果 CPU 在 GPU 核函数活动时访问统一内存分配，由于缺少 GPU 页错误机制，无法保证一致性。

4.1.3.1. 多 GPU
~~~~~~~~~~~~~~~

在具有低于 6.0 计算能力的设备或 Windows 平台的系统上，托管分配通过 GPU 的对等功能自动对系统中的所有 GPU 可见。托管内存分配的行为类似于使用 ``cudaMalloc()`` 分配的未托管内存：当前活动设备是物理分配的主页，但系统中的其他 GPU 将通过 PCIe 总线以降低的带宽访问内存。

在 Linux 上，只要程序积极使用的所有 GPU 都具有对等支持，就会在 GPU 内存中分配托管内存。如果应用程序在任何时候开始使用与任何具有托管分配的 GPU 都不具有对等支持的 GPU，驱动程序将所有托管分配迁移到系统内存。在这种情况下，所有 GPU 都会遇到 PCIe 带宽限制。

在 Windows 上，如果无法获得对等映射（例如，在不同架构的 GPU 之间），无论程序是否实际使用这两个 GPU，系统都会自动回退到使用映射内存。如果只使用一个 GPU，则需要在启动程序之前设置 ``CUDA_VISIBLE_DEVICES`` 环境变量。这限制了哪些 GPU 是可见的，并允许在 GPU 内存中分配托管内存。

或者，在 Windows 上，用户还可以将 ``CUDA_MANAGED_FORCE_DEVICE_ALLOC`` 设置为非零值，以强制驱动程序始终使用设备内存进行物理存储。当此环境变量设置为非零值时，该过程中使用的所有支持托管内存的设备必须彼此具有对等兼容性。如果使用了支持托管内存的设备，并且它与该过程中之前使用的任何其他托管内存支持设备不具有对等兼容性，即使已在这些设备上调用过 ``::cudaDeviceReset``，也会返回错误 ``::cudaErrorInvalidDevice``。这些环境变量在 `CUDA 环境变量 <../05-appendices/environment-variables.html#cuda-environment-variables>`_ 中描述。

4.1.3.2. 一致性和并发性
~~~~~~~~~~~~~~~~~~~~~~~

为确保一致性，统一内存编程模型对 CPU 和 GPU 并发执行时的数据访问施加约束。实际上，在任何核函数操作执行期间，GPU 拥有对所有托管数据的独占访问权，不允许 CPU 访问它，无论特定核函数是否正在积极使用数据。并发 CPU/GPU 访问，即使是对不同的托管内存分配，也会导致段错误，因为该页被认为对 CPU 无法访问。

例如，以下代码在具有 6.x 计算能力的设备上成功运行，因为 GPU 页错误功能取消了同时访问的所有限制，但在 6.x 之前的架构和 Windows 平台上失败，因为当 CPU 触及 ``y`` 时 GPU 程序核函数仍处于活动状态：

.. code-block:: c++

   __device__ __managed__ int x, y=2;
   __global__  void  kernel() {
       x = 10;
   }
   int main() {
       kernel<<< 1, 1 >>>();
       y = 20;            // Error on GPUs not supporting concurrent access

       cudaDeviceSynchronize();
       return  0;
   }

程序必须在访问 ``y`` 之前与 GPU 显式同步（无论 GPU 核函数是否实际触及 ``y`` 或任何托管数据）：

.. code-block:: c++

   __device__ __managed__ int x, y=2;
   __global__  void  kernel() {
       x = 10;
   }
   int main() {
       kernel<<< 1, 1 >>>();
       cudaDeviceSynchronize();
       y = 20;            //  Success on GPUs not supporting concurrent access
       return  0;
   }

注意，任何保证 GPU 完成其工作的函数调用在逻辑上都是有效的，以确保 GPU 工作完成，参见 `显式同步 <../03-advanced/advanced-host-programming.html#advanced-host-explicit-synchronization>`_。

注意，如果在 GPU 活动时使用 ``cudaMallocManaged()`` 或 ``cuMemAllocManaged()`` 动态分配内存，则在启动额外工作或同步 GPU 之前，内存的行为是未指定的。在此期间尝试在 CPU 上访问内存可能会导致也可能不会导致段错误。这不适用于使用标志 ``cudaMemAttachHost`` 或 ``CU_MEM_ATTACH_HOST`` 分配的内存。

4.1.3.3. 流关联的统一内存
~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA 编程模型提供流作为程序指示核函数启动之间依赖性和独立性的机制。启动到同一流中的核函数保证连续执行，而启动到不同流中的核函数允许并发执行。参见 `CUDA 流 <../02-basics/asynchronous-execution.html#cuda-streams>`_ 节。

4.1.3.3.1. 流回调
^^^^^^^^^^^^^^^^^

在流回调内，CPU 访问托管数据是合法的，前提是 GPU 上没有其他可能正在访问托管数据的流处于活动状态。此外，后面没有任何设备工作的回调可用于同步：例如，通过在回调内部发出条件变量信号；否则，CPU 访问仅在回调期间有效。有几个重要的注意点：

- 回调本身保证所有在回调之前启动的设备工作已完成，包括访问托管数据的核函数
- 如果在回调之后在同一 GPU 上有任何核函数启动，则回调内的 CPU 访问无效，因为后续核函数可能与回调并发执行
- 回调内的 CPU 访问必须限制在回调本身内，不能访问回调之外的托管数据

4.1.3.3.2. 托管内存与流关联允许更精细的控制
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``cudaStreamAttachMemAsync()`` API 允许将托管内存区域与特定流关联。这允许更精细地控制内存的并发访问，对于支持并发托管内存访问但希望在不同线程或任务之间隔离内存访问的程序很有用。

考虑以下示例，其中 ``y`` 是托管变量，由核函数写入并由 CPU 读取：

.. code-block:: c++

    __device__ __managed__ int x, y=2;
    __global__ void kernel() {
        x = 10;
    }
    int main() {
        kernel<<< 1, 1 >>>();
        y = 20;            // 在未支持并发访问的 GPU 上出错

        cudaDeviceSynchronize();
        return  0;
    }

程序必须在访问 ``y`` 之前与 GPU 显式同步（无论 GPU 核函数是否实际触及 ``y`` 或任何托管数据）：

.. code-block:: c++

    __device__ __managed__ int x, y=2;
    __global__ void kernel() {
        x = 10;
    }
    int main() {
        kernel<<< 1, 1 >>>();
        cudaDeviceSynchronize();
        y = 20;            // 在不支持并发访问的 GPU 上成功
        return  0;
    }

通过使用 ``cudaStreamAttachMemAsync()`` 将 ``y`` 与 NULL 流关联，核函数与 ``y`` 解耦，从而允许在核函数执行期间访问 ``y``：

.. code-block:: c++

    __device__ __managed__ int x, y=2;
    __global__ void kernel() {
        x = 10;
    }
    int main() {
        // 将 y 与 NULL 流关联
        cudaStreamAttachMemAsync(0, &y, 0, cudaMemAttachHost);
        cudaDeviceSynchronize();          // 等待主机关联发生。
        kernel<<< 1, 1 >>>();              // 启动到 NULL 流。
        y = 20;                           // 成功 – 核函数正在运行但 "y"
                                          // 已与任何流解关联。
        return  0;
    }

或者，``cudaStreamAttachMemAsync()`` 可用于将托管内存与特定流关联，从而允许在核函数执行期间从 CPU 访问：

.. code-block:: c++

    __device__ __managed__ int x, y=2;
    __global__ void kernel() {
        x = 10;
    }
    int main() {
        cudaStream_t stream1;
        cudaStreamCreate(&stream1);
        cudaStreamAttachMemAsync(stream1, &y, 0, cudaMemAttachHost);
        cudaDeviceSynchronize();          // 等待主机关联发生。
        kernel<<< 1, 1, 0, stream1 >>>(); // 注意：启动到 stream1。
        y = 20;                           // 成功 – 核函数正在运行但 "y"
                                          // 已与任何流解关联。
        return  0;
    }

4.1.3.3.3. 多线程主机程序的更复杂示例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``cudaStreamAttachMemAsync()`` 的主要用途是使用 CPU 线程启用独立的任务并行性。通常在这样的程序中，CPU 线程为其生成的所有工作创建自己的流，因为使用 CUDA 的 NULL 流会导致线程之间的依赖关系。托管数据对任何 GPU 流的默认全局可见性使得难以避免多线程程序中 CPU 线程之间的交互。因此，函数 ``cudaStreamAttachMemAsync()`` 用于将线程的托管分配与该线程自己的流关联，并且这种关联通常在线程的生存期内不改变。这样的程序只需添加一个调用 ``cudaStreamAttachMemAsync()`` 来使用统一内存进行数据访问：

.. code-block:: c++

    // 此函数在其自己的私有流中执行某些任务，可以并行运行
    void run_task(int *in, int *out, int length) {
        // 为我们使用的流创建。
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        // 分配一些托管数据并与我们的流关联。
        // 注意使用 cudaMallocManaged() 的主机关联标志；
        // 然后我们将分配与我们的流关联，以便
        // 我们的 GPU 核函数启动可以访问它。
        int *data;
        cudaMallocManaged((void **)&data, length, cudaMemAttachHost);
        cudaStreamAttachMemAsync(stream, data);
        cudaStreamSynchronize(stream);
        // 以某种方式迭代数据，使用主机和设备。
        for(int i=0; i<N; i++) {
            transform<<< 100, 256, 0, stream >>>(in, data, length);
            cudaStreamSynchronize(stream);
            host_process(data, length);    // CPU 使用托管数据。
            convert<<< 100, 256, 0, stream >>>(out, data, length);
        }
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        cudaFree(data);
    }

在此示例中，分配 - 流关联只建立一次，然后数据由主机和设备重复使用。结果是代码比显式在主机和设备之间复制数据的代码简单得多，尽管结果是相同的。

函数 ``cudaMallocManaged()`` 指定 cudaMemAttachHost 标志，该标志创建最初对设备端执行不可见的分配。（默认分配对所有流上的所有 GPU 核函数可见。）这确保在数据分配和当数据被获取到特定流之间的间隔中没有与其他线程执行的意外交互。

如果没有此标志，如果另一个线程启动的核函数恰好正在运行，则新分配将被视为在 GPU 上使用。这可能会影响线程在能够将数据显式附加到私有流之前从 CPU 访问新分配数据的能力。因此，为了实现线程之间的安全独立性，应指定此标志进行分配。

另一种选择是在分配附加到流后在所有线程之间放置一个进程范围的屏障。这将确保所有线程在任何核函数启动之前完成其数据/流关联，从而避免危险。在流销毁之前需要第二个屏障，因为流销毁会导致分配恢复其默认可见性。``cudaMemAttachHost`` 标志的存在既是为了简化此过程，也是因为在需要时并不总是可以插入全局屏障。

4.1.3.3.4. 流关联统一内存的数据移动
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 ``concurrentManagedAccess`` 未设置的设备上，使用流关联统一内存的 Memcpy()/Memset() 行为不同，适用以下规则：

如果指定了 ``cudaMemcpyHostTo*`` 并且源数据是统一内存，则如果它在复制流中可以从主机相干访问 [1]_；否则它将从设备访问。当指定 ``cudaMemcpy*ToHost`` 并且目的地是统一内存时，类似的规则适用于目的地。

如果指定了 ``cudaMemcpyDeviceTo*`` 并且源数据是统一内存，则它将从设备访问。源必须在复制流中可以从设备相干访问 [2]_；否则返回错误。当指定 ``cudaMemcpy*ToDevice`` 并且目的地是统一内存时，类似的规则适用于目的地。

如果指定了 ``cudaMemcpyDefault``，则统一内存将从主机访问，要么如果它不能在复制流中从设备相干访问 [2]_，要么如果数据的首选位置是 ``cudaCpuDeviceId`` 并且它可以在复制流中从主机相干访问 [1]_；否则，它将从设备访问。

当使用 ``cudaMemset*()`` 与统一内存时，数据必须在用于 ``cudaMemset()`` 操作的流中可以从设备相干访问 [2]_；否则返回错误。

当数据通过 ``cudaMemcpy*`` 或 ``cudaMemset*`` 从设备访问时，操作流被认为在 GPU 上处于活动状态。在此期间，任何与流关联的数据或具有全局可见性的数据的 CPU 访问，如果 GPU 的设备属性 ``concurrentManagedAccess`` 为零值，将导致段错误。程序必须适当同步以确保操作在完成之前从 CPU 访问任何关联数据。

.. [1] 在给定流中从主机相干访问意味着内存既没有全局可见性也没有与给定流关联。
.. [2] 在给定流中从设备相干访问意味着内存要么具有全局可见性，要么与给定流关联。


.. _um-perf-hints:

4.1.4. 性能提示
~~~~~~~~~~~~~~~~

性能提示允许程序员向 CUDA 提供有关统一内存使用的更多信息。CUDA 使用性能提示更有效地管理内存并提高应用程序性能。性能提示永远不会影响应用程序的正确性。性能提示只影响性能。

.. note::

   应用程序只应在统一内存性能提示提高性能时使用它们。

性能提示可用于任何统一内存分配，包括 CUDA 托管内存。在具有完整 CUDA 统一内存支持的系统上，性能提示可应用于所有系统分配的内存。

4.1.4.1. 数据预取
^^^^^^^^^^^^^^^^^

``cudaMemPrefetchAsync`` API 是一个异步流有序 API，可以迁移数据以驻留在更靠近指定处理器的位置。数据可以在预取时访问。迁移不会开始，直到流中的所有先前操作完成，并在流中的任何后续操作之前完成。

.. code-block:: c++

   cudaError_t cudaMemPrefetchAsync(const void *devPtr,
                                    size_t count,
                                    struct cudaMemLocation location,
                                    unsigned int flags,
                                    cudaStream_t stream=0);

包含 ``[devPtr, devPtr + count)`` 的内存区域可以迁移到目标设备 ``location.id``（如果 ``location.type`` 是 ``cudaMemLocationTypeDevice``）或 CPU（如果 ``location.type`` 是 ``cudaMemLocationTypeHost``），当预取任务在给定 ``stream`` 中执行时。有关 ``flags`` 的详细信息，请参阅当前的 `CUDA 运行时 API 文档 <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html>`_。

考虑下面的简单代码示例：

.. dropdown:: 系统分配器

   .. code-block:: c++

      void test_prefetch_sam(const cudaStream_t& s) {
        // 在 CPU 上初始化数据
        char *data = (char*)malloc(dataSizeBytes);
        init_data(data, dataSizeBytes);
        cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

        // 鼓励数据在使用前移动到 GPU
        const unsigned int flags = 0;
        cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

        // 在 GPU 上使用数据
        const unsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
        mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes);

        // 鼓励数据移回 CPU
        location = {.type = cudaMemLocationTypeHost};
        cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

        cudaStreamSynchronize(s);

        // 在 CPU 上使用数据
        use_data(data, dataSizeBytes);
        free(data);
      }

.. dropdown:: 托管

   .. code-block:: c++

      void test_prefetch_managed(const cudaStream_t& s) {
        // 在 CPU 上初始化数据
        char *data;
        cudaMallocManaged(&data, dataSizeBytes);
        init_data(data, dataSizeBytes);
        cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

        // 鼓励数据在使用前移动到 GPU
        const unsigned int flags = 0;
        cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

        // 在 GPU 上使用数据
        const unsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
        mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes);

        // 鼓励数据移回 CPU
        location = {.type = cudaMemLocationTypeHost};
        cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

        cudaStreamSynchronize(s);

        // 在 CPU 上使用数据
        use_data(data, dataSizeBytes);
        cudaFree(data);
      }

4.1.4.2. 数据使用提示
^^^^^^^^^^^^^^^^^^^^^

当多个处理器同时访问相同的数据时，``cudaMemAdvise`` 可用于提示 ``[devPtr, devPtr + count)`` 的数据将如何被访问：

.. code-block:: c++

   cudaError_t cudaMemAdvise(const void *devPtr,
                             size_t count,
                             enum cudaMemoryAdvise advice,
                             struct cudaMemLocation location);

示例展示了如何使用 ``cudaMemAdvise``：

.. code-block:: c++

   // test-prefetch-managed-begin
   void test_prefetch_managed(const cudaStream_t& s) {
     // 在 CPU 上初始化数据
     char *data;
     cudaMallocManaged(&data, dataSizeBytes);
     init_data(data, dataSizeBytes);
     cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

     // 鼓励数据在使用前移动到 GPU
     const unsigned int flags = 0;
     cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

     // 在 GPU 上使用数据
     const unsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
     mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes);

     // 鼓励数据移回 CPU
     location = {.type = cudaMemLocationTypeHost};
     cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

     cudaStreamSynchronize(s);

     // 在 CPU 上使用数据
     use_data(data, dataSizeBytes);
     cudaFree(data);
   }
   // test-prefetch-managed-end

   static const int maxDevices = 1;
   static const int maxOuterLoopIter = 3;
   static const int maxInnerLoopIter = 4;

其中 ``advice`` 可以采取以下值：

- ``cudaMemAdviseSetReadMostly``：

  这意味着数据主要将被读取，偶尔写入。通常，它允许在此区域上以读取带宽换取写入带宽。

- ``cudaMemAdviseSetPreferredLocation``：

  此提示将数据的首选位置设置为指定设备的物理内存。此提示鼓励系统将数据保留在首选位置，但不保证它。对 location.type 传递 ``cudaMemLocationTypeHost`` 的值将首选位置设置为主机内存。其他提示（如 ``cudaMemPrefetchAsync``）可能会覆盖此提示并允许内存从首选位置迁移。

- ``cudaMemAdviseSetAccessedBy``：

  在某些系统上，为了性能，在从给定处理器访问数据之前建立到内存的映射可能是有益的。此提示告诉系统数据将被 ``location.id`` 频繁访问（当 ``location.type`` 是 ``cudaMemLocationTypeDevice`` 时），使系统能够假设创建这些映射是值得的。此提示不暗示数据应该驻留在哪里，但可以与 ``cudaMemAdviseSetPreferredLocation`` 结合使用来指定。在硬件一致性系统上，此提示开启访问计数器迁移，参见 `访问计数器迁移 <#um-access-counters>`_。

每个提示也可以使用以下值之一取消设置：``cudaMemAdviseUnsetReadMostly``、``cudaMemAdviseUnsetPreferredLocation`` 和 ``cudaMemAdviseUnsetAccessedBy``。

示例展示了如何使用 ``cudaMemAdvise``：

.. dropdown:: 系统分配器

   .. code-block:: c++

      void test_advise_sam(cudaStream_t stream) {
        char *dataPtr;
        size_t dataSize = 64 * threadsPerBlock;  // 16 KiB

        // 使用 malloc 或 cudaMallocManaged 分配内存
        dataPtr = (char*)malloc(dataSize);

        // 在内存区域上设置提示
        cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
        cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);

        int outerLoopIter = 0;
        while (outerLoopIter < maxOuterLoopIter) {
          // 数据在每个外循环迭代中由 CPU 写入
          init_data(dataPtr, dataSize);

          // 数据通过预取使所有 GPU 可用。
          // 此处的预取导致数据读取重复而不是
          // 数据迁移
          cudaMemLocation location;
          location.type = cudaMemLocationTypeDevice;
          for (int device = 0; device < maxDevices; device++) {
            location.id = device;
            const unsigned int flags = 0;
            cudaMemPrefetchAsync(dataPtr, dataSize, location, flags, stream);
          }

          // 核函数在内循环中只读取此数据
          int innerLoopIter = 0;
          while (innerLoopIter < maxInnerLoopIter) {
            mykernel<<<32, threadsPerBlock, 0, stream>>>((const char *)dataPtr, dataSize);
            innerLoopIter++;
          }
          outerLoopIter++;
        }

        free(dataPtr);
      }

.. dropdown:: 托管

   .. code-block:: c++

      void test_advise_managed(cudaStream_t stream) {
        char *dataPtr;
        size_t dataSize = 64 * threadsPerBlock;  // 16 KiB

        // 使用 cudaMallocManaged 分配内存
        // （在具有完整 CUDA 统一内存支持的系统上可以使用 malloc）
        cudaMallocManaged(&dataPtr, dataSize);

        // 在内存区域上设置提示
        cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
        cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);

        int outerLoopIter = 0;
        while (outerLoopIter < maxOuterLoopIter) {
          // 数据在每个外循环迭代中由 CPU 写入
          init_data(dataPtr, dataSize);

          // 数据通过预取使所有 GPU 可用。
          // 此处的预取导致数据读取重复而不是
          // 数据迁移
          cudaMemLocation location;
          location.type = cudaMemLocationTypeDevice;
          for (int device = 0; device < maxDevices; device++) {
            location.id = device;
            const unsigned int flags = 0;
            cudaMemPrefetchAsync(dataPtr, dataSize, location, flags, stream);
          }

          // 核函数在内循环中只读取此数据
          int innerLoopIter = 0;
          while (innerLoopIter < maxInnerLoopIter) {
            mykernel<<<32, threadsPerBlock, 0, stream>>>((const char *)dataPtr, dataSize);
            innerLoopIter++;
          }
          outerLoopIter++;
        }

        cudaFree(dataPtr);
      }

4.1.4.3. 内存丢弃
^^^^^^^^^^^^^^^^^

``cudaMemDiscardBatchAsync`` API 允许应用程序通知 CUDA 运行时指定内存范围的内容不再有用。统一内存驱动程序执行自动内存传输以支持基于错误的迁移或内存驱逐以支持设备内存超额订阅。这些自动内存传输有时可能是冗余的，这会严重降低性能。将地址范围标记为"丢弃"将通知统一内存驱动程序应用程序已使用该范围的内容，无需在预取或页面驱逐时迁移此数据以为其他分配腾出空间。读取丢弃的页面而不进行后续写入访问或预取将产生不确定的值。而在丢弃操作之后的任何新写入保证被后续读取访问看到。对正在丢弃的地址范围的并发访问或预取将导致未定义的行为。

.. code-block:: c++

   cudaError_t cudaMemDiscardBatchAsync(void **dptrs,
                                        size_t *sizes,
                                        size_t count,
                                        unsigned long long flags,
                                        cudaStream_t stream);

该函数对 ``dptrs`` 和 ``sizes`` 数组中指定的地址范围执行一批内存丢弃。两个数组的长度必须与 ``count`` 指定的一样。每个内存范围必须引用通过 ``cudaMallocManaged`` 分配或通过 ``__managed__`` 变量声明的托管内存。

``cudaMemDiscardAndPrefetchBatchAsync`` API 结合了丢弃和预取操作。调用 ``cudaMemDiscardAndPrefetchBatchAsync`` 语义上等同于调用 ``cudaMemDiscardBatchAsync`` 后跟 ``cudaMemPrefetchBatchAsync``，但更优化。当应用程序需要内存位于目标位置但不需要内存内容时，这很有用。

.. code-block:: c++

   cudaError_t cudaMemDiscardAndPrefetchBatchAsync(void **dptrs,
                                                   size_t *sizes,
                                                   size_t count,
                                                   struct cudaMemLocation *prefetchLocs,
                                                   size_t *prefetchLocIdxs,
                                                   size_t numPrefetchLocs,
                                                   unsigned long long flags,
                                                   cudaStream_t stream);

``prefetchLocs`` 数组指定预取的目的地，而 ``prefetchLocIdxs`` 指示每个预取位置适用于哪些操作。例如，如果一批有 10 个操作，前 6 个应该预取到一个位置，其余 4 个预取到另一个位置，那么 ``numPrefetchLocs`` 将是 2，``prefetchLocIdxs`` 将是 {0, 6}，``prefetchLocs`` 将包含两个目标位置。

**重要考虑因素：**

- 从丢弃的范围读取而不进行后续写入或预取将返回不确定的值
- 丢弃操作可以通过写入范围或通过 ``cudaMemPrefetchAsync`` 预取它来撤销
- 与丢弃操作同时发生的任何读取、写入或预取将导致未定义的行为
- 所有设备必须具有非零值的 ``cudaDevAttrConcurrentManagedAccess``

4.1.4.4. 查询托管内存上的数据使用属性
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

程序可以使用以下 API 查询通过 ``cudaMemAdvise`` 或 ``cudaMemPrefetchAsync`` 分配的 CUDA 托管内存的内存范围属性：

.. code-block:: c++

   cudaMemRangeGetAttribute(void *data,
                            size_t dataSize,
                            enum cudaMemRangeAttribute attribute,
                            const void *devPtr,
                            size_t count);

此函数查询从 ``devPtr`` 开始、大小为 ``count`` 字节的内存范围的属性。内存范围必须引用通过 ``cudaMallocManaged`` 分配或通过 ``__managed__`` 变量声明的托管内存。可以查询以下属性：

- ``cudaMemRangeAttributeReadMostly``：如果整个内存范围设置了 ``cudaMemAdviseSetReadMostly`` 属性则返回 1，否则返回 0。

- ``cudaMemRangeAttributePreferredLocation``：如果整个内存范围将相应的处理器作为首选位置，则返回的结果将是 GPU 设备 ID 或 ``cudaCpuDeviceId``，否则返回 ``cudaInvalidDeviceId``。应用程序可以使用此查询 API 根据托管指针的首选位置属性来决定是通过 CPU 还是 GPU 暂存数据。注意，查询时内存范围的实际位置可能与首选位置不同。

- ``cudaMemRangeAttributeAccessedBy``：将返回为该内存范围设置了该提示的设备列表。

- ``cudaMemRangeAttributeLastPrefetchLocation``：将返回使用 ``cudaMemPrefetchAsync`` 显式预取内存范围的最后一个位置。注意，这仅仅返回应用程序请求预取内存范围的最后一个位置。它不指示预取操作是否已完成甚至开始。

- ``cudaMemRangeAttributePreferredLocationType``：它返回首选位置的位置类型，具有以下值：

  - ``cudaMemLocationTypeDevice``：如果内存范围中的所有页面都有相同的 GPU 作为其首选位置，
  - ``cudaMemLocationTypeHost``：如果内存范围中的所有页面都有 CPU 作为其首选位置，
  - ``cudaMemLocationTypeHostNuma``：如果内存范围中的所有页面都有相同的主机 NUMA 节点 ID 作为其首选位置，
  - ``cudaMemLocationTypeInvalid``：如果所有页面没有相同的首选位置或某些页面根本没有首选位置。

- ``cudaMemRangeAttributePreferredLocationId``：如果 ``cudaMemRangeAttributePreferredLocationType`` 查询对于相同地址范围返回 ``cudaMemLocationTypeDevice``，则返回设备序号。如果首选位置类型是主机 NUMA 节点，则返回主机 NUMA 节点 ID。否则，应忽略 ID。

- ``cudaMemRangeAttributeLastPrefetchLocationType``：返回所有页面通过 ``cudaMemPrefetchAsync`` 显式预取的最后一个位置类型。返回以下值：

  - ``cudaMemLocationTypeDevice``：如果内存范围中的所有页面都预取到相同的 GPU，
  - ``cudaMemLocationTypeHost``：如果内存范围中的所有页面都预取到 CPU，
  - ``cudaMemLocationTypeHostNuma``：如果内存范围中的所有页面都预取到相同的主机 NUMA 节点 ID，
  - ``cudaMemLocationTypeInvalid``：如果所有页面没有预取到相同的位置或某些页面从未预取过。

- ``cudaMemRangeAttributeLastPrefetchLocationId``：如果 ``cudaMemRangeAttributeLastPrefetchLocationType`` 查询对于相同地址范围返回 ``cudaMemLocationTypeDevice``，它将是有效的设备序号，或者如果它返回 ``cudaMemLocationTypeHostNuma``，它将是有效的主机 NUMA 节点 ID。否则，应忽略 ID。

此外，可以使用相应的 ``cudaMemRangeGetAttributes`` 函数查询多个属性。

4.1.4.5. GPU 内存超额订阅
^^^^^^^^^^^^^^^^^^^^^^^^^^^

统一内存使应用程序能够 *超额订阅* 任何单个处理器的内存：换句话说，
它们可以分配和共享大于系统中任何单个处理器内存容量的数组，从而能够（除其他外）对不适合单个 GPU 的数据集进行核外处理，而不会向编程模型添加显著的复杂性。

此外，可通过对应的 cudaMemRangeGetAttributes 函数查询多个属性。
