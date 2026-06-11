.. _um-details-intro:

4.1. 统一内存
=============

本节将详细解释当前各种统一内存（Unified Memory）范式的行为与用法。
前面 :ref:`memory-unified-memory` 已经介绍了如何判断适用哪种统一内存范式，并对每种范式进行了简要介绍。

如前所述，统一内存编程共有四种范式：

- 完全支持托管内存分配
- 完全支持所有分配的软件一致性
- 完全支持所有分配的硬件一致性
- 有限的统一内存支持

前三种涉及完全统一内存支持的范式具有非常相似的行为和编程模型，将在 :ref:`um-pageable-systems` 中涵盖，任何差异都会突出显示。

最后一种统一内存支持受限的范式在 :ref:`um-legacy-devices` 中详细讨论。

.. _um-pageable-systems:

4.1.1. 支持完整 CUDA 统一内存的设备上的统一内存
--------------------------------------------------------

这些系统包括硬件一致性内存系统，如 NVIDIA Grace Hopper 和启用了异构内存管理 (Heterogeneous Memory Management, HMM) 的现代 Linux 系统。
HMM 是一种基于软件的内存管理系统，提供与硬件一致性内存系统相同的编程模型。

Linux HMM 需要 Linux 内核版本 6.1.24+、6.2.11+ 或 6.3+，计算能力 7.5 或更高的设备，
以及安装了 `Open Kernel Modules <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#nvidia-open-gpu-kernel-modules>`_ 的 CUDA 驱动程序版本 535+。

.. note::

   我们将 CPU 和 GPU 共用同一张页表的系统称为 **硬件一致性系统** （hardware coherent systems）。
   而将 CPU 和 GPU 使用各自独立页表的系统称为 **软件一致性系统** （software-coherent）。

诸如 NVIDIA Grace Hopper 这类硬件一致性系统，为 CPU 和 GPU 提供了逻辑上统一的页表（详情请参阅 :ref:`um-hw-coherency` ）。
以下章节仅适用于硬件一致性系统：

- :ref:`um-access-counters`

.. _um-system-allocator:

4.1.1.1. 统一内存：深度案例解析
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

支持完整 CUDA 统一内存的系统，详见 :ref:`tab:unified-memory-levels` ，允许设备访问与其交互的主机进程拥有的任何内存。

本节将展示几个高级用例。这些用例使用了一个简单 kernel ，其功能仅仅是将输入字符数组的前 8 个字符打印到标准输出流中：

.. code-block:: cuda

   __global__ void kernel(const char* type, const char* data) {
     static const int n_char = 8;
     printf("%s - first %d characters: '", type, n_char);
     for (int i = 0; i < n_char; ++i) {
       printf("%c", data[i]);
     }
     printf("'\n");
   }

以下选项卡展示了使用系统分配内存调用该 kernel 的多种方式：

.. tab-set::

   .. tab-item:: Malloc

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

   .. tab-item:: Managed

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

   .. tab-item:: 栈变量

      .. code-block:: c++

         void test_stack() {
           const char test_string[] = "Hello World";
           kernel<<<1, 1>>>("stack", test_string);
           ASSERT(cudaDeviceSynchronize() == cudaSuccess,
             "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
         }

   .. tab-item:: 文件作用域静态变量

      .. code-block:: c++

          void test_static() {
           static const char test_string[] = "Hello World";
           kernel<<<1, 1>>>("static", test_string);
           ASSERT(cudaDeviceSynchronize() == cudaSuccess,
             "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
         }

   .. tab-item:: 全局作用域变量
 
      .. code-block:: c++

         const char global_string[] = "Hello World";

         void test_global() {
           kernel<<<1, 1>>>("global", global_string);
           ASSERT(cudaDeviceSynchronize() == cudaSuccess,
             "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
         }

   .. tab-item:: 全局作用域 extern 变量

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

请注意，对于 extern 变量，其声明、内存的所有权及管理可能完全由第三方库负责，而该第三方库甚至根本不会与 CUDA 发生任何交互。

另外需要注意的是，GPU 只能通过 **指针** 来访问栈变量（stack variables）以及文件作用域（file-scope）和全局作用域（global-scope）的变量。
在这个特定的示例中，这恰好非常方便，因为字符数组本身已经被声明为了指针类型（ ``const char*`` ）。
不过，请考虑以下使用全局整型变量的示例：

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

在上述示例中，我们需要确保将指向该全局变量的 **指针** 传递给 kernel ，而不是在 kernel 中直接访问该全局变量。
这是因为没有使用 ``__managed__`` 修饰符的全局变量默认被声明为仅限主机端（ ``__host__`` -only），因此就目前而言，大多数编译器不允许在设备代码中直接使用这些变量。

.. _um-file-backed-memory:

4.1.1.1.1. 基于文件的统一内存
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

由于支持完整 CUDA 统一内存的系统允许设备访问宿主进程所拥有的任意内存，因此它们可以直接访问基于文件的内存。

这里，我们对上一节中的初始示例进行了修改，使其使用基于文件的内存，以便 GPU 能够直接从输入文件中读取并打印字符串。
在下面的示例中，该内存由物理文件提供支持（backed），但该示例同样适用于以内存为支撑的文件（memory-backed files）。

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

请注意，在不具备 ``hostNativeAtomicSupported`` 属性的系统上（见 :ref:`um-host-native-atomics` ），包括启用了 Linux HMM 的系统，均不支持对基于文件的内存进行原子访问。

.. _um-ipc:

4.1.1.1.2. 使用统一内存的进程间通信 (IPC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   就目前而言，在统一内存中使用 IPC（进程间通信）可能会对性能产生显著影响。

许多应用程序倾向于为每个进程分配一个 GPU，但仍然需要使用统一内存（例如为了实现显存超额订阅），并且需要从多个 GPU 访问该内存。

CUDA IPC（见 :ref:`interprocess-communication-details` ）不支持托管内存（managed memory），本节中讨论的任何机制都不能用于共享此类内存句柄。
但在支持完整 CUDA 统一内存的系统上，系统分配的内存是支持 IPC 的。
一旦将系统分配内存的访问权限与其他进程共享，其适用的编程模型将与基于文件的统一内存（File-backed Unified Memory）类似。

有关在 Linux 下创建支持 IPC 的系统分配内存的各种方法，请参阅以下参考资料：

- `mmap with MAP_SHARED <https://man7.org/linux/man-pages/man2/mmap.2.html>`_
- `POSIX IPC APIs <https://pubs.opengroup.org/onlinepubs/007904875/functions/shm_open.html>`_
- `Linux memfd_create <https://man7.org/linux/man-pages/man2/memfd_create.2.html>`_

请注意，使用该技术无法在不同的主机及其设备之间共享内存。

.. _um-performance-tuning:

4.1.1.2. 性能调优
~~~~~~~~~~~~~~~~~

为了在使用统一内存时获得良好的性能，以下几点至关重要：

- 了解系统上的分页机制是如何工作的，如何避免不必要的缺页中断（page faults）；
- 了解各种能够让数据保留在访问该数据的处理器本地的机制；
- 考虑针对系统的内存传输粒度对应用程序进行调优。

作为一般性建议，使用 :ref:`um-perf-hints` 可能会提升性能，但如果使用不当，反而会导致性能低于默认行为。
此外请注意，任何提示都会给主机端带来一定的性能开销，因此，有用的提示至少必须能带来足够的性能提升，以抵消这部分开销。

.. _um-page-size:

4.1.1.2.1. 内存分页和页大小
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为了更好地理解统一内存对性能的影响，了解虚拟地址、内存页和页大小等概念至关重要。
本小节旨在定义所有必要的术语，并解释为什么分页机制对性能如此重要。

当前所有支持统一内存的系统都使用虚拟地址空间，这意味着应用程序使用的内存地址代表的是一个虚拟位置，该位置可能会被映射到内存实际驻留的物理位置。

此外，当前所有处理器（包括 CPU 和 GPU）都使用了内存 `分页机制` 。
由于所有系统都使用虚拟地址空间，因此存在两种类型的内存页：

- **虚拟页** ：这代表由操作系统跟踪的、属于每个进程的固定大小的连续虚拟内存块，它可以被映射到物理内存中。
  请注意，虚拟页是与 *映射关系* 绑定在一起的。例如，同一个虚拟地址可以使用不同的页大小来映射到物理内存。

- **物理页** ：这代表处理器的内存管理单元（MMU）所支持的固定大小的连续内存块，虚拟页可以被映射到其中。

目前，所有 x86_64 架构的 CPU 默认使用的物理页大小为 4KiB。
Arm 架构的 CPU 则根据具体型号的不同，支持多种物理页大小——包括 4KiB、16KiB、32KiB 和 64KiB。
最后，NVIDIA GPU 同样支持多种物理页大小，但更倾向于使用 2MiB 或更大的物理页。
请注意，这些规格在未来的硬件中可能会有所改变。

虚拟页的默认页大小通常与物理页大小相对应，但只要操作系统和硬件支持，应用程序也可以使用不同的页大小。
通常情况下，受支持的虚拟页大小必须是 2 的幂次方，并且是物理页大小的整数倍。

用于跟踪虚拟页到物理页映射关系的逻辑实体被称为页表（page table），而将具有特定虚拟大小的给定虚拟页映射到物理页的具体记录，则称为页表项（Page Table Entry, PTE）。
所有受支持的处理器都为页表提供了专用的缓存，以加速虚拟地址到物理地址的转换过程。
这些缓存被称为转换后备缓冲器（Translation Lookaside Buffers, TLB）。

性能调优有两个重要方面：

- 虚拟页大小的选择
- CPU 和 GPU 使用的统一页表，还是为每个 CPU 和 GPU 分别提供独立的页表。

.. _um-choosing-page-size:

4.1.1.2.1.1. 选择合适的页大小
"""""""""""""""""""""""""""""

通常来说，较小的页大小（Page Size）会导致较少的（虚拟）内存碎片，但会增加 TLB 未命中（TLB misses）的次数；
而较大的页大小则会导致更多的内存碎片，但能减少 TLB 未命中。
此外，由于系统通常以完整的内存页为单位进行迁移，因此与较小的页相比，使用较大页进行内存迁移的成本通常更高。
这可能会导致使用大页的应用程序出现更明显的延迟尖峰（latency spikes）。
有关缺页中断（page faults）的更多详细信息，请参阅下一节。

在性能调优中，一个非常重要的方面是：与 CPU 相比，GPU 上的 TLB 未命中带来的性能惩罚通常要大得多。
这意味着，如果 GPU 线程频繁地随机访问使用较小页大小映射的统一内存，其速度可能会明显慢于访问使用较大页大小映射的统一内存。
虽然 CPU 线程随机访问使用小页映射的大片内存区域时也会出现类似的性能下降，但其影响相对不那么显著。
因此，应用程序可能需要在承受这种 `性能下降` 与减少 `内存碎片化` 之间进行权衡。

需要注意的是，应用程序通常不应根据特定处理器的物理页大小来调整性能，因为物理页大小可能会随着硬件的更迭而发生改变。
上述建议仅适用于虚拟页大小。


.. _um-hw-coherency:

4.1.1.2.1.2. CPU 和 GPU 页表：硬件一致性与软件一致性
"""""""""""""""""""""""""""""""""""""""""""""""""""""

像 NVIDIA Grace Hopper 这样的硬件一致性（Hardware-coherent）系统，为 CPU 和 GPU 提供了逻辑上统一的页表。
这一点非常重要，因为当 GPU 需要访问由系统分配的内存时，它必须使用 CPU 为该内存创建的页表项（PTE）。
如果该页表项使用的是 CPU 默认的 4KiB 或 64KiB 页大小，那么在访问大段虚拟内存区域时，将会导致严重的 TLB 未命中，从而引发显著的性能下降。

另一方面，在 CPU 和 GPU 各自拥有独立逻辑页表的软件一致性（software-coherent）系统中，需要考虑不同的性能调优策略：
为了保证数据一致性，当一个处理器访问映射到另一个处理器物理内存中的地址时，这些系统通常会触发缺页中断（page fault）。
这种缺页中断意味着：

- 需确保当前持有该物理页的处理器（即该物理页当前所在的处理器）无法再访问此页面，具体可通过删除或更新页表项实现。
- 需确保发起访问请求的处理器能够访问该页面，可通过新建页表项或更新已有页表项使其转为有效 / 启用状态来实现。
- 支撑该虚拟页的物理页必须被移动（或迁移）到请求访问的处理器上：这是一项开销较大的操作，并且所需的工作量与页大小成正比。

总体而言，在 CPU 和 GPU 线程频繁并发访问同一内存页的情况下，硬件一致性系统相比软件一致性系统能提供显著的性能优势：

- 更少的缺页中断，这些系统不需要利用缺页中断来模拟数据一致性或迁移内存。
- 竞争更少，这类系统以 `缓存行` 为粒度维护一致性，而非以页面为粒度。
  也就是说，当多个处理器对同一缓存行产生访问竞争时，仅需传输该缓存行 —— 其尺寸远小于最小页面；而当不同处理器访问同一页面内的不同缓存行时，不会产生竞争。

这影响以下场景的性能：

- CPU 和 GPU 同时对同一地址进行原子更新
- 从 CPU 线程向 GPU 线程发送信号，反之亦然。

.. _um-mixed-coherency:

4.1.1.2.1.3. 混合硬件和软件一致性
"""""""""""""""""""""""""""""""""""

部分具备硬件一致性的系统（例如 NVIDIA DGX Station）也支持安装独立的非一致性的 GPU 硬件。
在这种情况下，硬件一致性 GPU 发起的内存访问将继续采用 :ref:`um-hw-coherency` 中描述的基于硬件的一致性机制；
而独立 GPU 发起的内存访问则将采用基于软件的一致性机制。

两类 GPU 可共用统一地址空间。受访问端 GPU 类型影响，系统性能与内存迁移表现会存在差异。
具体来说，软件一致性 GPU 发起的访问会产生更多缺页中断与内存迁移；而硬件一致性 GPU 的缺页中断更少，且会在可行情况下启用远程映射。

为实现最优性能，应尽量减少两类 GPU 间的数据共享，或通过显式拷贝完成数据交互。
也可调用 ``cudaMemAdviseSetPreferredLocation`` 等接口，将高频共享数据固定存放于 CPU 内存或硬件一致性 GPU 内存中。
这是因为默认状态下，访问软件一致性内存会触发缺页与数据迁移。

在混合一致性系统中， ``cudaHostRegister`` 以及其他主机内存访问接口针对软件一致性 GPU 的行为会有所不同。
该类系统中的软件一致性 GPU 不再使用锁页映射（pinned mappings），而是采用 CPU 页表软件镜像机制。
这会使得部分 GPU 访问操作触发缺页中断，这类情况在非混合一致性系统中不会出现；不过此类缺页中断并不常见，一般仅在内存资源吃紧时才会发生。

.. _um-host-direct-access:

4.1.1.2.2. 从主机直接访问统一内存
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

部分设备支持主机直接对 GPU 端驻留的统一内存执行一致性读、写及原子访问操作。
这类设备的属性 ``cudaDevAttrDirectManagedMemAccessFromHost`` 会被置为 1。
需注意：所有搭载硬件一致性、且通过 NVLink 互联的设备，该属性均默认启用。
在此类系统中，主机可直接访问 GPU 驻留内存，全程不会产生缺页中断与数据迁移。
另外，使用 CUDA 托管内存时，若想实现无缺页的直接访问，必须使用位置类型 ``cudaMemLocationTypeHost`` 为 ``cudaMemAdviseSetAccessedBy`` 的访问提示，具体示例见下文。

.. tab-set::

   .. tab-item:: 系统分配器

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

           write<<< 1, 1000 >>>(ret, 10, 100);  // pages populated in GPU memory
           cudaDeviceSynchronize();
           for(int i = 0; i < 1000; i++)
               printf("%d: A+B = %d\n", i, ret[i]);  // directManagedMemAccessFromHost=1: CPU accesses GPU memory directly without migrations
                                                     // directManagedMemAccessFromHost=0: CPU faults and triggers device-to-host migrations
           append<<< 1, 1000 >>>(ret, 10, 100);  // directManagedMemAccessFromHost=1: GPU accesses GPU memory without migrations
           cudaDeviceSynchronize();  // directManagedMemAccessFromHost=0: GPU faults and triggers host-to-device migrations
           free(ret);
         }

   .. tab-item:: 托管

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
 
           write<<< 1, 1000 >>>(ret, 10, 100);  // pages populated in GPU memory
           cudaDeviceSynchronize();
           for(int i = 0; i < 1000; i++)
               printf("%d: A+B = %d\n", i, ret[i]);  // directManagedMemAccessFromHost=1: CPU accesses GPU memory directly without migrations
                                                     // directManagedMemAccessFromHost=0: CPU faults and triggers device-to-host migrations
           append<<< 1, 1000 >>>(ret, 10, 100);  // directManagedMemAccessFromHost=1: GPU accesses GPU memory without migrations
           cudaDeviceSynchronize();  // directManagedMemAccessFromHost=0: GPU faults and triggers host-to-device migrations
           cudaFree(ret);
         }

在 ``write`` 核函数完成后， ``ret`` 将在 GPU 内存中创建和初始化。
接下来，CPU 将访问 ``ret`` ，然后 ``append`` 核函数使用相同的 ``ret`` 内存。
这段代码根据系统架构和硬件一致性支持显示出不同的行为：

- 在 ``directManagedMemAccessFromHost=1`` 的系统上：对托管缓冲区的 CPU 访问不会触发任何迁移；
  数据将保留在 GPU 内存中，任何后续的 GPU 核函数都可以继续直接访问它，而不会导致错误或迁移
- 在 ``directManagedMemAccessFromHost=0`` 的系统上：对托管缓冲区的 CPU 访问将页错误并启动数据迁移；
  任何尝试首次访问相同数据的 GPU 核函数将页错误并将页迁移回 GPU 内存

.. _um-host-native-atomics:

4.1.1.2.3. 主机原生原子操作
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

部分设备（采用 NVLink 互联的硬件一致性系统设备也包含在内）支持硬件原子访问驻留在 CPU 端的内存。
这意味着，对主机内存执行原子操作时，无需借助缺页异常来软件模拟实现。
这类设备对应的属性 ``cudaDevAttrHostNativeAtomicSupported`` 会被置为 1。


.. _um-atomics:

4.1.1.2.4. 原子访问和同步原语
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA 统一内存支持主机线程与设备线程可用的全部原子操作，让所有线程能够通过并发访问同一共享内存地址实现协同工作。
`libcu++ <https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives.html>`_  库提供了大量针对主机与设备线程并发交互优化的异构同步原语，
包括 ``cuda::atomic`` 、  ``cuda::atomic_ref`` 、 ``cuda::barrier`` 、 ``cuda::semaphore`` 等。

在软件一致性系统中，设备端无法对基于文件映射的主机内存执行原子访问。
下方示例代码在硬件一致性系统中可正常运行，但在其他系统上会产生未定义行为。

.. code-block:: c++

   #include <cuda/atomic>

   #include <cstdio>
   #include <fcntl.h>
   #include <sys/mman.h>

   #define ERR(msg, ...) { fprintf(stderr, msg, ##__VA_ARGS__); return EXIT_FAILURE; }

   __global__ void kernel(int* ptr) {
     // atomic accesses from the device to file-backed host memory are not supported
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

在软件一致性系统中，对统一内存执行原子访问可能触发缺页异常，进而带来显著的访问延迟。
需要注意：并非该类系统上所有 GPU 对主机内存的原子操作都会出现此问题；通过命令 ``nvidia-smi -q | grep "Atomic Caps Outbound"`` 查询到的受支持操作，可避免触发缺页。

在硬件一致性系统中，主机与设备间的原子访问不会因跨设备访问触发缺页异常，但仍可能因所有内存访问都会遇到的通用问题产生缺页。


.. _um-memcpy-memset-behavior:

4.1.1.2.5. 统一内存下 memcpy () /memset () 的行为
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``cudaMemcpy*()`` 与 ``cudaMemset*()`` 系列接口，均支持传入任意统一内存指针作为参数。

对于各类 ``cudaMemcpy*()`` 接口，由 ``cudaMemcpyKind`` 指定的数据传输方向仅作为性能提示；若入参中包含统一内存指针，该提示对性能的影响会更加明显。

因此，建议遵循以下性能建议：

- 当已知统一内存的物理位置时，请使用准确的 ``cudaMemcpyKind`` 提示。
- 如果无法提供准确的 ``cudaMemcpyKind`` 提示，请优先使用 ``cudaMemcpyDefault`` 。
- 务必使用已填充（已初始化）的缓冲区，不要借助这类接口做内存初始化操作。
- 当两个指针都指向系统分配的内存时，应避免使用 ``cudaMemcpy*()`` 函数：请改为启动核函数，或使用 CPU 内存拷贝算法（如 ``std::memcpy`` ）。

.. _um-allocator-overview:

4.1.1.2.6. 统一内存的内存分配器概述
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

对于支持完整 CUDA 统一内存的系统，可以使用各种不同的分配器（allocators）来分配统一内存。
下表概述了部分分配器及各自的功能特性。
请注意，本节中的所有信息在未来的 CUDA 版本中可能会发生变化。

.. list-table:: 不同分配器对统一内存的支持概览
   :widths: 25 20 15 20 20
   :header-rows: 1

   * - API
     - 放置策略
     - 可访问自
     - 基于访问迁移 [\ [#f2]_ ]
     - 页大小 [\ [#f4]_ ] [\ [#f5]_ ]
   * - | ``malloc`` 
       | ``new``
       | ``mmap``
     - | 首次访问 /
       | 内存提示 [\ [#f1]_ ]
     - CPU、GPU
     - 是 [\ [#f3]_ ]
     - 系统或大页 [\ [#f6]_ ]
   * - ``cudaMallocManaged``
     - | 首次访问 /
       | 内存提示
     - CPU、GPU
     - 是
     - | CPU 驻留：
       | 系统页大小
       |
       | GPU 驻留：
       | 2MB
   * - ``cudaMalloc``
     - GPU
     - GPU
     - 否
     - | GPU 页大小：
       | 2MB
   * - | ``cudaMallocHost``
       | ``cudaHostAlloc`` 
       | ``cudaHostRegister``
     - CPU
     - CPU、GPU
     - 否
     - | CPU 映射：
       | 系统页大小
       |
       | GPU 映射：
       | 2MB
   * - | 内存池，主机：
       | ``cuMemCreate``
       | ``cudaMemPoolCreate``
     - CPU
     - CPU、GPU
     - 否
     - | CPU 映射：
       | 系统页大小
       | 
       | GPU 映射：
       | 2MB
   * - | 内存池，设备：
       | ``cuMemCreate``
       | ``cudaMemPoolCreate``
       | ``cudaMallocAsync``
     - GPU
     - GPU
     - 否
     - 2MB

.. [#f1] 对于 ``mmap`` ，除非通过 ``cudaMemAdviseSetPreferredLocation`` （或 ``mbind`` ，见下方要点）另行指定，否则基于文件的内存默认会被放置在 CPU 端。
.. [#f2] 此功能可以通过 ``cudaMemAdvise`` 进行覆盖。即使禁用了基于访问的迁移，如果底层内存空间已满，内存仍可能会发生迁移。
.. [#f3] 基于文件的内存不会根据访问迁移。
.. [#f4] 在大多数系统上，默认的系统页大小为 4KiB 或 64KiB，除非显式指定了大页大小（例如，使用 ``mmap`` ``MAP_HUGETLB`` / ``MAP_HUGE_SHIFT``）。
         在这种情况下，支持在系统上配置的任何大页大小。
.. [#f5] 驻留在 GPU 端的内存所使用的页大小，可能会在后续 CUDA 版本中发生调整。
.. [#f6] 目前，当内存迁移至 GPU、或是因 GPU 端首次访问完成内存驻留时，原有的大页配置可能无法保留。

上面的表格展示了多种分配器在语义上的差异。这些分配器可用于分配能同时被多个处理器（包括主机和设备）访问的数据。
有关 ``cudaMemPoolCreate`` 的更多详细信息，请参阅 :ref:`soma-memory-pools` 章节；
有关 ``cuMemCreate`` 的更多详细信息，请参阅 :ref:`virtual-memory-management-details` 章节。

在硬件一致性系统中，设备内存会作为一个 NUMA 域暴露给操作系统。
此时，可以使用 ``numa_alloc_on_node`` 等专用分配器将内存固定到指定的 NUMA 节点（无论是主机节点还是设备节点）。
这部分内存可同时被主机和设备访问，且不会发生迁移。
类似地，也可以使用 ``mbind`` 将内存固定到指定的一个或多个 NUMA 节点，它还能使基于文件的内存在首次被访问之前，就提前放置在指定的 NUMA 节点上。

以下规则适用于共享内存相关的各类内存分配器：

- 系统分配器（如 ``mmap`` ）允许使用 ``MAP_SHARED`` 标志在进程之间共享内存。
  CUDA 支持这一特性，可用于在同一主机连接的不同设备之间共享内存。
  但是，目前不支持通过这种方式在多台主机之间或多台设备之间共享内存。有关详细信息，请参阅 :ref:`um-ipc` 。

- 如果需要在多台主机之间通过网络访问统一内存或其他 CUDA 内存，请查阅所使用的通信库的文档，例如：
  `NCCL <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html>`_ 、
  `NVSHMEM <https://docs.nvidia.com/nvshmem/api/index.html>`_ 、
  `OpenMPI <https://www.open-mpi.org/faq/?category=runcuda>`_ 、
  `UCX <https://docs.mellanox.com/category/hpcx>`_ 
  等。

.. _um-access-counters:

4.1.1.2.7. 基于访问计数的迁移
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在硬件一致性系统中，访问计数功能会统计 GPU 对其他处理器上内存的访问频次。
该机制用于将内存页面迁移至访问最频繁的处理器物理内存中，可调度页面在 CPU 与 GPU、以及对等 GPU 之间迁移，这一过程称为访问计数器迁移。

从 CUDA 12.4 开始，系统分配内存支持访问计数功能。
需要注意：文件映射内存不会根据访问行为触发页迁移。
对于系统分配内存，可通过 ``cudaMemAdviseSetAccessedBy`` 提示并指定对应设备 ID，启用访问计数器迁移。
若已开启该功能，调用 ``cudaMemAdviseSetPreferredLocation`` 并设置为主机端，即可阻止内存发生迁移。
默认情况下， ``cudaMallocManaged`` 采用缺页触发迁移机制完成页面搬迁。[\ [#f7]_ ]

驱动程序还可借助访问计数，更高效地缓解内存抖动，或应对内存超配场景。

.. [#f7] 当前系统，允许对设置了访问设备提示的托管内存使用访问计数迁移，但这仅仅是底层的实现细节，在未来的版本中不应依赖此实现以保证兼容性。

.. _um-traffic-hd:

4.1.1.2.8. 避免 CPU 频繁写入 GPU 驻留内存
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果主机（CPU）访问统一内存，缓存未命中（cache misses）可能会在主机与设备之间引入超出预期的通信流量。
许多 CPU 架构要求所有的内存操作（包括写入操作）都必须经过缓存层级。
如果当前内存驻留在 GPU 上，这意味着 CPU 对该内存的频繁写入会引发缓存未命中，导致数据必须先由 GPU 传输回 CPU，然后才能将实际的值写入目标内存。
在软件一致性系统上，这可能会引发额外的缺页异常；而在硬件一致性系统上，这可能会导致 CPU 操作之间的延迟显著增加。
因此，为了将主机生成的数据共享给设备，建议将数据写入驻留在 CPU 端的内存中，并让设备直接从该内存读取数据。下面的代码展示了如何使用统一内存来实现这一操作。

.. tab-set::

   .. tab-item:: 系统分配器

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

   .. tab-item:: 托管

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

.. _um-async-access:

4.1.1.2.9. 利用对系统内存的异步访问
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果应用程序需要将设备端的计算结果共享给主机，可选择以下多种实现方案：

1. 设备将计算结果写入 GPU 内存，再通过 ``cudaMemcpy*`` 接口完成数据搬运（至主机内存），最后由主机读取搬运后的数据。
2. 设备直接将其结果写入 CPU 内存，主机读取该数据。
3. 设备写入 GPU 内存，主机直接访问该数据。

若主机传输、读取结果期间，设备仍可调度执行独立任务，方案 1 与方案 3 为首选。
如果设备会因等待主机读取结果而陷入空闲，则建议选用方案 2 。
这是因为通常情况下，设备的写入带宽高于主机的读取带宽，除非使用大量主机线程并发读取数据。

.. tab-set::

   .. tab-item:: 1. 显式复制

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

   .. tab-item:: 2. 设备直接写入

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

   .. tab-item:: 3. 主机直接读取

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

最后，在上面显式拷贝示例中，除了使用 ``cudaMemcpy*`` 外，也可通过主机端或设备端 kernel 完成显式数据搬运。
对于连续数据，优先使用 CUDA 拷贝引擎，因为拷贝引擎可与主机、设备的计算任务并行执行。
``cudaMemcpy*`` 与 ``cudaMemPrefetchAsync`` 接口可能会调用拷贝引擎，但无法保证每次调用都启用。
基于同样原因，当数据量较大时，相比主机直接读取，更推荐采用显式拷贝：若主机与设备的运算负载均未打满各自的内存带宽，拷贝引擎便可在主机与设备执行业务的同时，并行完成数据传输。

在使用 NVLink 互联的系统中，拷贝引擎通常被用于主机与设备之间、以及对等设备之间的数据传输。
由于拷贝引擎总数有限，部分场景下 ``cudaMemcpy*`` 的传输带宽会低于由设备主动发起的显式拷贝（由 kernel 搬运）。
此时若数据传输处于应用的关键路径上，建议改用基于设备的显式拷贝方案。

.. _um-no-pageable-systems:

4.1.2. 仅支持 CUDA 托管内存的设备上的统一内存
-------------------------------------------------

对于计算能力 6.x 及以上、但不支持可分页内存访问的设备，请参考表 :ref:`tab:unified-memory-levels` 。
此类设备完整支持 CUDA 托管内存，且内存具备一致性，但 GPU 无法访问系统分配内存。
这类设备的统一内存编程模型与性能调优方式，和 :ref:`um-pageable-systems` 所描述的内容大体一致，主要区别在于：无法使用系统内存分配器。
因此，以下章节内容不适用：

- :ref:`um-system-allocator`
- :ref:`um-hw-coherency`
- :ref:`um-atomics`
- :ref:`um-access-counters`
- :ref:`um-traffic-hd`
- :ref:`um-async-access`

.. _um-legacy-devices:

4.1.3. Windows、WSL 和 Tegra 上的统一内存
--------------------------------------------

.. note::
   本节仅针对以下设备与平台：计算能力低于 6.0 的设备、Windows 平台以及 **concurrentManagedAccess** 属性值为 **0** 的设备。

计算能力低于 6.0 的设备、Windows 平台设备，以及 ``concurrentManagedAccess`` 属性为 **0** 的设备，请参见 :ref:`tab:unified-memory-levels` 。
这类设备支持 CUDA 托管内存，但存在以下限制：

- **数据迁移和一致性**：不支持按需将托管内存数据细粒度迁移至 GPU。
  每当启动 GPU 核函数时，所有托管内存通常都需提前传输至 GPU 显存，避免访问内存时触发缺页。
  仅 CPU 端支持内存缺页机制。
- **GPU 内存超额订阅**：可分配的托管内存总量，不得超过 GPU 物理显存容量。
- **一致性和并发性**：无法对托管内存进行并发访问。原因在于这类设备缺少 GPU 端缺页机制，若 GPU 内核正在运行时 CPU 同时访问统一内存，将无法保障内存一致性。

.. _um-legacy-multi-gpu:

4.1.3.1. 多 GPU
~~~~~~~~~~~~~~~

在搭载计算能力低于 6.0 的设备，或是 Windows 平台的系统中，托管内存分配区会借助 GPU 对等访问能力，自动对系统内所有 GPU 可见。
托管内存的表现形式与通过 ``cudaMalloc()`` 分配的非托管内存相近：物理内存会驻留在当前活跃设备上，系统内其他 GPU 需经由 PCIe 总线访问该内存，访问带宽会有所下降。

在 Linux 系统中，只要应用当前使用的所有 GPU 均支持对等访问，托管内存就会分配在 GPU 显存中。
一旦应用开始使用某块 GPU，而该 GPU 与其他存有托管内存的任一 GPU 之间不支持对等访问，驱动便会将全部托管内存迁移至系统内存。
此时，所有 GPU 访问该内存都会受到 PCIe 总线带宽的限制。

在 Windows 系统中，若对等映射不可用（例如不同架构的 GPU 之间），无论应用是否实际同时使用两块显卡，系统都会自动降级为映射内存模式。
如果程序仅需使用单块 GPU，需要在启动程序前配置 ``CUDA_VISIBLE_DEVICES`` 环境变量。
该变量可限定程序可见的 GPU 设备，从而让托管内存能够分配在 GPU 显存中。

在 Windows 系统中，还可以将环境变量 ``CUDA_MANAGED_FORCE_DEVICE_ALLOC`` 设为非零值，强制驱动程序始终使用设备显存作为托管内存的物理存储介质。
该环境变量启用后，当前进程内所有支持托管内存的设备，彼此之间必须兼容对等访问。
若新增的托管内存设备，与进程中此前已使用的其他同类型设备无法建立对等访问，接口将返回 ``cudaErrorInvalidDevice`` 错误；
即便已对旧设备调用过 ``::cudaDeviceReset`` ，该限制依然生效。
相关环境变量的详细说明请参见 :ref:`environment-variables-details` 。

.. _um-coherency-concurrency:

4.1.3.2. 一致性和并发性
~~~~~~~~~~~~~~~~~~~~~~~

为确保内存一致性，统一内存编程模型对 CPU、GPU 并发数据访问施加了限制。
实际规则为：只要有任意 kernel 正在执行，GPU 就独占全部托管内存数据，CPU 不允许访问这些数据，无论该 kernel 是否真正读写对应数据。
即便 CPU、GPU 分别访问两块互不相同的托管内存，这种并发访问行为也会触发段错误，因为此时内存页对 CPU 标记为不可访问。

举个例子：下面这段代码在计算能力 6.x 的设备上可以正常运行，这类设备具备 GPU 缺页机制，取消了并发访问限制；
但在 6.0 之前的架构设备以及 Windows 平台上运行会报错，原因是 CPU 访问变量 ``y`` 时，GPU kernel 仍处于执行状态。

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

程序在访问变量 ``y`` 之前，必须与 GPU 执行显式同步操作，无论该 GPU 内核是否实际读写了 ``y`` （甚至完全未访问任何托管内存）。

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

请注意：任何能够从逻辑上保证 GPU 任务执行完毕的函数调用，都可用于确认 GPU 运算已全部完成，详见 :ref:`explicit-synchronization` 。

注意：若在 GPU 处于运行状态时，通过 ``cudaMallocManaged()`` 或 ``cuMemAllocManaged()`` 动态分配内存，在提交新任务或完成 GPU 同步之前，该段内存的行为是未定义的。
在此期间 CPU 尝试访问这块内存，有可能触发段错误，也有可能不会。
该规则不适用于通过 ``cudaMemAttachHost`` 或 ``CU_MEM_ATTACH_HOST`` 标识分配的内存。

.. _um-stream-associated:

4.1.3.3. 流关联的统一内存
~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA 编程模型提供流机制来标识各个 kernel 启动操作之间的依赖或无依赖关系。
提交至同一个流的 kernel 一定会按顺序串行执行；而提交至不同流的 kernel 允许并发执行。详见 :ref:`cuda-streams` 。

.. _um-stream-callback:

4.1.3.3.1. 流回调
^^^^^^^^^^^^^^^^^

只要 GPU 上没有其他可能访问统一内存的流处于活动状态，CPU 在流回调函数（stream callback）中访问统一内存就是合法的。
此外，如果某个回调函数之后不再有任何设备端任务，则该回调可用于同步操作：例如，可以在回调内部触发条件变量；否则，CPU 仅在回调函数执行期间对统一内存的访问才是有效的。
以下是几个需要注意的重要事项：

- 当 GPU 处于活动状态时，CPU 始终可以访问非托管的映射内存数据。
- 只要 GPU 正在运行任何 kernel ，即被视为处于活动状态，即使该 kernel 并未使用统一内存。如果某个 kernel 可能会访问这些数据，则禁止 CPU 进行访问。
- 对于多 GPU 并发访问统一内存，除了适用于非托管内存的多 GPU 访问限制外，没有其他额外约束。
- 多个 GPU kernel 并发访问统一内存数据也没有任何限制。


请注意最后一条规则会导致多个 GPU kernel 之间产生数据竞争，这一点与非托管内存的现有表现一致。
从 GPU 的视角来看，托管内存和非托管内存的行为完全相同。
下面的代码示例将对以上要点进行演示说明：

.. code-block:: c++

   int main() {
       cudaStream_t stream1, stream2;
       cudaStreamCreate(&stream1);
       cudaStreamCreate(&stream2);
       int *non_managed, *managed, *also_managed;
       cudaMallocHost(&non_managed, 4);  // Non-managed, CPU-accessible memory
       cudaMallocManaged(&managed, 4);
       cudaMallocManaged(&also_managed, 4);
       
       kernel<<< 1, 1, 0, stream1 >>>(managed);
       *non_managed = 1;  // Point 1: CPU can access non-managed data.
       // Point 2: CPU cannot access any managed data while GPU is busy,
       //          unless concurrentManagedAccess = 1
       // Note we have not yet synchronized, so "kernel" is still active.
       *also_managed = 2;      // Will issue segmentation fault
       // Point 3: Concurrent GPU kernels can access the same data.
       kernel<<< 1, 1, 0, stream2 >>>(managed);
       // Point 4: Multi-GPU concurrent access is also permitted.
       cudaSetDevice(1);
       kernel<<< 1, 1 >>>(managed);
       return  0;
   }

.. _um-stream-fine-grained-control:

4.1.3.3.2. 与流关联的托管内存允许更精细的控制
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

统一内存基于流独立性模型，允许 CUDA 程序将托管内存分配块显式绑定到某条 CUDA 流。
通过这种方式，程序员可以根据 kernel 是否被提交到指定的流中，来表明该 kernel 对数据的使用情况。
这使得系统能够根据程序特定的数据访问模式，发掘并发执行的机会。用于控制该行为的接口如下：

.. code-block:: c++

   cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream,
                                        void *ptr,
                                        size_t length=0,
                                        unsigned int flags=0);

``cudaStreamAttachMemAsync()`` 函数将从 ``ptr`` 开始、长度为 ``length`` 字节的内存区域与指定的流相关联。
只要该流中的所有操作均已完成，CPU 即可访问该内存区域，而无需考虑其他流是否处于活动状态。
实际上，这将活动 GPU 对统一内存区域的独占所有权范围，从整个 GPU 级别缩小到了单个流的级别。
最为关键的是，如果某块内存分配未与特定流关联，则所有正在运行的 kernel （无论其属于哪个流）均可访问该内存。
这正是 ``cudaMallocManaged()`` 分配的内存或 ``__managed__`` 变量的默认行为；
因此，简单规则是：在任何 kernel 运行期间，CPU 不得访问这些数据。

.. note::

   通过将内存分配与特定流关联，程序即作出保证：只有提交到该流中的 kernel 才会访问该数据。
   统一内存系统不会对此进行任何错误检查。

.. note::

   除了能够实现更高程度的并发外，使用 ``cudaStreamAttachMemAsync()`` 还可以启用统一内存系统内部的数据传输优化，从而可能降低延迟并减少其他开销。

以下示例展示了如何将 ``y`` 显式关联为主机可访问，从而使 CPU 能够随时访问该内存。
（注意：kernel 调用后没有调用 ``cudaDeviceSynchronize()`` 。）
此时，若 GPU 上正在运行的核函数的同时访问 ``y`` ，将产生未定义的结果。

.. code-block:: c++

    __device__ __managed__ int x, y=2;
    __global__ void kernel() {
        x = 10;
    }
    int main() {
        cudaStream_t stream1;
        cudaStreamCreate(&stream1);
        cudaStreamAttachMemAsync(stream1, &y, 0, cudaMemAttachHost);
        cudaDeviceSynchronize();          // Wait for Host attachment to occur.
        kernel<<< 1, 1, 0, stream1 >>>(); // Note: Launches into stream1.
        y = 20;                           // Success – a kernel is running but "y"
                                          // has been associated with no stream.
        return  0;
    }

.. _um-stream-multithread-example:

4.1.3.3.3. 一个关于多线程主机程序的更复杂示例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``cudaStreamAttachMemAsync()`` 的主要用途是利用 CPU 多线程实现独立任务的并行。
通常在这类程序中，每个 CPU 线程都会为其生成的所有任务创建专属的流，因为如果使用 CUDA ``NULL stream`` ，将会导致线程间产生依赖关系。
在多线程程序中，由于托管内存默认对任何 GPU 流都具有全局可见性，因此很难避免 CPU 线程之间的相互干扰。
为此，可使用 ``cudaStreamAttachMemAsync()`` 函数将某个线程的托管内存分配与该线程自身的流相关联，且这种关联关系通常在线程的整个生命周期内保持不变。
对于此类程序而言，只需添加一次 ``cudaStreamAttachMemAsync()`` 调用，即可在其数据访问中使用统一内存：


.. code-block:: c++

    // This function performs some task in its own private stream 
    // and can be run in parallel
    void run_task(int *in, int *out, int length) {
        // Create a stream for us to use.
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        // Allocate some managed data and associate with our stream.
        // Note the use of the host-attach flag to cudaMallocManaged();
        // we then associate the allocation with our stream so that
        // our GPU kernel launches can access it.
        int *data;
        cudaMallocManaged((void **)&data, length, cudaMemAttachHost);
        cudaStreamAttachMemAsync(stream, data);
        cudaStreamSynchronize(stream);
        // Iterate on the data in some way, using both Host & Device.
        for(int i=0; i<N; i++) {
            transform<<< 100, 256, 0, stream >>>(in, data, length);
            cudaStreamSynchronize(stream);
            host_process(data, length);    // CPU uses managed data.
            convert<<< 100, 256, 0, stream >>>(out, data, length);
        }
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        cudaFree(data);
    }

在本示例中， ``data`` 内存分配与流的关联仅建立一次，随后主机和设备便可反复使用该内存。
尽管最终结果与在主机和设备之间显式复制数据的方式完全相同，但代码却简洁得多。

函数 ``cudaMallocManaged()`` 通过指定 ``cudaMemAttachHost`` 标志，创建的内存分配在初始状态下对设备端执行不可见。（而默认的内存分配对所有流上的所有 GPU 内核均可见。）
这确保了从数据分配到该数据被特定流获取之间的这段时间内，不会意外地与其他线程的执行发生交互。

若未指定此标志，当其他线程启动的 kernel 恰好在 GPU 上运行时，新分配的内存将被视为正在被 GPU 使用。
这可能会导致该线程在将新分配的内存显式关联到其私有流之前，无法从 CPU 端安全地访问这些数据。
因此，为确保各线程之间能够安全地独立运行，在分配内存时应指定此标志。

另一种实现思路是：在内存绑定至流完成后，设置一个作用于全部线程的进程级同步屏障。
该屏障能保证所有线程都完成各自内存与流的绑定操作后，再提交任何内核任务，以此规避风险。
销毁流之前还需要增设第二道同步屏障，原因是流销毁后，对应内存的可见性会恢复为默认全局托管模式。
``cudaMemAttachHost`` 标识的设计初衷，一是简化上述复杂的多屏障同步流程，二是很多业务场景下并不方便插入全局同步屏障。

.. _um-stream-data-movement:

4.1.3.3.4. 流关联统一内存的数据迁移
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

对于与流关联的统一内存，当设备的 ``concurrentManagedAccess`` 属性为 **0** ， ``Memcpy()/Memset()`` 的行为会有所不同，具体遵循以下规则：

1. 如果指定了 ``cudaMemcpyHostTo*`` 且源数据为统一内存，那么当该数据在拷贝流中可从主机端一致性访问时，将从主机端进行访问 [1]_；否则将从设备端进行访问。
类似地，当指定了 ``cudaMemcpy*ToHost`` 且目标为统一内存时，对目标数据的访问也遵循相同的规则。

1. 如果指定了 ``cudaMemcpyDeviceTo*`` 且源数据为统一内存，则将从设备端进行访问。
此时，源数据必须在拷贝流中可从设备端一致性访问 [2]_；否则将返回错误。
类似地，当指定了 ``cudaMemcpy*ToDevice`` 且目标为统一内存时，对目标数据的访问也遵循相同的规则。

1. 若拷贝方式指定为 ``cudaMemcpyDefault`` ，对于统一内存， 满足以下任一条件时，将从主机侧读取该内存：

- 该内存无法在拷贝流中从设备端进行一致性访问 [2]_ ；
- 该内存的优选存放位置为 ``cudaCpuDeviceId`` ，且能在拷贝流中从主机端一致性访问 [1]_ 。

其余情况下，均从设备侧读取该内存。

4. 对统一内存使用 ``cudaMemset*()`` 时，数据必须能够在用于执行 ``cudaMemset()`` 操作的流中从设备端被一致性访问 [2]_ ；否则将返回错误。

5. 当通过 ``cudaMemcpy*`` 或 ``cudaMemset*`` 在设备端访问数据时，执行该操作的流会被视为 GPU 上的活跃流。
   若当前设备属性 ``concurrentManagedAccess`` 值为 **0** （即不支持并发托管内存访问），在此期间 CPU 访问绑定该流的内存或全局可见的托管内存，都会触发段错误。
   程序必须做好同步处理，确保上述操作执行完毕后，CPU 再访问任何相关内存。

.. [1] 在指定流中可从主机一致性访问，指该内存既不具备全局可见性，也未与该指定流进行绑定关联。
.. [2] 在指定流中 **可从设备端一致性访问** 是指：该内存要么具有全局可见性，要么与该指定流相关联。


.. _um-perf-hints:

4.1.4. 性能提示
----------------

性能提示机制允许开发者向 CUDA 提供有关统一内存使用的更多信息。CUDA 借助这些性能提示更高效地管理内存，提升应用运行性能。
性能提示不会影响程序逻辑，仅对运行性能产生作用。

.. note::

   应用程序只应在统一内存性能提示能够带来性能提升时才使用。

性能提示可作用于任意统一内存分配，包括 CUDA 托管内存。在具备完整 CUDA 统一内存支持的系统上，性能提示可应用于所有系统分配的内存。

.. _um-data-prefetch:

4.1.4.1. 数据预取
~~~~~~~~~~~~~~~~~

``cudaMemPrefetchAsync`` 是一个异步的按流排序的 API，它可以将数据迁移至更靠近指定处理器的位置。
在数据被预取的过程中，这些数据仍可被访问。迁移操作不会在流中所有先前的操作完成之前开始，并且会在流中任何后续操作之前完成。

.. code-block:: c++

   cudaError_t cudaMemPrefetchAsync(const void *devPtr,
                                    size_t count,
                                    struct cudaMemLocation location,
                                    unsigned int flags,
                                    cudaStream_t stream=0);

当在指定流中执行预取任务时，
若 ``location.type`` 为 ``cudaMemLocationTypeDevice`` ，则内存区间 ``[devPtr, devPtr + count)`` 可迁移至目标设备 ``location.id`` ；
若 ``location.type`` 为 ``cudaMemLocationTypeHost`` ，则迁移至 CPU 端。
关于标识位的详细说明，请参阅当前 `CUDA 运行时 API 文档 <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html>`_。

考虑下面的简单代码示例：

.. tab-set::

   .. tab-item:: 系统分配器

      .. code-block:: c++

         void test_prefetch_sam(const cudaStream_t& s) {
           // initialize data on CPU
           char *data = (char*)malloc(dataSizeBytes);
           init_data(data, dataSizeBytes);
           cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

           // encourage data to move to GPU before use
           const unsigned int flags = 0;
           cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

           // use data on GPU
           const unsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
           mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes);

           // encourage data to move back to CPU
           location = {.type = cudaMemLocationTypeHost};
           cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

           cudaStreamSynchronize(s);

           // use data on CPU
           use_data(data, dataSizeBytes);
           free(data);
         }

   .. tab-item:: 托管

      .. code-block:: c++

         void test_prefetch_managed(const cudaStream_t& s) {
           // initialize data on CPU
           char *data;
           cudaMallocManaged(&data, dataSizeBytes);
           init_data(data, dataSizeBytes);
           cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

           // encourage data to move to GPU before use
           const unsigned int flags = 0;
           cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

           // use data on GPU
           const unsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
           mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes);

           // encourage data to move back to CPU
           location = {.type = cudaMemLocationTypeHost};
           cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

           cudaStreamSynchronize(s);

           // use data on CPU
           use_data(data, dataSizeBytes);
           cudaFree(data);
         }

.. _um-data-usage-hints:

4.1.4.2. 数据使用提示
~~~~~~~~~~~~~~~~~~~~~

当多个处理器并发访问相同的数据时，可以使用 ``cudaMemAdvise`` 来提示 ``[devPtr, devPtr + count)`` 内存区间内的数据将将如何被访问：

.. code-block:: c++

   cudaError_t cudaMemAdvise(const void *devPtr,
                             size_t count,
                             enum cudaMemoryAdvise advice,
                             struct cudaMemLocation location);

示例展示了如何使用 ``cudaMemAdvise`` ：

.. code-block:: c++

   // test-prefetch-managed-begin
   void test_prefetch_managed(const cudaStream_t& s) {
     // initialize data on CPU
     char *data;
     cudaMallocManaged(&data, dataSizeBytes);
     init_data(data, dataSizeBytes);
     cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

     // encourage data to move to GPU before use
     const unsigned int flags = 0;
     cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

     // use data on GPU
     const unsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
     mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes);

     // encourage data to move back to CPU
     location = {.type = cudaMemLocationTypeHost};
     cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

     cudaStreamSynchronize(s);

     // use data on CPU
     use_data(data, dataSizeBytes);
     cudaFree(data);
   }
   // test-prefetch-managed-end

   static const int maxDevices = 1;
   static const int maxOuterLoopIter = 3;
   static const int maxInnerLoopIter = 4;

其中 ``advice`` 可以采取以下值：

- ``cudaMemAdviseSetReadMostly`` ：

  表示对应内存数据以读为主，偶尔写入。总体而言，它能让这片内存区间以牺牲部分写带宽为代价，换取更高的读带宽。

- ``cudaMemAdviseSetPreferredLocation`` ：

  该提示会将数据的优选驻留位置设置为指定设备的物理显存。
  该提示仅建议系统尽量将数据保存在此优选位置，但无法做出保证。
  若入参 ``location.type`` 传入 ``cudaMemLocationTypeHost`` ，则会把 CPU 内存设为优选位置。
  其他性能提示（例如 ``cudaMemPrefetchAsync`` ）可覆盖本提示的策略，允许内存页面从优选位置迁移出去。

- ``cudaMemAdviseSetAccessedBy`` ：
   
   在部分系统中，若要从指定处理器访问数据，提前建立内存映射可提升性能。
   当 ``location.type`` 取值为 ``cudaMemLocationTypeDevice`` 时，
   该提示告知系统：设备 ``location.id`` 会频繁访问这片数据，让系统预先创建对应映射（该操作的开销可被后续收益抵消）。
   此提示并不规定数据应当驻留在哪一端，但可以与（其他提示搭配使用）。

   可搭配 ``cudaMemAdviseSetPreferredLocation`` 来指定数据驻留位置。
   在具备硬件一致性的系统上，该提示会启用访问计数器迁移机制，详情参阅 :ref:`um-access-counters` 。

每种内存建议也可以通过使用以下值之一来取消设置： ``cudaMemAdviseUnsetReadMostly`` 、 ``cudaMemAdviseUnsetPreferredLocation`` 和 ``cudaMemAdviseUnsetAccessedBy`` 。

示例展示了如何使用 ``cudaMemAdvise`` ：

.. tab-set::

   .. tab-item:: 系统分配器

      .. code-block:: c++

         void test_advise_sam(cudaStream_t stream) {
           char *dataPtr;
           size_t dataSize = 64 * threadsPerBlock;  // 16 KiB

           // Allocate memory using malloc or cudaMallocManaged
           dataPtr = (char*)malloc(dataSize);

           // Set hint on memory region
           cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
           cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);

           int outerLoopIter = 0;
           while (outerLoopIter < maxOuterLoopIter) {
             // Data is written by CPU in each outer loop iteration
             init_data(dataPtr, dataSize);

             // Data is made available to all GPUs via prefetch.
             // The prefetch here causes data reads to be duplicated rather than
             // data migration
             cudaMemLocation location;
             location.type = cudaMemLocationTypeDevice;
             for (int device = 0; device < maxDevices; device++) {
               location.id = device;
               const unsigned int flags = 0;
               cudaMemPrefetchAsync(dataPtr, dataSize, location, flags, stream);
             }

             // Kernel only reads this data in the inner loop
             int innerLoopIter = 0;
             while (innerLoopIter < maxInnerLoopIter) {
               mykernel<<<32, threadsPerBlock, 0, stream>>>((const char *)dataPtr, dataSize);
               innerLoopIter++;
             }
             outerLoopIter++;
           }

           free(dataPtr);
         }

   .. tab-item:: 托管

      .. code-block:: c++

         void test_advise_managed(cudaStream_t stream) {
           char *dataPtr;
           size_t dataSize = 64 * threadsPerBlock;  // 16 KiB

           // Allocate memory using cudaMallocManaged
           // (malloc can be used on systems with full CUDA unified memory support)
           cudaMallocManaged(&dataPtr, dataSize);

           // Set hint on memory region
           cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
           cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);

           int outerLoopIter = 0;
           while (outerLoopIter < maxOuterLoopIter) {
             // Data is written by CPU in each outer loop iteration
             init_data(dataPtr, dataSize);

             // Data is made available to all GPUs via prefetch.
             // The prefetch here causes data reads to be duplicated rather than
             // data migration
             cudaMemLocation location;
             location.type = cudaMemLocationTypeDevice;
             for (int device = 0; device < maxDevices; device++) {
               location.id = device;
               const unsigned int flags = 0;
               cudaMemPrefetchAsync(dataPtr, dataSize, location, flags, stream);
             }

             // Kernel only reads this data in the inner loop
             int innerLoopIter = 0;
             while (innerLoopIter < maxInnerLoopIter) {
               mykernel<<<32, threadsPerBlock, 0, stream>>>((const char *)dataPtr, dataSize);
               innerLoopIter++;
             }
             outerLoopIter++;
           }

           cudaFree(dataPtr);
         }

.. _um-memory-discard:

4.1.4.3. 内存丢弃
~~~~~~~~~~~~~~~~~~

``cudaMemDiscardBatchAsync`` API 允许应用程序告知 CUDA ：指定内存区域的数据内容已不再需要。
统一内存驱动会在缺页迁移、内存回收（为了支持设备显存超分）时自动执行数据搬运。
这类自动数据搬运有时会产生冗余开销，大幅降低性能。
将一段地址区间标记为 `丢弃（discard）` ，相当于通知统一内存驱动：应用已经用完该区间的数据，后续执行预取或页面回收、腾挪空间给其他内存分配时，无需再迁移这份数据。
若读取已标记丢弃的页面，且此前没有写入或预取操作，读取结果是未定义的随机值；但丢弃操作完成后执行新的写入，后续读取一定能拿到正确写入内容。
对正在执行丢弃操作的地址区间并发访问、并发预取，会产生未定义行为。


.. code-block:: c++

   cudaError_t cudaMemDiscardBatchAsync(void **dptrs,
                                        size_t *sizes,
                                        size_t count,
                                        unsigned long long flags,
                                        cudaStream_t stream);

该函数对 ``dptrs`` 和 ``sizes`` 数组中指定的地址范围执行一批内存丢弃。
两个数组的长度必须与 ``count`` 指定的一样。
每个内存范围必须引用通过 ``cudaMallocManaged`` 分配或通过 ``__managed__`` 变量声明的托管内存。

``cudaMemDiscardAndPrefetchBatchAsync`` API 结合了丢弃和预取操作。
调用 ``cudaMemDiscardAndPrefetchBatchAsync`` 语义上等同于调用 ``cudaMemDiscardBatchAsync`` 后跟 ``cudaMemPrefetchBatchAsync`` ，但更优。
当应用程序需要内存位于目标位置但不需要内存内容时，这很有用。

.. code-block:: c++

   cudaError_t cudaMemDiscardAndPrefetchBatchAsync(void **dptrs,
                                                   size_t *sizes,
                                                   size_t count,
                                                   struct cudaMemLocation *prefetchLocs,
                                                   size_t *prefetchLocIdxs,
                                                   size_t numPrefetchLocs,
                                                   unsigned long long flags,
                                                   cudaStream_t stream);

``prefetchLocs`` 数组指定预取的目的地，而 ``prefetchLocIdxs`` 指示每个预取位置适用于哪些操作。
例如，如果一批有 10 个操作，前 6 个应该预取到一个位置，其余 4 个预取到另一个位置，
那么 ``numPrefetchLocs`` 将是 2， ``prefetchLocIdxs`` 将是 {0, 6}， ``prefetchLocs`` 将包含两个目标位置。

**重要考虑因素：**

- 读一个丢弃之后没有继续执行写入或预取操作的内存地址，将会返回一个不确定的值。
- 可以通过向该内存范围写入数据，或通过 ``cudaMemPrefetchAsync`` ，来重置丢弃标记。
- 任何与丢弃操作同时发生的读取、写入或预取操作，都会导致未定义行为。
- 所有设备必须具有非零值的 ``cudaDevAttrConcurrentManagedAccess`` 。

.. _um-query-data-usage:

4.1.4.4. 查询托管内存上的数据使用属性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

应用程序可通过下述 API，查询 CUDA 托管内存上由 ``cudaMemAdvise`` 或 ``cudaMemPrefetchAsync`` 设置的内存区间属性：

.. code-block:: c++

   cudaMemRangeGetAttribute(void *data,
                            size_t dataSize,
                            enum cudaMemRangeAttribute attribute,
                            const void *devPtr,
                            size_t count);

此函数用于查询一段内存区间的属性：区间起始地址为 ``devPtr`` ，总长度 ``count`` 字节。
该内存区间必须是通过 ``cudaMallocManaged`` 分配的托管内存，或是以 ``__managed__`` 限定符声明的托管变量。
支持的 ``attribute`` 类型如下：

- ``cudaMemRangeAttributeReadMostly`` ：如果整个内存范围设置了 ``cudaMemAdviseSetReadMostly`` 属性则返回 1，否则返回 0。

- ``cudaMemRangeAttributePreferredLocation`` ：如果整个内存范围都将对应的处理器设置为首选位置，查询将返回 GPU 设备 ID 或 ``cudaCpuDeviceId``；
  否则将返回 ``cudaInvalidDeviceId``。应用程序可以利用此查询 API，根据托管指针的首选位置属性，来决定是通过 CPU 还是 GPU 来中转数据。
  请注意，在查询时该内存范围的实际物理位置可能与首选位置不同。

- ``cudaMemRangeAttributeAccessedBy`` ：将返回为该内存范围设置了此提示的设备列表。

- ``cudaMemRangeAttributeLastPrefetchLocation`` ：返回应用程序上次通过 ``cudaMemPrefetchAsync`` 显式预取这片内存区间的目标位置。
  请注意：该值仅记录应用程序上一次发起预取时指定的目标位置，无法反映该次预取操作是否已开始、或是是否执行完成。

- ``cudaMemRangeAttributePreferredLocationType`` ：该属性返回优选驻留位置类型，取值如下：

  - ``cudaMemLocationTypeDevice`` ：内存范围中的所有页面都使用相同的 GPU 作为其首选位置，
  - ``cudaMemLocationTypeHost`` ：内存范围中的所有页面都使用 CPU 作为其首选位置，
  - ``cudaMemLocationTypeHostNuma`` ：内存范围中的所有页面都使用相同的主机 NUMA 节点 ID 作为其首选位置，
  - ``cudaMemLocationTypeInvalid`` ：不是所有页面使用相同的首选位置或某些页面根本没有首选位置。

- ``cudaMemRangeAttributePreferredLocationId`` ：如果查询优选驻留位置类型返回 ``cudaMemLocationTypeDevice`` ，则本属性返回对应设备编号；
  若优选位置类型为 ``cudaMemLocationTypeHostNuma`` ，则返回主机 NUMA 节点 ID。其余场景下，返回值无意义，应当忽略。

- ``cudaMemRangeAttributeLastPrefetchLocationType`` ：返回上一次通过 ``cudaMemPrefetchAsync`` 显式预取该内存区间全部页面时，指定的目标位置类型，可选返回值如下：

  - ``cudaMemLocationTypeDevice`` ：内存范围中的所有页面都预取到相同的 GPU，
  - ``cudaMemLocationTypeHost`` ：内存范围中的所有页面都预取到 CPU，
  - ``cudaMemLocationTypeHostNuma`` ：内存范围中的所有页面都预取到相同的主机 NUMA 节点 ID，
  - ``cudaMemLocationTypeInvalid`` ：不是所有页面都预取到相同的位置或某些页面从未预取过。

- ``cudaMemRangeAttributeLastPrefetchLocationId`` ：如果查询预取位置类型返回 ``cudaMemLocationTypeDevice`` ，本属性返回对应设备编号；
  如果查询预取位置类型返回 ``cudaMemLocationTypeHostNuma`` ，本属性返回主机 NUMA 节点 ID。其余场景下，返回值无意义，应当忽略。

此外，可以使用 ``cudaMemRangeGetAttributes`` 函数单次查询多个属性。

.. _um-gpu-memory-oversubscription:

4.1.4.5. GPU 内存超额订阅
~~~~~~~~~~~~~~~~~~~~~~~~~~

统一内存使应用程序能够 *超额分配* 大于任何单个处理器的内存：换句话说，
它们可以分配和共享大于系统中任何单个处理器存储容量的数组，从而能够对单个 GPU 显存容纳不下的数据集进行处理，而不会显著增加编程模型的复杂度。
