.. _environment-variables-details:

5.2. CUDA 环境变量
==================

以下部分列出了 CUDA 环境变量。与多进程服务 (MPS) 相关的环境变量记录在 `GPU 部署和管理指南 <https://docs.nvidia.com/deploy/mps/index.html#environment-variables>`_ 中。

.. _device-enumeration-and-properties:

5.2.1. 设备枚举和属性
---------------------

.. _cuda-visible-devices:

5.2.1.1. ``CUDA_VISIBLE_DEVICES``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量控制哪些 GPU 设备对 CUDA 应用程序可见以及它们的枚举顺序。

- 如果未设置该变量，则所有 GPU 设备都可见。

- 如果将该变量设置为空字符串，则没有 GPU 设备可见。

**可选值**：以逗号分隔的 GPU 标识符序列。

GPU 标识符的提供方式如下：

- **整数索引**：这些对应于系统中 GPU 的序号，由 ``nvidia-smi`` 确定，从 0 开始。例如，设置 ``CUDA_VISIBLE_DEVICES=2,1`` 会使设备 0 不可见，并使设备 2 在设备 1 之前枚举。

  - 如果遇到无效索引，则只有索引在无效索引之前出现的设备才可见。例如，设置 ``CUDA_VISIBLE_DEVICES=0,2,-1,1`` 会使设备 0 和 2 可见，而设备 1 不可见，因为它出现在无效索引 ``-1`` 之后。

- **GPU UUID 字符串**：这些应遵循 ``nvidia-smi -L`` 给出的相同格式，例如 ``GPU-8932f937-d72c-4106-c12f-20bd9faed9f6`` 。但是，为了方便起见，允许使用缩写形式；只需指定 GPU UUID 开头的足够数字即可在目标系统中唯一标识该 GPU。例如，假设系统中没有其他 GPU 共享此前缀，则 ``CUDA_VISIBLE_DEVICES=GPU-8932f937`` 可能是引用上述 GPU UUID 的有效方式。

- `多实例 GPU (MIG) <https://docs.nvidia.com/datacenter/tesla/mig-user-guide/>`_ 支持： ``MIG-<GPU-UUID>/<GPU 实例 ID>/<计算实例 ID>`` 。例如， ``MIG-GPU-8932f937-d72c-4106-c12f-20bd9faed9f6/1/2`` 。仅支持单个 MIG 实例枚举。

``cudaGetDeviceCount()`` API 返回的设备计数仅包括可见设备，因此使用整数设备标识符的 CUDA API 仅支持 [0, 可见设备计数 - 1] 范围内的序号。GPU 设备的枚举顺序决定了序号值。例如，使用 ``CUDA_VISIBLE_DEVICES=2,1`` 时，调用 ``cudaSetDevice(0)`` 会将设备 2 设置为当前设备，因为它首先被枚举并分配了序号 0。之后调用 ``cudaGetDevice(&device_ordinal)`` 也会将 ``device_ordinal`` 设置为 0，这对应于设备 2。

**示例**：

.. code-block:: bash

   nvidia-smi -L  # 获取 GPU UUID 列表
   CUDA_VISIBLE_DEVICES=0,1
   CUDA_VISIBLE_DEVICES=GPU-8932f937-d72c-4106-c12f-20bd9faed9f6
   CUDA_VISIBLE_DEVICES=MIG-GPU-8932f937-d72c-4106-c12f-20bd9faed9f6/1/2

----

.. _cuda-device-order:

5.2.1.2. ``CUDA_DEVICE_ORDER``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量控制 CUDA 枚举可用设备的顺序。

**可选值**：

- ``FASTEST_FIRST`` ：使用简单的启发式方法从最快到最慢枚举可用设备（默认）。

- ``PCI_BUS_ID`` ：按 PCI 总线 ID 升序枚举可用设备。可以使用 ``nvidia-smi --query-gpu=name,pci.bus_id`` 获取 PCI 总线 ID。

**示例**：

.. code-block:: bash

   CUDA_DEVICE_ORDER=FASTEST_FIRST
   CUDA_DEVICE_ORDER=PCI_BUS_ID
   nvidia-smi --query-gpu=name,pci.bus_id  # 获取 PCI 总线 ID 列表

----

.. _cuda-managed-force-device-alloc:

5.2.1.3. ``CUDA_MANAGED_FORCE_DEVICE_ALLOC``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量改变 :ref:`memory-unified-memory` 在多 GPU 系统中的物理存储方式。

**可选值**：数值，为零或非零。

- **非零值**：强制驱动程序使用设备内存进行物理存储。进程中使用的所有支持托管内存的设备必须具有点对点兼容性。否则，将返回 ``cudaErrorInvalidDevice`` 。

- ``0`` ：默认行为。

**示例**：

.. code-block:: bash

   CUDA_MANAGED_FORCE_DEVICE_ALLOC=0
   CUDA_MANAGED_FORCE_DEVICE_ALLOC=1  # 强制使用设备内存

----

.. _jit-compilation:

5.2.2. JIT 编译
---------------

.. _cuda-cache-disable:

5.2.2.1. ``CUDA_CACHE_DISABLE``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量控制磁盘上 :ref:`just-in-time-compilation` 缓存的行为。禁用 JIT 缓存会强制 CUDA 应用程序每次执行时都进行 PTX 到 CUBIN 的编译，除非在二进制文件中找到运行架构的 CUBIN 代码。

禁用 JIT 缓存会增加应用程序在初始执行期间的加载时间。但是，它对于减少应用程序的磁盘空间以及诊断不同驱动程序版本或构建标志之间的差异很有用。

**可选值**：

- ``1`` ：禁用 PTX JIT 缓存。

- ``0`` ：启用 PTX JIT 缓存（默认）。

**示例**：

.. code-block:: bash

   CUDA_CACHE_DISABLE=1  # 禁用缓存
   CUDA_CACHE_DISABLE=0  # 启用缓存

----

.. _cuda-cache-path:

5.2.2.2. ``CUDA_CACHE_PATH``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量指定 :ref:`just-in-time-compilation` 缓存的目录路径。

**可选值**：缓存目录的绝对路径（具有适当的访问权限）。默认值为：

- 在 Windows 上， ``%APPDATA%\NVIDIA\ComputeCache``

- 在 Linux 上， ``~/.nv/ComputeCache``

**示例**：

.. code-block:: bash

   CUDA_CACHE_PATH=~/tmp

----

.. _cuda-cache-maxsize:

5.2.2.3. ``CUDA_CACHE_MAXSIZE``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量以字节为单位指定 :ref:`just-in-time-compilation` 缓存的大小。超过此大小的二进制文件不会被缓存。如果需要，会从缓存中驱逐较旧的二进制文件以为较新的文件腾出空间。

**可选值**：字节数。默认值为：

- 在桌面/服务器平台上， ``1073741824`` （1 GiB）

- 在嵌入式平台上， ``268435456`` （256 MiB）

最大大小为 ``4294967296`` （4 GiB）。

**示例**：

.. code-block:: bash

   CUDA_CACHE_MAXSIZE=268435456  # 256 MiB

----

.. _cuda-force-ptx-jit-and-cuda-force-jit:

5.2.2.4. ``CUDA_FORCE_PTX_JIT`` 和 ``CUDA_FORCE_JIT``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

这些环境变量指示 CUDA 驱动程序忽略应用程序中嵌入的任何 CUBIN，并对嵌入的 PTX 代码执行 :ref:`just-in-time-compilation`。

强制 JIT 编译会增加应用程序在初始执行期间的加载时间。但是，它可用于验证 PTX 代码是否嵌入在应用程序中以及其即时编译是否正常工作。这确保了与未来架构的 `前向兼容性 <https://docs.nvidia.com/deploy/cuda-compatibility/>`_。

``CUDA_FORCE_PTX_JIT`` 优先于 ``CUDA_FORCE_JIT`` 。

**可选值**：

- ``1`` ：强制 PTX JIT 编译。

- ``0`` ：默认行为。

**示例**：

.. code-block:: bash

   CUDA_FORCE_PTX_JIT=1

----

.. _cuda-disable-ptx-jit-and-cuda-disable-jit:

5.2.2.5. ``CUDA_DISABLE_PTX_JIT`` 和 ``CUDA_DISABLE_JIT``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

这些环境变量禁用嵌入 PTX 代码的 :ref:`just-in-time-compilation`，并使用应用程序中嵌入的兼容 CUBIN。

如果内核没有嵌入的二进制代码，或者嵌入的二进制是为不兼容的架构编译的，则内核将无法加载。这些环境变量可用于验证应用程序是否为每个内核生成了兼容的 CUBIN 代码。有关更多详细信息，请参阅 :ref:`binary-compatibility` 部分。

``CUDA_DISABLE_PTX_JIT`` 优先于 ``CUDA_DISABLE_JIT`` 。

**可选值**：

- ``1`` ：禁用 PTX JIT 编译。

- ``0`` ：默认行为。

**示例**：

.. code-block:: bash

   CUDA_DISABLE_PTX_JIT=1

----

.. _cuda-force-preload-libraries:

5.2.2.6. ``CUDA_FORCE_PRELOAD_LIBRARIES``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量影响 :ref:`just-in-time-compilation` 和 `NVVM <https://docs.nvidia.com/cuda/nvvm-ir-spec/>`_ 所需库的预加载。

**可选值**：

- ``1`` ：强制驱动程序在初始化期间预加载 :ref:`just-in-time-compilation` 和 `NVVM <https://docs.nvidia.com/cuda/nvvm-ir-spec/>`_ 所需的库。这会增加 CUDA 驱动程序初始化所需的内存占用和时间。设置此环境变量对于避免涉及多线程的某些死锁情况是必要的。

- ``0`` ：默认行为。

**示例**：

.. code-block:: bash

   CUDA_FORCE_PRELOAD_LIBRARIES=1

----

.. _execution:

5.2.3. 执行
-----------

.. _cuda-launch-blocking:

5.2.3.1. ``CUDA_LAUNCH_BLOCKING``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量指定是禁用还是启用异步内核启动。

禁用异步执行会导致执行速度变慢，但对调试很有用。它强制 GPU 工作从 CPU 的角度同步运行。这允许在触发 CUDA API 错误的确切 API 调用处观察它们，而不是在执行的后续阶段。同步执行对于调试目的很有用。

**可选值**：

- ``1`` ：禁用异步执行。

- ``0`` ：异步执行（默认）。

**示例**：

.. code-block:: bash

   CUDA_LAUNCH_BLOCKING=1

----

.. _cuda-device-max-connections:

5.2.3.2. ``CUDA_DEVICE_MAX_CONNECTIONS``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量控制并发计算和复制引擎连接（工作队列）的数量，将两者都设置为指定值。如果独立的 GPU 任务（即从不同 CUDA 流启动的内核或复制操作）映射到同一工作队列，则会创建虚假依赖关系，这可能导致 GPU 工作串行化，因为使用了相同的底层资源。为了减少此类虚假依赖的可能性，建议通过此环境变量控制的工作队列计数大于或等于每个上下文的活跃 CUDA 流数量。

设置此环境变量也会修改复制连接的数量，除非通过 ``CUDA_DEVICE_MAX_COPY_CONNECTIONS`` 环境变量显式设置。

**可选值**： ``1`` 到 ``32`` 个连接，默认为 ``8`` （假设没有 MPS）

**示例**：

.. code-block:: bash

   CUDA_DEVICE_MAX_CONNECTIONS=16

----

.. _cuda-device-max-copy-connections:

5.2.3.3. ``CUDA_DEVICE_MAX_COPY_CONNECTIONS``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量控制复制操作中涉及的并发复制连接（工作队列）的数量。它仅影响 :ref:`compute-capabilities` 8.0 及以上的设备。

如果同时设置了 ``CUDA_DEVICE_MAX_COPY_CONNECTIONS`` 和 ``CUDA_DEVICE_MAX_CONNECTIONS`` ，则 ``CUDA_DEVICE_MAX_COPY_CONNECTIONS`` 会覆盖通过 ``CUDA_DEVICE_MAX_CONNECTIONS`` 设置的复制连接值。

**可选值**： ``1`` 到 ``32`` 个连接，默认为 ``8`` （假设没有 MPS）

**示例**：

.. code-block:: bash

   CUDA_DEVICE_MAX_COPY_CONNECTIONS=16

----

.. _cuda-scale-launch-queues:

5.2.3.4. ``CUDA_SCALE_LAUNCH_QUEUES``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量指定用于启动工作（命令缓冲区）的队列大小的缩放因子，即可以在设备上排队的挂起内核或主机/设备复制操作的总数。

**可选值**： ``0.25x``、``0.5x``、``2x``、``4x``

- 任何非 ``0.25x``、``0.5x``、``2x`` 或 ``4x`` 的值都被解释为 ``1x`` 。

**示例**：

.. code-block:: bash

   CUDA_SCALE_LAUNCH_QUEUES=2x

----

.. _cuda-graphs-use-node-priority:

5.2.3.5. ``CUDA_GRAPHS_USE_NODE_PRIORITY``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量控制 CUDA 图相对于其启动的流的优先级的执行优先级。

``CUDA_GRAPHS_USE_NODE_PRIORITY`` 会覆盖图实例化时的 `cudaGraphInstantiateFlagUseNodePriority <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1gd4d586536547040944c05249ee26bc62>`_ 标志。

**可选值**：

- ``0`` ：继承图启动到的流的优先级（默认）。

- ``1`` ：遵循每节点启动优先级。CUDA 运行时将节点级优先级视为就绪运行的图节点的调度提示。

**示例**：

.. code-block:: bash

   CUDA_GRAPHS_USE_NODE_PRIORITY=1

----

.. _cuda-device-waits-on-exception:

5.2.3.6. ``CUDA_DEVICE_WAITS_ON_EXCEPTION``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量控制 CUDA 应用程序在发生异常（错误）时的行为。

启用后，当设备端异常发生时，CUDA 应用程序将暂停并等待，允许调试器（如 `cuda-gdb <https://docs.nvidia.com/cuda/cuda-gdb/index.html>`_）附加以在进程退出或继续之前检查实时 GPU 状态。

**可选值**：

- ``0`` ：默认行为。

- ``1`` ：发生设备异常时暂停。

**示例**：

.. code-block:: bash

   CUDA_DEVICE_WAITS_ON_EXCEPTION=1

----

.. _cuda-device-default-persisting-l2-cache-percentage-limit:

5.2.3.7. ``CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量控制 GPU 的 L2 缓存中为 :ref:`l2-set-aside` 保留的默认「预留」部分，以 L2 大小的百分比表示。

它与支持持久 L2 缓存的 GPU 相关，特别是使用 `CUDA 多进程服务 (MPS) <https://docs.nvidia.com/deploy/mps/index.html>`_ 时 :ref:`compute-capabilities` 8.0 或更高的设备。必须在启动 CUDA MPS 控制守护进程之前设置此环境变量，即在运行 ``nvidia-cuda-mps-control -d`` 命令之前。

**可选值**：0 到 100 之间的百分比值，默认为 0。

**示例**：

.. code-block:: bash

   CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT=25  # 25%

----

.. _cuda-disable-perf-boost:

5.2.3.8. ``CUDA_DISABLE_PERF_BOOST``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 Linux 主机上，将此环境变量设置为 1 可防止提升设备性能状态，而是可以根据各种启发式方法隐式选择 pstate。此选项可能有助于降低功耗，但由于动态性能状态选择，在某些情况下可能会导致更高的延迟。

**示例**：

.. code-block:: bash

   CUDA_DISABLE_PERF_BOOST=1  # 禁用性能提升，仅限 Linux
   CUDA_DISABLE_PERF_BOOST=0  # 默认行为

.. _cuda-auto-boost-deprecated:

5.2.3.9. ``CUDA_AUTO_BOOST`` [[已弃用]]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量影响 GPU 时钟「自动提升」行为，即动态时钟提升。它会覆盖 ``nvidia-smi`` 工具的「自动提升」选项，即 ``nvidia-smi --auto-boost-default=0`` 。

.. note::

   此环境变量已弃用。强烈建议使用 ``nvidia-smi --applications-clocks=<memory,graphics>`` 或 `NVML API <https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceCommands.html#group__nvmlDeviceCommands>`_ 代替 ``CUDA_AUTO_BOOST`` 环境变量。

----

.. _module-loading:

5.2.4. 模块加载
---------------

.. _cuda-module-loading:

5.2.4.1. ``CUDA_MODULE_LOADING``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量影响 CUDA 运行时加载模块的方式，特别是它如何初始化设备代码。

**可选值**：

- ``DEFAULT`` ：默认行为，等同于 ``LAZY`` 。

- ``LAZY`` ：特定内核的加载会延迟到使用 ``cuModuleGetFunction()`` 或 ``cuKernelGetFunction()`` API 调用提取 CUDA 函数句柄 ``CUfunc`` 时。在这种情况下，当加载 CUBIN 中的第一个内核或访问 CUBIN 中的第一个变量时，会加载 CUBIN 中的数据。

  - 驱动程序在首次调用内核时加载所需的代码；后续调用不会产生额外开销。这减少了启动时间和 GPU 内存占用。

- ``EAGER`` ：在程序初始化时完全加载 CUDA 模块和内核。CUBIN、FATBIN 或 PTX 文件中的所有内核和数据会在相应的 ``cuModuleLoad*`` 和 ``cuLibraryLoad*`` 驱动程序 API 调用时完全加载。

  - 启动时间和 GPU 内存占用较高。内核启动开销是可预测的。

**示例**：

.. code-block:: bash

   CUDA_MODULE_LOADING=EAGER
   CUDA_MODULE_LOADING=LAZY

----

.. _cuda-module-data-loading:

5.2.4.2. ``CUDA_MODULE_DATA_LOADING``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量影响 CUDA 运行时加载与模块关联的数据的方式。

这是 ``CUDA_MODULE_LOADING`` 中内核焦点设置的补充设置。此环境变量不影响内核的 ``LAZY`` 或 ``EAGER`` 加载。如果未设置此环境变量，则数据加载行为从 ``CUDA_MODULE_LOADING`` 继承。

**可选值**：

- ``DEFAULT`` ：默认行为，等同于 ``LAZY`` 。

- ``LAZY`` ：模块数据的加载会延迟到需要 CUDA 函数句柄 ``CUfunc`` 时。在这种情况下，当加载 CUBIN 中的第一个内核或访问 CUBIN 中的第一个变量时，会加载 CUBIN 中的数据。

  - 延迟数据加载可能需要上下文同步，这可能会减慢并发执行。

- ``EAGER`` ：CUBIN、FATBIN 或 PTX 文件中的所有数据会在相应的 ``cuModuleLoad*`` 和 ``cuLibraryLoad*`` API 调用时完全加载。

**示例**：

.. code-block:: bash

   CUDA_MODULE_DATA_LOADING=EAGER

.. _cuda-binary-loader-thread-count:

5.2.4.3. ``CUDA_BINARY_LOADER_THREAD_COUNT``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

设置加载设备二进制文件时使用的 CPU 线程数。设置为 0 时，使用的 CPU 线程数设置为默认值 1。

**可选值**：

- 要使用的线程整数。默认为 0，即使用 1 个线程。

**示例**：

.. code-block:: bash

   CUDA_BINARY_LOADER_THREAD_COUNT=4

----

.. _cuda-error-log-management:

5.2.5. CUDA 错误日志管理
-------------------------

.. _cuda-log-file:

5.2.5.1. ``CUDA_LOG_FILE``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

该环境变量指定一个位置，当支持的 CUDA API 调用返回错误时，将在该位置打印描述性错误日志消息。

例如，如果尝试使用无效的网格配置启动内核，例如 ``kernel<<<1, dim3(1,1,128)>>>(...)`` ，则该内核将无法启动， ``cudaGetLastError()`` 将返回通用的 ``invalid configuration argument`` 错误。如果设置了 ``CUDA_LOG_FILE`` 环境变量，用户可以在日志中看到以下描述性错误消息： ``[CUDA][E] Block Dimensions (1,1,128) include one or more values that exceed the device limit of (1024,1024,64)`` ，并轻松确定指定的块的 z 维度无效。有关更多详细信息，请参阅 :doc:`08-error-log-management`。

**可选值**： ``stdout``、``stderr`` 或有效文件路径（具有适当的访问权限）

**示例**：

.. code-block:: bash

   CUDA_LOG_FILE=stdout
   CUDA_LOG_FILE=/tmp/dbg_cuda_log