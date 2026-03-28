.. _compute-capabilities:

计算能力
========

计算设备的一般规格和功能取决于其计算能力（参见 :ref:`cuda-platform-compute-capability-sm-version` ）。

:ref:`compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability`、:ref:`compute-capabilities-table-device-and-streaming-multiprocessor-sm-information-per-compute-capability` 和 :ref:`compute-capabilities-table-memory-information-per-compute-capability` 展示了当前支持的每个计算能力相关联的功能和技术规格。

所有 NVIDIA GPU 架构都使用小端序（little-endian）表示。

.. _compute-capabilities-obtain-the-gpu-compute-capability:

获取 GPU 计算能力
------------------

`CUDA GPU 计算能力 <https://developer.nvidia.com/cuda-gpus>`__ 页面提供了 NVIDIA GPU 型号与其计算能力之间的完整映射。

或者，可以使用随 `NVIDIA 驱动程序 <https://www.nvidia.com/en-us/drivers/>`__ 提供的 `nvidia-smi <https://docs.nvidia.com/deploy/nvidia-smi/index.html>`__ 工具获取 GPU 的计算能力。例如，以下命令将输出系统上可用的 GPU 名称和计算能力：

.. code-block:: bash

   nvidia-smi --query-gpu=name,compute_cap

在运行时，可以使用 CUDA Runtime API `cudaDeviceGetAttribute() <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151>`__、CUDA Driver API `cuDeviceGetAttribute() <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266>`__ 或 NVML API `nvmlDeviceGetCudaComputeCapability() <https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g1f803a2fb4b7dfc0a8183b46b46ab03a>`__ 获取计算能力：

.. code-block:: c++

   #include <cuda_runtime_api.h>

   int computeCapabilityMajor, computeCapabilityMinor;
   cudaDeviceGetAttribute(&computeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, device_id);
   cudaDeviceGetAttribute(&computeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, device_id);

.. code-block:: c++

   #include <cuda.h>

   int computeCapabilityMajor, computeCapabilityMinor;
   cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_id);
   cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_id);

.. code-block:: c++

   #include <nvml.h> // 需要链接 -lnvidia-ml

   int computeCapabilityMajor, computeCapabilityMinor;
   nvmlDeviceGetCudaComputeCapability(nvmlDevice, &computeCapabilityMajor, &computeCapabilityMinor);

.. _compute-capabilities-feature-availability:

功能可用性
----------

大多数随计算架构引入的计算功能都旨在后续所有架构上可用。这在 :ref:`compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability` 中通过功能在其引入后的计算能力上标记「Yes」来表示。

.. _compute-capabilities-architecture-specific-features:

架构特定功能
~~~~~~~~~~~~

从计算能力 9.0 的设备开始，随架构引入的专用计算功能可能无法保证在所有后续计算能力上都可用。这些功能称为*架构特定*（architecture-specific）功能，旨在加速专用操作，如 Tensor Core 操作，这些操作并非适用于所有类型的计算能力，或者可能在未来的代次中发生重大变化。代码必须使用架构特定的编译器目标编译（参见 :ref:`compute-capabilities-feature-set-compiler-targets` ）才能启用架构特定功能。使用架构特定编译器目标编译的代码只能在其编译的目标计算能力上运行。

.. _compute-capabilities-family-specific-features:

系列特定功能
~~~~~~~~~~~~

从计算能力 10.0 的设备开始，某些架构特定功能对于多个计算能力的设备是通用的。包含这些功能的设备属于同一个系列（family），这些功能也可以称为*系列特定*（family-specific）功能。系列特定功能保证在同一系列的所有设备上可用。需要系列特定的编译器目标才能启用系列特定功能。参见 :ref:`compute-capabilities-feature-set-compiler-targets`。为系列特定目标编译的代码只能在该系列成员的 GPU 上运行。

.. _compute-capabilities-feature-set-compiler-targets:

功能集编译器目标
~~~~~~~~~~~~~~~~

编译器可以针对三组计算功能：

**基准功能集（Baseline Feature Set）**：主要的计算功能集，引入时旨在用于后续计算架构。这些功能及其可用性总结在 :ref:`compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability` 中。

**架构特定功能集（Architecture-Specific Feature Set）**：一小部分高度专业化的功能集，称为架构特定功能，引入用于加速专用操作，不保证在后续计算架构上可用或可能发生重大变化。这些功能总结在各自的「计算能力 #.#」小节中。架构特定功能集是系列特定功能集的超集。架构特定编译器目标随计算能力 9.0 设备引入，通过在编译目标中使用 **a** 后缀选择，例如指定 `compute_100a` 或 `compute_120a` 作为计算目标。

**系列特定功能集（Family-Specific Feature Set）**：某些架构特定功能对于多个计算能力的 GPU 是通用的。这些功能总结在各自的「计算能力 #.#」小节中。除少数例外，具有相同主计算能力的后一代设备属于同一系列。:ref:`compute-capabilities-family-specific-compatibility` 显示了系列特定目标与设备计算能力的兼容性，包括例外情况。系列特定功能集是基准功能集的超集。系列特定编译器目标随计算能力 10.0 设备引入，通过在编译目标中使用 **f** 后缀选择，例如指定 `compute_100f` 或 `compute_120f` 作为计算目标。

从计算能力 9.0 开始的所有设备都有一组架构特定功能。要在特定 GPU 上利用这些完整功能集，必须使用带 **a** 后缀的架构特定编译器目标。此外，从计算能力 10.0 开始，有些功能集出现在具有不同次计算能力的多个设备中。这些指令集称为系列特定功能，共享这些功能的设备被称为属于同一系列的一部分。系列特定功能是该 GPU 系列所有成员共享的架构特定功能的子集。带 **f** 后缀的系列特定编译器目标允许编译器生成使用架构特定功能公共子集的代码。

例如：

- ``compute_100`` 编译目标不允许使用架构特定功能。此目标将与计算能力 10.0 及更高版本的所有设备兼容。

- ``compute_100f`` *系列特定* 编译目标允许使用 GPU 系列中通用的架构特定功能子集。此目标仅与属于该 GPU 系列的设备兼容。在此示例中，它与计算能力 10.0 和计算能力 10.3 的设备兼容。系列特定 ``compute_100f`` 目标中可用的功能是基准 ``compute_100`` 目标中可用功能的超集。

- ``compute_100a`` *架构特定* 编译目标允许使用计算能力 10.0 设备中的完整架构特定功能集。此目标仅与计算能力 10.0 的设备兼容，不与其他设备兼容。 ``compute_100a`` 目标中可用的功能构成了 ``compute_100f`` 目标中可用功能的超集。

.. _compute-capabilities-family-specific-compatibility:

.. list-table:: 系列特定兼容性
   :header-rows: 1
   :widths: 40 60

   * - 编译目标
     - 兼容的计算能力
   * - ``compute_100f``
     - 10.0、10.3
   * - ``compute_103f``
     - 10.3 [#family2]_
   * - ``compute_110f``
     - 11.0 [#family2]_
   * - ``compute_120f``
     - 12.0、12.1
   * - ``compute_121f``
     - 12.1 [#family2]_

.. rubric:: 脚注

.. [#family2] 某些系列在创建时只包含一个成员。未来可能会扩展以包含更多设备。

.. _compute-capabilities-features-and-technical-specifications:

功能和技术规格
--------------

.. _compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability:

.. list-table:: 每个计算能力的功能支持
   :header-rows: 1
   :widths: 55 9 9 9 9 9 9

   * - **功能支持**
     - **计算能力**
     - 
     - 
     - 
     - 
     - 
   * - （未列出的功能支持所有计算能力）
     - 7.x
     - 8.x
     - 9.0
     - 10.x
     - 11.0
     - 12.x
   * - 对共享内存和全局内存中 128 位整数值的原子操作（ :ref:`atomic-functions` ）
     - No
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - 对全局内存中 ``float2`` 和 ``float4`` 浮点向量的原子加法（ :ref:`atomicadd` ）
     - No
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - Warp 归约函数（ :ref:`warp-reduce-functions` ）
     - No
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - Bfloat16 精度浮点运算
     - No
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - 128 位精度浮点运算
     - No
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - 硬件加速的 ``memcpy_async`` （ :ref:`pipelines` ）
     - No
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - 硬件加速的 Split Arrive/Wait 屏障（ :ref:`asynchronous-barriers` ）
     - No
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - L2 缓存驻留管理（ :ref:`advanced-kernels-l2-control` ）
     - No
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - 用于加速动态规划的 DPX 指令（ :ref:`dpx-instructions` ）
     - Multiple Instr.
     - Native
     - Multiple Instr.
     - Native
     - Native
     - Native
   * - 分布式共享内存
     - No
     - No
     - Yes
     - Yes
     - Yes
     - Yes
   * - 线程块集群（ :ref:`thread-block-clusters` ）
     - No
     - No
     - Yes
     - Yes
     - Yes
     - Yes
   * - Tensor Memory Accelerator (TMA) 单元（ :ref:`async-copies-tma` ）
     - No
     - No
     - Yes
     - Yes
     - Yes
     - Yes

注意，以下表格中使用的 KB 和 K 单位对应于 1024 字节（即 KiB）和 1024。

.. _compute-capabilities-table-device-and-streaming-multiprocessor-sm-information-per-compute-capability:

.. list-table:: 每个计算能力的设备和流多处理器 (SM) 信息
   :header-rows: 1
   :widths: 45 8 8 8 8 8 8 8 8 8 8

   * - **计算能力**
     - 7.5
     - 8.0
     - 8.6
     - 8.7
     - 8.9
     - 9.0
     - 10.0
     - 10.3
     - 11.0
     - 12.x
   * - FP32 与 FP64 吞吐量比率 [#fn-cc-throughput]_
     - 32:1
     - 2:1
     - 64:1
     - 2:1
     - 64:1
     - 32:1
     - 32:1
     - 32:1
     - 32:1
     - 32:1
   * - 每个设备的最大驻留 grid 数（并发内核执行）
     - 128
     - 128
     - 128
     - 128
     - 128
     - 128
     - 128
     - 128
     - 128
     - 128
   * - grid 的最大维度
     - 3
     - 3
     - 3
     - 3
     - 3
     - 3
     - 3
     - 3
     - 3
     - 3
   * - grid 的最大 x 维度
     - 2\ :sup:`31`\ -1
     - 2\ :sup:`31`\ -1
     - 2\ :sup:`31`\ -1
     - 2\ :sup:`31`\ -1
     - 2\ :sup:`31`\ -1
     - 2\ :sup:`31`\ -1
     - 2\ :sup:`31`\ -1
     - 2\ :sup:`31`\ -1
     - 2\ :sup:`31`\ -1
     - 2\ :sup:`31`\ -1
   * - grid 的最大 y 或 z 维度
     - 65535
     - 65535
     - 65535
     - 65535
     - 65535
     - 65535
     - 65535
     - 65535
     - 65535
     - 65535
   * - 线程块的最大维度
     - 3
     - 3
     - 3
     - 3
     - 3
     - 3
     - 3
     - 3
     - 3
     - 3
   * - 线程块的最大 x 或 y 维度
     - 1024
     - 1024
     - 1024
     - 1024
     - 1024
     - 1024
     - 1024
     - 1024
     - 1024
     - 1024
   * - 线程块的最大 z 维度
     - 64
     - 64
     - 64
     - 64
     - 64
     - 64
     - 64
     - 64
     - 64
     - 64
   * - 每个块的最大线程数
     - 1024
     - 1024
     - 1024
     - 1024
     - 1024
     - 1024
     - 1024
     - 1024
     - 1024
     - 1024
   * - Warp 大小
     - 32
     - 32
     - 32
     - 32
     - 32
     - 32
     - 32
     - 32
     - 32
     - 32
   * - 每个 SM 的最大驻留块数
     - 16
     - 32
     - 16
     - 24
     - 16
     - 32
     - 24
     - 24
     - 24
     - 24
   * - 每个 SM 的最大驻留 warp 数
     - 32
     - 64
     - 48
     - 64
     - 48
     - 64
     - 48
     - 48
     - 48
     - 48
   * - 每个 SM 的最大驻留线程数
     - 1024
     - 2048
     - 1536
     - 2048
     - 1536
     - 2048
     - 1536
     - 1536
     - 1536
     - 1536
   * - Green contexts：useFlags 为 0 时的最小 SM 分区大小
     - –
     - –
     - –
     - –
     - –
     - 2
     - 4
     - 4
     - 4
     - 8
   * - Green contexts：useFlags 为 0 时每个分区的 SM 共调度对齐
     - –
     - –
     - –
     - –
     - –
     - 2
     - 8
     - 8
     - 8
     - 8

.. rubric:: 脚注

.. [#fn-cc-throughput] 非 Tensor Core 吞吐量。有关吞吐量的更多信息，请参阅 `CUDA 最佳实践指南 <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions-throughput-native-arithmetic-instructions>`__。

.. _compute-capabilities-table-memory-information-per-compute-capability:

.. list-table:: 每个计算能力的内存信息
   :header-rows: 1
   :widths: 45 8 8 8 8 8 8 8 8 8 8

   * - **计算能力**
     - 7.5
     - 8.0
     - 8.6
     - 8.7
     - 8.9
     - 9.0
     - 10.x
     - 11.0
     - 12.x
     - 
   * - 每个 SM 的 32 位寄存器数量
     - 64 K
     - 64 K
     - 64 K
     - 64 K
     - 64 K
     - 64 K
     - 64 K
     - 64 K
     - 64 K
     - 
   * - 每个线程块的最大 32 位寄存器数量
     - 64 K
     - 64 K
     - 64 K
     - 64 K
     - 64 K
     - 64 K
     - 64 K
     - 64 K
     - 64 K
     - 
   * - 每个线程的最大 32 位寄存器数量
     - 255
     - 255
     - 255
     - 255
     - 255
     - 255
     - 255
     - 255
     - 255
     - 
   * - 每个 SM 的最大共享内存量
     - 64 KB
     - 164 KB
     - 100 KB
     - 164 KB
     - 100 KB
     - 228 KB
     - 100 KB
     - 100 KB
     - 100 KB
     - 
   * - 每个线程块的最大共享内存量 [#fn33]_
     - 64 KB
     - 163 KB
     - 99 KB
     - 163 KB
     - 99 KB
     - 227 KB
     - 99 KB
     - 99 KB
     - 99 KB
     - 
   * - 共享内存存储体数量
     - 32
     - 32
     - 32
     - 32
     - 32
     - 32
     - 32
     - 32
     - 32
     - 
   * - 每个线程的最大本地内存量
     - 512 KB
     - 512 KB
     - 512 KB
     - 512 KB
     - 512 KB
     - 512 KB
     - 512 KB
     - 512 KB
     - 512 KB
     - 
   * - 常量内存大小
     - 64 KB
     - 64 KB
     - 64 KB
     - 64 KB
     - 64 KB
     - 64 KB
     - 64 KB
     - 64 KB
     - 64 KB
     - 
   * - 每个 SM 的常量内存缓存工作集
     - 8 KB
     - 8 KB
     - 8 KB
     - 8 KB
     - 8 KB
     - 8 KB
     - 8 KB
     - 8 KB
     - 8 KB
     - 
   * - 每个 SM 的纹理内存缓存工作集
     - 32 或 64 KB
     - 28 KB ~ 192 KB
     - 28 KB ~ 128 KB
     - 28 KB ~ 192 KB
     - 28 KB ~ 128 KB
     - 28 KB ~ 256 KB
     - 28 KB ~ 128 KB
     - 28 KB ~ 128 KB
     - 28 KB ~ 128 KB
     - 

.. rubric:: 脚注

.. [#fn33] 依赖每个块超过 48 KB 共享内存分配的内核必须使用动态共享内存并需要显式选择，参见 :ref:`advanced-kernel-l1-shared-config`。

.. _compute-capabilities-table-shared-memory-capacity-per-compute-capability:

.. list-table:: 每个计算能力的共享内存容量
   :header-rows: 1
   :widths: 25 25 50

   * - 计算能力
     - 统一数据缓存大小 (KB)
     - SMEM 容量大小 (KB)
   * - 7.5
     - 96
     - 32, 64
   * - 8.0
     - 192
     - 0, 8, 16, 32, 64, 100, 132, 164
   * - 8.6
     - 128
     - 0, 8, 16, 32, 64, 100
   * - 8.7
     - 192
     - 0, 8, 16, 32, 64, 100, 132, 164
   * - 8.9
     - 128
     - 0, 8, 16, 32, 64, 100
   * - 9.0
     - 256
     - 0, 8, 16, 32, 64, 100, 132, 164, 196, 228
   * - 10.x
     - 256
     - 0, 8, 16, 32, 64, 100, 132, 164, 196, 228
   * - 11.0
     - 256
     - 0, 8, 16, 32, 64, 100, 132, 164, 196, 228
   * - 12.x
     - 128
     - 0, 8, 16, 32, 64, 100

:ref:`compute-capabilities-table-tensor-core-data-types-per-compute-capability` 显示了 Tensor Core 加速支持的输入数据类型。Tensor Core 功能集可通过内联 PTX 在 CUDA 编译工具链中使用。强烈建议应用程序通过 CUDA-X 库（如 cuDNN、cuBLAS 和 cuFFT）或通过 `CUTLASS <https://docs.nvidia.com/cutlass/index.html>`__ （一组 CUDA C++ 模板抽象和 Python 领域特定语言 （DSL），旨在支持 CUDA 各级别的高性能矩阵-矩阵乘法 （GEMM） 和相关计算）使用此功能集。

.. _compute-capabilities-table-tensor-core-data-types-per-compute-capability:

.. list-table:: 每个计算能力的 Tensor Core 加速支持的输入数据类型
   :header-rows: 1
   :widths: 12 8 8 8 8 8 8 8 8 8

   * - 计算能力
     - FP64
     - TF32
     - BF16
     - FP16
     - FP8
     - FP6
     - FP4
     - INT8
     - INT4
   * - 7.5
     - 
     - Yes
     - 
     - Yes
     - 
     - 
     - 
     - Yes
     - 
   * - 8.0
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - 
     - 
     - Yes
     - 
   * - 8.6
     - 
     - Yes
     - Yes
     - Yes
     - Yes
     - 
     - 
     - Yes
     - 
   * - 8.7
     - 
     - Yes
     - Yes
     - Yes
     - Yes
     - 
     - 
     - Yes
     - 
   * - 8.9
     - 
     - Yes
     - Yes
     - Yes
     - Yes
     - 
     - 
     - Yes
     - 
   * - 9.0
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - 
     - 
     - Yes
     - 
   * - 10.0
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - 10.3
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - 
   * - 11.0
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - 
   * - 12.x
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     -