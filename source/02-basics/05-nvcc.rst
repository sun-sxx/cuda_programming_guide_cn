.. _nvcc:

2.5. NVCC：NVIDIA CUDA 编译器
=============================

`NVIDIA CUDA 编译器 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>`_ ``nvcc`` 是 NVIDIA 提供的工具链，
用于编译 CUDA C/C++ 以及 `PTX <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>`_ 代码。
该工具链是 `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ 的一部分，包含多个工具，
包括编译器、链接器以及 PTX 和 :ref:`Cubin <cubins-and-fatbins>` 汇编器。
顶层工具 ``nvcc`` 协调编译过程，为每个编译阶段调用适当的工具。

``nvcc`` 驱动 CUDA 代码的离线编译，这与由 CUDA 运行时编译器 `nvrtc <https://docs.nvidia.com/cuda/nvrtc/index.html>`_ 驱动的在线或即时（JIT）编译形成对比。

本章介绍构建应用程序所需的 ``nvcc`` 最常见用法和详细信息。
关于 ``nvcc`` 的完整介绍请参阅 `nvcc 文档主页 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>`_。

.. _nvcc-source-files:

2.5.1. CUDA 源文件和头文件
--------------------------

使用 ``nvcc`` 编译的源文件可以包含主机代码（在 CPU 上执行）和设备代码（在 GPU 上执行）的组合。
``nvcc`` 接受常见的主机专用 C/C++ 源文件扩展名 ``.c``、``.cpp``、``.cc``、``.cxx`` ，以及 ``.cu`` 用于包含设备代码或主机和设备代码混合的文件。
包含设备代码的头文件通常采用 ``.cuh`` 扩展名，以区别于主机专用代码头文件 ``.h``、``.hpp``、``.hh``、``.hxx`` 等。

.. list-table::
   :header-rows: 1

   * - 文件扩展名
     - 描述
     - 内容
   * - ``.c``
     - C 源文件
     - 仅主机代码
   * - ``.cpp``、``.cc``、``.cxx``
     - C++ 源文件
     - 仅主机代码
   * - ``.h``、``.hpp``、``.hh``、``.hxx``
     - C/C++ 头文件
     - 设备代码、主机代码、主机/设备代码混合
   * - ``.cu``
     - CUDA 源文件
     - 设备代码、主机代码、主机/设备代码混合
   * - ``.cuh``
     - CUDA 头文件
     - 设备代码、主机代码、主机/设备代码混合

.. _nvcc-compilation-workflow:

2.5.2. NVCC 编译工作流程
-------------------------

在初始阶段， ``nvcc`` 将设备代码与主机代码分离，并分别将其编译分派给 GPU 编译器和主机编译器。

要编译主机代码，CUDA 编译器 ``nvcc`` 需要有可用的兼容主机编译器。
CUDA Toolkit 定义了 `Linux 平台 <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy>`_ 
和 `Windows 平台 <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements>`_ 的主机编译器支持策略。

仅包含主机代码的文件可以使用 ``nvcc`` 或直接使用主机编译器构建。生成的目标文件可以在链接时与包含 GPU 代码的 ``nvcc`` 目标文件组合。

GPU 编译器将 C/C++ 设备代码编译为 PTX 汇编代码。GPU 编译器会针对编译命令行中指定的每个虚拟机指令集架构（例如 ``compute_90`` ）运行。

然后将各个 PTX 代码传递给 ``ptxas`` 工具，该工具为目标硬件 ISA 生成 :ref:`Cubin <cubins-and-fatbins>` 。
硬件 ISA 由其 :ref:`SM 版本 <compute-capability-and-streaming-multiprocessor-versions>` 标识。

可以将多个 PTX 和 Cubin 目标嵌入到应用程序或库中的单个二进制 :ref:`Fatbin <cubins-and-fatbins>` 容器中，
以便单个二进制文件可以支持多个虚拟和目标硬件 ISA。

上述工具的调用和协调由 ``nvcc`` 自动完成。
可以使用 ``-v`` 选项显示完整的编译工作流程和工具调用。
可以使用 ``-keep`` 选项将编译期间生成的 `中间文件 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#keeping-intermediate-phase-files>`_ 保存在当前目录或由 ``--keep-dir`` 指定的目录中。

以下示例说明了 CUDA 源文件 ``example.cu`` 的编译工作流程：

.. code-block:: cuda
   :caption: example.cu

   // ----- example.cu -----
   #include <stdio.h>
   __global__ void kernel() {
       printf("Hello from kernel\n");
   }

   void kernel_launcher() {
       kernel<<<1, 1>>>();
       cudaDeviceSynchronize();
   }

   int main() {
       kernel_launcher();
       return 0;
   }

``nvcc`` 基本编译工作流程：

.. figure:: /_static/images/nvcc-flow.png
   :alt: nvcc 基本工作流程

   nvcc 基本编译工作流程

``nvcc`` 具有多个 PTX 和 Cubin 架构的编译工作流程：

.. figure:: /_static/images/nvcc-flow-multi-archs.png
   :alt: nvcc 多架构工作流程

   nvcc 具有多个 PTX 和 Cubin 架构的编译工作流程

关于 ``nvcc`` 编译工作流程的更详细描述可以在 `编译器文档 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#the-cuda-compilation-trajectory>`_ 中找到。

.. _nvcc-basic-usage:

2.5.3. NVCC 基本用法
---------------------

使用 ``nvcc`` 编译 CUDA 源文件的基本命令是：

.. code-block:: text

   nvcc <source_file>.cu -o <output_file>

``nvcc`` 接受常用的编译器标志，用于指定包含目录 ``-I <path>`` 和库路径 ``-L <path>``、链接其他库 ``-l<library>`` 以及定义宏 ``-D<macro>=<value>`` 。

.. code-block:: text

   nvcc example.cu -I path_to_include/ -L path_to_library/ -lcublas -o <output_file>

.. _nvcc-ptx-cubin-generation:

2.5.3.1. NVCC PTX 和 Cubin 生成
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

默认情况下， ``nvcc`` 为 CUDA Toolkit 支持的最早 GPU 架构（最低的 ``compute_XY`` 和 ``sm_XY`` 版本）生成 PTX 和 Cubin，以最大化兼容性。

- ``-arch`` `-arch 选项 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-architecture-arch>`__ 可用于为特定 GPU 架构生成 PTX 和 Cubin。

- ``-gencode`` `-gencode 选项 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#generate-code-specification-gencode>`__ 可用于为多个 GPU 架构生成 PTX 和 Cubin。

可以通过传递 ``--list-gpu-code`` 和 ``--list-gpu-arch`` 标志分别获取支持的虚拟和实际 GPU 架构的完整列表，
或参考 ``nvcc`` 文档中的 `虚拟架构列表 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list>`_ 
和 `GPU 架构列表 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list>`_ 部分。

.. code-block:: text

   nvcc --list-gpu-code  # 列出所有支持的实际 GPU 架构
   nvcc --list-gpu-arch  # 列出所有支持的虚拟 GPU 架构

   nvcc example.cu -arch=compute_<XY>  # 例如 -arch=compute_80 用于 NVIDIA Ampere GPU 及更高版本
                                       # 仅 PTX，GPU 前向兼容

   nvcc example.cu -arch=sm_<XY>       # 例如 -arch=sm_80 用于 NVIDIA Ampere GPU 及更高版本
                                       # PTX 和 Cubin，GPU 前向兼容

   nvcc example.cu -arch=native        # 自动检测并为当前 GPU 生成 Cubin
                                       # 无 PTX，无 GPU 前向兼容性

   nvcc example.cu -arch=all           # 为所有支持的 GPU 架构生成 Cubin
                                       # 还包括用于 GPU 前向兼容性的最新 PTX

   nvcc example.cu -arch=all-major     # 为所有主要支持的 GPU 架构生成 Cubin，例如 sm_80、sm_90
                                       # 还包括用于 GPU 前向兼容性的最新 PTX

更高级的用法允许单独指定 PTX 和 Cubin 目标：

.. code-block:: text

   # 为虚拟架构 compute_80 生成 PTX，并将其编译为实际架构 sm_86 的 Cubin，保留 compute_80 PTX
   nvcc example.cu -arch=compute_80 -gpu-code=sm_86,compute_80  # （PTX 和 Cubin）

   # 为虚拟架构 compute_80 生成 PTX，并将其编译为实际架构 sm_86、sm_89 的 Cubin
   nvcc example.cu -arch=compute_80 -gpu-code=sm_86,sm_89       # （无 PTX）
   nvcc example.cu -gencode=arch=compute_80,code=sm_86,sm_89    # 同上

   # (1) 为虚拟架构 compute_80 生成 PTX，并将其编译为实际架构 sm_86、sm_89 的 Cubin
   # (2) 为虚拟架构 compute_90 生成 PTX，并将其编译为实际架构 sm_90 的 Cubin
   nvcc example.cu -gencode=arch=compute_80,code=sm_86,sm_89 -gencode=arch=compute_90,code=sm_90

关于控制 GPU 代码生成的 ``nvcc`` 命令行选项的完整参考可以在 `GPU 代码生成选项文档 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation>`_ 中找到。

.. _host-code-compilation-notes:

2.5.3.2. 主机代码编译注意事项
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

不包含设备代码或符号的编译单元（即源文件及其头文件）可以直接使用主机编译器编译。
如果任何编译单元使用 CUDA 运行时 API 函数，则应用程序必须与 CUDA 运行时库链接。
CUDA 运行时同时提供静态库和共享库，分别为 ``libcudart_static`` 和 ``libcudart`` 。默认情况下， ``nvcc`` 链接到静态 CUDA 运行时库。
要使用 CUDA 运行时的共享库版本，请在编译或链接命令中向 ``nvcc`` 传递 ``--cudart=shared`` 标志。

``nvcc`` 允许通过 ``-ccbin <compiler>`` 参数指定用于主机函数的主机编译器。
还可以定义环境变量 ``NVCC_CCBIN`` 来指定 ``nvcc`` 使用的主机编译器。 ``nvcc`` 的 ``-Xcompiler`` 参数将参数传递给主机编译器。
例如，在下面的示例中， ``-O3`` 参数由 ``nvcc`` 传递给主机编译器。

.. code-block:: bash

   nvcc example.cu -ccbin=clang++

   export NVCC_CCBIN='gcc'
   nvcc example.cu -Xcompiler=-O3

.. _nvcc-separate-compilation:

2.5.3.3. GPU 代码的分离编译
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``nvcc`` 默认采用 **全程序编译** （whole-program compilation），期望所有 GPU 代码和符号都存在于使用它们的编译单元中。
CUDA 设备函数可以调用在其他编译单元中定义的设备函数或访问设备变量，但必须在 ``nvcc`` 命令行中指定 ``-rdc=true`` 或其别名 ``-dc`` 标志，以启用来自不同编译单元的设备代码链接。
从不同编译单元链接设备代码和符号的能力称为 **分离编译** （separate compilation）。

分离编译允许更灵活的代码组织，可以改善编译时间，并可以产生更小的二进制文件。
与全程序编译相比，分离编译可能会涉及一些构建时的复杂性。使用设备代码链接可能会影响性能，这就是默认情况下不使用它的原因。
:ref:`nvcc-link-time-optimization` 可以帮助减少分离编译的性能开销。

分离编译需要满足以下条件：

- 在一个编译单元中定义的非 ``const`` 设备变量在其他编译单元中必须使用 ``extern`` 关键字引用。

- 所有 ``const`` 设备变量必须定义并使用 ``extern`` 关键字引用。

- 所有 CUDA 源文件 ``.cu`` 必须使用 ``-dc`` 或 ``-rdc=true`` 标志编译。

主机和设备函数默认具有外部链接，不需要 ``extern`` 关键字。
注意，`从 CUDA 13 开始 <https://developer.nvidia.com/blog/cuda-c-compiler-updates-impacting-elf-visibility-and-linkage/>`_，
``__global__`` 函数和 ``__managed__``/``__device__``/``__constant__`` 变量默认具有内部链接。

在以下示例中， ``definition.cu`` 定义了一个变量和一个函数，而 ``example.cu`` 引用它们。
这两个文件分别编译并链接到最终二进制文件中。

.. code-block:: cuda
   :caption: definition.cu

   // ----- definition.cu -----
   extern __device__ int device_variable = 5;
   __device__        int device_function() { return 10; }

.. code-block:: cuda
   :caption: example.cu

   // ----- example.cu -----
   extern __device__ int  device_variable;
   __device__        int device_function();

   __global__ void kernel(int* ptr) {
       device_variable = 0;
       *ptr            = device_function();
   }

.. code-block:: bash

   nvcc -dc definition.cu -o definition.o
   nvcc -dc example.cu    -o example.o
   nvcc definition.o example.o -o program

.. _common-compiler-options:

2.5.4. 常用编译器选项
---------------------

本节介绍可与 ``nvcc`` 一起使用的最相关的编译器选项，涵盖语言特性、优化、调试、分析和构建方面。
所有选项的完整描述可以在 `nvcc 命令选项文档 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#command-option-description>`_ 中找到。

.. _language-features:

2.5.4.1. 语言特性
^^^^^^^^^^^^^^^^^

``nvcc`` 支持 C++ 核心语言特性，从 C++03 到 `C++20 <https://en.cppreference.com/w/cpp/compiler_support#cpp20>`_。可以使用 ``-std`` 标志指定要使用的语言标准：

- ``--std={c++03|c++11|c++14|c++17|c++20}``

此外， ``nvcc`` 支持以下语言扩展：

- ``-restrict`` ：断言所有内核指针参数都是 :ref:`restrict <restrict-pointers>` 指针。

- ``-extended-lambda`` ：允许在 lambda 声明中使用 ``__host__``、``__device__`` 注解。

- ``-expt-relaxed-constexpr`` ：（实验性标志）允许主机代码调用 ``__device__ constexpr`` 函数，设备代码调用 ``__host__ constexpr`` 函数。

关于这些特性的更多详细信息可以在 :ref:`扩展 lambda <extended-lambdas>` 和 :ref:`constexpr 函数 <constexpr-functions-restrictions>` 部分找到。

.. _debugging-options:

2.5.4.2. 调试选项
^^^^^^^^^^^^^^^^^

``nvcc`` 支持以下选项来生成调试信息：

- ``-g`` ：为主机代码生成调试信息。 ``gdb/lldb`` 和类似工具依赖此类信息进行主机代码调试。

- ``-G`` ：为设备代码生成调试信息。`cuda-gdb <https://docs.nvidia.com/cuda/cuda-gdb/index.html>`_ 依赖此类信息进行设备代码调试。该标志还定义了 ``__CUDACC_DEBUG__`` 宏。

- ``-lineinfo`` ：为设备代码生成行号信息。此选项不影响执行性能，与 `compute-sanitizer <https://developer.nvidia.com/compute-sanitizer>`_ 工具结合使用来跟踪内核执行时非常有用。

``nvcc`` 默认对 GPU 代码使用最高优化级别 ``-O3`` 。调试标志 ``-G`` 会阻止某些编译器优化，因此调试代码的性能预期低于非调试代码。
可以定义 ``-DNDEBUG`` 标志来禁用运行时断言，因为断言也会减慢执行速度。

.. _optimization-options:

2.5.4.3. 优化选项
^^^^^^^^^^^^^^^^^

``nvcc`` 提供了许多用于优化性能的选项。本节旨在简要介绍开发人员可能发现有用的一些选项，并提供进一步信息的链接。
完整介绍可以在 `nvcc 优化文档 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>`_ 中找到。

- ``-Xptxas`` 将参数传递给 PTX 汇编工具 ``ptxas`` 。
- ``nvcc`` 文档提供了 `ptxas 有用参数列表 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ptxas-options>`_。
- 例如， ``-Xptxas=-maxrregcount=N`` 指定每个线程使用的最大寄存器数量。

- ``-extra-device-vectorization`` ：启用更激进的设备代码向量化。

- 提供对浮点行为细粒度控制的其他标志在 :ref:`浮点计算 <mathematical-functions>` 部分
  和 `快速数学选项文档 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math>`_ 中介绍。

以下标志从编译器获取输出，在更高级的代码优化中可能有用：

- ``-res-usage`` ：编译后打印资源使用报告。它包括为每个内核函数分配的寄存器数量、共享内存、常量内存和本地内存。

- ``-opt-info=inline`` ：打印有关内联函数的信息。

- ``-Xptxas=-warn-lmem-usage`` ：如果使用了本地内存则发出警告。

- ``-Xptxas=-warn-spills`` ：如果寄存器溢出到本地内存则发出警告。

.. _nvcc-link-time-optimization:

2.5.4.4. 链接时优化（LTO）
^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`nvcc-separate-compilation` 可能比全程序编译性能更低，因为跨文件优化机会有限。
链接时优化（LTO）通过在链接时对单独编译的文件执行优化来解决此问题，代价是编译时间增加。LTO 可以恢复全程序编译的大部分性能，同时保持分离编译的灵活性。

``nvcc`` 需要 ``-dlto`` `-dlto 标志 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#dlink-time-opt-dlto>`__ 或 ``lto_<SM version>`` 链接时优化目标来启用 LTO：

.. code-block:: bash

   nvcc -dc -dlto -arch=sm_100 definition.cu -o definition.o
   nvcc -dc -dlto -arch=sm_100 example.cu    -o example.o
   nvcc -dlto definition.o example.o -o program

   nvcc -dc -arch=lto_100 definition.cu -o definition.o
   nvcc -dc -arch=lto_100 example.cu    -o example.o
   nvcc -dlto definition.o example.o -o program

.. _profiling-options:

2.5.4.5. 分析选项
^^^^^^^^^^^^^^^^^

可以使用 `Nsight Compute <https://developer.nvidia.com/nsight-compute>`_ 和 `Nsight Systems <https://developer.nvidia.com/nsight-systems>`_ 工具直接分析 CUDA 应用程序，
而无需在编译过程中使用额外的标志。
但是， ``nvcc`` 可以生成额外的信息，通过将源文件与生成的代码相关联来辅助分析：

- ``-lineinfo`` ：为设备代码生成行号信息；这允许在分析工具中查看源代码。分析工具要求原始源代码在编译代码的相同位置可用。

- ``-src-in-ptx`` ：在 PTX 中保留原始源代码，避免上述 ``-lineinfo`` 的限制。需要 ``-lineinfo`` 。

.. _fatbin-compression:

2.5.4.6. Fatbin 压缩
^^^^^^^^^^^^^^^^^^^^

``nvcc`` 默认压缩存储在应用程序或库二进制文件中的 :ref:`fatbins <cubins-and-fatbins>`。
可以使用以下选项控制 Fatbin 压缩：

- ``-no-compress`` ：禁用 fatbin 的压缩。

- ``--compress-mode={default|size|speed|balance|none}`` ：设置压缩模式。
  ``speed`` 专注于快速解压时间，而 ``size`` 旨在减小 fatbin 大小。 ``balance`` 在速度和大小之间提供权衡。默认模式是 ``speed`` 。
  ``none`` 禁用压缩。

.. _compiler-performance-controls:

2.5.4.7. 编译器性能控制
^^^^^^^^^^^^^^^^^^^^^^^

``nvcc`` 提供用于分析和加速编译过程本身的选项：

- ``-t <N>`` ：单个编译单元需要针对多个 GPU 架构并行编译时，使用 CPU 线程数。

- ``-split-compile <N>`` ：优化阶段并行化使用的 CPU 线程数。

- ``-split-compile-extended <N>`` ：更激进的分割编译形式。需要链路时优化。

- ``-Ofc <N>`` ：设备代码编译速度级别。

- ``-time <filename>`` ：生成包含每个编译阶段所耗时间的逗号分隔值（CSV）表。

- ``-fdevice-time-trace`` ：为设备代码编译生成时间跟踪。