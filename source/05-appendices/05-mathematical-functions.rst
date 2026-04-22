.. _mathematical-functions:

5.5. 浮点计算
=============

.. _floating-point-introduction:

5.5.1. 浮点介绍
---------------

自 1985 年采用 `IEEE-754 标准 <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229>`__ 进行二进制浮点运算以来，几乎所有主流计算系统（包括 NVIDIA 的 CUDA 架构）都已实现了该标准。IEEE-754 标准规定了浮点运算结果应如何近似。

要在所需精度下获得准确结果并实现最高性能，重要的是考虑浮点行为的许多方面。这在异构计算环境中尤为重要，因为操作在不同类型的硬件上执行。

以下章节回顾浮点计算的基本属性，并介绍融合乘加（FMA）运算和点积。这些示例说明不同的实现选择如何影响精度。

.. _floating-point-format:

5.5.1.1. 浮点格式
^^^^^^^^^^^^^^^^^

浮点格式和功能在 `IEEE-754 标准 <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229>`__ 中定义。

该标准规定二进制浮点数据在三个字段上编码：

- **符号（Sign）**：一位，指示正数或负数。
- **指数（Exponent）**：编码基数为 2 的指数，偏移一个数值偏差。
- **有效数字（Significand）**（也称为*尾数*或*小数*）：编码数值的小数部分。

最新的 IEEE-754 标准定义了以下二进制格式的编码和属性：

- 16 位，也称为半精度，对应 CUDA 中的 ``__half`` 数据类型。
- 32 位，也称为单精度，对应 C、C++ 和 CUDA 中的 ``float`` 数据类型。
- 64 位，也称为双精度，对应 C、C++ 和 CUDA 中的 ``double`` 数据类型。
- 128 位，也称为四精度，对应 CUDA 中的 ``__float128`` 或 ``_Float128`` 数据类型。

正规值的浮点编码关联的数值计算如下：

.. math::

   (-1)^{\mathrm{sign}} \times 1.\mathrm{mantissa} \times 2^{\mathrm{exponent} - \mathrm{bias}}

对于次正规值，公式修改为：

.. math::

   (-1)^{\mathrm{sign}} \times 0.\mathrm{mantissa} \times 2^{1-\mathrm{bias}}

指数分别对单精度和双精度偏移 127 和 1023。``1.`` 的整数部分在小数中是隐含的。

例如，值 :math:`-192 = (-1)^1 \times 2^7 \times 1.5` 编码为负号、指数 7 和小数部分 0.5。因此指数 7 用 ``float`` 的值 ``7 + 127 = 134 = 10000110`` 和 ``double`` 的值 ``7 + 1023 = 1030 = 10000000110`` 的位字符串表示。尾数 ``0.5 = 2^{-1}`` 用第一个位置为 ``1`` 的二进制值表示。-192 在单精度和双精度中的二进制编码如下图所示：

由于小数字段使用有限数量的位，并非所有实数都可以精确表示。例如，小数 :math:`2 / 3` 的数学值的二进制表示是 ``0.10101010...``，在二进制小数点后有无限多位。因此，:math:`2 / 3` 必须在可以表示为有限精度的浮点数之前舍入。舍入规则和模式在 IEEE-754 中指定。最常用的模式是*向偶数舍入*，缩写为向最近舍入。

.. _normal-and-subnormal-values:

5.5.1.2. 正规和次正规值
^^^^^^^^^^^^^^^^^^^^^^^

任何指数字段既不全为零也不全为一的浮点值都称为*正规*值。

浮点值的一个重要方面是最小可表示的正正规数 ``FLT_MIN`` 与零之间的巨大差距。这个差距比 ``FLT_MIN`` 与第二小正规数之间的差距大得多。

浮点*次正规*数（也称为*非正规数*）是为了解决这个问题而引入的。次正规浮点值用指数中所有位都设置为零且有效数字中至少设置一位来表示。次正规数是 IEEE-754 浮点标准的必要部分。

次正规数允许逐渐丢失精度，作为突然向零舍入的替代方案。然而，次正规数的计算成本更高。因此，不需要严格精度的应用程序可能会选择避免它们以提高性能。``nvcc`` 编译器允许通过设置 ``-ftz=true`` 选项（刷新为零）来禁用次正规数，这也包含在 ``--use_fast_math`` 中。

.. _special-values:

5.5.1.3. 特殊值
^^^^^^^^^^^^^^^

IEEE-754 标准为浮点数定义了三种特殊值：

**零：**

- 数学零。
- 注意浮点零有两种可能的表示：``+0`` 和 ``-0``。这与整数零的表示不同。
- ``+0 == -0`` 计算为 ``true``。
- 零用指数和有效数字中所有位都设置为 0 来编码。

**无穷大：**

- 浮点数根据饱和算术行为，其中超出可表示范围的操作结果为 ``+Infinity`` 或 ``-Infinity``。
- 无穷大用指数中所有位都设置为 1 且有效数字中所有位都设置为 0 来编码。无穷大值正好有两个编码。
- 涉及无穷大和非零有限值的算术运算通常结果为无穷大。不确定形式如 ``Inf * 0.0``、``Inf - Inf``、``Inf / Inf`` 和 ``0.0 / 0.0`` 结果为 NaN。

**非数字（NaN）：**

- NaN 是表示未定义或不可表示值的特殊符号。常见示例是 ``0.0 / 0.0``、``sqrt(-1.0)`` 或 ``+Inf - Inf``。
- NaN 用指数中所有位都设置为 1 且有效数字中有任何位模式（除了所有位都设置为 0）来编码。有 :math:`2^{\mathrm{mantissa} + 1} - 2` 个可能的编码。
- 任何涉及 NaN 的算术运算都将结果为 NaN。
- 任何涉及 NaN 的有序比较（``<``、``<=``、``>``、``>=``、``==``）都将结果为 ``false``，包括 ``NaN == NaN``（非自反）。无序比较 ``NaN != NaN`` 返回 ``true``。
- NaN 有两种形式：
  
  - 静默 NaN（``qNaN``）用于传播无效操作或值导致的错误。无效算术运算通常产生静默 NaN。它们用有效数字的最高有效位设置为 1 来编码。
  - 信号 NaN（``sNaN``）旨在引发无效操作异常。信号 NaN 通常显式创建。它们用有效数字的最高有效位设置为 0 来编码。
  - 静默和信号 NaN 的确切位模式是实现定义的。CUDA 提供 `cuda::std::numeric_limits<T>::quiet_NaN <https://en.cppreference.com/w/cpp/types/numeric_limits/quiet_NaN.html>`__ 和 `cuda::std::numeric_limits<T>::signaling_NaN <https://en.cppreference.com/w/cpp/types/numeric_limits/signaling_NaN.html>`__ 常量来获取其特殊值。

.. _associativity:

5.5.1.4. 结合律
^^^^^^^^^^^^^^^

重要的是要注意，数学算术的规则和属性由于其有限精度而不能直接应用于浮点算术。以下示例显示单精度值 ``A``、``B`` 和 ``C`` 以及使用不同结合律计算的数学精确值。

数学上，:math:`(A + B) + C` 等于 :math:`A + (B + C)`。

设 :math:`\mathrm{rn}(x)` 表示对 :math:`x` 的一次舍入步骤。根据 IEEE-754 在向最近舍入模式下以单精度浮点算术执行相同计算，我们得到：

对于参考，上面还计算了数学精确结果。根据 IEEE-754 计算的结果与数学精确结果不同。此外，对应于和 :math:`\mathrm{rn}(\mathrm{rn}(A + B) + C)` 和 :math:`\mathrm{rn}(A + \mathrm{rn}(B + C))` 的结果彼此不同。在这种情况下，:math:`\mathrm{rn}(A + \mathrm{rn}(B + C))` 比键 :math:`\mathrm{rn}(\mathrm{rn}(A + B) + C)` 更接近正确的数学结果。

此示例表明，看似相同的计算可以产生不同的结果，即使所有基本运算都符合 IEEE-754。

.. _fused-multiply-add-fma:

5.5.1.5. 融合乘加（FMA）
^^^^^^^^^^^^^^^^^^^^^^^^

融合乘加（FMA）运算仅用一次舍入步骤计算结果。没有 FMA，结果需要两次舍入步骤：一次用于乘法，一次用于加法。因为 FMA 只使用一次舍入步骤，它产生更准确的结果。

融合乘加运算可能以不同于两个单独运算的方式影响 NaN 的传播。然而，FMA NaN 处理在所有目标上并非普遍相同。具有多个 NaN 操作数的不同实现可能倾向于静默 NaN 或传播一个操作数的有效负载。此外，当存在多个 NaN 操作数时，IEEE-754 不严格要求确定性的有效负载选择顺序。NaN 也可能出现在中间计算中，例如 :math:`\infty \times 0 + 1` 或 :math:`1 \times \infty - \infty`，导致实现定义的 NaN 有效负载。

为清晰起见，首先考虑一个使用十进制算术的示例来说明 FMA 运算如何工作。我们将使用总共五位精度（小数点后四位）计算 :math:`x^2 - 1`。

- 对于 :math:`x = 1.0008`，正确的数学结果是 :math:`x^2 - 1 = 1.60064 \times 10^{-4}`。使用小数点后四位的最接近数字是 :math:`1.6006 \times 10^{-4}`。
- 融合乘加运算仅用一次舍入步骤实现正确结果 :math:`\mathrm{rn}(x \times x - 1) = 1.6006 \times 10^{-4}`。
- 替代方案是分别计算乘法和加法步骤。:math:`x^2 = 1.00160064` 转换为 :math:`\mathrm{rn}(x \times x) = 1.0016`。最终结果是 :math:`\mathrm{rn}(\mathrm{rn}(x \times x) -1) = 1.6000 \times 10^{-4}`。

分别舍入乘法和加法产生的结果偏差 :math:`0.00064`。相应的 FMA 计算仅偏差 :math:`0.00004`，其结果最接近正确的数学答案。结果总结如下：

下面是另一个使用二进制单精度值的示例：

- 分别计算乘法和加法会导致所有精度位丢失，产生 :math:`0`。
- 另一方面，计算 FMA 提供等于数学值的结果。

融合乘加有助于防止相减抵消期间的精度丢失。当添加具有相反符号的相似数量级时，会发生相减抵消。在这种情况下，许多前导位相互抵消，导致有意义位更少。融合乘加在乘法期间计算双宽度乘积。因此，即使在加法期间发生相减抵消，乘积中仍有足够的有效位剩余以产生精确结果。

**CUDA 中的融合乘加支持：**

CUDA 以多种方式为 ``float`` 和 ``double`` 数据类型提供融合乘加运算：

- 使用标志 ``-fmad=true`` 或 ``--use_fast_math`` 编译时的 ``x * y + z``。
- ``fma(x, y, z)`` 和 ``fmaf(x, y, z)`` `C 标准库函数 <https://en.cppreference.com/w/c/numeric/math/fma>`__。
- ``__fmaf_[rd, rn, ru, rz]``、``__fmaf_ieee_[rd, rn, ru, rz]`` 和 ``__fma_[rd, rn, ru, rz]`` `CUDA 数学内建函数 <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html>`__。
- ``cuda::std::fma(x, y, z)`` 和 ``cuda::std::fmaf(x, y, z)`` `CUDA C++ 标准库函数 <https://en.cppreference.com/w/cpp/numeric/math/fma.html>`__。

.. _dot-product-example:

5.5.1.6. 点积示例
^^^^^^^^^^^^^^^^^

考虑找到两个短向量 :math:`\vec{a}` 和 :math:`\vec{b}` 的点积的问题，两者都有四个元素。

尽管这个运算在数学上很容易写下来，但在软件中实现它涉及几种可能导致略有不同结果的替代方案。这里提出的所有策略都使用完全符合 IEEE-754 的运算。

**示例算法 1：** 计算点积的最简单方法是使用乘积的顺序和，保持乘法和加法分离。

> 最终结果可以表示为 :math:`((((a_1 \times b_1) + (a_2 \times b_2)) + (a_3 \times b_3)) + (a_4 \times b_4))`。

**示例算法 2：** 使用融合乘加顺序计算点积。

> 最终结果可以表示为 :math:`(a_4 \times b_4) + ((a_3 \times b_3) + ((a_2 \times b_2) + (a_1 \times b_1 + 0)))`。

**示例算法 3：** 使用分治策略计算点积。首先，我们找到向量前半部分和后半部分的点积。然后，我们使用加法组合这些结果。此算法称为"并行算法"，因为两个子问题可以并行计算，因为它们彼此独立。然而，该算法不需要并行实现；它可以用单个线程实现。

> 最终结果可以表示为 :math:`((a_1 \times b_1) + (a_2 \times b_2)) + ((a_3 \times b_3) + (a_4 \times b_4))`。

.. _rounding:

5.5.1.7. 舍入
^^^^^^^^^^^^^

IEEE-754 标准要求支持几种运算。这些包括算术运算如加法、减法、乘法、除法、平方根、融合乘加、求余数、转换、缩放、符号和比较运算。对于给定的格式和舍入模式，这些运算的结果保证在标准的所有实现中一致。

**舍入模式**

IEEE-754 标准定义了四种舍入模式：*向最近舍入*、*向正无穷舍入*、*向负无穷舍入*和*向零舍入*。CUDA 支持所有四种模式。默认情况下，运算使用*向最近舍入*。`内建数学函数 <#mathematical-functions-appendix-intrinsic-functions>`__ 可用于为单个运算选择其他舍入模式。

.. list-table:: 舍入模式
   :header-rows: 1
   :widths: 20 60

   * - 舍入模式
     - 解释
   * - ``rn``
     - 向最近舍入，平局向偶数
   * - ``rz``
     - 向零舍入
   * - ``ru``
     - 向 :math:`\infty` 舍入
   * - ``rd``
     - 向 :math:`-\infty` 舍入

.. _notes-on-host-device-computation-accuracy:

5.5.1.8. 主机/设备计算精度说明
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

浮点计算结果的精度受多种因素影响。本节总结了在浮点计算中获得可靠结果的重要考虑因素。其中一些方面在前面章节中有更详细的描述。

这些方面在比较 CPU 和 GPU 之间的结果时也很重要。主机和设备执行之间的差异必须仔细解释。存在差异并不一定意味着 GPU 的结果不正确或 GPU 有问题。

**结合律**：

> 有限精度中的浮点加法和乘法不是`可结合的 <#associativity>`__，因为它们经常产生无法直接以目标格式表示的数学值，需要舍入。评估这些运算的顺序会影响舍入误差的累积方式，并可能显著改变最终结果。

**融合乘加**：

> `融合乘加 <#fused-multiply-add>`__ 在单个运算中计算 :math:`a \times b + c`，从而产生更高的精度和更快的执行时间。最终结果的精度可能受其使用影响。融合乘加依赖于硬件支持，可以通过调用相关函数显式启用或通过编译器优化标志隐式启用。

**精度**：

> 增加浮点精度可能潜在地提高结果的精度。更高的精度减少了有效数字的丢失，并能够表示更广泛的值范围。然而，更高精度类型的吞吐量更低，消耗更多寄存器。此外，使用它们显式存储输入和输出会增加内存使用和数据移动。

**编译器标志和优化**：

> 所有主要编译器都提供各种优化标志来控制浮点运算的行为。
> 
> - GCC（``-O3``）、Clang（``-O3``）、nvcc（``-O3``）和 Microsoft Visual Studio（``/O2``）的最高优化级别不影响浮点语义。然而，内联、循环展开、向量化和公共子表达式消除可能会影响结果。NVC++ 编译器还需要标志 ``-Kieee -Mnofma`` 才能实现 IEEE-754 兼容语义。
> - 请参阅 `GCC <https://gcc.gnu.org/wiki/FloatingPointMath>`__、`Clang <https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior>`__、`Microsoft Visual Studio 编译器 <https://learn.microsoft.com/en-us/cpp/build/reference/fp-specify-floating-point-behavior>`__、`nvc++ <https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-user-guide/index.html#gpu>`__ 和 `Arm C/C++ 编译器 <https://developer.arm.com/documentation/101458/2404/Compiler-options?lang=en>`__ 文档，获取影响浮点行为的选项的详细信息。
> - 另请参阅 ``nvcc`` `用户手册 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math>`__，获取专门影响 CUDA 设备代码中浮点行为的编译器标志的详细描述：``-ftz``、``-prec-div``、``-prec-sqrt``、``-fmad``、``--use_fast_math``。除了这些浮点选项外，在用户程序的上下文中验证其他编译器优化的效果也很重要。鼓励用户通过广泛的测试验证其结果的正确性，并比较启用优化与禁用所有设备代码优化时获得的结果；另请参阅 ``-G`` 编译器标志。

**库实现**：

> IEEE-754 标准之外定义的函数不能保证正确舍入，并且取决于实现定义的行为。因此，结果可能因不同平台而异，包括主机、设备和不同设备架构之间。

**确定性结果**：

> 确定性结果是指在相同指定条件下使用相同输入运行时每次计算相同的逐位数值输出。这些条件包括：
> 
> - 硬件依赖性，例如在同一 CPU 处理器或 GPU 设备上执行。
> - 编译器方面，例如编译器版本和`编译器标志和优化 <#compiler-flags-and-optimizations>`__。
> - 影响计算的运行时条件，例如`舍入模式 <#floating-point-rounding>`__ 或环境变量。
> - 计算的相同输入。
> - 线程配置，包括参与计算的线程数量及其组织，例如块和 grid 大小。
> - `算术原子操作 <cpp-language-extensions.html#atomic-functions>`__ 的排序取决于硬件调度，这可能因运行而异。

**利用 CUDA 库**：

> `CUDA 数学库 <https://developer.nvidia.com/gpu-accelerated-libraries>`__、`C 标准库数学函数 <https://docs.nvidia.com/cuda/cuda-math-api/index.html>`__ 和 `C++ 标准库数学函数 <https://nvidia.github.io/cccl/libcudacxx/standard_api.html>`__ 旨在为常见功能提高开发人员生产力，特别是对于浮点数学和数值密集型例程。这些功能提供一致的高级接口，经过优化，并在平台和边缘情况下广泛测试。鼓励用户充分利用这些库，避免繁琐的手动重新实现。

.. _floating-point-data-types:

5.5.2. 浮点数据类型
-------------------

CUDA 支持 Bfloat16、半精度、单精度、双精度和四精度浮点数据类型。下表总结了 CUDA 中支持的浮点数据类型及其要求。

.. list-table:: 支持的浮点类型
   :header-rows: 1
   :widths: 20 20 15 35

   * - 精度/名称
     - 数据类型
     - IEEE-754
     - 头文件/内建
   * - Bfloat16
     - ``__nv_bfloat16``
     - ❌
     - ``<cuda_bf16.h>``
   * - 半精度
     - ``__half``
     - ✅
     - ``<cuda_fp16.h>``
   * - 单精度
     - ``float``
     - ✅
     - 内建
   * - 双精度
     - ``double``
     - ✅
     - 内建
   * - 四精度
     - ``__float128``/``_Float128``
     - ✅
     - 内建，数学函数需要 ``<crt/device_fp128_functions.h>``

CUDA 还支持 `TensorFloat-32 <https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/>`__（``TF32``）、`微缩放（MX） <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__ 浮点类型和其他 `低精度数值格式 <https://resources.nvidia.com/en-us-blackwell-architecture>`__，这些不用于通用计算，而是用于涉及张量核心的专用目的。这些包括 4、6 和 8 位浮点类型。有关更多详细信息，请参阅 `CUDA Math API <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/structs.html>`__。

.. list-table:: 支持的浮点类型属性
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - 精度/名称
     - 最大值
     - 最小正值
     - 最小正次正规值
     - Epsilon
   * - Bfloat16
     - :math:`\approx 2^{128}`
     - :math:`2^{-126}`
     - :math:`2^{-133}`
     - :math:`2^{-7}`
   * - 半精度
     - :math:`\approx 2^{16}` (65504)
     - :math:`2^{-14}`
     - :math:`2^{-24}`
     - :math:`2^{-10}`
   * - 单精度
     - :math:`\approx 2^{128}`
     - :math:`2^{-126}`
     - :math:`2^{-149}`
     - :math:`2^{-23}`
   * - 双精度
     - :math:`\approx 2^{1024}`
     - :math:`2^{-1022}`
     - :math:`2^{-1074}`
     - :math:`2^{-52}`
   * - 四精度
     - :math:`\approx 2^{16384}`
     - :math:`2^{-16382}`
     - :math:`2^{-16494}`
     - :math:`2^{-112}`

.. hint::

   `CUDA C++ 标准库 <cpp-language-support.html#cpp-standard-library>`__ 在 ``<cuda/std/limits>`` 头文件中提供 ``cuda::std::numeric_limits`` 来查询支持的浮点类型的属性和范围，包括 `微缩放格式（MX） <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__。有关可查询属性的列表，请参阅 `C++ 参考 <https://en.cppreference.com/w/cpp/types/numeric_limits.html>`__。

**复数支持：**

- `CUDA C++ 标准库 <cpp-language-support.html#cpp-standard-library>`__ 在 ``<cuda/std/complex>`` 头文件中通过 `cuda::std::complex <https://en.cppreference.com/w/cpp/numeric/complex>`__ 类型支持复数。有关更多详细信息，另请参阅 `libcu++ 文档 <https://nvidia.github.io/cccl/libcudacxx/standard_api/numerics_library/complex.html>`__。

- CUDA 还在 ``cuComplex.h`` 头文件中通过 ``cuComplex`` 和 ``cuDoubleComplex`` 类型提供对复数的基本支持。

.. _cuda-and-ieee-754-compliance:

5.5.3. CUDA 和 IEEE-754 兼容性
------------------------------

所有 GPU 设备都遵循 `IEEE 754-2019 <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229>`__ 二进制浮点运算标准，但有以下限制：

- 没有动态可配置的舍入模式；然而，大多数运算支持多种常量 IEEE 舍入模式，可通过特定命名的`设备内建函数 <#mathematical-functions-appendix-intrinsic-functions>`__ 选择。

- 没有检测浮点异常的机制，因此所有运算的行为就像 IEEE-754 异常总是被屏蔽一样。如果发生异常事件，则传递 IEEE-754 定义的默认屏蔽响应。因此，虽然支持信号 NaN（``SNaN``）编码，但它们不是信号的，而是作为静默异常处理。

- 浮点运算可能会更改输入 NaN 有效负载的位模式。绝对值和取反等运算也可能不符合 IEEE 754 要求，这可能导致 NaN 的符号以实现定义的方式更新。

为了最大化结果的可移植性，建议用户使用 ``nvcc`` 编译器浮点选项的默认设置：``-ftz=false``、``-prec-div=true`` 和 ``-prec-sqrt=true``，并且不使用 ``--use_fast_math`` 选项。请注意，浮点表达式重关联和收缩默认是允许的，类似于 ``--fmad=true`` 选项。另请参阅 ``nvcc`` `用户手册 <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math>`__ 获取这些编译标志的详细描述。

IEEE-754 和 C/C++ 语言标准没有明确解决在舍入到整数值超出目标整数格式范围的情况下将浮点值转换为整数值的问题。GPU 设备到范围的钳位行为在 `PTX ISA 转换指令 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt>`__ 部分中描述。然而，当超出范围的转换不是直接通过 PTX 指令调用时，编译器优化可能会利用未指定行为条款，从而导致未定义行为和无效的 CUDA 程序。CUDA Math 文档在每个函数/内建基础上向用户发出警告。例如，考虑 `__double2int_rz() <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html#_CPPv415__double2int_rzd>`__ 指令。这可能不同于主机编译器和库实现的行为方式。

**原子函数次正规值行为**：

原子运算关于浮点次正规值具有以下行为，无论编译器标志 ``-ftz`` 的设置如何：

- 全局内存上的原子单精度浮点加法始终在刷新为零模式下运算，即行为等同于 PTX ``add.rn.ftz.f32`` 语义。

- 共享内存上的原子单精度浮点加法始终支持次正规值，即行为等同于 PTX ``add.rn.f32`` 语义。

.. _cuda-and-c-c-compliance:

5.5.4. CUDA 和 C/C++ 兼容性
---------------------------

**浮点异常：**

与主机实现不同，设备代码中支持的数学运算符和函数不设置全局 ``errno`` 变量，也不报告`浮点异常 <https://en.cppreference.com/w/cpp/numeric/fenv/FE_exceptions>`__ 来指示错误。因此，如果需要错误诊断机制，用户应为函数实现额外的输入和输出筛选。

**浮点运算的未定义行为：**

数学运算的未定义行为的常见条件包括：

- 数学运算符和函数的无效参数：
  
  - 使用未初始化的浮点变量。
  - 在其生存期之外使用浮点变量。
  - 有符号整数溢出。
  - 解引用无效指针。

- 浮点特定的未定义行为：
  
  - 将浮点值转换为结果不可表示的整数类型是未定义行为。这也包括 NaN 和无穷大。

用户有责任确保 CUDA 程序的有效性。无效参数可能导致未定义行为，并受编译器优化影响。

与整数除以零相反，浮点除以零不是未定义行为，也不受编译器优化影响；相反，它是实现特定的行为。符合 `IEC-60559 <https://en.cppreference.com/w/cpp/types/numeric_limits/is_iec559.html>`__（IEEE-754）的 C++ 实现（包括 CUDA）产生无穷大。请注意，无效浮点运算产生 NaN，不应误解为未定义行为。示例包括零除以零和无穷大除以无穷大。

**浮点字面量可移植性：**

C 和 C++ 都允许以十进制或十六进制表示法表示浮点值。`C99 <https://en.cppreference.com/w/c/language/floating_constant.html>`__ 和 `C++17 <https://en.cppreference.com/w/cpp/language/floating_literal.html>`__ 支持的十六进制浮点字面量以科学记数法表示实数值，可以精确地以基数为 2 表示。然而，这并不保证字面量将映射到目标变量中存储的实际值（见下一段）。相反，十进制浮点字面量可能表示无法以基数为 2 表示的数值。

根据 `C++ 标准规则 <https://eel.is/c++draft/lex.fcon#3>`__，十六进制和十进制浮点字面量舍入到最接近的可表示值，较大或较小，以实现定义的方式选择。此舍入行为可能在主机和设备之间不同。

相同浮点表达式的运行时和编译时评估受以下可移植性问题影响：

- 浮点表达式的运行时评估可能受选定的舍入模式、浮点收缩（FMA）和重关联编译器设置以及浮点异常的影响。请注意，CUDA 不支持浮点异常，`舍入模式 <#floating-point-rounding>`__ 默认设置为*向偶数舍入*。可以使用`内建函数 <#mathematical-functions-appendix-intrinsic-functions>`__ 选择其他舍入模式。

- 编译器可能使用更高精度的内部表示来处理常量表达式。

- 编译器可能执行优化，如常量折叠、常量传播和公共子表达式消除，这可能导致不同的最终值或比较结果。

.. note::

   有关浮点计算的详细内容，请参考 `CUDA 官方文档 <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html>`__。