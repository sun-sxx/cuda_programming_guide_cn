.. _cpp-language-support:

5.3. C++ 语言支持
=================

``nvcc`` 根据以下规范处理 CUDA 和设备代码：

- **C++03** (ISO/IEC 14882:2003)，``--std=c++03`` 标志。

- **C++11** (ISO/IEC 14882:2011)，``--std=c++11`` 标志。

- **C++14** (ISO/IEC 14882:2014)，``--std=c++14`` 标志。

- **C++17** (ISO/IEC 14882:2017)，``--std=c++17`` 标志。

- **C++20** (ISO/IEC 14882:2020)，``--std=c++20`` 标志。

传递 ``nvcc`` ``-std=c++<version>`` 标志会启用与指定版本相关的所有 C++ 特性，并以相应的 C++ 方言选项调用主机预处理器、编译器和链接器。

编译器支持所支持标准的所有语言特性，但以下章节中报告的限制除外。

.. _c-11-language-features:

5.3.1. C++11 语言特性
---------------------

.. list-table:: NVCC 设备代码支持的 C++11 语言特性
   :header-rows: 1
   :widths: 60 25 15

   * - 语言特性
     - C++11 提案
     - NVCC/CUDA Toolkit 7.x
   * - 右值引用
     - `N2118 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n2118.html>`__
     - ✅
   * - ``*this`` 的右值引用
     - `N2439 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2439.htm>`__
     - ✅
   * - 通过右值初始化类对象
     - `N1610 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1610.html>`__
     - ✅
   * - 非静态数据成员初始化器
     - `N2756 <http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2008/n2756.htm>`__
     - ✅
   * - 可变参数模板
     - `N2242 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2242.pdf>`__
     - ✅
   * - 扩展可变参数模板模板参数
     - `N2555 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2555.pdf>`__
     - ✅
   * - 初始化列表
     - `N2672 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2672.htm>`__
     - ✅
   * - 静态断言
     - `N1720 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1720.html>`__
     - ✅
   * - ``auto`` 类型变量
     - `N1984 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1984.pdf>`__
     - ✅
   * - 多声明符 ``auto``
     - `N1737 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1737.pdf>`__
     - ✅
   * - 移除 ``auto`` 作为存储类说明符
     - `N2546 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2546.htm>`__
     - ✅
   * - 新函数声明器语法
     - `N2541 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2541.htm>`__
     - ✅
   * - Lambda 表达式
     - `N2927 <http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2927.pdf>`__
     - ✅
   * - 表达式的声明类型
     - `N2343 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2343.pdf>`__
     - ✅
   * - 不完整返回类型
     - `N3276 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3276.pdf>`__
     - ✅
   * - 右尖括号
     - `N1757 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1757.html>`__
     - ✅
   * - 函数模板的默认模板参数
     - `DR226 <http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#226>`__
     - ✅
   * - 解决表达式的 SFINAE 问题
     - `DR339 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2634.html>`__
     - ✅
   * - 别名模板
     - `N2258 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2258.pdf>`__
     - ✅
   * - 外部模板
     - `N1987 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1987.htm>`__
     - ✅
   * - 空指针常量
     - `N2431 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2431.pdf>`__
     - ✅
   * - 强类型枚举
     - `N2347 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2347.pdf>`__
     - ✅
   * - 枚举的前向声明
     - `N2764 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2764.pdf>`__
     - ✅
   * - 标准化属性语法
     - `N2761 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2761.pdf>`__
     - ✅
   * - 广义常量表达式
     - `N2235 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2235.pdf>`__
     - ✅
   * - 对齐支持
     - `N2341 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2341.pdf>`__
     - ✅
   * - 条件支持的行为
     - `N1627 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1627.pdf>`__
     - ✅
   * - 将未定义行为改为可诊断错误
     - `N1727 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1727.pdf>`__
     - ✅
   * - 委托构造函数
     - `N1986 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1986.pdf>`__
     - ✅
   * - 继承构造函数
     - `N2540 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2540.htm>`__
     - ✅
   * - 显式转换运算符
     - `N2437 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2437.pdf>`__
     - ✅
   * - 新字符类型
     - `N2249 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2249.html>`__
     - ✅
   * - Unicode 字符串字面量
     - `N2442 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm>`__
     - ✅
   * - 原始字符串字面量
     - `N2442 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm>`__
     - ✅
   * - 字面量中的通用字符名
     - `N2170 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2170.html>`__
     - ✅
   * - 用户定义字面量
     - `N2765 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2765.pdf>`__
     - ✅
   * - 标准布局类型
     - `N2342 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2342.htm>`__
     - ✅
   * - 默认函数
     - `N2346 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm>`__
     - ✅
   * - 删除函数
     - `N2346 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm>`__
     - ✅
   * - 扩展友元声明
     - `N1791 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1791.pdf>`__
     - ✅
   * - 扩展 ``sizeof``
     - `N2253 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2253.html>`__
     - ✅
   * - 内联命名空间
     - `N2535 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2535.htm>`__
     - ✅
   * - 无限制联合
     - `N2544 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2544.pdf>`__
     - ✅
   * - 局部和未命名类型作为模板参数
     - `N2657 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2657.htm>`__
     - ✅
   * - 基于范围的 for 循环
     - `N2930 <http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2930.html>`__
     - ✅
   * - 显式 ``virtual`` 覆盖
     - `N2928 <http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2928.htm>`__
     - ✅
   * - 允许移动构造函数抛出异常 [noexcept]
     - `N3050 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3050.html>`__
     - ✅

**并发**

.. list-table:: NVCC 设备代码支持的 C++11 并发特性
   :header-rows: 1
   :widths: 60 25 15

   * - 语言特性
     - C++11 提案
     - NVCC/CUDA Toolkit 7.x
   * - 序列点
     - `N2239 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2239.html>`__
     - ❌
   * - 原子操作
     - `N2427 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2427.html>`__
     - ❌
   * - 强比较并交换
     - `N2748 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2748.html>`__
     - ❌
   * - 双向栅栏
     - `N2752 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2752.htm>`__
     - ❌
   * - 内存模型
     - `N2429 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2429.htm>`__
     - ❌
   * - 数据依赖排序：原子和内存模型
     - `N2664 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2664.htm>`__
     - ❌
   * - 传播异常
     - `N2179 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2179.html>`__
     - ❌
   * - 允许在信号处理程序中使用原子
     - `N2547 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2547.htm>`__
     - ❌
   * - 线程局部存储
     - `N2659 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2659.htm>`__
     - ❌
   * - 并发动态初始化和销毁
     - `N2660 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2660.htm>`__
     - ❌

**C99 在 C++11 中的特性**

.. list-table:: NVCC 设备代码支持的 C99 特性
   :header-rows: 1
   :widths: 60 25 15

   * - 语言特性
     - C++11 提案
     - NVCC/CUDA Toolkit 7.x
   * - ``__func__`` 预定义标识符
     - `N2340 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2340.htm>`__
     - ✅
   * - C99 预处理器
     - `N1653 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1653.htm>`__
     - ✅
   * - ``long long``
     - `N1811 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1811.pdf>`__
     - ✅
   * - 扩展整数类型
     - `N1988 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1988.pdf>`__
     - ❌

.. _c-14-language-features:

5.3.2. C++14 语言特性
---------------------

.. list-table:: NVCC 设备代码支持的 C++14 语言特性
   :header-rows: 1
   :widths: 60 25 15

   * - 语言特性
     - C++14 提案
     - NVCC/CUDA Toolkit 9.x
   * - 某些 C++ 上下文转换的调整
     - `N3323 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3323.pdf>`__
     - ✅
   * - 二进制字面量
     - `N3472 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3472.pdf>`__
     - ✅
   * - 推导返回类型的函数
     - `N3638 <https://isocpp.org/files/papers/N3638.html>`__
     - ✅
   * - 广义 lambda 捕获（init-capture）
     - `N3648 <https://isocpp.org/files/papers/N3648.html>`__
     - ✅
   * - 泛型（多态）lambda 表达式
     - `N3649 <https://isocpp.org/files/papers/N3649.html>`__
     - ✅
   * - 变量模板
     - `N3651 <https://isocpp.org/files/papers/N3651.pdf>`__
     - ✅
   * - 放宽 constexpr 函数的要求
     - `N3652 <https://isocpp.org/files/papers/N3652.html>`__
     - ✅
   * - 成员初始化器和聚合
     - `N3653 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3653.html>`__
     - ✅
   * - 澄清内存分配
     - `N3664 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3664.html>`__
     - ❌
   * - 带大小的释放
     - `N3778 <https://isocpp.org/files/papers/n3778.html>`__
     - ❌
   * - ``[[deprecated]]`` 属性
     - `N3760 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3760.html>`__
     - ✅
   * - 单引号作为数字分隔符
     - `N3781 <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3781.pdf>`__
     - ✅

.. _c-17-language-features:

5.3.3. C++17 语言特性
---------------------

.. list-table:: NVCC 设备代码支持的 C++17 语言特性
   :header-rows: 1
   :widths: 60 25 15

   * - 语言特性
     - C++17 提案
     - NVCC/CUDA Toolkit 11.x
   * - 移除三字符组
     - `N4086 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4086.html>`__
     - ✅
   * - ``u8`` 字符字面量
     - `N4267 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4267.html>`__
     - ✅
   * - 折叠表达式
     - `N4295 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4295.html>`__
     - ✅
   * - 命名空间和枚举器的属性
     - `N4266 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4266.html>`__
     - ✅
   * - 嵌套命名空间定义
     - `N4230 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4230.html>`__
     - ✅
   * - 允许所有非类型模板参数的常量求值
     - `N4268 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4268.html>`__
     - ✅
   * - 扩展 ``static_assert``
     - `N3928 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3928.pdf>`__
     - ✅
   * - ``auto`` 从花括号初始化列表推导的新规则
     - `N3922 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3922.html>`__
     - ✅
   * - ``[[fallthrough]]`` 属性
     - `P0188R1 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0188r1.pdf>`__
     - ✅
   * - ``[[nodiscard]]`` 属性
     - `P0189R1 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0189r1.pdf>`__
     - ✅
   * - ``[[maybe_unused]]`` 属性
     - `P0212R1 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0212r1.pdf>`__
     - ✅
   * - 聚合初始化扩展
     - `P0017R1 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0017r1.html>`__
     - ✅
   * - ``constexpr`` lambda 的措辞
     - `P0170R1 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0170r1.pdf>`__
     - ✅
   * - 一元折叠和空参数包
     - `P0036R0 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0036r0.pdf>`__
     - ✅
   * - 泛化基于范围的 for 循环
     - `P0184R0 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0184r0.html>`__
     - ✅
   * - Lambda 按值捕获 ``*this``
     - `P0018R3 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0018r3.html>`__
     - ✅
   * - ``enum class`` 变量的构造规则
     - `P0138R2 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0138r2.pdf>`__
     - ✅
   * - C++ 十六进制浮点字面量
     - `P0245R1 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0245r1.html>`__
     - ✅
   * - 保证复制省略
     - `P0135R1 <https://wg21.link/p0135>`__
     - ✅
   * - ``constexpr if``
     - `P0292R2 <https://wg21.link/p0292>`__
     - ✅
   * - 带初始化器的选择语句
     - `P0305R1 <https://wg21.link/p0305>`__
     - ✅
   * - 类模板参数推导
     - `P0091R3 <https://wg21.link/p0091>`__
     - ✅
   * - 使用 ``auto`` 声明非类型模板参数
     - `P0127R2 <https://wg21.link/p0127>`__
     - ✅
   * - 结构化绑定
     - `P0217R3 <https://wg21.link/p0217>`__
     - ✅
   * - 内联变量
     - `P0386R2 <https://wg21.link/p0386r2>`__
     - ✅

.. _c-20-language-features:

5.3.4. C++20 语言特性
---------------------

需要 GCC 版本 ≥ 10.0、Clang 版本 ≥ 10.0、Microsoft Visual Studio ≥ 2022 和 nvc++ 版本 ≥ 20.7。

.. list-table:: NVCC 设备代码支持的 C++20 语言特性
   :header-rows: 1
   :widths: 60 25 15

   * - 语言特性
     - C++20 提案
     - NVCC/CUDA Toolkit 12.x
   * - 位域的默认成员初始化器
     - `P0683R1 <https://wg21.link/p0683r1>`__
     - ✅
   * - 修复 ``const`` 限定成员指针
     - `P0704R1 <https://wg21.link/p0704r1>`__
     - ✅
   * - 允许 lambda 捕获 ``[=, this]``
     - `P0409R2 <https://wg21.link/p0409r2>`__
     - ✅
   * - 预处理逗号省略的 ``__VA_OPT__``
     - `P0306R4 <https://wg21.link/p0306r4>`__
     - ✅
   * - 指定初始化器
     - `P0329R4 <https://wg21.link/p0329r4>`__
     - ✅
   * - 泛型 lambda 的熟悉模板语法
     - `P0428R2 <https://wg21.link/p0428r2>`__
     - ✅
   * - 概念
     - `P0734R0 <https://wg21.link/p0734r0>`__
     - ✅
   * - 一致比较（``operator<=>``）
     - `P0515R3 <https://wg21.link/p0515r3>`__
     - ✅
   * - 立即函数（``consteval``）
     - `P1073R3 <https://wg21.link/p1073r3>`__
     - ✅
   * - ``std::is_constant_evaluated``
     - `P0595R2 <https://wg21.link/p0595r2>`__
     - ✅
   * - constexpr 限制放宽
     - `P1002R1 <https://wg21.link/p1002r1>`__
     - ✅
   * - 特性测试宏
     - `P0941R2 <https://wg21.link/p0941r2>`__
     - ✅
   * - 模块
     - `P1103R3 <https://wg21.link/p1103r3>`__
     - ❌
   * - 协程
     - `P0912R5 <https://wg21.link/p0912r5>`__
     - ❌
   * - ``constinit``
     - `P1143R2 <https://wg21.link/p1143r2>`__
     - ✅

.. _cuda-c-standard-library:

5.3.5. CUDA C++ 标准库
----------------------

CUDA 提供了 C++ 标准库 (STL) 的实现，称为 `libcu++ <https://nvidia.github.io/cccl/libcudacxx/standard_api.html>`__。该库具有以下优势：

- 功能在主机和设备上都可用。

- 与 CUDA Toolkit 支持的所有 `Linux <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#id59>`__ 和 `Windows <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#id2>`__ 平台兼容。

- 与 CUDA Toolkit 最近两个主要版本支持的所有 `GPU 架构 <https://developer.nvidia.com/cuda-gpus>`__ 兼容。

- 与当前和以前主要版本的所有 `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit-archive>`__ 兼容。

- 提供最新标准版本（包括 C++20、C++23 和 C++26）中可用的 C++ 标准库特性的 C++17 后向移植。

- 支持扩展数据类型，如 128 位整数（``__int128``）、半精度浮点（``__half``）、Bfloat16（``__nv_bfloat16``）和四精度浮点（``__float128``）。

- 针对设备代码进行了高度优化。

此外，``libcu++`` 还提供了 C++ 标准库中不可用的 `扩展功能 <https://nvidia.github.io/cccl/libcudacxx/extended_api.html>`__，以提高生产力和应用程序性能。这些功能包括数学函数、内存操作、同步原语、容器扩展、CUDA 内建函数的高级抽象、C++ PTX 包装器等。

``libcu++`` 作为 `CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`__ 的一部分提供，也是开源 `CCCL <https://nvidia.github.io/cccl/>`__ 仓库的一部分。

.. _c-standard-library-functions:

5.3.6. C 标准库函数
-------------------

.. _clock-and-clock64:

5.3.6.1. ``clock()`` 和 ``clock64()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   __host__ __device__ clock_t   clock();
   __device__          long long clock64();

在设备代码中执行时，返回每个多处理器的计数器值，该计数器每个时钟周期递增一次。在内核开始和结束时采样此计数器，减去两个值，并为每个线程记录结果，可以估算设备执行该线程所花费的时钟周期数。但是，此值并不代表设备执行该线程指令所花费的实际时钟周期数。前者大于后者，因为线程是时间片轮转的。

.. hint::

   在 ``<cuda/std/ctime>`` 头文件中提供了相应的 `CUDA C++ 函数 <https://en.cppreference.com/w/cpp/chrono/c/clock.html>`__ ``cuda::std::clock()``。

   在 ``<cuda/std/chrono>`` `头文件 <https://nvidia.github.io/cccl/libcudacxx/standard_api/time_library.html#libcudacxx-standard-api-time>`__ 中也提供了可移植的 `C++ <https://en.cppreference.com/w/cpp/header/chrono>`__ ``<chrono>`` 实现，用于类似目的。

.. _printf:

5.3.6.2. ``printf()``
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   int printf(const char* format[, arg, ...]);

该函数将内核中的格式化输出打印到主机端输出流。

内核中的 ``printf()`` 函数行为类似于标准 C 库的 ``printf()`` 函数。用户应参考其主机系统的手册页面以获取 ``printf()`` 行为的完整描述。本质上，作为 ``format`` 传入的字符串被输出到主机上的流。

``printf()`` 命令像任何其他设备端函数一样执行：每个线程执行，并在调用线程的上下文中执行。在多线程内核中，对 ``printf()`` 的简单调用将由每个线程使用该线程指定的数据执行。因此，主机流中会出现多个版本的输出字符串，每个对应于遇到 ``printf()`` 的线程。

与返回打印字符数的 C 标准 ``printf()`` 不同，CUDA 的 ``printf()`` 返回解析的参数数量。如果格式字符串后没有参数，则返回 0。如果格式字符串为 ``NULL``，则返回 -1。如果发生内部错误，则返回 -2。

在内部，``printf()`` 使用共享数据结构，因此调用 ``printf()`` 可能会改变线程的执行顺序。特别是，调用 ``printf()`` 的线程可能比不调用 ``printf()`` 的线程执行路径更长，该路径的长度取决于 ``printf()`` 的参数。但是，请注意，CUDA 不保证线程执行顺序，除非在显式的 ``__syncthreads()`` 屏障处。因此，无法判断执行顺序是否被 ``printf()`` 或硬件中的其他调度行为修改。

**格式说明符**

与标准 ``printf()`` 一样，格式说明符采用以下形式：``%[flags][width][.precision][size]type``

支持以下字段。有关所有行为的完整描述，请参阅广泛可用的文档。

- 标志：``#``、``' '``、``0``、``+``、``-``
- 宽度：``*``、``0-9``
- 精度：``0-9``
- 大小：``h``、``l``、``ll``
- 类型：``%cdiouxXpeEfgGaAs``

**限制**

``printf()`` 输出的最终格式化在主机系统上进行。这意味着格式字符串必须被主机系统的编译器和 C 库理解。虽然已尽力确保 CUDA 的 ``printf()`` 函数支持的格式说明符是最常见主机编译器支持的通用子集，但确切的行为将取决于主机操作系统。

``printf()`` 接受所有有效的标志和类型组合。这是因为它无法确定在最终输出格式化的主机系统上什么有效和什么无效。因此，如果程序发出包含无效组合的格式字符串，输出可能是未定义的。

``printf()`` 函数最多可以接受 32 个参数，此外还有格式字符串。任何额外的参数将被忽略，格式说明符将按原样输出。

由于 Windows 平台（32 位）和 Linux 平台（64 位）上 ``long`` 类型的不同大小，在 Linux 机器上编译然后在 Windows 机器上运行的内核将产生包含 ``%ld`` 的所有格式字符串的损坏输出。为确保安全，建议编译和执行平台匹配。

**主机端缓冲区**

``printf()`` 的输出缓冲区在内核启动前设置为固定大小。缓冲区是循环的，因此如果在内核执行期间产生的输出多于缓冲区可以容纳的内容，则会覆盖较旧的输出。仅在执行以下操作之一时刷新缓冲区：

- 通过 ``<<< >>>`` 或 ``cuLaunchKernel()`` 启动内核：在启动开始时，以及如果 ``CUDA_LAUNCH_BLOCKING`` 环境变量设置为 1，则在启动结束时也会刷新，
- 通过 ``cudaDeviceSynchronize()``、``cuCtxSynchronize()``、``cudaStreamSynchronize()``、``cuStreamSynchronize()``、``cudaEventSynchronize()`` 或 ``cuEventSynchronize()`` 进行同步，
- 通过任何阻塞版本的 ``cudaMemcpy*()`` 或 ``cuMemcpy*()`` 进行内存复制，
- 通过 ``cuModuleLoad()`` 或 ``cuModuleUnload()`` 加载/卸载模块，
- 通过 ``cudaDeviceReset()`` 或 ``cuCtxDestroy()`` 销毁上下文。
- 在执行通过 ``cudaLaunchHostFunc()`` 或 ``cuLaunchHostFunc()`` 添加的流回调之前。

请注意，程序退出时缓冲区不会自动刷新。

以下 API 函数设置和检索用于将 ``printf()`` 参数和内部元数据传输到主机的缓冲区大小。默认大小为 1 MB。

- ``cudaDeviceGetLimit(size_t* size, cudaLimitPrintfFifoSize)``
- ``cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size_t size)``

**示例**

以下代码示例：

.. code-block:: c++

   #include <stdio.h>

   __global__ void helloCUDA(float value) {
       printf("Hello thread %d, value=%f\n", threadIdx.x, value);
   }

   int main() {
       helloCUDA<<<1, 5>>>(1.2345f);
       cudaDeviceSynchronize();
       return 0;
   }

将输出：

.. code-block:: text

   Hello thread 2, value=1.2345
   Hello thread 1, value=1.2345
   Hello thread 4, value=1.2345
   Hello thread 0, value=1.2345
   Hello thread 3, value=1.2345

注意每个线程都遇到 ``printf()`` 命令。因此，输出行数与网格中的线程数相同。

.. _memcpy-and-memset:

5.3.6.3. ``memcpy()`` 和 ``memset()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   __host__ __device__ void* memcpy(void* dest, const void* src, size_t size);

该函数从 ``src`` 指向的内存位置复制 ``size`` 字节到 ``dest`` 指向的内存位置。

.. code-block:: c++

   __host__ __device__ void* memset(void* ptr, int value, size_t size);

该函数将 ``ptr`` 指向的内存块的 ``size`` 字节设置为 ``value``，解释为 ``unsigned char``。

.. hint::

   建议使用 ``<cuda/std/cstring>`` `头文件 <https://nvidia.github.io/cccl/libcudacxx/standard_api/c_library/cstring.html#libcudacxx-standard-api-cstring>`__ 中提供的 ``cuda::std::memcpy()`` 和 ``cuda::std::memset()`` 函数作为 ``memcpy`` 和 ``memset`` 的更安全版本。

.. _malloc-and-free:

5.3.6.4. ``malloc()`` 和 ``free()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   __host__ __device__ void* malloc(size_t size);
   // 或 <cuda/std/cstdlib> 头文件中的 cuda::std::malloc()、cuda::std::calloc()

函数 ``malloc()``（设备端）、``cuda::std::malloc()`` 和 ``cuda::std::calloc()`` 从设备堆中分配至少 ``size`` 字节，并返回指向已分配内存的指针。如果内存不足以满足请求，则返回 ``NULL``。返回的指针保证对齐到 16 字节边界。

.. code-block:: c++

   __device__ void* __nv_aligned_device_malloc(size_t size, size_t align);
   // 或 <cuda/std/cstdlib> 头文件中的 cuda::std::aligned_alloc()

函数 ``__nv_aligned_device_malloc()`` 和 `C++ <https://en.cppreference.com/w/cpp/memory/c/aligned_alloc>`__ ``cuda::std::aligned_alloc()`` 从设备堆中分配至少 ``size`` 字节，并返回指向已分配内存的指针。如果内存不足以满足请求的大小或对齐，则返回 ``NULL``。已分配内存的地址是 ``align`` 的倍数。``align`` 必须是非零的 2 的幂。

.. code-block:: c++

   __host__ __device__ void free(void* ptr);
   // 或 <cuda/std/cstdlib> 头文件中的 cuda::std::free()

设备端函数 ``free()`` 和 ``cuda::std::free()`` 释放 ``ptr`` 指向的内存，该内存必须由先前对 ``malloc()``、``cuda::std::malloc()``、``cuda::std::calloc()``、``__nv_aligned_device_malloc()`` 或 ``cuda::std::aligned_alloc()`` 的调用返回。如果 ``ptr`` 为 ``NULL``，则忽略对 ``free()`` 或 ``cuda::std::free()`` 的调用。使用相同的 ``ptr`` 重复调用 ``free()`` 或 ``cuda::std::free()`` 具有未定义行为。

通过 ``malloc()``、``cuda::std::malloc()``、``cuda::std::calloc()``、``__nv_aligned_device_malloc()`` 或 ``cuda::std::aligned_alloc()`` 由给定 CUDA 线程分配的内存在 CUDA 上下文的生存期内保持分配状态，或直到通过调用 ``free()`` 或 ``cuda::std::free()`` 显式释放。此内存可由其他 CUDA 线程使用，即使是来自后续内核启动的线程。任何 CUDA 线程都可以释放由另一个线程分配的内存；但是，应注意确保同一指针不被多次释放。

**堆内存 API**

设备内存堆的大小必须在任何在设备代码中分配或释放内存的程序之前指定，包括 ``new`` 和 ``delete`` 关键字。如果任何程序使用设备内存堆而未显式指定堆大小，则分配 8 MB 的默认堆。

以下 API 函数获取和设置堆大小：

- ``cudaDeviceGetLimit(size_t* size, cudaLimitMallocHeapSize)``
- ``cudaDeviceSetLimit(cudaLimitMallocHeapSize, size_t size)``

授予的堆大小将至少为 ``size`` 字节。`cuCtxGetLimit() <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8>`__ 和 `cudaDeviceGetLimit() <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g720e159aeb125910c22aa20fe9611ec2>`__ 返回当前请求的堆大小。

堆的实际内存分配发生在模块加载到上下文时，无论是通过 CUDA 驱动程序 API 显式加载（参见 `模块 <../03-advanced/driver-api.html#driver-api-module>`__），还是通过 CUDA 运行时 API 隐式加载。如果内存分配失败，模块加载将生成 ``CUDA_ERROR_SHARED_OBJECT_INIT_FAILED`` 错误。

堆大小在模块加载后无法更改，并且不会根据需要动态调整大小。

为设备堆保留的内存是通过主机端 CUDA API 调用（如 ``cudaMalloc()``）分配的内存之外的。

**与主机内存 API 的互操作性**

通过设备端函数 ``malloc()``、``cuda::std::malloc()``、``cuda::std::calloc()``、``__nv_aligned_device_malloc()``、``cuda::std::aligned_alloc()`` 或 ``new`` 关键字分配的内存不能与运行时或驱动程序 API 调用（如 ``cudaMalloc``、``cudaMemcpy`` 或 ``cudaMemset``）一起使用或释放。同样，通过主机运行时 API 分配的内存不能使用设备端函数 ``free()``、``cuda::std::free()`` 或 ``delete`` 关键字释放。

**每线程分配示例：**

.. code-block:: c++

   #include <stdlib.h>
   #include <stdio.h>

   __global__ void single_thread_allocation_kernel() {
       size_t size = 123;
       char*  ptr  = (char*) malloc(size);
       memset(ptr, 0, size);
       printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
       free(ptr);
   }

   int main() {
       // 设置 128 MB 的堆大小。
       // 注意这必须在任何内核启动之前完成。
       cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
       single_thread_allocation_kernel<<<1, 5>>>();
       cudaDeviceSynchronize();
       return 0;
   }

将输出：

.. code-block:: text

   Thread 0 got pointer: 0x20d5ffe20
   Thread 1 got pointer: 0x20d5ffec0
   Thread 2 got pointer: 0x20d5fff60
   Thread 3 got pointer: 0x20d5f97c0
   Thread 4 got pointer: 0x20d5f9720

注意每个线程如何遇到 ``malloc()`` 和 ``memset()`` 命令，因此接收并初始化自己的分配。

.. _lambda-expressions:

5.3.7. Lambda 表达式
--------------------

编译器通过将 lambda 表达式或闭包类型（C++11）与最内层封闭函数作用域的执行空间相关联来确定其执行空间。如果没有封闭函数作用域，则执行空间指定为 ``__host__``。

执行空间也可以使用 `扩展 lambda 语法 <#extended-lambdas>`__ 显式指定。

示例：

.. code-block:: c++

   auto global_lambda = [](){ return 0; }; // __host__

   void host_function() {
       auto lambda1 = [](){ return 1; };   // __host__
       [](){ return 3; };                  // __host__，闭包类型（lambda 表达式的主体）
   }

   __device__ void device_function() {
       auto lambda2 = [](){ return 2; };   // __device__
   }

   __global__ void kernel_function(void) {
       auto lambda3 = [](){ return 3; };   // __device__
   }

   __host__ __device__ void host_device_function() {
       auto lambda4 = [](){ return 4; };   // __host__ __device__
   }

   using function_ptr_t = int (*)();

   __device__ void device_function(float          value,
                                   function_ptr_t ptr = [](){ return 4; } /* __host__ */) {}

.. _lambda-expressions-and-global-function-parameters:

5.3.7.1. Lambda 表达式和 ``__global__`` 函数参数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

只有当 lambda 表达式或闭包类型的执行空间为 ``__device__`` 或 ``__host__ __device__`` 时，才能将其用作 ``__global__`` 函数的参数。全局或命名空间作用域的 lambda 表达式不能用作 ``__global__`` 函数的参数。

示例：

.. code-block:: c++

   template <typename T>
   __global__ void kernel(T input) {}

   __device__ void device_function() {
       // 设备内核调用需要单独编译（-rdc=true 标志）
       kernel<<<1, 1>>>([](){});
       kernel<<<1, 1>>>([] __device__() {});          // 扩展 lambda
       kernel<<<1, 1>>>([] __host__ __device__() {}); // 扩展 lambda
   }

   auto global_lambda = [] __host__ __device__() {};

   void host_function() {
       kernel<<<1, 1>>>([] __device__() {});          // 正确，扩展 lambda
       kernel<<<1, 1>>>([] __host__ __device__() {}); // 正确，扩展 lambda
   //  kernel<<<1, 1>>>([](){});                      // 错误，主机执行空间的闭包类型
   //  kernel<<<1, 1>>>(global_lambda);               // 错误，扩展 lambda，但在全局作用域
   }

.. _extended-lambdas:

5.3.7.2. 扩展 Lambda
^^^^^^^^^^^^^^^^^^^^

``nvcc`` 标志 ``--extended-lambda`` 允许在 lambda 表达式中显式注释执行空间。这些注释应出现在 lambda 引导符之后和可选的 lambda 声明符之前。

当指定 ``--extended-lambda`` 标志时，``nvcc`` 定义宏 ``__CUDACC_EXTENDED_LAMBDA__``。

- *扩展 lambda* 定义在 ``__host__`` 或 ``__host__ __device__`` 函数的直接或嵌套块作用域内。

- *扩展设备 lambda* 是用 ``__device__`` 关键字注释的 lambda 表达式。

- *扩展主机设备 lambda* 是用 ``__host__ __device__`` 关键字注释的 lambda 表达式。

与标准 lambda 表达式不同，扩展 lambda 可以用作 ``__global__`` 函数中的类型参数。

示例：

.. code-block:: c++

   void host_function() {
       auto lambda1 = [] {};                      // 不是扩展 lambda：没有显式执行空间注释
       auto lambda2 = [] __device__ {};           // 扩展 lambda
       auto lambda3 = [] __host__ __device__ {};  // 扩展 lambda
       auto lambda4 = [] __host__ {};             // 不是扩展 lambda
   }

   __host__ __device__ void host_device_function() {
       auto lambda1 = [] {};                      // 不是扩展 lambda：没有显式执行空间注释
       auto lambda2 = [] __device__ {};           // 扩展 lambda
       auto lambda3 = [] __host__ __device__ {};  // 扩展 lambda
       auto lambda4 = [] __host__ {};             // 不是扩展 lambda
   }

   __device__ void device_function() {
       // 此函数中的所有 lambda 都不是扩展 lambda，
       // 因为封闭函数不是 __host__ 或 __host__ __device__ 函数。
       auto lambda1 = [] {};
       auto lambda2 = [] __device__ {};
       auto lambda3 = [] __host__ __device__ {};
       auto lambda4 = [] __host__ {};
   }

   auto global_lambda = [] __host__ __device__ { }; // 不是扩展 lambda，因为它不是定义在
                                                    // __host__ 或 __host__ __device__ 函数内

.. _extended-lambda-type-traits:

5.3.7.3. 扩展 Lambda 类型特性
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

编译器提供类型特性来在编译时检测扩展 lambda 的闭包类型。

.. code-block:: c++

   bool __nv_is_extended_device_lambda_closure_type(type);

如果 ``type`` 是为扩展 ``__device__`` lambda 创建的闭包类，则函数返回 ``true``，否则返回 ``false``。

.. code-block:: c++

   bool __nv_is_extended_device_lambda_with_preserved_return_type(type);

如果 ``type`` 是为扩展 ``__device__`` lambda 创建的闭包类，并且 lambda 使用尾随返回类型定义，则函数返回 ``true``，否则返回 ``false``。如果尾随返回类型引用任何 lambda 参数名称，则返回类型不会被保留。

.. code-block:: c++

   bool __nv_is_extended_host_device_lambda_closure_type(type);

如果 ``type`` 是为扩展 ``__host__ __device__`` lambda 创建的闭包类，则函数返回 ``true``，否则返回 ``false``。

.. _extended-lambda-restrictions:

5.3.7.4. 扩展 Lambda 限制
^^^^^^^^^^^^^^^^^^^^^^^^^

扩展 lambda 具有以下限制：

1. **嵌套限制**：扩展 lambda 不能定义在另一个扩展 lambda 内部。

2. **泛型 lambda 限制**：扩展 lambda 不能定义在泛型 lambda 内部。

3. **封闭函数要求**：扩展 lambda 的封闭函数必须是具名的，并且其地址必须是可访问的。

4. **主机 - 设备 lambda 限制**：``__host__ __device__`` 扩展 lambda 不能是泛型 lambda。

5. **初始化捕获限制**：初始化捕获（init-capture）不支持 ``__host__ __device__`` 扩展 lambda。

6. **模板参数限制**：扩展 lambda 不能用作模板模板参数的实参。

7. **命名空间限制**：扩展 lambda 不能定义在命名空间作用域内。

8. **constexpr 限制**：扩展 lambda 不能用于 constexpr 上下文。

.. _host-device-lambda-optimization:

5.3.7.5. 主机 - 设备 Lambda 优化注意事项
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

扩展 ``__host__ __device__`` lambda 在主机代码中使用间接函数调用，这可能会阻止内联优化。对于性能关键的代码，考虑使用单独的 ``__device__`` lambda 或 ``__host__`` lambda，而不是 ``__host__ __device__`` lambda。

.. _this-capture-by-value:

5.3.7.6. ``*this`` 按值捕获
^^^^^^^^^^^^^^^^^^^^^^^^^^^

C++17 的 ``*this`` 捕获模式复制对象而不是指针，避免在 GPU 代码中访问成员变量时出现问题：

.. code-block:: c++

   struct S {
       int var = 10;
       void method() {
           auto lambda1 = [=, *this] __device__ {
               return var + 1;  // 按值捕获对象
           };
       }
   };

.. _adl-with-extended-lambdas:

5.3.7.7. 参数依赖查找 (ADL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

扩展 lambda 可能导致额外的命名空间参与 ADL，可能导致函数解析歧义。在使用扩展 lambda 时应注意这一点。

.. _polymorphic-function-wrappers:

5.3.8. 多态函数包装器
---------------------

``nvfunctional`` 头文件提供 ``nvstd::function``，一个多态函数包装器，可在主机和设备代码中使用：

.. code-block:: c++

   #include <nvfunctional>

   __device__ int device_function() { return 10; }

   __global__ void kernel(int* result) {
       nvstd::function<int()> fn1 = device_function;
       nvstd::function<int()> fn2 = [](){ return 10; };
       *result = fn1() + fn2();
   }

**无效情况：**

- 主机 ``nvstd::function`` 不能用 ``__device__`` 函数初始化
- 设备 ``nvstd::function`` 不能用 ``__host__`` 函数初始化
- 不能在运行时在主机和设备之间传递 ``nvstd::function``

.. _c-c-language-restrictions:

5.3.9. C/C++ 语言限制
---------------------

.. _unsupported-features:

5.3.9.1. 不支持的特性
^^^^^^^^^^^^^^^^^^^^^

以下 C++ 特性在设备代码中不支持：

- RTTI（``typeid``、``dynamic_cast``）❌
- 异常（``try/catch/throw``）❌
- ``long double`` ❌
- 三字符组 ❌

.. _namespace-reservations:

5.3.9.2. 命名空间保留
^^^^^^^^^^^^^^^^^^^^^

向 ``cuda::``、``nv::`` 或 ``cooperative_groups::`` 命名空间添加定义是未定义行为。

.. _pointers-and-memory-addresses:

5.3.9.3. 指针和内存地址
^^^^^^^^^^^^^^^^^^^^^^^

- 不能在主机上解引用设备内存指针
- 不能在设备代码中解引用主机内存指针
- 不能在主机代码中获取 ``__device__`` 函数的地址

.. _variables:

5.3.9.4. 变量
^^^^^^^^^^^^^

**局部变量：** 内存空间说明符（``__device__``、``__shared__`` 等）根据执行上下文受到限制。

**``const`` 限定变量：** 设备代码不能引用或获取主机 ``const`` 变量的地址。改用 ``constexpr``。

**``volatile`` 限定变量：** 不适用于线程间同步（使用原子操作）或 MMIO（使用 PTX）。

**``static`` 变量：** 允许在设备代码中使用，但初始化受到限制。

.. _functions:

5.3.9.5. 函数
^^^^^^^^^^^^^

**递归：** ``__global__`` 函数不支持递归；``__device__`` 函数支持。

**``__global__`` 函数参数：**

- 不支持可变参数
- 总大小限制为 32,764 字节
- 不支持按引用传递
- 不支持 ``std::initializer_list`` 参数
- 不支持多态类参数

.. _classes:

5.3.9.6. 类
^^^^^^^^^^^

**多态类：** 在主机/设备之间复制是未定义行为。

**数据成员：** 内存空间说明符不允许用于类数据成员。

**函数成员：** ``__global__`` 函数不能是类成员。

.. _templates:

5.3.9.7. 模板
^^^^^^^^^^^^^

``__global__`` 函数模板参数的类型限制适用（不支持局部、未命名或私有类型）。

.. _c-11-restrictions:

5.3.10. C++11 限制
------------------

.. _inline-namespaces-restrictions:

5.3.10.1. ``inline`` 命名空间
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

不能在内联命名空间和外围命名空间中定义同名的 ``__global__`` 函数或设备变量。

.. _constexpr-functions-restrictions:

5.3.10.3. ``constexpr`` 函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

默认情况下，不能从主机代码调用仅设备的 ``constexpr`` 或从设备代码调用仅主机的 ``constexpr``。使用 ``--expt-relaxed-constexpr`` 放宽此约束。

.. _constexpr-variables:

5.3.10.4. ``constexpr`` 变量
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

可用于设备代码中的标量类型和具有 ``constexpr`` 构造函数的类类型。不允许 ``constexpr __managed__`` 和 ``constexpr __shared__``。

.. _defaulted-functions:

5.3.10.6. 默认函数 ``= default``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

显式默认函数上的执行空间说明符被忽略（除非是外联或虚函数）。

.. _c-14-restrictions:

5.3.11. C++14 限制
------------------

.. _functions-with-deduced-return-type:

5.3.11.1. 推导返回类型的函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``__global__`` 函数不能有推导返回类型 ``auto``
- 返回类型内省不允许在主机代码中使用

.. _variable-templates-restrictions:

5.3.11.2. 变量模板
^^^^^^^^^^^^^^^^^^

``__device__`` 或 ``__constant__`` 变量模板在使用 Microsoft 编译器时不能是 ``const`` 限定的。

.. _c-17-restrictions:

5.3.12. C++17 限制
------------------

.. _inline-variables-restrictions:

5.3.12.1. ``inline`` 变量
^^^^^^^^^^^^^^^^^^^^^^^^^

仅在分离编译模式下或对于具有内部链接的变量允许。

.. _structured-binding-restrictions:

5.3.12.2. 结构化绑定
^^^^^^^^^^^^^^^^^^^^^

不能用内存空间说明符（``__device__``、``__shared__`` 等）声明。

.. _c-20-restrictions:

5.3.13. C++20 限制
------------------

.. _three-way-comparison-operator:

5.3.13.1. 三向比较运算符（``<=>``）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在设备代码中支持，但可能需要 ``--expt-relaxed-constexpr`` 标志和主机实现兼容性。

.. code-block:: c++

   struct S {
       int x, y;
       auto operator<=>(const S&) const = default;
   };

.. _consteval-functions:

5.3.13.2. ``consteval`` 函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

可以从主机和设备代码独立调用，无论其执行空间如何：

.. code-block:: c++

   consteval int host_consteval() { return 10; }

   __device__ int device_function() {
       return host_consteval();  // 正确
   }

.. note::

   有关 C++ 语言支持的详细内容，请参考 `CUDA 官方文档 <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html>`_。