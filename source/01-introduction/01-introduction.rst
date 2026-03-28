.. _introduction:

1.1. 简介
=========

.. _the-graphics-processing-unit:

1.1.1. 图形处理器
-----------------

*图形处理器* (GPU) 最初是为 3D 图形设计的专用处理器，作为固定功能硬件来加速实时 3D 渲染中的并行操作。经过几代的发展，GPU 变得更加可编程。到 2003 年，图形流水线的某些阶段变得完全可编程，可以为 3D 场景或图像的每个组件并行运行自定义代码。

2006 年，NVIDIA 引入了 *统一计算设备架构* (CUDA)，使任何计算工作负载都能独立于图形 API 使用 GPU 的吞吐能力。

从那时起，CUDA 和 GPU 计算已被用于加速几乎每种类型的计算工作负载，从流体动力学或能量传输等科学模拟到数据库和分析等商业应用。此外，GPU 的能力和可编程性已成为从图像分类到扩散模型或大语言模型等生成式人工智能等新算法和技术进步的基础。

.. _the-benefits-of-using-gpus:

1.1.2. 使用 GPU 的好处
----------------------

GPU 在相似的价格和功耗范围内提供比 CPU 更高的指令吞吐量和内存带宽。许多应用程序利用这些能力在 GPU 上比在 CPU 上运行得更快（参见 `GPU 应用程序 <https://www.nvidia.com/en-us/accelerated-applications/>`_）。其他计算设备，如 FPGA，也非常节能，但提供的编程灵活性远低于 GPU。

GPU 和 CPU 的设计目标不同。CPU 旨在尽可能快地执行串行操作序列（称为线程），并且可以并行执行几十个这样的线程，而 GPU 旨在并行执行数千个线程，牺牲较低的单线程性能以实现更高的总吞吐量。

GPU 专门用于高度并行计算，并将更多晶体管用于数据处理单元，而 CPU 将更多晶体管用于数据缓存和流控制。:numref:`fig-gpu-devotes-more-transistors` 显示了 CPU 与 GPU 的芯片资源分配示例。

.. _fig-gpu-devotes-more-transistors:
.. figure:: /_static/images/gpu-devotes-more-transistors-to-data-processing.png
   :alt: GPU 将更多晶体管用于数据处理
   :width: 80%

   GPU 将更多晶体管用于数据处理

.. _getting-started-quickly:

1.1.3. 快速入门
---------------

有多种方法可以利用 GPU 提供的计算能力。本指南涵盖了使用 C++ 等高级语言为 CUDA GPU 平台编程。但是，有许多方法可以在不需要直接编写 GPU 代码的情况下在应用程序中使用 GPU。

来自不同领域的算法和例程的不断增加的集合可以通过专门的库获得。当库已经实现时——尤其是 NVIDIA 提供的库——使用它通常比从头重新实现算法更具生产力和性能。cuBLAS、cuFFT、cuDNN 和 CUTLASS 等库只是帮助开发者避免重新实现已确立算法的几个例子。这些库还有一个好处，即为每个 GPU 架构进行了优化，提供了生产力、性能和可移植性的理想组合。

还有框架，特别是用于人工智能的框架，提供了 GPU 加速的构建块。许多这些框架通过利用上述 GPU 加速库来实现加速。

此外，特定领域语言 (DSL) 如 NVIDIA 的 Warp 或 OpenAI 的 Triton 可以编译为直接在 CUDA 平台上运行。这提供了比本指南中介绍的高级语言更高级的 GPU 编程方法。

`NVIDIA 加速计算中心 <https://github.com/NVIDIA/accelerated-computing-hub>`_ 包含教授 GPU 和 CUDA 计算的资源、示例和教程。