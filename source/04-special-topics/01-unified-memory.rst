.. _um-details:

4.1. 统一内存
=============

本节详细解释每种可用统一内存范式的行为和使用方法。:doc:`../02-basics/understanding-memory` 展示了如何确定应用哪种统一内存范式，并简要介绍了每种范式。

如前所述，统一内存编程有四种范式：

- 托管内存分配的完全支持
- 软件一致性的所有分配完全支持
- 硬件一致性的所有分配完全支持
- 有限的统一内存支持

前三种涉及完全统一内存支持的范式具有非常相似的行为和编程模型。

.. note::

   有关统一内存的详细内容，请参考 `CUDA 官方文档 <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html>`_。