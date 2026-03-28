.. _programmatic-dependent-launch-details:

4.5. 程序化依赖启动与同步
================================

*Programmatic Dependent Launch*（程序化依赖启动）机制允许依赖的 *secondary*（次级）kernel 在同一个 CUDA stream 中其所依赖的 *primary*（主）kernel 完成执行之前启动。从计算能力 9.0 的设备开始可用，当 *secondary* kernel 可以完成大量不依赖于 *primary* kernel 结果的工作时，此技术可以提供性能优势。

.. _pdl-background:

4.5.1. 背景
-----------

CUDA 应用程序通过在 GPU 上启动和执行多个 kernel 来利用 GPU。典型的 GPU 活动时间线如 :numref:`fig:gpu-activity` 所示。

.. _fig:gpu-activity:
.. figure:: /_static/images/gpu-activity.png
   :alt: GPU activity timeline
   :align: center

   GPU 活动时间线

这里， `secondary_kernel` 在 `primary_kernel` 完成执行后启动。串行执行通常是必要的，因为 `secondary_kernel` 依赖于 `primary_kernel` 生成的结果数据。如果 `secondary_kernel` 不依赖于 `primary_kernel` ，则可以使用 :doc:`../02-basics/asynchronous-execution` 中描述的 CUDA Streams 并发启动它们。即使 `secondary_kernel` 依赖于 `primary_kernel` ，也存在一些并发执行的潜力。例如，几乎所有 kernel 都有某种 *preamble* （前导）部分，在此期间执行诸如清零缓冲区或加载常量值等任务。

.. _fig:secondary-kernel-preamble:
.. figure:: /_static/images/secondary-kernel-preamble.png
   :alt: Preamble section of secondary_kernel
   :align: center

   `secondary_kernel` 的前导部分

:numref:`fig:secondary-kernel-preamble` 展示了 `secondary_kernel` 中可以并发执行而不影响应用程序的部分。请注意，并发启动还允许我们将 `secondary_kernel` 的启动延迟隐藏在 `primary_kernel` 的执行之后。

.. _fig:preamble-overlap:
.. figure:: /_static/images/preamble-overlap.png
   :alt: Concurrent execution of primary_kernel and secondary_kernel
   :align: center

   `primary_kernel` 和 `secondary_kernel` 的并发执行

:numref:`fig:preamble-overlap` 中显示的 `secondary_kernel` 的并发启动和执行可以使用 *Programmatic Dependent Launch* 来实现。

*Programmatic Dependent Launch* 对 CUDA kernel 启动 API 进行了更改，如下节所述。这些 API 至少需要计算能力 9.0 才能提供重叠执行。

.. _pdl-api-description:

4.5.2. API 描述
---------------

在 Programmatic Dependent Launch 中，primary kernel 和 secondary kernel 在同一个 CUDA stream 中启动。primary kernel 应该在准备好启动 secondary kernel 时，由所有线程块执行 `cudaTriggerProgrammaticLaunchCompletion`。secondary kernel 必须使用可扩展启动 API 启动，如下所示。

.. code-block:: cuda
   :linenos:

   __global__ void primary_kernel() {
      // Initial work that should finish before starting secondary kernel

      // Trigger the secondary kernel
      cudaTriggerProgrammaticLaunchCompletion();

      // Work that can coincide with the secondary kernel
   }

   __global__ void secondary_kernel()
   {
      // Independent work

      // Will block until all primary kernels the secondary kernel is dependent on have completed and flushed results to global memory
      cudaGridDependencySynchronize();

      // Dependent work
   }

   cudaLaunchAttribute attribute[1];
   attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
   attribute[0].val.programmaticStreamSerializationAllowed = 1;
   configSecondary.attrs = attribute;
   configSecondary.numAttrs = 1;

   primary_kernel<<<grid_dim, block_dim, 0, stream>>>();
   cudaLaunchKernelEx(&configSecondary, secondary_kernel);

当使用 `cudaLaunchAttributeProgrammaticStreamSerialization` 属性启动 secondary kernel 时，CUDA 驱动程序可以安全地提前启动 secondary kernel，而不必等待 primary kernel 的完成和内存刷新。

当所有 primary 线程块都已启动并执行 `cudaTriggerProgrammaticLaunchCompletion` 时，CUDA 驱动程序可以启动 secondary kernel。如果 primary kernel 没有执行触发器，则在 primary kernel 中的所有线程块退出后会隐式触发。

无论哪种情况，secondary 线程块可能会在 primary kernel 写入的数据可见之前启动。因此，当 secondary kernel 配置为 *Programmatic Dependent Launch* 时，它必须始终使用 `cudaGridDependencySynchronize` 或其他方式来验证来自 primary 的结果数据是否可用。

请注意，这些方法为 primary kernel 和 secondary kernel 提供了并发执行的机会，但是这种行为是机会性的，不能保证并发 kernel 执行。以这种方式依赖并发执行是不安全的，可能导致死锁。

.. _pdl-use-in-cuda-graphs:

4.5.3. 在 CUDA Graphs 中的使用
------------------------------

Programmatic Dependent Launch 可以通过 :doc:`cuda-graphs` 中的 :ref:`sec:cuda-graphs-creating-a-graph-using-stream-capture` 或直接通过 :ref:`sec:cuda-graphs-edge-data` 在 CUDA Graphs 中使用。要使用边数据在 CUDA Graph 中编程此功能，请在连接两个 kernel 节点的边上使用 `cudaGraphDependencyType` 值 `cudaGraphDependencyTypeProgrammatic`。此边类型使上游 kernel 对下游 kernel 中的 `cudaGridDependencySynchronize()` 可见。此类型必须与 `cudaGraphKernelNodePortLaunchCompletion` 或 `cudaGraphKernelNodePortProgrammatic` 的输出端口一起使用。

流捕获的结果图等效项如下：

.. list-table:: 流捕获等效项
   :widths: 50 50
   :header-rows: 1

   * - 流代码（缩略）
     - 结果图边

   * - .. code-block:: cuda

        cudaLaunchAttribute attribute;
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;

        cudaGraphEdgeData edgeData;
        edgeData.type = cudaGraphDependencyTypeProgrammatic;
        edgeData.from_port = cudaGraphKernelNodePortProgrammatic;

     - 

   * - .. code-block:: cuda

        cudaLaunchAttribute attribute;
        attribute.id = cudaLaunchAttributeProgrammaticEvent;
        attribute.val.programmaticEvent.triggerAtBlockStart = 0;

        cudaGraphEdgeData edgeData;
        edgeData.type = cudaGraphDependencyTypeProgrammatic;
        edgeData.from_port = cudaGraphKernelNodePortProgrammatic;

     - 

   * - .. code-block:: cuda

        cudaLaunchAttribute attribute;
        attribute.id = cudaLaunchAttributeProgrammaticEvent;
        attribute.val.programmaticEvent.triggerAtBlockStart = 1;

        cudaGraphEdgeData edgeData;
        edgeData.type = cudaGraphDependencyTypeProgrammatic;
        edgeData.from_port = cudaGraphKernelNodePortLaunchCompletion;

     -