.. _programmatic-dependent-launch-details:

4.5. 程序化依赖启动与同步
================================

程序化依赖启动（Programmatic Dependent Launch， PDL）机制允许在一个 CUDA 流中，一个跟随的 `secondary kernel` 在其依赖的 `primary kernel` 完成执行之前提前启动。
该特性从计算能力 9.0 及以上的硬件设备开始支持；若从核可执行大量不依赖主核计算结果的任务，采用该技术能够获得性能增益。

.. _pdl-background:

4.5.1. 背景
-----------

CUDA 应用程序通过在 GPU 上启动并执行多个 kernel 来利用 GPU 进行计算。下图展示了一个典型的 GPU 活动时间线。

.. _fig-gpu-activity:
.. figure:: /_static/images/gpu-activity.png
   :alt: GPU activity timeline
   :align: center

   GPU 活动时间线

这里， ``secondary_kernel`` 在 ``primary_kernel`` 执行完成后启动。
串行执行通常是必要的，因为 ``secondary_kernel`` 依赖于 ``primary_kernel`` 生成的结果数据。
如果 ``secondary_kernel`` 不依赖 ``primary_kernel`` ，则可以使用 :ref:`CUDA Streams<cuda-streams>` 并发启动。
即使 ``secondary_kernel`` 依赖 ``primary_kernel`` ，也存在一些并发执行的潜力。
例如，几乎所有 kernel 都有某种 *preamble* （前导）部分，在此期间执行诸如清零缓冲区或加载常量值等任务。

.. _fig-secondary-kernel-preamble:
.. figure:: /_static/images/secondary-kernel-preamble.png
   :alt: Preamble section of secondary_kernel
   :align: center

   `secondary_kernel` 的前导部分

:numref:`fig-secondary-kernel-preamble` 展示了 ``secondary_kernel`` 中可并行执行且不会影响应用程序运行的代码段。
请注意，并行启动还允许我们将 ``secondary_kernel`` 的启动延迟隐藏在 ``primary_kernel`` 的执行过程中。

.. _fig-preamble-overlap:
.. figure:: /_static/images/preamble-overlap.png
   :alt: Concurrent execution of primary_kernel and secondary_kernel
   :align: center

   `primary_kernel` 和 `secondary_kernel` 的并行执行

如 :numref:`fig-preamble-overlap` 所示， ``secondary_kernel`` 的并发启动与执行，可以通过 PDL 机制来实现。

PDL 启动对 CUDA kernel 启动接口进行了修改，具体说明见下文。这至少需要计算能力 9.0+ 才能实现重叠执行。


.. _pdl-api-description:

4.5.2. API 描述
---------------

在 PDL 机制中， ``primary_kernel`` 和 ``secondary_kernel``  被提交至同一个 CUDA 流。
当 ``primary_kernel`` 准备就绪、允许 ``secondary_kernel`` 启动时，需要其所有线程块全部执行 ``cudaTriggerProgrammaticLaunchCompletion`` ；
``secondary_kernel`` 则必须采用下文所示的扩展式调度 API 进行提交。

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

      // Will block until all primary kernels the secondary kernel is dependent on 
      // have completed and flushed results to global memory
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

启动 ``secondary_kernel`` 时，需要指定 ``cudaLaunchAttributeProgrammaticStreamSerialization`` 属性。
当执行 ``primary_kernel`` 的所有线程块都已启动并执行 ``cudaTriggerProgrammaticLaunchCompletion`` 时，CUDA 驱动程序可以启动 ``secondary_kernel`` 。
如果 ``primary_kernel`` 没有执行触发器，则在 ``primary_kernel`` 中的所有线程块执行完成后会隐式触发。

无论哪种情况， ``secondary_kernel`` 都可能在 ``primary_kernel`` 写入数据之前就已经启动。
因此，当 ``secondary_kernel`` 使用 PDL 时，它必须始终使用 ``cudaGridDependencySynchronize`` 或其他同步机制，来确保 ``primary_kernel`` 的结果数据已经准备就绪。

请注意， PDL 仅为并行执行提供了可能性，但这个是一种是投机行为，不能保证一定会能并行执行。
并且以这种方式依赖并行执行是不安全的，并可能导致死锁（ 比如 ``secondary_kernel`` 提前执行占用了计算资源； 而 ``primary_kernel`` 执行需要更多的计算资源）。

.. _pdl-use-in-cuda-graphs:

4.5.3. 在 CUDA Graphs 中的使用
------------------------------

PDL 可以通过流捕获或直接通过 :ref:`边数据<cuda-graphs-edge-data>` 的方式，应用于 CUDA Graphs 中。
如果要在 CUDA Graph 中通过边数据来编程实现该特性，需要在连接两个 kernel 节点的边上，
将 ``cudaGraphDependencyType`` 设置为 ``cudaGraphDependencyTypeProgrammatic`` 。
这种边类型会使上游 kernel 对下游 kernel 中的 ``cudaGridDependencySynchronize()`` 可见。
此外，该类型必须配合 ``cudaGraphKernelNodePortLaunchCompletion`` 或 ``cudaGraphKernelNodePortProgrammatic`` 出端口一起使用。

通过流捕获生成的等效图如下

.. list-table:: 流捕获等效项
   :widths: 50 50
   :header-rows: 1

   * - 流捕获（简化）
     - 数据边

   * - .. code-block:: cuda

        cudaLaunchAttribute attribute;
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;

     - .. code-block:: cuda
       
        cudaGraphEdgeData edgeData;
        edgeData.type = cudaGraphDependencyTypeProgrammatic;
        edgeData.from_port = cudaGraphKernelNodePortProgrammatic;

   * - .. code-block:: cuda

        cudaLaunchAttribute attribute;
        attribute.id = cudaLaunchAttributeProgrammaticEvent;
        attribute.val.programmaticEvent.triggerAtBlockStart = 0;

     - .. code-block:: cuda

        cudaGraphEdgeData edgeData;
        edgeData.type = cudaGraphDependencyTypeProgrammatic;
        edgeData.from_port = cudaGraphKernelNodePortProgrammatic;

   * - .. code-block:: cuda

        cudaLaunchAttribute attribute;
        attribute.id = cudaLaunchAttributeProgrammaticEvent;
        attribute.val.programmaticEvent.triggerAtBlockStart = 1;

     - .. code-block:: cuda

        cudaGraphEdgeData edgeData;
        edgeData.type = cudaGraphDependencyTypeProgrammatic;
        edgeData.from_port = cudaGraphKernelNodePortLaunchCompletion;

