.. _cuda-graphs:

4.2. CUDA Graphs
================

CUDA 图（CUDA Graphs）提供了另一种 CUDA 任务提交模型。
一张图由一系列存在依赖关系的操作（如核函数启动、数据传输等）构成，其定义过程与执行过程相互分离。
这意味着一张图只需定义一次，便可反复提交执行。将图的定义与执行拆分开，能够实现多项性能优化：

1. 与流相比，CPU 侧的任务提交开销大幅降低，绝大多数配置工作都可提前完成；
2. 向 CUDA 暴露完整的工作流，可以实现一些使用流的分段任务提交机制无法达成的优化。

想要理解 CUDA 图所能带来的优化，我们先看 CUDA 流的执行过程：
当你向流中提交一个 kernel 时，主机驱动会执行一系列操作，为该 kernel 在 GPU 上运行做准备。
这些用于配置并启动 kernel 的操作属于额外开销，每提交一次 kernel 就要承担一次开销。
如果某个 GPU kernel 本身执行耗时很短，这份启动开销可能是整体端到端执行时间的很大一部分。
而如果把需要反复执行的完整工作流封装成一张 CUDA 图，所有启动开销仅会在图实例化阶段支付一次；后续重复提交这张图时，仅会产生极低的额外开销。

.. _cuda-graphs-graph-structure:

4.2.1. 图结构
-------------

各类操作构成图中的节点（ node ），操作之间的依赖关系为图的边（edge），这些依赖约束了操作的执行顺序。

一旦某个操作所依赖的全部节点执行完成，该操作即可随时被调度执行。调度由 CUDA 系统负责。

.. _cuda-graphs-node-types:

4.2.1.1. 节点类型
~~~~~~~~~~~~~~~~~

图节点可以是以下类型之一：

- kernel
- CPU function call
- memory copy
- memset
- empty node
- waiting on a :ref:`CUDA Event<cuda-events>`
- recording a :ref:`CUDA Event<cuda-events>`
- signalling an :ref:`external semaphore<external-resource-interoperability>`
- waiting on an :ref:`external semaphore<external-resource-interoperability>`
- :ref:`conditional node<cuda-graphs-conditional-nodes>`
- :ref:`memory node<cuda-graphs-memory-nodes>`
- 子图（child graph）：执行单独的嵌套图，如下图所示。

.. figure:: /_static/images/child-graph.png
   :alt: 子图示例
   :align: center

   子图示例

.. _cuda-graphs-edge-data:

4.2.1.2. 边数据
~~~~~~~~~~~~~~~

CUDA 12.3 为 CUDA 图新增了边数据功能（edge data）。
目前，非默认边数据仅有一种用途：:ref:`启用可编程依赖启动（Programmatic Dependent Launch）<programmatic-dependent-launch-details>`。

总体而言，边数据用于修改一条边所定义的依赖关系，由三部分组成：输出端口、输入端口与类型。
输出端口用于指定对应边的触发时机；
输入端口用于指定一个节点内哪一部分逻辑依赖该边；
类型则用于修改两端节点之间的依赖关联规则。

端口取值由节点类型与数据传输方向决定，且边类型仅能搭配特定节点类型使用。所有场景下，全部字段置零初始化的边数据代表默认行为：
输出端口 0 表示等待整个任务执行完毕；
输入端口 0 表示阻塞整个任务；
边类型 0 对应完整依赖关系，附带内存同步语义。

作为可选参数，各类图相关 API 可通过一组与对应节点并行的数组，按需传入边数据。
若作为入参时省略该数组，则自动使用全零初始化的默认边数据；
若作为出参（查询参数）时省略该数组：
当被忽略的边数据全部为默认值（零初始化值时），API 调用正常执行；
否则接口返回 ``cudaErrorLossyQuery`` 错误码（即边数据非默认值，但是调用者没有提供出参，本次查询会丢失有效边数据信息）。

部分流捕获 API 同样支持边数据，如 ``cudaStreamBeginCaptureToGraph()`` 、 ``cudaStreamGetCaptureInfo()`` 、 ``cudaStreamUpdateCaptureDependencies()`` 。
这类场景下尚不存在下游节点，边数据会绑定在 **悬空边（半边）** 上：该悬空边后续要么接入捕获过程中新生成的节点，要么在流捕获结束时被丢弃。
请注意：部分边类型无需等待上游节点完全执行完毕。在判断流捕获是否完全回连至原始流时，这类边不会纳入考量，且捕获流程结束后也无法被丢弃。
详情参考 :ref:`cuda-graphs-stream-capture` 。

所有节点类型均未定义额外输入端口；仅有 kernel 节点支持自定义输出端口。
目前仅支持一种非默认依赖类型即 ``cudaGraphDependencyTypeProgrammatic`` ，用于在两个 kernel 节点之间启用 :ref:`可编程依赖启动<programmatic-dependent-launch-details>`。

.. _cuda-graphs-build-and-run:

4.2.2. 构建和运行图
-------------------

使用计算图提交任务分为三个独立阶段：定义、实例化、执行。

- 在定义（创建）阶段，程序会构建图中所有操作及其相互依赖关系的描述信息。
- 实例化操作会对计算图模板生成快照、完成合法性校验，并预先执行大部分任务配置与初始化工作，目的是尽可能减少后续启动时的处理步骤。
  经过该流程生成的产物称为可执行图（executable graph）。
- 可执行图可像其他任意 CUDA 任务一样提交至流中执行，无需重复执行实例化操作，能够不限次数反复启动。

.. _cuda-graphs-graph-creation:

4.2.2.1. 图创建
~~~~~~~~~~~~~~~

计算图有两种创建方式：显式调用图专用 API，以及通过流捕获生成。

.. _cuda-graphs-graph-api:

4.2.2.1.1. 图 API
`````````````````

以下是一个创建图的示例代码（省略变量声明与其他标准化模板代码）。
示例中通过 ``cudaGraphCreate()`` 创建图对象，并调用 ``cudaGraphAddNode()`` 添加节点与节点间依赖关系。
`CUDA 运行时 API 文档 <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html>`_ 中列出了所有可用于新增节点和依赖关系的接口函数。

.. _fig-cuda-graphs-create-a-graph:
.. figure:: /_static/images/create-a-graph.png
   :alt: 使用图 API 创建图的示例
   :align: center

   使用图 API 创建图的示例

.. code-block:: cpp

   // Create the graph - it starts out empty
   cudaGraphCreate(&graph, 0);

   // Create the nodes and their dependencies
   cudaGraphNode_t nodes[4];
   cudaGraphNodeParams kParams = { cudaGraphNodeTypeKernel };
   kParams.kernel.func         = (void *)kernelName;
   kParams.kernel.gridDim.x    = kParams.kernel.gridDim.y  = kParams.kernel.gridDim.z  = 1;
   kParams.kernel.blockDim.x   = kParams.kernel.blockDim.y = kParams.kernel.blockDim.z = 1;

   cudaGraphAddNode(&nodes[0], graph, NULL, NULL, 0, &kParams);
   cudaGraphAddNode(&nodes[1], graph, &nodes[0], NULL, 1, &kParams);
   cudaGraphAddNode(&nodes[2], graph, &nodes[0], NULL, 1, &kParams);
   cudaGraphAddNode(&nodes[3], graph, &nodes[1], NULL, 2, &kParams);

上面的示例包含四个 kernel 节点，并配置了节点间的依赖关系，用于演示如何创建极简计算图。
在常规业务程序中，还需要添加各类内存操作节点，例如 ``cudaGraphAddMemcpyNode()`` 等接口。
如需查阅全部用于创建节点的计算图 API，可参考 `CUDA 运行时 API 文档 <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html>`_ 。

.. _cuda-graphs-stream-capture:

4.2.2.1.2. 流捕获
`````````````````

流捕获（Stream capture）提供了一种基于现有流 API 生成计算图的机制。
一段向流提交任务的代码（包括已有业务代码），可以用 ``cudaStreamBeginCapture()`` 和 ``cudaStreamEndCapture()`` 两个函数包裹起来，详见下文示例。

.. code-block:: cpp

   cudaGraph_t graph;

   cudaStreamBeginCapture(stream);

   kernel_A<<< ..., stream >>>(...);
   kernel_B<<< ..., stream >>>(...);
   libraryCall(stream);
   kernel_C<<< ..., stream >>>(...);

   cudaStreamEndCapture(stream, &graph);

调用 ``cudaStreamBeginCapture()`` 会将流切换为捕获模式。
当流处于捕获状态时，提交至该流的任务不会入队等待执行，而是追加到一个内部计算图中，该图会逐步构建完善。
随后调用 ``cudaStreamEndCapture()`` 可返回这张完整的计算图，同时结束流的捕获模式。
正在通过流捕获流程构建的计算图，称为捕获图（capture graph）。

流捕获可用于任意 CUDA 流，但传统默认流 ``cudaStreamLegacy`` （又称 ``NULL`` 流）除外。
注意该功能支持线程私有流 ``cudaStreamPerThread`` 。
如果程序正在使用传统默认流，可以将 0 号流重新设置为线程私有流，且不会改变原有业务逻辑行为。
更多细节参见 :ref:`async-execution-blocking-non-blocking-default-stream` 。

可通过 ``cudaStreamIsCapturing()`` 查询某个流是否处于捕获状态。

可调用 ``cudaStreamBeginCaptureToGraph()`` 将任务捕获至一张已存在的计算图。
该接口不会捕获到系统内部临时图，而是把所有任务录制到用户传入的目标图中。

.. _cuda-graphs-cross-stream-dependencies-and-events:

4.2.2.1.2.1. 跨流依赖和事件
***************************

流捕获机制能够处理通过 ``cudaEventRecord()`` 与 ``cudaStreamWaitEvent()`` 表达的跨流依赖关系，前提是被等待的事件已记录至同一张捕获图中。

若在处于捕获模式的流中记录事件，会生成捕获事件（ `captured event` ）。
捕获事件对应捕获图内的一组节点。

若某一流等待一个捕获事件，如果该流尚未开启捕获模式，则会自动将其切换至捕获模式；
该流的下一个任务会新增依赖，依赖于该捕获事件所对应的图节点。
此时两条流会被捕获到同一张捕获图中。

流捕获中存在跨流依赖时， ``cudaStreamEndCapture()`` 仍必须在调用 ``cudaStreamBeginCapture()`` 的同一条流中执行，该流称为初始流（ `origin stream` ）。
因事件依赖关系而被捕获至同一张捕获图的其他所有流，都必须回流合并至初始流，下文会对此进行说明。
调用 ``cudaStreamEndCapture()`` 后，所有捕获至该捕获图的流都会退出捕获模式。
若未将其他流合并回初始流，整个捕获操作将会失败。

.. code-block:: cpp

   // stream1 is the origin stream
   cudaStreamBeginCapture(stream1);

   kernel_A<<< ..., stream1 >>>(...);

   // Fork into stream2
   cudaEventRecord(event1, stream1);
   cudaStreamWaitEvent(stream2, event1);

   kernel_B<<< ..., stream1 >>>(...);
   kernel_C<<< ..., stream2 >>>(...);

   // Join stream2 back to origin stream (stream1)
   cudaEventRecord(event2, stream2);
   cudaStreamWaitEvent(stream1, event2);

   kernel_D<<< ..., stream1 >>>(...);

   // End capture in the origin stream
   cudaStreamEndCapture(stream1, &graph);

   // stream1 and stream2 no longer in capture mode

上述代码返回的图如图 :numref:`fig-cuda-graphs-create-a-graph` 所示。

.. note::
   当一条流退出捕获模式后，该流中后续首个非捕获任务（若存在），仍会依赖于上一条最近的非捕获任务，即便二者中间的任务已被剥离至捕获图内。

.. _cuda-graphs-prohibited-and-unhandled-operations:

4.2.2.1.2.2. 禁止和未处理的操作
*******************************

对处于捕获过程的流或捕获事件执行同步、查询执行状态均属于非法操作，因为它们并非已调度、待执行的任务单元。
若某个设备句柄、上下文句柄关联的任意一条流正处于捕获模式，这类包含了正在进行流捕获的高层句柄，同样不允许同步或查询其执行状态。

同一上下文内若存在任意一条处于捕获模式、且创建时未指定 ``cudaStreamNonBlocking`` 的流，则任何尝试使用传统默认流的操作均为非法操作。
原因在于传统默认流句柄始终关联该上下文下其他所有普通流；向传统默认流提交任务会与正在捕获的流生成依赖关系，而对传统默认流执行状态查询或同步操作，等价于对正在捕获的流执行查询、同步。

因此，在该场景下调用具有同步属性的 API 同样属于非法操作。
``cudaMemcpy()`` 就是同步 API 的典型示例：它会将任务提交至传统默认流，并在函数返回前执行同步等待。

.. note::

   若某条依赖关系一边是捕获阶段记录的任务、另一边是直接入队待执行的常规任务，CUDA 会直接返回错误。
   有一个例外：对流开启或退出捕获模式时，模式切换前后添加到该流的任务之间的依赖关系会被切断。

若从一个正在被捕获流上等待另一个不同的捕获图中的捕获事件，试图通过这种方式合并这两张不同的捕获图的操作属于非法行为。
当流处于捕获模式时等待非捕获事件，且未传入 ``cudaEventWaitExternal`` 标志，该操作同样非法。

少数用于向流中提交异步操作的 API 当前不支持在图捕获场景使用；若在处于捕获模式的流上调用这类接口，会直接返回错误， ``cudaStreamAttachMemAsync()`` 便是其中一例。


.. _cuda-graphs-invalidation:

4.2.2.1.2.3. 失效
*****************

在流捕获过程中若执行非法操作，相关的捕获图就会失效。
捕获图一旦失效，继续使用该图对应的、处于捕获状态的流或捕获事件均属于非法操作，调用会返回错误；该状态持续到调用 ``cudaStreamEndCapture()`` 结束流捕获为止。
该接口调用会将关联的流退出捕获模式，同时会返回错误码，并输出空图对象（ ``NULL graph`` ）。


.. _cuda-graphs-capture-introspection:

4.2.2.1.2.4. 捕获探查
*********************

可通过 ``cudaStreamGetCaptureInfo()`` 查询正在进行的流捕获操作。
该接口能够获取捕获状态、当前捕获流程的进程内唯一标识、底层图对象，以及流中下一个待捕获节点的依赖与边信息。
借助这份依赖信息，开发者可以拿到该流中最后捕获生成的节点句柄。

.. _cuda-graphs-comprehensive-example:

4.2.2.1.3. 综合示例
```````````````````

:numref:`fig-cuda-graphs-create-a-graph` 是一个简化示例，用于直观展示小型 CUDA 图的基础概念。
在实际使用 CUDA 图的应用程序里，无论是直接调用图原生 API，还是采用流捕获方式构建图，实际使用逻辑都会复杂得多。
下方代码片段并列对比了图原生 API与流捕获两种方式，二者均用于创建 CUDA 图，实现简单的两段式归约算法。


:numref:`fig-cuda-graphs-reduction` 为该 CUDA 计算图的示意图。
该图通过对下述代码调用 ``cudaGraphDebugDotPrint`` 函数生成，为便于阅读做了少量调整，最后经由 `Graphviz <https://graphviz.org/>`_ 渲染输出。

.. _fig-cuda-graphs-reduction:
.. figure:: /_static/images/cuda_graph_reduction.png
   :alt: 使用两阶段归约 kernel 的 CUDA 图示例
   :align: center

   使用两阶段归约 kernel 的 CUDA 图示例


.. tab-set::

   .. tab-item:: 图 API

      .. code-block:: cpp

         void cudaGraphsManual(float  *inputVec_h,
                               float  *inputVec_d,
                               double *outputVec_d,
                               double *result_d,
                               size_t  inputSize,
                               size_t  numOfBlocks)
         {
            cudaStream_t                 streamForGraph;
            cudaGraph_t                  graph;
            std::vector<cudaGraphNode_t> nodeDependencies;
            cudaGraphNode_t              memcpyNode, kernelNode, memsetNode;
            double                       result_h = 0.0;

            cudaStreamCreate(&streamForGraph);

            cudaKernelNodeParams kernelNodeParams = {0};
            cudaMemcpy3DParms    memcpyParams     = {0};
            cudaMemsetParams     memsetParams     = {0};

            memcpyParams.srcArray = NULL;
            memcpyParams.srcPos   = make_cudaPos(0, 0, 0);
            memcpyParams.srcPtr   = make_cudaPitchedPtr(inputVec_h, sizeof(float) * inputSize, inputSize, 1);
            memcpyParams.dstArray = NULL;
            memcpyParams.dstPos   = make_cudaPos(0, 0, 0);
            memcpyParams.dstPtr   = make_cudaPitchedPtr(inputVec_d, sizeof(float) * inputSize, inputSize, 1);
            memcpyParams.extent   = make_cudaExtent(sizeof(float) * inputSize, 1, 1);
            memcpyParams.kind     = cudaMemcpyHostToDevice;

            memsetParams.dst         = (void *)outputVec_d;
            memsetParams.value       = 0;
            memsetParams.pitch       = 0;
            memsetParams.elementSize = sizeof(float); // elementSize can be max 4 bytes
            memsetParams.width       = numOfBlocks * 2;
            memsetParams.height      = 1;

            cudaGraphCreate(&graph, 0);
            cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams);
            cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams);

            nodeDependencies.push_back(memsetNode);
            nodeDependencies.push_back(memcpyNode);

            void *kernelArgs[4] = {(void *)&inputVec_d, (void *)&outputVec_d, &inputSize, &numOfBlocks};

            kernelNodeParams.func           = (void *)reduce;
            kernelNodeParams.gridDim        = dim3(numOfBlocks, 1, 1);
            kernelNodeParams.blockDim       = dim3(THREADS_PER_BLOCK, 1, 1);
            kernelNodeParams.sharedMemBytes = 0;
            kernelNodeParams.kernelParams   = (void **)kernelArgs;
            kernelNodeParams.extra          = NULL;

            cudaGraphAddKernelNode(
               &kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams);

            nodeDependencies.clear();
            nodeDependencies.push_back(kernelNode);

            memset(&memsetParams, 0, sizeof(memsetParams));
            memsetParams.dst         = result_d;
            memsetParams.value       = 0;
            memsetParams.elementSize = sizeof(float);
            memsetParams.width       = 2;
            memsetParams.height      = 1;
            cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams);

            nodeDependencies.push_back(memsetNode);

            memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
            kernelNodeParams.func           = (void *)reduceFinal;
            kernelNodeParams.gridDim        = dim3(1, 1, 1);
            kernelNodeParams.blockDim       = dim3(THREADS_PER_BLOCK, 1, 1);
            kernelNodeParams.sharedMemBytes = 0;
            void *kernelArgs2[3]            = {(void *)&outputVec_d, (void *)&result_d, &numOfBlocks};
            kernelNodeParams.kernelParams   = kernelArgs2;
            kernelNodeParams.extra          = NULL;

            cudaGraphAddKernelNode(
               &kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams);

            nodeDependencies.clear();
            nodeDependencies.push_back(kernelNode);

            memset(&memcpyParams, 0, sizeof(memcpyParams));

            memcpyParams.srcArray = NULL;
            memcpyParams.srcPos   = make_cudaPos(0, 0, 0);
            memcpyParams.srcPtr   = make_cudaPitchedPtr(result_d, sizeof(double), 1, 1);
            memcpyParams.dstArray = NULL;
            memcpyParams.dstPos   = make_cudaPos(0, 0, 0);
            memcpyParams.dstPtr   = make_cudaPitchedPtr(&result_h, sizeof(double), 1, 1);
            memcpyParams.extent   = make_cudaExtent(sizeof(double), 1, 1);
            memcpyParams.kind     = cudaMemcpyDeviceToHost;

            cudaGraphAddMemcpyNode(
               &memcpyNode, graph, nodeDependencies.data(), nodeDependencies.size(), &memcpyParams);
            nodeDependencies.clear();
            nodeDependencies.push_back(memcpyNode);

            cudaGraphNode_t    hostNode;
            cudaHostNodeParams hostParams = {0};
            hostParams.fn                 = myHostNodeCallback;
            callBackData_t hostFnData;
            hostFnData.data     = &result_h;
            hostFnData.fn_name  = "cudaGraphsManual";
            hostParams.userData = &hostFnData;

            cudaGraphAddHostNode(
               &hostNode, graph, nodeDependencies.data(), nodeDependencies.size(), &hostParams);
         }

   .. tab-item:: 流捕获

      .. code-block:: cpp

         void cudaGraphsUsingStreamCapture(float  *inputVec_h,
                                           float  *inputVec_d,
                                           double *outputVec_d,
                                           double *result_d,
                                           size_t  inputSize,
                                           size_t  numOfBlocks)
         {
            cudaStream_t stream1, stream2, stream3, streamForGraph;
            cudaEvent_t  forkStreamEvent, memsetEvent1, memsetEvent2;
            cudaGraph_t  graph;
            double       result_h = 0.0;

            cudaStreamCreate(&stream1);
            cudaStreamCreate(&stream2);
            cudaStreamCreate(&stream3);
            cudaStreamCreate(&streamForGraph);

            cudaEventCreate(&forkStreamEvent);
            cudaEventCreate(&memsetEvent1);
            cudaEventCreate(&memsetEvent2);

            cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

            cudaEventRecord(forkStreamEvent, stream1);
            cudaStreamWaitEvent(stream2, forkStreamEvent, 0);
            cudaStreamWaitEvent(stream3, forkStreamEvent, 0);

            cudaMemcpyAsync(inputVec_d, inputVec_h, sizeof(float) * inputSize, cudaMemcpyDefault, stream1);

            cudaMemsetAsync(outputVec_d, 0, sizeof(double) * numOfBlocks, stream2);

            cudaEventRecord(memsetEvent1, stream2);

            cudaMemsetAsync(result_d, 0, sizeof(double), stream3);
            cudaEventRecord(memsetEvent2, stream3);

            cudaStreamWaitEvent(stream1, memsetEvent1, 0);

            reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0, stream1>>>(inputVec_d, outputVec_d, inputSize, numOfBlocks);

            cudaStreamWaitEvent(stream1, memsetEvent2, 0);

            reduceFinal<<<1, THREADS_PER_BLOCK, 0, stream1>>>(outputVec_d, result_d, numOfBlocks);
            cudaMemcpyAsync(&result_h, result_d, sizeof(double), cudaMemcpyDefault, stream1);

            callBackData_t hostFnData = {0};
            hostFnData.data           = &result_h;
            hostFnData.fn_name        = "cudaGraphsUsingStreamCapture";
            cudaHostFn_t fn           = myHostNodeCallback;
            cudaLaunchHostFunc(stream1, fn, &hostFnData);
            cudaStreamEndCapture(stream1, &graph);
         }

.. _cuda-graphs-graph-instantiation:

4.2.2.2. 图实例化
~~~~~~~~~~~~~~~~~

无论通过图原生 API 还是流捕获方式创建图对象后，都必须对该图执行实例化操作，生成可执行图之后才能运行。
假设 ``cudaGraph_t`` 类型的图对象已创建成功，下方代码将完成图的实例化，并生成可执行图对象 ``cudaGraphExec_t graphExec`` ：

.. code-block:: cpp

   cudaGraphExec_t graphExec;
   cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

.. _cuda-graphs-graph-execution:

4.2.2.3. 图执行
~~~~~~~~~~~~~~~

创建图并实例化得到可执行图之后，便可提交执行。
假设可执行图对象 ``cudaGraphExec_t graphExec`` 创已建成功，以下代码片段会将该图提交至指定流运行：

.. code-block:: cuda

   cudaGraphLaunch(graphExec, stream);

结合起来，使用 :ref:`cuda-graphs-stream-capture` 节中的流捕获示例，以下代码片段将创建图、实例化并启动它：

.. code-block:: cuda

   cudaGraph_t graph;

   cudaStreamBeginCapture(stream);

   kernel_A<<< ..., stream >>>(...);
   kernel_B<<< ..., stream >>>(...);
   libraryCall(stream);
   kernel_C<<< ..., stream >>>(...);

   cudaStreamEndCapture(stream, &graph);

   cudaGraphExec_t graphExec;
   cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
   cudaGraphLaunch(graphExec, stream);

.. _cuda-graphs-updating-instantiated:

4.2.3. 更新实例化的图
---------------------

当业务流程发生变动时，原图就会过时，必须进行修改。
若图结构发生大幅变更（例如拓扑结构、节点类型改动），则需要重新实例化，因为和拓扑相关的优化策略需要重新生成。
但多数场景下，仅节点参数（如核函数入参、内存地址）发生变化，图拓扑保持不变。
针对这种场景，CUDA 提供轻量化的图更新机制，支持就地修改指定节点参数，无需重建整张图，效率远高于重新实例化。

参数更新会在下一次提交执行该图时生效，不会影响此前已经发起的图执行任务，即便更新操作执行时这些旧任务仍在运行。
同一张可执行图支持反复更新参数并重新提交运行，因此可以在同一个流上连续排队执行多次更新与图启动操作。

CUDA 提供两种更新已实例化图参数的机制：整图更新与单节点更新。
整图更新允许开发者传入一张拓扑结构完全一致的静态图对象 ``cudaGraph_t`` ，该图内各节点已配置更新后的参数。
单节点更新则支持开发者显式修改单个节点的参数。
当需要修改大量节点，或是调用方不清楚图拓扑（例如通过库函数流捕获生成的图）时，使用更新后的 ``cudaGraph_t`` 做整图更新会更便捷。
若修改项较少，且开发者持有待更新节点的句柄，则优先选用单节点更新。
单节点更新会跳过未改动节点的拓扑校验与比对流程，多数场景下性能更高。

CUDA 还提供一种机制，可单独启用或禁用指定节点，且不会改动节点当前已设置的参数。

下文各小节将对每种方案展开详细说明。

.. _cuda-graphs-whole-graph-update:

4.2.3.1. 整图更新
~~~~~~~~~~~~~~~~~

``cudaGraphExecUpdate()`` 函数可以利用一张拓扑完全相同的图（称为更新图）的参数，对已实例化完成的可执行图（即原始图）进行参数更新。
更新图的拓扑结构必须与当初用来实例化 ``cudaGraphExec_t`` 的原始静态图完全一致；除此之外，两处图中声明依赖关系的顺序也必须一一对应。
最后，CUDA 要求汇点节点（不存在后继依赖的末端节点）拥有稳定、统一的排序规则。CUDA 会依靠各类接口的调用顺序，保证汇点节点排序固定不变。

更明确地说，遵循下述规则， ``cudaGraphExecUpdate()`` 就能以确定的方式完成原始图与更新图之间节点的一一匹配：

1. 对于任意捕获流，所有作用于该流的 API 调用顺序必须保持完全一致，其中包含事件等待以及其他并非直接创建图节点的 API。
2. 所有用于直接操作指定图节点入边的 API 调用（包括流捕获过程中的流接口、节点创建、增删依赖边），调用顺序必须完全一致。
   此外，如果这类接口通过数组批量传入依赖关系，数组内部各依赖项的排列顺序也必须一一对应。
3. 汇点节点的排序必须保持一致。汇点节点指调用 ``cudaGraphExecUpdate()`` 时，最终图中不存在后继依赖节点、也无出边的节点。
   以下操作会影响汇点节点的排序（若存在汇点），且这一系列操作整体调用顺序必须完全相同：

   - 会生成汇点节点的节点创建接口。
   - 删除依赖边后导致某个节点变为汇点节点的操作。
   - ``cudaStreamUpdateCaptureDependencies()`` 接口（当该操作会从捕获流的依赖集合中移除某个汇点节点时）
   - ``cudaStreamEndCapture()`` 。

以下示例展示了如何使用 API 更新实例化的图：

.. code-block:: cpp

   cudaGraphExec_t graphExec = NULL;

   for (int i = 0; i < 10; i++) {
       cudaGraph_t graph;
       cudaGraphExecUpdateResult updateResult;
       cudaGraphNode_t errorNode;

       // In this example we use stream capture to create the graph.
       // You can also use the Graph API to produce a graph.
       cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

       // Call a user-defined, stream based workload, for example
       do_cuda_work(stream);

       cudaStreamEndCapture(stream, &graph);

       // If we've already instantiated the graph, try to update it directly
       // and avoid the instantiation overhead
       if (graphExec != NULL) {
           // If the graph fails to update, errorNode will be set to the
           // node causing the failure and updateResult will be set to a
           // reason code.
           cudaGraphExecUpdate(graphExec, graph, &errorNode, &updateResult);
       }

       // Instantiate during the first iteration or whenever the update
       // fails for any reason
       if (graphExec == NULL || updateResult != cudaGraphExecUpdateSuccess) {

           // If a previous update failed, destroy the cudaGraphExec_t
           // before re-instantiating it
           if (graphExec != NULL) {
               cudaGraphExecDestroy(graphExec);
           }
           // Instantiate graphExec from graph. The error node and
           // error message parameters are unused here.
           cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
       }

       cudaGraphDestroy(graph);
       cudaGraphLaunch(graphExec, stream);
       cudaStreamSynchronize(stream);
   }

标准执行流程如下：通过流捕获或图原生接口创建初始静态图 ``cudaGraph_t`` ，随后对该静态图执行实例化，并按常规方式提交运行。
首次执行完成后，采用与创建初始图完全相同的方式构建一张全新的静态图，再调用 ``cudaGraphExecUpdate()`` 执行整图更新。
若更新成功（可通过上例中的 ``updateResult`` 参数判断），即可运行更新后的可执行图 ``cudaGraphExec_t`` 。
倘若更新因任意原因失败，则依次调用 ``cudaGraphExecDestroy()`` 、 ``cudaGraphInstantiate()`` ，销毁原有可执行图并重新实例化生成新的可执行图。

也可以直接修改静态图 ``cudaGraph_t`` 中的节点参数（如调用 ``cudaGraphKernelNodeSetParams()`` ），再基于修改后的静态图去更新可执行图 ``cudaGraphExec_t`` ；
但相比之下，使用下一节介绍的单节点更新接口效率更高。

条件句柄标识与默认值会随同图更新操作一并完成更新。

如需了解更多使用方法与现有约束限制，请查阅 `CUDA 图相关 API <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH>`_ 文档。

.. _cuda-graphs-single-node-update:

4.2.3.2. 单节点更新
~~~~~~~~~~~~~~~~~~~~~

实例化的图节点参数可以直接更新。这消除了实例化的开销以及创建新 cudaGraph_t 的开销。如果需要更新的节点数量相对于图中的节点总数较小，则最好单独更新节点。以下方法可用于更新 cudaGraphExec_t 节点：

可直接修改已实例化图的节点参数。
该方式省去了实例化的开销，也无需新建静态图。若需要更新的节点数量远少于整张图的总节点数，优先采用单节点更新方式。
以下接口可用于更新可执行图的节点参数：

.. _table-cuda-graphs-single-node-update:

.. table:: 单节点更新 API

   =========================================================== ============
   接口                                                         类型
   =========================================================== ============
   ``cudaGraphExecKernelNodeSetParams()``                      kernel 节点
   ``cudaGraphExecMemcpyNodeSetParams()``                      内存拷贝节点
   ``cudaGraphExecMemsetNodeSetParams()``                      内存填充节点
   ``cudaGraphExecHostNodeSetParams()``                        主机节点
   ``cudaGraphExecChildGraphNodeSetParams()``                  子图节点
   ``cudaGraphExecEventRecordNodeSetEvent()``                  事件记录节点
   ``cudaGraphExecEventWaitNodeSetEvent()``                    事件等待节点
   ``cudaGraphExecExternalSemaphoresSignalNodeSetParams()``    外部信号量发出节点
   ``cudaGraphExecExternalSemaphoresWaitNodeSetParams()``      外部信号量等待节点
   =========================================================== ============

请参阅图 API 以获取有关用法和当前限制的更多信息。

更多使用说明与当前存在的限制，请参阅 `CUDA 图相关 API <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH>`_ 文档。

.. _cuda-graphs-single-node-enable:

4.2.3.3. 单节点启用
~~~~~~~~~~~~~~~~~~~~~

可通过 ``cudaGraphNodeSetEnabled()`` 接口启用或禁用已实例化图中的核函数、内存置零、内存拷贝节点。
借助该特性，我们可以创建一张包含全部所需功能的完整图，后续每次执行时按需定制启用 / 关闭部分逻辑。
可调用 ``cudaGraphNodeGetEnabled()`` 接口查询节点当前的启用状态。

被禁用的节点在重新启用前，功能上等同于空节点。节点参数不会因启用 / 禁用操作发生改变。
节点的启用状态不会受单节点更新，或是整图更新影响。若在节点处于禁用状态时更新了其参数，这些新参数会在该节点重新启用后生效。

更多使用方式与当前存在的限制，请查阅 `CUDA 图相关 API <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH>`_ 文档。

.. _cuda-graphs-update-limitations:

4.2.3.4. 图更新限制
~~~~~~~~~~~~~~~~~~~

**kernel 节点：**

- 函数的所属上下文不能更改。
- 若某节点绑定的原始核函数未使用 CUDA 动态并行机制，则无法通过更新操作将其替换为启用了 CUDA 动态并行的核函数。

**cudaMemset 和 cudaMemcpy 节点：**

- 操作数所分配 / 映射至的 CUDA 设备不可变更。
- 分配源内存与目标内存的 CUDA 上下文必须和原始源 / 目标内存保持一致。
- 仅一维的 ``cudaMemset`` 、 ``cudaMemcpy`` 节点支持参数修改。

**内存拷贝节点的其他限制如下：**

- 不支持修改源内存或目标内存的类型（如 ``cudaPitchedPtr`` 、 ``cudaArray_t`` 等），也不支持修改传输类型（即 ``cudaMemcpyKind`` ）。

**外部信号量等待节点和记录节点：**

- 不支持更改信号量数量。

**条件节点：**

- 两张图之间，句柄的创建顺序与赋值顺序必须保持一致。
- 不支持修改节点参数（例如条件节点内包含的子图数量、节点绑定的 CUDA 上下文等）。
- 条件分支子图内部节点的参数修改，仍需遵循上述各项规则。

**内存节点：**

- 如果 ``cudaGraph_t`` 原图已实例化生成另一个 ``cudaGraphExec_t`` 可执行图，则无法使用该原图去更新别的 ``cudaGraphExec_t`` 。

主机节点、事件记录节点、事件等待节点的参数更新无相关限制。

.. _cuda-graphs-conditional-nodes:

4.2.4. 条件节点
-----------------

条件节点支持对其内部封装的子图进行条件分支执行与循环执行。
借助该特性，完整的动态迭代业务流程均可在图内部实现，无需主机 CPU 介入，CPU 可并行处理其他任务。

当条件节点的所有依赖项均就绪后，会在设备端完成条件值的判定。条件节点分为以下几种类型：

- :ref:`IF 节点 <cuda-graphs-conditional-if>` ：如果执行 ``if`` 节点时条件值非零，则执行一次其主体子图。
  也可额外提供第二份主体子图，若执行时条件值为零，则执行该第二份子图一次。

- :ref:`WHILE 节点 <cuda-graphs-conditional-while>` ：如果执行 ``while`` 节点时条件值非零，若条件值非零则运行其主体子图，并会反复执行该主体子图，直至条件值变为零为止。
- :ref:`条件 SWITCH 节点 <cuda-graphs-conditional-switch>` ： 执行 ``witch`` 节点时， 若条件值等于 `n` ，则执行下标为 `n` 的对应主体子图一次；
  若条件值不存在匹配的主体子图，则不调度执行任何子图。

条件值通过 :ref:`条件句柄 <cuda-graphs-conditional-handles>` 访问，该句柄必须在创建条件节点前先行创建。
设备端代码可调用 ``cudaGraphSetConditional()`` 接口设置条件值；创建句柄时也可指定一个默认值，每次启动图执行时都会使用该默认值。

创建条件节点时，将创建一个空图，并将句柄返回给用户，以便可以填充图。此条件体图可以使用图 API 或 cudaStreamBeginCaptureToGraph() 填充。

创建条件节点时，会同步生成一张空白子图并返回对应句柄给用户，供用户向其中填充任务节点。
开发者可通过 :ref:`图接口 <cuda-graphs-graph-api>` ，或是调用 ``cudaStreamBeginCaptureToGraph()`` 向该条件主体子图录入节点。

条件节点支持嵌套使用。

.. _cuda-graphs-conditional-handles:

4.2.4.1. 条件句柄
~~~~~~~~~~~~~~~~~

条件值由 ``cudaGraphConditionalHandle`` 类型表示，通过 ``cudaGraphConditionalHandleCreate()`` 接口创建。

一个条件句柄只能绑定单个条件节点。条件句柄无需手动销毁，因此开发者不必维护其生命周期。

创建句柄时若指定 ``cudaGraphCondAssignDefault`` 标志位，则每次图执行开始时，条件值都会被初始化为设定的默认值。
若未传入该标志位，每次图启动时条件值处于未定义状态，代码不可假定条件值会在多次执行间保留原有数值。

在执行 :ref:`整图更新 <cuda-graphs-whole-graph-update>` 操作时，句柄对应的默认值与关联标志位会同步更新。

.. _cuda-graphs-conditional-body-graph:

4.2.4.2. 条件节点体图要求
~~~~~~~~~~~~~~~~~~~~~~~~~

**一般要求：**

- 图中的所有节点必须归属同一设备。
- 图只能包含 kernel 节点、空节点、内存拷贝节点、内存填充节点、子图节点和条件节点。

**kernel 节点：**

- 图内的 kernel 节点不允许使用 CUDA 动态并行，也不允许在设备端发起图执行。
- 只要未启用 MPS （Multi-Process Service），协同组启动操作是允许的。

**内存拷贝/内存填充节点：**

- 仅允许操作设备内存、锁页内存（pinned host memory）之间的拷贝 / 内存填充操作。
- 不允许执行涉及 CUDA 数组的内存拷贝 / 内存置值操作。
- 实例化可执行图时，两个操作数对应的内存都必须能被当前设备访问。请注意：即便拷贝目标内存位于其他设备，拷贝操作仍会在该图所属的设备上执行。

.. _cuda-graphs-conditional-if:

4.2.4.3. IF 节点
~~~~~~~~~~~~~~~~~~~~~

执行 `IF` 节点时，若条件值非零，则其主体子图会执行一次。下图展示了一张包含三个节点的图，其中中间节点 B 为条件节点：

.. figure:: /_static/images/conditional-if-node.png
   :alt: 条件 IF 节点
   :align: center

   条件 IF 节点

以下代码演示如何创建包含 IF 条件节点的 CUDA 图。
通过前置 kernel 设置条件的默认值，并借助图相关接口填充该条件节点的主体子图。

.. code-block:: cpp

   __global__ void setHandle(cudaGraphConditionalHandle handle, int value)
   {
       ...
       // Set the condition value to the value passed to the kernel
       cudaGraphSetConditional(handle, value);
       ...
   }

   void graphSetup() {
       cudaGraph_t graph;
       cudaGraphExec_t graphExec;
       cudaGraphNode_t node;
       void *kernelArgs[2];
       int value = 1;

       // Create the graph
       cudaGraphCreate(&graph, 0);

       // Create the conditional handle; because no default value is provided,
       // the condition value is undefined at the start of each graph execution
       cudaGraphConditionalHandle handle;
       cudaGraphConditionalHandleCreate(&handle, graph);

       // Use a kernel upstream of the conditional to set the handle value
       cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
       params.kernel.func = (void *)setHandle;
       params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
       params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
       params.kernel.kernelParams = kernelArgs;
       kernelArgs[0] = &handle;
       kernelArgs[1] = &value;
       cudaGraphAddNode(&node, graph, NULL, 0, &params);

       // Create and add the conditional node
       cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
       cParams.conditional.handle = handle;
       cParams.conditional.type   = cudaGraphCondTypeIf;
       cParams.conditional.size   = 1; // There is only an "if" body graph
       cudaGraphAddNode(&node, graph, &node, 1, &cParams);

       // Get the body graph of the conditional node
       cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

       // Populate the body graph of the IF conditional node
       ...
       cudaGraphAddNode(&node, bodyGraph, NULL, 0, &params);

       // Instantiate and launch the graph
       cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
       cudaGraphLaunch(graphExec, 0);
       cudaDeviceSynchronize();

       // Clean up
       cudaGraphExecDestroy(graphExec);
       cudaGraphDestroy(graph);
   }

IF 节点还可配置可选的第二个分支子图；执行该节点时，若条件值为零，则运行此分支子图一次。


.. code-block:: cpp

   void graphSetup() {
       cudaGraph_t graph;
       cudaGraphExec_t graphExec;
       cudaGraphNode_t node;
       void *kernelArgs[2];
       int value = 1;

       // Create the graph
       cudaGraphCreate(&graph, 0);

       // Create the conditional handle; because no default value is provided, 、
       // the condition value is undefined at the start of each graph execution
       cudaGraphConditionalHandle handle;
       cudaGraphConditionalHandleCreate(&handle, graph);

       // Use a kernel upstream of the conditional to set the handle value
       cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
       params.kernel.func = (void *)setHandle;
       params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
       params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
       params.kernel.kernelParams = kernelArgs;
       kernelArgs[0] = &handle;
       kernelArgs[1] = &value;
       cudaGraphAddNode(&node, graph, NULL, 0, &params);

       // Create and add the IF conditional node
       cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
       cParams.conditional.handle = handle;
       cParams.conditional.type   = cudaGraphCondTypeIf;
       cParams.conditional.size   = 2; // There is both an "if" and an "else" body graph
       cudaGraphAddNode(&node, graph, &node, 1, &cParams);

       // Get the body graphs of the conditional node
       cudaGraph_t ifBodyGraph = cParams.conditional.phGraph_out[0];
       cudaGraph_t elseBodyGraph = cParams.conditional.phGraph_out[1];

       // Populate the body graphs of the IF conditional node
       ...
       cudaGraphAddNode(&node, ifBodyGraph, NULL, 0, &params);
       ...
       cudaGraphAddNode(&node, elseBodyGraph, NULL, 0, &params);

       // Instantiate and launch the graph
       cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
       cudaGraphLaunch(graphExec, 0);
       cudaDeviceSynchronize();

       // Clean up
       cudaGraphExecDestroy(graphExec);
       cudaGraphDestroy(graph);
   }

.. _cuda-graphs-conditional-while:

4.2.4.4. WHILE 节点
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

只要条件值不为零， `WHILE` 节点的主体子图就会循环执行。
系统会在节点首次执行时、以及每一轮主体子图运行完毕后，对条件进行判断。
下图展示了一张包含三个节点的图，其中中间节点 B 为条件节点：

.. figure:: /_static/images/conditional-while-node.png
   :alt: 条件 WHILE 节点
   :align: center

   WHILE 节点

以下代码演示如何创建包含 `WHILE` 节点的图。
创建句柄时传入 ``cudaGraphCondAssignDefault`` 标志，无需借助上游内核来初始化条件值；并通过:ref:`图接口<cuda-graphs-graph-api>` 填充该条件节点的主体子图。

.. code-block:: cpp

   __global__ void loopKernel(cudaGraphConditionalHandle handle, char *dPtr)
   {
      // Decrement the value of dPtr and set the condition value to 0 once dPtr is 0
      if (--(*dPtr) == 0) {
         cudaGraphSetConditional(handle, 0);
      }
   }

   void graphSetup() {
       cudaGraph_t graph;
       cudaGraphExec_t graphExec;
       cudaGraphNode_t node;
       void *kernelArgs[2];

       // Allocate a byte of device memory to use as input
       char *dPtr;
       cudaMalloc((void **)&dPtr, 1);

       // Create the graph
       cudaGraphCreate(&graph, 0);

       // Create the conditional handle with a default value of 1
       cudaGraphConditionalHandle handle;
       cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault);

       // Create and add the WHILE conditional node
       cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
       cParams.conditional.handle = handle;
       cParams.conditional.type   = cudaGraphCondTypeWhile;
       cParams.conditional.size   = 1;
       cudaGraphAddNode(&node, graph, NULL, 0, &cParams);

       // Get the body graph of the conditional node
       cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

       // Populate the body graph of the conditional node
       cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
       params.kernel.func = (void *)loopKernel;
       params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
       params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
       params.kernel.kernelParams = kernelArgs;
       kernelArgs[0] = &handle;
       kernelArgs[1] = &dPtr;
       cudaGraphAddNode(&node, bodyGraph, NULL, 0, &params);

       // Initialize device memory, instantiate, and launch the graph
       cudaMemset(dPtr, 10, 1); // Set dPtr to 10; the loop will run until dPtr is 0
       cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
       cudaGraphLaunch(graphExec, 0);
       cudaDeviceSynchronize();

       // Clean up
       cudaGraphExecDestroy(graphExec);
       cudaGraphDestroy(graph);
       cudaFree(dPtr);
   }

.. _cuda-graphs-conditional-switch:

4.2.4.5. SWITCH 节点
~~~~~~~~~~~~~~~~~~~~~~~~~

执行 `SWITCH` 节点时，若条件值等于 ``n`` ，则会执行索引为 ``n`` （从 0 开始计数）的分支子图一次。
下图展示了一张包含三个节点的图，其中中间节点 B 为条件节点：

.. figure:: /_static/images/conditional-switch-node.png
   :alt: 条件 SWITCH 节点
   :align: center

   SWITCH 节点

以下代码展示如何创建包含 `SWITCH` 节点的图。通过前置任务设置条件数值，并使用 :ref:`图接口<cuda-graphs-graph-api>` 填充该条件节点的各分支子图。

.. code-block:: cpp

   __global__ void setHandle(cudaGraphConditionalHandle handle, int value)
   {
       ...
       // Set the condition value to the value passed to the kernel
       cudaGraphSetConditional(handle, value);
       ...
   }

   void graphSetup() {
       cudaGraph_t graph;
       cudaGraphExec_t graphExec;
       cudaGraphNode_t node;
       void *kernelArgs[2];
       int value = 1;

       // Create the graph
       cudaGraphCreate(&graph, 0);

       // Create the conditional handle; because no default value is provided,
       // the condition value is undefined at the start of each graph execution
       cudaGraphConditionalHandle handle;
       cudaGraphConditionalHandleCreate(&handle, graph);

       // Use a kernel upstream of the conditional to set the handle value
       cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
       params.kernel.func = (void *)setHandle;
       params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
       params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
       params.kernel.kernelParams = kernelArgs;
       kernelArgs[0] = &handle;
       kernelArgs[1] = &value;
       cudaGraphAddNode(&node, graph, NULL, 0, &params);

       // Create and add the conditional SWITCH node
       cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
       cParams.conditional.handle = handle;
       cParams.conditional.type   = cudaGraphCondTypeSwitch;
       cParams.conditional.size   = 5;
       cudaGraphAddNode(&node, graph, &node, 1, &cParams);

       // Get the body graphs of the conditional node
       cudaGraph_t *bodyGraphs = cParams.conditional.phGraph_out;

       // Populate the body graphs of the SWITCH conditional node
       ...
       cudaGraphAddNode(&node, bodyGraphs[0], NULL, 0, &params);
       ...
       cudaGraphAddNode(&node, bodyGraphs[4], NULL, 0, &params);

       // Instantiate and launch the graph
       cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
       cudaGraphLaunch(graphExec, 0);
       cudaDeviceSynchronize();

       // Clean up
       cudaGraphExecDestroy(graphExec);
       cudaGraphDestroy(graph);
   }

.. _cuda-graphs-memory-nodes:

4.2.5. 图内存节点
-----------------

.. _cuda-graphs-memory-nodes-intro:

4.2.5.1. 简介
~~~~~~~~~~~~~

图内存节点允许 CUDA 图创建并持有内存分配对象。
图内存节点具备 GPU 执行时序的生命周期语义，该语义规定了设备上允许访问对应内存的时段。
这套 GPU 执行时序的生命周期机制支持驱动自动复用内存，其行为与流序分配接口 ``cudaMallocAsync`` 、 ``cudaFreeAsync`` 保持一致，而这两个异步内存接口在构建图时可被捕获为图节点。

在整张图的完整生命周期内（包含多次实例化与执行），图分配内存的虚拟地址保持固定不变。
即便 CUDA 更换底层承载的物理内存，图内其他操作仍可直接引用该内存，无需执行图更新操作。
在同一张图中，若多个内存分配的图时序生命周期互不重叠，它们可复用同一块底层物理内存。

CUDA 可在多张不同的图之间复用同一块物理内存，并依据 GPU 执行时序生命周期语义做虚拟地址映射别名复用。
例如，若多张不同的图提交至同一个流执行，对于生命周期仅局限于单张图内的内存分配，CUDA 会通过虚拟地址别名复用同一块物理内存以节省资源。

.. _cuda-graphs-memory-nodes-api:

4.2.5.2. API 基础
~~~~~~~~~~~~~~~~~

图内存节点是用于表示内存分配或释放操作的图节点。
为简便起见，执行内存分配的节点称为分配节点；同理，执行内存释放的节点称为释放节点。由分配节点创建的内存空间称作图分配内存。
CUDA 会在节点创建阶段为图分配内存分配虚拟地址。
该虚拟地址在分配节点的整个生命周期内保持固定，但内存中的数据在释放操作执行后不会持久留存，可能会被指向其他内存分配块的读写操作覆盖。

可以认为，每次图执行时图分配都被重新创建。
图分配内存的生命周期与对应分配节点的生命周期并不等同；该内存生命周期从 GPU 执行到分配节点时开启，并在满足以下任一条件时结束：

- GPU 执行至释放节点。
- GPU 执行到达 ``cudaFreeAsync()`` 调用。
- 在调用 ``cudaFree()`` 时立即释放

.. note::
   销毁图并不会自动释放任何处于活跃状态的图分配内存，尽管销毁图会终止分配节点自身的生命周期。
   必须在另一张图中执行释放操作，或是调用 ``cudaFreeAsync()`` / ``cudaFree()`` 来释放该内存。

与其他图中节点相同，图内存节点依靠依赖边在图内形成执行先后顺序。
程序必须保证访问图内存的各类操作满足以下要求：

- 在分配节点之后
- 在释放内存之前

图分配生命周期根据 GPU 执行开始和通常结束（与 API 调用相反）。GPU 排序是工作在 GPU 上运行的顺序，而不是工作入队或描述的顺序。因此，图分配被认为是"GPU 有序"的。

图分配内存的生命周期均依据 GPU 的实际执行时序开启，且通常也依据该时序结束（而非主机端 API 的调用顺序）。
GPU 时序指任务在 GPU 硬件上实际运行的先后顺序，区别于任务入队、任务定义的先后顺序。
因此，图分配内存属于遵循 `GPU 执行时序` 。

4.2.5.2.1. 图节点 API
`````````````````````

可通过 ``cudaGraphAddNode`` 显式创建图内存节点。
当添加类型为 ``cudaGraphNodeTypeMemAlloc`` 的内存分配节点时，分配得到的地址会存入传入的 ``cudaGraphNodeParams`` 结构体的 ``alloc::dptr`` 字段并返回给用户。
在该分配节点所属的图中，所有使用这块图分配内存的操作，执行顺序都必须排在分配节点之后。
同理，图内所有释放节点，执行顺序都必须排在该内存的全部读写操作之后。
释放节点同样通过 ``cudaGraphAddNode`` 创建，节点类型指定为 ``cudaGraphNodeTypeMemFree``。

如下图所示，节点 ``a`` 、 ``b`` 、 ``c`` 的执行顺序排在分配节点之后、释放节点之前，因此这些节点可以正常访问该内存；
节点 ``e`` 的执行顺序并未排在分配节点之后，故而无法安全访问这块内存；
节点 ``d`` 的执行顺序未排在释放节点之前，因此也无法安全访问该内存。

.. figure:: /_static/images/kernel-nodes.png
   :alt: 内核节点示例
   :align: center

   图内存申请节点和释放节点

代码如下：

.. code-block:: cpp

   // Create the graph - it starts out empty
   cudaGraphCreate(&graph, 0);

   // parameters for a basic allocation
   cudaGraphNodeParams params = { cudaGraphNodeTypeMemAlloc };
   params.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
   params.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
   // specify device 0 as the resident device
   params.alloc.poolProps.location.id = 0;
   params.alloc.bytesize = size;

   cudaGraphAddNode(&allocNode, graph, NULL, NULL, 0, &params);

   // create a kernel node that uses the graph allocation
   cudaGraphNodeParams nodeParams = { cudaGraphNodeTypeKernel };
   nodeParams.kernel.kernelParams[0] = params.alloc.dptr;
   // ...set other kernel node parameters...

   // add the kernel node to the graph
   cudaGraphAddNode(&a, graph, &allocNode, 1, NULL, &nodeParams);
   cudaGraphAddNode(&b, graph, &a, 1, NULL, &nodeParams);
   cudaGraphAddNode(&c, graph, &a, 1, NULL, &nodeParams);
   cudaGraphNode_t dependencies[2];
   // kernel nodes b and c are using the graph allocation,
   // so the freeing node must depend on them.
   // Since the dependency of node b on node a establishes an indirect dependency,
   // the free node does not need to explicitly depend on node a.
   dependencies[0] = b;
   dependencies[1] = c;
   cudaGraphNodeParams freeNodeParams = { cudaGraphNodeTypeMemFree };
   freeNodeParams.free.dptr = params.alloc.dptr;
   cudaGraphAddNode(&freeNode, graph, dependencies, NULL, 2, freeNodeParams);
   // free node does not depend on kernel node d,
   // so it must not access the freed graph allocation.
   cudaGraphAddNode(&d, graph, &c, NULL, 1, &nodeParams);

   // node e does not depend on the allocation node,
   // so it must not access the allocation.
   // This would be true even if the freeNode depended on kernel node e.
   cudaGraphAddNode(&e, graph, NULL, NULL, 0, &nodeParams);

4.2.5.2.2. 流捕获
`````````````````

对捕获流执行 ``cudaMallocAsync`` 与 ``cudaFreeAsync`` ，会自动生成对应的图内存节点。
此种场景下，被捕获的分配接口所返回的虚拟地址，可供给图内其他操作使用。
由于流的时序依赖关系会一并捕获至图中，只要原流操作代码书写规范，流序分配接口自带的时序约束规则，就能保证图内存节点与捕获的各类流操作之间具备正确的执行先后关系。

为便于理解，此处暂且忽略节点 `d` 和 `e` 。下方代码片段演示如何使用流捕获，构建上文中示意图对应的 CUDA 图：

.. code-block:: c++

   cudaMallocAsync(&dptr, size, stream1);
   kernel_A<<< ..., stream1 >>>(dptr, ...);

   // Fork into stream2
   cudaEventRecord(event1, stream1);
   cudaStreamWaitEvent(stream2, event1);

   kernel_B<<< ..., stream1 >>>(dptr, ...);
   // event dependencies translated into graph dependencies,
   // so the kernel node created by the capture of kernel C
   // will depend on the allocation node created by capturing the cudaMallocAsync call.
   kernel_C<<< ..., stream2 >>>(dptr, ...);

   // Join stream2 back to origin stream (stream1)
   cudaEventRecord(event2, stream2);
   cudaStreamWaitEvent(stream1, event2);

   // Free depends on all work accessing the memory.
   cudaFreeAsync(dptr, stream1);

   // End capture in the origin stream
   cudaStreamEndCapture(stream1, &graph);

.. _cuda-graphs-accessing-and-freeing-graph-memory-outside:

4.2.5.2.3. 在分配图之外访问和释放图内存
`````````````````````````````````````````

图分配内存不强制要求在创建它的图中完成释放。
若一张图没有释放某块分配内存，该内存会在这张图执行完毕后继续留存，可供后续 CUDA 操作访问。
这类内存既可在另一张图内访问，也能直接通过普通流操作访问，只需要通过 CUDA 事件或其他流排序机制，确保访问操作排在分配操作之后即可。
后续释放该内存有多种方式：直接调用常规接口 ``cudaFree`` 、 ``cudaFreeAsync`` ；执行另一张包含对应释放节点的图；
或是重新执行创建该内存的原图（前提是实例化原图时传入了 :ref:`cudaGraphInstantiateFlagAutoFreeOnLaunch <cuda-graphs-auto-free-on-launch>` 标识）。
内存释放后再对其访问属于非法行为。
所有的内存读写操作，都应借助依赖边、CUDA 事件或其他流排序机制，在释放操作之前完成。

.. note::

   由于多张图可能共享底层同一块物理显存，释放操作的执行时序必须排在所有设备运算操作完成之后。
   带外同步（例如计算核函数内部基于内存的同步方式）不足以保证内存写操作与释放操作之间的执行先后关系。
   如需了解更多细节，请参阅与内存一致性、缓存一致性相关的 :ref:`virtual-aliasing-support` 规范。

下面三个代码片段演示如何在外部访问图分配内存，分别通过三种方式建立合法时序依赖：使用单一流、跨流搭配 CUDA 事件、将事件嵌入分配与释放对应的图中。

.. tab-set::

   .. tab-item:: 使用单一流

      .. code-block:: c++

         // Contents of allocating graph
         void *dptr;
         cudaGraphNodeParams params = { cudaGraphNodeTypeMemAlloc };
         params.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
         params.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
         params.alloc.bytesize = size;
         cudaGraphAddNode(&allocNode, allocGraph, NULL, NULL, 0, &params);
         dptr = params.alloc.dptr;

         cudaGraphInstantiate(&allocGraphExec, allocGraph, NULL, NULL, 0);

         cudaGraphLaunch(allocGraphExec, stream);
         kernel<<< ..., stream >>>(dptr, ...);
         cudaFreeAsync(dptr, stream);

   .. tab-item:: 通过 CUDA 事件

      .. code-block:: c++

         // Contents of allocating graph
         void *dptr;
         cudaGraphAddNode(&allocNode, allocGraph, NULL, NULL, 0, &allocNodeParams);
         dptr = allocNodeParams.alloc.dptr;

         // Contents of consuming/freeing graph
         kernelNodeParams.kernel.kernelParams[0] = allocNodeParams.alloc.dptr;
         cudaGraphAddNode(&freeNode, freeGraph, NULL, NULL, 1, dptr);

         cudaGraphInstantiate(&allocGraphExec, allocGraph, NULL, NULL, 0);
         cudaGraphInstantiate(&freeGraphExec, freeGraph, NULL, NULL, 0);

         cudaGraphLaunch(allocGraphExec, allocStream);

         // Establish stream2's dependency on the allocation node
         cudaEventRecord(allocEvent, allocStream);
         cudaStreamWaitEvent(stream2, allocEvent);

         kernel<<< ..., stream2 >>> (dptr, ...);

         // Establish dependency between stream3 and the allocation use
         cudaStreamRecordEvent(streamUseDoneEvent, stream2);
         cudaStreamWaitEvent(stream3, streamUseDoneEvent);

         // Now it is safe to launch the free graph, which can also access the memory
         cudaGraphLaunch(freeGraphExec, stream3);

   .. tab-item:: 使用图外部事件节点

      .. code-block:: c++

         // Contents of allocating graph
         void *dptr;
         cudaEvent_t allocEvent;  // event indicating when allocation is ready for use
         cudaEvent_t streamUseDoneEvent;  // event indicating when stream operations are done

         // Allocating graph contents with event record node
         cudaGraphAddNode(&allocNode, allocGraph, NULL, NULL, 0, &allocNodeParams);
         dptr = allocNodeParams.alloc.dptr;
         // Note: this event record node depends on the allocation node

         cudaGraphNodeParams allocEventNodeParams = { cudaGraphNodeTypeEventRecord };
         allocEventNodeParams.eventRecord.event = allocEvent;
         cudaGraphAddNode(&recordNode, allocGraph, &allocNode, NULL, 1, allocEventNodeParams);
         cudaGraphInstantiate(&allocGraphExec, allocGraph, NULL, NULL, 0);

         // Consuming/freeing graph contents with event wait node
         cudaGraphNodeParams streamWaitEventNodeParams = { cudaGraphNodeTypeEventWait };
         streamWaitEventNodeParams.eventWait.event = streamUseDoneEvent;
         cudaGraphAddNode(&streamUseDoneEventNode,
                          waitAndFreeGraph,
                          NULL,
                          NULL,
                          0,
                          streamWaitEventNodeParams);

         cudaGraphNodeParams allocWaitEventNodeParams = { cudaGraphNodeTypeEventWait };
         allocWaitEventNodeParams.eventWait.event = allocEvent;
         cudaGraphAddNode(&allocReadyEventNode,
                          waitAndFreeGraph,
                          NULL,
                          NULL,
                          0,
                          allocWaitEventNodeParams);

         kernelNodeParams->kernelParams[0] = allocNodeParams.alloc.dptr;

         // allocReadyEventNode provides ordering for the allocation node in the consuming graph
         cudaGraphAddNode(&kernelNode,
                          waitAndFreeGraph,
                          &allocReadyEventNode,
                          NULL,
                          1,
                          &kernelNodeParams);

         // Free node must be ordered after both external and internal users
         dependencies[0] = kernelNode;
         dependencies[1] = streamUseDoneEventNode;

         cudaGraphNodeParams freeNodeParams = { cudaGraphNodeTypeMemFree };
         freeNodeParams.free.dptr = dptr;
         cudaGraphAddNode(&freeNode, waitAndFreeGraph, &dependencies, NULL, 2, freeNodeParams);
         cudaGraphInstantiate(&waitAndFreeGraphExec, waitAndFreeGraph, NULL, NULL, 0);

         cudaGraphLaunch(allocGraphExec, allocStream);

         // Establish stream2's dependency on the event node to satisfy ordering requirements
         cudaStreamWaitEvent(stream2, allocEvent);
         kernel<<< ..., stream2 >>> (dptr, ...);
         cudaStreamRecordEvent(streamUseDoneEvent, stream2);

         // Event wait node in waitAndFreeGraphExec establishes dependency on required events
         cudaGraphLaunch(waitAndFreeGraphExec, stream3);

.. _cuda-graphs-auto-free-on-launch:

4.2.5.2.4. cudaGraphInstantiateFlagAutoFreeOnLaunch
````````````````````````````````````````````````````

在正常情况下，如果一张图存在未释放的内存分配，CUDA 会阻止图被重新启动，因为同一虚拟地址重复分配会造成显存泄漏。
使用 ``cudaGraphInstantiateFlagAutoFreeOnLaunch`` 标志实例化图允许图在仍有未释放分配的情况下重新启动。
在这种情况下，启动时会自动插入针对未释放分配的异步释放操作。

启动时自动释放对于单生产者多消费者算法很有用。在每次迭代中，生产者图创建多个分配，并且根据运行时条件，不同的消费者集合访问这些分配。这种可变执行序列意味着消费者无法释放分配，因为后续消费者可能需要访问。启动时自动释放意味着启动循环不需要跟踪生产者的分配——相反，该信息保持隔离在生产者的创建和销毁逻辑中。一般来说，启动时自动释放简化了原本需要在每次重新启动前释放图拥有的所有分配的算法。

在启动时自动释放（Auto free on launch）功能对于单生产者多消费者算法非常有用。
每一轮迭代中，生产者图（producer graph）会创建多个内存分配，并且根据运行时条件，会有不定数量的消费者访问这些内存。
这种可变执行序列决定了消费者无法释放内存，因为后续可能还有消费者需要读取该内存。
`在启动时自动释放` 意味着启动循环无需追踪生产者的内存分配——相反，这些信息被隔离在生产者自身的创建与销毁逻辑中。
通常来说，该功能简化了算法设计，否则程序需要在每次重新启动图之前手动释放该图持有的全部内存。

.. note::

   ``cudaGraphInstantiateFlagAutoFreeOnLaunch`` 标识不会改变图销毁时的行为。
   即便图是通过该标识完成实例化，应用程序仍必须显式释放所有未回收的内存，以此避免显存泄漏。

下方代码演示如何使用 ``cudaGraphInstantiateFlagAutoFreeOnLaunch`` 简化单生产者 / 多消费者算法：

.. code-block:: c++

   // Create producer graph that allocates memory and fills it with data
   cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
   cudaMallocAsync(&data1, blocks * threads, cudaStreamPerThread);
   cudaMallocAsync(&data2, blocks * threads, cudaStreamPerThread);
   produce<<<blocks, threads, 0, cudaStreamPerThread>>>(data1, data2);
   ...
   cudaStreamEndCapture(cudaStreamPerThread, &graph);
   cudaGraphInstantiateWithFlags(&producer,
                                 graph,
                                 cudaGraphInstantiateFlagAutoFreeOnLaunch);
   cudaGraphDestroy(graph);

   // Create first consumer graph by capturing an asynchronous library call
   cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
   consumerFromLibrary(data1, cudaStreamPerThread);
   cudaStreamEndCapture(cudaStreamPerThread, &graph);
   cudaGraphInstantiateWithFlags(&consumer1, graph, 0);  // regular instantiation
   cudaGraphDestroy(graph);

   // Create second consumer graph
   cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
   consume2<<<blocks, threads, 0, cudaStreamPerThread>>>(data2);
   ...
   cudaStreamEndCapture(cudaStreamPerThread, &graph);
   cudaGraphInstantiateWithFlags(&consumer2, graph, 0);
   cudaGraphDestroy(graph);

   // Launch in a loop
   bool launchConsumer2 = false;
   do {
       cudaGraphLaunch(producer, myStream);
       cudaGraphLaunch(consumer1, myStream);
       if (launchConsumer2) {
           cudaGraphLaunch(consumer2, myStream);
       }
   } while (determineAction(&launchConsumer2));

   // free the unfreed memory, orderd by myStream
   cudaFreeAsync(data1, myStream);
   cudaFreeAsync(data2, myStream);

   cudaGraphExecDestroy(producer);
   cudaGraphExecDestroy(consumer1);
   cudaGraphExecDestroy(consumer2);

.. _cuda-graphs-memory-nodes-in-child-graphs:

4.2.5.2.5. 子图中的内存节点
````````````````````````````

CUDA 12.9 新增了将子图的所有权转移给父图的功能。
被转移至父图的子图现在允许包含内存分配和释放节点。
这使得包含分配或释放节点的子图，可以在被添加到父图之前进行独立构建。

子图完成所有权转移后，需遵守以下限制条件：

- 不能独立实例化或销毁。
- 不能再作为子图添加到其他父图中。
- 不能用作 ``cuGraphExecUpdate`` 的参数。
- 不能添加额外的内存分配或释放节点。


.. code-block:: c++

   // Create child graph
   cudaGraphCreate(&child, 0);

   // Parameters for a basic allocation
   cudaGraphNodeParams allocNodeParams = { cudaGraphNodeTypeMemAlloc };
   allocNodeParams.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
   allocNodeParams.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
   // Specify device 0 as the resident device
   allocNodeParams.alloc.poolProps.location.id = 0;
   allocNodeParams.alloc.bytesize = size;

   cudaGraphAddNode(&allocNode, child, NULL, NULL, 0, &allocNodeParams);
   // Additional nodes using this allocation can be added here
   cudaGraphNodeParams freeNodeParams = { cudaGraphNodeTypeMemFree };
   freeNodeParams.free.dptr = allocNodeParams.alloc.dptr;
   cudaGraphAddNode(&freeNode, child, &allocNode, NULL, 1, freeNodeParams);

   // Create parent graph
   cudaGraphCreate(&parent, 0);

   // Move child graph into parent graph
   cudaGraphNodeParams childNodeParams = { cudaGraphNodeTypeGraph };
   childNodeParams.graph.graph = child;
   childNodeParams.graph.ownership = cudaGraphChildGraphOwnershipMove;
   cudaGraphAddNode(&parentNode, parent, NULL, NULL, 0, &childNodeParams);

.. _cuda-graphs-memory-nodes-optimization:

4.2.5.3. 优化内存重用
~~~~~~~~~~~~~~~~~~~~~

CUDA 以两种方式重用内存：

- 图内的虚拟和物理内存重用基于虚拟地址分配，与流有序分配器（stream ordered allocator）机制类似。
- 图之间的物理内存重用通过虚拟别名实现：不同的图可以将各自唯一的虚拟地址映射到相同的物理内存。

.. _cuda-graphs-address-reuse-within-a-graph:

4.2.5.3.1. 图内地址重用
````````````````````````

CUDA 可能会为生命周期不重叠的不同内存分配分配相同的虚拟地址，从而在图内部实现内存复用。
由于虚拟地址可能会被重复使用，因此指向具有非重叠生命周期的不同分配的指针，并不保证具有唯一性。

下图展示了添加一个新的分配节点（2），它可以重用其依赖节点（1）释放的地址。

.. figure:: /_static/images/new-alloc-node.png
   :alt: Adding New Alloc Node 2
   :align: center
   :scale: 80

   添加新的分配节点 （2）

下图展示了添加一个新的分配节点（3）。新的分配节点不依赖于释放节点（2），因此无法重用关联分配节点（2）的地址。
如果分配节点（2）使用了释放节点（1）释放的地址，则新的分配节点 3 将需要一个新地址。

.. figure:: /_static/images/adding-new-alloc-nodes.png
   :alt: Adding New Alloc Node 3
   :align: center
   :scale: 60

   添加新的分配节点 3

.. _cuda-graphs-physical-memory-management-and-sharing:

4.2.5.3.2. 物理内存管理和共享
``````````````````````````````

在 GPU 执行至分配节点之前，CUDA 负责将物理内存映射到虚拟地址。
为优化显存占用与映射开销，若多张图不会同时运行，它们各自独立的内存分配可复用同一块物理显存；但如果物理页同时绑定多个正在执行的图，或是绑定了未释放的图内存分配，则该物理页无法被复用。

CUDA 可在图实例化、启动或执行阶段的任意时刻更新物理显存映射关系。
CUDA 还可能在后续多次图启动操作之间插入同步逻辑，避免存活的图内存分配指向同一块物理显存。
对于任何 **分配 - 释放 - 再分配** 的使用模式，若程序在某段内存分配的生命周期之外访问其指针，该非法访问可能在无报错的情况下读写另一块内存分配的有效数据（即使分配的虚拟地址是唯一的）。
可借助计算检查工具捕获此类错误。

下图展示了在同一个流中顺序执行的多张图。
在此示例里，每张图都会释放自身申请的全部内存。
由于同一流内的多张图不会并发运行，CUDA 能够而且应该复用同一块物理显存来承载所有内存分配需求。

.. figure:: /_static/images/sequentially-launched-graphs.png
   :alt: Sequentially Launched Graphs
   :align: center
   :scale: 60

   顺序启动的图

.. _cuda-graphs-performance-considerations:

4.2.5.4. 性能考虑
~~~~~~~~~~~~~~~~~

当多张图提交至同一个流中顺序执行时，CUDA 会尝试让它们复用同一块物理显存，原因是这些图的执行时段不会重叠。
作为优化手段，一张图的物理地址映射关系会在多次启动之间保留，省去重复映射的开销。
若后续其中某张图被提交到其他流，导致其执行时段可能与其它图产生重叠（例如分配至不同流启动），CUDA 就必须重新执行地址映射操作：并发运行的多张图需要独立的物理内存，以此防止数据损坏。

CUDA 中图内存的重新映射通常由以下操作触发：

- 更改执行图的流
- 对图内存池执行修剪操作，该操作显式释放未使用的内存（在 :ref:`物理内存占用 <cuda-graphs-physical-memory-footprint>` 中讨论）
- 若另一张图存在未释放的内存分配且该分配映射至同一块物理内存，此时重新启动图，系统会在启动前执行内存重映射操作。

重新映射（Remapping）必须严格按照执行顺序进行，而且得等该图之前的执行彻底完成后才能开始（不然的话，还在使用中的内存可能会被意外取消映射）。
正因为有这种顺序依赖，再加上映射操作本身需要调用操作系统接口，所以它的开销相对比较大。
如果应用能把包含内存分配节点的图，始终提交到同一个流中执行，就能避开这笔额外的开销。

.. _cuda-graphs-first-launch-cuda-graph-upload:

4.2.5.4.1. 首次启动
````````````````````

在图实例化阶段，是无法分配或映射物理内存的，因为此时还不知道这个图最终会在哪个流中执行。
因此，映射操作会被推迟到图执行时进行。
不过，你可以调用 ``cudaGraphUpload`` 来将内存分配的开销与启动开销分离开来——它会立即执行该图的所有映射操作，并将图与指定的上传流绑定。
这样一来，如果后续将该图提交到同一个流中启动，就不会再产生任何额外的重新映射开销了。

如果在上传图和启动图时使用了不同的流，其表现就类似于切换了流，这很可能会触发重新映射操作。
此外，由于内存池的管理机制允许从处于空闲状态的流中抽调内存，这可能会抵消掉提前上传所带来的优化效果。

.. _cuda-graphs-physical-memory-footprint:

4.2.5.5. 物理内存占用
~~~~~~~~~~~~~~~~~~~~~

由于异步分配的内存池管理机制，销毁一个包含内存节点的图（即使这些节点的分配已经处于空闲状态），物理内存也不会立即归还给操作系统供其他进程使用。
如果应用程序需要显式地将内存释放回操作系统，应该调用 ``cudaDeviceGraphMemTrim`` 。

``cudaDeviceGraphMemTrim`` 会 ``unmap`` 并释放那些由图内存节点保留、但当前并未活跃使用的物理内存。
图中尚未释放的分配，以及已处于调度中或正在运行的图，都被视为正在活跃使用物理内存，不会受到该操作的影响。
使用此 API 可以使物理内存重新可用，供其他分配 API、其他应用程序或进程使用；但这也会导致 CUDA 在下一次启动这些被清理过的图时，不得不重新分配和映射内存。
需要注意的是， ``cudaDeviceGraphMemTrim`` 操作的内存池与 ``cudaMemPoolTrimTo()`` 是完全不同的，图内存池对基于流顺序的内存分配器是不可见的。

此外，应用程序可以使用 ``cudaDeviceGetGraphMemAttribute`` 来查询其图的内存占用情况：

- 查询 ``cudaGraphMemAttrReservedMemCurrent`` 属性，返回驱动程序为当前进程中的图分配所保留的物理内存总量。
- 查询 ``cudaGraphMemAttrUsedMemCurrent`` 属性，返回当前至少被一个图映射的物理内存总量（正在使用的）。

这两个属性都可以用来追踪 CUDA 何时为了满足图的内存分配需求而获取了新的物理内存。
同时，它们也非常有助于评估内存共享机制到底节省了多少内存。

.. _cuda-graphs-peer-access:

4.2.5.6. 对等访问
~~~~~~~~~~~~~~~~~

图分配可以配置为从多个 GPU 访问，在这种情况下，CUDA 将根据需要将分配映射到对等 GPU。CUDA 允许需要不同映射的图分配重用相同的虚拟地址。当这种情况发生时，地址范围会映射到不同分配所需的所有 GPU。这意味着分配有时可能允许比创建期间请求的更多对等访问；但是，依赖这些额外映射仍然是错误的。

图内存分配可以被配置为允许多个 GPU 访问，在这种情况下，CUDA 会根据需要将分配映射到对等 GPU 上。
CUDA 允许拥有不同映射关系的图分配复用相同的虚拟地址。
当发生这种情况时，该地址范围将被映射到不同分配所需的所有 GPU 设备上。
这意味着，图内存分配时可能提供超出请求的对等访问能力；依赖这些额外的映射（访问能力）属于错误行为。

.. admonition:: 译注

   例如， 图 A 分配的内存只允许 GPU0、GPU1 访问； 图B 分配的内存只允许 GPU0、GPU2 访问。
   这个两个图在同一条流上顺序执行，所以这两块图分配生命周期不重叠，CUDA 会给它们分配同一个虚拟地址。
   此时这个虚拟地址区间，会在 GPU0、1、2 三块卡全都建立映射，即 `提供超出请求的对等访问能力` 。

.. _cuda-graphs-peer-access-with-graph-node-apis:

4.2.5.6.1. 使用图节点 API 的对等访问
`````````````````````````````````````

``cudaGraphAddNode`` 接口通过内存分配节点参数结构体中的 ``accessDescs`` 数组字段接收映射访问请求。
内嵌结构体 ``poolProps.location`` 用于指定该内存分配的驻留设备。
系统默认该内存驻留的 GPU 必然需要访问该显存，因此应用程序无需在 ``accessDescs`` 数组中为驻留设备额外填写条目。

.. code-block:: c++

   cudaGraphNodeParams allocNodeParams = { cudaGraphNodeTypeMemAlloc };
   allocNodeParams.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
   allocNodeParams.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
   // specify device 1 as the resident device
   allocNodeParams.alloc.poolProps.location.id = 1;
   allocNodeParams.alloc.bytesize = size;

   // allocate an allocation resident on device 1 accessible from device 1
   cudaGraphAddNode(&allocNode, graph, NULL, NULL, 0, &allocNodeParams);

   accessDescs[2];
   // access descs (only ReadWrite and Device access supported by the add node api)
   accessDescs[0].flags = cudaMemAccessFlagsProtReadWrite;
   accessDescs[0].location.type = cudaMemLocationTypeDevice;
   accessDescs[1].flags = cudaMemAccessFlagsProtReadWrite;
   accessDescs[1].location.type = cudaMemLocationTypeDevice;

   // access being requested for device 0 & 2.
   // Device 1 access requirement left implicit.
   accessDescs[0].location.id = 0;
   accessDescs[1].location.id = 2;

   // access request array has 2 entries.
   allocNodeParams.alloc.accessDescCount = 2;
   allocNodeParams.alloc.accessDescs = accessDescs;

   // allocate an allocation resident on device 1 accessible from devices 0, 1 and 2.
   // (0 & 2 from the descriptors, 1 from it being the resident device).
   cudaGraphAddNode(&allocNode, graph, NULL, NULL, 0, &allocNodeParams);

.. _cuda-graphs-peer-access-with-stream-capture:

4.2.5.6.2. 使用流捕获的对等访问
````````````````````````````````

对于流捕获场景，内存分配节点会在捕获时记录对应内存池的对等 GPU 访问权限。
若在 ``cudaMallocFromPoolAsync`` 被捕获之后，修改内存池的对等访问权限，不会改变图执行时为这段内存创建的映射关系

.. code-block:: c++

   // access descs (only ReadWrite and Device access supported by the add node api)
   accessDesc.flags = cudaMemAccessFlagsProtReadWrite;
   accessDesc.location.type = cudaMemLocationTypeDevice;
   accessDesc.location.id = 1;

   // let memPool be resident and accessible on device 0

   cudaStreamBeginCapture(stream);
   cudaMallocAsync(&dptr1, size, memPool, stream);
   cudaStreamEndCapture(stream, &graph1);

   cudaMemPoolSetAccess(memPool, &accessDesc, 1);

   // The graph node allocating dptr1 will only have accessibility from device 0, even though
   // memPool now has accessibility from device 1.

   cudaStreamBeginCapture(stream);

   // The graph node allocating dptr2 will have accessibility from devices 0 and 1,
   // because that was the pool accessibility at the time of the cudaMallocAsync call.

   cudaMallocAsync(&dptr2, size, memPool, stream);
   cudaStreamEndCapture(stream, &graph2);

.. _cuda-graphs-device-graph-launch:

4.2.6. 从设备图启动
-------------------

许多工作流需要在运行时根据数据做出决策，并根据这些决策执行不同的操作。
与其将这一决策过程交给主机执行（这可能需要数据在设备与主机之间进行往返传输），用户通常更倾向于直接在设备上完成决策。
为此，CUDA 提供了一种从设备端启动图的机制。

设备端图执行提供了便捷的方式，可在设备侧实现动态控制流，小到简单循环，大到复杂的设备端任务调度器，都能依托该机制实现。

因此，能够从设备端启动的图将被称为“设备图”（device graphs），而不能从设备端启动的图将被称为“主机图”（host graphs）。

设备图既可以从主机启动，也可以从设备启动；而主机图只能从主机启动。
与主机端启动不同，如果在一个设备图的前一次启动仍在运行时，再次从设备端启动该图，将会导致错误，并返回 ``cudaErrorInvalidValue`` ；
因此，同一张设备图不允许在设备端并发两次执行。
此外，如果同时从主机端和设备端启动同一个设备图，将导致未定义行为。

.. _cuda-graphs-device-graph-creation:

4.2.6.1. 设备图创建
~~~~~~~~~~~~~~~~~~~

为了使图能够从设备端启动，必须在实例化时显式指定为设备端启动。
通过在调用 ``cudaGraphInstantiate()`` 时传入 ``cudaGraphInstantiateFlagDeviceLaunch`` 标志来实现。
与主机图的情况相同，设备图的结构在实例化时即被固定，若需更新则必须重新进行实例化，且实例化操作只能在主机端执行。
此外，为了使图能够被实例化为设备端启动，它必须满足一系列特定的要求。

.. _cuda-graphs-device-graph-requirements:

4.2.6.1.1. 设备图要求
``````````````````````

一般要求：

- 图的节点必须全部驻留在单个设备上。
- 图只能包含 kernel 节点、memcpy 节点、memset 节点和子图节点。

内核节点：

- 不允许图中的 kernel 使用 CUDA 动态并行。
- 只要未启用多进程服务（MPS），就允许协同组启动。

Memcpy 节点：

- 只允许涉及设备内存和/或固定内存（pinned host memory，即已映射到设备地址空间的页锁定主机内存）的拷贝操作。
- 不允许涉及 CUDA 数组（CUDA array）的拷贝。
- 实例化时，源地址和目标地址都必须能够从当前设备访问。
  需要注意的是，即使该拷贝操作的目标是另一块设备上的内存，实际的拷贝操作也会由该图所在的设备来执行。

.. _cuda-graphs-device-graph-upload:

4.2.6.1.2. 设备图加载
``````````````````````

为了在设备端启动图，必须先将其加载到设备，以分配必要的设备资源。这可以通过以下两种方式之一实现。

- 显式地上传。调用 ``cudaGraphUpload()`` 或者在调用 ``cudaGraphInstantiateWithParams()`` 进行实例化时一并请求上传。
- 主机端启动该图。这会在启动过程中隐式地执行这一上传步骤。

所有三种方法的示例：

.. code-block:: c++

   // Explicit upload after instantiation
   cudaGraphInstantiate(&deviceGraphExec1, deviceGraph1, cudaGraphInstantiateFlagDeviceLaunch);
   cudaGraphUpload(deviceGraphExec1, stream);

   // Explicit upload as part of instantiation
   cudaGraphInstantiateParams instantiateParams = {0};
   instantiateParams.flags = cudaGraphInstantiateFlagDeviceLaunch | cudaGraphInstantiateFlagUpload;
   instantiateParams.uploadStream = stream;
   cudaGraphInstantiateWithParams(&deviceGraphExec2, deviceGraph2, &instantiateParams);

   // Implicit upload via host launch
   cudaGraphInstantiate(&deviceGraphExec3, deviceGraph3, cudaGraphInstantiateFlagDeviceLaunch);
   cudaGraphLaunch(deviceGraphExec3, stream);

.. _cuda-graphs-device-graph-update:

4.2.6.1.3. 设备图更新
``````````````````````

设备图仅能在主机端进行更新；若已实例化的可执行图发生更新，必须重新将其上传至设备，修改才能生效
这可以通过 :ref:`cuda-graphs-device-graph-upload` 中方法来实现。
与主机图不同的是，如果在更新过程中从设备端启动设备图，将导致未定义的行为。

.. _cuda-graphs-device-launch:

4.2.6.2. 设备图启动
~~~~~~~~~~~~~~~~~~~

设备图可以从主机端或设备端通过 ``cudaGraphLaunch()`` 函数来启动，该函数在设备端与主机端的签名相同。
设备图在主机端和设备端均通过相同的句柄进行启动。
当从设备端启动设备图时，必须由另一个图来发起启动。

设备端图的启动操作是以线程为单位独立执行的，不同线程可同时发起多次图执行；用户需要指定由单个线程来负责启动某个特定的图。

与主机端启动不同，设备图不能启动到常规的 CUDA 流中，而只能启动到特定的流中，每种流代表一种特定的启动模式。
下表列出了可用的启动模式。

.. table:: 仅设备的图启动流

   =========================================== ==================
   流                                          启动模式
   =========================================== ==================
   ``cudaStreamGraphFireAndForget``            即发即弃启动
   ``cudaStreamGraphTailLaunch``               尾部启动
   ``cudaStreamGraphFireAndForgetAsSibling``   同级启动
   =========================================== ==================

.. _cuda-graphs-fire-and-forget-launch:

4.2.6.2.1. 即发即弃启动
````````````````````````

顾名思义，即发即弃启动（Fire and Forget Launch）被立即提交到 GPU，并且它独立于启动图运行。在即发即弃场景中，启动图是父图，启动的图是子图。

.. figure:: /_static/images/fire-and-forget-simple.png
   :alt: Fire and forget launch
   :scale: 80%
   :align: center

   即发即弃启动

示例代码：

.. code-block:: c++

   __global__ void launchFireAndForgetGraph(cudaGraphExec_t graph) {
       cudaGraphLaunch(graph, cudaStreamGraphFireAndForget);
   }

   void graphSetup() {
       cudaGraphExec_t gExec1, gExec2;
       cudaGraph_t g1, g2;

       // Create, instantiate, and upload the device graph.
       create_graph(&g2);
       cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch);
       cudaGraphUpload(gExec2, stream);

       // Create and instantiate the launching graph.
       cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
       launchFireAndForgetGraph<<<1, 1, 0, stream>>>(gExec2);
       cudaStreamEndCapture(stream, &g1);
       cudaGraphInstantiate(&gExec1, g1);

       // Launch the host graph, which will in turn launch the device graph.
       cudaGraphLaunch(gExec1, stream);
   }

一张图在其单次完整执行周期内，最多可发起总计 120 个即发即弃子图。
当同一张父图重新发起执行时，该计数会清零重置。

.. _cuda-graphs-graph-execution-environments:

4.2.6.2.1.1. 图执行环境
************************

要深入理解设备端同步模型，首先必须掌握 **执行环境** 的概念。

从设备端发起图执行时，该图会运行在专属执行环境中。
某张图的执行环境会封装图内所有任务，以及由此产生的全部即发即弃任务。
只有当图自身执行完毕、且所有衍生的子图任务全部完成时，该图才算真正执行完毕。

下图展示了上一节中即发即弃示例代码所生成的执行环境封装关系。

.. figure:: /_static/images/fire-and-forget-environments.png
   :alt: Fire and forget launch, with execution environments
   :scale: 70%
   :align: center

   即发即弃启动，带有执行环境

这些执行环境也是分层的，因此一个图的执行环境可以包含由即发即弃启动所产生的多层子环境。

.. figure:: /_static/images/fire-and-forget-nested-environments.png
   :alt: Nested fire and forget environments
   :scale: 70%
   :align: center

   嵌套的即发即弃环境

当图从主机启动时，存在一个流环境，它是启动图的执行环境的父环境。流环境封装了作为整体启动一部分生成的所有工作。当整体流环境标记为完成时，流启动完成（即下游依赖工作现在可以运行）。

当从主机端启动一个图时，会存在一个流环境（Stream Environment），作为被启动子图的执行环境的父级。
该流环境封装了子图启动过程中所生成的所有工作。
当整个流环境被标记为完成时，流启动才算完成（即此时下游的依赖工作才可以开始运行）。

.. figure:: /_static/images/device-graph-stream-environment.png
   :alt: The stream environment, visualized
   :scale: 70%
   :align: center

   流环境可视化

.. _cuda-graphs-tail-launch:

4.2.6.2.2. 尾部启动
````````````````````

与主机端不同，无法通过传统接口如 ``cudaDeviceSynchronize()`` 或 ``cudaStreamSynchronize()`` 在 GPU 上完成设备图执行同步。
为了实现串行任务依赖，CUDA 提供了另一种执行模式 —— 尾部启动（tail launch），用以提供相近的同步能力。

尾部启动在图的环境被认为完成时执行——即当图及其所有子图完成时。当图完成时，尾部启动列表中下一个图的环境将替换已完成的环境作为父环境的子环境。与即发即忘启动一样，一个图可以有多个图排队进行尾部启动。

尾部启动会在一个图的执行环境完全完成时（当该图及其所有子图都执行完毕）才会触发下一个图执行。
当一个图执行完成时，尾部启动列表中下一个图的环境将作为父环境的子环境，接替刚刚完成的那个图的环境。
与即发即弃启动类似，单张图可以向尾部启动队列加入多张待执行子图。

.. figure:: /_static/images/tail-launch-simple.png
   :alt: A simple tail launch
   :scale: 80%
   :align: center

   简单的尾部启动

示例代码：

.. code-block:: c++

   __global__ void launchTailGraph(cudaGraphExec_t graph) {
       cudaGraphLaunch(graph, cudaStreamGraphTailLaunch);
   }

   void graphSetup() {
       cudaGraphExec_t gExec1, gExec2;
       cudaGraph_t g1, g2;

       // Create, instantiate, and upload the device graph.
       create_graph(&g2);
       cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch);
       cudaGraphUpload(gExec2, stream);

       // Create and instantiate the launching graph.
       cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
       launchTailGraph<<<1, 1, 0, stream>>>(gExec2);
       cudaStreamEndCapture(stream, &g1);
       cudaGraphInstantiate(&gExec1, g1);

       // Launch the host graph, which will in turn launch the device graph.
       cudaGraphLaunch(gExec1, stream);
   }

由同一张图提交的尾部启动将按入队顺序逐个执行。也就是说，最先入队的图会最先运行，接着是第二个，依此类推。

.. figure:: /_static/images/tail-launch-ordering-simple.png
   :alt: Tail launch ordering
   :scale: 60%
   :align: center

   尾部启动排序

如下图所示，图 `G1` 在执行时向尾部启动队列提交 `G2` 和 `G3` 的启动请求。
图 `G2` 在执行时，生成新的尾部启动子图 `X` 和 `Y`。
虽然 `G3` 已经存在于尾部启动队列中，但是 `X` 和 `Y` 会先与 `G3` 执行。
因为 `G3` 在 `G2` **完全** 执行完成之后才能执行。 **只有由同一张图提交的尾部启动才按入队顺序逐个执行。**

.. figure:: /_static/images/tail-launch-ordering-complex.png
   :alt: Tail launch ordering when enqueued from multiple graphs
   :scale: 60%
   :align: center

   多图入队场景下尾随执行的调度顺序

一个图最多可以拥有 255 个待执行的尾部启动。

.. _cuda-graphs-tail-self-launch:

4.2.6.2.2.1. 尾部自启动
************************

设备图可以将自己提交至尾部启动队列，尽管同一个图在同一时间只能有一个自启动排队。
为了查询当前正在运行的设备图以便重新启动，系统添加了一个新的设备端函数：

.. code-block:: c++

   cudaGraphExec_t cudaGetCurrentGraphExec();

如果当前正在运行的是设备图，该函数将返回其句柄。如果当前正在执行的 kernel 并非设备图中的节点，该函数将返回 NULL。

下方为示例代码，展示如何使用该函数实现循环重新执行逻辑。

.. code-block:: c++

   __device__ int relaunchCount = 0;

   __global__ void relaunchSelf() {
       int relaunchMax = 100;

       if (threadIdx.x == 0) {
           if (relaunchCount < relaunchMax) {
               cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
           }

           relaunchCount++;
       }
   }

.. _cuda-graphs-sibling-launch:

4.2.6.2.3. 兄弟启动
````````````````````

兄弟启动（Sibling Launch）是即发即弃启动的一种变体。
在这种模式下，被启动的图不会作为启动图执行环境的子环境，而是作为启动图的父环境的子环境。
兄弟启动等效于从启动图的父环境发起的即发即弃启动。

.. figure:: /_static/images/sibling-launch-simple.png
   :alt: A simple sibling launch
   :align: center

   简单的兄弟启动

示例代码：

.. code-block:: c++

   __global__ void launchSiblingGraph(cudaGraphExec_t graph) {
       cudaGraphLaunch(graph, cudaStreamGraphFireAndForgetAsSibling);
   }

   void graphSetup() {
       cudaGraphExec_t gExec1, gExec2;
       cudaGraph_t g1, g2;

       // Create, instantiate, and upload the device graph.
       create_graph(&g2);
       cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch);
       cudaGraphUpload(gExec2, stream);

       // Create and instantiate the launching graph.
       cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
       launchSiblingGraph<<<1, 1, 0, stream>>>(gExec2);
       cudaStreamEndCapture(stream, &g1);
       cudaGraphInstantiate(&gExec1, g1);

       // Launch the host graph, which will in turn launch the device graph.
       cudaGraphLaunch(gExec1, stream);
   }

由于兄弟启动不会被启动到发起图的执行环境中，因此它们不会阻塞（或拦截）由该发起图提交的尾部启动队列。

.. _cuda-graphs-using-graph-apis:

4.2.7. 使用图 API
-----------------

``cudaGraph_t`` 对象不是线程安全的。用户有责任确保多个线程不会同时访问同一个 ``cudaGraph_t`` 。

同一个 ``cudaGraphExec_t`` 实例不能并发运行。
对同一个可执行图的启动操作，将严格排在它之前的启动操作之后执行。

图执行和其他的异步任务在流中按序执行。
但该流仅用于管控该流上任务的执行顺序；不会限制图内部的并行能力，也不会改变图中各节点的执行硬件位置。

请参阅 `图 API <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH>`_ 。

.. _cuda-graphs-cuda-user-objects:

4.2.8. CUDA 用户对象
--------------------

CUDA 用户对象（CUDA User Objects）可用于辅助管理 CUDA 异步任务所使用资源的生命周期。该特性对 CUDA 图以及流捕获场景尤为实用。

各种资源管理方案并不总是与 CUDA 图兼容。例如，基于事件的资源池（event-based pool），或是 `同步创建、异步销毁` 的方案。

.. code-block:: c++

   // Library API with pool allocation
   void libraryWork(cudaStream_t stream) {
       auto &resource = pool.claimTemporaryResource();
       resource.waitOnReadyEventInStream(stream);
       launchWork(stream, resource);
       resource.recordReadyEvent(stream);
   }

   // Library API with asynchronous resource deletion
   void libraryWork(cudaStream_t stream) {
       Resource *resource = new Resource(...);
       launchWork(stream, resource);
       cudaLaunchHostFunc(
           stream,
           [](void *resource) {
               delete static_cast<Resource *>(resource);
           },
           resource,
           0);
       // Error handling considerations not shown
   }

这类方案难以兼容 CUDA 图场景。
一方面资源的指针或句柄并非固定值，需要间接寻址或更新图；另一方面每次提交任务都要执行同步 CPU 代码。
这类方案也无法适配流捕获场景，调用者不感知实现细节，而流捕获阶段调用了禁用 API。
目前已有多种解决方案，例如向调用者直接暴露资源句柄。而 CUDA 用户对象则提供了一种全新的解决思路。

CUDA 用户对象将用户自定义的析构回调函数与内部引用计数绑定，机制类似于 C++ 的 ``shared_ptr`` 。
引用可由 CPU 侧用户代码持有，也可由 CUDA 图持有。
需要注意，对于用户持有的引用，和 C++ 智能指针不同，不存在专门对象来代表该引用；用户必须手动维护自身持有的引用。
典型使用场景为，创建用户对象后，立刻将唯一的用户持有引用转移至 CUDA 图。

当引用与 CUDA 图绑定时，CUDA 会自动管理图中的相关操作。
被克隆的 ``cudaGraph_t`` 会保留源 ``cudaGraph_t`` 所拥有的每个引用的副本，且保持相同的数量（多重性）。
被实例化的 ``cudaGraphExec_t`` 也会保留源 ``cudaGraph_t`` 中每个引用的副本。
如果 ``cudaGraphExec_t`` 在未进行同步的情况下就被销毁，这些引用会被继续保留，直到图的执行彻底完成。

使用示例：

.. code-block:: c++

   cudaGraph_t graph;  // pre-existing graph

   Object *object = new Object;  // C++ object with possibly non-trivial destructor
   cudaUserObject_t cuObject;
   cudaUserObjectCreate(
       &cuObject,
       object,  // here we use CUDA's provided template wrapper for this API,
                // which provides a callback to delete the C++ object pointer
       1,  // initial reference count
       cudaUserObjectNoDestructorSync  // acknowledge the callback cannot be waited on by CUDA
   );
   cudaGraphRetainUserObject(
       graph,
       cuObject,
       1,  // number of references
       cudaGraphUserObjectMove  // transfer caller-owned reference (does not
                                // modify total reference count)
   );
   // This thread no longer owns a reference; no need to call release API
   cudaGraphExec_t graphExec;
   cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);  // will retain
                                                                  // new reference
   cudaGraphDestroy(graph);  // graphExec still owns a reference
   cudaGraphLaunch(graphExec, 0);  // async launch can access user object
   cudaGraphExecDestroy(graphExec);  // launch is unsynchronized; if needed, release
                                     // will be deferred
   cudaStreamSynchronize(0);  // After synchronizing the launch, the remaining
                              // references are released and the destructor will
                              // execute. Note this happens asynchronously.
   // If the destructor callback signaled a synchronization object, at this point
   // it is safe to wait on it.

由子图所持有的引用，是归属于子图本身的，而不是父图。
如果子图被更新或删除，其持有的引用也会发生相应的变化。
如果使用 ``cudaGraphExecUpdate`` 或 ``cudaGraphExecChildGraphNodeSetParams`` 更新可执行图或子图，新图中的引用将被克隆，并替换目标图中原有的引用。
任何情况下，如果之前的启动操作尚未完成同步，任何本应被释放的引用都会被继续保留，直到这些启动操作彻底执行完毕。

目前没有可供调用的 CUDA 接口，能够等待用户对象的析构回调执行完毕。
开发者可以在析构函数代码中手动发送同步对象信号。
此外，不允许在析构回调内部调用任何 CUDA 接口，该限制与 ``cudaLaunchHostFunc`` 的约束一致。
此举是为了避免阻塞 CUDA 内部共享线程，进而导致整个 GPU 任务停滞无法推进。
但允许通过信号通知其他线程执行 CUDA 接口调用，前提是依赖关系为单向，且执行接口调用的线程不会阻碍 CUDA 任务的正常推进。

用户对象通过 ``cudaUserObjectCreate`` 函数创建，这也是一个了解相关 API 的良好切入点。
