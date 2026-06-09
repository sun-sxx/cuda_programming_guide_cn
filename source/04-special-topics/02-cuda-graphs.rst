.. _cuda-graphs-details:

4.2. CUDA Graphs
================

CUDA Graphs 提出了另一种 CUDA 工作提交模型。图（graph）是一系列操作（如内核启动、数据移动等）通过依赖关系连接而成，其定义与执行分离。这使得图可以定义一次，然后重复启动多次。将图的定义与执行分离实现了许多优化：首先，与流相比，CPU 启动开销降低了，因为大部分准备工作都是提前完成的；其次，将整个工作流程呈现给 CUDA 可以实现一些使用流的分段工作提交机制无法实现的优化。

要了解图可以实现的优化，可以考虑流中发生的事情：当你将内核放入流中时，主机驱动程序会执行一系列操作来准备在 GPU 上执行内核。这些操作对于设置和启动内核是必要的，是每次发出内核时都必须支付的开销成本。对于执行时间较短的 GPU 内核，此开销成本可能是整体端到端执行时间的很大一部分。通过创建涵盖将多次启动的工作流程的 CUDA 图，这些开销成本可以在实例化期间为整个图支付一次，然后图本身可以以非常小的开销重复启动。

.. _cuda-graphs-graph-structure:

4.2.1. 图结构
-------------

操作在图中形成节点（node）。操作之间的依赖关系是边（edge）。这些依赖关系约束了操作的执行顺序。

一旦操作所依赖的节点完成，就可以随时调度该操作。调度由 CUDA 系统负责。

.. _cuda-graphs-node-types:

4.2.1.1. 节点类型
~~~~~~~~~~~~~~~~~

图节点可以是以下类型之一：

- 内核（kernel）
- CPU 函数调用
- 内存拷贝（memory copy）
- 内存填充（memset）
- 空节点（empty node）
- 等待 CUDA 事件（waiting on a CUDA Event）
- 记录 CUDA 事件（recording a CUDA Event）
- 发出外部信号量（signalling an external semaphore）
- 等待外部信号量（waiting on an external semaphore）
- 条件节点（conditional node）
- 内存节点（memory node）
- 子图（child graph）：执行单独的嵌套图，如下图所示。

.. figure:: https://docs.nvidia.com/cuda/cuda-programming-guide/_images/child-graph-example.png
   :alt: 子图示例

   图 21 子图示例

.. _cuda-graphs-edge-data:

4.2.1.2. 边数据
~~~~~~~~~~~~~~~

CUDA 12.3 在 CUDA Graphs 中引入了边数据（edge data）。目前，非默认边数据的唯一用途是支持编程式依赖启动（Programmatic Dependent Launch）。

一般而言，边数据修改由边指定的依赖关系，由三部分组成：输出端口（outgoing port）、输入端口（incoming port）和类型（type）。输出端口指定关联的边何时被触发。输入端口指定节点的哪一部分依赖于关联的边。类型修改端点之间的关系。

端口值特定于节点类型和方向，边类型可能仅限于特定节点类型。在所有情况下，零初始化的边数据表示默认行为。输出端口 0 等待整个任务，输入端口 0 阻塞整个任务，边类型 0 与具有内存同步行为的完整依赖关系相关联。

边数据在各种图 API 中通过关联节点的并行数组可选地指定。如果作为输入参数省略，则使用零初始化的数据。如果作为输出（查询）参数省略，如果忽略的边数据全部为零初始化，则 API 接受此情况；如果调用将丢弃信息，则返回 cudaErrorLossyQuery。

某些流捕获 API 中也提供边数据：cudaStreamBeginCaptureToGraph()、cudaStreamGetCaptureInfo() 和 cudaStreamUpdateCaptureDependencies()。在这些情况下，还没有下游节点。数据与悬空边（半条边）相关联，该边将连接到未来的捕获节点或在流捕获终止时被丢弃。

请注意，某些边类型不会等待上游节点完全完成。在考虑流捕获是否已完全重新加入原始流时，这些边将被忽略，并且在捕获结束时不能被丢弃。参见流捕获（Stream Capture）。

没有节点类型定义额外的输入端口，只有内核节点定义额外的输出端口。有一个非默认依赖类型 cudaGraphDependencyTypeProgrammatic，用于在两个内核节点之间启用编程式依赖启动。

.. _cuda-graphs-build-and-run:

4.2.2. 构建和运行图
-------------------

使用图进行工作提交分为三个不同的阶段：定义（definition）、实例化（instantiation）和执行（execution）。

- 在定义或创建阶段，程序创建图中操作的描述以及它们之间的依赖关系。
- 实例化获取图模板的快照，验证它，并执行大量的工作设置和初始化，目的是最小化启动时需要完成的工作。生成的实例称为可执行图（executable graph）。
- 可执行图可以启动到流中，类似于任何其他 CUDA 工作。它可以启动任意次数，而无需重复实例化。

.. _cuda-graphs-graph-creation:

4.2.2.1. 图创建
~~~~~~~~~~~~~~~

图可以通过两种机制创建：使用显式图 API 和通过流捕获。

4.2.2.1.1. 图 API
`````````````````

以下是一个创建下图的示例（省略声明和其他样板代码）。注意使用 cudaGraphCreate() 创建图，使用 cudaGraphAddNode() 添加内核节点及其依赖关系。CUDA Runtime API 文档列出了所有可用于添加节点和依赖关系的函数。

.. figure:: https://docs.nvidia.com/cuda/cuda-programming-guide/_images/creating-graph-using-apis.png
   :alt: 使用图 API 创建图的示例

   图 22 使用图 API 创建图的示例

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

上面的示例展示了四个内核节点及其之间的依赖关系，以说明非常简单图的创建。在典型的用户应用程序中，还需要为内存操作添加节点，例如 cudaGraphAddMemcpyNode() 等。有关添加节点的所有图 API 函数的完整参考，请参阅 CUDA Runtime API 文档。

4.2.2.1.2. 流捕获
`````````````````

流捕获提供了一种从现有基于流的 API 创建图的机制。可以使用 cudaStreamBeginCapture() 和 cudaStreamEndCapture() 调用来框住启动工作到流中的代码段（包括现有代码）。如下所示：

.. code-block:: cpp

   cudaGraph_t graph;

   cudaStreamBeginCapture(stream);

   kernel_A<<< ..., stream >>>(...);
   kernel_B<<< ..., stream >>>(...);
   libraryCall(stream);
   kernel_C<<< ..., stream >>>(...);

   cudaStreamEndCapture(stream, &graph);

调用 cudaStreamBeginCapture() 会将流置于捕获模式。当流被捕获时，启动到该流中的工作不会排队执行。而是附加到正在逐步构建的内部图中。然后通过调用 cudaStreamEndCapture() 返回此图，同时也结束流的捕获模式。正在通过流捕获积极构建的图称为捕获图（capture graph）。

流捕获可用于任何 CUDA 流，除了 cudaStreamLegacy（"NULL 流"）。请注意，它可用于 cudaStreamPerThread。如果程序使用传统流，则可以重新定义流 0 为每线程流，而不会发生功能变化。请参阅阻塞和非阻塞流以及默认流。

可以使用 cudaStreamIsCapturing() 查询流是否正在被捕获。

可以使用 cudaStreamBeginCaptureToGraph() 将工作捕获到现有图中。工作不是捕获到内部图，而是捕获到用户提供的图。

4.2.2.1.2.1. 跨流依赖和事件
***************************

流捕获可以处理使用 cudaEventRecord() 和 cudaStreamWaitEvent() 表示的跨流依赖关系，前提是所等待的事件已记录到同一捕获图中。

当在处于捕获模式的流中记录事件时，会产生捕获事件（captured event）。捕获事件表示捕获图中的一组节点。

当流等待捕获事件时，如果流尚未处于捕获模式，它会将其置于捕获模式，并且流中的下一项将对捕获事件中的节点具有额外的依赖关系。然后两个流被捕获到同一个捕获图。

当流捕获中存在跨流依赖关系时，仍必须在 cudaStreamBeginCapture() 被调用的同一流中调用 cudaStreamEndCapture()；这是原始流（origin stream）。由于基于事件的依赖关系而被捕获到同一捕获图的任何其他流也必须重新加入原始流。如下图所示。所有被捕获到同一捕获图的流在 cudaStreamEndCapture() 时都会退出捕获模式。如果未能重新加入原始流，将导致整个捕获操作失败。

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

上述代码返回的图如图 22 所示。

.. note::
   当流退出捕获模式时，流中的下一个非捕获项（如果有）仍将依赖于最近的先前非捕获项，尽管中间项已被移除。

4.2.2.1.2.2. 禁止和未处理的操作
*******************************

同步或查询正在被捕获的流或捕获事件的执行状态是无效的，因为它们不代表已调度执行的项目。查询或同步包含活动流捕获的更广泛句柄（例如当任何关联流处于捕获模式时的设备或上下文句柄）也是无效的。

当同一上下文中的任何流正在被捕获，并且它不是使用 cudaStreamNonBlocking 创建时，任何尝试使用传统流都是无效的。这是因为传统流句柄始终包含这些其他流；入队到传统流将创建对正在被捕获的流的依赖关系，查询或同步它将查询或同步正在被捕获的流。

因此，在这种情况下调用同步 API 也是无效的。同步 API 的一个示例是 cudaMemcpy()，它在返回之前将工作入队到传统流并同步。

.. note::
   作为一般规则，当依赖关系连接被捕获的内容和未被捕获而是入队执行的内容时，CUDA 倾向于返回错误而不是忽略依赖关系。对于将流置于或移出捕获模式有一个例外；这切断了在模式转换之前和之后立即添加到流的项目之间的依赖关系。

通过等待来自正在被捕获的流的捕获事件来合并两个单独的捕获图是无效的，该流与事件关联的捕获图不同。在不指定 cudaEventWaitExternal 标志的情况下，从正在被捕获的流等待非捕获事件是无效的。

少数将异步操作入队到流中的 API 当前不支持图，如果使用正在被捕获的流调用它们将返回错误，例如 cudaStreamAttachMemAsync()。

4.2.2.1.2.3. 失效
*****************

当在流捕获期间尝试无效操作时，任何关联的捕获图都将失效。当捕获图失效时，任何正在被捕获的流或捕获事件的进一步使用都是无效的，并将返回错误，直到使用 cudaStreamEndCapture() 结束流捕获。此调用将使关联的流退出捕获模式，但也将返回错误值和 NULL 图。

4.2.2.1.2.4. 捕获内省
*********************

可以使用 cudaStreamGetCaptureInfo() 检查活动流捕获操作。这允许用户获取捕获状态、捕获的唯一（每进程）ID、底层图对象，以及流中要捕获的下一个节点的依赖关系/边数据。此依赖关系信息可用于获取上一个捕获在流中的节点的句柄。

4.2.2.1.3. 综合示例
```````````````````

图 22 中的示例是一个简单的示例，旨在概念性地展示一个小图。在利用 CUDA 图的应用程序中，使用图 API 或流捕获会有更多的复杂性。以下代码片段并排展示了图 API 和流捕获，以创建执行简单两阶段归约算法的 CUDA 图。

图 23 是此 CUDA 图的插图，使用 cudaGraphDebugDotPrint 函数应用于以下代码生成，并进行了小的调整以提高可读性，然后使用 Graphviz 渲染。

.. figure:: https://docs.nvidia.com/cuda/cuda-programming-guide/_images/cuda-graph-example-reduction.png
   :alt: 使用两阶段归约内核的 CUDA 图示例

   图 23 使用两阶段归约内核的 CUDA 图示例

**图 API：**

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

      cudaStreamCreate(&streamForGraph));

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

      cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(), nodeDependencies.size(), &memcpyParams);
      nodeDependencies.clear();
      nodeDependencies.push_back(memcpyNode);

      cudaGraphNode_t    hostNode;
      cudaHostNodeParams hostParams = {0};
      hostParams.fn                 = myHostNodeCallback;
      callBackData_t hostFnData;
      hostFnData.data     = &result_h;
      hostFnData.fn_name  = "cudaGraphsManual";
      hostParams.userData = &hostFnData;

      cudaGraphAddHostNode(&hostNode, graph, nodeDependencies.data(), nodeDependencies.size(), &hostParams);
   }

**流捕获：**

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

一旦创建了图（通过使用图 API 或流捕获），就必须实例化该图以创建可执行图，然后可以启动它。假设 cudaGraph_t graph 已成功创建，以下代码将实例化图并创建可执行图 cudaGraphExec_t graphExec：

.. code-block:: cpp

   cudaGraphExec_t graphExec;
   cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

.. _cuda-graphs-graph-execution:

4.2.2.3. 图执行
~~~~~~~~~~~~~~~

在创建和实例化图以创建可执行图之后，可以启动它。假设 cudaGraphExec_t graphExec 已成功创建，以下代码片段将图启动到指定的流中：

.. code-block:: cpp

   cudaGraphLaunch(graphExec, stream);

综合起来，使用 4.2.2.1.2 节中的流捕获示例，以下代码片段将创建图、实例化它并启动它：

.. code-block:: cpp

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

当工作流程发生变化时，图就会过时，必须修改。图结构的重大更改（如拓扑或节点类型）需要重新实例化，因为必须重新应用与拓扑相关的优化。然而，通常只有节点参数（如内核参数和内存地址）发生变化，而图拓扑保持不变。对于这种情况，CUDA 提供了一种轻量级的"图更新"机制，允许就地修改某些节点参数，而无需重建整个图，这比重新实例化要高效得多。

更新在图下次启动时生效，因此它们不会影响之前的图启动，即使在更新时它们正在运行。图可以重复更新和重新启动，因此多个更新/启动可以在流上排队。

CUDA 提供了两种更新实例化图参数的机制：整图更新（whole graph update）和单个节点更新（individual node update）。整图更新允许用户提供拓扑相同的 cudaGraph_t 对象，其节点包含更新的参数。单个节点更新允许用户显式更新单个节点的参数。当大量节点正在更新或调用者不知道图拓扑时（即图来自库调用的流捕获），使用更新的 cudaGraph_t 更方便。当更改数量很少且用户拥有需要更新的节点的句柄时，首选单个节点更新。单个节点更新跳过未更改节点的拓扑检查和比较，因此在许多情况下可能更高效。

CUDA 还提供了一种启用和禁用单个节点而不影响其当前参数的机制。

以下章节更详细地解释每种方法。

.. _cuda-graphs-whole-graph-update:

4.2.3.1. 整图更新
~~~~~~~~~~~~~~~~~

cudaGraphExecUpdate() 允许使用拓扑相同的图（"更新"图）的参数更新实例化的图（"原始"图）。更新图的拓扑必须与用于实例化 cudaGraphExec_t 的原始图相同。此外，指定依赖关系的顺序必须匹配。最后，CUDA 需要一致地排序汇点节点（没有依赖关系的节点）。CUDA 依赖于特定 API 调用的顺序来实现一致的汇点节点排序。

更明确地说，遵循以下规则将使 cudaGraphExecUpdate() 确定性地配对原始图和更新图中的节点：

1. 对于任何捕获流，在该流上运行的 API 调用必须按相同的顺序进行，包括事件等待和其他不直接对应于节点创建的 API 调用。
2. 直接操作给定图节点传入边的 API 调用（包括捕获流 API、节点添加 API 和边添加/移除 API）必须按相同的顺序进行。此外，当在这些 API 的数组中指定依赖关系时，数组内指定依赖关系的顺序必须匹配。
3. 汇点节点必须一致地排序。汇点节点是在调用 cudaGraphExecUpdate() 时在最终图中没有依赖节点/传出边的节点。以下操作影响汇点节点排序（如果存在），并且必须（作为组合集）按相同的顺序进行：

   - 导致汇点节点的节点添加 API。
   - 导致节点成为汇点节点的边移除。
   - cudaStreamUpdateCaptureDependencies()（如果它从捕获流的依赖关系集中移除汇点节点）。
   - cudaStreamEndCapture()。

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

典型的工作流程是使用流捕获或图 API 创建初始 cudaGraph_t。然后实例化 cudaGraph_t 并正常启动。在初始启动之后，使用与初始图相同的方法创建新的 cudaGraph_t，并调用 cudaGraphExecUpdate()。如果图更新成功（由上述示例中的 updateResult 参数指示），则启动更新的 cudaGraphExec_t。如果更新因任何原因失败，则调用 cudaGraphExecDestroy() 和 cudaGraphInstantiate() 来销毁原始 cudaGraphExec_t 并实例化新的图。

也可以直接更新 cudaGraph_t 节点（即使用 cudaGraphKernelNodeSetParams()），然后更新 cudaGraphExec_t，但是使用下一节中介绍的显式节点更新 API 更高效。

条件句柄标志和默认值作为图更新的一部分进行更新。

请参阅图 API 以获取有关用法和当前限制的更多信息。

.. _cuda-graphs-single-node-update:

4.2.3.2. 单个节点更新
~~~~~~~~~~~~~~~~~~~~~

实例化的图节点参数可以直接更新。这消除了实例化的开销以及创建新 cudaGraph_t 的开销。如果需要更新的节点数量相对于图中的节点总数较小，则最好单独更新节点。以下方法可用于更新 cudaGraphExec_t 节点：

**表 8 单个节点更新 API**

- cudaGraphExecKernelNodeSetParams(): 内核节点
- cudaGraphExecMemcpyNodeSetParams(): 内存拷贝节点
- cudaGraphExecMemsetNodeSetParams(): 内存填充节点
- cudaGraphExecHostNodeSetParams(): 主机节点
- cudaGraphExecChildGraphNodeSetParams(): 子图节点
- cudaGraphExecEventRecordNodeSetEvent(): 事件记录节点
- cudaGraphExecEventWaitNodeSetEvent(): 事件等待节点
- cudaGraphExecExternalSemaphoresSignalNodeSetParams(): 外部信号量发出节点
- cudaGraphExecExternalSemaphoresWaitNodeSetParams(): 外部信号量等待节点

请参阅图 API 以获取有关用法和当前限制的更多信息。

.. _cuda-graphs-single-node-enable:

4.2.3.3. 单个节点启用
~~~~~~~~~~~~~~~~~~~~~

实例化图中的内核、内存填充和内存拷贝节点可以使用 cudaGraphNodeSetEnabled() API 启用或禁用。这允许创建包含所需功能超集的图，可以为每次启动自定义。可以使用 cudaGraphNodeGetEnabled() API 查询节点的启用状态。

禁用的节点在功能上等同于空节点，直到它被重新启用。节点的参数不受启用/禁用节点的影响。启用状态不受单个节点更新或使用 cudaGraphExecUpdate() 的整图更新的影响。节点禁用时的参数更新将在节点重新启用时生效。

请参阅图 API 以获取有关用法和当前限制的更多信息。

.. _cuda-graphs-update-limitations:

4.2.3.4. 图更新限制
~~~~~~~~~~~~~~~~~~~

**内核节点：**

- 函数的所属上下文不能更改。
- 原本不使用 CUDA 动态并行的函数节点不能更新为使用 CUDA 动态并行的函数。

**cudaMemset 和 cudaMemcpy 节点：**

- 分配/映射操作数的 CUDA 设备不能更改。
- 源/目标内存必须与原始源/目标内存从同一上下文分配。
- 只能更改 1D cudaMemset/cudaMemcpy 节点。

**额外的 memcpy 节点限制：**

- 不支持更改源或目标内存类型（即 cudaPitchedPtr、cudaArray_t 等）或传输类型（即 cudaMemcpyKind）。

**外部信号量等待节点和记录节点：**

- 不支持更改信号量数量。

**条件节点：**

- 句柄创建和分配的顺序在图之间必须匹配。
- 不支持更改节点参数（即条件中的图数量、节点上下文等）。
- 更改条件体图内节点的参数受上述规则约束。

**内存节点：**

- 如果 cudaGraph_t 当前实例化为不同的 cudaGraphExec_t，则无法使用 cudaGraph_t 更新 cudaGraphExec_t。

对主机节点、事件记录节点或事件等待节点的更新没有限制。

.. _cuda-graphs-conditional-nodes:

4.2.4. 条件图节点
-----------------

条件节点允许条件执行和循环条件节点内包含的图。这允许动态和迭代工作流完全在图内表示，并释放主机 CPU 并行执行其他工作。

条件值的评估在设备上执行，当条件节点的依赖关系已满足时。条件节点可以是以下类型之一：

- **条件 IF 节点**：如果执行节点时条件值非零，则执行其体图一次。可以提供可选的第二个主体图，如果执行节点时条件值为零，则该图将执行一次。
- **条件 WHILE 节点**：如果执行节点时条件值非零，则执行其体图，并将继续执行其体图，直到条件值为零。
- **条件 SWITCH 节点**：如果条件值等于 n，则执行零索引的第 n 个体图一次。如果条件值不对应于体图，则不启动体图。

条件值通过条件句柄访问，该句柄必须在节点之前创建。条件值可以使用 cudaGraphSetConditional() 由设备代码设置。默认值（在每次图启动时应用）也可以在创建句柄时指定。

创建条件节点时，将创建一个空图，并将句柄返回给用户，以便可以填充图。此条件体图可以使用图 API 或 cudaStreamBeginCaptureToGraph() 填充。

条件节点可以嵌套。

.. _cuda-graphs-conditional-handles:

4.2.4.1. 条件句柄
~~~~~~~~~~~~~~~~~

条件值由 cudaGraphConditionalHandle 表示，并由 cudaGraphConditionalHandleCreate() 创建。

句柄必须与单个条件节点关联。句柄不能被销毁，因此无需跟踪它们。

如果在创建句柄时指定了 cudaGraphCondAssignDefault，则条件值将在每次图执行开始时初始化为指定的默认值。如果未提供此标志，则条件值在每次图执行开始时未定义，代码不应假设条件值在执行之间持续存在。

与句柄关联的默认值和标志将在整图更新期间更新。

.. _cuda-graphs-conditional-body-graph:

4.2.4.2. 条件节点体图要求
~~~~~~~~~~~~~~~~~~~~~~~~~

**一般要求：**

- 图的所有节点必须位于单个设备上。
- 图只能包含内核节点、空节点、内存拷贝节点、内存填充节点、子图节点和条件节点。

**内核节点：**

- 不允许图中的内核使用 CUDA 动态并行或设备图启动。
- 只要不使用 MPS，就允许协作启动。

**内存拷贝/内存填充节点：**

- 只允许涉及设备内存和/或固定设备映射主机内存的拷贝/填充。
- 不允许涉及 CUDA 数组的拷贝/填充。
- 在实例化时，两个操作数必须可从当前设备访问。请注意，拷贝操作将从图所在的设备执行，即使它针对另一个设备上的内存。

.. _cuda-graphs-conditional-if:

4.2.4.3. 条件 IF 节点
~~~~~~~~~~~~~~~~~~~~~

IF 节点的体图将在执行节点时如果条件非零则执行一次。下图描述了一个 3 节点图，其中中间节点 B 是条件节点：

.. figure:: https://docs.nvidia.com/cuda/cuda-programming-guide/_images/conditional-if-node.png
   :alt: 条件 IF 节点

   图 24 条件 IF 节点

以下代码说明了创建包含 IF 条件节点的图。条件的默认值使用上游内核设置。条件的主体使用图 API 填充。

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

       // Create the conditional handle; because no default value is provided, the condition value is undefined at the start of each graph execution
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

IF 节点也可以有一个可选的第二个主体图，当执行节点时如果条件值为零则执行一次。

.. code-block:: cpp

   void graphSetup() {
       cudaGraph_t graph;
       cudaGraphExec_t graphExec;
       cudaGraphNode_t node;
       void *kernelArgs[2];
       int value = 1;

       // Create the graph
       cudaGraphCreate(&graph, 0);

       // Create the conditional handle; because no default value is provided, the condition value is undefined at the start of each graph execution
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

4.2.4.4. 条件 WHILE 节点
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WHILE 节点的体图将只要条件非零就执行。条件将在执行节点时和体图完成后进行评估。下图描述了一个 3 节点图，其中中间节点 B 是条件节点：

.. figure:: https://docs.nvidia.com/cuda/cuda-programming-guide/_images/conditional-while-node.png
   :alt: 条件 WHILE 节点

   图 25 条件 WHILE 节点

以下代码说明了创建包含 WHILE 条件节点的图。使用 cudaGraphCondAssignDefault 创建句柄以避免需要上游内核。条件的主体使用图 API 填充。

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

4.2.4.5. 条件 SWITCH 节点
~~~~~~~~~~~~~~~~~~~~~~~~~

SWITCH 节点的零索引第 n 个体图将在执行节点时如果条件等于 n 则执行一次。下图描述了一个 3 节点图，其中中间节点 B 是条件节点：

.. figure:: https://docs.nvidia.com/cuda/cuda-programming-guide/_images/conditional-switch-node.png
   :alt: 条件 SWITCH 节点

   图 26 条件 SWITCH 节点

以下代码说明了创建包含 SWITCH 条件节点的图。条件值使用上游内核设置。条件的主体使用图 API 填充。

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

       // Create the conditional handle; because no default value is provided, the condition value is undefined at the start of each graph execution
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

图内存节点允许图创建和拥有内存分配。图内存节点具有 GPU 有序生命周期语义，它规定何时允许在设备上访问内存。这些 GPU 有序生命周期语义实现了驱动程序管理的内存重用，并与流有序分配 API cudaMallocAsync 和 cudaFreeAsync 匹配，这些 API 在创建图时可以被捕获。

图分配在图的整个生命周期内（包括重复实例化和启动）具有固定地址。这允许内存被图内的其他操作直接引用，而无需图更新，即使 CUDA 更改了后备物理内存。在图内，生命周期不重叠的分配可以使用相同的底层物理内存。

CUDA 可以在多个图之间重用相同的物理内存进行分配，根据 GPU 有序生命周期语义虚拟化地址映射。例如，当不同的图启动到同一流中时，CUDA 可以虚拟化地混叠相同的物理内存以满足具有单图生命周期的分配需求。

.. _cuda-graphs-memory-nodes-api:

4.2.5.2. API 基础
~~~~~~~~~~~~~~~~~

图内存节点是表示内存分配或释放操作的图节点。作为简写，分配内存的节点称为分配节点。同样，释放内存的节点称为释放节点。由分配节点创建的分配称为图分配。CUDA 在节点创建时为图分配分配虚拟地址。虽然这些虚拟地址在分配节点的整个生命周期内是固定的，但分配内容在释放操作之后不持久，并且可能被引用不同分配的访问覆盖。

图分配每次图运行时都被视为重新创建。图分配的生命周期（与节点的生命周期不同）开始于 GPU 执行到达分配图节点时，并在以下情况之一发生时结束：

- GPU 执行到达释放图节点
- GPU 执行到达释放 cudaFreeAsync() 流调用
- 立即在调用 cudaFree() 时

.. note::
   图销毁不会自动释放任何活动的图分配内存，即使它结束了分配节点的生命周期。分配必须在另一个图中随后释放，或使用 cudaFreeAsync()/cudaFree() 释放。

与其他图结构一样，图内存节点在图中通过依赖边排序。程序必须保证访问图内存的操作：

- 在分配节点之后排序
- 在释放内存的操作之前排序

图分配生命周期根据 GPU 执行开始和通常结束（与 API 调用相反）。GPU 排序是工作在 GPU 上运行的顺序，而不是工作入队或描述的顺序。因此，图分配被认为是"GPU 有序"的。

4.2.5.2.1. 图节点 API
`````````````````````

可以使用节点创建 API cudaGraphAddNode 显式创建图内存节点。添加 cudaGraphNodeTypeMemAlloc 节点时分配的地址在传递的 cudaGraphNodeParams 结构的 alloc::dptr 字段中返回给用户。在分配图内使用图分配的所有操作必须在分配节点之后排序。同样，任何释放节点必须在图内分配的所有使用之后排序。释放节点使用 cudaGraphAddNode 和 cudaGraphNodeTypeMemFree 节点类型创建。

在下图中，有一个带有分配和释放节点的示例图。内核节点 a、b 和 c 在分配节点之后排序并在释放节点之前排序，以便内核可以访问分配。内核节点 e 在分配节点之后没有排序，因此不能安全地访问内存。内核节点 d 在释放节点之前没有排序，因此它不能安全地访问内存。

.. figure:: https://docs.nvidia.com/cuda/cuda-programming-guide/_images/kernel-nodes-example.png
   :alt: 内核节点示例

   图 27 内核节点

以下代码片段建立了此图中的图：

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
   // kernel nodes b and c are using the graph allocation, so the freeing node must depend on them.  Since the dependency of node b on node a establishes an indirect dependency, the free node does not need to explicitly depend on node a.
   dependencies[0] = b;
   dependencies[1] = c;
   cudaGraphNodeParams freeNodeParams = { cudaGraphNodeTypeMemFree };
   freeNodeParams.free.dptr = params.alloc.dptr;
   cudaGraphAddNode(&freeNode, graph, dependencies, NULL, 2, freeNodeParams);
   // free node does not depend on kernel node d, so it must not access the freed graph allocation.
   cudaGraphAddNode(&d, graph, &c, NULL, 1, &nodeParams);

   // node e does not depend on the allocation node, so it must not access the allocation.  This would be true even if the freeNode depended on kernel node e.
   cudaGraphAddNode(&e, graph, NULL, NULL, 0, &nodeParams);

4.2.5.2.2. 流捕获
`````````````````

图内存节点也可以使用流捕获创建。当流处于捕获模式时，调用 cudaMallocAsync 和 cudaFreeAsync 会创建相应的分配和释放节点。在流捕获中，cudaMallocAsync 节点的行为类似于创建具有相同 GPU 有序生命周期语义的分配节点。

图分配的虚拟地址在流捕获期间是未知的，并且在捕获结束时在 cudaStreamEndCapture 时分配。当图被实例化时，地址被固定。这意味着在流捕获期间，不能在内核节点中直接使用 cudaMallocAsync 返回的指针。相反，必须使用图更新机制在实例化后设置指针值。

当使用流捕获创建图内存节点时，节点在捕获图中创建，就像使用图 API 显式创建它们一样。生成的图具有相同的 GPU 有序生命周期语义。

.. _cuda-graphs-accessing-and-freeing-graph-memory-outside:

4.2.5.2.3. 在分配图之外访问和释放图内存
`````````````````````````````````````````

图分配不必由分配它的图来释放。当图没有释放分配时，该分配会在图执行完成后持续存在，并且可以被后续的 CUDA 操作访问。这些分配可以在另一个图中访问，也可以直接使用流操作访问，只要访问操作通过 CUDA 事件和其他流排序机制在分配之后排序即可。分配随后可以通过常规的 ``cudaFree`` 、 ``cudaFreeAsync`` 调用，或通过启动另一个具有相应释放节点的图，或分配图的后续启动（如果它使用 ``cudaGraphInstantiateFlagAutoFreeOnLaunch`` 标志实例化）来释放。在内存被释放后访问它是非法的——释放操作必须使用图依赖、CUDA 事件和其他流排序机制在所有访问内存的操作之后排序。

.. note::

   由于图分配可能共享底层物理内存，释放操作必须在所有设备操作完成后排序。带外同步（如计算内核中基于内存的同步）不足以在内存写入和释放操作之间排序。更多信息，请参阅虚拟别名支持中关于一致性和相干性的规则。

以下三个代码片段演示了通过以下方式建立正确排序来在分配图之外访问图分配：使用单个流、使用流之间的事件，以及使用烘焙到分配和释放图中的事件。

**首先，通过使用单个流建立排序：**

.. code-block:: c++

   // 分配图的内容
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

**其次，通过记录和等待 CUDA 事件建立排序：**

.. code-block:: c++

   // 分配图的内容
   void *dptr;
   cudaGraphAddNode(&allocNode, allocGraph, NULL, NULL, 0, &allocNodeParams);
   dptr = allocNodeParams.alloc.dptr;

   // 消费/释放图的内容
   kernelNodeParams.kernel.kernelParams[0] = allocNodeParams.alloc.dptr;
   cudaGraphAddNode(&freeNode, freeGraph, NULL, NULL, 1, dptr);

   cudaGraphInstantiate(&allocGraphExec, allocGraph, NULL, NULL, 0);
   cudaGraphInstantiate(&freeGraphExec, freeGraph, NULL, NULL, 0);

   cudaGraphLaunch(allocGraphExec, allocStream);

   // 建立 stream2 对分配节点的依赖
   cudaEventRecord(allocEvent, allocStream);
   cudaStreamWaitEvent(stream2, allocEvent);

   kernel<<< ..., stream2 >>> (dptr, ...);

   // 建立 stream3 和分配使用之间的依赖
   cudaStreamRecordEvent(streamUseDoneEvent, stream2);
   cudaStreamWaitEvent(stream3, streamUseDoneEvent);

   // 现在可以安全地启动释放图，它也可以访问内存
   cudaGraphLaunch(freeGraphExec, stream3);

**第三，通过使用图外部事件节点建立排序：**

.. code-block:: c++

   // 分配图的内容
   void *dptr;
   cudaEvent_t allocEvent;  // 指示分配何时可供使用的事件
   cudaEvent_t streamUseDoneEvent;  // 指示流操作完成的事件

   // 带有事件记录节点的分配图内容
   cudaGraphAddNode(&allocNode, allocGraph, NULL, NULL, 0, &allocNodeParams);
   dptr = allocNodeParams.alloc.dptr;
   // 注意：此事件记录节点依赖于分配节点

   cudaGraphNodeParams allocEventNodeParams = { cudaGraphNodeTypeEventRecord };
   allocEventNodeParams.eventRecord.event = allocEvent;
   cudaGraphAddNode(&recordNode, allocGraph, &allocNode, NULL, 1, allocEventNodeParams);
   cudaGraphInstantiate(&allocGraphExec, allocGraph, NULL, NULL, 0);

   // 带有事件等待节点的消费/释放图内容
   cudaGraphNodeParams streamWaitEventNodeParams = { cudaGraphNodeTypeEventWait };
   streamWaitEventNodeParams.eventWait.event = streamUseDoneEvent;
   cudaGraphAddNode(&streamUseDoneEventNode, waitAndFreeGraph, NULL, NULL, 0, streamWaitEventNodeParams);

   cudaGraphNodeParams allocWaitEventNodeParams = { cudaGraphNodeTypeEventWait };
   allocWaitEventNodeParams.eventWait.event = allocEvent;
   cudaGraphAddNode(&allocReadyEventNode, waitAndFreeGraph, NULL, NULL, 0, allocWaitEventNodeParams);

   kernelNodeParams->kernelParams[0] = allocNodeParams.alloc.dptr;

   // allocReadyEventNode 为消费图中的分配节点提供排序
   cudaGraphAddNode(&kernelNode, waitAndFreeGraph, &allocReadyEventNode, NULL, 1, &kernelNodeParams);

   // 释放节点必须在外部和内部用户之后排序
   dependencies[0] = kernelNode;
   dependencies[1] = streamUseDoneEventNode;

   cudaGraphNodeParams freeNodeParams = { cudaGraphNodeTypeMemFree };
   freeNodeParams.free.dptr = dptr;
   cudaGraphAddNode(&freeNode, waitAndFreeGraph, &dependencies, NULL, 2, freeNodeParams);
   cudaGraphInstantiate(&waitAndFreeGraphExec, waitAndFreeGraph, NULL, NULL, 0);

   cudaGraphLaunch(allocGraphExec, allocStream);

   // 建立 stream2 对事件节点的依赖以满足排序要求
   cudaStreamWaitEvent(stream2, allocEvent);
   kernel<<< ..., stream2 >>> (dptr, ...);
   cudaStreamRecordEvent(streamUseDoneEvent, stream2);

   // waitAndFreeGraphExec 中的事件等待节点建立了对所需事件的依赖
   cudaGraphLaunch(waitAndFreeGraphExec, stream3);

.. _cuda-graphs-auto-free-on-launch:

4.2.5.2.4. cudaGraphInstantiateFlagAutoFreeOnLaunch
````````````````````````````````````````````````````

在正常情况下，如果图有未释放的内存分配，CUDA 会阻止图被重新启动，因为同一地址的多次分配会导致内存泄漏。使用 ``cudaGraphInstantiateFlagAutoFreeOnLaunch`` 标志实例化图允许图在仍有未释放分配的情况下重新启动。在这种情况下，启动会自动插入未释放分配的异步释放。

启动时自动释放对于单生产者多消费者算法很有用。在每次迭代中，生产者图创建多个分配，并且根据运行时条件，不同的消费者集合访问这些分配。这种可变执行序列意味着消费者无法释放分配，因为后续消费者可能需要访问。启动时自动释放意味着启动循环不需要跟踪生产者的分配——相反，该信息保持隔离在生产者的创建和销毁逻辑中。一般来说，启动时自动释放简化了原本需要在每次重新启动前释放图拥有的所有分配的算法。

.. note::

   ``cudaGraphInstantiateFlagAutoFreeOnLaunch`` 标志不会改变图销毁的行为。应用程序必须显式释放未释放的内存以避免内存泄漏，即使对于使用此标志实例化的图也是如此。

以下代码展示了使用 ``cudaGraphInstantiateFlagAutoFreeOnLaunch`` 来简化单生产者/多消费者算法：

.. code-block:: c++

   // 创建分配内存并用数据填充的生产者图
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

   // 通过捕获异步库调用创建第一个消费者图
   cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
   consumerFromLibrary(data1, cudaStreamPerThread);
   cudaStreamEndCapture(cudaStreamPerThread, &graph);
   cudaGraphInstantiateWithFlags(&consumer1, graph, 0);  // 常规实例化
   cudaGraphDestroy(graph);

   // 创建第二个消费者图
   cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
   consume2<<<blocks, threads, 0, cudaStreamPerThread>>>(data2);
   ...
   cudaStreamEndCapture(cudaStreamPerThread, &graph);
   cudaGraphInstantiateWithFlags(&consumer2, graph, 0);
   cudaGraphDestroy(graph);

   // 在循环中启动
   bool launchConsumer2 = false;
   do {
       cudaGraphLaunch(producer, myStream);
       cudaGraphLaunch(consumer1, myStream);
       if (launchConsumer2) {
           cudaGraphLaunch(consumer2, myStream);
       }
   } while (determineAction(&launchConsumer2));

   cudaFreeAsync(data1, myStream);
   cudaFreeAsync(data2, myStream);

   cudaGraphExecDestroy(producer);
   cudaGraphExecDestroy(consumer1);
   cudaGraphExecDestroy(consumer2);

.. _cuda-graphs-memory-nodes-in-child-graphs:

4.2.5.2.5. 子图中的内存节点
````````````````````````````

CUDA 12.9 引入了将子图所有权移动到父图的能力。移动到父图的子图允许包含内存分配和释放节点。这允许包含分配或释放节点的子图在添加到父图之前独立构建。

以下限制适用于移动后的子图：

- 不能独立实例化或销毁。
- 不能作为单独父图的子图添加。
- 不能用作 ``cuGraphExecUpdate`` 的参数。
- 不能添加额外的内存分配或释放节点。

.. code-block:: c++

   // 创建子图
   cudaGraphCreate(&child, 0);

   // 基本分配的参数
   cudaGraphNodeParams allocNodeParams = { cudaGraphNodeTypeMemAlloc };
   allocNodeParams.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
   allocNodeParams.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
   // 指定设备 0 为驻留设备
   allocNodeParams.alloc.poolProps.location.id = 0;
   allocNodeParams.alloc.bytesize = size;

   cudaGraphAddNode(&allocNode, child, NULL, NULL, 0, &allocNodeParams);
   // 可以在此处添加使用该分配的其他节点
   cudaGraphNodeParams freeNodeParams = { cudaGraphNodeTypeMemFree };
   freeNodeParams.free.dptr = allocNodeParams.alloc.dptr;
   cudaGraphAddNode(&freeNode, child, &allocNode, NULL, 1, freeNodeParams);

   // 创建父图
   cudaGraphCreate(&parent, 0);

   // 将子图移动到父图
   cudaGraphNodeParams childNodeParams = { cudaGraphNodeTypeGraph };
   childNodeParams.graph.graph = child;
   childNodeParams.graph.ownership = cudaGraphChildGraphOwnershipMove;
   cudaGraphAddNode(&parentNode, parent, NULL, NULL, 0, &childNodeParams);

.. _cuda-graphs-memory-nodes-requirements:

4.2.5.3. 图内存节点要求
~~~~~~~~~~~~~~~~~~~~~~~~~

图内存节点必须满足以下要求：

**分配节点：**

- 分配节点必须指定有效的内存池属性。
- 分配节点必须指定正的字节大小。

**释放节点：**

- 释放节点必须指定有效的图分配指针。
- 释放节点必须在分配节点之后排序。

**访问：**

- 访问图分配的所有操作必须在分配节点之后排序。
- 访问图分配的所有操作必须在释放节点之前排序。

**生命周期：**

- 图分配的生命周期在 GPU 执行到达分配节点时开始。
- 图分配的生命周期在 GPU 执行到达释放节点、cudaFreeAsync 流调用或 cudaFree 调用时结束。

.. _cuda-graphs-memory-nodes-optimization:

4.2.5.4. 优化内存重用
~~~~~~~~~~~~~~~~~~~~~

CUDA 可以在图内和跨图重用图分配的物理内存。这使得可以更有效地使用 GPU 内存，并减少内存碎片。

.. _cuda-graphs-address-reuse-within-a-graph:

4.2.5.4.1. 图内地址重用
````````````````````````

CUDA 可以通过将相同的虚拟地址范围分配给生命周期不重叠的不同分配来在图内重用内存。由于虚拟地址可能被重用，因此不保证具有不相交生命周期的不同分配的指针是唯一的。

下图展示了添加一个新的分配节点（2），它可以重用依赖节点（1）释放的地址。

.. figure:: /_static/images/new-alloc-node.png
   :alt: Adding New Alloc Node 2
   :width: 400px

   添加新的分配节点 2

下图展示了添加一个新的分配节点（4）。新的分配节点不依赖于释放节点（2），因此无法重用关联分配节点（2）的地址。如果分配节点（2）使用了释放节点（1）释放的地址，则新的分配节点 3 将需要一个新地址。

.. figure:: /_static/images/adding-new-alloc-nodes.png
   :alt: Adding New Alloc Node 3
   :width: 400px

   添加新的分配节点 3

.. _cuda-graphs-physical-memory-management-and-sharing:

4.2.5.4.2. 物理内存管理和共享
``````````````````````````````

CUDA 负责在 GPU 顺序中到达分配节点之前将物理内存映射到虚拟地址。作为内存占用和映射开销的优化，如果多个图不会同时运行，它们可以为不同的分配使用相同的物理内存；但是，如果物理页面同时绑定到多个正在执行的图，或绑定到仍未释放的图分配，则无法重用。

CUDA 可以在图实例化、启动或执行期间的任何时候更新物理内存映射。CUDA 还可能在未来的图启动之间引入同步，以防止活动的图分配引用相同的物理内存。对于任何分配-释放-分配模式，如果程序在分配生命周期之外访问指针，错误的访问可能会静默读取或写入另一个分配拥有的实时数据（即使分配的虚拟地址是唯一的）。使用计算清理工具可以捕获此错误。

下图展示了在同一流中顺序启动的图。在此示例中，每个图释放它分配的所有内存。由于同一流中的图永远不会同时运行，CUDA 可以并且应该使用相同的物理内存来满足所有分配。

.. figure:: /_static/images/sequentially-launched-graphs.png
   :alt: Sequentially Launched Graphs
   :width: 400px

   顺序启动的图

.. _cuda-graphs-performance-considerations:

4.2.5.5. 性能考虑
~~~~~~~~~~~~~~~~~

.. _cuda-graphs-first-launch-cuda-graph-upload:

4.2.5.5.1. 首次启动 / cudaGraphUpload
``````````````````````````````````````

在图实例化期间无法分配或映射物理内存，因为图将在其中执行的流是未知的。映射改为在图启动期间完成。调用 ``cudaGraphUpload`` 可以通过立即执行该图的所有映射并将图与上传流关联来将分配成本与启动分离。如果然后将图启动到同一流中，它将无需任何额外重新映射即可启动。

对图上传和图启动使用不同的流的行为类似于切换流，可能会导致重新映射操作。此外，不相关的内存池管理允许从空闲流中提取内存，这可能会抵消上传的影响。

.. _cuda-graphs-updating-graph-memory-nodes:

4.2.5.6. 图内存节点更新
~~~~~~~~~~~~~~~~~~~~~~~

图内存节点可以使用图更新机制更新。但是，更新图内存节点有一些限制：

- 不能更改分配节点的字节大小。
- 不能更改分配节点的位置或类型。
- 可以更改释放节点的指针，但它必须指向同一图分配。

有关图更新的更多信息，请参阅 4.2.3 节。

.. _cuda-graphs-peer-access:

4.2.5.7. 对等访问
~~~~~~~~~~~~~~~~~

图分配可以配置为从多个 GPU 访问，在这种情况下，CUDA 将根据需要将分配映射到对等 GPU。CUDA 允许需要不同映射的图分配重用相同的虚拟地址。当这种情况发生时，地址范围会映射到不同分配所需的所有 GPU。这意味着分配有时可能允许比创建期间请求的更多对等访问；但是，依赖这些额外映射仍然是错误的。

.. _cuda-graphs-peer-access-with-graph-node-apis:

4.2.5.7.1. 使用图节点 API 的对等访问
`````````````````````````````````````

``cudaGraphAddNode`` API 在分配节点参数结构的 ``accessDescs`` 数组字段中接受映射请求。 ``poolProps.location`` 嵌入结构指定分配的驻留设备。假定需要从分配 GPU 访问，因此应用程序不需要在 ``accessDescs`` 数组中为驻留设备指定条目。

.. code-block:: c++

   cudaGraphNodeParams allocNodeParams = { cudaGraphNodeTypeMemAlloc };
   allocNodeParams.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
   allocNodeParams.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
   // 指定设备 1 为驻留设备
   allocNodeParams.alloc.poolProps.location.id = 1;
   allocNodeParams.alloc.bytesize = size;

   // 分配一个驻留在设备 1 上、可从设备 1 访问的分配
   cudaGraphAddNode(&allocNode, graph, NULL, NULL, 0, &allocNodeParams);

   accessDescs[2];
   // 访问描述的样板代码（添加节点 API 仅支持 ReadWrite 和 Device 访问）
   accessDescs[0].flags = cudaMemAccessFlagsProtReadWrite;
   accessDescs[0].location.type = cudaMemLocationTypeDevice;
   accessDescs[1].flags = cudaMemAccessFlagsProtReadWrite;
   accessDescs[1].location.type = cudaMemLocationTypeDevice;

   // 请求设备 0 和 2 的访问。设备 1 的访问需求保持隐式。
   accessDescs[0].location.id = 0;
   accessDescs[1].location.id = 2;

   // 访问请求数组有 2 个条目。
   allocNodeParams.accessDescCount = 2;
   allocNodeParams.accessDescs = accessDescs;

   // 分配一个驻留在设备 1 上、可从设备 0、1 和 2 访问的分配。
   // （0 和 2 来自描述，1 来自它是驻留设备）。
   cudaGraphAddNode(&allocNode, graph, NULL, NULL, 0, &allocNodeParams);

.. _cuda-graphs-peer-access-with-stream-capture:

4.2.5.7.2. 使用流捕获的对等访问
````````````````````````````````

对于流捕获，分配节点记录捕获时分配池的对等可访问性。在 ``cudaMallocFromPoolAsync`` 调用被捕获后更改分配池的对等可访问性不会影响图将为分配进行的映射。

.. code-block:: c++

   // 访问描述的样板代码（添加节点 API 仅支持 ReadWrite 和 Device 访问）
   accessDesc.flags = cudaMemAccessFlagsProtReadWrite;
   accessDesc.location.type = cudaMemLocationTypeDevice;
   accessDesc.location.id = 1;

   // 让 memPool 驻留在设备 0 上并可从设备 0 访问

   cudaStreamBeginCapture(stream);
   cudaMallocAsync(&dptr1, size, memPool, stream);
   cudaStreamEndCapture(stream, &graph1);

   cudaMemPoolSetAccess(memPool, &accessDesc, 1);

   cudaStreamBeginCapture(stream);
   cudaMallocAsync(&dptr2, size, memPool, stream);
   cudaStreamEndCapture(stream, &graph2);

   // 分配 dptr1 的图节点将只有设备 0 的可访问性，即使
   // memPool 现在具有设备 1 的可访问性。
   // 分配 dptr2 的图节点将具有设备 0 和设备 1 的可访问性，因为那是
   // cudaMallocAsync 调用时的池可访问性。

.. _cuda-graphs-device-graph-launch:

4.2.6. 设备图启动
-----------------

有许多工作流需要在运行时进行数据相关的决策，并根据这些决策执行不同的操作。与其将此决策过程卸载到主机（可能需要从设备往返），用户可能更希望在设备上执行它。为此，CUDA 提供了一种从设备启动图的机制。

设备图启动提供了一种从设备进行动态控制流的便捷方式，无论是简单的循环还是复杂的设备端工作调度器。

可以从设备启动的图将被称为设备图，不能从设备启动的图将称为主机图。

设备图可以从主机和设备启动，而主机图只能从主机启动。与主机启动不同，在前一次图启动仍在运行时从设备启动设备图将导致错误，返回 ``cudaErrorInvalidValue`` ；因此，设备图不能同时从设备启动两次。同时从主机和设备启动设备图将导致未定义行为。

.. _cuda-graphs-device-graph-creation:

4.2.6.1. 设备图创建
~~~~~~~~~~~~~~~~~~~

为了使图能够从设备启动，必须显式为设备启动实例化它。这是通过将 ``cudaGraphInstantiateFlagDeviceLaunch`` 标志传递给 ``cudaGraphInstantiate()`` 调用来实现的。与主机图一样，设备图结构在实例化时固定，无法在不重新实例化的情况下更新，并且实例化只能在主机上执行。为了使图能够为设备启动实例化，它必须遵守各种要求。

.. _cuda-graphs-device-graph-requirements:

4.2.6.1.1. 设备图要求
``````````````````````

一般要求：

- 图的节点必须全部驻留在单个设备上。
- 图只能包含内核节点、memcpy 节点、memset 节点和子图节点。

内核节点：

- 不允许图中的内核使用 CUDA 动态并行。
- 只要不使用 MPS，就允许协作启动。

Memcpy 节点：

- 只允许涉及设备内存和/或固定设备映射主机内存的拷贝。
- 不允许涉及 CUDA 数组的拷贝。
- 两个操作数在实例化时必须可从当前设备访问。请注意，即使拷贝操作针对另一个设备上的内存，拷贝操作也将从图驻留的设备执行。

.. _cuda-graphs-device-graph-upload:

4.2.6.1.2. 设备图上传
``````````````````````

为了在设备上启动图，必须首先将图上传到设备以填充必要的设备资源。这可以通过以下两种方式之一实现。

首先，可以通过 ``cudaGraphUpload()`` 显式上传图，或者通过 ``cudaGraphInstantiateWithParams()`` 在实例化时请求上传。

或者，可以先从主机启动图，这将作为启动的一部分隐式执行此上传步骤。

所有三种方法的示例：

.. code-block:: c++

   // 实例化后显式上传
   cudaGraphInstantiate(&deviceGraphExec1, deviceGraph1, cudaGraphInstantiateFlagDeviceLaunch);
   cudaGraphUpload(deviceGraphExec1, stream);

   // 作为实例化的一部分显式上传
   cudaGraphInstantiateParams instantiateParams = {0};
   instantiateParams.flags = cudaGraphInstantiateFlagDeviceLaunch | cudaGraphInstantiateFlagUpload;
   instantiateParams.uploadStream = stream;
   cudaGraphInstantiateWithParams(&deviceGraphExec2, deviceGraph2, &instantiateParams);

   // 通过主机启动隐式上传
   cudaGraphInstantiate(&deviceGraphExec3, deviceGraph3, cudaGraphInstantiateFlagDeviceLaunch);
   cudaGraphLaunch(deviceGraphExec3, stream);

.. _cuda-graphs-device-graph-update:

4.2.6.1.3. 设备图更新
``````````````````````

设备图只能从主机更新，并且在可执行图更新后必须重新上传到设备才能使更改生效。这可以使用设备图上传中概述的相同方法实现。与主机图不同，在应用更新时从设备启动设备图将导致未定义行为。

.. _cuda-graphs-device-launch:

4.2.6.2. 设备启动
~~~~~~~~~~~~~~~~~

设备图可以通过 ``cudaGraphLaunch()`` 从主机和设备启动，该函数在设备上与主机上具有相同的签名。设备图通过主机和设备上的相同句柄启动。设备图从设备启动时必须从另一个图启动。

设备端图启动是每线程的，多个启动可能同时从不同线程发生，因此用户需要选择单个线程来启动给定的图。

与主机启动不同，设备图不能启动到常规 CUDA 流中，只能启动到不同的命名流中，每个流表示特定的启动模式。下表列出了可用的启动模式。

.. table:: 仅设备的图启动流

   ========================================= ==================
   流                                        启动模式
   ========================================= ==================
   ``cudaStreamGraphFireAndForget``          即发即忘启动
   ``cudaStreamGraphTailLaunch``             尾部启动
   ``cudaStreamGraphFireAndForgetAsSibling`` 兄弟启动
   ========================================= ==================

.. _cuda-graphs-fire-and-forget-launch:

4.2.6.2.1. 即发即忘启动
````````````````````````

顾名思义，即发即忘启动立即提交到 GPU，并且它独立于启动图运行。在即发即忘场景中，启动图是父图，启动的图是子图。

.. figure:: /_static/images/fire-and-forget-simple.png
   :alt: Fire and forget launch
   :width: 400px

   即发即忘启动

示例代码：

.. code-block:: c++

   __global__ void launchFireAndForgetGraph(cudaGraphExec_t graph) {
       cudaGraphLaunch(graph, cudaStreamGraphFireAndForget);
   }

   void graphSetup() {
       cudaGraphExec_t gExec1, gExec2;
       cudaGraph_t g1, g2;

       // 创建、实例化和上传设备图。
       create_graph(&g2);
       cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch);
       cudaGraphUpload(gExec2, stream);

       // 创建和实例化启动图。
       cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
       launchFireAndForgetGraph<<<1, 1, 0, stream>>>(gExec2);
       cudaStreamEndCapture(stream, &g1);
       cudaGraphInstantiate(&gExec1, g1);

       // 启动主机图，它将依次启动设备图。
       cudaGraphLaunch(gExec1, stream);
   }

一个图在其执行过程中最多可以有 120 个即发即忘图。此总数在同一父图的启动之间重置。

.. _cuda-graphs-graph-execution-environments:

4.2.6.2.1.1. 图执行环境
************************

为了完全理解设备端同步模型，首先需要理解执行环境的概念。

当图从设备启动时，它被启动到自己的执行环境中。给定图的执行环境封装了图中的所有工作以及所有生成的即发即忘工作。当图完成执行并且所有生成的子工作完成时，可以认为图已完成。

这些环境也是分层的，因此图环境可以包含来自即发即忘启动的多级子环境。

.. figure:: /_static/images/fire-and-forget-environments.png
   :alt: Fire and forget launch, with execution environments
   :width: 400px

   即发即忘启动，带有执行环境

.. figure:: /_static/images/fire-and-forget-nested-environments.png
   :alt: Nested fire and forget environments
   :width: 400px

   嵌套的即发即忘环境

当图从主机启动时，存在一个流环境，它是启动图的执行环境的父环境。流环境封装了作为整体启动一部分生成的所有工作。当整体流环境标记为完成时，流启动完成（即下游依赖工作现在可以运行）。

.. figure:: /_static/images/device-graph-stream-environment.png
   :alt: The stream environment, visualized
   :width: 400px

   流环境可视化

.. _cuda-graphs-tail-launch:

4.2.6.2.2. 尾部启动
````````````````````

与主机不同，无法通过传统方法（如 ``cudaDeviceSynchronize()`` 或 ``cudaStreamSynchronize()`` ）从 GPU 与设备图同步。相反，为了启用串行工作依赖，提供了不同的启动模式——尾部启动——以提供类似的功能。

尾部启动在图的环境被认为完成时执行——即当图及其所有子图完成时。当图完成时，尾部启动列表中下一个图的环境将替换已完成的环境作为父环境的子环境。与即发即忘启动一样，一个图可以有多个图排队进行尾部启动。

.. figure:: /_static/images/tail-launch-simple.png
   :alt: A simple tail launch
   :width: 400px

   简单的尾部启动

示例代码：

.. code-block:: c++

   __global__ void launchTailGraph(cudaGraphExec_t graph) {
       cudaGraphLaunch(graph, cudaStreamGraphTailLaunch);
   }

   void graphSetup() {
       cudaGraphExec_t gExec1, gExec2;
       cudaGraph_t g1, g2;

       // 创建、实例化和上传设备图。
       create_graph(&g2);
       cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch);
       cudaGraphUpload(gExec2, stream);

       // 创建和实例化启动图。
       cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
       launchTailGraph<<<1, 1, 0, stream>>>(gExec2);
       cudaStreamEndCapture(stream, &g1);
       cudaGraphInstantiate(&gExec1, g1);

       // 启动主机图，它将依次启动设备图。
       cudaGraphLaunch(gExec1, stream);
   }

由给定图排队的尾部启动将按它们排队的顺序一次执行一个。因此第一个排队的图将首先运行，然后是第二个，依此类推。

.. figure:: /_static/images/tail-launch-ordering-simple.png
   :alt: Tail launch ordering
   :width: 400px

   尾部启动排序

由尾部图排队的尾部启动将在尾部启动列表中先前图排队的尾部启动之前执行。这些新的尾部启动将按它们排队的顺序执行。

.. figure:: /_static/images/tail-launch-ordering-complex.png
   :alt: Tail launch ordering when enqueued from multiple graphs
   :width: 400px

   从多个图排队时的尾部启动排序

一个图最多可以有 255 个待处理的尾部启动。

.. _cuda-graphs-tail-self-launch:

4.2.6.2.2.1. 尾部自启动
************************

设备图可以将自己排队进行尾部启动，尽管给定的图一次只能有一个自启动排队。为了查询当前正在运行的设备图以便重新启动，添加了一个新的设备端函数：

.. code-block:: c++

   cudaGraphExec_t cudaGetCurrentGraphExec();

如果当前运行的图是设备图，此函数返回当前运行图的句柄。如果当前执行的内核不是设备图中的节点，此函数将返回 NULL。

展示重新发布循环用法的示例代码：

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

兄弟启动是即发即忘启动的变体，其中图不是作为启动图执行环境的子图启动，而是作为启动图父环境的子图启动。兄弟启动等同于从启动图父环境进行的即发即忘启动。

.. figure:: /_static/images/sibling-launch-simple.png
   :alt: A simple sibling launch
   :width: 400px

   简单的兄弟启动

示例代码：

.. code-block:: c++

   __global__ void launchSiblingGraph(cudaGraphExec_t graph) {
       cudaGraphLaunch(graph, cudaStreamGraphFireAndForgetAsSibling);
   }

   void graphSetup() {
       cudaGraphExec_t gExec1, gExec2;
       cudaGraph_t g1, g2;

       // 创建、实例化和上传设备图。
       create_graph(&g2);
       cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch);
       cudaGraphUpload(gExec2, stream);

       // 创建和实例化启动图。
       cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
       launchSiblingGraph<<<1, 1, 0, stream>>>(gExec2);
       cudaStreamEndCapture(stream, &g1);
       cudaGraphInstantiate(&gExec1, g1);

       // 启动主机图，它将依次启动设备图。
       cudaGraphLaunch(gExec1, stream);
   }

由于兄弟启动不是启动到启动图的执行环境中，它们不会阻止由启动图排队的尾部启动。

.. _cuda-graphs-using-graph-apis:

4.2.7. 使用图 API
-----------------

``cudaGraph_t`` 对象不是线程安全的。用户有责任确保多个线程不会同时访问同一个 ``cudaGraph_t`` 。

``cudaGraphExec_t`` 不能与自身同时运行。 ``cudaGraphExec_t`` 的启动将在同一可执行图的前次启动之后排序。

图执行在流中完成，以便与其他异步工作排序。但是，流仅用于排序；它不限制图的内部并行性，也不影响图节点执行的位置。

请参阅 `图 API <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH>`_ 。

.. _cuda-graphs-cuda-user-objects:

4.2.8. CUDA 用户对象
--------------------

CUDA 用户对象可用于帮助管理 CUDA 中异步工作使用的资源的生命周期。特别是，此功能对 CUDA Graphs 和流捕获很有用。

各种资源管理方案与 CUDA graphs 不兼容。例如，考虑基于事件的池或同步创建、异步销毁方案。

.. code-block:: c++

   // 具有池分配的库 API
   void libraryWork(cudaStream_t stream) {
       auto &resource = pool.claimTemporaryResource();
       resource.waitOnReadyEventInStream(stream);
       launchWork(stream, resource);
       resource.recordReadyEvent(stream);
   }

   // 具有异步资源删除的库 API
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
       // 未显示错误处理考虑
   }

这些方案与 CUDA graphs 难以配合，因为资源的非固定指针或句柄需要间接引用或图更新，以及每次提交工作时所需的同步 CPU 代码。如果这些考虑对库的调用者隐藏，它们也不适用于流捕获，并且因为在捕获期间使用了不允许的 API。存在各种解决方案，例如将资源暴露给调用者。CUDA 用户对象提供了另一种方法。

CUDA 用户对象将用户指定的析构函数回调与内部引用计数关联，类似于 C++ ``shared_ptr`` 。引用可以由 CPU 上的用户代码和 CUDA graphs 拥有。请注意，对于用户拥有的引用，与 C++ 智能指针不同，没有表示引用的对象；用户必须手动跟踪用户拥有的引用。典型用例是在创建用户对象后立即将唯一的用户拥有引用移动到 CUDA graph。

当引用与 CUDA graph 关联时，CUDA 将自动管理图操作。克隆的 ``cudaGraph_t`` 保留源 ``cudaGraph_t`` 拥有的每个引用的副本，具有相同的重数。实例化的 ``cudaGraphExec_t`` 保留源 ``cudaGraph_t`` 中每个引用的副本。当 ``cudaGraphExec_t`` 在未同步的情况下被销毁时，引用将保留直到执行完成。

使用示例：

.. code-block:: c++

   cudaGraph_t graph;  // 预先存在的图

   Object *object = new Object;  // 具有可能非平凡析构函数的 C++ 对象
   cudaUserObject_t cuObject;
   cudaUserObjectCreate(
       &cuObject,
       object,  // 这里我们使用 CUDA 提供的此 API 的模板包装器，
                // 它提供一个回调来删除 C++ 对象指针
       1,  // 初始引用计数
       cudaUserObjectNoDestructorSync  // 确认回调无法通过 CUDA 等待
   );
   cudaGraphRetainUserObject(
       graph,
       cuObject,
       1,  // 引用数量
       cudaGraphUserObjectMove  // 转移调用者拥有的引用（不
                                // 修改总引用计数）
   );
   // 此线程不再拥有引用；无需调用释放 API
   cudaGraphExec_t graphExec;
   cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);  // 将保留
                                                                  // 新引用
   cudaGraphDestroy(graph);  // graphExec 仍然拥有一个引用
   cudaGraphLaunch(graphExec, 0);  // 异步启动可以访问用户对象
   cudaGraphExecDestroy(graphExec);  // 启动未同步；如果需要，释放
                                     // 将被延迟
   cudaStreamSynchronize(0);  // 在启动同步后，剩余的
                              // 引用被释放，析构函数将
                              // 执行。请注意这是异步发生的。
   // 如果析构函数回调发出了同步对象信号，此时
   // 等待它是安全的。

子图节点中图拥有的引用与子图关联，而不是与父图关联。如果子图被更新或删除，引用会相应更改。如果使用 ``cudaGraphExecUpdate`` 或 ``cudaGraphExecChildGraphNodeSetParams`` 更新可执行图或子图，新源图中的引用将被克隆并替换目标图中的引用。在任何一种情况下，如果先前的启动未同步，任何将被释放的引用将保留直到启动完成执行。

目前没有机制通过 CUDA API 等待用户对象析构函数。用户可以从析构函数代码手动发出同步对象信号。此外，从析构函数调用 CUDA API 是非法的，类似于 ``cudaLaunchHostFunc`` 的限制。这是为了避免阻塞 CUDA 内部共享线程并阻止前进进度。如果依赖是单向的并且执行调用的线程不能阻塞 CUDA 工作的前进进度，则可以发出信号让另一个线程执行 API 调用。

用户对象使用 ``cudaUserObjectCreate`` 创建，这是浏览相关 API 的良好起点。
