.. _cuda-graphs-details:

4.2. CUDA Graphs
================

CUDA Graphs 提出了另一种 CUDA 工作提交模型。图（graph）是一系列操作（如内核启动、数据移动等）通过依赖关系连接而成，其定义与执行分离。这使得图可以定义一次，然后重复启动多次。将图的定义与执行分离实现了许多优化：首先，与流相比，CPU 启动开销降低了，因为大部分准备工作都是提前完成的；其次，将整个工作流程呈现给 CUDA 可以实现一些使用流的分段工作提交机制无法实现的优化。

要了解图可以实现的优化，可以考虑流中发生的事情：当你将内核放入流中时，主机驱动程序会执行一系列操作来准备在 GPU 上执行内核。这些操作对于设置和启动内核是必要的，是每次发出内核时都必须支付的开销成本。对于执行时间较短的 GPU 内核，此开销成本可能是整体端到端执行时间的很大一部分。通过创建涵盖将多次启动的工作流程的 CUDA 图，这些开销成本可以在实例化期间为整个图支付一次，然后图本身可以以非常小的开销重复启动。

4.2.1. 图结构
-------------

操作在图中形成节点（node）。操作之间的依赖关系是边（edge）。这些依赖关系约束了操作的执行顺序。

一旦操作所依赖的节点完成，就可以随时调度该操作。调度由 CUDA 系统负责。

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

4.2.1.2. 边数据
~~~~~~~~~~~~~~~

CUDA 12.3 在 CUDA Graphs 中引入了边数据（edge data）。目前，非默认边数据的唯一用途是支持编程式依赖启动（Programmatic Dependent Launch）。

一般而言，边数据修改由边指定的依赖关系，由三部分组成：输出端口（outgoing port）、输入端口（incoming port）和类型（type）。输出端口指定关联的边何时被触发。输入端口指定节点的哪一部分依赖于关联的边。类型修改端点之间的关系。

端口值特定于节点类型和方向，边类型可能仅限于特定节点类型。在所有情况下，零初始化的边数据表示默认行为。输出端口 0 等待整个任务，输入端口 0 阻塞整个任务，边类型 0 与具有内存同步行为的完整依赖关系相关联。

边数据在各种图 API 中通过关联节点的并行数组可选地指定。如果作为输入参数省略，则使用零初始化的数据。如果作为输出（查询）参数省略，如果忽略的边数据全部为零初始化，则 API 接受此情况；如果调用将丢弃信息，则返回 cudaErrorLossyQuery。

某些流捕获 API 中也提供边数据：cudaStreamBeginCaptureToGraph()、cudaStreamGetCaptureInfo() 和 cudaStreamUpdateCaptureDependencies()。在这些情况下，还没有下游节点。数据与悬空边（半条边）相关联，该边将连接到未来的捕获节点或在流捕获终止时被丢弃。

请注意，某些边类型不会等待上游节点完全完成。在考虑流捕获是否已完全重新加入原始流时，这些边将被忽略，并且在捕获结束时不能被丢弃。参见流捕获（Stream Capture）。

没有节点类型定义额外的输入端口，只有内核节点定义额外的输出端口。有一个非默认依赖类型 cudaGraphDependencyTypeProgrammatic，用于在两个内核节点之间启用编程式依赖启动。

4.2.2. 构建和运行图
-------------------

使用图进行工作提交分为三个不同的阶段：定义（definition）、实例化（instantiation）和执行（execution）。

- 在定义或创建阶段，程序创建图中操作的描述以及它们之间的依赖关系。
- 实例化获取图模板的快照，验证它，并执行大量的工作设置和初始化，目的是最小化启动时需要完成的工作。生成的实例称为可执行图（executable graph）。
- 可执行图可以启动到流中，类似于任何其他 CUDA 工作。它可以启动任意次数，而无需重复实例化。

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

4.2.2.2. 图实例化
~~~~~~~~~~~~~~~~~

一旦创建了图（通过使用图 API 或流捕获），就必须实例化该图以创建可执行图，然后可以启动它。假设 cudaGraph_t graph 已成功创建，以下代码将实例化图并创建可执行图 cudaGraphExec_t graphExec：

.. code-block:: cpp

   cudaGraphExec_t graphExec;
   cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

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

4.2.3. 更新实例化的图
---------------------

当工作流程发生变化时，图就会过时，必须修改。图结构的重大更改（如拓扑或节点类型）需要重新实例化，因为必须重新应用与拓扑相关的优化。然而，通常只有节点参数（如内核参数和内存地址）发生变化，而图拓扑保持不变。对于这种情况，CUDA 提供了一种轻量级的"图更新"机制，允许就地修改某些节点参数，而无需重建整个图，这比重新实例化要高效得多。

更新在图下次启动时生效，因此它们不会影响之前的图启动，即使在更新时它们正在运行。图可以重复更新和重新启动，因此多个更新/启动可以在流上排队。

CUDA 提供了两种更新实例化图参数的机制：整图更新（whole graph update）和单个节点更新（individual node update）。整图更新允许用户提供拓扑相同的 cudaGraph_t 对象，其节点包含更新的参数。单个节点更新允许用户显式更新单个节点的参数。当大量节点正在更新或调用者不知道图拓扑时（即图来自库调用的流捕获），使用更新的 cudaGraph_t 更方便。当更改数量很少且用户拥有需要更新的节点的句柄时，首选单个节点更新。单个节点更新跳过未更改节点的拓扑检查和比较，因此在许多情况下可能更高效。

CUDA 还提供了一种启用和禁用单个节点而不影响其当前参数的机制。

以下章节更详细地解释每种方法。

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

4.2.3.3. 单个节点启用
~~~~~~~~~~~~~~~~~~~~~

实例化图中的内核、内存填充和内存拷贝节点可以使用 cudaGraphNodeSetEnabled() API 启用或禁用。这允许创建包含所需功能超集的图，可以为每次启动自定义。可以使用 cudaGraphNodeGetEnabled() API 查询节点的启用状态。

禁用的节点在功能上等同于空节点，直到它被重新启用。节点的参数不受启用/禁用节点的影响。启用状态不受单个节点更新或使用 cudaGraphExecUpdate() 的整图更新的影响。节点禁用时的参数更新将在节点重新启用时生效。

请参阅图 API 以获取有关用法和当前限制的更多信息。

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

4.2.4.1. 条件句柄
~~~~~~~~~~~~~~~~~

条件值由 cudaGraphConditionalHandle 表示，并由 cudaGraphConditionalHandleCreate() 创建。

句柄必须与单个条件节点关联。句柄不能被销毁，因此无需跟踪它们。

如果在创建句柄时指定了 cudaGraphCondAssignDefault，则条件值将在每次图执行开始时初始化为指定的默认值。如果未提供此标志，则条件值在每次图执行开始时未定义，代码不应假设条件值在执行之间持续存在。

与句柄关联的默认值和标志将在整图更新期间更新。

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

4.2.5. 图内存节点
-----------------

4.2.5.1. 简介
~~~~~~~~~~~~~

图内存节点允许图创建和拥有内存分配。图内存节点具有 GPU 有序生命周期语义，它规定何时允许在设备上访问内存。这些 GPU 有序生命周期语义实现了驱动程序管理的内存重用，并与流有序分配 API cudaMallocAsync 和 cudaFreeAsync 匹配，这些 API 在创建图时可以被捕获。

图分配在图的整个生命周期内（包括重复实例化和启动）具有固定地址。这允许内存被图内的其他操作直接引用，而无需图更新，即使 CUDA 更改了后备物理内存。在图内，生命周期不重叠的分配可以使用相同的底层物理内存。

CUDA 可以在多个图之间重用相同的物理内存进行分配，根据 GPU 有序生命周期语义虚拟化地址映射。例如，当不同的图启动到同一流中时，CUDA 可以虚拟化地混叠相同的物理内存以满足具有单图生命周期的分配需求。

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

4.2.5.4. 图内存重用
~~~~~~~~~~~~~~~~~~~

CUDA 可以在图内和跨图重用图分配的物理内存。这使得可以更有效地使用 GPU 内存，并减少内存碎片。

**图内重用：**

在图内，如果两个分配的 GPU 有序生命周期不重叠，CUDA 可以为它们分配相同的物理内存。这可以减少图的总内存使用量。

**跨图重用：**

CUDA 可以在不同图之间重用物理内存，前提是分配具有兼容的生命周期语义。例如，如果两个图启动到同一流中，并且它们具有单图生命周期的分配，CUDA 可以为它们分配相同的物理内存。

4.2.5.5. 图内存节点更新
~~~~~~~~~~~~~~~~~~~~~~~~~

图内存节点可以使用图更新机制更新。但是，更新图内存节点有一些限制：

- 不能更改分配节点的字节大小。
- 不能更改分配节点的位置或类型。
- 可以更改释放节点的指针，但它必须指向同一图分配。

有关图更新的更多信息，请参阅 4.2.3 节。

.. note::
   有关图内存节点的更多详细信息，请参阅 `CUDA 官方文档 <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>`_。
