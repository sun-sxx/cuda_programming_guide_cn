.. _green-contexts-details:

4.6. Green Contexts
===================

Green Contexts（绿色上下文）是一种轻量级的执行上下文，与特定的 GPU 资源集合关联。用户可以在创建 Green Context 时分配 GPU 资源（当前是流式多处理器 (SM) 和工作队列 (WQ)），以便针对 Green Context 的 GPU 工作只能使用其分配的资源。这样做有助于减少或更好地控制由于使用共享资源而导致的干扰。应用程序可以拥有多个 Green Contexts。

使用 Green Contexts 不需要任何 GPU 代码（核函数）更改，只需要少量主机端更改（例如，为 Green Context 创建 Green Context 和流）。Green Context 功能在各种场景中很有用。例如，它可以帮助确保某些 SM 始终可用于延迟敏感的核函数开始执行（假设没有其他约束），或者提供了一种快速方法来测试使用较少 SM 的效果，而无需任何核函数修改。

Green Context 支持最初通过 `CUDA Driver API <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS>`_ 提供。从 CUDA 13.1 开始，上下文通过执行上下文 (EC) 抽象在 CUDA 运行时中暴露。目前，执行上下文可以对应于主上下文（运行时 API 用户一直隐式交互的上下文）或 Green Context。本节将在提到 Green Context 时互换使用术语 *执行上下文* 和 *Green Context*。

随着 Green Contexts 在运行时中的暴露，强烈建议直接使用 CUDA 运行时 API。本节也将仅使用 CUDA 运行时 API。

本节其余部分组织如下：`动机/何时使用`_ 提供示例，`Green Contexts：易用性`_ 突出易用性，`Green Contexts：设备资源和描述符`_ 介绍设备资源和资源描述符结构体。`Green Context 创建示例`_ 解释如何创建 Green Context，`Green Contexts：启动工作`_ 介绍如何启动针对它的工作，`Green Contexts：API`_ 突出一些额外的 Green Context API。最后，`Green Contexts：示例`_ 总结示例。

.. _green-contexts-motivation:

4.6.1. 动机/何时使用
--------------------

启动 CUDA 核函数时，用户无法直接控制该核函数将在多少个 SM 上执行。只能通过更改核函数的启动几何形状或任何可能影响每个 SM 的最大活动线程块数的因素来间接影响这一点。此外，当多个核函数在 GPU 上并行执行时（在不同 CUDA 流上运行的核函数或作为 CUDA 图的一部分），它们可能会争用相同的 SM 资源。

然而，有些用例需要确保 GPU 资源始终可用于延迟敏感的工作尽快开始，从而尽快完成。Green Contexts 通过分配 SM 资源提供了一种方法，使得给定的 Green Context 只能使用特定的 SM（在创建期间分配的那些）。

例如，假设一个应用程序，两个独立的核函数 A 和 B 在两个不同的非阻塞 CUDA 流上运行。首先启动核函数 A 并开始执行，占用所有可用的 SM 资源。当稍后启动延迟敏感的核函数 B 时，没有可用的 SM 资源。因此，核函数 B 只能等到核函数 A 减少（即来自核函数 A 的线程块完成执行）后才能开始执行。第一个图说明了这种情况，关键工作 B 被延迟。y 轴显示占用的 SM 百分比，x 轴表示时间。

使用 Green Contexts，可以分配 GPU 的 SM，使得针对核函数 A 的 Green Context A 可以访问 GPU 的一些 SM，而针对核函数 B 的 Green Context B 可以访问剩余的 SM。在这种情况下，核函数 A 只能使用为 Green Context A 分配的 SM，与其启动配置无关。因此，当启动关键核函数 B 时，保证将有可用的 SM 供其立即开始执行，除非有其他资源约束。如图所示，即使核函数 A 的持续时间可能增加，延迟敏感的工作 B 也不会因为不可用的 SM 而延迟。

这种行为无需对核函数 A 和 B 进行任何代码修改。只需确保它们在与适当的 Green Context 所属的 CUDA 流上启动。每个 Green Context 将访问的 SM 数量应由用户在创建 Green Context 时根据具体情况决定。

**工作队列**：

流式多处理器是可以为 Green Context 分配的一种资源类型。另一种资源类型是工作队列。将工作队列视为一个黑盒资源抽象，它也会影响 GPU 工作执行并发性以及其他因素。如果独立的 GPU 工作任务（例如，在不同 CUDA 流上提交的内核）映射到相同的工作队列，则可能会在这些任务之间引入错误的依赖性，从而导致它们的序列化执行。用户可以通过 ``CUDA_DEVICE_MAX_CONNECTIONS`` 环境变量来影响 GPU 上工作队列数量的上限。

**Green Contexts 与 MIG 或 MPS 的比较**

为了完整起见，本节简要比较 Green Contexts 与另外两种资源分配机制：`MIG（多实例 GPU）<https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html>`_ 和 `MPS（多进程服务）<https://docs.nvidia.com/deploy/mps/index.html>`_。

MIG 将支持 MIG 的 GPU 静态划分为多个 MIG 实例（"较小的 GPU"）。这种分配必须在启动应用程序之前完成，不同的应用程序可以使用不同的 MIG 实例。使用 MIG 对于应用程序持续未充分利用可用 GPU 资源的用户可能有益；随着 GPU 变得更大，这个问题更加明显。使用 MIG，用户可以在不同的 MIG 实例上运行这些不同的应用程序，从而提高 GPU 利用率。MIG 对云服务提供商 (CSP) 具有吸引力，不仅因为对这些应用程序提高了 GPU 利用率，还因为它可以在不同 MIG 实例上运行的客户端之间提供服务质量 (QoS) 和隔离。

但使用 MIG 无法解决前面描述的有问题的场景，即关键工作 B 因为所有 SM 资源被同一应用程序的其他 GPU 工作占用而延迟。对于在单个 MIG 实例上运行的应用程序，此问题仍可能存在。为了解决这个问题，可以与 MIG 一起使用 Green Contexts。在这种情况下，可用于分配的 SM 资源将是给定 MIG 实例的资源。

MPS 主要针对不同进程（例如 MPI 程序），允许它们同时在 GPU 上运行而不进行时间切片。它需要在启动应用程序之前运行 MPS 守护进程。默认情况下，MPS 客户端将争用 GPU 或 MIG 实例的所有可用 SM 资源。在这种多客户端进程设置中，MPS 可以支持使用活动线程百分比选项动态分配 SM 资源，这限制了 MPS 客户端进程可以使用的 SM 百分比。与 Green Contexts 不同，使用活动线程百分比的分配在进程级别发生在 MPS 上，百分比通常由环境变量在启动应用程序之前指定。MPS 活动线程百分比表示给定客户端应用程序不能使用超过 x% 的 GPU SM，比如 N 个 SM。但是，这些 SM 可以是 GPU 的任何 N 个 SM，也可以随时间变化。另一方面，在创建期间分配有 N 个 SM 的 Green Context 只能使用这 N 个特定 SM。

.. _green-contexts-ease-of-use:

4.6.2. Green Contexts：易用性
-----------------------------

为了突出 Green Contexts 的使用有多简单，假设您有以下代码片段，创建两个 CUDA 流，然后调用在这些 CUDA 流上使用 ``<<<>>>`` 启动核函数的函数。如前所述，除了更改核函数的启动几何形状外，无法影响这些核函数可以使用多少个 SM。

.. code-block:: c++

   int gpu_device_index = 0; // GPU 序号
   CUDA_CHECK(cudaSetDevice(gpu_device_index));

   cudaStream_t strm1, strm2;
   CUDA_CHECK(cudaStreamCreateWithFlags(&strm1, cudaStreamNonBlocking));
   CUDA_CHECK(cudaStreamCreateWithFlags(&strm2, cudaStreamNonBlocking));

   // 无法控制每个流上运行的核函数可以使用多少个 SM
   code_that_launches_kernels_on_streams(strm1, strm2); // 这个函数中的抽象内容+核函数是你代码的绝大多数

   // 清理代码未显示

从 CUDA 13.1 开始，可以使用 Green Contexts 控制给定核函数可以访问的 SM 数量。下面的代码片段展示了这样做有多容易。通过几行额外的代码，无需任何核函数修改，你就可以控制这些不同流上启动的核函数可以使用的 SM 资源。

.. code-block:: c++

   int gpu_device_index = 0; // GPU 序号
   CUDA_CHECK(cudaSetDevice(gpu_device_index));

   /* ------------------ 创建 Green Contexts 所需的代码 --------------------------- */

   // 获取所有可用的 GPU SM 资源
   cudaDevResource initial_GPU_SM_resources {};
   CUDA_CHECK(cudaDeviceGetDevResource(gpu_device_index, &initial_GPU_SM_resources, cudaDevResourceTypeSm));

   // 分割 SM 资源。此示例创建一个包含 16 个 SM 的组和一个包含 8 个 SM 的组。假设您的 GPU 有 >= 24 个 SM
   cudaDevSmResource result[2] {{}, {}};
   cudaDevSmResourceGroupParams group_params[2] =  {
           {.smCount=16, .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0},
           {.smCount=8,  .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0}};
   CUDA_CHECK(cudaDevSmResourceSplit(&result[0], 2, &initial_GPU_SM_resources, nullptr, 0, &group_params[0]));

   // 为每个资源生成资源描述符
   cudaDevResourceDesc_t resource_desc1 {};
   cudaDevResourceDesc_t resource_desc2 {};
   CUDA_CHECK(cudaDevResourceGenerateDesc(&resource_desc1, &result[0], 1));
   CUDA_CHECK(cudaDevResourceGenerateDesc(&resource_desc2, &result[1], 1));

   // 创建 Green Contexts
   cudaExecutionContext_t my_green_ctx1 {};
   cudaExecutionContext_t my_green_ctx2 {};
   CUDA_CHECK(cudaGreenCtxCreate(&my_green_ctx1, resource_desc1, gpu_device_index, 0));
   CUDA_CHECK(cudaGreenCtxCreate(&my_green_ctx2, resource_desc2, gpu_device_index, 0));

   /* ------------------ 修改的代码 --------------------------- */

   // 只需使用不同的 CUDA API 创建流
   cudaStream_t strm1, strm2;
   CUDA_CHECK(cudaExecutionCtxStreamCreate(&strm1, my_green_ctx1, cudaStreamDefault, 0));
   CUDA_CHECK(cudaExecutionCtxStreamCreate(&strm2, my_green_ctx2, cudaStreamDefault, 0));

   /* ------------------ 未更改的代码 --------------------------- */

   // 无需修改此函数或核函数中的任何代码。
   // 提醒：这个函数+核函数中的抽象内容是你代码的绝大多数
   // 现在在流 strm1 上运行的核函数最多使用 16 个 SM，在 strm2 上的核函数最多使用 8 个 SM
   code_that_launches_kernels_on_streams(strm1, strm2);

   // 清理代码未显示

各种执行上下文 API（其中一些在前面的示例中显示）采用显式的 ``cudaExecutionContext_t`` 句柄，从而忽略当前调用线程的上下文。到目前为止，不使用 Driver API 的 CUDA 运行时用户默认只与通过 ``cudaSetDevice()`` 隐式设置为当前线程的主上下文交互。这种向基于显式上下文的编程的转变提供了更容易理解的语义，与之前依赖线程局部状态 (TLS) 的基于隐式上下文的编程相比，可以带来额外的好处。

.. _green-contexts-device-resource-and-desc:

4.6.3. Green Contexts：设备资源和资源描述符
--------------------------------------------

Green Context 的核心是与特定 GPU 设备关联的设备资源 (``cudaDevResource``)。资源可以组合并封装到描述符 (``cudaDevResourceDesc_t``) 中。Green Context 只能访问用于创建它的描述符中封装的资源。

目前 ``cudaDevResource`` 数据结构定义为：

.. code-block:: c++

   struct {
        enum cudaDevResourceType type;
        union {
            struct cudaDevSmResource sm;
            struct cudaDevWorkqueueConfigResource wqConfig;
            struct cudaDevWorkqueueResource wq;
        };
    };

支持的有效资源类型是 ``cudaDevResourceTypeSm``、``cudaDevResourceTypeWorkqueueConfig`` 和 ``cudaDevResourceTypeWorkqueue``，而 ``cudaDevResourceTypeInvalid`` 标识无效的资源类型。

有效的设备资源可以与以下内容关联：

- 特定的一组流式多处理器 (SM)（资源类型 ``cudaDevResourceTypeSm``）
- 特定的工作队列配置（资源类型 ``cudaDevResourceTypeWorkqueueConfig``）
- 预先存在的工作队列资源（资源类型 ``cudaDevResourceTypeWorkqueue``）

可以使用 ``cudaExecutionCtxGetDevResource`` 和 ``cudaStreamGetDevResource`` API 分别查询给定的执行上下文或 CUDA 流是否与给定类型的 ``cudaDevResource`` 资源关联。执行上下文可以关联不同类型的设备资源（例如 SM 和工作队列），而流只能与 SM 类型的资源关联。

给定的 GPU 设备默认具有所有三种设备资源类型：包含 GPU 所有 SM 的 SM 类型资源、包含所有可用工作队列的工作队列配置资源及其相应的工作队列资源。这些资源可以通过 ``cudaDeviceGetDevResource`` API 检索。

**相关设备资源结构体概述**

不同的资源类型结构体具有由用户或相关 CUDA API 调用设置的字段。建议将所有设备资源结构体归零初始化。

- **SM 类型设备资源** (``cudaDevSmResource``) 有以下相关字段：

  - ``unsigned int smCount``：此资源中可用的 SM 数量
  - ``unsigned int minSmPartitionSize``：分配此资源所需的最小 SM 数量
  - ``unsigned int smCoscheduledAlignment``：保证在同一 GPU 处理集群上共同调度的 SM 数量，与线程块集群相关。当 ``flags`` 为零时，``smCount`` 是此值的倍数。
  - ``unsigned int flags``：支持的标志是 0（默认）和 ``cudaDevSmResourceGroupBackfill``

  这些字段将通过用于创建此 SM 类型资源的适当分割 API（``cudaDevSmResourceSplitByCount`` 或 ``cudaDevSmResourceSplit``）设置，或者由检索给定 GPU 设备 SM 资源的 ``cudaDeviceGetDevResource`` API 填充。这些字段不应由用户直接设置。

- **工作队列配置设备资源** (``cudaDevWorkqueueConfigResource``) 有以下相关字段：

  - ``int device``：工作队列资源可用的设备
  - ``unsigned int wqConcurrencyLimit``：预期避免错误依赖的流排序工作负载数量
  - ``enum cudaDevWorkqueueConfigScope sharingScope``：工作队列资源的共享范围。支持的值是：``cudaDevWorkqueueConfigScopeDeviceCtx``（默认）和 ``cudaDevWorkqueueConfigScopeGreenCtxBalanced``。使用默认选项，所有工作队列资源在所有上下文中共享，而使用平衡选项，驱动程序尽可能尝试在 Green Contexts 之间使用非重叠的工作队列资源，使用用户指定的 ``wqConcurrencyLimit`` 作为提示。

  这些字段需要由用户设置。

- **预先存在的工作队列资源** (``cudaDevResourceTypeWorkqueue``) 没有可以由用户设置的字段。

.. _green-contexts-creation-example:

4.6.4. Green Context 创建示例
-----------------------------

Green Context 创建涉及四个主要步骤：

- **步骤 1**：从一组初始资源开始，例如通过获取 GPU 的可用资源
- **步骤 2**：将 SM 资源分配到一个或多个分区（使用可用的分割 API 之一）
- **步骤 3**：创建组合不同资源（如果需要）的资源描述符
- **步骤 4**：从描述符创建 Green Context，分配其资源

Green Context 创建后，您可以创建属于该 Green Context 的 CUDA 流。随后在此类流上启动的 GPU 工作（如通过 ``<<< >>>`` 启动的核函数）将只能访问此 Green Context 分配的资源。

.. _green-contexts-creation-example-step1:

4.6.4.1. 步骤 1：获取可用的 GPU 资源
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Green Context 创建的第一步是获取可用的设备资源并填充 ``cudaDevResource`` 结构体。目前有三个可能的起点：设备、执行上下文或 CUDA 流。

相关 CUDA 运行时 API 函数签名如下：

- 对于 **设备**：``cudaError_t cudaDeviceGetDevResource(int device, cudaDevResource* resource, cudaDevResourceType type)``
- 对于 **执行上下文**：``cudaError_t cudaExecutionCtxGetDevResource(cudaExecutionContext_t ctx, cudaDevResource* resource, cudaDevResourceType type)``
- 对于 **流**：``cudaError_t cudaStreamGetDevResource(cudaStream_t hStream, cudaDevResource* resource, cudaDevResourceType type)``

所有这些 API 都允许所有有效的 ``cudaDevResourceType`` 类型，但 ``cudaStreamGetDevResource`` 除外，它仅支持 SM 类型资源。

通常，起点是 GPU 设备。以下代码片段展示了如何获取给定 GPU 设备的可用 SM 资源。

.. code-block:: c++

   int current_device = 0; // 假设设备序号为 0
   CUDA_CHECK(cudaSetDevice(current_device));

   cudaDevResource initial_SM_resources = {};
   CUDA_CHECK(cudaDeviceGetDevResource(current_device /* GPU 设备 */,
                                      &initial_SM_resources /* 要填充的设备资源 */,
                                      cudaDevResourceTypeSm /* 资源类型*/));

   std::cout << "初始 SM 资源：" << initial_SM_resources.sm.smCount << " 个 SM" << std::endl;
   std::cout << "最小 SM 分区大小：" <<  initial_SM_resources.sm.minSmPartitionSize << " 个 SM" << std::endl;
   std::cout << "SM 共同调度对齐：" <<  initial_SM_resources.sm.smCoscheduledAlignment << " 个 SM" << std::endl;

.. _green-contexts-split-sm-resources:

4.6.4.2. 步骤 2：分配 SM 资源
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Green Context 创建的第二步是静态地将可用的 ``cudaDevResource`` SM 资源分割成一个或多个分区，可能还有一些 SM 留在剩余分区中。可以使用 ``cudaDevSmResourceSplitByCount()`` 或 ``cudaDevSmResourceSplit()`` API 进行此分配。

**cudaDevSmResourceSplitByCount API**

此 API 请求将 ``input`` SM 类型设备资源分割成 ``*nbGroups`` 个具有 ``minCount`` 个 SM 的同质组。但是，最终结果将包含潜在更新的 ``*nbGroup`` 个具有 ``N`` 个 SM 的同质组。由于一些粒度和对齐要求（这是特定于架构的），这些调整可能会发生。

**cudaDevSmResourceSplit API**

``cudaDevSmResourceSplitByCount`` API 调用只能创建同质分区（即具有相同数量 SM 的分区），加上剩余分区。对于异构工作负载（在不同 Green Contexts 上运行的工作具有不同的 SM 数量要求），这可能会受到限制。

``cudaDevSmResourceSplit`` API 旨在通过允许用户在单次调用中创建非重叠的异构分区来解决这些限制。

.. code-block:: c++

   cudaError_t cudaDevSmResourceSplit(cudaDevResource* result, unsigned int nbGroups, 
                                      const cudaDevResource* input, cudaDevResource* remainder, 
                                      unsigned int flags, cudaDevSmResourceGroupParams* groupParams)

.. _green-contexts-add-workqueue-resources:

4.6.4.3. 步骤 2（续）：添加工作队列资源
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果还想指定工作队列资源，则需要显式执行。以下示例展示了如何为特定设备创建工作队列配置资源，具有平衡的共享范围和四个并发限制。

.. code-block:: c++

   cudaDevResource split_result[2] = {{}, {}};
   // 填充 split_result[0] 的代码未显示；使用 split API，nbGroups=1

   // 最后一个资源将是工作队列资源
   split_result[1].type = cudaDevResourceTypeWorkqueueConfig;
   split_result[1].wqConfig.device = 0; // 假设设备序号为 0
   split_result[1].wqConfig.sharingScope = cudaDevWorkqueueConfigScopeGreenCtxBalanced;
   split_result[1].wqConfig.wqConcurrencyLimit = 4;

.. _green-contexts-create-resource-desc:

4.6.4.4. 步骤 3：创建资源描述符
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

分割 SM 资源（以及可选地添加工作队列配置资源）后，下一步是创建资源描述符。这可以通过 ``cudaDevResourceGenerateDesc`` API 完成。

.. code-block:: c++

   cudaDevResourceDesc_t resource_desc {};
   CUDA_CHECK(cudaDevResourceGenerateDesc(&resource_desc, split_result, 2));

.. _green-contexts-create-green-ctx:

4.6.4.5. 步骤 4：创建 Green Context
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

最后一步是使用 ``cudaGreenCtxCreate`` API 创建 Green Context。

.. code-block:: c++

   cudaExecutionContext_t green_ctx {};
   CUDA_CHECK(cudaGreenCtxCreate(&green_ctx, resource_desc, device_id, 0));

.. _green-contexts-launching-work:

4.6.5. Green Contexts：启动工作
------------------------------

Green Context 创建后，您可以创建属于该 Green Context 的 CUDA 流。随后在此类流上启动的 GPU 工作（如通过 ``<<< >>>`` 启动的核函数）将只能访问此 Green Context 分配的资源。

.. code-block:: c++

   cudaStream_t stream;
   CUDA_CHECK(cudaExecutionCtxStreamCreate(&stream, green_ctx, cudaStreamDefault, 0));
   
   // 在此流上启动的核函数将只使用 Green Context 的分配资源
   my_kernel<<<grid, block, 0, stream>>>(args);

库也可以轻松利用 Green Contexts，只要用户将属于 Green Context 的流传递给他们。

.. _green-contexts-apis:

4.6.6. Green Contexts：API
--------------------------

以下是与 Green Contexts 相关的一些额外 API：

- ``cudaGreenCtxDestroy``：销毁 Green Context
- ``cudaExecutionCtxGetDevResource``：获取执行上下文的设备资源
- ``cudaStreamGetDevResource``：获取 CUDA 流的设备资源
- ``cudaDevSmResourceSplitByCount``：通过计数分割 SM 资源
- ``cudaDevSmResourceSplit``：分割 SM 资源为异构组
- ``cudaDevResourceGenerateDesc``：生成资源描述符

.. _green-contexts-example:

4.6.7. Green Contexts：示例
----------------------------

以下是一个完整的示例，展示了如何使用 Green Contexts：

.. code-block:: c++

   #include <cuda_runtime.h>
   #include <iostream>

   __global__ void kernel_a() {
       // 长时间运行的核函数
       for (int i = 0; i < 1000000; i++) {
           // 一些计算
       }
   }

   __global__ void kernel_b() {
       // 延迟敏感的核函数
       // 一些关键计算
   }

   int main() {
       int device = 0;
       cudaSetDevice(device);

       // 获取初始 SM 资源
       cudaDevResource initial_resources {};
       cudaDeviceGetDevResource(device, &initial_resources, cudaDevResourceTypeSm);

       // 分割 SM 资源：80% 用于 Green Context A，20% 用于 Green Context B
       cudaDevSmResource result[2] {{}, {}};
       cudaDevSmResourceGroupParams group_params[2] =  {
               {.smCount=initial_resources.sm.smCount * 80 / 100, .flags=0},
               {.smCount=initial_resources.sm.smCount * 20 / 100, .flags=0}};
       cudaDevSmResourceSplit(&result[0], 2, &initial_resources, nullptr, 0, &group_params[0]);

       // 创建资源描述符
       cudaDevResourceDesc_t desc_a {}, desc_b {};
       cudaDevResourceGenerateDesc(&desc_a, &result[0], 1);
       cudaDevResourceGenerateDesc(&desc_b, &result[1], 1);

       // 创建 Green Contexts
       cudaExecutionContext_t ctx_a {}, ctx_b {};
       cudaGreenCtxCreate(&ctx_a, desc_a, device, 0);
       cudaGreenCtxCreate(&ctx_b, desc_b, device, 0);

       // 创建属于 Green Contexts 的流
       cudaStream_t stream_a, stream_b;
       cudaExecutionCtxStreamCreate(&stream_a, ctx_a, cudaStreamDefault, 0);
       cudaExecutionCtxStreamCreate(&stream_b, ctx_b, cudaStreamDefault, 0);

       // 启动长时间运行的核函数
       kernel_a<<<grid, block, 0, stream_a>>>();

       // 立即启动延迟敏感的核函数
       // 即使核函数 A 正在运行，它也有保证的 SM 可用
       kernel_b<<<grid, block, 0, stream_b>>>();

       // 清理
       cudaStreamDestroy(stream_a);
       cudaStreamDestroy(stream_b);
       cudaGreenCtxDestroy(ctx_a);
       cudaGreenCtxDestroy(ctx_b);

       return 0;
   }

.. note::

   有关 Green Contexts 的更多详细信息，请参考 `CUDA 官方文档 <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html>`_。
