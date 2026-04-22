.. _driver-api:

3.3. CUDA Driver API
====================

本指南前面的章节介绍了 CUDA runtime。如 :ref:`cuda-platform-driver-and-runtime` 中所述，CUDA runtime 是在较低级别的 CUDA driver API 之上构建的。本节介绍 CUDA runtime 和 driver API 之间的一些区别，以及如何混合使用它们。大多数应用程序无需与 CUDA driver API 交互即可达到最佳性能。但是，新接口有时会更早地在 driver API 中提供，而一些高级接口（如 :doc:`../04-special-topics/16-virtual-memory-management` ）仅在 driver API 中公开。

Driver API 在 `cuda` 动态库（`cuda.dll` 或 `cuda.so` ）中实现，该库在安装设备驱动程序时复制到系统中。其所有入口点都以 `cu` 为前缀。

这是一个基于句柄的、命令式 API：大多数对象由不透明句柄引用，可以将这些句柄指定给函数来操作对象。

Driver API 中可用的对象总结在 :numref:`driver-api-objects-available-in-cuda-driver-api` 中。

.. _driver-api-objects-available-in-cuda-driver-api:
.. list-table:: CUDA Driver API 中可用的对象
   :header-rows: 1

   * - 对象
     - 句柄
     - 描述
   * - Device
     - CUdevice
     - 支持 CUDA 的设备
   * - Context
     - CUcontext
     - 大致相当于 CPU 进程
   * - Module
     - CUmodule
     - 大致相当于动态库
   * - Function
     - CUfunction
     - Kernel
   * - Heap memory
     - CUdeviceptr
     - 指向设备内存的指针
   * - CUDA array
     - CUarray
     - 设备上一维或二维数据的不透明容器，可通过纹理或表面引用读取
   * - Texture object
     - CUtexref
     - 描述如何解释纹理内存数据的对象
   * - Surface reference
     - CUsurfref
     - 描述如何读取或写入 CUDA 数组的对象
   * - Stream
     - CUstream
     - 描述 CUDA 流的对象
   * - Event
     - CUevent
     - 描述 CUDA 事件的对象

在调用 driver API 中的任何函数之前，必须使用 `cuInit()` 初始化 driver API。然后必须创建一个 CUDA context，并将其附加到特定设备并使其成为调用主机线程的当前 context，详见 :ref:`driver-api-context`。

在 CUDA context 中，kernel 由主机代码显式加载为 PTX 或二进制对象，如 :ref:`driver-api-module` 中所述。因此，用 C++ 编写的 kernel 必须单独编译为 *PTX* 或二进制对象。使用 API 入口点启动 kernel，如 :ref:`driver-api-kernel-execution` 中所述。

任何希望在未来设备架构上运行的应用程序都必须加载 *PTX*，而不是二进制代码。这是因为二进制代码是特定于架构的，因此与未来架构不兼容，而 *PTX* 代码在加载时由设备驱动程序编译为二进制代码。

以下是使用 driver API 编写的 :ref:`kernels` 示例的主机代码：

.. code-block:: cuda

   int main()
   {
       int N = ...;
       size_t size = N * sizeof(float);

       // Allocate input vectors h_A and h_B in host memory
       float* h_A = (float*)malloc(size);
       float* h_B = (float*)malloc(size);

       // Initialize input vectors
       ...

       // Initialize
       cuInit(0);

       // Get number of devices supporting CUDA
       int deviceCount = 0;
       cuDeviceGetCount(&deviceCount);
       if (deviceCount == 0) {
           printf("There is no device supporting CUDA.\n");
           exit (0);
       }

       // Get handle for device 0
       CUdevice cuDevice;
       cuDeviceGet(&cuDevice, 0);

       // Create context
       CUcontext cuContext;
       cuCtxCreate(&cuContext, 0, cuDevice);

       // Create module from binary file
       CUmodule cuModule;
       cuModuleLoad(&cuModule, "VecAdd.ptx");

       // Allocate vectors in device memory
       CUdeviceptr d_A;
       cuMemAlloc(&d_A, size);
       CUdeviceptr d_B;
       cuMemAlloc(&d_B, size);
       CUdeviceptr d_C;
       cuMemAlloc(&d_C, size);

       // Copy vectors from host memory to device memory
       cuMemcpyHtoD(d_A, h_A, size);
       cuMemcpyHtoD(d_B, h_B, size);

       // Get function handle from module
       CUfunction vecAdd;
       cuModuleGetFunction(&vecAdd, cuModule, "VecAdd");

       // Invoke kernel
       int threadsPerBlock = 256;
       int blocksPerGrid =
               (N + threadsPerBlock - 1) / threadsPerBlock;
       void* args[] = { &d_A, &d_B, &d_C, &N };
       cuLaunchKernel(vecAdd,
                      blocksPerGrid, 1, 1, threadsPerBlock, 1, 1,
                      0, 0, args, 0);

       ...
   }

完整代码可以在 `vectorAddDrv` CUDA 示例中找到。

.. _driver-api-context:

3.3.1. Context
--------------

CUDA context 类似于 CPU 进程。在 driver API 中执行的所有资源和操作都封装在 CUDA context 内，当 context 被销毁时，系统会自动清理这些资源。除了模块和纹理或表面引用等对象外，每个 context 都有自己的独立地址空间。因此，来自不同 context 的 `CUdeviceptr` 值引用不同的内存位置。

主机线程一次只能有一个设备 context 为当前 context。当使用 `cuCtxCreate()` 创建 context 时，它成为调用主机线程的当前 context。在 context 中操作的 CUDA 函数（大多数不涉及设备枚举或 context 管理的函数）如果线程没有有效的当前 context，将返回 `CUDA_ERROR_INVALID_CONTEXT`。

每个主机线程都有一个当前 context 的栈。`cuCtxCreate()` 将新 context 推入栈顶。可以调用 `cuCtxPopCurrent()` 将 context 从主机线程分离。然后 context 处于「浮动」状态，可以作为任何主机线程的当前 context 推入。`cuCtxPopCurrent()` 还会恢复之前的当前 context（如果有的话）。

每个 context 还维护一个使用计数。`cuCtxCreate()` 创建使用计数为 1 的 context。`cuCtxAttach()` 增加使用计数，`cuCtxDetach()` 减少使用计数。当调用 `cuCtxDetach()` 或 `cuCtxDestroy()` 时使用计数变为 0 时，context 被销毁。

Driver API 与 runtime 可互操作，可以通过 `cuDevicePrimaryCtxRetain()` 从 driver API 访问 runtime 管理的 primary context（参见 :ref:`intro-cpp-runtime-initialization` ）。

使用计数便于在同一个 context 中运行的第三方代码之间的互操作。例如，如果加载三个库使用同一个 context，每个库都会调用 `cuCtxAttach()` 增加使用计数，并在库使用完 context 时调用 `cuCtxDetach()` 减少使用计数。对于大多数库，预期应用程序会在加载或初始化库之前创建 context；这样，应用程序可以使用自己的启发式方法创建 context，库只需操作传递给它的 context。希望创建自己 context 的库——其 API 客户端可能不知道或可能没有创建自己的 context——将使用 `cuCtxPushCurrent()` 和 `cuCtxPopCurrent()`，如下图所示。

.. _library-context-management:
.. figure:: /_static/images/library-context-management.png
   :alt: 库 Context 管理

   库 Context 管理

.. _driver-api-module:

3.3.2. Module
-------------

Module 是设备代码和数据的动态可加载包，类似于 Windows 中的 DLL，由 nvcc 输出（参见 :ref:`compilation-with-nvcc` ）。所有符号的名称，包括函数、全局变量以及纹理或表面引用，都在 module 范围内维护，以便由独立的第三方编写的 module 可以在同一个 CUDA context 中互操作。

此代码示例加载一个 module 并检索某个 kernel 的句柄：

.. code-block:: cuda

   CUmodule cuModule;
   cuModuleLoad(&cuModule, "myModule.ptx");
   CUfunction myKernel;
   cuModuleGetFunction(&myKernel, cuModule, "MyKernel");

此代码示例从 PTX 代码编译并加载新 module，并解析编译错误：

.. code-block:: cuda

   #define BUFFER_SIZE 8192
   CUmodule cuModule;
   CUjit_option options[3];
   void* values[3];
   char* PTXCode = "some PTX code";
   char error_log[BUFFER_SIZE];
   int err;
   options[0] = CU_JIT_ERROR_LOG_BUFFER;
   values[0]  = (void*)error_log;
   options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
   values[1]  = (void*)BUFFER_SIZE;
   options[2] = CU_JIT_TARGET_FROM_CUCONTEXT;
   values[2]  = 0;
   err = cuModuleLoadDataEx(&cuModule, PTXCode, 3, options, values);
   if (err != CUDA_SUCCESS)
       printf("Link error:\n%s\n", error_log);

此代码示例从多个 PTX 代码编译、链接并加载新 module，并解析链接和编译错误：

.. code-block:: cuda

   #define BUFFER_SIZE 8192
   CUmodule cuModule;
   CUjit_option options[6];
   void* values[6];
   float walltime;
   char error_log[BUFFER_SIZE], info_log[BUFFER_SIZE];
   char* PTXCode0 = "some PTX code";
   char* PTXCode1 = "some other PTX code";
   CUlinkState linkState;
   int err;
   void* cubin;
   size_t cubinSize;
   options[0] = CU_JIT_WALL_TIME;
   values[0] = (void*)&walltime;
   options[1] = CU_JIT_INFO_LOG_BUFFER;
   values[1] = (void*)info_log;
   options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
   values[2] = (void*)BUFFER_SIZE;
   options[3] = CU_JIT_ERROR_LOG_BUFFER;
   values[3] = (void*)error_log;
   options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
   values[4] = (void*)BUFFER_SIZE;
   options[5] = CU_JIT_LOG_VERBOSE;
   values[5] = (void*)1;
   cuLinkCreate(6, options, values, &linkState);
   err = cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
                       (void*)PTXCode0, strlen(PTXCode0) + 1, 0, 0, 0, 0);
   if (err != CUDA_SUCCESS)
       printf("Link error:\n%s\n", error_log);
   err = cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
                       (void*)PTXCode1, strlen(PTXCode1) + 1, 0, 0, 0, 0);
   if (err != CUDA_SUCCESS)
       printf("Link error:\n%s\n", error_log);
   cuLinkComplete(linkState, &cubin, &cubinSize);
   printf("Link completed in %fms. Linker Output:\n%s\n", walltime, info_log);
   cuModuleLoadData(cuModule, cubin);
   cuLinkDestroy(linkState);

可以使用多线程加速 module 链接/加载过程的某些部分，包括加载 cubin 时。此代码示例使用 `CU_JIT_BINARY_LOADER_THREAD_COUNT` 加速 module 加载。

.. code-block:: cuda

   #define BUFFER_SIZE 8192
   CUmodule cuModule;
   CUjit_option options[3];
   void* values[3];
   char* cubinCode = "some cubin code";
   char error_log[BUFFER_SIZE];
   int err;
   options[0] = CU_JIT_ERROR_LOG_BUFFER;
   values[0]  = (void*)error_log;
   options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
   values[1]  = (void*)BUFFER_SIZE;
   options[2] = CU_JIT_BINARY_LOADER_THREAD_COUNT;
   values[2]  = 0; // Use as many threads as CPUs on the machine
   err = cuModuleLoadDataEx(&cuModule, cubinCode, 3, options, values);
   if (err != CUDA_SUCCESS)
       printf("Link error:\n%s\n", error_log);

完整代码可以在 `ptxjit` CUDA 示例中找到。

.. _driver-api-kernel-execution:

3.3.3. Kernel 执行
------------------

`cuLaunchKernel()` 以给定的执行配置启动 kernel。

参数可以作为指针数组（`cuLaunchKernel()` 的倒数第二个参数）传递，其中第 n 个指针对应第 n 个参数并指向从中复制参数的内存区域，或作为额外选项之一（`cuLaunchKernel()` 的最后一个参数）传递。

当参数作为额外选项传递（`CU_LAUNCH_PARAM_BUFFER_POINTER` 选项）时，它们作为指向单个缓冲区的指针传递，其中假设参数根据设备代码中每种参数类型的对齐要求彼此正确偏移。

设备代码中内建向量类型的对齐要求列在 :numref:`vector-types-alignment-requirements-in-device-code` 中。对于所有其他基本类型，设备代码中的对齐要求与主机代码中的对齐要求匹配，因此可以使用 `__alignof()` 获得。唯一的例外是当主机编译器在单字边界而不是双字边界上对齐 `double` 和 `long long`（以及 64 位系统上的 `long` ）时（例如，使用 `gcc` 的编译标志 `-mno-align-double` ），因为在设备代码中这些类型始终在双字边界上对齐。

`CUdeviceptr` 是整数，但表示指针，因此其对齐要求是 `__alignof(void*)`。

以下代码示例使用宏（`ALIGN_UP()` ）调整每个参数的偏移以满足其对齐要求，并使用另一个宏（`ADD_TO_PARAM_BUFFER()` ）将每个参数添加到传递给 `CU_LAUNCH_PARAM_BUFFER_POINTER` 选项的参数缓冲区。

.. code-block:: cuda

   #define ALIGN_UP(offset, alignment) \
         (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

   char paramBuffer[1024];
   size_t paramBufferSize = 0;

   #define ADD_TO_PARAM_BUFFER(value, alignment)                   \
       do {                                                        \
           paramBufferSize = ALIGN_UP(paramBufferSize, alignment); \
           memcpy(paramBuffer + paramBufferSize,                   \
                  &(value), sizeof(value));                        \
           paramBufferSize += sizeof(value);                       \
       } while (0)

   int i;
   ADD_TO_PARAM_BUFFER(i, __alignof(i));
   float4 f4;
   ADD_TO_PARAM_BUFFER(f4, 16); // float4's alignment is 16
   char c;
   ADD_TO_PARAM_BUFFER(c, __alignof(c));
   float f;
   ADD_TO_PARAM_BUFFER(f, __alignof(f));
   CUdeviceptr devPtr;
   ADD_TO_PARAM_BUFFER(devPtr, __alignof(devPtr));
   float2 f2;
   ADD_TO_PARAM_BUFFER(f2, 8); // float2's alignment is 8

   void* extra[] = {
       CU_LAUNCH_PARAM_BUFFER_POINTER, paramBuffer,
       CU_LAUNCH_PARAM_BUFFER_SIZE,    &paramBufferSize,
       CU_LAUNCH_PARAM_END
   };
   cuLaunchKernel(cuFunction,
                  blockWidth, blockHeight, blockDepth,
                  gridWidth, gridHeight, gridDepth,
                  0, 0, 0, extra);

结构的对齐要求等于其字段对齐要求的最大值。因此，包含内建向量类型、`CUdeviceptr` 或非对齐的 `double` 和 `long long` 的结构，在设备代码和主机代码之间可能有不同的对齐要求。这种结构也可能有不同的填充。例如，以下结构在主机代码中根本没有填充，但在设备代码中在字段 `f` 之后填充了 12 个字节，因为字段 `f4` 的对齐要求是 16。

.. code-block:: cuda

   typedef struct {
       float  f;
       float4 f4;
   } myStruct;

.. _driver-api-interop-with-runtime: 

3.3.4. Runtime 和 Driver API 之间的互操作性
--------------------------------------------

应用程序可以混合使用 runtime API 代码和 driver API 代码。

如果通过 driver API 创建并设置为当前 context，后续的 runtime 调用将使用此 context 而不是创建新的 context。

如果 runtime 已初始化，可以使用 `cuCtxGetCurrent()` 检索初始化期间创建的 context。此 context 可以被后续的 driver API 调用使用。

从 runtime 隐式创建的 context 称为 primary context（参见 :ref:`intro-cpp-runtime-initialization` ）。可以使用 `Primary Context Management <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html>`__ 函数从 driver API 管理它。

可以使用任一 API 分配和释放设备内存。`CUdeviceptr` 可以转换为常规指针，反之亦然：

.. code-block:: cuda

   CUdeviceptr devPtr;
   float* d_data;

   // Allocation using driver API
   cuMemAlloc(&devPtr, size);
   d_data = (float*)devPtr;

   // Allocation using runtime API
   cudaMalloc(&d_data, size);
   devPtr = (CUdeviceptr)d_data;

特别是，这意味着使用 driver API 编写的应用程序可以调用使用 runtime API 编写的库（如 cuFFT、cuBLAS 等）。

参考手册的设备和版本管理部分中的所有函数可以互换使用。
