.. _driver-api:

3.3. CUDA Driver API
====================

本指南前面的章节介绍了 CUDA runtime。
如前面 :ref:`cuda-runtime-api-and-cuda-driver-api` 中所述，CUDA runtime 是在更底层的 driver API 之上构建的。
本节将探讨这两者之间的一些区别，以及如何在代码中混合使用它们。
对于大多数应用程序来说，即使完全不使用 Driver API，也依然能发挥出 GPU 的全部性能。
不过，Driver API 有时候会比 Runtime 更早推出一些新接口； 而一些高级接口（如 :ref:`虚拟内存管理 <virtual-memory-management-details>` ）仅在 driver API 中提供。

Driver API 在 ``cuda`` 动态库（ ``cuda.dll`` 或 ``cuda.so``）中提供，该库在安装设备驱动程序时复制到系统中。其所有入口函数都以 ``cu`` 为前缀。

这是一种基于句柄（handle-based）的命令式 API：大多数对象都是通过不透明的句柄来引用的，你可以将这些句柄作为参数传递给各种函数，从而操作相应的对象。

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
     - | 设备上的一维或二维数据的不透明容器，
       | 可以通过纹理（texture）或表面（surface）引用进行读取。
   * - Texture object
     - CUtexref
     - 描述如何解读纹理内存数据的对象
   * - Surface reference
     - CUsurfref
     - 描述如何读写 CUDA 数组的对象
   * - Stream
     - CUstream
     - 描述 CUDA 流的对象
   * - Event
     - CUevent
     - 描述 CUDA 事件的对象

在调用任何 driver API 函数之前，必须先通过 ``cuInit()`` 来初始化 driver API。
然后，需要创建一个绑定到特定设备（GPU）的 CUDA ``Context`` （上下文），并将其设置为当前主机线程的活动上下文，具体细节可以参考 :ref:`Context<driver-api-context>` 。

在 CUDA context 中，kernels 以 PTX 或二进制对象的形式被主机代码显示加载，见 :ref:`driver-api-module` 。
因此，用 C++ 编写的 kernel 必须被编译为 **PTX** 或二进制对象。 
Kernel 执行则使用 :ref:`driver-api-kernel-execution` 提供的 API。

任何希望在未来设备架构上运行的应用程序，都应该使用 **PTX** 代码，而不是二进制代码。
这是因为二进制代码是特定于具体硬件架构的，因此无法兼容未来的新架构；而 **PTX** 代码则会在加载时，由设备驱动程序实时编译成适配当前硬件的二进制代码。

以下是使用 driver API 编写的 :ref:`kernels` 章节示例中的主机代码部分。

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

完整代码可以在 ``vectorAddDrv`` CUDA 示例中找到。

.. _driver-api-context:

3.3.1. Context
----------------

CUDA Context （上下文） 类似于 CPU 进程。
在 driver API 中执行的所有操作以及使用的所有资源，都被封装进一个 CUDA context 中；当这个 context 被销毁时，系统会自动清理掉这些资源。
除了模块、纹理或表面引用等对象之外，每个 context 还拥有自己独立的地址空间。
因此，来自不同 context 的 ``CUdeviceptr`` 指向的是完全不同的内存位置。

一个主机线程在同一时刻只能有一个处于当前状态的设备上下文 （device context current）。
当使用 ``cuCtxCreate()`` 创建一个 context 时，它会自动成为调用该函数的主机线程的当前 context。
绝大多数需要在 context 中运行的 CUDA 函数（除了那些用来枚举设备或管理上下文的函数之外），
如果当前线程没有绑定一个有效的 context，将会返回 ``CUDA_ERROR_INVALID_CONTEXT`` 错误。

每个主机线程内部都维护着一个 **当前上下文** 的栈（stack）。
当你调用 ``cuCtxCreate()`` 时，新创建的 context 会被压入到这个栈的顶端。
你可以使用 ``cuCtxPopCurrent()`` 来将 context 从主机线程中剥离出来。
此时，这个 context 就变成了一个 **游离状态（floating）**，随后它可以被压入到任意其他主机线程的栈顶，成为那个线程的当前上下文。
此外， ``cuCtxPopCurrent()`` 还会自动恢复之前处于栈中的上一个当前上下文（如果有的话）。

每个上下文都会维护一个使用计数。
``cuCtxCreate()`` 在创建上下文时，将使用计数初始化为 `1` 。
``cuCtxAttach()`` 会让这个使用计数加 `1` ，而 ``cuCtxDetach()`` 则会使其减 `1` 。
只有当调用 ``cuCtxDetach()`` 或 ``cuCtxDestroy()`` 导致使用计数降为 `0` 时，这个上下文才会被真正销毁。

Driver API 与 Runtime API 是可以互相兼容（互操作）的。
Runtime API 管理的那个主上下文（ 见 :ref:`intro-cpp-runtime-initialization` ）可以通过 Driver API 中的函数 ``cuDevicePrimaryCtxRetain()`` 来访问。

使用计数机制极大地方便了多个第三方代码库在同一个 context 中协同工作。
举个例子，假设应用程序有三个库需要使用 context ，那么每个库都可以通过 ``cuCtxAttach()`` 来增加使用计数，并在该库不再需要时通过 ``cuCtxDetach()`` 来减少计数。
对于大多数库来说，通常预期应用程序会在加载或初始化这些库之前，就已经创建好了一个 context 。
这样，应用程序可以按照自己的策略来创建 context ，而库只需要直接使用传递给它的 context 即可。
不过，如果某些库想要创建独立的 context （而且不希望被调用者感知，毕竟调用者可能创建、也可能没创建），
那么这些库就需要使用 ``cuCtxPushCurrent()`` 和 ``cuCtxPopCurrent()`` 来管理，具体操作就像下图所示的那样。

.. _library-context-management:
.. figure:: /_static/images/library-context-management.png
   :alt: 库 Context 管理

   库 Context 管理

.. _driver-api-module:

3.3.2. Module
-------------

Module 是动态可加载的设备侧代码和数据的集合包，类似于 Windows 系统中的 DLL ，由 nvcc 编译生成（参见 :ref:`compilation-with-nvcc` ）。
所有的符号名称——包括函数、全局变量以及纹理或表面引用，都保持在模块的作用域范围内。
这样一来，由不同第三方独立编写的模块就可以在同一个 CUDA context 中互相协作了。

以下示例展示加载一个 module 并检索某个 kernel 的句柄：

.. code-block:: cuda

   CUmodule cuModule;
   cuModuleLoad(&cuModule, "myModule.ptx");
   CUfunction myKernel;
   cuModuleGetFunction(&myKernel, cuModule, "MyKernel");

以下示例展示使用 PTX 代码编译并加载新 module，并解析编译错误：

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

以下示例展示从多个 PTX 代码编译、链接并加载新 module，并解析链接和编译错误：

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

可以使用多线程加速 module 链接/加载过程的某些部分，包括加载 cubin 时。
此代码示例使用 ``CU_JIT_BINARY_LOADER_THREAD_COUNT`` 加速 module 加载。

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

完整代码可以在 ``ptxjit`` CUDA 示例中找到。

.. _driver-api-kernel-execution:

3.3.3. Kernel 执行
------------------

``cuLaunchKernel()`` 以给定的执行配置启动 kernel。

参数的传递方式有两种：
一种是通过一个指针数组来传递（对应 ``cuLaunchKernel()`` 函数的倒数第二个参数），在这个数组中，第 n 个指针对应着第 n 个参数，并且指向一块内存区域，参数值就是从这块区域拷贝过去的；
另一种则是作为 **额外选项** 之一来传递（对应 ``cuLaunchKernel()`` 函数的最后一个参数）。

当参数作为额外选项（即 ``CU_LAUNCH_PARAM_BUFFER_POINTER`` 选项）来传递时，它们会被当作一个指向缓冲区的指针。
在这个缓冲区中，各个参数需要满足设备端代码中对应参数类型的内存对齐要求，彼此之间保持正确的偏移量。

内置向量类型在设备端代码中的对齐要求已在 :ref:`表格 43<built-in-types>` 中列出。
对于所有其他的基本类型，它们在设备端代码中的对齐要求与主机端代码的对齐要求是一致的，因此可以直接使用 ``__alignof()`` 来获取。
唯一的例外情况是：当主机编译器将 ``double``、  ``long long`` （以及 64 位系统上的 ``long`` ）按照单字（one-word）而不是双字（two-word）对齐时（例如使用了 ``gcc`` 编译选项 ``-mno-align-double`` ）。
因为无论什么情况，这些类型在设备端代码中始终是按照双字边界来对齐的。

``CUdeviceptr`` 虽然是整数，但它用来表示指针，所以它的内存其对齐要求是 ``__alignof(void*)``。

下面的代码示例使用了两个宏：一个是 ``ALIGN_UP()`` ，用来调整每个参数的偏移量，使其满足各自的内存对齐要求；
另一个是 ``ADD_TO_PARAM_BUFFER()`` ，用来把每个参数添加到传给 ``CU_LAUNCH_PARAM_BUFFER_POINTER`` 选项的参数缓冲区中。

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

结构的对齐要求等于其字段对齐要求的最大值。因此，包含内建向量类型、``CUdeviceptr`` 或非对齐的 ``double`` 和 ``long long`` 的结构，在设备代码和主机代码之间可能有不同的对齐要求。这种结构也可能有不同的填充。例如，以下结构在主机代码中根本没有填充，但在设备代码中在字段 ``f`` 之后填充了 12 个字节，因为字段 ``f4`` 的对齐要求是 16。

一个结构体的对齐要求，等于其所有字段中对齐要求的最大值。
因此，如果一个结构体里包含了内置向量类型、 ``CUdeviceptr`` 或者未对齐的 ``double`` 和 ``long long`` ，那么它在设备端代码和主机端代码中的对齐要求可能会有所不同。
这种结构体的填充方式（ ``padding`` ）也可能不一样。
比如下面这个结构体，在主机端代码中是完全不需要填充的；但在设备端代码中，由于字段 ``f4`` 的对齐要求是 ``16`` 字节，所以会在字段 ``f`` 后面额外填充 12 个字节。

.. code-block:: cuda

   typedef struct {
       float  f;
       float4 f4;
   } myStruct;


.. _driver-api-interop-with-runtime:

3.3.4. Runtime 和 Driver API 之间的互操作性
--------------------------------------------

应用程序可以混合使用 runtime API 和 driver API 。

如果通过 Driver API 创建 context 并设为当前的，那么后续调用 Runtime API 时就会直接使用这个现有的 context ，而不会再创建一个新的。

同样如果 Runtime 已经被初始化，就可以使用 Driver API ``cuCtxGetCurrent()`` 来获取在 runtime 初始化期间创建的 context，并被后续的 driver API 使用。

由 Runtime （ :ref:`初始化<intro-cpp-runtime-initialization>` 时）隐式创建的 context ，被称为主上下文（primary context）。
它可以通过 Driver API 中的 `Primary Context Management <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html>`__ 进行管理。

设备内存可以使用任意一套 API（Driver API 或 Runtime API）来进行分配和释放。 ``CUdeviceptr`` 可以被强制转换为普通的指针，反之亦然。

.. code-block:: cuda

   CUdeviceptr devPtr;
   float* d_data;

   // Allocation using driver API
   cuMemAlloc(&devPtr, size);
   d_data = (float*)devPtr;

   // Allocation using runtime API
   cudaMalloc(&d_data, size);
   devPtr = (CUdeviceptr)d_data;

特别是，这意味着使用 Driver API 编写的应用程序，可以调用那些使用 Runtime API 编写的库（例如 cuFFT、cuBLAS 等）。

参考手册中设备管理和版本管理章节里的所有函数，都可以互相替换使用。
