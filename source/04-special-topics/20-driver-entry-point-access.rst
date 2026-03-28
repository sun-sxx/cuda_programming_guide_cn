.. _driver-entry-point-access:

Driver Entry Point Access
==========================

.. _introduction-driver-entry-point-access:

简介
----

``Driver Entry Point Access APIs`` 提供了一种获取 CUDA driver 函数地址的方法。从 CUDA 11.3 开始，用户可以使用通过这些 API 获取的函数指针调用可用的 CUDA driver API。

这些 API 提供了类似于 POSIX 平台上 dlsym 和 Windows 上 GetProcAddress 的功能。提供的 API 允许用户：

- 使用 ``CUDA Driver API.`` 获取 driver 函数的地址。
- 使用 ``CUDA Runtime API.`` 获取 driver 函数的地址。
- 请求 CUDA driver 函数的 *per-thread default stream* 版本。更多详情请参阅 :ref:`retrieve-per-thread-default-stream-versions`。
- 在旧版 toolkit 但新版 driver 的情况下访问新的 CUDA 功能。

.. _driver-function-typedefs:

Driver Function Typedefs
-------------------------

为了帮助获取 CUDA Driver API 入口点，CUDA Toolkit 提供了包含所有 CUDA driver API 函数指针定义的头文件访问。这些头文件随 CUDA Toolkit 一起安装，位于 toolkit 的 ``include/`` 目录中。下表总结了每个 CUDA API 头文件对应的包含 ``typedefs`` 的头文件。

.. _tbl:typedefs-header-files:

.. table:: CUDA driver API 的 Typedefs 头文件

   =====================  ==============================
   API header file        API Typedef header file
   =====================  ==============================
   ``cuda.h``             ``cudaTypedefs.h``
   ``cudaGL.h``           ``cudaGLTypedefs.h``
   ``cudaProfiler.h``     ``cudaProfilerTypedefs.h``
   ``cudaVDPAU.h``        ``cudaVDPAUTypedefs.h``
   ``cudaEGL.h``          ``cudaEGLTypedefs.h``
   ``cudaD3D9.h``         ``cudaD3D9Typedefs.h``
   ``cudaD3D10.h``        ``cudaD3D10Typedefs.h``
   ``cudaD3D11.h``        ``cudaD3D11Typedefs.h``
   =====================  ==============================

上述头文件本身并不定义实际的函数指针；它们定义的是函数指针的 typedef。例如， ``cudaTypedefs.h`` 中有 driver API ``cuMemAlloc`` 的以下 typedef：

.. code-block:: c++

   typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v3020)(CUdeviceptr_v2 *dptr, size_t bytesize);
   typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v2000)(CUdeviceptr_v1 *dptr, unsigned int bytesize);

CUDA driver 符号采用基于版本的命名方案，在名称中带有 ``_v*`` 扩展（第一个版本除外）。当特定 CUDA driver API 的签名或语义发生变化时，我们会增加相应 driver 符号的版本号。以 ``cuMemAlloc`` driver API 为例，第一个 driver 符号名称是 ``cuMemAlloc`` ，下一个符号名称是 ``cuMemAlloc_v2`` 。在 CUDA 2.0 (2000) 中引入的第一个版本的 typedef 是 ``PFN_cuMemAlloc_v2000`` 。在 CUDA 3.2 (3020) 中引入的下一个版本的 typedef 是 ``PFN_cuMemAlloc_v3020`` 。

``typedefs`` 可用于更轻松地在代码中定义适当类型的函数指针：

.. code-block:: c++

   PFN_cuMemAlloc_v3020 pfn_cuMemAlloc_v2;
   PFN_cuMemAlloc_v2000 pfn_cuMemAlloc_v1;

如果用户对特定版本的 API 感兴趣，上述方法是首选。此外，头文件中还为安装 CUDA toolkit 发布时可用的所有 driver 符号的最新版本预定义了宏；这些 typedef 没有 ``_v*`` 后缀。对于 CUDA 11.3 toolkit， ``cuMemAlloc_v2`` 是最新版本，因此我们也可以如下定义其函数指针：

.. code-block:: c++

   PFN_cuMemAlloc pfn_cuMemAlloc;

.. _driver-function-retrieval:

Driver Function Retrieval
--------------------------

使用 Driver Entry Point Access API 和适当的 typedef，我们可以获取任何 CUDA driver API 的函数指针。

.. _using-the-driver-api:

使用 Driver API
^^^^^^^^^^^^^^^

driver API 需要 CUDA 版本作为参数，以获取请求的 driver 符号的 ABI 兼容版本。CUDA Driver API 具有按函数划分的 ABI，用 ``_v*`` 扩展表示。例如，考虑 ``cuStreamBeginCapture`` 的版本及其在 ``cudaTypedefs.h`` 中对应的 ``typedefs`` ：

.. code-block:: c++

   // cuda.h
   CUresult CUDAAPI cuStreamBeginCapture(CUstream hStream);
   CUresult CUDAAPI cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode);

   // cudaTypedefs.h
   typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10000)(CUstream hStream);
   typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10010)(CUstream hStream, CUstreamCaptureMode mode);

从上面代码片段中的 ``typedefs`` 来看，版本后缀 ``_v10000`` 和 ``_v10010`` 表示上述 API 分别在 CUDA 10.0 和 CUDA 10.1 中引入。

.. code-block:: c++

   #include <cudaTypedefs.h>

   // Declare the entry points for cuStreamBeginCapture
   PFN_cuStreamBeginCapture_v10000 pfn_cuStreamBeginCapture_v1;
   PFN_cuStreamBeginCapture_v10010 pfn_cuStreamBeginCapture_v2;

   // Get the function pointer to the cuStreamBeginCapture driver symbol
   cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_v1, 10000, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
   // Get the function pointer to the cuStreamBeginCapture_v2 driver symbol
   cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_v2, 10010, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);

参考上面的代码片段，要获取 driver API ``cuStreamBeginCapture`` 的 ``_v1`` 版本的地址，CUDA 版本参数应该正好是 10.0 (10000)。同样，获取 API 的 ``_v2`` 版本地址的 CUDA 版本应该是 10.1 (10010)。指定更高的 CUDA 版本来获取特定版本的 driver API 可能并不总是可移植的。例如，在这里使用 11030 仍然会返回 ``_v2`` 符号，但如果在 CUDA 11.3 中发布了假设的 ``_v3`` 版本，当与 CUDA 11.3 driver 配对时， ``cuGetProcAddress`` API 将开始返回较新的 ``_v3`` 符号。由于 ``_v2`` 和 ``_v3`` 符号的 ABI 和函数签名可能不同，使用为 ``_v2`` 符号设计的 ``_v10010`` typedef 调用 ``_v3`` 函数将表现出未定义行为。

要获取给定 CUDA Toolkit 的 driver API 的最新版本，我们还可以指定 CUDA_VERSION 作为 ``version`` 参数，并使用无版本号的 typedef 来定义函数指针。由于 ``_v2`` 是 CUDA 11.3 中 driver API ``cuStreamBeginCapture`` 的最新版本，下面的代码片段展示了获取它的另一种方法。

.. code-block:: c++

   // Assuming we are using CUDA 11.3 Toolkit

   #include <cudaTypedefs.h>

   // Declare the entry point
   PFN_cuStreamBeginCapture pfn_cuStreamBeginCapture_latest;

   // Initialize the entry point. Specifying CUDA_VERSION will give the function pointer to the
   // cuStreamBeginCapture_v2 symbol since it is latest version on CUDA 11.3.
   cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_latest, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);

注意，使用无效的 CUDA 版本请求 driver API 将返回错误 ``CUDA_ERROR_NOT_FOUND`` 。在上面的代码示例中，传入小于 10000 (CUDA 10.0) 的版本将是无效的。

.. _using-the-runtime-api:

使用 Runtime API
^^^^^^^^^^^^^^^^

runtime API ``cudaGetDriverEntryPoint`` 使用 CUDA runtime 版本获取请求的 driver 符号的 ABI 兼容版本。在下面的代码片段中，所需的最低 CUDA runtime 版本是 CUDA 11.2，因为 ``cuMemAllocAsync`` 是在当时引入的。

.. code-block:: c++

   #include <cudaTypedefs.h>

   // Declare the entry point
   PFN_cuMemAllocAsync pfn_cuMemAllocAsync;

   // Initialize the entry point. Assuming CUDA runtime version >= 11.2
   cudaGetDriverEntryPoint("cuMemAllocAsync", &pfn_cuMemAllocAsync, cudaEnableDefault, &driverStatus);

   // Call the entry point
   if(driverStatus == cudaDriverEntryPointSuccess && pfn_cuMemAllocAsync) {
       pfn_cuMemAllocAsync(...);
   }

runtime API ``cudaGetDriverEntryPointByVersion`` 使用用户提供的 CUDA 版本获取请求的 driver 符号的 ABI 兼容版本。这允许更精确地控制请求的 ABI 版本。

.. _retrieve-per-thread-default-stream-versions:

获取 Per-thread Default Stream 版本
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

某些 CUDA driver API 可以配置为具有 *default stream* 或 *per-thread default stream* 语义。具有 *per-thread default stream* 语义的 driver API 在其名称中带有 *_ptsz* 或 *_ptds* 后缀。例如， ``cuLaunchKernel`` 有一个名为 ``cuLaunchKernel_ptsz`` 的 *per-thread default stream* 变体。使用 Driver Entry Point Access API，用户可以请求 driver API ``cuLaunchKernel`` 的 *per-thread default stream* 版本，而不是 *default stream* 版本。配置 CUDA driver API 的 *default stream* 或 *per-thread default stream* 语义会影响同步行为。更多详情可以在 `这里 <https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream>`_ 找到。

driver API 的 *default stream* 或 *per-thread default stream* 版本可以通过以下方式之一获取：

- 使用编译标志 ``--default-stream per-thread`` 或定义宏 ``CUDA_API_PER_THREAD_DEFAULT_STREAM`` 来获取 *per-thread default stream* 行为。
- 使用标志 ``CU_GET_PROC_ADDRESS_LEGACY_STREAM/cudaEnableLegacyStream`` 或 ``CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM/cudaEnablePerThreadDefaultStream`` 分别强制 *default stream* 或 *per-thread default stream* 行为。

.. _access-new-cuda-features:

访问新的 CUDA 功能
^^^^^^^^^^^^^^^^^^

始终建议安装最新的 CUDA toolkit 以访问新的 CUDA driver 功能，但如果由于某种原因用户不想更新或无法访问最新的 toolkit，该 API 可以仅通过更新的 CUDA driver 来访问新的 CUDA 功能。为了讨论，假设用户使用的是 CUDA 11.3，并且想要使用 CUDA 12.0 driver 中可用的新 driver API ``cuFoo`` 。下面的代码片段说明了这个用例：

.. code-block:: c++

   int main()
   {
       // Assuming we have CUDA 12.0 driver installed.

       // Manually define the prototype as cudaTypedefs.h in CUDA 11.3 does not have the cuFoo typedef
       typedef CUresult (CUDAAPI *PFN_cuFoo)(...);
       PFN_cuFoo pfn_cuFoo = NULL;
       CUdriverProcAddressQueryResult driverStatus;

       // Get the address for cuFoo API using cuGetProcAddress. Specify CUDA version as
       // 12000 since cuFoo was introduced then or get the driver version dynamically
       // using cuDriverGetVersion
       int driverVersion;
       cuDriverGetVersion(&driverVersion);
       CUresult status = cuGetProcAddress("cuFoo", &pfn_cuFoo, driverVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);

       if (status == CUDA_SUCCESS && pfn_cuFoo) {
           pfn_cuFoo(...);
       }
       else {
           printf("Cannot retrieve the address to cuFoo - driverStatus = %d. Check if the latest driver for CUDA 12.0 is installed.\n", driverStatus);
           assert(0);
       }

       // rest of code here

   }

.. _potential-implications-with-cugetprocaddress:

cuGetProcAddress 的潜在影响
----------------------------

下面是 ``cuGetProcAddress`` 和 ``cudaGetDriverEntryPoint`` 潜在问题的一系列具体和理论示例。

.. _implications-with-cugetprocaddress-vs-implicit-linking:

cuGetProcAddress 与隐式链接的影响
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``cuDeviceGetUuid`` 在 CUDA 9.2 中引入。该 API 有一个在 CUDA 11.4 中引入的更新版本 (``cuDeviceGetUuid_v2``)。为了保持次版本兼容性，在 CUDA 12.0 之前， ``cuDeviceGetUuid`` 不会在 cuda.h 中版本升级为 ``cuDeviceGetUuid_v2`` 。这意味着通过 ``cuGetProcAddress`` 获取函数指针来调用它可能会有不同的行为。直接使用 API 的示例：

.. code-block:: c++

   #include <cuda.h>

   CUuuid uuid;
   CUdevice dev;
   CUresult status;

   status = cuDeviceGet(&dev, 0); // Get device 0
   // handle status

   status = cuDeviceGetUuid(&uuid, dev) // Get uuid of device 0

在此示例中，假设用户正在使用 CUDA 11.4 进行编译。请注意，这将执行 ``cuDeviceGetUuid`` 的行为，而不是 _v2 版本。现在看一个使用 ``cuGetProcAddress`` 的示例：

.. code-block:: c++

   #include <cudaTypedefs.h>

   CUuuid uuid;
   CUdevice dev;
   CUresult status;
   CUdriverProcAddressQueryResult driverStatus;

   status = cuDeviceGet(&dev, 0); // Get device 0
   // handle status

   PFN_cuDeviceGetUuid pfn_cuDeviceGetUuid;
   status = cuGetProcAddress("cuDeviceGetUuid", &pfn_cuDeviceGetUuid, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
   if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuid) {
       // pfn_cuDeviceGetUuid points to ???
   }

在此示例中，假设用户正在使用 CUDA 11.4 进行编译。这将获取 ``cuDeviceGetUuid_v2`` 的函数指针。然后调用该函数指针将调用新的 _v2 函数，而不是前一个示例中显示的相同 ``cuDeviceGetUuid`` 。

.. _compile-time-vs-runtime-version-usage-in-cugetprocaddress:

cuGetProcAddress 中编译时与运行时版本使用的影响
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

让我们看同一个问题并做一个小调整。最后一个示例使用编译时常量 CUDA_VERSION 来确定要获取哪个函数指针。如果用户使用 ``cuDriverGetVersion`` 或 ``cudaDriverGetVersion`` 动态查询 driver 版本并传递给 ``cuGetProcAddress`` ，会出现更多复杂性。示例：

.. code-block:: c++

   #include <cudaTypedefs.h>

   CUuuid uuid;
   CUdevice dev;
   CUresult status;
   int cudaVersion;
   CUdriverProcAddressQueryResult driverStatus;

   status = cuDeviceGet(&dev, 0); // Get device 0
   // handle status

   status = cuDriverGetVersion(&cudaVersion);
   // handle status

   PFN_cuDeviceGetUuid pfn_cuDeviceGetUuid;
   status = cuGetProcAddress("cuDeviceGetUuid", &pfn_cuDeviceGetUuid, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
   if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuid) {
       // pfn_cuDeviceGetUuid points to ???
   }

在此示例中，假设用户正在使用 CUDA 11.3 进行编译。用户会使用获取 ``cuDeviceGetUuid``（不是 _v2 版本）的已知行为来调试、测试和部署此应用程序。由于 CUDA 保证了次版本之间的 ABI 兼容性，在 driver 升级到 CUDA 11.4 后（无需更新 toolkit 和 runtime），此应用程序应该可以运行而无需重新编译。但这将产生未定义行为，因为现在 ``PFN_cuDeviceGetUuid`` 的 typedef 仍然是原始版本的签名，但由于 ``cudaVersion`` 现在是 11040 (CUDA 11.4)， ``cuGetProcAddress`` 将返回 _v2 版本的函数指针，这意味着调用它可能会产生未定义行为。

注意在这种情况下，原始（不是 _v2 版本）typedef 看起来像：

.. code-block:: c++

   typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v9020)(CUuuid *uuid, CUdevice_v1 dev);

但 _v2 版本的 typedef 看起来像：

.. code-block:: c++

   typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v11040)(CUuuid *uuid, CUdevice_v1 dev);

所以在这种情况下，API/ABI 将是相同的，runtime API 调用可能不会导致问题——只有可能返回未知的 uuid。在 :ref:`implications-to-api-abi` 中，我们讨论了一个更成问题的 API/ABI 兼容性情况。

.. _api-version-bumps-with-explicit-version-checks:

带有显式版本检查的 API 版本升级
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

上面是一个具体的示例。现在让我们使用一个理论示例，它仍然存在跨 driver 版本的兼容性问题。示例：

.. code-block:: c++

   CUresult cuFoo(int bar); // Introduced in CUDA 11.4
   CUresult cuFoo_v2(int bar); // Introduced in CUDA 11.5
   CUresult cuFoo_v3(int bar, void* jazz); // Introduced in CUDA 11.6

   typedef CUresult (CUDAAPI *PFN_cuFoo_v11040)(int bar);
   typedef CUresult (CUDAAPI *PFN_cuFoo_v11050)(int bar);
   typedef CUresult (CUDAAPI *PFN_cuFoo_v11060)(int bar, void* jazz);

注意，自 CUDA 11.4 原始创建以来，API 已被修改两次，CUDA 11.6 中的最新版本还修改了函数的 API/ABI 接口。针对 CUDA 11.5 编译的用户代码中的用法是：

.. code-block:: c++

   #include <cuda.h>
   #include <cudaTypedefs.h>

   CUresult status;
   int cudaVersion;
   CUdriverProcAddressQueryResult driverStatus;

   status = cuDriverGetVersion(&cudaVersion);
   // handle status

   PFN_cuFoo_v11040 pfn_cuFoo_v11040;
   PFN_cuFoo_v11050 pfn_cuFoo_v11050;
   if(cudaVersion < 11050 ) {
       // We know to get the CUDA 11.4 version
       status = cuGetProcAddress("cuFoo", &pfn_cuFoo_v11040, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
       // Handle status and validating pfn_cuFoo_v11040
   }
   else {
       // Assume >= CUDA 11.5 version we can use the second version
       status = cuGetProcAddress("cuFoo", &pfn_cuFoo_v11050, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
       // Handle status and validating pfn_cuFoo_v11050
   }

在此示例中，如果没有针对 CUDA 11.6 中新 typedef 的更新以及使用这些新 typedef 和情况处理重新编译应用程序，应用程序将获得返回的 cuFoo_v3 函数指针，并且使用该函数将导致未定义行为。此示例的重点是说明，即使对 ``cuGetProcAddress`` 进行显式版本检查，也可能无法安全覆盖 CUDA 主版本内的次版本升级。

.. _issues-with-runtime-api-usage:

Runtime API 使用的问题
^^^^^^^^^^^^^^^^^^^^^^

上面的示例集中在使用 Driver API 获取 driver API 函数指针的问题上。现在我们将讨论使用 Runtime API 获取 ``cudaApiGetDriverEntryPoint`` 的潜在问题。

我们将首先使用与上面类似的 Runtime API。

.. code-block:: c++

   #include <cuda.h>
   #include <cudaTypedefs.h>
   #include <cuda_runtime.h>

   CUresult status;
   cudaError_t error;
   int driverVersion, runtimeVersion;
   CUdriverProcAddressQueryResult driverStatus;

   // Ask the runtime for the function
   PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidRuntime;
   error = cudaGetDriverEntryPoint ("cuDeviceGetUuid", &pfn_cuDeviceGetUuidRuntime, cudaEnableDefault, &driverStatus);
   if(cudaSuccess == error && pfn_cuDeviceGetUuidRuntime) {
       // pfn_cuDeviceGetUuid points to ???
   }

此示例中的函数指针比上面仅使用 driver 的示例更加复杂，因为无法控制要获取哪个版本的函数；它总是获取当前 CUDA Runtime 版本的 API。有关更多信息，请参见下表：

.. table:: 静态 Runtime 版本链接

   ========================  ===========  ===========
   Driver Version Installed  V11.3        V11.4
   ========================  ===========  ===========
   V11.3                     v1           v1x
   V11.4                     v1           v2
   ========================  ===========  ===========

.. code-block:: text

   V11.3 => 11.3 CUDA Runtime and Toolkit (includes header files cuda.h and cudaTypedefs.h)
   V11.4 => 11.4 CUDA Runtime and Toolkit (includes header files cuda.h and cudaTypedefs.h)
   v1 => cuDeviceGetUuid
   v2 => cuDeviceGetUuid_v2

   x => Implies the typedef function pointer won't match the returned
        function pointer.  In these cases, the typedef at compile time
        using a CUDA 11.4 runtime, would match the _v2 version, but the
        returned function pointer would be the original (non _v2) function.

表中问题出现在使用较新的 CUDA 11.4 Runtime 和 Toolkit 与较旧 driver (CUDA 11.3) 组合的情况，在上表中标记为 v1x。这种组合会让 driver 返回指向旧函数（非 _v2）的指针，但应用程序中使用的 typedef 将是针对新函数指针的。

.. _issues-with-runtime-api-and-dynamic-versioning:

Runtime API 和动态版本控制的问题
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当我们考虑应用程序编译的 CUDA 版本、CUDA runtime 版本以及应用程序动态链接的 CUDA driver 版本的不同组合时，会出现更多复杂性。

.. code-block:: c++

   #include <cuda.h>
   #include <cudaTypedefs.h>
   #include <cuda_runtime.h>

   CUresult status;
   cudaError_t error;
   int driverVersion, runtimeVersion;
   CUdriverProcAddressQueryResult driverStatus;
   enum cudaDriverEntryPointQueryResult runtimeStatus;

   PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidDriver;
   status = cuGetProcAddress("cuDeviceGetUuid", &pfn_cuDeviceGetUuidDriver, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
   if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuidDriver) {
       // pfn_cuDeviceGetUuidDriver points to ???
   }

   // Ask the runtime for the function
   PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidRuntime;
   error = cudaGetDriverEntryPoint ("cuDeviceGetUuid", &pfn_cuDeviceGetUuidRuntime, cudaEnableDefault, &runtimeStatus);
   if(cudaSuccess == error && pfn_cuDeviceGetUuidRuntime) {
       // pfn_cuDeviceGetUuidRuntime points to ???
   }

   // Ask the driver for the function based on the driver version (obtained via runtime)
   error = cudaDriverGetVersion(&driverVersion);
   PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidDriverDriverVer;
   status = cuGetProcAddress ("cuDeviceGetUuid", &pfn_cuDeviceGetUuidDriverDriverVer, driverVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
   if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuidDriverDriverVer) {
       // pfn_cuDeviceGetUuidDriverDriverVer points to ???
   }

预期以下函数指针矩阵：

.. table:: 函数指针版本矩阵 (3 => CUDA 11.3, 4 => CUDA 11.4)

   =====================================  =======  =======  =======  =======  =======  =======  =======  =======
   Function Pointer                        3/3/3    3/3/4    3/4/3    3/4/4    4/3/3    4/3/4    4/4/3    4/4/4
   =====================================  =======  =======  =======  =======  =======  =======  =======  =======
   pfn_cuDeviceGetUuidDriver              t1/v1    t1/v1    t1/v1    t1/v1    N/A      N/A      t2/v1    t2/v2
   pfn_cuDeviceGetUuidRuntime             t1/v1    t1/v1    t1/v1    t1/v2    N/A      N/A      t2/v1    t2/v2
   pfn_cuDeviceGetUuidDriverDriverVer     t1/v1    t1/v2    t1/v1    t1/v2    N/A      N/A      t2/v1    t2/v2
   =====================================  =======  =======  =======  =======  =======  =======  =======  =======

.. code-block:: text

   tX -> Typedef version used at compile time
   vX -> Version returned/used at runtime

如果应用程序是针对 CUDA 版本 11.3 编译的，它将具有原始函数的 typedef，但如果是针对 CUDA 版本 11.4 编译的，它将具有 _v2 函数的 typedef。因此，请注意 typedef 与实际返回/使用版本不匹配的情况数量。

.. _issues-with-runtime-api-allowing-cuda-version:

Runtime API 允许 CUDA 版本的问题
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

除非另有说明，CUDA runtime API ``cudaGetDriverEntryPointByVersion`` 将具有与 driver 入口点 ``cuGetProcAddress`` 类似的影响，因为它允许用户请求特定的 CUDA driver 版本。

.. _implications-to-api-abi:

对 API/ABI 的影响
^^^^^^^^^^^^^^^^^^

在上面的 ``cuDeviceGetUuid`` 示例中，API 不匹配的影响很小，对于许多用户来说可能完全不明显，因为 _v2 是为了支持多实例 GPU (MIG) 模式而添加的。因此，在没有 MIG 的系统上，用户甚至可能没有意识到他们正在获取不同的 API。

更成问题的是 API 更改其应用程序签名（以及因此的 ABI），例如 ``cuCtxCreate`` 。在 CUDA 3.2 中引入的 _v2 版本目前在使用 ``cuda.h`` 时作为默认的 ``cuCtxCreate`` 使用，但现在有一个在 CUDA 11.4 中引入的更新版本 (``cuCtxCreate_v3``)。API 签名也被修改了，现在接受额外的参数。因此，在上面的某些情况下，函数指针的 typedef 与返回的函数指针不匹配，这可能会导致非明显的 ABI 不兼容，从而导致未定义行为。

例如，假设以下代码是针对 CUDA 11.3 toolkit 编译的，并安装了 CUDA 11.4 driver：

.. code-block:: c++

   PFN_cuCtxCreate cuUnknown;
   CUdriverProcAddressQueryResult driverStatus;

   status = cuGetProcAddress("cuCtxCreate", (void**)&cuUnknown, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
   if(CUDA_SUCCESS == status && cuUnknown) {
       status = cuUnknown(&ctx, 0, dev);
   }

运行此代码时，如果 ``cudaVersion`` 设置为任何 >=11040（表示 CUDA 11.4）的值，可能会产生未定义行为，因为没有为 ``cuCtxCreate_v3`` API 的 _v3 版本提供所需的全部参数。

.. _determining-cugetprocaddress-failure-reasons:

确定 cuGetProcAddress 失败原因
-------------------------------

cuGetProcAddress 有两种类型的错误。它们是 (1) API/使用错误和 (2) 无法找到请求的 driver API。第一种错误类型将通过 CUresult 返回值从 API 返回错误代码。比如将 NULL 作为 ``pfn`` 变量传递或传递无效的 ``flags`` 。

第二种错误类型编码在 ``CUdriverProcAddressQueryResult *symbolStatus`` 中，可用于帮助区分 driver 无法找到请求符号的潜在问题。请看以下示例：

.. code-block:: c++

   // cuDeviceGetExecAffinitySupport was introduced in release CUDA 11.4
   #include <cuda.h>
   CUdriverProcAddressQueryResult driverStatus;
   cudaVersion = ...;
   status = cuGetProcAddress("cuDeviceGetExecAffinitySupport", &pfn, cudaVersion, 0, &driverStatus);
   if (CUDA_SUCCESS == status) {
       if (CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT == driverStatus) {
           printf("We can use the new feature when you upgrade cudaVersion to 11.4, but CUDA driver is good to go!\n");
           // Indicating cudaVersion was < 11.4 but run against a CUDA driver >= 11.4
       }
       else if (CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND == driverStatus) {
           printf("Please update both CUDA driver and cudaVersion to at least 11.4 to use the new feature!\n");
           // Indicating driver is < 11.4 since string not found, doesn't matter what cudaVersion was
       }
       else if (CU_GET_PROC_ADDRESS_SUCCESS == driverStatus && pfn) {
           printf("You're using cudaVersion and CUDA driver >= 11.4, using new feature!\n");
           pfn();
       }
   }

第一个返回代码 ``CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT`` 表示在 CUDA driver 中搜索时找到了 ``symbol`` ，但它是在提供的 ``cudaVersion`` 之后添加的。在示例中，将 ``cudaVersion`` 指定为 11030 或更低，并在 CUDA driver >= CUDA 11.4 上运行时，会得到 ``CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT`` 的结果。这是因为 ``cuDeviceGetExecAffinitySupport`` 是在 CUDA 11.4 (11040) 中添加的。

第二个返回代码 ``CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND`` 表示在 CUDA driver 中搜索时未找到 ``symbol`` 。这可能是由于多种原因，例如由于 driver 较旧而不支持 CUDA 函数，以及仅仅是拼写错误。在后一种情况下，类似于最后一个示例，如果用户将 ``symbol`` 设为 CUDeviceGetExecAffinitySupport——注意开头的字母 CU 是大写的——``cuGetProcAddress`` 将无法找到该 API，因为字符串不匹配。在前一种情况下，示例可能是用户针对支持新 API 的 CUDA driver 开发应用程序，并将应用程序部署在较旧的 CUDA driver 上。使用最后一个示例，如果开发人员针对 CUDA 11.4 或更高版本进行开发，但部署在 CUDA 11.3 driver 上，在开发期间他们可能成功调用了 ``cuGetProcAddress`` ，但在部署针对 CUDA 11.3 driver 运行的应用程序时，该调用将不再工作，并在 ``driverStatus`` 中返回 ``CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND`` 。