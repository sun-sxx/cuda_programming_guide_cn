.. _graphics-interop-details:

4.19. CUDA 与图形 API 互操作
============================

在 CUDA 中直接访问 API 的 GPU 数据可以读取和写入带有 CUDA 核函数的数据，从而提供 CUDA 功能，同时从其他 API 使用它们。主要有两个概念：直接方法，`图形互操作性`_ 与 OpenGL 和 Direct3D[9-11] 使得可以将资源从 OpenGL 和 Direct3D 映射到 CUDA 地址空间；以及更灵活的 `外部资源互操作性`_，其中可以通过导入和导出操作系统级句柄来访问内存和同步对象。这支持以下 API：Direct3D[11-12]、Vulkan 和 NVIDIA 软件通信接口互操作性。

.. _graphics-interoperability:

4.19.1. 图形互操作性
--------------------

在使用 CUDA 访问 Direct3D 或 OpenGL 资源（例如 VBO（顶点缓冲对象））之前，必须先注册和映射它。使用相应的 CUDA 函数（见下面的示例）进行注册将返回类型为 ``struct cudaGraphicsResource`` 的 CUDA 图形资源，该资源保存 CUDA 设备指针或数组。要访问核函数中的设备数据，必须映射资源。资源注册后，可以根据需要多次映射和取消映射。映射的资源通过使用 ``cudaGraphicsResourceGetMappedPointer()`` 返回的设备内存地址（用于缓冲区）和 ``cudaGraphicsSubResourceGetMappedArray()``（用于 CUDA 数组）由核函数访问。一旦 CUDA 不再需要该资源，可以取消注册。这些是主要步骤：

1. 使用 CUDA 注册图形缓冲区
2. 映射资源
3. 访问映射资源的设备指针或数组
4. 在 CUDA 核函数中使用设备指针或数组
5. 取消映射资源
6. 取消注册资源

请注意，注册资源代价高昂，因此理想情况下每个资源只调用一次，但是对于打算使用该资源的每个 CUDA 上下文，需要分别注册资源。可以调用 ``cudaGraphicsResourceSetMapFlags()`` 来指定使用提示（只写、只读），CUDA 驱动程序可以使用这些提示来优化资源管理。另外请注意，在资源映射时通过 OpenGL、Direct3D 或不同的 CUDA 上下文访问资源会产生未定义的结果。

.. _opengl-interoperability:

4.19.1.1. OpenGL 互操作性
^^^^^^^^^^^^^^^^^^^^^^^^^

可以映射到 CUDA 地址空间的 OpenGL 资源是 OpenGL 缓冲区、纹理和渲染缓冲区对象。使用 ``cudaGraphicsGLRegisterBuffer()`` 注册缓冲区对象，在 CUDA 中，它显示为普通设备指针。使用 ``cudaGraphicsGLRegisterImage()`` 注册纹理或渲染缓冲区对象，在 CUDA 中，它显示为 CUDA 数组。

如果使用 ``cudaGraphicsRegisterFlagsSurfaceLoadStore`` 标志注册了纹理或渲染缓冲区对象，则可以写入它。``cudaGraphicsGLRegisterImage()`` 支持具有 1、2 或 4 个分量且内部类型为 float（例如 ``GL_RGBA_FLOAT32``）、归一化整数（例如 ``GL_RGBA8, GL_INTENSITY16``）和非归一化整数（例如 ``GL_RGBA8UI``）的所有纹理格式。

**示例：simpleGL 互操作性**

以下代码示例使用核函数动态修改存储在顶点缓冲对象 (VBO) 中的 2D ``width`` x ``height`` 顶点网格，并经过以下主要步骤：

1. 使用 CUDA 注册 VBO
2. 循环：从 CUDA 映射 VBO 进行写入
3. 循环：运行 CUDA 核函数修改顶点位置
4. 循环：取消映射 VBO
5. 循环：使用 OpenGL 渲染结果
6. 取消注册并删除 VBO

此部分的完整示例 simpleGL 可以在这里找到：`NVIDIA/cuda-samples <https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleGL>`_

.. code-block:: cuda

   __global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time)
   {
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
       unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

       // 计算 uv 坐标
       float u = x / (float)width;
       float v = y / (float)height;
       u = u * 2.0f - 1.0f;
       v = v * 2.0f - 1.0f;

       // 计算简单的正弦波模式
       float freq = 4.0f;
       float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;

       // 写入输出顶点
       pos[y * width + x] = make_float4(u, w, v, 1.0f);
   }

.. code-block:: c++

   void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags)
   {
       assert(vbo);

       // 创建缓冲区对象
       glGenBuffers(1, vbo);
       glBindBuffer(GL_ARRAY_BUFFER, *vbo);

       // 初始化缓冲区对象
       unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
       glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

       glBindBuffer(GL_ARRAY_BUFFER, 0);

       // 使用 CUDA 注册此缓冲区对象
       checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

       SDK_CHECK_ERROR_GL();
   }

   void display()
   {
       float4 *dptr;
       // 2. 从 CUDA 映射 VBO 进行写入
       checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
       size_t num_bytes;
       checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cuda_vbo_resource));

       // 3. 运行 CUDA 核函数修改顶点位置
       dim3 block(8, 8, 1);
       dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
       simple_vbo_kernel<<<grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);

       // 4. 取消映射 VBO
       checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

       glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

       // 设置视图矩阵
       glMatrixMode(GL_MODELVIEW);
       glLoadIdentity();
       glTranslatef(0.0, 0.0, translate_z);
       glRotatef(rotate_x, 1.0, 0.0, 0.0);
       glRotatef(rotate_y, 0.0, 1.0, 0.0);

       // 5. 使用 OpenGL 渲染更新的内容
       glBindBuffer(GL_ARRAY_BUFFER, vbo);
       glVertexPointer(4, GL_FLOAT, 0, 0);

       glEnableClientState(GL_VERTEX_ARRAY);
       glColor3f(1.0, 0.0, 0.0);
       glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
       glDisableClientState(GL_VERTEX_ARRAY);

       glutSwapBuffers();

       g_fAnim += 0.01f;
   }

   void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
   {
       // 6. 取消注册并删除 VBO
       checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

       glBindBuffer(1, *vbo);
       glDeleteBuffers(1, vbo);

       *vbo = 0;
   }

**限制和注意事项**：

- 正在共享其资源的 OpenGL 上下文必须对进行任何 OpenGL 互操作性 API 调用的主机线程当前有效。
- 当 OpenGL 纹理设置为无绑定（例如，通过使用 ``glGetTextureHandle`` 或 ``glGetImageHandle`` API 请求图像或纹理句柄）时，它不能与 CUDA 注册。应用程序需要在请求图像或纹理句柄之前为互操作注册纹理。

.. _direct3d-interoperability:

4.19.1.2. Direct3D 互操作性
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Direct3D 互操作性支持 Direct3D9、Direct3D10 和 Direct3D11，但不支持 Direct3D12，这里我们关注 Direct3D11，对于 Direct3D9 和 Direct3D10 请参阅 CUDA 编程指南 12.9。可以映射到 CUDA 地址空间的 Direct3D 资源是 Direct3D 缓冲区、纹理和表面。这些资源使用 ``cudaGraphicsD3D11RegisterResource()`` 注册。

CUDA 上下文只能与使用 ``DriverType`` 设置为 ``D3D_DRIVER_TYPE_HARDWARE`` 创建的 Direct3D11 设备互操作。

**示例：2D 纹理 Direct3D11 互操作性**

以下代码片段来自 simpleD3D11Texture 示例：`NVIDIA/cuda-samples <https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleD3D11Texture>`_。完整示例包括大量 DX11 样板代码，这里我们关注 CUDA 方面。

CUDA 核函数 ``cuda_kernel_texture_2d`` 在频闪蓝色背景上绘制移动的红色/绿色阴影图案的 2D 纹理，它依赖于先前的纹理值。底层数据是 2D CUDA 数组，其中行偏移由 pitch 定义。

.. code-block:: cuda

   /*
    * 在频闪蓝色背景上绘制移动的红色/绿色阴影图案的 2D 纹理。
    * 请注意，此核函数读取和写入纹理，因此为什么此纹理未映射为 WriteDiscard。
    */
   __global__ void cuda_kernel_texture_2d(unsigned char *surface, int width,
                                         int height, size_t pitch, float t) {
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
     float *pixel;

     // 在由于量化为网格而我们有比像素更多线程的情况下，跳过不对应有效像素的线程
     if (x >= width || y >= height) return;

     // 获取 (x,y) 处像素的指针
     pixel = (float *)(surface + y * pitch) + 4 * x;

     // 填充它
     float value_x = 0.5f + 0.5f * cos(t + 10.0f * ((2.0f * x) / width - 1.0f));
     float value_y = 0.5f + 0.5f * cos(t + 10.0f * ((2.0f * y) / height - 1.0f));
     pixel[0] = 0.5 * pixel[0] + 0.5 * pow(value_x, 3.0f);  // 红色
     pixel[1] = 0.5 * pixel[1] + 0.5 * pow(value_y, 3.0f);  // 绿色
     pixel[2] = 0.5f + 0.5f * cos(t);                       // 蓝色
     pixel[3] = 1;                                          // 阿尔法
   }

为了使指针和数据缓冲区保持在一起，使用以下数据结构：

.. code-block:: c++

   // 在 DX11 和 CUDA 之间共享的 2D 纹理数据结构
   struct {
     ID3D11Texture2D *pTexture;
     ID3D11ShaderResourceView *pSRView;
     cudaGraphicsResource *cudaResource;
     void *cudaLinearMemory;
     size_t pitch;
     int width;
     int height;
     int offsetInShader;
   } g_texture_2d;

在初始化 Direct3D 设备和纹理后，资源向 CUDA 注册一次。为了匹配 Direct3D 像素格式，CUDA 数组分配有相同的宽度和高度，以及与 Direct3D 纹理行 pitch 匹配的 pitch。

.. code-block:: c++

   // 注册 CUDA 核函数中使用的 Direct3D 资源
   // 我们将读取和写入 g_texture_2d，因此不为其设置任何特殊映射标志
   cudaGraphicsD3D11RegisterResource(&g_texture_2d.cudaResource,
                                     g_texture_2d.pTexture,
                                     cudaGraphicsRegisterFlagsNone);
   getLastCudaError("cudaGraphicsD3D11RegisterResource (g_texture_2d) failed");
   
   // CUDA 无法直接写入纹理：纹理被视为 cudaArray，只能映射为纹理
   // 创建缓冲区以便 CUDA 可以写入
   // 像素格式为 DXGI_FORMAT_R32G32B32A32_FLOAT
   cudaMallocPitch(&g_texture_2d.cudaLinearMemory, &g_texture_2d.pitch,
                   g_texture_2d.width * sizeof(float) * 4,
                   g_texture_2d.height);
   getLastCudaError("cudaMallocPitch (g_texture_2d) failed");
   cudaMemset(g_texture_2d.cudaLinearMemory, 1,
              g_texture_2d.pitch * g_texture_2d.height);

在渲染循环中，映射资源，启动 CUDA 核函数更新纹理数据，然后取消映射资源。在此步骤之后，使用 Direct3D 设备在屏幕上绘制更新的纹理。

.. code-block:: c++

   cudaStream_t stream = 0;
   const int nbResources = 3;
   cudaGraphicsResource *ppResources[nbResources] = {
       g_texture_2d.cudaResource, g_texture_3d.cudaResource,
       g_texture_cube.cudaResource,
   };
   cudaGraphicsMapResources(nbResources, ppResources, stream);
   getLastCudaError("cudaGraphicsMapResources(3) failed");

   // 运行将填充这些纹理内容的核函数
   RunKernels();

   // 取消映射资源
   cudaGraphicsUnmapResources(nbResources, ppResources, stream);
   getLastCudaError("cudaGraphicsUnmapResources(3) failed");

最后，一旦 CUDA 不再需要这些资源，它们将取消注册并释放设备数组。

.. code-block:: c++

   // 取消注册 Cuda 资源
   cudaGraphicsUnregisterResource(g_texture_2d.cudaResource);
   getLastCudaError("cudaGraphicsUnregisterResource (g_texture_2d) failed");
   cudaFree(g_texture_2d.cudaLinearMemory);
   getLastCudaError("cudaFree (g_texture_2d) failed");

.. _sli-interoperability:

4.19.1.3. 可扩展链接接口 (SLI) 配置中的互操作性
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在具有多个 GPU 的系统中，所有支持 CUDA 的 GPU 都可以通过 CUDA 驱动程序和运行时作为单独的设备访问。当系统处于 SLI 模式时，这是不同的。SLI 是一种硬件配置的多 GPU 配置，通过跨多个 GPU 划分工作负载来提供更高的渲染性能。隐式 SLI 模式（驱动程序进行假设）不再受支持，但显式 SLI 仍受支持。显式 SLI 意味着应用程序通过 API（例如 Vulkan、DirectX、GL）了解和管理 SLI 组中所有设备的 SLI 状态。

当系统处于 SLI 模式时有特殊考虑：

- 一个 GPU 上的一个 CUDA 设备中的分配将消耗属于 Direct3D 或 OpenGL 设备 SLI 配置的其他 GPU 上的内存。因此，分配可能比预期更早失败。
- 应用程序应创建多个 CUDA 上下文，SLI 配置中的每个 GPU 一个。虽然这不是严格要求，但可以避免设备之间不必要的数据传输。应用程序可以使用 ``cudaD3D[9|10|11]GetDevices()``（用于 Direct3D）和 ``cudaGLGetDevices()``（用于 OpenGL）调用来识别当前帧和下一帧中执行渲染的设备的 CUDA 设备句柄。鉴于此信息，应用程序通常会选择合适的设备，并在 ``deviceList`` 参数设置为 ``cudaD3D[9|10|11]DeviceListCurrentFrame`` 或 ``cudaGLDeviceListCurrentFrame`` 时将 Direct3D 或 OpenGL 资源映射到 ``cudaD3D[9|10|11]GetDevices()`` 或 ``cudaGLGetDevices()`` 返回的 CUDA 设备。
- 从 ``cudaGraphicsD3D[9|10|11]RegisterResource`` 和 ``cudaGraphicsGLRegister[Buffer|Image]`` 返回的资源必须仅在发生注册的设备上使用。因此，在 SLI 配置中，当在不同 CUDA 设备上计算不同帧的数据时，需要分别为每个资源注册。

.. _external-resource-interoperability:

4.19.2. 外部资源互操作性
--------------------------

外部资源互操作性允许 CUDA 导入由 API 显式导出的某些资源。这些对象通常使用操作系统原生的句柄导出，如 Linux 上的文件描述符或 Windows 上的 NT 句柄。这允许在其他 API 和 CUDA 之间高效共享资源，而无需在之间复制或复制。它支持以下 API：Direct3D[11-12]、Vulkan 和 NVIDIA 软件通信接口互操作性。有两种类型的资源可以导入：

**内存对象**

可以使用 ``cudaImportExternalMemory()`` 导入到 CUDA 中。导入的内存对象可以使用 ``cudaExternalMemoryGetMappedBuffer()`` 映射到内存对象上的设备指针或使用 ``cudaExternalMemoryGetMappedMipmappedArray()`` 映射的 CUDA 分层数组在内核中访问。根据内存对象的类型，可能在单个内存对象上设置多个映射。映射必须与导出 API 的映射设置匹配。任何不匹配的映射都会导致未定义行为。导入的内存对象必须使用 ``cudaDestroyExternalMemory()`` 释放。释放内存对象不会释放对该对象的任何映射。因此，必须使用 ``cudaFree()`` 显式释放映射到该对象的任何设备指针，并且必须使用 ``cudaFreeMipmappedArray()`` 显式释放映射到该对象的任何 CUDA 分层数组。在对象被销毁后访问该对象的映射是非法的。

**同步对象**

可以使用 ``cudaImportExternalSemaphore()`` 导入到 CUDA 中。导入的同步对象可以使用 ``cudaSignalExternalSemaphoresAsync()`` 发出信号并使用 ``cudaWaitExternalSemaphoresAsync()`` 等待。在发出相应信号之前发出等待是非法的。另外，根据导入的同步对象的类型，对如何发出信号和等待可能有额外的约束，如后续部分所述。导入的信号量对象必须使用 ``cudaDestroyExternalSemaphore()`` 释放。在销毁信号量对象之前，所有未完成的信号和等待必须已完成。

.. _vulkan-interoperability:

4.19.2.1. Vulkan 互操作性
^^^^^^^^^^^^^^^^^^^^^^^^^

Vulkan 图形和计算工作负载在同一硬件上的耦合执行可以最大化 GPU 利用率并避免不必要的复制。注意，这不是 Vulkan 指南，我们只关注与 CUDA 的互操作性，有关 Vulkan 指南请参阅 `https://www.vulkan.org/learn#vulkan-tutorials <https://www.vulkan.org/learn#vulkan-tutorials>`_。

使 Vulkan-CUDA 互操作性工作的主要步骤涉及：

1. 初始化 Vulkan，创建并导出外部缓冲区和/或同步对象
2. 使用匹配的 UUID 设置 Vulkan 正在运行的 CUDA 设备
3. 获取内存和/或同步句柄
4. 使用这些句柄在 CUDA 中导入内存和/或同步对象
5. 将设备指针或分层数组映射到内存对象上
6. 通过对同步对象发出信号和等待来定义执行顺序，从而在 CUDA 和 Vulkan 中交替使用导入的内存对象。

在本节中，将借助 *simpleVulkan* 示例逐步解释上述步骤：`NVIDIA/cuda-samples <https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleVulkan>`_

**设置 Vulkan 设备**

为了导出内存对象，必须创建启用了 ``VK_KHR_external_memory_capabilities`` 扩展的 Vulkan 实例，以及启用了 ``VK_KHR_external_memory`` 的设备。此外，必须启用特定于平台的句柄类型，对于 Windows 为 ``VK_KHR_external_memory_win32``，对于基于 UNIX 的系统为 ``VK_KHR_external_memory_fd``。

类似地，对于导出同步对象，需要在设备级别启用 ``VK_KHR_external_semaphore_capabilities`` 和 ``VK_KHR_external_semaphore``，在实例级别也需要启用。以及用于句柄的特定于平台的扩展，即 Windows 的 ``VK_KHR_external_semaphore_win32`` 和基于 Unix 的系统的 ``VK_KHR_external_semaphore_fd``。

**使用匹配的 UUID 初始化 CUDA**

导入 Vulkan 导出的内存和同步对象时，必须在创建它们的同一设备上导入和映射。可以通过比较 CUDA 设备的 UUID 与 Vulkan 物理设备的 UUID 来确定与 Vulkan 物理设备对应的 CUDA 设备。

**导出 Vulkan 内存对象**

为了导出 Vulkan 内存对象，必须创建具有相应导出标志的缓冲区。

**导出 Vulkan 同步对象**

Vulkan API 在 GPU 上执行的调用是异步的。为了定义执行顺序，Vulkan 中有信号量和围栏可以与 CUDA 共享。与内存对象类似，信号量可以由 Vulkan 导出，它们需要使用导出标志创建，具体取决于信号量的类型。有二进制信号量和时间轴信号量。

**导入内存对象**

Vulkan 导出的专用和非专用内存对象都可以导入到 CUDA 中。导入 Vulkan 专用内存对象时，必须设置 ``cudaExternalMemoryDedicated`` 标志。

**将缓冲区映射到导入的内存对象**

导入内存对象后，必须映射它们才能使用。可以将设备指针映射到导入的内存对象上。

**将分层数组映射到导入的内存对象**

可以将 CUDA 分层数组映射到导入的内存对象上。

**导入同步对象**

可以使用与对象关联的文件描述符或 NT 句柄将 Vulkan 信号量对象导入到 CUDA 中。

**对导入的同步对象发出信号/等待**

可以按需要对导入的 Vulkan 信号量发出信号和等待。

.. note::

   有关 CUDA 与图形 API 互操作性的更多详细信息，请参考 `CUDA 官方文档 <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html>`_。
