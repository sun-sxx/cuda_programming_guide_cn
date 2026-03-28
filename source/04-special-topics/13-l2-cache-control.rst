.. _l2-cache-control:

L2 缓存控制
============

.. _advanced-kernels-l2-control:

当 CUDA kernel 反复访问全局内存中的某个数据区域时，这种数据访问可以被认为是持久化的。另一方面，如果数据只被访问一次，这种数据访问可以被认为是流式的。

计算能力 8.0 及以上的设备能够影响数据在 L2 缓存中的持久性，从而可能提供更高的全局内存访问带宽和更低的延迟。

此功能通过两个主要 API 公开：

- CUDA runtime API（从 CUDA 11.0 开始）提供对 L2 缓存持久性的编程控制。
- ``libcu++`` 库中的 ``cuda::annotated_ptr`` API（从 CUDA 11.5 开始）在 CUDA kernel 中用内存访问属性注释指针，以达到类似的效果。

以下部分主要介绍 CUDA runtime API。关于 ``cuda::annotated_ptr`` 方法的详细信息，请参阅 `libcu++ 文档 <https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_access_properties/annotated_ptr.html>`_。

.. _l2-set-aside:

L2 缓存预留空间用于持久化访问
------------------------------

可以将一部分 L2 缓存预留出来，用于全局内存的持久化数据访问。持久化访问优先使用这部分预留的 L2 缓存，而普通或流式访问全局内存时只能在预留部分未被持久化访问使用时才能利用它。

用于持久化访问的 L2 缓存预留大小可以在限制范围内调整：

.. code-block:: c++

   cudaGetDeviceProperties(&prop, device_id);
   size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
   cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* 将 L2 缓存的 3/4 预留用于持久化访问，或使用最大允许值 */

当 GPU 配置为多实例 GPU（Multi-Instance GPU，MIG）模式时，L2 缓存预留功能将被禁用。

当使用多进程服务（Multi-Process Service，MPS）时，L2 缓存预留大小不能通过 ``cudaDeviceSetLimit`` 更改。相反，预留大小只能在 MPS 服务器启动时通过环境变量 ``CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT`` 指定。

.. _l2-access-policy:

持久化访问的 L2 策略
--------------------

访问策略窗口指定全局内存的连续区域以及该区域内访问在 L2 缓存中的持久性属性。

下面的代码示例展示了如何使用 CUDA Stream 设置 L2 持久化访问窗口。

**CUDA Stream 示例**

.. code-block:: c++

   cudaStreamAttrValue stream_attribute;                                          // Stream 级别属性数据结构
   stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr);  // 全局内存数据指针
   stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                     // 持久化访问的字节数
                                                                                  // （必须小于 cudaDeviceProp::accessPolicyMaxWindowSize）
   stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                           // 缓存命中率的提示
   stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;  // 缓存命中时的访问属性类型
   stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;   // 缓存未命中时的访问属性类型

   // 将属性设置到 cudaStream_t 类型的 CUDA stream
   cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

当 kernel 随后在 CUDA ``stream`` 中执行时，全局内存范围 ``[ptr..ptr+num_bytes)`` 内的内存访问比其他全局内存位置的访问更有可能持久化在 L2 缓存中。

L2 持久性也可以为 CUDA Graph Kernel Node 设置，如下例所示：

**CUDA GraphKernelNode 示例**

.. code-block:: c++

   cudaKernelNodeAttrValue node_attribute;                                        // Kernel 级别属性数据结构
   node_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr);    // 全局内存数据指针
   node_attribute.accessPolicyWindow.num_bytes = num_bytes;                       // 持久化访问的字节数
                                                                                  // （必须小于 cudaDeviceProp::accessPolicyMaxWindowSize）
   node_attribute.accessPolicyWindow.hitRatio  = 0.6;                             // 缓存命中率的提示
   node_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;    // 缓存命中时的访问属性类型
   node_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;     // 缓存未命中时的访问属性类型

   // 将属性设置到 cudaGraphNode_t 类型的 CUDA Graph Kernel 节点
   cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute);

``hitRatio`` 参数可用于指定接收 ``hitProp`` 属性的访问比例。在上面的两个示例中，全局内存区域 ``[ptr..ptr+num_bytes)`` 中 60% 的内存访问具有持久化属性，40% 的内存访问具有流式属性。哪些特定的内存访问被分类为持久化（即 ``hitProp`` ）是随机的，概率约为 ``hitRatio`` ；概率分布取决于硬件架构和内存范围。

例如，如果 L2 预留缓存大小为 16KB，而 ``accessPolicyWindow`` 中的 ``num_bytes`` 为 32KB：

- 当 ``hitRatio`` 为 0.5 时，硬件将随机选择 32KB 窗口中的 16KB 被指定为持久化，并缓存在预留的 L2 缓存区域中。
- 当 ``hitRatio`` 为 1.0 时，硬件将尝试将整个 32KB 窗口缓存在预留的 L2 缓存区域中。由于预留区域小于窗口，缓存行将被驱逐以保持 32KB 数据中最近使用的 16KB 留在 L2 缓存的预留部分。

因此，可以使用 ``hitRatio`` 来避免缓存行的颠簸，并总体上减少移入和移出 L2 缓存的数据量。

低于 1.0 的 ``hitRatio`` 值可用于手动控制来自并发 CUDA stream 的不同 ``accessPolicyWindow`` 可以在 L2 中缓存的数据量。例如，假设 L2 预留缓存大小为 16KB；两个不同 CUDA stream 中的两个并发 kernel，各自有 16KB 的 ``accessPolicyWindow`` ，且 ``hitRatio`` 值都为 1.0，在竞争共享的 L2 资源时可能会相互驱逐对方的缓存行。但是，如果两个 ``accessPolicyWindow`` 的 hitRatio 值都为 0.5，它们就不太可能驱逐自己或对方的持久化缓存行。

.. _l2-access-prop:

L2 访问属性
-----------

为不同的全局内存数据访问定义了三种类型的访问属性：

1. ``cudaAccessPropertyStreaming`` ：以流式属性发生的内存访问不太可能持久化在 L2 缓存中，因为这些访问会被优先驱逐。

2. ``cudaAccessPropertyPersisting`` ：以持久化属性发生的内存访问更有可能持久化在 L2 缓存中，因为这些访问会被优先保留在 L2 缓存的预留部分。

3. ``cudaAccessPropertyNormal`` ：此访问属性强制将先前应用的持久化访问属性重置为正常状态。来自先前 CUDA kernel 的具有持久化属性的内存访问可能会在其预期用途之后很长时间仍保留在 L2 缓存中。这种使用后的持久性减少了不使用持久化属性的后续 kernel 可用的 L2 缓存量。使用 ``cudaAccessPropertyNormal`` 属性重置访问策略窗口可以移除先前访问的持久化（优先保留）状态，就像先前访问没有访问属性一样。

.. _l2-simple-example:

L2 持久化示例
-------------

以下示例展示了如何为持久化访问预留 L2 缓存，在 CUDA kernel 中通过 CUDA Stream 使用预留的 L2 缓存，然后重置 L2 缓存。

.. code-block:: c++

   cudaStream_t stream;
   cudaStreamCreate(&stream);                                                                  // 创建 CUDA stream

   cudaDeviceProp prop;                                                                        // CUDA 设备属性变量
   cudaGetDeviceProperties( &prop, device_id);                                                 // 查询 GPU 属性
   size_t size = min( int(prop.l2CacheSize * 0.75) , prop.persistingL2CacheMaxSize );
   cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size);                                  // 将 L2 缓存的 3/4 预留用于持久化访问，或使用最大允许值

   size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes);                        // 选择用户定义的 num_bytes 和最大窗口大小中的较小值

   cudaStreamAttrValue stream_attribute;                                                       // Stream 级别属性数据结构
   stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data1);            // 全局内存数据指针
   stream_attribute.accessPolicyWindow.num_bytes = window_size;                                // 持久化访问的字节数
   stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                                        // 缓存命中率的提示
   stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;               // 持久化属性
   stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;                // 缓存未命中时的访问属性类型

   cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);  // 将属性设置到 CUDA Stream

   for(int i = 0; i < 10; i++) {
       cuda_kernelA<<<grid_size,block_size,0,stream>>>(data1);                                 // data1 被 kernel 多次使用
   }                                                                                           // [data1 + num_bytes) 从 L2 持久化中受益
   cuda_kernelB<<<grid_size,block_size,0,stream>>>(data1);                                      // 同一 stream 中的不同 kernel 也可以
                                                                                               // 从 data1 的持久化中受益

   stream_attribute.accessPolicyWindow.num_bytes = 0;                                          // 将窗口大小设置为 0 以禁用它
   cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // 覆盖 CUDA Stream 的访问策略属性
   cudaCtxResetPersistingL2Cache();                                                            // 移除 L2 中的任何持久化行

   cuda_kernelC<<<grid_size,block_size,0,stream>>>(data2);                                      // data2 现在可以在正常模式下受益于完整的 L2

.. _l2-reset-to-normal:

将 L2 访问重置为正常
----------------------

来自先前 CUDA kernel 的持久化 L2 缓存行可能会在使用后很长时间仍持久化在 L2 中。因此，将 L2 缓存重置为正常状态对于流式或正常内存访问以正常优先级利用 L2 缓存很重要。有三种方法可以将持久化访问重置为正常状态。

1. 使用访问属性 ``cudaAccessPropertyNormal`` 重置先前的持久化内存区域。

2. 通过调用 ``cudaCtxResetPersistingL2Cache()`` 将所有持久化 L2 缓存行重置为正常。

3. **最终** 未触及的行会自动重置为正常。强烈不建议依赖自动重置，因为自动重置所需的时间长度不确定。

.. _l2-managing-utilization:

管理 L2 预留缓存的利用率
------------------------

在不同 CUDA stream 中并发执行的多个 CUDA kernel 可能有不同的访问策略窗口分配给它们的 stream。然而，L2 预留缓存部分在所有这些并发 kernel 之间是共享的。因此，该预留缓存部分的净利用率是所有并发 kernel 单独使用的总和。随着持久化访问量超过预留 L2 缓存容量，将内存访问指定为持久化的好处会减少。

要管理预留 L2 缓存部分的利用率，应用程序必须考虑以下几点：

- L2 预留缓存的大小。
- 可能并发执行的 CUDA kernel。
- 所有可能并发执行的 CUDA kernel 的访问策略窗口。
- 何时以及如何需要 L2 重置，以允许正常或流式访问以同等优先级利用先前预留的 L2 缓存。

.. _l2-cache-query:

查询 L2 缓存属性
----------------

与 L2 缓存相关的属性是 ``cudaDeviceProp`` 结构的一部分，可以使用 CUDA runtime API ``cudaGetDeviceProperties`` 查询。

CUDA 设备属性包括：

- ``l2CacheSize`` ：GPU 上可用的 L2 缓存量。
- ``persistingL2CacheMaxSize`` ：可以为持久化内存访问预留的最大 L2 缓存量。
- ``accessPolicyMaxWindowSize`` ：访问策略窗口的最大大小。

.. _l2-cache-getset-size:

控制持久化内存访问的 L2 缓存预留大小
-------------------------------------

用于持久化内存访问的 L2 预留缓存大小使用 CUDA runtime API ``cudaDeviceGetLimit`` 查询，并使用 CUDA runtime API ``cudaDeviceSetLimit`` 作为 ``cudaLimit`` 设置。设置此限制的最大值是 ``cudaDeviceProp::persistingL2CacheMaxSize`` 。

.. code-block:: c++

   enum cudaLimit {
       /* 其他字段未显示 */
       cudaLimitPersistingL2CacheSize
   };