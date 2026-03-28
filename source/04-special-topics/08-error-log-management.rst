.. _error-log-management-details:

错误日志管理
=============

错误日志管理（Error Log Management）机制允许 CUDA API 错误以通俗易懂的英语格式向开发者报告，描述问题的原因。

.. _error-log-management-background:

背景
----

传统上，CUDA API 调用失败的唯一指示是非零代码的返回。截至 CUDA Toolkit 12.9，CUDA Runtime 为错误条件定义了超过 100 种不同的返回代码，但其中许多是通用的，无法帮助开发者调试原因。

.. _error-log-management-activation:

激活
----

设置 ``CUDA_LOG_FILE`` 环境变量。可接受的值为 ``stdout``、``stderr`` 或系统中用于写入文件的有效路径。即使程序执行前未设置 ``CUDA_LOG_FILE`` ，也可以通过 API 转储日志缓冲区。

.. note::
   无错误的执行可能不会打印任何日志。

.. _error-log-management-output:

输出
----

日志按以下格式输出：

.. code-block:: c++

   [Time][TID][Source][Severity][API Entry Point] Message

以下是一行实际的错误消息，当开发者尝试将错误日志管理日志转储到未分配的缓冲区时会生成：

.. code-block:: c++

   [22:21:32.099][25642][CUDA][E][cuLogsDumpToMemory] buffer cannot be NULL

而在之前，开发者只能从返回代码中获得 ``CUDA_ERROR_INVALID_VALUE`` ，如果调用 ``cuGetErrorString`` 可能会得到"invalid argument"。

.. _error-log-management-api-description:

API 描述
--------

CUDA Driver 提供两类 API 用于与错误日志管理功能交互。

此功能允许开发者注册回调函数，以便在生成错误日志时使用，回调签名如下：

.. code-block:: c++

   void callbackFunc(void *data, CUlogLevel logLevel, char *message, size_t length)

使用此 API 注册回调：

.. code-block:: c++

   CUresult cuLogsRegisterCallback(CUlogsCallback callbackFunc, void *userData, CUlogsCallbackHandle *callback_out)

其中 ``userData`` 不经修改地传递给回调函数。 ``callback_out`` 应由调用者存储，以便在 ``cuLogsUnregisterCallback`` 中使用。

.. code-block:: c++

   CUresult cuLogsUnregisterCallback(CUlogsCallbackHandle callback)

另一组 API 函数用于管理日志输出。一个重要的概念是日志迭代器（log iterator），它指向缓冲区的当前末尾：

.. code-block:: c++

   CUresult cuLogsCurrent(CUlogIterator *iterator_out, unsigned int flags)

在不需要转储整个日志缓冲区的情况下，调用软件可以保留迭代器位置。目前，flags 参数必须为 0，其他选项保留供未来 CUDA 版本使用。

随时可以使用以下函数将错误日志缓冲区转储到文件或内存：

.. code-block:: c++

   CUresult cuLogsDumpToFile(CUlogIterator *iterator, const char *pathToFile, unsigned int flags)
   CUresult cuLogsDumpToMemory(CUlogIterator *iterator, char *buffer, size_t *size, unsigned int flags)

如果 ``iterator`` 为 NULL，则转储整个缓冲区，最多 100 条条目。如果 ``iterator`` 不为 NULL，日志将从该条目开始转储，并且 ``iterator`` 的值将更新为日志的当前末尾，就像调用了 ``cuLogsCurrent`` 一样。如果缓冲区中有超过 100 条日志条目，将在转储开头添加一条注释说明这一点。

flags 参数必须为 0，其他选项保留供未来 CUDA 版本使用。

``cuLogsDumpToMemory`` 函数还有其他注意事项：

#. 缓冲区本身将以 null 结尾，但每个日志条目之间仅用换行符（ ``\\n`` ）分隔。

#. 缓冲区的最大大小为 25600 字节。

#. 如果 ``size`` 中提供的值不足以存储所有所需的日志，将添加一条注释作为第一个条目，并且无法容纳的最旧条目将不会被转储。

#. 返回后， ``size`` 将包含实际写入所提供缓冲区的字节数。

.. _error-log-management-limitations:

限制和已知问题
--------------

#. 日志缓冲区限制为 100 条条目。达到此限制后，最旧的条目将被替换，日志转储将包含一条说明翻转的行。

#. 并非所有 CUDA API 都已覆盖。这是一个持续进行的项目，旨在为所有 API 提供更好的使用错误报告。

#. 错误日志管理的日志位置（如果给定）将不会测试其有效性，直到/除非生成日志。

#. 错误日志管理 API 目前仅通过 CUDA Driver 提供。等效的 API 将在未来版本中添加到 CUDA Runtime。

#. 日志消息未本地化为任何语言，所有提供的日志均为美式英语。