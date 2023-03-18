# 如何修复无法分配请求的地址:服务“sparkDriver”在 16 次重试后失败

> 原文：<https://towardsdatascience.com/spark-fix-cant-assign-driver-32406580375>

## 了解如何解决最常见的 PySpark 问题之一

![](img/0604ae9c6b51435a5f275d491e7decd7.png)

艾蒂安·吉拉尔代在 [Unsplash](https://unsplash.com/s/photos/error?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

## 介绍

当涉及到配置时，Spark networking 有时会变得相当具有挑战性，有时可能会出现一些奇怪的错误。我个人看到的最常见的错误之一是`Service 'sparkDriver' failed after 16 retries`错误。错误的完整追溯分享如下。

```
Exception in thread "main" java.net.BindException: Can't assign requested address: Service 'sparkDriver' failed after 16 retries (on a random free port)! Consider explicitly setting the appropriate binding address for the service 'sparkDriver' (for example spark.driver.bindAddress for SparkDriver) to the correct binding address.
    at sun.nio.ch.Net.bind0(Native Method)
    at sun.nio.ch.Net.bind(Net.java:433)
    at sun.nio.ch.Net.bind(Net.java:425)
    at sun.nio.ch.ServerSocketChannelImpl.bind(ServerSocketChannelImpl.java:223)
    at io.netty.channel.socket.nio.NioServerSocketChannel.doBind(NioServerSocketChannel.java:127)
    at io.netty.channel.AbstractChannel$AbstractUnsafe.bind(AbstractChannel.java:501)
    at io.netty.channel.DefaultChannelPipeline$HeadContext.bind(DefaultChannelPipeline.java:1218)
    at io.netty.channel.AbstractChannelHandlerContext.invokeBind(AbstractChannelHandlerContext.java:496)
    at io.netty.channel.AbstractChannelHandlerContext.bind(AbstractChannelHandlerContext.java:481)
    at io.netty.channel.DefaultChannelPipeline.bind(DefaultChannelPipeline.java:965)
    at io.netty.channel.AbstractChannel.bind(AbstractChannel.java:210)
    at io.netty.bootstrap.AbstractBootstrap$2.run(AbstractBootstrap.java:353)
    at io.netty.util.concurrent.SingleThreadEventExecutor.runAllTasks(SingleThreadEventExecutor.java:399)
    at io.netty.channel.nio.NioEventLoop.run(NioEventLoop.java:446)
    at io.netty.util.concurrent.SingleThreadEventExecutor$2.run(SingleThreadEventExecutor.java:131)
    at io.netty.util.concurrent.DefaultThreadFactory$DefaultRunnableDecorator.run(DefaultThreadFactory.java:144)
    at java.lang.Thread.run(Thread.java:745)

Process finished with exit code 1
```

在今天的简短教程中，我们将探索一些潜在的解决方法，最终可以帮助您处理这个错误。

## 设置 SPARK_LOCAL_IP 环境变量

Spark 使用驱动程序中指定的配置参数或作为位于安装目录中的`conf/spark-env.sh`(或`conf/spark-env.cmd`)脚本的一部分加载的环境变量，确定 JVM 如何在工作节点上初始化。

在运行本地 Spark 应用程序或提交脚本时，也会用到这个脚本文件。注意安装 Spark 时`conf/spark-env.sh`默认不存在。然而，GitHub 上有一个名为`[conf/spark-env.sh.template](https://github.com/apache/spark/blob/master/conf/spark-env.sh.template)`的模板文件，你可以用它作为起点。您还必须确保该文件是可执行的。

`SPARK_LOCAL_IP`环境变量对应于要绑定到的机器的 IP 地址。因此，您可以通过向`conf/spark-env.sh`脚本添加以下命令来指定 Spark 在该节点上绑定的 IP 地址

```
export SPARK_LOCAL_IP="127.0.0.1"
```

或者，由于`conf/spark-env.sh`是一个 shell 脚本，您可以通过查找特定网络接口的 IP 来编程计算`SPARK_LOCAL_IP`，而不是将其硬编码为一个特定的值。

## 配置 spark.driver.bindAddress

如前一节所述，Spark 还可以使用驱动程序中指定的配置来确定 JVM 如何在工作节点上初始化。

换句话说，通过为`spark.driver.bindAddress`提供一个配置，您可以在初始化`SparkSession`时以编程方式设置绑定监听套接字的主机名或 IP 地址。

但是请注意，这种方法将覆盖`SPARK_LOCAL_IP`环境变量的值，如果它已经在我们在上一节描述的`conf/spark-env.sh`脚本中导出的话。

> `*spark.driver.bindAddress*`:绑定监听套接字的主机名或 IP 地址。这个配置覆盖了`*SPARK_LOCAL_IP*`环境变量。
> 
> 它还允许将与本地地址不同的地址公布给执行器或外部系统。例如，当运行具有桥接网络的容器时，这是有用的。为了正常工作，驱动程序使用的不同端口(RPC、块管理器和 UI)需要从容器的主机转发。
> 
> —来源:[火花文件](https://spark.apache.org/docs/latest/configuration.html)

在 Scala 中，您可以配置 IP 地址来绑定监听套接字，如下所示。

```
import org.apache.spark.sql.SparkSessionval spark: SparkSession = SparkSession.builder() \
    .appName("Test application") \
    .master("local[*]") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()
```

PySpark 中的等效表达式是

```
from pyspark.sql import SparkSessionspark = SparkSession.builder \
    .appName("Test application") \    
    .master("local[1]") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()
```

## 在本地机器上设置主机名

第三个选项是通过命令行在本地机器上手动指定`hostname`。主机名是作为计算机或服务器名称的唯一标识符。

可以使用以下命令调整该名称:

```
sudo hostname -s 127.0.0.1
```

## 检查您的 VPN 连接

最后，如果您在本地机器上运行 Spark，另一种可能性是您的虚拟专用网络可能会影响绑定监听套接字的主机名或 IP 地址。

如果是这种情况，您可以通过简单地断开 VPN 或禁用任何其他可能影响网络的工具来解决这个问题。

## 最后的想法

在今天的简短教程中，我们讨论了 Spark 中一个常见的报告错误，即与 Spark 网络相关的`java.net.BindException: Can't assign requested address: Service 'sparkDriver' failed after 16 retries`。

我们展示了可以采取哪些措施来解决这个问题。总而言之，这可以通过以下方式实现

*   导出在工作节点上初始化 JVM 时加载的相应的`SPARK_LOCAL_IP`环境变量
*   在`SparkSession`中设置相应的`spark.driver.bindAddress`配置(注意这种方法将覆盖`SPARK_LOCAL_IP`环境变量)
*   更新本地机器上的`hostname`
*   检查您是否启用了虚拟专用网络(VPN)或任何其他可能影响本地计算机联网的工具，因为这有时可能会影响绑定地址

一旦应用了本文中描述的任何解决方案，您最终应该能够顺利运行 Spark 应用程序。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读媒体上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership) [## 通过我的推荐链接加入 Medium-Giorgos Myrianthous

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

gmyrianthous.medium.com](https://gmyrianthous.medium.com/membership) 

**相关文章你可能也喜欢**

[](/how-to-efficiently-convert-a-pyspark-dataframe-to-pandas-8bda2c3875c3) [## 加快 PySpark 和 Pandas 数据帧之间的转换

### 将大火花数据帧转换为熊猫时节省时间

towardsdatascience.com](/how-to-efficiently-convert-a-pyspark-dataframe-to-pandas-8bda2c3875c3) [](/sparksession-vs-sparkcontext-vs-sqlcontext-vs-hivecontext-741d50c9486a) [## spark session vs spark context vs SQLContext vs hive context

### SparkSession、SparkContext HiveContext 和 SQLContext 有什么区别？

towardsdatascience.com](/sparksession-vs-sparkcontext-vs-sqlcontext-vs-hivecontext-741d50c9486a) [](/apache-spark-3-0-the-five-most-exciting-new-features-99c771a1f512) [## Apache Spark 3.0:5 个最激动人心的新特性

### Apache Spark 3.0 新版本中最激动人心的 5 个特性

towardsdatascience.com](/apache-spark-3-0-the-five-most-exciting-new-features-99c771a1f512)