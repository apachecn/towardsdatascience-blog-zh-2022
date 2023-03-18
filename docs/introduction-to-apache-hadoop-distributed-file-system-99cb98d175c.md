# Apache Hadoop 分布式文件系统简介

> 原文：<https://towardsdatascience.com/introduction-to-apache-hadoop-distributed-file-system-99cb98d175c>

## 关于这个 Hadoop 组件，您需要了解的一切

![](img/877d5b4c58bd1e159534dec0d917ec45.png)

马丁·比约克在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

Apache HDFS 是一个分布式文件系统，用于存储大数据领域的大量数据，并将其分布在不同的计算机上。该系统使得 [Apache Hadoop](https://databasecamp.de/en/data/hadoop-explained) 能够以分布式方式跨大量节点(即计算机)运行。

# 什么是 Apache 框架 Hadoop？

Apache Hadoop 是一个软件框架，可以用来快速处理分布式系统上的大量数据。它具有确保稳定和容错功能的机制，因此该工具非常适合在[大数据](https://databasecamp.de/en/data/big-data-basics)环境中进行数据处理。软件框架本身由四个部分组成。

**Hadoop Common** 是各种模块和库的集合，支持其他组件并使它们能够协同工作。其中，Java 归档文件(JAR 文件)存储在这里，这是启动 Hadoop 所需要的。此外，该集合允许提供基本服务，如文件系统。

[**Map-Reduce 算法**](https://databasecamp.de/en/data/mapreduce-algorithm) 源自谷歌，有助于将复杂的计算任务划分为更易于管理的子流程，然后将这些子流程分布到多个系统中，即横向扩展。这大大减少了计算时间。最后，子任务的结果必须再次组合成整体结果。

**另一个资源协商器(YARN)** 通过跟踪计算机集群中的资源并将子任务分配给各个计算机来支持 Map-Reduce 算法。此外，它还为各个进程分配容量。

Apache **Hadoop 分布式文件系统(HDFS)** 是一个用于存储中间或最终结果的可伸缩文件系统，我们将在本文中详细讨论。

# 我们需要 HDFS 做什么？

在集群中，HDFS 分布在多台计算机上，以快速高效地处理大量数据。这背后的想法是，大数据项目和数据分析基于大量数据。因此，应该有一个系统能够批量存储数据并快速处理数据。HDFS 确保存储数据记录的副本，以便能够应对计算机故障。

根据自己的[文档](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html) , [Hadoop 使用 HDFS 的](https://databasecamp.de/en/data/hadoop-explained)目标如下:

*   从硬件故障中快速恢复
*   启用流数据处理
*   巨大数据集的处理
*   易于迁移到新的硬件或软件

# Hadoop 分布式文件系统的结构

Hadoop 分布式文件系统的核心是将数据分布在不同的文件和计算机上，以便快速处理查询，并且用户不会等待很长时间。为了确保群集中单台机器的故障不会导致数据丢失，在不同的计算机上进行有针对性的复制以确保恢复能力。

Hadoop 一般按照所谓的主从原则工作。在计算机集群中，我们有一个承担所谓主节点角色的节点。在我们的示例中，这个节点不执行任何直接计算，而只是将任务分配给所谓的从节点，并协调整个过程。从节点依次读取书籍并存储词频和词分布。

这个原理也用于数据存储。主节点将数据集中的信息分发到不同的从节点，并记住它在哪些计算机上存储了哪些分区。它还冗余地存储数据，以便能够弥补故障。当用户查询数据时，主节点然后决定它必须查询哪些从节点，以便获得想要的信息。

Apache Hadoop 分布式文件系统中的主服务器称为 Namenode。从节点就是所谓的 datanodes。从下到上，示意结构可以理解如下:

客户端将数据写入不同的文件，这些文件可以位于不同的系统上，在我们的示例中是机架 1 和机架 2 上的 datanodes。集群中的每台计算机通常有一个 datanode。它们主要管理计算机上可供它们使用的内存。几个文件通常存储在内存中，这些文件又被分成所谓的块。

名称节点的任务是记住哪些块存储在哪个 datanode 中。此外，他们管理文件，并可以根据需要打开、关闭和重命名文件。

datanodes 又负责客户端(即用户)的读写过程。在发生查询时，客户端也从它们那里接收所需的信息。同时，datanodes 还负责数据的复制，以保证系统的容错性。

# Hadoop 分布式文件系统有什么优势？

对于许多公司来说， [Hadoop 框架](https://databasecamp.de/en/data/hadoop-explained)作为[数据湖](https://databasecamp.de/en/data/data-lakes)也变得越来越有趣，即由于 HDFS，作为大量数据的非结构化存储。各种因素在这里起着决定性的作用:

*   在分布式集群中存储大量数据的能力。在大多数情况下，这比在一台机器上存储信息要便宜得多。
*   高容错能力，因此系统高度可用。
*   Hadoop 是开源的，因此可以免费使用，源代码可以查看

这些要点解释了 Hadoop 和 HDFS 在许多应用程序中的日益普及。

# 这是你应该带走的东西

*   HDFS 是一个分布式文件系统，用于存储大数据领域的大量数据，并将其分布在不同的计算机上。
*   它是 Apache Hadoop 框架的一部分。
*   主节点将数据集分成更小的分区，分布在不同的计算机上，即所谓的从节点。

*如果你喜欢我的作品，请在这里订阅*[](https://medium.com/subscribe/@niklas_lang)**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！此外，媒体允许你每月免费阅读三篇文章***。如果你想让***无限制地访问我的文章和数以千计的精彩文章，请不要犹豫，通过点击我的推荐链接:*[https://medium.com/@niklas_lang/membership](https://medium.com/@subhinajar/membership)每月获得***5 美元*** *的会员资格****

**[](/learn-coding-13-free-sites-to-help-you-do-it-9b2c1b92e573) [## 学习编码:13 个免费网站帮助你开始

### 一旦你决定要学习编码，你会被众多的在线工具宠坏，这些工具可以帮助你…

towardsdatascience.com](/learn-coding-13-free-sites-to-help-you-do-it-9b2c1b92e573) [](/introduction-to-random-forest-algorithm-fed4b8c8e848) [## 随机森林算法简介

### 算法是如何工作的，我们可以用它来做什么

towardsdatascience.com](/introduction-to-random-forest-algorithm-fed4b8c8e848) [](/understanding-mapreduce-with-the-help-of-harry-potter-5b0ae89cc88) [## 借助《哈利·波特》理解 MapReduce

### MapReduce 是一种允许并行处理大型数据集的算法，例如，在多台计算机上…

towardsdatascience.com](/understanding-mapreduce-with-the-help-of-harry-potter-5b0ae89cc88)**