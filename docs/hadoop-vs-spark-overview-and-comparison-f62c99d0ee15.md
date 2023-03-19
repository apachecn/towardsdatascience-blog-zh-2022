# Hadoop 与 Spark:概述和比较

> 原文：<https://towardsdatascience.com/hadoop-vs-spark-overview-and-comparison-f62c99d0ee15>

## Spark 和 Hadoop 的总结与比较

![](img/866b68441d25fbf426d92ff54df6febc.png)

Wolfgang Hasselmann 在 [Unsplash](https://unsplash.com/s/photos/elephant?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

Hadoop 和 Spark 都是开源软件的集合，由 Apache 软件基金会维护，用于大规模数据处理。Hadoop 是两者中较老的一个，曾经是处理大数据的首选。然而，自从 Spark 推出以来，它的增长速度远远超过了 Hadoop，后者不再是该领域无可争议的领导者。

随着 Spark 越来越受欢迎，在 Spark 和 Hadoop 之间做出选择是现实世界中许多公司面临的问题。不幸的是，这个问题的答案并不简单。两种系统都有优点和缺点，正确的选择将取决于所讨论的用例的复杂性。

在本次讨论中，我们将简要介绍 Spark 和 Hadoop，讨论两者之间的主要技术差异，并比较它们的优势和劣势，以确定在哪些情况下应该选择其中之一。

# Hadoop 概述

![](img/4bea491ab403af14f4e43434cc64f899.png)

照片由 [Wolfgang Hasselmann](https://unsplash.com/@wolfgang_hasselmann?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/elephant?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄——由作者编辑

Hadoop 允许其用户利用由许多计算机组成的网络，目的是利用它们的综合计算能力来处理涉及大量数据的问题。Hadoop 框架有两个主要元素，即分布式存储和处理。分布式存储使用 Hadoop 分布式文件系统( **HDFS** )，而处理使用另一个资源协商器( **YARN** )来调度任务和分配资源，从而实现 MapReduce 编程模型。

HDFS 的建立有许多目标。首先，由于一个 HDFS 实例可能包含数千台机器，硬件故障被视为常态而非例外。因此，通过确保快速检测到故障，并且恢复过程平稳且自动，可以对这种故障进行规划。其次，HDFS 的设计考虑的是批处理，而不是用户的交互使用。因此，HDFS 优先考虑高吞吐量，而不是对数据的低延迟访问，从而实现对数据的流式访问。第三，HDFS 确保包含巨大数据集(例如许多 TB)的用例被容纳。最后，HDFS 的另一个优势是易用性，这源于它与许多操作系统的兼容性以及跨硬件平台的可移植性。

Hadoop 最初发布时没有 YARN，仅仅依赖于 MapReduce 框架。YARN 的加入意味着 Hadoops 的潜在用例扩展到了 MapReduce 之外。YARN 的关键添加是将集群资源管理和调度从 MapReduce 的数据处理组件中分离出来。这导致 Hadoop 集群比 MapReduce 更严格的方法更好地分配资源(在内存和 CPU 负载方面)。YARN 在 HDFS 和运行应用的处理引擎(如 Spark)之间提供了更高效的链接，使 Hadoop 能够运行更广泛的应用，如流数据和交互式查询。

Hadoop 的真正基础是 **MapReduce** ，它的关键特征是批处理、对数据传递没有限制、没有时间或内存限制。有许多想法可以实现这些特性并定义 Hadoop MapReduce。首先，设计是这样的，硬件故障是预料之中的，并且将被快速处理，不会丢失或损坏数据。第二，优先考虑横向扩展而不是纵向扩展，这意味着增加更多的商用机器比减少高端机器更可取。因此，Hadoop 中的可伸缩性相对便宜且无缝。此外，Hadoop 按顺序处理数据，避免了随机访问，还提高了数据位置意识。这些属性确保处理速度提高几个数量级，并尽可能避免移动大量数据的昂贵过程。

# Spark 概述

![](img/e0c7985963e1816d6273f224f180e023.png)

由[克里斯蒂安·埃斯科瓦尔](https://unsplash.com/@cristian1?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/sparks?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

Hadoop 的简单 MapReduce 编程模型很有吸引力，并在行业中得到广泛应用，但是，某些任务的性能仍然不是最佳的。这导致了 Spark 的出现，Spark 的引入是为了提供对 Hadoop 的加速。需要注意的是，Spark 不依赖于 Hadoop，但可以利用它。在比较和对比这两种技术之前，我们将简要介绍一下 Spark。

Spark 是一个用于大数据集的数据处理引擎，也是开源的，由 Apache 基金会维护。称为弹性分布式数据集(RDDs)的抽象概念的引入是 Spark 在某些任务上超越 Hadoop 并获得巨大加速的基础。

rdd 是容错的元素集合，通过分布在集群中的多个节点上，可以并行处理这些元素。Spark 速度的关键在于，在 RDD 上执行的任何操作都是在内存中完成的，而不是在磁盘上。Spark 允许对 rdd 进行两种类型的操作，即转换和操作。动作用于应用计算并获得结果，而变换导致新 RDD 的创建。这些操作的分配由 Spark 完成，不需要用户的指导。

通过使用有向无环图(DAG)来管理在 RDD 上执行的操作。在 Spark DAG 中，每个 RDD 被表示为一个节点，而操作形成边。RDDs 的容错特性来自于这样一个事实，即 RDD 的一部分丢失了，那么可以通过使用存储在图中的操作谱系从原始数据集中重新计算它。

# Spark 和 Hadoop 的主要技术差异和选择

![](img/d93e80fd163f6613298d2a2e8d3e4b58.png)

作者图片

如前所述，Spark 为某些任务带来了巨大的加速。这种情况的主要技术原因是由于 Spark 在 RAM(随机存取存储器)中处理数据，而 Hadoop 在磁盘上向 HDFS 读写文件(我们在这里注意到 Spark 可以使用 HDFS 作为数据源，但仍然会在 RAM 中处理数据，而不是像 Hadoop 那样在磁盘上处理数据)。RAM 比磁盘快得多有两个原因。首先，RAM 使用固态技术来存储信息，而磁盘通过磁性来存储。其次，RAM 比存储在磁盘上的信息更接近 CPU，并且具有更快的连接，因此 RAM 中的数据被更快地访问。

这种技术上的差异导致应用程序的速度提高了许多个数量级，在这些应用程序中，同一个数据集被多次重用。Hadoop 导致这些任务的显著延迟(等待时间),因为每次查询都需要单独的 MapReduce 作业，这涉及到每次从磁盘重新加载数据。然而，使用 Spark，数据保留在 RAM 中，因此从那里而不是从磁盘读取。这导致在我们多次重用相同数据的某些情况下，Spark 的速度比 Hadoop 快 100 倍。因此，在这种情况下，我会选择 Spark 而不是 Hadoop。这种情况的常见例子是迭代作业和交互式分析。

重复使用相同数据集的迭代任务的一个具体且非常常见的示例是机器学习(ML)模型的训练。ML 模型通常通过迭代地经过相同的训练数据集来训练，以便通过使用诸如梯度下降的优化算法来尝试并达到误差函数的全局最小值。数据被查询的次数越多，Spark 在此类任务中实现的性能提升水平就越显著。例如，如果您在 Hadoop 和 Spark 上仅使用一次数据传递(epoch)来训练 ML 模型，将不会有明显的加速，因为 Spark 上的第一次迭代需要将数据从磁盘加载到 RAM 中。然而，Spark 上的每个后续迭代将在一小部分时间内运行，而每个后续 Hadoop 迭代将花费与第一次迭代相同的时间，因为每次都从磁盘检索数据。因此，在处理 ML 应用程序时，Spark 通常优于 Hadoop。

尽管在许多应用程序中这是一个巨大的优势，但值得注意的是，在有些情况下，内存中的 Spark 计算并不尽如人意。例如，如果我们处理的数据集非常大，超过了可用的 RAM，那么 Hadoop 就是首选。此外，同样由于内存和磁盘的差异，与 Spark 相比，Hadoop 的扩展相对容易且便宜。因此，尽管 Spark 可能最适合时间有限的企业，但 Hadoop 更便宜的设置和可扩展性可能更适合资金有限的企业。

如果你从这些文章中获得了价值，考虑使用下面的链接注册 medium！👇

[](https://medium.com/@riandolphin/membership) 