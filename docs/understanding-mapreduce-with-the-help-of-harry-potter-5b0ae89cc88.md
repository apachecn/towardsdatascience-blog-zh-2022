# 借助《哈利·波特》理解 MapReduce

> 原文：<https://towardsdatascience.com/understanding-mapreduce-with-the-help-of-harry-potter-5b0ae89cc88>

## 通过一个简单的例子从头开始学习 MapReduce

![](img/400805015b449387de434be9cc67d717.png)

伊恩·杜利在 [Unsplash](https://unsplash.com/s/photos/map-reduce?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

MapReduce 是一种允许并行处理大型数据集的算法，即同时在多台计算机上处理。这大大加快了大型数据集的查询速度。

# 我们用 MapReduce 做什么？

MapReduce 最初是由 Google 推出的，用于高效地查询大量的搜索结果。然而，该算法真正出名是因为它在 Hadoop 框架中的使用。它在 Hadoop 分布式文件系统(HDFS)中存储大量数据，并使用 MapReduce 来查询或聚合 TB 或 Pb 范围内的信息。

假设我们已经将《哈利·波特》小说的所有部分以 PDF 格式存储在 [Hadoop](https://databasecamp.de/en/data/hadoop-explained) 中，现在想要统计书中出现的单个单词。这是一个经典的任务，分解成一个映射函数和一个归约函数可以帮助我们。

# 以前是怎么做的？

在有可能将如此复杂的查询拆分到整个计算机集群中并并行计算它们之前，人们被迫一个接一个地运行完整的数据集。当然，数据集越大，查询时间越长。

假设我们已经在 Python 列表中逐字记录了哈利波特文本:

我们可以通过使用 For 循环遍历这个列表，并将每个单词从 Python“Collections”模块加载到“Counter”中，来计算出现的单词数。然后，这个函数为我们进行单词计数，并输出十个最常用的单词。使用 Python 模块“Time”，我们可以显示我们的计算机执行这个函数花了多长时间。

据网站 [wordcounter.io](https://wordcounter.io/blog/how-many-words-are-in-harry-potter/) 统计，哈利波特第一部共有 76944 个单词。由于我们的例句只有 20 个单词(包括句号)，这意味着我们必须重复例句大约 3850 次(76944/20 ~ 3847)才能得到一个和哈利波特第一本书一样多的单词列表:

我们的函数需要 64 毫秒来遍历第一部分的所有单词，并计算它们出现的频率。如果我们对所有总共 3397170 个单词的哈利波特书籍进行同样的查询(来源: [wordcounter.io](https://wordcounter.io/blog/how-many-words-are-in-harry-potter/) )，总共需要 2.4 秒。

这个查询需要相对较长的时间，对于较大的数据集，自然会变得越来越长。加快功能执行的唯一方法是为计算机配备更强大的处理器(CPU)，即改进其硬件。当一个人试图通过改进设备的硬件来加速算法的执行时，这被称为**垂直缩放**。

# MapReduce 算法是如何工作的？

在 MapReduce 的帮助下，通过将任务拆分成更小的子任务，可以显著加快这样的查询。这反过来又有一个优点，即子任务可以在许多不同的计算机之间划分和执行。这意味着我们不必改进单个设备的硬件，而是可以使用许多功能相对较弱的计算机，并且仍然可以减少查询时间。这种方法被称为**水平缩放**。

让我们回到我们的例子:到目前为止，我们已经形象地以这样一种方式进行，我们阅读了所有的哈利波特书籍，并简单地在我们阅读的每个单词后将单个单词的计数表扩展一个计数。这样做的问题是我们不能并行化这种方法。假设有第二个人想要帮助我们，她不能这样做，因为她需要我们正在处理的计数表来继续。只要她没有，就支持不了。

然而，她可以支持我们，从哈利波特系列的第二部分开始，为第二本书创建一个单独的清单。最后，我们可以合并所有单独的计数表，例如，将单词“Harry”在所有计数表上的出现频率相加。

![](img/8d63e533358e61b760c1d322fcae8f4e.png)

作者照片

这也使得让一个人在每本哈利波特书上工作相对容易横向扩展任务。如果我们想工作得更快，我们也可以让更多的人参与进来，让每个人都做一章。最后，我们只需要把每个人的所有结果结合起来，就可以得出一个整体的结果。

# Python 中的 MapReduce 示例

总之，我们需要两个函数一个映射器和一个缩减器来用 Python 编码这种方法。我们以这样一种方式定义映射器，它为接收到的每个单词返回一个字典，以单词为关键字，值为 1:

与我们的示例类似，映射器返回一个计数列表，上面写着:“传递给我的单词恰好出现一次”。在第二步中，reducer 将所有单独的计数表合并成一个大的总计数表。它区分了两种情况:如果传递给它的单词已经出现在它的大计数列表中，那么它只是在相应的行中添加一个破折号。如果新单词还没有出现在列表中，reducer 只需在大计数列表中添加一个新行。

如果我们合并这两个子任务，与之前相同的查询只需要 1.4 秒:

因此，使用 MapReduce 算法，我们能够在不进行任何水平或垂直缩放的情况下，将所有哈利波特书籍的查询时间减少一半以上。但是，如果 1.4 秒的查询时间对于我们的应用程序来说仍然太长，我们可以简单地任意拆分单词列表，并在不同的计算机上并行运行映射器，以进一步加快这个过程。没有 MapReduce 算法，这是不可能的。

# MapReduce 的缺点

我们的例子令人印象深刻地表明，我们可以使用 MapReduce 更快地查询大量数据，同时为水平缩放准备算法。然而，MapReduce 并不总是可以使用，或者根据使用情况的不同也会带来一些缺点:

*   某些查询无法引入 MapReduce 架构。
*   地图功能彼此独立运行。因此，这些进程不可能相互通信。
*   分布式系统比单台计算机更难管理和控制。因此，应该仔细考虑是否真的需要计算集群。Kubernetes 软件工具可用于控制计算机集群。

# 这是你应该带走的东西

*   MapReduce 是一种允许并行快速处理大型数据集的算法。
*   MapReduce 算法将一个大的查询拆分成几个小的子任务，然后可以在不同的计算机上分发和处理这些子任务。
*   不是每个应用程序都可以转换成 MapReduce 方案，所以有时甚至不可能使用这种算法。

*如果你喜欢我的作品，请在这里订阅*[](https://medium.com/subscribe/@niklas_lang)**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，medium 允许你每月免费阅读* ***3 篇*** *。如果你想让***无限制地访问我的文章和数以千计的精彩文章，请不要犹豫，通过点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格***

**[](https://medium.com/@niklas_lang/what-are-deepfakes-and-how-do-you-recognize-them-f9ab1a143456) [## 什么是 deepfakes，怎么识别？

### Deepfakes 是使用深度学习模型人工创建的视频、图像或音频文件。比如说…

medium.com](https://medium.com/@niklas_lang/what-are-deepfakes-and-how-do-you-recognize-them-f9ab1a143456) [](https://medium.com/@niklas_lang/what-are-convolutional-neural-networks-cnn-faf948b5a98a) [## 理解卷积神经网络

### 卷积神经网络(CNN 或 ConvNet)是神经网络的一个子类型，主要用于神经网络的分类

medium.com](https://medium.com/@niklas_lang/what-are-convolutional-neural-networks-cnn-faf948b5a98a) [](https://medium.com/@niklas_lang/intuitive-guide-to-artificial-neural-networks-5a2925ea3fa2) [## 人工神经网络直观指南

### 人工神经网络(ANN)是人工智能和人工智能领域最常用的术语

medium.com](https://medium.com/@niklas_lang/intuitive-guide-to-artificial-neural-networks-5a2925ea3fa2)**