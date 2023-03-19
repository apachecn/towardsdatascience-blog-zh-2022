# 熊猫还不够？替代数据争论解决方案的全面指南

> 原文：<https://towardsdatascience.com/pandas-is-not-enough-a-comprehensive-guide-to-alternative-data-wrangling-solutions-a4730ba8d0e4>

![](img/0c61bd9d120209526a9a6382c9cbe7ca.png)

照片由[马修·施瓦茨](https://unsplash.com/@cadop?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/data?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄

## 包括达斯克、摩丁、polars、Vaex、Terality 等 6 人

我认为`pandas`不需要介绍。这是数据科学家使用的一个伟大而通用的工具，并且很可能会继续在日常生活中使用。但是在使用`pandas`时，我们可能会面临一些潜在的挑战。

最大的问题与数据量有关，这在大数据时代肯定会成为一个问题。虽然有许多任务不涉及如此大量的数据，但我们迟早会遇到这种情况。在这种情况下，我们可以试几招。首先，我们可以尝试优化存储在数据帧中的变量的数据类型，以使数据适合内存。或者，我们可以一次只加载整个数据的一部分。

这些解决方案通常会有所帮助，但有时它们仅仅是不够的，我们可能会耗尽内存或操作变得慢得令人无法忍受。在这种情况下，我们可能想要远离`pandas`，使用更好的工具来完成工作。

本文的目标不是提供所有可能方法的性能比较。相反，我想介绍一下`pandas`的可能替代方案，并简要介绍它们的潜在用例，以及它们的优缺点。然后，您可以选择哪个解决方案符合您的需求，并且可以更深入地了解实现的本质细节。

# 回顾线程与流程

在整篇文章中，我们将提到使用线程或进程运行并行操作。我认为这需要快速复习一下:

*   *进程*不共享内存，在单个内核上运行。它们更适合于不需要相互通信的计算密集型任务。
*   *线程*共享内存。在 Python 中，由于全局解释器锁(GIL)，两个线程不能在同一程序中同时运行。因此，只有一些操作可以使用线程并行运行。

有关线程与进程的更多信息，请参考本文。

# 熊猫替代品列表

在本节中，我们将介绍最受欢迎的`pandas`替代品(截至 2022 年初)。列表的顺序不是从最好到最差的排序，也不是事实上的任何排序。我只是尝试提出这些方法，并在这样做的同时，当各种解决方案之间存在逻辑桥梁时，介绍一些结构。开始吧！

## Dask — ~10k GitHub stars

Dask 是一个用于分布式计算的开源库。换句话说，它有助于在一台机器或许多独立的计算机(集群)上同时运行许多计算。对于前者，Dask 允许我们使用线程或进程并行运行计算。

![](img/0ff24e8ad059260ce5e086d787275a79.png)

[来源](https://dask.org/)

Dask 依靠的是被称为懒惰评估的原理。这意味着直到我们明确地要求(使用`compute`函数)操作才会被执行。通过延迟操作，Dask 创建了一个转换/计算队列，以便它们可以在以后并行执行。

在幕后，Dask 将一个大型数据处理任务分解成许多较小的任务，然后由`numpy`或`pandas`处理。之后，该库将结果重新组合成一个连贯的整体。

关于 Dask 的一些要点:

*   将数据处理工作负载从单台机器扩展到分布式集群(可能扩展到具有 1000 个内核的集群)的良好选择。我们可以很容易地使用完全相同的代码在本地机器上用整个数据集的样本来测试运行一些任务。然后，我们可以在完整数据上重复使用完全相同的代码，并在群集上运行它。
*   数据不必放入内存，而是需要放在磁盘上。
*   它建立在现有的众所周知的对象之上，如`numpy`数组和`pandas`数据帧——没有必要放弃当前的方法并从头重写
*   API 与`pandas`非常相似，除了它有懒惰的行为。
*   Dask 为更多的定制工作负载和与其他项目的集成提供了任务调度接口。此外，它还提供了大量交互式图表和任务分布的可视化，以便进行深入的分析和诊断。
*   Dask 不仅仅是数据处理。Dask-ML(一个独立的库)使用 [Dask](https://dask.org/) 以及流行的机器学习库，如`scikit-learn`、`xgboost`、`lightgbm`等，提供了 Python 中可扩展的机器学习。它有助于缩放数据大小和模型大小。一个例子是，许多`scikit-learn`算法是使用`joblib`为并行执行而编写的(它支持众所周知的`n_jobs`参数)。Dask 通过提供另一个`joblib`后端将这些算法扩展到一个机器集群。

**有用参考:**

*   [https://github.com/dask/dask](https://github.com/dask/dask)
*   【https://dask.org/】

## 摩丁——约 7k GitHub 星

Modin 是一个库，旨在通过在系统的所有可用 CPU 内核之间自动分配计算来并行化`pandas`数据帧。由于这一点，摩丁声称能够获得接近线性的速度提升到你的系统上的 CPU 核心的数量。

那么这是怎么发生的呢？Modin 只是将现有的数据帧分成不同的部分，这样每个部分都可以发送到不同的 CPU 内核。更准确地说，Modin 将数据帧划分为行和列，这使得它的并行处理对于任何大小和形状的数据帧都是高度可扩展的。

该库的作者专注于将数据科学家的时间优先于硬件时间。这就是为什么摩丁:

*   具有与`pandas`相同的 API，因此不会增加数据科学家的学习成本。与大多数其他库不同，它的目标是全面覆盖`pandas` API。在撰写本文时，它提供了`pd.DataFrame`90%的>功能和`pd.Series`88%的>功能。如果某个功能/方法没有实现，Modin 默认为`pandas`，所以最终所有命令都被执行。
*   运行非常简单，可作为`pandas`的替代产品。其实我们只需要把一个导入行改成`import modin.pandas as pd`。
*   提供与 Python 生态系统的流畅集成。
*   不仅可以在本地机器上运行，还可以在 Ray/Dask 集群上运行。我们之前已经介绍过 Dask，所以提到 Ray 是有意义的。Ray 是一个高性能的分布式执行框架。完全相同的代码可以运行在一台机器上(高效的多处理)和一个专用集群上进行大规模计算。
*   支持核外模式，在这种模式下，Modin 使用磁盘作为内存的溢出存储。这样，我们可以处理比内存大得多的数据集。

那么摩丁和达斯克有什么不同呢？有一些差异值得一提:

*   与 Dask 相反，Modin 提供了完全的兼容性。为了可伸缩性，Dask 数据帧提供基于行的存储。这就是为什么他们不能完全支持所有的`pandas`功能。相比之下，Modin 被设计成一个灵活的列存储。
*   Dask 数据帧需要使用`compute`方法显式计算(因为它们处于惰性模式)。在 Modin 中，对用户查询的所有优化都是在幕后执行的，不需要用户的任何输入。
*   必须明确说明 Dask 中的分区数量。
*   Modin 可以在 Dask 上运行，但它最初是为了与 Ray 一起工作而构建的。

有关这些差异的更多信息，请参见下面的链接。

**有用的参考资料:**

*   [https://github.com/modin-project/modin](https://github.com/modin-project/modin)
*   使用 Modin 透明地扩展交互式数据科学:[https://www2 . eecs . Berkeley . edu/Pubs/techr pts/2018/EECS-2018-191 . pdf](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2018/EECS-2018-191.pdf)
*   [https://rise . cs . Berkeley . edu/blog/pandas-on-ray-early-lessons/](https://rise.cs.berkeley.edu/blog/pandas-on-ray-early-lessons/)
*   描述 Dask 和 Modin 区别的帖子:[https://github . com/Modin-project/Modin/issues/515 # issue comment-477722019](https://github.com/modin-project/modin/issues/515#issuecomment-477722019)

## 更快—约 2k 颗 GitHub 星

`swifter`是一个开源库，它试图以最快的方式将任何函数有效地应用到`pandas`数据帧或系列中。`apply`是一个非常有用的函数，因为它允许我们轻松地将任何函数应用于`pandas`对象。然而，这是有代价的——该函数充当 for 循环，导致速度很慢。

除了首先对函数进行矢量化之外，本文中已经提到了相当多的并行替代方法。那么`swifter`在这一切中扮演什么角色呢？

我们已经提到，它试图以最快的方式应用该功能。首先，如果可能的话，`swifter`对函数进行矢量化。如果这是不可能的，它估计什么更快:使用 Dask/Modin 或简单的`pandas`应用并行处理。

`swifter`的主要特点:

*   低学习曲线——这是在`apply`方法链中添加`swifter`的问题。例如:

```
df["col_out"] = df["col_in"].swifter.apply(some_function)
```

*   截至目前，`swifter`支持以下加速方式:`apply`、`applymap`、`rolling().apply()`和`resample().apply()`
*   它受益于 Dask/Modin 等库的潜力
*   我们不应该盲目地把任何功能扔在`swifter`上，抱最好的希望。这就是为什么当我们写 UDF 时，我们应该考虑到函数的矢量化。一个例子是使用`np.where`代替 if-else 条件流。

**有用参考:**

*   [https://github.com/jmcarpenter2/swifter](https://github.com/jmcarpenter2/swifter)
*   [https://medium . com/@ jmcarpenter 2/swifter-1-0-0-自动-高效-pandas-and-modin-data frame-applies-cfbd 9555 e7c 8](https://medium.com/@jmcarpenter2/swifter-1-0-0-automatically-efficient-pandas-and-modin-dataframe-applies-cfbd9555e7c8)
*   [https://github . com/jmcarpenter 2/swifter/blob/master/examples/swifter _ apply _ examples . ipynb](https://github.com/jmcarpenter2/swifter/blob/master/examples/swifter_apply_examples.ipynb)

## vaex — 7k GitHub stars

Vaex 是另一个开源的数据帧库，专门研究懒惰的核外数据帧。

该库最大的亮点可能是 Vaex 只需要很少的 RAM 来检查和与任意大小的数据集交互。这是可能的，因为结合了惰性评估和内存映射。后者是一种技术，你告诉操作系统你想要一块内存与磁盘上的内容同步。当一段时间内没有修改或使用某块内存时，它将被丢弃，以便可以重用 RAM。

实际上，当我们用 Vaex 打开一个文件时，实际上并没有读取任何数据。相反，Vaex 只读取文件的元数据:数据在磁盘上的位置、数据的结构(行数/列数、列名和类型)、文件描述等。

这就是为什么受益于 Vaex 的要求之一是以内存可映射文件格式存储数据，例如 Apache Arrow、Apache Parquet 或 HDF5。如果我们满足这个要求，Vaex 将立即打开这样一个文件，不管它有多大，也不管我们有多少 RAM。

Vaex 的主要特点:

*   类似于`pandas`的 API。
*   易于处理非常大的数据集—通过结合内存映射和惰性评估，Vaex 只受我们可用硬盘空间的限制。
*   Dask 等库侧重于让我们将代码从本地机器扩展到集群，而 Vaex 侧重于使在单台机器上处理大型数据集变得更容易。
*   Vaex 不会创建内存副本，因为过滤后的数据帧只是原始数据的浅层副本。这意味着过滤只需要很少的内存。假设我们有一个 50GB 的文件。许多工具需要 50GB 来读取文件，过滤后的数据帧也需要大约 50GB。
*   虚拟列是在我们转换 Vaex 数据帧的现有列时创建的。它们的行为就像普通的一样，主要区别是它们根本不使用内存。这是因为 Vaex 只记住它们的定义，并不实际计算这些值。虚拟列仅在必要时才进行延迟评估。
*   Vaex 非常快，因为虚拟列的计算是完全并行的，并且使用一些流行的列方法(`value_counts`、`groupby`等)的 C++实现。).此外，所有这些都在核外工作，这意味着我们可以在使用所有可用内核的同时，处理比 RAM 中所能容纳的更多的数据。
*   速度还来自于智能优化，它允许我们只通过一次数据就可以为多个选择计算一些统计数据(无需每次创建新的参考数据帧)。更好的是，我们可以将这些与`groupby`聚合结合起来，同时仍然只传递一次数据。
*   Vaex 可以通过 Numba、Pythran 或 CUDA(需要支持 CUDA 的 NVIDIA 显卡)使用即时编译来进一步加速函数的评估。
*   Vaex 坚持只在必要时检查整个数据集的策略，然后尽可能少地检查数据。例如，当显示 Vaex 数据帧或列时，Vaex 仅从磁盘中读取前 5 行和后 5 行。
*   Vaex 还提供了非常快速且节省内存的字符串操作(几乎支持所有的`pandas`操作)。
*   还有一个`vaex.ml`库，它实现了一些常见的数据转换，例如 PCA、分类编码器和数字定标器。它们具有熟悉的 API、并行化和核外执行的优势。该库还提供了几个流行的机器学习库的接口，如`scikit-learn`或`xgboost`。通过使用它，我们在处理数据争论部分(清理、特征工程和预处理)时不会浪费任何内存。这使我们能够最大限度地利用可用内存来训练模型。

这是相当多的信息。我们还将简要介绍 Vaex 和前面提到的方法之间的一些差异。

*   虽然 Dask 与`pandas`并不完全兼容，但摩丁的目标是兼容，因此，这些库带有一些`pandas`固有的包袱。通过更多地偏离源(但仍然非常相似)，Vaex 在功能方面受到的限制更少(内存映射的查询方式等)。)
*   Dask 和 Modin 扩展到集群，而 Vaex 试图通过内存映射文件和使用本地机器的所有可用核心来帮助用户避免对集群的需求。
*   Vaex 的作者将 Vaex 和 Dask 之间的关系描述为正交。Dask(和 Modin)主要关注数据处理和争论，而 Vaex 也提供了在 N 维网格上快速计算统计数据的能力，并具有一些易于可视化和绘制大型数据集的功能。

关于 Vaex 和 Dask 更深入的对比，请看[这篇文章](/dask-vs-vaex-a-qualitative-comparison-32e700e5f08b)。

**有用参考:**

*   [https://github.com/vaexio/vaex](https://github.com/vaexio/vaex)
*   [https://vaex.io/](https://vaex.io/)
*   [https://towards data science . com/dask-vs-vaex-a-qualitative-comparison-32e 700 e 5 f 08 b](/dask-vs-vaex-a-qualitative-comparison-32e700e5f08b)

## 数据表— 1.5k GitHub stars

`datatable`是一个用于操作二维表格数据的 Python 库。它是由 [H2O.ai](https://www.h2o.ai/) 开发的，它的第一个用户是[无人驾驶. ai](https://www.h2o.ai/driverless-ai/) 。在许多方面，它类似于`pandas`，特别强调速度和单节点机器上的大数据(高达 100GB)支持。

如果您使用过 R，您可能已经熟悉了相关的包`data.table`，这是 R 用户在快速聚合大数据时的首选包。Python 的实现试图模仿其核心算法和 API。

说到 API，其实就是`datatable`和`pandas`(以及 R 的`data.frame`)的“爱它还是恨它”的区别。在`datatable`中，执行所有操作的主要方式是方括号符号，这是受传统矩阵索引的启发。一个例子是:

`DT[i, j, ...]`

其中`i`是行选择器，`j`是列选择器，`...`表示可能添加的附加修饰符。虽然这已经很熟悉了，因为它与在 R/ `pandas` / `numpy`中索引矩阵或对象时遇到的符号完全相同，但还是有一些不同。

其中之一是`i`可以是任何可以被解释为行选择器的东西:一个整数、一个切片、一个范围、一个整数列表、一个切片列表、一个表达式、一个布尔值/整数值框架、一个生成器等等。但这仍然是熟悉的，不应该是一个大问题。

当我们想要执行更高级的操作时，棘手的部分就来了，因为`datatable`的语法与我们大多数人习惯的语法相差甚远。例如:

```
DT[:, sum(f.quantity), by(f.product_id)]
```

该代码片段计算产品数量的总和。而不熟悉的`f`是必须从`datatable`模块导入的特殊变量。它提供了引用给定框架中任何列的快捷方式。

关于`datatable`的要点:

*   `datatable`中的数据帧被称为帧，和`pandas`中的数据帧一样，它们是柱状数据结构。
*   与`pandas`相反，该库为所有数据类型提供了 native-C 实现，包括字符串。`pandas`仅对数值类型有效。
*   它提供了从 CSV 和其他文件格式快速读取数据。
*   使用`datatable`时，我们应该将数据以与内存相同的格式存储在磁盘上。得益于此，我们可以使用磁盘上数据的内存映射，并处理内存不足的数据集。这样，我们就避免了为每个特定的操作加载过多的数据到内存中。
*   `datatable`使用多线程数据处理来实现最高效率。
*   该库最大限度地减少了数据复制量。
*   很容易将`datatable`的帧转换成`pandas` / `numpy`的对象。

**有用参考:**

*   [https://github.com/h2oai/datatable](https://github.com/h2oai/datatable)
*   [https://datatable.readthedocs.io/en/latest/](https://datatable.readthedocs.io/en/latest/)

## cuDF —约 4.5k GitHub stars

`[cuDF](https://github.com/rapidsai/cudf)`是一个 GPU 数据框架库，是 NVIDIA 的 RAPIDS 的一部分，这是一个跨多个开源库并利用 GPU 功能的数据科学生态系统。cuDF 提供了一个类似于 pandas 的 API，允许我们从性能提升中获益，而无需深入 CUDA 编程的细节。

关于`cuDF`的要点:

*   `pandas`-像 API 一样——在很多情况下，我们只需要修改一行代码就可以开始受益于 GPU 的强大功能。
*   使用 Apache Arrow 列内存格式构建。
*   `cuDF`是单 GPU 库。然而，它可以利用多 GPU 设置结合 Dask 和专用的`dask-cudf`库。有了它，我们能够在一台机器上的多个 GPU 之间扩展`cuDF`，或者在一个集群中的许多机器之间扩展多个 GPU。
*   使用`cuDF`需要一个兼容的 NVIDIA GPU 和一些额外的设置(更新驱动程序，安装 CUDA 等)。)
*   我们应该记住，只要数据合理地适合 GPU 内存，就可以获得最佳性能。

**有用参考:**

*   [https://github.com/rapidsai/cudf](https://github.com/rapidsai/cudf)
*   【https://docs.rapids.ai/api/cudf/stable/】
*   [https://docs . rapids . ai/API/cudf/stable/user _ guide/10min-cudf-cupy . html](https://docs.rapids.ai/api/cudf/stable/user_guide/10min-cudf-cupy.html)
*   [https://docs . rapids . ai/API/cudf/stable/user _ guide/10min . html](https://docs.rapids.ai/api/cudf/stable/user_guide/10min.html)

## pyspark

与之前的库相比，我们实际上首先需要后退一步，描述 Spark 是什么。

Apache Spark 是一个用于大规模数据处理的统一分析引擎，用 Scala 编写。它基本上是为数据科学处理大型数据集(比如 100GB 以上)的*和*库。其受欢迎有多种原因，包括以下原因:

*   它比 Hadoop 快 100 倍，
*   它实现了静态、批处理和流式数据的高性能，
*   它使用最先进的 DAG ( [有向无环图](https://en.wikipedia.org/wiki/Directed_acyclic_graph))调度器、查询优化器和物理执行引擎。

Spark 采用主从架构工作，其中主设备实际上被称为“驱动器”，从设备被称为“工人”。运行 Spark 应用程序时，Spark driver 会创建一个上下文，作为应用程序的入口点。然后，所有操作都在工作节点上执行，而资源由集群管理器管理。

Spark 自带数据帧风格。虽然它们具有类似于`pandas`数据帧的功能，但主要区别在于它们是分布式的，它们具有惰性评估并且是不可变的(不允许覆盖数据)。

对 Spark 的介绍已经足够了，让我们把重点放在与本文最相关的部分，即缩放数据帧。为此，我们可以使用 PySpark，这是一个用于 Spark 的 Python API。

关于 PySpark 需要了解的关键事项:

*   PySpark 是一个通用的、内存中的分布式处理引擎，用于以分布式方式进行高效的数据处理。
*   速度大幅提升——在 PySpark 上运行计算比使用传统系统快 100 倍。
*   它有一个不同于`pandas`的 API，并且它不能很好地与其他库集成(例如用于绘图的`matplotlib`等)。).一般来说，它的学习曲线比`pandas`更陡。
*   当使用*宽转换*(查看所有节点的全部数据，例如，排序或使用`groupby`)时，我们应该小心，因为它们比*窄转换*(查看每个节点中的单个数据)计算量更大。
*   要使用 PySpark，我们需要克服一些开销——设置一个 Spark(本地或集群),在我们的计算机上有一个 JVM (Java 虚拟机),等等。当我们的组织中还没有运行时，这可能是一个阻碍，并且对于一些较小的实验来说，设置它将是一个大材小用。或者，我们可以使用托管云解决方案，如 [Databricks](https://databricks.com/) 。
*   使用 PySpark，我们可以轻松处理来自 Hadoop HDFS、AWS S3 和许多其他文件系统的数据。这也包括使用流媒体和 Kafka 处理实时数据。
*   是 PySpark 的包装器，基本上是 Spark 的机器学习库。由`MLlib`库提供的 API 非常容易使用，并且支持许多分类、回归、聚类、维度减少等算法。
*   Spark 允许我们用 SQL 和Python 查询数据帧。这很方便，因为有时用 SQL 编写一些逻辑比记住确切的 PySpark API 更容易。因为工作可以互换，你可以使用任何你喜欢的。
*   由于 Spark 运行在几乎无限的计算机集群上，它可以处理的数据集大小实际上没有限制。

**有用参考:**

*   [https://www.youtube.com/watch?v=XrpSRCwISdk](https://www.youtube.com/watch?v=XrpSRCwISdk)
*   [https://towards data science . com/the-most-complete-guide-to-py spark-data frames-2702 c 343 B2 E8](/the-most-complete-guide-to-pyspark-dataframes-2702c343b2e8)

## 考拉——约 3k GitHub 星

我们提到过 PySpark 最大的痛点是语法，它不同于`pandas`，并且有一个相当陡峭的学习曲线。这正是 Databricks 想用考拉解决的问题。该库的目标是通过在 Spark 上实现`pandas` API，让数据科学家在与大数据交互时更有效率。

关于考拉要知道的一些事情:

*   你可以拥有一个既能与`pandas`(较小的数据集)又能与 Spark(分布式数据集)一起工作的单一代码库。你只需要把导入语句从`pandas`替换到`koalas`。
*   考拉支持 Spark ≤ 3.1，在 Spark 3.2 中正式被 PySpark 收录为`pyspark.pandas`
*   空值的处理方式可能稍有不同。`pandas`使用 NaNs(特殊常量)来表示缺失值，而 Spark 在每个值上都有一个特殊的标志来表示是否有值缺失。
*   惰性求值——由于 Spark 本质上是惰性的，一些操作(例如创建新列)只有在 Spark 需要打印或写入数据帧时才会执行。
*   很容易将考拉数据帧转换成`pandas` /PySpark 数据帧。
*   考拉还支持带有`ks.sql()`的标准 SQL 语法，允许执行 Spark SQL 查询，并以数据帧的形式返回结果。
*   考拉和 PySpark 的性能结果非常相似，因为它们都在幕后使用 Spark。但是，与纯 PySpark 相比，性能会略有下降。大多数情况下，它与构建默认索引的开销有关，或者与一些`pandas`和 PySpark APIs 共享相同的名称，但具有不同的语义(例如，`count`方法)的事实有关。
*   树袋熊数据帧与 PySpark 数据帧在外观上略有不同。为了实现需要隐式排序的`pandas`数据帧结构及其丰富的 API，考拉数据帧具有表示类似`pandas`的索引的内部元数据和映射到 PySpark 数据帧列的列标签。另一方面，PySpark 对应物往往更符合关系数据库中的关系/表。因此，它们没有唯一的行标识符。
*   在内部，考拉数据帧建立在 PySpark 数据帧之上。考拉将`pandas`API 翻译成 Spark SQL 的逻辑计划。然后 Spark SQL 引擎优化并执行该计划。

**有用的参考资料:**

*   [https://github.com/databricks/koalas](https://github.com/databricks/koalas)
*   [https://koalas.readthedocs.io/en/latest/](https://koalas.readthedocs.io/en/latest/)
*   [https://koalas . readthe docs . io/en/latest/getting _ started/10min . html](https://koalas.readthedocs.io/en/latest/getting_started/10min.html)

## 极地星——约 5k GitHub 星

`polars`是一个强调速度的开源数据框架库。为了实现这一点，它在 Rust 中以 Apache Arrow 作为内存模型来实现。直到最近，作为`polars`的 Python 包装器的库被称为`pypolars`，然而，为了简单起见，现在它也被称为`polars`。

`polars`的一些关键特性:

*   API 类似于`pandas`，然而，它实际上更接近 R 的`dplyr`。
*   有两个 API——渴望和懒惰。前者与`pandas`非常相似，因为结果是在执行完成后立即产生的。另一方面，lazy API 更类似于 Spark，在 Spark 中，计划是在执行查询时形成的。但是当我们调用`collect`方法时，该计划直到在 CPU 的所有核心上并行执行时才真正看到数据。
*   绘图很容易生成并与最流行的可视化工具集成。
*   `polars`是目前最快的(如果不是最快的)数据帧库之一(根据这个[基准](https://h2oai.github.io/db-benchmark/))，并且支持对于`pandas`来说可能太大的数据帧。
*   `polars`的速度来自于利用你机器所有可用的内核。它与其他解决方案的区别在于，`polars`是从底层开始编写的，并考虑到了数据帧查询的并行化，而 Dask 等工具则并行化了现有的单线程库(如`numpy`和`pandas`)。

**有用参考:**

*   [https://github.com/pola-rs/polars](https://github.com/pola-rs/polars)
*   [https://pola-RS . github . io/polars-book/user-guide/introduction . html](https://pola-rs.github.io/polars-book/user-guide/introduction.html)
*   [https://www . ritchievink . com/blog/2021/02/28/I-written-one-of-the-fast-data frame-libraries/](https://www.ritchievink.com/blog/2021/02/28/i-wrote-one-of-the-fastest-dataframe-libraries/)

## 潘达平行星——约 2k 颗 GitHub 星

`pandarallel`(承认吧，这个听起来有点像神奇宝贝)是一个开源库，可以在所有可用的 CPU 上并行化`pandas`操作。

当我们使用`pandarallel`调用并行化函数时，下面的步骤会在幕后发生:

*   该库初始化 PyArrow 等离子体共享存储器，
*   它为每个 CPU 创建一个子进程，然后要求它们处理原始数据帧的一部分，
*   它合并了父流程中的所有结果。

关于`pandarallel`的一些要点:

*   该库允许您并行化下面的`pandas`方法:`apply`、`applymap`、`groupby`、`map`和`rolling`。
*   如果您的 CPU 使用超线程(例如，8 个核心和 16 个线程)，则只会使用 8 个核心。
*   `pandarallel`需要两倍于标准`pandas`操作使用的内存。不言而喻，如果数据最初不适合使用`pandas`的内存，则不应使用该库。

**有用参考:**

[https://github.com/nalepae/pandarallel](https://github.com/nalepae/pandarallel)

## Terality

在谈到`pandas`替代品时，Terality 是一个新手。它是一个无服务器的数据处理引擎，使`pandas`像 Apache Spark 一样可扩展和快速(比`pandas`快 100 倍，能够处理 100 多 GB 的数据)，既没有基础设施要求，也不涉及任何代码更改。听起来已经很棒了！有什么条件？

与其他库/方法的最大区别是 Terality 不是开源软件。有不同类型的订阅(包括一个免费的游戏！)但总的来说，你是按处理的数据量收费的。有关定价的更多信息，请参见[本页](https://www.terality.com/pricing)。

关于 Terality 需要知道的一些事情:

*   `pandas` API 的 100%覆盖率。
*   Terality 在现有的`pandas`功能的基础上提供了两个新方法:`to_csv_folder`和`to_parquet_folder`。它们允许我们轻松地将原始数据集分割成多个更小的数据集。当将数据分割成块，然后分别分析每个块时，这个特性特别有用。
*   由于该项目不是开源的，我们不能对其快速性能背后的底层架构说太多。我们所知道的是，Terality 团队开发了一个专有的数据处理引擎，因此它不是 Spark 或 Dask 的分支/风格。
*   由于是托管的，所以不需要管理基础设施，内存实际上是无限的。
*   它消除了`pandas`的可扩展性问题，Spark 的复杂性(设置+不同的语法)，以及 Dask/Modin 的局限性。
*   截至 2022 年 2 月，您可以将 Terality 与 Google Colab 结合使用。
*   Terality 是可自动扩展的——无论数据大小如何，我们的操作都会以极高的速度自动处理。不需要手动调整处理能力来匹配数据集的大小。所有的基础设施都是在 Terality 这边管理的，包括在你完成你的处理后关闭东西。
*   Terality 与针对您的数据的云存储解决方案(亚马逊 S3、Azure Data Lake 等)结合使用时效果最佳。).这是因为另一种选择是加载一个本地文件，这可能需要相当长的时间(取决于你的网速)。
*   自然，在使用具有潜在敏感数据的第三方解决方案时，会有约束/限制，尤其是在涉及公司数据时。就安全性而言，Terality 提供了安全隔离，我们的数据在传输和计算过程中都得到了充分保护。更多安全信息请参考[本网站](https://www.terality.com/security)。
*   不久前，Terality 可以部署在您自己的 AWS 帐户中。由于这种自托管部署，您的数据永远不会离开您的 AWS 帐户。这种功能可以帮助您遵守数据保护要求，并消除对数据安全性的任何疑虑。

**有用参考:**

*   [https://www.terality.com/](https://www.terality.com/)
*   [https://docs.terality.com/](https://docs.terality.com/)
*   [https://www . terality . com/post/terality-beats-spark-and-dask-H2O-benchmark](https://www.terality.com/post/terality-beats-spark-and-dask-h2o-benchmark)

# 结论

不要误解我，`pandas`是一个很棒的工具，我每天都在使用并将继续使用。然而，对于某些特定的用例来说，这可能是不够的。这就是为什么在这篇文章中，我提供了最流行的`pandas`选择的概述。

让我先回答一个你可能会想到的问题:哪个解决方案是最好的？你可能已经猜到了，答案是:视情况而定。如果你想简单地在本地机器上加速你的`pandas`代码，也许 Modin 是一个很好的起点。如果已经有 Spark 集群在运行，可以试试 py Spark/考拉。对于计算一些统计数据或可视化大规模数据集，Vaex 可能是一个很好的起点。或者，如果您想使用最大速度，而不必担心设置任何基础设施，那么 Terality 可能是一个不错的选择。

最后但同样重要的是，如果没有某种基准测试的参考，这样的文章是不完整的。说到纯速度，H2O.ai 为大部分用于数据处理的 Python 库准备了这样的[一个基准测试](https://h2oai.github.io/db-benchmark/)。为了评估这些库，他们对 2 个数据集进行了数据聚合和连接。对于这些任务，他们使用了不同大小的数据集:0.2 GB、5 GB 和 50 GB。

你对文章中提到的图书馆有经验吗？还是我错过了一个你知道的图书馆？我很想听听你的经历！你可以在[推特](https://twitter.com/erykml1?source=post_page---------------------------)或评论中联系我。

喜欢这篇文章吗？成为一个媒介成员，通过无限制的阅读继续学习。如果你使用[这个链接](https://eryk-lewinson.medium.com/membership)成为会员，你将支持我，而不需要额外的费用。提前感谢，再见！

您可能还会对以下内容感兴趣:

</make-working-with-large-dataframes-easier-at-least-for-your-memory-6f52b5f4b5c4>  </8-more-useful-pandas-functionalities-for-your-analyses-ef87dcfe5d74>  </9-useful-pandas-methods-you-probably-have-not-heard-about-28ff6c0bceee> 