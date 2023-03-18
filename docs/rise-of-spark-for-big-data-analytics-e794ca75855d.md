# 大数据科学火花的升起

> 原文：<https://towardsdatascience.com/rise-of-spark-for-big-data-analytics-e794ca75855d>

## Apache Spark 已经成为处理大数据的首选解决方案。让我们来看看 Spark 受欢迎背后的三个原因。

随着可用于处理和分析的数据量增加，我们看到了向分布式系统的缓慢但明确的转变(查看我关于分布式系统崛起的文章，特别是 Hadoop [这里](https://medium.com/codex/journey-to-hadoop-e2dbc30acc))。然而，在 21 世纪初，数据科学和“大数据”的机器学习仍然具有挑战性。当时的尖端解决方案(如 Hadoop)依赖于 Map Reduce，这在一些关键方面存在不足

*   在数据科学过程中，大部分时间花在探索性数据分析、特征工程和选择上。这需要对数据进行复杂的多步转换，很难仅使用 Hadoop 中的 Map 和 Reduce 函数来表示，这将花费大量开发时间并产生复杂的代码库。因此，需要一种支持复杂数据转换的解决方案。

![](img/c22a4f63e848b88811329b39fceeeb7c.png)

使用 MapReduce 分析数据可以…..令人沮丧(照片由 [Siora 摄影](https://unsplash.com/@siora18?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/frustrated?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄)

*   数据科学是一个迭代过程，Hadoop 中典型的 Map-Reduce 操作需要每次从磁盘读取数据，这使得每次迭代都非常耗时且成本高昂。因此，需要一种解决方案来减少数据科学过程每次迭代的时间。
*   模型需要生产、部署和维护。因此，我们需要一个框架，不仅允许我们分析数据，还允许我们在生产中开发和部署模型。Hadoop 不支持前面提到的迭代分析，R/Python 等框架也不能很好地扩展到大型数据集。因此，需要一种解决方案来支持大数据的迭代分析，并生产生成的 ML 模型。

进入阿帕奇火花。

它在 2014 年首次发布时就考虑到了上述需求。Spark 保留了 Hadoop 的可伸缩性和容错性(查看我的 [Hadoop](https://medium.com/codex/journey-to-hadoop-e2dbc30acc) 文章了解更多细节),并在此基础上构建了以下特性

*   包括[广泛的操作列表](https://spark.apache.org/docs/latest/rdd-programming-guide.html#transformations)(除了 map 和 reduce 之外),允许仅用几行代码构建复杂的数据处理/分析系统。此外，为 Spark 开发的 MLlib 库有助于像使用 Scikit learn 一样直观地构建 ML 模型。这减少了开发人员/科学家的时间，并使代码库更易于维护。
*   Spark 利用一个[有向无环图](https://data-flair.training/blogs/dag-in-apache-spark/#:~:text=What%20is%20DAG%20in%20Apache%20Spark?,to%20later%20in%20the%20sequence.)或‘DAG’(把它想象成一个流程图)来跟踪你想要对你的数据执行的操作。因此，与 Hadoop 不同的是，在 Hadoop 中，您会将一系列 Map Reduce 作业串在一起，每个作业都需要从磁盘读取和写入，而 Spark DAG 可以帮助您将操作串在一起，而不必写出中间结果。这意味着多步数据处理/分析作业将运行得更快。Spark 还能够在内存中缓存中间结果。这在机器学习中特别有用，在机器学习中，您可以执行预处理并缓存结果训练数据，以便在优化期间可以从内存中重复访问它(因为梯度下降等优化算法将多次迭代训练数据)。在 Hadoop Map-Reduce 中，必须从磁盘访问列车数据，这使得该过程非常耗时。

![](img/03b4bffeaf3b26790995d7cad0fea0eb.png)

使用 Map Reduce 优化逻辑回归？可能要等很久了……(图片由 [Unsplash](https://unsplash.com/s/photos/waiting?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的[13 拍摄)](https://unsplash.com/@13on?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

*   Spark 不仅可以以迭代和交互的方式分析数据(可以与 jupyter 笔记本集成)，还可以构建生产级数据处理和机器学习管道。

这些特性使 Apache Spark 在过去十年中成为分布式数据科学和机器学习的首选框架，现在几乎可以在所有处理真正大数据的组织中找到。

本系列的下一篇文章将介绍如何使用 Apache Spark，从 Scala 的基础知识开始，Scala 是编写 Spark 程序的理想语言。

参考文献。

Sandy Ryza 等人在《Spark 高级分析:大规模数据学习模式》一书中指出了 Spark 在大数据分析方面的优势。