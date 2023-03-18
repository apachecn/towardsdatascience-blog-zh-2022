# 如何安装 Spark NLP

> 原文：<https://towardsdatascience.com/how-to-install-spark-nlp-5fcd36fab378>

## 环境设置

## 如何让 Spark NLP 在您的本地计算机上工作的分步教程

![](img/9ed202e24aa163ca32140068702d57a5.png)

照片由[西格蒙德](https://unsplash.com/@sigmund?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

Apache Spark 是一个用于快速和通用数据处理的开源框架。它提供了一个统一的引擎，可以以快速和分布式的方式运行复杂的分析，包括机器学习。

Spark NLP 是一个 Apache Spark 模块，为 Spark 应用程序提供高级自然语言处理(NLP)功能。它可以用于构建复杂的文本处理管道，包括标记化、句子分割、词性标注、解析和命名实体识别。

尽管描述如何安装 Spark NLP 的文档非常清楚，但有时您可能会在安装时遇到困难。出于这个原因，在本文中，我尝试描述一个逐步的过程来使 Spark NLP 在您的计算机上工作。

要安装 Spark NLP，您应该安装以下工具:

*   计算机编程语言
*   Java 语言(一种计算机语言，尤用于创建网站)
*   斯卡拉
*   阿帕奇火花
*   PySpark
*   火花 NLP。

# 1 条蟒蛇

您已经按照技术要求一节中描述的步骤安装了 Python。所以，我们可以从第二步开始安装软件，Java。

# 2 Java

Spark NLP 构建在 Apache Spark 之上，可以安装在任何支持 Java 8 的操作系统上。通过在终端中运行以下命令，检查您是否安装了 Java 8:

```
java –version
```

如果已经安装了 Java，您应该会看到以下输出:

```
openjdk version “1.8.0_322”
OpenJDK Runtime Environment (build 1.8.0_322-bre_2022_02_28_15_01-b00)OpenJDK 64-Bit Server VM (build 25.322-b00, mixed mode)
```

如果没有安装 Java 8，可以从[这个链接](https://java.com/en/download/)下载 Java 8，按照向导进行操作。

在 **Ubuntu** 中，可以通过包管理器安装`openjdk-8`:

```
sudo apt-get install openjdk-8-jre
```

在 Mac OS 中，可以通过`brew`安装`openjdk-8`:

```
brew install openjdk@8
```

如果安装了另一个版本的 Java，可以下载 Java 8，如前所述，然后将`JAVA_HOME`环境变量设置为 Java 8 目录的路径。

# 3 斯卡拉

Apache Spark 需要 scala 2.12 或 2.13 才能正常工作。您可以按照此处 [中](https://www.scala-lang.org/download/2.12.15.html.)[描述的步骤安装 scala 2.12.15。](https://www.scala-lang.org/download/2.12.15.html)

安装完成后，您可以通过运行以下命令来验证 scala 是否正常工作:

```
scala -version
```

# **4 阿帕奇 Spark**

你可以从 Apache Spark 的官方网站下载，这里是。Apache Spark 有很多版本。个人安装了 3.1.2 版本，这里[有](https://archive.apache.org/dist/spark/spark-3.1.2/)。

您下载这个包，然后，您可以提取它，并把它放在文件系统中您想要的任何地方。然后，您需要将 spark 目录中包含的 bin 目录的路径添加到`PATH`环境变量中。在 Unix 中，您可以导出`PATH`变量:

```
export PATH=$PATH:/path/to/spark/bin
```

然后，导出`SPARK_HOME`环境变量以及 spark 目录的路径。在 Unix 中，您可以按如下方式导出`SPARK_HOME`变量:

```
export SPARK_HOME=”/path/to/spark”
```

要检查 Apache Spark 是否安装正确，可以运行以下命令:

```
spark-shell
```

外壳应该打开:

```
Welcome to____ __/ __/__ ___ _____/ /___\ \/ _ \/ _ `/ __/ ‘_//___/ .__/\_,_/_/ /_/\_\ version 3.1.2/_/Using Scala version 2.12.15 (OpenJDK 64-Bit Server VM, Java 1.8.0_322)Type in expressions to have them evaluated.Type :help for more information.scala>
```

要退出 shell，可以使用 Ctrl+C。

# **5 PySpark 和 Spark NLP**

PySpark 和 Spark NLP 是两个 Python 库，可以通过 pip 安装:

```
pip install pyspark
pip install spark-nlp
```

现在 Spark NLP 应该已经在你的电脑上准备好了！

# 摘要

恭喜你！您刚刚在计算机上安装了 Spark NLP！你已经安装了 Java，Scala，Apache Spark，Spark NLP，PySpark！

现在是玩 Spark NLP 的时候了。网上有很多教程。建议你从以下几本笔记本开始:

*   [如何使用 Spark NLP 预处理管道](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/quick_start_google_colab.ipynb#scrollTo=aaVmDt1TEXdh)
*   [深入 Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/1hr_workshop/Deep_Dive_on_SparkNLP_3.0.ipynb) 。

你也可以查看这个教程，它解释了如何将 Spark NLP 与 Comet 集成，Comet 是一个用来监控机器学习实验的平台

如果你读到这里，对我来说，今天已经很多了。谢谢！你可以在[这个链接](https://alod83.medium.com/my-most-trending-articles-4fbfbe107fb)阅读我的趋势文章。

# 相关文章

[](/how-to-speed-up-your-python-code-through-pyspark-e3296e39da6) [## 如何通过 PySpark 加速您的 Python 代码

### 关于如何安装和运行 Apache Spark 和 PySpark 以提高代码性能的教程。

towardsdatascience.com](/how-to-speed-up-your-python-code-through-pyspark-e3296e39da6) [](/have-you-ever-thought-about-using-python-virtualenv-fc419d8b0785) [## 有没有想过用 Python virtualenv？

### 在终端和 Jupyter 笔记本上安装和使用 Python virtualenv 的实用指南。

towardsdatascience.com](/have-you-ever-thought-about-using-python-virtualenv-fc419d8b0785) [](https://medium.datadriveninvestor.com/how-to-restore-the-original-layout-of-a-text-document-after-a-manipulation-in-python-8f3de41e8e95) [## 如何在 Python 中操作后恢复文本文档的原始布局

### 少于 10 行的代码，用于在操作后保留文本文档的布局，例如文本…

medium.datadriveninvestor.com](https://medium.datadriveninvestor.com/how-to-restore-the-original-layout-of-a-text-document-after-a-manipulation-in-python-8f3de41e8e95)