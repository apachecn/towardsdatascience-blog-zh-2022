# 将你的笔记本改编成脚本

> 原文：<https://towardsdatascience.com/modularise-your-notebook-into-scripts-5d5ccaf3f4f3>

## 将您的代码从笔记本转换为可执行脚本的简单指南

![](img/95f72ac58cf4721a0925dc9b8816de1a.png)

照片由[詹姆斯·哈里逊](https://unsplash.com/@jstrippa?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/python-code?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

ello 世界！在这篇文章中，我将介绍一个简单的指南，如何将你的笔记本变成可执行脚本。

之前 [Geoffrey Hung](https://medium.com/@geoffreyhung) 分享了一篇关于如何将你的 [Jupyter 笔记本转换成脚本](/from-jupyter-notebook-to-sc-582978d3c0c)的极其全面的文章。然而，在我寻求生产模型的过程中，我发现在将笔记本从`.ipynb`升级到`.py`并运行整个脚本管道的模块中存在一些缺口。

如果您在分析领域工作，您可能会发现自己在某个时间点在笔记本上编写 python 代码。您可能遇到过这种方法的问题，也可能没有，但是如果您正在寻找一种用脚本执行笔记本的方法，那么这篇文章就是为您准备的。

我不会关注用脚本编写代码的好处，也不会试图比较这两种方法，因为笔记本和脚本各有利弊。如果你想知道为什么你应该作出改变，这篇文章可能会提供更多的清晰度。

[](/5-reasons-why-you-should-switch-from-jupyter-notebook-to-scripts-cb3535ba9c95) [## 你应该从 Jupyter 笔记本转向脚本的 5 个理由

### 使用脚本帮助我认识到 Jupyter 笔记本的缺点

towardsdatascience.com](/5-reasons-why-you-should-switch-from-jupyter-notebook-to-scripts-cb3535ba9c95) 

我已经创建了一个演示库来对从 [Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) 获得的信用卡数据集执行聚类分析。我将在整篇文章中使用这个存储库来分享示例片段。

**目录:**

1.  项目和代码结构
2.  抽象和重构
3.  执行管道

## 项目和代码结构

拥有适当的存储库结构是必不可少的。我们不需要一个巨大的笔记本，或者多个包含从数据提取到建模的整个流程的不同模型的笔记本，我们首先必须通过将它们分解成不同的目的来划分这种复杂性。

典型的数据分析工作流程大致包括 3 个部分:提取、转换/预处理和分析/建模。这意味着你已经可以将笔记本分成至少 3 个独立的脚本——`extraction.py`、`preprocessing.py`和`model.py`。

在我的演示存储库中，`extraction.py`缺失，因为数据集是从 Kaggle 获得的，所以提取脚本是不必要的。但是，如果您正在利用 API、web 抓取或从数据湖中转储数据，那么拥有一个提取脚本将会非常有用。根据您的团队采用的数据模型的类型，您可能会发现自己必须编写一系列查询来提取数据，并执行`join`语句来将它们合并到一个单独的表或数据帧中。

项目的典型结构可能如下所示。要查看项目结构，请执行以下操作:

```
$ tree.
├── LICENSE
├── README.md
├── config.yml
├── data
│   ├── CC_GENERAL.csv
│   └── data_preprocessed.csv
├── main.py
├── notebooks
│   ├── dbscan.ipynb
│   ├── kmeans.ipynb
│   └── preprocessing.ipynb
├── requirements.txt
└── src
    ├── dbscan.py
    ├── executor.py
    ├── kmeans.py
    ├── preprocessing.py
    └── utility.py
```

在这个项目存储库中，我们将管道分解成它的关键组件。让我们来看看它们的每一个目的。

*   `/notebooks`:这个文件夹是一个游戏场，在这里你的原始代码是以一种扁平的结构编写的，以便更简单地显示输出和遍历代码块。
*   这个文件夹包含了你的脚本将要使用的各种数据文件。这里存储的一些流行的数据格式包括`.csv`、`.parquet`、`.json`等。
*   这个文件夹存储了你所有的可执行脚本。
*   `main.py`:运行整个管道的主脚本，代码从笔记本中抽象和重构。
*   `config.yml`:一个人类可读的文件，存储用于运行脚本的可配置参数

## 抽象和重构

一旦你有了你的项目结构，下一步就是重构和抽象你的代码来降低复杂性。将代码抽象成函数和类(加上适当的变量命名)有助于区分复杂性，而不是编写迫使读者理解如何操作的代码。

下面的文章提供了一个如何重构你的笔记本的惊人总结。

[](https://www.thoughtworks.com/en-sg/insights/blog/coding-habits-data-scientists) [## 数据科学家的编码习惯

### 如果你尝试过机器学习或数据科学，你就会知道代码会很快变得混乱。通常情况下…

www.thoughtworks.com](https://www.thoughtworks.com/en-sg/insights/blog/coding-habits-data-scientists) 

**预处理**

提取数据后，在将数据用于分析或模型之前，通常会对其进行清理。一些常见的预处理包括输入缺失值、去除异常值和转换数据等。

演示存储库中涉及的一些预处理包括移除异常值和输入缺失值。这些特定的任务可以被抽象成函数并存储在一个`utility.py`脚本中，该脚本稍后可以被导入到`preprocessing.py`中。

例如，用中位数输入缺失值的函数被放在了`utility.py`文件中。

**型号**

如果您必须对同一组预处理数据使用不同的模型，我们也可以创建*类*来创建一个模型实例。在演示存储库中，我研究了执行集群时的两种算法，每种模型都被分成可执行的脚本。例如，kmeans 被抽象为`kmeans.py`，而 DBSCAn 被抽象为`dbscan.py`。

让我们导入必要的包并为 kmeans 模型创建一个类。

如果我们想要创建一个模型实例，我们可以简单地定义一个对象来初始化一个模型并存储 kmeans 模型实例。

```
kmeans = kmeans_model(df) # instantiate kmeans modelkmeans_models = kmeans.kmeans_model(min_clusters=1, max_clusters=10) # run multiple iterations of kmeans model
```

由 Sadrach Pierre 撰写的这篇文章对如何在构建模型时利用类进行了广泛的阐述。

[](/using-classes-for-machine-learning-2ed6c0713305) [## 使用类进行机器学习

### 使用面向对象编程来构建模型

towardsdatascience.com](/using-classes-for-machine-learning-2ed6c0713305) 

## 执行管道

随着分析管道的各种关键组件被抽象成函数和类，并转换成模块化的脚本，我们现在可以简单地运行整个管道。这是使用两个脚本实现的— `main.py`和`executor.py`。

**主**

主脚本`main.py`将在执行时运行整个管道，接收已加载的必要配置。在演示存储库中，我利用一个配置文件来存储参数，并点击[与它交互。](https://click.palletsprojects.com/en/8.1.x/)

**执行者**

一旦加载了模型选择及其相应的参数，我们就可以解析这些模型输入，并使用执行脚本`executor.py`来执行它。实例化模型、优化模型和拼接集群标签的步骤将在 executor 函数中展开。

要运行整个管道:

```
# execute entire pipeline with default model
python3 main.py# execute entire pipeline using DBSCAN
python3 main.py --model=dbscan# execute entire pipeline using another config file and DBSCAN
python3 main.py another_config.yml --model=dbscan
```

## 结论

将它们放在一起，我们现在有了一个逻辑项目结构，每个模块脚本都执行其特定的目的，底层代码被抽象和重构。然后，可以使用存储模型输入参数的配置文件来执行管道。

请留下评论💬如果有更多要添加的，我会很高兴将它们编辑在一起！

感谢阅读！:)

## 参考

[](/from-jupyter-notebook-to-sc-582978d3c0c) [## 从朱庇特笔记本到剧本

### 不要玩玩具模型；准备好生产你的作品吧！

towardsdatascience.com](/from-jupyter-notebook-to-sc-582978d3c0c) [](/from-jupyter-notebook-to-deployment-a-straightforward-example-1838c203a437) [## 从 Jupyter 笔记本到部署—一个简单的例子

### 一步一步的例子，采用典型的机器学习研究代码，构建一个生产就绪的微服务。

towardsdatascience.com](/from-jupyter-notebook-to-deployment-a-straightforward-example-1838c203a437)