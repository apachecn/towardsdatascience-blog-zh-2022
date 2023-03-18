# 每个数据科学管道的 4 个步骤

> 原文：<https://towardsdatascience.com/the-4-steps-for-every-data-science-pipeline-b426b91c4945>

## 意见

## 大多数数据科学项目比你想象的更相似

![](img/f147045ea49b205f46ffe8afbc196502.png)

Max Ostrozhinskiy 在[Unsplash](https://unsplash.com/s/photos/steps?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【1】上拍摄的照片。

# 目录

1.  介绍
2.  查询/数据
3.  预处理
4.  火车
5.  部署
6.  摘要
7.  参考

# 介绍

本文的目的是介绍适用于每个数据科学项目的主要步骤。这个讨论不仅对初学的数据科学家有帮助，对产品经理、软件工程师，甚至高级数据科学家也有帮助，他们也可以从新的角度受益。虽然对于给定的用例，有无数的机器学习算法可供选择，但其他项目之间的共同点是获取数据、处理数据、训练算法，然后部署数据的主要部分。话虽如此，让我们通过一些清晰的例子来更深入地了解这些步骤。

# 查询/数据

![](img/d6284de9550d2e80fdcac27b17ebb3ee.png)

照片由[卡斯帕·卡米尔·鲁宾](https://unsplash.com/@casparrubin?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在[Unsplash](https://unsplash.com/s/photos/query?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【2】上拍摄。

对于任何数据科学项目，您当然需要数据来启动。然而，数据可能没有我们想象的那么容易获得。有时，它是在学术环境中给你的，这样你就可以专注于学习算法、测试以及它们如何工作，所以这是可以理解的。但是，在专业环境中，在很大程度上，有人不会给你提供一个完美的数据集，而这正是我们一直努力追求的。

> 数据库表

根据您的使用情况，理想的数据集应该由易于阅读和理解的数据组成，并带有功能和目标变量。除了来自某人的直接 CSV 文件(在某些情况下是*)之外，更好的场景是数据库中连接良好的表。对于某些项目，如果幸运的话，您的数据可以来自一个表，但是在大多数情况下，您将不得不执行带有连接的查询，由几个子查询组成，以创建一些模型特征。*

> *外部来源*

*获取数据的下一个选择可能是从数据库或数据湖中尚不存在的外部来源获取数据。例如，您可以自己或在数据工程师的帮助下，通过访问 API 将新数据加载到表中。这个过程可能需要更长的时间，你可能需要向公司证明为什么你需要留出一点时间来收集这些数据，以便它是可查询和可访问的。(*一些外部资源可能更简单，比如一个包含静态但重要的关键信息的 CSV 文件——字面意思是*)。*

*无论您是从第一个选项开始，还是不得不求助于下一个选项，这两个选项最终都可以实现按特定时间表(例如每小时、每天、每周等)查询数据的自动化任务。*

> *总而言之，以下是数据科学项目第一部分的一些步骤:*

*   *从数据库/数据湖等中查询数据。*
*   *如果没有，那么数据将需要从外部来源进入其中一个地方*
*   *一旦有了这些选项，您就可以创建一个自动化的任务，比如用 Python 来查询一个时间表*

# *预处理*

*![](img/335d599ad9f2346a2dc625cc5dc2a68a.png)*

*照片由[施洋](https://unsplash.com/@ltmonster?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在[Unsplash](https://unsplash.com/s/photos/panda?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【3】上拍摄。*

*既然您已经将数据保存在一个地方，那么很可能可以在一个文件中对所有数据进行预处理。这一步已经在特征工程之后，更确切地说，是我正在讨论的生产中预处理的自动化过程。也就是说，您已经知道您希望模型中包含哪些特性。*

*但是，在您的查询数据位于 pandas dataframe 中之后，您可能仍然需要创建这些特性(例如，在 Python 中的*，在 R* 中还有其他选项)。*

> *预处理的一些示例包括但不限于:*

*   *用数值特征的最小值、平均值、最大值填充 NA 值，或者用孤立的字符串填充 NA 值，如用于分类特征的“null ”,以告诉算法它是除其他类别值之外的特定类别值*
*   *分割或分组已查询的列来创建一个新的特性，例如，`count of orders`和`day`列，您可以创建`order_count_per_day`——有时您可以在查询本身中更容易地做到这一点，或者最好在 pandas 中这样做*

*同样，就像查询步骤一样，您将使这成为一个自动任务，该任务将获取一个 CSV 文件，该文件很可能随后被转换为 pandas 数据帧。*

# *火车*

*![](img/c4216c66fd8b86a123c30a23822619b1.png)*

*照片由[查尔斯先行者](https://unsplash.com/@charles_forerunner?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在[Unsplash](https://unsplash.com/s/photos/train?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【4】拍摄。*

*因为此时您已经选择了一种算法，所以这将是您的数据科学项目的第三个主要任务。*

*训练您的算法可以通过多种方式进行，利用不同的平台，但在这一步中相似的是，您将根据您查询的内容开始训练工作，并预先进行预处理。您还可以训练和测试打印出您的准确性和/或其他误差指标，以供进一步分析。*

> *这一步相当简单，下面总结了可能适用于您的要点:*

*   *训练经过查询和预处理的训练数据集(基于数据集位置的*)**
*   **如果您选择进行某种形式的分割，也要进行测试，以便尽早停止，从而减少模型的过度拟合**

# **部署**

**![](img/5fbf7370eaee98ebb798ace712476620.png)**

**照片由[张秀坤·吕克曼](https://unsplash.com/@exdigy?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在[Unsplash](https://unsplash.com/s/photos/docker?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【5】上拍摄。**

**数据科学管道的最后一步很可能是部署过程，在该过程中，您将为训练好的模型端点提供服务，以便可以使用最近的训练作业进行预测。**

**类似于培训工作，这一步被广泛使用和常见。一旦你有了你的终点，或者仅仅是一个训练好的模型，你就可以做出你的预测了。当然，您也可以在这里创建一个自动任务，选择最近的培训工作，这样您的模型就是最新的。**

**部署可能是软件工程方面的更多工作，但实际上，这取决于您的过程所使用的平台。**

# **摘要**

**数据科学可能有很多部分需要消化，但当你后退一步时，你会意识到，它主要归结为相同的四个步骤，其中包括查询数据或获取数据，预处理数据，训练数据，然后部署数据。**

> **以下是数据科学流程的总结步骤:**

```
*** Query/Data* Preprocess* Train* Deploy**
```

**我希望你觉得我的文章既有趣又有用。如果您同意或不同意这四个主要的数据科学步骤，请随时在下面发表评论。为什么或为什么不？关于典型的数据科学流程，您认为还有哪些重要的步骤需要指出？这些当然可以进一步澄清，但我希望我能够阐明任何数据科学项目的四个主要步骤。**

*****我不属于这些公司中的任何一家。*****

***请随时查看我的个人资料、* [Matt Przybyla](https://medium.com/u/abe5272eafd9?source=post_page-----b426b91c4945--------------------------------) 、*和其他文章，并通过以下链接订阅接收我的博客的电子邮件通知，或通过点击屏幕顶部的订阅图标* *的* ***，如果您有任何问题或意见，请在 LinkedIn 上联系我。*****

****订阅链接:**[https://datascience2.medium.com/subscribe](https://datascience2.medium.com/subscribe)**

****引荐链接:【https://datascience2.medium.com/membership】T22****

**(*如果你在 Medium* 上注册会员，我会收到一笔佣金)**

# **参考**

**[1]Max Ostrozhinskiy 在 [Unsplash](https://unsplash.com/s/photos/steps?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片，(2017)**

**[2]照片由[卡斯帕·卡米尔·鲁宾](https://unsplash.com/@casparrubin?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/query?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄，(2017)**

**[3]照片由[徐世洋](https://unsplash.com/@ltmonster?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在[un splash](https://unsplash.com/s/photos/panda?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)(2020)上拍摄**

**[4]查尔斯·弗劳恩德在 [Unsplash](https://unsplash.com/s/photos/train?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片，(2014)**

**[5]照片由[张秀坤·吕克曼](https://unsplash.com/@exdigy?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/docker?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄，(2020)**