# 随机森林算法简介

> 原文：<https://towardsdatascience.com/introduction-to-random-forest-algorithm-fed4b8c8e848>

## 算法是如何工作的，我们可以用它来做什么

![](img/d3a5b7ce536169e0505eaaa413786685.png)

杰里米·毕晓普在 [Unsplash](https://unsplash.com/s/photos/random-forest?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

![](img/25bb3b9b837db4d65ab6643919b2883c.png)

随机森林是由个体[决策树](https://databasecamp.de/en/ml/decision-trees)组成的[监督](https://databasecamp.de/en/ml/supervised-learning-models) [机器学习](https://databasecamp.de/en/machine-learning)算法。这种类型的模型被称为集合模型，因为独立模型的“集合”用于计算结果。

# 什么是决策树？

随机森林的基础由许多单独的决策树形成，即所谓的决策树。树由不同的决策层和分支组成，用于对数据进行分类。

决策树算法尝试将训练数据划分到不同的类中，以便类内的对象尽可能相似，而不同类的对象尽可能不同。这会产生多个决策级别和响应路径，如下例所示:

![](img/1b3290dbd7ec4b34c4dcf30879fbc981.png)

决策树示例|作者照片

这棵树有助于决定是否在户外进行运动，这取决于天气变量“天气”、“湿度”和“风力”。决策树将答案的分类可视化为“是”和“否”,并非常简单地阐明何时可以户外运动，何时不可以。你可以在我们关于[决策树](https://databasecamp.de/en/ml/decision-trees)的帖子中找到详细的解释。

不幸的是，决策树很快就会过度适应。这意味着算法变得太习惯于训练数据，并记住了它。因此，它在新的、看不见的数据上表现很差。

在机器学习中，目标实际上总是训练一种算法，该算法从训练数据集中学习某些能力，然后可以将它们应用于新数据。由于这个原因，现在很少使用决策树，取而代之的是非常相似的随机森林。这是通过所谓的[系综](/ensemble-methods-in-machine-learning-what-are-they-and-why-use-them-68ec3f9fef5f)方法实现的，这将在下一节详细解释。

# 随机森林如何工作

随机森林由大量的这些决策树组成，它们作为所谓的[集合](/ensemble-methods-in-machine-learning-what-are-they-and-why-use-them-68ec3f9fef5f)一起工作。每个单独的决策树做出预测，例如分类结果，并且森林使用大多数决策树支持的结果作为整个集合的预测。为什么多个决策树比单个决策树好得多？

随机森林背后的秘密是所谓的[群体智慧](https://en.wikipedia.org/wiki/The_Wisdom_of_Crowds)原理。基本思想是，许多人的决定总是比单个人或单个决策树的决定更好。这个概念最初是在连续集的估计中认识到的。

1906 年，一头公牛在一次交易会上被展示给 800 人。他们被要求在真正称重之前估计这头牛的重量。结果表明，800 个估计值的中间值与牛的实际重量只相差 1%左右。没有任何一个估计接近正确。因此，作为一个整体，观众比任何一个人都估计得更好。

这可以以完全相同的方式应用于随机森林。大量的决策树和它们的聚合预测将总是优于单个决策树。

![](img/9b3316a5f2c3cc7bc2f3e0901b7da28e.png)

随机森林合集|作者照片

然而，这只有在这些树彼此不相关并且因此单个树的错误被其他决策树补偿的情况下才是真实的。让我们回到集市上牛的重量的例子。

如果参与者彼此不一致，即不相关，则所有 800 个人的估计值的中值只有可能比每个人更好。然而，如果参与者在评估之前一起讨论，并因此相互影响，则多数人的智慧不再出现。

# 什么是装袋？

为了让随机森林产生好的结果，我们必须确保各个决策树彼此不相关。我们使用所谓的装袋法。它是集成算法中的一种方法，确保在数据集的不同子集上训练不同的模型。

决策树对它们的训练数据非常敏感。数据中的一个小变化已经可以导致明显不同的树结构。我们在装袋时利用了这一特性。因此，在训练数据集的[样本](https://databasecamp.de/en/statistics/population-and-sample)上训练森林中的每棵树，这防止了这些树彼此相关。

即使采集了样本，每个树仍然在具有原始数据集长度的训练数据集上进行训练。这是通过替换丢失的值来完成的。假设我们的原始数据集是长度为 6 的列表[1，2，3，4，5，6]。一个可能的例子是[1，2，4，6]，我们将其扩展为 2 和 6，这样我们再次得到一个长度为 6 的列表:[1，2，2，4，6，6]。装袋是从数据集中抽取一个样本并用样本中的元素将其“扩充”回原始大小的过程。

# 随机森林算法的应用领域

类似于决策树，随机森林模型用于分类任务和回归分析。这些技术在许多领域都有应用，如医药、电子商务和金融。具体应用例如:

*   预测股票价格
*   评估银行客户的信誉
*   根据医疗记录诊断疾病
*   根据购买历史预测消费者偏好

# 随机森林有什么好处？

在分类任务中使用随机森林有一些很好的理由。以下是最常见的几种:

*   **更好的性能**:正如我们在这一点上多次解释的那样，集成算法的性能平均要比单个模型的性能好。
*   **较低的过拟合风险**:决策树有很强的记忆训练数据集的倾向，即陷入过拟合。另一方面，不相关决策树的中值不容易受到影响，因此为新数据提供了更好的结果。
*   **可重复的决策**:虽然在随机森林中寻找结果比用单一决策树更令人困惑，但它的核心仍然是可以理解的。类似的算法，如[神经网络](https://databasecamp.de/en/ml/artificial-neural-networks)，无法理解结果是如何得出的。
*   **较低的计算能力**:随机森林可以在今天的计算机上相对快速地训练，因为硬件要求没有其他机器学习模型那么高。

# 什么时候不应该使用随机森林？

尽管在许多用例中，随机森林是一个可以考虑的备选方案，但也有不适合它们的情况。

随机森林应该主要用于分类任务，在这种任务中，所有带有少量示例的类都出现在训练数据集中。然而，它们不适合预测新的类或值，例如，我们从[线性回归](https://databasecamp.de/en/ml/linear-regression-basics)或[神经网络](https://databasecamp.de/en/ml/artificial-neural-networks)中知道它们。

虽然训练随机森林相对较快，但是单个分类需要相对较长的时间。因此，如果您有一个需要进行实时预测的用例，其他算法可能更适合。

如果训练数据集的填充非常不均匀，这意味着某些类只有很少的记录。装袋过程中的样品受到这种影响，进而对模型性能产生负面影响。

# 这是你应该带走的东西

*   随机森林是由个体决策树组成的监督机器学习算法。
*   它基于群体智慧的原则，即许多不相关组件的联合决策优于单个组件的决策。
*   Bagging 用于确保决策树彼此不相关。
*   随机森林用于医药以及金融和银行部门。

*如果你喜欢我的作品，请在这里订阅*[](https://medium.com/subscribe/@niklas_lang)**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，medium 允许你每月免费阅读* ***3 篇*** *。如果你希望有****无限制的*** *访问我的文章和数以千计的精彩文章，请不要犹豫，点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格**

*[](https://medium.com/illumination/intuitive-guide-to-artificial-neural-networks-5a2925ea3fa2) [## 人工神经网络直观指南

### 人工神经网络(ANN)是人工智能和人工智能领域最常用的术语

medium.com](https://medium.com/illumination/intuitive-guide-to-artificial-neural-networks-5a2925ea3fa2) [](https://medium.com/nerd-for-tech/what-are-deepfakes-and-how-do-you-recognize-them-f9ab1a143456) [## 什么是 deepfakes，怎么识别？

### Deepfakes 是使用深度学习模型人工创建的视频、图像或音频文件。比如说…

medium.com](https://medium.com/nerd-for-tech/what-are-deepfakes-and-how-do-you-recognize-them-f9ab1a143456) [](https://medium.com/@niklas_lang/membership) [## 通过我的推荐链接加入媒体- Niklas Lang

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

medium.com](https://medium.com/@niklas_lang/membership) 

*最初发布于*[*https://database camp . de*](https://databasecamp.de/en/ml/random-forests)*。**