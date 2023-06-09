# 我能相信我的模型的概率吗？深入探讨概率校准

> 原文：<https://towardsdatascience.com/can-i-trust-my-models-probabilities-a-deep-dive-into-probability-calibration-fc3886cfc677>

## 数据科学统计学

## 概率校准实用指南

![](img/6393973567cfc5bdc6a7c4d4f853e28d.png)

由 [Edge2Edge 媒体](https://unsplash.com/@edge2edgemedia?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

假设你有一个二元分类器和两个观察值；模型将它们分别评分为`0.6`和`0.99`。具有`0.99`分数的样本属于阳性类的可能性更大吗？对某些模型来说，这是真的，但对其他模型来说可能不是。

这篇博文将深入探讨概率校准——这是每个数据科学家和机器学习工程师的必备工具。概率校准允许我们确保来自我们的模型的较高分数更可能属于正类。

这篇文章将提供可复制的开源软件代码示例，这样你就可以用你的数据运行它了！我们将使用 [sklearn-evaluation](https://github.com/ploomber/sklearn-evaluation?utm_source=medium&utm_medium=blog&utm_campaign=calibration-curve) 进行绘图，使用 [Ploomber](https://github.com/ploomber/ploomber?utm_source=medium&utm_medium=blog&utm_campaign=calibration-curve) 并行执行我们的实验。

> 嗨！我叫爱德华多，我喜欢写关于数据科学的所有东西。如果您想了解我的最新内容。在[媒体](https://medium.com/@edublancas?utm_source=medium&utm_medium=blog&utm_campaign=calibration-curve)或[推特](https://twitter.com/edublancas?utm_source=medium&utm_medium=blog&utm_campaign=calibration-curve)上关注我。感谢阅读！

# **什么是概率校准？**

当训练一个二元分类器时，我们感兴趣的是发现一个特定的观察值是否属于正类。*正类*的意思取决于上下文。例如，如果处理电子邮件过滤器，这可能意味着某个特定的邮件是垃圾邮件。如果致力于内容审核，这可能意味着*有害帖子*。

使用一个实数值范围内的数字比是/否答案提供了更多的信息。幸运的是，大多数二元分类器可以输出分数(注意，这里我使用的是单词*分数*，而不是*概率*，因为后者有严格的定义)。

让我们看一个逻辑回归的例子:

`predict_proba`函数允许我们输出分数(对于逻辑回归的情况，这是独立的概率):

**控制台输出(1/1):**

输出中的每一行代表属于类别`0`(第一列)或类别`1`(第二列)的概率。不出所料，行加起来是`1`。

直觉上，我们期望模型在对特定预测更有信心时给出更高的概率。例如，如果属于类别`1`的概率是`0.6`，我们可以假设该模型不像一个概率估计为`0.99`的例子那样有信心。这是校准良好的模型所表现出的特性。

这个属性是有利的，因为它允许我们对干预进行优先级排序。例如，如果致力于内容审核，我们可能有一个模型将内容分类为*无害*或*有害*；一旦我们获得了预测，我们可能会决定只要求审查团队检查那些被标记为*有害的*，而忽略其余的。但是团队能力有限，最好只关注危害概率大的帖子。为了做到这一点，我们可以对所有的新帖子进行评分，取分数最高的前`N`，然后将这些帖子交给评审团队。

然而，模型并不总是表现出这种特性，因此，如果我们想要根据输出概率对预测进行优先排序，我们必须确保我们的模型是校准良好的。

让我们看看我们的逻辑回归是否被校准。

**控制台输出(1/1):**

![](img/5d48b4d9698dc7cb9be36862ed820673.png)

现在让我们按概率箱分组，并检查该箱内属于正类的样本的比例:

**控制台输出(1/1):**

我们可以看到该模型得到了合理的校准。对于`0.0`和`0.1`之间的输出，没有样本属于阳性类别。对于其余部分，实际正类样本的比例接近值边界。比如`0.3`到`0.4`之间的，29%属于正类。逻辑回归由于其[损失函数](https://en.wikipedia.org/wiki/Loss_functions_for_classification?utm_source=medium&utm_medium=blog&utm_campaign=calibration-curve)而返回精确校准的概率。

很难评估表格中的数字；这就是校准曲线的用武之地，它允许我们直观地评估校准。

# 什么是校准曲线？

校准曲线是模型校准的图形表示。它允许我们将我们的模型与一个目标进行比较:一个完美校准的模型。

一个完全校准的模型将在 10%确信该模型属于正类时输出得分`0.1`，在 20%确信时输出得分`0.2`，以此类推。所以如果我们画这个，我们会有一条直线:

![](img/fc248b24f0553ba35ff2a6aa59f46397.png)

完美校准的模型。图片作者。

此外，校准曲线允许我们比较几个模型。例如，如果我们想要将一个校准良好的模型部署到生产中，我们可能会训练几个模型，然后部署一个校准更好的模型。

# 恋恋笔记本

我们将使用笔记本来运行我们的实验并更改模型类型(例如，逻辑回归、随机森林等)。)和数据集大小。这里可以看到[源代码](https://github.com/ploomber/posts/blob/master/calibration-curve/fit.ipynb?utm_source=medium&utm_medium=blog&utm_campaign=calibration-curve)。

笔记本很简单:它生成样本数据，拟合模型，对样本外预测进行评分，并保存它们。运行所有实验后，我们将下载模型的预测，并使用它们绘制校准曲线和其他曲线。

# 运行实验

为了加速我们的实验，我们将使用 [Ploomber Cloud](https://www.cloud.ploomber.io/signin.html?utm_source=medium&utm_medium=blog&utm_campaign=calibration-curve) ，它允许我们参数化并并行运行笔记本。

*注意:本节中的命令是 bash 命令。在终端中运行它们，或者添加* `*%%sh*` *魔法，如果你在 Jupyter 中执行它们的话。*

让我们下载笔记本:

**控制台输出(1/1):**

现在，让我们运行我们的参数化笔记本。这将触发我们所有的平行实验:

**控制台输出(1/1):**

大约一分钟后，我们将看到所有 28 个实验都已执行完毕:

**控制台输出(1/1):**

让我们下载概率估计:

**控制台输出(1/1):**

# 加载输出

每个实验都将模型的预测存储在一个`.parquet`文件中。让我们加载数据来生成一个数据框，其中包含模型类型、样本大小和模型概率的路径(由`predict_proba`方法生成)。

**控制台输出(1/1):**

![](img/d06d97a1db5efaf36d2753f122f84192.png)

`name`是型号名称。`n_samples`是样本大小，`path`是每个实验生成的输出数据的路径。

# 逻辑回归:一个校准良好的模型

逻辑回归是一个特例，因为它的目标函数是最小化对数损失函数，所以通过设计得到了很好的校准。

让我们看看它的校准曲线:

**控制台输出(1/1):**

![](img/0fd1faab864ddb9bae3f4dbb81dbf956.png)

逻辑回归校准曲线。图片作者。

您可以看到概率曲线非常类似于一个完美校准的模型。

# 样本大小的影响

在上一节中，我们展示了逻辑回归是用来产生校准概率的。但是要注意样本大小。如果没有足够大的训练集，模型可能没有足够的信息来校准概率。下图显示了随着样本量的增加，逻辑回归模型的校准曲线:

**控制台输出(1/1):**

![](img/aa8d5d022e926a27b0061fe60ffe6692.png)

不同样本量的逻辑回归校准曲线。图片作者。

可以看到，对于 1，000 个样本，校准效果很差。然而，一旦你通过了 10，000 个样本，更多的数据不会显著改善校准。请注意，这种影响取决于您的数据的动态性；在您的用例中，您可能需要更多或更少的数据。

# 非校准估计量

虽然逻辑回归被设计为产生校准的概率，但是其他模型不显示这种属性。让我们看看 AdaBoost 分类器的校准图:

**控制台输出(1/1):**

![](img/6ff73d7644f656744ce575b3ef817872.png)

不同样本量下的 AdaBoost 校准曲线。图片作者。

可以看到校准曲线看起来高度失真:阳性分数(y 轴)与其对应的平均预测值(x 轴)相差甚远；此外，该模型甚至不产生沿整个`0.0`到`1.0`轴的值。

即使样本量为 1000，000，曲线也可以更好。在接下来的章节中，我们将看到如何解决这个问题，但是现在，记住这一点:不是所有的模型都会默认产生校准的概率。特别是，最大间隔方法，如 boosting (AdaBoost 是其中之一)、支持向量机和朴素贝叶斯产生未校准的概率(Niculescu-Mizil 和 Caruana，2005)。

AdaBoost(不同于逻辑回归)有一个不同的优化目标，不产生校准概率。然而，这并不意味着模型不准确，因为在创建二元响应时，分类器是通过其准确性来评估的。我们来对比一下两款机型的性能。

现在，我们绘制并比较分类指标。AdaBoost 的指标显示在每个方块的上半部分，而逻辑回归的指标显示在下半部分。我们将看到两种型号具有相似的性能:

**控制台输出(1/1):**

![](img/fa984970d577eaf18357ac09eede5313.png)

AdaBoost 和逻辑回归度量比较。图片作者。

# 概率分布的重要性

到目前为止，我们仅使用校准曲线来判断分类器是否经过校准。然而，另一个需要考虑的关键因素是模型预测的分布。也就是分值有多常见或者多罕见。

让我们看看随机森林校准曲线:

**控制台输出(1/1):**

![](img/65456ba93d273079ede286ae17f021df.png)

随机森林与逻辑回归校准曲线。图片作者。

随机森林遵循与逻辑回归相似的模式:样本量越大，校准越好。众所周知，随机森林能够提供精确的概率(Niculescu-Mizil 和 Caruana，2005 年)。

然而，这只是一部分情况。首先，让我们看看输出概率的分布:

**控制台输出(1/1):**

![](img/bcbefeafb972dcb57287f693db532c33.png)

随机森林与概率的逻辑回归分布。图片作者。

我们可以看到，随机森林将概率推向`0.0`和`1.0`，而来自逻辑回归的概率则不那么偏斜。当随机森林被校准时，在`0.2`到`0.8`区域没有很多观察值。另一方面，逻辑回归在`0.0`到`1.0`区域一直有支撑。

一个更极端的例子是当使用一棵树时:我们会看到一个更加偏斜的概率分布。

**控制台输出(1/1):**

![](img/5f24d3e1687736dfbc76b03c7485d6a6.png)

概率的决策树分布。图片作者。

让我们看看概率曲线:

**控制台输出(1/1):**

![](img/121bea878dec0e5510c8cf114d0e870e.png)

不同样本量的决策树概率曲线。图片作者。

你可以看到我们的两个点(`0.0`和`1.0`)被校准了(它们相当接近虚线)。但是，由于模型没有输出具有其他值的概率，因此不再存在更多数据。

# 校准分类器

![](img/2871b6cb7e49815d390499fa3705c861.png)

培训/校准/测试分割。图片作者。

有几种技术可以校准分类器。它们通过使用您的模型的未校准预测作为输入来训练第二个模型，该模型将未校准分数映射到校准概率。我们必须使用一组新的观察数据来拟合第二个模型。否则，我们将在模型中引入偏差。

有两种广泛使用的方法:普拉特的方法和保序回归。当数据较少时，建议使用 Platt 的方法。相反，当我们有足够的数据来防止过度拟合时，保序回归更好(Niculescu-Mizil 和 Caruana，2005)。

考虑到校准不会自动产生校准良好的模型。可以更好地校准预测的模型是提升树、随机森林、支持向量机、袋装树和神经网络(Niculescu-Mizil 和 Caruana，2005)。

请记住，校准分类器会增加开发和部署过程的复杂性。因此，在尝试校准模型之前，确保没有更直接的方法来进行更好的数据清理或使用逻辑回归。

让我们看看如何使用 Platt 的方法训练、校准和测试分割来校准分类器:

**控制台输出(1/1):**

![](img/be9b884d3e0f74f71aae392b70983473.png)

未校准与校准模型。图片作者。

或者，您可以使用交叉验证和测试折叠来评估和校准模型。让我们看一个使用交叉验证和保序回归的例子:

![](img/c801afe1bfafcaa32a7475808a413b30.png)

使用交叉验证进行校准。图片作者。

**控制台输出(1/1):**

![](img/2f713fc05f5ab64395c4a99b8311583d.png)

未校准与校准模型(使用交叉验证)。图片作者。

# 校准多类别模型

在上一节中，我们讨论了用于校准分类器的方法(普拉特方法和保序回归)，这些方法仅支持二元分类。

然而，校准方法可以通过遵循*一对一* [策略](https://scikit-learn.org/stable/modules/multiclass.html?utm_source=medium&utm_medium=blog&utm_campaign=calibration-curve#ovr-classification)扩展到支持多个类别，如下例所示:

**控制台输出(1/1):**

![](img/67f454e831a243326f8449aeac2dcd92.png)

未校准与校准的多级模型。图片作者。

# 结束语

在这篇博文中，我们深入探讨了概率校准，这是一个实用的工具，可以帮助你开发更好的预测模型。我们还讨论了为什么有些模型无需额外步骤就能显示校准预测，而其他模型则需要第二个模型来校准其预测。通过一些模拟，我们还演示了样本大小的影响，并比较了几个模型的校准曲线。

为了并行运行我们的实验，我们使用了 [Ploomber Cloud](https://www.cloud.ploomber.io/signin.html?utm_source=medium&utm_medium=blog&utm_campaign=calibration-curve) ，为了生成我们的评估图，我们使用了 [sklearn-evaluation](https://github.com/ploomber/sklearn-evaluation?utm_source=medium&utm_medium=blog&utm_campaign=calibration-curve) 。Ploomber Cloud 有一个免费层，sklearn-evaluation 是开源的，所以你可以从这里获取[这篇笔记本格式的文章，获得](https://github.com/ploomber/posts/blob/master/calibration-curve/fit.ipynb?utm_source=medium&utm_medium=blog&utm_campaign=calibration-curve) [API 密钥](https://www.cloud.ploomber.io/signin.html?utm_source=medium&utm_medium=blog&utm_campaign=calibration-curve)，然后用你的数据运行代码。

如果您有任何问题，欢迎加入我们的[社区](https://ploomber.io/community?utm_source=medium&utm_medium=blog&utm_campaign=calibration-curve)！

# 参考

*   [概率校准(scikit-learn 文档)](https://scikit-learn.org/stable/modules/calibration.html?utm_source=ploomber&utm_medium=blog&utm_campaign=calibration-curve)
*   [校准分类器(scikit-learn 文档)](https://scikit-learn.org/stable/modules/calibration.html?utm_source=ploomber&utm_medium=blog&utm_campaign=calibration-curve#calibrating-a-classifier)
*   【David S. Rosenberg 的概率校准笔记
*   [分类:预测偏差](https://developers.google.com/machine-learning/crash-course/classification/prediction-bias?utm_source=ploomber&utm_medium=blog&utm_campaign=calibration-curve)
*   [用监督学习预测好的概率。亚历山德鲁·尼古列斯库-米齐尔和里奇·卡鲁阿纳(2005 年)](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf?utm_source=ploomber&utm_medium=blog&utm_campaign=calibration-curve)

# 使用的包

以下是我们用于代码示例的版本:

**控制台输出(1/1):**

*最初发表于*[*ploomber . io*](https://ploomber.io/blog/calibration-curve/?utm_source=medium&utm_medium=blog&utm_campaign=calibration-curve)*。*