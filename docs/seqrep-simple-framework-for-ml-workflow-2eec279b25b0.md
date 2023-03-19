# seq rep——ML 工作流的简单框架

> 原文：<https://towardsdatascience.com/seqrep-simple-framework-for-ml-workflow-2eec279b25b0>

## 从标记数据到用几行代码评估模型

你有没有担心过**构建正确的机器学习(ML)流水线**？
你是想专注于模型还是预处理部分，你**不想费心把所有东西都放在一起**？
你想只使用**几行代码**来测试你的想法吗？

如果你对任何一个问题的回答是肯定的，或者你只是好奇，继续读下去！

![](img/24a891b304172e4e69bbe7c645f24b4b.png)

[Goran Ivos](https://unsplash.com/@goran_ivos?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

# SeqRep 是什么？

SeqRep 是一个简单的 Python 库，用于简化 ML 任务的编码。它主要侧重于序列(时间序列)数据；但是，它也可以用于其他类型的数据。

在任何 ML 任务的开始，我们都有**数据**。最终，我们要得到一个模型的**性能结果**。在这两个阶段之间，有几个步骤。有时候我们需要做更多，有时候几步就够了。SeqRep 就是来缓解这种情况的。我们可以像玩具积木一样构建工作流。

# `PipelineEvaluator` -主程序块

*序列*的基本类是`PipelineEvaluator`类。这是放置我们特殊积木的地方。通常，我们会插入一个`Splitter`,因为我们希望在不同于用于训练的数据上评估模型。然后我们添加一个模型。 *SeqRep* 指望模型有`fit`和`predict`方法。因此，您可以轻松使用任何型号的 *Scikit* 。最后，可以添加`Evaluator`对象来获得一些评估度量值。您可以使用一些预定义的指标，也可以实现自己的指标(例如，使用您想要的指标，比如[精度和](/finally-remember-precision-and-recall-94b4d481f9bf))。

代码可能如下。

```
pipe_eval = PipelineEvaluator(labeler = NextColorLabeler(),
                              splitter = TrainTestSplitter(),
                              model = SVC(),
                              evaluator = ClassificationEvaluator(),
                              )
result = pipe_eval.run(data=data)
```

# 添加其他块

让我们通过**添加其他模块**来构建更大更复杂的东西。

![](img/b2063029f83aa5141ea340997ddb995f.png)

由[米歇尔·bożek](https://unsplash.com/@bozu?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

## 标记

有时，我们有数据，但我们没有标签(我们将使用监督模型)。例如，它适用于任务是预测下一个值的时间序列数据(例如，[温度预测](https://github.com/MIR-MU/seqrep/blob/main/examples/RegressionExample-Temperature.ipynb)或[功耗预测](https://github.com/MIR-MU/seqrep/blob/main/examples/RegressionExample-Electric_Power_Constumption.ipynb))。这可以使用`Labeler`对象来解决。同样，您可以通过继承`Labeler`类或使用任何实现的。

那么当**您分别拥有数据和标签**时，情况会怎样呢？没问题，`labeler`不需要使用。您可以直接将数据和标签添加到主对象中。请参见下面的代码片段。

```
pipe_eval.data = X_data
pipe_eval.labels = labels
```

同样，如果您的数据已经分离，您就不必使用`TrainTestSplitter`。

```
pipe_eval.X_train = X_train
pipe_eval.y_train = y_train
pipe_eval.X_test = X_test
pipe_eval.y_test = y_test
```

这里要注意的是，预定义的`TrainTestSplitter`并没有对数据进行洗牌(参数`shuffle`的默认值为 *False* )。这是因为我们通常不想重新排列时序数据**。然而，在某些情况下，它可能是有意义的。所以要照顾好这个参数！**

## 预处理

有时**适当的预处理比模型选择更重要。**

在 *SeqRep* 中，我们使用了来自 *ScikitLearn* 的`Pipeline`；但是，你可以定义你的`FeatureExtractor`。通常，预处理的最后一步(有时是唯一的一步)是*缩放*。您可能想要(根据可用数据)创建要素。一些实现的提取器(例如，用于计算价格、心率变异性特征等技术指标)已经可用。

可以使用简单的`PreviousValuesExtractor`，它是为顺序数据设计的。它只是增加了前一个示例的功能。尽管如此，您可以通过继承类`FeatureExtractor`来定义自己的提取器。唯一需要实现的方法是`transform`方法。

## 特征的减少或选择

添加有价值的功能是值得的；但是，有时候，**有无价值的特性**。

`FeatureReductor`可用于选择(不知何故)好的特征。在输入模型之前，应该省略一些(原始)特征(例如，文本特征)。`FeatureSelector`就是来帮你做这件事的！

这里就不赘述了。然而，基本的特征约简方法已经在包中实现。

# 形象化

*SeqRep* 包能够**为您可视化各个步骤**(如果您想要的话)。在主类(`PipelineEvaluator`)初始化期间，将您想要绘制的模块名称写入参数`visualize`。

请注意，看到可视化的训练数据可能会很好；但是，处理从 x 要素到二维的向下投影可能需要一些时间。

![](img/ee85157ff9b7cf977bee966fbd8429b0.png)

[Firmbee.com](https://unsplash.com/@firmbee?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

# SeqRep 的高级使用

*SeqRep* 包非常**可定制**。您可以定义自己的单个步骤的实例。

一些步骤可以省略，包括学习和评估。因此您可以创建仅用于预处理的工作流。但是，您可以将输出用于主类的其他实例(`PipelineEvaluator`)以及一些模型和评估器。如果我们想要测试更多的模型(或者不同的模型设置)，这可能是有益的。你可以在[这个例子](https://github.com/MIR-MU/seqrep/blob/main/examples/RegressionExample-Electric_Power_Constumption.ipynb)中看到这个用法。

# 下一步是什么？

您可以查看存储库中的[示例。](https://github.com/MIR-MU/seqrep/tree/main/examples)

你可以通过打开[模板 Jupyter 笔记本](https://github.com/MIR-MU/seqrep/blob/main/examples/TEMPLATE.ipynb)开始使用 *SeqRep* 包。

欢迎反馈！没有什么是完美的，所以我会感谢你的评论。如果你觉得 *SeqRep* 包有帮助，我会很高兴你**给** [**仓库**](https://github.com/MIR-MU/seqrep) 打一颗星。

[](https://github.com/MIR-MU/seqrep)  

今天，我们简单介绍了一下 *SeqRep* 包的功能。希望可以理解。我计划写另一篇文章，关于使用基因组数据包**的**案例研究**。**

## **新到中？[解锁](https://medium.com/@jakubrysavy/membership)无限访问(并支持我)！**