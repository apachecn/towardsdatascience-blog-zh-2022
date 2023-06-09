# 评估您的 NLP 系统

> 原文：<https://towardsdatascience.com/evaluating-your-nlp-systems-2ba0d3f09f04>

## 如何为您的 NLP 场景建立一套度量和评估

![](img/1ba5846e735a07ddce77fb8f466071cf.png)

由 iMattSmart 在 [Unspash](https://unsplash.com/photos/sm0Bkoj5bnA) 拍摄的照片

所有的组织都希望围绕实验和度量标准来构建，但是这并不像听起来那么容易，因为每个度量标准只呈现了现实的某个视图。通常需要收集正确的指标来充分描述您的数据。度量标准不仅是您所做工作的可测量的结果，而且它们实际上是您业务的杠杆。这是因为一旦您选择了一个指标，您就为它进行了优化。您对要优化的度量标准的选择对项目成功的影响通常比您为其进行优化的能力更大。

这篇文章将讨论机器学习生命周期中使用的一些常见指标和评估。我将描述的所有这些指标就像工具箱中的工具。机器学习从业者有责任选择正确的工具，并构建一套适用于关键用例的指标。

# 评估您的业务效用

每一个好的企业都会有一套你的 ML 应该影响的产品指标。仅仅有很好的模型质量指标是远远不够的，因为机器学习应该显示出对你的业务和客户的可测量的改进。拥有一个框架来将您的机器学习基础设施与您的客户联系起来是关键。我在我的另一篇[博客文章](/why-accurate-models-arent-always-useful-382f0cd64cfb)中对此进行了更详细的讨论。不管你的模型看起来有多棒，它们必须为你的客户提供好处。

# 评估您的数据

在 NLP 的世界中，评估数据的质量通常是一项严格但重要的工作。在这一阶段，数据科学家会逐渐熟悉构建模型所需的知识。本节的其余部分将介绍一些计算技术，您可以尝试这些技术来获得关于数据质量的更多信息。

## 基本统计测量

测量和清除数据中噪声的最简单方法是返回到基本统计数据来评估数据集。去除停用词，理解语料库中的热门词、二元词和三元词可能会有所帮助。如果应用词汇化或词干化，可能会对数据中出现的常见单词和短语有更深入的了解。

## 主题建模

使用 LDA 或 LSI 等技术，从文本中提取主题是可能的。将这些技术与词云结合起来可能会让你对主题有更深入的了解。然而，选择主题粒度的超参数相当棘手，主题本身的可解释性也很混乱。然而，在正确的情况下，主题建模可能是一个有用的工具。

## 使聚集

与主题建模类似，有时可以使用聚类方法。如果您使用 USE 或 GloVe 嵌入您的语料库，您可以尝试构建语义集群。一些流行的聚类选项是 k-means 和 hdbscan。对于 k-means，您可能需要对您的数据有一些先验知识，以便对聚类数(k)做出有根据的猜测。Hdbscan 不要求您设置大量的集群，但是集群本身可能需要进行调整，如果引入更多的数据，集群可能不会一般化。在某些情况下，聚类可以帮助您了解数据中的噪声，甚至可以识别几个类。

## 人工数据评估

评估文本数据质量不是一个简单或自动的过程。很多时候，它需要通读数千行，并使用直觉来猜测质量。将上述方法的一些结果与定性分析相结合通常是衡量数据质量的最佳方式。

# 评估您的模型

这些是大多数数据科学家熟悉的指标，通常也是你在机器学习课上学到的内容。一大堆其他补充指标可以与其中的每一个配对，还有像模型校准这样的完整主题，它们可以为这一讨论添加细微差别。我不会在这篇博文中讨论这些话题，但是我会在这一部分的最后提到一个应该更多使用的度量框架。

## 准确(性)

准确性是最简单的衡量标准。它只是告诉我们，在我们的系统做出的所有预测中，有多少是正确的。如果你仅仅基于准确性来解释你的模型的有效性，会有很多陷阱，包括不平衡的数据集或高灵敏度的用例。精确度可以是一个很好的开始指标，但是它通常应该与其他测量技术相结合，以对您的模型进行适当的评估。在现实世界中，很少发现只有准确性才是足够的。

## 罗马纪元

AUC 代表曲线下面积。AUC 所指的曲线被称为 ROC(接收操作特性)曲线。ROC 曲线测量假阳性率对真阳性率。要真正理解 ROC 和 AUC 的本质，你可以参考我的老同事关于这个的精彩的[博文](https://blog.revolutionanalytics.com/2016/11/calculating-auc.html)。通常 roc 和 AUC 是在二元分类器的上下文中解释的，但是许多 NLP 场景通常有两个以上的意图。为了使 AUC 适应多类场景，你可以使用**一对一**或**一对一**技术。OvR 和 OvO 本质上将你的多类分解成许多不同的二元分类器。

## 精确度、召回率和 F1

许多 NLP 系统依赖于意图分类。这就是模型的工作是预测特定文本的意图。在分类器中，精确度、召回率和 F1 分数是衡量这些意图预测质量的最常见方法。Precision 告诉你所有你预测为正(TP + FP)的项目，有多少实际上是正的。回忆告诉你所有实际的阳性标记项目(TP + FN)，你预测有多少是阳性的。F1 是精确度和召回率的调和平均值。

在多类场景中，您通常可以获得每个类的精度、召回率和 F1。`sklearn`有一个分类报告，可以在一行代码中计算所有这些指标。如果你的多类场景有太多意图，可读性可能会有限制。F1 是不平衡数据集问题的一个很好的度量，可以帮助对抗一些**精度**的限制。

## 混淆矩阵

混淆矩阵没有单点度量，您可以使用它来评估您的模型在看不见的数据上的表现。但是，它们提供了一种定性评估模型预测能力的好方法。通常，在聊天机器人中构建基于意图的模型时，您可能会遇到这些意图的定性问题。话语有多相似？意向 X 是意向 Y 的子集吗？混淆矩阵可以帮助您相对快速地分析和诊断数据问题。我经常把它作为我的每级精度/召回/F1 报告的伴侣。混淆矩阵的一个缺点是，如果你有太多的意图，可解释性就会受到影响。混淆矩阵还会导致定性分析，因此评估存在一定的偏差风险。

## BLEU 评分

如果你正在处理语言翻译的情况，“双语评估替角”分数是一个很好理解的评估翻译质量的方法。分数由机器生成的翻译与专业人工翻译的接近程度决定。当然，如果你没有专业的人工翻译，你可以找到其他方法来获得高质量的翻译，以非常好地近似真实情况。

## 清单失败率

有时候，点度量是不够的。从论文 [**行为准确性:用清单**](https://arxiv.org/abs/2005.04118) 进行 NLP 模型的行为测试来看，清单是一种完全不同的、伟大的评估你的 NLP 模型的方式**。**本文深入探讨了对 NLP 系统中可能出现的常见行为进行单元测试的想法。

Checklist 本身是一个框架，在这个框架中，您可以用一种可量化、可测量的方式实际测试不同维度的 NLP 模型。它允许您测试您的模型对拼写错误、随机标点错误、命名实体识别问题以及现实文本中出现的其他常见错误的响应情况。本质上，使用像`nlpaug`这样的包或`checklist`包本身，您可以实际模拟您想要识别和测量数据中这些不同维度的**失败率**的测试类型。实施纸质清单的缺点是您必须自己生成文本，这可能是一个昂贵的过程。检查表也简单地描述了问题，但是解决检查表发现的问题可能是复杂的。

# 结论

这篇文章涵盖了一些流行的方法来衡量和评估你的 ML 生命周期的不同部分。这不是一个详尽的列表，因为有许多子领域和主题都有一组测量值。然而，我在这里列出的评估工具应该是构建一个整体的度量套件来探索 NLP 系统的不同维度的良好开端。