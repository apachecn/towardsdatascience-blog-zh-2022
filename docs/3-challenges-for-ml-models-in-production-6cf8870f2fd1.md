# ML 模型在生产中面临的 3 大挑战

> 原文：<https://towardsdatascience.com/3-challenges-for-ml-models-in-production-6cf8870f2fd1>

## 这些都是不容忽视的。

![](img/1f1549a771afe19dabfb4a6d19731ab0.png)

照片由 [Unsplash](https://unsplash.com/s/photos/factory?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的[德韦恩·希尔斯](https://unsplash.com/@dhillssr?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)拍摄

很大一部分机器学习(ML)模型从未投入生产。在 Jupyter 笔记本中创建一个模型是一回事，但将它部署到生产中并作为一项持续服务来维护它则是另一回事。这是一个涉及许多相关步骤的过程，这些步骤可以归为机器学习操作这一术语， [MLOps](/from-jupyter-notebooks-to-real-life-mlops-9f590a7b5faa) 。

MLOps 如此重要，我认为它是数据科学中的下一件大事，或者说它已经是一件大事了😊。为了通过机器学习产生长期的商业价值，我们需要从数据采集到模型监控都坚持 MLOps 原则。

在生产中维护 ML 模型有一些挑战，需要解决和正确处理这些挑战，以使整个系统可靠和健壮。在本文中，我们将讨论其中的 3 个挑战。

## 1.没有人为干预

ML 模型极有可能做出比我们更好的预测。他们比我们工作得更快，可扩展性更强。另一方面，ML 模型的一个缺点是它们没有常识。因此，ML 模型有可能产生不切实际的结果。

ML 模型的核心是训练数据。它们反映了它所训练的数据中存在的结构和关系。在揭示数据中的依赖关系方面，ML 模型比我们好得多。然而，当他们遇到新事物时，他们可能会产生荒谬的结果。

由于生产中的 ML 系统很可能没有人工干预，因此总是存在结果不合理的风险。

这些风险的影响因应用而异。在推荐引擎的情况下，这并不重要。然而，如果我们有一个自动反馈供应变化的预测引擎，那么事情可能会变得非常糟糕。

这些风险不应被视为赤字。我们只需要记住它们。我们可以通过频繁地重新训练我们的模型，基于某些限制检查结果，并在生产中监控模型来克服这一挑战。

## 2.数据更改

世界在不断变化。唯一不变的是变化本身。

> 生活中唯一不变的是变化——赫拉克利特。

一切都随世界而变，数据也是如此。不可避免地会有以前看不到的数据点，我们希望我们的模型能为它们产生结果。

考虑一个垃圾邮件检测模型。诈骗者改变他们的策略，创造新的垃圾邮件。没有经过这些新例子训练的模型很可能无法捕捉到它们。这个概念被称为数据漂移，如果处理不当，可能会导致严重的问题。

另一个对模型准确性有影响的变化叫做概念漂移。当因变量和目标变量之间的关系发生变化时，就会发生这种情况。在欺诈检测模型中可以观察到概念漂移的典型例子。如果一个过去没有被归类为欺诈的交易，现在也可以是欺诈。

数据和概念漂移都可以通过实施稳健可靠的监控系统和反馈回路来处理。然后我们可以决定何时重新训练我们的模型。

## 3.利益相关者之间的沟通

建立一个 ML 系统需要不同的技能。一般来说，数据科学家、数据工程师、软件工程师、DevOps 工程师和主题专家参与机器学习生命周期。

主题专家与数据科学家合作，定义要用机器学习解决的业务问题。他们设置了 KPI 来衡量模型的性能。

数据工程师的主要职责是数据采集和 ETL 操作。软件工程师和 DevOps 工程师处理与 IT 相关的任务。

角色和任务可能因公司而异。不变的是，不同的利益相关者之间需要稳健有效的沟通，才能使事情顺利进行。否则，我们最终会有不必要的时间间隔，甚至会使项目失败。

为了产生商业价值，需要将 ML 模型部署到生产中。这绝对不是一件容易的事。创建这样一个系统超出了数据科学家的技能范围。然而，了解与在生产中维护 ML 模型相关的风险并采取预防措施可以使事情变得更容易。

*你可以成为* [*媒介会员*](https://sonery.medium.com/membership) *解锁我的全部写作权限，外加其余媒介。如果你已经是了，别忘了订阅*<https://sonery.medium.com/subscribe>**如果你想在我发表新文章时收到电子邮件。**

*<https://sonery.medium.com/membership>  

感谢您的阅读。如果您有任何反馈，请告诉我。*