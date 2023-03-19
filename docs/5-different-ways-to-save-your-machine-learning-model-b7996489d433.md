# 保存机器学习模型的 5 种不同方法

> 原文：<https://towardsdatascience.com/5-different-ways-to-save-your-machine-learning-model-b7996489d433>

## 简化重用模型的过程

![](img/e57e65cb37b42f8bc7e479a330075223.png)

照片由[弗雷迪·雅各布](https://unsplash.com/@thefredyjacob?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

保存经过训练的机器学习模型是机器学习工作流程中的一个重要步骤:它允许您在将来重用它们。例如，您很可能必须比较模型，以确定投入生产的冠军模型——在模型经过培训后保存模型会使这一过程更容易。

另一种方法是在每次需要使用模型时对其进行训练，这可能会极大地影响生产率，尤其是在模型需要长时间训练的情况下。

在本帖中，我们将介绍 5 种不同的方法来保存你训练过的模型。

# 头号泡菜

Pickle 是 Python 中最流行的序列化对象的方法之一；你可以用 Pickle 把你训练好的机器学习模型序列化，保存到一个文件里。稍后或在另一个脚本中，您可以反序列化该文件以访问定型模型并使用它进行预测。

下面是我创建的用于保存模型管道的实用函数，我还将在培训管道中演示它的功能:

让我们看看这在我们的培训管道脚本中是如何工作的:

您可以看到我们在第 50 行调用了`save_pipeline`实用函数。

为了加载管道，我创建了另一个名为`load_pipeline`的实用函数。

在我们的预测脚本中，我们将管道加载到一个名为`_fraud_detection_pipe`的变量中:我们现在可以使用这个变量作为我们训练的管道对象的实例来进行预测。让我们看看剧本的其余部分…

***注*** *:第 14 行加载训练好的模型。*

# #2 工作库

Joblib 是 pickle 的替代工具，我们可以用它来保存[和加载]我们的模型。它是 SciPy 生态系统的一部分，在携带大型 NumPy 数组的对象上更有效——在这个 [StackOverflow 讨论](https://stackoverflow.com/questions/12615525/what-are-the-different-use-cases-of-joblib-versus-pickle#:~:text=joblib%20is%20usually,zlib%20or%20lz4.)中了解更多关于 Joblib 的好处。

> “Joblib 是一套工具，用于在 Python 中提供轻量级流水线操作。特别是:函数的透明磁盘缓存和惰性重求值(memoize 模式)简单易行的并行计算。”
> — **来源** : [Joblib 文档](https://joblib.readthedocs.io/)

为了用 Joblib 保存我们的模型，我们只需要对我们的`save_pipeline()`函数进行修改。

注意，我们有重要的`joblib`而不是`pickle`，在第 16 行，我们用`joblib`序列化了我们的模型管道。

***注*** *:有兴趣的读者可以在我的*[*GitHub*](https://github.com/kurtispykes/fraud-detection-project)*上看到完整代码。*

# 第三名 JSON

另一种保存模型的方法是使用 JSON。与 Joblib 和 Pickle 不同，JSON 方法不一定会直接保存拟合的模型，而是保存构建模型所需的所有参数——当需要完全控制保存和恢复过程时，这是一个好方法。

***注*** *:有兴趣的读者不妨详细了解一下* [*继承*](https://medium.com/geekculture/inheritance-getting-to-grips-with-oop-in-python-2ec35b52570#:~:text=the%20super()%20function.-,The%20super()%20function,-The%20super()) *来了解我们是如何构建这个类的。*

现在我们可以在我们的`MyLogisticRegression`实例上调用`save_json()`方法来保存参数。

我们可以在另一个脚本中调用它，如下所示:

您可以使用所有这些数据来重现之前构建的模型。

# 第四名 PMML

预测模型标记语言(PMML)是从业者用来保存他们的机器学习模型的另一种格式。它比 Pickle 健壮得多，因为 PMML 模型不依赖于创建它们的类——这与 Pickle 模型不同。

我们也可以如下加载模型:

# 第五名 Tensorflow Keras

Tensorflow Keras 可用于将 Tensorflow 模型保存到`SavedModel`或`HDF5`文件。

让我们建立一个简单的模型，并用 Tensorflow Keras 保存它—首先，我们将从生成一些训练数据开始。

现在让我们建立一个序列模型:

在上面代码的第 16 行，我们在顺序模型实例上使用了`save()`方法，并传递了一个路径到我们想要保存模型的目录。

为了加载模型，我们简单地在`models`对象上使用了`load_model()`方法。

# 包裹

在本文中，我们讨论了保存机器学习模型的 5 种不同方法。记录用于构建模型的 Python 版本以及所用库的版本也很重要-这些数据将简化重新创建模型构建环境的过程，以便日后可以重现它们。

*感谢阅读。*

**联系我:**
[LinkedIn](https://www.linkedin.com/in/kurtispykes/)
[Twitter](https://twitter.com/KurtisPykes)
[insta gram](https://www.instagram.com/kurtispykes/)

如果你喜欢阅读这样的故事，并希望支持我的写作，可以考虑成为一名灵媒。每月支付 5 美元，你就可以无限制地阅读媒体上的故事。如果你使用[我的注册链接](https://kurtispykes.medium.com/membership)，我会收到一小笔佣金。

已经是会员了？[订阅](https://kurtispykes.medium.com/subscribe)在我发布时得到通知。

<https://kurtispykes.medium.com/subscribe> 