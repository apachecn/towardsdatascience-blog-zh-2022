# PyTorch 多权重支持 API 让迁移学习再次变得琐碎

> 原文：<https://towardsdatascience.com/pytorch-multi-weight-support-api-makes-transfer-learning-trivial-again-899071fa2844>

## 一个新的 Pytorch API 使微调流行的神经网络架构变得容易，并使它们为您工作

![](img/6cf6da8655d7b226929d87a7db5bcf50.png)

艾莉娜·格鲁布尼亚克在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

微调深度学习(DL)模型从未如此简单。像 [TensorFlow](https://www.tensorflow.org/) 和 [PyTorch](https://pytorch.org/) 这样的现代 DL 框架使得这个任务变得微不足道。你可以在几分钟内拥有一个经过战斗考验的适合你需求的人工神经网络。

然而，微调您的模型只是您实验中的一步。因此，像这样的实验通常遵循以下算法:

1.  实例化您想要微调的神经网络架构。例如，如果您的领域是计算机视觉，您可能希望加载 ResNet 架构。
2.  为此架构加载一组权重。坚持我们的计算机视觉示例，通过在 ImageNet 数据集上训练模型获得的一组权重通常是首选。
3.  创建预处理函数或操作组合，将数据塑造成所需的形状。
4.  将你在第一步中实例化的神经网络与你的数据相匹配。您可以根据需要调整模型的输出层，或者冻结一些层以保持权重子集不变。这些决定取决于您的用例。
5.  在保留的测试数据集上评估您训练的神经网络。您必须以处理定型数据集的相同方式处理测试数据集。任何微小的差异都会导致模型性能的恶化，这是非常难以调试的。
6.  存储实验的元数据，例如数据集的类名，以便在以后的应用程序中使用它们，甚至执行健全性检查。

[](/xresnet-from-scratch-in-pytorch-e64e309af722) [## Pytorch 中从头开始的 xResNet

### 从你的 ResNet 架构中挤出一点额外的东西。

towardsdatascience.com](/xresnet-from-scratch-in-pytorch-e64e309af722) 

这就是你说的重点，"*等一下，这不是那么直截了当的！*“你说得对，细节决定成败；如果你什么都自己做，有很多陷阱要避免。

因此，让我们看看一个新的 PyTorch API 如何使这一切变得更加机械化，从而节省您的时间和精力。

> [Learning Rate](https://www.dimpo.me/newsletter?utm_source=medium&utm_medium=article&utm_campaign=pytorch-pretrained-api) 是为那些对 AI 和 MLOps 的世界感到好奇的人准备的时事通讯。你会在每周五收到我关于最新人工智能新闻和文章的更新和想法。订阅[这里](https://www.dimpo.me/newsletter?utm_source=medium&utm_medium=article&utm_campaign=pytorch-pretrained-api)！

# 古老的方式:徒步旅行

为了对正在发生的事情有一个坚实的理解，我们将首先检查旧的方式。我们不会训练一个模特，但我们会做几乎所有其他的事情:

*   加载神经网络架构的预训练版本
*   预处理数据集
*   使用神经网络在测试集上获得预测
*   使用数据集的元数据获得人类可读的结果

下面的代码片段总结了您需要做什么来勾选上面列表中的所有框:

在此示例中，首先加载 ResNet 神经网络体系结构。您将`pretrained`标志设置为`True`,告诉 PyTorch 您不希望它随机初始化模型的权重。相反，它应该使用通过在 ImageNet 数据集上训练模型而获得的权重。

然后，定义并初始化数据转换的组合。因此，在将图像输入模型之前，PyTorch 将:

*   将图像调整为`224x224`
*   把它转换成张量
*   将张量的每个值转换为类型`float`
*   使用一组给定的平均值和标准偏差将其标准化

接下来，您准备处理图像，并通过神经网络层获得输出。这一步是最简单的一步。

最后，您希望以人类可读的方式打印结果。这意味着你不想打印出你的模型预测的图像是`4`类。这对您或您的应用程序的用户来说没有任何意义。您希望打印出您的模型预测图像显示一只狗有 95%的可信度。

为此，您首先需要加载包含类名的元数据文件，并为您的预测确定正确的类名。

我承认这个剧本不算太大的工作量；然而，它有两个缺点:

1.  您处理测试数据集的方式的微小变化可能会导致难以调试的错误。请记住，您应该像处理训练数据集一样转换测试数据集。如果你不知道你是怎么做到的，或者别人运行了训练程序，你就完了。
2.  您必须随身携带元数据文件。同样，对该文件的任何篡改都可能导致意想不到的结果。这些错误会让你的头在键盘上撞上几天。

现在让我们看看一个新的 PyTorch API 如何让这一切变得更好。

# 走向

正如我们之前看到的，您有两个痛点需要解决:(I)总是以相同的方式处理您的训练和测试子集，(ii)消除携带单独的元数据文件的需要。

让我们通过一个示例来看看新的 PyTorch API 如何应对这些挑战:

您会立即看到脚本明显变小了。但这不是重点；该脚本更小，因为您不必从头定义一个`preprocess`函数，也不必加载任何元数据文件。

在第 9 行，您会看到一些新的东西。不是告诉 PyTorch 你需要一个预训练版本的 ResNet，首先，你实例化一个新的`weights`对象，然后用它实例化模型。在这个`weights`对象中，您可以找到在训练期间应用于数据的转换和数据集的元数据。这太棒了。

另一个观察结果是，您现在可以轻松地选择要预加载的重量。例如，您可以选择加载一组不同的权重，只需对代码进行微小的更改:

```
*# New weights with accuracy 80.674%* model **=** resnet50(weights**=**ResNet50_Weights.ImageNet1K_V2)
```

或者询问在 ImageNet 上产生最佳结果的答案:

```
*# Best available weights (currently alias for ImageNet1K_V2)* model **=** resnet50(weights**=**ResNet50_Weights.default)
```

# 结论

微调深度学习(DL)模型从未如此简单。像 TensorFlow 和 PyTorch 这样的现代 DL 框架使得这个任务变得微不足道。

然而，仍然有一些陷阱要避免。最值得注意的是，您应该总是以相同的方式处理您的训练和测试子集，并且消除携带单独的元数据文件的需要。

此外，如果您需要使用一组不同的权重作为起点，或者您已经有了一组权重，并且希望在某个中央存储库中共享它们，会发生什么情况呢？有没有一种简单的方法来实现这一点？

如您所见，新的 PyTorch 多权重支持 API 涵盖了所有这些挑战。你可以通过安装 PyTorch 的夜间版本进行试验，并对这个 [GitHub 问题](https://github.com/pytorch/vision/issues/5088)提供反馈。

# 关于作者

我的名字是[迪米特里斯·波罗普洛斯](https://www.dimpo.me/?utm_source=medium&utm_medium=article&utm_campaign=pytorch-pretrained-api)，我是一名为[阿里克托](https://www.arrikto.com/)工作的机器学习工程师。我曾为欧洲委员会、欧盟统计局、国际货币基金组织、欧洲央行、经合组织和宜家等主要客户设计和实施过人工智能和软件解决方案。

如果你有兴趣阅读更多关于机器学习、深度学习、数据科学和数据操作的帖子，请关注我的 [Medium](https://towardsdatascience.com/medium.com/@dpoulopoulos/follow) 、 [LinkedIn](https://www.linkedin.com/in/dpoulopoulos/) 或 Twitter 上的 [@james2pl](https://twitter.com/james2pl) 。

所表达的观点仅代表我个人，并不代表我的雇主的观点或意见。