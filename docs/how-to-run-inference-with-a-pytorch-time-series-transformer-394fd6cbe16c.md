# 如何使用 PyTorch 时序转换器运行推理

> 原文：<https://towardsdatascience.com/how-to-run-inference-with-a-pytorch-time-series-transformer-394fd6cbe16c>

## 在您不知道解码器输入的情况下，在推断时间使用 PyTorch 转换器进行时间序列预测

![](img/a3c24d1314f123d98a2397d943ea8dc2.png)

[郭锦恩](https://unsplash.com/@spacexuan?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

在这篇文章中，我将展示如何使用 PyTorch 转换器进行时间序列预测。具体来说，我们将使用 PyTorch 时间序列转换器，我在上一篇文章[中描述了如何使用 PyTorch](/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e) 制作用于时间序列预测的转换器。

这篇文章的结构如下:首先，我将简要描述 PyTorch 时序转换器需要什么输入。然后，我将向您展示当您不知道解码器输入值时，如何使用模型进行推理。最后，我将指出所示方法的一些缺点。

[](/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e) [## 如何制作用于时间序列预测的 PyTorch 转换器

### 这篇文章将向你展示如何一步一步地将时序转换器架构图转换成 PyTorch 代码。

towardsdatascience.com](/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e) 

# 时序转换器模型需要什么输入？

如果你没有读过我的文章“如何为时间序列预测制作 PyTorch 转换器”，让我们先简单回顾一下时间序列转换器需要什么输入。关于更详细的演示，请参见上面提到的帖子。请注意，术语`trg`和`tgt`有时会在这篇文章和另一篇文章中互换使用。

变压器模型需要以下输入:

`src`编码器所使用的。`src`的形状必须是【批量大小， *n* ，输入特征的数量】或 *n* ，批量大小，输入特征的数量(取决于`batch_first`构造函数参数的值)，其中 *n* 是输入序列中数据点的数量。例如，如果你正在预测每小时的电价，并且你想根据上周的数据进行预测，那么 n=168。

`tgt`是变压器需要的另一个输入。它被解码器使用。`tgt`由`src`中输入序列的最后一个值和目标序列除最后一个值以外的所有值组成。换句话说，它将具有[批量大小， *m* ，预测变量数]或[ *m* ，批量大小，预测变量数]的形状，其中 *m* 是预测范围。继续电价预测的例子，如果你想提前 48 小时预测电价，那么 *m=48。*

此外，编码器和解码器需要所谓的掩模。读者可以参考上面提到的关于屏蔽的介绍。

在我们继续之前，另一件重要的事情是，我们在这篇博客文章中使用的特定时序转换器实现总是输出形状为[批量大小， *m* ，预测变量数]或[ *m，*，批量大小，预测变量数]，【T22 的张量，即模型输出序列的长度由在 `tgt` *张量中给解码器的输入序列的长度决定。*

因此，如果`tgt`具有形状【72，批量大小，1】，这意味着`tgt`中序列的长度是 72，因此模型也将输出 72 的序列。

[](/multi-step-time-series-forecasting-with-xgboost-65d6820bec39) [## XGBoost 多步时间序列预测

### 本文展示了如何使用 XGBoost 生成多步时间序列预测和 24 小时电价预测…

towardsdatascience.com](/multi-step-time-series-forecasting-with-xgboost-65d6820bec39) 

# 如何使用时间序列转换器进行推理

好了，准备工作已经就绪，现在让我们考虑一下为什么会有一篇关于如何使用转换器进行时间序列预测的博文:

在训练期间，直接产生`tgt`，因为我们知道目标序列的值。然而，在推理过程中(例如在生产环境中)，我们在进行预测时当然不知道目标序列的值——否则我们就不需要首先进行预测。因此，我们需要找到一种方法来产生一个合理的`tgt`，它可以在推理过程中用作模型的输入。

现在我们知道了时间序列转换器需要什么输入，以及为什么我们需要以某种方式生成`tgt`，让我们看看实际上如何做。在接下来的内容中，请记住，总体目的是生成一个`tgt`张量，该张量一旦生成，就可以用作模型的输入来进行预测。

举一个简单的例子来说明，假设在推断时间 *t* ，我们想要根据序列的 5 个最近观察值来预测序列的下 3 个值。

下面是`src`的样子:

`src = [xt-4, xt-3, xt-2, xt-1, xt]`

其中`x`表示我们正在处理的系列，例如电价。

目标是预测`tgt_y`，它将是:

`tgt_y = [xt+1, xt+2, xt+3]`

因此，我们的`tgt`，模型需要它作为输入，以便对`tgt_y`进行预测，应该是:

`tgt = [xt, xt+1, xt+2]`

我们知道`xt`的值，但不知道`xt+1`和`xt+2`的值，所以我们需要以某种方式估计这些值。在本文中，我们将首先预测`xt+1`，然后将此预测添加到`tgt`中，这样`tgt = [xt, xt+1]`将使用此`tgt`预测`xt+2`，然后将此预测添加到`tgt`中，这样`tgt = [xt, xt+1, xt+2]`，最后使用此`tgt`生成最终预测。

下面的函数是您在 PyTorch 中使用时序转换器模型运行推理所需的代码。该函数根据上述方法生成预测。您传入一个 Transformer 模型和`src`以及 docstring 中描述的一些其他参数。然后，该函数迭代生成`tgt`，并基于由时间 *t* 的最后已知观测值和剩余 *m-1* 数据点的估计值组成的`tgt`生成最终预测。

该函数设计用于验证或测试循环中。您可以调用推理函数，而不是调用模型来生成预测。下面是如何使用它的一个简单示例:

请注意，您不能照原样使用该脚本。这仅仅是一个展示整体思想的例子，并不意味着你可以复制粘贴并期望工作。例如，在让脚本工作之前，您需要实例化模型和数据加载器。在这篇博文的 GitHub repo 中，参见文件 *sandbox.py* 中关于如何做的例子。如果你以前从未训练、验证和测试过 PyTorch 神经网络，我建议你看看 PyTorch 的初级教程。

# 使用时序转换器运行推理的所示方法的缺点

假设推理函数依赖于一个循环来迭代产生`tgt`，如果 *m* 很大，该函数会很慢，因为这会增加循环中的迭代次数。这是上述方法的主要缺点。我没有足够的想象力想出一个更有效的方法，但是如果你有任何想法，我很乐意在评论区听到你的意见。也欢迎你直接参与回购。

假设推理函数每批调用模型 *m-1* 次，您可能需要警惕一些增加调用模型的计算时间的事情，例如使用具有许多参数的模型或使用大的 *n.* 此外，您拥有的批越多，推理函数将被调用的次数就越多，总的训练或测试脚本运行的时间就越长。

使用时序转换器运行推理的代码以及 PyTorch 转换器实现可以在以下报告中找到:

[](https://github.com/KasperGroesLudvigsen/influenza_transformer) [## GitHub-KasperGroesLudvigsen/influence _ Transformer:py torch 实现的变压器模型…

### PyTorch 实现的变压器模型用于“时间序列预测的深度变压器模型:流感…

github.com](https://github.com/KasperGroesLudvigsen/influenza_transformer) 

就是这样！我希望你喜欢这篇文章🤞

请留下评论让我知道你的想法。

关注更多与[时间序列预测](/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e)、[绿色软件工程](https://kaspergroesludvigsen.medium.com/the-10-most-energy-efficient-programming-languages-6a4165126670)和数据科学[环境影响](/8-podcast-episodes-on-the-climate-impact-of-machine-learning-54f1c19f52d)相关的帖子🍀

请随时在 LinkedIn 上与我联系。