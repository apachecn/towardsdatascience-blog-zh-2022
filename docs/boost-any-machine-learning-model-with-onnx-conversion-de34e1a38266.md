# 通过 ONNX 转换提升任何机器学习模型

> 原文：<https://towardsdatascience.com/boost-any-machine-learning-model-with-onnx-conversion-de34e1a38266>

## 关于如何将模型转换为 ONNX 的简单理解

**开放神经网络交换** (ONNX)是一个开源生态系统，旨在**跨各种平台标准化和优化**人工智能模型。

它是一种**机器可读格式**，可用于在**不同软件应用程序和框架**(例如 TensorFlow、PyTorch 等)之间交换信息。).

![](img/a0f2507be3a5150e9fe8ff67fd355a2a.png)

照片由 [Sammy Wong](https://unsplash.com/@vr2ysl?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/running-animal?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

历史上，ONNX 格式被命名为**太妃糖**，由脸书的 PyTorch 团队开发。该框架于 2017 年底发布，由微软和脸书共同撰写。

从那以后，ONNX 格式已经被其他几家公司支持，包括英特尔、AMD 和 IBM。

我个人已经和`onnxruntime`一起工作了几个月，我很高兴它得到了大量框架和项目的支持。

**互操作性**作为 ONNX 的关键特性，是该工具的最佳资产之一，并且使**包含在所有 ML 项目**中变得非常有趣。

> “ONNX 运行时是一个跨平台的推理和训练机器学习加速器”——[ONNX 运行时](https://github.com/microsoft/onnxruntime)

在这篇文章中，我将向您展示将您的模型转换为 ONNX 需要遵循和理解的步骤。我还将列出所有可用的库，根据您使用的框架，这些库可用于将您的模型转换为 ONNX。

此外，我将包括一个将 PyTorch LSTM 模型转换为 ONNX 的例子。

# 👣转换步骤

所有的框架都有自己的方式将它们的模型转换成 ONNX。但是您需要遵循一些常见的步骤。

让我们看看将您的模型转换为 ONNX 需要遵循哪些步骤。

## -🏋️训练你的模型

很明显，您需要一个经过训练的模型来将其转换为 ONNX，但是您需要注意模型架构和模型权重的可用性。

如果您只保存模型权重，您将无法将其转换为 ONNX，因为模型架构是将您的模型转换为 ONNX 所必需的，并且非常重要。

有了模型架构，ONNX 就能够跟踪模型的不同层，并将其转换为图形(也称为*中间表示*)。

模型权重是用于计算模型输出的不同层的权重。因此，它们对于成功转换您的模型同样重要。

## -📝输入名称和输出名称

您将需要定义模型的输入名称和输出名称。这些元数据用于描述模型的输入和输出。

## - 🧊输入样本

如前所述，ONNX 将跟踪模型的不同层，以便创建这些层的图形。

在追踪图层时，ONNX 还需要一个输入样本来了解模型如何工作以及使用什么运算符来计算输出。

所选样本将作为模型第一层的输入，并用于定义模型的输入形状。

## -🤸‍♀️动态轴

然后，ONNX 需要知道模型的动态轴。在转换过程中的大多数时间，您将使用批量大小 1。

但是如果您想要一个可以接受一批 N 个样本的模型，您将需要定义模型的动态轴来接受一批 N 个样本。

*例如，这样导出的模型将接受大小为[batch_size，1，224，224]的输入，其中“batch_size”可以是 1 到 n 之间的任何值*

## -🔄转换评估

最后，您将需要评估转换后的模型，以确保它是一个可持续的 ONNX 模型，并且按照预期工作。评估转换后的模型有两个独立的步骤。

**第一步**是使用 ONNX 的 API 来检查模型的有效性。这是通过调用`onnx.checker.check_model`函数来完成的。这将验证模型的结构，并确认模型是否具有有效的 ONNX 方案。

通过检查节点的输入和输出来评估模型中的每个节点。

**第二步**是将转换模型的输出与原始模型的输出进行比较。这通过用`numpy.testing.assert_allclose`功能比较两个输出来完成。

该功能将比较两个输出，如果两个输出不相等，将根据`rtol`和`atol`参数产生错误。

常用 *1e-03* 的`rtol`和 *1e-05* 的`atol`进行比较，其中`rtol`代表相对公差，`atol`代表绝对公差。

# 🔧将模型转换为 ONNX 的工具

对于每个框架，都有不同的工具将您的模型转换成 ONNX。我们将列出每个框架可用的工具。

*列表可能会随着时间的推移而改变，所以如果您发现了没有列出的工具，请在评论中告诉我。*

*   **PyTorch** : `[torch.onnx](https://pytorch.org/docs/stable/onnx.html)`
*   **张量流** : `[onnx/tensorflow-onnx](https://github.com/onnx/tensorflow-onnx)`
*   **喀拉斯** : `[onnx/keras-onnx](https://github.com/onnx/keras-onnx)`
*   **Scikit-learn**:T3
*   **咖啡馆** : `[htshinichi/caffe-onnx](https://github.com/htshinichi/caffe-onnx)`
*   **MXNet** : `[mx2onnx](https://mxnet.apache.org/versions/1.9.0/api/python/docs/tutorials/deploy/export/onnx.html)`
*   **py torch——闪电** : `[onnx-pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/common/production_inference.html#convert-to-onnx)`
*   **变形金刚**:🤗`[Exporting Transformers models](https://huggingface.co/docs/transformers/serialization#onnx)`
*   **PaddlePaddle** : `[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)`

现在您已经知道了转换是如何工作的，并且拥有了将您的模型转换为 ONNX 的所有工具，您已经准备好进行实验，看看它是否对您的 ML 项目有所帮助。我打赌你会发现它很有用！

您可以查看我在下一节提供的完整 PyTorch 示例，或者直接阅读本文的结论。

# 🧪完整 PyTorch 示例

我最近需要将 PyTorch 模型转换为 ONNX。该模型是一个简单的 LSTM 预测模型，用于预测输入序列的下一个值。

该模型来自我参与的一个更大的开源项目，该项目旨在使训练和服务预测模型自动化。那个项目叫做[让我们变富](https://github.com/ChainYo/make-us-rich)。

需要注意的是，我使用 **PyTorch Lightning** 来训练模型。这个框架是 PyTorch 的一个包装器，允许你提高你的训练过程。

如果你想了解更多关于 PyTorch Lightning 的知识，可以查看 [PyTorch Lightning 文档](https://pytorch-lightning.readthedocs.io/en/latest/)。

首先，您需要下载我们将用于训练和转换模型的数据示例。您可以用这个简单的命令下载它:

```
$ wget https://raw.githubusercontent.com/ChainYo/make-us-rich/master/example.csv
```

该文件包含 365 天内每小时的 BTC(比特币)值。我们将使用这些数据创建序列来训练我们的 LSTM 模型。

为了训练一个 LSTM 模型，输入数据应该被转换为数据序列。我们将使用这些预处理函数将数据转换为模型使用的序列:

创建序列所需的所有预处理功能

我们将使用这些函数来获得我们的`train_sequences`、`val_sequences`和`test_sequences`。

从数据中创建所需的序列

既然我们已经准备好了模型要使用的序列，那么让我们详细看看模型架构、**数据集**和**数据加载器**类。

处理模型使用的数据的数据集和数据加载器类

这两个类用于加载将用于训练模型的数据。我不想描述 PyTorch 和 PyTorch Lightning 中加载数据的所有细节，因为这不是本教程的重点。

*如果你需要更多关于* `*Dataset*` *和* `*DataLoader*` *类的解释，这里有一篇很棒的* [*文章*](/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00) *关于它。*

这是我们将要使用的 LSTM 模型架构:

示例的模型架构

既然我们已经定义了模型架构和数据加载方式，我们就可以开始训练模型了。

以下是我使用的训练循环:

用 PyTorch Lightning 创建的训练循环

经过训练，我们现在有了一个模型检查点，我们希望将其转换为 ONNX。我们将从检查点文件加载模型，通过数据加载器加载数据样本，并将模型转换为 ONNX。

下面是转换的代码:

您现在应该在当前目录中有一个名为`model.onnx`的文件。

最后一步是评估转换后的模型，以确保它是一个可持续的 ONNX 模型，并按预期工作。

如果一切顺利，您应该会看到以下输出:

```
 🎉 ONNX model is valid. 🎉
```

您已成功将您的模型转换为 ONNX。现在，您可以在任何支持 ONNX 的框架中和任何机器上使用转换后的模型。

# 结论

在这篇文章中，我们介绍了如何将你的模型转换成 ONNX，以及如何评估转换后的模型。另外，我们已经看到了一个预测 LSTM PyTorch 模型的例子。

我希望这篇文章能帮助你理解一个模型到 ONNX 的转换过程。我也希望你对 ONNX 的个人体验能帮助你提升你的 ML 项目，就像我帮助我提升我自己的项目一样。

ONNX 框架还包括许多其他的特性，这些特性在这篇文章中没有提到，比如为你的模型和硬件优化训练和推理的能力。这可以在下一篇文章中讨论，让我知道你是否感兴趣。

如果您有任何问题或面临任何问题，请随时联系我。我很乐意帮助你。