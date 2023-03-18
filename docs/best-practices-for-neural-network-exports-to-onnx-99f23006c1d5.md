# 向 ONNX 导出神经网络的最佳实践

> 原文：<https://towardsdatascience.com/best-practices-for-neural-network-exports-to-onnx-99f23006c1d5>

![](img/f84f9675127d45290db491188fa54247.png)

Artem Sapegin 在 [Unsplash](https://unsplash.com/) 上拍摄的照片。

> ONNX 是一种开放格式，用于表示机器学习模型。ONNX 定义了一组通用的运算符——机器学习和深度学习模型的构建块——和一种通用的文件格式，使 AI 开发人员能够将模型与各种框架、工具、运行时和编译器一起使用。
> 
> — [onnx.ai](https://github.com/Q-AMeLiA/documents/blob/main/onnx.ai)

# 为什么要导出到 ONNX？

将您的模型导出到 ONNX 有助于您将(经过训练的)模型从项目的其余部分中分离出来。此外，导出还避免了对 python 解释器、框架版本、使用的库等环境的依赖。导出的 ONNX-model 可以存储模型的架构和参数。这意味着向您的同事发送一个文件来交换您的模型就足够了。

# 出口

我们的经验表明，出口 PyTorch 模型更容易。如果可能的话，选择 PyTorch 源，并使用内置的`torch.onnx`模块进行转换。或者，您可以使用较新的独立`onnx` python 包(在下面的代码示例中，只需用`onnx`替换`torch.onnx`)。

## 来自 PyTorch

PyTorch 模型只能以编程方式导出:

请注意，PyTorch 在运行时计算计算图，因此转换引擎需要一批正确的形状(数据在大多数情况下可以是随机的)，它将通过网络来理解架构。`torch.onnx`使用`torch.jit.trace`找到您的数据通过网络的路径。

*最佳实践:*

*   小心`TracerWarnings`。这可能表明追踪器无法跟踪您的批次。
*   如果您在运行时做出路由决策，请确保使用一个批处理/配置来处理您尝试导出的所有路由。
*   如果你需要做路线决定，你应该在张量上做。tracer 找不到 Pythons 的默认类型。
*   如果您正在使用 torchhub 模型，请检查它们是否提供了一个`exportable`参数(或类似的)来替换不兼容的操作。

## 来自 TensorFlow (1/2/lite/js/tf。Keras)

我们推荐微软`[tf2onnx](https://github.com/onnx/tensorflow-onnx)`包用于 TensorFlow 模型的转换。在 ONNX 导出之前，必须将模型存储为 TensorFlows 支持的文件格式之一。支持的格式包括`saved model`、`checkpoint`、`graphdef`或`tflite`。

将*保存的模型*文件导出到 ONNX:

```
python -m tf2onnx.convert --saved-model tensorflow-model-path --output model.onnx
```

这是为 *tflite* 所做的事情(或者使用 [*tflite2onnx*](https://pypi.org/project/tflite2onnx/) ):

```
python -m tf2onnx.convert --opset 13 --tflite tflite--file --output model.onnx
```

对于其他格式，您需要提供输入和输出张量的名称。tf2onnx 将使用它们来跟踪网络。提供错误或不完整的标签列表可能会导致导出损坏。如果不知道模型的输入和输出节点名，可以使用[summary _ graph](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms)tensor flow 实用程序。以下是安装和使用它的方法:

或者，从显著位置检查源项目或询问原作者。对于下面的例子，我们假设有两个名为`input0:0,input1:0`的张量流输入和一个名为`output0:0`的输出。

对于*检查点*格式:

```
python -m tf2onnx.convert --checkpoint tensorflow-model-meta-file-path --output model.onnx --inputs input0:0,input1:0 --outputs output0:0
```

对于 *graphdef* 格式:

```
python -m tf2onnx.convert --graphdef tensorflow-model-graphdef-file --output model.onnx --inputs input0:0,input1:0 --outputs output0:0
```

*注意:*导出供重用的模型(如 TensorFlow Hub 模型)不可使用`summarize_graphs`进行分析，可能根本无法导出。

# 潜在的出口障碍

ONNX 协议或所使用的转换器可能不支持源模型的所有操作。

*可能的解决方案:*

*   检查是否有更新的[操作集版本](https://github.com/onnx/onnx/blob/master/docs/Operators.md) (opset)支持您的相关操作。
*   检查不支持的操作是否可以被支持的操作替换，例如，通常用`ReLU`替换`Hardswish`或`SiLU`激活(从 opset 11 开始)。
*   如果您的转换器没有映射该操作，但 ONNX 协议支持该操作，请实现映射或尝试不同的转换器。
*   如果 ONNX 协议不支持该操作，请尝试使用支持的操作重写您的操作，或者使用支持的操作实现映射。另外，考虑向 ONNX 提交一份 [PR。](https://github.com/onnx/onnx/blob/master/docs/AddNewOp.md)

# 验证导出的模型

我们建议使用 PyTorch 加载模型，并使用内置的验证引擎。

该代码块将只验证模式。这并不保证您的架构是完整的，也不保证所有的参数都被(正确地)导出。因此，我们建议您在几个样本上运行推理，并将它们与您原始框架的推理进行比较。请注意，由于导出过程和潜在的不同执行框架，可能会略有不同。要使用推理，请确保安装带有`pip`的`onnxruntime` python 包或您的 python 包管理器。

此外，您可以通过使用像 [Netron](https://netron.app/) 这样的外部工具可视化导出的模型来运行健全性检查。Netron 还允许您浏览存储的参数。请记住，转换器可能会根据需要减少或扩大操作，因此可能会触及原始架构。*注意:* Netron 在加载大型模型时可能会有问题。

# 至理名言

无论您是否选择 ONNX，都要考虑发布您训练好的模型，尤其是如果您正在准备一份科学出版物。这有助于他人轻松复制你的发现，也为他们的项目提供了一个良好的开端。

*这项工作得到了德国巴登-符腾堡州(MWK)科学、研究和艺术部的部分资助，资助项目为 32–7545.20/45/1*[*机器学习应用的质量保证(Q-AMeLiA)*](https://q-amelia.in.hs-furtwangen.de/) *。*