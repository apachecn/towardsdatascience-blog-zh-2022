# 在 Pytorch 和 Tensorflow 中将图像转换为张量

> 原文：<https://towardsdatascience.com/convert-images-to-tensors-in-pytorch-and-tensorflow-f0ab01383a03>

## 学习以本机方式转换数据

![](img/9fc169f0bd974333234241a24b7b7fbd.png)

由 [Unsplash](https://unsplash.com/s/photos/machine-learning?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的 [Clarisse Croset](https://unsplash.com/@herfrenchness?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄的照片

如你所知，张量代表了你的机器学习项目的构建模块。它们是不可变的，只能被创建，不能被更新。处理张量是复杂的，因为它在很大程度上依赖于你使用的框架，并且没有关于如何编码它的标准化。

因此，您将经常陷入这样一种情况，即您必须执行自定义操作，例如从 Pytorch 转换到 Tensorflow 或改变张量的维度。

在本教程中，您将学习:

*   在 Pytorch 和 Tensorflow 框架中将图像转换为张量
*   使用`torch.permute`改变 Pytorch 张量的维度顺序
*   使用`tf.transpose`改变张量流张量的维度顺序
*   将张量从 Pytorch 转换为 Tensorflow，反之亦然

让我们继续下一节，开始安装所有必要的 Python 包。

# 设置

强烈建议您在继续安装之前创建一个新的虚拟环境。

## Pytorch

运行以下命令来安装`torch`和`torchvision`软件包。

```
pip install torch torchvision
```

`torchvision`是一个基本的软件包，它提供了相当多的图像[变换功能](https://pytorch.org/vision/stable/transforms.html)，如调整大小和裁剪。

## Python 图像库

此外，您需要安装 Python 图像库(PIL ),它在加载您的图像时补充了`torchvision`。您可以按如下方式安装它:

```
pip install Pillow
```

## 张量流

从版本 2 开始，Tensorflow 的安装变得简单多了。使用以下命令安装它:

```
pip install tensorflow
```

完成安装后，继续下一节的实施。

# 将图像转换为张量流张量

在本节中，您将学习为 Pytorch 和 Tensorflow 框架实现图像到张量的转换代码。

供您参考，张量流中图像张量的典型轴顺序如下:

```
shape=(N, H, W, C)
```

*   `N` —批量大小(每批图像的数量)
*   `H` —图像的高度
*   `W` —图像的宽度
*   `C` —通道数(RGB 通常使用 3 个通道)

您可以轻松地加载图像文件(PNG、JPEG 等。)和下面的代码:

# 更改张量流张量维度

此外，Tensorflow 确实提供了一个名为`tf.transpose`的有用函数，可以根据`perm`的值改变张量的维数。默认情况下，`perm`设置为`[n-1…0]`，其中 n 代表维度的数量。

假设你想改变张量的形状

```
(224, 224, 3) -> (3, 224, 224)
```

简单地调用`tf.transpose`函数如下:

```
tensor = tf.transpose(tensor, perm=[2, 0, 1])
```

这在对源自 Pytorch 的模型(例如，从 Pytorch 到 ONNX 到 Tensorflow 的转换)执行推理时很方便，因为两种框架之间图像张量的标准结构不同。

# 将图像转换为 Pytorch 张量

另一方面，Pytorch 中图像张量的形状与张量流张量略有不同。而是基于以下`torch.Size`:

```
torch.Size([N, C, H, W])
```

*   `N` —批量大小(每批图像的数量)
*   `C` —通道数(RGB 通常使用 3 个通道)
*   `H` —图像的高度
*   `W` —图像的宽度

加载图像数据有多种方式，其中一种如下:

*   通过枕形包装装载
*   设置转换功能
*   对图像应用调整大小变换
*   设置张量转换函数
*   对图像应用张量转换函数

看看下面的转换脚本:

与 Tensorflow 使用术语扩展维度来添加新维度不同，Pytorch 基于挤压和取消挤压。挤压意味着你将通过截断来减少维度，而 unsqueeze 将为相应的张量增加一个新的维度。

# 更改 Pytorch 张量维度

此外，您可以通过`torch.permute`功能轻松地重新排列张量的维度。假设您想要按如下方式更改形状:

```
(3, 224, 224) -> (224, 224, 3)
```

调用`torch.permute`函数时，应传入以下输入:

```
torch.permute(tensor, (1, 2, 0))
```

# 在 Pytorch 和 Tensorflow 之间转换张量

张量转换的一个最简单的基本工作流程如下:

*   将张量(A)转换为 numpy 数组
*   将 numpy 数组转换为张量(B)

## Pytorch 到 Tensorflow

Pytorch 中的 Tensors 自带了一个名为`numpy()`的内置函数，该函数会将其转换为 numpy 数组。

```
py_tensor.numpy()
```

然后，简单地调用 [tf.convert_to_tensor()](https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor) 函数如下:

```
import tensorflow as tf...tf_tensors = tf.convert_to_tensor(py_tensor.numpy())
```

## 张量流向 Pytorch

另一方面，tensors 从 Tensorflow 到 Pytorch 的转换遵循相同的模式，因为它有自己的`numpy()`函数:

```
tf_tensors.numpy()
```

随后，您可以调用 [torch.from_numpy()](https://pytorch.org/docs/stable/generated/torch.from_numpy.html) 进行张量转换:

```
import torch...py_tensors = torch.from_numpy(tf_tensors.numpy())
```

# 结论

让我们回顾一下你今天所学的内容。

本文首先简要解释了张量和操作张量的困难。

接下来，它通过`pip install`进入安装过程。

它继续逐步指导将图像数据转换为 Tensorflow 张量，以及如何根据自己的需要更改张量的维度。

此外，还介绍了 Pytorch 框架的张量转换和相应的维数置换。

最后，它突出显示了一些代码片段，用于将 tensors 从 Tensorflow 转换为 Pytorch，反之亦然。

感谢你阅读这篇文章。祝你有美好的一天！

# 参考

1.  [tensor flow——张量介绍](https://www.tensorflow.org/guide/tensor)
2.  [Tensorflow — tf.io](https://www.tensorflow.org/api_docs/python/tf/io)
3.  [py torch—torch vision . transform](https://pytorch.org/vision/stable/transforms.html)