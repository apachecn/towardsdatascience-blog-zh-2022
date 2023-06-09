# 物体检测的基础:YOLO，SSD，R-CNN

> 原文：<https://towardsdatascience.com/the-basics-of-object-detection-yolo-ssd-r-cnn-6def60f51c0b>

## 概述对象检测的工作原理，以及从哪里开始

![](img/4c89f1c3ac0aefaea09f11070318f5c7.png)

亨利·迪克在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

**注意:**这里假设你对卷积神经网络有一个大致的了解。如果你需要复习，这篇 [IBM 帖子](https://www.ibm.com/cloud/learn/convolutional-neural-networks)非常好。

你知道卷积神经网络(或 CNN)如何检测和分类图像。但是你可以扩展 CNN 来探测图像中的物体。

> 物体检测和图像识别有什么不同？

你有 4 头牛的图像。图像识别将图像作为一个整体来检测。上面说图像被归类为奶牛。

但是，图像识别无法分辨出牛在图像中的位置。此外，它将无法分辨出有 4 头牛而不是 1 头。

另一方面，物体检测并不意味着计数。还有其他计算机视觉技术，如密度估计和提取用于计数的斑块。对象检测只是检测同一幅图像中是否有多个对象。

> 为什么我需要学习这个？这在现实世界中有什么用？

有很多骗子。有些人用假的银行本票或美钞混日子。图像检测只能说假复制品看起来是支票还是美钞。对象检测可以帮助搜索有助于确定支票或账单是否是伪造的单个组件。

如果你对医学着迷，那么物体检测可以识别 X 射线中的阴影或异常。即使它不能诊断疾病，物体检测也可以协助医生发现肺部组织的异常。医生可以在肺癌变得恶性之前发现它。

如果你对无人机和自动驾驶汽车着迷，那么对象检测对于检测视频流中的障碍和边界非常有用。它使汽车保持在车道内，避免事故。

> 听起来棒极了！但是物体检测是如何工作的呢？

图像已经包含了如此多的信息。CNN 旨在减少图像数据并保留重要信息。这可以通过不同的卷积层、池层和密集层来实现。

在这些层的末端，图像被缩小到足以进行预测。对于图像识别，这是一个类似 Softmax 的激活函数，用于对图像进行分类。对于对象检测，它有一种算法来预测检测到的对象周围的边界框，然后对对象进行分类。

> 有哪些算法？

有许多对象检测算法，但我们将涵盖三个主要的。

*   YOLO——你只能看一次
*   SSD —单次检测器
*   r-CNN——基于区域的卷积神经网络

YOLO 是最简单的对象检测架构。在对象通过 CNN 后，它通过基于网格的方法预测边界框。它将每个图像分成一个 SxS 网格，每个网格预测包含任何对象的 N 个盒子。从这些 SxSxN 盒子中，它为每个类分类每个盒子，并挑选最高的类概率。

SSD 类似于 YOLO，但是使用每个卷积层的特征图(每个滤波器/层的输出)来预测边界框。在整合所有的特征图之后，它对它们运行 3×3 卷积核来预测包围盒和分类概率。SSD 是一个算法家族，流行的选择是 RetinaNet。

R-CNN 采用不同的方法，对构成识别的边界框/区域中的对象的像素进行分类。它使用一个神经网络来建议要检测的对象的潜在位置(基于区域的网络)。它使用第二个神经网络来基于这些建议的区域分类和检测对象。第二个神经网络添加了一个像素掩码，为需要分类的对象提供形状。

注意，一些研究者对这些算法有不同的语义。一些人认为 YOLO 是 SSD 家族的一部分，因为它们都只处理一次图像。一些人将 YOLO 从固态硬盘家族中分离出来。有人说像 R-CNN 这样的基于区域的网络是与对象检测方法相对的*实例分割*方法。

> 我应该使用哪一个？

这在很大程度上取决于您的数据、目标和计算使用情况。

YOLO 速度极快，使用的处理内存很少。虽然 YOLOv1 不如 SSD 准确，但 YOLOv3 和 YOLOv5 在准确性和速度上都超过了 SSD。此外，YOLO 只能预测每个网格 1 个类别。如果网格中有多个对象，YOLO 失败。最后，YOLO 努力探测小物体。

SSD 可以处理各种规模的对象。它利用来自所有卷积层的特征图，并且每一层在不同的尺度上操作。它的计算量也不大。然而，固态硬盘也很难检测到小物体。此外，如果 SSD 包含更多的卷积层，它会变得更慢。

R-CNN 是最准确的。然而，它在计算上是昂贵的。它需要大量的存储和处理能力来进行检测。它也比 YOLO 和固态硬盘慢。

两者都有利弊。如果准确性不是一个大问题，YOLO 是最好的选择。如果你的图像是黑白的，或者在清晰的背景上有容易识别的物体，YOLO 会在这些场景上非常准确。如果您有复杂的图像，并关心准确性(如从 X 射线检测癌症)，那么 R-CNN 将是最适合的。

> 我必须从头开始重写这些算法吗？

不。这些都是开源和预先训练。它们都可以在 [OpenCV](https://opencv.org/) 上获得。OpenCV 文档和样本适用于 [YOLO、](https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html)SSD 和 [Mask R-CNN](https://pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/) 。

> 等等，如果这些模型是预先训练的，那么我如何将它们用于特定于我的用例的新数据？

您需要创建两个数据集

*   自定义训练数据集，包括图像和每个图像中对象的标签/注释
*   用于模型预测的自定义测试数据集，以及用于验证模型预测准确性的标签/注释

强烈建议您创建自定义验证数据集。

> 这听起来工作量很大！我该如何着手做这件事？

您可以使用工具来添加注释。完成后，您可以将数据集导出为模型的任何格式。

我最喜欢的工具是 Label Studio。您可以为计算机视觉的图像对象、自然语言处理的文本、转录的音频等添加注释。我只用它来注释对象，它非常棒。

您可以以各种格式输出数据集。CSV，JSON，XML，Pascal VOC XML 等。甚至还有专门针对 YOLO 的格式。

你可以阅读更多关于如何开始使用 Label Studio 的内容。设置起来超级简单。您通过命令`pip install -U label-studio`下载 label-studio，然后通过`label-studio`启动它。用户界面非常直观，可以随时随地解决问题。

<https://labelstud.io/guide/index.html#Quick-start>  

> 太棒了。但是我如何可视化这些对象的注释呢？

用 python 包`matplotlib`！

如果你需要帮助来解决这个问题，其他一些来自数据科学家的贡献者写了他们自己的注释图像的方法。见下文。

</how-to-use-matplotlib-for-plotting-samples-from-an-object-detection-dataset-5877fe76496d>  

现在，您拥有了在自己的自定义数据集上训练现有对象检测模型所需的工具。

感谢阅读！如果你想阅读更多我的作品，请查看我的目录。

如果你不是一个中等收入的会员，但对订阅《走向数据科学》感兴趣，只是为了阅读类似的教程和文章，[单击此处](https://hd2zm.medium.com/membership)注册成为会员。注册这个链接意味着我可以通过向你推荐 Medium 来获得报酬。

# 参考

[计算机视觉的实用机器学习— O'Reilly Media](https://www.amazon.com/Practical-Machine-Learning-Computer-Vision/dp/1098102363) (特别是第四章:目标检测和图像分割)

[R-CNN，快速 R-CNN，更快 R-CNN，YOLO——目标检测算法](/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e)

[用 YOLO5 进行自定义物体检测训练](http://Custom Object Detection Training using YOLOv5 In this blog post, we are fine tuning YOLOv5 models for custom object detection training and inference. Introduction…learnopencv.com)