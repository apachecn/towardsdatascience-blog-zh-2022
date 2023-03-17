# PyTorch 定制数据集在 5 分钟内帮助詹姆斯·邦德

> 原文：<https://towardsdatascience.com/pytorch-custom-datasets-in-5-minutes-to-help-james-bond-c7e454aa1aab>

## 如何利用自定义数据集来提高我们最喜爱的英雄的机会

![](img/23ed67a6f362b4e608a534c0a8b4e8de.png)

马塞尔·埃伯勒在 [Unsplash](https://unsplash.com/s/photos/007?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

> 最近我看了詹姆斯·邦德的新电影《T2》。因为詹姆斯·邦德在电影中有很多敌人，所以我一直很担心。在一个场景中，他正驾驶着他的阿斯顿·马丁 DB 5 穿过一片美丽的雪景，突然几辆吉普车不知从哪里冒了出来，瞄准了他。他所有敌人的吉普车都是路虎的。如果詹姆斯·邦德事先简单地训练了一个神经网络来探测敌人的汽车会怎么样？电影中有一个简单的逻辑:好人开阿斯顿马丁，坏人开路虎，其他人都开其他品牌(好吧詹姆斯·邦德也开过一次经典的路虎和一次丰田，但我们想在这一点上保持简单……)。q 也有同样的想法，但是无法正确地组织数据。让我们帮助他，创建一个自定义数据集类。

![](img/604b8900cf5dab6a5ed69c0e03571598.png)

照片由[雷·哈灵顿](https://unsplash.com/@raymondo600?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/aston-martin?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

在下面的文章中，我们将经历几个步骤:

1.  我们将使用 PyTorch 创建一个自定义数据集。
2.  我们将利用迁移学习的优势，我们将调整一个结果。
3.  最后，我们将在 PyTorch 中有一个多类分类问题的蓝图。

您将学到的内容:

1.  几乎所有的 CNN 教程都从 MNIST 或 CIFAR10 数据集开始。不要误解我，这些数据集是了解 CNN 的很好的资源。但是在现实世界中，数据的组织方式不允许您用一行程序将其加载到 Python 中。因此，我们将进一步了解自定义数据集。
2.  我们还将进一步探讨如何利用迁移学习在有限的资源下取得非常好的效果。

## 数据集

我们将使用斯坦福大学的汽车数据集。它包含 196 个汽车类别的 16，185 幅图像。你可以直接从他们的网站下载所有图片。让我们通过 Python 获取文件:

正如你所看到的，这些图片是有编号的，除了图片本身，不包含任何其他信息。所以我们需要某种密钥文件来添加关于单个图像的信息。幸运的是，斯坦福大学提供了这样一个关键文件。否则，我们不得不自己给每张图片贴标签。我们可以得到*。mat* 文件用 Python 也一样:

现在，我们将图像复制到我们的图像文件夹，并存储元信息。下一步，我们将结合这两个来源。mat-file 包含一个名为 *annotations* 的键，它保存了关于相应文件、所有图像的边界框和类名的信息。我们将忽略由每个数组中的前四个元素定义的边界框。第五个元素包含我们需要的汽车名称 id。让我们创建一个完成这项工作的函数:

因此，我们会收到一个以图像文件名为关键字的字典，其中包含以下值:品牌和汽车品牌名称、每个品牌的数字表示(例如，奥迪为 3)以及我们感兴趣的 3 个类别的数字表示。 *brand_translation_dict* 帮助我们跟踪哪个数字代表哪个品牌。

幸运的是，奥迪 R8 跑车在跑车世界里是众所周知的，我们可以确认我们的形象标签是正确的。我们还看到元组中的最后一个整数是 2，代表类别*其他*，这也是正确的。

现在我们可以创建 PyTorch 数据集类了。PyTorch 数据集是 [*表示数据集*](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) 的抽象类。它允许我们向数据加载器提供正确的信息，数据加载器用于将训练样本发送到我们的神经网络。必要的信息是数据集的长度和神经网络的输入结构。在我们的例子中，我们希望提取图像作为数字表示和每个图像的标签:

让我们一行一行地检查代码。在 init-part 中，我们指定图像的路径。我们还添加了一个转换操作。这将用于预处理图像(例如，调整大小，将 numpy 数组转换为张量等)。我们还添加了我们的 *translation_dict* 来从文件名中提取正确的标签。

下一部分是关于数据集的长度。这个函数将返回数据集的大小。最后一部分是 getitem 函数。这个函数帮助我们组织数据集的索引，允许数据加载器使用数据集[x]获取第 x 个样本。第一步，我们定义路径和文件夹。然后，我们打开索引图像并对其应用变换操作。在我们通过访问我们的 *translation_dict* 加载两个定义好的标签之后。最后，我们创建了一个包含三个键的字典，调用这个函数将返回这三个键。让我们试一试:

不出所料，我们得到了一个包含关键字 *image* 、 *label_brand* 和 *label_james_bond* 的字典。图像张量的形状是[16，3，224，224]，其中 16 是批量大小，3 是通道的数量，以及[224，224]每个图像的大小。标签张量的大小为[16]，因为每个图像有一个标签。

## 模型

最后，我们可以定义一个模型。这项任务并不容易，因为一些模型和汽车看起来非常相似。因此，我们将使用预训练网络并对其进行微调:

现在我们可以创建一个非常基本的训练循环。我们将以 1e-4 的学习速率训练网络 5 个时期:

训练结束后，我们将分别计算网络的总精度和每个类别的精度:

阿斯顿·马丁的准确率约为 87%，路虎的准确率约为 88%。其他类别的正确识别率为 99%。对于一个非常简单的结构来说一点也不差。有很大的改进空间:

*   使用其他变换操作，例如随机翻转、随机裁剪
*   创建一个更复杂的自定义头，而不是只添加一个线性层
*   使用更复杂的训练策略，例如冻结主干层，使用逐层区别学习率，使用其他优化器
*   尝试其他一些预先训练好的神经网络
*   组装

不管怎样，Q 是个天才，我相信他会让我们的网络变得更好。在我们结束之前，让我们看一下混淆矩阵。

对詹姆斯·邦德来说最重要的是:我们在阿斯顿·马丁和路虎之间没有分类失误。然而，8 辆阿斯顿马丁和 4 辆路虎被错误地归类为*其他*。

## 结论

PyTorch 自定义数据集类是一个强大的工具，可以为非完美结构化的数据创建管道。我们在 *getitem* 部分使用了自己的翻译词典。我们能够创建和加载我们自己的定制培训标签。甚至有更多的可能性。我们可以创建修改后的文件列表(例如，只过滤特定的品牌)并将它们添加为条件。我们也可以在 *getitem* 部分中集成 *if else* 语句。如果您希望将数据集类用于生产中的未标记数据，这将非常有用。例如，通过添加一个 *test == true* 语句，我们可以返回一个没有标签的字典。PyTorch 自定义数据集类提供了无限的可能性来单独构建您的训练数据。感谢阅读！

## 进一步阅读

**笔记本:**[https://jovian . ai/droste-benedikt/01-article-py torch-custom-datasets](https://drive.google.com/file/d/14AuCCmXW-Tjk02NCQONrHouRb2gVEiEw/view?usp=sharing)
**FastAI CNN 简介:**[https://colab . research . Google . com/github/FastAI/fastbook/blob/master/13 _ convolutions . ipynb](https://colab.research.google.com/github/fastai/fastbook/blob/master/13_convolutions.ipynb)
**py torch 定制数据集简介:** [https://py torch . org 澳大利亚悉尼。2013 年 12 月 8 日。](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) [【pdf】](https://ai.stanford.edu/~jkrause/papers/3drr13.pdf)[【BibTex】](https://ai.stanford.edu/~jkrause/papers/3drr13.bib)[【幻灯片】](https://ai.stanford.edu/~jkrause/papers/3drr_talk.pdf)

[如果您喜欢中级数据科学，并且还没有注册，请随时使用我的推荐链接加入社区。](https://medium.com/@droste.benedikt/membership)