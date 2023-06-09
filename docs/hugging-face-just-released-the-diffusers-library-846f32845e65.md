# 拥抱脸刚刚发布了扩散器库

> 原文：<https://towardsdatascience.com/hugging-face-just-released-the-diffusers-library-846f32845e65>

## 使 DALL-E 2 和 Imagen 等扩散器型号比以往任何时候都更容易使用

![](img/f182510868ddac569b5d94604bb0aef3.png)

由作者编辑，背景照片由 [Michael Dziedzic](https://unsplash.com/@lazycreekimages) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

拥抱脸，*变形金刚*库的创建者，刚刚发布了一个全新的库，用于构建扩散器模型。如果你不确定扩散器模型是什么，就把它们想象成今年一些最受关注的人工智能模型背后不那么秘密的秘密。

你在互联网上看到的那些美得令人发狂的、有创意的、几乎人性化的艺术作品？它们是来自 OpenAI 的 DALL-E 2、Google 的 Imagen 和 Midjourney 的图像。所有这些服务都使用*扩散器模型*生成它们的图像。

![](img/263420b4b65e133db31f8d4b9080a37d.png)

一些 DALL-E 2 生成的图像[1]。

现在拥抱脸发布了一个专注于扩散器的开源库。有了它，我们只需几行代码就可以下载并生成图像。

新的*扩散器*库使得这些极其复杂的模型变得直观而简单。在本文中，我们将探索新的库作品，生成一些我们自己的图像，并看看它们与上面提到的最先进的模型相比如何。

如果您喜欢视频，可以观看本文的视频演示:

# 入门指南

首先，我们需要`pip install diffusers`并初始化扩散模型或管道(通常由预处理/编码步骤和扩散器组成)。我们将使用文本到图像的扩散管道:

现在，我们需要做的就是创建一个提示，并通过我们的`ldm`管道运行它。从拥抱脸的介绍性笔记本中获得灵感，我们将尝试生成一幅松鼠吃香蕉的画。

我们走吧。这简直太容易了。现在图像不像 DALL-E 2 那样令人印象深刻，但我们用五行代码做到了这一点，而且是免费的，在新库的第一个版本中。如果这不酷，我不知道什么是酷。

这是另一幅松鼠吃香蕉的画:

也许是现代艺术？

## 快速工程

自三大扩散模型(DALL-E 2、Imagen 和 Midjourney)发布以来，出现了一个有趣的趋势，即人们越来越关注所谓的*“即时工程”*。

快速工程师就是罐头上写的。字面上的“工程”的提示，以达到预期的结果。例如，许多人发现添加“在 4K”或“在 Unity 中渲染”可以增强三巨头生成的图像的真实感(尽管它们都不是 4K 分辨率的)。

如果我们用简单的扩散器模型做同样的尝试会发生什么？

每张图片都有这样或那样的怪异之处，这些香蕉的摆放位置当然也有问题。但是，你得给模特加分；一些松鼠的细节相当不错，看看图片 1 中香蕉的倒影。

摄影师和画家们，注意了，你们来了。

## 在罗马时

我目前住在罗马，这在盛夏是个糟糕的主意。尽管如此，比萨是首屈一指的，没有比罗马圆形大剧场更具标志性的建筑了，我想，太好了，如果一个意大利人在那个标志性建筑上吃比萨会是什么样子？

当然，我们并没有坐在竞技场的顶端，但是我很感激你的努力。斗兽场本身看起来很棒，尽管拱门之间的天空颜色不匹配。

除了奇怪的热狗手和小披萨，我们的意大利披萨看起来很棒。太阳镜的选择给他一种 90 年代父亲的感觉。

很难从这个单一的提示中做出任何有把握的判断，但我认为有趣的是，该模型没有生成任何女性吃披萨的图像，尽管约 51%的意大利人是女性[2]。该模型也没有生成非白人男性的图像——然而，我没有充分运行该模型来确定后者是否具有统计学意义。

这种模型和其他未来拥抱脸托管模型中的偏见的影响无疑将是图书馆现在和未来的一个重要焦点。

## Squirrelzilla

回到松鼠，试图生成更抽象的图像，如“一只巨大的松鼠摧毁了一座城市”，会导致混合的结果:

对我来说，这个模型似乎很难融合两个典型的不相关的概念，一只(巨大的)松鼠和一座城市。从同一个提示生成的这两个图像似乎强调了这种行为:

在这里，我们可以看到一个城市的天际线，或者在一个更常见的与松鼠相关的环境中看到一个类似松鼠的物体。在运行这些提示几次后，我发现它在两者之间切换，并且从未将两者合并在一起。

只是为了好玩，下面是 DALL-E 2 从提示`"a dramatic shot of a giant squirrel destroying a modern city"`中产生的内容:

![](img/edcbdc48b8bf934c8afaddc260cc2812.png)

由作者使用 OpenAI 的 DALL-E 2 生成。

这些都非常令人印象深刻，但正如已经提到的，我们不能指望这两个选项之间的可比性能，目前*。*

*这就是第一次看拥抱脸的最新图书馆。总的来说，我非常兴奋看到这个图书馆的发展。现在，最好的扩散器模型都被锁在紧闭的门后，我认为这个框架是一把钥匙，可以释放人工智能推动的创造力的一些令人敬畏的水平。*

*这并不是说这个框架即将取代 DALL-E 2、Imagen 或 Midjourney。事实并非如此，随着商业和开源产品之间的选择越来越多，世界变得越来越美好。*

*这些开源模型允许像你我这样的普通人获得深度学习的一些最新进展。当许多人自由尝试新技术时，很酷的事情就会发生。*

*我很期待这一天的到来。如果你有兴趣看到更多，我会定期在 YouTube 上发帖，并和其他许多对 ML 感兴趣的人一起活跃在 Discord here 上。*

*感谢阅读！*

# ***参考文献***

*【1】[DALL-E insta gram](https://www.instagram.com/openaidalle/?hl=en)*

*[2] [意大利人口统计数据](https://statisticstimes.com/demographics/country/italy-demographics.php#:~:text=The%20Sex%20Ratio%20in%20Italy,million%20more%20females%20than%20males.) (2019)，联合国《世界人口展望》*

*[Github 上的抱紧面部扩散器](https://github.com/huggingface/diffusers)*