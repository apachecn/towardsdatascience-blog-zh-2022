# 如何使用计算机视觉读取酒瓶上的标签？(第一部分)

> 原文：<https://towardsdatascience.com/how-to-read-a-label-on-a-wine-bottle-using-computer-vision-part-1-25447f97a761>

欢迎阅读本系列文章，在这里我将解释我构建第一个计算机视觉项目的历程。这里的目标是让计算机从一张简单的照片上读出一瓶酒的标签。你可能会问，为什么这很复杂？首先，我们不能直接调用像 tesseract 这样的 OCR(光学字符识别)库，因为标签上的文本在圆柱体上是扭曲的，正因为如此，我们不能正确地提取字符，从而提取单词和句子。

为了实现这一点，我们需要将我们的问题分解成不同的具体任务，以成功地阅读写在酒瓶标签上的内容。但首先我们需要向我们的机器解释什么是酒瓶标签，以及如何检测它。

# 第 1 部分—检测标签，图像处理方法

在我最初的研究阶段，我偶然发现了一个很棒的 mathematica 论坛帖子，作者试图做或多或少相同的任务，但是是在果酱罐上。在这个过程中，作者使用不同的图像处理技术来尝试定位标签的位置。所以我当然尝试用 python、opencv 和我自己的酒瓶照片来复制这个。让我们以这张照片为例:

![](img/435edd86502d41adf80d5e20ee3bac04.png)

首先，从计算机视觉的角度来看，这张照片有一个巨大的好处:标签和背景之间有很多对比度，如果我们试图用图像处理来分离标签，我们可以希望得到一些好的结果。

因此，让我们开始检测标签的位置:首先，我们将图像转换为灰度，并应用一个自适应阈值函数来帮助我们简化图像的结构。只考虑图像的对比度。

![](img/7ed491926f85ea8f2078bab48d34b08b.png)![](img/6a8258ef1c905ec5fb6b6bbe9be3d8a8.png)

该[自适应阈值](https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html)功能参数对该方法的成功非常重要，目标是获得标签周围的粗轮廓线。我必须迭代并找到适合我的参数:{blockSize : 35，C : 4}。

一旦这样做了，下一步我发现有用的是模糊图像一点点，以平滑图像轮廓，并应用一个简单的二进制阈值，只得到白色和黑色。在这些步骤中，参数也是手动调整的，以获得可能的最佳结果。

![](img/99614e135cff62f131c6d3f5db23c33e.png)![](img/5de44a6f54524461418ab33062b3346e.png)

一旦我们可以清楚地看到标签的轮廓，我们可以希望 opencv 上可用的 [Canny 函数](https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html)将在我们的图像上绘制所有检测到的边缘，并允许我们通过使用按最大面积排序的 [findCountours](https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a) 函数来精确选择它的位置:

![](img/d71eddc0fb0771f8c83e72f717c24a72.png)![](img/6bbe23f19b05342c6640d58acef40678.png)

在这里，我们成功地找到了我们的标签的相对位置，所以我们可以开始工作，并希望阅读上面写的东西！

…等等，让我们先在其他照片上试试我们的算法，当然它也适用于其他图像。

![](img/dffb25960692d4b335c045128dd020aa.png)

…是的，它对其他图像根本不起作用，因为每个上下文都是唯一的，我们的算法是参数化的，以对我们最初选择的图像起作用。在我的下一篇文章中，我将在神经网络魔法的帮助下，为这个问题找到一个系统化的解决方案，在一个广义的环境中更好地工作！

与此同时，你已经可以在我的[网站](https://plural.run/wineReader)上亲自尝试该项目的当前版本的现场应用，只需用你的智能手机连接到该网站，上传一张照片，等待一分钟，你应该会得到结果。

[更新]下面是第 2 部分的链接:[https://medium . com/@ leroyantonin . pro/how-to-read-a-label-on-a-a-a-wine-bottle-using-computer-vision-part-2-8bd 047 D2 a945](https://medium.com/@leroyantonin.pro/how-to-read-a-label-on-a-wine-bottle-using-computer-vision-part-2-8bd047d2a945)

*所有图片均由作者*