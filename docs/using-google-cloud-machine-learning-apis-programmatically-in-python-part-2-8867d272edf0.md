# 用 python 编程使用 Google Cloud 机器学习 APIs 第 2 部分

> 原文：<https://towardsdatascience.com/using-google-cloud-machine-learning-apis-programmatically-in-python-part-2-8867d272edf0>

![](img/9155404b192a23da1823ceac10352ac1.png)

照片由[米切尔罗](https://unsplash.com/@mitchel3uo?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 介绍

在本文的[前一部分(第 1 部分)](/using-google-cloud-machine-learning-apis-programmatically-in-python-part-1-430f608af6a5)中，我们探讨了 Google Cloud Vision、文本到语音以及 python 中的语音到文本 API。

在本文的这一部分，我们将探索其余的 API。

1.  翻译 API
2.  视频智能 API
3.  自然语言 API

## 注意

*在 Medium 中，Jupyter 笔记本上的小部件无法在手机这样的小屏幕上正常显示。它们只有在笔记本电脑这样的大屏幕上才能正常显示。*

# 谷歌云机器学习 API

## 1.翻译 API

顾名思义，这个 API 主要用于文本翻译。这还有一个附加功能，可以从文本中检测语言。使用之前，在[谷歌云控制台](https://console.cloud.google.com/marketplace/product/google/translate.googleapis.com?project=text-analysis-323506)中启用该 API。

我试验过的这个 API 的特性列表:

*   **翻译**

这项功能可以将文本翻译成数百种语言。它也非常准确，因为该模型是由谷歌研究人员在谷歌的超大数据集上构建和训练的。此处提供了所有支持语言的列表[。](http://www.mathguide.de/info/tools/languagecode.html)

下面举一个小例子来说明它有多好。在这个例子中，我使用翻译 API 将一篇法语文章翻译成英语和日语。

翻译

*   **语言检测**

该功能检测给定文本的语言。返回检测到的语言代码。这里有一个例子:

语言检测

## 2.视频智能 API

视频智能 API 用于常见的视频处理问题。使用之前，在[谷歌云控制台](https://console.cloud.google.com/marketplace/product/google/videointelligence.googleapis.com)中启用该 API。

我试验过的这个 API 的特性列表:

*   **镜头变化检测**

此功能可检测视频镜头的变化。对于每一个镜头，它返回它的开始和结束时间。

在这个实验中，我使用了奥迪的一个广告视频，是从 [youtube](https://www.youtube.com/watch?v=uWXyAgP1sJQ&ab_channel=AudiNederland) 上为其 e-tron 电动车拍摄的。响应太大，无法在笔记本中显示，必须转储到文本文件中。这里看一下[的详细输出](https://github.com/Subrahmanyajoshi/Google-Cloud-Machine-Learning-APIs/blob/main/cloud-video-intelligence-api/results/shot_change_detection.txt)。

镜头变化检测

*   **语音转录**

这项功能可以转录视频中的语音。不需要手动将视频转换为音频，然后输入语音转文本 API。

我再次用奥迪的广告视频作为例子。

语音转录

*   **标签检测**

此功能可检测视频中的标签。它检测在视频上能找到的一切并返回。

我也用奥迪的广告视频来做这个实验。

标签检测

*   **标志检测**

检测视频中出现的所有流行标志。

在这个实验中，我使用了从 Youtube[上截取的百事可乐的广告视频。](https://www.youtube.com/watch?v=hPcQ5lCTe2U&ab_channel=JocoFlimz)

徽标检测

*   **人物检测**

该功能检测视频中的人，并提供坐标以在检测到的人周围绘制边界框。给出每个检测到的人出现在视频中的时间范围。还给出附加的细节，如每个被检测的人穿什么样的衣服，衣服的颜色等。

我再次用百事可乐的广告视频作为例子。回复太大，无法在笔记本中显示，必须转储到文本文件中。这里看一下[详细输出](https://raw.githubusercontent.com/Subrahmanyajoshi/Google-Cloud-Machine-Learning-APIs/main/cloud-video-intelligence-api/results/people_detection.txt)。

人物检测

*   **人脸检测**

该功能可以检测视频中的人脸。返回坐标以在每个检测到的人脸周围绘制边界框，每个检测到的人脸出现的时间范围，以及诸如此人是否戴着头带、此人是否在微笑等等。它还可以进行人脸标注。

我也用百事可乐的广告视频来做这个实验。这里的响应也太大，必须转储到一个文本文件中。这里看一下[详细输出](https://github.com/Subrahmanyajoshi/Google-Cloud-Machine-Learning-APIs/blob/main/cloud-video-intelligence-api/results/face_detection.txt)。

人脸检测

## **3。自然语言 API**

自然语言 API 用于解决一些常见的 NLP 问题。使用前在[谷歌云控制台](https://console.cloud.google.com/marketplace/product/google/language.googleapis.com)中启用该 API。

我试验过的功能列表:

*   **实体分析**

该特性分析输入文本中的已知实体，如专有名词或普通名词。这里有一个例子:

实体分析

*   **情感分析**

该特征检测输入文本的情感。一个例子:

情感分析

*   **语法分析**

这个特性分析输入文本并返回语法细节。一个例子:

语法分析

*   **文本分类**

该特征将输入文本分类成不同的已知类别。一个例子:

文本分类

# 一个示例用例

多个 API 可以组合起来解决一个复杂的问题。想象一个问题，我们需要从图像中读取文本并检测这些文本的情感。下面是如何仅使用 API 来解决这个问题:

*   使用**视觉 API** 的文本检测功能从图像中读取文本。
*   使用**翻译 API** 的语言检测功能检查文本是否为英文。
*   如果文本不是英文，使用**翻译 API** 的翻译功能进行翻译。
*   然后使用**自然语言 API** 的情感分析功能获取翻译文本的情感。

我们通过结合使用三种不同的 API 解决了这个问题。此外，请注意，为这个问题构建一个定制的机器学习系统将过于复杂。

# 尾注

这些 API 最好的一点是，我们让谷歌来处理一切。从构建和训练准确的模型到管理和扩展部署基础设施。我们唯一担心的两件事是发送请求和处理响应。

在上面的大部分实验中，我只用了一个例子。也可以在一个请求中发送一批示例，API 将在一个响应中返回每个示例的结果。

Python 客户端并不是访问这些 API 的唯一方式。这些也可以通过 curl 命令来访问。JSON 文件需要以特定的格式创建，包含示例的路径以及关于使用什么 API 特性/方法的规范。然后可以使用 curl 命令将其发送到 API 端点，响应将显示在命令行上。

这总结了我已经试验过的机器学习 API 及其特性的列表。正如我们在本文中看到的，大多数常见的机器学习问题都可以使用这些 API 本身来解决。

许多行业已经在使用这些 API 来自动化各种手工任务。关于他们的信息可以在 Google Cloud 的官网上找到。

谢谢