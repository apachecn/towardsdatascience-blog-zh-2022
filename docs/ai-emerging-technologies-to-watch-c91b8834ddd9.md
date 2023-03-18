# 人工智能新兴技术值得关注

> 原文：<https://towardsdatascience.com/ai-emerging-technologies-to-watch-c91b8834ddd9>

## 通过新颖的软件和硬件技术改变人工智能的使用

![](img/efd980ae09f2a618b377886290b4444d.png)

佛罗里达州威尼斯海滩图片由作者提供

# 介绍

我有幸在英特尔参与的最激动人心的项目之一是领导工程团队设计、实施和部署软件平台，该平台支持基于哈瓦那高迪处理器的 MLPerf 大规模培训。我学到了很多关于如何在大型硬件集群中扩展人工智能培训的知识，以及构建数据中心的挑战。最突出的一点是，驱动这样一项计算密集型工作需要大量的硬件、人力和电力。

现代 AI/ML 解决方案已经表明，给定大量的计算资源，我们可以为复杂的问题创建惊人的解决方案。利用像 [DALL E](https://openai.com/dall-e-2/) 和 [GPT-3](https://openai.com/blog/gpt-3-apps/) 这样的解决方案来生成图像或创建类似人类的研究论文的应用程序确实令人兴奋。

当我在度假时(见上图)，我花了一些时间思考过去几年中模型参数大小的爆炸，以及计算加速器如何快速跟上。我发现了这个帖子，[大型语言模型:新摩尔定律？](https://huggingface.co/blog/large-language-models)，来自 HuggingFace 的人们，我觉得他们很有趣，特别是因为他们比我更了解深度学习模型。

我发现他们与摩尔定律的比较特别恰当，摩尔定律假设晶体管数量和计算能力每两年翻一番。我想到的另一个类似的类比是，在 20 世纪 90 年代末和 21 世纪初，CPU 频率升级竞赛是如何进行的。在这两种情况下，计算硬件增长迅速(甚至呈指数增长)，但最终开始遇到众所周知的瓶颈。

HuggingFace 的文章描述了如何减少将模型投入生产所需的计算量。这些是非常重要的技术，但我认为同样重要的是找到更有效地解决相同问题的替代方法。就像当 CPU 频率达到极限时，计算机行业转向关注并行性/每周期指令数(IPC)/等。过去十年的 AI/ML 工作已经向我们展示了一些可能的例子，现在的问题是我们能否使用不同的、更优化的技术提供相同的解决方案。

# 一些有趣的竞争者

显然，我并不了解这个领域正在进行的所有研究，我也不会自称了解，所以这里列出了一些有趣的技术，它们至少在各个领域提供了一些希望。与其深入研究并试图重现人们已经完成的一些伟大的工作，我主要是试图涵盖我们何时可以使用这些技术来解决可能也映射到 AI/ML 领域的问题。

## 相似性搜索

相似性搜索是一种近似的最近邻(ANN)算法，可用于识别将大向量作为输入的数据组。它采用这个向量和相似向量的现有数据库，并试图识别数据库中的哪些向量最像输入向量。最酷的是向量可以是任何东西，所以相似性搜索的应用可以很广泛。例如，它可用于确定某些数字化资产(图像/声音等)是否与其他资产相似。

为了更好地描述如何利用相似性搜索以及新技术如何提供良好的准确性和良好的功耗，请查看这篇文章:

[](https://www.linkedin.com/pulse/power-intel-hardware-billion-scale-similarity-search-mariano-tepper/) [## 英特尔硬件在十亿级相似性搜索中的强大功能

### 与 Sourabh Dongaonkar、Mark Hildebrand、Cecilia Aguerrebere、Jawad Khan 和 Ted Willke 共同进行相似性搜索…

www.linkedin.com](https://www.linkedin.com/pulse/power-intel-hardware-billion-scale-similarity-search-mariano-tepper/) 

## 多模态认知人工智能

顾名思义，多模态认知人工智能是一个试图采用存在于各种空间中的解决方案，并将它们相互逻辑组合以提供上下文的领域。这使得将大量训练有素的模型结合起来提供更好的解决方案成为可能。我也一直在考虑这个问题，想知道这是否为一些更小、更不准确、计算密集度更低的模型打开了大门，这些模型可以在解决方案中得到利用，而以前如果没有上下文，它们可能不够准确。老实说，我不确定，我很想听听更多领域专家对这个想法是否有意义的看法…

[](/multimodality-a-new-frontier-in-cognitive-ai-8279d00e3baf) [## 多模态:认知人工智能的新前沿

### 现代多模态 ML 体系结构及其应用

towardsdatascience.com](/multimodality-a-new-frontier-in-cognitive-ai-8279d00e3baf) 

有趣的是，就在我准备发表这篇文章的时候，英特尔的 VL 解释器在 IEEE/CVF 计算机视觉和模式识别国际会议(CVPR) 2022 活动上获得了最佳演示奖！这个工具帮助我们理解视觉语言转换器的行为。

[](https://github.com/IntelLabs/VL-InterpreT) [## GitHub——英特尔实验室/VL——翻译

### VL 解释提供了互动的可视化解释的关注和隐藏的代表…

github.com](https://github.com/IntelLabs/VL-InterpreT) 

## 神经形态计算

神经形态计算可能是这三种方法中我最喜欢的，可能是因为它有硬件组件，也可能是因为在我看来它是 AI/ML 建模的另一个阶段。

20 多年前，当我在大学时，我第一次了解了神经网络，这是深度学习算法的主要构建模块之一。这个概念在当时是有意义的，离散层通过节点/层内的连接和数学运算相互传递信息。当在教科书中被描述为类似于大脑如何工作时，我接受了它的表面价值。

神经形态计算的酷之处在于，它使用硬件来更准确地模拟神经系统中神经元的工作方式。神经形态芯片没有集中的共享存储和内存，而是有几个独立的内核，它们有自己的存储和本地内存。内核完全相互连接，这允许内核之间的无数通信流。此外，用于神经形态计算系统的算法可以考虑各种输入因素，包括例如输入随时间的速率或尖峰间隔。

鉴于这是一种相对较新的解决问题的方法，它需要一些新的编程范例。作为硬件研究领域的领导者之一，英特尔已经开源了一个名为 [Lava](https://github.com/lava-nc) 的软件框架，以帮助程序员利用硬件功能。Lava 支持类似深度学习的算法，以及其他类型的算法。

神经形态计算的一个潜在的巨大好处是，这种硬件通常可以以极小的功耗为人工智能领域的一些问题提供解决方案。有很多工作要做，但这一领域的初步工作非常有趣，非常有前途。此外，神经形态计算可以为通过传统计算方法不容易解决的问题提供解决方案。这不仅仅是一种人工智能硬件优化，而是一种全新的思考计算的方式。

这里的第一个链接是一个非常棒的资源，他们有一个非常好的概述，还有一些研讨会，有一些非常容易理解的概述和深入的视频。

[](https://intel-ncl.atlassian.net/wiki/spaces/INRC/overview?homepageId=196610) [## INRC 公众

### 欢迎来到 INRC 网络中心，我们正在为我们的网站带来全新的外观和感觉，以确保您…

intel-ncl.atlassian.net](https://intel-ncl.atlassian.net/wiki/spaces/INRC/overview?homepageId=196610)  [## Lava 软件框架— Lava 文档

### 一个用于神经形态计算的软件框架 Lava 是一个开源软件框架，用于开发神经启发…

lava-nc.org](https://lava-nc.org) 

# 结论

在 AI/ML 领域有许多令人兴奋的研究领域，它们似乎有助于创建需要更少计算能力的更有效的解决方案。计算密集型人工智能/人工智能将永远有一个空间，大量的工作表明了这个空间可以提供的结果的可能性。

然而，随着模型变得越来越大，我们在准确性和可训练性方面达到收益递减点，解决类似问题的新替代方法变得越来越成熟，这些方法需要更少的大规模计算。

相似性搜索、多模态认知人工智能和神经形态计算都是这种方法的例子，我很乐意听到我应该检查的其他新方法，请在评论或 Twitter 上告诉我！

```
**Want to Connect?**If you want to see what random tech news I’m reading, you can [follow me](https://twitter.com/tonymongkolsmai) on Twitter.Tony is a Software Architect and Technical Evangelist at Intel. He has worked on several software developer tools and most recently led the software engineering team that built the data center platform which enabled Habana’s scalable MLPerf solution.Intel, the Intel logo and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.
```