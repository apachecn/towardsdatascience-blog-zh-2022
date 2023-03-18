# 摆脱数据科学常规的 5 种方法

> 原文：<https://towardsdatascience.com/5-ways-to-escape-a-data-science-rut-4db5148adc72>

## 用这些成长的想法打开你的发展之门

![](img/45bb3cb00609938fe68cfecfd0e4d544.png)

托马斯·马图在 [Unsplash](https://unsplash.com/photos/WajTuzeanUk) 上的照片

## 介绍

我们都知道墨守成规的感觉。某一天，你有动力去学习新的东西，开始新的项目，一头扎进数据科学的世界——第二天，你就感觉自己只是在走过场。过了一段时间后，你看起来不像是在朝着目标努力，而更像是在消磨时间或无休止地做额外的工作。

当我们发现自己像这样在兜圈子时，重要的是检查我们的日常工作，并找到新的方法让事情变得有趣。这里有 5 个想法可以让你走出创造力的低谷，并帮助你重新爱上数据科学。

## 1.进行一些数据科学交叉培训

我最近读了一篇关于领导力的文章，其中的主要观点是，当提高力量时，交叉训练比加倍有氧运动更好。

> “要想从好变得更好，你需要从事相当于交叉培训的商业活动。例如，如果你是技术专家，即使深入钻研技术手册也不会让你获得像沟通这样的互补技能，这种技能会让你的专业知识更明显，更容易被同事了解。”([发表在《哈佛商业评论》上](https://hbr.org/2011/10/making-yourself-indispensable)

机器学习中改变游戏规则的创新也来自交叉训练。以图像转换器模型为例:这些最先进的计算机视觉(CV)模型([优于 CNN](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html))在很大程度上基于最初为自然语言处理(NLP)设计的模型。数据科学交叉培训不仅限于连接 CV 和 NLP 等专业，它还跨越多个领域。原来[中的*注意就是你所需要的*](https://arxiv.org/abs/1706.03762) 执行的语言翻译。现在，变形金刚模型正被应用于从[医学图像分割](https://paperswithcode.com/paper/stepwise-feature-fusion-local-guides-global)到[语音识别](https://github.com/retrocirce/hts-audio-transformer)的方方面面。

感觉卡住了？自己进行一些交叉训练。如果你的专业是监督学习，查一下强化学习技术，看看你能学到什么。或者如果你对面部识别的对象检测感兴趣，看看你能从医学机器学习中的对象检测研究中学到什么。

## 2.收集你自己的数据并将其可视化

数据可视化是一项重要的技能。无论您是在建模前使用数据可视化作为探索工具，还是创建交互式工具来讲述数据故事，清晰有效地传达信息的可视化都至关重要。

我最喜欢的磨练数据可视化技能的方法之一是收集我自己的数据，勾画一些可视化，然后看看我如何使用数据可视化工具实现它们。首先，绞尽脑汁寻找你一直想回答的问题。你邻居的狗一周吠几次？你一天检查多少次你的电子邮件？相对于黑色汽车，有多少辆白色汽车从你的窗前经过？世界是你的！

然后，开始收集数据。当您收集自己的数据而不是使用在线开放数据时，您可以完全创造性地控制要收集的功能和收集的节奏。我还发现，当你自己创建数据集时，更容易知道数据集的哪些方面是有趣的。一旦你得到了你的数据，拿起纸和笔，勾画出一些潜在的形象。数量重于质量——设定一个 5 分钟左右的计时器，然后在纸上画一堆草图。当时间到了，花一两个小时试着编写你最喜欢的可视化代码。如果你想在开始素描之前寻找一些灵感，我推荐你查看一下[视觉复杂性](http://www.visualcomplexity.com/vc/)或[信息是美丽的](https://informationisbeautiful.net/)中一些有趣的插图。

## 3.从基本原则中学习数据科学方法

从基本原则开始学习数据科学方法是一种很好的方式，可以在摆脱常规的同时提高您的数学和统计技能。像《走向数据科学的 T4》这样的出版物是一个很好的地方，可以找到关于数据科学主题的可访问的指南和温和的介绍。以下是我最喜欢的几个，可以帮助你开始，但你可以随意搜索，找到你自己的主题:

*   SHAP 价值观准确地解释了你希望别人如何向你解释。如果你对可解释的人工智能感兴趣，你需要知道沙普利附加解释值或 SHAP 值。本指南介绍了这种与模型无关的特性重要性技术的基础及其博弈论基础。在进入 [SHAP python 库](https://github.com/slundberg/shap)及其文档以了解更多信息之前，请阅读马赞蒂的指南。
*   [理解梯度下降](https://medium.com/towards-data-science/understanding-the-mathematics-behind-gradient-descent-dde5dc9be06e)背后的数学原理。梯度下降是机器学习中最重要的优化技术之一。它包含了如此多的重要算法和技术，所以花一个下午来确保你理解它是如何工作的是值得的。
*   [吉布斯采样由作者解释](/gibbs-sampling-explained-b271f332ed8d)。我为这个动手教程感到骄傲。使用二元正态的例子和一系列的可视化，我走过 Gibbs 抽样和如何从复杂的目标分布中抽样。如果你对马尔可夫链蒙特卡罗(MCMC)算法和贝叶斯统计感兴趣，本教程是一个很好的地方来建立对幕后涉及的统计的直觉。

## 4.阅读一篇最新的研究论文

好吧，听我说完。我知道数据科学和机器学习研究论文可能会令人生畏——它们通常信息密集，理论性太强，在行业中没有用处，或者完全无聊。然而，能够打开一篇你感兴趣的前沿研究论文是有真正价值的。数据科学和机器学习的研究进展以闪电般的速度发生，跟上可能是让你的工作脱颖而出的真正优势。如果你做得对，阅读研究也可以很有趣。

让研究变得有趣的关键是选择正确的论文。您希望找到(a)与您感兴趣的主题相关，(b)您和您的同行感兴趣，以及(c)有相关源代码供您探索和学习的研究。我最喜欢的寻找这些宝石的网站是代码为的[论文。这个网站汇编了机器学习研究的最新成果，并按主题进行了分类，这样你就可以浏览你感兴趣的内容。除了全文和摘要链接，目录中的每个列表都有社区中其他人如何与研究互动的指标。正如所暗示的，所有的研究论文都有代码库和数据集。](https://paperswithcode.com/)

## 5.跟随一些新的数据科学灵感

在社交媒体(Twitter、Medium、LinkedIn、YouTube 等)上关注正确的数据科学家、机器学习工程师和技术专家。)是跟上最新技术、工艺和趋势的好方法。合适的个人也创造迫使我们深入思考数据科学如何与世界互动的内容。识别一些新的声音可能正是你所需要的，以疏通你的发展。

社交媒体可能是一个可怕的地方，所以如果你是数据科学社交媒体的新手，这里有一些技巧和资源可以帮助你开始。

*   **关注那些做着你尊敬的工作的消息来源。像 ML 系统设计专家 [Chip Huyen](https://twitter.com/chipro) 这样的个人和像 GovTech 公司 [Civis Analytics](https://www.linkedin.com/company/civis-analytics/) 这样的公司博客都属于这一类。**
*   **关注那些让你了解专业领域最新动态的账号。Andrew Gelman 的推特账户让我了解他实验室的工作。 [Mike Lopez 的 Twitter](https://twitter.com/StatsbyLopez) 帮助我了解体育分析领域有趣的新发展。**
*   **关注那些让你思考或挑战你的假设的报道。**尝试 [Cortnie Abercrombie](https://www.linkedin.com/in/cortnieabercrombie/) 人工智能伦理以及人工智能/人工智能系统如何放大现有的社会和人口偏见。或者跟随 Allie Miller 了解一切。
*   关注有趣的或让你发笑的账户。见[神经网络猜测模因](https://twitter.com/ResNeXtGuesser)(不客气)。

```
*Acknowledgments: A big thank you to the wonderful, soon-to-be-doctor Claire Hoffman for proofreading and editing this article. If you liked this story and want to read more, follow me on Medium and subscribe via my referred members link here:* [https://sethbilliau.medium.com/membership](https://sethbilliau.medium.com/membership).
```