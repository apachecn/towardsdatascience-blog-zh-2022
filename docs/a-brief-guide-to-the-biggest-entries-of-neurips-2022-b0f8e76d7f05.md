# NeurIPS 2022 最大条目的简要指南

> 原文：<https://towardsdatascience.com/a-brief-guide-to-the-biggest-entries-of-neurips-2022-b0f8e76d7f05>

## NeurIPS'22 提供了许多很好的选择

![](img/edd8958a3062b3d3470d375480da494b.png)

安德烈·斯特拉图在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

神经信息处理系统会议和研讨会(NeurIPS)是机器学习(ML)和计算神经科学方面最受尊敬的国际会议之一。对于[neur IPS’22](https://nips.cc/)(11 月 28 日—12 月 9 日)，新奥尔良被选为活动的主办城市，随后的第二周是虚拟部分。

自 1987 年成立以来，大会已经看到了相当多的突破性提交，包括[墨菲](https://papers.nips.cc/paper/1987/file/92cc227532d17e56e07902b254dfad10-Paper.pdf) (1988)和[神经象棋](https://papers.nips.cc/paper/1994/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf) (1994)，以及最近的 [Word2Vec](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf) (2013)和 [GPT-3](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) (2020)。今年，近 3000 篇论文被接受。neur IPS’22 的议程如此之多，以下是对您有所帮助的内容——关于特别令人兴奋的主题的简要指南:

**#1 联合学习**

联合学习是当今的一个热门话题——它是一种解决与训练大型语言模型(如 GPT-3)相关的资源不足问题的方法。这些模型不仅非常昂贵([高达 1 亿美元](https://huggingface.co/blog/large-language-models))，而且他们目前的训练方式也是不可持续的。

联合学习是一种涉及在边缘设备上进行 ML 模型训练而无需在它们之间进行数据交换的技术，这使得整个过程更便宜并且计算要求更低。今年有 3 个提交来解决这个问题:

来自阿里巴巴的这篇论文的作者提出了一个个性化联合学习方法的基准。[另一篇论文](https://arxiv.org/pdf/2201.13097.pdf)提出了一种使联合和协作学习更有效的理论方法。最后[这篇文章](https://arxiv.org/pdf/2206.08307.pdf)解释了如何通过联合学习获得更好的结果。

对于对这个话题感兴趣的人来说，还有一件事值得一试，那就是关于联合学习的国际研讨会。顺便说一句，所有 NeurIPS 研讨会基本上都是在主要活动中更侧重于主题的小型会议，因此您总能找到符合您兴趣的内容。

**#2 基础和自回归模型**

基础模型是对大量非结构化数据进行训练的模型，随后使用标记数据进行微调，以满足各种应用的需求(例如，BERT)。一个主要问题是，为了微调这些模型，必须引入额外的参数。这意味着在专业集群中持续使用 GPU，这很难获得和融资。

[本文](https://arxiv.org/pdf/2206.01288.pdf)提出了一种分散且成本较低的方法来训练大型基础模型。[另一篇论文](https://arxiv.org/pdf/2209.07526.pdf)提出了一种新的图像语言和视频语言任务的多模态基础模型。本文作者[来自微软，探索如何从图像中提取书面信息，这涉及到连接计算机视觉(CV)和语言模型，从而产生一个能够产生可靠描述性段落的新系统。](https://arxiv.org/pdf/2206.01843.pdf)

还有这个综合性的 [FMDM 研讨会](https://sites.google.com/view/fmdm-neurips/)，其主题围绕着调查基础模型和决策如何能够一起解决大规模的复杂任务。

**#3 具有人类反馈的强化学习**

强化学习一直是 NeurIPS 的主旋律。我们今天面临的一个主要问题是，大型模型生成的输出往往不符合用户的需求或意图。

撰写本文的研究人员介绍了他们使用人在回路方法对大型语言模型进行微调的情况，即如何利用管理人群来训练强化学习的奖励模型。这导致下游应用中预测质量的显著提高。另一个优势是预算更少——与最初的 GPT-3 模型相比，需要的可训练参数更少。

同一个主题推动了关于生成模型人工评估的[研讨会](https://humaneval-workshop.github.io/)，即如何成功地进行人工评估，以支持语言和 CV 的生成模型(如 GPT-3、DALL-E、CLIP 和 OPT)。

**#4 更多研讨会、教程和比赛**

除了我提到的研讨会，还有[这个](https://offline-rl-neurips.github.io/2022/)研究如何建立更可扩展的强化学习系统。还有[这个](https://neurips-hill.github.io/)深入探讨了如何建立更好的人在回路系统的问题。此外，如果你想从技术话题中抽身出来，看看未来的人工智能研究合作，也可以参加这个研讨会。

neur IPS’22 还提供 [13 教程](https://neurips.cc/virtual/2022/events/tutorial)，提供实践培训和实践指导。我推荐查看[这个关于数据集构建的教程](https://neurips.cc/virtual/2022/tutorial/55810)，这个关于基础模型稳健性的教程[，还有这个关于贝叶斯优化的教程](https://neurips.cc/virtual/2022/tutorial/55796)。还有关于算法公平和社会责任人工智能的有用教程。

今年的大会有许多有趣的挑战和竞赛。其中一个是关于填充虚拟环境的最有效策略的[视频游戏挑战](https://www.aicrowd.com/challenges/neurips-2022-the-neural-mmo-challenge)。还有优化建模的自然语言[挑战](https://nl4opt.github.io/) (NL4Opt)，以及大规模图形基准测试的 [OGB-LSC](https://ogb.stanford.edu/neurips2022/) ，这两个挑战都很有趣。

**# 5 neur IPS’22 Socials**

我也强烈推荐今年的社交活动，这是学习新事物的好方法，有机会亲身参与。与大多数研讨会相比，NeuroIPS 的社交活动更为非正式，每个参与者都有机会参与和表达自己的观点。每场社交活动都由一组组织者主持，他们引导讨论，总结所有的意见，然后发表总结性的评论。

neuro IPS’22 充满了有趣的条目——从 [ML 和气候变化](https://neurips.cc/virtual/2022/social/56276)到[neuro IPS 的 K-Pop 爱好者](https://neurips.cc/virtual/2022/social/56275)(是的，你没听错)。[例如，这个圆桌会议](https://neurips.cc/virtual/2022/social/56273)是关于注释者授权和数据优化，即如何解决数据标注者之间的分歧，获得采样多样性，以及建立对偏见免疫的 ML 系统。

**总结**

正如你所看到的，NeurIPS'22 提供了许多很棒的选项，我已经在这里列出了，还有一些我没有空间提及的。希望我的推荐能帮助你更好的组织时间，让你不要错过任何重要的东西。