# 简单随机抽样:实际上简单吗？

> 原文：<https://towardsdatascience.com/simple-random-sampling-is-it-actually-simple-71014e58e0d1>

## 如何为您的数据项目创建抽样计划

无论你多么努力地试图忘记你的 STAT101 课程，当你的膝跳接近时，你很可能倾向于默认简单随机抽样(SRS)。毕竟，这是你被告知为每一个家庭作业问题所做的假设。我不怪你——在可行的情况下，SRS 是一个很好的选择。

但我发现令人悲喜交加的是，每当我问一批新学生他们如何建议应对数据收集挑战时，我听到的部分答案是“*只是*”。比如，**“只是完全随机地选择它们。”**

让我们在现实世界里呆一会儿，好吗？

# 数据科学家的第一天工作

假设你是一名[数据科学家](http://bit.ly/quaesita_datascim)，被雇佣[估算](http://bit.ly/quaesita_vocab)下图森林中松树的平均高度，并描述[分布](http://bit.ly/quaesita_distributions)。

![](img/5af370c2a7fb4ecc89eea1a990bcac89.png)

这片森林里的树有多高？照片由[丹·奥蒂斯](https://unsplash.com/@danotis?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

鉴于你可以在互联网上找到的树高信息的自助餐，很明显你不会是第一个勇敢的树木测量员来处理这种工作。很多人都测量过高大的树木…这能有多难呢？

*(注意:本文中的链接会带你到我对出现的任何行话术语的轻松解释。)*

# 事实还是统计？

如果你完美地测量了每一棵树，你就不需要统计数据了；你已经掌握了事实。但是你需要事实吗？还是愿意满足于统计？

即使你没有你想要的所有数据，统计学也给你一个前进的方法。测量几棵树([样本](http://bit.ly/quaesita_vocab))而不是整个受祝福的森林([人口](http://bit.ly/quaesita_vocab))会导致对你感兴趣的信息的不太完美但希望更便宜的观点。这是一种解脱，因为你甚至不知道在这片巨大的森林里有多少棵树。

> 让我们测量足够好的树木样本，这样我们就不必测量所有的树木了！

从统计学的角度来看，你的老板让你对 **20 棵树**的[随机样本](http://bit.ly/quaesita_gistlist)进行精确到英尺的测量，所以你遵循了我们[以前的文章](https://bit.ly/quaesita_planck)中的建议，并确认这些规格对你的项目有意义。他们有；舞台已经搭好了！

STAT101 告诉你下一步做什么？

![](img/2bb5c183dc8a2b51315012071e869029.png)

照片由[蒂姆·莫斯霍尔德](https://unsplash.com/@timmossholder?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

我已经给 100 多个班级的学生讲过这个例子，当我问他们我们应该如何挑选树木时，我每次都会听到人群中有人的一个或两个(相当的)回答:

**“随便选[全部/全部]。”**

和/或

**“随便拿个简单的随机样本。”**

# 采取一个简单的随机样本

我不怪你默认简单随机抽样(SRS)作为你的膝跳反应。在可行的情况下，这是一个绝妙的选择。那不是我反对的一点。

令我感到悲喜交集的是，每一次，我都会听到“*只是*这个词作为答案的一部分。

# 只是……没有。

无论是谁告诉你，对这些树进行简单随机抽样的方法是“完全随机地选择它们”……他都不知道如何正确使用英语单词“just”。这件事没有“只是”！当现实世界露出丑陋的一面时，数据设计会变得异常棘手。

![](img/5af370c2a7fb4ecc89eea1a990bcac89.png)

再看看我们的森林吧！丹·奥蒂斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

想象一下，你非常讨厌户外，所以你偷偷把测量树木的工作外包给一个能忍受新鲜空气的人。你雇佣了一个没有技术背景的狂热的徒步旅行者，他渴望遵循你给的任何指示，所以你告诉这个人，呃，“只是”完全随机地选择 20 棵树？！

如果我是徒步旅行者，我会“只是”抓住看起来很方便的前 20 棵树“只是”给你上一课，告诉你要小心你的指示。

# **取样程序**

***简单随机抽样*** 和 ***简单随机抽样******完全随机******SRS***都是专业术语。它们指的是一种抽样程序，其中**每个抽样单位(树)与任何其他树被选中的概率相同**。

> 只有来自相同选择概率的样本才是真正的简单随机样本。否则只是闪闪发光的废话。

SRS 是我们教给统计学新手的第一个(有时也是唯一的)抽样程序，这是有原因的，因为它…简单。就计算而言很简单。还有其他抽样程序，但它们需要调整计算，这通常超出了你第一年统计课程的范围。

> 还有其他采样程序，但它们需要更高级的计算。

如果它来自一个对每棵树都有相等选择概率的森林，那么它只是一个真正的简单随机样本(SRS ),否则当你对它使用 SRS 计算时，它只是一派胡言。

不幸的是，如果你按照 STAT101 教你的方法分析数据，但你实际上没有使用真正简单的随机抽样程序来获得数据，那么从技术上讲，你的结果将是错误的。

> 永远努力给出万无一失的指令，因为你永远不知道一个野傻子什么时候会出现。

如果你的徒步旅行者选择了靠近森林边缘的更方便的树，那绝对不是一个简单的随机样本。这是一种叫做 ***便利样本*** 的东西——这是一种你应该像躲避瘟疫一样避免的程序——在以后的文章中会有更多的介绍。用 SRS 数学分析这样的数据在统计学上是不合适的…如果那些树木得到的阳光量不同，因此不能代表整个森林呢？以此为基础进行推论会使你得出错误的结论。

那么，专业统计学家的回答会是什么样的呢？要找到答案，请前往[第二部分](https://bit.ly/quaesita_srstrees2)！

<https://kozyrkov.medium.com/how-to-create-a-sampling-plan-for-your-data-project-3b14bfd81f3a>  

你有没有试过在 Medium 上不止一次按下拍手按钮，看看会发生什么？ ❤️

# 喜欢作者？与凯西·科兹尔科夫联系

让我们做朋友吧！你可以在 [Twitter](https://twitter.com/quaesita) 、 [YouTube](https://www.youtube.com/channel/UCbOX--VOebPe-MMRkatFRxw) 、 [Substack](http://decision.substack.com) 和 [LinkedIn](https://www.linkedin.com/in/kozyrkov/) 上找到我。有兴趣让我在你的活动上发言吗？使用[表格](http://bit.ly/makecassietalk)联系。