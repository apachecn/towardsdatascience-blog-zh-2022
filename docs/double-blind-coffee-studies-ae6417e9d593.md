# 双盲咖啡研究

> 原文：<https://towardsdatascience.com/double-blind-coffee-studies-ae6417e9d593>

## 咖啡数据科学

## 关于我为什么不做的讨论

在过去的三年里，我收集了很多关于咖啡的数据，并且发表了很多关于改善味道和萃取的文章。然而，我没有做过双盲研究或三角研究(在双盲中使用两个不同的样本和一些对照样本)。我想我也不会，我想说说为什么。

# 定义

盲法研究是指我会冲两杯咖啡，在下面贴一个标签，然后让其他人移动杯子，这样直到研究结束我才知道哪个是哪个。

一项关于咖啡的**双盲研究**是咖啡师制作两份浓缩咖啡(变量不同)，标签藏在杯子下面。然后他们把镜头混在一起，走开，测试者进来。两者都不知道测试者将各自尝试的顺序。

一项在咖啡中进行的**三角研究**使用了与双盲相同的两张照片，但增加了第三张不相关的照片作为对照。

双盲研究的目的是当人们知道哪些变量将进入食谱时，消除偏见。这对于有多个临床试验的药物是非常重要的。对于咖啡，我不认为每个实验都应该遵循临床试验标准。

![](img/fb296e50f373cd80f0e552ee5b918bfc.png)

所有图片由作者提供。我做的一个没有双盲甚至盲测的研究样本。

# 绩效指标

我使用两个[指标](/metrics-of-performance-espresso-1ef8af75ce9a)来评估技术之间的差异:最终得分和咖啡萃取。

[**最终得分**](https://towardsdatascience.com/@rmckeon/coffee-data-sheet-d95fd241e7f6) 是评分卡上 7 个指标(辛辣、浓郁、糖浆、甜味、酸味、苦味和余味)的平均值。当然，这些分数是主观的，但它们符合我的口味，帮助我提高了我的拍摄水平。分数有一些变化。我的目标是保持每个指标的一致性，但有时粒度很难确定。

[](/coffee-solubility-in-espresso-an-initial-study-88f78a432e2c)**使用折射仪测量总溶解固体量(TDS)，该数字结合咖啡的输出重量和输入重量，用于确定提取到杯中的咖啡的百分比，称为**提取率(EY)** 。**

# **咖啡的挑战**

**出于几个原因，双盲研究具有挑战性。**

1.  **你需要不止一个人。**
2.  **你需要调整像浓咖啡这样的东西对舌头的影响，这就是为什么要对稀释的咖啡进行 Q 分级。**
3.  **你需要多个人来品尝世界各地的美食。**
4.  **你需要多种咖啡，不同的烘焙程度，不同的年龄。**
5.  **你需要多台研磨机/浓缩咖啡机。**

**也许它会帮助你看到，我的思想在哪里看到一个实验，更好地理解变量空间。**

# **实验实例设计**

**来说说我最近的 [Rok vs 小众磨床](/rok-beats-niche-zero-part-1-7957ec49840d)实验吧。我最初的设计使用相同的浓缩咖啡机，在多次烘焙和不同烘焙时间进行配对，我没有连续进行配对，而是更换研磨机，因为我整天都在进行烘焙。我发现韩国研磨机在味道和提取上都更好，然后我做了进一步的调查，找到了[的根本原因](/rok-defeats-niche-zero-part-3-2fbcc18397af)。**

**但是，这个实验只是我品尝，并不是双盲的。我们暂且把提取收益率(EY)放在一边。EY 为韩国展示了更好的结果，但是人们对味道有争议。让我们只看一个主观的衡量标准，比如味道。**

**我们可以重新设计这个实验。我们可以假设最初的实验花费了 N 个人-小时，这说明了有多少人花了多少时间做某事。**

1.  **首先，让我们做一个双盲实验，现在需要 2*N 个人-小时。**
2.  **让我们在时间上将镜头推得更近一些，这不会导致时间增加，但必须漱口会大大影响时间。**
3.  **将品酒师的人数增加到 10 人，所以现在我们是 11*N 人(假设 1 名咖啡师为每个人制作饮品)。**
4.  **我做了六次烘烤和多次烘烤，但没有达到烘烤的程度。将烘焙等级的数量改变 4，这将时间增加到 44*N**
5.  **我们可以用一台像样的咖啡机模仿其他机器，做 4 台普通的机器。时间增加到 176*N。**
6.  **我们应该尝试不同的压力/温度曲线还是只尝试一个？对于倍数，N 当然增加。**
7.  **因为这些都是磨床，我们还在乎刀片钝吗？我们是否只在研磨机磨合后才关心它们？**

**我不想夸张。我想用这个例子来说明咖啡数据科学面临的挑战。**

**通常，人们认为我没有提取率来支持我的发现，当我有了，他们会说我的味觉发现不是双盲的。我很少在 espresso 中看到双盲研究，当有双盲研究时，只品尝了几个样品，这也是不够的。所以我觉得那些不应该算标准。**

**这种想法的问题是，它限制了能够对咖啡做出有数据支持的意见的人的数量，只限于少数有专业知识、时间和金钱的人。除了人们为自己重复一个或一组实验之外，没有其他的同行评审过程。**

**我的经验是，即使面对拥有数据的人，人们也会向缺乏数据的知名咖啡人寻求帮助。然后当面对数据时，他们自己没有数据也争论。**

**![](img/7a6000bbf89706dd80cbdb0da0aca6de.png)**

**用粉笔进行第一次微粒迁移试验**

**一个很好的例子就是[微粒迁移](https://medium.com/nerd-for-tech/rebuking-fines-migration-in-espresso-6790e6c964de)。我用多种方法进行了搜索，看看粉末是否会迁移，我发现只有少量粉末迁移，大约 4%。然而，许多咖啡从业者的反对意见是，我并没有否定微粒迁移的理论，同时忽略了该理论缺乏证据。在过去的二十年里，咖啡界已经证明了这一点，但这只是一个没有证据的理论。甚至对该理论起源的讨论也是模糊的，表明它最初是一个口头神话，用来解释有问题的镜头。**

# **如何适应**

**我提出品味和提取。我的目标不是证明某个产品或方法更好，因为我希望它是。我的目标是改善我自己的咖啡体验。为此，我必须以诚实为目标。本着这种精神，我也发表过关于怪异和失败实验的文章，因为我相信它们的结果可能也很重要。**

**我在一天和一周的时间里尝试拍摄，并试图随机化相关的变量。此外，在我的分析中，我采取措施比其他人更深入，所以我不想只是量化一种方法是好是坏，而是要了解原因和条件。**

**espresso 的世界还不是一个充满数据的世界，所以任何数据都比没有数据好，欢迎任何人收集任何数量的数据来拍摄我错了。我愿意讨论数据驱动的结论。**

**如果你愿意，可以在[推特](https://mobile.twitter.com/espressofun?source=post_page---------------------------)、 [YouTube](https://m.youtube.com/channel/UClgcmAtBMTmVVGANjtntXTw?source=post_page---------------------------) 和 [Instagram](https://www.instagram.com/espressofun/) 上关注我，我会在那里发布不同机器上的浓缩咖啡照片和浓缩咖啡相关的视频。你也可以在 [LinkedIn](https://www.linkedin.com/in/dr-robert-mckeon-aloe-01581595) 上找到我。也可以在[中](https://towardsdatascience.com/@rmckeon/follow)关注我，在[订阅](https://rmckeon.medium.com/subscribe)。**

# **[我的进一步阅读](https://rmckeon.medium.com/story-collection-splash-page-e15025710347):**

**[我未来的书](https://www.kickstarter.com/projects/espressofun/engineering-better-espresso-data-driven-coffee)**

**[浓缩咖啡系列文章](https://rmckeon.medium.com/a-collection-of-espresso-articles-de8a3abf9917?postPublishedType=repub)**

**[工作和学校故事集](https://rmckeon.medium.com/a-collection-of-work-and-school-stories-6b7ca5a58318?source=your_stories_page-------------------------------------)**

**[个人故事和关注点](https://rmckeon.medium.com/personal-stories-and-concerns-51bd8b3e63e6?source=your_stories_page-------------------------------------)**

**[乐高故事启动页面](https://rmckeon.medium.com/lego-story-splash-page-b91ba4f56bc7?source=your_stories_page-------------------------------------)**

**[摄影启动页面](https://rmckeon.medium.com/photography-splash-page-fe93297abc06?source=your_stories_page-------------------------------------)**

**[改进浓缩咖啡](https://rmckeon.medium.com/improving-espresso-splash-page-576c70e64d0d?source=your_stories_page-------------------------------------)**

**[断奏生活方式概述](https://rmckeon.medium.com/a-summary-of-the-staccato-lifestyle-dd1dc6d4b861?source=your_stories_page-------------------------------------)**

**[测量咖啡磨粒分布](https://rmckeon.medium.com/measuring-coffee-grind-distribution-d37a39ffc215?source=your_stories_page-------------------------------------)**

**[浓缩咖啡中的粉末迁移](https://medium.com/nerd-for-tech/rebuking-fines-migration-in-espresso-6790e6c964de)**

**[咖啡萃取](https://rmckeon.medium.com/coffee-extraction-splash-page-3e568df003ac?source=your_stories_page-------------------------------------)**

**[咖啡烘焙](https://rmckeon.medium.com/coffee-roasting-splash-page-780b0c3242ea?source=your_stories_page-------------------------------------)**

**[咖啡豆](https://rmckeon.medium.com/coffee-beans-splash-page-e52e1993274f?source=your_stories_page-------------------------------------)**

**[浓缩咖啡滤纸](https://rmckeon.medium.com/paper-filters-for-espresso-splash-page-f55fc553e98?source=your_stories_page-------------------------------------)**

**[浓缩咖啡篮及相关主题](https://rmckeon.medium.com/espresso-baskets-and-related-topics-splash-page-ff10f690a738?source=your_stories_page-------------------------------------)**

**[意式咖啡观点](https://rmckeon.medium.com/espresso-opinions-splash-page-5a89856d74da?source=your_stories_page-------------------------------------)**

**[透明 Portafilter 实验](https://rmckeon.medium.com/transparent-portafilter-experiments-splash-page-8fd3ae3a286d?source=your_stories_page-------------------------------------)**

**[杠杆机维修](https://rmckeon.medium.com/lever-machine-maintenance-splash-page-72c1e3102ff?source=your_stories_page-------------------------------------)**

**[咖啡评论和想法](https://rmckeon.medium.com/coffee-reviews-and-thoughts-splash-page-ca6840eb04f7?source=your_stories_page-------------------------------------)**

**[咖啡实验](https://rmckeon.medium.com/coffee-experiments-splash-page-671a77ba4d42?source=your_stories_page-------------------------------------)**