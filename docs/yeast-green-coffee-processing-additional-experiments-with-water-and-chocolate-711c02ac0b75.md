# 酵母绿咖啡加工:水和巧克力的附加实验

> 原文：<https://towardsdatascience.com/yeast-green-coffee-processing-additional-experiments-with-water-and-chocolate-711c02ac0b75>

## 咖啡数据科学

## 用酵母胡闹

在之前的中，我用酵母做了实验，展示了它对生咖啡的影响。在咖啡果和绿豆上使用酵母加工已经完成，但是没有太多关于绿豆发酵的公开信息。一般来说，酵母加工减少了大大影响咖啡豆甜味的酸度和苦味。

在用酵母做实验时，我做了两个额外的实验:水和可可粉。

在**的水实验**中，我很好奇将咖啡豆水合，然后单独脱水是否是酵母咖啡更好的部分原因。最终，事实证明并非如此，但我很高兴我完成了这个实验，因为它有助于分离一个变量。

对于**可可粉**，我随机想到看看酵母和可可粉如何在发酵过程中影响咖啡风味可能会很有趣。我希望创造一种巧克力咖啡，但可可粉似乎对酵母没有影响。

# 设备/技术

[浓缩咖啡机](/taxonomy-of-lever-espresso-machines-f32d111688f1) : [像样的浓缩咖啡](/developing-a-decent-profile-for-espresso-c2750bed053f)

咖啡研磨机:[小生零](/rok-beats-niche-zero-part-1-7957ec49840d)

咖啡:[家庭烘焙咖啡](https://rmckeon.medium.com/coffee-roasting-splash-page-780b0c3242ea)，中杯(第一口+ 1 分钟)

镜头准备:[断奏夯实](/staccato-tamping-improving-espresso-without-a-sifter-b22de5db28f6)和[断奏](https://medium.com/overthinking-life/staccato-espresso-leveling-up-espresso-70b68144f94)

[预灌注](/pre-infusion-for-espresso-visual-cues-for-better-espresso-c23b2542152e):长，约 25 秒

输液:[压力脉动](/pressure-pulsing-for-better-espresso-62f09362211d)

[过滤篮](https://rmckeon.medium.com/espresso-baskets-and-related-topics-splash-page-ff10f690a738) : 20g VST

其他设备: [Atago TDS 计](/affordable-coffee-solubility-tools-tds-for-espresso-brix-vs-atago-f8367efb5aa4)、 [Acaia 比重秤](/data-review-acaia-scale-pyxis-for-espresso-457782bafa5d)、 [Kruve 筛](https://www.kruveinc.com/pages/kruve-sifter)

# 绩效指标

我使用两个[指标](/metrics-of-performance-espresso-1ef8af75ce9a)来评估技术之间的差异:最终得分和咖啡萃取。

[**最终得分**](https://towardsdatascience.com/@rmckeon/coffee-data-sheet-d95fd241e7f6) 是评分卡上 7 个指标(辛辣、浓郁、糖浆、甜味、酸味、苦味和余味)的平均值。当然，这些分数是主观的，但它们符合我的口味，帮助我提高了我的拍摄水平。分数有一些变化。我的目标是保持每个指标的一致性，但有时粒度很难确定。

[](/coffee-solubility-in-espresso-an-initial-study-88f78a432e2c)**使用折射仪测量总溶解固体量(TDS)，该数字结合咖啡的输出重量和输入重量，用于确定提取到杯中的咖啡的百分比，称为**提取率(EY)** 。**

****强度半径(IR)** 定义为 TDS vs EY 控制图上原点的半径，所以 IR = sqrt( TDS + EY)。这一指标有助于标准化产量或酿造比的击球性能。**

# **水实验**

**我设置了同样的豆子，经历了同样的过程，但是我为其中一个豆子留下了酵母。**

**![](img/1583d3625fd87b962dd230dd3ce9b4b6.png)****![](img/476f52c5473609f7f6e4471e4b816057.png)****![](img/38431c858f0c3141506a66b18dae333f.png)****![](img/e1d035096549f7f5bf024a8f0a2fe4a9.png)****![](img/08fb357e5292f66efb1011e77bd5f2ef.png)****![](img/4abdbaeb689d02a7d554fef7309cd11a.png)**

**每张图片的左边和右边分别是水和酵母处理。所有图片由作者提供。**

**它们在罐子里呆了 24 小时，豆子吸收了所有的水分。然而，经过水处理的有液体流出。我用 TDS 仪测试过，甚至还尝过。它非常涩，不好吃。因为糖的含量，我不确定酵母是否在消耗这种提取物，但是当我冲洗和干燥咖啡豆时，我没有把它洗掉。我也担心它会使咖啡豆脱去咖啡因，但是我没有任何证据。**

**![](img/075ae984178472acf599b4a2cfaa7ea5.png)****![](img/490a35127518c310e4c9cb743173a830.png)**

**水加工过的豆子烘烤起来非常不同。我应该把它们放久一点，它们的密度要大得多，这表明它们在烘烤中没有被充分开发。这使得味道比较变得困难。**

**![](img/dfafdf6a51e1fbd9e7b094a9217a929b.png)****![](img/5c5cdf4b68a895d0ef14963c03a813b3.png)**

**水加工(左)和酵母加工(右)**

**部分由于烘烤，酵母比水加工的豆子好得多。我很快结束了实验，因为很明显，味觉得分分布不会有重叠。大多数用水加工过的豆子很难直接饮用。**

**![](img/5c4f71eaaf0285b9478b1fc5b46a5970.png)****![](img/cda88373cef380177a172dd828331cbe.png)**

# **向酵母和绿豆中加入可可**

**我看到了可可包，说“也许”，所以我试了一下。这是有趣的照片。**

**![](img/02a0ebc2aec4b594d769969471227505.png)****![](img/ab77715baa839f6287217a8cad098ed0.png)****![](img/520c79613486b5a6df959ee20adc3a9d.png)****![](img/2a3fe7f8f8b9819f59379a39dfd42989.png)**

**酵母加工成白色，可可+酵母看起来像巧克力棕色。**

**冲洗后这两个看起来很相似。**

**![](img/58643f32eb98ca686a59fec46eef44a0.png)****![](img/5b4a80c7e262ba7221b46e8bc050614a.png)**

**他们烤得非常相似，这让我认为品尝结果不会显示任何重要的东西。**

**![](img/d353362a6a74f0ca0ca841f58669a45e.png)**

**就口味(最终得分)而言，他们差不多。就 TDS/EY/IR 而言，可可豆稍微好一些，但总的来说，我很快就结束了这个实验，因为看不到好的味道。**

**![](img/5b3df08c3388bfc1db98684007416293.png)****![](img/b413400f300339fc9696a749d8386eec.png)**

**即使这两个实验都没有发现新的东西，我仍然喜欢写它们，因为我的失败造就了今天的我。研究就是为了一个绝妙的想法把你的头往墙上撞 100 次。**

**如果你愿意，可以在推特、 [YouTube](https://m.youtube.com/channel/UClgcmAtBMTmVVGANjtntXTw?source=post_page---------------------------) 和 [Instagram](https://www.instagram.com/espressofun/) 上关注我，我会在那里发布不同机器上的浓缩咖啡照片和浓缩咖啡相关的视频。你也可以在 [LinkedIn](https://www.linkedin.com/in/dr-robert-mckeon-aloe-01581595) 上找到我。也可以关注我在[中](https://towardsdatascience.com/@rmckeon/follow)和[订阅](https://rmckeon.medium.com/subscribe)。**

# **[我的进一步阅读](https://rmckeon.medium.com/story-collection-splash-page-e15025710347):**

**[我未来的书](https://www.kickstarter.com/projects/espressofun/engineering-better-espresso-data-driven-coffee)**

**[我的链接](https://rmckeon.medium.com/my-links-5de9eb69c26b?source=your_stories_page----------------------------------------)**

**[浓缩咖啡系列文章](https://rmckeon.medium.com/a-collection-of-espresso-articles-de8a3abf9917?postPublishedType=repub)**

**工作和学校故事集**