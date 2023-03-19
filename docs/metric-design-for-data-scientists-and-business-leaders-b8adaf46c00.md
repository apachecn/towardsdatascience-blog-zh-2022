# 面向数据科学家和商业领袖的指标设计

> 原文：<https://towardsdatascience.com/metric-design-for-data-scientists-and-business-leaders-b8adaf46c00>

## 公制设计最难的部分是什么？

为了做出好的[数据驱动的决策](http://bit.ly/quaesita_inspired)，你需要三样东西:

1.  *决策标准基于精心设计的* [***度量***](http://bit.ly/quaesita_dmguide) *。*
2.  *收集* [***数据的能力***](http://bit.ly/quaesita_hist) *那些指标都将基于此。*
3.  [***统计***](http://bit.ly/quaesita_statistics) *技能在* [*不确定性*](http://bit.ly/quaesita_uncertainty) *下计算那些指标并解读结果。*

需求#2 和#3 已经写了很多(包括由 [me](http://bit.ly/quaesita_damnedlies) 写的)，但是需求#1 呢？

现在收集数据比以往任何时候都容易，许多领导感到有压力要把数字拖到每个会议上。不幸的是，在喂养狂潮中，他们中的许多人没有给予 ***公制设计*** 应有的重视。在那些愿意付出努力的人当中，大多数人都是边走边补，就好像这是全新的一样。

它不是。

心理学——对思维和行为的科学研究——已经花了一个多世纪的时间来应对试图测量尚未正确定义的模糊数量的危险，因此该领域已经学到了一些坚实的金块，商业领袖和数据科学家在设计指标时借鉴这些金块是明智的。

</why-arguing-about-metrics-is-a-waste-of-time-b1c6f9026724>  

如果你不相信公制设计很难，拿起笔和纸。我挑战你写下幸福的定义，它是如此的铁，以至于没人会反对你衡量幸福的方式…

![](img/34d41f1eb709a01cc323b8977041cdb2.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [D Jonez](https://unsplash.com/@cooljonez?utm_source=medium&utm_medium=referral) 拍摄的照片

很棘手，对吧？现在，试着用人们日常使用的一些抽象名词，比如“记忆”、“智力”、“爱”和“注意力”等等。我们中的任何一个人都了解自己，这简直是不可思议的，更别说相互了解了。

然而，这恰恰是心理学研究者为了取得科学进步而必须清除的第一个障碍。为了研究心理过程，他们必须创造出精确的、可测量的替代物——度量标准。那么，心理学家和其他社会科学家是如何看待公制设计的？

![](img/64ec3111720b477f13c634777cf86857.png)

图片来源: [Pixabay](https://pixabay.com/photos/measure-yardstick-tape-ruler-1509707/) 。

# 像心理学家一样思考

你如何严谨、科学地研究那些你无法轻易定义的概念？像*注意力*、*满意度*、*创造力*这样的概念？答案是……你没有！相反，你[操作](http://bit.ly/quaesita_opera)。为了这个例子的目的，让我们假设你对测量**用户幸福度**感兴趣。

## 什么是操作化？

什么是操作化？我已经为你写了一篇关于它的介绍文章[这里](http://bit.ly/quaesita_opera)，但结果是当你操作时，你首先对自己说*“我永远不会去衡量幸福，我已经平静地接受了这一点。”哲学家已经研究这个问题几千年了，所以你不可能突然提出一个让所有人都满意的定义*。**

*</operationalization-the-art-and-science-of-making-metrics-31770d94998f>  

接下来，你将你的概念的可测量的本质提取到一个代理中。

> 永远记住，你实际上并不是在衡量幸福。或者记忆。或者注意力。或者智力。或者任何其他诗意的模糊词语，不管它听起来有多伟大。

既然我们已经接受了我们永远无法衡量幸福和它的朋友的事实，那么是时候问问我们自己为什么我们会首先考虑这个词了。这个概念是什么——以模糊的形式——看起来与我们想要做出的决定相关？什么混凝土(和可获得的！)信息会让我们选择[一个行动方案，而不是另一个](http://bit.ly/quaesita_damnedlies)？(当你在开始之前心中有了[动作](http://bit.ly/quaesita_hypexample)，度量设计就容易多了。如果可能的话，在试图设计一个指标之前，考虑一下可能的决策。)

![](img/a59a9a0de62e2a2fa9f435083337e0e1.png)

照片由[阿道夫·费利克斯](https://unsplash.com/@adolfofelix?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

然后，我们提炼出我们所追求的核心理念，以创建一个可衡量的代理——一个捕捉我们所关心的核心本质的指标。

> 在命名之前，先定义您的指标。

现在有趣的部分来了！我们可以用任何我们喜欢的名字来命名我们的指标:“博客工作”或者“用户快乐”或者“X”或者其他什么。

我们被语言警察逮捕没有意义的原因是，无论我们多么努力地设计它，我们的代理将****而不是**** 成为用户快乐的柏拉图式形式。

虽然它可能适合我们的需求，但重要的是要记住，我们的指标不太可能适合其他人的需求。这就是为什么在关于我们的衡量标准是否捕捉到真正的幸福的无用辩论中争论不休是愚蠢的。并没有。如果你迫切需要某种标准来统治所有人，那么有一首迪士尼歌曲适合你。

![](img/49032aac63312a1313a23b0b9ac1ded3.png)

让·维默林在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄的照片

我们创建的任何度量标准都只是一个适合我们自己需求的代理(可能没有其他人的需求)。这是我们达到个人目的的个人手段:做出明智的决定或总结一个概念，这样我们就不必每次提到它都写一整段。我们可以在没有语言警察介入的情况下相处得很好。

# 最难的部分

到目前为止，一切顺利。你只需确定你的决策需要哪些信息，然后你想出一种方法，以一种对你的需求有意义的方式总结这些信息(ta-da，*这是*你的度量标准)，然后给它起你喜欢的名字。对吗？对，但是…

那里的 ***是*** 这一切中最难的一部分。继续到[下一期](https://bit.ly/quaesita_hardmetrics)来找出它是什么…

# 关于指标设计的视频

如果你渴望了解更多，请观看我的《与机器交朋友》学习课程中的课程[039](http://bit.ly/quaesita_039)–047。它们都是几分钟长的短视频。从此处开始，并在附加的播放列表中继续:

# 感谢阅读！人工智能课程怎么样？

如果你在这里玩得开心，并且你正在寻找一个为初学者和专家设计的有趣的应用人工智能课程，这里有一个我为你制作的娱乐课程:

在这里欣赏课程播放列表，它被分成 120 个单独的一口大小的课程视频:[bit.ly/machinefriend](http://bit.ly/machinefriend)

<https://kozyrkov.medium.com/membership>  

*附言:你有没有试过在 Medium 上不止一次点击这里的鼓掌按钮，看看会发生什么？* ❤️

# 喜欢作者？与凯西·科兹尔科夫联系

让我们做朋友吧！你可以在 [Twitter](https://twitter.com/quaesita) 、 [YouTube](https://www.youtube.com/channel/UCbOX--VOebPe-MMRkatFRxw) 、 [Substack](http://decision.substack.com) 和 [LinkedIn](https://www.linkedin.com/in/kozyrkov/) 上找到我。有兴趣让我在你的活动上发言吗？使用此表格取得联系。

# 寻找动手 ML/AI 教程？

以下是我最喜欢的 10 分钟演练:

*   [AutoML](https://console.cloud.google.com/?walkthrough_id=automl_quickstart)
*   [顶点 AI](https://bit.ly/kozvertex)
*   [人工智能笔记本](https://bit.ly/kozvertexnotebooks)
*   [ML 为表格数据](https://bit.ly/kozvertextables)
*   [文本分类](https://bit.ly/kozvertextext)
*   [图像分类](https://bit.ly/kozverteximage)
*   [视频分类](https://bit.ly/kozvertexvideo)*