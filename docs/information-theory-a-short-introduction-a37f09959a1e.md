# 信息论——简介

> 原文：<https://towardsdatascience.com/information-theory-a-short-introduction-a37f09959a1e>

## [理解数据背后的东西&人工智能](http://towardsdatascience.com/tagged/twg144)

![](img/116f5ce5963b3d470f4d2e212e46c66c.png)

照片由[在](https://unsplash.com/@thisisengineering?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) [Unsplash](https://unsplash.com/s/photos/information-technology?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

# 数据和人工智能的核心

当我举办关于人工智能的研讨会或在座谈会上发表论文时，我总是试图用几分钟的理论来引导话题。信息论，贝叶斯定理，语言学，偏见，混沌，复杂性，创新，破坏，以及其他一大堆。我这样做是因为我发现，在许多人列出的高科技职位头衔与该头衔所应具备的知识之间存在严重差距。

似乎所有放在我办公桌上申请编程职位的简历都附有相同的流行语。人工智能、大数据和机器学习(ML)是其中的热门。虽然这些才华横溢的个人确实知道人工智能编程堆栈，并且是 python 和算法的“忍者”，但令人痛苦的是，大多数人显然没有足够的理论知识和他们所做事情背后的结构。

缺乏这方面的知识会产生创造力缺口。在人工智能中，加上海量的数据和预测分析，有这么多因素要考虑，无法展示创造性思维可能是一个巨大的障碍。

我真的希望以下关于人工智能核心的各种主题和理论的短文有助于阐明和教育。人们不应该认为它们是全面的，而是进一步深入感兴趣的领域的起点。

# 布尔函数——慢慢成长的种子

![](img/2095f533d0a42e4fad9d3e40544c66f4.png)

罗伯特·阿纳施在 [Unsplash](https://unsplash.com/s/photos/choice?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

通常情况下，重大的发现和创新需要一颗种子来发展。这种种子本身往往非常重要。信息论就是这种情况，其重要性是无法量化的。如果没有信息论，任何形式的数字信息都不会存在。

它始于 1854 年乔治·布尔关于代数逻辑的论文，“对思维规律的研究，这是逻辑和概率的数学理论的基础。”布尔的代数和逻辑概念今天被称为“布尔函数”,并且从很早的时候就渗透到我们的思维过程中。计算机程序员完全依赖于布尔逻辑操作符，如果没有这些用代码表示的命题，就不可能发展出任何水平的编程复杂度。

> 布尔通过找到用符号和等式表达逻辑语句的方法，彻底改变了逻辑。他给真命题赋值 1，给假命题赋值 0。“一组基本的逻辑运算——比如 and、or、not、要么/or 和 if/then——可以使用这些命题来执行，就像它们是数学方程一样。”⁴

“与或”和“真-假”命题的迷人简单性经历了近 100 年的重大变化。存在两种选择:真和假。线性发展始于一个与或命题。*“在数学中，布尔函数是这样一种函数，它的自变量和结果采用二元集合的值(通常是{真，假}、{0，1}或{-1，1})。”⁵*

上图有两扇门。红色和黄色。每一个都有一个门把手。然而，除了颜色之外，它们在所有方面都是一样的。把它们想象成开关。如果一个打开，另一个保持关闭，这可能意味着或情况。如果两个都打开，这可能意味着一个和的情况。穿过一系列红色和黄色的门，每一对门或者打开或者关闭，或者一个门打开而另一个门关闭，可以近似一个布尔函数。

然而，通过 Claude Shannon 的想法，当真-假世界向编程和/或构造敞开大门时，信息和数据的历史永远改变了。

# 信息论的核心

![](img/dda05fee765b4cafb57a3ba902daa6c8.png)

照片由[罗曼·卡夫](https://unsplash.com/@iamromankraft?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/information?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

从布尔提出他的数学命题开始，信息论的种子萌发了将近一个世纪。1948 年，在贝尔实验室，克劳德·香农发表了一篇论文，对现代技术和数据分析产生了不可估量的影响。在如今被称为“信息时代的大宪章”的“communication',⁶的数学理论”中，香农引入了令人难以置信的概念，即信息可以量化和测量。他将布尔逻辑应用于一个全新的宇宙，同时加入了他个人的天才。

> 香农发现电路可以通过开关的排列来执行这些逻辑运算。例如，为了执行 and 功能，可以将两个开关按顺序放置，这样两个开关都必须打开，电流才能流动。为了实现“或”功能，开关可以并联，这样，如果其中一个开关接通，电流就会流动。“功能稍微多一点的逻辑门开关可以简化这个过程，”⁷
> 
> 在香农的论文发表之前，信息被视为一种定义模糊的瘴气液体。但是在香农的论文之后，很明显，信息是一个明确定义的，最重要的是，可测量的量…
> 
> …香农的信息论提供了信息的数学定义，并精确地描述了系统的不同元素之间可以交流多少信息。这听起来可能没什么，但香农的理论支持了我们对信号和噪声之间关系的理解，以及为什么信息在任何系统(无论是人造系统还是生物系统)中的传播速度都有一定的限制。'⁸
> 
> 香农写道，由此产生的单位可以被称为二进制数字，或者更简单地说，比特。⁹
> 
> 比特现在加入了英寸、磅、夸脱和分，成为一个确定的量——一个基本的计量单位。但是衡量什么呢？“测量信息的单位”，香农写道，就好像信息是可以测量和量化的一样。⁰

在此之前，没有人认为信息可以服从数学公式或计算分析。随着这篇开创性论文的发表，在香农之前有了世界，在香农之后有了 T2。

> 也许香农最大的成就是独立于“意义”分析“信息”的反直觉方法。简而言之，在处理信息时，人们不需要考虑信息的含义。事实上，“意义”对于实际内容来说是多余的。“意义”实际上是没有意义的。

正如他在《交流的数学理论》的第二段引言中所写的:

> 交流的基本问题是在一点上准确地或近似地再现在另一点上选择的信息。**通常这些信息是有意义的；也就是说，它们指的是或根据某些系统与某些物理或概念实体相关联。交流的这些语义方面与工程问题无关。**重要的方面是实际消息是从一组可能的消息中选择的一个。系统的设计必须能适应每一种可能的选择，而不仅仅是实际选择的那一种，因为这在设计时是未知的。

值得关注的是概率和不确定性。随着 bit 的诞生，Shannon 将布尔函数的 0，1->真/假结构提升到了一个全新的高度。香农理论的核心是“噪音”和“惊奇”。所有的交流——人类的、计算机的、通过电线的、信号的、数字的——作为一个普遍的基础——都有“噪音”和“惊喜”的成分。令人惊讶的是，噪音消除后留下的是什么(任何沟通渠道都有噪音)，而不会干扰原始信息。

> 那么，什么是信息呢？它是从信息中挤出每一点自然冗余，去除每一个无目的的噪音之后剩下的东西。“这是一种不受约束的本质，从计算机传递到计算机，从卫星传递到地球，从眼睛传递到大脑，并且(经过许多代的自然选择)从自然界传递到每个物种的集体基因库。”

香农的信息论实际上催生了数字时代。没有它，人们将淹没在噪音和对他们分享的信息的真实性的不确定性中。没有它，所有的交流方式都将陷入混乱的信息和不连贯的含义中。

> 矛盾的是，通过忽略信息的意义，通过展示“意义”对实际信息来说是多么微不足道，香农给了世界真正的意义和安全一致地处理大量数据的能力。

简单来说，信息论是一切的基础。

> 但是在香农之前，信息作为一种思想，一种可测量的量，一种适合于硬科学的对象，几乎没有什么意义。在香农之前，信息是一封电报，一张照片，一段文字，一首歌。香农之后，信息完全被抽象成比特。发送者不再重要，意图不再重要，媒介不再重要，甚至意义也不再重要:一次电话交谈，一份莫尔斯电报，一页侦探小说都被纳入一个共同的代码

在他的论文中，以及在他的余生中，香农向世界介绍了一系列全新的概念:

1.  他将一条信息——二进制单位——命名为“比特”
2.  他向我们展示了意义与信息无关。事实上，我们越关注意义，“信息中的噪音”就变得越大。
3.  通过忽略噪音，他能够产生一种通用的传递和破译信息的方法。
4.  最后，香农给我们留下了一个几乎无法理解的信息噪音术语。这个想法也将动摇科学、技术和数字时代的基础，并在随后的时代留下足迹。他称之为“信息熵”

在本系列的下一篇文章中，我希望揭开熵和“信息熵”的神秘面纱。然而，正如任何研究过熵的人都会告诉你的，热力学第二定律是不可能确定一个定义的。然而，熵在混沌和复杂性理论中扮演着重要的角色，而混沌和复杂性理论是人工智能的核心。

# 关于作者:

泰德·格罗斯是“假设-假设”的联合创始人兼首席执行官。Ted 担任 R&D 首席技术官兼副总裁多年，擅长数据库技术，专注于 NoSQL 系统、NodeJS、MongoDB、加密、人工智能、颠覆、混沌和复杂性理论以及奇点事件。他在虚拟世界技术领域有超过 15 年的专业经验，在增强现实领域有 6 年的经验。Ted 继续在专业学术期刊和脸书 [If-What-if Group](https://www.facebook.com/groups/ifwhatif/) 、 [Medium](https://medium.com/@tedwgross) 和 [LinkedIn](https://www.linkedin.com/in/tedwgross) 网站上撰写许多关于技术主题的文章。你也可以在这里或者在 [Substack](https://ifwhatif.substack.com/) 上注册[免费的 If-What-If 时事通讯。](https://mailchi.mp/110a2d8c8e6a/ifwhatifnews)

# 参考资料:

1.维基百科(未注明)“信息论”，可在:[https://en.wikipedia.org/wiki/Information_theory](https://en.wikipedia.org/wiki/Information_theory)(2021 年 7 月 29 日访问)。

2.维基百科(未注明)，思想法则，可在:[https://en.wikipedia.org/wiki/The_Laws_of_Thought](https://en.wikipedia.org/wiki/The_Laws_of_Thought)(2021 年 8 月 19 日访问)。

3.维基百科(未注明)“布尔函数”，可在:[https://en.wikipedia.org/wiki/Boolean_function](https://en.wikipedia.org/wiki/Boolean_function)(2021 年 8 月 19 日访问)。

4.艾萨克森，w .(2014)“[创新者:一群黑客、天才和极客如何创造数字革命](https://www.amazon.com/Innovators-Hackers-Geniuses-Created-Revolution-ebook/dp/B00JGAS65Q/ref=sr_1_1?crid=2AS0CKI7LO72Z&amp;keywords=the+innovators+walter+isaacson&amp;qid=1650189188&amp;sprefix=the+innovators+%252Caps%252C229&amp;sr=8-1&_encoding=UTF8&tag=virgeartreale-20&linkCode=ur2&linkId=85dd0550b3edc3b283beb931af7ecf20&camp=1789&creative=9325)”，西蒙&舒斯特，纽约，纽约，Kindle 版，位置 943。

5.维基百科(未注明)‘布尔函数’，可在:[https://en.wikipedia.org/wiki/Boolean_function](https://en.wikipedia.org/wiki/Boolean_function)获得(2021 年 8 月 19 日访问)。

6.c . Shannon(1948)“通信的数学理论”，*贝尔系统技术杂志*，第 27 卷，7 月/10 月，第 379–423 页

7.艾萨克森，参考。4 以上，位置 943

8.斯通，j . v .(2018)[信息论:教程介绍](https://www.amazon.com/Information-Theory-Introduction-James-Stone-ebook/dp/B07CBM8B3B/ref=sr_1_1?crid=1WSNSG9MI407X&amp;keywords=Information+Theory%253A+A+Tutorial+Introduction&amp;qid=1650189760&amp;sprefix=information+theory+a+tutorial+introduction%252Caps%252C944&amp;sr=8-1&_encoding=UTF8&tag=virgeartreale-20&linkCode=ur2&linkId=fda705f19d001c68777a881fa082c086&camp=1789&creative=9325)，塞伯特出版社，Kindle 版，位置 82

9.香农裁判。6 以上。

10.Gleick，J. (2011) ' [The Information](https://www.amazon.com/Information-History-Theory-Flood-ebook/dp/B004DEPHUC/ref=sr_1_1?crid=3U6YHRVBW90F2&amp;keywords=the+information+james+gleick&amp;qid=1650190078&amp;sprefix=The+Information%252Caps%252C497&amp;sr=8-1&_encoding=UTF8&tag=virgeartreale-20&linkCode=ur2&linkId=a904d720d96db2500fe390b19cbb40f8&camp=1789&creative=9325) '，纽约，万神殿图书公司，Kindle 版，位置 66。

11.香农裁判。6 以上。

12.斯通，参考文献 8，地点 359。

13.Soni，j .和 Goodman，r .(2017)“[一种思维在发挥作用:克劳德·香农如何发明了信息时代](https://www.amazon.com/Mind-Play-Shannon-Invented-Information-ebook/dp/B01M5IJN1P/ref=sr_1_1?crid=OAZKXPE1PXE2&amp;keywords=A+Mind+at+Play%253A+How+Claude+Shannon+Invented+the+Information+Age&amp;qid=1650190290&amp;sprefix=a+mind+at+play+how+claude+shannon+invented+the+information+age%252Caps%252C214&amp;sr=8-1&_encoding=UTF8&tag=virgeartreale-20&linkCode=ur2&linkId=a05a1b6cf7f11aa41c33a91cc5e2d051&camp=1789&creative=9325)”,西蒙&舒斯特，纽约，纽约，Kindle 版，位置 69。

*为了与 Medium 的披露政策保持一致，上面列出的所有亚马逊图书链接都是假设分析的附属链接。