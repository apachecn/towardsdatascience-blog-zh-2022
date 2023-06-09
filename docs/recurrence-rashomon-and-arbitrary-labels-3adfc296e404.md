# 递归、罗生门和任意标签

> 原文：<https://towardsdatascience.com/recurrence-rashomon-and-arbitrary-labels-3adfc296e404>

![](img/6e7be51e12cf3e404fec536820c61143.png)

彼得·奥勒克萨在 [Unsplash](https://unsplash.com/s/photos/abstract?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

## 计算社会科学阅读

# 介绍

本文对与计算社会科学研究相关的三篇论文进行了评论:(1) *级联会复发吗？*(程等，2016)；(2) *整合文本和图像:确定 Instagram 帖子中的多模态文档意图* (Kruk et al .，2019)；和(3) *统计建模:两种文化* (Breiman，2001)；这些论文集中于(1)社交媒体上内容共享的重现(这里，脸书)；(2)使用来自社交媒体(这里是 Instagram)的多模态数据(图像-标题对)来改进作者意图的模型；(3)统计领域内从数据建模向算法建模的转变。

下面，我不提供这些工作的总结，所有这些都是我强烈推荐的，而是集中在以下主题上:(1)数据集内相似性(相对于同一性)的模糊性；(2)建模在理解(和影响)系统中的作用；(3)标签作为文化类别的稳定性。结合起来，这些主题旨在探索数据科学中模糊性的来源。

# 模糊事物的重现

这篇论文通过它的标题*问级联会重现吗？、* Cheng 等人[1]讨论了上图片分享的长期模式，以回答随之而来的问题，如*曾经病毒式传播的内容能否重获病毒式传播？*和*哪些因素影响复发？*。在这项工作中，作者探索了一些现象，如(1)原始内容(即帖子)和复制内容(即转发)的共享瀑布(即使在静止期之后)的重复出现，(2)观众对内容的饱和度(此后兴趣下降)，以及(3)内容的吸引力广度(近似为内容的初始分享量激增的幅度)对重复出现行为的调节作用。

如前所述，该研究考虑了原创帖子和复制或模仿之前帖子的转发。为此，将精确副本和几乎相同的副本与原始内容分组，以研究组内的级联循环。通过对(重新)帖子进行分组，作者能够广泛地研究(低方差)内容*类型*在网络上共享的过程；然而，分组的方法(在这种情况下，二进制*k*-意味着聚类)不是一个中性的选择，因此提出了许多关于内容相似意味着什么的问题。更具体地说，虽然这项研究主要关注几乎相同的图像(包括具有不同覆盖文本的图像)，但内容也可能在其他方面类似，如象征意义或风格。

在这项研究中，图像集群被随机采样，以确认(通过人工检查)它们包含足够相似的图像；94%的样本聚类包含几乎相同的图像，剩下的 6%在文本方面有所不同)。然而，对这一数据或类似数据的后续研究可能会质疑帖子相似的维度(例如，它们是否共享主题？它们以同样的方式转换不同人的形象吗？它们传达相似的信息或激发相似的情感吗？)，以及如何通过机器学习来辨别这些维度。围绕内容模仿的问题——不同的用户如何玩相同的主题以及他们这样做的意图——可能是有趣的问题，因为它们可能与创造力和社会行为有关；然而，从技术角度来看，这些问题也很有趣，因为它们可以从聚类算法的角度来看，聚类算法可能只能成功地捕捉某些相似性维度，而在很大程度上遗漏了其他维度。

# 罗生门效应与叙事空间

在*统计建模:两种文化*【2】中，Leo Breiman 解释了不同的模型如何能够产生大致相同精度的预测(相对于正在考虑的响应)，从而产生不同但同样引人注目的对所研究系统的描述；这种效应被称为罗生门效应(源自电影*罗生门*，其中一个故事是从多个角色的角度讲述的)。从这个想法(从具有相同预测值的不同模型得到的系统的多种描述)延伸，这些模型的解释可以说形成了一个*叙事空间*，讲述了他们可用的完整故事；换句话说，如果每个模型只能从一个角度讲述故事(平行于*罗生门*中的人物)，那么通过组合这些故事(以某种重叠的方式)，就形成了一个包含所有同样引人注目的故事的叙事空间(平行于电影本身)。

(关于不同准确性的模型(不同引人注目的故事)似乎也有一些值得说的东西，但我认为，这样做需要另一个方面的考虑:不仅每个角色都有自己的故事版本，而且一些角色作为叙述者比其他人或多或少更可靠，这影响了观众(或分析师)对故事的解释；然而，以这种方式添加维度会使解释更加复杂，因此，为了简单起见，只考虑获得竞争性能的模型可能更有意义。此外，一个所谓的叙事空间似乎有可能被映射为若干维度上的密度，使得从许多模型的视角可解释的故事的方面(或模型的特征)最为突出。[作为题外话的补充，我最近看了 Emmanuel Candès 在神经信息处理系统会议(NeurIPS)上发表的关于共形预测的演讲，似乎其中概述的关于预测区间的工作和我在这里提出的密度概念之间可能有很好的联系(除了缺点可能没有其他区别)。])

虽然叙事空间的想法与统计学的想法(例如，预测区间、不确定性、集合)完全一致，但在大数据和多参数模型的范式下，模型解释(或应用)可能不太涉及发现~真相(即，在叙事空间中定位高密度区域)，而是更多地涉及在不必发现~真相的情况下调节决策；因此，与其将模型解释为所研究系统的代表以获得对系统的理解，不如根据模型作为决策者的角色对其进行评估，并对其进行修改以限制哪些叙事(在叙事空间内)可以通过预测得到进一步(放大)。

作为一个例子，数据可以显示给定模型的某一组输入总是产生相同的响应，并且模型可以学习这些输入应该总是预测该响应，但是这并不一定意味着给定系统的那些输入就必须总是产生响应(可能没有规定响应的自然法则，样本可能偏向于仅包括系统内发生的某些类型的情况，响应可能完全不希望重复)；因此，在模型作为决策者的情况下，预测可以维持现状，除非模型可讲述的叙述是有限的，并且为了使可讲述的叙述是有限的，模型必须是可解释的和可调的，或者是充分可检查的和可否决的。

# 任意标签的文化稳定性

为了探索被称为意义倍增的符号学概念，Kruk 等人[3]考虑了从社交媒体收集的图像-标题对的多模态数据集。字幕既不是图像的纯转录，也不是图像对字幕的纯描绘，因此，与其假设这两种数据类型之间存在直接和不对称的关系，不如将图像和字幕视为具有复杂的关系，这种关系取决于作者想要通过两者的组合来传达的信息。作者用三组标题(一组捕捉作者意图，一组捕捉上下文关系，一组捕捉符号关系)注释了来自 Instagram 的相当小的图像数据集(n=1299 ),然后根据这种分类法建立了一个注释帖子的模型；他们表明，当同时给定图像和标题时，该模型比只给定其中一个时表现更好(并且当图像和标题在语义上不同时，提升最大)，这应该表明意义是成倍增加的，但听起来也像是银行出纳员琳达的情况的反转。

从我收集的情况来看，在本文撰写之时(2019 年)，这一系列工作还处于早期阶段，因此有许多方法来分割和扩展这项研究；例如，了解图像-标题对的哪些~特征解释了数据类型组合下的准确性提升，这些~特征在多大程度上作为交互术语而不是个体，以及这些~特征在社会中是否是文化稳定的，这可能是有趣的。更具体地关于最后一点:因为标签(作为附加到数据的平面符号，而不是文化中定义的深层概念)本质上是任意的(通过从标签到数据的路径隐式定义，而不是通过预定义的特征和相对于其他标签显式定义)， 考虑本文中用作标签的概念的明确文化定义是否与模型选择的~特征稳定相关，而不是从数据到没有稳定文化基础的(任意)标签的方便途径，这可能是有趣的，因为毕竟，当前引起争议或有争议的(用于注释数据集的两个类别)可能在其他地方或不同时间仅仅是表达性或娱乐性的(另外两个类别)。

# 参考

1.  布雷曼，利奥。2001."统计建模:两种文化."*统计学*16(3):199–231。
2.  Cheng，Justin，Lada A. Adamic，Jon M. Kleinberg 和 Jure Leskovec。2016."瀑布会复发吗？"在*第 25 届万维网国际会议论文集*，671–81 页。加拿大魁北克省蒙特利尔:国际万维网会议指导委员会。【https://doi.org/10.1145/2872427.2882993】T4。
3.  克鲁克，朱莉娅，乔纳·卢宾，卡兰·西卡，肖林，丹·茹拉夫斯基，阿贾伊·迪瓦卡兰。2019."整合文本和图像:确定 Instagram 帖子中的多模态文档意图."在*2019 自然语言处理经验方法会议和第九届自然语言处理国际联合会议(EMNLP-IJCNLP)* 的会议录中，4621–31。中国香港:计算语言学协会。https://doi.org/10.18653/v1/D19-1469[。](https://doi.org/10.18653/v1/D19-1469)