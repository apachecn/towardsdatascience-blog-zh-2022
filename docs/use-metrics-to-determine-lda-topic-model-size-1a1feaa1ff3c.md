# 使用度量来确定 LDA 主题模型大小

> 原文：<https://towardsdatascience.com/use-metrics-to-determine-lda-topic-model-size-1a1feaa1ff3c>

## 一篇超过 3 分钟的主题建模文章

![](img/0bbde82afdf3b5fa9aba3d58852cee2b.png)

马特·布里内在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

opic 建模是自动发现正文中语义上有意义的主题。主题模型产生类别，以单词列表的形式表示，可以用来将文本划分成有用的组。目前用于主题建模的最常见算法是潜在狄利克雷分配( *Blei et al. 2003* )。

有大量的文章、博客帖子、教程和视频涵盖了 LDA 主题建模的基础知识。他们详细描述了语料库的准备、模型的创建，并经常以在 [pyLDAvis](https://github.com/bmabey/pyLDAvis) 中可视化模型结束。这篇文章不是 LDA 基础知识的入门读物，事实上假设读者已经熟悉了这些步骤。本文没有回到这些老生常谈的话题，而是集中在这个过程中一个关键的、较少讨论的部分，使用度量来帮助为一个模型选择正确数量的话题。

读者被警告说，即使在这篇长文中，也只能触及这个主题的表面。本文只讨论了如何使用度量来指导选择最佳主题数量的过程。正如读者将在下面的文章中看到的，在这个过程的某个时刻，度量不再提供足够的信息来评估主题模型。本文在任务完全完成之前就结束了，此时自动化的度量标准必须让位于更细粒度、自动化程度更低的流程，这些流程用于评估主题列表本身的语义内容以及它们与被建模文本的关系。尽管有这些警告，还是希望认真的从业者能从读完它中受益。

虽然这不是一个教程，但是本文中涉及的每一步的数据和代码都在 Kaggle 和 GitHub 上发布和提供。有一款 [Jupyter 笔记本](https://github.com/drob-xx/TopicModelTuning/blob/main/TopicModelTuning.ipynb)将在 [Google Colab](https://colab.research.google.com/) 运行。笔记本以及配置和实现细节可以在 GitHub [库](https://github.com/drob-xx/TopicModelTuning)中找到。本文使用的数据集是一个随机选择的， [30，000 篇文章的子集](https://www.kaggle.com/datasets/danrobinson707/newsdf)，它是一个更大的公开许可的数据集，称为[新闻文章](https://www.kaggle.com/harishcscode/all-news-articles-from-home-page-media-house)。

# 度量不能解决你的问题

(但它们会让你的生活更轻松)

LDA 模型的一个关键特征是开发人员必须选择要建模的主题数量，以获得最佳结果。然而，关于如何确定这一价值的文章却很少。原因之一是，答案永远是:视情况而定。给定模型的最佳主题大小的选择将与文本(语料库)本身的特征以及模型的用途相关。被建模的文档是长的还是短的？它们包含许多不同的想法、主题和主题吗？或者，语料库中的文档是关于一个特定的、小的主题集的精心准备的学术论文摘要吗？也许这些文件是随机的推文，没有特别统一的韵律或原因？

本文将带领读者了解使用行业标准度量来确定少量候选主题大小以供进一步评估的过程。然而，正如将要展示的，这些度量标准不能代替人类的判断。最终，建模者将不得不应用主观标准。他们必须深入了解主题与文本的契合程度，以决定使用哪种模式。虽然这个耗时的任务是不可避免的，但是使用度量来减少任务的大小是非常值得的。

# 工具包

本文的其余部分将关注度量标准的实际应用，以便为一个 30，000 篇文章的新闻剪辑数据库确定一组合理的候选主题大小。目标是建立一组主题，既反映文档中主题的多样性，又理想地将这些主题限制在一般读者可能认为重要或有意义的主题范围内。

# 数据集

所提供的语料库数据集既有原始新闻文章，也有适合 LDA 主题模型处理的经过处理、词条化和删减的文本版本。代码是为希望自己生成这个输出(或者尝试不同的文本准备策略)的读者提供的。

除了核心数据集之外，还提供了由数据集的 90 多次运行产生的指标的副本。最后，在最初创建数据之后，还引入了一个补充的停用词表。

# Google Colab

虽然本文使用了“pro”帐户，但一切都是在标准模式下运行的，没有利用额外的 GPU 处理器或内存。

# [Gensim](https://radimrehurek.com/gensim/)

自始至终使用了 *Gensim* [LDA 模型实现](https://radimrehurek.com/gensim/models/ldamodel.html)。

# [OCTIS](https://github.com/MIND-Lab/OCTIS)

*优化比较话题模型*用于其广泛收集话题模型评估方案。OCTIS 为模型构建和评估提供了一个端到端的环境，包括一个可视化界面。本文只使用了 OCTIS Gensim LDA 包装器和 OCTIS 指标。

# [kneed](https://github.com/arvkevi/kneed)

*Python 中的拐点检测*用于识别各种度量结果中的“最大点”曲率，从而选择特定的模型构建作为候选模型。

# [阴谋地](https://plotly.com/)

来制作图表。

# 评估模型

评估了两个常用的衔接指标:NPMI 和 CV(*Lau 等人，2014 年*和*roder 等人，2015 年*)。此外，还描述了一系列衡量主题多样性和相似性的指标。以前的内聚度量在计算上是昂贵的。这里运行两个常见的变量来确定它们是否产生不同的结果。主题度量——分析模型的主题输出，而不是其内部结构，比基于 PMI 的内聚度量要便宜得多。这三个多样性指标是:主题多样性( *Dieng 等人 2019* )、基于倒排的重叠( *Webber 等人 2010* )和 Kullback-Liebler 散度。还运行相似性度量:成对 [Jaccard 相似性](https://en.wikipedia.org/wiki/Jaccard_index)和三个主题显著性度量:KL 统一、KL 空泡和 KL 背景( *AlSumait 等人，2009 年*)。

重要的是要记住 LDA 模型是随机的。这意味着模型的每次运行都会产生不同的结果。因此，在本次评估中，在每个选定的主题规模下运行了多个模型及其完整的度量套件。总共进行了 90 次实验。LDA 主题模型被创建用于主题数量大小为 5 到 150，增量为 5 (5，10，15…150).每次运行都捕获了所有九个指标。所有 90 次运行的指标都绘制在这里:

![](img/13019b8bcdf80f1f2ab2e6deb48ce569.png)

图片作者。

对每个主题模型尺寸的三次运行进行平均，得到:

![](img/a6c9757ecf7bbfe4a949086b0cf7bda6.png)

图片作者。

不幸的是，解释他们的结果并不简单。NPMI 和 CV 产生了一个最佳结果，但是大多数其他的似乎继续改进超过了选择的样本大小。在继续从机器学习的其他领域借用的直觉之前，确定这些图表的[膝盖或肘部](https://en.wikipedia.org/wiki/Elbow_method_(clustering))将会有所帮助。幸运的是有一个包可以做到这一点: [kneed](https://github.com/arvkevi/kneed)

kneed 的输出揭示了六种候选主题大小:5、10、20、35、50 和 80。

```
Jaccard Similarity: 5
Topic Diversity: 10
NPMI: 20
C_V: 20
Inverted RBO: 35
KLBac: 50
KLDiv: 50
KLVac: 50
KLUni: 80
```

六个很多，但比三十个好。在某些时候，有必要对每个主题进行抽样，并将其与一组文档进行比较，以判断是否合适。这六个模型代表了两百个主题。如果我们只对每个主题的 10 个文档进行抽样，这将意味着对 2000 个文档进行审查、评级和比较。如果能进一步缩小候选人名单就更好了。主题分布的概况可能指向可以不考虑的模型。

# 似然分布

因为主题是跨整个语料库创建的，所以文档将包含多个主题。LDA 模型计算给定文档中存在一组主题的可能性。例如，一个文档可能被评估为包含十几个主题，没有一个主题的可能性超过 10%。另一个文档可能与四个主题相关联。在这四个主题中，有一个主题可能被评估为 90%的可能性，剩下的 10%分布在其余三个主题中。这种现实会使手头的工作变得复杂，产生大量需要评估的变体(每个文档不同主题的混合)。

为了简化这个问题，有可能应用另一个被广泛接受的直觉——占主导地位的*话题。主导主题是最有可能出现在文档中的主题。例如，假设一个文档的可能性分布为 62，33，4，1。主导主题是具有最大可能贡献的主题，在本例中为 62%。当我们绘制一个主题成为我们的数据集生成的文档的主导主题的次数时:*

![](img/c2c7ccaf36b4e749246011b6575e3fb8.png)

图片作者。

请注意，在较大的模型中，最高代表值和最低代表值之间的比率变化很大。在五主题模型中，最主要的主题出现在 28%的文档中，最不主要的主题出现在 15%的文档中。在 80 个主题的模型中，范围是从 7%到 0.001%。在较小的模型中，即使是最少出现的主题也可以在数百篇文章中找到。在较大的模型中，长尾包含非常少量的文档。这意味着即使这些主题是“好的”,它们也只代表整个文档的很小一部分，相对于我们声明的对文档进行良好的总体分块的目标来说，它们很可能只是噪音。测量模型如何在主题内分发文档*的进一步剖析可能揭示重要信息。*

上图显示，在一个给定的模型中，有一些主题在许多文档中占主导地位，而在其他文档中则相对较少。但是它并没有揭示*一个给定的主题对于它所有的主题有多重要。例如，在一个文档中占主导地位 20%的主题与在另一个文档中占主导地位 80%的主题被视为相同。幸运的是，描述给定主题*内*的分布是很简单的:*

![](img/2e14b881c552193d33aff50c18229910.png)

图片作者。

请注意，在五个和十个主题模型中，每个主题的上限和下限大约在 20%和 100%之间。在每个后续型号中，此窗口会下移。还要注意的是，在两个较小的模型中，中值大约在 60%以上，统计异常值的数量非常少。

在这个层次的分析中，数字表明小型模型确实非常强大。他们总体上是统计上有信心的，每个主题的平均信心都超过 50%，信心得分的低端比所有其他模型都好。较大的型号就不是这样了。随着主题大小的增加，数据变得越来越嘈杂，离群值越来越多，平均值降低，以高置信度预测的主题越来越少，低预测主题的数量增加。

基于这些统计数据，可以从评估中排除两个最大的模型。请记住，这个模型的任务是得出一组主题，这些主题将允许语料库被分成可管理的块。取消 50 和 80 主题模型的理由是:

*   大量的主题对于任何类型的一般组块来说都是不实用的。许多较大的模型主题只占语料库的不到 2%。即使这些主题代表了语义上有意义的类别，它们也不能服务于总体目标。
*   在较大的模型中，平均可能性主题优势几乎总是小于 50%。较大的模型对它们的预测不是很有信心。
*   两个最大的模型中的数据非常嘈杂。

在这一点上，有必要从使用度量和统计跨越到基于对模型本身的相似性的更主观的度量来评估模型。然而，在结束本文之前，有必要快速浏览一下模型输出，以确定是否有可能合理地进一步减少候选主题大小的数量，并使这项工作变得更加容易。

# 评估主题模型的语义

主题模型本身的评估可以不同于它们所分类的文档。事实上，上面使用的基于非 PMI 的指标正是通过评估包含该主题最常见单词的单词来做到这一点的。我们可以基于主题的内部连贯性、单词是否一起工作及其可理解性、单词是否一起形成语义上有意义的概念或类别来评估主题。

主题列表被评估为内聚性和可理解性。衔接是对单词是否能很好地结合在一起的判断。例如，“汽车、船、火车”或“男人、女人”是内聚性列表，而“汽车、船、火车、男人、女人”的内聚性较低。可理解性衡量某个想法、主题或活动是否清晰地出现在主题列表中。

这是五个主题模型中每个主题的前十个单词(第一个数字是主题占主导地位的文档数，第二个数字是主题 id):

```
============== Five ===============
8609 3 family old leave man know life mother see woman call
6614 4 first win club come back leave second world england match
5706 1 president official call country law former cnn party leader include
4570 2 company school uk business include cent service high money come
4501 0 woman world study many know health first see life even
```

五类模型的主题似乎在衔接性和可理解性上都是混杂的。例如在主题 3 中，例如`family, man, mother, woman`在语义上与`old, life`一起工作。`Leave, know, see, call`不清楚。主题 4 似乎是最有凝聚力的，该组中的单词没有一个是不合适的。话题 4 也是最容易理解的，显然是关于足球的。主题 2 似乎与商业有关(大概在`uk`)，但是`include, cent, come`这几个字降低了它的整体凝聚力。同样，主题 1 可能是关于政治的，但同样，它的列表中有一些词很难与其他词相协调。话题 0 最难解读，连贯性最差。

这很有趣，但似乎对我们快速消除该模型的任务没有帮助。然而，看看题目和课文的例子有多吻合。文档的随机抽样产生(主题列表在每个文本样本的开头，“贡献:”是主题在文本中出现的可能性):

```
************************
Model:  Five
Document ID:  19866
Topic:  1
Contribution:  0.5211269855499268president official call country law former cnn party leader includeDaily Mail Reporter . PUBLISHED: . 14:04 EST, 6 June 2013 . | . UPDATED: . 14:04 EST, 6 June 2013 . The government says one in 10 youths at juvenile detention facilities around the country reported having been sexually victimized by staff or by other youths. The study by the Bureau of Justice Statistics found that among the more than************************
Model:  Five
Document ID:  12506
Topic:  4
Contribution:  0.4131307005882263first win club come back leave second world england match1 June 2016 Last updated at 16:20 BST  The Gotthard base tunnel is
57km (35-miles) long and took seventeen years to build.  Engineers dug deep under the Swiss Alps mountains to make it and links northern Europe to Italy in the South.  The tunnel will be used for freight trains transporting goods and passenger trains.  It's estimated around************************
Model:  Five
Document ID:  12890
Topic:  3
Contribution:  0.5673818588256836family old leave man know life mother see woman callBy . Daily Mail Reporter . PUBLISHED: . 07:25 EST, 29 November 2013\. UPDATED: . 07:25 EST, 29 November 2013 . He had his hind legs
amputated at just a few weeks old after being born with a severe
deformity. But not only has the Boxer puppy overcome his disability by running on his front paws, he also has a specially adapted wheelchair************************
Model:  Five
Document ID:  11310
Topic:  4
Contribution:  0.573677122592926first win club come back leave second world england match(CNN) -- "Glee" will likely end its run after season 6 the final year in the drama's current deal on Fox. "I would not anticipate it goes beyond two more seasons," Fox  entertainment chairman Kevin Reilly told reporters on Thursday. "Never say never, but there's two very clear [story] arcs to get to that end and conclude. If we discover a************************
Model:  Five
Document ID:  4728
Topic:  1
Contribution:  0.580642819404602president official call country law former cnn party leader includeBy . Simon Walters, Glen Owen and Brendan Carlin . PUBLISHED: . 18:25 EST, 27 April 2013 . | . UPDATED: . 18:38 EST, 27 April 2013 . David Cameron's election guru believes that Tory chairman Grant Shapps and Chancellor George Osborne are ‘liabilities’ who will cost the party votes in this week’s crucial town hall polls, it was claimed last
```

这些例子表明，这些主题与它们相应的主题分类有很大的偏差。基于此，从考虑中去除五个主题模型是合理的。

转向十话题模型，我们发现话题连贯性是混合的:

```
=========== Ten =============
4684 3 old leave family man miss car officer see back come
3928 8 club first win match england leave score back side come
3924 7 film see first star come think know world even well
3304 5 world water high china country large area first see many
3053 6 official country military attack security cnn president call leader american
2749 4 party uk bbc service election vote council public company labour
2618 9 woman school student know family parent life call girl want
1979 0 charge judge sentence prison trial murder arrest prosecutor drug month
1900 1 president trump republican obama race campaign first come run car
1861 2 hospital health patient doctor medical dr care treatment die risk
```

五个随机选择的文档示例揭示了:

```
************************
Model:  Ten
Document ID:  13787
Topic:  7
Contribution:  0.4396437108516693film see first star come think know world even wellLOS ANGELES, California (CNN) -- When director Antoine Fuqua rolls
into a community to shoot a movie, he becomes part of that community. Filmmaker Antoine Fuqua began a program to foster young moviemakers in poor communities. This isn't the case of a Hollywood filmmaker cherry-picking glamorous locations like Beverly Hills or Manhattan. Fuqua's************************
Model:  Ten
Document ID:  19146
Topic:  7
Contribution:  0.4848936200141907film see first star come think know world even wellShinjuku has a population density of about 17,000 people per square
kilometre but undeterred by this it has granted citizenship to a new
resident, who only goes by one name - Godzilla.  Name: Godzilla
Address: Shinjuku-ku, Kabuki-cho, 1-19-1  Date of birth: April 9, 1954 Reason for special residency: Promoting the entertainment of and************************
Model:  Ten
Document ID:  1482
Topic:  1
Contribution:  0.3362347483634949president trump republican obama race campaign first come run car(CNN) -- "An unconditional right to say what one pleases about public affairs is what I consider to be the minimum guarantee of the First Amendment." -- U.S. Supreme Court Justice Hugo L. Black, New York Times Co. vs. Sullivan, 1964 . It's downright disgusting to listen to conservative and Republican lawmakers, presidential candidates,************************
Model:  Ten
Document ID:  28462
Topic:  5
Contribution:  0.5035414695739746world water high china country large area first see manyThe emergency services were called out at about 10:00, and the CHC
helicopter landed at about 10:15\.  A CHC spokesperson said: "In
accordance with operating procedures, the crew requested priority
landing from air traffic control.  "This is normal procedure, a light illuminated in the cockpit."  The spokesperson added: "The aircraft************************
Model:  Ten
Document ID:  16179
Topic:  3
Contribution:  0.5338488221168518old leave family man miss car officer see back comeThe 31-year-old right-armer joined from Hampshire ahead of the 2014
campaign, but missed most of the 2015 season with triceps and back
injuries.  Griffiths was Kent's leading wicket-taker in the T20 Blast this season, with 13 at an average of 33.61\.  He also played three times in the One-Day Cup, but did not feature in the Count
```

总的来说，主题似乎比五主题模型更符合代表性文档。虽然存在明显的问题，但在没有进一步证据的情况下，放弃十个主题的模型似乎为时过早。

# 摘要

本文演示了如何使用度量来帮助确定 LDA 主题模型的大小。生成了许多不同大小的模型及其附带的度量和概要统计。计算度量输出的拐点，其识别六个候选主题模型大小。两个较大模型的相对较高的统计噪声，加上它们的剪切大小不适合手头的任务的判断，导致它们被排除在考虑之外。分析转向了五个和十个主题模型，这些模型在统计学上似乎是可行的。在这一点上，纯粹的度量和统计评估工具已经用尽。本文介绍了判断模型语义的初步过程。五个主题的模式显然不适合这项任务，很容易被排除在外。虽然十个主题的模型显示出一些弱点，但在缺乏更彻底分析的情况下，不考虑它似乎是不合理的。

网上大多数关于 LDA 主题模型创建的文章要么是基础教程，要么是关于 LDA 数学或其评估的密集的理论论文。本文试图为开发人员提供一个模板，让他们超越基础知识，立足于实际应用。应该清楚的是，这不是一个简单的“三分钟”任务。本文描述了使用行业标准度量和统计数据为任务提供指导的过程。希望寻求加深他们对如何在现实世界中使用 LDA 的理解的从业者将发现这些信息有助于加深他们对该主题的理解，并且它将提供对如何改进他们自己的工作的洞察力。

# 文献学

AlSumait，l .，Barbará，d .，Gentle，j .，和 Domeniconi，C. (2009 年)。LDA 生成模型的主题重要性排序。*数据库中的机器学习和知识发现*，67–82。

布莱博士，Ng，A. Y .，&乔丹，M. I. (2003 年)。潜在狄利克雷分配。*机器学习研究杂志:JMLR* 。【https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf? ref=https://githubhelp.com

刘，J. H .，纽曼博士和鲍德温博士(2014 年)。机器阅读茶叶:自动评估主题连贯性和主题模型质量。计算语言学协会欧洲分会第 14 届会议论文集，530–539。

罗德尔，m .，两者，a .，和欣内堡，A. (2015 年)。探索话题连贯性测量的空间。*第八届 ACM 网络搜索和数据挖掘国际会议论文集*，399–408。

特拉尼、费尔西尼、加卢齐、特罗佩亚诺和坎代列里(2021 年)。OCTIS:比较和优化主题模型很简单！263–270。

w .韦伯、a .莫法特和 j .佐贝尔(2010 年)。不确定排序的相似性度量。 *ACM 信息与系统安全汇刊*， *28* (4)，1–38。

# 附加参考

这篇文章的灵感来自于:

Selva Prabhakaran (2018) [使用 Gensim (Python)进行主题建模](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#17howtofindtheoptimalnumberoftopicsforlda)

Shashank Kapadia (2019) [评估主题模型:潜在狄利克雷分配(LDA)](/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0)