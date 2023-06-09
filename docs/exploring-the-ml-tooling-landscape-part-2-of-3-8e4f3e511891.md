# 探索 ML 工具的前景(第 2 部分，共 3 部分)

> 原文：<https://towardsdatascience.com/exploring-the-ml-tooling-landscape-part-2-of-3-8e4f3e511891>

## 当前的 ML 工具和采用

![](img/b2df41f2a1fa6a1006b8345184c1b536.png)

[附身摄影](https://unsplash.com/@possessedphotography?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

在本系列的[前一篇博文](/exploring-the-ml-tooling-landscape-part-1-of-3-53f8c39e6e4e)中，我们考察了行业中整体机器学习(ML)的成熟度，特别关注机器学习操作(MLOps)。两个主要的问题是，作为一个整体，ML 在行业中明显缺乏成熟，以及完全接受 MLOps 所涉及的复杂性，这可以被视为 ML 成熟的最高点。在这篇博文中，我们将考虑工具在工业和更广泛的 ML 工具市场中的应用。

在这一系列博客文章中，我的目标是解决以下问题:

1.  在工业中，ML 的成熟程度如何？
2.  ML 工具和采用的当前状态是什么？
3.  ML 工具的可能趋势是什么？

这篇博文关注的是**第二个**问题。

与前一篇文章一样，同样的免责声明也适用:这一系列的博客文章绝不是详尽无遗的——甚至不一定在某些地方是正确的！我写这篇文章是为了组织我对最近几周阅读的思考，我希望这能成为进一步讨论的起点。这是一个绝对迷人的领域，我真的很想了解这个行业，所以请联系我们！

# 地貌

谈论拥挤的 ML 工具环境绝不是夸大其词。名副其实的 MAD Landscape(即机器学习、人工智能和数据)列出了截至 2021 年底的 2000 多个(这个数字包括一些重复的)商业和开源项目和公司(图尔克，2021)。而这只是为数不多的类似综述之一:(TWIML，2021)，(LF AI & Data Foundation，2022)，(Hewage，2022)，(Ruf，2021)。

在进一步讨论之前，有必要澄清一下这个主题的一些术语。经常发现端到端的 ML 平台被称为 MLOps 平台(thoughtworks，2021)，数据科学和机器学习(DSML)平台(Turck，2020)，等等。然而，根据前一篇博文中的讨论，使用术语“ML 平台”来包含所有这样的产品是有意义的，因为可以论证 MLOps 通常是 ML 成熟度的最终目标。我也倾向于去掉限定词“端到端”，因为 ML 平台通常旨在服务 ML 工作流程的所有步骤。独立于平台，术语工具用于指处理 ML 工作流中特定步骤的软件。我可能偶尔会使用工具将它们联系在一起，这在上下文中应该是清楚的，或者使用“解决方案”来将两者联系在一起。

虽然这几项审查在方法和广度上有所不同，但也有一些共同点。首先，工具生态系统本身缺乏成熟度，指的是没有任何一个平台支持完全自动化的 ML 工作流，或者至少没有一个平台可供广泛的利益相关者访问(Ruf，2021)。与此相关的一点是专家工具，根据他们的方法，可能支持一个以上的 ML 任务(TWIML，2021)。其次，一个 ML 平台被认为是“好的”,因为它覆盖了 ML 工作流的所有方面，允许自动化，并支持多种语言、框架和库(Hewage，2022)。第三，在现有的解决方案中，不同 ML 任务的实施水平存在很大差异(Felipe & Maya，2021)。正如上一篇博文中所讨论的那样，我们有理由推测，这些问题中有许多是由于在实践中对如何实现 ML 工作流缺乏共识而导致的直接后果。

此外，许多文章根据解决方案的完整性，即它支持多少 ML 工作流，以及产品的“种类”,构建了各种产品。也就是说，

1.  它是一个平台还是专门的/零碎的工具？
2.  它是特定于 ML 的还是适用于软件工程(SWE)的？
3.  它适用于 ML 生命周期的哪个(些)阶段？

后一个标准特别主观，因为对于给定工具与什么特定任务相关是有争议的，例如，数据可观察性几乎触及所有事物(TWIML，2021)。(thoughtworks，2021 年)和(Miranda，2021 年 a)给出了两个这样的框架。我更喜欢后者，因为它提供了一种在更广泛的 SWE 环境中理解 ML 工具的方法，并且提供了一个简单框架的基础来理解工具成熟度的当前状态，即使它没有直接将工具链接到 ML 工作流状态。当我们考虑如何评价下面的工具选择时，这个框架将被证明是有洞察力的。

# ML 工具采用

我们通过检查行业中的 ML 成熟度结束了上一篇文章，注意到企业公司中有两个不同的群体:处于 ML 前沿的成熟公司和近年来才推出 ML/AI 产品的不成熟公司。此外，我们检查内部工具的优势，至少在成熟的公司。事实上，在这个领域的成熟和不成熟的公司中，内部工具化都很突出，但原因不同。

对于成熟的公司来说，这在很大程度上是因为缺乏(成熟的)替代品——毕竟，他们是这个领域的领导者。当具体考虑 MLOps 工具时，尤其如此:特性存储、模型服务等。在对 MLOps 特定平台组件的审查中，在占主导地位的美国大型 B2C 中，只有 ML 平台的工作流程编排组件经常使用开源解决方案，通常是 Kubeflow 管道或气流(Chan，2021)。这可能是因为这是考虑的最通用的组件(dotscience，2019)。在这些公司中，开发内部 ML 平台的目标是双重的(Symeonidis，2022)，首先，目标是减少构建和交付模型所需的时间，其次是保持预测的稳定性和可重复性；或者换句话说，分别处理 MLOps 的交付和部署阶段。随着工具生态系统整体的成熟，这种情况会改变吗？可能没有那么多。这有三个主要原因。首先，任何现存的工具都是直接响应给定公司的特定战略响应而出现的，如果考虑到前面给出的两个需求，可能不希望更广泛地使用这些工具，或者不容易在外部复制这些工具。其次，与此相关的是，可能无法从外部生成一个功能齐全的替代方案。第三，这一级别的成功不是由一种技术决定的，而是由它的端到端集成决定的，这给第三方工具带来了额外的障碍。正如马特·图尔克(Matt Turck)所说:“大数据(或 ML/AI)的成功不是实施一项技术……而是需要将技术、人员和流程组装成一条流水线。”(图尔克，2016)。

在不成熟的公司中，内部解决方案的趋势在很大程度上正是由于这种不成熟:他们可能没有技术技能或预算来获取第三方工具(dotscience，2019)。与此相关，他们可能会发现很难证明需要这些工具，因为没有比较基准，特别是如果考虑到这些工具在很大程度上是用来优化一个已知的过程。正如我们将在下面看到的，要么以渐进的方式创建一个 ML 工作流，要么选择一个覆盖端到端工作流的平台，这两者之间有一个不可逾越的权衡；这两种方法都必然引入技术债务(Ruf，2021)。无论哪种方式，都可能很难找到符合公司要求的产品，这源于对 ML 工作流实现缺乏共识(见上一篇文章)和缺乏功能完整的 ML 平台(TWIML，2021)。相反，围绕现有的 ML 管道实现匹配第三方选项可能太具挑战性(将在下面详细讨论)。一个相关的问题:对于这些公司中的许多来说，可用的工具可能不适合他们的需求，原因有很多，特别是由于各种各样的遵从性问题。例如，当前的数据标签解决方案通常使用外包或合同工，在许多情况下可能不适合，如金融服务等受监管领域(Vanian，2021)。对于许多企业来说，通常更希望确保对数据的完全控制和所有权，并避免将数据分发给多方。像浮潜人工智能这样的公司旨在直接解决这些问题。

综上所述，这些点可能表明当前工具环境的多样性是需要解决许多不同的实现的直接结果，这些实现可能是高度特定于公司的。除此之外，普遍的风险投资资金过剩和其他可用资金(图尔克，2021 年)使得该领域的公司和解决方案迅速扩张。

# 决定决定决定

考虑到前面的讨论，人们不禁要问，如何为不成熟的 ML 实现选择合适的平台和/或工具集？正如所料，没有真正正确的答案，但是可以做出明智的选择，以确保你能够在这个领域取得有意义的进展，同时尽量减少以后的问题。

全面采用 MLOps 的最快途径是关注 ML 生命周期的完整端到端，或者使用(Stiebellehner，2022a)中使用的术语，首先关注功能的深度，然后才关注功能的广度。这样做还有一个额外的好处，就是减少了与想法验证相关的时间和成本，这样就可以尽早获得有意义的反馈。

正如已经讨论过的，目前没有广泛接受的完整的端到端 ML 平台可用。这就需要在现有的 ML 平台和/或从各种独立的专业工具中拼凑一个 ML 工作流程之间做出选择。从战略角度来看，与一家平台提供商合作可能并不可取，因为他们可能不符合您的长期需求，并可能导致供应商锁定，这意味着任何后续变化都意味着更大的机会成本。相反，围绕独立工具开发过程可能意味着在内部产生比期望的更多的工作，特别是如果交付速度是主要目标的话。在任何情况下，按照上面介绍的框架，给定的工具或平台越成熟，并且在一定程度上，工具越零碎，选择就越安全(Miranda，2021b)。

特别是对于 ML 不是核心功能的企业，一般的建议是购买而不是构建，并在这些组件之间添加定制的集成器，无论这些组件是平台还是专用工具(Miranda，2021b)。工具的选择应该基于即时交付目标，重点是这些组件之间的集成和重叠；目标是选择最小的令人满意的选项集。由于关注的重点是深度而不是广度，所以诸如可伸缩性或自动化之类的因素不应该是主要关注点。要问的问题包括，每种工具支持哪种语言来帮助交互，以及一些工具的选择是否会由于它们的重叠功能而引入冗余；诸如此类的问题只能用经验来回答，因此非常需要迭代并从您的过程中学习(Ruf，2021)。Kubeflow 提供了一个工具采用的有启发性的例子，它虽然提供了对端到端 ML 平台的支持，但只看到了像 Kubeflow 管道这样的特定元素被广泛采用。

# 地平线上有什么？

围绕当前工具星座的一个核心争论是，我们是否以及何时应该期待看到某种形式的整合。大多数观察家会期待某种形式的合理化，然而参数远不清楚(图尔克，2021)。一种整合，即“功能整合”，已经在成熟和成功的解决方案中观察到，如 Databricks 和 Snowflake，它们有向功能齐全的端到端产品发展的趋势(Turck，2021)。这很可能是许多成功的第三方解决方案的更普遍趋势的一部分，这些解决方案通常是一个更大的生态系统的组成部分，例如雪花和 AWS。除此之外，我们可能希望看到大型云平台通过内部创新或并购提供额外的服务。

到目前为止，工具和过程的任何更广泛的合理化一般都被数据和工具日益增长的复杂性以及这种变化的速度所抵消。这在 MLOps 和 DataOps 工具方面很明显，这两种工具目前通常由内部工具处理(Turck，2020)。进一步阻碍进步的原因是广泛观察到的人才短缺，预计这种情况将持续更长时间(O'Shaughnessy，2022)，尽管在一定程度上，这可以通过增加工具来缓解。

数据质量测试和可观察性工具目前在市场上取得了最大的成功，这正是由于市场在整体 ML 成熟度方面所处的位置(Stiebellehner，2022b)。然而，我们似乎正在进入 ML 工具和 ML/AI 的新阶段。鉴于致力于 MLOps 的初创公司的显著增长，可以预计运营化将在短期内占据主导地位(Shankar，2021)。然而，随着人工智能招聘的普遍降温(Huyen，2020 年)和风投资金的普遍撤出(Turck，2022 年)，整合似乎即将到来。至于技术发展，我将在本系列的第三篇博文中讨论，但是我希望看到以数据为中心的人工智能、MLOps 和 AutoML 在不久的将来成为 ML 工具的主要趋势。

# 总结

在这篇博客文章中，我们继续我们之前关于 ML 在行业中的成熟度的讨论，展示了复杂 ML 采用的普遍低水平与 ML 工具产品的数量和不完整性之间的联系。此外，我们暗示了近期的一些关键趋势:对数据测试和可观测性的持续兴趣，以及对 MLOps 初创公司的不断增加的资助。在此基础上，本系列的下一篇博文将研究工具领域中出现的一些关键技术。

# 参考

费利佩，a .，&玛雅，V. (2021)。MLOps 的状态。

Hewage，n .等人(2022 年)。机器学习操作:关于 MLOps 工具支持的调查。

呼延，C. (2020 年 12 月 30 日)。*机器学习工具 Landscape v2 (+84 新工具)*。奇普·胡恩。2022 年 5 月 7 日检索，来自 https://huyenchip.com/2020/12/30/mlops-v2.html

人工智能和数据基金会。(2022). *LF AI &数据基础互动景观*。LF AI &数据景观。2022 年 5 月 6 日，从[https://landscape.lfai.foundation/](https://landscape.lfai.foundation/)检索

l .米兰达(2021a，5 月 15 日)。*浏览 MLOps 工具领域(第 2 部分:生态系统)*。Lj 米兰达。2022 年 5 月 6 日检索，来自[https://ljvmiranda 921 . github . io/notebook/2021/05/15/navigating-the-mlops-landscape-part-2/](https://ljvmiranda921.github.io/notebook/2021/05/15/navigating-the-mlops-landscape-part-2/)

l .米兰达(2021b，5 月 30 日)。*在 MLOps 工具领域导航(第 3 部分:策略)*。Lj 米兰达。2022 年 5 月 6 日检索，来自[https://ljvmiranda 921 . github . io/notebook/2021/05/30/navigating-the-mlops-landscape-part-3/](https://ljvmiranda921.github.io/notebook/2021/05/30/navigating-the-mlops-landscape-part-3/)

奥肖内西，P. (2022，4 月 12 日)。亚历山大·巴甫洛夫·王——人工智能入门。YouTube。2022 年 5 月 8 日从 https://open.spotify.com/episode/0jFd4L8nvDROu05lk2kv6y?[检索 si = 06 E4 af 52 baff 44 be&nd = 1](https://open.spotify.com/episode/0jFd4L8nvDROu05lk2kv6y?si=06e4af52baff44be&nd=1)

Ruf，p .等人(2021 年)。揭开 MLOps 的神秘面纱，展示选择开源工具的诀窍。

s . Shankar(2021 年 12 月 13 日)。*现代洗钱监测混乱:对部署后问题进行分类(2/4)* 。史瑞雅·尚卡尔。于 2022 年 5 月 8 日从[https://www.shreya-shankar.com/rethinking-ml-monitoring-2/](https://www.shreya-shankar.com/rethinking-ml-monitoring-2/)检索

斯蒂贝勒纳，S. (2022a，2 月 27 日)。*【MLOps 工程师】纵向第一，横向第二。为什么在开发机器学习系统时应该尽早突破生产，以及 MLOps 如何促进这一点。|作者 Simon Stiebellehner | Medium* 。西蒙·斯蒂贝勒纳。2022 年 5 月 7 日检索，来自[https://sistel . medium . com/the-mlops-engineer-vertical-first-horizontal-second-306 fa 7 b 7 a 80 b](https://sistel.medium.com/the-mlops-engineer-vertical-first-horizontal-second-306fa7b7a80b)

斯蒂贝勒纳，S. (2022b，4 月 10 日)。未来的“数据狗”。数据质量如何，监测&可观测波正在建立。2022 年 4 月，。ITNEXT。2022 年 5 月 9 日检索，来自[https://it next . io/the-mlops-engineer-the-data dogs-of-tomorrow-614 a88a 374 E0](https://itnext.io/the-mlops-engineer-the-datadogs-of-tomorrow-614a88a374e0)

思想工厂。(2021).MLOps 平台评估指南 2021 年 11 月。

m .图尔克(2016 年 2 月 1 日)。*大数据还是个东西吗？(2016 年大数据格局)*。马特·图尔克。2022 年 5 月 4 日检索，来自 https://mattturck.com/big-data-landscape/

m .图尔克(2017 年 4 月 5 日)。*全力以赴:2017 年大数据前景*。马特·图尔克。2022 年 5 月 4 日从[https://mattturck.com/bigdata2017/](https://mattturck.com/bigdata2017/)检索

m .图尔克(2020 年 10 月 21 日)。*2020 年的数据和 AI 景观*。风险投资。2022 年 5 月 6 日检索，来自[https://venturebeat . com/2020/10/21/the-2020-data-and-ai-landscape/](https://venturebeat.com/2020/10/21/the-2020-data-and-ai-landscape/)

m .图尔克(2021 年 9 月 28 日)。*红热:2021 年的机器学习、AI 和数据(MAD)格局*。马特·图尔克。于 2022 年 5 月 4 日从[https://mattturck.com/data2021/](https://mattturck.com/data2021/)检索

m .图尔克(2022 年 4 月 28 日)。*2022 年风投大撤退——马特·图尔克*。马特·图尔克。于 2022 年 5 月 8 日从[https://mattturck.com/vcpullback/](https://mattturck.com/vcpullback/)检索

TWIML。(2021 年 6 月 16 日)。*介绍 TWIML 新的 ML 和 AI 解决方案指南*。TWIML。2022 年 5 月 6 日检索，来自[https://twimlai . com/solutions/introducing-twiml-ml-ai-solutions-guide/](https://twimlai.com/solutions/introducing-twiml-ml-ai-solutions-guide/)