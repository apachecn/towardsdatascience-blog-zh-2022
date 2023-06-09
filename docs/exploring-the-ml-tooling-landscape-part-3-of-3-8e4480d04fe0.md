# 探索 ML 工具的前景(第 3 部分，共 3 部分)

> 原文：<https://towardsdatascience.com/exploring-the-ml-tooling-landscape-part-3-of-3-8e4480d04fe0>

## ML 工具的未来

![](img/4639f822f1b74eb6a938c2201fc8eddc.png)

帕特·克鲁帕在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

本系列的[前一篇博文](/exploring-the-ml-tooling-landscape-part-2-of-3-8e4f3e511891)考虑了 ML 工具生态系统的当前状态，以及这如何反映在 ML 在行业中的采用。主要的收获是专有工具在该领域的公司中的广泛使用，以及相应的多样化和分散的 ML 工具市场。这篇文章最后看了一些新兴的近期趋势，强调了数据可观测性和相关工具的优势，以及 MLOps 初创公司的出现。这篇博客文章将继续讨论 ML 工具的一些关键趋势，这些趋势很可能在不久的将来占据主导地位——或者至少是我想谈论的那些！正如在之前的博客文章中所指出的，我想专注于 MLOps，AutoML 和以数据为中心的 AI。

在这一系列博客文章中，我的目标是解决以下问题:

1.  在工业中，ML 的成熟程度如何？
2.  ML 工具和采用的当前状态是什么？
3.  ML 工具的可能趋势是什么？

这篇博文关注的是**第三个**问题。

与本系列中的所有其他帖子一样，同样的免责声明也适用:本系列的博客帖子绝不是详尽无遗的，甚至在某些地方也不一定是正确的！我写这篇文章是为了组织我对最近几周阅读的思考，我希望这能成为进一步讨论的起点。这是一个绝对迷人的领域，我真的很想了解这个行业，所以请联系我们！

# MLOps

虽然 MLOps 一直是贯穿本系列的主题，但我想花一些空间来研究 MLOps 承诺引入的变化更普遍地代表了 ML。

MLOps 旨在解决部署和维护面向生产的 ML 产品的特定问题。与更传统的软件工程(SWE)不同，ML 模型以及为其提供动力的管道由多个高度集成的组件组成，这些组件依赖于历史输入，即“有状态的”；这意味着实时端到端监控是全面解决生产问题的唯一手段(Shankar，2021)。

“漂移”是一个只能用 MLOps 真正解决的问题的例子。漂移是一个总括术语，可以涉及一系列相关的现象；简而言之，通常通过模型的性能随着时间的推移而下降来观察，这是由于模型的训练数据与模型用于推理的数据之间不匹配。只有拥有一个真正敏捷的端到端开发管道，才能提供解决这个问题的完整方法。

更一般地说，MLOps 的核心代表了从数据导向的模型开发向更接近 SWE 的转变，在 SWE 中，操作日志记录成为监控和开发的基础(Shankar，2021)。

# AutoML

自动机器学习(AutoML)是指一系列旨在消除 ML 工作流程中大多数到所有步骤的人工工作的系统。“AutoML”一词最初是由谷歌在 2017 年推广的(TWiML，2019)，并在 2018 年谷歌 NASNet 的成功之后得到了巨大的推动，其表现优于所有现有的人类创造的模型(人工智能报告状态:2018 年 6 月 29 日)。

一般来说，AutoML 系统的目标是双重的。首先，AutoML 系统旨在使更多的用户能够使用 ML，否则这些用户可能不具备使用 ML 的专业知识。其次，AutoML 旨在通过使端到端开发过程尽可能无缝来加快模型开发的速度(Hutter et al .，2019)。

AutoML 本质上是组合算法选择和超参数(CASH)优化的超集。CASH optimisation 基于这样一种理解，即没有一个最大似然模型对所有任务都表现最佳(Hutter et al .，2019)，它将数据准备、最大似然算法的选择和超参数优化视为一个单一的全局优化问题(Brownlee，2020)。除了这些元素之外，AutoML 在管道创建问题(即端到端 ML 问题)中引入了“管道结构搜索”:尽管 CASH 方法将返回线性管道，但 AutoML 算法可能会生成具有多个并行分支的管道，然后将这些管道重新组合(ller，2021)。

对不属于端到端平台的现有 AutoML 库的调查普遍发现，CASH 算法优于它们；至少从纯车型性能来说是这样的。此外，在对经典 ML 模型(即非神经网络)的调查中，自动 ML 生成的管道通常非常简单，仅限于监督学习，并且尚未完全解决真正自动化 ML 管道所需的所有元素(ller，2021)。为 AutoML 生成的结果应用于神经网络，神经架构搜索(NAS)，返回类似的结果，观察到 NAS 可以生成新颖的，虽然简单的架构(何，2021)。跨云供应商、AutoML 平台(H2O、DataRobot 等)的 AutoML 产品。)和开源软件，尽管许多发行商声称(辛，2021)；目前没有一个开源工具能够创建完全自动化的 MLOps 工作流(Ruf，2021)。

尽管经常提到的采用 AutoML 的目的是为了更广泛地实现 ML 功能，但我可以想象，至少在目前的形式下，由于资源需求和工具的黑箱特性，具有讽刺意味的是，AutoML 可以更好地服务于已经建立了 ML 能力的公司。特别是，如果 AutoML 工具希望使技术能力较差的用户能够利用 ML，它就对可解释的人工智能(XAI)技术的并行改进提出了硬性要求。不考虑这些问题，一些观察者已经建议了一种更有限的、渐进的方法，在这种方法中，AutoML 仅仅辅助 ML 工作流的元素，其中可定制性和通用性不是关键的关注点。出于战略原因，手动任务持续存在的理由可能一直存在(辛，2021)。

# 以数据为中心的人工智能

“以数据为中心的人工智能”一词的具体含义可能很难准确界定。然而，它通常可以被理解为倡导一种整体方法，强调支持模型性能的基础设施、过程和文化，当然包括数据本身，而不是孤立地优化模型(Kurlansik，2021)。最近以数据为中心的人工智能相关性增加的一个更具体的原因，可能部分是由于自动化标签和稳定的架构性能(人工智能报告状态:2021 年 10 月 12 日)。

即使以数据为中心的人工智能可能不属于任何专有的工具或流程，但它确实更加重视适合数据运营的技术，包括数据可观察性、数据血统、数据质量和数据治理工具。更详细地说，数据可观测性可以被认为不仅仅是典型的软件监控，而是使用户能够对历史数据进行特别分析(Hu，2022)。

另一个这样的领域与数据质量有关，数据质量通常通过某种数据验证工具来维护。有两种广泛的方法:使用一组手动维护的声明性规则，如 Great Expectations，或者使用某种 ML 层(可能在规则集之上)来自动检测数据质量问题，如 Anomalo。这两种方法各有利弊。一方面，尽管维护一组声明性规则可能很费力，但它确保了数据的预期是明确的，并有助于澄清理解——类似于代码测试如何帮助文档。另一方面，利益相关者可能很难明确表达他们对数据的要求，因此需要自动化(数据驱动的纽约，2021)。

# 摘要

从本系列的第一篇博文来看，我们发现绝大多数应用 ML 原则的公司在该领域还相当不成熟，而且对于所有公司来说，与数据收集和处理相关的初始步骤仍然是主要的绊脚石。在此基础上，我们预计获得牵引力的工具和解决方案将有望直接解决这些问题，以数据为中心的人工智能的兴趣证明了这一点。虽然这篇博客文章将 MLOps、AutoML 和以数据为中心的人工智能分开处理，但前两者在以数据为中心的保护伞下大致相符:合在一起，目标是推动 ML/AI 产生的商业价值。除了这些主题领域，还有一个使用人工智能来增强数据迁移和数据清理的案例。

随着时间的推移，我们看到越来越多的公司参与到人工智能/人工智能中，但作为一个比例，总的人工智能/人工智能能力保持不变，因为新来者的数量超过了更成熟的参与者。尽管如此，人们普遍认为该领域的创新是由成熟的 ML 公司的需求和经验驱动的，这也可能意味着一定程度的碎片化将持续存在，正如我们在本系列的[第二部分](/exploring-the-ml-tooling-landscape-part-2-of-3-8e4f3e511891)中看到的那样。

还应考虑公司在该领域面临的问题在多大程度上可以通过另一种工具解决方案充分解决:在许多情况下，问题是由更基本的系统性问题引起的(Brinkmann & Rachakonda，2021)。虽然这可能对该领域的大多数初创公司不公平，但在许多情况下，它们都是从解决创始人自己以前经历过的问题开始的，正如我们现在所预期的那样，这些问题可能并不代表更广泛的行业(Brinkmann & Rachakonda，2022)。虽然过去的经验可以形成有效的概念证明的基础，但它并不是统一适用的，并且可能有成为所谓的“寻找问题的解决方案”的风险(Friedman，2021)。ML 采用的下一个时代也可能看到已被证明交付真正商业价值的基本原则的重申，值得注意的是结构化数据比非结构化数据更有价值，过程比工具更有价值(Brinkmann & Rachakonda，2021)。这也很可能意味着特定于数据的工具在不久的将来将继续占据主导地位。

# 参考

Brinkmann，d .，& Rachakonda，V. (2021，4 月 6 日)。*m lops Investments//Sarah Catanzaro//第 33 场咖啡会*。YouTube。检索于 2022 年 6 月 24 日，发自 https://www.youtube.com/watch?v=twvHm8Fa5jk

Brinkmann，d .，& Rachakonda，V. (2022，4 月 20 日)。*穿越数据成熟度谱:创业视角//马克·弗里曼//咖啡会议#94* 。YouTube。于 2022 年 6 月 24 日从[https://www.youtube.com/watch?v=vZ96dGM3l2k](https://www.youtube.com/watch?v=vZ96dGM3l2k)检索

j .布朗利(2020 年 9 月 16 日)。*结合算法选择和超参数优化(CASH 优化)*。机器学习精通。检索于 2022 年 5 月 16 日，来自[https://machine learning mastery . com/combined-algorithm-selection-and-hyperparameter-optimization/](https://machinelearningmastery.com/combined-algorithm-selection-and-hyperparameter-optimization/)

数据驱动的纽约。(2021 年 6 月 21 日)。*炉边谈话:Abe Gong(超导创始人& CEO)与 Matt Turck(first mark 合伙人)*。YouTube。检索于 2022 年 6 月 24 日，来自[https://www.youtube.com/watch?v=oxN9-G4ltgk](https://www.youtube.com/watch?v=oxN9-G4ltgk)

弗里德曼，J. (2021)。*如何获得创业创意*。YouTube。2022 年 6 月 24 日检索，发自 https://www.youtube.com/watch?time_continue=1[&v = uvw-u 99 yj 8 w&feature = emb _ logo](https://www.youtube.com/watch?time_continue=1&v=uvw-u99yj8w&feature=emb_logo)

何，X. (2021)。AutoML:最新技术水平的调查

胡，王(2022)。*数据可观性:从 1788 年到 2032 年*。BrightTALK。于 2022 年 6 月 24 日从[https://www.brighttalk.com/webcast/18160/534019](https://www.brighttalk.com/webcast/18160/534019)检索

哈特，f .，科特霍夫，l .，，范肖伦，j。).(2019).*自动化机器学习:方法、系统、挑战*。斯普林格国际出版公司。

库尔兰西克河(2021 年 6 月 23 日)。*以数据为中心的平台如何解决 MLOps 面临的最大挑战*。数据砖。2022 年 6 月 24 日检索，来自[https://databricks . com/blog/2021/06/23/need-for-data-centric-ml-platforms . html](https://databricks.com/blog/2021/06/23/need-for-data-centric-ml-platforms.html)

Ruf，P. (2021 年)。揭开 MLOps 的神秘面纱，展示选择开源工具的诀窍。

s . Shankar(2021 年 12 月 13 日)。*现代洗钱监测混乱:部署后问题分类(2/4)* 。史瑞雅·尚卡尔。于 2022 年 5 月 8 日从[https://www.shreya-shankar.com/rethinking-ml-monitoring-2/](https://www.shreya-shankar.com/rethinking-ml-monitoring-2/)检索

TWiML。(2019).*机器学习平台权威指南*。

辛，丁(2021)。汽车向何处去？理解自动化在机器学习工作流中的作用。CHI’21:2021 CHI 计算系统中人的因素会议论文集。

泽勒，文学硕士(2021)。自动机器学习框架的基准和综述。*人工智能研究杂志*， *70 期*。