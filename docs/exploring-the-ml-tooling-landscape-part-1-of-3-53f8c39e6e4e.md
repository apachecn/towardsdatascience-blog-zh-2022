# 探索 ML 工具的前景(第 1 部分，共 3 部分)

> 原文：<https://towardsdatascience.com/exploring-the-ml-tooling-landscape-part-1-of-3-53f8c39e6e4e>

## 行业成熟度

![](img/051dbfece731bea3f5f1af2de7d96e92.png)

[附身摄影](https://unsplash.com/@possessedphotography?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

这一系列的博客文章开始时是我试图掌握机器学习操作(MLOps ),然而这个任务很快变成了对机器学习(ML)的采用和工具的广泛回顾。正如我很快发现的那样，MLOps 代表了生产中 ML 的前沿，因此只有相对较少的领先科技公司(想想 FAANG)、前瞻性思维的初创公司以及越来越多的研究论文专门关注这一领域。这就引出了一个问题，为什么在 2022 年，大多数公司的 MLOps 采用以及更一般的 ML 采用都处于早期阶段？我相信解决这个问题对于全面理解 ML 工具的当前状态和不久的将来的趋势是至关重要的。

我在这一系列中要解决的关键问题是，

1.  在工业中，ML 的成熟程度如何？
2.  ML 工具和采用的状态是什么？
3.  ML 工具的可能趋势是什么？

这篇博文关注的是**的第一个**问题。

这一系列的博客文章绝不是详尽无遗的，甚至在某些地方也不一定是正确的！我写这篇文章是为了组织我对最近几周阅读的思考，我希望这能成为进一步讨论的起点。这是一个绝对迷人的领域，我真的很想了解这个行业，所以请联系我们！

# MLOps

MLOps 指的是用于帮助在生产中部署和维护 ML 模型的实践和工具的集合。顾名思义，它是 DevOps 的表亲，devo PS 同样涉及管理软件质量和部署的实践和工具。MLOps 的出现是由于人们越来越意识到在生产中部署和维护 ML 模型的独特挑战，以及意识到在任何部署过程中，ML 特定元素只是必要基础设施的非常小的一部分(Scully，2015)。

与 DevOps 类似，MLOps 代表了行业内的一种文化转变，促进了敏捷实践以及产品和服务的端到端所有权。特别是后一种考虑有助于解释端到端 ML 平台的普遍性，它提供了一系列涉及典型 ML 工作流所有主要组件的服务。ML 平台的效用通常遵循其抽象出较低级别细节的能力(Felipe & Maya，2021)，这意味着这些平台通常部署在托管基础设施层之上，专门用于减轻工程团队的运营负担(Paleyes，2021)。这些平台旨在减少构建和交付模型所需的时间，并保持预测的稳定性和可重复性。看看这个领域中更成熟的公司，我们看到一种在内部开发 ML 平台的趋势(Symeonidis，2022)，这在很大程度上是由于管道实施的高度上下文特定的性质。在下一篇博文中，我们将更加关注内部平台和第三方平台以及工具。

与 DevOps 不同，MLOps 利用三种主要的人工制品:数据、模型和代码(Felipe & Maya，2021)。ML 项目对数据有硬性要求，这不太适合现有的软件管理实践(Paleyes，2021)，特别是数据准备的初始步骤遵循一种更瀑布式的方法(kinen，2021)。此外，每个产品都带来了不同的挑战，并具有不同的开发周期:数据开发周期通常比代码开发周期快得多，因为软件工程编码是最难的部分。不同工件及其伴随需求的组合在某种程度上解释了 MLOps 的复杂性和工具生态系统的规模(Symeonidis，2022)。在流程层面，MLOps 将持续培训(CT)的原则添加到 CI/CD 组合中。

此外，这只可能是我，但围绕 MLOps 术语及其范围的共识存在一定程度的混乱，特别是考虑到 ML 的迭代性质，很难在不同的 MLOps 关注点和数据 Ops 之间划清界限。与此相关，谈论 MLOps 成熟度和 ML 成熟度会变得困难。出于本博客的目的，我将使用“ML 成熟度”来描述 ML 工作流程中所有元素的不断增长的经验、标准化和操作，以及与具体操作方面相关的 MLOps 成熟度，即一旦模型投入生产。

# ML 工作流

在直接讨论 ML 成熟度之前，首先介绍 ML 工作流的概念是有意义的，它是 ML 成熟度的定义元素之一。ML 工作流对应于一组正式的和可重复的过程，用于在生产中开发和部署 ML 模型。尽管这些步骤的具体步骤和编排仍有争议(Brinkmann & Rachakonda，2021)，但(Ashmore，2019)给出了一个经常被引用的工作流程大纲，它强调了关键的高级阶段。

*   数据管理:对应于模型训练、数据采集预处理、扩充、分析等获取数据状态的所有步骤。
*   模型学习:模型训练和优化发生在这里。
*   模型验证:根据各种业务指标和监管框架进行评估。
*   模型部署:包括监控和更新生产模型。

所列的阶段可能被分解成更小、更明确的步骤，并且如前所述，不确定特定的顺序或次序(Paleyes，2021)。请注意，我在这里将术语“工作流”与“生命周期”互换使用，然而后者偶尔用于特别强调模型验证后的一切(Ashmore，2019)。

关于行业中 ML 工作流的具体实施，英伟达、脸书、Spotify 和谷歌的例子突出了“规范的”ML 工作流的新兴共识(Felipe & Maya，2021)。至少在架构层面上，这些差异很大程度上是特定用例以及其他组织关注点的结果，它们可能并不代表整个行业。然而，目前还没有相应的“规范的”ML 技术栈(Jeffries，2020)，其中许多记录的 ML 工作流是用内部工具实现的(Chan，2021)——剧透！

# ML 成熟度框架

有许多不同的框架试图说明不同的 ML/MLOps 成熟度级别，最著名的是谷歌( *MLOps:机器学习中的连续交付和自动化管道*，2020)和微软(*机器学习运营成熟度模型——Azure 架构中心*)。它们非常相似，展示了全面采用 MLOps 的途径如何需要围绕开发、部署和监控流程提高自动化水平。然而，由于这两个框架都只关注事情的运营方面，所以没有一个真正有助于澄清在公司全面采用 ML 的过程中，公司何时应该特别关注运营问题。(Algorithmia，2018)和(kinen，2021)提供了更接近这一点的框架。Algoritmia 白皮书提供了最通用的 ML 定义:“我们将 MLOps 成熟度定义为一个组织通过其 ML 模型推动重大商业价值的能力。”这是根据六个维度来衡量的:组织调整、数据、培训、部署、管理、治理，由(kinen，2021)撰写的论文简单得多，指出提高 ML 成熟度可以理解为经历以下阶段:

*   以数据为中心:关注各种数据管理问题
*   以模型为中心:重点是建立和生产最初的几个模型
*   以管道为中心:在生产中有模型，关注运营问题

只有在“以流水线为中心”的阶段，MLOps 的问题才会得到具体解决。此外，这些阶段之间的运动应该看到相应的组织变化(kinen，2021)。这个框架在整体上更好地记录了 ML 成熟度，但是我要说的是这些类别并没有被很好地命名。具体来说，术语“以数据为中心”和“以模型为中心”通常是指作为一个整体的 ML 工作流的焦点，而不是 ML 采用水平的代表(Strickland，2022)。

总的来说，后两个框架适用范围更广，并强调了以下几点，

1.  ML 成熟度通常伴随着 ML 工作流程的不断进步和效率的不断提高
2.  MLOps 成熟度是 ML 成熟度的延续，在许多方面是 ML 成熟度的最终目标，也就是说，谈论一个而不谈论另一个是没有意义的
3.  只有解决了 ML 成熟度的其他要素，才能实现真正的 MLOps 成熟度

# 行业成熟度

当把这些放在一起时，最令我震惊的是除了技术领导者之外，ML 普遍缺乏成熟和意识；在较大的公司中，约 70%的公司近年来才开始人工智能/人工智能投资(dimensional research，2019)，这些公司通常不是人工智能/人工智能/数据公司。很难推测为什么会出现这种情况，但是尽管大数据、ML 和 AI 已经在商业上存在了十多年(Hadoop 最初是在 2006 年发布的(维基百科))，但它在最近几年才变得对大多数公司实用。这主要是由于越来越多的基于云的数据仓库和湖边小屋，以及足够的工具(Turck，2021)。这可以被理解为一个长期趋势的一部分，正如马特·图尔克所说，“每个公司都不仅仅是一个软件公司，也是一个数据公司。”(图尔克，2021 年)

就普遍采用 ML 而言，一般来说，有两类公司处于 ML 成熟度谱的两个极端:那些刚刚迈出第一步的公司和处于前沿的公司。回到上面讨论的框架，我们可以分别称之为“以数据为中心”和“以管道为中心”。无论如何，就(Ashmore，2019)给出的 ML 工作流而言，大多数公司报告称，他们项目的大部分时间都花在了数据管理阶段(Shankar，2021)。(Paleyes，2021)给出了按 ML 工作流程阶段细分的问题和顾虑的完整列表。

具体到 ML 中不太成熟的公司，针对相对较大(dimensional research，2019)和相对较小(dotscience，2019)组织的一系列调查强调，数据问题是数据项目的主要障碍，96%的受访者遇到了一些数据质量问题(dimensional research，2019)。这些问题包括数据可用性、数据质量和数量、标签和可访问性。其他经常提到的问题突出了适当的工具、专业知识或预算限制的普遍缺乏。整体来看，78%的项目在部署前就停滞了(dimensional research，2019)。

另一个有趣的发现涉及超参数优化，这是模型训练的一个关键步骤:24.6%的受访者完全忽略了这一点，59.2%的受访者手动执行，很少有受访者报告使用第三方工具(dotscience，2019)。这可能再次是这些组织在该领域不成熟的结果，在该领域，从配置空间手动选择更有效，特别是对于早期的 ML 项目。与此相关的是，计算能力是超参数优化的瓶颈，而不是工具或设置(Huyen，2020)。

# 总结

这篇文章讨论了行业中 ML 成熟度的当前状态以及一般的 MLOps。主要的收获是，从整体上考虑行业应用时，ML 采用的不成熟程度，以及全面采用 MLOps 所带来的技术、流程和文化挑战的程度。这两个元素与理解 ML 工具的当前状态高度相关，ML 工具是高度多样化的，并且是一个尚未找到共识的行业的象征。

# 参考

算法 ia。(2018).生产中的机器学习:成功路线图。

阿什莫尔等人(2019 年)。确保机器学习生命周期:需求、方法和挑战。 *ACM 计算调查*， *54* (5)，1–30。

Brinkmann，d .，& Rachakonda，V. (2021，4 月 6 日)。 *MLOps 投资//莎拉·卡坦扎罗//第 33 期咖啡*。YouTube。2022 年 5 月 2 日从[https://www.youtube.com/watch?v=twvHm8Fa5jk](https://www.youtube.com/watch?v=twvHm8Fa5jk)检索

Chan，E. (2021 年 5 月 12 日)。*ML 平台上的课程——来自网飞、DoorDash、Spotify 等等*。走向数据科学。2022 年 4 月 28 日检索，来自[https://towards data science . com/lessons-on-ml-platforms-from-网飞-door dash-Spotify-and-more-f 455400115 C7](/lessons-on-ml-platforms-from-netflix-doordash-spotify-and-more-f455400115c7)

维度研究。(2019).*人工智能和机器学习项目因数据问题受阻*。

网络科学。(2019).*AI 应用的开发运营状况*。

费利佩，a .，&玛雅，V. (2021)。MLOps 的状态。

呼延，C. (2020 年 6 月 22 日)。*看 200 个机器学习工具学到的东西*。[https://huyenchip.com/2020/06/22/mlops.html](https://huyenchip.com/2020/06/22/mlops.html)

杰弗里斯博士(2020 年 10 月 13 日)。*机器学习中典范栈的崛起*。走向数据科学。2022 年 5 月 2 日检索，来自[https://towardsdatascience . com/rise-of-the-canonical-stack-in-machine-learning-724 e 7 D2 FAA 75](/rise-of-the-canonical-stack-in-machine-learning-724e7d2faa75)

*机器学习运营成熟度模型——Azure 架构中心*。微软文档。2022 年 5 月 2 日检索，来自[https://docs . Microsoft . com/en-us/azure/architecture/example-scenario/ml ops/ml ops-maturity-model](https://docs.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model)

Mä kinen，s .等人(2021 年)。谁需要 MLOps:数据科学家寻求完成什么，MLOps 如何提供帮助？[https://arxiv.org/abs/2103.08942](https://arxiv.org/abs/2103.08942)

*MLOps:机器学习中的连续交付和自动化管道*。(2020 年 1 月 7 日)。谷歌云。2022 年 5 月 2 日检索，来自[https://cloud . Google . com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

Paleyes，a .等人(2021 年)。部署机器学习的挑战:案例研究调查。

Scully，d .等人(2015 年)。机器学习系统中隐藏的技术债务。

s . Shankar(2021 年 12 月 13 日)。*现代 ML 监控混乱:对部署后问题进行分类(2/4)* 。史瑞雅·尚卡尔。于 2022 年 4 月 30 日从[https://www.shreya-shankar.com/rethinking-ml-monitoring-2/](https://www.shreya-shankar.com/rethinking-ml-monitoring-2/)检索

斯特里克兰，E. (2022 年 2 月 9 日)。*吴恩达:Unbiggen AI* 。[https://spectrum.ieee.org/andrew-ng-data-centric-ai](https://spectrum.ieee.org/andrew-ng-data-centric-ai)

Symeonidis，g .等人(2022 年)。MLOps——定义、工具和挑战。

m .图尔克(2021 年 9 月 28 日)。*红热:2021 年的机器学习、人工智能和数据(MAD)格局*。[https://mattturck.com/data2021/](https://mattturck.com/data2021/)

维基百科。 *Apache Hadoop* 。维基百科。2022 年 5 月 2 日从 https://en.wikipedia.org/wiki/Apache_Hadoop 检索