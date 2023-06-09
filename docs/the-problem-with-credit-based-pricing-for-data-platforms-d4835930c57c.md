# 数据平台基于信用的定价问题

> 原文：<https://towardsdatascience.com/the-problem-with-credit-based-pricing-for-data-platforms-d4835930c57c>

![](img/3fee54d740e90519b752f4a2e8a68003.png)

安妮·斯普拉特在 Unsplash 上的照片

当您选择构成现代数据基础设施的平台和系统时，定价模式是一个需要考虑的*主要*因素。

**你的数据仓库和数据管道每天都要处理大量的数据，它们可以很快地决定你的预算。准确评估如何以及为什么向你收费是一个关键的计划步骤。**

可以说，定价对话没有得到应有的关注。相反，花言巧语经常集中在系统的性能、闪亮的新功能以及在我们的数据堆栈中使用它的“正确”方式上。(数据厂商对这种重定向尤其心虚。)

归根结底，您的数据的工作是推动您的业务底线。如果您必须每年向供应商支付数十万甚至数百万美元才能获得您需要的数据结果，那该怎么办？

嗯，你实际上可能根本没有达到你的商业目标。

在本文中，我们将研究数据仓库和数据管道平台的不透明定价模型的流行，以及这是如何造成问题的。在此过程中，我们将分解数据平台定价的其他方面:基于计算、基于存储以及其中的变化。

通过了解当今市场上流行的定价模式，您将能够更明智地选择您的供应商。即使你最终选择了一款不太好的产品，你也会知道会发生什么，并在合同谈判中做好更充分的准备。

# 通过数据仓库定价了解市场

数据仓库定价模型比数据管道定价模型讨论得多，但两者密切相关。

通常，数据管道的主要功能是将数据移入仓库。因此，这两个系统将由相似的客户购买，并处理相似数量的数据。

这个目标客户？大部分是大型企业，加上一些中小型企业。

数据量？巨大的。TB 级，并且还在快速增长。

这是关键。**数据仓库(和相关技术)的目标市场是已经拥有大量数据的公司，随着时间的推移，他们将继续获取和使用越来越多的数据。**

2021 年汽水数据和存储趋势[调查了企业的总数据增长。](https://www.linuxfoundation.org/wp-content/uploads/LFResearch_SODADataStorageTrends_Report_102821.pdf)它发现:

> *“根据 62%的样本报告，主流年数据增长率在 1-100tb 之间。但是，样本中有 9%的数据年增长率为 1PB 或更高。这是主流增长的 10 到 100 倍，很可能预示着许多企业将在几年内找到自己的位置。”*

当然，这是总数据，而不是存储在仓库中的数据，但它表明了行业的整体趋势。调查还发现，46%的受访者“一直”在数据仓库上运行工作负载

数据仓库是[现代数据栈的基石，](https://www.estuary.dev/2022/03/29/understanding-the-modern-data-stack-and-why-open-source-matters/)和[的研究也表明](https://reports.valuates.com/market-reports/ALLI-Auto-2Q343/data-warehousing)数据仓库的销售正在蓬勃发展。各行各业的公司*需要*数据仓库来支持高级分析、低延迟视图和运营分析等功能，这些功能正迅速变得不可或缺。

厂商对此心知肚明。他们还知道，一旦您迁移到某个产品，您就会被锁定—您依赖于该产品并围绕它构建您的基础架构。从长远来看，它们的潜在利润取决于它们如何构建定价模型。随着您的总数据量的增长，以及您使用的数据源和数据工具数量的增加，您的仓库和管道账单的增长可能会超出您的预期。

但是一旦你被锁定，你更有可能忍受不断增长的账单，而不是迁移到不同的平台。

# 计算与基于量的数据仓库定价模型

考虑到这一点，我们来看看一些常见的数据仓库定价模型。

您的数据仓库账单通常包含两个部分:**存储**和**计算。**

存储定价非常简单:不可否认的是数据量的问题。您可以测量存储数据的字节数，并据此收费。这是意料之中的事，所以我们先把这个话题放在一边。

另一方面，对仓库*计算*的收费是事情变得有趣的地方。

数据仓库存储数据的方式[是为分析查询性能](https://www.estuary.dev/2021/11/02/database-vs-data-warehouse-vs-data-lake-key-differences-and-usage/)而设计的。你特意把它放在那里，这样你就可以查询它，而这些查询是有成本的。

仓库供应商可以通过两种主要方式向您收取计算费用:

*   通过运行查询在**卷上扫描了**个数据。
*   在**上实际计算**用于运行查询**。**

# BigQuery:按扫描数据量收费

Google BigQuery [按查询处理期间读取的 TB 数](https://cloud.google.com/bigquery/pricing#on_demand_pricing)收费。这是许多云提供商使用的一种简单、可预测的模型。

但是按数据量收取查询费用给谷歌制造了一个难题。如果谷歌的一个工程团队找到了一种让他们的查询更有效的方法——这通常是任何查询技术的目标——这实际上可能对他们的底线不利。

事实上，一些用户已经注意到[在 BigQuery](https://blog.devgenius.io/why-is-snowflake-so-expensive-92b67203945#:~:text=idea%20to%20let-,BigQuery,-perform%20a%20full) 中查询时的有趣行为，这些行为会导致它不必要地扫描整个数据集，但这些行为在生产中可能无法解决。

# 雪花:按计算收费…按“信用”

有人可能会说，根据供应商用来运行您的查询的实际*计算资源*来收费更有意义。毕竟，供应商是为支持您消费的计算资源而付费的，因此将定价与此挂钩可能会鼓励更公平的模式，而不会积极鼓励公司降低运营效率。

重要的是，公司在量化这些计算资源实际上是什么时要透明。

公平地说，衡量计算比衡量数据量要复杂一些。[它包含许多因素](https://aws.amazon.com/what-is/compute/)，包括处理能力、网络和内存。但这并不能免除供应商记录其定价结构的技术细节的责任。

让我们看一个例子。数据仓库供应商 Snowflake 在最近几年获得了巨大的利润。一个特别有趣的指标是其净收入保持率——根据 Q1 2022 年的估计，雪花[仅来自现有客户](https://cloudedjudgement.substack.com/i/62699354/net-revenue-retention) 的收入同比增长**174%，远远超过其同行。**

当然，这在很大程度上与产品质量和前面提到的企业数据量的增长有关。尽管如此，像这样的统计数据还是令人吃惊，雪花因其定价受到了一些审查。

雪花收费通过一个称为“信用”的单位来计算信用有点像垄断货币。它们与现实世界中任何可量化的事物都没有直接联系:比如说，所使用的硬件。这种缺乏透明度的情况引发了一些危险信号。客户如何才能真正知道他们是否为雪花公司正在做的事情支付了合理的价格？(他们不能。)

现在，我们已经了解了简单的基于数量的定价和基于信用的数据仓库定价，让我们将这些原则应用于数据管道定价。

# 基于行的数据管道定价

与仓库不同，计算与数据管道定价并不真正相关。几乎所有的供应商都根据数据量收费。尽管数据量是一个相对直接的指标，但许多定价结构会带来复杂性，正如我们在雪花的信用模型中看到的那样。

很多厂商不是按纯量收费，而是按**排收费。**这有一定的意义——数据管道读取的数据行数很容易预测和估计，大多数[批处理管道](https://www.estuary.dev/2021/09/07/real-time-and-batch-data-processing-an-introduction/)工具都是按照行来考虑数据的。

包括 [Fivetran](https://www.fivetran.com/pricing) 、 [Airbyte](https://airbyte.com/pricing) 和 [Stitch Data](https://www.stitchdata.com/pricing/) 在内的流行厂商使用基于行的定价的变体。

通过基于行的定价，您可以安全地预测您接收的数据量和运行管道的价格之间的线性关系。但是你可能无法预测那段关系的细节。

这是因为基于行的定价是数量的代理。基于行的数据管道定价的一些问题包括:

*   并非所有行的大小都相同。
*   对于较小的数据源(如 SaaS 应用程序)，总体数据量通常非常小；使用行为供应商提供了回旋余地，可以对每个小的集成收取更多的费用。
*   数据源系统以不同的方式对数据建模。有些不是基于行的；数据管道将这些数据重组为行的方式带来了另一层复杂性。

行虽然看起来是一种简单的数据量计费方式，但实际上并没有看起来那么透明。

除此之外，许多供应商在这种定价模型的基础上增加了额外的抽象，通常是——你猜对了！—学分。

# 数据管道基于数量的定价

当任何数据供应商对代理收费时，比如 rows 或 credits，作为消费者，您就失去了一些代理权。这并不一定意味着你会被占便宜，但这*意味着*你有责任询问你的销售代表，以确定*确切地*你将支付什么。

你应该可以随意协商，尤其是当你要介绍一个拥有大量数据的大客户时。

然而，最理想的情况是找到一个数据管道平台，依靠*纯*数据量收费。

当你按纯体积充电时:

*   您不会为在管道中添加许多更小的数据系统而多付钱。
*   无论您的源数据是如何建模的，您都可以更容易地预测您将支付的费用。
*   你不需要非常警惕来确保你没有被多收了钱。

此时，您可能想知道:什么是数据管道中每单位数据量的合理价格？

这是一个很难回答的问题:这取决于市场，并且会随着时间的推移而不断变化。

它还取决于供应商的利润率，而利润率反过来又与他们提供的管道架构是否高效和高度可伸缩有关。

在理想的情况下，你的供应商可能希望降低自己的运营成本。如果它这样做了，它可以将这些节省下来的钱转给你，同时仍能盈利，并因其合理的定价而赢得更多的客户。

在任何情况下，您的供应商绝对应该提供大量折扣，这些折扣将随着您的数据量的增长而生效。

# 如何评价大数据工具的价格

在购买数据仓库和数据管道工具时，最好的办法是从多家供应商那里获取报价并进行比较。

当您这样做时，需要考虑以下四点:

小心小妖精。

正如我们在讨论数据仓库定价时所展示的，几乎任何定价模型都会鼓励供应商对系统中的某些低效率视而不见。

当然，目标应该是与那些你信任的拥有超级可靠产品的公司做生意，但你应该始终保持警惕。

最终，B2B 客户-供应商关系可以是积极的和互利的。但这需要双方认真谈判、诚实和关注细节。

关于这一点:

**避免不透明的定价模式**

您的数据账单肯定会随着时间的推移而增长，因为您的数据会随着时间的推移而增长。通过只同意透明和可扩展的定价模式来创建最佳方案。避免增加复杂层次的模型(如积分或行),这可能会掩盖您实际支付的费用。

随着时间的推移，继续谈判…

**寻求大批量交易。**

它们就在那里，最好的可能没有被宣传。

最后，如果你从这篇博文中只拿走*一件*东西，应该是这个:

**在评估定价时，现实地考虑一下您数据的未来。**

您的组织现在可能拥有大量数据。T4 变少的可能性极小。因此，问问你自己:定价模式是否以有利于你或供应商的方式扩展？

*本帖原载于* [*河口博客。*](https://www.estuary.dev/blogs/)