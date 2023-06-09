# 在这个问题上投入更多的计算

> 原文：<https://towardsdatascience.com/throwing-more-compute-at-the-problem-1ceec0cb178a>

![](img/931ac55ed0e89df1df41688f0f99e2d8.png)

奥利维尔·科莱在 [Unsplash](https://unsplash.com/s/photos/processor?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

## 为什么这不是一个长期或短期的好策略

# 背景

是的，如果可能的话，这是查询编写人员在面对他们编写了一个糟糕的查询这一事实时通常采取的第一种方法。有时，他们有能力获得更多的计算来解决问题。大多数时候，不是的。在过去的十年里，我已经为多个产品和公司设计和扩展了数据库和数据仓库，我已经非常详细地理解了这个问题。在这篇文章中，我将分享我的观察，并提出一些解决这个问题的好策略。

在我们深入研究解决方案之前，我想从我自己的工作中给你一些例子，你可能会很容易联系到。这会帮助你更好地理解这个问题。考虑以下场景:

😕MySQL 混合使用本地和第三方复制器，具有复杂的复制拓扑。数据库托管在 EC2 实例上(这不是最近的；RDS 当时还不流行)。对于您所拥有的负载，这些实例是过度调配的。尽管如此，当有人查询终端从属数据库实例时，他们有时会由于负载峰值而停止复制。由于各种各样的锁，它们还会阻止其他想要查询数据库的用户。这直接影响了业务。

😞你刚刚加入了一个团队。该团队已经建立了一个红移“数据仓库”。他们实际上从关系数据库、第三方集成 API 和应用程序流中转储所有数据。将数据转储到 Redshift 中的过程非常有效，但是当分析师在这些数据上编写查询，连接所有数据源时，这一过程似乎不起作用。为什么？嗯，将所有数据转储到一个允许数据仓库的平台上并不会自动使转储成为一个数据仓库。一半的表没有正确的分布和排序键。他们中的一些有巨大的倾斜。*了解为了什么目的使用什么工具也是非常重要的。*

</the-new-data-engineering-stack-78939850bb30>  

😭你有很多数据。其中大部分来自结构化来源。您希望通过 Spark 利用并行处理。你给你的团队一个 Spark EMR 集群。集群被设置为自动扩展，但这是有上限的。您的数据在从中提取数据的对象存储中被正确分区，但是仍然有一点倾斜。您使用 Python 和 SQL 的组合来到达您想要的地方，但是您注意到您在执行查询上比在小得多的关系数据库上花费更多的时间。这怎么可能呢？

让我们一个接一个地看看这些例子。

# 识别问题

乍一看，基础架构似乎配置不足，但仔细观察，您可以清楚地从这些示例中发现几个问题:

*   对于第一个例子，有一个查询优化问题需要尽快解决。
*   在第二个例子中，有一个术语“数据仓库”的误用和对查询性能成功的错误期望。
*   在第三个例子中，还有一个查询优化问题需要解决。

这只是三个例子。我相信你们中的许多人已经看到了更多。

# 一些问题和建议

通过询问以下问题，您可以做一些简单的事情来解决这个问题:

*   *查询数量/强度的激增是否导致了该问题？*如果是，并且您的系统大部分时间都没有得到充分利用，那么最好转移到无服务器基础设施。
*   *你真的需要数据仓库吗？*如果是的话，如果您构建一个而不是将红移垃圾站称为数据仓库不是更好吗？那是不诚实的工程。
*   *通过查看执行计划来分析查询有帮助吗？*是的，可以，而且确实如此。不管您使用什么样的数据库技术，您都可以选择查看查询优化器，看看数据库是如何执行查询的；这将告诉你什么需要修理。

# **上投多算**

对上述问题投入更多的计算不会有多大帮助。有时候会，但是不会值这个钱。如果您的查询很糟糕，或者您使用了错误的数据库来解决问题，数据库系统甚至不会记录计算(资源)的增加。它们会吃掉任何它们遇到的东西。

你可能会问，什么时候投入更多计算是好的？答案并不简单，但也不复杂。您需要同时考虑几个因素，比如数据库配置、计算和内存利用率、数据量、并发用户、数据访问模式(尤其是读取模式)、索引(和/或分区)等等。在某种程度上，即使在短期内，投入更多的计算可能是最糟糕的解决方案，因为它也可能无法解决糟糕的查询问题。

</the-art-of-discarding-data-4948ae3b3d14>  

# 利用可观察性了解更多信息

市场上的许多工具可以让你了解这些事情。一个例子是 NewRelic 和 Percona 工具包的组合。据我所知，这将是应用程序可观察性和数据库性能管理工具的结合，它将为您提供数据库发生情况的完整视图。如果您将数据库托管在 AWS 这样的云平台上，那么您可以使用 RDS Performance Insights 这样的工具来查看数据库。这就是为什么使用云数据库进行扩展更容易一些，因为它们从一开始就打包了一些支持工具。

</numbers-every-data-engineer-should-know-cc5c1a0bc3ec>  

# 结论

一旦您对数据库性能有了更全面的了解，您就能更好地判断是否需要为数据库提供更多的功能！

如果你觉得我的文章有用，请订阅并查看我的文章🌲 [**Linktree**](linktree.com/kovid) 。你也可以考虑用我的推荐链接购买一个中级会员来支持我。

<https://kovidrathee.medium.com/membership> 