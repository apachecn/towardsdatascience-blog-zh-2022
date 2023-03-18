# 弹性研究导论

> 原文：<https://towardsdatascience.com/an-introduction-to-elasticsearch-19f081380d14>

## 从弹性搜索开始你需要知道的一切

![](img/b95431fa1663caf861f64a5858c0edd3.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上 [Marten Newhall](https://unsplash.com/@laughayette?utm_source=medium&utm_medium=referral) 拍摄的照片

Elasticsearch 是一个基于 [Apache Lucene](https://lucene.apache.org/) 的分布式搜索引擎。它是一个流行的搜索引擎，用于全文搜索或日志分析，因此被许多大公司使用，如网飞、Slack 和优步。

# Elasticsearch 是如何工作的？

这个搜索引擎基于这样一个事实，即原始数据和要搜索的文档被组合在一个索引中。为此，它们在索引步骤中被规范化和解析，以便最终的搜索可以运行得更快。这种预处理的索引可以比原始文档和数据更快地被搜索。

假设我们有一个网上商店，里面有各种各样的家具，我们想让我们的顾客通过搜索就能找到它们。对于每件家具，都有在搜索中应该考虑的信息。其中包括产品属性(如尺寸、颜色或特殊功能)和产品描述。为了确保可以快速搜索这些文本属性，我们使用全文搜索 Elasticsearch。

为此，我们必须将数据和文档存储在其索引中。这可以被认为是一个存储所有信息的数据库。在索引中，有几个所谓的类型，可与数据库中的表相比较。

在我们的示例中，只有产品属性可以存储在一种类型中，而产品描述存储在另一种类型中。在这些类型中，最后还有保存单个数据记录的文档。尽管索引不是一个严格的关系数据库，但仍必须保持一定的结构，以便快速搜索。

在我们的例子中，每件家具都在文档中准备好，并存储在一个结构中，以便在最后的搜索中更容易找到它们。为此，所谓的属性是在文档中定义的，它最接近表中的列。

# 你用 Elasticsearch 做什么？

任何需要搜索功能的地方都可以使用 Elasticsearch。此外，它因其高可伸缩性和快速搜索过程而脱颖而出。可以考虑的应用有:

*   在网站上搜索
*   应用程序中的搜索引擎
*   企业数据搜索引擎
*   搜索日志文件
*   在地理数据中搜索
*   在安全和监控文件中搜索

# Elasticsearch 有哪些组成部分？

由于它的广泛使用和许多好处，围绕弹性搜索已经形成了一整套工具，不仅仅是搜索。

在 **Logstash** 的帮助下，可以收集和准备数据，以便更好地适用于后续索引。开源程序可以被理解为 Elastic 的 [ETL](https://databasecamp.de/en/data/etl-en) 工具，它通过将不同来源的数据汇集在一起，进行转换，并将其带到最终的存储位置来提供类似的功能。

弹性搜索的下游工具是 Kibana。它提供了可视化和分析来自搜索索引的信息的可能性。因此，这种所谓的 ELK (Elastic，Logstash，Kibana)堆栈提供了覆盖从获取数据到分析指数的完整范围的可能性。

# Elasticsearch 有什么好处？

Elasticsearch 是当今非常流行的搜索引擎，因为它有很多优点。其中一些是:

*   **速度**:由于索引的原因，它比同类算法要快得多，尤其是在全文搜索方面。此外，预备索引也不需要很长时间，这意味着从包含在索引中到在搜索中可找到的整个过程非常快。这对于搜索速度是重要标准的应用来说是非常有利的。
*   **分布式架构**:索引分布在不同的物理机器上，称为碎片。还会创建单个文档的副本，以弥补单台机器的故障。这种集群结构允许扩展搜索的性能。
*   **其他功能** : Elasticsearch 还提供许多其他功能，有助于确保搜索性能非常高。例如，这些包括数据汇总或索引生命周期管理。
*   **业务分析**:已经描述的组件提供了可视化和处理索引或已处理数据的可能性。这提供了一个整体的方法。

# Elasticsearch 有什么缺点？

尽管 Elastic 的搜索算法有巨大的优势，但在实施之前，也有一些问题需要考虑和权衡:

*   搜索并非与所有商店系统和基础设施兼容。
*   使用自托管服务器，实现 Elasticsearch 会变得非常昂贵和复杂。
*   索引在整个集群中的分布有利于扩展，但也可能很快成为缺点。如果使用了太多所谓的主碎片，并且索引分布在许多机器上，就会出现这种情况。因此，当索引一个新文档时，所有这些机器都必须处于活动状态，这导致仅索引一项就给系统带来很高的负载。

# 这是你应该带走的东西

*   Elasticsearch 是一个针对各种应用程序的流行全文搜索。
*   基本原则是索引数据，从而使搜索算法更容易、更快地找到数据。
*   这种搜索算法的特点是处理速度快，并且索引可以划分到一个计算机集群中，因此是可伸缩的。

*如果你喜欢我的作品，请在这里订阅*[](https://medium.com/subscribe/@niklas_lang)**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，medium 允许你每月免费阅读* ***3 篇*** *。如果你希望有****无限制的*** *访问我的文章和数以千计的精彩文章，请不要犹豫，点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格**

*[](/comprehensive-guide-to-principal-component-analysis-bb4458fff9e2) [## 主成分分析综合指南

### 主成分分析的理论解释

towardsdatascience.com](/comprehensive-guide-to-principal-component-analysis-bb4458fff9e2) [](/why-you-should-know-big-data-3c0c161b9e14) [## 为什么您应该了解大数据

### 定义大数据及其潜在威胁

towardsdatascience.com](/why-you-should-know-big-data-3c0c161b9e14) [](/what-are-deepfakes-and-how-do-you-recognize-them-f9ab1a143456) [## 什么是 Deepfakes？

### 如何应对人工智能制造的误传

towardsdatascience.com](/what-are-deepfakes-and-how-do-you-recognize-them-f9ab1a143456)*