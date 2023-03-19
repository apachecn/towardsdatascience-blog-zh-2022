# Redis:内存数据存储很容易解释

> 原文：<https://towardsdatascience.com/redis-in-memory-data-store-easily-explained-3b92457be424>

## 了解基于键值的 NoSQL 数据存储

![](img/f060bb553dc4ea1e6badd281fa27d4ca.png)

由 [Khadeeja Yasser](https://unsplash.com/@k_yasser?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

Redis 是一个基于键值的 [NoSQL](https://databasecamp.de/en/data/nosql-databases) 数据库，它将数据存储在内存中，即 RAM 中。这个数据存储是当今最常用的键值数据库之一，例如用于缓存。缩写代表**Re**mote**Di**dictionary**S**server。

# 什么是 NoSQL 数据库？

[NoSQL](https://databasecamp.de/en/data/nosql-databases) (“不仅是 SQL”)描述了与 [SQL](https://databasecamp.de/daten/sql) 不同的数据库，它们是非关系型的，也就是说，不能被组织在表中，等等。这些方法也可以分布在不同的计算机系统上，因此具有高度的可扩展性。NoSQL 解决方案非常适合许多大数据应用。

[数据库](https://databasecamp.de/en/data/database)的特点在于两个特别的标准，这两个标准非常宽泛。首先，数据不是存储在表中的，其次，查询语言不是 SQL，这一点从名称“不仅是 SQL”也可以看得很清楚。

Redis 属于所谓的**键值存储**，这是 NoSQL 数据库中的一个类别。它们是一种非常简单的数据结构，其中每条记录都存储为一个具有唯一键的值。使用该键，可以专门检索信息。

# Redis 是如何工作的？

Redis 被开发为具有可靠的数据存储，可以在短时间内存储和输出信息。这个数据库的特别之处在于已经提到的**键值存储**和**内存存储**的组合。

内存数据库将数据存储在计算机的随机存取存储器(RAM)上，而不是将其写入硬盘，如 HDD 或 SSD。这大大加快了读写的速度，但代价是数据的安全性和成本。RAM 通常比类似的硬盘存储更贵，当机器关机或系统崩溃时，RAM 会被完全擦除。

该存储器中的每个条目被分配一个唯一的密钥，该密钥可用于访问数据记录。由于计算机的工作内存通常是一种有限的资源，因此必须小心使用。这也包括使用消耗尽可能少的内存的特殊数据结构。

在大多数情况下，字符串是作为数据结构使用和存储的。此外，Redis 还可以处理其他数据类型(参见 [IONOS (2020)](https://www.ionos.com/digitalguide/hosting/technical-matters/what-is-redis/) ):

*   **字符串**:最大内存 512 MB 的字符串。
*   **散列**:散列表示和相关字符串之间的映射。
*   **列表**:存储在列表中的字符串集合。
*   **位图**:布尔值的紧凑表示。
*   **Streams** :专门为 Redis 开发的一种日志文件。

# Redis 用于哪些应用？

尽管 Redis 的使用案例非常有限，但由于其优越的属性，它们很难在这一领域被取代。如前所述，Redis 主要用于缓存，比如在 Twitter 上。缓存通常被理解为保存中间状态，以便将来的查询可以运行得更快。例如，在 Twitter 的情况下，这可能意味着已经加载的个人资料图片或推文会保存在缓存中，以便在再次查询时可以更快地获得它们。

这些功能在聊天或消息服务中尤其有利，因为新消息可以几乎实时地发送给用户。

# Redis 有什么优势？

与传统的关系数据库相比，NoSQL 数据库有几个优点。这些优势包括更好的性能和跨多个系统的大数据量和分布。Redis 还可以利用以下优势进行评分:

*   通过内存中的内存快速访问
*   支持最常见的编程语言
*   借助各种工具，Redis 提供了高度的用户友好性
*   数据也可以分布在多个集群中，并保存在其他计算机的内存中
*   开源

# 使用 Redis 的缺点是什么？

与大多数数据库一样，使用 Redis 有一些缺点，在实施之前必须权衡这些缺点。除了系统崩溃时数据丢失的风险之外，以下几点也很重要:

*   主存储器是一种昂贵的硬件组件
*   数据只能通过密钥访问
*   更复杂的数据集很难绘制

# 这是你应该带走的东西

*   Redis 是在内存中存储数据的键值存储。
*   快速的读写过程使数据库非常适合缓存应用程序或存储聊天消息。

*如果你喜欢我的作品，请在这里订阅*[](https://medium.com/subscribe/@niklas_lang)**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，medium 允许你每月免费阅读* ***3 篇*** *。如果你希望有****无限制的*** *访问我的文章和数以千计的精彩文章，不要犹豫，点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格**

*[](/comprehensive-guide-to-principal-component-analysis-bb4458fff9e2)  [](/why-you-should-know-big-data-3c0c161b9e14)  [](/what-are-deepfakes-and-how-do-you-recognize-them-f9ab1a143456) *