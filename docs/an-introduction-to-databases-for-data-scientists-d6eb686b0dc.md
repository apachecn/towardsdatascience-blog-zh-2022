# 面向数据科学家的数据库介绍

> 原文：<https://towardsdatascience.com/an-introduction-to-databases-for-data-scientists-d6eb686b0dc>

## 您需要了解的关于数据库的一切都在一篇文章中

![](img/98865677584de945ce1c5ef183d5626b.png)

照片由[叶小开·克里斯托弗·古特瓦尔德](https://unsplash.com/@project2204?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

数据库是通常存储在计算机系统中的有组织、有结构的信息集合(来源:[甲骨文](https://www.oracle.com/de/database/what-is-database/#:~:text=Eine%20Datenbank%20ist%20eine%20organisierte,einem%20Datenbankverwaltungssystem%20(DBMS)%20gesteuert.))。数据库的操作和管理通常发生在数据库管理系统(DBMS)中。

# 什么是数据库？

在数据库中，大量数据通常以结构化的方式存储，并可供检索。这几乎总是一个电子系统。然而，从理论上讲，模拟信息收集，如图书馆，也是数据库。

早在 20 世纪 60 年代，就出现了对集中式数据存储的需求，因为像数据访问授权或数据验证这样的事情不应该在应用程序中完成，而应该与应用程序分开。

# 什么是数据库管理系统？

数据库由两个主要部分组成。一个是实际的数据存储，一个是所谓的数据库管理系统(简称 DBMS)。简单地说，它充当数据和最终用户之间的接口。MySQL 是 Oracle 数据库管理系统的一个具体例子。

数据库管理系统的中心任务包括，例如:

*   数据的存储、修改和删除
*   数据模型的定义和符合性
*   添加用户并创建相应的权限

该管理系统进一步确保所谓的 ACID 属性在数据存储中得到维护。其中包括以下几点:

*   **原子性(A)** :数据事务，例如新数据记录的输入或旧数据记录的删除，要么完全执行，要么根本不执行。对于其他用户，事务只有在完全执行后才可见。例如，在金融机构的数据库中，从一个帐户到另一个帐户的转帐只有在两个表中的交易都完全执行时才可见。
*   **Consistency (C)** :当每个数据事务将数据存储从一致状态移动到一致状态时，满足该属性。
*   **隔离(I)** :当多个事务同时发生时，最终状态必须与事务分别发生时相同。也就是说，数据库应该通过压力测试。换句话说，它不应该由于过载而导致不正确的数据库事务。
*   **持久性(D)** :数据只能因交易而改变，不能因外部影响而改变。例如，软件更新不得无意中导致数据更改或可能被删除。

[](/database-basics-acid-transactions-bf4d38bd8e26) [## 数据库基础:ACID 事务

### 了解数据库的 ACID 属性

towardsdatascience.com](/database-basics-acid-transactions-bf4d38bd8e26) 

# 数据库有哪些类型？

有许多不同类型的数据收集，这也主要取决于组织或公司内的使用类型。各种影响因素都在发挥作用，例如潜在用户和数据查询的数量，以及要存储的数据类型:

*   **关系数据库**:这是存储数据的地方，这些数据可以以表格的形式存储，即以行和列的形式存储。
*   **分布式数据库**:如果数据要存储在几台不同的计算机上，这就叫做分布式数据库。这很有用，例如，如果您想使数据收集无故障，或者如果您需要处理大量的数据查询。
*   [**数据仓库**](https://databasecamp.de/en/data/data-warehouses) :如果数据要在公司内部集中访问，这就称为数据仓库。在这里，来自不同源系统的数据被存储并形成统一的数据形式。
*   [**NoSQL 数据库**](https://databasecamp.de/en/data/nosql-databases) :如果要存储的数据不对应于关系模式，例如在非结构化数据的情况下，它被存储在所谓的 NoSQL(“不仅是 SQL”)数据集合中。

这些只是一些最常见的数据库类型。随着时间的推移，出现了更多的类型，但是我们不能在本文中详细讨论它们。最常见的数据库类型是关系数据库和 NoSQL 数据库。

[](/comprehensive-guide-to-data-warehouses-6374617f45d5) [## 数据仓库综合指南

### 您需要知道的一切，包括与数据湖的比较

towardsdatascience.com](/comprehensive-guide-to-data-warehouses-6374617f45d5) 

# SQL 和 NoSQL 的区别是什么？

关系数据库存储数据，这些数据组织在具有列和行的表中。通常，它用于组织中的许多应用程序，如存储销售数据、客户信息或仓库中的当前库存。这些数据库可以通过 SQL 语言查询，并且它们满足所介绍的 ACID 属性。然而，该数据库只能在一台设备上实现，这意味着如果需要更多的存储，则必须改进该计算机的硬件，这通常更昂贵。

NoSQL 的原理(“不仅是 SQL”)最早出现在 2000 年代末，泛指所有不在关系表中存储数据、查询语言不是 [SQL](https://databasecamp.de/en/data/sql-definition) 的[数据库](https://databasecamp.de/en/data/database)。除了 [MongoDB](https://www.mongodb.com/) 之外，NoSQL 数据库最著名的例子还有 [Apache Cassandra](https://cassandra.apache.org/_/index.html) 、 [Redis](https://redis.io/) 和 [Neo4j](https://neo4j.com/) 。

由于其结构的原因，NoSQL 数据库的可伸缩性远远高于传统的 SQL 解决方案，因为它们还可以分布在不同的系统和计算机上。此外，大多数解决方案都是开源的，支持关系系统无法覆盖的数据库查询。

有关 NoSQL 数据库的更多信息，请查看我们关于该主题的文章:

[](/introducing-nosql-databases-with-mongodb-d46c976da5bf) [## 使用 MongoDB 介绍 NoSQL 数据库

### NoSQL 数据库实用指南

towardsdatascience.com](/introducing-nosql-databases-with-mongodb-d46c976da5bf) 

# 数据库挑战

如果在组织中引入大型数据仓库，管理员将面临各种各样的挑战。创建数据集合时，应考虑以下几点:

*   **增加数据量的能力**:由于公司内部生成和存储的数据量不断增加，系统必须有足够的资源来扩展数据量。
*   **数据安全**:当部分机密信息存储在中央位置时，自然会成为未经授权访问的目标。这不仅包括防止外部访问，还包括为组织内的用户分配权限。
*   **可扩展性**:随着公司的发展，信息量自然也会增长。数据库解决方案应该为此做好准备，并且能够处理更多的用户查询和数据。
*   **数据时效性**:当今世界，我们习惯了无延迟地接收信息，同样的道理自然也适用于数据存储。因此，必须构建能够尽快处理和提供信息的体系结构。

# 谁使用结构化查询语言？

[结构化查询语言(SQL)](https://databasecamp.de/en/data/sql-definition) 是处理关系数据库时最常用的语言。不管它的名字是什么，这种语言不仅仅可以用于简单的查询。它还可以用于执行创建和维护数据集合所需的所有操作。

SQL 提供了许多读取、修改或删除数据的函数。它实际上用在所有常见的关系数据库系统中，并且应用广泛。此外，非关系系统还提供了扩展，因此即使数据没有排列在表中，也可以使用查询语言。这可能是因为 SQL 提供了许多优势:

*   它在语义上非常容易阅读和理解。即使是初学者也能在很大程度上理解这些命令。
*   这种语言可以直接在数据库环境中使用。对于信息的基本工作，数据不必首先从集合转移到另一个工具。
*   简单的计算和查询可以直接在数据集合中进行。
*   与其他电子表格工具(如 Excel)相比，使用结构化查询语言的数据分析可以很容易地复制和拷贝，因为每个人都可以访问集合中的相同数据。因此，相同的查询总是导致相同的结果。

在我们的博客中，我们提供了一篇关于结构化查询语言的详细文章:

[](https://databasecamp.de/en/data/sql-definition) [## 结构化查询语言(SQL) |数据库

### 结构化查询语言(SQL)是处理关系数据库时最常用的语言。语言…

数据库营](https://databasecamp.de/en/data/sql-definition) 

# 为什么数据库对数据科学家如此重要？

如果您读到这里，您可能会想，既然有像数据工程师这样的同事在做数据库方面的工作，为什么数据科学家应该了解数据库呢？然而，这只是部分正确。在大多数公司中，不可能填补两个单一的职位，即数据科学家和数据工程师。因此，即使作为一名数据科学家，您也应该具备数据库的基础知识。

然而，另一点更重要:几乎所有用作评估来源的数据都来自数据库。因此，数据库决定了数据科学家如何获得他需要的信息。例如，查询语言、数据结构，以及数据是否已经准备好以及如何准备好。所有这些信息对数据科学家来说或多或少都很耗时，因此对他来说至关重要。

# 这是你应该带走的东西

*   数据库是一个系统，用于以有组织和结构化的方式收集信息。
*   关系存储系统仍然是最常见的。然而，NoSQL 解决方案或数据仓库也越来越受欢迎。
*   在创建这样的数据集合时，需要考虑许多不同的挑战，比如可伸缩性或数据安全性。
*   对于查询和维护数据库，结构化查询语言(SQL)在许多情况下仍然被使用。

*如果你喜欢我的作品，请在这里订阅*[](https://medium.com/subscribe/@niklas_lang)**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，medium 允许你每月免费阅读* ***3 篇*** *。如果你希望有****无限制的*** *访问我的文章和数以千计的精彩文章，不要犹豫，点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格**

*[](/beginners-guide-to-gradient-descent-47f8d0f4ce3b) [## 梯度下降初学者指南

### 关于梯度下降法你需要知道的一切

towardsdatascience.com](/beginners-guide-to-gradient-descent-47f8d0f4ce3b) [](/the-difference-between-correlation-and-causation-51d44c102789) [## 相关性和因果关系的区别

### 你需要知道的关于偶然推断的一切

towardsdatascience.com](/the-difference-between-correlation-and-causation-51d44c102789) [](/beginners-guide-extract-transform-load-etl-49104a8f9294) [## 初学者指南:提取、转换、加载(ETL)

### 了解数据分析中的大数据原理

towardsdatascience.com](/beginners-guide-extract-transform-load-etl-49104a8f9294)*