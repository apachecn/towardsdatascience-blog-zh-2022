# 如何构建您的 dbt 项目和数据模型

> 原文：<https://towardsdatascience.com/dbt-models-structure-c31c8977b5fc>

## 对 dbt 数据模型实施有意义的结构

![](img/658d061a20cad10b300c36e04e6b5865.png)

阿兰·范在 [Unsplash](https://unsplash.com/s/photos/structure?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

数据模型是表示应用程序域中特定对象的抽象结构。这种模型可以是特定组织的用户、产品和订单。在现代组织中，应该制定数据驱动的决策，因此能够高效和有效地管理数据模型非常重要。

随着公司收集和处理的数据量不断增长，需要维护数百甚至数千个数据模型。这意味着，考虑到一些模型可能是其他数据模型的上游或下游依赖关系，也需要管理它们的相互依赖关系。

在蓬勃发展的数据行业中，大量的工具可供数据专业人员和团队使用。数据构建工具(dbt)是由 Fishtown Analytics 开发的开源工具，无疑是最强大的工具之一，我强烈推荐给每个团队，如果他们对现代组织中可扩展的有效数据管理感兴趣的话。 **dbt 可以帮助数据团队创建数据集和模型，作为分析、报告、机器学习建模和一般数据工作流的一部分**。

> dbt 是一个开发框架，它将模块化 SQL 与软件工程最佳实践结合起来，使数据转换变得可靠、快速和有趣。- dbt [文档](https://www.getdbt.com/analytics-engineering/transformation/)

除了在工作中使用正确的工具，同样重要的是确保你也以正确的方式使用它们。在我最近的一篇文章中，我们讨论了关于[如何安装 dbt 命令行接口以及您需要访问](/install-dbt-1bd6a4259a14)的特定数据平台所需的适配器。

在今天的文章中，我们将讨论如何在 dbt 项目中正确构建数据模型。数据通常是组织中混乱的资产，因此尽可能加强结构总是很重要的。在接下来的几节中，我们将讨论三种类型的数据模型——在 dbt 的上下文中——以及如何以一种有意义且可伸缩的方式构建它们，这将允许数据团队的团队成员保持一致。

## 分期、中间和集市模型

数据模型各不相同——我的意思是，一些数据模型可能对应于一些特定的数据源，另一些可能将多个数据源甚至其他数据模型组合在一起，等等。因此，创建数据模型的**层非常重要。**拟分层由三类模型组成，即**分期**模型、**中间**模型和**集市**模型。

**阶段模型**，是您的 dbt 项目中所有数据模型的构建块。分期模型应该包括**基本计算**(如字节到千兆字节)**重命名**、**类型转换**和**分类**(使用`CASE WHEN`语句)。然而，您应该**避免在分段模型上执行任何连接和聚合**。作为一般的经验法则，你应该在你的数据源和阶段模型之间有 1-1 的映射。因为阶段模型不应该代表最终的工件，所以建议**将它们具体化为视图**。

**中间模型**，被认为是将阶段模型甚至其他中间模型集合在一起，它们往往比阶段模型更复杂一些。换句话说，这些模型代表了更有意义的构建模块，将来自多个模型的信息集合在一起。但是请注意，它们不应该向最终用户公开(即由 BI 工具使用，如 Tableau、PowerBI 或 Looker)。同样重要的是要提到，作为一个经验法则，如果一个中间模型在几个地方被引用，那么你可能不得不考虑构建一个宏，或者重新考虑你设计模型的方式。

**集市模型**，是商业定义的实体，应该由终端用户和商业智能工具消费。每个集市模型都代表一个细粒度的实体——支付、客户、用户、订单只是我们作为集市所代表的一些例子。

## 构建您的 dbt 模型

既然我们已经对数据构建环境中的三种主要模型类型有了坚实的理解，那么让我们开始讨论如何以一种有意义的方式构建这些模型，以帮助数据团队以一种简单直观的方式维护和扩展它们。

在您的 dbt 项目中，您需要有一个名为`models`的父目录，由三个目录组成，每个目录代表我们前面讨论的一个模型类型:

```
models
|---intermediate
|---marts
|---staging
```

现在让我们从**分期模型**开始。

*   对于每个不同的源，您需要在`staging`目录下创建一个子目录
*   每一个模型，都必须遵循`stg_[source]__[entity]s.sql`符号
*   模型目录下的一个`base`子目录，以防您需要将登台模型连接在一起

举个例子，假设我们有三个独立的来源——一个是脸书广告(营销活动),另一个来自 Stripe(支付),第三个包含我们的在线商店的商业实体。

```
models/staging
|---facebook_ads
|   |---_facebook_ads__models.yml
|   |---_facebook_ads__sources.yml
|   |---_facebook_ads__events.yml
|---my_shop
|   |---_my_shop__models.yml
|   |---_my_shop__sources.yml
|   |---base
|   |  |---base_my_shop__deleted_products.sql
|   |  |---base_my_shop__deleted_users.sql
|   |  |---base_my_shop__products.sql
|   |  |---base_my_shop__users.sql
|   |---stg_my_shop__orders.sql
|   |---stg_my_shop__products.sql
|   |---stg_my_shop__users.sql
|---stripe
    |---_stripe_models.yml
    |---_stripe_models.yml
    |---stg_stripe__payments.yml
```

请注意我们是如何为每个不同的源创建一个单独的子目录的，每个子目录都由两个 yml 文件组成——一个用于定义模型，另一个用于源——以及您为每个源拥有的尽可能多的登台模型。

现在让我们继续讨论**中级车型**。

*   对于每个不同的业务组，我们创建一个子目录——非常类似于我们前面介绍的分级结构
*   每一款中级车型，都必须遵循`int_[entity]s_[verb]s.sql`的命名惯例。请注意，使用动词作为命名的一部分将有助于您构建名称，这有助于读者和维护人员清楚地了解特定模型应该做什么。这样的动词有`joined`、`aggregated`、`summed`等。

例如，假设我们有两个业务组，即`finance`和`marketing`:

```
models/intermediate
|---finance
|   |---_int_finance__models.yml
|   |---int_payments_pivoted_to_orders.sql
|---marketing
|   |---_int_marketing__models.yml
|   |---int_events_aggregated_per_user_platform.sql
```

最后，让我们看看如何构建我们的最终构件，即对应于业务定义的实体的**集市模型**。

*   为每个部门、业务单位或实体创建一个子目录
*   每个 mart 模型都应该简单地以它所代表的实体命名。例如`orders`、`users`或`payments`
*   避免跨多个不同业务单元的重复实体(这通常是一种反模式)。

```
models/marts
|---finance
|   |---_finance__models.yml
|   |---orders.sql
|   |---payments.sql
|   |---payroll.sql
|   |---revenue.sql
|---marketing
|   |---_marketing__models.yml
|   |---campaigns.sql
|   |---users.sql
```

## 命名约定:概述

这是一篇相当长的文章，包含了太多的信息——尤其是如果您是 dbt 新手的话——所以让我回顾一下关于命名约定的一些要点。

*   在`models`目录下，为每个数据模型类型创建三个子目录
*   分级模型需要遵循`stg_[source]__[entity]s.sql`命名约定
*   中间型号需要遵循`int_[entity]s_[verb]s.sql`惯例
*   集市模型需要以它们所代表的实体命名
*   暂存模型目录下的一个`base`子目录，以防您需要将暂存模型连接在一起

这是我们在前面几节中经历的示例的最终结构。

```
models
|---intermediate
   |---finance
   |   |---_int_finance__models.yml
   |   |---int_payments_pivoted_to_orders.sql
   |---marketing
   |   |---_int_marketing__models.yml
   |   |---int_events_aggregated_per_user_platform.sql
|---marts
    |---finance
    |   |---_finance__models.yml
    |   |---orders.sql
    |   |---payments.sql
    |   |---payroll.sql
    |   |---revenue.sql
    |---marketing
    |   |---_marketing__models.yml
    |   |---campaigns.sql
    |   |---users.sql
|---staging
   |---facebook_ads
   |   |---_facebook_ads__models.yml
   |   |---_facebook_ads__sources.yml
   |   |---_facebook_ads__events.yml
   |---my_shop
   |   |---_my_shop__models.yml
   |   |---_my_shop__sources.yml
   |   |---base
   |   |  |---base_my_shop__deleted_products.sql
   |   |  |---base_my_shop__deleted_users.sql
   |   |  |---base_my_shop__products.sql
   |   |  |---base_my_shop__users.sql
   |   |---stg_my_shop__orders.sql
   |   |---stg_my_shop__products.sql
   |   |---stg_my_shop__users.sql
   |---stripe
       |---_stripe_models.yml
       |---_stripe_models.yml
       |---stg_stripe__payments.yml
```

显然，我们今天演示的结构可能不是 100%适合您的用例，所以请随意相应地修改它——但是无论什么情况，都要确保清楚地定义这种结构背后的逻辑，并坚持下去。

## 最后的想法

数据模型管理是现代数据团队**必须**做好的最重要的支柱之一。对此类模型的薄弱管理可能会导致数据质量下降、数据停机以及难以扩展和丰富您的数据资产。鉴于现代公司需要基于数据做出决策，糟糕的数据模型管理可能会带来灾难性的后果，并导致错误的决策。

在今天的文章中，我们对数据构建工具(dbt)如何帮助现代组织和数据团队更高效地管理数据模型进行了高度概括。但最重要的是，我们讨论了如何在 dbt 本身中构建项目和数据模型。数据构建工具是最强大的工具之一，但是使用正确的工具还不够，以正确的方式使用正确的工具也很重要。

关于 dbt 项目结构的更全面的阅读，你可以参考[官方 dbt 文档](https://docs.getdbt.com/guides/best-practices/how-we-structure/1-guide-overview)。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读媒介上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

<https://gmyrianthous.medium.com/membership>  

**相关文章你可能也喜欢**

</visual-sql-joins-4e3899d9d46c>  </2-rules-groupby-sql-6ff20b22fd2c>  </sql-select-distinct-277c61012800> 