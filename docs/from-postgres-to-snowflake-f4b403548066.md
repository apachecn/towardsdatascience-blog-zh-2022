# 从明信片到雪花

> 原文：<https://towardsdatascience.com/from-postgres-to-snowflake-f4b403548066>

## 当我把 DWH 从波斯格里斯迁移到雪花时遇到的有趣的点

![](img/c6eea07b8f233eff5b15efa986a61787.png)

图片由 [Paul Owens](https://www.istockphoto.com/pt/portfolio/Oenz?mediatype=photography) 在 [Unsplash](https://unsplash.com/) 上拍摄

我最近遇到了一个任务，似乎是许多使用 PostgreSQL 作为主数据库的公司在某个时候需要做的任务。

这是 ETL 脚本从 PostgreSQL 到雪花的迁移。

整个过程包括基于辛格和 DBT 技术从零开始构建 ETL 管道，我稍后可能会详细介绍，但现在我想集中讨论 PostgreSQL 和雪花 SQL 语法之间的差异。我希望它对那些发现自己处于类似情况并需要重写大量 SQL 的人有用。

# 数字

PostgreSQL 是一个关系数据库，具有所有优点和缺点，有时需要微调，包括转换到最适合的类型。但雪花的情况并非如此。

首先需要将`int2`、`int4`、`int8`等类型转换为`int`。第二个也是更容易出错的区别是，尽管两个数据库中都有一个类型`numeric`,但雪花将这种类型的数据视为整数，而在 Postgres 中，它也可以包含小数。换句话说，雪花不会给出任何误差，但一切都会被四舍五入。

# 不同于

我相信对于大多数分析数据库来说，只需要获得最后一行或第一行是很常见的情况。Postgres 有一个针对它的`distinct on`构造，但是 Snowflake 不支持它，所以必须使用`row_number`窗口函数或`qualify`构造来解决这个问题([https://docs . snow flake . com/en/SQL-reference/constructs/qualify . html](https://docs.snowflake.com/en/sql-reference/constructs/qualify.html))。例如，这就是我们如何获得第一个用户会话开始的时间戳。

# JSON 空值

Snowflake 使用 JSON 很棒，但是有时那里的字段有`null`值，结果 SQL `is null`不起作用。有两种选择:要么使用特殊函数`is_null_value (`[https://docs . snow flake . com/en/SQL-reference/functions/is _ null _ value . html](https://docs.snowflake.com/en/sql-reference/functions/is_null_value.html)，要么通过`strip_null_value (`[https://docs . snow flake . com/en/SQL-reference/functions/strip _ null _ value . html](https://docs.snowflake.com/en/sql-reference/functions/strip_null_value.html)函数去除空值，然后使用普通的`is null` SQL 检查。我个人认为后一种解决方案更有吸引力。

# 过滤

在分析中向度量(聚合函数)添加过滤器是一种常见的做法，在 SQL:2003 的 Postgres 中有一个很好的语法。不幸的是，Snowflake 不支持它，所以解决方法是通过`case`构造来走老路。例如，在这里，我们计算不同设备类型的会话数。

# 横向连接

雪花支持横向连接，此外，当需要解析 JSON 数组时，它被大量使用，但是有一些限制。其中一个限制是我们不能在横向连接中使用`limit`。一种常见的情况是，我们只需要获得第一行就行不通了。解决方法是将其转换为 CTE。实际上，CTE 将在 Postgres 和雪花中工作，所以横向连接的解决方案只是对 Postgres 的优化，但我们在大多数情况下不需要对雪花进行这样的优化，因为引擎的工作方式不同。在下面的代码片段中，我们得到了每个会话的第一个事件。

# 递归

好消息是，雪花完全支持递归，它会工作。问题在于局限性。在 Snowflake 中，默认的递归深度是 100，它可以增加，但仍然有一些限制(我们在这里将 Snowflake 视为托管服务)。不幸的是，由于这些限制，雪花递归根本不能解决一些问题，例如库存操作列表的加权平均成本/价格。(在 Postgres 中，我们可以使用这个解决方案:[https://stack overflow . com/questions/22426878/calculating-the-weighted-average-cost-of-products-stock](https://stackoverflow.com/questions/22426878/calculating-the-weighted-average-cost-of-products-stock))

我找到的唯一解决方案是要么在雪花之外进行计算，要么创建一个存储过程。

# 其他…

当然，在函数，过程，写 UDF 的方式，和存储过程方面还有很多不同，但是这篇文章的目的是分享我的经验，所以我希望它会有用。