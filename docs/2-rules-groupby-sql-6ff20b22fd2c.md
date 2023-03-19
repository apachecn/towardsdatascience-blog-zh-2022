# 在 SQL 中使用 GROUP BY 时要遵循的 2 条规则

> 原文：<https://towardsdatascience.com/2-rules-groupby-sql-6ff20b22fd2c>

## 了解 GROUP BY 子句中要包含哪些列，以及如何在 WHERE 子句中包含聚合

![](img/c8c6b9961d9d6b5988fa306c6bdb55ed.png)

由 [Markus Spiske](https://unsplash.com/@markusspiske?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/group?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

SQL 中的`GROUP BY`子句允许用户应用函数，并通过将具有相同值的记录分组在一起来对数据执行聚合，如结果查询中所指定的。

SQL 新手在使用`GROUP BY`子句和聚合函数时，通常很难找到正确的查询语法。

在接下来的几节中，我们将介绍两个基本原则，这两个原则最终可以帮助您解决 SQL 中聚合和 GROUP BY 子句的任何困惑。

首先，让我们创建一个示例表，我们将在这个简短的教程中引用它，以便用实际的例子来演示一些概念。

```
> SELECT * FROM orders;+----+-------------+---------+-----------+----------+
| id | customer_id |  amount | order_date| store_id |
+----+-------------+---------+-----------+----------+
|  1 |          20 |   10.99 | 2022-02-02|        1 |
|  2 |         138 |   21.85 | 2022-09-10|        1 |
|  3 |          31 |    9.99 | 2021-12-28|        2 |
|  4 |          20 |   12.99 | 2022-04-29|        2 |
|  5 |          89 |    5.00 | 2022-03-03|        3 |
|  6 |          31 |    4.99 | 2022-11-09|        1 |
|  7 |          20 |   15.00 | 2022-09-10|        3 |
|  8 |          15 |  120.00 | 2022-06-07|        2 |
|  9 |          15 |   32.00 | 2022-04-01|        2 |
+----+-------------+---------+-----------+----------+
```

## SQL 中的 GROUP BY 子句

现在,`GROUP BY`子句通常应用于分类列。尽管可以对具有连续值的列上的记录进行分组，但这样做通常没有太大意义。

当调用`GROUP BY`子句时，查询将根据`GROUP BY`子句中指定的列的值将记录分组，然后将聚合函数应用于各个组，以计算所需的结果，并最终返回结果，在这些结果中，我们将看到`GROUP BY`子句中指定的每组列的一个值。

一些最常见的聚合函数包括`SUM`、`AVG`、`COUNT`、`MAX`和`MIN`。

现在使用我们的示例表，假设我们希望看到每个客户的最大订单量，最后根据最大订单量对结果进行降序排序。下面的查询可以解决这个问题:

```
SELECT
    customer_id,
    MAX(amount) as max_order_amount
FROM
    orders
GROUP BY
    customer_id
ORDER BY 
    max_order_amount DESC
;
```

结果会是

```
+-------------+------------------+
| customer_id | max_order_amount |
+-------------+------------------+
|          15 |              120 |
|         138 |            21.85 |
|          20 |               15 |
|          31 |             9.99 |
|          89 |                5 |
+------------+-------------------+
```

## GROUP BY 子句中要包含哪些列

现在，当我们需要在我们的`SELECT`子句中包含更多列时，事情可能会变得有点复杂。继续我们之前编写的查询，现在让我们假设我们想要查看每个商店的每个客户的最小和最大订单。

让我们尝试运行以下查询:

```
SELECT
    customer_id,
    store_id,
    MIN(amount) as min_order_amount,
    MAX(amount) as max_order_amount
FROM
    orders
GROUP BY
    customer_id
ORDER BY 
    max_order_amount DESC
;
```

显然，我们有一个语法错误，应该类似于下面报告的错误:

```
ERROR:  column "orders.store_id" must appear in the GROUP BY clause or be used in an aggregate function LINE 3:     store_id,             ^ SQL state: 42803 Character: 29
```

这是因为我们违反了涉及`GROUP BY`子句的 SQL 查询的一个基本原则。在`SELECT`语句中指定的列，要么必须应用聚合函数，要么必须包含在`GROUP BY`子句中。

回到我们的例子，这个错误是因为我们没有将`staff_id`和`customer_id`一起包含在`GROUP BY`子句中。另一方面，`amount`不应该包含在`GROUP BY`子句中，因为它应用了聚合函数(`MIN`和`MAX`)。

> `SELECT`语句中的列必须应用聚合函数，或者包含在`GROUP BY`子句中

因此，为了修复我们的查询，我们需要做的就是在`GROUP BY`子句中包含`store_id`(假设我们想要观察的是每个客户和商店的最大和最小订单金额)。

```
SELECT
    customer_id,
    store_id,
    MIN(amount) as min_order_amount,
    MAX(amount) as max_order_amount
FROM
    orders
GROUP BY
    customer_id,
 store_id
ORDER BY 
    max_order_amount DESC
;
```

## WHERE 语句中的聚合以及如何使用 HAVING

回到我们的第一个例子，让我们假设我们想计算每位消费超过 5.00 英镑的顾客的最大订单金额。

```
SELECT
    customer_id,
    MAX(amount) as max_order_amount
FROM
    orders
WHERE 
    MAX(amount) > 5
GROUP BY
    customer_id
ORDER BY 
    max_order_amount DESC
;
```

如果我们尝试运行上面的查询，将会出现以下错误:

```
ERROR:  aggregate functions are not allowed in WHERE
```

上述查询的问题是，我们试图在一个`WHERE`子句中包含一个聚合函数(即`MAX`)。SQL 的语法明确规定`WHERE`语句不能引用`SELECT`子句中指定的聚合。

但是解决方法是什么呢？

> WHERE 语句不能引用 SELECT 子句中指定的聚合

我们可以使用`HAVING`子句，而不是在`WHERE`子句中指定聚合列的条件，该子句允许我们使用`GROUP BY`子句，然后对聚合结果进行筛选。

因此，我们应该将`WHERE`替换为`HAVING`子句，如下所示:

```
SELECT
    customer_id,
    MAX(amount) as max_order_amount
FROM
    orders
GROUP BY
    customer_id
HAVING
 MAX(amount) > 5
ORDER BY 
    max_order_amount DESC
;
```

生成的记录集应该如下所示:

```
+-------------+------------------+
| customer_id | max_order_amount |
+-------------+------------------+
|          15 |              120 |
|         138 |            21.85 |
|          20 |               15 |
|          31 |             9.99 |
+------------+-------------------+
```

## 最后的想法

作为一名 SQL 初学者，使用 GROUP BY 子句有时会非常令人沮丧。在今天的简短教程中，我们讨论了在 SQL 查询中使用聚合时需要牢记的两个最基本的原则。

概括地说，重要的是要确保在`SELECT`子句中指定的所有列都将应用聚合函数(例如`SUM`、`AVG`等)。)或在`GROUP BY`条款中指定。

此外，您需要记住聚合(在`SELECT`子句中指定)不能在`WHERE`子句中引用。如果您希望这样做，您需要在`HAVING`条款中指定您的条件。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**相关文章您可能也喜欢**

[](/ddl-dml-e802a25076c6)  [](/standard-vs-legacy-sql-bigquery-6d01fa3046a9)  [](/sql-select-distinct-277c61012800) 