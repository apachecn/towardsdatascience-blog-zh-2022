# 最常用的 Pandas 函数的 SQL 版本

> 原文：<https://towardsdatascience.com/sql-versions-of-the-most-frequently-used-pandas-functions-bb6399f87461>

## 实用指南

![](img/b8122b6ee60a30668c3ec9083c972743.png)

[张杰](https://unsplash.com/@jay_zhang?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/computer-keyboard?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

我使用熊猫已经 3 年了，最近我发表了一篇关于我使用最多的 [8 熊猫功能](https://medium.com/p/4e54f4db5656)和它们的 [R 版本](/r-versions-of-the-most-frequently-used-pandas-functions-f658cbcabaf7)的文章。

是时候在数据科学生态系统的另一个大玩家 SQL 中介绍相同的操作了。

由于大多数公司将数据存储在关系数据库中，SQL 是数据分析师和数据科学家的必备技能。在本文中，我们将学习如何在 SQL 中执行最常用的 Pandas 函数。

## 1.阅读 _csv

在 Pandas 中，我们讨论了使用 read_csv 函数从 csv 文件中读取数据。在关系数据库中，数据存储在表中，这些表可以看作是熊猫数据帧的等价物。

我创建了一个名为 sales 的表。我们将举例来查询这个表。

为了从 SQL 表中读取数据，我们使用 select 语句。

```
SELECT * FROM sales
```

该查询选择 sales 表中的所有数据。假设销售表中有几列，但我们只需要其中的几列。在这种情况下，我们写列名而不是“*”。

我们还可以用选定的数据创建一个临时表。当我们需要处理存储在大表中的数据子集时，它特别有用。

以下查询从 sales 表中选择 4 列，并将它们保存到名为#temp 的临时表中。

```
select
   product_group,
   product,
   sales_qty,
   sales_rev
into #temp
from sales
```

**注:**关系数据库管理系统(RDBMS)是使用 SQL 来管理关系数据库中的数据的软件。尽管 SQL 语法在不同的 RDBMSs 中基本相同，但还是有一些细微的差别，但它确实对您学习 SQL 有着重大的影响。因此，这两种 RDBMS 都适合学习 SQL。在本文中，我们将使用 Microsoft SQL Server 作为 RDBMS。

## 2.值计数

在 Pandas 中，value_counts 函数返回一列中的唯一值及其出现的次数。我们可以在 SQL 中使用 group by 语句和 count 函数来完成这个操作。

```
SELECT product_group, COUNT(1) AS value_count
FROM #temp
GROUP BY product_group**# output**
  product_group  value_count
1 heater         34
2 laptop         237
3 phone          377
4 printer        32
```

该查询根据产品组列中的不同值对行进行分组，并计算每列中的行数，这基本上就是 value_counts 函数的工作。

## 3.astype

在 Pandas 中，astype 函数转换列的数据类型。我们可以使用 cast 函数在 SQL 中做同样的事情。它将任何类型的值转换为指定的数据类型。

下面是一个将十进制数转换为整数的快速示例。

```
SELECT CAST(12.5 AS int)**# Output**
12
```

当处理一个表时，我们在 cast 函数中写入列名和所需的数据类型。以下查询将 product 列转换为小数点数字(这没有意义，但我只想演示如何使用 cast 函数)。

```
SELECT top 5 CAST(product as decimal(10,2))
FROM #temp**# Output** (No column name)
1 17794.00
2 15669.00
3 18091.00
4 17801.00
5 18105.00
```

## 4.isna

我们经常需要处理丢失的值，这些值仅仅表示我们没有的数据。如果单元格中缺少值，Pandas 中的 isna 函数将返回 True。

我们可以通过在 where 语句中使用“is null”谓词来检查缺少的值。例如，以下查询选择 sales_qty 列中的值为 null 的行。

```
SELECT *
FROM #temp
WHERE sales_qty IS NULL
```

如果我们对非缺失值感兴趣，我们只需将“is null”改为“is not null”。

## 5.德罗普纳

在进行任何分析之前，需要正确处理缺失的值。我们基本上有两个选项来处理缺失值:drop 或 fill。

在 Pandas 中，dropna 函数用于删除缺少值的行或列。我们可以通过使用 delete 和 where 语句来执行这项任务。

where 语句查找丢失的值，而 delete，顾名思义，删除它们。以下查询删除 sales_qty 值缺失的行。

```
DELETE FROM #temp
WHERE sales_qty IS NULL
```

如果多列中有缺失值，我们可能希望删除一行。在这种情况下，我们只需要在 where 语句中编写多个条件。下面是一个在删除一行之前检查 sales_qty 和 sales_rev 列的示例。

```
DELETE FROM #temp
WHERE sales_qty IS NULL AND sales_rev is NULL
```

## 6.菲尔娜

处理缺失值的另一个选择是用合适的值替换它们。在 Pandas 中，fillna 函数执行此操作。

我们在 SQL 中没有 fillna 函数，但是我们可以通过使用 update 和 where 语句轻松完成它的功能。

我们在 Pandas [文章](/i-have-been-using-pandas-for-3-years-here-are-the-8-functions-i-use-the-most-4e54f4db5656)中做的例子是用一列的平均值替换该列中缺失的值。该任务可以在 SQL 中完成，如下所示:

```
UPDATE #temp
SET sales_rev = (SELECT AVG(sales_rev) FROM #temp)
WHERE sales_rev IS NULL
```

where 语句查找 sales_rev 列中缺少值的行。update 语句用该列的平均值更新缺失值。您可能已经注意到，我们还在这个查询中找到了 sales_rev 列的平均值。

## 7.分组依据

Pandas 中的 groupby 函数允许根据列中的不同值对行进行分组。然后，我们可以为每个组计算一个大范围的聚合。这可能是探索性数据分析中最常用的函数之一。

SQL 有一个 group by 语句和一组聚合函数。因此，对于这样的任务，SQL 就像熊猫一样能干。

让我们首先找出每个产品组的平均销售收入。

```
SELECT
   product_group,
   AVG(sales_rev) AS avg_sales
FROM #temp
GROUP BY product_group**# output**
  product_group  avg_sales
1 heater         3096.717647
2 laptop         147766.657130
3 phone          481470.797002
4 printer        27642.223750
```

我们在 select 语句中写入列名。选择列时应用聚合函数。最后，用于分组的列在 group by 语句中指定。

我们也可以进行多重聚合。以下查询查找每组中的产品数量以及平均销售收入。

```
SELECT
   product_group,
   AVG(sales_rev) AS avg_sales,
   COUNT(product) AS prod_count
FROM #temp
GROUP BY product_group**# output**
  product_group  avg_sales     prod_count
1 heater         3096.717647   34
2 laptop         147766.657130 237
3 phone          481470.797002 377
4 printer        27642.223750  32
```

就像我们可以进行多个聚合一样，我们可以按多个列对行进行分组。我们只需要在 select 语句和 group by 语句中写入这些列。

## 8.独一无二的

熊猫的这些功能是:

*   unique 返回不同的值
*   nunique 返回不同值的数量

我们可以在 SQL 中找到非重复值和非重复值的数量，如下所示:

*   独特:独特
*   努尼克:计数和独特的

让我们在 product_group 列中找到它们。

```
SELECT DISTINCT(product_group) FROM #temp**# output**
  product_group
1 heater
2 laptop
3 phone
4 printer------------------------------------------
SELECT COUNT(DISTINCT(product_group)) FROM #temp**# output**
4
```

我们已经介绍了如何在 SQL 中复制最常用的 Pandas 函数的功能。SQL 是数据科学领域中非常有价值的技能，因为大量数据存储在关系数据库中。

*你可以成为* [*媒介会员*](https://sonery.medium.com/membership) *解锁我的全部写作权限，外加其余媒介。如果你已经是了，别忘了订阅*<https://sonery.medium.com/subscribe>**如果你想在我发表新文章时收到电子邮件。**

*<https://sonery.medium.com/membership>  

感谢您的阅读。如果您有任何反馈，请告诉我。*