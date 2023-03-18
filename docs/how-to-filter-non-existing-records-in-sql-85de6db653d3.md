# 如何过滤 SQL 中不存在的记录

> 原文：<https://towardsdatascience.com/how-to-filter-non-existing-records-in-sql-85de6db653d3>

## 了解如何筛选 SQL 表中不存在的记录

![](img/c085a9b8623f1cc94a2fb4ae80c65515.png)

图片来自 [Unsplash](https://unsplash.com/photos/fPkvU7RDmCo)

# 介绍

除了所有的特别查询之外，开发预定义的报告已经成为我日常工作的一部分。我经常发现我的利益相关者要求我准备报告，他们想知道一些从未发生过的事情。这听起来可能很奇怪，但的确，这就是他们所关心的问题。简而言之，最常出现的问题如下

1.  哪些是我从未购买过产品的客户？
2.  哪些产品是从来不卖的？
3.  哪些活动从不预订？

了解这些答案有助于分析这些问题的症结所在，而且他们可能还会提出一些有利于整体销售的新想法。

# 我的方法

当我开始准备这样一个问题的答案时，我通常会审核所有我可能感兴趣的表格。例如，在第一个场景中，客户和销售表是最重要的。类似地，在第二个场景中，人们会对产品和销售感兴趣等等。

在任何一种情况下，理解我们需要从一个表中提取另一个表中不存在的信息是很重要的。我将试着解释获得这样一个答案的基本方法，并强调我认为的最佳选择。

# 解决方法

从技术上来说，当事情发展到编写 SQL 查询时，我感到非常兴奋。SQL 中有多种方法可以实现上述场景，下面提到了其中一些方法:

1.  使用 NOT IN 语句
2.  使用外部敷剂
3.  使用左外部连接
4.  使用 EXCEPT 语句
5.  使用不存在

所有这些方法可能产生相同的结果，但是，有可能一种方法比另一种方法执行得更好。在最终确定一种方法之前，我不得不逐一尝试所有的方法。

就本文而言，我想举一个微软提供的全球进口商数据库的例子。这个数据库可以从[微软 SQL Server 的官方 Github 页面下载。](https://github.com/microsoft/sql-server-samples/tree/master/samples/databases/wide-world-importers)

# 使用 NOT IN 语句

虽然这是每个 SQL 开发人员尝试应用的最常见的方法，但这可能不是最好的方法。典型的查询如下所示。

```
SELECT CustomerName FROM Sales.Customers
WHERE CustomerID NOT IN (SELECT CustomerID FROM Sales.Orders);
```

这似乎是一个相当不错的查询，只要销售中有 CustomerID，就会返回正确的结果。订单不为空。如果有空值，此方法可能不起作用，因为数据库引擎将其视为左反半连接，但是，它不会反映右侧的空值是否与左侧的空值匹配。

# 使用外部敷剂

表达相同查询的另一种方式是使用外部 APPLY 子句，查询如下。

```
SELECT CustomerName FROM Sales.Customers cus
OUTER APPLY (
SELECT CustomerID FROM Sales.Orders ord
WHERE ord.CustomerID = cus.CustomerID
) as sales
WHERE sales.CustomerID IS NULL;
```

这种方法也使用左反半连接，但是，产生的计划似乎缺少连接操作符。我认为这种方法比前一种方法稍微贵一点，因为它实际上是以不同的方式处理的。外部连接操作符首先引入所有与条件匹配或不匹配的数据，然后应用过滤器，只返回匹配的记录。

# 使用左外部连接

外部应用的一个常用替代方法是左外部连接。这里，假设因为连接条件是基于 CustomerID 的，所以销售中的所有 CustomerID。对于那些还没有购买的人，订单表将为空。查询计划与前面方法中使用的几乎相似。这里要注意的最重要的事情是确定要对哪一列应用空过滤器。此外，建议在连接条件中使用索引列，因为这有助于获得更好的性能。

```
SELECT CustomerName FROM Sales.Customers cus
LEFT OUTER JOIN Sales.Orders ord
ON ord.CustomerID = cus.CustomerID
WHERE ord.CustomerID IS NULL;
```

# 使用 EXCEPT 语句

另一种方法是使用 EXCEPT 子句，尽管我并不经常使用。在我看来，唯一的问题是，它只能在两列相同的情况下使用，在本例中是 CustomerID。不能在第一个表的 SELECT 语句中使用任何其他列，因为数据库引擎无法识别要对哪一列应用联接条件。

```
SELECT CustomerID FROM Sales.Customers cus
EXCEPT
SELECT CustomerID FROM Sales.Orders ord;
```

# 使用不存在

我处理这种情况的首选是使用 NOT EXISTS 语句。请注意，我在内部查询中使用了“SELECT 1 ”,这有两个原因。首先，SQL Server 并不真正关心 EXISTS 子查询中的内容，其次，从文档的角度来看，它帮助我认识到，仅仅通过查看语句，内部查询并不返回任何记录。就性能而言，我觉得它几乎等同于 NOT IN 和 EXCEPT 语句，但是，一个潜在的优势是它消除了空值或重复的风险。

```
SELECT CustomerName FROM Sales.Customers cus
WHERE NOT EXISTS (
SELECT 1 FROM Sales.Orders ord
WHERE ord.CustomerID = cus.CustomerID
);
```

# 拿走

尽管有多种方法来处理 SQL Server 中不存在的记录的筛选，但我最喜欢的选择是使用 NOT EXISTS 语句。将来，我可能会对所有这些语句使用的所有执行计划进行详细的比较，并对其中每一个语句的性能进行比较。

# 参考

[https://sqlinthewild . co . za/index . PHP/2010/02/18/not-exists-vs-not-in/](https://sqlinthewild.co.za/index.php/2010/02/18/not-exists-vs-not-in/)