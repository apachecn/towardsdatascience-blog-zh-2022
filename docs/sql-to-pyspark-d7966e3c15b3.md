# SQL 到 PySpark

> 原文：<https://towardsdatascience.com/sql-to-pyspark-d7966e3c15b3>

## 从 SQL 到 PySpark 的快速指南。

如果你知道 SQL 但是需要在 PySpark 中工作，这篇文章就是为你准备的！

![](img/7e3c41a95fb61fe1664494479eec3381.png)

照片由 [Miki Fath](https://unsplash.com/@m_fath?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

Spark 正迅速成为大数据处理最广泛采用的框架之一。但是为什么要使用本机 PySpark 而不是 SQL 呢？

嗯，你不需要。PySpark 允许您[创建一个不牺牲运行时性能的 tempView](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.createOrReplaceTempView.html) 。在后端，spark 以完全相同的方式运行相同的转换，而不考虑语言。因此，如果你想坚持使用 SQL，你的代码不会有任何不同。

然而，当在 DataFrame API 中工作时，您会得到编译时错误，而使用原始 SQL 时，您会得到运行时错误。**如果您正在处理大量数据，那么在使用本机 PySpark 时，同样的错误可能会更早出现。**

在这篇文章中，我们将利用 [Spark:权威指南](https://www.amazon.com/Spark-Definitive-Guide-Processing-Simple/dp/1491912219)，顺序处理基本 SQL 查询中的每个子句，并解释如何在 PySpark 中复制这个逻辑。

事不宜迟，让我们开始吧…

# 挑选

任何好的 SQL 查询都是从 SELECT 语句开始的——它决定了哪些列将被提取，以及它们是否应该被转换或重命名。

## 结构化查询语言

```
SELECT 
  column_1,
  CASE WHEN column_2 IS NULL THEN 0 ELSE 1 END AS is_not_null,
  SUM(column_3) OVER(PARTITION BY column_1)
```

## PySpark

如上所示，SQL 和 PySpark 的结构非常相似。`df.select()`方法接受作为位置参数传递的一系列字符串。每一个 SQL 关键字在 PySpark 中都有对应的符号:点符号，例如`df.method()`、`pyspark.sql`或`pyspark.sql.functions`。

几乎任何 SQL select 结构都很容易复制，只需搜索一些 SQL 关键字。

> 提示:使用`[*df.selectExpr()*](https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrame.selectExpr.html)`来运行带有 SQL 字符串的 SQL 命令。

# 从

现在，如果没有一个好的 FROM 子句，我们的 SELECT 语句就毫无价值。

## 结构化查询语言

```
FROM df
```

## PySpark

很复杂，对吧？

如上所示，FROM 表是由方法之前引用的数据帧定义的。

如果您习惯于在 SQL 代码中使用 CTE，您可以通过将一组转换赋值给变量来复制 CTE 逻辑。

# 在哪里

WHERE 子句是一个被低估的子句，它可以显著提高查询时间。在 PySpark 中，有两个[相同的方法](https://stackoverflow.com/questions/38867472/spark-select-where-or-filtering)允许你过滤数据:`df.where()`和`df.filter()`。

## 结构化查询语言

```
WHERE column_2 IS NOT NULL 
  AND column_1 > 5
```

## PySpark

如上所述，两者都支持 SQL 字符串和本机 PySpark，因此利用 SQL 语法有助于平稳过渡到 PySpark。但是，出于可读性和减少错误的目的，完全原生的 PySpark 应该(可能)是最终目标。

# 加入

连接是另一个被低估的子句——如果你真的很擅长连接，你代码中的 bug 数量会大大减少。根据 [Spark:权威指南](https://www.amazon.com/Spark-Definitive-Guide-Processing-Simple/dp/1491912219)，有 8 大类连接，其中一些包括内连接和左外连接。

我们不会一一介绍，但通常 PySpark 连接遵循以下语法:

```
<LEFT>.join(<RIGHT>, <JOIN_EXPRESSION>, <JOIN_TYPE>)
```

*   `<LEFT>`和`<RIGHT>`是 PySpark 数据帧
*   `<JOIN_EXPRESSION>`是两个数据帧中的列之间的布尔比较
*   `<JOIN_TYPE>`是确定连接类型的字符串

## 结构化查询语言

```
FROM table_1
INNER JOIN table_2
  ON table_1.x = table_2.y
```

## PySpark

> 提示:使用`<DF>.dropDuplicates().count() == <DF>.count()`来检查左表、右表或连接表中是否有重复项。这些错误有时很难被发现，因为你并没有在寻找它们。

# 分组依据

转到更复杂的 SQL 分组概念，PySpark 的语法与这个领域中的 pandas 非常相似。

## 结构化查询语言

```
SELECT
  column_1,
  SUM(column_3) AS col_3_sum
FROM df
GROUP BY 1
```

## PySpark

在 PySpark 中有许多不同的方法来对数据进行分组，但是最通用的语法是上面的方法。我们利用`.agg()`并传递许多定义如何转换列的位置参数。注意，我们可以链接`.alias()`来重命名我们的列，使其比`sum(column_3)`更有用。

如果你记住了这个语法，你就可以随时进行任何你想要的转换。非常清楚地说，语法是…

```
df.groupBy(['<col_1>','<col_2>',...]).agg(
  F.<agg_func>('<col_3>').alias('<name_3>'),
  F.<agg_func>('<col_4>').alias('<name_4>'),
  ...
)
```

有关聚合函数的列表和每个函数的示例，请查看 [sparkbyexamples](https://sparkbyexamples.com/pyspark/pyspark-aggregate-functions/) 。

# 结论

在这里，我们讨论了从 SQL 迁移到 PySpark 的基础知识。有了上面的结构和 google 的一些帮助，您可以用本机 PySpark 编写几乎任何 SQL 查询。

注意，有很多 SELECT 语句关键字，比如 CASE、COALESCE 或 NVL，所有这些都可以使用`df.selectExpr()`编写。如果你想迁移到原生的 PySpark，对 google 来说很简单。

希望这有所帮助，祝你好运！

*感谢阅读！我会再写 13 篇文章，把学术研究带到 DS 行业。查看我的评论，链接到这篇文章的主要来源和一些有用的资源。*