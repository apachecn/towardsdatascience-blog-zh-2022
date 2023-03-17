# 减少您的担忧:使用 PySpark 的“Reduce”

> 原文：<https://towardsdatascience.com/reduce-your-worries-using-reduce-with-pyspark-642106d6ae50>

# 减少您的担忧:使用 PySpark 的“Reduce”

## 使用 python 轻松重复 PySpark 操作

![](img/8ae4854cb338483f7c8fbdce0d34f81a.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [S Migaj](https://unsplash.com/@simonmigaj?utm_source=medium&utm_medium=referral) 拍摄的照片

如果您使用 PySpark，您可能已经熟悉了它编写类似 SQL 的查询的能力。您可以轻松地对常见的 SQL 子句进行方法链，如。select()，。filter/where()/，。join()，。withColumn()，。groupBy()和。agg()来转换 Spark 数据帧。它本身就很强大，但是当您将它与 python 风格的脚本结合使用时，它的功能就变得无限了。我将展示两个例子，在这两个例子中，我使用 python 的 functools 库中的“reduce”来重复地对 Spark 数据帧应用操作。

## 堆叠桌子

第一个技巧是使用类似 SQL 的 union all 堆叠任意数量的数据帧。假设您有一系列结构相同的表，并且您想将它们堆叠在一起。出于共享的目的，可以将这些表划分成许多较小的表，或者每个表可以代表一个月，或者其他任何原因。在这个例子中，我们假设有一个 parquet 文件路径列表，其中包含一系列需要组合的表。我们可以编写一个不必要的 for 循环来一个接一个地堆叠它们，但是更好的方法是利用 functools 库中的“reduce”。

reduce 函数需要两个参数。第一个参数是我们想要重复的函数，第二个参数是我们想要重复的 iterable。通常当你使用 reduce 时，你使用一个需要两个参数的函数。一个常见的例子是

```
reduce(lambda x, y : x + y, [1,2,3,4,5])
```

它会这样计算:

```
((((1+2)+3)+4)+5)
```

对于这个例子，我们将使用 DataFrame *方法*来代替，并在 iterable 上重复链接它。

这个方法链按照我们的期望组合了我们所有的数据框架。

```
(dfs[0].unionAll(dfs[1])).unionAll(dfs[2])...
```

## 嵌套 OR/AND

在下一个例子中，我们将需要应用一个包含一系列条件的过滤器，这些条件可能全部由 or 或 and 连接。一个常见的 SQL 示例是，您可能希望查询与特定市场相关的所有行，您只需要知道“市场”字段的前三个字符。

```
SELECT * FROM TABLE
WHERE (MARKET LIKE 'AQ1%' OR MARKET LIKE 'AW3%' OR MARKET LIKE 'M89%' OR ...)
```

对于我的表，每个人都有一行，一系列列表示他们在那个月是否被覆盖。如果他们在那个月被覆盖，他们在列中有 1，否则他们有 0。我想查询至少在某个时间范围的某个点上有覆盖率的所有成员，所以我想编写的类似 SQL 的查询如下所示:

```
SELECT * FROM TABLE
WHERE (COV_2007 = 1 OR COV_2008 = 1 OR COV_2009 = 1 OR ... OR COV_2106 = 1)
```

在 SQL 中单独编写这个是一件痛苦的事情，但是使用 python 我们可以很容易地编写这个重复 OR 条件的脚本。

为此，我首先创建一个我感兴趣的列的字符串列表。在示例中，我想要 2020 年 7 月到 2021 年 6 月(它们被命名为“cov _ 2007”—“cov _ 2106”)。

接下来，我创建一个列级过滤器的列表，在本例中，我希望该列等于值 1。使用 reduce 之前的最后一步是创建我想要重复的函数。在这种情况下，我创建了一个 lambda 函数，它只接受两列的逻辑或(如果需要，您可以使用' & '来代替 AND)。

这就够了！使用' reduce '节省了我大量的时间来写出不必要的条件，或者写一个不好的 for 循环。我希望你喜欢阅读！