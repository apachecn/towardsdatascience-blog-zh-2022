# BigQuery SQL 优化 2:使用临时表快速获得结果

> 原文：<https://towardsdatascience.com/bigquery-sql-optimization-2-with-temp-tables-to-fast-results-41869b15fcff>

## 何时使用临时表而不是 WITH

查询的最大性能杀手之一是在不应该使用`CREATE TEMP TABLE`的情况下使用`WITH`!在阐明了我们应该尽早使用 [*过滤器之后，让我们继续讨论何时使用或避免`WITH`。*](/bigquery-sql-optimization-1-filter-as-early-as-possible-60dfd65593ff)

![](img/f576d79ce95e6f72e3f06f0965d2d13d.png)

(作者供图)

# 患有急性健忘症

`WITH`语句也叫*常用表表达式* (CTE)。它们有助于消除查询的混乱，使查询更具可读性，因为它们将子查询从上下文中提取出来，并为它们命名。

即

```
SELECT a, b, c
FROM (SELECT x, y, z FROM ...)
```

变成了

```
WITH my_CTE AS (SELECT x, y, z FROM ...)SELECT a, b, c
FROM my_CTE
```

`my_CTE`看起来像一张桌子，因为它在一个表单的后面——但它不是桌子。它更像是实时指令，无论何时调用它，都会在运行时动态创建一个结果表。

# *每次你引用一个 CTE，它就会被执行*

太疯狂了，对吧？cte 不会记住任何以前执行的结果！所以如果你这样做…

```
WITH ***a*** AS (...),a1 AS (SELECT aggr_1 FROM ***a***),a2 AS (SELECT aggr_2 FROM ***a***) SELECT ... FROM a1 LEFT JOIN a2
```

…然后你需要在读完这篇文章后立即修改这个查询，因为你通过计算两次`***a***`给你的查询引擎和计算槽带来了很多*不必要的负载*！

# 不要忘记:临时表

那么我们该怎么办呢？`***a***`应该是一个临时表，因为它们会记住结果——至少在查询运行期间。

之前的查询应该是这样的:

```
CREATE TEMP TABLE ***a*** AS (...)WITH a1 AS (SELECT aggr_1 FROM ***a***),a2 AS (SELECT aggr_2 FROM ***a***) SELECT ... FROM a1 LEFT JOIN a2
```

我们只计算一次`***a***`，并将其用于`a1`和`a2`中的两个不同聚合。

这和之前的例子没什么不同，对吧？但是它会执行得更好，因为我们省去了`***a***`的第二次计算。

您也不需要担心在特定数据集中创建表或删除表——它将由 BigQuery 处理，并在您的 SQL 语句运行完毕后消失。

我见过 CTE 被引用超过 5 次的查询。至少可以说，将该表重构为临时表很有帮助。为了证明它有帮助，我们可以对几次运行进行采样:

*   准备好已优化和未优化的查询
*   停用缓存
*   通过减少查询的数据量来保持合理的总工作量
*   大约同时运行两个查询 5 次，比较它们的平均槽时间

# 如何在创建临时表时重构旧查询

除了上面显示的变化之外，您可能会遇到希望将 cte 与临时表混合的情况。如果您真的只需要运行一次 CTE，那么它会比运行然后临时存储它稍微快一些。因此，如果我们可以跳过临时存储这一步，我们应该这样做。那么我们如何混合 cte 和临时表呢？

你可以把`CREATE TEMP TABLE`看作是更根本的操作。为了一起使用它们，它将简单地包含 CTE 定义，因为它们只是使子查询更可读，但本质上与子查询是一样的:

```
CREATE TEMP TABLE a AS ( WITH x AS (...),
  y as (...) SELECT ... FROM x LEFT JOIN y ON ...) SELECT ... FROM a ...
```

这个查询将使用 CTE `x`(在`a`的定义中定义)来创建临时表`a`。

总结一下:使用 cte 整理您的 SQL 语句，使它们更具可读性。但是不要多次引用 CTE，因为每次查询引擎都会重新计算结果。在这种情况下，请使用临时表——它们会给处理成本增加额外的存储步骤，但这(从临时表中读取)可能比重新计算整个查询更便宜。

别忘了尽早将这个最佳实践与[过滤结合起来](/bigquery-sql-optimization-1-filter-as-early-as-possible-60dfd65593ff)！

重构快乐！