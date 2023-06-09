# SQL + Jinja 是不够的—为什么我们需要数据帧

> 原文：<https://towardsdatascience.com/sql-jinja-is-not-enough-why-we-need-dataframes-4d71a191936d>

2021 年 10 月，Apache Airflow 的创始人 Max Beauchemin 写了一篇关于目前影响数据工程师的 12 种趋势的优秀文章。其中一篇文章被漂亮地命名为“模板化的 SQL 和 YAML 之山”，这和我自己的理解非常吻合。他将 SQL + Jinja 方法与早期的 PHP 时代进行了比较——感谢上帝——我从未亲眼目睹过，并解释道:

> 如果您采用以数据框架为中心的方法，您将拥有更多“合适的”对象，以及围绕数据集、列和转换的编程抽象和语义。
> 这与 SQL+jinja 方法非常不同，在后者中，我们本质上是将 SQL 代码片段作为字符串的拼贴来处理

因此，我用 Python 开发了一个开源 POC 来说明这一点，特别是展示以数据帧为中心的方法可以给我们带来多大的进步。

我把我的 POC 项目称为 [bigquery-frame](https://github.com/FurcyPin/bigquery-frame) ，它包括为 Google Big Query 提供一个 DataFrame API。通过分享它，我希望它能更好地展示 DataFrames 的力量，**如果有人在谷歌读到这篇文章，说服他们应该为 BigQuery 开发一个合适的 DataFrame API，并赶上 Spark 和 Snowflake** (如果他们还没有开始工作的话)。

在简要解释了 DataFrame 的来源以及我的 POC 项目 [bigquery-frame](https://github.com/FurcyPin/bigquery-frame) 如何工作以及如何使用它之后，我将给出几个实际的例子，展示使用 data frame 比使用 SQL(甚至使用 Jinja，甚至使用 dbt)可以更简单、更优雅地完成的事情。

这是一个很长的帖子，所以我做了一个目录，如果你现在没有时间阅读，为什么不把它加入书签，在合适的时候再回来呢？还有，既然这次找不到任何关于 DataFrame [的](/modern-data-stack-which-place-for-spark-8e10365a8772) [xkcd 漫画](https://xkcd.com/2582/)，我就用了无版权图片，让大家时不时放松一下。我希望你喜欢它们。还有，[希望你喜欢狗](https://random.dog/)。

![](img/ca3462c6bf82929538ac72f8c431cdad.png)

如果你想给这篇文章做书签，就去这一页的顶部。我也发现这个功能很好，可以找到我喜欢的文章。

# **目录**

## 介绍

*   数据帧的快速历史
*   bigquery-frame:它是如何工作的？
*   bigquery-frame:如何尝试？

## 可以用 DataFrame 做的事情，不能用 SQL 做(或者至少不那么优雅)

*   前言:DataFrame 是 SQL 的超集
*   即时自省
*   链接操作
*   通用转换
*   更高级别的抽象

## 结尾部分

*   大查询框架方法的局限性
*   SQL 也有优势
*   走向 SQL 的统一库？

## 结论

![](img/3c750d6eb4e40a5fd4fde3d3eb61273c.png)

这张图中的小狗几乎和这篇博文中要讨论的话题一样多。(图片由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的 [Jametlene Reskp](https://unsplash.com/@reskp?utm_source=medium&utm_medium=referral) 拍摄)

# 介绍

## **数据帧的快速历史**

DataFrame 的概念最早是由 2009 年开源的 Pandas 推广的。2015 年，Spark 1.3.0 发布了 [DataFrame API](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/Dataset.html) 以及 [Spark Catalyst](https://databricks.com/blog/2015/04/13/deep-dive-into-spark-sqls-catalyst-optimizer.html) ，首次将 SQL 和 DataFrame API 统一在同一个优化器下，允许数据工程师根据需要在 SQL 和 DataFrame 之间轻松切换。2021 年，雪花最近发布了 Snowpark，一个运行在雪花上的 [DataFrame API](https://docs.snowflake.com/en/developer-guide/snowpark/reference/scala/com/snowflake/snowpark/DataFrame.html) ，显然是受了 Spark 的启发(甚至它的名字:S *现在是* PARK)。

## Bigquery-frame:它是如何工作的？

bigquery-frame 背后的想法很简单:我想创建一个 DataFrame 类，它的外观和行为尽可能地类似于 Spark DataFrame，只是它可以在 bigquery 上运行。因为 BigQuery 只理解 SQL，所以我的方法很简单。每个数据帧对应于一个 SQL 查询，该查询仅在一个动作( *df.show()* ， *df.collect()* 等)被执行时才被执行。)被执行。

下面是一个小示例代码，展示了使用 PySpark(左边)和 bigquery-frame(右边)时的代码。

![](img/77130967b8cf8b821fe1d0c5c42ae83c.png)

左边:pyspark(此处有代码)；右边:bigquery-frame(此处提供代码)

## Bigquery-frame:如何尝试？

我的项目在 Pypi 上有[可用，你用 pip 或者你喜欢的 python 包管理器安装就可以了(顺便说一句，如果你不知道](https://pypi.org/project/bigquery-frame/) [Python 诗](https://python-poetry.org/)，试试吧，很牛逼的)。

```
pip install bigquery-frame
```

*(我建议安装在一个* [*的 Python 虚拟环境*](https://realpython.com/python-virtual-environments-a-primer/) *中，以避免与你的主 Python 安装发生任何冲突)*

然后，去看一下 [AUTH.md](https://github.com/FurcyPin/bigquery-frame/blob/main/AUTH.md) 文件，它解释了如何配置您的项目来访问 Big Query。如果你有一个空无一物的虚拟测试 GCP 项目(创建一个项目需要 5 分钟，附带一个 gmail 地址，而且完全免费)，你可以使用[方法 1](https://github.com/FurcyPin/bigquery-frame/blob/main/AUTH.md#method-1--use-application-default-credentials) 。否则，如果你使用一个真实的 GCP 项目，我建议使用[方法 2](https://github.com/FurcyPin/bigquery-frame/blob/main/AUTH.md#method-2--use-a-service-account) 来使用一个具有最小特权的合适的服务帐户。

[*示例*](https://github.com/FurcyPin/bigquery-frame/tree/main/examples) 文件夹包含几个代码示例，其中大部分我将在本文中使用。

# 可以用 DataFrame 做的事情，不能用 SQL 做(或者至少不那么优雅)

**前言:DataFrame 是 SQL 查询的超集**

在我们深入探讨数据框架能做 SQL 不能做的事情(或者至少没有 SQL 做得好)之前，我想指出**数据框架 API 是 SQL 查询的*超集*:**

可以用 SQL 查询表达的一切都可以简单地复制粘贴到 DataFrame 代码中，就像这样:`bigquery.sql(my_sql_query)`。

如果您的 SQL 查询中包含 Jinja，您也可以简单地用 Python 代码编译它，就像这样:

```
from jinja2 import Template
query_template = Template(my_sql_query_with_jinja)
compiled_query = query_template.render(**my_jinja_context)
df = bigquery.sql(compiled_query)
```

当然，这对于使用[进行循环并立即执行](/loops-in-bigquery-db137e128d2d)来运行多阶段 SQL 逻辑的 [SQL 脚本](https://cloud.google.com/bigquery/docs/reference/standard-sql/scripting)可能不起作用，但是好消息是*您可以使用 Python 和 DataFrames 以一种更加干净的方式来完成这项工作！*

**关于 SQL *脚本*** 再多说一句

我很清楚，我将在下面展示的所有展示 DataFrame 优越性的例子都可能以某种方式用[纯 SQL *脚本*](https://cloud.google.com/bigquery/docs/reference/standard-sql/scripting) 重新编码。但是我也很确定结果会看起来*很恐怖*。这就是为什么我添加了足迹“(至少没有那么优雅)”。因此，在下面的示例中，我将重点关注 DataFrames 和 SQL *查询*之间的差异，而不会讨论 SQL [*脚本*](https://cloud.google.com/bigquery/docs/reference/standard-sql/scripting) 的替代方案。我邀请那些不相信的人尝试用通用的、干净的和经过单元测试的纯 SQL 脚本重写下面一些最复杂的例子。祝你好运。*(万一有人疯狂到接受挑战并证明我是错的:请确保同时监控你花了多长时间。作为参考，在撰写本文时，我花了几个周末的时间来完成整个 bigquery-frame POC。*

**现在，让我们看看数据框架能为您做哪些 SQL 查询做不到的事情。**

![](img/400673ad996c7291cdc61391be4889c6.png)

这是我们开始给出真实例子的地方。顺便说一句，他是不是很可爱？(照片由 [Alvan Nee](https://unsplash.com/@alvannee?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄)

# 即时自省

DataFrames 的一个优点是，在转换链中的任何一点，程序都可以检查当前结果的模式，并对其进行操作。

## 例如:sort_columns

让我们从一个简单的例子开始:您有一个 SQL 查询，您希望输出表的列按字母顺序排序。当然，如果表只有几列，您可以用 SQL 自己完成。即使它有很多列，用 Sublime 文本编写一个 SELECT 语句，并按名称对列进行排序(只需按 F9 键)，也不到 30 秒。

SQL 的经典方法

虽然它确实解决了问题，但是这种方法有几个缺点:

*   如果在 T 中添加或重命名一个新列，就必须考虑更新最终语句。
*   如果另一个开发人员更新了您的查询，可能他们不知道或没有考虑到这种约定，而是在最终选择的末尾添加新列，而不是保持名称排序。

另一方面，使用数据帧，解决方案更加简单和可靠:

```
df = bq.sql("/* SOME TRANSFORMATION */")
**df_with_sorted_columns = df.select(*sorted(df.columns))**
```

`df.columns`返回列名列表，`sorted`对其进行排序，`df.select`生成一个 select 语句，其中列按字母顺序排序。如果您在上游转换中添加或重命名列，它不会断开。

您甚至可以编写一个`sort_columns` [函数](https://github.com/FurcyPin/bigquery-frame/blob/v0.2.7/bigquery_frame/transformations_impl/sort_columns.py)来使代码更加简单

```
from bigquery_frame.transformations import sort_columnsdf = bq.sql("/* SOME TRANSFORMATION */")
**df_with_sorted_columns = sort_columns(df)**
```

## 其他应用

这种自省能力不仅仅适用于列名:通过`df.schema`，您可以访问中间结果的整个模式(列名、类型和描述)。有了它，编写通用代码变得很容易，例如:

*   [仅选择具有给定前缀或正则表达式的列](https://stackoverflow.com/a/59778548/2087478)。
*   将转换应用于某一类型的所有列(例如，删除超出某一范围的日期:`0001–01–01`可能不是正确的出生日期，并且可能会使一些工具崩溃。)

![](img/10d40d4cb2797104f8d2cb9a38049c33.png)

没有数据帧的 SQL 让我像一条被拴着的狗一样难过。(照片由[曲赛·阿库德](https://unsplash.com/@qusaiakoud?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄)

# 链接操作

正如我们刚刚看到的，DataFrames 允许您检查中间结果的模式 的 ***，并根据它进行下一次转换，但是您也可以直接检查数据*** *的**。这也不能用 SQL *查询*来完成，这是有原因的:无论您使用哪种 SQL 引擎，SQL 查询在运行之前都必须完全*编译——并且输出模式必须完全*已知。这使得 SQL 优化器能够完全规划和优化查询。一些高级引擎能够自适应查询执行，例如在运行时而不是编译时选择正确的连接类型: [Spark 在 3.0](https://databricks.com/blog/2020/05/29/adaptive-query-execution-speeding-up-spark-sql-at-runtime.html) 中添加了这个特性，可能 BigQuery 和 Snowflake 也有类似的优化，但它们都保守秘密。但是这样的 SQL 优化已经非常先进了。
下面的例子将演示 DataFrames 如何允许用户根据中间结果有条件地调整他们的查询:让我们做一个 **pivot** ！*****

## 一个例子:pivot

Pivot 是一个非常常见的转换示例，Excel 用户希望使用 SQL 进行转换，但却无法轻松完成。当然，大多数先进的 SQL 引擎，如 [Spark](https://spark.apache.org/docs/3.2.0/sql-ref-syntax-qry-select-pivot.html) 或 [BigQuery](https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#pivot_operator) ，最终都在它们的 SQL 语法*中增加了一个 PIVOT 语法(在*[*2018*](https://spark.apache.org/releases/spark-release-2-4-0.html)*中用于 Spark，在*[*2021*](https://cloud.google.com/bigquery/docs/release-notes)*中用于 BigQuery)。* **但是……这里面有猫腻！Spark 的 pivot 语句在 SQL 中不如在 DataFrame 中强大，正如我们将看到的，我们在 BigQuery 和 bigquery-frame 之间获得了相同的结果。**

为了理解为什么，让我们看一下 BigQuery 的 PIVOT 语法。[在他们的文档](https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#pivot_operator)中，他们给出了下面的例子:在生成一个看起来像这样的表格之后:

```
+---------+-------+---------+------+
| product | sales | quarter | year |
+---------+-------+---------+------|
| Kale    | 51    | Q1      | 2020 |
| Kale    | 23    | Q2      | 2020 |
| Kale    | 45    | Q3      | 2020 |
| Kale    | 3     | Q4      | 2020 |
| Kale    | 70    | Q1      | 2021 |
| Kale    | 85    | Q2      | 2021 |
| Apple   | 77    | Q1      | 2020 |
| Apple   | 0     | Q2      | 2020 |
| Apple   | 1     | Q1      | 2021 |
+---------+-------+---------+------+
```

他们邀请我们尝试这样一个支点:

```
SELECT * FROM
  Produce
  PIVOT(SUM(sales) FOR quarter **IN ('Q1', 'Q2', 'Q3', 'Q4')**)
+---------+------+----+------+------+------+
| product | year | Q1 | Q2   | Q3   | Q4   |
+---------+------+----+------+------+------+
| Apple   | 2020 | 77 | 0    | NULL | NULL |
| Apple   | 2021 | 1  | NULL | NULL | NULL |
| Kale    | 2020 | 51 | 23   | 45   | 3    |
| Kale    | 2021 | 70 | 85   | NULL | NULL |
+---------+------+----+------+------+------+
```

这里我强调了有问题的部分`FOR quarter **IN ('Q1', 'Q2', 'Q3', 'Q4')**` : BigQuery *希望您知道*您想要透视的列中的值。他们很好地选择了他们的例子，因为没有人期望有一天看到`**'Q5'**` 出现在那些值中。但假设你按国家做了一个支点，你的公司推出了一个新的国家。您真的希望每次发生这种情况时都必须更新您的透视查询吗？如果查询能够首先自动推断透视列中的值，然后再进行透视，不是更简单吗？仔细想想，纯 SQL PIVOT 语法做不到这一点并不奇怪:还记得我们说过“在运行之前，SQL 查询必须完全编译，输出模式必须完全已知”吗？这是这种限制的一个很好的例子。

如果你看看 [Spark-SQL 的 pivot 语法](https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-pivot.html)，它也有同样的问题。然而，如果你看看 [Spark DataFrame 的 pivot 方法](https://github.com/apache/spark/blob/4f25b3f71238a00508a356591553f2dfa89f8290/sql/core/src/main/scala/org/apache/spark/sql/RelationalGroupedDataset.scala#L330-L379)，你会看到它有两种风格:`pivot(pivotColumn)`和`pivot(pivotColumn, values)`。这些方法的文档解释了原因:

> pivot 函数有两个版本:一个要求调用者指定要透视的不同值的列表，另一个不要求。后者更简洁，但效率较低，因为 Spark 需要首先在内部计算不同值的列表。

正如我们刚刚看到的，在 SQL 版本或 BigQuery 中，只有第一种类型的 pivot 可用。

因此，我在 bigquery-frame [中实现了一个 pivot(和 unpivot)](https://github.com/FurcyPin/bigquery-frame/blob/v0.2.8/bigquery_frame/transformations_impl/pivot_unpivot.py) 方法，其工作方式类似于 Spark DataFrame 的方法。我也用[一个例子](https://github.com/FurcyPin/bigquery-frame/blob/v0.2.8/examples/pivot.py)来说明。我不会在细节上停留，因为我们还有两个例子要讲，我邀请你去看一看，了解更多的信息。有一点需要注意:我做了两个不同的实现:[第一个](https://github.com/FurcyPin/bigquery-frame/blob/7651e2f1a2a1c7d644b4e436cd49039048e3cafa/bigquery_frame/transformations_impl/pivot_unpivot.py#L77-L93)甚至没有使用 BigQuery 的 PIVOT 语句，只有 GROUP BY，而[第二个](https://github.com/FurcyPin/bigquery-frame/blob/7651e2f1a2a1c7d644b4e436cd49039048e3cafa/bigquery_frame/transformations_impl/pivot_unpivot.py#L96-L111)使用了。正如您所看到的，这两个实现只需要不到 15 行代码，这表明用 DataFrames 编写 pivot 方法比用 PIVOT 语句扩展 SQL 语法要简单得多(这需要在开始之前更新 SQL lexer/parser)。

顺便说一下， [dbt-utils 的 pivot](https://github.com/dbt-labs/dbt-utils/blob/0.8.0/macros/sql/pivot.sql) 受到了与纯 SQL 语法相同的限制:[将被转换为列的行值必须事先知道](https://github.com/dbt-labs/dbt-utils/blob/68b4b4dadc20cd5cc2a894bd2ad62aa1b8176dc7/macros/sql/pivot.sql#L31)。

## 其他应用

这里的应用是无限的:一旦您可以链接多个转换，并使下一个转换的性质依赖于前一个转换，可能性是巨大的。这里我们做了一个简单的旋转，但是如果我们在组合中添加循环的**，我们可以做一些惊人的事情，比如:**

*   实现在 BigQuery 上运行的图形算法，就像 [graphframes](https://graphframes.github.io/graphframes/docs/_site/index.html) 在 Spark 上做的那样。
*   实现高级功能工程逻辑，首先分析 BigQuery 表，然后根据列的分布自动选择如何转换每一列。所有这些都运行在 BigQuery 上。

![](img/29c0f1641c51cd4ddaf547441d4a0e8a.png)

数据帧有助于减少代码重复。(照片由[巴拉蒂·坎南](https://unsplash.com/@bk010397?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄)

# 通用转换

使用数据帧可以做的另一件事是对我们的中间表应用通用转换。这主要归功于上面描述的动态自省特性，但是它可以比简单地对数据帧的列进行排序更进一步。

## 一个例子:扁平化

SQL 从一开始就非常不擅长处理嵌套数据。在 2010 年之前，大多数 SQL 数据库甚至不支持嵌套数据格式，直到 Apache Hive 开始这样做。如今，我发现在处理嵌套数据时，BigQuery [比其他 SQL 引擎(包括 Spark-SQL)做得更好。但这完全是另一个话题，我打算以后再写一篇关于这个话题的文章。即使使用 BigQuery，我们也可能经常需要*展平*一个嵌套结构，以便能够将其导出为 csv 格式，或者使其可以被 Excel 或 Power BI 等不支持*记录*(Spark 中的*结构*)类型的“过时”工具使用。](https://cloud.google.com/bigquery/docs/reference/standard-sql/arrays)

在 bigquery-frame 中，我添加了一个 [flatten](https://github.com/FurcyPin/bigquery-frame/blob/main/bigquery_frame/transformations_impl/flatten.py) 方法，该方法可用于将所有不重复的记录(也称为 structs)自动展平为简单的列。

我前阵子给 pySpark 写过一个类似的函数，还有一个 unflatten 方法，就是反向操作。Spark 比 BigQuery 简单，因为它支持列名中的任何字符，而 [BigQuery 只支持字母、数字和下划线](https://cloud.google.com/bigquery/docs/schemas)。

如果您尝试使用它，有一点需要注意:它不会拉平重复的记录(在 Spark 中也称为`array<struct<...>>`)。原因很简单:如果您有一个包含两列重复记录的表，取消嵌套(在 Spark 中也称为`lateral view explode(...)`)将为您提供两个数组的 *笛卡尔积*中的*每个元素一行。这可能很快导致组合爆炸，正确处理每个数组取决于用户。*

## 其他应用

我希望有一天能实现的一个应用程序是一种方法，它能使在嵌套结构中应用转换变得容易。

假设您有一个嵌套了 5 层或更多层的表，您的列类型看起来像`(array<struct<array<struct<array<struct<…>>>>>>)`一样难看，您只想对结构最深层的列应用 *coalesce()* 。祝你用纯 SQL 写这篇文章好运。我可以很容易地想象出一种像 [dpath](https://github.com/dpath-maintainers/dpath-python) 处理 JSON 那样处理 SQL 的方法。看起来像这样的东西:

```
df.transform_column(**"a.b[*].c"**, lambda c: f.coalesce(c, f.lit(""))
```

![](img/897dfd8f4403d815531d3cc5dcf38b8b.png)

这就是当你达到抽象的极限时的感觉。(图片由[杰米街](https://unsplash.com/@jamie452?utm_source=medium&utm_medium=referral)上 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄)

# 更高级别的抽象

通过最后一个例子，我们将展示更高级别的抽象如何帮助我们自动化数据剖析。

## *举例:分析*

我在 bigquery-frame 中实现了一个我称之为 [analyze](https://github.com/FurcyPin/bigquery-frame/blob/7651e2f1a2a1c7d644b4e436cd49039048e3cafa/bigquery_frame/transformations_impl/analyze.py#L124) 的方法。它类似于[熊猫。DataFrame.describe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) 和[py spark . SQL . data frame . summary](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.summary.html)，但是它没有给出完全相同的统计数据。Analyze 给出每一列:列名、类型、计数、非重复计数、空值数量、最小值、最大值和 100 个最频繁值的近似值(使用 BigQuery 的 [APPROX_TOP_COUNT](https://cloud.google.com/bigquery/docs/reference/standard-sql/approximate_aggregate_functions#approx_top_count) 方法)。它还支持嵌套结构，并自动取消嵌套记录和重复记录。

下面是一个简单的例子:
*(* [*完整的例子包括进口和 df 的创造这里*](https://gist.github.com/FurcyPin/fdab5ab1e71c98c5507e9f2525356982) *)*

如果你安装了 [*【熊猫】*](https://pypi.org/project/pandas/)[*py arrow*](https://pypi.org/project/pyarrow/)和 [*xlsxwriter*](https://pypi.org/project/XlsxWriter/) ，你甚至可以用这个单行程序将结果导出到 Excel 文件中:

```
analyzed_df.toPandas().to_excel(“output.xlsx”, engine=’xlsxwriter’)
```

## 其他应用

在计算机科学的历史上，更高层次的抽象就像工业革命:它们释放了如此多的可能性，以至于它们可以完全改变人们的工作方式。这里有一些更高级抽象的例子，人们可以用数据框架更容易地构建它们。

*   自动比较两个数据帧的构建工具([这里是 Spark](https://github.com/univalence/spark-tools/tree/master/spark-test#comparing-dataframes) 中的一个开源例子)。这对于执行非回归测试特别有用。假设我重构了一个 SQL 查询，使其更加清晰。我想确定的第一件事是查询的结果将保持不变。今天，每个软件工程师都使用 git 的 diff 来审查他们对代码所做的更改。明天，每个分析工程师还应该区分表，并审查他们对数据所做的更改。希望将来我有时间将这个特性添加到 bigquery-frame 中。我也刚刚发现 Datafold 现在提议用带有用户界面的来呈现不同的结果，这非常令人兴奋。
*   建立一个完整的[数据争论](https://sonra.io/wp-content/uploads/2016/02/DSS_Wrangle_Recipe.png)接口，就像早期的 Dataiku 一样。自那以后，它可能发展了很多，但在 2013 年，它曾被称为 [Dataiku Shaker](https://annuaire.cnll.fr/societes/791012081) ，只是一个链接数据帧转换的 GUI(首先是 Pandas，然后是 Spark 以获得更大的可扩展性)。
*   构建更高级别的管道转换工具，如 [Hamilton](https://github.com/stitchfix/hamilton) (基于 pandas dataframes)，它允许获得列级血统，以帮助管理复杂的数据科学特性转换。具有讽刺意味的是，我认为使用 SQL 比使用 DataFrames 更容易实现列级血统，因为可以对 SQL 查询进行静态分析。

![](img/917cab8400cc521161b2684720e0a408.png)

不管事情发展到哪一步，我都很期待。(在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由[安德鲁·庞斯](https://unsplash.com/@imandrewpons?utm_source=medium&utm_medium=referral)拍摄)

# 结尾部分

## 该方法的局限性

**【查询太复杂】**

如果您深入分析方法的[代码，您会发现我不得不使用一些奇怪的变通方法来处理下面的错误消息:](https://github.com/FurcyPin/bigquery-frame/blob/v0.2.8/bigquery_frame/transformations_impl/analyze.py)

> *没有足够的资源用于查询规划——子查询太多或查询太复杂*

在我第一次实现中，当我试图分析包含数百列的表时，出现了这条消息。所以我不得不在返回整个结果之前通过[将中间结果保存到临时表](https://github.com/FurcyPin/bigquery-frame/blob/7651e2f1a2a1c7d644b4e436cd49039048e3cafa/bigquery_frame/transformations_impl/analyze.py#L202)来解决这个问题。这使得执行速度变慢，因为我运行了几个查询而不是一个，但是由于 BigQuery 的计费模型，这并没有改变计费成本，因为我只读取每一列一次，并且我存储在临时表中的中间结果非常小。

这显然是我的 POC 方法的局限性:与 Spark 不同，BigQuery 目前仅适用于 SQL 查询，而不是具有 300 个子步骤的疯狂复杂的转换。我找不到任何文档来解释到底是哪个 [BigQuery 的硬阈值](https://cloud.google.com/bigquery/quotas#query_jobs)触发了这个错误:例如，我仍然远远低于“最大未解析标准 SQL 查询长度”的 1MB 限制。我认为这个查询有太多的子阶段，查询计划器可能消耗了太多的 RAM 而崩溃，或者类似的情况。我找到了一个解决方法，但是与 Spark 相比，它确实使得实现高级抽象更加困难。

**添加 Python UDFs 将会很困难**

pySpark DataFrames 的另一个优点是，您可以轻松地添加 Python UDFs，将复杂的业务逻辑应用于您的行。诚然，这在性能方面不是最佳的，但对开发人员来说，生产率的提高是巨大的:经过单元测试的 Python UDF 通常比(通常)看起来像 SQL 的实现更容易编写和维护，即使您用 CREATE FUNCTION 语句将它封装在 dbt 宏或 BigQuery SQL UDF 中:单元测试、维护甚至部署将更加困难(如果您使用 SQL UDF)。

在 bigquery-frame 中添加 Python UDFs 听起来很困难，尽管可能并非不可能:bigquery 最近发布了一个名为 [Remote Functions](https://cloud.google.com/bigquery/docs/reference/standard-sql/remote-functions) 的新特性，它允许您在云函数中部署任意代码(包括 Python 代码！)然后通过 BigQuery UDF 调用这些函数。乍一看，这似乎需要很多步骤来建立一个新的 UDF，但也许有一天这一切都可以自动化，就像 pySpark 所做的那样。

**调试你的代码**

生成 SQL 字符串的框架的另一个缺点是有时会使调试变得困难。错误消息将描述生成的 SQL 字符串中的问题，而不是您的 Python 代码。但是 pySpark DataFrame 的错误信息也并不总是很好，所以这可能是平衡的。

## SQL 也有优势

如果我不对纯 SQL 相对于 DataFrame 的实际优势说几句话，这种比较就不太公平了。

*   SQL 编写速度更快:当我需要编写一次性代码来执行调查时，我会用 SQL 比 DataFrame 快得多。但是我可以在数据帧中写 SQL，所以我很好。
*   **SQL 让审计变得更加容易:**一些公司必须遵从监管机构的要求，能够审计他们的数据。他们希望能够知道谁查询了数据集，以及他们对数据集做了什么。这对于 SQL 来说要容易得多，因为转换的完整描述保存在一个 SQL 字符串中(当然，除非您调用外部函数)。使用 pySpark，由于您的 DataFrame 代码可以做任何事情，调用任何库，并且分布在数百个文件之间，因此您必须存储曾经在您的数据上运行的代码的每个版本。
*   **SQL 可以进行静态分析:**我已经提到过这一点，但是使用 SQL 构建[列级血统](https://sqlflow.gudusoft.com/#/)要比使用 DataFrames 容易得多。Monzo 写了一篇文章解释他们内部是如何做的，这是一篇非常有趣的文章。
*   **对于数据帧，你不能使用 dbt:** 显然，以数据帧为中心的管道需要一个像 dbt 这样的工具，使它们更容易构建和维护，就像 dbt 使用 SQL 一样。我确信在接下来的几年里，更多像 [fal](https://github.com/fal-ai/fal) 这样的开源项目将会出现。很难说他们是否能赶上 dbt 令人难以置信的受欢迎程度。

正如我们所看到的，bigquery-frame 方法确实有一些限制，因为它会生成 SQL 字符串。但是由于这一点，它也受益于 SQL 的一些优点，比如更容易的审计和编译查询的静态分析。

## 走向 SQL 的统一库？

也许有一天，某个足够疯狂的人会试图将我的 POC 扩展成一个统一的数据框架抽象，它可以与任何 SQL 方言(不仅仅是 BigQuery，还有 Snowflake、Postgres、Azure Synapse、Spark-SQL 等)一起工作。这将有一个很大的优势:任何构建在这个抽象之上的人都可以将他们的技术应用到任何 SQL 引擎中。就拿 [Malloy](https://github.com/looker-open-source/malloy) 来说吧，目前只对 BigQuery 和 Postgres 有效。如果他们使用像 DataFrames 这样的中间抽象级别，他们可以更快地扩展到其他语言，并与其他项目合作解决这个问题。

在我写完这篇文章的几个小时后，我看到了 [George Fraser 为 Fivetran](https://www.fivetran.com/blog/can-sql-be-a-library-language) 写的最后一篇博客。他声称 SQL 可以成为一种库语言。我绝对同意这将是伟大的。但是他接着说:

> 首先，我们必须接受这样一个事实:不同的 SQL 实现实际上是不同的语言，而且需要为每个数据库管理系统单独构建开源库

**好吧，如果这真的是业界所需要的，为什么不构建一个通用的数据框架抽象，负责翻译成任何 SQL 方言，让每个人都从中受益，并专注于在它的基础上构建伟大的库呢？**

这种方法的一个明显的缺陷是获得一个[泄漏抽象](https://www.joelonsoftware.com/2002/11/11/the-law-of-leaky-abstractions/)的风险。但我不太担心这个。还记得我说过数据帧是 SQL 的超集吗？这意味着无论何时你需要，你仍然可以回到普通的 SQL 代码。所以，如果你真的需要利用这个只有 BigQuery 拥有而 bigquery-frame 还不支持的特性，你仍然可以用纯 SQL 编写它。

![](img/c8ece0b5d662d89efb42f2e6d2e4057c.png)

斯巴基和比格并排跑着。(照片由 [Alvan Nee](https://unsplash.com/@alvannee?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄)

# 结论

通过写这篇文章，我想传达以下信息:

*   Dataframe API + Python 提供了比 pure-SQL + Jinja 更强大的抽象层次，我希望用几个例子来证明这一点。
*   它运行在*相同的基础设施上，*允许访问更高阶的转换，而不需要使用另一种技术(如 Spark、Dask 或 Vaex)设置集群。
*   Spark 用户和 Databricks 已经知道这个*很多年了*。
*   雪花正在迅速赶上[雪场](https://docs.snowflake.com/en/developer-guide/snowpark/index.html)。
*   *dbt 实验室*应该承认这是他们最大的弱点之一——事实上，[他们已经这么做了](https://blog.getdbt.com/why-doesn-t-dbt-support-python/)——*，如果他们想扩大他们的影响范围，超越分析工程师，就要建立一个长期计划来解决这个问题*。但是将数据帧支持直接添加到 dbt 中似乎与最初的设计相去甚远，这可能会非常困难。有些，比如 fal.ai，[已经在朝这个方向努力了](https://github.com/fal-ai/fal)，提供了一种在 dbt 作业之间运行 pandas 任务的方法。
*   **最后但同样重要的是:** *我觉得谷歌现在落后了，作为 BigQuery 的超级粉丝，这让我很难过。*

**这最后一点确实是促使我首先开始这个**[**biqquery-frame**](https://github.com/FurcyPin/bigquery-frame)**POC 的原因**。我真诚地希望它能帮助说服谷歌的一些人开始一个类似的项目，如果他们还没有这样做的话。BigQuery 用户需要做一些正确的事情，而不仅仅是像这样的概念验证。我看到了几种可能的方法:

1.  *继续改进*[*big query-frame*](https://github.com/FurcyPin/bigquery-frame)*，生成 SQL 字符串*。毕竟，像 Tableau 或 Looker 这样的数十亿美元的公司大多基于自动生成的 SQL(和 dataviz)，Looker 创始人的新项目 [Malloy](https://github.com/looker-open-source/malloy) 也是如此。我认为这种方法的主要问题是臭名昭著的“*查询太复杂*”错误( *c.f.* *方法的限制*)。BigQuery 团队将不得不进一步推动这种硬限制，以防止这种错误经常发生。这可能不太符合 BigQuery 的按需定价模型，即每读取 1tb 的数据进行计费。
2.  *为 BigQuery 提出一个真正的 DataFrame API，像 Spark Catalyst 那样直接编译逻辑计划*。我不知道 Snowpark 的实现细节，但我怀疑他们就是这么做的(而不是我这个可怜人的生成 SQL 字符串的解决方案)。一个好处是它可以返回比 bigquery-frame 中的 SQL 编译错误更好的错误消息。如果谷歌不想暴露太多 BigQuery 的内部信息，这对他们来说可能是一个挑战(但他们确实开源了 BigQuery 的 lexer/parser ，所以他们可能不会介意)。Spark 具有开源的优势，这使得它可以在本地运行。因此，我在 bigquery-frame 中实现的转换的单元测试比在 PySpark 中实现的转换要慢得多。
3.  *在 Apache Beam 的数据框架及其与 BigQuery 的集成上投入了大量精力*。这种方法很有意义，因为在 GCP，Apache Beam 运行在数据流上，这看起来像是谷歌内部建立的最接近 Spark 的东西。也许甚至 BigQuery 和 DataFlow 在表的后面共享一些共同的部分。理论上，Apache Beam 已经有了 DataFrame API，并且已经与 BigQuery 很好地集成在一起。不过，可能是我看的不够仔细，我还是没有找到任何一个用 Apache Beam 看起来这么简单的代码示例:`bigquery.table("source_table").withColumn(...).write("dest_table")`，
4.  将他们的用户导向 Dataproc 上的 Spark。这确实有效，但让我很难过，因为我认为 BigQuery 有一些 Spark-SQL 没有的巨大品质(但那是另一个故事了)。

**项目链接**

*   在 github 上:[https://github.com/FurcyPin/bigquery-frame](https://github.com/FurcyPin/bigquery-frame)
*   关于 Pypi:[https://pypi.org/project/bigquery-frame/](https://pypi.org/project/bigquery-frame/)

我不认为在不久的将来我会有太多的时间花在改进 bigquery-frame 上，尽管如此，我会试着关注任何问题或拉请求。当然，任何用户反馈或贡献都是最受欢迎的。请记住，在您决定将其用于生产之前，这只是一个概念验证；-).

**感谢您的阅读！**

![](img/21d3e67807c31cfe6420469cd32f0770.png)

写完这篇文章后的我(图片由 [Lucas Expedidor](https://unsplash.com/@lucasexpedidor?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄)