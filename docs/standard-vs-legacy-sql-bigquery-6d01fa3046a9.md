# BigQuery 中的标准 SQL 与传统 SQL

> 原文：<https://towardsdatascience.com/standard-vs-legacy-sql-bigquery-6d01fa3046a9>

## 理解标准 SQL 和遗留 SQL 在 Google Cloud BigQuery 环境中的区别

![](img/bf48c9bc557477057a30bf2d02fde831.png)

照片由 [Unsplash](https://unsplash.com/s/photos/sql?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的 [Sunder Muthukumaran](https://unsplash.com/@sunder_2k25?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄

BigQuery 是谷歌云平台上的一项托管数据仓库服务，它允许组织持久保存他们的数据，也允许分析师访问这些数据并提取有价值的信息。

当我开始使用 BigQuery 时，我遇到了两个当时对我来说不太清楚的术语，即**标准**和**遗留** SQL。这本质上是 BigQuery 支持的两种方言，具有不同的语法、语义和功能。

## 传统与标准 SQL

过去，BigQuery 使用一种非标准的 SQL 方言执行查询，称为 **BigQuery SQL** 。然而，自从 BigQuery 2.0 发布以来，该服务现在支持标准 SQL，而以前的 **BigQuery SQL 被重命名为遗留 SQL** 。

标准 SQL 是一种 ANSI 兼容的查询语言。值得一提的是，目前，在 BigQuery 上运行查询的首选语言是标准 SQL。

保留传统 SQL 主要是为了向后兼容，因此，建议迁移到标准 SQL，因为我们预计在将来的某个时候，传统 SQL 将被弃用。数据定义语言(DDL)和数据模型语言(DML)等功能仅受标准 SQL 支持

## 主要区别

遗留 SQL 中的每种类型在标准 SQL 中都有对应的类型(反之亦然)，这意味着遗留 SQL 中的类型在标准方言中有不同的名称。关于标准方言和传统方言之间的精确映射，你可以参考[官方文件](https://cloud.google.com/bigquery/docs/reference/standard-sql/migrating-from-legacy-sql#type_differences)。

此外，与传统 SQL 相比，标准方言的类型`TIMESTAMP`的有效值范围更小。前者只接受范围在`0001-01-01 00:00:00.000000`和`9999-12-31 23:59:59.999999`之间的值。

标准方言和传统方言在查询中转义字符(如连字符)的方式也不同。在前者中，我们使用反斜杠(```)字符，而在后者中使用方括号(`[]`)。

此外，传统 SQL 在引用项目名称时使用冒号`:`作为分隔符，而标准方言需要句点`.`

```
#standardSQL
SELECT *
FROM `bigquery-public-data.samples.shakespeare`;#legacySQL
SELECT * 
FROM `bigquery-public-data:samples.shakespeare`;
```

另外值得一提的是，标准方言不支持[表装饰器](https://cloud.google.com/bigquery/docs/reference/standard-sql/migrating-from-legacy-sql#table_decorators)和其他一些[通配符函数](https://cloud.google.com/bigquery/docs/reference/standard-sql/migrating-from-legacy-sql#wildcard_functions)。

## 标准 SQL 的优势

如前所述，标准 SQL 比传统 SQL 方言有几个优点。更具体地说，它支持

*   `WITH`条款
*   用户定义的 SQL 函数
*   `SELECT`和`WHERE`条款中的子查询
*   插入、更新和删除
*   更多数据类型，如`ARRAY`和`STRUCT`
*   更准确的`COUND(DISTINCT ..)`子句(与过去有许多重大限制的遗留 SQL 方言的`EXACT_COUNT_DISTINCT`相比)
*   相关子查询
*   和更复杂的`JOIN`谓词

关于标准 SQL 方言的这个功能的实际例子，你可以参考官方 BigQuery 文档[。](https://cloud.google.com/bigquery/docs/reference/standard-sql/migrating-from-legacy-sql#standard_sql_highlights)

## 更改默认方言

BigQuery 上的默认方言是标准 SQL。但是，这可以通过在 SQL 查询中包含前缀来改变。如果您希望切换到遗留 SQL，您需要在指定查询之前包含前缀`#legacySQL`。标准 SQL 对应的前缀是`#standardSQL`。

请注意，这些前缀必须在查询之前，不区分大小写，并且在查询和前缀本身之间应该有一个换行符。

例如，考虑以下使用传统 SQL 方言的查询:

```
#legacySQL
SELECT
  weight_pounds, 
  state, 
  year, 
  gestation_weeks
FROM
  [bigquery-public-data:samples.natality]
ORDER BY
  weight_pounds DESC
LIMIT
  10;
```

## 最后的想法

BigQuery 无疑是 Google 云平台上最受欢迎的云服务之一，因为几乎每个现代组织都需要托管数据仓库服务。

因此，利用大多数可用的特性来帮助您有效地、大规模地解决问题是非常重要的。

在今天的文章中，我们讨论了 BigQuery 的一个最基本的方面，即用于运行服务操作的 SQL 方言。请注意，我们只讨论了遗留和标准 SQL 方言之间总体差异的一小部分。

如果你还在运行传统的 SQL，我个人强烈推荐迁移到标准的 SQL 方言，因为它是 Google 推荐的方言，也提供了更强大的功能。有关差异的完整列表，请参考[官方指南“迁移到标准 SQL”。](https://cloud.google.com/bigquery/docs/reference/standard-sql/migrating-from-legacy-sql)

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**相关文章你可能也喜欢**

[](/ddl-dml-e802a25076c6)  [](/oltp-vs-olap-9ac334baa370)  [](/connect-airflow-worker-gcp-e79690f3ecea) 