# DISTINCT 不是 SQL 函数

> 原文：<https://towardsdatascience.com/sql-select-distinct-277c61012800>

## 在 SQL 中使用 DISTINCT 关键字时，括号的使用如何会导致混淆

![](img/5f4e7b536a1bf6537ef0056878368c84.png)

斯科特·韦伯在 [Unsplash](https://unsplash.com/s/photos/unique?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

许多 SQL 用户(甚至是更有经验的用户)最常犯的一个错误是将`DISTINCT`应用于指定列的方式。常见的情况是，SQL 查询试图对查询需要返回的列子集应用一个`SELECT DISTINCT`子句。

误解在于这样一种观点，即`DISTINCT`是一个基本上接受它将要应用的列名的函数，而在计算结果时不会考虑在“函数调用”之外指定的列。

但是在现实中，当试图使用圆括号来使`DISTINCT`只对圆括号中的列有效时，并不会像你期望的那样工作。

现在让我们创建一个示例表，我们将在本文中引用它来演示一些概念，并帮助我们准确地阐明`DISTINCT`关键字在 SQL 中是如何工作的。

创建示例租赁表—来源:作者

现在让我们在新创建的`address`表中添加一些记录。

将示例行添加到我们的示例租赁表中—来源:Author

现在让我们查询结果以查看最终的示例表:

示例租赁表的内容—来源:作者

## 对 SQL 中 DISTINCT 的误解

现在假设我们想从我们的`rental`表中获得不同的行，使用两个不同的字段，即`customer_id`和`store_id`。换句话说，我们想回答以下问题:

> 我们的租赁表中`customer_id`和`store_id`有哪些独特的组合？

为了回答上面的查询，我们可以简单地查询我们的表并获取`customer_id`和`store_id`列的`DISTINCT`值:

选择租赁表中 customer_id 和 store_id 字段的唯一组合-来源:作者

现在，如果我们只想检索一组唯一的客户，这样在上面的查询结果中，我们只能看到每个客户的一行，那么我们需要细化我们的查询来做到这一点。

这正是对`DISTINCT`的误解所在。很多用户，都有(错！)印象中，`DISTINCT`是一个函数，在这个函数中，我们可以指定在将它应用于目标表时要考虑的列。

如果您在“调用”时试图将`customer_id`括在圆括号中(这里不是正确的动词，因为这不是一个函数)`DISTINCT`，您会注意到它根本不起作用:

使用括号中的列选择 DISTINCT 来源:作者

我们仍然可以在查询结果中看到“重复的”客户 id。这是因为`SELECT DISTINCT`子句，将总是考虑所有指定的列名，不管它们是否被括在括号中。

事实上，下面显示的所有表达式确实是等价的:

*   `SELECT DISTINCT customer_id, store_id FROM rental;`
*   `SELECT DISTINCT (customer_id), store_id FROM rental;`
*   `SELECT DISTINCT (customer_id), (store_id) FROM rental;`
*   `SELECT DISTINCT (store_id), customer_id FROM rental;`
*   `SELECT DISTINCT ((customer_id)), store_id FROM rental;`

最后，我强烈建议**在将** `**SELECT**` **子句与** `**DISTINCT**` **限定词**一起使用时避免使用括号，因为这可能会使其他人(他们可能不知道我们今天讨论的内容)误解查询，并意外地认为您的意图是在单个列上应用`DISTINCT`，尽管我们已经演示过这是不可能的。

## PostgreSQL 和 DISTINCT ON

如果您正在使用 Postgres，并且希望将`DISTINCT`仅应用于您想要在结果中检索的列的子集，那么您可以利用`DISTINCT ON`。

> `SELECT DISTINCT`从结果中删除重复行。
> 
> `SELECT DISTINCT ON`消除与所有指定表达式匹配的行。
> 
> — Postgres [文档](https://www.postgresql.org/docs/current/sql-select.html)

这是建立在标准 SQL 的`DISTINCT`之上的扩展，它返回匹配指定表达式的每组行的第一行。

SQL 中带有 DISTINCT ON 子句的示例—来源:作者

但是请注意，当使用`DISTINCT ON`时，使用`ORDER BY`子句也是有意义的。这样，您将能够指定从冲突的行中挑选所需结果的条件。例如，如果有两行匹配您的表达式(在上面的例子中，我们有两条记录符合 id 为`100`的客户)。

现在，让我们假设我们想要获取具有相应商店 ID 的唯一客户 ID，但是这一次，如果存在多个竞争行，我们想要获取具有最小数量的行:

选择 DISTINCT ON with ORDER BY 来源:作者

请注意，对应于`customer_id=100`的`store_id`已经更改，因为金额最小的租赁行已经不同，因为我们已经按金额升序对结果进行了排序。

但是一般来说，如果你真的不在乎顺序，那么你可以忽略它。

## 最后的想法

理解 SQL 中的`DISTINCT`关键字如何与`SELECT`语句一起工作是很重要的，因为这对许多用户来说是一个困惑的来源——我甚至可以说对有经验的用户来说也是如此。

当编写带有`SELECT DISTINCT`子句的查询时，许多用户倾向于像使用适当的 SQL 函数一样使用`DISTINCT`。换句话说，它们将一列括在括号中，同时在子句后提供更多的列名，例如`SELECT DISTINCT(user_id), first_name FROM ...`。

当阅读这样的查询时(显然还有编写它们的人)，您可能最终会认为`SELECT DISTINCT`只适用于指定的列(例如`user_id`)，而不适用于剩余的列(例如`first_name`)。正如我们在今天的文章中看到的，在编写查询时，这是一个误解和非常危险的假设。

最后，我们讨论了 PostgreSQL 数据库中的一个特例，它允许用户使用特殊的`DISTINCT ON`子句明确指定在应用`DISTINCT`时要考虑的列。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

<https://gmyrianthous.medium.com/membership>  

**相关文章你可能也喜欢**

</ddl-dml-e802a25076c6>  </star-schema-924b995a9bdf>  </standard-vs-legacy-sql-bigquery-6d01fa3046a9> 