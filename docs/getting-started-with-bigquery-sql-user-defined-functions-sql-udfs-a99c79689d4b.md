# BigQuery SQL 用户定义函数(SQL UDFs)入门

> 原文：<https://towardsdatascience.com/getting-started-with-bigquery-sql-user-defined-functions-sql-udfs-a99c79689d4b>

## 一个强大的基本构建块，支持核心 BigQuery 平台功能的自定义扩展

![](img/8235e41ce8196c005e855f3b0513ca0d.png)

迷茫？不要害怕。照片由 [charlesdeluvio](https://unsplash.com/@charlesdeluvio?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

# 动机

可扩展性是 BigQuery 的关键特性之一——伴随着超级大国般的新功能的不断发展——这使得它成为一个强大的平台，可以处理任何类型的数据工作，无论大小。

然而，任何平台的功能集都可能是压倒性的，甚至很难弄清楚从哪里开始。这一系列的目的是揭开一些基本 BigQuery 特性的神秘面纱，并让您以更快的速度朝着您的特定目标前进。

# 情况

BigQuery 中有几个不同种类的“函数”,所以如果没有通过全面的(但有时是压倒性的)官方文档弄清楚每个函数的细微差别，请原谅。

首先，`User-Defined Functions (UDFs)`是被称为`Routines`的大查询资源类型家族的一部分，该家族还包括`Stored Procedures`、(也就是通常所说的脚本)、`Remote Functions`和`Table Functions`。

每一种都有不同的种类，每一种都有自己的特质。下面这篇文章是对开发`Routines`的一个温和的介绍:

</getting-started-with-bigquery-scripting-45bdd968010c>  

为了简单起见，让我们从最简单的开始:SQL 用户定义函数。

# 解决办法

在深入函数构造和语法的细节之前，一个有用的起点是展示如何实际使用它。从 SQL UDF 获得结果的语法非常简单且一致:

```
SELECT project_id.dataset_id.function_name([OPTIONAL PARAMETERS])
```

如果这个函数没有参数，那么你仍然需要包括括号，因此:

```
SELECT project_id.dataset_id.function_name()
```

您可以将它用作普通查询的一部分(在这种情况下，该函数将为查询响应的每一行返回一个值)，或者用作工作流的一部分，其中只需要一个响应，并且您希望将一些计算复杂性封装在一个易于使用的函数中。

在本例中，我们将构建一个 SQL UDF，使表引用(通常缩写为 refs)的工作变得非常简单，这对开发 BigQuery 自动化脚本和其他函数很有帮助。将使用以下语法调用该函数:

```
SELECT flowfunctions.ref.build('project_id', 'dataset_id', 'table_name')**RETURNS:**
'project_id.dataset_id.table_name'
```

我们按照地理位置(美国:`flowfunctions`)将所有`Routines`分组到单个项目中，并按照功能类型分组到数据集中。

UDF 最简单的开发工作流程实际上是*而不是*从开发实际功能开始，它是编写底层查询。通过在开始时声明您将用作参数的变量，将查询转换成函数就变成了微不足道的最后一步。

在这种情况下，我期望三个输入参数，`project_id`、`dataset_id`和`table_name`，都是`STRINGs`。我像在 BigQuery 脚本中一样声明它们，在这种情况下，使用`DECLARE`关键字在声明点设置它们的值:

```
DECLARE project_id STRING DEFAULT 'my-project';
DECLARE dataset_id STRING DEFAULT 'my_dataset';
DECLARE table_name STRING DEFAULT 'my_view';
```

我现在甚至可以运行这个脚本，它会成功运行——但是实际上什么也不会发生，因为我没有选择任何东西。如果我想检查这些值，我可以执行以下查询:

```
SELECT project_id, dataset_id, table_name
```

然而，这很无聊，让我们把他们建成一个参考！这并不难——然而，即使对于这个简单的例子，也有几种不同的方法可以实现这一点，或者使用 [**CONCAT**](https://cloud.google.com/bigquery/docs/reference/standard-sql/string_functions#concat) 函数:

```
SELECT CONCAT(project_id, ".", dataset_id, '.', table_name) AS ref
```

或者使用 [**串联运算符**](https://cloud.google.com/bigquery/docs/reference/standard-sql/operators#concatenation_operator) :

```
SELECT project_id||"."||dataset_id||'.'||table_name AS ref;
```

这两个例子都很可读，但是因为第二个例子有点短，我们就用它吧。执行这个(连同变量声明)，我们将得到我们想要的结果:

```
**RETURNS:**
my-project.my_dataset.my_view
```

太好了！现在把它放到一个函数中，这样我们可以重用和简化我们未来的代码。`User-Defined Functions`的官方文档是[这里是](https://cloud.google.com/bigquery/docs/reference/standard-sql/user-defined-functions)，然而我们只需要一个非常简单的语法来实现我们的简单函数:

```
CREATE OR REPLACE FUNCTION flowfunctions.ref.build (
project_id STRING, 
dataset_id STRING, 
table_name STRING
) AS ((
SELECT project_id||"."||dataset_id||'.'||table_name AS ref
))
```

没有必要将参数定义放在不同的行上，但是如果有很多这样的定义，会有助于提高可读性。其他需要注意的事项有:

*   您不能在查询末尾使用分号，否则会出错
*   您需要双括号，否则您会得到以下错误消息:

```
*The body of each CREATE FUNCTION statement is an expression, not a query; to use a query as an expression, the query must be wrapped with additional parentheses to make it a scalar subquery expression at [y:x]*
```

这听起来很奇怪，但却是*非常重要的*！您可以更改函数体以删除`SELECT`和`AS ref`来消除错误，但是将 SQL 查询放在双括号中(使其成为“标量子查询表达式”)现在使您能够编写任意复杂的 SQL 查询，并将它们封装到可重用的函数中。这意味着您可以以连续的方式编写美观、可读的查询(即使用通用的表表达式，而不求助于无法理解的嵌套子查询)。这是一件非常好的事情。

限制是您需要从查询中返回一个(且只有一个)“对象”，然而实际上您可以简单地将多列构建到`STRUCTs`中，将多行构建到`ARRAYs`中，这应该可以解决任何用例。

当然，那时你必须`UNNEST`他们才能使用他们，但那肯定是另一天的主题。

# 执行

要执行此功能，只需在控制台中执行以下查询:

```
SELECT flowfunctions.ref.build('project_id', 'dataset_id', 'table_name')
```

拥有`BigQuery Data Viewer`和`BigQuery Metadata Viewer`权限的`allAuthenticatedUsers`也可以使用包含该函数的数据集，因此只需几次点击，就可以对全世界开放。相当神奇！

如果您觉得这(以及其他相关材料)有用和/或有趣，请跟我来！

如果你还不是会员，那么[加入 Medium](https://jim-barlow.medium.com/membership) 吧，每月只需 5 美元，就能从这个活跃、充满活力和激情的数据人社区中获得无限的故事。也有很多其他人，但是如果你对数据感兴趣，那么这里就是你要去的地方…