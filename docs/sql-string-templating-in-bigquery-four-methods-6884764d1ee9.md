# BigQuery 脚本中的 SQL 字符串模板:四种方法

> 原文：<https://towardsdatascience.com/sql-string-templating-in-bigquery-four-methods-6884764d1ee9>

# BigQuery 脚本中的 SQL 字符串模板:四种方法

## 强大的基础技术有助于释放 BigQuery 脚本和自动化的力量

![](img/7ab89e452d0e3fcd383cf19b943f7fd8.png)

你应该避免这种混乱。[万用眼](https://unsplash.com/@universaleye?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

随着[脚本](https://jimbeepbeep.medium.com/getting-started-with-bigquery-scripting-45bdd968010c)的出现，尤其是[执行立即](https://cloud.google.com/bigquery/docs/reference/standard-sql/scripting#execute_immediate)语句的出现，在 BigQuery 中使用字符串变得更加强大。这使您能够将 SQL 查询构造为字符串，然后在临时或预定脚本中，或者在可调用的[过程](https://cloud.google.com/bigquery/docs/reference/standard-sql/scripting-concepts) 或[函数](https://cloud.google.com/bigquery/docs/reference/standard-sql/user-defined-functions) **中执行构造的 SQL 查询。**

这意味着，通过一点基于字符串的技巧，您可以突然自动执行以前需要 GCloud 或 API 访问的操作，简化您的工作流程，这样您就永远不需要离开控制台。

然而，正如生活和软件开发中的大多数事情一样，有许多方法可以达到相同的结果，每种方法都有其优缺点。在这篇文章中，我将概述将字符串值注入 SQL 字符串的四种不同方法，并讨论何时以及如何使用它们来编写干净、清晰且易于理解的 SQL 代码。

本文中的所有代码片段都可以复制粘贴并直接在 BigQuery 中执行，所以请随意打开您的查询编辑器和代码，或者将它们作为您自己实验的基础。

最后还有一个这些技术的示例实现，它将在自动化和功能方面为您打开一个机会的世界。

# 1.串联(激活)

![](img/c0f3e436f59c21b90330ca73ab9abd40.png)

不要试图连接真正的猫，它们有非常锋利的爪子。在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Manja Vitolic](https://unsplash.com/@madhatterzone?utm_source=medium&utm_medium=referral) 拍摄的照片

第一种方式对于任何使用电子表格的人来说都是熟悉的(也就是说，世界上几乎所有曾经使用过数据的人):[CONCAT](https://cloud.google.com/bigquery/docs/reference/standard-sql/string_functions#concat)(Concatenate 的缩写)。在最简单的形式中，它将多个逗号分隔的字符串作为参数，将它们连接起来并作为单个字符串返回:

```
SELECT 
CONCAT("This ", "is ", "a ", "useless ", "example.") AS example
```

这将产生以下输出:

```
This is a useless example.
```

这是正确的。

让我们来看一个更有用的例子，用粗体突出显示 CONCAT 函数语法:

```
WITH 
example_project_metadata AS (
SELECT
"flowfunctions" AS project_id,
"examples" AS dataset_name,
"a_more_useful_example" AS table_name
)SELECT 
**CONCAT("`", project_id, ".", dataset_name, ".", table_name, "`") 
AS table_ref**
FROM example_project_metadata
```

如果你认不出以**和**开头的结构，那么今天对你来说是非常重要的一天，可能比你在 Excel 中学习 VLOOKUP 的那一天还要重要。希望你现在不需要，因为你已经意识到电子表格是人类数据滥用的易错工具。

这是一个常见的表表达式(CTE)，一个简单但奇妙的构造，它使您能够对数据执行连续的原子操作，编写当从上到下阅读时实际上可以理解的代码(不像令人困惑的嵌套查询)，并迫使您为每个操作取别名。这意味着您可以通过 CTE 名称来解释您在每一步所做的事情(没有注释),并且还可以在查询的任何后续点重新引用来自该特定 CTE 的结果。非常整洁。

事实上，我打算稍微重写一下这个查询，因为从工作流的角度来看它会有所帮助，下面我来解释一下原因:

```
WITH 
example_project_metadata AS (
SELECT
"flowfunctions" AS project_id,
"examples" AS dataset_name,
"a_more_useful_example" AS table_name
),build_table_ref AS (
SELECT 
**CONCAT("`", project_id, ".", dataset_name, ".", table_name, "`") 
AS table_ref**
FROM example_project_metadata
)SELECT * 
FROM build_table_ref
```

这意味着在 SQL 查询编辑器中开发时，您可以通过将最终选择引用更改为不同的 CTE 名称(即 SELECT * FROM example _ project _ metadata)并执行查询，在任何阶段直观地检查数据。并不是说当数据集超过一定的(令人惊讶的小)大小时，直观地检查数据就特别有用，但是我们都是人，都喜欢观察事物。有时候这很有帮助。

上面的代码将在 **table_ref** 列中返回以下值:

```
`flowfunctions.examples.a_more_useful_example`
```

这本身并不是特别有用，但希望您能看到这是如何开始有一点点自动化的潜力。

# 2.管道连接

![](img/2a825d63938d72ec40df55fcf3af8f25.png)

我找不到一张猫抽烟斗的照片。照片由 [CHUTTERSNAP](https://unsplash.com/@chuttersnap?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

这一个非常相似，但是语法略有不同(以粗体突出显示)，使用[连接运算符](https://cloud.google.com/bigquery/docs/reference/standard-sql/operators)代替 CONCAT 函数来实现完全相同的输出:

```
WITH 
example_project_metadata AS (
SELECT
"flowfunctions" AS project_id,
"examples" AS dataset_name,
"a_more_useful_example" AS table_name
),build_table_ref AS (
SELECT 
**"`"||project_id||"."||dataset_name||"."||table_name||"`" 
AS table_ref**
FROM example_project_metadata
)SELECT * 
FROM build_table_ref
```

在我看来，这差不多了，但是根据你的使用情况、个人偏好或视力，它可以更具可读性。

# 3.格式()

![](img/7b3cf0c1f59f0c6155af3999921689ec.png)

格式化这个。费尔南多·拉文在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

[FORMAT](https://cloud.google.com/bigquery/docs/reference/standard-sql/string_functions#format_string) 函数是一个极其强大和通用的机制，用于构建严格控制注入元素的字符串。在这个例子中，我们只是将一个字符串注入到另一个字符串中(使用“%s”格式说明符)，但是您可以对可以注入的不同数据类型的格式进行细粒度控制——查看文档[这里](https://cloud.google.com/bigquery/docs/reference/standard-sql/string_functions#format_specifiers)。无论如何，在这种情况下，SQL 看起来像:

```
WITH 
example_project_metadata AS (
SELECT
"flowfunctions" AS project_id,
"examples" AS dataset_name,
"a_more_useful_example" AS table_name
),build_table_ref AS (
SELECT 
**FORMAT("`%s.%s.%s`",
project_id, dataset_name, table_name)
AS table_ref**
FROM example_project_metadata
)SELECT * 
FROM build_table_ref
```

每个“%s”标识符按顺序被每个字符串变量值替换，如果变量数量或任何变量数据类型不匹配，查询将出错。对于更复杂的结构，可以使用三引号多行字符串(类似的实现见下面的例子)，在更长的实例中，为了可读性和可追溯性，将注入的变量分隔到不同的行是有用的。

这里有一个警告:对于长语句，很难看出哪个变量映射到哪个标识符，所以另一种方法可能更容易编写和读取。

# 4.管道注射

![](img/6b6f779984b160e36546c17ef351dc8a.png)

可爱的烟斗。[西格蒙德](https://unsplash.com/@sigmund?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

最后一种变化非常类似于管道连接，在将变量注入到更长的多行 SQL 语句中时非常有用。注意，三重引号用于多行字符串(与 Python 中的完全一样)，但是它们在单行字符串中也很有用，在单行字符串中它们可能包含额外的引号字符。使用这种技术的示例代码如下:

```
WITH 
example_project_metadata AS (
SELECT
"flowfunctions" AS project_id,
"examples" AS dataset_name,
"a_more_useful_example" AS table_name,
"name" AS username_field,
"Jim" AS my_name
),build_sql_query AS (
SELECT 
"""SELECT * FROM  
`"""||project_id||"""."""||dataset_name||"""."""||table_name||"""`
WHERE """||username_field||""" = '"""||my_name||"""'
""" AS sql_query
FROM example_project_metadata
)SELECT *
FROM build_sql_query
```

这不是语法上最漂亮的代码，但是当它被打包成一个用户定义的函数(UDF)时，它就可以被编写、测试，并且永远不会再被看到。在这里，我实际上是在构造一个 SQL 查询来执行，这比构造随机的无用句子或表引用更好地使用了字符串模板。事实上，这个查询的结果将是另一个查询:

```
SELECT *
FROM `flowfunctions.examples.a_more_useful_example`
WHERE name = 'Jim'
```

在您的 SQL 查询编辑器中尝试一下，如果您有任何权限问题，请给我发一封私信(通过突出显示一些文本并单击带有挂锁的语音气泡)。

如果您想开始使用这些技术做实际有用的事情，请查看我的【BigQuery Scripting 入门文章。

虽然有更好的方法来构造代码(我更喜欢使用 UDF 来构造 SQL 并将其作为字符串返回，然后使用一个过程来执行 SQL)，但是要以最简单的方式执行该查询，您只需将其包含在 EXECUTE IMMEDIATE 语句中:

```
EXECUTE IMMEDIATE (WITH 
example_project_metadata AS (
SELECT
"flowfunctions" AS project_id,
"examples" AS dataset_name,
"a_more_useful_example" AS table_name,
"name" AS username_field,
"Jim" AS my_name
),build_sql_query AS (
SELECT 
"""SELECT * FROM  
`"""||project_id||"""."""||dataset_name||"""."""||table_name||"""`
WHERE """||username_field||""" = '"""||my_name||"""'
""" AS sql_query
FROM example_project_metadata
)SELECT *
FROM build_sql_query
);
```

就像变魔术一样，你写了一些 SQL 来编写和执行 SQL！

希望这将对一些人有所帮助，请关注我在 2022 年推出的更具结构性的系列，以帮助解锁您隐藏的数据超能力！

如果您觉得这(以及其他相关材料)有用和/或有趣，请跟我来！

如果你还不是会员，[加入 Medium](https://jim-barlow.medium.com/membership) ，每月只需 5 美元就能获得这个活跃、充满活力和激情的数据人社区的无限故事。也有很多其他人，但是如果你对数据感兴趣，那么这里就是你要去的地方…