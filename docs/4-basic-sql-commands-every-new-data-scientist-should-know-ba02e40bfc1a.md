# 每个新数据科学家都应该知道的 4 个基本 SQL 命令

> 原文：<https://towardsdatascience.com/4-basic-sql-commands-every-new-data-scientist-should-know-ba02e40bfc1a>

## 结构化查询语言简介

![](img/66e673718a6cc9e78be37894bcd778af.png)

迈克尔·泽兹奇在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

在这篇文章中，您将发现对于关系型[数据库](https://databasecamp.de/en/data/database)的基本工作来说最重要的 SQL 命令。[结构化查询语言(SQL)](https://databasecamp.de/en/data/sql-definition) 是处理关系数据库时最常用的语言。不管它的名字是什么，这种语言不仅仅可以用于简单的查询。它还可以用于执行创建和维护数据库所需的所有操作。

# 结构化查询语言的优势是什么？

结构化查询语言提供了许多读取、修改或删除数据的功能。此外，许多[分析师](https://databasecamp.de/en/ml-blog/business-analysts)认为它优于其他语言，原因如下:

*   它在语义上非常容易阅读和理解。即使是初学者也能在很大程度上理解这些命令。
*   这种语言可以直接在数据库环境中使用。对于信息的基本工作，数据不必首先从数据库转移到另一个工具。简单的计算和查询可以直接在数据库中进行。
*   与其他电子表格工具(如 Excel)相比，使用结构化查询语言的数据分析可以很容易地复制和拷贝，因为每个人都可以访问数据库中的相同数据。因此，相同的查询总是导致相同的结果。

SQL 提供了 Excel 电子表格中大多数执行的汇总和计算的替代方法，例如总计、平均或在列中查找最大值。这些计算也可以在多个数据集上同时进行。

# 我们使用什么样的样本数据？

为了能够实时测试最常见的 SQL 命令，我们使用来自 Kaggle 的[信用记录数据集](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)，它列出了各种匿名人的信用数据，如他们的收入、教育或职业。

我们将这个表作为 [Pandas DataFrame](https://databasecamp.de/en/python-coding/pandas-dataframe-basics) 加载到我们的笔记本中，然后可以在上面测试常见的 SQL 命令:

# 如何查询数据？

使用“选择”可以查询完整的表。由于不需要查询任何特定的列，所以我们只需使用星号“*”，以便输出所有可用的列。在“LIMIT 10”的帮助下，我们确保 SQL 命令只返回前 10 行，否则会变得太混乱:

由于有大量的列，我们只限于“姓名 _ 收入 _ 类型”和“子女”列。我们可以通过显式指定来查询它们:

正如我们所看到的，一些收入类型在这些行中出现了两次。如果多个信用嫌疑人的收入类型相同，就会出现这种情况。

为了只获得收入类型的唯一条目，我们使用附加参数“DISTINCT”:

# 如何筛选数据？

可以使用“WHERE”参数过滤数据。根据[数据类型](https://databasecamp.de/en/data/data-types)，对此有不同的查询:

*   可以使用大于号或小于号来比较数值，例如，“AMT_INCOME_TOTAL < 427500”. These can be supplemented with an equal sign, e.g. “AMT_INCOME_TOTAL < 427500”.
*   For texts or strings, the comparisons “=” or “<>”用于检查文本是否匹配(“=”)或不同(“<>”)。

对于所有收入超过$427，500 的信贷申请人，我们得到以下 SQL 命令:

为了能够在一个 SQL 命令中使用多个过滤器，我们可以用“and”或“or”将它们连接起来。通过这种方式，我们可以获得所有高收入的女性申请者:

# 如何对结果进行排序？

可以使用“ORDER BY”根据列对每个输出进行排序。数字和字符串都可以排序，然后按字母顺序排序:

默认情况下，输出总是按升序排序。要更改这一点，您还必须为降序指定“DESC ”:

# 记录怎么统计？

使用 SQL 命令“count ”,您可以计算列中或整个[数据库](https://databasecamp.de/en/data/database)中的值:

在这种情况下，我们在数据库中有 438，557 条记录。为了对列中的值进行计数，我们使用列的名称而不是星号“*”。此外，我们可以使用参数“DISTINCT”来只计算列中的唯一值:

因此，数据集中有 866 种不同的收入水平。

# 这是你应该带走的东西

*   在处理关系数据库时，结构化查询语言是使用最广泛的语言。
*   在 SQL 命令的帮助下，大量的数据查询可以被设计和个性化。

## 参考

1.  [信用卡审批预测](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)在 [CC0:公共域](https://creativecommons.org/publicdomain/zero/1.0/)许可下

*如果你喜欢我的作品，请在这里订阅*[](https://medium.com/subscribe/@niklas_lang)**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，medium 允许你每月免费阅读* ***3 篇*** *。如果你希望有****无限制的*** *访问我的文章和数以千计的精彩文章，不要犹豫，点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格**

*[](/redis-in-memory-data-store-easily-explained-3b92457be424)  [](/an-introduction-to-tensorflow-fa5b17051f6b)  [](/software-as-a-service-the-game-changer-for-small-it-departments-f841b292b02a) *