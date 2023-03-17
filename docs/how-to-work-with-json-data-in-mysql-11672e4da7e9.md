# 如何在 MySQL 中使用 JSON 数据

> 原文：<https://towardsdatascience.com/how-to-work-with-json-data-in-mysql-11672e4da7e9>

# 如何在 MySQL 中使用 JSON 数据

## 在 MySQL 中学习“NoSQL”

MySQL 支持原生的 [JSON](https://en.wikipedia.org/wiki/JSON) 数据类型，该数据类型支持 JSON 文档的自动验证、优化存储和访问。尽管 JSON 数据应该更好地存储在 NoSQL 数据库中，比如 [MongoDB](https://www.mongodb.com/nosql-explained) ，但是您仍然会不时地遇到包含 JSON 数据的表。在本文的第一部分，我们将介绍如何用简单的语句从 MySQL 的 JSON 字段中提取数据。在第二部分，我们将介绍如何将 MySQL 表中的数据聚集到 JSON 数组或对象中，以便在您的应用程序中方便地使用。

![](img/faca629b98c5c0f73d662ca617320018.png)

图片来自 [Pixabay](https://pixabay.com/photos/hand-business-technology-data-3044387/) 。

要设置的系统类似于[上一篇文章](https://medium.com/codex/how-to-execute-plain-sql-queries-with-sqlalchemy-627a3741fdb1)中介绍的关于如何用 Python 执行 SQL 查询的系统。如果您已经根据该文章中的说明设置了系统，那么您可以继续下一部分。如果没有，您可以按照下面的简化说明来设置您的系统。有关命令和选项的详细解释，请参考[上一篇文章](https://medium.com/codex/how-to-execute-plain-sql-queries-with-sqlalchemy-627a3741fdb1)。

本质上，我们将在 Docker 容器中启动一个本地 MySQL 服务器:

您可以直接在上面启动的控制台中执行 SQL 查询。或者，如果您更喜欢使用图形界面，您可以安装并使用 [DBeaver](https://dbeaver.io/) ，这是一个非常棒的图形数据库管理器，适用于所有类型的数据库。如果你一直在纠结 MySQL workbench，那真的值得一试。关于如何安装和设置 DBeaver 的更多细节，本文有一个简短但有用的总结。

让我们首先探索一下可以用来从 JSON 字段中提取数据的常见 MySQL 函数和操作符。

MySQL 中有两种主要类型的 [JSON 值](https://dev.mysql.com/doc/refman/8.0/en/json.html):

*   JSON 数组—用逗号分隔并用方括号([])括起来的值列表。
*   JSON object——一个字典/hashmap/object(在不同的编程语言中名称是不同的),它有一组由逗号分隔的键值对，并包含在花括号({})中。

JSON 数组和对象可以相互嵌套，我们将在后面看到。

我们可以使用`JSON_EXTRACT`函数从 JSON 字段中提取数据。基本语法是:

```
**JSON_EXTRACT(json_doc, path)**
```

对于 JSON 数组，路径用`$[index]`指定，其中索引从 0 开始:

```
mysql> **SELECT JSON_EXTRACT('[10, 20, 30, 40]', '$[0]')**;
+------------------------------------------+
| JSON_EXTRACT('[10, 20, 30, 40]', '$[0]') |
+------------------------------------------+
| 10                                       |
+------------------------------------------+
```

对于 JSON 对象，路径用`$.key`指定，其中`key`是对象的一个键。

```
mysql> **SELECT JSON_EXTRACT('{"name": "John", "age": 30}', '$.name')**;
+-------------------------------------------------------+
| JSON_EXTRACT('{"name": "John", "age": 30}', '$.name') |
+-------------------------------------------------------+
| "John"                                                |
+-------------------------------------------------------+
```

如果上面使用的`JSON_EXTRACT`只有两个参数，我们可以使用`->`操作符，它是`JSON_EXTRACT`的别名。为了演示这个操作符的用法，我们需要一个带有 JSON 字段的表。请复制以下 SQL 查询，并在 MySQL 控制台或 DBeaver 中执行它们:

特别是，MySQL 使用`utf8mb4`字符集和`utf8mb4_bin`排序规则处理 JSON 上下文中使用的字符串。一个[字符集](https://dev.mysql.com/doc/refman/8.0/en/charset-general.html)是一组符号和编码，一个校对是一组比较字符集中字符的规则。最好使用相应的字符集和排序规则创建带有 JSON 字段的表。

因为`utf8mb4_bin`是二进制排序规则，所以键是区分大小写的，我们需要用正确的大小写来指定它们:

现在我们可以使用`->`操作符从 JSON 字段中提取数据:

正如我们所见，`->`只是`JSON_EXTRACT`的一个快捷方式或别名。

有趣的是，`test_name`和`test_id`的引号仍然存在。这不是我们想要的。我们希望引号被删除，类似于`name`字段。

要删除提取值的引号，我们需要使用`JSON_UNQUOTE`函数。由于`JSON_UNQUOTE(JSON_EXTRACT(…))`如此常用，该组合也有一个快捷操作符，即`->>`。让我们在实践中看看:

证明`->>`和`JSON_UNQUOTE(JSON_EXTRACT(...))`结果相同。由于`->>`很少打字，所以在大多数情况下它是首选。

但是，如果想从嵌套的 JSON 数组或 JSON 对象中提取数据，就不能使用 chained `->`或`->>`。你只能使用`->`和`->>`作为顶层，需要使用`JSON_EXTRACT`作为嵌套层。让我们提取每个学生的分数:

干杯！它像预期的那样工作。

从 MySQL 的 JSON 字段中提取数据的关键要点:

*   使用`$.key`从 JSON 对象中提取键值。
*   使用`$[index]`从 JSON 数组中提取元素的值。
*   如果值不是字符串，使用`->`作为`JSON_EXTRACT`的快捷方式。
*   如果值是一个字符串，并且您想要删除提取的字符串的引号，使用`->>`作为`JSON_UNQUOTE(JSON_EXTRACT(...))`的快捷方式。
*   如果要从嵌套的 JSON 数组或 JSON 对象中提取数据，就不能使用链式`->`或`->>`。您只能对顶层使用`->`和`->>`，需要对嵌套层使用`JSON_EXTRACT`。

在 MySQL 中有很多其他的[函数](https://dev.mysql.com/doc/refman/8.0/en/json-functions.html)用于处理 JSON 数据。但是，如果您需要使用这些函数来验证/搜索您的 JSON 字段或对其执行 CRUD 操作，您应该认真考虑使用 [MongoDB](https://www.mongodb.com/) 来存储 JSON 字段。MongoDB 在处理非结构化数据([文档](https://docs.mongodb.com/manual/core/document/))方面要专业和方便得多。

上面我们介绍了如何从 MySQL 的 JSON 字段中提取值。现在我们将学习相反的内容，并探索如何从 MySQL 表中选择 JSON 数据。为了继续这一部分，我们需要一些虚拟数据来玩。请复制以下 SQL 查询，并在 MySQL 控制台或 DBeaver 中运行它们:

对于该表，使用默认字符[和校对](https://dev.mysql.com/doc/refman/8.0/en/charset-applications.html)。通过这两个查询，我们创建了一个存储从第一部分提取的数据的表。这是数据管道和分析的常见任务，即在数据清理之后执行一些数据分析。实际上，您可能希望将分数存储在一个单独的表中，这样这些表就更加[规范化](https://en.wikipedia.org/wiki/Database_normalization)。然而，为了简化演示，这里将数据放在同一个表中。

我们现在可以用`JSON_ARRARYAGG`函数将数据聚集到一个 JSON 数组中:

我们还可以使用`JSON_OBJECTAGG`函数将数据聚合到一个 JSON 对象中:

然后，可以在您的应用程序中直接使用聚合的数据。`JSON_ARRARYAGG`和`JSON_OBJECTAGG`可以节省您在应用程序中聚集数据的努力，有时非常方便。例如，您可以使用`json.loads()`方法将 JSON 字符串转换成 Python 中的数组或字典。

如果需要在 Python 中执行`JSON_ARRARYAGG`和`JSON_OBJECTAGG`的普通 SQL 查询，可以使用 SQLAlchemy 包，如本文[中的](https://medium.com/codex/how-to-execute-plain-sql-queries-with-sqlalchemy-627a3741fdb1)所示。

在本文中，我们介绍了如何在 MySQL 中处理 JSON 数据。在第一部分中，通过简单的例子讨论了用于从 JSON 字段中提取数据的函数和操作符。在第二部分中，我们反其道而行之，将规范化数据聚合到 JSON 数组或对象中，然后可以直接在您的程序中使用。通常我们应该避免在 MySQL 中存储非结构化数据(文档)。但是，如果无法避免，本文中的知识应该对你的工作有所帮助。

相关文章:

*   [如何用 Python 中的 SQLAlchemy 执行普通 SQL 查询](https://medium.com/codex/how-to-execute-plain-sql-queries-with-sqlalchemy-627a3741fdb1?source=your_stories_page----------------------------------------)