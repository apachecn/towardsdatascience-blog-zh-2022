# 如何在 Python 中使用 SQLAlchemy 高效地执行批量插入

> 原文：<https://towardsdatascience.com/how-to-perform-bulk-inserts-with-sqlalchemy-efficiently-in-python-23044656b97d>

## 学习用 Python 高效地将大量记录插入数据库的不同方法

![](img/624c6c4ea3f983124f70b9418d92358a.png)

图片由 Pixabay 中的 [PublicDomainPictures](https://pixabay.com/photos/freighter-cargo-ship-industry-port-315201/) 提供

使用 SQLAlchemy 通过[普通 SQL 查询](https://betterprogramming.pub/how-to-execute-plain-sql-queries-with-sqlalchemy-627a3741fdb1)或[对象关系映射器(ORM)](https://levelup.gitconnected.com/learn-the-basics-and-get-started-with-sqlalchemy-orm-from-scratch-66c8624b069) 与关系数据库交互非常方便。然而，当涉及到批量插入时，也就是说，将大量记录插入到一个表中，我们可能经常会遇到性能问题。在本帖中，我们将介绍批量插入的不同方法，并通过实践教程比较它们的性能。您将对这些方法有更好的理解，并可以选择最适合您实际情况的方法。

## 安装 SQLAlchemy

为了使它成为可以跟随的实践教程，我们需要在我们的计算机上安装所需的库。建议在虚拟环境[中安装软件包](https://lynn-kwong.medium.com/how-to-create-virtual-environments-with-venv-and-conda-in-python-31814c0a8ec2)，这样就不会弄乱你的系统库。我们将使用 [*conda*](https://docs.conda.io/en/latest/) 创建一个虚拟环境，因为我们可以在虚拟环境中安装特定版本的 Python:

为本教程安装的软件包:

*   [SQLAlchemy](https://pypi.org/project/SQLAlchemy/) —将用于与数据库交互的主包。
*   [MySQL client](https://pypi.org/project/mysqlclient/)—MySQL 数据库的高性能驱动程序。如果您在安装 *mysqlclient* 或使用它时遇到一些问题，您可以安装 [PyMySQL](https://pypi.org/project/PyMySQL/) ，它没有系统依赖性问题。如果需要使用 PyMySQL 作为驱动程序，请查看这篇文章。
*   [*密码术*](https://pypi.org/project/cryptography/) —由 SQLAlchemy 用于认证。
*   [*ipython*](https://pypi.org/project/ipython/) —用于更方便地执行交互式 python 代码。

## 设置本地 MySQL 服务器

在本教程中，我们将使用本地 MySQL 服务器，而不是 SQLite，以使其更类似于实际用例。可以使用 Docker 方便地设置服务器:

请注意，MySQL Docker 容器附带了一个卷，因此即使容器重新启动，数据也可以持久保存。此外，root 密码被指定为一个环境变量，因此可以在以后用于身份验证。最后，为容器分配了一个高端口(13306 ),因此它不会与其他现有的 MySQL 服务器发生潜在冲突。

## 设置数据库连接

现在让我们设置数据库连接元数据，它将在即将引入的测试中使用。两个[上下文管理器](/understand-context-managers-in-python-and-learn-to-use-them-in-unit-tests-66cff907ce8e)被创建，分别产生一个`[Session](https://docs.sqlalchemy.org/en/14/orm/session_api.html#sqlalchemy.orm.Session)`和一个`[Connection](https://docs.sqlalchemy.org/en/14/core/connections.html#sqlalchemy.engine.Connection)`对象。`Session`对象将用于执行 orm 模型的操作，而`Connection`对象用于处理 SQLAlchemy 核心 API 或直接执行普通 SQL 查询。一些清理工作也在上下文管理器中完成，因此我们可以连续运行多个测试。有关 SQLAlchemy 引擎、连接和会话的更详细介绍，请查看[这篇文章](https://levelup.gitconnected.com/learn-the-basics-and-get-started-with-sqlalchemy-orm-from-scratch-66c8624b069)。

设置数据库连接元数据的代码片段如下:

警报:

*   应该使用`127.0.0.1`而不是`localhost`作为上面 DB URL 中的主机名，否则，可能会出现连接问题。

## 为测试创建一个 ORM 类

我们将创建一个简单的`customers`表，它有两个字段，即`id`和`name`，其中`id`的主键默认自动递增。顺便说一下，这个表将位于上面 DB URL 中指定的`data`模式中。该表的 ORM 类如下所示。该表将在上下文管理器创建会话或连接时创建，并将在`cleanup`参数为`True`时删除。

## 逐个添加 ORM 对象

现在让我们用不同的方法将大量记录添加到表中，并比较它们的性能。第一个是`Session.add()`，当您使用 ORM 与数据库交互时，这是非常常用的。

我们将首先向数据库添加 20，000 条记录，但不指定主键:

该测试功能大约需要 5 秒钟。所花费的时间会因电脑的性能而异，并且每次运行时会略有不同。如果太快或太慢，可以微调`num`参数。如果你想检查数据库中插入的数据，设置`cleanup`为`False`。

对于 ORM，有一个快捷方法`Session.add_all()`，它将 ORM 实例列表作为参数:

使用`Session.add()`和`Session.add_all()`的性能应该非常相似，因为数据直到运行`Session.commit()`才保存到数据库，而运行`Session.commit()`是真正的时间限制步骤。

实际上，用 5 秒钟插入 20，000 条记录可能是应用程序的一个主要性能问题。如果数据库位于远程服务器上，情况可能会更严重。低性能有两个主要原因:

*   需要为每条记录创建一个 ORM 实例。
*   由于 ORM 的[工作单元](https://martinfowler.com/eaaCatalog/unitOfWork.html)设计，主键和其他默认值需要返回到 ORM 实例。

第二个更有影响力，如果我们为创建的 ORM 实例提供主键，就可以证明这一点:

警报:

*   如果主键是自动递增的，并且是像这里这样显式指定的，则它不能为零，否则数据可能无法成功插入。你可以试着把`id=idx+1`改成`id=idx`，看看自己是否也会这样。

事实证明，如果提供主键，性能可以得到显著提高。这太棒了！然而，这并不是使用 SQLAlchemy 执行批量插入的最有效的方式，有时它可能不适用于指定主键，所以请耐心等待。

## 使用 Session.bulk _ save _ 对象

SQLAlchemy 有一些专门为批量操作设计的方法。对于批量插入，有`Session.bulk_save_objects()`和`Session.bulk_insert_mappings()`。`Session.bulk_save_objects()`将 ORM 实例列表作为参数，类似于`Session.add_all()`，而`Session.bulk_insert_mappings()`将映射/字典列表作为参数。我们这里用`Session.bulk_save_objects()`，后面用`Session.bulk_insert_mappings()`。

在开始使用它之前，我们应该知道`Session.bulk_save_objects()`的两个主要注意事项:

*   大多数 ORM 的好处，比如外键关系和属性的自动更新，对于通过这种方法传递的 ORM 实例是不可用的。如果我们想有这些好处，那么就不应该用这种方法，而应该用`Session.add_all()`来代替。
*   我们不应该返回插入的 ORM 实例的主键，否则，性能会大大降低。如果我们需要返回主键，我们也应该使用`Session.add_all()`来代替。

在下面的代码片段中，我们将执行三个测试并比较它们的性能:

*   使用返回主键的`Session.bulk_save_objects()`。
*   使用`Session.bulk_save_objects()`而不返回主键。
*   使用`Session.bulk_save_objects()`并明确指定主键。

当运行这三个测试时，它表明当返回主键时，性能确实急剧下降。然而，与`Session.add_all()`的情况不同，是否为 ORM 实例指定主键并没有多大关系。

## 使用 bulk_insert_objects

另一个用于批量插入的 SQLAlchemy 方法是`Session.bulk_insert_mappings()`。顾名思义，映射列表(Python 中的字典)作为该方法的参数传递。直接使用映射的好处是避免创建 ORM 实例的开销，这通常不是问题，但是当需要创建和保存大量 ORM 实例时，这就变得很重要了。

在下面的代码片段中，我们将执行两个测试并比较它们的性能:

*   使用没有指定主键的`Session.bulk_insert_mappings()`。
*   使用指定主键的`Session.bulk_insert_mappings()`。

以上测试速度极快。如果没有指定主键，它比`Session.bulk_save_objects()`快两倍，而【】又比`Session.add_all()`快五倍。此外，与`Session.bulk_save_objects()`类似，如果为要保存的映射指定了主键，也没有多大关系。

## 使用 SQLAlchemy 核心 API

SQLAlchemy ORM 模型建立在核心 API 之上。如果性能是唯一的目标，我们应该使用直接插入的核心 API，避免 ORM 模型的所有开销。

我们可以使用 [SQL 表达式语言](https://docs.sqlalchemy.org/en/14/core/tutorial.html)来访问 SQLAlchemy 的核心 API。使用 SQL 表达式语言的好处是能够直接访问核心 API，从而实现高性能，同时提供适用于所有类型的关系数据库的后端/数据库中立语言。我们将在下一节介绍普通 MySQL 查询的直接用法。

我们可以使用 ORM 类的`__table__`属性来访问提供`[Insert](https://docs.sqlalchemy.org/en/14/core/dml.html#sqlalchemy.sql.expression.Insert)`结构的底层`[Table](https://docs.sqlalchemy.org/en/14/core/metadata.html#sqlalchemy.schema.Table)`对象。类似于`Session.bulk_insert_mappings()`，映射/字典列表可以传递给`Insert`构造。然而，一个 SQLAlchemy `[Connection](https://docs.sqlalchemy.org/en/14/core/connections.html#sqlalchemy.engine.Connection)`对象用于执行插入表达式，而不是一个`[Session](https://docs.sqlalchemy.org/en/14/orm/session_api.html#sqlalchemy.orm.Session)`对象。

在下面的代码片段中，我们将执行两个测试并比较它们的性能:

*   使用核心 API 插入没有主键的字典列表。
*   使用核心 API 插入指定了主键的字典列表。

上面的测试甚至比使用`Session.bulk_insert_mappings()`、*还要快，但比*快不了多少，因为这里完全避免了使用 ORM 模型的开销。此外，如果为要保存的映射指定了主键，也没有多大关系。

## 使用普通 SQL 查询

如果你是一个只想处理普通 SQL 查询而根本不想处理核心 API 或 ORM 的守旧派，你可以使用`Connection.exec_driver_sql()`来执行批量插入，它直接利用底层 DBAPI，与使用上面所示的核心 API 具有相同的性能:

如果你想了解更多关于在 SQLAlchemy 中执行普通 SQL 查询的信息，请查看这篇文章。

所有例子的代码都可以在[这里](https://gist.github.com/lynnkwong/be9532672302eed25675e2adbfa5a1c2)找到。一旦安装了库并设置了 MySQL 服务器，就可以直接运行它。

在本文中，介绍了批量插入的不同 SQLAlchemy 方法。以简单易懂的方式介绍了它们的代码，并系统地比较了它们的性能。

总之，如果您使用普通的 SQL 查询，您不需要担心 SQLAlchemy 的性能，因为它直接调用底层的 DBAPI。应该优化的是查询本身。

如果你坐在中间，不使用普通的 SQL 查询或 ORM 模型，而是使用所谓的表达式语言，你可以使用直接访问核心 API 的`Insert`构造来执行批量插入，这也非常有效。

最后，如果您使用 ORM 模型，并且希望在插入后访问 ORM 实例的更新状态，那么您应该使用`Session.add_all()`。如果可能，请提供主键，因为这样可以显著提高性能。另一方面，如果您使用 ORM 模型，并且不需要访问更新的数据，您可以使用与核心 API 效率相当的`Session.bulk_insert_mappings()`。

相关文章:

*   [如何用 Python 中的 SQLAlchemy 执行普通 SQL 查询](https://betterprogramming.pub/how-to-execute-plain-sql-queries-with-sqlalchemy-627a3741fdb1)
*   [学习基础知识并开始使用 SQLAlchemy ORM](https://levelup.gitconnected.com/learn-the-basics-and-get-started-with-sqlalchemy-orm-from-scratch-66c8624b069)