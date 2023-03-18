# 为绝对初学者创建 Python PostgreSQL 连接

> 原文：<https://towardsdatascience.com/creating-a-python-postgresql-connection-for-absolute-beginners-501b97f73de1>

## Python 脚本如何与 Postgres 数据库通信

![](img/bd93966de3d685b595579b27934f031c.png)

Postgres 来了(图片由[四月宠物杂交](https://unsplash.com/@apriiil)在 [Unsplash](https://unsplash.com/photos/jmAGGE6MDeA) 拍摄)

让 Python 连接到您的数据库并交换信息是一项非常基本的技能，但是创建实际的连接一开始可能会有点奇怪。在这篇文章中，我们将了解如何以一种高效安全的方式创建和使用连接。我们将使用一个名为`psycopg2`的包，并通过一些实例演示如何与数据库交互。我们来编码吧！

# 在开始之前..

想要连接到非 PostgreSQL 的数据库？查看下面的文章。

[](https://mikehuls.medium.com/how-to-make-a-database-connection-in-python-for-absolute-beginners-e2bfd8e4e52) [## 绝对初学者如何用 Python 做数据库连接

### 连接 MS SQL Server、MySQL、Oracle 和其他数据库的 3 个步骤(+示例)

mikehuls.medium.com](https://mikehuls.medium.com/how-to-make-a-database-connection-in-python-for-absolute-beginners-e2bfd8e4e52) 

## 生成 SQL 查询

本文重点介绍如何使用 pyodbc 创建与数据库的连接。然后可以使用这个连接来执行 SQL。有些数据库使用不同的语法:

*   **MS SQL Server**
    `SELECT TOP 1 colname FROM tablename LIMIT 10`
*   **PostgreSQL**T2
    。

这意味着您必须为特定的数据库编写 SQL 语句。

有一种**与数据库无关的**方法，可以用更 Pythonic 化的方式定义查询，然后针对某个数据库进行编译。这将**为那个数据库**生成特定的 SQL，这样您就不会受限于您当前使用的数据库的特定内容。这意味着，如果你将来决定从 MySQL 转换到 Postgres，你不必修改你的代码。

我正在写这篇文章，所以请务必[关注我](https://mikehuls.medium.com/membership)！

## 关于心理战 2

Psycopg2 是一个为 Python 编写的 PostgreSQL 数据库适配器。它是为高度多线程的应用程序而设计的，是用 C 语言编写的，侧重于有效和安全的通信。

如果你像我一样，认为*“心理战 2..*这个名字很怪；这是因为作者犯了一个小错误([来源](https://www.postgresql.org/message-id/flat/CA%2BRjkXFL6Jy7actUy%2BS%3DdGfjpuD_jpUYYofGpON%2B1Nq9S2Y75w%40mail.gmail.com#f9771ec89bf350e0594cf9640a1f912c)):

> 我想称之为 psychopg(指他们的精神病司机)，但我打错了名字。

尽管这个有趣的背景故事，psycopg2 是一个非常优化，安全和有效的软件包。在接下来的几个部分中，我们将使用它来建立到 Postgres 数据库的连接并执行一些 SQL。

![](img/92c95589dd998d1aac9a8edf9c484a15.png)

让我们进入我们的数据库(图片由[克林特·帕特森](https://unsplash.com/@cbpsc1)在 [Unsplash](https://unsplash.com/photos/exfrR9KkzlE) 上提供)

# 创建数据库连接

下面我们将介绍如何建立和使用连接。首先，我们将通过安装依赖项来做准备:

## 准备:依赖性

首先，让我们创建一个虚拟环境，并安装我们唯一的依赖项。看看下面的文章，了解为什么使用 venv 非常重要，以及如何创建它们。

```
pip install pscopg2
```

[](/virtual-environments-for-absolute-beginners-what-is-it-and-how-to-create-one-examples-a48da8982d4b) [## 绝对初学者的虚拟环境——什么是虚拟环境，如何创建虚拟环境(+例子)

### 深入探究 Python 虚拟环境、pip 和避免纠缠依赖

towardsdatascience.com](/virtual-environments-for-absolute-beginners-what-is-it-and-how-to-create-one-examples-a48da8982d4b) 

## 第一步。创建连接

第一个目标是创建一个连接(通常每个应用程序有一个连接)。

首先，我们将通过从环境中加载凭据来获取凭据，这样我们的应用程序就可以访问凭据，而无需将它们硬编码到脚本中。

通过学习如何应用下面文章中的`env files`来防止您的密码泄露:

[](/keep-your-code-secure-by-using-environment-variables-and-env-files-4688a70ea286) [## 通过使用环境变量和 env 文件来保证代码的安全

### 安全地加载一个文件，其中包含我们的应用程序所需的所有机密数据，如密码、令牌等

towardsdatascience.com](/keep-your-code-secure-by-using-environment-variables-and-env-files-4688a70ea286) 

## 第二步。创建游标并执行

现在我们有了连接，我们可以创建一个光标。这是一个用于迭代结果集(由查询产生)的对象。

上面的代码在上下文管理器(`with`)语句中创建了一个游标，它有两个主要优点:

1.  上下文管理器中的所有查询都作为一个事务来处理。任何失败的`cursor.execute`都将导致先前的回滚
2.  当我们退出程序块时，光标将自动关闭

[](https://medium.com/geekculture/understanding-python-context-managers-for-absolute-beginners-4873b6249f16) [## 绝对初学者理解 Python 上下文管理器

### 理解关于光剑的 WITH 语句

medium.com](https://medium.com/geekculture/understanding-python-context-managers-for-absolute-beginners-4873b6249f16) 

## 第三步。示例查询

并不是说我们的数据库连接已经建立，我们可以开始查询我们的数据库。在下面的文章中，我们配置了一个到 MS SQL Server(以及其他数据库)的连接。本文还包含一些非常有用的示例查询，您也可以将其应用于我们的 psycopg2 连接。请务必查看它们，以便开始跑步。

[](https://mikehuls.medium.com/how-to-make-a-database-connection-in-python-for-absolute-beginners-e2bfd8e4e52) [## 绝对初学者如何用 Python 做数据库连接

### 连接 MS SQL Server、MySQL、Oracle 和其他数据库的 3 个步骤(+示例)

mikehuls.medium.com](https://mikehuls.medium.com/how-to-make-a-database-connection-in-python-for-absolute-beginners-e2bfd8e4e52) 

# 重要的

为了防止 SQL 注入攻击，一定要清理放入 SQL 语句中的变量，尤其是在允许用户输入语句的情况下。这一点被 [**这部 XKCD 著名漫画**](https://www.explainxkcd.com/wiki/images/5/5f/exploits_of_a_mom.png) **诠释得很漂亮。**

# 后续步骤

现在您的脚本可以连接到数据库，您可以开始编写 SQL，如下文所示。查看 [**此链接**](https://mikehuls.com/articles?tags=sql) 了解许多便捷查询的概述。另请查看本文 中的 [**部分，该部分详细介绍了如何实现**数据库迁移**；这样你就可以使用 Python 来控制数据库结构的版本！非常有用！**](https://mikehuls.medium.com/python-database-migrations-for-beginners-getting-started-with-alembic-84e4a73a2cca)

[](/postgresql-how-to-upsert-safely-easily-and-fast-246040514933) [## PostgreSQL —如何安全、轻松、快速地进行升级

### 防止重复，插入新记录，更新现有记录

towardsdatascience.com](/postgresql-how-to-upsert-safely-easily-and-fast-246040514933) 

# 结论

我希望已经阐明了如何用 psycopg2 连接到 Postgres 数据库，并使用该连接来处理数据库。我希望一切都像我希望的那样清楚，但如果不是这样，请让我知道我能做些什么来进一步澄清。与此同时，请查看我的关于各种编程相关主题的其他文章:

*   [在一条语句中插入、删除和更新—使用 MERGE 同步您的表](https://mikehuls.medium.com/sql-insert-delete-and-update-in-one-statement-sync-your-tables-with-merge-14814215d32c)
*   [Python 为什么这么慢，如何加速](https://mikehuls.medium.com/why-is-python-so-slow-and-how-to-speed-it-up-485b5a84154e)
*   [Git 绝对初学者:借助视频游戏理解 Git](https://mikehuls.medium.com/git-for-absolute-beginners-understanding-git-with-the-help-of-a-video-game-88826054459a)
*   [Docker 对于绝对初学者——Docker 是什么以及如何使用它(+例子)](https://mikehuls.medium.com/docker-for-absolute-beginners-what-is-docker-and-how-to-use-it-examples-3d3b11efd830)
*   面向绝对初学者的虚拟环境——什么是虚拟环境，如何创建虚拟环境(+示例
*   [创建并发布自己的 Python 包](https://mikehuls.medium.com/create-and-publish-your-own-python-package-ea45bee41cdc)
*   [用 FastAPI 用 5 行代码创建一个快速自动记录、可维护且易于使用的 Python API](https://mikehuls.medium.com/create-a-fast-auto-documented-maintainable-and-easy-to-use-python-api-in-5-lines-of-code-with-4e574c00f70e)

编码快乐！

—迈克

*又及:喜欢我正在做的事吗？* [*跟我来！*](https://mikehuls.medium.com/membership)

[](https://mikehuls.medium.com/membership) [## 通过我的推荐链接加入 Medium—Mike Huls

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

mikehuls.medium.com](https://mikehuls.medium.com/membership)