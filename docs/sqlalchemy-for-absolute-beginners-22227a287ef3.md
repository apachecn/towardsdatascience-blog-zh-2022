# 绝对初学者的 SQLAlchemy

> 原文：<https://towardsdatascience.com/sqlalchemy-for-absolute-beginners-22227a287ef3>

## 创建数据库引擎并从 Python 执行 SQL

![](img/bae906252f184afb1332c5e6e490acc7.png)

一些适当的炼金术(图片由[埃琳娜·莫日维洛](https://unsplash.com/@miracleday)在 [Unsplash](https://unsplash.com/photos/JWtNpyXy-fQ) 上拍摄)

SqlAlchemy 是一种允许 Python 处理来自数据库的数据的简单快捷的方法。用引擎**创建到数据库**的连接非常简单。此外，用引擎和 ORM 查询数据库和检索数据也非常容易。

在本文中，我们将做两件事:

1.  用引擎创建到我们数据库的连接)
2.  使用引擎执行原始 SQL

最后，我们将关注接下来的步骤:我们可以使用数据库引擎的所有其他事情；比如一个迁移模型。我们将开始创建连接；我们来编码吧！

# 1.创建数据库引擎

数据库引擎允许我们与数据库通信。创造一个并不难。

## 步骤 1-创建连接字符串

首先，我们将创建一个连接字符串。这指定了关于我们的数据库的详细信息:它的托管位置和凭证。

```
import sqlalchemy as sa# Step 1: Create connection string
constring: sa.engine.url.URL = sa.engine.URL.create(
    drivername="postgresql",
    username="postgres",
    password="mysecretpassword",
    host="localhost",
    port=5432,
    database="mydb"
)
```

如您所见，我们正在连接到本地主机上的 Postgres 数据库。

> 通常，硬编码您的凭证是一个非常糟糕的想法。如果我在 Github 上把这个文件提交给我的 repo，每个人都可以看到我的凭证。强烈建议将您的凭证保存在一个`*.env*`文件中，这是一个**简单的技术，可以防止您的机密信息落入坏人之手**。查看下面的文章，了解如何做到这一点。

[](/keep-your-code-secure-by-using-environment-variables-and-env-files-4688a70ea286) [## 通过使用环境变量和 env 文件来保证代码的安全

### 安全地加载一个文件，其中包含我们的应用程序所需的所有机密数据，如密码、令牌等

towardsdatascience.com](/keep-your-code-secure-by-using-environment-variables-and-env-files-4688a70ea286) 

## 步骤 2:创建引擎

然后我们使用`constring`来创建一个引擎。

```
# Step 2: Create and configure engine with the connection string
dbEngine= sa.create_engine(
    url=constring,
    # fast_executemany=True,
)
```

我通过`fast_executemany`作为补充论证。这是 SQL Server 针对更快插入的特定优化。阅读以下文章了解更多信息:

[](/dramatically-improve-your-database-inserts-with-a-simple-upgrade-6dfa672f1424) [## 通过简单的升级，显著提高数据库插入速度

### 用 Python 创建速度惊人的数据库连接的 4 个层次

towardsdatascience.com](/dramatically-improve-your-database-inserts-with-a-simple-upgrade-6dfa672f1424) 

## 步骤 3:连接和测试发动机

在上一步中，我们已经定义了引擎。在这一步中，我们将尝试连接并检查连接是否有效。

```
# Step 3: Testing my engine
try:
    with dbEngine_sqlserver_localhost.connect() as con:
        con.execute("SELECT 1")
    print("Engine valid")
except Exception as e:
    print(f"Engine invalid: {e}")
```

我们`try`执行一些简单的 SQL。如果这没有引起错误，我们知道我们连接正确。否则，我们可以检查错误，看看哪里出错了。

例如，您可能会得到一个错误，因为 SQLAlchemy 缺少必需的包。例如，为了连接到 Postgres 数据库，我们需要`pip install psycopg2`。SQL Server 需要`pyodbc`。SQLAlchemy 非常清楚这一点，所以只需遵循说明。

[](/python-to-sql-upsert-safely-easily-and-fast-17a854d4ec5a) [## Python 到 SQL —安全、轻松、快速地向上插入

### 使用 Python 进行闪电般的插入和/或更新

towardsdatascience.com](/python-to-sql-upsert-safely-easily-and-fast-17a854d4ec5a) 

# 2.使用引擎执行 SQL

现在我们已经创建了数据库引擎，我们可以开始使用它了。在前一章中，我们已经先睹为快地执行了`SELECT 1`。在本章中，我们将讨论在引擎上执行原始 SQL 的优点和缺点。

## 原始 SQL 示例:创建表

在下面的示例中，我们将看到执行原始 SQL 的缺点:

```
statement_create = """
    CREATE TABLE IF NOT EXISTS Students (
        ID SERIAL PRIMARY KEY,
        Name TEXT,
        Age INT
    )
"""
with dbEngine_sqlserver_localhost.connect() as con:
    con.execute(statement_create)
```

我们将使用一些 PostgreSQL 来创建一个语句，然后使用连接来执行它。

## 原始 SQL 的问题是

尽管这很容易；执行原始 SQL 有一些缺点:

1.  容易出错:在 SQL 语法中很容易出错
2.  **特定于数据库的**:也许您只想在 Postgres 上测试和开发，但是您的生产数据库是另一种类型。例如，如果你使用 SQL Server，你不能使用`ID SERIAL PRIMARY KEY`，而应该使用类似`ID INT IDENTITY(1,1) PRIMARY KEY`的东西。
3.  **组织**:你的存储库将会充满 SQL 语句。想象一下，如果一个列名改变了；您需要浏览所有这些 SQL 语句并调整您的查询。

[](/sql-understand-how-indices-work-under-the-hood-to-speed-up-your-queries-a7f07eef4080) [## SQL——理解索引如何在幕后加速查询。

### 不用再等待缓慢的查询完成

towardsdatascience.com](/sql-understand-how-indices-work-under-the-hood-to-speed-up-your-queries-a7f07eef4080) 

## 解决方案:与数据库无关的模型

SqlAlchemy 通过创建映射到数据库中的表的**对象来解决这个问题(这是 ORM 中的对象关系映射)。这有许多优点:**

1.  模型是**数据库不可知的**。这意味着对象不知道需要哪个数据库和语法。例如，在插入 SQLAlchemy 时，它会将对象编译成一条语句，这条语句适合我们正在使用的数据库(-engine)。这意味着**您的 Python 代码*会运行*，即使您的开发环境使用与您的生产环境相比的另一个数据库**。这使得使用数据库更加灵活，在使用[迁移模型](https://mikehuls.medium.com/python-database-migrations-for-beginners-getting-started-with-alembic-84e4a73a2cca)时尤其有用。
2.  不易出错: **SQL 是为您生成的**,因此几乎不可能出现语法错误。此外，你只在你的 Python-IDE 中使用 Python 类，所以很难打错字。
3.  你所有的模型都整齐地**组织在一个地方**，并导入到你的项目中。如果有人更改了数据库表的列名，您只需要调整一个模型，而不是修复许多原始 SQL 查询。

我目前正在写一篇文章，演示如何将 SQLAlchemy 的 ORM 用于数据库引擎。 [**跟我来**](https://mikehuls.medium.com/membership) **敬请关注**！

[](/find-the-top-n-most-expensive-queries-48e46d8e9752) [## 查找数据库中前 n 个最慢的查询

### 找到降低数据库处理速度的瓶颈查询

towardsdatascience.com](/find-the-top-n-most-expensive-queries-48e46d8e9752) 

# 结论

在本文中，我们探索了 SQL 炼金术的起源；我们知道如何连接到数据库，以及如何执行原始 SQL。我们还讨论了执行原始 SQL 的利弊，并理解对于大多数项目来说，使用 ORM 是更好的选择。在下一篇文章中，我们将了解 ORM 以及如何使用它。

我希望一切都像我希望的那样清楚，但如果不是这样，请让我知道我能做些什么来进一步澄清。同时，看看我的其他关于各种编程相关主题的文章:

*   [Python 为什么这么慢，如何加速](https://mikehuls.medium.com/why-is-python-so-slow-and-how-to-speed-it-up-485b5a84154e)
*   [**Git** 绝对初学者:借助视频游戏理解 Git](https://mikehuls.medium.com/git-for-absolute-beginners-understanding-git-with-the-help-of-a-video-game-88826054459a)
*   [创建并发布自己的 **Python 包**](https://mikehuls.medium.com/create-and-publish-your-own-python-package-ea45bee41cdc)
*   [**虚拟环境**面向绝对初学者——什么是虚拟环境以及如何创建虚拟环境(+示例](https://mikehuls.medium.com/virtual-environments-for-absolute-beginners-what-is-it-and-how-to-create-one-examples-a48da8982d4b))
*   [用 FastAPI](https://mikehuls.medium.com/create-a-fast-auto-documented-maintainable-and-easy-to-use-python-api-in-5-lines-of-code-with-4e574c00f70e) 用 5 行代码创建一个快速自动归档、可维护且易于使用的 Python **API**

*   [**Docker** 适合绝对初学者——Docker 是什么，怎么用(+举例)](https://mikehuls.medium.com/docker-for-absolute-beginners-what-is-docker-and-how-to-use-it-examples-3d3b11efd830)

编码快乐！

—迈克

附注:喜欢我正在做的事吗？ [*跟我来！*](https://mikehuls.medium.com/membership)

[](https://mikehuls.medium.com/membership) [## 通过我的推荐链接加入媒体-迈克·赫斯

### 阅读迈克·赫斯(以及媒体上成千上万的其他作家)的每一个故事。你的会员费直接支持麦克…

mikehuls.medium.com](https://mikehuls.medium.com/membership)