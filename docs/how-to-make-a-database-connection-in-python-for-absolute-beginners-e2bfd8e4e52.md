# 绝对初学者如何用 Python 做数据库连接

> 原文：<https://towardsdatascience.com/how-to-make-a-database-connection-in-python-for-absolute-beginners-e2bfd8e4e52>

## 连接 MS SQL Server、MySQL、Oracle 和许多其他数据库的 3 个步骤(+示例)

![](img/f991da2c0f0de51c40ecc65a66736818.png)

连接时间(图片由[马库斯·乌尔本斯](https://unsplash.com/@marcusurbenz)在 [Unsplash](https://unsplash.com/photos/qY6ikaS8L38) 上拍摄)

使用 Python，我们可以自动编写和执行 SQL。但是要做到这一点，我们需要 Python 能够与数据库通信。在本文中，我们将重点关注使用名为`pyodbc`的包，通过 ODBC 协议与关系数据库进行通信。阅读完本文后，你将能够在你的 Python 应用程序中编写和执行 SQL。我们来编码吧！

# 在开始之前..

让我们定义一下这篇文章的范围。

## Postgres 用户？

有许多数据库符合 ODBC 和 pyodbc，在本文中我们将使用 MS SQL。如果您正在使用 PostgreSQL，那么请查看下面的文章，了解更优化的方法。

<https://mikehuls.medium.com/creating-a-python-postgresql-connection-for-absolute-beginners-501b97f73de1>  

## 生成 SQL 查询

本文重点介绍如何使用 pyodbc 创建与数据库的连接。然后可以使用这个连接来执行 SQL。有些数据库使用不同的语法:

*   **MS SQL Server**
    `SELECT TOP 1 colname FROM tablename LIMIT 10`
*   **PostgreSQL**T2
    。

这意味着您必须为特定的数据库编写 SQL 语句。

有一种**与数据库无关的**方法，可以用更 Pythonic 化的方式定义查询，然后针对某个数据库进行编译。这将**为那个数据库**生成特定的 SQL，这样您就不会受限于您当前使用的数据库的特定内容。这意味着，如果你将来决定从 MySQL 转换到 Postgres，你不必修改你的代码。

我正在写这篇文章，所以请务必[关注我](https://mikehuls.medium.com/membership)！

## 关于 pyodbc 和 odbc 协议

关于我们正在使用的软件包及其工作原理的一些背景知识。Pyodbc 是一个允许您与(关系)数据库通信的包。它使用**的*开放式数据库通信*的**协议。这个协议定义了客户端(就像您的 Python 脚本)和数据库如何通信。

您可以将这种通信协议与 HTTP 协议进行比较，HTTP 协议有助于计算机之间通过 internet 进行通信:客户端知道如何请求资源，服务器知道如何响应，客户端知道响应是什么样的，因此它们可以使用这些信息。同样，**客户端可以使用 *ODBC 协议*与数据库**通信。

</docker-for-absolute-beginners-the-difference-between-an-image-and-a-container-7e07d4c0c01d>  

# 连接到数据库—代码部分

我们将经历几个简单的步骤。我们需要使用我们的凭证来创建一个连接字符串。用那根线我们可以建立联系。在连接上，您可以创建一个我们将用来执行查询的游标。首先做一些准备:

## 准备:依赖性

首先，让我们创建一个虚拟环境，并安装我们唯一的依赖项。

```
pip install pyodbc
```

</virtual-environments-for-absolute-beginners-what-is-it-and-how-to-create-one-examples-a48da8982d4b>  

## 第一步。收集我们的凭证

这是我们检索数据库凭证的地方。在下面的例子中，我们尽可能以最安全的方式处理我们的凭证:我们从环境中加载它们，这样我们的应用程序就可以访问它们，而无需将它们硬编码到我们的脚本中。

```
import pyodbc

driver: str = 'ODBC Driver 17 for SQL Server'
username = os.environ.get("SQLSERVER_USER")
password = os.environ.get("SQLSERVER_PASS")
host = os.environ.get("SQLSERVER_HOST")
port = os.environ.get("SQLSERVER_PORT")
database = os.environ.get("SQLSERVER_DB")
```

通过学习如何应用下面文章中的`env files`来防止您的密码泄露:

</keep-your-code-secure-by-using-environment-variables-and-env-files-4688a70ea286>  

## 第二步。创建连接字符串

Pyodbc 需要一个包含我们的凭据的格式化字符串来连接到数据库。在下面的例子中，我们使用 f 字符串来创建连接字符串，同时保持代码的整洁。

```
password = "}}".join(password.split("}"))

constring = f"DRIVER={driver};" \
            f"SERVER={host};" \
            f"DATABASE={database};" \
            f"UID={username};" \
            f"PWD={{{password}}};" \
            f"port={port};"
```

**修复密码..**
`password`发生了一些诡异的事情。首先，我们将密码中的所有`}`花括号折叠起来，然后我们再次用花括号将它括起来。这是因为密码可能包含一些奇怪的字符，例如`/\{};`。用花括号把它们括起来可以避免所有的错误。我们不希望我们的密码中的任何花括号被转义，所以我们将它们加倍，以便“避免转义”。

有点奇怪，但这显然是它的工作方式。还要注意，这不仅适用于密码，也适用于任何参数，所以如果您的用户名包含特殊字符，您也可以使用这种技术。

产生的连接字符串如下所示(注意，`my_}password`现在被正确地翻译成了`{my_}}password}`)。

```
DRIVER=ODBC Driver 17 for SQL Server;SERVER=my_host;DATABASE=my_database;UID=my_username;PWD={my_}}password};port=my_port;
```

</getting-started-with-cython-how-to-perform-1-7-billion-calculations-per-second-in-python-b83374cfcf77>  

## 第三步。连接

这是最简单的部分，我们将创建一个连接(通常每个应用程序有一个连接)。然后我们在连接上创建一个光标。游标用于迭代查询产生的结果集。处理完结果集后，关闭光标。

```
cnxn:pyodbc.Connection = pyodbc.connect(constring)cursor:pyodbc.Cursor = cnxn.cursor()
try:
    cursor.execute("SELECT @@VERSION")
    print(cursor.fetchone())
except Exception as e:
    print(f"Connection could not be established: {e}")
finally:
    cursor.close()
```

接下来，我们可以使用光标执行一些 SQL，在本例中，我们打印出 SQL Server 数据库的版本。如果有任何失败，我们打印出错误，并且在任何情况下，我们关闭我们的光标。

有一种更短、更好的方法可以做到这一点:

```
with cnxn.cursor() as cursor:
    try:
        cursor.execute("SELECT @@VERSION")
        print(cursor.fetchone())
    except Exception as e:
        print(f"Connection could not be established: {e}")
```

使用上下文管理器(`with`部分)会使光标自动关闭。此外，它将提交(您必须`commit`插入；看下面的例子)你在`try`块中执行的任何东西。如果它检测到错误，它将回滚所有查询。请注意，这仅适用于使用`autocommit=False`(默认设置)创建连接的情况。

<https://medium.com/geekculture/understanding-python-context-managers-for-absolute-beginners-4873b6249f16>  

# 额外收获:示例查询

这里有一些示例查询可以帮助您。

## **查询 1:选择记录**

使用`cursor.fetchone()`检索单个行。

```
with cnxn.cursor() as cursor:
    cursor.execute("SELECT * FROM INFORMATION_SCHEMA.TABLES")
    for row in cursor.fetchall():
        print(row)
```

</sql-understand-how-indices-work-under-the-hood-to-speed-up-your-queries-a7f07eef4080>  

## **查询 2:选择记录；转换为字典列表，其中每个字典为一行**

```
with cnxn.cursor() as cursor:
    cursor.execute("SELECT * FROM INFORMATION_SCHEMA.TABLES")
    colnames = [col[0] for col in cursor.description]
    coltypes = [col[1] for col in cursor.description]
    for rowvalues in cursor.fetchall():
        # convert types
        rowvalues = [coltypes[i](rowvalues[i]) for i in range(len(coltypes))]
        # make dicts from the colnames and rowvalues
        row = dict(zip(colnames, rowvalues))
        print(row)
```

## 查询 3:常规插入

因为我们在上下文管理器块中(即`with`)，所以如果没有错误发生，执行将被提交。光标也会自动关闭。

```
with cnxn.cursor() as cursor:
    cursor.execute("INSERT INTO med.mytable (name, age)  VALUES (?,?)", "mike", 32)
```

</sql-insert-delete-and-update-in-one-statement-sync-your-tables-with-merge-14814215d32c> [## SQL —在一条语句中插入、删除和更新:用 MERGE 同步表

towardsdatascience.com](/sql-insert-delete-and-update-in-one-statement-sync-your-tables-with-merge-14814215d32c) 

## 查询 4:回滚插入

前两个被插入，然后#3 失败，所以#1 和# 2 再次回滚。数据库里不会有 a 先生，b 先生，c 先生的踪迹。

```
with cnxn.cursor() as cursor:
    cursor.execute("INSERT INTO med.mytable (name, age)  VALUES (?,?)", "mr. a", 44)
    cursor.execute("INSERT INTO med.mytable (name, age)  VALUES (?,?)", "mr. b", 33)
    cursor.execute("INSERT INTO med.mytable (name, age)  VALUES (?,?)", "mr. c", 55, "toomany")
```

</sql-rolling-back-statements-with-transactions-81937811e7a7>  

## 查询 5:超快速插入

上面的插入将一次插入一行。使用下面的选项，我们可以为多行创建一条语句，从而大大提高插入速度。

```
with cnxn.cursor() as cursor:
    cursor.fast_executemany = True
    SQL = "INSERT INTO med.mytable (name, age)  VALUES (?,?)"
    values = [
        ('mr. y', 21),
        ('mr. x', 98)
    ]
    cursor.executemany(SQL, values)
```

阅读下面的文章，了解`fast_executemany`的内部工作原理。

</dramatically-improve-your-database-inserts-with-a-simple-upgrade-6dfa672f1424>  

# 重要的

为了防止 SQL 注入攻击，一定要清理放入 SQL 语句中的变量，尤其是在允许用户输入语句的情况下。这一点被 [**这部 XKCD 著名漫画**](https://www.explainxkcd.com/wiki/images/5/5f/exploits_of_a_mom.png) **诠释得很漂亮。**

# 后续步骤

现在您的脚本可以连接到数据库了，您可以开始编写 SQL 了。比如你可以用 5 行代码 做一个 [**API。通过这种方式，您可以为用户提供对数据库的受控访问，定义用户可以请求哪些信息。**](https://mikehuls.medium.com/create-a-fast-auto-documented-maintainable-and-easy-to-use-python-api-in-5-lines-of-code-with-4e574c00f70e)

请务必查看 [**此链接**](https://mikehuls.com/articles?tags=sql) 以获得许多有用查询的详细概述。最后，下面的文章详细介绍了如何实现数据库迁移。这使得以编程方式设计和版本控制数据库成为可能。

</python-database-migrations-for-beginners-getting-started-with-alembic-84e4a73a2cca>  

# 结论

我希望已经阐明了如何用 pyodbc 连接到您的数据库，并使用它来执行一些 SQL。我希望一切都像我希望的那样清楚，但如果不是这样，请让我知道我能做些什么来进一步澄清。与此同时，请查看我的关于各种编程相关主题的其他文章:

*   [面向绝对初学者的 cy thon——通过简单的两步将代码速度提高 30 倍](https://mikehuls.medium.com/cython-for-absolute-beginners-30x-faster-code-in-two-simple-steps-bbb6c10d06ad)
*   [Python 为什么这么慢，如何加速](https://mikehuls.medium.com/why-is-python-so-slow-and-how-to-speed-it-up-485b5a84154e)
*   [Git 绝对初学者:借助视频游戏理解 Git](https://mikehuls.medium.com/git-for-absolute-beginners-understanding-git-with-the-help-of-a-video-game-88826054459a)
*   [Docker 对于绝对初学者——Docker 是什么以及如何使用它(+例子)](https://mikehuls.medium.com/docker-for-absolute-beginners-what-is-docker-and-how-to-use-it-examples-3d3b11efd830)
*   绝对初学者的虚拟环境——什么是虚拟环境，如何创建虚拟环境(+示例
*   [创建并发布你自己的 Python 包](https://mikehuls.medium.com/create-and-publish-your-own-python-package-ea45bee41cdc)
*   [用 FastAPI 用 5 行代码创建一个快速自动记录、可维护且易于使用的 Python API](https://mikehuls.medium.com/create-a-fast-auto-documented-maintainable-and-easy-to-use-python-api-in-5-lines-of-code-with-4e574c00f70e)

编码快乐！

—迈克

附注:喜欢我正在做的事吗？ [*跟我来！*](https://mikehuls.medium.com/membership)

<https://mikehuls.medium.com/membership> 