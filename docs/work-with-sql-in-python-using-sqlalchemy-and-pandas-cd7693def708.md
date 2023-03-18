# 如何使用 SQLAlchemy 和 Pandas 从 Python 连接到 SQL 数据库

> 原文：<https://towardsdatascience.com/work-with-sql-in-python-using-sqlalchemy-and-pandas-cd7693def708>

## 通过 SQLAlchemy 提取 SQL 表，在 SQL 数据库中插入、更新和删除行

![](img/7f594dbc39a464b175acbdbd98d3da4b.png)

帕斯卡尔·米勒在 [Unsplash](https://unsplash.com/?utm_source=medium&utm_medium=referral) 上的照片(作者修改)

在一个数据科学项目中，我们经常需要与**关系数据库**进行交互，例如，在 SQL 表中提取表、插入、更新和删除行。为了完成这些任务，Python 有一个这样的库，叫做 **SQLAlchemy** 。它支持流行的 SQL 数据库，如 **PostgreSQL** 、 **MySQL** 、 **SQLite** 、 **Oracle** 、**微软 SQL Server** 等。更好的是，它有内置功能，可以与**熊猫**集成。SQLAlchemy 和 Pandas 是处理数据管理的完美组合。

# 安装库

除了 SQLAlchemy 和 pandas，我们还需要安装一个 SQL 数据库适配器来实现 **Python 数据库 API** 。例如，我们需要为 PostgreSQL 安装“psycopg2”或“pg8000”，为 mysql 安装“mysql-connector-python”或“oursql”，为 Oracle SQL 数据库安装“cx-Oracle”，为 Microsoft SQL Server 安装“pyodbc”或“pymssql”等。在本文中，我将讨论如何将 PostgreSQL 与 Python 集成，因此，让我们安装“ **psycopg2** ”。

打开 anaconda 提示符或命令提示符，并键入以下命令。

```
pip install SQLAlchemy
pip install pandas 
pip install psycopg2
```

# 导入库

```
import sqlalchemy
import pandas as pd
```

# 创建到数据库的连接

首先，让我们基于一个 **URL** 使用“ **create_engine()** ”函数创建一个与 PostgreSQL 数据库的连接。URL 通常由方言、驱动程序、用户名、密码、主机名、数据库名以及用于附加配置的可选参数组成。数据库 URL 的典型形式看起来像“**方言+驱动://用户名:密码@主机:端口/数据库**”。比如微软 SQL Server 的“MSSQL+py odbc://username:password @ host:port/database”，mysql 的“MySQL+MySQL connector://username:password @ host:port/database”，postgresql 的“PostgreSQL+psycopg 2://username:password @ host:port/database”。

```
url = '**postgresql+psycopg2://username:password@host:port/database**'
engine = sqlalchemy.**create_engine**(url)
```

我们还可以在“create_engine()”函数中包含可选参数。例如，我们可以添加“-csearch_path=schema_name”来覆盖 PostgreSQL 中当前会话的搜索路径。这相当于写一个查询，“将 search_path 设置为 schema_name”。

```
engine = sqlalchemy.create_engine(params, **connect_args**={'options': '**-csearch_path=schema_name**'})
```

# 使用 SQLAlchemy 运行 SQL 查询

一旦创建了连接，我们就可以用 Python 与 SQL 数据库进行交互。让我们从最简单的查询开始，“SELECT * FROM table”。

```
from sqlalchemy.sql import **text**
sql = '''
    SELECT * FROM table;
'''
**with engine.connect()**.execution_options(**autocommit=True**) as conn:
    query = conn.execute(**text**(sql))         
df = pd.DataFrame(query.**fetchall()**)
```

我们将使用几个关键功能。

*   **text():** SQLAlchemy 允许用户通过函数“text()”使用 Python 中的**原生 SQL 语法**。它会将文本语句原封不动地传递给 SQL 数据库。因此，我们可以在一个 Python 框架内使用原生 SQL 语法，比如， **DELETE** ， **UPDATE** ， **INSERT** ， **SELECT，全文搜索**等。
*   **engine.connect():** 该函数返回一个 SQL **连接**对象。通过使用 Python 上下文管理器(例如，**和**语句)，" **Connection.close()** "函数将自动包含在代码块的末尾。
*   **autocommit=True:** 函数内部的可选参数。execution_options()"允许我们打开自动提交特性。这意味着我们不需要编写额外的代码，例如，“connection.commit()”和“connection.rollback()”。" **connection.commit()** "将提交对 SQL 数据库的任何更改，而" **connection.rollback()** "将放弃任何更改。使用自动提交的一个好处是我们有更少的代码行，并且解决了忘记提交变更的潜在问题。
*   **fetchall():** 这个函数将返回 row 对象，这些对象可以与 Pandas 集成在一起，创建一个数据框。

在下面的例子中，我们将**更新、插入、删除 SQL 表中的**行。与 **SELECT** 唯一不同的是我们编写了“conn.execute(text(sql))”而不是“query = conn.execute(text(sql))”，因为我们不提取表。

```
# **Update** rows in a SQL table
sql = '''
    UPDATE table 
    SET col='abc'
    WHERE condition;
'''
with engine.connect().execution_options(autocommit=True) as conn:
    **conn.execute(text(sql))**# **Insert** new rows in a SQL table
sql = '''
    INSERT INTO df
    VALUES 
       (1, 'abc'),
       (2, 'xyz'),
       (1, 'abc');
'''
with engine.connect().execution_options(autocommit=True) as conn:
    **conn.execute(text(sql))**# **Delete** rows in a SQL table
sql = '''
    DELETE FROM df
    WHERE condition;
'''
with engine.connect().execution_options(autocommit=True) as conn:
    **conn.execute(text(sql))**
```

在 Python 中运行 SQL 查询可以非常**灵活**。我们可以设置一个 for 循环来基于不同的条件运行多个 SQL 查询。例如:

```
For i in [value_1, value_2, value_3, ...]:
 if condition_1:
  sql = '''**sql_query_1**'''
 elif condition_2:
  sql = '''**sql_query_2**'''
 else:
  sql = '''**sql_query_3**'''

 with engine.connect().execution_options(autocommit=True) as conn:
  conn.execute(text(sql))
```

# 运行多个 SQL 查询

在单个块中运行多个 SQL 查询也很简单。我们只需要用**分号**分隔语句。SQLAlchemy 的简单实现使得在 Python 中与 SQL 交互变得容易。

```
sql = '''
    DROP TABLE IF EXISTS df;
    CREATE TABLE df(
            id SERIAL PRIMARY KEY,
            salary integer
    );
    INSERT INTO df (salary)
    VALUES 
            (400),
            (200),
            (3001);
    SELECT * FROM df;
'''
with engine.connect().execution_options(autocommit=True) as conn:
    query = conn.execute(text(sql))         
df = pd.DataFrame(query.fetchall())
```

# 在 Pandas 数据框中存储 SQL 表

我们已经提到了在 pandas 数据框中保存 SQL 表的“fetchall()”函数。**或者**，我们也可以使用“ **pandas.read_sql** ”来实现。由于 SQLAlchemy 是和 Pandas 集成的，所以我们可以用“con = conn”直接使用它的 SQL 连接。

```
with engine.connect().execution_options(autocommit=True) as conn:
    df = pd.**read_sql**(f"""SELECT * FROM table_name WHERE condition""", **con = conn**)
```

# 将数据帧插入现有的 SQL 数据库

要将新行插入到现有的 SQL 数据库中，我们可以使用带有原生 SQL 语法 insert 的代码，如上所述。或者，我们可以用“T2”熊猫。DataFrame.to_sql ，带有选项“*if _ exists = ' append***'**”将**行大容量插入到 sql 数据库中。这种方法的一个好处是我们可以充分利用 Pandas 的功能，比如导入外部数据文件和转换原始数据。因此，我们可以有一个兼容的 Pandas 数据帧(例如，具有与 SQL 表相同的列和数据类型),并准备好插入到现有的 SQL 数据库中。**

```
df = pd.read_excel('sample.xlsx')
with engine.connect().execution_options(autocommit=True) as conn:
    df.**to_sql**('table_name', con=conn, **if_exists=**'**append**', index= False)
```

# 创建新的 SQL 数据库

**熊猫。DataFrame.to_sql** ”也用于创建新的 sql 数据库。正如您在下面的示例中看到的，我们从 excel 电子表格中导入外部数据，并从 pandas 数据框架中创建新的 SQL 表。

```
from **sqlalchemy.types** import Integer, Text, String, DateTimedf = pd.read_excel('sample.xlsx')
df.**to_sql**(
    "table_name", 
    con = engine,
    **if_exists = "replace"**,
    schema='shcema_name',   
    index=False,
    chunksize=1000,
    dtype={
       "col_1_name": Integer,
       "col_2_name": Text,
       "col_3_name": String(50),
       "col_4_name": DateTime
     }
)
```

为了正确地创建一个新的 sql 表，我们需要用" **to_sql()** "函数指定几个重要的参数。

*   **if_exists** :如果数据库中已经存在一个名为“table_name”的表，该参数将指示如何处理。传递“ **replace** ”将删除现有表中的所有行，并将其替换为当前的 pandas 数据框。如上所述，传递“ **append** ”只会将 pandas 数据框追加到现有的 SQL 表中。
*   **schema** :该参数将获取保存新 SQL 表的模式名。如果已经在连接中指定了模式名，则不需要。
*   **index** :该参数表示我们是否要在新的 SQL 表中为 DataFrame 的索引创建一列。
*   **chuncksize** :该参数将指定每批中一次要插入的行数。默认情况下，所有行都将一次写入。
*   **dtype** :该参数将指定新 SQL 表中列的数据类型。我们使用的数据类型来自 **sqlalchemy.types.**

# 感谢您的阅读！！！

如果你喜欢这篇文章，并且想**请我喝杯咖啡，**请[点击这里](https://ko-fi.com/aaronzhu)。

您可以注册一个 [**会员**](https://aaron-zhu.medium.com/membership) 来解锁我的文章的全部访问权限，并且可以无限制访问介质上的所有内容。如果你想在我发表新文章时收到电子邮件通知，请订阅。