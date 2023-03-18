# SQL 方言快速指南

> 原文：<https://towardsdatascience.com/a-quick-guide-to-the-dialects-of-sql-7249125f9228>

## T-SQL、PL/SQL、PL/pgSQL 等之间的区别

![](img/81235ee37174c4d45d68c9983626c0e3.png)

伊恩·巴塔格利亚在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

SQL 于 20 世纪 70 年代首次开发，并于 20 世纪 80 年代在[得到广泛应用，很快成为与数据库通信的行业标准方法。直到最近受到 NoSQL 崛起的威胁，SQL 仍然是一股普遍的力量，也是任何数据专业人士的必备条件。](https://en.wikipedia.org/wiki/SQL#History)

然而，自最初发布以来，SQL 发展成了不同的形式。虽然许多核心和基本的关键字和功能是相同的，但不同的供应商利用同一种语言的不同变体来扩展其功能。这些差异通常很小，但有时很大，并且通常会给需要管理来自两个独立数据库(每个数据库都有自己的方言)的查询的数据专业人员或需要转换到新公司首选供应商的求职者带来困难。

虽然没有详尽地介绍每一种可能的变体或其细微差别的深度，但我们将探索由 Microsoft (T-SQL)、Oracle (PL/SQL)和 PostgreSQL (PL/sgSQL)实现的方言之间的基本差异。此外，还将简要介绍一些其他变体。

# T-SQL

[Transact-SQL](https://docs.microsoft.com/en-us/sql/t-sql/language-reference?view=sql-server-ver15) ，简称 t-SQL，是微软对 SQL 的[扩展。编写 T-SQL 是为了在其生态系统中为查询提供更大的能力，它可以与](https://en.wikipedia.org/wiki/Transact-SQL)[SQL Server Management Studio](https://docs.microsoft.com/en-us/sql/ssms/download-sql-server-management-studio-ssms?view=sql-server-ver15)、[基于 Azure 的产品](https://azure.microsoft.com/en-us/)以及各种其他微软软件一起使用。主要的例外是[访问](https://docs.microsoft.com/en-us/office/client-developer/access/desktop-database-reference/overview-of-the-access-sql-reference)，它使用 [Jet SQL](https://documentation.help/MS-Jet-SQL/) 来代替。

虽然大多数标准的 SQL 查询在翻译时不会出现大的问题，但是 T-SQL 为使用它的人提供了很大的好处。例如，它提供了创建[变量](https://docs.microsoft.com/en-us/sql/t-sql/language-elements/variables-transact-sql?view=sql-server-ver15)的能力，允许更简洁的查询。

```
DECLARE @myVariable INT
SET @myVariable = 5
```

此外，T-SQL 引入了创建 [while 循环](https://docs.microsoft.com/en-us/sql/t-sql/language-elements/while-transact-sql?view=sql-server-ver15)的能力，这允许查询中的迭代。

```
DECLARE @Counter INT
SET @Counter = 0WHILE (@Counter < 10)
BEGIN
  PRINT 'Example'
  SET @Counter = @Counter + 1
END
```

它还提供了 [Try/Catch](https://docs.microsoft.com/en-us/sql/t-sql/language-elements/try-catch-transact-sql?view=sql-server-ver15) 逻辑来更动态地处理错误。

```
BEGIN TRY
  SELECT * FROM MisspelledTableName;
END TRY
BEGIN CATCH
  SELECT 
    ERROR_NUMBER() AS ErrorNumber,
    ERROR_MESSAGE() AS ErrorMessage;
END CATCH
```

虽然详述微软在 T-SQL 中包含的所有额外特性本身就值得一篇[文章来讨论](https://docs.microsoft.com/en-us/sql/t-sql/language-reference?view=sql-server-ver15)，但这足以说明当与他们的产品线一起工作时，这种扩展使 SQL 成为一种更动态的语言。

# PL/SQL

[SQL](https://en.wikipedia.org/wiki/PL/SQL)的过程化语言，缩写为 PL/SQL，是 Oracle 对 SQL 的[扩展。与微软的 T-SQL 非常相似，Oracle 的版本试图在使用他们的解决方案时赋予 SQL 更多的功能。因此，PL/SQL 既可以在他们著名的数据库上工作，也可以在最近的云服务上工作。](https://www.oracle.com/database/technologies/appdev/plsql.html)

就与 T-SQL 的能力而言，PL/SQL 有很多重叠，它还具有[变量](https://docs.oracle.com/en/database/oracle/oracle-database/21/sqpug/VARIABLE.html)的特性。

```
DECLARE
  myVariable NUMBER := 5;
BEGIN
  NULL;
END;
```

同样，PL/SQL 允许使用循环。然而，与 T-SQL 不同的是，除了 [while 循环](https://docs.oracle.com/en/database/oracle/oracle-database/21/lnpls/WHILE-LOOP-statement.html)之外，PL/SQL 还具有两个独立的[循环](https://docs.oracle.com/en/database/oracle/oracle-database/21/lnpls/basic-LOOP-statement.html) [结构](https://docs.oracle.com/en/database/oracle/oracle-database/21/lnpls/FOR-LOOP-statement.html)，为实现提供了不同的选项。

```
DECLARE 
  i NUMBER := 0;
BEGIN
  LOOP
    dbms_output.put_line(i);
    i := i + 1
    IF i > 5 THEN
      exit;
    END IF
  END LOOP
END;
```

然而，PL/SQL 最有趣和最有利的特性之一是[面向对象编程](https://docs.oracle.com/en/database/oracle/oracle-database/21/adobj/about-oracle-objects.html#GUID-8F0BA083-FA6D-4373-B440-50FDDA4D6E90)。那些习惯于这种编程范式的人将很快利用这种独特的选项。

```
CREATE OR REPLACE TYPE person AS OBJECT (
  firstName VARCHAR2(30),
  lastName VARCHAR2(30)
);
/
```

然后可以调用该对象。

```
DECLARE
  customer person;
BEGIN
  customer := person("John", "Doe");
  dbms_output.put_line(customer.firstName);
  dbms_output.put_line(customer.lastName);
END;
```

在[官方文档](https://docs.oracle.com/en/)中可以更深入地探索更多的特性，但是很容易看出 Oracle 创建 PL/SQL 是为了让数据专业人员以比简单的通用 SQL 更强大的方式利用他们的数据库。

# PL/sgSQL

从 SQL 的公司风格转向免费和开源，[过程语言/postgreSQL](https://www.postgresql.org/docs/current/plpgsql.html) ，或 PL/sgSQL，是 postgreSQL 使用的方言。PL/sgSQL 不仅仅是一个相似的名称，它与 Oracle 的 PL/SQL 有许多共同的特性。事实上，这两者非常相似，以至于 PostgreSQL 甚至在其官方[文档](https://www.postgresql.org/docs/current/plpgsql-porting.html)中专门用了一页来说明它们之间相对较少的差异。

列出的一些差异包括字符串变量的不同关键字，使用 REVERSE 关键字时处理循环的略有不同的方法，以及如何编写函数体的一些细微差别。

总的来说，这两个扩展非常相似，并且允许使用 SQL 进行更多的动态编程。

# SQL 的其他变体

尽管 SQL 或多或少是数据库行业的标准，但不同的实现不可避免地会产生一些差异。此外，许多数据库、工具和技术在某种程度上使用了 SQL 这个名称，这导致了进一步的混淆。虽然不全面，但还有一些其他类型的 SQL 值得一提:

[**CQL**](https://cassandra.apache.org/doc/latest/cassandra/cql/)—NoSQL 的一个特定变体，用于 Cassandra 数据库。官方文件[详细描述了它的细节。](https://cassandra.apache.org/doc/latest/cassandra/cql/)

[**DSQL**](https://www.geeksforgeeks.org/dynamic-sql/) —动态 SQL 的简称，这不是一种特定的语言，而是一种允许在运行时动态编写查询的通用技术。顺便说一下，它还与一个[查询构建器](https://dsql.readthedocs.io/en/develop/#)共享一个名称。

[**火鸟 SQL**](https://firebirdsql.org/)——一个自由开放源码的数据库，通常只被称为火鸟，但有时会收到 SQL 后缀。因为它是一个数据库，所以它不是 SQL 的一种独特方言，而是接受几种变体的 [ANSI SQL 方言](https://firebirdsql.org/file/documentation/chunk/en/refdocs/fblangref30/fblangref30-structure.html)。

[**Jet SQL**](https://documentation.help/MS-Jet-SQL/)——前面已经简要提到，Jet SQL 是构建 Microsoft Access 的变体。通常不需要手动编写查询，因为 Access 的内置查询生成器通常会生成代码并处理任何差异。

[**MySQl**](https://www.mysql.com/)——另一个免费的开源数据库，使用了一种同名的 SQL 风格。它与 SQL 之间只有细微的区别，详见[正式文件](https://dev.mysql.com/doc/refman/8.0/en/differences-from-ansi.html)。值得注意的是，另一个流行的开源数据库 MariaDB 是 MySQL 的一个分支，并且保持了与它的高度兼容性。

[**NoSQL**](https://en.wikipedia.org/wiki/NoSQL)——一个用来描述非关系数据查询方法的总称，NoSQL 仍然使用与常规 SQL 重叠的查询命令。

[**SQLAlchemy**](https://www.sqlalchemy.org/)**——虽然不是真正的 SQL 风格，但它是一个 Python 工具包，可以将面向对象的语言转换为针对各种不同数据库实现的 SQL 查询，无需了解它们的差异。**

**[**SQLite**](https://www.sqlite.org/index.html)——SQLite 是基于文件的，而不是基于服务器的，它是一个轻量级数据库，不是语言本身的变体。主要用于本地应用程序，它的 SQL 风格在数据类型、关键字方面有一些小的变化，并且不包括常规 SQL 的一些特性。**

# **结论**

**虽然 SQL 是一个行业标准，但是它的各种实现方式给那些进入这个领域的人带来了很多困惑和挫折。虽然这不一定是一件坏事，但很大程度上是因为这些实现极大地扩展了基础语言的能力，可能需要额外的教育来充分理解它们的差异。**

**不幸的是，许多工具和数据库通过在其名称中选择使用 SQL 来进一步混淆这个问题。然而，通过一些研究，区分语言和数据库或工具是可能的。**