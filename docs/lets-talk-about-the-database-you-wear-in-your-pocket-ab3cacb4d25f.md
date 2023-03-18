# 让我们来谈谈你放在口袋里的数据库

> 原文：<https://towardsdatascience.com/lets-talk-about-the-database-you-wear-in-your-pocket-ab3cacb4d25f>

## 了解亿万人使用的未知数据库的更多信息

![](img/11eb43afd45c9e2c3300356e8fc8133c.png)

图片 Saskia van Baaren

当我们这个行业的人想到数据库时，很多人会想到 Oracle、MySQL、Postgresql，也许？但是使用量超过所有其他数据库总和的数据库呢？再来说说 SQLite！

# 什么是 SQLite？

SQLite 是一个实现 SQL 数据库引擎的库。它有几个吸引人的特性:

*   用 C 写的:速度极快
*   它很小:SQLite 运行在小设备上(比如电话和微波炉)
*   可靠:这个库每天被数十亿人使用，因为几乎每个手机应用程序都以这样或那样的方式使用 SQLite
*   它功能齐全:尽管名为 SQLite，但它有很多令人印象深刻的功能。

## 这是一个图书馆

与其他数据库不同，SQLite 是无服务器的。毕竟是*库*！SQLite 可以与您的代码一起提供，并将其数据存储在文件系统的文件中。

SQLite 通常包含在其他语言中。[比如 Python](https://python.land/) 就提供了开箱即用。因此，您可以使用 SQLite 来存储和分析数据，而无需设置数据库，我认为人们应该更经常地使用它！

> 您可以使用 SQLite 来存储和分析数据，而无需设置数据库，我认为人们应该更经常地使用它！

# SQLite 怎么发音

没有什么比发音错误更令人困惑，有时甚至尴尬的了。SQLite 有两种发音方式，都被认为 OK。

SQL 是结构化查询语言的缩写，通常读作“ess queue ell”或“sequel”因此，SQLite 可以读作:

1.  ess 队列灯
2.  续集灯

挑你喜欢的风格；我更喜欢最后一个。

# 浏览现有的 SQLite 数据库

我们可以使用 SQLite shell 来浏览 SQLite 数据库。要打开 SQLite shell，我们必须在终端或命令提示符下输入`sqlite3`命令。这在所有操作系统(Windows、MacOS、Linux)上都是一样的:

```
C:\> sqlite3
SQLite version 3.35.5 2021-04-19 18:32:05 
Enter ".help" **for** usage hints.
Connected to a transient **in**-memory database.
Use ".open FILENAME" to reopen on a persistent database. 
sqlite>
```

当不带任何参数打开 SQLite 时，它将创建一个内存数据库。正如 SQLite 指出的，我们可以通过使用`.open FILENAME`命令打开一个现有的数据库。但是，如果您已经有了一个基于文件的 SQLite 数据库，那么直接打开它会更容易:

```
C:\> sqlite3 customers.db
SQLite version 3.35.5 2021-04-19 18:32:05
Enter ".help" **for** usage hints.
sqlite>
```

# 从头开始创建 SQLite 数据库

如果您没有数据库，让我们先从头创建一个，这样我们就有东西可以使用了。我们需要用`sqlite3`命令再次启动 SQLite 来获得 SQLite shell。我们可以通过将数据库作为第一个参数直接给它命名:

```
$ sqlite3 customers.db
SQLite version 3.35.5 2021-04-19 18:32:05
Enter ".help" **for** usage hints
sqlite>
```

注意，我喜欢使用`.db`文件扩展名。您可以选择任何您想要的文件名，但是我建议使用`.db`扩展名来清楚地将它标记为数据库文件。

现在创建一个表，并通过输入以下行插入一些数据:

```
sqlite> **create** **table** customers(**name** text, age int);
sqlite> **insert** **into** customers **values**('Erik', 40);
sqlite> **insert** **into** customers **values**('Mary', 53);
```

我们可以通过一个简单的 SELECT 语句看到结果:

```
sqlite> **select** * **from** customers;
Erik|40
Mary|53Code language: SQL (Structured Query Language) (sql)
```

要关闭外壳，请按`control+D`，它会将文件尾字符发送到您的终端，或者键入。退出命令。

如果您检查文件系统，您会看到 SQLite 已经为我们创建了`customers.db`文件。

# SQLite 浏览器:打开一个数据库

现在我们有一个现有的数据库可以使用。让我们首先在文件系统的一个文件中打开这个 SQLite 数据库。我们将使用之前的数据库`customers.db`。我们可以像这样用`sqlite3`命令打开它:

```
C:\> sqlite3 customers.db
SQLite version 3.35.5 2021-04-19 18:32:05
Enter ".help" for usage hints.
sqlite>Code language: plaintext (plaintext)
```

现在我们可以浏览我们的 SQLite 数据库了！

# 有用的 SQLite 命令

SQLite 没有大张旗鼓地打开数据库，但是它在屏幕上显示了一条有用的消息。我们可以输入`.help`来获得用法提示，但是它会输出一个很大的命令列表，其中大部分我们在这一点上并不关心。在下面的摘录中，我截取了输出的大部分内容，只包含了将 SQLite 用作浏览器时有用的命令:

```
sqlite> .help
...
.databases          List names and files of attached databases
...
.mode MODE ?TABLE? Set output mode
.schema ?PATTERN?  Show the CREATE statements matching PATTERN.show              Show the current values for various settings
...
.tables ?TABLE?    List names of tables matching LIKE pattern TABLE
...Code language: plaintext (plaintext)
```

## 。数据库

使用`.databases`,我们可以看到哪个数据库连接到这个会话。您可以一次打开多个数据库，将数据从一个数据库复制到另一个数据库，或者连接不同数据库中的表中的数据。

当输入这个命令时，它将输出如下内容:

```
sqlite> .databases
main: C:\customers.db r/wCode language: SQL (Structured Query Language) (sql)
```

## 。桌子

`.tables`命令显示所有可用的表。在我们的例子中，这个命令的输出如下所示:

```
sqlite> .tables
customersCode language: SQL (Structured Query Language) (sql)
```

## 。 (计划或理论的)纲要

`.schema`命令打印用于创建表的 CREATE 语句。当不带参数运行时，它将打印所有表的模式。您可以通过提供特定表的名称来打印该表的模式:

```
sqlite> .schema customers
**CREATE** **TABLE** customers(**name** text, age int);Code language: SQL (Structured Query Language) (sql)
```

# 浏览 SQLite 表

此时，我们知道有哪些表，甚至查看了这些表背后的模式。如果您想查看这些表中的内容，您需要使用 SQL 语法。如果您不熟悉 SQL 语法，您可以使用并调整我下面提供的示例来安全地浏览数据。这些命令都不会改变数据库。

## SELECT * FROM table _ name

SELECT 语句从表中“选择”数据。我们可以给它各种选项来过滤输出。最简单快捷的开始方式是选择表格中的所有内容。我们通过使用通配符符号`*`来做到这一点:

```
sqlite> **select** * **from** customers;
Erik|40
Mary|53Code language: SQL (Structured Query Language) (sql)
```

您在输出中看到的是表中的行。然而，输出并不清楚。我们可以用`.mode column`命令来解决这个问题:

```
sqlite> .mode column
sqlite> **select** * **from** customers;
name  age
*----  ---*
Erik  40 
Mary  53Code language: SQL (Structured Query Language) (sql)
```

## 限制行数

如果表格中有大量数据，您可能想要限制您看到的行数。这很容易做到，只需在 select 语句的末尾添加一个限制:

```
sqlite> **select** * **from** customers **limit** 1;
Erik|40Code language: SQL (Structured Query Language) (sql)
```

LIMIT 接受两个值，所以上面的命令是`limit 0, 1`的简写。含义:限制行数，从第 0 行开始，返回 1 行。

所以极限的语法是:

```
sqlite> **select** * **from** **TABLE** **limit** START_ROW, NUMBER_OF_ROWS;Code language: SQL (Structured Query Language) (sql)
```

记住计算机从零开始计数，所以如果我们只想看到第二行，我们可以用这个:

```
sqlite> **select** * **from** customers **limit** 1, 1;
name  age
*----  ---*
Mary  53Code language: SQL (Structured Query Language) (sql)
```

# 结论

您已经学习了如何创建、打开和浏览 SQLite 数据库。我们已经研究了一些 SQLite 命令，它们帮助我们检查数据库及其表。最后，您学习了使用 SELECT 语句查看表中的数据。

下次当您开始一个项目或需要分析一些关系数据时，请考虑 SQLite！很有可能，对于你所需要的来说，这已经足够了，而且可能会为你节省一些麻烦和宝贵的时间！