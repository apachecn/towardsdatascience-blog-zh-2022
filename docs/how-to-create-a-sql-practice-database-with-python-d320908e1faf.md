# 如何用 Python 创建 SQL 实践数据库

> 原文：<https://towardsdatascience.com/how-to-create-a-sql-practice-database-with-python-d320908e1faf>

## 亲自动手

## 最后，开始用您自己的数据库练习 SQL

![](img/576eac5191fa3b5f5b56594caf9bf21b.png)

人物插图由[故事集](https://storyset.com/people)

编写 SQL 很重要。作为一名有抱负的数据分析师/科学家，高效查询数据库的能力通常被认为是最重要的技能之一。

SQL 不仅重要，而且非常常用。根据 2021 年 Stackoverflow 开发者调查[，SQL 位列使用的五大编程语言之一。](https://insights.stackoverflow.com/survey/2021)

所以，我们或许应该投入一些时间来学习 SQL。

> 但只有一个问题:首先，在没有数据库的情况下，如何练习查询数据库？

在接下来的几节中，我们将解决这个基本问题，并学习如何从头开始创建我们自己的 MySQL 数据库。在 Python 和一些外部库的帮助下，我们将创建一个简单的脚本，自动创建并使用随机生成的数据填充我们的表。

但是在深入实现细节之前，我们必须先了解一些先决条件。

> **注意**:当然还有其他方法可以获得 SQL 数据库用于实践目的(例如，简单地通过[下载](https://www.sqlservertutorial.net/sql-server-sample-database/))。然而，使用 Python 和一些外部库为我们提供了额外的、有价值的实践机会。

# 先决条件

让我们从基础开始。

首先我们需要安装 [MySQL 工作台](https://dev.mysql.com/downloads/workbench/)并设置一个[连接](https://dev.mysql.com/doc/workbench/en/wb-mysql-connections-new.html)。一旦我们有了一个可以用来建立数据库的连接:

```
CREATE DATABASE IF NOT EXISTS your_database_name;
```

现在，我们只需要安装必要的 python 库，基本设置已经完成。我们将要使用的库如下，可以通过终端轻松安装。

1.  数字`pip install numpy`
2.  sqlalchemy `pip install sqlalchemy`
3.  骗子`pip install faker`

# 创建脚本

完成基本设置后，我们终于可以开始编写 python 脚本了。

让我们先用一些样板代码创建一个类，为我们提供一个蓝图，指导我们完成其余的实现。

目前为止没什么特别的。

我们基本上只是创建了一个类，存储了供以后使用的数据库凭证，导入了库，并定义了一些方法。

## 建立连接

我们必须完成的第一件事是创建一个数据库连接。

幸运的是，我们可以利用 python 库 sqlalchemy 来完成大部分工作。

该方法创建并存储 3 个对象作为实例属性。

首先，我们创建一个引擎，作为任何 sqlalchemy 应用程序的起点，描述如何与特定种类的数据库/ DBAPI 组合进行对话。

在我们的例子中，我们指定一个 MySQL 数据库并传递我们的凭证。

接下来，我们创建一个连接，它只允许我们执行 SQL 语句和一个元数据对象，元数据对象是一个容器，它将数据库的不同特性保存在一起，允许我们关联和访问数据库表。

## 创建表格

现在，我们需要创建我们的数据库表。

我们创建了 3 个表，并将它们存储在一个字典中，供以后参考。

在 sqlalchemy 中创建一个表也非常简单。我们简单地实例化一个新的表类，提供一个表名、元数据对象，并指定不同的列。

在我们的例子中，我们创建了一个职务、一个公司和一个人员表。person 表还通过外键引用其他表，这使得数据库在实践 SQL 连接方面更加有趣。

一旦我们定义了表，我们只需要调用元数据对象的`create_all()`方法。

## 生成一些随机数据

我们创建了数据库表，但是我们仍然没有任何数据可以使用。因此，我们需要生成一些随机数据并插入到我们的表中。

现在，我们利用 Faker 库来生成随机数据。

我们只是在一个 for 循环中用随机生成的数据创建一个新记录，用一个字典表示。然后将单个记录追加到可用于(多个)insert 语句的列表中。

接下来，我们从 connection 对象中调用`execute()`方法，并将字典列表作为参数传递。

这就是了！我们完成了我们的类实现——我们只需要实例化这个类并调用它的方法来创建我们的数据库。

# 进行查询

唯一剩下的事情—我们需要验证我们的数据库已经启动并正在运行，并且确实包含了一些数据。

从一个基本查询开始:

```
SELECT *
FROM jobs
LIMIT 10;
```

![](img/d33406d1675f5f1fb924562c57cf6e8f.png)

基本查询结果[图片由作者提供]

看起来我们的脚本成功了，我们有了一个包含实际数据的数据库。

现在，让我们尝试一个更复杂的 SQL 语句:

```
SELECT
  p.first_name,
  p.last_name,
  p.salary,
  j.description
FROM
  persons AS p
JOIN
  jobs AS j ON
  p.job_id = j.job_id
WHERE 
  p.salary > 130000
ORDER BY
  p.salary DESC;
```

![](img/d86657f7705959780300b7d81b78e5df.png)

更多相关查询结果[图片由作者提供]

这看起来也很有希望——我们的数据库还活着，而且运行良好。

# 结论

在本文中，我们学习了如何利用 Python 和一些外部库，用随机生成的数据创建我们自己的练习数据库。

尽管人们可以很容易地下载一个现有的数据库来开始练习 SQL，但是使用 Python 从头开始创建我们自己的数据库提供了额外的学习机会。因为 SQL 和 Python 经常是紧密联系在一起的，所以这些学习机会会被证明是特别有用的。

你可以在我的 GitHub 上找到完整的代码。

随意调整代码，创建一个更大、更好、更复杂的数据库。

![Marvin Lanhenke](img/5b1b337a332bf18381aa650edc2190bd.png)

[马文·兰亨克](https://medium.com/@marvinlanhenke?source=post_page-----d320908e1faf--------------------------------)

## # 30 日

[View list](https://medium.com/@marvinlanhenke/list/30daysofnlp-3974a0c731d6?source=post_page-----d320908e1faf--------------------------------)30 stories![](img/97f1fa41b4f518ab6047fde7d260d65f.png)![](img/d605569da3c842a21a16b568b04cf244.png)![](img/382a4c66fbeaa2d71de204436d7b4f68.png)

*喜欢这篇文章吗？成为* [*中等会员*](https://medium.com/@marvinlanhenke/membership) *继续无限学习。如果你使用下面的链接，我会收到你的一部分会员费，不需要你额外付费。*

[](https://medium.com/@marvinlanhenke/membership) [## 通过我的推荐链接加入 Medium-Marvin Lanhenke

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

medium.com](https://medium.com/@marvinlanhenke/membership)