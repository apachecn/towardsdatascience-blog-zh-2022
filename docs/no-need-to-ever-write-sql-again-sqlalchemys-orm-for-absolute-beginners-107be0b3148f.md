# 再也不需要写 SQL 了:SQLAlchemy 的 ORM 完全适合初学者

> 原文：<https://towardsdatascience.com/no-need-to-ever-write-sql-again-sqlalchemys-orm-for-absolute-beginners-107be0b3148f>

## 有了这个 ORM，你可以创建一个表，插入、读取、删除和更新数据，而不用写一行 SQL

![](img/d835176c3d339720499712088b3a56db.png)

用 ORM 开发你的生活会有多美好(图片由 [Christian Joudry](https://unsplash.com/@cjoudrey) 在 [Unsplash](https://unsplash.com/photos/DuD5D3lWC3c) 上提供)

与执行原始 SQL 相比，使用 SQLAlchemy 的 ORM 有许多优点。在本文中，我们将演示使用 ORM 的基础知识。阅读完本文后，您将能够创建表、插入、选择、更新和删除数据。理解 ORM 将为你节省很多错误和调试时间，所以不要再浪费时间了，让我们开始编码吧！

## 什么是 ORM，它有什么作用？

ORM 代表对象关系映射。这是一种在 Python 代码和 SQL 表之间充当某种翻译器的技术。它基本上允许您使用数据库表中的记录作为 Python 中的 Python 对象。可以把它想象成把你的表转换成一个列表，其中每个条目都是一个代表一行的 Python 对象。

使用 ORM 有很多优点:

*   **数据库无关:**即使在迁移数据库时，您的代码也能工作
*   **不易出错:**没有语法错误，因为生成了 SQL
*   **通过模型实现的额外功能**(例如验证)
*   **组织:**通过保持项目中所有模型的整洁组织

现在我们知道使用 ORM 是个好主意，让我们来看看如何使用。

[](/keep-your-code-secure-by-using-environment-variables-and-env-files-4688a70ea286)  

# 使用 ORM

在这一部分中，我们将关注基础知识，并使用 SQLAlchemy ORM 来:

1.  创建表格
2.  插入记录
3.  选择一条记录
4.  删除记录
5.  更新记录

**注意** : SQLALchemy 需要一个允许 Python 从我们的数据库中读取数据的数据库引擎。看看下面这篇关于如何创建一个的短文。

[](/sqlalchemy-for-absolute-beginners-22227a287ef3)  

## 1.创建表格

在这一部分中，我们将为我们的表定义模型。在下面的部分中，我们以 Student 类的形式创建一个 Python 对象，然后使用 SQLAlchemy 实际创建表。

[*查看如何创建 dbEngine_PG* ***此处***](https://mikehuls.medium.com/sqlalchemy-for-absolute-beginners-22227a287ef3)

就这么简单！Students 类描述了我们的表:表的名称、模式和每一列的配置。

接下来我们导入 declarative_base。这是一个返回基类的函数，所有的表模型都必须继承这个基类。把它想象成给你所有的模型提供工具箱。然后，我们使用基本的“工具箱”通过我们的数据库连接来`create_all`表格。

注意，这是我们定义表结构的唯一地方。这意味着您可以将所有的表模型捆绑在一起，而不是在整个项目中使用原始 SQL。

[](/sql-insert-delete-and-update-in-one-statement-sync-your-tables-with-merge-14814215d32c) [## SQL —在一条语句中插入、删除和更新:用 MERGE 同步表

towardsdatascience.com](/sql-insert-delete-and-update-in-one-statement-sync-your-tables-with-merge-14814215d32c) 

## 2.使用 ORM 插入记录

在这一部分中，我们将使用上面定义的模型在新创建的表中插入一些新记录。

[*查看如何创建 dbEngine_PG* ***此处***](https://mikehuls.medium.com/sqlalchemy-for-absolute-beginners-22227a287ef3)

我们通过创建以前的 student 类的新实例来创建一个新的学生。然后，我们使用导入的 sessionmaker 添加并提交到我们的表中。这不仅非常容易；请注意，我们没有编写一行 SQL！

[](/applying-python-multiprocessing-in-2-lines-of-code-3ced521bac8f)  

## 3.使用 ORM 选择记录

选择数据甚至比插入数据更容易:

[*点击*](https://mikehuls.medium.com/sqlalchemy-for-absolute-beginners-22227a287ef3) 查看如何创建 dbEngine_PG

*使用我们的模型，定义我们的语句相当容易。请注意，我们添加了一个限制。当使用连接到 Postgres 数据库的 dbEngine 时，它被编译为`SELECT * FROM students LIMIT 5`。
如果我们切换到 SQL Server dbEngine，这条语句将被编译成`SELECT TOP 18 * FROM students`。这就是数据库无关的 ORM 的威力！*

*接下来，我们在一个会话中执行该语句，并遍历结果:*

```
*Student mike is 33 years old*
```

*[](/dramatically-improve-your-database-inserts-with-a-simple-upgrade-6dfa672f1424) * 

## *4.删除记录*

*删除数据非常类似于选择:*

*[*查看如何创建 dbEngine_PG* ***此处***](https://mikehuls.medium.com/sqlalchemy-for-absolute-beginners-22227a287ef3)*

*请注意，我们对 delete 语句应用了一些过滤。我们当然可以在前一部分的 select 语句中做同样的事情。*

*[](/getting-started-with-cython-how-to-perform-1-7-billion-calculations-per-second-in-python-b83374cfcf77) * 

## *5.更新记录*

*更新也很简单:*

*[*查看如何创建 dbEngine_PG* ***这里***](https://mikehuls.medium.com/sqlalchemy-for-absolute-beginners-22227a287ef3)*

*与`delete`和`select`语句的不同之处在于，我们还添加了用来更新所选记录的新值。这些在 6 号线上整齐的传递。*

*[](/git-for-absolute-beginners-understanding-git-with-the-help-of-a-video-game-88826054459a) * 

# *ORM 的更多优势:验证*

*因为我们已经在类中定义了我们的表，并且还使用该类来创建新记录，所以我们可以添加一些非常高级的验证。想象一下有人试图添加一个 300 岁的学生。我们可以轻松地更新我们的学生表模型，以验证所提供的年龄:*

*如你所见，我们检查提供的年龄是否大于 110，如果是，则抛出一个错误。这将防止该记录被插入我们的数据库。*

*[](/docker-for-absolute-beginners-the-difference-between-an-image-and-a-container-7e07d4c0c01d) * 

# *关系呢？*

*目前我们只有一张桌子:学生。大多数应用程序都有更多相互关联的表。学生是班级的一部分，有多门课程。每门课程有一名教师等。*

*SQLAlchemy ORM 允许您在一条语句中轻松地查询相关的表:选择一个学生还将返回与该学生相关的所有班级、课程和教师。请关注我的下一篇文章[](https://mikehuls.medium.com/membership)**！***

***[](/image-analysis-for-beginners-destroying-duck-hunt-with-opencv-e19a27fd8b6) *** 

# ***后续步骤:***

***有了数据库引擎和 ORM 的知识，我们就可以专注于下一步了。***

1.  ***[使用**迁移模型**](https://mikehuls.medium.com/python-database-migrations-for-beginners-getting-started-with-alembic-84e4a73a2cca)***
2.  ***[了解**索引**如何作用于表格](https://mikehuls.medium.com/sql-understand-how-indices-work-under-the-hood-to-speed-up-your-queries-a7f07eef4080)***
3.  ***[**跟踪数据库中的慢速查询**](https://mikehuls.medium.com/how-to-track-statistics-on-all-queries-in-your-postgres-database-to-prevent-slow-queries-or-730d3f94076c)***
4.  ***[使用**Docker**部署您的数据库](https://mikehuls.medium.com/getting-started-with-postgres-in-docker-616127e2e46d)***
5.  ***[在你的数据库上添加一个**API**](https://mikehuls.medium.com/create-a-fast-auto-documented-maintainable-and-easy-to-use-python-api-in-5-lines-of-code-with-4e574c00f70e)***
6.  ***[还有更多](https://mikehuls.com/articles?tags=database)***

***在以后的文章中，我们将探索如何在 ORM 中动态生成模型，以及如何使用通过外键相关的表，所以请确保 [**跟随我**](http://mikehuls.medium.com/membership) 。***

***[](/virtual-environments-for-absolute-beginners-what-is-it-and-how-to-create-one-examples-a48da8982d4b) *** 

# ***结论***

***在本文中，我们探索了 SQL 炼金术的起源；我们知道如何连接到数据库，以及如何执行原始 SQL。我们还讨论了执行原始 SQL 的利弊，并理解对于大多数项目来说，使用 ORM 是更好的选择。在下一篇文章中，我们将了解 ORM 以及如何使用它。***

***我希望一切都像我希望的那样清楚，但如果不是这样，请让我知道我能做些什么来进一步澄清。同时，看看我的其他关于各种编程相关主题的文章，比如:***

*   ***[Python 为什么这么慢，如何**加速**](https://mikehuls.medium.com/why-is-python-so-slow-and-how-to-speed-it-up-485b5a84154e)***
*   ***[了解一下**巨蟒装饰者**如何在 6 个关卡中工作](https://mikehuls.medium.com/six-levels-of-python-decorators-1f12c9067b23)***
*   ***[创建并发布自己的 **Python 包**](https://mikehuls.medium.com/create-and-publish-your-own-python-package-ea45bee41cdc)***
*   ***[**Docker** 适合绝对初学者——Docker 是什么，怎么用(+举例)](https://mikehuls.medium.com/docker-for-absolute-beginners-what-is-docker-and-how-to-use-it-examples-3d3b11efd830)***

***编码快乐！***

***—迈克***

***附注:喜欢我正在做的事吗？ [*跟我来！*](https://mikehuls.medium.com/membership)***

***[](https://mikehuls.medium.com/membership) ***