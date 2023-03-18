# 编写定制的 Django 数据库函数

> 原文：<https://towardsdatascience.com/writing-custom-django-database-functions-4358e6030df7>

## 通过定制 Django 数据库函数，可以更好地控制查询过滤

![](img/b419ff2ed49ad469b422e151b4bb351c.png)

根据[公共许可证](https://undraw.co/license)下的 [Undraw](https://undraw.co/) 生成的图像

Django 的数据库函数表示将在数据库中运行的函数。它为用户提供了一种将底层数据库提供的函数用作注释、聚合或过滤器的方式。函数也是[表达式](https://docs.djangoproject.com/en/4.0/ref/models/expressions/)，所以可以和[聚合函数](https://docs.djangoproject.com/en/4.0/ref/models/querysets/#aggregation-functions)等其他表达式一起使用和组合。

Django 丰富的特性之一是可以定制它的各种功能。是的，你答对了！🙌我们可以根据需要定制 Django 数据库功能。

📝在本帖中，我们将通过下面的几个例子来了解根据我们的业务需求编写自定义函数的要点。

👉让我们先了解一下`Django Func()`类，它是我们前进的基础。

# 📜姜戈`Func(*expressions, **extra)`级

*   类 [Func()](https://docs.djangoproject.com/en/4.0/ref/models/expressions/#django.db.models.Func) 是`Django Query Expressions`最通用的部分
*   它允许以某种方式将几乎任何函数或操作符实现到`Django ORM`中
*   [Func()表达式](https://docs.djangoproject.com/en/4.0/ref/models/expressions/#django.db.models.Func)是所有涉及数据库函数如`COALESCE`和`LOWER`或聚合如`SUM`的表达式的基本类型
*   我推荐在使用`Func()`之前先阅读[避开 SQL 注入](https://docs.djangoproject.com/en/4.0/ref/models/expressions/#avoiding-sql-injection)

以下是编写自定义数据库函数的一些方法:

# 🔹自定义数据库函数

我们可以使用 Django 的 [Func 类](https://docs.djangoproject.com/en/4.0/ref/models/expressions/#django.db.models.Func)创建自定义数据库函数。在我的一个项目中，我想用特定的日期格式将 Django 过滤器中的`UTC`时间戳转换成`IST`。编写两个简单的 Django 数据库函数帮助我`reuse`在多个实例中使用它，如下所示:

```
from django.db.models import Func

class TimestampToIST(Func):
    """ Converts the db (UTC) timestamp value to IST equivalent timestamp
    """
    function = 'timezone'
    template = "%(function)s('Asia/Calcutta', %(expressions)s)"

class TimestampToStr(Func):
    """ Converts the timestamp to string using the given format
    """
    function = 'to_char'
    template = "%(function)s(%(expressions)s, 'DD/MM/YYYY HH24:MI:SS')"  # 21/06/2021 16:08:34 # Usage
Author.objects.annotate(last_updated=TimestampToStr(TimestampToIST(F('updated_at'))))
```

# 🔹数据库功能的部分实现

另一个很好的定制例子是创建一个新版本的函数，其中已经填充了一两个参数。例如，让我们创建一个专门的`SubStr`，它从字符串中提取第一个字符:

```
from functools import partial
from django.db.models.functions import Substr

ExtractFirstChar = partial(Substr, pos=1, length=1) # Usage
User.objects.annotate(name_initial=ExtractFirstChar('first_name'))
```

# 🔹执行没有聚合函数的`GROUP BY`

想象一种情况，我们想使用`GROUP BY`而不使用任何聚合函数。`Django ORM`不允许我们在没有集合函数的情况下使用`GROUP BY`🤨因此，为了实现这一点，我们可以创建一个 Django 函数，它被 Django 视为一个聚合函数，但在一个`SQL query`中计算为`NULL`，来源于 [StackOverflow](https://stackoverflow.com/a/65066965/9334209)

```
from django.db.models import CharField, Func

class NullAgg(Func):
    """Annotation that causes GROUP BY without aggregating.

    A fake aggregate Func class that can be used in an annotation to cause
    a query to perform a GROUP BY without also performing an aggregate
    operation that would require the server to enumerate all rows in every
    group.

    Takes no constructor arguments and produces a value of NULL.

    Example:
        ContentType.objects.values('app_label').annotate(na=NullAgg())
    """
    template = 'NULL'
    contains_aggregate = True
    window_compatible = False
    arity = 0
    output_field = CharField()
```

# 📑资源

*   [Django 数据库功能](https://docs.djangoproject.com/en/4.0/ref/models/database-functions/)
*   [Django Func()类](https://docs.djangoproject.com/en/4.0/ref/models/expressions/#django.db.models.Func)
*   [stack overflow response GROUP BY without aggregate](https://stackoverflow.com/a/65066965/9334209)

*原载于 2022 年 5 月 19 日*[*https://dev . to*](https://dev.to/idrisrampurawala/writing-custom-django-database-functions-4dmb)*。*