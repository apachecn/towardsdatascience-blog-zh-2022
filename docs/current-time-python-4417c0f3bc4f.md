# 如何在 Python 中获取当前时间

> 原文：<https://towardsdatascience.com/current-time-python-4417c0f3bc4f>

## 使用 Python 以编程方式计算当前日期和时间

![](img/23dac43a96b5fed6dd5a39ca747b1182.png)

照片由 [Djim Loic](https://unsplash.com/@loic?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/time?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

如果你已经编程了一段时间，我敢肯定你遇到过必须处理日期(时间)和时间戳的用例。

当使用这样的结构时，确保使用**时区敏感的**对象是非常重要的。在生产环境中，为了完成某项任务，全球范围内许多不同的服务相互连接是很常见的。这意味着我们——作为程序员——需要确保避免时区无关的构造，这些构造可能会引入不必要的错误。

在接下来的几节中，我们将介绍几种不同的方法，您可以遵循这些方法以编程方式在 Python 中计算当前时间。

更具体地说，我们将演示如何使用`datetime`和`time`模块推断当前日期和/或时间以及时间戳。此外，我们将讨论如何将日期时间作为字符串处理，以及如何考虑时区。

## 如何用 Python 计算当前时间

现在为了以编程方式推断 Python 中的当前日期时间，我们可以使用如下概述的`datetime`模块:

```
>>> from datetime import datetime
>>>
>>> now = datetime.now()
>>> now
datetime.datetime(2022, 9, 30, 16, 34, 24, 88687)
```

上面的表达式将返回一个类型为`datetime.datetime`的对象。如果您想以更直观、更易读的格式打印日期时间，您需要做的就是将它转换为`str`(例如`str(now)`)或者调用`strftime()`来指定您希望新字符串对象具有的确切字符串格式。

例如，假设我们希望只保留日期时间中的日期部分，而放弃时间信息。以下内容应该可以解决问题:

```
>>> now_dt = now.strftime('%d-%m-%Y')
>>> now_dt
'30-09-2022'
```

类似地，如果您想只保留 datetime 对象的时间部分，您可以使用`time()`方法:

```
>>> from datetime import datetime
>>> now_time = datetime.now().time()
>>> now_time
datetime.time(16, 43, 12, 192517)
```

同样，我们可以将上面的 datetime 对象格式化为一个字符串。如果我们想要丢弃毫秒记录部分，只保留小时、分钟和秒，那么下面的表达式就可以满足我们的要求:

```
>>> now_time.strftime('%H:%M:%S')
'16:43:12'
```

另一个可以帮助我们处理时间的有用模块是内置的`time`。根据操作系统和主机的日期时间配置，`ctime`方法将返回当前时间的字符串。

```
>>> import time
>>> time.ctime()
'Fri Sep 30 16:48:22 2022'
```

## 引入支持时区的日期时间对象

现在，我们在上一节中展示的问题是，我们创建的 datetime 对象是**时区无关的**。例如，我住在伦敦—这意味着如果我和另外一个住在美国或印度的人在同一时间点运行我们之前演示的相同命令，我们最终都会得到不同的结果，因为上面的所有表达式都将根据主机的时区来计算当前时间(这显然会因地点而异)。

通用协调时间(UTC)是一个全球标准，也被程序员群体所采用。UTC(几乎)等同于 GMT，它不会因为夏令时等而改变。用 UTC 交流与日期时间相关的需求是很常见的，因为它是通用时区，采用 UTC 可以帮助人们更容易地交流日期时间和日程安排。其他常见的编程结构(如时间戳/unix 纪元时间)也使用 UTC 进行计算。

回到前面的例子，让我们试着推断构造的 datetime 对象的时区。请再次注意，我的工作地点在伦敦，在撰写本文时，我们正处于英国夏令时(BST):

```
>>> from datetime import datetime
>>> now = datetime.now()
>>> tz = now.astimezone().tzinfo
>>> tz
datetime.timezone(datetime.timedelta(seconds=3600), 'BST')
```

现在，如果你在不同的时区运行上面的命令，你会得到不同的`now`和`tz`的值——这正是问题所在。

相反，我们可以用 UTC 时区计算当前的日期时间，这样我们所有人都会得到相同的计算结果。假设我住在伦敦(目前是英国夏令时)，我们预计当前的 UTC 日期时间将比我当地的 BST 时区晚一个小时:

```
>>> from datetime import datetime
>>> now = datetime.utcnow()
>>> now
datetime.datetime(2022, 9, 30, 16, 22, 22, 386588)
```

注意，也可以通过调用`replace()`方法并提供`datetime.timezone`模块中可用的时区选项之一来更改日期时间对象的时区信息。

例如，让我们创建一个当前日期时间的日期时间对象(BST 格式):

```
>>> from datetime import datetime, timezone
>>> now = datetime.now()
>>> now
datetime.datetime(2022, 9, 30, 17, 26, 15, 891393)
>>> now_utc =  now.replace(tzinfo=timezone.utc)
datetime.datetime(2022, 9, 30, 17, 26, 15, 891393, tzinfo=datetime.timezone.utc)
```

## 最后的想法

在今天的文章中，我们展示了用 Python 计算日期时间和时间戳的几种不同方法。这可以通过使用两个内置模块中的一个来实现，即`datetime`和`time`。

此外，我们讨论了使用**时区感知构造**的重要性。在生产环境中运行的现代系统通常涉及全球托管的许多不同的服务。

这意味着托管在不同国家的服务将位于不同的时区，因此我们需要以一致和准确的方式处理这种不规则性。

这就是我们通常想要使用时区感知对象的原因。此外，在编程环境中，坚持 UTC 时区和 unix 时间戳是很常见的。

在我即将发表的一篇文章中，我们将讨论更多关于 UTC 和 Unix 纪元时间的内容，所以请保持关注:)

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**相关文章你可能也喜欢**

[](/requirements-vs-setuptools-python-ae3ee66e28af)  [](/diagrams-as-code-python-d9cbaa959ed5)  [](/python-poetry-83f184ac9ed1) 