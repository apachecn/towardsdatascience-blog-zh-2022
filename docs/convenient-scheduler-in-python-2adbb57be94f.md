# Python 中方便的调度程序

> 原文：<https://towardsdatascience.com/convenient-scheduler-in-python-2adbb57be94f>

## 用 python 调度 ETL 任务的小规模便捷方法。

![](img/7b8a716f0a89e8f8d10d9ceef3de2ca8.png)

一个永远悬而未决的任务的例子，波兰 [2021](https://private.zerowithdot.com/travelling) 。(作者供图)。

# 介绍

Python 已经成为一种通用语言。它尤其常用于数据科学中的分析和解决算法问题，但在 web 开发中也很流行。这种组合使它成为各种提取-转换-加载(ETL)任务的合理选择。

然而，这些任务中的许多都相当小，不需要大型框架，如[气流](https://airflow.apache.org/)或 [Luigi](https://luigi.readthedocs.io/en/stable/) 。当轮询一个或多个 web 页面的数据时，一个简单的 python 脚本加上 [crontab](https://docs.oracle.com/en/learn/oracle-linux-crontab/) 就足够了。尽管如此，当一个项目变得稍微大一点时，使用 cron 管理多个作业可能会变得很麻烦。同时，“小作业”的 Airflow 裸装至少需要 4GB RAM 和 2 个 CPU(此处)。考虑到 AWS 的成本，它至少是一个一直运行的 *t2.small* 实例。

有中间的吗？小到可以使用，比如说 *t2.nano* (非常便宜)并且相当“可维护”和“可扩展”？

在这篇文章中，我想和你分享一个简单的方法，它使用 python 的 [schedule](https://schedule.readthedocs.io/en/stable/examples.html) 包，并做了一些修改。

# Python 调度程序

Python [schedule](https://schedule.readthedocs.io/en/stable/examples.html) 库提供了简单的任务调度。它可以使用`pip`来安装，并且非常容易使用。不幸的是，文档没有提供在更大的项目中使用它的例子:

```
import schedule
import time

def job():
    print("I'm working...")

    # Run job every 3 second/minute/hour/day/week,
    # Starting 3 second/minute/hour/day/week from now
    schedule.every(3).seconds.do(job)
    schedule.every(3).minutes.do(job)
    schedule.every(3).hours.do(job)
    schedule.every(3).days.do(job)
    schedule.every(3).weeks.do(job)

    # Run job every minute at the 23rd second
    schedule.every().minute.at(":23").do(job)

    # Run job every hour at the 42rd minute
    schedule.every().hour.at(":42").do(job)

    # Run jobs every 5th hour, 20 minutes and 30 seconds in.
    # If current time is 02:00, first execution is at 06:20:30
    schedule.every(5).hours.at("20:30").do(job)

    # Run job every day at specific HH:MM and next HH:MM:SS
    schedule.every().day.at("10:30").do(job)
    schedule.every().day.at("10:30:42").do(job)

    # Run job on a specific day of the week
    schedule.every().monday.do(job)
    schedule.every().wednesday.at("13:15").do(job)
    schedule.every().minute.at(":17").do(job)

    while True:
        schedule.run_pending()
            time.sleep(1)
```

正如您所看到的，所有的函数都在模块级别被调用，这对于将它放在脚本中是可以的。但是，如果您有几个不同的作业，代码很快就会变得混乱，特别是如果不同的调用需要不同的参数。

换句话说，利用面向对象的方法并围绕它定义一些“架构”可能更好。

# 在项目中使用它

为了便于讨论，假设我们有一组专用的 ETL 任务，使用下面的抽象类建模:

```
from abc import ABC, abstractmethod
from typing import Any, Dict, TypeVar

E = TypeVar("ETL")

class BaseETL(ABC):
    def __init__(self, **kwargs: Dict) -> None:
        self.raw_data = None
        self.transformed_data = None

    @abstractmethod
    def extract(self, **kwargs: Dict) -> E:
        ...

    @abstractmethod
    def transform(self, **kwargs: Dict) -> E:
        ...

    @abstractmethod
    def load(self, **kwargs: Dict) -> Any:
        ...

    def run(self, **kwargs: Dict) -> None:
        self.extract(**kwargs).transform(**kwargs).load(**kwargs)
```

任何实现 ETL 过程的类都会继承这个基类。例如，`extract`方法可以获取一个网站。然后`transform`将原始 HTML 转换成数据库可接受的格式。最后，`load`会将数据保存到数据库中。按照这个顺序执行的所有方法都可以使用`run`方法包装。

现在，在定义了 ETL 类之后，我们希望通过`schedule`模块以一种很好的方式来调度它们。

# 两个示例 ETL 任务

为了简洁，在下面的例子中，让我们跳过继承，只关注`run`方法。假设他们的`extract`、`transform`和`load`方法在别处实现。

## etl.py

```
class DummyETL:  # normally DummyETL(BaseETL)
    def __init__(self, init_param: int) -> None:
        # super().__init__()  # - not needed here
        self.init_param = init_param

    def run(self, p1: int, p2: int) -> None:
        name = self.__class__.__name__
        print(f"{name}({self.init_param}, p1={p1}, p2={p1})")

class EvenDummierETL:  # same...
    def __init__(self, init_param: int) -> None:
        # super().__init__()  # - same
        self.init_param = init_param

    def run(self, p1: int) -> None:
        name = self.__class__.__name__
        print(f"{name}({self.init_param}, p1={p1})")
```

例如，构造函数的参数可以指定用于抓取的页面的 URL。方法的参数可以用来传递秘密。

现在，我们已经定义了 ETL 类，让我们创建一个单独的*注册中心*来将流程与某种时间表关联起来。

## 注册表. py

```
import schedule

from etl import DummyETL, EvenDummierETL

def get_registry():
    dummy_etl = DummyETL(init_param=13)
    dummier_etl = EvenDummierETL(init_param=15)

    return [
        (dummy_etl, schedule.every(1).seconds),
        (dummier_etl, schedule.every(1).minutes.at(":05")),
    ]
```

`get_registry`功能是定义时间表的地方。尽管参数的值是硬编码的，但是您可以考虑函数从配置文件中加载它们的情况。无论哪种方式，它都会返回一个元组列表，这些元组匹配带有`Job`的 ETL 对象(来自`schedule`)。注意，这是我们的**约定。**作业尚未与任何特定的`Scheduler`相关联(同样来自`schedule`)。然而，公约允许我们在项目的任何其他部分这样做。我们不必将它们与模块级对象绑定，如文档示例所示。

# 我们基于调度程序的调度程序

最后，让我们创建一个新的类来激活整个机制。

## scheduler.py

```
import time
from typing import Dict, List, Tuple, TypeVar

from schedule import Job, Scheduler

from etl import DummyETL, EvenDummierETL
from etl import E  # we could do so from e.g. etl.base

S = TypeVar("Scheduler")

class TaskScheduler:
    def __init__(self, registry: List[Tuple[E, Job]]) -> None:
        self.scheduler = Scheduler()
        self.registry = []

        for task, job in registry:
            self.registry.append(task)
            self.scheduler.jobs.append(job)

    def register(self, run_params: Dict) -> S:
        jobs = self.scheduler.get_jobs()
        for task, job in zip(self.registry, jobs):
            params = run_params.get(task.__class__.__name__)
            job.do(task.run, **params)

        return self

    def run(self, polling_seconds: int) -> None:
        while True:
            time.sleep(polling_seconds)
            self.scheduler.run_pending()
```

我们的`TaskScheduler`使用组合来创建一个单独的`Scheduler`实例，并向其中添加先前注册的作业。尽管不是强制的，我们使用`typing`来给出一个强有力的提示，告诉构造函数应该提供什么来正确地注册作业。然后，`register`方法是一个提供绑定的独立方法。最后，同样重要的是，`run`激活机器。

使用此实现的脚本如下所示:

## run.py

```
from registry import get_registry
from scheduler import TaskScheduler

if __name__ == "__main__":
    run_params = {
        "DummyETL": dict(p1=1, p2=2),  # e.g. from env vars
        "EvenDummierETL": dict(p1=3),
    }

    registry = get_registry()  # e.g. from script's args or config
    task_scheduler = TaskScheduler(registry).register(run_params)
    task_scheduler.run()
```

这个解决方案最弱的一点可能是在`run_params`字典中使用`__class__.__name__`作为键的约定。然而，考虑到这种方法的简单性，它可能是可以的，特别是如果这些参数是在运行时定义的。有很多选择，其中之一是创建一个额外的抽象层，比如像`DummyTask`这样的对象，作为 ETL 对象和注册中心之间的桥梁。

# 任务调度的另一种方法

回到`TaskScheduler`，我们也可以通过继承来定义它，而不是像以前一样通过组合来定义。这将意味着扩展`schedule`的原生`Scheduler`类的功能。在这种情况下，`TaskScheduler`如下所示:

```
class TaskScheduler(Scheduler):  # <- here
    def __init__(self, registry: List[Tuple[E, Job]]) -> None:
        super().__init__()  # <- here
        self.registry = []

        for task, job in registry:
            self.registry.append(task)
            self.jobs.append(job)  # <- here

    def register(self, run_params: Dict) -> S:
        jobs = self.get_jobs()  # <- here
        for task, job in zip(self.registry, jobs):
            params = run_params.get(task.__class__.__name__)
            job.do(task.run, **params)

        return self

    def run(self, polling_seconds: int) -> None:
        while True:
            time.sleep(polling_seconds)
            self.run_pending()  # <- and here
```

你决定哪种方式更好，如果有的话；).

# 结论

在这篇简短的文章中，我们展示了如何扩展简单的`schedule`模块来创建一个小型 ETL 工作机器。最重要的是，这种方法允许在一个小项目中更好地组织代码，而不必去找大炮。

*最初发表于*[*https://zerowithdot.com*](https://zerowithdot.com/scheduler-in-python/)*。*