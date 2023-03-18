# Python:面向对象还是面向编程？

> 原文：<https://towardsdatascience.com/python-to-oop-or-to-fp-13ac79a43b16>

## 这是一个问题

![](img/6a200bf69efb92c512297266e87c01ae.png)

苏珊·霍尔特·辛普森在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄的照片

程序员永远无法在任何事情上达成一致，但迄今为止，不断困扰互联网的最大争论之一是面向对象编程(OOP)和函数式编程(FP)之间的斗争。

提醒一下，OOP 围绕着将所有的业务逻辑和数据包装在类中，然后可以创建共享相同功能的对象。它还包括继承和多态等概念，这使得拥有功能相似但略有不同的类变得更加容易。

通常用来演示 OOP 的语言是 Java。在 Java 中，所有东西都必须包装在一个类中，包括程序的主执行循环。

另一方面，函数式编程更关心——你猜对了——函数。在函数式编程中，数据通常通过管道从一个函数传递到另一个函数，每个函数对数据执行一个操作。如果给定相同的输入，函数通常被设计成产生完全相同的输出。

最流行的函数式编程语言是 Clojure、Elixir 和 Haskell。

# 但是 Python 呢？

Python 是一个有趣的例子。它共享了面向对象语言的许多共同特性，允许您创建类并从超类继承，但它也具有您通常在函数式语言中看到的功能。你可以在程序的主体中定义函数，函数也是一等公民，这意味着你可以把它们作为对象传递。

事实是，Python 非常灵活。如果您来自 Java，并且想用纯面向对象的风格编写所有东西，那么您将能够完成您想要的大部分事情。如果您以前是 Clojure 开发人员，用 Python 复制 FP 模式也不会有太大的困难。

然而，Python 的美妙之处在于，你不局限于任何一种做事方式。您可以使用这两种范例的特性来创建可读的、可扩展的代码，这将保持您的代码库的可维护性，即使它在增长。

下面是三个用 OOP、FP 和两者混合编写的相同(非常简单)程序的例子。我将强调每种方法的优点和缺点，这将为您设计下一个 Python 项目打下良好的基础。

# 该计划

用于演示的程序非常简单——它创建了两只动物(一只狗和一条鱼),并让它们执行一些非常简单的动作。在这个例子中，动作只是记录到`stdout`，但是它们显然可以做得更多。

## OOP 示例

```
from abc import ABC, abstractmethod

class Logger(ABC):
    @abstractmethod
    def log(self, message: str):
        ...

class MyLogger(Logger):
    def __init__(self, name: str):
        self.name = name

    def log(self, message: str):
        print(f'{self.name}: {message}')

class Animal:
    def __init__(self, name: str, logger: Logger):
        self.name = name
        self.logger = logger

    def speak(self):
        self.logger.log('Speaking')
        ...

class Dog(Animal):
    def speak(self):
        self.logger.log('Woof!')
        ...

    def run(self):
        self.logger.log('Running')
        ...

class Fish(Animal):
    ...

class App:
    @staticmethod
    def run():
        fido = Dog(name='Fido', logger=MyLogger('Fido'))
        goldie = Fish(name='Goldie', logger=MyLogger('Goldie'))

        fido.speak()
        fido.run()

        goldie.speak()

if __name__ == '__main__':
    App.run()

# Fido: Woof!
# Fido: Running
# Goldie: Speaking
```

如您所见，代码创建了一个用于将事件记录到`stdout`的`MyLogger`类，一个`Animal`基类，然后是用于更具体动物的`Dog`和`Fish`类。

为了更好地遵循 OOP 范例，它还定义了一个具有运行程序的单一方法`run`的`App`类。

OOP 和继承的好处是我们不必在`Fish`类上定义一个`speak`方法，它仍然能够说话。

然而，如果我们想要有更多可以运行的动物，我们就必须在`Animal`和`Dog`之间引入一个定义`run`方法的`RunningAnimal`类，并且可能为`Fish`引入一个类似的`SwimmingAnimal`类，但是之后我们的层次结构开始变得越来越复杂。

另外，`MyLogger`和`App`类在这里几乎没有用。每个都只做一件事，实际上使代码可读性稍差。这些最好放在一个`log`和一个`main`(或`run`)函数中。

我们还必须创建一个纯粹的`Logger`抽象基类，这样代码就可以正确地进行类型提示，并允许我们 API 的用户传入其他日志程序，如果他们想登录到`stdout`之外的地方，或者如果他们想用不同的格式登录。

## FP 示例

只是提醒一下——我对 FP 的熟悉程度不如 OOP，所以这可能不是实现这种行为的最像 FP 的方式，但这是我要做的。

```
import functools
from typing import Callable

Logger = Callable[[str], None]

def log(message: str, name: str):
    print(f'{name}: {message}')

def bark(
    name: str,
    log_fn: Logger,
) -> (str, Logger):
    log_fn('Woof!')
    return name, log_fn

def run(
    name: str,
    log_fn: Logger,
) -> (str, Logger):
    log_fn('Running')
    return name, log_fn

def speak(
    name: str,
    log_fn: Logger,
) -> (str, Logger):
    log_fn('Speaking')
    return name, log_fn

def main():
    run(
        *bark(
            'Fido',
            functools.partial(log, name='Fido'),
        ),
    )

    speak(
        'Goldie',
        functools.partial(log, name='Goldie'),
    )

if __name__ == '__main__':
    main()

# Fido: Woof!
# Fido: Running
# Goldie: Speaking
```

很快，我们可以看到我们的`Logger`类已经成为`Callable[[str], None]`的一个方便的类型别名。我们还定义了一个`log`函数来处理我们的打印。我们没有为我们的动物定义类，而是简单地定义了以动物的名字和一个`Logger`函数命名的函数。

您会注意到，`run`、`speak`和`bark`函数也都返回它们的名称和日志记录函数参数，以便将它们组合到管道中，就像我们对 Fido 的`run`和`bark`所做的那样。

我们还将我们的逻辑移到了一个`main`函数中，消除了仅仅为了运行我们的程序而定义整个类的需要。

为了避免我们的`log`函数与`Logger`类型不匹配的事实，我们使用`functools.partial`来创建一个匹配的部分函数。这允许我们用我们喜欢的任何东西来替换我们的记录器，只要我们可以使用一个部分函数来减少它，以便它匹配我们的`Logger`类型。

然而，由于我们没有将数据封装在任何东西中，如果我们想给我们的动物添加更多的属性，我们可能不得不开始使用`dict`对象来表示它们并传递它们，但这样总是会担心字典创建不正确，从而缺少我们的一个函数所依赖的键。

为了避开*和*，我们需要为我们的动物创建初始化函数，这时代码又变得越来越乱。

## 两者都有点

那么，如果我们将一点 OOP 和一点 FP 结合起来，会发生什么呢？我将引入一些 Pythonic 代码来脱离传统的 OOP 和 FP 范例，并希望使代码更加简洁易读。

```
from dataclasses import dataclass
from functools import partial
from typing import Callable

Logger = Callable[[str], None]

def log(message: str, name: str):
    print(f'{name}: {message}')

@dataclass
class Animal:
    name: str
    log: Logger

    def speak(self):
        self.log('Speaking')

@dataclass
class Dog(Animal):
    breed: str = 'Labrador'

    def speak(self):
        self.log('Woof!')

    def run(self):
        self.log('Running')

@dataclass
class Fish(Animal):
    ...

def main():
    fido = Dog('Fido', partial(log, name='Fido'))
    goldie = Fish('Goldie', partial(log, name='Goldie'))

    fido.speak()
    fido.run()

    goldie.speak()

if __name__ == '__main__':
    main()

# Fido: Woof!
# Fido: Running
# Goldie: Speaking
```

在这个例子中，我使用 python`dataclasses`模块来避免为我的类编写构造函数。这不仅减少了我需要编写的一些代码，而且如果我需要的话，还可以更容易地添加新的属性。

类似于 OOP 的例子，我们有一个带有`Dog`和`Fish`子类的`Animal`基类。然而，就像在 FP 示例中一样，我使用了`Logger`类型别名和`functools.partial`来为动物创建记录器。Python 支持作为一等公民的函数使得这变得更加容易。

还有，`main`函数只是一个函数。我永远不会明白为什么 Java 是 Java 的样子。

# 在生产中混合 OOP 和 FP

好的，我承认这个例子非常简单，虽然它为我们的讨论提供了一个很好的起点，但是我现在想给你们一个例子来说明这些概念是如何在生产中使用的，我将使用我最喜欢的两个 Python 库: [FastAPI](https://fastapi.tiangolo.com/) 和 [Pydantic](https://pydantic-docs.helpmanual.io/) 。FastAPI 是 Python 的轻量级 API 框架，Pydantic 是数据验证和设置管理库。

我不打算详细介绍这些库，但是 Pydantic 有效地允许您使用 Python 类定义数据结构，然后验证传入的数据并通过对象属性访问它。这意味着您不会遇到使用字典带来的问题，并且您总是知道您的数据是您期望的格式。

FastAPI 允许您将 API 路由定义为函数，用一个装饰器(这是一个非常类似 FP 的概念)包装每一个路由，以封装您的逻辑。

下面是如何使用它的一个例子。同样，这是一个简单的示例，但它相当能代表您在生产中可能看到的情况。

```
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Baz(BaseModel):
    qux: int

class Foo(BaseModel):
    bar: str
    baz: Baz

@app.get('/foo')
async def get_foo(name: str, age: int) -> Foo:
    ...  # Some logic here
    return Foo(
        bar=name,
        baz=Baz(qux=age),
    )

# GET /foo?name=John&age=42
# {
#   "bar": "John",
#   "baz": {
#     "qux": 42
#   }
# }
```

如您所见，FastAPI 使用 Pydantic 的能力将嵌套对象转换为 JSON，从而为我们的端点创建一个 JSON 响应。`app.get`装饰器还向`app`对象注册了我们的`get_foo`函数，允许我们向`/foo`端点发出`GET`请求。

我希望这篇文章对你有所帮助。我很想听听你们的想法，以及你们在编写 Python 时倾向于哪种范式。

显然，这并不是 Python 中结合 FP 和 OOP 的唯一方式，使用这种结合可以实现和改进很多设计模式。

我将在未来写下这些，通过在[媒体](https://medium.isaacharrisholt.com/)上跟随我，你将不会错过。我也在 Twitter 上发关于 Python 和我当前项目的推文，并且(最近)也在乳齿象上发关于它们的帖子。

我相信我们很快会再见的！

-艾萨克