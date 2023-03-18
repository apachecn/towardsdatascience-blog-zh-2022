# 理解 Python 中异步编程的 async/await with asyncio

> 原文：<https://towardsdatascience.com/understand-async-await-with-asyncio-for-asynchronous-programming-in-python-e0bc4d25808e>

## 用一种新的方式编写异步代码

![](img/92993285d88e5d9a17d6aff6aedd279e.png)

图片由[帕特里克·亨德利](https://unsplash.com/photos/vuOqMK4NWc8)在 Unsplash 拍摄

大多数 Python 开发人员可能只使用过 Python 中的同步代码，甚至是一些资深的 Python 爱好者。然而，如果你是一名数据科学家，你可能使用过*多处理*库来并行运行一些计算。如果你是一名 web 开发人员，你可能有机会通过*线程*实现并发。*多处理*和*线程*都是 Python 中的高级概念，都有自己特定的应用领域。

除了*多处理*和*线程*之外，Python 的并发家族中还有一个相对较新的成员——*asyncio*，这是一个使用`async` / `await`语法编写并发代码的库。与*线程*类似， *asyncio* 适用于实际中非常常见的 io 绑定任务。在本帖中，我们将介绍 *asyncio* 的基本概念，并演示如何使用这个新库来编写异步代码。

## CPU 受限和 IO 受限的任务

在我们开始使用 asyncio 库之前，有两个概念我们应该弄清楚，因为它们决定了应该使用哪个库来解决您的特定问题。

受 CPU 限制的任务大部分时间都在用 CPU 进行繁重的计算。如果你是一名数据科学家，需要通过处理大量数据来训练一些机器学习模型，那么这是一项 CPU 密集型任务。如果是这种情况，你应该使用 [*多处理*](https://docs.python.org/3/library/multiprocessing.html) 来并行运行你的作业，充分利用你的 CPU。

另一方面，IO 绑定的任务花费大部分时间等待 IO 响应，这些响应可能是来自网页、数据库或磁盘的响应。对于需要从 API 或数据库获取数据的 web 开发，这是一个 IO 绑定的任务，可以通过 [*线程*](https://levelup.gitconnected.com/how-to-write-concurrent-python-code-with-multithreading-b24dec228c43) 或 [*asyncio*](https://docs.python.org/3/library/asyncio.html) 实现并发，以最大限度地减少外部资源的等待时间。

## 线程与异步

好了，我们知道*线程*和*异步*都适合 io 绑定的任务，但是它们有什么区别呢？

首先，乍看之下你可能觉得难以置信，*线程*使用多线程，而 *asyncio* 只用一个线程。对于*线程化*，更容易理解，因为线程轮流运行代码，从而实现并发。但是如何用单线程实现并发呢？

嗯，*线程化*通过抢占式多任务处理实现并发，这意味着我们无法确定何时在哪个线程中运行哪个代码。决定哪个代码应该在哪个线程中运行的是操作系统。操作系统可以在线程之间的任何点切换控制。这就是为什么我们经常看到*线程*的随机结果。[如果你想了解更多关于*线程*的知识，这篇文章](https://levelup.gitconnected.com/how-to-write-concurrent-python-code-with-multithreading-b24dec228c43)会很有帮助。

另一方面， *asyncio* 通过协作多任务实现并发。我们可以决定代码的哪一部分可以等待，从而控制切换到运行代码的其他部分。这些任务需要协作并宣布何时控制将被切换出去。所有这些都是在单线程中用`await`命令完成的。现在它可能看起来难以捉摸，但当我们稍后看到代码时，它会变得更加清晰。

## 什么是协程？

这是 asyncio 中一个奇特的名字。很难解释它是什么。许多教程根本没有解释这个概念，只是用一些代码向你展示它是什么。不过，我们先试着了解一下是什么。

Python 中协程的[定义是:](https://docs.python.org/3.10/glossary.html#term-coroutine)

> 协程是子程序的一种更一般化的形式。子程序在一点进入，在另一点退出。协程可以在许多不同的点进入、退出和恢复。

当你第一次看到它的时候，这可能看起来很奇怪。然而，当你和*阿辛西奥*合作越来越多的时候，就会越来越有意义。

在这个定义中，我们可以把子程序理解为函数，尽管两者之间有差异。通常，一个函数只在被调用时进入和退出一次。但是 Python 中有一个特殊的函数叫做[生成器](/demystify-iterators-and-generators-in-python-f21878c9897)，可以多次进出。

协程的行为很像生成器。实际上，在旧版本的 Python 中，协程是由生成器定义的。这些协程被称为基于生成器的协程。然而，协程现在已经成为 Python 的一个原生特性，可以用新的`async def`语法来定义。尽管基于生成器的协程现在已经被弃用，但它们的历史和存在可以帮助我们理解什么是协程，以及如何在代码的不同部分之间切换或产生控制。如果你想了解更多关于 Python 中协程的历史和规范， [PEP 492](https://peps.python.org/pep-0492/) 是一个很好的参考。然而，对于初学者来说，阅读和理解可能并不容易。

好了，现在抽象的概念已经够多了。如果你不知何故迷路了，不能理解所有的概念，没关系。随着时间的推移，当你用 asyncio 库编写和读取越来越多的异步代码时，它们会变得更加清晰。

## 定义协程函数

既然已经介绍了基本概念，我们可以开始编写我们的第一个协程函数了:

```
async def coro_func():
    print("Hello, asyncio!")
```

`coro_func()`是一个协程函数，当调用它时，它将返回一个协程对象:

```
coro_obj = coro_func()

type(coro_obj)
# coroutine
```

注意，术语*协程*可以指协程函数或协程对象，这取决于上下文。

您可能已经注意到，当调用协程函数时，不会调用`print`函数。如果您使用过生成器，您不会感到惊讶，因为它的行为与生成器功能相似:

```
def gen_func():
    yield "Hello, generator!"

generator = gen_func()
type(generator)
# generator
```

为了在生成器中运行代码，您需要迭代它。例如，您可以使用`next`函数来迭代它:

```
next(generator)
# 'Hello, generator!'
```

同样，要运行协程函数中定义的代码，你需要 ***等待*** 它。但是，您不能像迭代生成器一样等待它。一个协程只能在由`async def`语法定义的另一个协程中等待:

```
async def coro_func():
    print("Hello, asyncio!")

async def main():
    print("In the entrypoint coroutine.")
    await coro_func()
```

现在的问题是我们如何运行`main()`协程函数。显然，我们不能把它放在另一个协程函数中等待它。

对于顶级入口点协程函数，通常命名为`main()`，我们需要使用`asyncio.run()`来运行它:

```
import asyncio

async def coro_func():
    print("Hello, asyncio!")

async def main():
    print("In the entrypoint coroutine.")
    await coro_func()

asyncio.run(main())
# In the entrypoint coroutine.
# Hello, asyncio!
```

注意，我们需要在这里导入内置的 *asyncio* 库。

在引擎盖下，它由一个叫做[事件循环](https://docs.python.org/3/library/asyncio-eventloop.html)的东西处理。然而，有了现代 Python，你再也不需要担心这些细节了。

## 在协程函数中返回值

我们可以在协程函数中返回值。该值通过`await`命令返回，并可分配给一个变量:

```
import asyncio

async def coro_func():
    return "Hello, asyncio!"

async def main():
    print("In the entrypoint coroutine.")
    result = await coro_func()
    print(result)

asyncio.run(main())
# In the entrypoint coroutine.
# Hello, asyncio!
```

## 同时运行多个协程

在你的代码中只有一个协程并不有趣，也没有用。当有多个协同程序应该并发运行时，它们会大放异彩。

让我们首先看一个协程等待错误的例子:

```
import asyncio
from datetime import datetime

async def async_sleep(num):
    print(f"Sleeping {num} seconds.")
    await asyncio.sleep(num)

async def main():
    start = datetime.now()

    for i in range(1, 4):
        await async_sleep(i)

    duration = datetime.now() - start
    print(f"Took {duration.total_seconds():.2f} seconds.")

asyncio.run(main())
# Sleeping 1 seconds.
# Sleeping 2 seconds.
# Sleeping 3 seconds.
# Took 6.00 seconds.
```

首先，注意我们需要在协程函数中使用`asyncio.sleep()`函数来模拟 IO 阻塞时间。

其次，创建的三个协程对象被逐个等待。因为当等待的协程对象已经完成时，控制只被处理到下一行代码(这里是下一个循环)，所以这三个协程实际上是一个接一个等待的。因此，运行代码需要 6 秒钟，这与同步运行代码是一样的。

我们应该用`async.gather()` 函数同时运行多个协程。

`async.gather()`用于同时运行多个 awaitables。顾名思义，一个[可用的](https://docs.python.org/3.10/library/asyncio-task.html#awaitables)是可以通过`await`命令等待的东西。它可以是一个协程、一个任务、一个未来，或者任何实现了`__await__()`魔法方法的东西。

我们来看看`async.gather()`的用法:

```
import asyncio
from datetime import datetime

async def async_sleep(num):
    print(f"Sleeping {num} seconds.")
    await asyncio.sleep(num)

async def main():
    start = datetime.now()

    coro_objs = []
    for i in range(1, 4):
        coro_objs.append(async_sleep(i))

    await asyncio.gather(*coro_objs)

    duration = datetime.now() - start
    print(f"Took {duration.total_seconds():.2f} seconds.")

asyncio.run(main())
# Sleeping 1 seconds.
# Sleeping 2 seconds.
# Sleeping 3 seconds.
# Took 3.00 seconds.
```

注意，我们需要解包`async.gather()`函数的 awaitables 列表。

这一次协程对象是并发运行的，代码只花了 3 秒钟。

如果你检查`asyncio.gather()`的返回类型，你会看到它是一个 [Future](https://docs.python.org/3.10/library/asyncio-future.html#asyncio.Future) 对象。未来对象是一种特殊的数据结构，表示某些工作在其他地方完成，可能已经完成，也可能尚未完成。当等待未来对象时，会发生三种情况:

*   当 future 已成功解决(意味着底层工作已成功完成)时，它将立即返回返回值(如果可用)。
*   当未来未成功解决并且引发异常时，该异常将传播到调用方。
*   当未来还没有解决时，代码会一直等到它解决。

## 一个更实用的 async with 和 aiohttp 的例子

上面我们刚刚写了一些伪代码来演示 *asyncio* 的基础。现在让我们编写一些更实用的代码来进一步演示 *asyncio* 的用法。

我们将编写一些代码来同时从对一些网页的请求中获取响应，这是一个经典的 IO 绑定任务，正如本文开头所解释的。

请注意，我们不能使用我们熟悉的*请求*库来获得来自网页的响应。这是因为*请求*库不支持 *asynico* 库。这实际上是 asynico 库的一个主要限制，因为许多经典的 Python 库仍然不支持 asyncio 库。然而，随着时间的推移，这将会变得更好，更多的异步库将会出现。

为了解决*请求*库的问题，我们需要使用 *aiohttp* 库，它是为异步 http 请求(以及更多请求)而设计的。

我们需要首先安装 aiohttp，因为它仍然是一个外部库:

```
pip install aiohttp
```

强烈建议在[虚拟环境](https://lynn-kwong.medium.com/how-to-create-virtual-environments-with-venv-and-conda-in-python-31814c0a8ec2)中安装新的库，这样它们就不会影响系统库，你也不会有兼容性问题。

这是使用 *aiohttp* 库执行 http 请求的代码，它也大量使用了`async with`语法:

```
import asyncio
import aiohttp

async def scrape_page(session, url):
    print(f"Scraping {url}")
    async with session.get(url) as resp:
        return len(await resp.text())

async def main():
    urls = [
        "https://www.superdataminer.com/posts/66cff907ce8e",
        "https://www.superdataminer.com/posts/f21878c9897",
        "https://www.superdataminer.com/posts/b24dec228c43"
    ]

    coro_objs = []

    async with aiohttp.ClientSession() as session:
        for url in urls:
            coro_objs.append(
                scrape_page(session, url)
            )

        results = await asyncio.gather(*coro_objs)

    for url, length in zip(urls, results):
        print(f"{url} -> {length}")

asyncio.run(main())
# Scraping https://www.superdataminer.com/posts/66cff907ce8e
# Scraping https://www.superdataminer.com/posts/f21878c9897
# Scraping https://www.superdataminer.com/posts/b24dec228c43
# https://www.superdataminer.com/posts/66cff907ce8e -> 12873
# https://www.superdataminer.com/posts/f21878c9897 -> 12809
# https://www.superdataminer.com/posts/b24dec228c43 -> 12920
```

`async with`语句使得在进入或退出上下文时执行异步调用成为可能。在引擎盖下，它是通过`async def __aenter__()`和`async def __aexit__()`魔法方法实现的，这是一个相当高级的话题。有兴趣的话，先了解一下 Python 中的[正则上下文管理器](/understand-context-managers-in-python-and-learn-to-use-them-in-unit-tests-66cff907ce8e)的一些知识。而在那之后，[这个帖子](https://bbc.github.io/cloudfit-public-docs/asyncio/asyncio-part-3)如果你想更深一层的话，可以是一个很好的参考。然而，通常您不需要深入研究，除非您想创建自己的异步上下文管理器。

除了`async with`语法， *aiohttp* 库的用法实际上与*请求*库非常相似。

在这篇文章中，我们介绍了异步编程的基本概念。用简单易懂的例子介绍了带有`async/await`、`asyncio.run()`和`asyncio.gather()`语句的 *asyncio* 库的基本用法。有了这些知识，你将能够使用 *asyncio* 库读写基本的异步代码，并且能够更舒适地使用异步 API 框架，如 [FastAPI](https://fastapi.tiangolo.com/) 。

## 相关文章:

*   [解开 Python 中迭代器和生成器的神秘面纱](https://www.superdataminer.com/posts/f21878c9897)
*   [如何用多线程编写并发 Python 代码](https://www.superdataminer.com/posts/b24dec228c43)