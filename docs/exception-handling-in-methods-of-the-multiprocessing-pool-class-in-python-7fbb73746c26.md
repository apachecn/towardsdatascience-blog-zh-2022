# Python 中多处理池类的方法中的异常处理

> 原文：<https://towardsdatascience.com/exception-handling-in-methods-of-the-multiprocessing-pool-class-in-python-7fbb73746c26>

## 使用 map、imap 和 imap_unordered 方法

![](img/fac3fb35185bc1eb96226666fdb0285e.png)

由 [Marek Piwnicki](https://unsplash.com/@marekpiwnicki?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄的照片

## 介绍

处理大数据时，通常需要并行计算。在 python 中，标准的 [**多处理**](https://docs.python.org/3/library/multiprocessing.html) 模块通常用于需要大量计算资源的任务。在 DS 中，我们必须不断地解决容易并行化的问题。示例可以是[引导](https://medium.com/towards-data-science/bootstrap-and-statistical-inference-in-python-a06d098a8bfd)、多重预测(多个示例的模型预测)、[数据预处理](/parallelization-w-multiprocessing-in-python-bd2fc234f516)等。

在本文中，我想谈谈在 python 中使用多重处理`[Pool](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing.pool)`类时需要考虑的一些有趣而重要的事情:

*   Pool 类的方法中的异常处理
*   python 中悬挂函数的处理
*   进程使用的内存限制(仅适用于 Unix 系统)

我将在 OS Ubuntu 20.04 上使用 3.9 版本的 Python。

所以让我们开始吧！

## Pool 类的方法中的异常处理

在我的实践中，我经常不得不使用**多重处理**模块。把电脑的所有能力都用上，把处理器的汁液都挤出来，感觉很不错吧？让我们想象一下，你写了非常复杂的代码，你的计算量非常大，以至于你决定在晚上运行它们，希望醒来后能看到你工作的精彩结果。所以，这就是我们美丽的函数(假设我们忘记了不可能除以一个零，有谁没发生呢？)

早上你会看到什么？我想你会非常沮丧，因为很明显，你会看到下面的追溯:

```
multiprocessing.pool.RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.9/multiprocessing/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
  File "/usr/lib/python3.9/multiprocessing/pool.py", line 48, in mapstar
    return list(map(*args))
  File "/home/PycharmProjects/myproject/main.py", line 9, in my_awesome_foo
    1 / 0
ZeroDivisionError: division by zero
"""The above exception was the direct cause of the following exception:Traceback (most recent call last):
  File "/home/PycharmProjects/myproject/main.py", line 19, in <module>
    result = p.map(my_awesome_foo, tasks)
  File "/usr/lib/python3.9/multiprocessing/pool.py", line 364, in map
    return self._map_async(func, iterable, mapstar, chunksize).get()
  File "/usr/lib/python3.9/multiprocessing/pool.py", line 771, in get
    raise self._value
ZeroDivisionError: division by zero
```

有人会说，这不奇怪，这应该发生，而且绝对正确。但是让我们稍微修改一下我们的代码，试着更详细地了解当一个进程中发生异常时，池内部发生了什么。我们将在我们的功能中添加打印消息的功能，该过程已经开始，我将完成这项工作。使用**多重处理**模块的函数`[current_procces().name](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.current_process)`可以获得进程名称。

```
Process ForkPoolWorker-1 started working on task 0
Process ForkPoolWorker-2 started working on task 1
Process ForkPoolWorker-4 started working on task 3
Process ForkPoolWorker-3 started working on task 2
Process ForkPoolWorker-1 started working on task 4
Process ForkPoolWorker-2 ended working on task 1
Process ForkPoolWorker-2 started working on task 5
Process ForkPoolWorker-4 ended working on task 3
Process ForkPoolWorker-4 started working on task 6
Process ForkPoolWorker-2 ended working on task 5
Process ForkPoolWorker-2 started working on task 7
Process ForkPoolWorker-1 ended working on task 4
Process ForkPoolWorker-1 started working on task 8
Process ForkPoolWorker-4 ended working on task 6
Process ForkPoolWorker-4 started working on task 9
Process ForkPoolWorker-3 ended working on task 2
Process ForkPoolWorker-1 ended working on task 8
Process ForkPoolWorker-4 ended working on task 9
Process ForkPoolWorker-2 ended working on task 7
multiprocessing.pool.RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.9/multiprocessing/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
  File "/usr/lib/python3.9/multiprocessing/pool.py", line 48, in mapstar
    return list(map(*args))
  File "/home/PycharmProjects/myproject/main.py", line 9, in my_awesome_foo
    1 / 0
ZeroDivisionError: division by zero
"""The above exception was the direct cause of the following exception:Traceback (most recent call last):
  File "/home/PycharmProjects/myproject/main.py", line 19, in <module>
    result = p.map(my_awesome_foo, tasks)
  File "/usr/lib/python3.9/multiprocessing/pool.py", line 364, in map
    return self._map_async(func, iterable, mapstar, chunksize).get()
  File "/usr/lib/python3.9/multiprocessing/pool.py", line 771, in get
    raise self._value
ZeroDivisionError: division by zeroProcess finished with exit code 1
```

**嘭！**

所以我们的函数在第一次迭代时捕捉到了一个异常，但是我们看到了什么呢？我们看到所有的过程都在我们看到出错之前开始并成功完成了它们的工作。事实上，这意味着你的程序真的会通宵工作，但是最后，它仍然以一个错误结束，你不会得到任何结果。很遗憾，不是吗？

这个例子清楚地显示了在使用 Pool 类的 map 方法时处理异常是多么重要。那么`imap`和`imap_unordered`方法呢？这里我们看到更多可预测的行为:

```
Process ForkPoolWorker-1 started working on task 0
Process ForkPoolWorker-3 started working on task 2
Process ForkPoolWorker-2 started working on task 1
Process ForkPoolWorker-4 started working on task 3
Process ForkPoolWorker-1 started working on task 4
multiprocessing.pool.RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.9/multiprocessing/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
  File "/home/PycharmProjects/myproject/main.py", line 8, in my_awesome_foo
    1 / 0
ZeroDivisionError: division by zero
"""The above exception was the direct cause of the following exception:Traceback (most recent call last):
  File "/home/PycharmProjects/myproject/main.py", line 21, in <module>
    result = list(p.imap(my_awesome_foo, tasks))
  File "/usr/lib/python3.9/multiprocessing/pool.py", line 870, in next
    raise value
ZeroDivisionError: division by zeroProcess finished with exit code 1
```

不幸的是，正确处理`map`方法中出现的异常超出了本文的范围。有像 [**pebble**](https://github.com/noxdafox/pebble) 这样的库可以让你这么做。

下面是`imap`方法的一个异常处理选项的例子(也适用于`imap_unordered`)

```
Process ForkPoolWorker-1 started working on task 0
Process ForkPoolWorker-2 started working on task 1
Process ForkPoolWorker-4 started working on task 3
Process ForkPoolWorker-3 started working on task 2
Process ForkPoolWorker-1 started working on task 4
Process ForkPoolWorker-4 ended working on task 3
Process ForkPoolWorker-4 started working on task 5
Process ForkPoolWorker-2 ended working on task 1
Process ForkPoolWorker-3 ended working on task 2
Process ForkPoolWorker-2 started working on task 6
Process ForkPoolWorker-3 started working on task 7
Process ForkPoolWorker-1 ended working on task 4
Process ForkPoolWorker-1 started working on task 8
Process ForkPoolWorker-4 ended working on task 5
Process ForkPoolWorker-4 started working on task 9
Process ForkPoolWorker-2 ended working on task 6
Process ForkPoolWorker-3 ended working on task 7
Process ForkPoolWorker-1 ended working on task 8
Process ForkPoolWorker-4 ended working on task 9
time took: 3.0
[ZeroDivisionError('division by zero'), 1, 2, 3, 4, 5, 6, 7, 8, 9]Process finished with exit code 0
```

然后，您可以打印出完整的回溯，看看哪里出错了:

```
Traceback (most recent call last):
  File "/usr/lib/python3.9/multiprocessing/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
  File "/home/PycharmProjects/myproject/main.py", line 9, in my_awesome_foo
    1 / 0
ZeroDivisionError: division by zero
"""The above exception was the direct cause of the following exception:Traceback (most recent call last):
  File "/home/PycharmProjects/myproject/main.py", line 23, in <module>
    result.append(next(iterator))
  File "/usr/lib/python3.9/multiprocessing/pool.py", line 870, in next
    raise value
ZeroDivisionError: division by zero
```

因此，我们成功地捕获了异常，我们的池完成了它的工作，并给了我们结果。此外，我们可以打印整个异常堆栈，并查看代码中发生错误的地方。

## Python 中悬挂函数的处理

让我们改变我们美丽的功能:

对于 *n=0* ，我们的函数休眠 *5* 秒，对于所有其他 *n* ，休眠 *1* 秒。现在想象一下，举例来说，不是 5 秒，而是 5 小时。或者更糟，对于一些输入数据，你的函数陷入了一个无限循环。我们不想永远等下去，不是吗？那么在这种情况下该怎么办呢？下面是针对`imap`方法的 python 文档摘录:

> 同样，如果 *chunksize* 为`1`，那么`[imap()](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.imap)`方法返回的迭代器的`next()`方法有一个可选的*超时*参数:如果在*超时*秒内不能返回结果，`*next(timeout)*`将引发`[multiprocessing.TimeoutError](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.TimeoutError)`。

因此，让我们尝试使用文档中描述的带有超时参数的迭代器`next()`方法。在前一章中，我们学习了如何处理错误，理论上，我们应该正确处理 [**TimeoutError**](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.TimeoutError) :

这次我们应该看什么？

```
Process ForkPoolWorker-1 started working on task 0
Process ForkPoolWorker-2 started working on task 1
Process ForkPoolWorker-3 started working on task 2
Process ForkPoolWorker-4 started working on task 3
Process ForkPoolWorker-2 ended working on task 1
Process ForkPoolWorker-3 ended working on task 2
Process ForkPoolWorker-4 ended working on task 3
Process ForkPoolWorker-2 started working on task 4
Process ForkPoolWorker-3 started working on task 5
Process ForkPoolWorker-4 started working on task 6
Process ForkPoolWorker-2 ended working on task 4
Process ForkPoolWorker-3 ended working on task 5
Process ForkPoolWorker-2 started working on task 7Process ForkPoolWorker-4 ended working on task 6
Process ForkPoolWorker-3 started working on task 8
Process ForkPoolWorker-4 started working on task 9
Process ForkPoolWorker-2 ended working on task 7
Process ForkPoolWorker-4 ended working on task 9
Process ForkPoolWorker-3 ended working on task 8
Process ForkPoolWorker-1 ended working on task 0
time took: 6.0
[TimeoutError(), TimeoutError(), TimeoutError(), TimeoutError(), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]Process finished with exit code 0
```

**双啵！**

我们捕获了超时错误异常 *4* 次并处理了它，而函数在 *n=0* 时仍然工作。也就是说，`ForkPoolWorker-1`进程本身并没有停止等待 *5* 秒，每隔 *1.5* 秒就会出现一个异常，我们拦截了这个异常。然后`ForkPoolWorker-1`进程成功完成其工作并返回值 *0* 。这根本不是我们想要的，是吗？

这种情况下我们该怎么办？超时过期后如何强行终止进程？

让我们考虑一下如何中断函数的执行。可能很多，我知道这甚至可以从键盘上使用键盘快捷键 **Ctr+C** 来完成。如何在 python 中强制中断呢？我们需要向我们的进程发送一个中断信号。让我们看看`os`模块的`kill`功能的文档:

> `[os.**kill**](https://docs.python.org/3/library/os.html#os.kill)` [(pid，sig)](https://docs.python.org/3/library/os.html#os.kill)
> 
> 向进程 *pid* 发送信号 *sig* 。主机平台上可用的特定信号的常数在`[signal](https://docs.python.org/3/library/signal.html#module-signal)`模块中定义。

查阅 [**信号**](https://docs.python.org/3/library/signal.html) 模块的文档，可以看到`[SIGINT](https://docs.python.org/3/library/signal.html#signal.SIGINT)`负责从键盘中断(默认动作是抬起`[KeyboardInterrupt](https://docs.python.org/3/library/exceptions.html#KeyboardInterrupt)`

**注意:**这种方法只适用于 Unix 系统。稍后我将描述如何在**窗口**中完成这项工作。

> [*类*](https://docs.python.org/3/library/threading.html#threading.Timer) `[threading.**Timer**](https://docs.python.org/3/library/threading.html#threading.Timer)` [(区间，函数，args=None，kwargs=None)](https://docs.python.org/3/library/threading.html#threading.Timer)
> 
> 创建一个计时器，在经过*间隔*秒后，该计时器将运行*函数*，其参数为 *args* 和关键词参数 *kwargs* 。如果*参数*为`None`(默认)，那么将使用空列表。如果 *kwargs* 为`None`(默认值)，那么将使用空字典。

很好，现在需要的是，我们将创建一个函数来模拟来自键盘的中断，并且我们将在一个等于超时的计时器上运行这个函数。如果它没有来，我们将简单地取消计时器。让我们以装饰器的形式实现我们的想法:

让我们看看它是如何为我们的功能工作的:

```
Process MainProcess started working on task 0
function my_awesome_foo took longer than 1.5 s.
time took: 1.5Process finished with exit code 0
```

一切都如我们所愿！对于基于 **Windows** 的系统，可以用`[_thread.interrupt_main()](https://docs.python.org/3/library/_thread.html#thread.interrupt_main)`代替`os.kill()`。我在 Windows 11 上进行了测试，一切正常。让我们看看我们的修饰函数如何与 Pool 类的`imap`方法一起工作:

```
Process ForkPoolWorker-2 started working on task 1
Process ForkPoolWorker-1 started working on task 0
Process ForkPoolWorker-4 started working on task 3
Process ForkPoolWorker-3 started working on task 2
Process ForkPoolWorker-2 ended working on task 1
Process ForkPoolWorker-4 ended working on task 3
Process ForkPoolWorker-3 ended working on task 2
Process ForkPoolWorker-4 started working on task 4
Process ForkPoolWorker-2 started working on task 6
Process ForkPoolWorker-3 started working on task 5
Process ForkPoolWorker-1 started working on task 7
Process ForkPoolWorker-3 ended working on task 5
Process ForkPoolWorker-4 ended working on task 4
Process ForkPoolWorker-2 ended working on task 6
Process ForkPoolWorker-4 started working on task 9
Process ForkPoolWorker-3 started working on task 8
Process ForkPoolWorker-1 ended working on task 7
Process ForkPoolWorker-4 ended working on task 9
Process ForkPoolWorker-3 ended working on task 8time took: 3.0['function my_awesome_foo took longer than 1.5 s.', 1, 2, 3, 4, 5, 6, 7, 8, 9]Process finished with exit code 0
```

这就是我们想要的！

## 进程使用的内存限制(仅适用于 Unix 系统)

现在让我们设想一种情况，您想要限制一个进程可以使用的内存。这可以在 Unix 系统上使用 [**资源**](https://docs.python.org/3/library/resource.html) 模块轻松完成。

```
Process ForkPoolWorker-1 started working on task 0
Process ForkPoolWorker-2 started working on task 1
Process ForkPoolWorker-3 started working on task 2
Process ForkPoolWorker-4 started working on task 3
Process ForkPoolWorker-1 started working on task 4
Process ForkPoolWorker-4 ended working on task 3
Process ForkPoolWorker-2 ended working on task 1
Process ForkPoolWorker-3 ended working on task 2
Process ForkPoolWorker-2 started working on task 5
Process ForkPoolWorker-3 started working on task 6
Process ForkPoolWorker-4 started working on task 7
Process ForkPoolWorker-1 ended working on task 4
Process ForkPoolWorker-1 started working on task 8
Process ForkPoolWorker-2 ended working on task 5
Process ForkPoolWorker-3 ended working on task 6
Process ForkPoolWorker-4 ended working on task 7
Process ForkPoolWorker-2 started working on task 9
Process ForkPoolWorker-1 ended working on task 8
Process ForkPoolWorker-2 ended working on task 9time took: 3.0[MemoryError(), 1, 2, 3, 4, 5, 6, 7, 8, 9]Process finished with exit code 0
```

嗯，就像蛋糕上的樱桃一样，让我们把所有的例子收集到一个例子中，看看我们在这里讨论的所有事情是如何通过使用 [**parallelbar**](https://pypi.org/project/parallelbar/) 库的一个命令来完成的:

![](img/c140061a6517dcdc86f5b575ce71726e.png)

作者图片

结果是:

```
time took: 8.2
[MemoryError(), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, TimeoutError('function my_awesome_foo took longer than 1.5 s.'), 21, 22, 23, 24, 25, 26, 27, 28, 29]Process finished with exit code 0
```

因此，多亏了进度条，我们能够估计到执行结束还剩下多少时间，他还向我们展示了被拦截的错误。

您可以在我的文章中了解更多关于 **parallelbar** 的信息:

<https://medium.com/pythoneers/visualize-your-multiprocessing-calculations-in-python-with-parallelbar-5395651f35aa>  

或者您可以查看[文档](http://or you can check the documentation)

## 结论

*   在本文中，我们以**多重处理**模块的`Pool`类为例，简要回顾了 python 中的多重处理。
*   我们已经看到了如何使用`imap`函数在进程池中处理异常。
*   我们实现了一个装饰器，允许你在指定的超时后中断函数的执行
*   我们以限制使用的内存为例，研究了如何限制进程池中某个进程使用的资源
*   我们看了一个使用 [**parallelbar**](https://github.com/dubovikmaster/parallelbar) 库实现异常处理和限制进程使用资源的小例子

我希望这篇文章对你有用！