# 通过多重处理使用参数化装饰器限制 Python 函数的执行时间

> 原文：<https://towardsdatascience.com/limiting-a-python-functions-execution-time-using-a-decorator-and-multiprocessing-6fcfe01da6f8>

## 限制 Python 函数执行时间的装饰器

![](img/67bc0d46f7be4eff45ab60e59ef42aa9.png)

丹尼尔·利维斯·佩鲁西 https://unsplash.com/photos/WxmZT3sIe4gT2**摄影**

在本文中，我将带您创建一个装饰器，通过多重处理来限制 Python 程序中函数的执行时间。我构建这个装饰器的主要动机是用简单的语法和最少的依赖性来限制 Python 函数的执行时间。

一种简单的方法是在 Python 函数中使用一个计时器，定期检查正在执行的 Python 函数是否超出了限制，然后退出。对于简单的一次性解决方案来说，这种方法可能是可行的，但是任何对第三方库的调用都会阻止检查时间限制。

我还想要一个尽可能不引人注目的解决方案，并且可以在整个代码库中轻松应用。装饰者提供了很好的语法和抽象来实现这个目标。

考虑到这一点，我知道我想要创建一个可以附加到我的项目中的任何函数的装饰器。装饰器会将函数的执行时间限制在某个特定的数量。我还想让所有东西都用 Python 编写，以限制添加这种调度程序的依赖性/复杂性。

这样做的主要挑战是

1.  装饰者应该为最大执行时间取一个参数，使其易于扩展。
2.  修饰函数能够具有任意的输入/输出。
3.  即使正在执行的函数调用了第三方库，计时器也应该工作。

首先，我需要创建一个可以接受参数作为自变量的装饰器。经过一些研究，我发现了一个优秀的[堆栈溢出线程](https://stackoverflow.com/questions/5929107/decorators-with-parameters.)，人们在那里提出了几种解决方案。

我按照彼得·莫滕森在评论中给出的架构为装饰者创建了一个装饰器。我不会深入讨论这是如何工作的，但是你可以进入这个线程来获得更详细的解释。为了获得更多关于装饰者的信息，我经常去[这里](https://realpython.com/primer-on-python-decorators/)复习。

然后，您可以将这个装饰器附加到您想要应用到您的函数的装饰器上，允许您参数化该装饰器。我想创建一个 **run_with_timer** decorator，它将最大执行时间作为一个参数。看起来是这样的。

接下来，我们可以填充代码来限制执行时间。逻辑如下；主进程将使用 Python 的多重处理在一个单独的进程中运行修饰函数。主进程将设置一个定时器，如果定时器超时，则终止执行该函数的子进程。

设置多重处理的代码由两部分组成。第一个是我称为 **function_runner，**的函数，它充当在新进程中运行的包装器，处理 Python 函数的运行并返回多处理函数可以处理的结果。第二个是多重处理代码，它产生新的进程，设置一个计时器，如果没有及时完成，就终止产生的进程。

最后，我可以创建函数来包装我的 **run_with_timer** 装饰器。我叫它睡熊。

当我们运行 **sleeping_bear** 函数时，如果它超过了装饰参数中设置的时间限制，它就会终止。如果 Python 函数在时间限制之前完成，那么 **send_end** 处理程序将返回结果。

```
**sleeping_bear**("Grizzly", hibernation=10)>> Grizzly is going to hibernate
>> 0 zZZ
>> 1 zZZzZZ
>> 2 zZZzZZzZZ
>> 3 zZZzZZzZZzZZ
>> 4 zZZzZZzZZzZZzZZ
>>
>> TimeExceededException: **Exceeded Execution Time****sleeping_bear**("Grizzly", hibernation=2)>> Grizzly is going to hibernate
>> 0 zZZ
>> 1 zZZzZZ
>> 
>> "Grizzly is waking up!"
```

总之，我已经向您展示了如何创建一个装饰器来限制使用多处理作为调度器的 Python 函数的执行时间。我能够解决三个主要问题。

1.  创建一个参数化装饰器来限制最大执行时间。
2.  允许向被包装的 Python 函数输入任意参数。
3.  通过利用多处理模块中的系统级调度程序来限制任何函数的执行时间。

另外，这一切都是用 Python 完成的，没有第三方依赖。启动多处理会有一些开销，但是我的目标是限制长时间运行的函数，所以这不是一个问题。

如果你喜欢这个，一定要关注我，在未来支持更多这样的内容。感谢您的阅读，一如既往，如果您有任何建议或反馈，请在评论中告诉我。