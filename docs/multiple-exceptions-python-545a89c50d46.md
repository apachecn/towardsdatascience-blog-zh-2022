# 如何在 Python 中捕捉多个异常

> 原文：<https://towardsdatascience.com/multiple-exceptions-python-545a89c50d46>

## 在 Python 中处理多个异常

![](img/5f57a2bd82648d6c278bc427bde45d44.png)

照片由[丘特尔斯纳普](https://unsplash.com/@chuttersnap?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/error?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

## 介绍

一个开发良好的应用程序必须总是能够以适当的方式处理意外事件——比如异常。这对于在开发期间调试源代码也很重要，对于在应用程序启动并运行(最终进入生产)时检查应用程序日志也很重要。

在今天的简短教程中，我们将展示如何在 Python 中处理多个异常。我们还将探索 Python 中的一些新特性，这些特性可以帮助您以一种更直观、更简洁的方式完成这项工作。我们开始吧！

## 对于 Python < 3.11

Now let’s suppose that we have the following (fairly dumb) code snippet, that raises 【 , 【 and 【 .

```
def my_function(x):
    if x == 1:
        raise AttributeError('Example AttributeError')
    elif x == 2:
        raise ValueError('Example ValueError')
    elif x == 3:
        raise TypeError('Example TypeError')
    else:
        print('Hello World')
```

And let’s also suppose that we want to to call the function 【 but at the same time, we also need to ensure that we handle any unexpected errors appropriately. To do so, we can use the 【 clauses.

But let’s assume that we want to perform a certain action if an 【 is being thrown and a different action when either of 【 or 【 are raised by 【 .

```
try:
    my_function(x)
except AttributeError:
    # Do something
    ...
except (ValueError, TypeError):
    # Do something else
    ...
```

> A 【 statement may have more than one except clause, to specify handlers for different exceptions. At most one handler will be executed. Handlers only handle exceptions that occur in the corresponding try clause, not in other handlers of the same 【 statement. An except clause may name multiple exceptions as a parenthesized tuple.
> 
> — [Python 文档](https://docs.python.org/2/tutorial/errors.html#handling-exceptions)

如果您不打算对引发的任何错误做任何特殊处理(例如，您只需`pass`)，您甚至可以使用如下所示的`suppress`上下文管理器:

```
from contextlib import suppress

with suppress(AttributeError, ValueError, TypeError):
     my_function(x)
```

注意`[suppress()](https://docs.python.org/3/library/contextlib.html#contextlib.suppress)`从 Python 3.4 开始可用。此外，只有当您希望程序的特定部分无声地失败并继续执行时，才必须使用这种方法。但是在大多数情况下，您可能希望对某些异常采取某些措施。

## 用 Python 3.11 处理多个异常

从 Python 3.11 开始，引入了新的标准异常类型，即`ExceptionGroup`。这个新异常用于**一起传播一组不相关的异常**。

在下面的创建示例中，我们创建了一个包含四种不同类型错误的`ExceptionGroup`实例，即`TypeError`、`ValueError`、`KeyError`和`AttributeError`。然后，我们使用多个`except*`子句来处理`ExceptionGroup`，无论是针对单个异常类型还是多个异常类型。

```
try:
    raise ExceptionGroup('Example ExceptionGroup', (
        TypeError('Example TypeError'),
        ValueError('Example ValueError'),
        KeyError('Example KeyError'),
        AttributeError('Example AttributeError')
    ))
except* TypeError:
    ...
except* ValueError as e:
    ...
except* (KeyError, AttributeError) as e:
    ...
```

但是请注意，一个`except*`子句中提出的异常不适合匹配同一`try`语句中的其他子句

关于`ExceptionGroup`和`except*`条款背后的基本原理的更多细节，您可以参考 [PEP-654](https://peps.python.org/pep-0654/) 。

此外，要更全面地阅读 Python 3.11 中的新增内容和更新，包括我们之前讨论的内容，您可以参考我最近在下面分享的一篇文章。

[](/python-3-11-f62708eed569)  

## 最后的想法

在今天的简短教程中，我们展示了在 Python 中处理多个异常的各种不同方法。我们已经看到了如何使用传统的`except`子句来捕获多个异常，但是我们也展示了如何使用 Python 3.11 中将要引入的新的`except*`子句来这样做。

最后，您应该始终记住，在整个源代码中处理意外事件是一个重要的方面，如果执行得当，可以显著提高代码质量。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**相关文章你可能也喜欢**

[](/switch-statements-python-e99ea364fde5)  [](/how-to-merge-pandas-dataframes-221e49c41bec)  [](/requirements-vs-setuptools-python-ae3ee66e28af) 