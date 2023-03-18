# Python 中 __pycache__ 是什么？

> 原文：<https://towardsdatascience.com/pycache-python-991424aabad8>

## 了解运行 Python 代码时创建的 __pycache__ 文件夹

![](img/0e84f4327dbdd65b31245359fa25f9a4.png)

照片由 [dimas aditya](https://unsplash.com/@dimasadityawicaksana?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/box?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

## 介绍

您可能已经注意到，在执行 Python 代码时，会(有时)创建一个名为`__pycache__`的目录，其中包含许多扩展名为`.pyc`的文件。

在今天的简短教程中，我们将讨论由 Python 解释器创建的这些文件的用途。我们将首先讨论为什么生成它们，如何抑制它们的创建，以及如何确保它们不会被提交给任何远程存储库。

## 。pyc 文件和 __pycache__ 文件夹

Python 是一种解释型语言，这意味着你的源代码在运行时被翻译成一组能够被 CPU 理解的指令。当运行你的 Python 程序时，源代码被编译成**字节码**其中是 CPython 的一个实现细节(Python 的原始实现)。字节码也缓存并存储在`.pyc`文件中，这样下次重新运行代码时，同一个文件的执行速度会更快。

*请注意，本节中围绕解释器和字节码讨论的几个概念过于简单，仅部分正确，但它们足以帮助您理解* `*pyc*` *文件和* `*__pycache__*` *文件夹。我计划在以后的文章中更详细地介绍这些概念。*

因此，在第一次执行您的源代码之后，将会创建一个`__pycache__`文件夹，其中包含几个与您的`.py`文件同名的`.pyc`字节码文件。如上所述，这些将在后续执行中使用，以便您的程序启动得更快一些。

每次修改源代码时，都会重新编译，并再次创建新的字节码文件。请注意，在某些情况下，这可能不是真的，Python 将使用缓存文件执行代码，这给你带来了一些麻烦。例如，您可能已经修复了一个 bug，但是 Python 可能运行在一个有 bug 的缓存版本上。在这种情况下，您可能必须删除`__pycache__`文件夹，或者甚至禁止创建这些文件，如下节所示。

## 禁止创建 __pycache__

当使用 CPython 解释器(不管怎样，它是 Python 的原始实现)时，您可以通过两种方式禁止创建该文件夹。

第一个选项是在运行 Python 文件时传递`-B`标志。当提供该标志时，Python 不会试图在导入源模块时写入`.pyc`文件:

```
**python3 -B my_python_app.py**
```

或者，您可以将`PYTHONDONTWRITEBYTECODE`环境变量设置为任何非空字符串。同样，这将阻止 Python 试图写`.pyc`文件。

```
**export PYTHONDONTWRITEBYTECODE=abc**
```

请注意，这两种方法是等效的。

## 正在将 __pycache__ 添加到。gitignore 文件

当在本地存储库中工作时，Git 将跟踪 Git repo 下的每个文件。每个文件都可以被**跟踪**(即已经暂存并提交)**未跟踪**(未暂存或提交)或**忽略**。

在大多数情况下，您应该忽略特定的文件，例如包含敏感数据的文件、特定于系统的文件或由 IDE 或特定工作区创建的自动生成的文件。

最优雅的方式是通过位于远程 Git 存储库顶层目录中的`.gitignore`文件，在该文件中，您可以显式指定 Git 将忽略并且不再跟踪的文件或目录(也可以应用正则表达式)。

`__pycache__`是不应该推送到远程存储库的目录之一。因此，您需要做的就是在`.gitignore`文件中指定目录。

```
# .gitignore__pycache__/
```

注意，对于一般的 Python 项目，有更多的文件需要放入`.gitignore`。更全面的列表，请参考本文件。

## 最后的想法

在今天的文章中，我们讨论了在`.pyc`文件中缓存的字节码，以及它们和`__pycache__`目录的用途。此外，我们探讨了如何抑制这个目录的创建，以及如何避免将它们包含在 Git 提交中(从而意外地将它们推送到远程存储库)。

*关于字节码的一个重要警告是，* `*.pyc*` *文件将在对等的* `*.py*` *文件不再存在的情况下使用。例如，如果您删除或重命名了一个* `*.py*` *文件，但出于某种原因，您仍然可以看到它们以任何可能的方式被执行，那么这可能是实际原因。*

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership) [## 通过我的推荐链接加入 Medium-Giorgos Myrianthous

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

gmyrianthous.medium.com](https://gmyrianthous.medium.com/membership) 

**你可能也会喜欢**

[](/python-linked-lists-c3622205da81) [## 如何在 Python 中实现链表

### 探索如何使用 Python 从头开始编写链表和节点对象

towardsdatascience.com](/python-linked-lists-c3622205da81) [](/python-iterables-vs-iterators-688907fd755f) [## Python 中的 Iterables vs 迭代器

### 理解 Python 中 Iterables 和迭代器的区别

towardsdatascience.com](/python-iterables-vs-iterators-688907fd755f) [](/duck-typing-python-7aeac97e11f8) [## Python 中的鸭式打字是什么？

### 理解动态类型编程语言(如 Python)中鸭类型的概念

towardsdatascience.com](/duck-typing-python-7aeac97e11f8)