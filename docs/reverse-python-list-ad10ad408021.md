# 如何更有效地反转 Python 列表

> 原文：<https://towardsdatascience.com/reverse-python-list-ad10ad408021>

## Python 中列表的高效反转

![](img/1ccbe703e64f54f6140460466e1b17d5.png)

Photo by [愚木混株 cdd20](https://unsplash.com/@cdd20?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/reverse?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

列表是 Python 中最基本和最常用的数据结构之一。列表是可变的有序对象集合，也可以存储重复值。它们甚至可以用作队列和堆栈(尽管`deque`可能更有效)。

列表反转是开发人员在编写 Python 应用程序时执行的一项相当常见的任务。在今天的简短教程中，我们将演示反转列表的几种不同方法，并讨论哪种方法执行得更好，尤其是在处理包含大量数据的列表时。

## 使用负步长的切片

现在让我们假设我们有以下由一些数值组成的列表:

```
>>> my_lst = [10, 0, 30, 25, 40, 100, 80]
```

我们可以使用切片和步长 1 来反转列表。

```
>>> my_lst_reversed = my_lst[::-1]
>>> my_lst_reversed
[80, 100, 40, 25, 30, 0, 10]
```

`[::-1]`符号主要做两件事:

*   它选择列表中的所有元素(第一个`:`)
*   它指定负的步长为 1，这意味着以相反的顺序(从结束到开始)检索元素

如果你想了解更多关于 Python 中切片和索引的知识，你可以参考我以前的一篇文章。

[](/mastering-indexing-and-slicing-in-python-443e23457125) [## 掌握 Python 中的索引和切片

### 深入研究有序集合的索引和切片

towardsdatascience.com](/mastering-indexing-and-slicing-in-python-443e23457125) 

请注意，这种方法看起来更像 Pythonic，但是只有当您想要创建原始列表的副本，以便它包含逆序的元素时，才应该使用这种方法。

如果您想就地反转一个列表(或者迭代反转列表的元素)，那么有两个其他的选择，它们比使用负步长进行切片要有效得多。

## 使用 list.reverse()方法

如果您正在考虑就地反转列表的元素(这意味着您实际上不想创建原始列表的另一个副本)，那么`list.reverse()`方法是最有效的方法。

```
>>> my_lst = [10, 0, 30, 25, 40, 100, 80]
>>> my_lst.reverse()
>>> my_lst
[80, 100, 40, 25, 30, 0, 10]
```

注意，由于`reverse()`方法就地发生，我们不需要将操作的结果赋回给变量。除了性能影响之外，我还发现这种方法可读性更好，并且清楚地表明列表被颠倒了。

## 使用 reversed()函数

最后，另一种选择是返回反向迭代器的`[reversed()](https://docs.python.org/3/library/functions.html#reversed)`内置函数。

> 返回一个反向`[iterator](https://docs.python.org/3/glossary.html#term-iterator)`。
> 
> *seq* 必须是具有`__reversed__()`方法或支持序列协议的对象(从`0`开始的`__len__()`方法和`__getitem__()`方法，带有整数参数)。

在对列表执行向后迭代时，推荐使用这种方法。

```
>>> my_lst = [10, 0, 30, 25, 40, 100, 80]
>>> my_lst_reverse_iter = reversed(my_lst)
>>> my_lst_reverse_iter
<list_reverseiterator object at 0x10afcae20>
```

如你所见，`reversed()`函数返回了一个迭代器，我们可以对它进行循环:

```
>>> for element in my_lst_reverse_iter:
...     print(element)
... 
80
100
40
25
30
0
10
```

请注意，您甚至可以将迭代器转换为列表:

```
>>> my_lst = [10, 0, 30, 25, 40, 100, 80]
>>> my_lst_reverse = list(reversed(my_lst))
>>> my_lst_reverse
[80, 100, 40, 25, 30, 0, 10]
```

但是如果这是你的最终目标，我个人会选择`list.reverse()`方法。

## 最后的想法

在今天的简短教程中，我们演示了在 Python 中反转对象列表的两种不同方法。更具体地说，我们展示了如何使用步长切片、`list`的`reverse()`方法以及内置的`reversed()` Python 方法来实现这一点。

总结一下，

*   如果您想就地反转列表，请选择`list.reverse()`
*   如果您想以相反的顺序创建列表的副本，请使用负步长(即`[::-1]`)进行切片
*   如果你想遍历一个反向列表，使用`reversed()`函数

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership) [## 通过我的推荐链接加入 Medium-Giorgos Myrianthous

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

gmyrianthous.medium.com](https://gmyrianthous.medium.com/membership) 

**相关文章你可能也喜欢**

[](/diagrams-as-code-python-d9cbaa959ed5) [## Python 中作为代码的图

### 用 Python 创建云系统架构图

towardsdatascience.com](/diagrams-as-code-python-d9cbaa959ed5) [](/setuptools-python-571e7d5500f2) [## Python 中的 setup.py 与 setup.cfg

### 使用 setuptools 管理依赖项和分发 Python 包

towardsdatascience.com](/setuptools-python-571e7d5500f2) [](/augmented-assignments-python-caa4990811a0) [## Python 中的扩充赋值

### 了解增强赋值表达式在 Python 中的工作方式，以及为什么在使用它们时要小心…

towardsdatascience.com](/augmented-assignments-python-caa4990811a0)