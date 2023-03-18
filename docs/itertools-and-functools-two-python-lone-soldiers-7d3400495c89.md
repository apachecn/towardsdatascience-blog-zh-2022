# itertools 和 functools:两个 Python 独行侠

> 原文：<https://towardsdatascience.com/itertools-and-functools-two-python-lone-soldiers-7d3400495c89>

## Python 很酷的一点是它对函数式编程的支持，函数式编程声明处理步骤是通过函数来完成的。

![](img/1908479be27f80dac709911d1c75810a.png)

由[马库斯·斯皮斯克](https://unsplash.com/@markusspiske)在 [Unsplash](https://unsplash.com/) 上拍摄的照片

幸运的是，Python 附带了成熟的包，从而增强了它的多范式制胜牌，其中一些(如果不是全部的话)最初是用 c 实现的。实际上，您可以在 Github Python 项目的 CPython repo 中的 [Lib](https://github.com/python/cpython/tree/main/Lib) / [Modules](https://github.com/python/cpython/tree/main/Modules) 文件夹中阅读这些实现。

在本文中，我将讨论 Python 为我们提供的两个函数式编程模块，用于执行各种基于函数的计算:`itertools and functools`。

我不会复制你在官方文档中发现的相同模式。我只是想提到一些你在互联网上不常遇到的功能，但不知何故在我最需要的时候设法在那里拯救了我。

带着我对他们的感激之情，让我们开始吧。

# 1.Itertools

简单地说，`itertools`允许高效的循环。

> 该模块标准化了一组快速、内存高效的核心工具，这些工具可以单独使用，也可以组合使用。它们一起形成了一个“迭代器代数”,使得用纯 Python 简洁高效地构造专用工具成为可能。

## 1.1.循环

我记得我在哈斯克尔身上看到过类似的东西。很明显，需要用 Python 实现一个类似的构件。
无论何时你想运行一个无限循环的循环，这个循环会在一个异常或特定的条件下停止，这就是你要做的。
简单有效:

```
>> from itertools import cycle
>> for el in cycle(range(10)): print(el, end="")
>> 0123456789012345 ...
>> ..
```

最终，如果你不中断地运行你的循环，你的内存缓冲区会发生一些事情。不知道，没试过。

## 1.2.积聚

如果您想用尽可能少的代码行设计某种累加器:

```
>> from itertools import accumulate
>> list(accumulate(range(10)))
>> [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]
>> ...
```

基本上，累加器中的每个元素都等于所有先前元素的总和，包括它自己。

就我个人而言，我会自豪地用至少 6 行代码来实现纯内置函数。然而，由于几乎每个数据帧聚合中都需要一个合适的累加器，您可能会喜欢它的工作方式。

## 1.3.分组依据

实际上，在我的几个项目中，我用它获得了很多乐趣。这里有一个例子:

```
>>> from itertools import groupby
>>> data = 'AAAABBBCCDAABBB'
>>> for k, v in groupby(data):
...   print(k, len(list(v)))
... 
A 4
B 3
C 2
D 1
A 2
B 3
```

它的操作与`numpy.unique`函数大致相似，但功能更多。
每次 key 函数值发生变化，都会创建一个 break 或一个新组。这与`SQL`的`GROUP BY`形成对比，后者将相似的数据分组，而不考虑顺序。所以在将数据传递给`itertools.groupby`之前，首先对数据进行排序是很重要的。

## 1.4.成对地

当你在一个 iterable 上循环时，你对单独处理每个元素不感兴趣。`**Python 3.10**`为更有趣的用例提供了一个新特性。
这里有一个成对的例子:

```
>>> from itertools import pairwise
>>> list(pairwise('ABCDEFG'))
... ['AB','BC','CD','DE','EF','FG']
```

拥有额外程度的执行迭代总是一种特权和奢侈。通过两个元素的连续块来迭代你的列表允许更多的递增或递减处理。

## 1.5.星图

如果你曾经使用过 map 来操作一个 iterable，你可以把 starmap 看作是一个 map 操作符，它扩展到了更小的 iterable。让我们看一个小例子:

```
>>> from itertools import starmap
>>> v = starmap(sum, [[range(5)], [range(5, 10)], [range(10, 15)]])
>>> list(v)
[10, 35, 60]
>>> ...
```

在本例中，`sum`应用于每组参数，并生成一个具有相同编号的结果列表。将每个 iterable 看作一个参数而不是一个简单的输入是很重要的，这就是为什么每个 iterable 都被放在括号中，否则就会出现错误。

任何时候你都想拥有一个函数，它可以根据不同的数据组获取参数，`**starmap**`可能会非常有效，非常有表现力。

## 1.6.zip_longest

Haskell 中也使用了`zip`。对于并行迭代是相当有用的。
然而，它期望每个可迭代参数的长度相同。
如果你试着滑过两个不同长度的物体，`zip`会把长的物体超出的部分剪掉。
`zip_longest`将较短的可重复项填充到与较长的可重复项相同的长度:

```
>>> from itertools import zip_longest
>>> list(zip_longest('ABCD', 'xy', fillvalue='-')) 
... [('A', 'x'), ('B', 'y'), ('C', '-'), ('D', '-')]
```

当你最初有一个长度不匹配的时候，这是很实际的。然后，您可以轻松地循环两个或更多的可迭代对象，并确保这将在运行时保持不变。

## 1.7.产品

假设您有一个矩阵等待更新，您必须先遍历行，然后遍历列(反之亦然)。
你会怎么做:

```
for i in ncolumns: 
    for j in nrows :
        ...
```

有一种更酷的方法可以用`itertools`中一个叫做`product`的特性来复合这两个循环:

```
>>> from itertools import product
>>> for i,j in product([0,1], [10,11]):
...   print(i,j)
... 
0 10
0 11
1 10
1 11
```

`**product**`沿可迭代对象执行笛卡尔乘积，在矩阵的情况下，您可以替换嵌套循环，如下所示:

```
>>> for i, j in product(ncolumns, nrows):
    ...
```

不利的一面是，这消除了在两个循环之间初始化中间变量或临时变量的可能性。归根结底，这取决于你想要实现什么。

这可能是我能想到的让我的一天更轻松、更酷的每一件事情。我不会在这方面做更多的阐述，因为这个库永远不会出其不意，特别是 [itertools recipes](https://docs.python.org/3/library/itertools.html#itertools-recipes) :它们是由作者编写的，以充分利用之前看到的函数，这些函数充当了矢量化和高度优化的构建块。在菜谱中，您实际上看到了这些函数在多大程度上有潜力创建更强大的工具，并具有与底层工具集相同的高性能。那里正在发生一些非常有趣的事情，你最好马上去看看...

# 2 .功能工具

最简单地说，`functools`是高阶函数的模块:它们将函数作为参数和输出。装饰器、属性、缓存都是高阶函数的例子，它们作为注释放置在函数定义之上。让我们看一些其他的例子。

## 2.1.部分的

因此，您需要在您的程序中实现一个将被视为一等公民的函数。问题是，它接受了太多的参数，因此在调用时会导致异常。
`partial`是`functools`中的一个特性，允许通过给至少一个参数赋值来冻结一部分参数。

让我们考虑下一组数组和权重:

```
>>> import numpy as np
>>> x1 = np.array([1,2,1])
>>> w1 = np.array([.2, .3, .2])
>>> n1 = 3
>>> 
>>> x2 = np.array([2,1,2])
>>> w2 = np.array([.1, .1, .05])
>>> n2 = 3
>>>
```

然后让我们考虑这个函数，它计算这些数组的加权平均值:

```
>>> def weighted_means(x1, w1, n1, x2, w2, n2): 
...   return np.dot(x1,w1)/n1 , np.dot(x2,w2)/n2
```

我们将该功能付诸实施:

```
>>> weighted_means(x1, w1, n1, x2, w2, n2)
... (0.3333333333333333, 0.13333333333333333)
```

假设您想通过冻结`x2, w2 and n2`来减少变量参数的数量，这样做:

```
>>> from functools import partial
>>> reduced_weighted_means = partial(weighted_means, x2 = x2 , w2 = w2 , n2 = n2)
```

然后，您可以使用参数数量减少的新 reduced 函数:

```
>>> reduced_weighted_means(x1, w1, n1)
... (0.3333333333333333, 0.13333333333333333)
```

请注意，这是和以前一样的结果，表明该函数的工作方式就好像固定参数是静态输入的一样。当函数在与参数多样性相关的特定约束下使用时，这似乎是一个完美的变通方法。

## 2.2.部分方法

`partialmethod`是 partial 的外推，只是它被用作类方法。让我们用一个例子来说明这一点(我承认这是一个非常愚蠢的例子):

让我们测试一下:

```
>>> mn = ModelName('clustering')
>>> mn.set_cls()
>>> print(mn.algorithm)
>>> 'classification'
>>>
>>> mn.set_regr()
>>> print(mn.algorithm)
>>> 'regression'
```

该类用于存储字符串，并将 setter 角色委托给两个 partialmethod。`set_cls`和`set_regr`的工作方式就好像它们是正确定义的方法，并将`algorithm`分别设置为不同的值。一个小提示:定义了`algorithm`属性后，应该没有`algorithm.setter`。

## 2.3.单一调度

假设您定义了一个执行一些指令的简单函数，然后决定使它更加通用:

然后，通过使用现在用作装饰器的`zen_of_python`的`register()`属性，用额外的实现重载该函数。

您可以根据参数的类型确定不同的实现:

请注意泛型函数在接受不同类型的参数时的行为:

```
>>> zen_of_python('hello')
... Beautiful is better than ugly.>>> zen_of_python(1)
... There should be one-- and preferably only one --obvious way to do it.>>> zen_of_python(1.0)
... Readability counts.>>> zen_of_python([1, 2, "a"])
... Namespaces are one honking great idea -- let's do more of those!
```

结果与我们的实现一致。然而，由于通用函数还不知道`dict`类型的适当实现，它将跳转到默认实现:

```
>>> zen_of_python(dict())
... Beautiful is better than ugly.
```

这东西很酷。我知道。

## 2.4.singledispatchmethod 方法

我们的最后一位客人:`singledispatchmethod`它可以在类定义中使用。
举个例子也许？
假设您有一个基于您输入的参数类型输出字符串的类:

以一个小演示结束:

```
>>> gen = Generic()>>> gen.generic_method(1)
...'First case'>>> gen.generic_method(1.0)
>>> 'Second case'>>> gen.generic_method([1,2,3])
>>> 'Third case'
```

让我们试试`dict`式:

```
>>> gen.generic_method(dict(a=1, b=2))
Traceback (most recent call last):
    ...
  NotImplementedError: Never heard of this type ..
```

如果你在泛型类上声明了另一个类型，你会遇到一个错误。如果你想让它消失，你应该添加一个小而温和的功能来照顾它。

# 参考

正式文档— [itertools](https://docs.python.org/3/library/itertools.html) 和 [functools](https://docs.python.org/3/library/functools.html)