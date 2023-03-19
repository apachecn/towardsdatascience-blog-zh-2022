# 独一无二的 Python Itertools 指南

> 原文：<https://towardsdatascience.com/a-guide-to-python-itertools-like-no-other-454da1ddd5b8>

## 通过动画 gif 使你对这个惊人的库的理解具体化，并学习如何编写更优雅的代码

![](img/f10b29f2fdda7da1d06fc386b4eecea8.png)

照片由[埃琳娜·鲁梅](https://unsplash.com/@roum?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

# 目录

1.  [简介](#a8a7)
2.  [itertools.product()](#8830)
3.  [ITER tools . permutations()](#9d43)
4.  [ITER tools . combinations()](#ca31)
5.  [ITER tools . combinations _ with _ replacement()](#4cd3)
6.  [itertools.count()](#b308)
7.  [itertools.cycle()](#03db)
8.  [itertools.repeat()](#5155)
9.  [itertools.accumulate()](#90fb)
10.  [itertools.chain()](#1106)
11.  [itertools.compress()](#eb22)
12.  [itertools.dropwhile()](#45c6)
13.  [itertools.takewhile()](#cb16)
14.  [ITER tools . filter false()](#642e)
15.  [itertools.starmap()](#c753)
16.  [itertools.tee()](#e923)
17.  [ITER tools . zip _ longest()](#0362)
18.  [itertools.pairwise()](#2dd3)
19.  [itertools.groupby()](#9b2d)
20.  [itertools.islice()](#7d31)
21.  [结论](#a20a)

# 介绍

`itertools`是 Python 中用于处理可重复项的内置模块。它提供了许多快速、节省内存的方法来循环遍历 iterables，以获得不同的预期结果。这是一个强大但被低估的模块，每个数据科学家都应该知道，以便用 Python 编写干净、优雅和可读的代码。

虽然有大量关于`itertools`及其功能的资源，但它们通常专注于代码，使得不熟悉的读者很难立即理解每个方法的内部工作原理。本文采用了一种不同的方法——我们将使用动画 gif 带您浏览每一个`itertools`方法，以说明它们实际上是如何工作的。希望这个指南能帮助你更好地想象和理解如何使用`itertools`。

*注意:因为我们采用了这种方法，许多动画插图被故意过度简化，以帮助读者理解。例如，如果 GIF 中的输出显示为“ABC”，并不意味着代码输出是字符串“ABC”。而是代表代码输出，* `*[('A', 'B', 'C')]*` *。同样，* `*itertools*` *方法通常返回一个生成器(它不会立即显示结果元素)作为输出。然而，在 gif 中，我们将输出表示为将输出包装在* `*list()*` *函数后得到的结果。*

说到这里，让我们开始行动吧！

# itertools.product()

`itertools.product()`是一种组合迭代器，给出给定可迭代列表的笛卡尔积。每当您在代码中嵌套了 for 循环时，都是使用`itertools.product()`的好机会。

![](img/4f27d71a65b613dd10ce2d307450e7e6.png)

图 itertools.product()`的动画演示

要计算 iterable 与其自身的乘积，可以用可选的`repeat`参数指定重复的次数。

# ITER tools . p**er mutations()**

给你一个 iterable 的所有可能的排列，即没有重复元素的所有可能的排序。

![](img/ec652aaba6aa71467645e1b27b696453.png)

图 itertools.permutations()`的动画演示

# **ITER tools . combinations()**

对于给定的 iterable，`itertools.combinations()`返回长度为 *r* 且没有重复元素的所有可能组合。

![](img/e18525cb4ef143a4637196967c15dddd.png)

图 itertools.combinations()`的动画演示

图 3 中的 GIF 假定了`r=3`，因此返回了`('A','B','C')`的唯一组合。如果`r=2`，`itertools.combinations('ABC', 2)`将返回`[('A','B'), ('A','C'),('B','C')]`。

# **ITER tools . combinations _ with _ replacement()**

对于给定的 iterable，`itertools.combinations_with_replacement()`返回长度为 *r* 的所有可能组合，每个元素允许重复多次。

![](img/d76dc47ee44ee9fe399a12b53585cc6c.png)

图 4:ITER tools . combinations _ with _ replacement()的动画演示

# **itertools.count()**

`itertools.count()`返回给定一个输入数的均匀间隔值，直到无穷大。因此，它被称为“无限迭代器”。默认情况下，这些值将平均间隔 1，但这可以用`step`参数设置。

![](img/1aa53d83b2859c0831e1547865d91768.png)

图 itertools.count()的动画演示

# **itertools.cycle()**

`itertools.cycle()`是另一个无限迭代器，它连续“循环”遍历一个可迭代对象，产生一个无限序列。

![](img/44d5c3e14a9254a14096d43d57e898ab.png)

图 itertools.cycle()的动画演示

# **itertools.repeat()**

`itertools.repeat()`是第三种无限迭代器，它一遍又一遍地重复一个可迭代对象，产生一个无限序列，除非指定了`times`。比如`itertools.repeat('ABC', times=3)`会产生`['ABC', 'ABC', 'ABC']`。

![](img/e076221541fcd8c915c6e35ea2b050cc.png)

图 itertools.repeat()`的动画演示

# **itertools.accumulate()**

`itertools.accumulate()`生成一个迭代器，它累加 iterable 中每个元素的总和。

![](img/be91ebf3b4b332f4f1b70b5f360695ce.png)

图 itertools.accumulate()'的动画演示

默认情况下，它通过加法或串联进行累加。您还可以使用带有两个参数的`func`参数来指定一个定制函数。例如，`itertools.accumulate('ABCD', func=lambda x, y: y.lower()+x)`会产生`['A', 'bA', 'cbA', 'dcbA']`。

# **itertools.chain()**

`itertools.chain()`获取多个可迭代对象，并将它们链接在一起，生成一个可迭代对象。

![](img/f97934481c76188aa5e65208e1bb763a.png)

图 itertools.chain()`的动画演示

与此稍有不同的是`itertools.chain.from_iterable()`，它接受一个 iterables 中的一个 iterable，并将它的单个元素链接在一个 iterable 中。因此，`itertools.chain.from_iterable([‘ABC’, ‘DEF’])`将产生与`itertools.chain(‘ABC’, ‘DEF’)`相同的结果，即`[‘A’, ‘B’, ‘C’, ‘D’, ‘E’, ‘F’]`。

# **itertools.compress()**

`itertools.compress()`基于布尔值的一个可迭代项过滤另一个可迭代项(称为“选择器”)。结果 iterable 将只包含来自输入 iterable 的元素，这些元素的位置对应于选择器的`True`值。

![](img/ec5bfdf31cd86640fac61260276b8bcf.png)

图 itertools.compress()的动画演示

# **itertools.dropwhile()**

在`itertools.dropwhile`中，您“删除”条件为`True`的“while”元素，并在条件第一次变为`False`后“获取”元素。

![](img/b4a77129b4fc29f7f9aa08fa9e173e9c.png)

图 11:ITER tools . drop while()的动画演示

对于图 10 所示的例子:

*   第一要素:条件为`True` —下降
*   第二元素:条件为`True` —下降
*   第三个元素:条件为`False` —保持所有元素

# **itertools.takewhile()**

`itertools.takewhile()`以相反的方式工作——在条件第一次变成`False`之后，您“取”元素“而”条件是`True`并且“放”元素。

![](img/6ab7aa543a2e2e890fb5d7a1ceef246f.png)

图 12:ITER tools . take while()的动画演示

对于图 11 所示的例子:

*   第一个元素:条件为`True` —保持
*   第二元素:条件为`True` —保持
*   第三个元素:条件为`False`——从此丢弃所有元素

# **ITER tools . filter false()**

`itertools.filterfalse()`顾名思义，只有在条件为`False`的情况下，才保持输入的元素是可迭代的。

![](img/d4628bb80e4765176fc03a5926b6c5cb.png)

图 13:ITER tools . filter false()`的动画演示

# **itertools.starmap()**

通常，您可以使用`map`将一个函数映射到一个 iterable，比如一个列表。例如，`map(lambda x: x*x, [1, 2, 3, 4])`会产生`[1, 4, 9, 16]`。但是，如果你有一个 iterables 的 iterable，比如一个元组列表，并且你的函数需要使用内部 iterable 的每个元素作为参数，你可以使用`itertools.starmap()`。

![](img/e80967eaa6cb771a09ce539bf46d8f4f.png)

图 14:ITER tools . star map()`的动画演示

如果你感兴趣，可以看看下面这篇由 [Indhumathy Chelliah](https://medium.com/u/720e3a4ac60c?source=post_page-----454da1ddd5b8--------------------------------) 撰写的文章，这篇文章详细分析了`map`和`starmap`之间的区别:

[](https://betterprogramming.pub/exploring-map-vs-starmap-in-python-6bcf32f5fa4a)  

# **itertools.tee()**

给定一个 iterable，`itertools.tee()`产生多个独立的迭代器，由它的`n`参数指定。

![](img/067d73db0cf5e842d398e7629153761b.png)

图 15:ITER tools . tee()`的动画演示

# **ITER tools . zip _ longest()**

内置的`zip()`函数接受多个可迭代对象作为参数，并返回一个迭代器，我们可以用它来生成一系列由每个可迭代对象中的元素组成的元组。它要求输入的 iterables 长度相等。对于不同长度的可重复项，`zip()`会导致一些信息的丢失。例如，`zip(‘ABCD’, ‘12’)`将只返回`[(‘A’, ‘1’), (‘B’, ‘2’)]`。

`itertools.zip_longest()`减轻了这种限制。它的行为与`zip()`完全相同，除了它基于最长的输入 iterable“压缩”。默认情况下，不匹配的元素用`None`填充，除非使用`fillvalue`参数指定。

![](img/73febbf5b464aa161b6cac881c31c1de.png)

图 16:ITER tools . tee()`的动画演示

# itertools.pairwise()

在 Python 3.10 中新引入的`itertools.pairwise()`从一个输入 iterable 生成连续的重叠对。如果您有一个可迭代的对象，比如一个列表或一个字符串，并且您想用一个包含两个元素的滚动窗口来迭代它，这是非常有用的。

![](img/523f20c7b194a5660a9a79b04497ae96.png)

图 itertools.pairwise()'的动画演示

这是奖金！如果你还没有使用 Python 3.10，你可以定义自己的`pairwise`函数(致谢: [Rodrigo](https://mathspp.com/about) )。

```
>>> from itertools import tee
>>> def pairwise(it):
>>>    """Mimicks `itertools.pairwise()` method in Python 3.10."""
>>>     prev_, next_ = tee(it, 2) # Split `it` into two iterables.
>>>     next(next_) # Advance once.
>>>     yield from zip(prev_, next_) # Yield the pairs.
```

# itertools.groupby()

给定一个输入 iterable，`itertools.groupby()`返回连续的键和相应组的 iterable。

![](img/af56c1eed61c3c52b320fdfeba048099.png)

图 18:ITER tools . group by()的动画演示

默认情况下，`itertools.groupby()`会在每次键值改变时生成一个中断或新组。对于图 17 中的例子，它将单个“A”(绿色)分组为一个单独的组，而不是将 4 个“A”分组在一起。如果期望的行为是根据 iterable 中的唯一元素进行分组，那么首先需要对输入 iterable 进行排序。

# itertools.islice()

`itertools.islice()`是一个迭代器，在给定`start`、`stop` 和`step` 参数的情况下，返回输入 iterable 中所需的元素。

![](img/b12bd2cee5a28a9235ed336d537e5c3c.png)

图 19:` ITER tools . is lice()`的动画演示

您可能会想，“使用常规索引切片也可以做到这一点！”。比如`‘AAABBACCC’[1:8:2]`会返回`‘ABAC’`。事实证明，`itertools.islice()`和常规的索引切片是有区别的:

1.  常规索引切片支持 start、stop 和 step 的负值，但`itertools.islice()`不支持。
2.  常规索引切片创建了一个新的可迭代对象，而`itertools.islice()`创建了一个迭代现有可迭代对象的交互器。
3.  由于前面的原因，`itertools.islice()`更节省内存，尤其是对于大的可重复项。

# 结论

恭喜你走到这一步！这是大量的 gif，但我希望它们能帮助你更好地欣赏令人惊叹的`itertools`库，并且你正在编写优雅的 Python 代码！

如果你觉得这篇文章有用，请在评论中告诉我。我也欢迎讨论、问题和建设性反馈。以下是更多相关资源，可进一步加深您的理解:

1.  的官方文档`[itertools](https://docs.python.org/3/library/itertools.html#)`
2.  [Python 中的 Iterables vs 迭代器](/python-iterables-vs-iterators-688907fd755f)由 [Giorgos Myrianthous](https://medium.com/u/76c21e75463a?source=post_page-----454da1ddd5b8--------------------------------)
3.  [高级 Python: Itertools 库——Python 语言的瑰宝](https://medium.com/fintechexplained/advanced-python-itertools-library-the-gem-of-python-language-99da37dfcca2)作者[法尔哈德·马利克](https://medium.com/u/d9b237bc89f0?source=post_page-----454da1ddd5b8--------------------------------)
4.  你应该如何——为什么——使用 Python 生成器

# 在你走之前

如果你对类似的内容感兴趣，可以看看我下面列出的其他文章。通过[媒体](https://zeyalt.medium.com/)关注我，或者通过 [LinkedIn](https://www.linkedin.com/in/zeyalt/) 或 [Twitter](https://twitter.com/zeyalt_) 联系我。祝您愉快！

[](https://betterprogramming.pub/7-lesser-known-python-tips-to-write-elegant-code-fa06476e3959)  [](https://betterprogramming.pub/demystifying-look-ahead-and-look-behind-in-regex-4a604f99fb8c)  [](/a-quick-and-easy-guide-to-conditional-formatting-in-pandas-8783035071ee)  [](/precision-and-recall-made-simple-afb5e098970f) 