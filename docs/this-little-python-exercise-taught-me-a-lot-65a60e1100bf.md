# 这个小 Python 练习教会了我很多

> 原文：<https://towardsdatascience.com/this-little-python-exercise-taught-me-a-lot-65a60e1100bf>

## 如果你想成为一名优秀的程序员，不要满足于你想到的第一个解决方案

![](img/37848e08ba51008d41fd1fed95d9e147.png)

约翰·施诺布里奇在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

解决编程挑战是提高技能和学习新工具的好方法，即使是用你已经知道的语言。这就是为什么我觉得这个小问题很吸引人:

> 给定一个包含重复字母的字符串 *s* ，例如“hhhhhhhaaaaaaaaabbbbbbbvvvvvaaaahhhh”，输出一个摘要列表，显示每个相同字符序列的起始位置和字符

对于上面的字符串，输出将是:

```
[(0, 'h'), (7, 'a'), (15, 'b'), (22, 'v'), (27, 'a'), (31, 'h')]
```

## 解决方案 1

第一种解决方法是根据每个字符在字符串中各自的位置，这可以通过

```
enumerate(s)
```

然而，这还不够，因为我们需要扫描“变化”——当前角色与其前任不同的位置。让我们迭代字符串的两个副本，移动一个字符，如下所示:

```
enumerate(zip(s[1:],s),1)
```

枚举现在从 1 开始，因为我们从相对于前一个字符的索引 1 开始查找字符。

为了完成这个解决方案，我们需要只保留代表一个新字符的元组。我们还需要确保在输出中包含第一个字符(第 0 个索引处的字符)。空字符串的情况需要单独处理，以避免异常。

这是我们的功能:

```
def summarize_string1(s):
    if not s:
        return []
    return [(0,s[0])] + \
[(i, c) for (i, (c, pc)) in enumerate(zip(s[1:],s),1) if c != pc]
```

注 *i* 代表索引， *c* 代表字符， *pc* 代表前一个字符。

## itertools.groupby

事实证明，Python 的 *itertools* 模块可以帮助我们让这段代码更加优雅。

该模块中的 *groupby* 函数将一个序列作为输入，并将其分解成子序列，其中每个子序列中的项目共享同一个键。这听起来有点抽象，所以让我们分解一下，看看在这种情况下如何让它工作。

我们再看看

```
enumerate(s)
```

它根据单个字符及其位置生成一个项目序列，例如(0，' h ')、(1，' h')…等等。

现在，我们希望将这个序列中字母相同的项目组合在一起，因此我们编写这个 lambda 函数来提取字符作为键:

```
lambda x: x[1]
```

现在我们准备编写如下代码:

```
from itertools import groupby
def summarize_string2(s):
    result = []
    for key, group in groupby(enumerate(s),lambda x: x[1]):
        result.append(next(group))
    return result
```

理解这段代码的关键是理解 groupby 函数在这里是如何工作的。它检查我们的字符串 *s* 的枚举字符，并将 lambda 函数应用于元组(0，' h ')、(1，' h ')等。当它意识到它到达了一个键不同的项时，它“打包”到目前为止它处理过的所有项，并把它们放入一个 iterable(我们在变量*组中接收到它)*和它们相关的公共键中。因为*组*是一个*生成器*而不是一个合适的列表，访问它的第一个元素的最简单的方法是使用 Python 的内置 *next，*以上面代码*中描述的方式。*下一个(组)的可能替代项是

```
list(group)[0]
```

提取第一个元素。

这是我们对这个问题的第二个解决方案。我认为代码更好，但是在性能方面，它比解决方案 1 慢 50%。

## 使用 numpy

numpy 用于数值计算，通常不用于处理字符串。但是它有一个 diff 函数，我们可以在这里利用它。

下面是我们使用 numpy 的代码:

```
import numpy as npdef summarize_string3(s):
    str_arr = np.array(list(s.encode('ascii')))
    loc = np.diff(str_arr,prepend=0).nonzero()[0]
    return [(i, s[i]) for i in loc]
```

str_arr 是字符串到整数数组的转换。每个整数代表字符串中的一个字母，通过它的 ASCII 码(如果字符串包含 ASCII 子集中没有的 unicode 字符，它就不起作用)。

np.diff 是什么？它查找 numpy 数组中连续条目之间的差异。无论相同的字符在哪里，显然相同的 ASCII 码将填充数组，差异将为零。非零元素只会出现在字符序列的开头。预先添加一个任意的零条目允许我们自动捕获第一个元素，因为没有一个字母或可打印字符的 ASCII 码是 0。

虽然 numpy 被认为非常快，但对于我在本文中使用的特定示例数组，我们的解决方案的 numpy 版本大约比前面的解决方案慢 4 倍。转换成数组一定消耗了大量的处理器时间。

## 一锤定音

这个有趣的编码小挑战让我们看到了解决同一个问题的不同方法。当选择工具来解决特定的编码问题时，每个工具所提供的效用应该与最终解决方案的可读性及其解决特定问题的效率相权衡。

## 资源

*   本文中三个函数的代码也可以作为要点[获得](https://gist.github.com/kishkash555/020ede77fff9c2849c20eb175382a365)。
*   参见文档 [itertools.groupby](https://docs.python.org/3/library/itertools.html#itertools.groupby) ， [numpy.diff](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) ， [next](https://docs.python.org/3/library/functions.html#next)
*   如何使用 [timeit](https://docs.python.org/3/library/timeit.html#timeit.timeit) 模块为您的代码计时