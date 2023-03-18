# 应该用 lambda 定义命名的 Python 函数吗？

> 原文：<https://towardsdatascience.com/should-you-define-named-python-functions-using-lambda-fb5806c0898b>

## PYTHON 编程

## 这样做是违背 PEP8 的，那么为什么这么多作者这么建议呢？

![](img/69c57d003bb888001af20ed7f1ecc42d.png)

在 Python 中，lambda 关键字允许定义匿名函数。由 [Saad Ahmad](https://unsplash.com/@saadahmad_umn?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

许多作者声称，为了缩短你的代码，你可以使用`[lambda](/lambda-functions-with-practical-examples-in-python-45934f3653a8)` [函数](/lambda-functions-with-practical-examples-in-python-45934f3653a8)来定义函数并给它们分配一个名字。一些初学者可能想尝试这种方法；说实话，我不仅在某个时候尝试过，而且乐此不疲。

请注意，这与使用未命名的`lambda`函数作为高阶函数的参数的情况不同，如下所示:

```
>>> x = [
...     ("Johny Faint", 197),
...     ("Jon Krohn", 187),
...     ("Anna Vanna", 178),
...     ("Michael Ginger", 165),
...     ("Viola Farola", 189)
... ]
>>> sorted(x, key=lambda x: -x[1])
[('Johny Faint', 197), ('Viola Farola', 189), ('Jon Krohn', 187), ('Anna Vanna', 178), ('Michael Ginger', 165)]
```

我们将讨论给一个名字分配一个`lambda`功能，如下所示:

```
>>> sort_fun = lambda x: -x[1]
```

不同之处在于赋值，用`=`表示，这意味着我们定义函数`sort_fun(x)`，它在形式上是一个有一个参数`x`的函数。该函数假设`x`是一个至少包含两个元素的 iterable。当然，我们知道这个函数是为与上面使用的函数相同的排序而构建的，如下所示:

```
>>> sorted(x, key=sort_fun)
[('Johny Faint', 197), ('Viola Farola', 189), ('Jon Krohn', 187), ('Anna Vanna', 178), ('Michael Ginger', 165)]
```

这两种方法不同。当我们在高阶函数内部定义一个`lambda`函数时，它不能在其他地方使用；并且它严格地专用于这种特殊的用途。当我们定义一个`lambda`函数并给它指定一个名字(`sort_fun`)时，它可以在代码的其他地方使用。同样，在 Python 中，函数就是函数，不管它们是如何定义的，是使用`def`还是`lambda`；因此，`sort_fun()`是一个有效的 Python 函数。

本文讨论命名的`lambda`函数；也就是说，`lambda`被定义的功能*和*被赋予一个名称。我们将讨论您是否应该这样定义函数，或者更确切地说，您应该避免这样做。我们将会看到 PEP8 对此会说些什么，但是我们也会对`def`定义的函数和`lambda`定义的函数进行比较。这将帮助我们决定`lambda`是否确实提供了一种在 Python 中定义简单函数的替代方式。

# PEP8 对此怎么说？

尽管已经 20 多年了，我们仍然使用 PEP8 作为“Python 代码的风格指南”所以，如果你需要检查一些与编码风格相关的东西，PEP8 应该是你的第一来源。

使用分配的`lambda`函数的想法与 PEP8 的建议相违背，PEP8 使用精确的措辞来声明您不应该这样做:

> 总是使用 def 语句，而不是将 lambda 表达式直接绑定到标识符的赋值语句…

那么，这个*真的*意味着使用一个命名的`lambda`函数确实是违背 PEP8 的吗？我认为有两种可能性:

1.  不，不是的！这个基于`lambda`的 PEP8 推荐已经过时了。例如，许多开发人员不同意 PEP8 推荐的 79 个字符的最大行长度，认为这是旧的气息。同样,`lambda`建议已经过时，我们不应该担心与 PEP8 背道而驰。PEP8 不是在 20 多年前，早在 2001 年就被创造出来了吗？在像 Python 编程语言这样现代的东西中，我们真的应该遵循如此古老的风格指南吗？
2.  是的，确实如此！建议打破这种基于`lambda`的推荐的作者，只依赖一个前提:字符数。这就意味着，越短越好。世界没有那么简单，编程也没有什么不同:仅仅是`lambda`函数的简洁还不够。听 PEP8，不要用命名的`lambda`函数！

老实说，前一段时间我自己也偶尔使用这样命名的`lambda`定义。我只是喜欢它们的样子:简洁和与众不同。现在，我认为正是这种差异吸引了我。我知道我倾向于*喜欢*复杂的解决方案，那些看起来与众不同、非典型的解决方案，因为它们让我觉得我是一个优秀的程序员。但是这太误导人了！写复杂的代码并不意味着成为一名优秀的程序员。现在我知道了这一点，我试着小心并仔细检查我是否做了不必要的复杂。

所以，让我们客观地分析一下，基于命名的`lambda`的函数定义是否比基于`def`的函数定义更好——如果是的话，什么时候更好。

请注意，我们*不*讨论使用`lambda`的一般情况，而只是命名的`lambda`定义，它将`lambda`功能分配给一个名称，如下所示:

```
foo = lambda x, u: f"{x} and {u}"
```

无论我们想出什么，都与在高阶函数中使用 lambdas 无关。

# 使用 lambda 和`def`定义函数

让我们来做一个实际的比较。为此，我将使用四个函数:

*   没有参数的函数
*   只有一个参数的函数
*   有两个参数的函数
*   有三个参数的函数

对于最后一个，我们将使用其中一个参数的默认值。以下是以传统方式定义的函数:

```
def weather():
    return "beautiful weather"

def square(x):
    return x**2

def multiply(x, m):
    """Multiply x by m.

    >>> multiply(10, 5)
    50
    >>> multiply("a", 3)
    'aaa'
    """
    return x * m

def join_sep(iter1, iter2, sep="-"):
    return [f"{v1}{sep}{v2}" for v1, v2 in zip(iter1, iter2)]
```

让我们看看这些函数在使用`lambda` s 定义时是什么样子的:

```
weather = lambda: "beautiful weather"

square = lambda x: x**2

multiply = lambda x, m: x * m

join_sep =lambda iter1, iter2, sep="-": [f"{v1}{sep}{v2}" for v1, v2 in zip(iter1, iter2)]
```

我们将从几个方面比较这两种方法。

## 代码简洁和视觉混乱

代码看起来更加密集。是好事吗？注意，分配的`lambda`函数并没有短多少:它们用`= lambda`替换了以下字符:`def`、`():`和`return`。因此，他们使用`7`而不是`12`字符，所以只有`5`字符更少(忽略空白)。那不是很多。

我认为`lambda`定义更密集的感觉来自于在一行中呈现它们。然而，我们可以使用`def`做同样的事情:

```
def square(x): return x**2
```

这个定义和`lambda`的定义在长度上没有太大区别，是吗？然而，使用`def`定义这样的短函数，我们在第一行之后使用新的一行是有原因的——对我来说，不是更多的字符；相反，它有助于避免视觉混乱。

因此，就代码简洁而言，我真的看不出选择`lambda`定义而不是`def`定义有什么意义。

就视觉杂乱而言，`lambda`的定义在我看来更糟糕，意思是更杂乱。再次比较使用`lambda`定义的`multiply()`函数:

```
multiply = lambda x, m: x * m
```

和`def`:

```
def multiply(x, m):
    return x * m
```

后者中的额外空白有助于将功能直观地分为两个部分:

1.  函数的签名:`def multiply(x, m):`
2.  函数体:`return x * m`

在`lambda`定义中，没有这样的视觉区分。相反，我们得到的是视觉混乱和密度，这需要额外的脑力劳动来区分函数的签名和主体。更糟糕的是，我们做的*不是*直接看到的签名！我们必须在头脑中创造它。

我不喜欢`lambda`定义的另一个视觉方面，这也让它们看起来很混乱。也就是说，函数名和它的参数不在一起，而是用“`= lambda` ”隔开。

## 清晰，无注释(类型提示)

因此，尽管有些作者说，`lambda`定义的简洁带来了视觉混乱和代码密度的增加。因此，对我来说，这种简洁是缺点而不是优点。

也许`lambda`函数的代码更清晰？在下面的代码块中，后面的定义是否比前面的更清晰？

```
def square(x):
    return x**2

square = lambda x: x**2
```

虽然我在阅读和理解这个`lambda`函数方面没有问题，但我不会说它比使用`def`关键字定义的函数更清晰。说实话，由于我已经习惯了`def`函数定义，所以前者对我来说更加清晰。

我们上面关于视觉混乱和代码密度的讨论也适用于此。我喜欢两行的`def`定义，它立即显示了函数的签名和主体。相反，`lambda`定义要求我在一行中通读整个代码，以便看到函数的签名和主体。这降低了代码的可读性——即使对于这个简单的函数也是如此。

让我们看看它在一个有两个参数的函数中是什么样子的:

```
def multiply(x, m):
    return x * m

multiply = lambda x, m: x * m
```

这一次，`def`版本对我来说似乎更加清晰。这也是由于在视觉上区分函数签名(包括函数参数)和函数体的一些问题。

这不是没有原因的，无论是谁提出了`lambda`定义，都是为了简短的函数。我们可以在`join_sep`函数的定义中看到这一点:

```
def join_sep(iter1, iter2, sep="-"):
    return [f"{v1}{sep}{v2}" for v1, v2 in zip(iter1, iter2)]

join_sep =lambda iter1, iter2, sep="-": [f"{v1}{sep}{v2}" for v1, v2 in zip(iter1, iter2)]
```

有人选择用`lambda`关键字定义的`join_sep()`作为更好的吗？对我来说，只有前者可以被认为是可读的，而后者过于困难——尽管我确实理解这个`lambda`定义。在这种情况下，理解不等于欣赏。

我提供这个例子只是为了强调一点，即使您决定使用`lambda`定义，您也应该为简单和简短的函数这样做；`join_sep()`是不是*不够*简单。

然而，`def`定义使我们能够做更多的事情来增加它的清晰度:文档字符串。让我重复一下`multiply()`定义的例子:

```
def multiply(x, m):
    """Multiply x by m.

    >>> multiply(10, 5)
    50
    >>> multiply("a", 3)
    'aaa'
    """
    return x * m
```

文档字符串是记录功能和用单元测试丰富功能的强大方法(例如，`[doctest](https://docs.python.org/3/library/doctest.html)` [s](https://docs.python.org/3/library/doctest.html) )。您可以向命名的`lambda`函数添加任何类型的文档，除了行内注释。另一方面，我想你不会对需要 docstring 的函数使用`lambda`。然而，我会说`multiply()`不需要*而*需要 docstring——但是有了它就更清楚了。

## 清晰，带注释(类型提示)

我们知道，当明智地使用时，类型提示可以增加函数的清晰度。考虑这两个例子:

```
from numbers import Number

def square(x: Number) -> Number:
    return x**2

square = lambda x: x**2
```

```
def multiply(x: Number, m: Number) -> Number:
    return x * m

multiply = lambda x, m: x * m
```

我们不能在`lambda`定义中使用函数注释。所以，每当你有一个函数，它的签名使用了*有用的*类型提示，基于`def`的定义总是更好，因为它比它的`lambda`定义传达了更多关于函数的信息。

现在，做一个简短的心理练习:想象一下既有 docstring 又有 type 提示的`multiply()`,看看这个定义比简单的命名的`lambda`定义好多少。

上面的例子表明，具有两个(或者更多)参数的函数可能比单参数函数更缺少类型提示。

# 结论

我们比较了 Python 中基于`lambda`和基于`def`的函数定义。在这样做的时候，我们只使用了两个方面:简洁和视觉混乱；和清晰度。这种对比并不十分富有——因为我们不需要让它变得富有。无论从哪个角度来看这两种定义，`def`总是比`lambda`更好。

即使在最简单的函数的情况下，我也没有找到足够的理由来使用`lambda`定义它们。通常情况下，相应的`def`定义更清晰，即使稍微长一点，它的代码在视觉上也不那么混乱和密集。

我很难想象一个双参数函数，特别是至少有一个参数的默认值，定义`lambda`时比定义`def`时更清晰。

最后，`def`定义使开发者能够使用`lambda`函数中没有的三个强大工具:

*   函数注释(类型提示)
*   文档字符串
*   单元测试，通过 doctest 内部的 doctest

多亏了它们，我们可以让`def`函数提供更多信息。

感谢阅读这篇文章。我希望你喜欢它。

我真的希望我已经说服了你。即使这两种类型的函数定义对你来说都一样好——即使这样，我还是会*而不是*使用命名的`lambda`定义。这是因为使用它们，你什么也得不到，同时也冒着别人不同意你的风险。如果连这都不能说服你，记住这样做是违背 PEP8 的。

那么，在给一个函数命名时，为什么要使用命名的`lambda`定义呢？

# 资源

 [## PEP 8 风格的 Python 代码指南

### 本文档给出了 Python 代码的编码约定，包括主 Python 中的标准库…

peps.python.org](https://peps.python.org/pep-0008/) [](https://betterprogramming.pub/pythons-type-hinting-friend-foe-or-just-a-headache-73c7849039c7) [## Python 的类型暗示:朋友，敌人，还是只是头痛？

### 类型提示在 Python 社区越来越受欢迎。这将把我们引向何方？我们能做些什么来使用它…

better 编程. pub](https://betterprogramming.pub/pythons-type-hinting-friend-foe-or-just-a-headache-73c7849039c7) [](/lambda-functions-with-practical-examples-in-python-45934f3653a8) [## Python 中的 Lambda 函数及其实例

### 如何、何时使用以及何时不使用 Lambda 函数

towardsdatascience.com](/lambda-functions-with-practical-examples-in-python-45934f3653a8)  [## doctest -测试交互式 Python 示例- Python 3.11.0 文档

### 源代码:Lib/doctest.py 检查模块的文档字符串是否是最新的

docs.python.org](https://docs.python.org/3/library/doctest.html) [](https://medium.com/@nyggus/membership) [## 加入我的介绍链接媒体-马尔钦科萨克

### 阅读马尔钦·科萨克(以及媒体上成千上万的其他作家)的每一个故事。您的会员费直接支持…

medium.com](https://medium.com/@nyggus/membership)