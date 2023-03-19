# pre gex:Python 中普通英语的正则表达式

> 原文：<https://towardsdatascience.com/pregex-regular-expressions-in-plain-english-in-python-4670425d0eb5>

## 用 Python 创建正则表达式从未如此简单

![](img/3216ec6ff6d7054ff0d48f45b8c0c174.png)

由[像素](https://www.pexels.com/@pixabay/)在[像素](https://www.pexels.com/photo/alphabet-close-up-communication-conceptual-278887/)上拍摄的照片

记住正则表达式(regex) [中的元字符并不难](/a-simple-and-intuitive-guide-to-regular-expressions-404b057b1081)，但是构建一个匹配复杂文本模式的元字符有时很有挑战性。

如果我们可以用简单的英语构建正则表达式会怎么样？

现在，您可以使用名为 PRegEx 的 Python 库编写易于理解的正则表达式。这个库可以温和地向初学者介绍 regex 的世界，甚至帮助那些已经知道 regex 的人。

它是这样工作的。

## 安装库

首先，我们需要安装 PRegEx(它需要 Python >= 3.9)

```
pip install pregex
```

## 探索 PRegEx 库

假设我们只想捕获包含姓氏、头衔和以下文本名称的行。

```
text = """
    Here are the full name of some people:
    Smith, Mr. Robert
    Johnson, Ms Mary
    Wilson, Mrs. Barbara
    Taylor, Dr Karen
    Lewis, Mr. John

"""
```

我们可以通过编写以下代码用 PRegEx 解决这个问题。

```
from pregex.core.classes import AnyButWhitespace
from pregex.core.quantifiers import OneOrMore, Optional
from pregex.core.operators import Either

family_name = OneOrMore(AnyButWhitespace())
title = Either("Mrs", "Mr", "Ms", "Dr")
given_name = OneOrMore(AnyButWhitespace())

pre = (
    family_name +
    ', ' +
    title +
    Optional(".") +
    ' ' +
    given_name
)

pre.get_matches(text)
```

以下是输出结果:

```
['Smith, Mr. Robert',
 'Johnson, Ms Mary',
 'Wilson, Mrs. Barbara',
 'Taylor, Dr Karen',
 'Lewis, Mr. John']
```

这是等价的正则表达式模式。

```
>>> pre.get_pattern()
'\\S+, (?:Mrs|Mr|Ms|Dr)\\.? \\S+'
```

让我们看看我们从 pregex 导入的那些元素意味着什么:

*   `AnyButWhitespace`匹配除空白以外的任何字符(相当于 regex 中的`\S`)
*   `OneOrMore`匹配一个字符一次或多次(相当于 regex 中的`+`)
*   `Either`匹配所提供的模式之一(相当于 regex 中的`|`)
*   `Optional`零次或一次匹配一个字符(相当于 regex 中的`?`)

虽然现在我们使用普通英语来匹配文本模式，但我们仍然需要熟悉可以从 pregex 导入的所有元素。

让我们通过一些例子来探索 pregex 库。

## 示例 1:获取正确的日期格式

假设我们希望在下面的文本中获得格式 DD-MM-YYYY。

```
text = """
    04-13-2021
    2021-04-13
    2021-13-04
"""
```

这是我们如何用 pregex 解决的。

```
from pregex.core.classes import AnyDigit
from pregex.core.quantifiers import Exactly

two_digits = Exactly(AnyDigit(), 2) 
four_digits = Exactly(AnyDigit(), 4)

pre = (
    two_digits +
    "-" +
    two_digits +
    "-" +
    four_digits
)

pre.get_matches(text)
```

以下是输出结果:

```
['04-13-2021']
```

这是等价的正则表达式模式。

```
>>> pre.get_pattern()
'\\d{2}-\\d{2}-\\d{4}'
```

让我们看看我们从 pregex 导入的那些元素意味着什么:

*   `AnyDigit`匹配从 0 到 9 的任何数字(相当于 regex 中的`\d`)
*   `Exactly`匹配重复 n 次的精确字符数(类似于 regex 中的`{n}`)

## 示例 2:获得正确的电子邮件格式

假设我们希望在下面的文本中得到正确的电子邮件格式。

```
text = """
    example@python.com
    example@@python.com
    example@python.com.
"""
```

这是我们如何用 pregex 解决的。

```
from pregex.core.classes import AnyButFrom
from pregex.core.quantifiers import OneOrMore, AtLeast
from pregex.core.assertions import MatchAtLineEnd

non_at_sign_space = OneOrMore(AnyButFrom("@", ' '))
non_at_sign_space_dot = OneOrMore(AnyButFrom("@", ' ', '.'))
domain = MatchAtLineEnd(AtLeast(AnyButFrom("@", ' ', '.'), 2))

pre = (
    non_at_sign_space +
    "@" +
    non_at_sign_space_dot +
    '.' +
    domain
)

pre.get_matches(text)
```

以下是输出结果:

```
['example@python.com']
```

这是等价的正则表达式模式。

```
>>> pre.get_pattern()
'[^ @]+@[^ .@]+\\.[^ .@]{2,}$'
```

让我们看看我们从 pregex 导入的那些元素意味着什么:

*   `AnyButFrom`匹配除括号内的字符之外的任何字符(相当于 regex 中的`[^]`)
*   `AtLeast`匹配至少一定数量的重复 n 次的字符(类似于 regex 中的`{n,}`)
*   `MatchAtLineEnd`在行尾断言位置(相当于当标志“多行”打开时 regex 中的`$`)

恭喜你！您已经完成了掌握正则表达式的第一步！如果你想学习如何构建标准的正则表达式，[查看本指南](/a-simple-and-intuitive-guide-to-regular-expressions-404b057b1081)。

用 Python 学习数据科学？ [**通过加入我的 10k+人电子邮件列表，获取我的免费 Python for Data Science 备忘单。**](https://frankandrade.ck.page/26b76e9130)

如果你喜欢阅读这样的故事，并想支持我成为一名作家，可以考虑报名成为一名媒体成员。每月 5 美元，让您可以无限制地访问数以千计的 Python 指南和数据科学文章。如果你使用[我的链接](https://frank-andrade.medium.com/membership)注册，我会赚一小笔佣金，不需要你额外付费。

[](https://frank-andrade.medium.com/membership) 