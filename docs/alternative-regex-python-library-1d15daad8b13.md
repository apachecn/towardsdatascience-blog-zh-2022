# 如何用 Python 编写易读、优雅的正则表达式模式

> 原文：<https://towardsdatascience.com/alternative-regex-python-library-1d15daad8b13>

## 用友好的声明性语法编写更容易理解的正则表达式

![](img/3520563cdcc4cce6110e02262ceaf3eb.png)

来自[佩克斯](https://www.pexels.com/photo/white-and-black-wooden-board-963278/)的[大卫·巴图斯](https://www.pexels.com/photo/white-and-black-wooden-board-963278/)的照片

毫无疑问，Regex 是有史以来最有用的文本处理工具。它帮助我们找到文本中的模式，而不是精确的单词或短语。正则表达式引擎也明显更快。

然而，困难的部分是定义一个模式。有经验的程序员可以随时定义它。但是大多数开发人员将不得不花时间搜索和阅读文档。

不管经验如何，每个人都发现阅读别人定义的模式很困难。

这就是 PRegEx 解决的问题。

PRegEx 是一个 Python 库，它使得正则表达式模式更加优雅和可读。现在它是我最喜欢的[清理器 python 代码](https://www.the-analytics.club/python-project-structure-best-practices)库之一。

您可以从 [PyPI 库](https://pypi.org/project/pregex/)安装它。

```
pip install pregex# [Poetry](https://www.the-analytics.club/virtualenv-alternative-for-python-dependency-management) users can install
# poetry add pregex 
```

# 开始编写可读性更强的正则表达式。

这里有一个例子来说明 PRegEx 有多酷。

从地址中提取邮政编码的需求非常普遍。如果地址是标准化的，这并不困难。否则，我们需要使用一些巧妙的技术来提取它们。

美国的邮政编码通常是五位数。此外，一些邮政编码的扩展名可能是由连字符分隔的四位数。

例如，88310 是新墨西哥州的邮政编码。有些人更喜欢使用带有分机 88310–7241 的地理段。

下面是寻找这种模式的典型方法(使用 [re 模块](https://docs.python.org/3/library/re.html))。

使用正则表达式在 Python 中查找我们的邮政编码

这些步骤似乎很简单。然而，如果你要向一个编程新手解释你是如何定义这个模式的，你将不得不做一个小时的演讲。

我也不打算解释了。因为我们有 PRegEx。这是它的 PRegEx 版本。

使用 PRegEx 模块在 Python 中查找我们的邮政编码。

如您所见，这段代码定义和理解都很简单。

该模式有两个部分。第一段应该正好有五个数字，第二段是可选的。此外，第二段(如果有)应该有一个连字符和四个数字。

# 理解子模块以创建更令人兴奋的正则表达式模式。

这里我们使用了 PRegEx 库的几个子模块——类和量词。“类”子模块确定匹配什么，量词子模块帮助指定执行多少次重复。

您可以使用其他类，比如 AnyButDigit 来匹配非数字值，或者 AnyLowercaseLetter 来匹配小写字符串。要创建更复杂的正则表达式模式，您还可以使用不同的量词，比如 OneOrMore、至少、AtMost 或不定。

这是另一个有更多精彩比赛的例子。我们需要找出短信中的电子邮件地址。这很简单。但是除了匹配模式之外，我们还对捕获电子邮件地址的域感兴趣。

使用正则表达式匹配 Python 中的电子邮件地址，并捕获电子邮件域。

在上面的例子中，我们使用了“groups”子模块中的 Capture 类。它允许我们在一场比赛中收集片段，这样你就不必做任何后处理来提取它们。

您经常需要的另一个子模块是操作符模块。它帮助您连接模式或选择一组选项中的任何一个。

这是上面同一个例子的一个稍微修改的版本。

在上面的例子中，我们已经将顶级域名限制为。我们已经使用了 operator 子模块中的“要么”类来构建这个模式。如你所见，它与 thanos@wierdland.err 不匹配，因为它的顶级域名是。呃，“不是”。' com '或' . org '

# 最后的想法

对于有经验的开发人员来说，定义正则表达式可能不是一项艰巨的任务。但即使对他们来说，阅读和理解别人创造的模式也是困难的。对于初学者来说，这两者都令人望而生畏。

此外，正则表达式是一个优秀的文本挖掘工具。任何开发人员或数据科学家几乎肯定会遇到正则表达式的使用。

如果您是 Python 程序员，PRegEx 涵盖了复杂的部分。

> 感谢阅读，朋友！在[**LinkedIn**](https://www.linkedin.com/in/thuwarakesh/)[**Twitter**](https://twitter.com/Thuwarakesh)[**Medium**](https://thuwarakesh.medium.com/)上跟我打招呼。
> 
> 还不是中等会员？请使用此链接 [**成为会员**](https://thuwarakesh.medium.com/membership) 因为，不需要你额外付费，我为你引荐赚取一小笔佣金。