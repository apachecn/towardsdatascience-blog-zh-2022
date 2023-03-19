# Python f-strings 比您想象的更强大

> 原文：<https://towardsdatascience.com/python-f-strings-are-more-powerful-than-you-might-think-8271d3efbd7d>

## 了解 Python 的 f 字符串(格式化的字符串文字)的未知特性，并提升您的文本格式化知识和技能

![](img/b3ce34e064c854042ca1e1acf04f1ff6.png)

照片由[阿玛多·洛雷罗](https://unsplash.com/@amadorloureiro?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

格式化字符串文字——也称为 *f 字符串*——从 Python 3.6 开始就已经存在，所以我们都知道它们是什么以及如何使用它们。然而，有一些事实和 f 弦方便的特点，你可能不知道。所以，让我们来看看一些令人敬畏的 f-string 特性，你会想在你的日常编码中使用它们。

# 日期和时间格式

用 f 字符串应用数字格式很常见，但是您知道您还可以格式化日期和时间戳字符串吗？

f 字符串可以格式化日期和时间，就像你使用`datetime.strftime`方法一样。当你意识到有更多的格式，而不仅仅是文档中提到的几种格式时，这就更好了。Python 的`strftime`也支持底层 C 实现所支持的所有格式，这可能因平台而异，这也是为什么在文档中没有提到它的原因。也就是说，你可以利用这些格式，例如使用`%F`，它相当于`%Y-%m-%d`或`%T`，它相当于`%H:%M:%S`，同样值得一提的是`%x`和`%X`，它们分别是本地首选的日期和时间格式。这些格式的使用显然不仅限于 f 字符串。有关格式的完整列表，请参考 [Linux 联机帮助页](https://manpages.debian.org/bullseye/manpages-dev/strftime.3.en.html)。

# 变量名和调试

f-string 特性(从 Python 3.8 开始)最近增加的一项功能是打印变量名和值的能力:

该功能称为*“调试”*，可与其他修改器结合使用。它还保留了空格，所以`f"{x = }"`和`f"{x=}"`会产生不同的字符串。

# 字符串表示

当打印类实例时，默认使用该类的`__str__`方法进行字符串表示。然而，如果我们想强制使用`__repr__`，我们可以使用`!r`转换标志:

我们也可以只在 f 字符串中调用`repr(some_var)`，但是使用转换标志是一个很好的原生且简洁的解决方案。

# 性能优越

强大的功能和语法糖经常伴随着性能损失，然而，当涉及到 f 字符串时，情况并非如此:

上面的例子是用`timeit`模块测试的，比如:`python -m timeit -s 'x, y = "Hello", "World"' 'f"{x} {y}"'`，正如你所看到的，f 字符串实际上是 Python 提供的所有格式化选项中最快的。因此，即使您更喜欢使用一些旧的格式选项，您也可以考虑切换到 f 字符串来提高性能。

# 格式化规范的全部功能

f 字符串支持 Python 的[格式规范迷你语言](https://docs.python.org/3/library/string.html#formatspec)，所以你可以在它们的修饰符中嵌入很多格式化操作:

Python 的格式化迷你语言不仅仅包括格式化数字和日期的选项。它允许我们对齐或居中文本，添加前导零/空格，设置千位分隔符等等。显然，所有这些不仅适用于 f 字符串，也适用于所有其他格式选项。

# 嵌套 F 字符串

如果基本的 f 字符串不能满足您的格式需求，您甚至可以将它们嵌套在一起:

您可以将 f 字符串嵌入到 f 字符串中，以解决棘手的格式问题，比如向右对齐的浮点数添加美元符号，如上所示。

如果需要在格式说明符部分使用变量，也可以使用嵌套的 f 字符串。这也可以使 f 弦更具可读性:

# 条件格式

在上述嵌套 f 字符串示例的基础上，我们可以更进一步，在内部 f 字符串中使用三元条件运算符:

这可能会很快变得非常不可读，所以你可能想把它分成多行。

# λ表达式

如果你想突破 f 字符串的限制，并让阅读你代码的人感到愤怒，那么——稍加努力——你也可以使用 lambdas:

在这种情况下，lambda 表达式两边的括号是强制的，因为有了`:`，f-string 就会解释它。

# 结束语

正如我们在这里看到的，f 弦真的非常强大，比大多数人想象的要多得多。然而，这些*【未知】*特性中的大多数在 Python 文档中都有提及，所以我建议不要只是通读 f 字符串的文档页面，还要通读您可能正在使用的 Python 的任何其他模块/特性。钻研文档通常会帮助你发现一些非常有用的特性，这些特性是你在钻研 *StackOverflow* 时不会发现的。

*本文原帖*[*martinheinz . dev*](https://martinheinz.dev/blog/70?utm_source=medium&utm_medium=referral&utm_campaign=blog_post_70)

[成为会员](https://medium.com/@martin.heinz/membership)阅读 Medium 上的每一个故事。**你的会员费直接支持我和你看的其他作家。**你还可以在媒体上看到所有的故事。

[](https://medium.com/@martin.heinz/membership)  

你可能也喜欢…

[](/ultimate-ci-pipeline-for-all-of-your-python-projects-27f9019ea71a)  [](/optimizing-memory-usage-in-python-applications-f591fc914df5) 