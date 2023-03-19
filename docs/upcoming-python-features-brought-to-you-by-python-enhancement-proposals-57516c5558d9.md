# Python 增强提案为您带来的即将到来的 Python 特性

> 原文：<https://towardsdatascience.com/upcoming-python-features-brought-to-you-by-python-enhancement-proposals-57516c5558d9>

## 看看最近的 Python 增强提案(pep)以及它们可能带来的所有令人兴奋的新特性、变化和改进

![](img/058c82ba79170bfc40ee57b69454c43a.png)

照片由[张丽](https://unsplash.com/@sunx?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

在任何新特性、变化或改进进入 Python 之前，需要有一个 *Python 增强提议*，也称为 PEP，概述提议的变化。这些 pep 是获取关于即将发布的 Python 版本中可能包含的最新信息的好方法。因此，在这篇文章中，我们将回顾所有将在不久的将来带来一些令人兴奋的新 Python 特性的提案！

# 语法变化

所有这些提议都可以分成几类，第一类是语法改变提议，它肯定会带来有趣的特性。

这个类别中的第一个是 [PEP 671](https://www.python.org/dev/peps/pep-0671/) ，它提出了后期绑定函数参数默认值的语法。那是什么意思呢？

Python 中的函数可以将其他函数作为参数。然而，没有好的方法来设置这些参数的默认值。通常使用`None`或 sentinel 值(全局常量)作为默认值，这有缺点，包括不能在参数上使用`help(function)`。这个 PEP 描述了使用`=>` ( `param=>func()`)符号指定函数作为默认参数的新语法。

对我来说，这种改变看起来合理且有用，但我认为我们应该小心添加太多新的语法符号/改变。像这样的小改进是否保证了另一个赋值操作符是有问题的。

另一个语法改变提议是 [PEP 654](https://python.org/dev/peps/pep-0654/) ，它提议将`except*`作为引发异常组的新语法。这种方法的基本原理是 Python 解释器一次只能传播一个异常，但是当栈展开时，有时需要传播多个不相关的异常。一种这样的情况是来自并发任务的`asyncio`的并发错误或在执行重试逻辑时引发的多个不同异常，例如，当连接到某个远程主机时。

这是使用这个新特性的一个非常简单的例子。如果你看一看 PEP 中的[处理异常组](https://www.python.org/dev/peps/pep-0654/#id38)的例子，你会发现使用它的很多方法，包括递归匹配和链接。

# 打字

下一个类别——在最近的 Python 版本中大量出现——是类型/类型注释。

让我们从 [PEP 673](https://www.python.org/dev/peps/pep-0673/) 开始——它不需要对 Python 的`typing`模块有广泛的了解(通常情况下就是这样)。让我们用一个例子来解释一下:假设你有一个方法为`set_name`的类`Person`，它返回`self`——类型为`Person`的实例。如果你用同样的`set_name`方法创建子类`Employee`，你会期望它返回类型`Employee`的实例，而不是`Person`。然而，这并不是类型检查目前的工作方式——在 Python 3.10 中，类型检查器推断子类中的返回类型是基类的类型。该 PEP 通过允许我们使用带有以下语法的*“Self Type”*来帮助类型检查器正确推断类型，从而解决了这个问题:

如果您遇到了这个问题，那么您可以期待很快使用这个特性，因为这个 PEP 已经被接受，并将作为 Python 3.11 版本的一部分来实现。

另一个类型变化出现在 [PEP 675](https://www.python.org/dev/peps/pep-0675/) 中，标题为*任意文字字符串*。

引入这个 PEP 源于这样一个事实，即当前不可能指定函数参数的类型可以是一个*任意文字*字符串(只能使用特定的文字字符串，例如`Literal["foo"]`)。您可能想知道为什么这甚至是一个问题，为什么有人需要指定参数应该是*文字字符串*，而不是 *f 字符串*(或其他插入的字符串)。这主要是安全问题——要求参数是字面量*有助于避免注入攻击，无论是 SQL/命令注入还是 XSS。PEP 的[附录](https://www.python.org/dev/peps/pep-0675/#appendix-a-other-uses)中显示了一些例子。实现这一点将有助于像`sqlite`这样的库在字符串插值被用在不该用的地方时向用户提供警告，所以让我们期待这一点很快被接受。*

# 排除故障

接下来是 pep，帮助我们更有效地调试代码。从标题为*CPython*的 [PEP 669](https://www.python.org/dev/peps/pep-0669/) 开始。这个 PEP 建议为 CPython 实现低成本监控，在运行调试器或分析器时不会影响 Python 程序的性能。考虑到在进行基本调试时不会有很大的性能损失，这不会对 Python 的最终用户产生很大的影响。然而，这在某些特殊情况下非常有用，例如:

*   调试只能在生产环境中重现的问题，而不会影响应用程序性能。
*   调试竞争条件，计时会影响问题是否会发生。
*   运行基准测试时进行调试。

我个人可能不会从中受益太多，但我相信试图提高 Python 本身性能的人肯定会喜欢这种变化，因为这将使调试和测试性能问题/改进变得更容易。

下一个是 [PEP 678](https://www.python.org/dev/peps/pep-0678/) ，它建议将`__note__`属性添加到`BaseException`类中。该属性将用于保存附加的调试信息，这些信息可以作为回溯的一部分显示。

如上例所示，这在重新引发异常时特别有用。正如 PEP 所描述的，这对于有重试逻辑的库也是有用的，为每次失败的尝试增加额外的信息。类似地，测试库可以利用这一点向失败的断言添加更多的上下文，比如变量名和值。

最后一个与调试相关的提议是 [PEP 657](https://www.python.org/dev/peps/pep-0657/) ，它想给 Python 程序的每个字节码指令添加额外的数据。该数据可用于生成更好的追溯信息。它还建议应该公开 API，这将允许其他工具(如分析器或静态分析工具)使用这些数据。

这听起来可能没什么意思，但实际上——在我看来——这是这里介绍的最有用的激励。这个 PEP 的最大好处肯定是拥有更好的回溯信息，例如:

我认为这对于回溯可读性和调试来说是一个惊人的改进，我真的很高兴这是作为 Python 3.11 的一部分实现的，所以我们很快就会使用它。

# 生活质量改变

最后一个主题是致力于带来某些*“生活质量”改善的 pep。其中之一是 [PEP 680](https://www.python.org/dev/peps/pep-0680/) ，它提议在 Python 的标准库中添加对解析 TOML 格式的支持。*

*默认情况下，TOML 作为一种格式被许多 Python 工具使用，包括构建工具。这给他们制造了一个自举问题。此外，许多流行的工具如`flake8`不包含 TOML 支持，理由是它在标准库中缺乏支持。这个 PEP 提议在标准库的基础上增加 TOML 支持，这个标准库已经被像`pip`或者`pytest`这样的包使用。*

*我个人喜欢这个提议，我认为在标准库中包含通用/流行格式的库是有意义的，特别是当它们对 Python 的工具和生态系统如此重要的时候。问题是，我们什么时候能在 Python 标准库中看到 YAML 支持？*

*最后一点，就是 [PEP 661](https://www.python.org/dev/peps/pep-0661/) ，与所谓的 [*【哨兵值】*](https://python-patterns.guide/python/sentinel-object/#sentinel-value) 有关。在 Python 中没有创建这种值的标准方法。通常用`_something = object()`(普通习语)来完成，如前面的 PEP 671 所示。本 PEP 提出了标准库哨兵值的规范/实施:*

*除了新的解决方案可读性更好之外，这也有助于类型注释，因为它将为所有 sentinels 提供不同的类型。*

# *结束语*

*从上面提出的建议来看，很明显，很多好东西正在向 Python 走来。然而，并不是所有的这些特性都会被纳入 Python(至少在目前的状态下不会)，所以一定要关注这些提议，看看它们会走向何方。为了及时了解上述 PEP 以及任何新添加的内容，您可以偶尔浏览一下[索引](https://www.python.org/dev/peps/#numerical-index)，或者通过订阅 [PEP RSS feed](https://www.python.org/dev/peps/peps.rss/) 获得关于每个新添加内容的通知。*

*此外，我只包括了一些对该语言提出一些新功能/变化的 pep，然而还有其他一些指定最佳实践、过程或烤箱 Python 的[发布时间表](https://www.python.org/dev/peps/pep-0664/)，所以如果你对这些主题感兴趣，请确保查看上述索引。*

**本文最初发布于*[*martinheinz . dev*](https://martinheinz.dev/blog/67?utm_source=medium&utm_medium=referral&utm_campaign=blog_post_67)*

*[](https://python.plainenglish.io/creating-beautiful-tracebacks-with-pythons-exception-hooks-c8a79e13558d)  [](/profiling-and-analyzing-performance-of-python-programs-3bf3b41acd16)  [](/exploring-google-analytics-realtime-data-with-python-8625849c7d7a) *