# Python 中的异常处理

> 原文：<https://towardsdatascience.com/exception-handling-in-python-8cc8f69f16ad>

## 了解如何使用 Python Try Except

![](img/ef33826bbfcbc3296f343ff9b0448bc8.png)

穆斯塔法·梅拉吉在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

Python Try Except 是一种在 [Python](https://databasecamp.de/en/python-coding) 程序中处理所谓异常的方法，这样应用程序就不会崩溃。借助 Python Try Except，可以捕捉和处理异常。

# 语法错误和异常有什么区别？

我们可能都不得不痛苦地学习，一旦出现错误或异常， [Python](https://databasecamp.de/en/python-coding) 就会终止一个正在运行的程序。这里必须做一个重要的区分。代码可能由于语法错误而停止，即代码只是被错误地编写并且不能被解释，或者由于异常而停止，即语法正确的代码组件在执行期间引起问题。

例如，错误地设置括号可能会导致语法错误:

另一方面，异常是在应用程序执行期间发生的错误，即使代码在语法上是正确的。例如，当试图将一个字符串和一个数字相加时，就会发生这种情况。代码本身写得很正确，但是这个操作的实现是不可能的:

# Python 中的异常类型有哪些？

在 [Python](https://databasecamp.de/en/python-coding) 中，有许多不同类型的异常可能在执行代码时发生。下面的列表描述了最常见的异常类型，但不是全部:

*   **AssertionError** :当命令“assert”没有被正确使用或产生错误时，该异常发生。
*   **ImportError** :如果导入模块有问题，会出现导入错误。例如，如果模块(如 [Pandas](https://databasecamp.de/en/python-coding/pandas-introduction-1) )尚未安装，要从模块加载不正确或不存在的功能，或者指定名称的模块根本不存在，就会发生这种情况。
*   **IndexError** :当使用 [Python](https://databasecamp.de/en/python-coding) 有索引的数据对象时，比如 [Python 元组](https://databasecamp.de/en/python-coding/python-tuples)或者 [Python 列表](https://databasecamp.de/en/python-coding/python-lists)，如果使用了在对象中找不到的索引，就会发生 IndexError。
*   **KeyError** :类似于 IndexError，在使用 [Python 字典](https://databasecamp.de/en/python-coding/python-dictionarys)时，如果在字典对象中找不到某个键，就会发生 KeyError。
*   **内存错误**:当机器内存不足以继续运行 [Python](https://databasecamp.de/en/python-coding) 程序时，会出现内存错误。
*   **unboundlocalrerror**:使用局部变量时，一旦引用了尚未定义的变量，即尚未赋值的变量，就会发生 unboundlocalrerror。

# Python 除了工作如何尝试？

Python 的 Try Except 功能通过定义一个精确描述如何处理异常的例程，使得有针对性地处理可能的异常成为可能。如果出现异常的概率非常高，或者要不惜一切代价防止程序中断，这就特别有用。

为此使用了两个块:Try 和 Except。Try 块用于添加在执行过程中可能导致异常的代码。如果这个代码块运行没有问题，下面的 Except 块将被跳过，代码将在后面执行。但是，如果 Try 块中有异常，Except 块中的代码会自动执行。

在这种情况下，总是执行 a 和 b 的加法，除非为两个没有编号的变量传递值。那么实际上会有一个 ValueError，但这是由我们的 Python Try Except 循环捕获的。文本“a、b 或两者都不是数字”,而不是值 Error。请用不同的值重试。

然而，在 Python 的 Try Except 循环中，您不必指定要响应的特定异常，也可以将其定义为对任何异常执行 Except 块。此外，可以使用“finally”定义一个例程，以防程序没有出现异常。

# 在哪些应用中使用 Python Try 有意义，除了？

在数据科学中，在许多应用程序中使用 Python Try Except 循环来避免过早终止程序是有意义的:

*   **准备大型数据集**:如果你想为[机器学习](https://databasecamp.de/en/machine-learning)准备大型数据集，用作训练数据集，准备工作通常需要几个小时。根据数据集的质量，并非所有的[数据类型](https://databasecamp.de/en/data/data-types)都与给定的匹配。同时，你要避免程序中途停止而你没有注意到。为此，您可以使用 Python Try Except 循环简单地跳过数据质量不正确的单个记录，这将导致异常。
*   软件的日志记录:这里也需要 Python 的 Try Except 循环，因为生产应用程序应该继续工作。使用 Except 块，可以将错误输出到日志中，然后进行评估。在新版本中修复错误后，可以部署新状态，并确保应用程序的停机时间尽可能低。

# 这是你应该带走的东西

*   Python Try Except 是一种处理 Python 程序中所谓异常的方法，这样应用程序就不会崩溃。
*   Try 块包含了可能导致异常的行。另一方面，Except 块定义了在发生错误时应该执行的代码。如果没有异常发生，Python Try Except 循环之后的代码将被简单地执行。
*   此外，您可以使用“finally”命令来定义如果 Try 块中没有错误发生时应该发生什么。

*如果你喜欢我的作品，请在这里订阅*[](https://medium.com/subscribe/@niklas_lang)**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！另外，媒体允许你每月免费阅读* ***3 篇文章*** *。如果你想让***无限制地访问我的文章和数以千计的精彩文章，请不要犹豫，通过点击我的推荐链接:*[https://medium.com/@niklas_lang/membership](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格***

**[](/5-basic-commands-for-working-with-python-lists-3088e57bace6)  [](/6-fundamental-questions-when-working-with-a-pandas-series-1d142b5fba4e)  [](/6-pandas-dataframe-tasks-anyone-learning-python-should-know-1aadce307d26) **