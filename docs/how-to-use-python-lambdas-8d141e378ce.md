# 如何使用 Python Lambdas

> 原文：<https://towardsdatascience.com/how-to-use-python-lambdas-8d141e378ce>

## Python 中 Lambda 函数的介绍

![](img/b320cebb1f3a46b97368ef786099f8bb.png)

[亨利&公司](https://unsplash.com/@hngstrm?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

Python lambdas 是所谓的匿名函数，使用它可以快速定义有多个输入但只有一个输出的函数。这样的匿名函数不仅在 [Python](https://databasecamp.de/en/python-coding) 中使用，在其他编程语言如 Java、C#或 C++中也使用。

# Python 中普通函数是如何定义的？

在 [Python](https://databasecamp.de/en/python-coding) 中，函数的定义用“def”标记。然后定义其名称，带名称的参数列在圆括号中。以下名为“sum”的函数只接受两个参数并将它们相加:

这些函数的优点是可以通过唯一的名称调用它们。这有助于更好地描述功能，并在以后再次引用该功能。此外，参数还可以有名称来帮助更好地理解结果。另一个优点是显式定义的函数也可以输出多个参数。

例如，我们的初始函数除了输出和之外，还可以输出两个参数之间的差:

# 什么是匿名函数？

然而，在 [Python](https://databasecamp.de/en/python-coding) 中还有第二种定义函数的方式。在所谓的匿名函数的帮助下，这些可以用几行来定义，通常甚至只有一行。这些匿名函数也存在于其他编程语言中，比如 Java 或 C#。在 [Python](https://databasecamp.de/en/python-coding) 中，参数“lambda”用于此，这也是为什么 [Python](https://databasecamp.de/en/python-coding) 中的匿名函数经常被简单地称为 Python lambdas。

这些函数可以很容易地用参数“lambda”、变量的命名以及函数应该计算的表达式来定义:

这个例子也清楚地说明了为什么 Python lambdas 是匿名函数:函数本身不能被赋予名称，它只能存储在一个变量(“function”)中。

因此，如果我们想使用 Python Lambdas 从最初的示例中重新创建 sum 函数，如下所示:

如您所见，可以向 Python Lambdas 传递多个参数。但是，我们不能用 Python lambdas 重新创建函数“sum_difference ”,因为匿名函数只能输出一个结果。因此，必须定义两个不同的函数并调用两次:

# 为什么应该使用 Python Lambdas？

在一些情况下，使用 Python lambdas 会很有用:

*   当定义了只有一个输出和少量输入的简单函数时。在大型项目中，为了节省空间和避免不必要的混乱，不明确定义这样的函数是有意义的。此外，还可以避免继承等问题。在广泛的课堂上。
*   如果函数只使用一次，同样的论点也是有效的。那么没有显式定义也可以，因为无论如何没有人必须再次访问该函数。
*   此外，匿名函数可以确保函数的内容更快更容易理解，因为它是在一行中定义的。显式定义函数会导致失去一些可理解性。
*   Python 中有一些函数，比如“filter”或者“map”，是把函数作为输入的。所以在这些情况下，使用 Python Lambda 是有意义的。

# filter()函数如何与 Python lambdas 配合使用？

函数“filter()”可用于过滤满足特定条件的[列表](https://databasecamp.de/en/python-coding/python-lists)。结果是一个新的[列表](https://databasecamp.de/en/python-coding/python-lists)，它只包含那些满足条件的元素。如果没有显式函数，您也可以按如下方式求解:

函数“filter()”总共有两个输入，首先是用于过滤的条件，其次是需要过滤的输入列表。这个过滤条件必须是一个函数，这就是 Python lambdas 适合这个的原因:

对于这样的应用程序，Python Lambdas 当然也是一个最佳选择。

# 这是你应该带走的东西

*   匿名函数用于快速定义不需要名字的函数。
*   如果函数只使用一次，或者如果您希望保持变量和函数的命名空间较小，这将非常有用。
*   在 Python 中，匿名函数与参数“lambda”一起使用，这就是为什么 Python 中的匿名函数也被称为 Python lambdas。

*如果你喜欢我的作品，请在这里订阅*[](https://medium.com/subscribe/@niklas_lang)**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，medium 允许你每月免费阅读* ***3 篇*** *。如果你希望有****无限制的*** *访问我的文章和数以千计的精彩文章，请不要犹豫，点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格**

*[](/4-basic-sql-commands-every-new-data-scientist-should-know-ba02e40bfc1a)  [](/beginners-introduction-to-python-for-loops-7df0f6bdbcc8)  [](/4-basic-commands-when-working-with-python-dictionaries-1152e0331604) *