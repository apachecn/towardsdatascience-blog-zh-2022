# 使用 Python 列表的 5 个基本命令

> 原文：<https://towardsdatascience.com/5-basic-commands-for-working-with-python-lists-3088e57bace6>

## 让您了解 Python 列表的特征以及如何处理它们

![](img/96ab952135dd29bb96c2ae524be10cfb.png)

格伦·卡斯滕斯-彼得斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

Python 列表用于在单个变量中存储多个项目。它们是预装在 [Python](https://databasecamp.de/en/python-coding) 中的总共四种数据结构之一。除了列表，它们还包括[元组](https://databasecamp.de/en/python-coding/python-tuples)、[集合](https://databasecamp.de/en/python-coding/python-sets)和[字典](https://databasecamp.de/en/python-coding/python-dictionarys)。

# 1.定义列表

我们通过将元素写在方括号中来定义一个 [Python](https://databasecamp.de/en/python-coding) 列表。我们可以在一个列表中存储不同数据类型的元素。创建列表的第二种方法是调用“list()”函数。为此，元素必须写在双圆括号中。然而，这种书写方式很少使用。

# 2.查询列表的元素

一个 [Python](https://databasecamp.de/en/python-coding) 列表的元素有一个决定每个元素顺序的索引。列表中的单个元素可以通过它们对应的索引来调用。这里当然要注意，我们的电脑从 0 开始计数。所以列表的第一个元素的索引为 0。

使用索引还可以调用列表的最后一个元素，而不知道它的长度以及具体的索引。在负索引的帮助下，我们可以从后面开始调用链表中的对象。使用这种计数方法，我们从 1 开始，而不是从 0 开始。

如果我们不仅想查询列表中的单个元素，还想查询整个范围，我们可以通过指定开始和结束索引来定义它。应该注意，开始索引是输出的一部分，而结束索引不是。

这里值得注意的是，列表范围的查询总是返回一个列表作为结果，即使 [Python](https://databasecamp.de/en/python-coding) 列表只有一个元素。

该查询也可以在不指定结束索引的情况下执行。这有两种可能性:

在第一个选项中，方括号包含起始索引，然后是冒号。在我们的示例中，这将返回列表中的第二个元素以及第二个元素之后的所有其他元素，无论后面还有多少个元素，即“Tokyo”、“Montreal”和“Berlin”。

在第二个变体中，方括号首先包含一个冒号，然后是初始索引。在这种情况下，我们从 Python 列表中获取列表第三个元素之前的所有元素，即“纽约”和“东京”。

作为这些类型查询的辅助，可以使用下面的小算法:

1.  将光标放在前面(！)的元素定义的数字。
2.  如果冒号在数字后面，即在数字的右边，那么结果由一个列表组成，所有元素都在光标的右边。如果冒号在数字的前面，即在数字的左边，那么结果由光标左边的所有元素的列表组成。

对于我们的第一个查询“list_1[1:]”，我们将光标放在第二个元素的前面，即“Tokyo”的前面。因为冒号在 1 的右边，所以我们必须使用结果中光标右边的所有元素。因此，结果由包含元素“东京”、“蒙特利尔”和“柏林”的 Python 列表组成。

# 3.更改列表项目？

如果我们想改变一个 [Python](https://databasecamp.de/en/python-coding) 列表中的单个或多个元素，我们可以像上面描述的那样调用它们，并简单地重新定义它们的值。

同时，我们还可以使用“insert()”方法在列表中的任意位置插入一个元素，而无需更改任何现有条目。以下值的索引将相应地增加 1。如果我们只想将一个元素追加到列表的末尾，我们可以用“append()”方法来实现。

# 4.从列表中删除元素

当然，我们也可以从一个 [Python](https://databasecamp.de/en/python-coding) 列表中删除值，而不是覆盖它们。为此，有“pop()”和“remove()”方法。这两种方法的区别在于“pop()”将索引作为输入,“remove()”将具体的元素作为输入。

# 5.对列表进行排序

虽然列表中的元素是排序的，但是您可以按字母顺序或数字顺序对它们进行排序。但是，这改变了它们的索引，即列表中的顺序:

Python 列表中的顺序也可以颠倒，即从 Z 到 A 或从大到小排序。为此，您只需要额外的参数“reverse = True”:

然而，只有当 Python 列表由具有统一的[数据类型](https://databasecamp.de/en/data/data-types)的数据组成时，排序算法才有效。数字和字符串的混合会导致所谓的“类型错误”:

# 为什么这对数据科学家很重要？

Python 列表对许多数据科学家来说非常重要，因为它们可以用作大量数据操作的初始起点。因为列表可以在一个中存储不同类型的数据，所以它也可以在数据清理之前用于缓存大量数据。

数据科学家的主要优势是允许重复。例如，对于字典或集合来说，这是不可能的。使用 Python 元组，您还可以存储副本，但是元组在定义后不能更改，而列表可以。因此，Python 列表也是处理可能出现重复的数据的最方便的数据结构。

# 这是你应该带走的东西

*   Python 列表可用于在 Python 变量中存储项目集合。
*   该列表是有序的，并使用索引来引用其元素。
*   它能够像 Python 元组一样存储副本。然而，Python 列表可以改变，而元组不能。
*   对于数据科学家来说，这是一个非常有用的工具，因为它可以用于数据清理和操作。

*如果你喜欢我的作品，请在这里订阅*<https://medium.com/subscribe/@niklas_lang>**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，medium 允许你每月免费阅读* ***3 篇*** *。如果你希望有****无限制的*** *访问我的文章和数以千计的精彩文章，不要犹豫，点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格**

*</6-fundamental-questions-when-working-with-a-pandas-series-1d142b5fba4e>  </8-machine-learning-algorithms-everyone-new-to-data-science-should-know-772bd0f1eca1>  </4-basic-commands-when-working-with-python-tuples-8edd3787003f> *