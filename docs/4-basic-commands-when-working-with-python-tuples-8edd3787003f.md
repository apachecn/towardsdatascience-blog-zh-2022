# 使用 Python 元组时的 4 个基本命令

> 原文：<https://towardsdatascience.com/4-basic-commands-when-working-with-python-tuples-8edd3787003f>

## 让您了解 Python 元组的特征以及如何处理它们

![](img/da8638a845cbaa748052f3742a457895.png)

照片由[华盛顿·奥利维拉·🇧🇷](https://unsplash.com/@washingtonolv?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

Python 元组用于在一个变量中存储多个值。它是预装在 [Python](https://databasecamp.de/en/python-coding) 中的四种数据结构之一。除了元组之外，这些还包括[字典](https://databasecamp.de/en/python-coding/python-dictionarys)，集合，以及[列表](https://databasecamp.de/en/python-coding/python-lists)。

元组是有序的，不能更改。一方面，这意味着元素具有特定且恒定的顺序。由于顺序固定，元组也允许重复。另一方面，元组在被定义后不能被改变。因此，不能删除或添加任何元素。

# 为什么了解不同的数据类型如此重要？

Python 总共有四种不同的数据类型，它们已经存在于基本安装中，因此不能仅通过安装模块来使用，例如 Panda 的 DataFrames。这些可以用于各种各样的用例，也是模块中使用的许多其他数据类型的基础。

Python 中的四种基本数据类型是:

*   [**列表**](https://databasecamp.de/en/python-coding/python-lists) 是元素的有序集合，是可变的，也可以包含重复的元素。
*   **元组**实际上是一个列表，不同之处在于它不再是可变的。因此以后不能添加或删除任何元素。
*   **集合**不允许重复输入。同时，集合中元素的排列是可变的。集合本身可以更改，但是单个元素以后不能更改。
*   从 Python 版本开始，一个 [**字典**](https://databasecamp.de/en/python-coding/python-dictionarys) 就是可以改变的元素的有序集合。在早期版本中，字典是无序的。

# 1.定义一个元组

我们可以通过在圆括号中定义元素并用逗号分隔它们来创建 Python 元组。具有不同[数据类型](https://databasecamp.de/en/data/data-types)的元素可以毫无问题地存储在一个元组中。

# 2.查询元素

由于元组的顺序，我们可以借助索引从元组中检索单个元素。应该注意的是，元素的计数从 0 开始。另一方面，如果我们想从末尾检索一个值，我们从 1 开始计数。

如果还不知道某个元素的索引，可以用“index”的方法去求。因为 Python 元组中的顺序不变，所以该值也保持不变。

# 3.更改元素

我们已经知道，元组实际上是不可变的。也就是说，一旦我们定义了一个元组，就不能再添加或删除元素了。

为了能够改变元组，我们使用了一个小技巧。我们首先将元组转换成一个 [Python](https://databasecamp.de/en/python-coding/python-lists) 列表。由于这是可变的，我们可以简单地在这里添加或删除元素。然后我们将列表转换回一个元组。这样，我们间接地改变了 Python 元组的元素。

# 4.合并元组

如果我们想要合并两个或更多的元组，我们可以简单地使用“+”操作符。因此，第一个命名元组的顺序在第二个命名元组之前。

# 这是你应该带走的东西

*   Python 元组是 Python 中预装的四种数据结构之一。
*   它用于在单个变量中存储多个值。
*   元组创建后不能修改。它也是有序的，这意味着值具有预定义的顺序。

*如果你喜欢我的作品，请在这里订阅*[](https://medium.com/subscribe/@niklas_lang)**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！此外，媒体允许你每月免费阅读三篇文章***。如果你想让***无限制地访问我的文章和数以千计的精彩文章，不要犹豫，通过点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$ ***5*** 获得会员资格***

**[](/an-introduction-to-tensorflow-fa5b17051f6b)  [](/why-do-we-use-xml-in-data-science-99a730c46adb)  [](/redis-in-memory-data-store-easily-explained-3b92457be424) **