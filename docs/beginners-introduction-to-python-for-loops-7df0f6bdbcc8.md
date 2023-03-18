# Python for-loops 入门

> 原文：<https://towardsdatascience.com/beginners-introduction-to-python-for-loops-7df0f6bdbcc8>

## Python 和 Loops 新手的基本命令

![](img/9a5dd16d18b525179ff1cea34ce8dbf3.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由[Tine ivani](https://unsplash.com/@tine999?utm_source=medium&utm_medium=referral)拍摄的照片

例如，Python for-loop 用于动态迭代一系列对象，并使用它们执行计算。它自动处理不同长度的物体，因此可以节省[编程](https://databasecamp.de/en/ml-blog/learn-coding)的工作。

# 为什么需要 for 循环？

在 [Python](https://databasecamp.de/en/python-coding) 编程语言中，for 循环主要用于处理 [Python 列表](https://databasecamp.de/en/python-coding/python-lists)，以便改变[列表](https://databasecamp.de/en/python-coding/python-lists)中的对象或者能够将它们用于另一个计算。使用循环的优点是不需要知道[列表](https://databasecamp.de/en/python-coding/python-lists)的长度或对象的单个索引。

Python 中的 for 循环与其他编程语言中的 for 循环有很大不同。在这些情况下，它仅用作 while 循环的替代，即只要满足条件就执行计算。另一方面，在 [Python](https://databasecamp.de/en/python-coding) 中，它专门用于处理对象，尤其是列表。

# Python for-loop 的语法是什么？

for 循环的结构总是相对相似，以单词“for”开头。接下来是变量名，它的值会随着每次运行而改变。在我们的例子中，我们称之为“变量”,并贯穿对象“对象”。在每一次传递中，变量取当前队列中元素的值。单词“in”分隔变量名和被遍历对象的名称:

这一行以冒号结尾，下一行以缩进开始。在 for 语句后缩进的每一行都在一次循环中执行。在这里，可以执行计算，或者像在我们的示例中一样，可以输出列表中的元素:

# range()函数是什么？

如果您不想使用 for 循环对一个具体对象进行迭代，而只是为了多次执行一个命令，那么就使用 range()函数。它代替了对象名，并定义了一个被迭代的值的范围。此功能有三种规格:范围(开始、停止、步进)。因此，您可以指定函数应该从哪个数字开始，默认情况下是 0。此外，您可以指定函数应该在哪个值处停止，以及步长有多大，这里默认为 1。

如果只向 range()函数传递一个值，那么这个值将自动成为起始值为 0、步长为 1 的停止值。

通过传递两个值，可以设置起始值和终止值:

# 如何形成更复杂的 for 循环？

如前所述，在 for 循环之后缩进的所有代码行都在每一次循环中执行。这允许您映射更复杂的关系，并包含 if-else 循环，例如:

# 如何把 for-loop 写成一行？

到目前为止，我们已经了解到 Python for-loop 在冒号后总是有一个换行符，并在下一行继续缩进。为了使代码更紧凑或者节省时间，也可以在一行中编写一个 Python for-loop，这取决于它的复杂程度。上面的例子输出 2 到 5 之间的数字，可以很容易地写成一行:

对于更复杂的 for 循环，一行中的表示也可能很快变得混乱，如我们的偶数和奇数示例所示:

特别优雅的是使用 for 循环创建一个[列表](https://databasecamp.de/en/python-coding/python-lists)或一个[字典](https://databasecamp.de/en/python-coding/python-dictionarys)，这通常也在一行中定义:

如果你要用一个“常规”Python for-loop 来解决这个问题，你必须首先创建一个空的[列表](https://databasecamp.de/en/python-coding/python-lists)，然后一点一点地填充它。这要麻烦得多:

# 在 for 循环中，break 和 continue 做什么？

Python for 循环不应该总是运行到循环的末尾。在某些情况下，提前结束序列也是有意义的。为此，您可以使用“中断”命令。在下面的示例中，只要当前数字大于 4，循环就会中断。

命令“continue”是“break”的反义词，它使循环继续运行。如果您没有针对某个条件执行的直接命令，但只想在循环中开始一轮，这种方法尤其有意义。

在下面的例子中，我们只想输出大于 4 的数字。为此，我们使用“continue”跳过所有小于或等于 4 的值。这将只输出 5 以上的数字:

# 枚举在循环中是如何工作的？

在“枚举”的帮助下，你不仅可以遍历一个对象的元素，比如一个[列表](https://databasecamp.de/en/python-coding/python-lists)，还可以同时获得相应元素的索引。这是有意义的，例如，如果你想直接改变一个[列表](https://databasecamp.de/en/python-coding/python-lists)的元素。

假设我们要将“数字”[列表](https://databasecamp.de/en/python-coding/python-lists)中的每个奇数乘以 2。为此，我们使用“enumerate”遍历“numbers”对象，如果元素是奇数，则更改[列表](https://databasecamp.de/en/python-coding/python-lists)中的数字。这里需要注意的是，我们现在必须分配两个名称，因为每个迭代步骤都有两个变量，即索引和元素本身。

在这里，重要的是要理解对正在迭代的对象的更改对 Python for-loop 没有影响。它仍然像第一次那样看着物体。否则，下面的命令将导致一个无限循环，因为“numbers”元素在每次循环中的长度都是以前的两倍。然而，Python for-loop 只有九个步骤，因为在循环的开始，对象“numbers”只有九个元素。

# 这是你应该带走的东西

*   Python for-loop 用于动态遍历对象的元素。
*   在“中断”的帮助下，你可以提前结束循环。
*   命令“enumerate”不仅返回每一轮中的一个元素，还返回该元素在对象中的索引。例如，这使得从循环内改变对象成为可能。

*如果你喜欢我的作品，请在这里订阅*[](https://medium.com/subscribe/@niklas_lang)**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，medium 允许你每月免费阅读* ***3 篇*** *。如果你希望有****无限制的*** *访问我的文章和数以千计的精彩文章，不要犹豫，点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格**

*[](/3short-introduction-to-numpy-3a65ec23eaba) [## NumPy 简介

### 数字图书馆和 ufuncs 的一些基本知识

towardsdatascience.com](/3short-introduction-to-numpy-3a65ec23eaba) [](/5-basic-commands-when-working-with-python-sets-875f71dcc85b) [## 使用 Python 集合时的 5 个基本命令

### 让您了解 Python 列表的特征以及如何处理它们

towardsdatascience.com](/5-basic-commands-when-working-with-python-sets-875f71dcc85b) [](/exception-handling-in-python-8cc8f69f16ad) [## Python 中的异常处理

### 了解如何使用 Python Try Except

towardsdatascience.com](/exception-handling-in-python-8cc8f69f16ad)*