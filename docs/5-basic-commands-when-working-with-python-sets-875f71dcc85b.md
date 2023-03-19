# 使用 Python 集合时的 5 个基本命令

> 原文：<https://towardsdatascience.com/5-basic-commands-when-working-with-python-sets-875f71dcc85b>

## 让您了解 Python 列表的特征以及如何处理它们

![](img/6ad6249de7ec9dadf9dafe106838dd45.png)

萨姆·麦克格在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

[Python](https://databasecamp.de/en/python-coding) sets 用于在单个变量中存储多个元素。set 主要用于不允许元素多次重复的情况。因此，集合中的元素是唯一的。

# 1.定义 Python 集

Python 集合是通过在花括号中用逗号分隔各个值来定义的。这些元素也可以有不同的数据类型，但仍然存储在同一个集合中。

作为定义 Python 集合的第二种方法，您也可以使用“set”操作符并将元素写在双圆括号中。然而，这种方法很少使用。

# 2.查询 Python 集合中的元素

由于集合是无序的，因此顺序是不相关的，正如我们从 [Python 列表](https://databasecamp.de/en/python-coding/python-lists)或[字典](https://databasecamp.de/en/python-coding/python-dictionarys)中所知，单个元素不能通过索引或键进行查询。剩下的唯一可能性是查询特定元素是否出现在集合中。

# 3.添加元素

现有集合可以通过单个元素进行扩展，甚至可以将整个集合添加到现有集合中。

值得注意的是，值“新元素”在最终集合中只出现一次。因此，如果我们想要向 Python 集合中添加已经存在的值，它们会被忽略。

由于四种[数据类型](https://databasecamp.de/en/data/data-types)在 [Python](https://databasecamp.de/en/python-coding) 中紧密相连，我们也可以连接不同[数据类型](https://databasecamp.de/en/data/data-types)的变量。例如，[列表](https://databasecamp.de/en/python-coding/python-lists)也可以添加到现有的集合中。[列表](https://databasecamp.de/en/python-coding/python-lists)中的重复元素当然在集合中只考虑一次。

# 4.删除元素

如果我们想从 Python 集合中删除元素，也有不同的可能性。使用“删除”或“丢弃”命令，我们可以通过指定某个值来删除元素。另一方面，对于“pop ”,我们删除最后添加到集合中的元素。最后是“明确”。此命令删除集合中的所有元素，并留下一个空集合的变量。

# 5.合并多个集合

当合并多个 Python 集合时，只保留在两个集合中至少出现一次的元素。这发生在所谓的“联合”函数中:

“union”函数会自动创建一个新的对象，除了前面两个对象之外，还会存储一个新的对象。如果不希望这样，也可以显式地将元素添加到现有的 Python 集合中。为此，请使用“更新”功能:

# 这是你应该带走的东西

*   Python Set 是 Python 中预装的四种数据结构之一。
*   它用于在单个变量中存储几个唯一的元素。元素的顺序暂时不重要。
*   Python 集合与数学集合相当，因此可以用它执行相同的函数，比如 union。

*如果你喜欢我的作品，请在这里订阅*<https://medium.com/subscribe/@niklas_lang>**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，medium 允许你每月免费阅读* ***3 篇*** *。如果你希望有****无限制的*** *访问我的文章和数以千计的精彩文章，不要犹豫，点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格**

*</6-pandas-dataframe-tasks-anyone-learning-python-should-know-1aadce307d26>  </4-basic-commands-when-working-with-python-tuples-8edd3787003f>  </an-introduction-to-tensorflow-fa5b17051f6b> *