# 使用 Python 字典时的 4 个基本命令

> 原文：<https://towardsdatascience.com/4-basic-commands-when-working-with-python-dictionaries-1152e0331604>

## 让您了解 Python 字典的特征以及如何处理它们

![](img/106ceecedde79d88839dbb607c4a9dbf.png)

Emmanuel Ikwuegbu 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

[Python](https://databasecamp.de/en/python-coding) 字典用于存储变量中的键值对。它是预装在 [Python](https://databasecamp.de/en/python-coding) 中的总共四种数据结构之一。除了字典，这些还包括[元组](https://databasecamp.de/en/python-coding/python-tuples)、[集合](https://databasecamp.de/en/python-coding/python-sets)和[列表](https://databasecamp.de/en/python-coding/python-lists)。

# Python 字典的基本特征是什么？

从 [Python](https://databasecamp.de/en/python-coding) 版本 3.7 开始，字典被订购。这意味着我们存储键值对的顺序也起着作用。相比之下，在之前的版本中，顺序没有任何意义。此外， [Python](https://databasecamp.de/en/python-coding) 字典也是可修改的，即在它被创建后，可以从字典中修改、添加或删除元素。

Python 字典最重要的特性是不允许重复的键值对。然而，在 [Python](https://databasecamp.de/en/python-coding) 的其他数据格式中，重复元素是允许的。如果我们想将一个键-值对添加到字典中，而字典中的键已经存在，那么旧的键-值对将被覆盖，而不会发出通知。

# 1.定义和查询字典

我们定义了一个 [Python](https://databasecamp.de/en/python-coding) 字典，将键值对写在花括号中，并用冒号分隔。我们可以在一个字典中存储不同数据类型的元素。

我们可以通过指定方括号中的键来查询字典的元素。然后我们得到为这个键存储的相应值。我们可以从字典中查询各种信息和元素。

正如我们已经看到的，我们可以通过在方括号中定义相关的键来查询值。类似地，“get()”方法返回相同的结果:

# 2.获取字典的键和值

用命令”。按键()"和"。values()" [Python](https://databasecamp.de/en/python-coding) 返回给我们一个所有键和值的列表。列表的顺序也对应于它们在字典中的存储方式。这也意味着值列表可能包含重复项。

另一方面，如果我们想要检索完整的键-值对，我们使用“.items()"方法，该方法将对作为元组列表返回:

# 3.更改字典中的元素

如果我们想改变 Python 字典中的单个值，我们可以直接通过键来实现。因为不能有重复的键，所以旧值会被简单地覆盖。如果我们想一次改变多个线对，我们使用“.”。update()"方法，并在其中定义新的键值对。

# 4.删除元素或整个字典

如果我们想从 [Python](https://databasecamp.de/en/python-coding) 字典中删除单个元素，我们可以指定键并使用“pop()”方法专门删除元素，或者使用“popitem()”删除最后添加的键-值对:

最后，您可以用“clear()”方法清除整个 [Python](https://databasecamp.de/en/python-coding) 字典:

# 这是你应该带走的东西

*   Python 字典是 Python 中预装的四种数据结构之一。
*   它用于在单个变量中存储键值对。
*   字典的值可以有不同的值。除了单个标量，列表、元组或新字典也可以存储为值。

*如果你喜欢我的作品，请在这里订阅*[](https://medium.com/subscribe/@niklas_lang)**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！此外，媒体允许你每月免费阅读* ***3 篇*** *。如果你想让***无限制地访问我的文章和数以千计的精彩文章，不要犹豫，通过点击我的推荐链接:*[https://medium.com/@niklas_lang/membership](https://medium.com/@niklas_lang/membership)获得会员资格，每个月只需支付 ***5*****

***[](/6-pandas-dataframe-tasks-anyone-learning-python-should-know-1aadce307d26)  [](/4-basic-commands-when-working-with-python-tuples-8edd3787003f)  [](/an-introduction-to-tensorflow-fa5b17051f6b) ***