# 计算 Python 列表中项目频率的 3 种方法

> 原文：<https://towardsdatascience.com/3-ways-to-count-the-item-frequencies-in-a-python-list-89975f118899>

## 实用指南

![](img/2e5f12af8cf91505b09984e0093bd59d.png)

Ibrahim Rifath 在 [Unsplash](https://unsplash.com/s/photos/count?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

数据结构在编程语言中至关重要。如何存储、管理和操作数据是创建健壮高效的程序的关键因素。

Python 中内置的数据结构之一是[列表](/11-must-know-operations-to-master-python-lists-f03c71b6bbb6)，它被表示为方括号中的数据点集合。列表可用于存储任何数据类型或不同数据类型的混合。

以下是 Python 列表的一个示例:

```
mylist = ["a", "a", "b", "c", "c", "c", "c", "d", "d"]
```

在这篇文章中，我们将学习 3 种计算列表中项目的方法，这对于很多任务来说可能是非常有用的输入。更具体地说，我们将计算项目出现的次数，以获得它们的分布。否则，使用 len 函数可以很容易地计算出项目的总数。

```
len(mylist)
**# output**
9
```

## 计数方法

第一个是 count 方法，它返回给定项目的出现次数。

```
mylist = ["a", "a", "b", "c", "c", "c", "c", "d", "d"]mylist.count("c")
**# output**
4
```

如果您要查找特定的项目，此方法非常有用。要一次获得所有项目的频率，我们可以使用以下两种方法之一。

## 计数器功能

集合模块中的计数器函数可用于查找列表中项目的频率。它是 dictionary 的子类，用于计算可散列对象的数量。

Counter 返回一个 counter 对象，其中项目存储为键，频率存储为值。它非常类似于 Python 字典，这是另一种内置的数据结构。

```
from collections import Countermylist = ["a", "a", "b", "c", "c", "c", "c", "d", "d"]Counter(mylist)
**# output**
Counter({'a': 2, 'b': 1, 'c': 4, 'd': 2})
```

我们可以按如下方式访问项目的出现次数:

```
mycounter = Counter(mylist)mycounter["a"]
**# output**
2
```

counter 函数也可以用来计算字符串中的字符数。这里有一个例子:

```
mystring = "Data science ????"Counter(mystring)
**# output**
Counter({'D': 1,
         'a': 2,
         't': 1,
         ' ': 2,
         's': 1,
         'c': 2,
         'i': 1,
         'e': 2,
         'n': 1,
         '?': 4})
```

## 熊猫重视计数

Pandas 的 value counts 函数返回所有项目及其出现次数。结果，我们得到了列表分布的概况。

因为这是一个熊猫函数，我们首先需要将列表转换成一个系列。

```
import pandas as pdmylist = ["a", "a", "b", "c", "c", "c", "c", "d", "d"]pd.Series(mylist).value_counts()
**# output**
c    4
a    2
d    2
b    1
dtype: int64
```

value counts 函数的输出是一个序列，其索引包含列表中的项目。我们可以按如下方式访问项目的出现次数:

```
myseries = pd.Series(mylist).value_counts()myseries["d"]
**# output**
2
```

我们还可以使用 normalize 参数获得项目的百分比份额。

```
pd.Series(mylist).value_counts(normalize=True)**# output**
c    0.444444
a    0.222222
d    0.222222
b    0.111111
dtype: float64
```

在这个简短的指南中，我们学习了在 Python 中执行一个简单而重要的操作的 3 种不同方法。

感谢您的阅读。如果您有任何反馈，请告诉我。