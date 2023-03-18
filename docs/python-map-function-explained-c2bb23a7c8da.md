# Python map()函数解释

> 原文：<https://towardsdatascience.com/python-map-function-explained-c2bb23a7c8da>

## 在本文中，我们将探索如何使用 Python **map()** 函数

![](img/9542496601f9400fb0b627a437c2bccf.png)

由[安妮·斯普拉特](https://unsplash.com/@anniespratt?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/photos/AFB6S2kibuk?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

**目录**

*   介绍
*   使用 Map()将函数映射到列表上
*   使用 Map()在多个列表上映射一个函数
*   使用 Map()在多个列表上映射 lambda 函数
*   结论

# 介绍

在 Python 中，当我们处理多个可迭代对象([列表](https://pyshark.com/python-list-data-structure/)、[元组](https://pyshark.com/python-tuple-data-structure/)或字符串)时，我们经常需要对可迭代对象中的每个元素应用一个函数。

Python **map()** 函数是一个内置函数，它允许在 iterable 中的每个元素上“映射”或应用特定的函数，并返回一个包含修改元素的新 iterable。

**map()** 功能过程定义为:

```
map(function, iterable) -> map object
```

其中*映射对象*可以被转换成一个 iterable(例如使用 **list()** 构造函数的 list)。

# 使用 Map()将函数映射到列表上

让我们从一个简单的例子开始，使用 **map()** 函数将一个函数应用于一个列表。

我们将定义一个 *square* 函数，该函数将一个数字作为参数并返回其平方值，然后将该函数应用于一系列数字以获得它们的平方值列表。

首先，我们将创建一个数字列表:

```
#Create a list of numbers
nums = [1, 2, 3, 4, 5]
```

接下来，我们将定义一个*平方*函数并测试它:

```
#Define a function to square values
def square(x):
    return x*x

#Test function
squared_val = square(3)
print(squared_val)
```

您应该得到:

```
9
```

现在这个函数工作了，我们可以在 **nums** 列表中应用它。

```
#Square values in nums list
squared_vals = map(square, nums)

#Convert map object to a list
squared_vals = list(squared_vals)

#Print squared values
print(squared_vals)
```

您应该得到:

```
[1, 4, 9, 16, 25]
```

如您所见，我们已经能够快速应用 **nums** 列表上的 *square* 函数来获得平方值。

# 使用 Map()在多个列表上映射一个函数

在上一节中，我们有一个*平方*函数，它接受一个参数并将其应用于一个 **nums** 列表。

在本节中，我们将创建一个带两个参数的函数，并使用 **map()** 函数将它应用于两个列表。

首先，我们将创建两个数字列表:

```
#Create two lists of numbers
nums1 = [1, 3, 5, 7, 9]
nums2 = [2, 4, 6, 8, 10]
```

接下来，我们将定义一个 *add* 函数并测试它:

```
#Define a function to add values
def add(x, y):
    return x+y

#Test function
add_val = add(3, 5)
print(add_val)
```

您应该得到:

```
8
```

现在该函数已经工作，我们可以在 **nums1** 和 **nums2** 列表上应用它。

```
#Add values from two lists
add_vals = map(add, nums1, nums2)

#Convert map object to a list
add_vals = list(add_vals)

#Print added values
print(add_vals)
```

您应该得到:

```
[3, 7, 11, 15, 19]
```

这是将列表 **nums1** 和 **nums2** 中的数字相加后的单个值列表。

# 使用 Map()在多个列表上映射 lambda 函数

在上一节中，我们创建了一个 *add* 函数，它接受两个参数，然后可以应用于两个列表。

但是我们能在不定义一个多参数函数的情况下做同样的操作吗？

我们可以使用一个 [Python **lambda** 函数](https://pyshark.com/python-lambda-functions/)和 **map()** 函数来实现。

让我们重复使用上一节中的数字列表:

```
#Create two lists of numbers
nums1 = [1, 3, 5, 7, 9]
nums2 = [2, 4, 6, 8, 10]
```

现在我们可以在 **nums1** 和 **nums2** 列表上应用一个 **lambda** 函数。

```
#Add values from two lists
add_vals = map(lambda x, y: x + y, nums1, nums2)

#Convert map object to a list
add_vals = list(add_vals)

#Print added values
print(add_vals)
```

您应该得到:

```
[3, 7, 11, 15, 19]
```

这与[前一节](https://pyshark.com/python-map-function/#map-a-function-over-multiple-lists)中的结果相同。

# 结论

在本文中，我们探索了如何使用 [Python **map()** 函数](https://docs.python.org/3/library/functions.html?highlight=map#map)。

现在你已经知道了基本的功能，你可以练习使用它和其他可迭代的数据结构来完成更复杂的用例。

如果你有任何问题或对编辑有任何建议，请随时在下面留下评论，并查看我的更多 [Python 函数](https://pyshark.com/category/python-functions/)教程。

*原载于 2022 年 12 月 24 日 https://pyshark.com**[*。*](https://pyshark.com/python-map-function/)*