# 你应该知道的 4 个 Python 技巧

> 原文：<https://towardsdatascience.com/4-python-hacks-you-should-know-29acbc45addd>

## 数据科学

## 列表上的一堆用例以及字典理解

![](img/b1f5a9e07cbaef9f1bbb38ad364c8dae.png)

[卡洛斯](https://unsplash.com/es/@folkcarlos?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍照

**Python 更好！**

在现实世界中，对于数据科学来说，Python 通常被广泛用于转换数据。它提供了强大的数据结构，使得处理数据更加灵活。

> 我所说的“灵活性”是什么意思？

这意味着，在 Python 中，总是有多种方法可以获得相同的结果。总是选择易于使用、节省时间并能更好地控制转换的方法。

当然，不可能全部掌握。因此，我列出了在处理任何类型的数据时都应该知道的 4 个 Python 技巧。

这里有一个快速索引供你参考。

```
· [Duplicate a List using List Comprehension](#0451)
· [Multiply elements in a list using List Comprehension](#a8b8)
· [Remove negative elements in a List using List Comprehension](#4c90)
· [Convert two lists into Dictionary Key-Value pair using dict()](#11af)
```

先从经典的东西开始。

[**列表理解**](/3-python-tricks-for-better-code-511c82600ee1) 是创建列表的一种优雅且最有技巧的方式。与 for 循环和 if 语句相比，list comprehensions 具有更短的语法来基于现有列表的值创建新列表。因此，让我们看看这个特性如何获得一个列表的副本。

# 使用列表理解复制列表

有时您需要创建现有列表的副本。最简单的答案是`.copy()`,它让你将一个列表的内容复制到另一个(新的)列表中。

例如，假设您有一个由整数组成的列表— `original_list`。

```
**original_list = [10,11,20,22,30,34]**
```

你可以简单地用下面的`.copy()`方法复制这个列表。

```
duplicated_list = **original_list.copy()**
```

然而，使用列表理解方法可以得到完全相同的输出。复制列表是理解列表理解工作的最佳用例。

你需要做的就是执行下面这段代码。

```
duplicated_list = **[item for item in original_list]**
```

当然，决定使用哪个选项是你的选择。知道实现相同结果的多种方法是有益的。

接下来，让我们看看当您想要对列表中的每个元素执行数学运算时，列表理解是如何让您的生活变得简单的。

# 使用列表理解将列表中的元素相乘

乘法最简单或直接的方法是使用乘法运算符，即`*****`

例如，假设您想用一个标量(即数字 5)乘以列表中的每一项。在这里，你可以不做`**original_list*5**`,因为它将简单地创建列表的 5 个副本。

在这种情况下，最好的答案是列表理解，如下所示。

```
original_list = [10,11,20,22,30,34]
multiplied_list = **[item*5 for item in original_list]**# Output
[50, 55, 100, 110, 150, 170]
```

简单！

乘法不仅限于一个数字。相反，您可以对原始列表的每个元素执行复杂的操作。

例如，假设您想计算每一项的平方根的立方。使用列表理解，你可以只用一行就解决它。

```
multiplied_list = **[math.sqrt(item)**3 for item in original_list]**# Output
[31.6227766016838,
 36.4828726939094,
 89.4427190999916,
 103.18914671611546,
 164.31676725154983,
 198.25236442474025]
```

请注意，用于计算数字平方根的函数`**sqrt**`属于库 math，因此在使用它之前需要导入。

类似于上面显示的内置函数，您也可以在列表的每个元素上使用用户定义的函数。

例如，假设您有一个如下所示的简单函数。

```
def simple_function(item):
    item1 = item*10
    item2 = item*11
    return math.sqrt(item1**2 + item2**2)
```

您可以对列表中的每个项目应用此用户定义的函数。

```
multiplied_list = [simple_function(item) for item in original_list]# Output
[148.66068747318505,
 163.52675622050356,
 297.3213749463701,
 327.0535124410071,
 445.9820624195552,
 505.4463374088292]
```

嗯，列表理解在实际场景中会更加有用。通常，在您的分析任务中，您需要从列表中删除某种类型的元素，例如负面或正面元素。列表理解是完成这些任务的完美工具。

# 使用列表理解删除列表中的负面元素

根据特定条件过滤数据是选择所需数据集的常见任务之一。并且在列表理解中使用相同的逻辑来从列表中移除负面元素。

假设你有下面提到的数字列表。

```
original_list = [10, 22, -43, 0, 34, -11, -12, -0.1, 1]
```

并且您希望只保留列表中积极的项目。因此，从逻辑上讲，您希望只保留那些评估条件`**item > 0**`为真的项目。

你可以如下使用列表理解。

```
new_list = **[item for item in original_list if item > 0]**# Output
[10, 22, 34, 1]
```

上表理解中的`**if clause**`是去除负面价值的真正原因。这意味着您可以使用这个`if clause`应用任何条件从列表中删除任何项目。

例如，当您想要删除所有平方小于 200 的项目时。你所需要做的就是在下面的列表理解中提到条件`**item**2 > 200**`。

```
new_list = [item for item in original_list **if item**2 > 200**]# Output
[22, -43, 34]
```

在处理真实数据集时，过滤列表项的条件可能会复杂得多，因此最好知道最快的方法。

接下来，让我们了解字典理解如何将两个列表转换成一个键-值对，即转换成一个字典。

# 使用 dict()将两个列表转换成字典键值对

有时，您需要从两个列表中的值创建一个字典。不用一个一个打，可以用字典理解法。

> 字典理解**是一种优雅简洁的创建字典的方法！**

它的工作方式与列表理解完全相似，唯一的区别是——要创建列表理解，您需要将所有内容括在方括号内，而在字典理解中，您需要将所有内容括在花括号内

让我们用一个例子来更好地理解它。

假设您有两个列表——`fields`和`details`——如下所示。

```
fields = [‘name’, ‘country’, ‘age’, ‘gender’]
details = [‘pablo’, ‘Mexico’, 30, ‘Male’]
```

您想用它创建一个字典，其中键是来自字段的项，值是来自细节的项。

一个简单的方法是使用字典理解，就像这样—

```
new_dict = **{key: value for key, value in zip(fields, details)}**# Output
{'name': 'pablo', 'country': 'Mexico', 'age': 30, 'gender': 'Male'}
```

这里需要理解的重要事情是函数 zip 是如何工作的。

在 Python 中，zip 函数将字符串、列表或字典等可重复项作为输入，然后将它们聚合为元组返回。

所以在这种情况下，zip 已经形成了列表`fields`和`details`中每个项目的配对。当您在字典理解中提到`**key: value**`时，简单地说，这个元组被分解成单独的键-值对。

当您使用 Python 中的内置`**dict()**`构造函数(用于创建字典)时，这个过程会变得更快。

> dict()比字典理解至少快 1.3 倍！

同样，您需要将这个构造函数与 zip()函数一起使用，以获得与上面完全相同的输出。它有更简单的语法— `**dict(zip(fields, details))**`

仅此而已！

正如我最初提到的，Python 是灵活的，因为有多种方法可以达到相同的结果。根据任务的复杂程度，你需要选择最好的方法来完成它。

因此，了解所有这些备选方案以获得相同的结果是一个更好的想法。

希望这篇文章能让你耳目一新，大有裨益。让我也知道，如果有任何其他的方法来做同样的事情，正如我在这篇文章中提到的。

看看我以前写的关于 Python 中的列表和字典的文章，就知道更多的技巧了。

1.  [**Python 字典:你需要知道的 10 个实用方法**](https://pub.towardsai.net/python-dictionary-10-practical-methods-you-need-to-know-cbeb1c962bed)
2.  [**轻松掌握 Python 的 5 种方法**](/5-promising-ways-to-easily-master-the-lists-in-python-bed64cd43bc1)

请注意，免费媒体会员只能阅读 3 篇会员专用的文章。在媒体上阅读无限故事 [**今天用我下面的推荐链接成为媒体会员**](https://medium.com/@17.rsuraj/membership) 。当你这么做的时候，我会从你的费用中得到一小部分。

[](https://medium.com/@17.rsuraj/membership)  

还有，别忘了 [**订阅我的邮件列表**](https://medium.com/subscribe/@17.rsuraj) 。

**感谢阅读！**