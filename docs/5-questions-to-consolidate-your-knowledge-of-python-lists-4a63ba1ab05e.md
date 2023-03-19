# 巩固 Python 列表知识的 5 个问题

> 原文：<https://towardsdatascience.com/5-questions-to-consolidate-your-knowledge-of-python-lists-4a63ba1ab05e>

## 选自 Stackoverflow

![](img/65b0ba434e087b2dc2e4104e0a2880a0.png)

[Zan](https://unsplash.com/@zanilic?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/five?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

Python 有 4 种内置的数据结构:列表、元组、集合和字典。这些数据结构极其重要，即使您使用 Python 进行数据科学研究。

所有这些本质上都是数据的集合。然而，它们之间的差异使得每个数据结构都是独特的。例如，列表是可变的，而元组是不可变的。我们不能向元组追加新项。

在本文中，我们将讨论 5 个问题，展示 Python 列表的一个特殊特性。我已经挑选了在 Stackoverflow 上有大量投票的问题，所以我们可以假设它们是 Python 社区经常问的问题。

Python 列表是一个有序的元素序列，写在方括号中，用逗号分隔。以下是 4 个元素的列表。

```
mylist = [4, True, "a", 15]
```

正如我们在这个例子中看到的，列表可以存储不同数据类型的对象。

## 1.如何检查一个列表是否为空？

有多种方法可以检查列表是否为空。例如，我们可以用 len 函数找到一个列表的长度，如果它是 0，那么这个列表就是空的。

```
mylist = []if len(mylist) == 0:
    print("The list is empty!")**# output**
The list is empty!
```

更 pythonic 化的方法是基于空序列是假的这一事实。

```
mylist = []if not mylist:
    print("The list is empty!")**# output**
The list is empty!
```

## 2.append 和 extend 方法之间有什么区别？

append 方法用于向列表添加新项目(例如，追加),而 extend 方法向列表添加另一个集合中的项目。换句话说，extend 方法通过使用给定集合中的项目来扩展列表。

让我们看几个例子来演示它们是如何使用的。

```
a = [1, 2, 3]a.append(4)print(a)**# output**
[1, 2, 3, 4]
```

如果将集合传递给 append 方法，它将作为单个元素添加到列表中。

```
a = [1, 2, 3]a.append([4, 5])print(a)**# output**
[1, 2, 3, [4, 5]]
```

列表“a”有 4 项。如果我们使用 extend 方法做同样的例子，结果列表将有 5 个条目，因为 4 和 5 将作为单独的条目添加。

```
a = [1, 2, 3]a.extend([4, 5])print(a)**# output**
[1, 2, 3, 4, 5]
```

## 3.如何获取列表中的最后一项？

有一个非常简单的方法来获取列表中的最后一项。列表第一项的索引是 0。最后一项的索引当然取决于列表的长度。但是，我们可以选择从最后一项开始索引。在这种情况下，索引从-1 开始。

```
a = [1, 2, 3, 4]a[-1]**# output**
4
```

这个方法有一个例外，就是空列表的情况。如果列表为空，上述方法会产生一个索引错误。我们可以用如下简单的方法解决这个问题:

```
a = []a[-1:]**# output**
[]
```

输出仍然是一个空列表，但是代码没有抛出错误。

## 4.你如何展平一个列表？

假设我们有一个如下的列表列表:

```
a = [[1, 2], [3, 4], ["a", "b"]]
```

我们希望将它展平，这意味着将它转换为如下所示的项目列表:

```
a = [1, 2, 3, 4, "a", "b"]
```

同样，执行这项任务有多种方式。我们将使用[列表理解](/crystal-clear-explanation-of-python-list-comprehensions-ac4e652c7cfb)。

```
a = [[1, 2], [3, 4], ["a", "b"]]b = [item for sublist in a for item in sublist]print(b)**# output**
[1, 2, 3, 4, 'a', 'b']
```

如果列表理解的语法看起来令人困惑，试着把它想象成嵌套的 for 循环。例如，上面的列表理解可以通过下面的嵌套 For 循环来实现:

```
b = []for sublist in a:
    for item in sublist:
        b.append(item)

print(b)**# output**
[1, 2, 3, 4, 'a', 'b']
```

## 5.如何删除列表中的重复项？

同样，完成这项任务的方法不止一种。最简单的方法可能是使用 set 构造函数。Set 是 Python 中的另一种内置数据结构。集合的一个识别特征是它们不包含重复项。

```
a = [1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7]b = list(set(a))print(b)**# output**
[1, 2, 3, 4, 5, 6, 7]
```

我们首先将列表转换为集合，然后再转换回列表。

Python 中经常使用列表。它们是通用的和高度实用的，这使它们在许多情况下成为首选的数据结构。

*你可以成为* [*媒介会员*](https://sonery.medium.com/membership) *解锁我的全部写作权限，外加其余媒介。如果你已经是了，别忘了订阅**如果你想在我发表新文章时收到电子邮件。***

*[](https://sonery.medium.com/membership)  

感谢您的阅读。如果您有任何反馈，请告诉我。*