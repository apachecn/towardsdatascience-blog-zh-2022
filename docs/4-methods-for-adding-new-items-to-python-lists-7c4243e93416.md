# 向 Python 列表添加新项目的 4 种方法

> 原文：<https://towardsdatascience.com/4-methods-for-adding-new-items-to-python-lists-7c4243e93416>

## 列表是 Python 的核心数据结构

![](img/37b7af736531d661d732e23977fce5dd.png)

凯利·西克玛在 [Unsplash](https://unsplash.com/s/photos/list?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

数据结构对于任何编程语言都非常重要，因为如何存储、访问和管理数据是设计高效程序的关键因素之一。

Python 有 4 种内置数据结构:

*   目录
*   一组
*   元组
*   词典

列表表示为方括号中的数据点集合，可用于存储不同数据类型的值。下面是一个列表示例:

```
mylist = [1, 2, 3, "Jane", True]
```

在本文中，我们将学习 4 种向 Python 列表添加新项的方法。

# 1.附加

append 方法用于向列表中添加单个项目。

```
mylist.append(7)

print(mylist)

# output
[1, 2, 3, 'Jane', True, 7]
```

如果您尝试使用 append 方法添加项目集合(例如列表、元组)，该集合将作为单个项目追加到列表中。

```
mylist.append(["Max", "John"])

print(mylist)

# output
[1, 2, 3, 'Jane', True, 7, ['Max', 'John']]
```

变量 mylist 不单独包含项目“Max”和“John”。

```
"Max" in mylist
# output
False

["Max", "John"] in mylist
# output
True
```

***提示*** :如果你不小心把一个物品列表追加为单个物品，但需要新的物品在原列表中作为单独的物品，可以使用熊猫的分解功能。

```
import pandas as pd

mylist = [1, 2, 3, 'Jane', True, 7, ['Max', 'John']]

pd.Series(mylist).explode(ignore_index=True)
# output
0       1
1       2
2       3
3    Jane
4    True
5       7
6     Max
7    John
dtype: object
```

您可以使用 list 构造函数将输出转换回 list:

```
import pandas as pd

mylist = [1, 2, 3, 'Jane', True, 7, ['Max', 'John']]

list(pd.Series(mylist).explode(ignore_index=True))
# output
[1, 2, 3, 'Jane', True, 7, 'Max', 'John']
```

# 2.插入

append 方法将新项添加到列表的末尾(即作为最后一项)。如果要在开头或特定索引处添加项目，请改用 insert 方法。

它需要两个参数:

1.  要添加的项的索引
2.  项目本身

索引从 0 开始，所以我们需要使用索引 0 在开头添加一个新项。

```
mylist = ["Jane", "Jennifer"]

mylist.insert(0, "Matt")

print(mylist)
# output
['Matt', 'Jane', 'Jennifer']
```

让我们再做一个例子，添加新的条目作为第三个条目。

```
mylist = [1, 2, 3, 4]

mylist.insert(2, 10000)

print(mylist)
# output
[1, 2, 10000, 3, 4]
```

就像 append 方法一样，insert 方法只能用于向列表中添加单个项目。

# 3.扩展

extend 方法使用给定集合中的项扩展列表。

它也可以用于向列表中添加单个项目，但是 append 方法更适合于此任务。

```
mylist = ["Jane", "Jennifer"]

mylist.extend(["Max", "John"])

print(mylist)
# output
['Jane', 'Jennifer', 'Max', 'John']
```

现在“Max”和“John”作为单独的项目添加到列表中。

```
"Max" in mylist
# output
True
```

使用 extend 方法添加字符串的单个项时要小心。因为字符串可以被看作是字符的集合，所以 extend 方法将每个字符作为单独的项添加到列表中。

这里有一个例子:

```
mylist = ["Jane", "Jennifer"]

mylist.extend("Matt")

print(mylist)
# output
['Jane', 'Jennifer', 'M', 'a', 't', 't']
```

使用 append 方法将单个项目添加到列表中总是更好。

# 4.加法运算符(+)

加法运算符可用于连接列表，这也意味着将一个项目列表添加到另一个列表中。

```
mylist = [1, 2, 3, 4]

mylist = mylist + [8, 9]

print(mylist)
# output
[1, 2, 3, 4, 8, 9]
```

但是，我们只能使用这个方法将一个列表中的项目添加到另一个列表中。它不能用于向列表中添加不同类型的项目(例如，整数、元组、字符串)。

我们来做一个例子，用一个整数来演示这种情况。

```
mylist = [1, 2, 3, 4]

mylist = mylist + 10

# output
TypeError: can only concatenate list (not "int") to list
```

正如我们在上面的输出中看到的，这引发了一个类型错误。

# 结论

列表是 Python 中最常用的数据结构之一。因此，学习与列表交互的方法来创建健壮的程序是很重要的。在本文中，我们学习了如何通过添加新项目来修改列表。

*你可以成为* [*媒介会员*](https://sonery.medium.com/membership) *解锁我的全部写作权限，外加其余媒介。如果你已经是了，别忘了订阅*<https://sonery.medium.com/subscribe>**如果你想在我发表新文章时收到电子邮件。**

*感谢您的阅读。如果您有任何反馈，请告诉我。*