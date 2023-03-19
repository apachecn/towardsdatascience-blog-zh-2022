# NumPy 数组的简单介绍

> 原文：<https://towardsdatascience.com/an-easy-introduction-to-numpy-arrays-6ed950edd389>

什么，怎样，为什么。

![](img/b11007479aa6f7b7c6a5e7ddc70f54f0.png)

米卡·鲍梅斯特在 [Unsplash](https://unsplash.com/s/photos/numbers?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

```
**Index:**
1.Introduction
2.Indexing an array
3.Slicing array
4.Operations on a array
5.Arithmetic functions in Numpy
6.Concatenation of array
Splitting of an array
```

# 1.介绍

> NumPy 代表数字，Pandas 代表数据

NumPy 是一个免费的开源 Python 库，可以帮助你操作数字和应用数值函数。熊猫擅长处理数据框架，NumPy 擅长应用数学。

**数组**是一个由相同类型的值组成的网格，由一个正整数索引。在 Numpy 中，维度称为轴(0 和 1 表示二维数组)，轴是秩。

数组的秩就是它拥有的轴(或维度)的数量。简单列表的秩为 1:二维数组(有时称为矩阵)的秩为 2:三维数组的秩为 3。

过多地投入数组的海洋可能是危险的，也是技术性的。因此，对于本文，我们将把自己限制在一个或两个数组(1 维或 2 维)上，因为它们是数据分析中最常见的。

*   **NumPy 数组对比列表**

更好的做法是使用 Numpy 数组而不是列表。它们分别是**更快**，消耗**更少内存**，更**方便**。

<https://medium.com/@LVillepinte/introduction-to-list-a49a4a73f8ce>  

# 2.索引数组

*   **一维步进**

索引数组类似于索引列表。第一个值是 0，需要用方括号来调用你的数据。

```
my_array = np.arrange(5)print(my_array)
[0,1,2,3,4]print(my_array[0])0
```

*   **二维分度**

如上所述，一个数组可以有几个维度，索引数组的方法保持不变。我们可以使用选择行和列的两个索引来检索元素。

```
my_array = np.array([[0,1,2],[3,4,5]])
my_arrayarray([[0,1,2],
      [3,4,5]])my_array[0][1]
1
```

*   **选择一行或一列**

您也可以选择一行或一列:

```
my_array
array([[0,1,2]
      ,[3,4,5]])my_array[0]
[0 1 2]my_array[:,1]
[1 4]
```

# 3.切片数组

*   **切片 1D 阵**

切片系统是这样做的[ *start:end* ]，其中 start 是包含性的，end 是排他性的。

```
my_array = [1,2,3,4,5,6,7,8] 
my_array[1,3]
[2, 3]my_array[1:]
[2,3,4,5,6,7,8]my_array[:]
[1,2,3,4,5,6,7,8]
```

*   **切片 2D 阵**

多维数组也是如此。

```
my_array = np.array([[1,2,3,4]
                     ,[5,6,7,8]
                     ,[9,10,11,12]])print(my_array[:,1:3])
[[2,3]
 [6,7]
 [10,11]]
```

# 4.数组上的操作

*   **将两个数组相加**

可以对数组进行操作。我们可以把它们加在一起。请注意，如果大小不同，该操作将不起作用:

```
array_1 = np.arrange(5)
array_2 = np.array([100,101,102,103,104])array_3 = array_1 + array_2
print(array_3)
array([100,102,104,106,108])
```

您也可以将数组相乘或应用任何类型的数值运算，逻辑保持不变。

*   **使用带有比较表达式的 Numpy】**

比较数组时，NumPy 应用布尔逻辑。

```
my_array = [0,1,2,3,4,5]
new_array = my_array > 1
array([False, False, True, True, True, True)my_array[new_array]
array([2,3,4,5])my_array[my_array > 1]
array([2,3,4,5])
```

# 5.NumPy 中的算术函数

在内置的 Python 函数之上，NumPy 提供了算术函数。它们遵循前面看到的相同的数组操作规则:

```
np.add() 
np.subtract()
np.negative()
np.multiply() 
np.divide()
np.floor_divide()
np.power()
```

# 6.数组的串联

*   **连接一个 1D 数组**

只有当数组具有相同的形状时，它们才能连接在一起。您可以一次添加多个阵列。

```
array_a = np.array([1,2,3])
array_b = np.array([4,5,6])
array_c = np.concatenate([array_a, array_b])
array([1,2,3,4,5,6])
```

*   **连接一个 2D 数组**

2D 阵列对矩阵特别感兴趣。默认情况下，串联设置在`axis=0`上。这将返回更多的行，但列数相同。如果我们希望数组彼此相邻，我们需要设置`axis=1`。

**如果数组不匹配，**比如，1 上的行数和轴数不相同，或者 0 上的行数和轴数不相同，那么它将不会工作并返回一个错误消息。

# 7.数组的拆分

可以将一个阵列分成多个子阵列。让我们把这看作串联的反义词。如果你简单地将一个数字与`split()`函数联系起来，NumPy 将会很聪明地将你的数组分成大小相同的子数组。

如果不可能创建相同大小的子阵列。必须使用`array_split()`功能。

```
my_array = [1,2,3,4,5,6]
split_array = np.split(my_array, 2)
print(split_array)
[array(1,2,3)),[array(4,5,6)]my_new_array = [1,2,3,4,5,6,7]
mna = np.split_array(my_new_array)
[array(1,2,3)),[array(4,5,6)], [array(7)]
```

你有可能去定制的结果，并在指定的位置分裂 1D 数组。

```
my_array = [1,2,3,4,5,6,7]
split_array = np.split(my_array, [2,4])
print(split_array)
[array([1,2])],array([3,4]),array([5,6,7])]
```

您也可以使用`vsplit()`和`hsplit()`功能决定垂直或水平分割阵列

# 结论

NumPy 提出了一种操作数据的简便方法。数组比列表更好、更快、更有效，是一种广泛使用的解决方案。这个介绍展示了它是如何工作的，以及一些有用的操作和可能的查询。

# 在你走之前👋

*感谢您阅读本文。这是一份关于数据科学的每周时事通讯。免费订阅，不要错过下一篇:*[https://medium.com/subscribe/@LVillepinte](https://medium.com/subscribe/@LVillepinte)

*📚先前发布:*

<https://faun.pub/how-to-become-the-best-data-scientist-6b1334f53244>  <https://faun.pub/this-one-thing-is-causing-the-next-black-swan-e734e6e710c1> 