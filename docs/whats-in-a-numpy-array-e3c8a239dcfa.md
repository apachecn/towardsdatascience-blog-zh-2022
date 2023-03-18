# NumPy 数组中有什么？

> 原文：<https://towardsdatascience.com/whats-in-a-numpy-array-e3c8a239dcfa>

## Python 的数组数据结构概述以及为什么应该使用它

![](img/c95dfbf06b8d74b34bcf0cd9948a07ba.png)

尼克·希利尔在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

这是我讨论 Python 独特特性系列的第六篇文章；请务必查看 [lambdas](/whats-in-a-lambda-c8cdc67ff107) 、 [list comprehensions](/whats-in-a-list-comprehension-c5d36b62f5) 、[dictionary](/whats-in-a-dictionary-87f9b139cc03)、[tuple](/whats-in-a-tuple-5d4b2668b9a1)和 [f-strings](/whats-in-an-f-string-a435db4f477a) 中的前五个。

如果您对数据处理感兴趣，那么您可能会使用 Python(如果您没有，那么您应该考虑这样做)。谈到用 Python 处理数据，理解如何使用 NumPy 数组是绝对必要的。上帝禁止你使用传统的 Python 列表来分析你的数据；特别是在数据科学的环境中，NumPy 数组比列表有很多优点。

## NumPy 数组到底是什么？

NumPy 数组是通过 Python 的 NumPy 模块可用的顺序数据结构。乍一看，它的功能与列表非常相似:

```
>>> import numpy as np
>>> np.array([1, 2, 3])
array([1, 2, 3])
```

事实上，定义数组的一种方法是传入一个列表，如上所述。您还可以使用内置函数，如`np.zeros`、`np.ones`和`np.arange`，用指定的元素定义数组:

```
>>> np.ones(5) # An 5-element array with all ones
array([1., 1., 1., 1., 1.])>>> np.zeros(3) # A 3-element array with all zeros
array([0., 0., 0.])>>> np.arange(3, 10) # An array of numbers from 3 to 10 (not right-inclusive)
array([3, 4, 5, 6, 7, 8, 9])
```

您可以像访问列表一样访问、更改和分割 NumPy 数组的元素:

```
>>> my_arr = np.array([3, 5, 2, 4, 12])
>>> my_arr[3]
4>>> my_arr[2] = 33
>>> my_arr
array([ 3,  5, 33,  4, 12])>>> my_arr[2:4]
array([33,  4])
```

好吧，这很酷，但是到目前为止，看起来并没有比列表更好的地方，那么拥有这些数组有什么意义呢？

## NumPy 数组为什么有用？

数组的主要优势在于它们能够快速操作和处理数据。当您处理数字时，您经常需要进行算术运算或快速生成汇总数据。在 NumPy 中，这可以使用许多内置函数来完成。这里有几个例子(你可以在这里查看完整的列表:

```
>>> my_arr = np.arange(0, 20)
>>> np.mean(my_arr) # Get the mean of numbers in array
9.5>>> np.sum(my_arr) # Get the sum
190

>>> np.sqrt(my_arr) # Square root of each element
array([0\.        , 1\.        , 1.41421356, 1.73205081, 2\.        ,
       2.23606798, 2.44948974, 2.64575131, 2.82842712, 3\.        ,
       3.16227766, 3.31662479, 3.46410162, 3.60555128, 3.74165739,
       3.87298335, 4\.        , 4.12310563, 4.24264069, 4.35889894])>>> np.cumsum(my_arr) # Cumulative sum up to each element
array([  0,   1,   3,   6,  10,  15,  21,  28,  36,  45,  55,  66,  78, 91, 105, 120, 136, 153, 171, 190], dtype=int32)
```

我非常喜欢最后一个——它将所有元素的总和*增加到当前元素*。因此，例如，输出数组的索引`4`处的元素是`10`，因为`10 = 0 + 1 + 2 + 3 + 4`(它们是输入数组的元素`0`到`4`)。如果你点击上面的链接，还有很多像这样的功能。

您还可以在数组之间执行非常方便的元素操作:

```
>>> np.array([1, 2, 3]) + np.array([-1, -2, -3])
array([0, 0, 0])>>> np.array([1, 2, 3]) * np.array([-1, -2, -3])
array([-1, -4, -9])
```

有琴弦收藏吗？想以某种方式操纵它们？NumPy 数组没有问题:

```
>>> strings = np.array(['MY', 'UPPERCASE', 'STRINGS'])>>> np.char.lower(strings)
array(['my', 'uppercase', 'strings'], dtype='<U9') # not so big now
```

不管怎样，我太兴奋了。您明白了——NumPy 数组是一种非常有用的数据结构，因为它们操作数据的逻辑简单明了。总的来说，NumPy 是一个非常适合 Pandas 的模块，如果你已经是 Pandas 的用户，我强烈建议你学习如何使用它。如果你两者都不使用，那么我建议两者都学！

和往常一样，你的目标应该是生成 Pythonic 代码，它不仅能完成手头的任务，而且易于阅读、理解和维护(人们经常忽略最后一点，但尝试这样做可能是个好主意，这样理解你的代码对将来接手它的人来说就不是一项可怕的任务)。NumPy 数组只是朝着这个方向迈出的又一步。

下次见，伙计们！

**想擅长 Python？** [**获取独家、免费获取我简单易懂的指南点击**](https://witty-speaker-6901.ck.page/0977670a91) **。**

## 参考

[1] *计算和推理思维。*[https://inferentialthinking.com/chapters/05/1/Arrays.html](https://inferentialthinking.com/chapters/05/1/Arrays.html)

[2][https://www . plural sight . com/guides/different-ways-create-numpy-arrays](https://www.pluralsight.com/guides/different-ways-create-numpy-arrays)