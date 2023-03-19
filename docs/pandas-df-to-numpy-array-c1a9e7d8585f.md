# 如何将 Pandas 数据帧转换成 NumPy 数组

> 原文：<https://towardsdatascience.com/pandas-df-to-numpy-array-c1a9e7d8585f>

## 将熊猫数据帧转换为 NumPy 数组

![](img/e2df43a0ee110a2422b7afb7abd8f2ef.png)

照片由[徐世洋](https://unsplash.com/@ltmonster?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/pandas?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

## 介绍

当处理 pandas 数据帧时，如果我们将它们转换成 NumPy 数组，有时会有好处。后者似乎更节省内存，尤其是在对数据执行一些复杂的数学运算时。

当您处理相对少量的数据(比如 50K 行或更少)时尤其如此。在涉及大量数据的情况下(比如说超过 50 万行)，Pandas 通常会优于 ndarrays。不过这只是一个经验法则——在大多数情况下，根据您的特定需求和使用案例，测试两个选项并查看哪一个在性能和内存使用方面更好会更好。

在今天的简短教程中，我们将展示如何有效地将熊猫数据帧转换成 NumPy 数组。

首先，让我们创建一个示例 pandas DataFrame，我们将使用它来演示几种可能用于将其转换为 numpy 数组的不同方法。

```
import pandas as pd df = pd.DataFrame(
    [
        (1, 'A', 10.5, True),
        (2, 'B', 10.0, False),
        (3, 'A', 19.2, False),
        (4, 'C', 21.1, True),
        (5, 'A', 15.5, True),
        (6, 'C', 14.9, False),
        (7, 'C', 13.1, True),
        (8, 'B', 12.5, False),
        (9, 'C', 11.2, False),
        (10, 'A', 31.4, False),
        (11, 'D', 10.4, True),
    ],
    columns=['colA', 'colB', 'colC', 'colD']
)print(df)
 ***colA colB  colC   colD*** *0      1    A  10.5   True
1      2    B  10.0  False
2      3    A  19.2  False
3      4    C  21.1   True
4      5    A  15.5   True
5      6    C  14.9  False
6      7    C  13.1   True
7      8    B  12.5  False
8      9    C  11.2  False
9     10    A  31.4  False
10    11    D  10.4   True*
```

## 用熊猫。DataFrame.to_numpy()

当谈到将 pandas 数据帧转换成 NumPy 数组时，我们的第一个选择是`[pandas.DataFrame.to_numpy()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html)`方法。

```
**ndarray = df.to_numpy()**print(ndarray)*array([[1, 'A', 10.5, True],
       [2, 'B', 10.0, False],
       [3, 'A', 19.2, False],
       [4, 'C', 21.1, True],
       [5, 'A', 15.5, True],
       [6, 'C', 14.9, False],
       [7, 'C', 13.1, True],
       [8, 'B', 12.5, False],
       [9, 'C', 11.2, False],
       [10, 'A', 31.4, False],
       [11, 'D', 10.4, True]], dtype=object)*
```

并且返回对象的类型将是`numpy.ndarray`:

```
>>> type(ndarray)
<class 'numpy.ndarray'>
```

## 用熊猫。DatFrame.to_records()

这里的另一个选项是`[pandas.DataFrame.to_records()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_records.html)`方法，它将把 pandas 数据帧转换成一个 **NumPy 记录数组**:

```
**recarray = df.to_records()**print(recarray)rec.array([( 0,  1, 'A', 10.5,  True), 
           ( 1,  2, 'B', 10\. , False),
           ( 2,  3, 'A', 19.2, False), 
           ( 3,  4, 'C', 21.1,  True),
           ( 4,  5, 'A', 15.5,  True), 
           ( 5,  6, 'C', 14.9, False),
           ( 6,  7, 'C', 13.1,  True), 
           ( 7,  8, 'B', 12.5, False),
           ( 8,  9, 'C', 11.2, False), 
           ( 9, 10, 'A', 31.4, False),
           (10, 11, 'D', 10.4,  True)],
          dtype=[('index', '<i8'), ('colA', '<i8'), ('colB', 'O'), ('colC', '<f8'), ('colD', '?')])
```

如上所述，与`to_numpy()`相反，`to_records()`方法将返回一个类型为`nympy.recarray`的对象:

```
>>> type(recarray)
<class 'numpy.recarray'>
```

## 使用 numpy.asarray()

第三个选项是`[numpy.asarray()](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html#numpy-asarray)`方法，它将输入的 pandas 数据帧转换成一个 NumPy 数组:

```
import numpy as np**ndarray = np.asarray(df)**print(ndarray)array([[1, 'A', 10.5, True],
      [2, 'B', 10.0, False],
      [3, 'A', 19.2, False],
      [4, 'C', 21.1, True],
      [5, 'A', 15.5, True], 
      [6, 'C', 14.9, False],
      [7, 'C', 13.1, True],
      [8, 'B', 12.5, False],
      [9, 'C', 11.2, False],
      [10, 'A', 31.4, False],
      [11, 'D', 10.4, True]], dtype=object)
```

返回的对象将再次成为`numpy.ndarray`的实例:

```
>>> type(ndarray)
<class 'numpy.ndarray'>
```

## 避免使用 df.values

在较早的 pandas 版本中，将 pandas 数据帧转换成 NumPy 数组的另一种方法是通过`[pandas.DataFrame.values](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.values.html#pandas.DataFrame.values)`属性。但是请注意，即使是官方文档也鼓励您不要再使用它:

> 我们建议使用`[**DataFrame.to_numpy()**](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_numpy.html#pandas.DataFrame.to_numpy)`来代替。

这是因为该属性的行为不一致。要了解更多细节，可以阅读 0.24.0 版本的[发行说明。](https://pandas-docs.github.io/pandas-docs-travis/whatsnew/v0.24.0.html#accessing-the-values-in-a-series-or-index)

> 从历史上看，这可以用`series.values`来完成，但是用`.values`还不清楚返回值是实际的数组、它的某种转换，还是熊猫定制的数组之一(像`Categorical`)。

## 最后的想法

在今天的文章中，我们讨论了将 pandas 数据帧转换成 NumPy 数组，以及在什么情况下这样做是有益的。

此外，我们展示了如何使用`pandas.DataFrame`对象的`to_numpy()`和`to_records()`方法以及`numpy.asarray()`方法将 DataFrame 转换为 ndarray。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

<https://gmyrianthous.medium.com/membership>  

**相关文章你可能也喜欢**

</how-to-efficiently-convert-a-pyspark-dataframe-to-pandas-8bda2c3875c3>  </how-to-merge-pandas-dataframes-221e49c41bec>  </random-seed-numpy-786cf7876a5f> 