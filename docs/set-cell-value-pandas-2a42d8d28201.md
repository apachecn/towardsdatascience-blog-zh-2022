# 如何使用索引设置熊猫数据框中的单元格值

> 原文：<https://towardsdatascience.com/set-cell-value-pandas-2a42d8d28201>

## 在熊猫数据帧的特定单元格中设置值

![](img/94c1b8da2b0f5261d57bdf51c44c7aea.png)

[Kai Dahms](https://unsplash.com/@dilucidus?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/cell?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

## 介绍

在我以前的一篇文章中，我讨论了关于[的两个熊猫最常用的属性，即](/loc-vs-iloc-in-pandas-92fc125ed8eb) `[loc](/loc-vs-iloc-in-pandas-92fc125ed8eb)` [和](/loc-vs-iloc-in-pandas-92fc125ed8eb) `[iloc](/loc-vs-iloc-in-pandas-92fc125ed8eb)`。这些属性可用于**分别通过标签和整数位置查找来访问一组行**。

在今天的简短教程中，我们将展示如何使用**标签**或**整数位置**在 pandas 数据帧中的**特定单元格**赋值(即特定列和行的值)。

首先，让我们创建一个 pandas 数据框架，我们将使用它作为一个例子来演示在 pandas 数据框架中设置单元格值时的一些概念。

```
import pandas as pd df = pd.DataFrame(
    [
        (1, 'A', 150, True),
        (2, 'A', 120, False),
        (3, 'C', 130, False),
        (4, 'A', 150, False),
        (5, 'B', 140, True),
        (6, 'B', 115, False),
    ], 
    columns=['colA', 'colB', 'colC', 'colD']
) print(df)
 ***colA colB  colC   colD
0     1    A   150   True
1     2    A   120  False
2     3    C   130  False
3     4    A   150  False
4     5    B   140   True
5     6    B   115  False***
```

## 使用标签设置特定行和列的值

为了访问行/列标签对的单个值，你可以使用`[pandas.DataFrame.at](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.at.html)`属性。注意，这个属性类似于`loc`，因为两者都允许基于标签的查找。然而，如果您想访问 DataFrame 或 Series 对象中的单个值就应该使用`at`属性。

例如，假设我们想要将索引`1`处的列`colB`的值设置为`B`。此时，该值被设置为`'A'`:

```
>>> df.at[1, 'colB']
'A'
```

为了将它更改为`B`，下面的表达式可以解决这个问题:

```
**df.at[1, 'colB'] = 'B'** print(df)
 *colA colB  colC   colD
0     1    A   150   True* ***1     2    B   120  False*** *2     3    C   130  False
3     4    A   150  False
4     5    B   140   True
5     6    B   115  False*
```

## 使用整数位置设置特定行和列的值

或者，您可以使用`[pandas.DataFrame.iat](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iat.html#pandas.DataFrame.iat)`属性通过**整数位置**访问行/列。这类似于`iloc`属性，因为它们都提供基于整数的查找。

同样，让我们访问第二行的列`colB`的值(即索引`1`)，但这一次，我们将使用整数位置，而不是使用列标签:

```
>>> df.iat[1, 1]
'B'
```

现在让我们使用`iat`属性和一个整数位置将值改回`'A'`:

```
**df.iat[1, 1] = 'A'**print(df)*colA colB  colC   colD
0     1    A   150   True* ***1     2    A   120  False*** *2     3    C   130  False
3     4    A   150  False
4     5    B   140   True
5     6    B   115  False*
```

## 最后的想法

在今天的简短教程中，我们展示了如何使用基于标签的查找和`at`属性或整数位置查找和`iat`属性来设置特定列和行的值。

注意，这两个属性非常类似于`loc`和`iloc`属性，但是`at`和`iat`可以用来访问(和改变)特定的列/行对，而不是访问一组行。

关于`loc`和`iloc`属性的更全面的阅读，你可以参考我最近的一篇文章，分享如下:

[](/loc-vs-iloc-in-pandas-92fc125ed8eb)  **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**你可能也会喜欢**

[](/random-seed-numpy-786cf7876a5f)  [](/how-to-efficiently-convert-a-pyspark-dataframe-to-pandas-8bda2c3875c3) 