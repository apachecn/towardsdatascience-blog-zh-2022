# 如何对熊猫数据帧进行分组以计算平均值

> 原文：<https://towardsdatascience.com/pandas-groupby-mean-fd34885a3d72>

## 用分组表达式计算熊猫的平均值

![](img/d967f9ea3c1a43bc907c6acb25ab56bc.png)

Diana Polekhina 在 [Unsplash](https://unsplash.com/s/photos/measure?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

## 介绍

当使用 pandas 数据框架时，我们通常需要为特定的组计算某些度量。实现这一点的典型方法是通过传统的分组表达式，然后是我们需要计算的相关计算。

在今天的简短教程中，我们将展示如何对 pandas 数据帧执行分组操作，以计算每组的平均值。

首先，让我们在 pandas 中创建一个示例数据框架，我们将在本文中使用它来演示一些概念，并理解为了获得目标结果我们需要遵循什么步骤。

```
import pandas as pd df = pd.DataFrame(
    [ 
        (1, 'B', 121, 10.1, True),
        (2, 'C', 145, 5.5, False),
        (3, 'A', 345, 4.5, False),
        (4, 'A', 112, 3.0, True),
        (5, 'C', 105, 2.1, False),
        (6, 'A', 435, 7.8, True),
        (7, 'B', 521, 9.1, True),
        (8, 'B', 322, 8.7, True),
        (9, 'C', 213, 5.8, True),
        (10, 'B', 718, 9.1, False),
    ],
    columns=['colA', 'colB', 'colC', 'colD', 'colE']
)print(df)
 **colA colB  colC  colD   colE** *0     1    B   121  10.1   True
1     2    C   145   5.5  False
2     3    A   345   4.5  False
3     4    A   112   3.0   True
4     5    C   105   2.1  False
5     6    A   435   7.8   True
6     7    B   521   9.1   True
7     8    B   322   8.7   True
8     9    C   213   5.8   True
9    10    B   718   9.1  False*
```

## 使用 mean()方法

我们这里的第一个选项是对感兴趣的列执行`groupby`操作，然后使用我们想要执行数学计算的列分割结果，最后调用`mean()`方法。

现在让我们假设对于列`colB`中出现的每个值，我们想要计算列`colC`的平均值。下面的表达式将为我们解决这个问题。

```
>>> df.groupby('colB')['colC'].mean()
colB
A    297.333333
B    420.500000
C    154.333333
Name: colC, dtype: float64
```

结果将是一个熊猫系列，包含列`colB`中出现的每个值的`colC`的平均值。

## 使用 agg()方法

同样，我们可以使用`agg()`方法来为指定的操作执行聚合——在我们的例子中是平均值计算。

```
>>> df.groupby('colB')['colC'].agg('mean')
colB
A    297.333333
B    420.500000
C    154.333333
Name: colC, dtype: float64
```

结果将与我们之前展示的方法完全相同。

## 计算中位数

同样，您可以使用相同的策略来计算其他指标，如计算组的中位数、计数或总和。

在下面的例子中，我们使用了与本教程第一部分中展示的相同的方法来计算列`colB`中出现的每个值的`colC`的中值。

```
>>> df.groupby('colB')['colC'].median()
colB
A    345.0
B    421.5
C    145.0
Name: colC, dtype: float64
```

同样，您可以使用其他方法，如`count()`和`sum()`来计算相应的指标。

这同样适用于涉及`agg()`方法的第二种方法:

```
>>> df.groupby('colB')['colC'].agg('median')
colB
A    345.0
B    421.5
C    145.0
Name: colC, dtype: float64
```

## 最后的想法

在今天的文章中，我们讨论了 pandas 中最常执行的操作之一，它要求我们对感兴趣的数据帧执行分组操作。

此外，我们展示了如何计算有用的指标，例如感兴趣的组的平均值和中值。当然，这只是您可以计算的度量值的一个示例—实际上，可以使用相同的方法来计算计数、总和等。

[](https://levelup.gitconnected.com/how-to-group-by-pandas-dataframes-to-compute-sum-82a6bd890cbf)  

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**相关文章你可能也喜欢**

[](/pandas-in-notin-ff2415f1e3e1)  [](/us-market-bank-holidays-pandas-fbb15c693fcc)  [](/make-class-iterable-python-4d9ec5db9b7a) 