# 如何在 Pandas 中按缺少值的列分组

> 原文：<https://towardsdatascience.com/groupby-with-nan-values-pandas-a05e53f410b>

## 在 pandas 列中分组时合并空值

![](img/7bc17eee805c2c4281ac65c0df20fd6b.png)

照片由[Firmbee.com](https://unsplash.com/@firmbee?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/graph?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

使用 group by 子句执行聚合可能是您在日常工作中遇到的事情。Python 中的 Pandas 也不例外，因为这是一个你肯定会在库的许多不同地方看到的操作。

然而，pandas 的默认行为是从结果中排除空值/缺失值。在今天的简短教程中，我们将演示这种默认行为以及将缺失值合并到结果聚合中的方法。

首先，让我们创建一个示例

```
import numpy as np
import pandas as pd 

df = pd.DataFrame(
    [ 
        (1, 'B', 121, 10.1, True),
        (2, 'C', 145, 5.5, False),
        (3, 'A', 345, 4.5, False),
        (4, 'A', 112, np.nan, True),
        (5, 'C', 105, 2.1, False),
        (6, np.nan, 435, 7.8, True),
        (7, np.nan, 521, np.nan, True),
        (8, 'B', 322, 8.7, True),
        (9, 'C', 213, 5.8, True),
        (10, 'B', 718, 9.1, False),
    ],
    columns=['colA', 'colB', 'colC', 'colD', 'colE']
)

print(df)
   colA colB  colC  colD   colE
0     1    B   121  10.1   True
1     2    C   145   5.5  False
2     3    A   345   4.5  False
3     4    A   112   NaN   True
4     5    C   105   2.1  False
5     6  NaN   435   7.8   True
6     7  NaN   521   NaN   True
7     8    B   322   8.7   True
8     9    C   213   5.8   True
9    10    B   718   9.1  False
```

## 默认行为

现在让我们假设我们想要计算`colB`中每个值的总和。这很简单，可以用下面的表达式来完成

```
df.groupby('colB')['colD'].sum()
```

这将返回以下结果:

```
>>> df.groupby('colB')['colD'].sum()
colB
A     4.5
B    27.9
C    13.4
Name: colD, dtype: float64
```

但是我们可以注意到，输出中缺少值(`None`)。

## 在聚合中合并空值

您可能仍然希望将缺失的值合并到聚合中。为了做到这一点，您需要做的就是在调用`groupby`函数时显式指定`dropna=False`——该值默认为`True`。请注意，这对于 **pandas 版本≥ 1.1** 是可能的。

```
df.groupby('colB', dropna=False)['colD'].sum()
```

并且结果序列还将包括缺失值的计数:

```
>>> df.groupby('colB', dropna=False)['colD'].sum()
colB
A       4.5
B      27.9
C      13.4
NaN     7.8
Name: colD, dtype: float64
```

显然，同样的概念也适用于其他聚合类型，比如 count。举个例子，

```
>>> df.groupby('colB').count()
      colA  colC  colD  colE
colB                        
A        2     2     1     2
B        3     3     3     3
C        3     3     3     3
```

而不是

```
>>> df.groupby('colB', dropna=False).count()
      colA  colC  colD  colE
colB                        
A        2     2     1     2
B        3     3     3     3
C        3     3     3     3
NaN      2     2     1     2
```

## 最后的想法

使用最新的 pandas 版本，您现在可以在对 pandas 数据帧执行聚合时合并缺失值。但是请注意，如果您运行的是不支持`groupby`方法中的`dropna`的旧版本，那么(除了升级到最新版本之外，这总是一个好的做法！)您也可以找到一些解决方法，如下所示:

```
# Fill the missing values with a particular placeholder value
# Note though that you must be careful when selecting such 
# value in order to avoid any collisions with pre-existing values
# in the dataframe
# In the example below I'll use -1, but make sure to amend if needed
>>> df.fillna(-1).groupby('colB').sum()
```

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**相关文章你可能也喜欢**

[](/how-to-merge-pandas-dataframes-221e49c41bec)  [](/data-engineer-tools-c7e68eed28ad)  [](/visual-sql-joins-4e3899d9d46c) 