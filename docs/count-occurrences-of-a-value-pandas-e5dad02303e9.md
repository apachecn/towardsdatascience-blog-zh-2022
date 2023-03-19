# 如何计算熊猫数据帧列中某个值的出现次数

> 原文：<https://towardsdatascience.com/count-occurrences-of-a-value-pandas-e5dad02303e9>

## 计算熊猫数据帧列中特定值的出现频率

![](img/ab739194055c092b4911cc06e94fb5ed.png)

照片由[纳丁·沙巴纳](https://unsplash.com/@nadineshaabana?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/count?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

## 介绍

当处理 pandas 数据框架时，我们通常需要检查数据并提取一些指标，这些指标最终将帮助我们更好地理解数据，甚至识别一些异常情况。我们在日常工作中需要执行的一个非常简单但常见的任务是计算某个值在数据帧中出现的次数。

在今天的简短教程中，我们将展示如何计算熊猫数据帧列中特定值的频率。我们将探讨如何计算该列中出现的所有唯一值的频率，或者只计算某个特定值的频率。

首先，让我们创建一个示例数据框架，我们将在本教程中引用它来演示几个概念。

```
import pandas a pddf = pd.DataFrame(
    [
        (1, 5, 'A', True),
        (2, 15, 'A', False),
        (3, 5, 'B', True),
        (4, 6, 'A', False),
        (5, 15, 'C', True),
        (6, None, 'B', True),
        (7, 15, 'A', False),
    ],
    columns=['colA', 'colB', 'colC', 'colD']
)print(df)
 ***colA  colB colC   colD
0     1   5.0    A   True
1     2  15.0    A  False
2     3   5.0    B   True
3     4   6.0    A  False
4     5  15.0    C   True
5     6   NaN    B   True
6     7  15.0    A  False***
```

## 使用 groupby

在计算某个特定值在特定列中出现的次数时，我们的第一个选择是用`groupby`和`count`来计算该特定值。让我们假设我们想要计算列`colB`中的每个值出现了多少次。下面的表达式可以帮我们解决这个问题:

```
**>>> df.groupby('colB')['colB'].count()**5.0     2
6.0     1
15.0    3
Name: colB, dtype: int64
```

请注意，上面的表达式将给出指定列中出现的每个非空值的频率。

## 使用值计数

或者，我们可以使用`[pandas.Series.value_counts()](https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html)`方法，该方法将返回包含唯一值计数的熊猫`Series`。

```
**>>> df['colB'].value_counts()**15.0    3
5.0     2
6.0     1
Name: colB, dtype: int64
```

默认情况下，`value_counts()`将返回非空值的频率。如果您还想包括`None`值的频率，您可以简单地将`dropna`参数设置为`False`:

```
**>>> df['colB'].value_counts(dropna=False)**15.0    3
5.0     2
NaN     1
6.0     1
Name: colB, dtype: int64
```

## 将计数添加为新列

如果您想将结果添加回原始数据帧，您需要使用`transform()`方法，如下图所示:

```
**>>>** **df['colB_cnt'] =** **df.groupby('colB')['colB'].transform('count')
>>> df
**   colA  colB colC   colD  colB_cnt
0     1   5.0    A   True       2.0
1     2  15.0    A  False       3.0
2     3   5.0    B   True       2.0
3     4   6.0    A  False       1.0
4     5  15.0    C   True       3.0
5     6   NaN    B   True       NaN
6     7  15.0    A  False       3.0
```

现在，每一行都将具有在新创建的列`colB_cnt`下的列`colB`中出现的值的相关频率。

## 获取特定值的频率

在前面的部分中，我们展示了如何计算出现在特定列中的所有唯一值的频率。如果您只对单个值的频率感兴趣，那么您可以使用以下任何一种方法(假设我们希望值`15`出现在列`colB`中的次数):

```
**>>> (df.colB.values == 15).sum()**
3>>> **(df.colB == 15).sum()**
3>>> **len(df[df['colB'] == 15])** 3
```

## 最后的想法

在今天的短文中，我们探讨了在计算特定 pandas DataFrame 列中值的频率时的一些不同选项。此外，我们展示了如何在现有数据框架中添加计算的计数/频率作为新列。

最后，我们探索了如何使用一些不同的选项计算 DataFrame 列中特定值的频率。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**你可能也会喜欢**

[](/how-to-efficiently-convert-a-pyspark-dataframe-to-pandas-8bda2c3875c3)  [](https://betterprogramming.pub/11-python-one-liners-for-everyday-programming-f346a0a73f39) 