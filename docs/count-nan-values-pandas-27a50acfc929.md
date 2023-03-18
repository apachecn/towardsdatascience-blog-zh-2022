# 如何计算熊猫数据帧列的 NaN 值

> 原文：<https://towardsdatascience.com/count-nan-values-pandas-27a50acfc929>

## 计算 pandas DataFrames 列中的空值

![](img/8d48cf43a301d5fbbdedaf32c052769f.png)

由 [Kelly Sikkema](https://unsplash.com/@kellysikkema?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/empty?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

## 介绍

在今天的短文中，我们将讨论 Python 和 Pandas 中一个简单但常见的任务。具体来说，我们将展示在特定的 DataFrame 列中计算空值的多种方法。更具体地说，我们将讨论如何计数

*   `NaN`特定列中的值
*   `NaN`整个数据帧中的值
*   `NaN`每个单独列中的值
*   `NaN`列子集中的值
*   所有列中包含`NaN`值的行
*   特定列中包含`NaN`值的行

首先，让我们创建一个示例 DataFrame，我们将在本教程中引用它，以便理解如何计算空值。

```
import pandas as pd 
df = pd.DataFrame(
    [
        (1, 100, None, 'A'),
        (2, None, True, 'B'),
        (3, 150, None, None), 
        (4, 100, None, 'B'),
        (5, None, False, 'B'),
        (6, 120, False, 'A'),
        (7, 45, True, 'C'),
    ], 
    columns=['colA', 'colB', 'colC', 'colD']
)print(df) ***colA   colB   colC  colD
0     1  100.0   None     A
1     2    NaN   True     B
2     3  150.0   None  None
3     4  100.0   None     B
4     5    NaN  False     B
5     6  120.0  False     A
6     7   45.0   True     C***
```

## 计算特定列中的 NaN 值

现在，为了计算特定列中包含`NaN`值的行数，您可以使用`[pandas.Series.isna()](https://pandas.pydata.org/docs/reference/api/pandas.Series.isna.html#pandas.Series.isna)`方法后跟`sum()`，如下所示:

```
>>> **df['colB'].isna().sum()**
2>>> **df['colA'].isna().sum()**
0
```

或者，您也可以使用`[isnull()](https://pandas.pydata.org/docs/reference/api/pandas.isnull.html)`方法:

```
>>> **df['colB'].isnull().sum()**
2
```

## 计算每一列的 NaN 值

现在，如果您想要获得每个单独列的缺失值的计数，那么您可以使用后面跟有`sum()`的`[pandas.DataFrame.isna()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html#pandas.DataFrame.isna)`方法。输出将是一个 Series 对象，包含原始数据帧中每一列的计数:

```
>>> **df.isna().sum()**
colA    0
colB    2
colC    3
colD    1
dtype: int64
```

## 对列子集中的 NaN 值进行计数

如果只需要原始数据帧中的一部分列的计数，可以执行与上一节相同的操作，但是这次对数据帧进行切片:

```
>>> **df[['colA', 'colD']].isna().sum()** colA    0
colD    1
dtype: int64
```

## 对指定列中只有 NaN 值的行进行计数

现在，如果您想计算所有指定列中缺少值的行数，可以使用下面的符号。例如，让我们假设我们想要计算在`colC`和`colD`中有多少行丢失了值:

```
>>> **df[['colC', 'colD']].isna().all(axis=1).sum()**
1
```

## 对每列中只包含 NaN 值的行进行计数

同样，如果您想计算整个数据帧中每一列只包含缺失值的行数，可以使用下面的表达式。注意，在我们的示例数据帧中，不存在这样的行，因此输出将是 0。

```
>>> **df.isnull().all(axis=1).sum()**
0
```

## 计算整个数据帧内的 NaN 值

最后，如果要计算整个数据帧中每一列包含的缺失值的数量，可以使用以下表达式:

```
>>> **df.isna().sum().sum()**
6
```

## 最后的想法

在今天的短文中，我们讨论了使用 Pandas 数据帧时的一个非常简单但常见的任务，并展示了如何计算特定列或整个数据帧中的空值。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership) [## 通过我的推荐链接加入 Medium-Giorgos Myrianthous

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

gmyrianthous.medium.com](https://gmyrianthous.medium.com/membership) 

**你可能也会喜欢**

[](/how-to-efficiently-convert-a-pyspark-dataframe-to-pandas-8bda2c3875c3) [## 加快 PySpark 和 Pandas 数据帧之间的转换

### 将大火花数据帧转换为熊猫时节省时间

towardsdatascience.com](/how-to-efficiently-convert-a-pyspark-dataframe-to-pandas-8bda2c3875c3) [](/dynamic-typing-in-python-307f7c22b24e) [## Python 中的动态类型

### 探索 Python 中对象引用的工作方式

towardsdatascience.com](/dynamic-typing-in-python-307f7c22b24e) [](/mastering-indexing-and-slicing-in-python-443e23457125) [## 掌握 Python 中的索引和切片

### 深入研究有序集合的索引和切片

towardsdatascience.com](/mastering-indexing-and-slicing-in-python-443e23457125)