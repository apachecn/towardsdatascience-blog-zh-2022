# 如何在熊猫中添加新的空列

> 原文：<https://towardsdatascience.com/add-empty-col-pandas-23d323f0fcc7>

## 使用 Python 向现有 pandas 数据框架添加空列

![](img/58d3a811e1d2a5645722065eb738ea47.png)

由[凯利·西克玛](https://unsplash.com/@kellysikkema?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/add?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

在我最近关于 Medium 的一些文章中，我们讨论了如何基于其他列的值在 Pandas 数据帧中添加[新列。](/create-new-column-based-on-other-columns-pandas-5586d87de73d)

另一个通常用户需要在数据帧上应用的类似操作是在现有的帧中添加一个新的空列。当我们希望以后填充该列，而不是仅仅用一些特定的值初始化它时，这通常是有用的，无论是硬编码的还是基于一些其他的计算。

在今天的简短教程中，我们将演示如何在现有的 pandas 数据框架中添加这样的空列。

首先，让我们创建一个示例数据框架，我们将在本文中引用它来演示一些概念。

```
import pandas as pddf = pd.DataFrame(
    [
        (1, 121, True), 
        (2, 425, False),
        (3, 176, False),
        (4, 509, False),
        (5, 120, True), 
        (6, 459, False),
        (7, 981, True),
        (8, 292, True),
    ], 
    columns=['colA', 'colB', 'colC']
)print(df)
 ***colA  colB   colC*** *0     1   121   True
1     2   425  False
2     3   176  False
3     4   509  False
4     5   120   True
5     6   459  False
6     7   981   True
7     8   292   True*
```

## 在 Pandas 中创建新的空列

因为我们希望新列为空，所以我们实际上可以为每条记录分配`numpy.nan`值。插入一个没有值的新列非常简单

```
import numpy as npdf['colD'] = np.nanprint(df)
 ***colA  colB   colC  colD*** *0     1   121   True   NaN
1     2   425  False   NaN
2     3   176  False   NaN
3     4   509  False   NaN
4     5   120   True   NaN
5     6   459  False   NaN
6     7   981   True   NaN
7     8   292   True   NaN*
```

## 指定空列的数据类型

现在，尽管 DataFrame 中新创建的列是空的，但我们可能希望指定特定的 dtype。期望是在某个时候这个空列将被填充一些值(否则一开始创建它有什么意义！).

我们可以通过调用`dtypes`属性来打印出 DataFrame 中每一列的数据类型:

```
>>> df.dtypes
>>> df.dtypes
*colA      int64
colB      int64
colC       bool
colD    float64
dtype: object*
```

Pandas 自动创建了一个类型为`float64`的空列——这是因为默认情况下`np.nan`的类型是 float。

```
>>> type(np.nan)
*<class 'float'>*
```

相反，我们可能打算在新创建的空列中存储布尔值。在这种情况下，我们可以在创建列后直接转换它，如下所示:

```
import numpy as npdf['colD'] = np.nan
df['colD'] = df['colD'].astype('boolean')print(df.dtypes)
*colA      int64
colB      int64
colC       bool* ***colD    boolean*** *dtype: object*
```

## 最后的想法

有时，我们可能需要在现有的 pandas 数据帧中添加一个空列，以便以后填充。在今天的简短教程中，我们演示了如何创建这样的列，以及如何将其数据类型更改为所需的类型。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership) [## 通过我的推荐链接加入 Medium-Giorgos Myrianthous

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

gmyrianthous.medium.com](https://gmyrianthous.medium.com/membership) 

**相关文章你可能也喜欢**

[](/args-kwargs-python-d9c71b220970) [## * Python 中的 args 和**kwargs

### 讨论位置参数和关键字参数之间的区别，以及如何在 Python 中使用*args 和**kwargs

towardsdatascience.com](/args-kwargs-python-d9c71b220970) [](/oltp-vs-olap-9ac334baa370) [## OLTP 与 OLAP:他们的区别是什么

### 在数据处理系统的上下文中理解 OLTP 和 OLAP 之间的区别

towardsdatascience.com](/oltp-vs-olap-9ac334baa370) [](/run-airflow-docker-1b83a57616fb) [## 如何使用 Docker 在本地运行气流

### 在本地机器上使用 Docker 运行 Airflow 的分步指南

towardsdatascience.com](/run-airflow-docker-1b83a57616fb)