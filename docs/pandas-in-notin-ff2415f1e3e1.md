# 如何使用“在”和“不在”筛选熊猫数据帧

> 原文：<https://towardsdatascience.com/pandas-in-notin-ff2415f1e3e1>

## 理解相当于 SQL“in”和“not in”表达式的熊猫

![](img/f6f764f658b81bf296f7e999fd543556.png)

丹尼尔·利维斯·佩鲁西在 [Unsplash](https://unsplash.com/s/photos/object?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄的照片

## 介绍

在 pandas 中处理数据帧时，我们通常需要根据某些条件来过滤行，比如某列的行值是否在某组指定值中(或者不在其中)。您可以将这样的表达式想象成对同一列有多个`OR`操作符。

在今天的简短教程中，我们将展示如何使用相当于 SQL `IN`和`NOT IN`表达式的熊猫。此外，我们还将讨论如何在一个熊猫表达式中组合这些条件。

首先，让我们创建一个示例数据框架，我们将在本教程中使用它来演示一些概念。

```
import pandas as pddf = pd.DataFrame(
    [
        (1, 'A', 10, True),
        (2, 'B', 12, False),
        (3, 'B', 21, False),
        (4, 'C', 18, False),
        (5, 'A', 13, True),
        (6, 'C', 42, True),
        (7, 'B', 19, True),
        (8, 'A', 21, False),
    ],
    columns=['colA', 'colB', 'colC', 'colD']
)print(df) ***colA colB  colC   colD*** 0     1    A    10   True
1     2    B    12  False
2     3    B    21  False
3     4    C    18  False
4     5    A    13   True
5     6    C    42   True
6     7    B    19   True
7     8    A    21  False
```

## 熊猫相当于 SQL `IN`表达式

`[pandas.Series.isin()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.isin.html)`方法，是相当于 SQL 中众所周知的`IN`表达式的熊猫。该方法返回一个布尔值`Series`,指示元素是否包含在指定的值中。

例如，让我们假设在我们的示例数据帧中，我们希望只保留在列`colB`中具有值`A`或`C`的行。为此，我们可以使用通过`isin()`方法获得的结果对数据帧进行切片，如下所示:

```
df = df[df.colB.isin(['A', 'C'])]print(df)
 ***colA colB  colC   colD***
0     1    A    10   True
3     4    C    18  False
4     5    A    13   True
5     6    C    42   True
7     8    A    21  False
```

## 熊猫相当于 SQL `NOT IN`表达式

同样，我们可以简单地否定来自`isin()`方法的结果，以获得与`NOT IN`表达式等价的熊猫。`~`求反运算符可用于实现这一点。

现在让我们假设我们想要过滤掉在列`colB`中具有值`A`或`C`的所有行。下面的表达式可以解决这个问题。

```
df = df[~df.colB.isin(['A', 'C'])]print(df)
 ***colA colB  colC   colD*** 1     2    B    12  False
2     3    B    21  False
6     7    B    19   True
```

## 将条件应用于多列

现在让我们假设我们想要在熊猫数据帧上组合多个条件。我们可以使用逻辑 AND 运算符`&`将所有的连接起来。举个例子，

```
df = df[df.colB.isin(['A', 'C']) & df.colC.isin([10, 13, 16])]print(df)
 ***colA colB  colC  colD*** 0     1    A    10  True
4     5    A    13  True
```

## isin()方法的替代方法

另一种方法是使用`query()`方法，让您指定类似 SQL 的表达式。

```
vals_to_keep = ['A', 'B']
df = df.query('colB in @vals_to_keep')print(df)
 ***colA colB  colC   colD*** 0     1    A    10   True
1     2    B    12  False
2     3    B    21  False
4     5    A    13   True
6     7    B    19   True
7     8    A    21  False
```

同样，我们也可以使用`not in`表达式:

```
vals_to_drop = ['A', 'B']
df = df.query('colB not in @vals_to_keep')print(df)
 ***colA colB  colC   colD*** 1     2    B    12  False
2     3    B    21  False
6     7    B    19   True
```

注意，SQL 表达式是区分大小写的，所以如果您改为指定`IN`或`NOT IN`(即大写字符)，您将以`SyntaxError`结束。

## 最后的想法

在今天的文章中，我们讨论了相当于 SQL `IN`和`NOT IN`表达式的熊猫。在过滤掉不符合指定标准的数据帧行时，这样的条件可能非常有用。

特别是`isin()`表达式非常强大，因为它可以帮助您指定多个条件，为了不从结果中过滤掉，数据帧的行必须至少满足其中一个条件。换句话说，你可以认为它有多个`OR`表达式。

最后，我们讨论了`isin()`方法的替代方法，即`query()`，它允许您在 pandas 数据帧上指定类似 SQL 的表达式。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

<https://gmyrianthous.medium.com/membership>  

**相关文章你可能也喜欢**

</how-to-merge-pandas-dataframes-221e49c41bec>  </how-to-iterate-over-rows-in-a-pandas-dataframe-6aa173fc6c84>  </how-to-efficiently-convert-a-pyspark-dataframe-to-pandas-8bda2c3875c3> 