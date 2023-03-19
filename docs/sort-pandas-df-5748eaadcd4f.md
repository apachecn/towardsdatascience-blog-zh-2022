# 如何对熊猫数据帧进行排序

> 原文：<https://towardsdatascience.com/sort-pandas-df-5748eaadcd4f>

## 使用一列或多列对熊猫数据帧进行排序

![](img/674e33bc6eb138ca4ad901ab1a06f925.png)

由 [Markus Spiske](https://unsplash.com/@markusspiske?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/order?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

当检查我们的数据时，我们有时可能想要或者甚至不得不基于一个或多个列对它进行排序。这个简单的过程可以帮助我们调查一个特定的用例，探索边缘案例等等。

在今天的教程中，我们将详细解释如何对熊猫数据帧进行升序或降序排序。此外，我们还将演示在对数据进行排序时如何使用一个或多个列。我们甚至将讨论如何按升序对列的子集进行排序，并按降序对剩余的子集进行排序。

首先，让我们创建一个示例数据帧，我们将在整个教程中引用它，以演示一些概念并展示如何有效地对 pandas 数据帧进行排序。

```
import pandas as pd df = pd.DataFrame(
    [ 
        (1, 'A', 140, False, 3.5),
        (2, 'B', 210, True, 4.0),
        (3, 'A', 562, True, 1.1),
        (4, 'D', 133, False, 2.3),
        (5, 'C', 109, False, 9.8),
        (6, 'C', None, True, 3.9),
        (7, 'B', 976, False, 7.8),
        (8, 'D', 356, False, 4.5),
        (9, 'C', 765, True, 2.1),
    ],
    columns=['colA', 'colB', 'colC', 'colD', 'colE']
)print(df)
 ***colA colB   colC   colD  colE*** *0     1    A  140.0  False   3.5
1     2    B  210.0   True   4.0
2     3    A  562.0   True   1.1
3     4    D  133.0  False   2.3
4     5    C  109.0  False   9.8
5     6    C    NaN   True   3.9
6     7    B  976.0  False   7.8
7     8    D  356.0  False   4.5
8     9    C  765.0   True   2.1*
```

`[pandas.DataFrame.sort_values()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html)`是根据特定条件对数据帧进行排序时需要使用的方法。在接下来的几节中，我们将讨论一些潜在的用例，并演示如何使用`sort_values`来推断出想要的结果。

## 按一列排序

现在让我们假设我们想要根据列`colC`的值对我们刚刚创建的数据帧进行排序。我们需要做的就是在`by`参数中指定列名:

```
>>> df.sort_values(by='colC')
```

结果将包含按`colC`升序排列的所有记录(默认)。

```
 ***colA colB   colC   colD  colE*** *4     5    C* ***109.0*** *False   9.8
3     4    D* ***133.0*** *False   2.3
0     1    A* ***140.0*** *False   3.5
1     2    B* ***210.0*** *True   4.0
7     8    D* ***356.0*** *False   4.5
2     3    A* ***562.0*** *True   1.1
8     9    C* ***765.0*** *True   2.1
6     7    B* ***976.0*** *False   7.8
5     6    C    NaN   True   3.9*
```

或者，您可以指定`ascending=False`来对列`colC`上的数据帧进行降序排序:

```
>>> **df.sort_values(by='colC', ascending=False)**
 *colA colB   colC   colD  colE
6     7    B  976.0  False   7.8
8     9    C  765.0   True   2.1
2     3    A  562.0   True   1.1
7     8    D  356.0  False   4.5
1     2    B  210.0   True   4.0
0     1    A  140.0  False   3.5
3     4    D  133.0  False   2.3
4     5    C  109.0  False   9.8
5     6    C    NaN   True   3.9*
```

## 按多列排序

现在让我们假设我们想要基于两列，即`colA`和`colC`，对 DataFrame 进行排序。这次我们需要做的就是以列表的形式提供列名，并将其传递给`by`参数:

```
>>> df.sort_values(by=['colB', 'colC'])
```

上面的语句将根据列`colB`和`colC`的值对数据帧进行升序排序:

```
 ***colA colB   colC   colD  colE*** *0     1    A  140.0  False   3.5
2     3    A  562.0   True   1.1
1     2    B  210.0   True   4.0
6     7    B  976.0  False   7.8
4     5    C  109.0  False   9.8
8     9    C  765.0   True   2.1
5     6    C    NaN   True   3.9
3     4    D  133.0  False   2.3
7     8    D  356.0  False   4.5*
```

但是请注意，列名的指定顺序很重要——换句话说,`sort_values(by=['colB', 'colC']`和`sort_values(by=['colC', 'colB']`不会产生相同的结果:

```
**>>> df.sort_values(by=['colC', 'colB'])
**   colA colB   colC   colD  colE4     5    C  109.0  False   9.8
3     4    D  133.0  False   2.3
0     1    A  140.0  False   3.5
1     2    B  210.0   True   4.0
7     8    D  356.0  False   4.5
2     3    A  562.0   True   1.1
8     9    C  765.0   True   2.1
6     7    B  976.0  False   7.8
5     6    C    NaN   True   3.9
```

## 对多列按升序或降序排序

接下来，我们甚至可以在对多个列进行排序时指定每个列的排序顺序。这意味着我们可以对一列进行升序排序，而对其他列可以进行降序排序。

为了实现这一点，我们需要做的就是指定一个包含布尔值的列表，这些值对应于在`by`参数中指定的每一列，并表示我们是否要按升序排序。

以下命令将根据`colB`(升序)和`colC`(降序)对数据帧进行排序。

```
>>> df.sort_values(by=['colB', 'colC'], ascending=[True, False])
 ***colA colB   colC   colD  colE*** *2     3    A  562.0   True   1.1
0     1    A  140.0  False   3.5
6     7    B  976.0  False   7.8
1     2    B  210.0   True   4.0
8     9    C  765.0   True   2.1
4     5    C  109.0  False   9.8
5     6    C    NaN   True   3.9
7     8    D  356.0  False   4.5
3     4    D  133.0  False   2.3*
```

## 处理缺失值

你可能已经注意到，空值总是被放在结果的最后，不管我们选择什么顺序。

我们可以通过简单地指定`na_position='first'`(默认为`'last'`)来改变这种行为，将它们放在结果的顶部。

```
>>> df.sort_values(by='colC', ascending=True, na_position='first')
 ***colA colB   colC   colD  colE
5     6    C    NaN   True   3.9*** *4     5    C  109.0  False   9.8
3     4    D  133.0  False   2.3
0     1    A  140.0  False   3.5
1     2    B  210.0   True   4.0
7     8    D  356.0  False   4.5
2     3    A  562.0   True   1.1
8     9    C  765.0   True   2.1
6     7    B  976.0  False   7.8*
```

## 最后的想法

在今天的简短教程中，我们演示了熊猫如何进行排序。给定一个相当简单的数据框架，我们展示了如何基于一列甚至多列对数据进行排序。

此外，我们演示了如何按降序或升序排序，甚至决定哪些列需要按降序排序，哪些应该按升序排序。

最后，我们展示了如何选择空值是出现在结果的顶部还是底部，而不管我们选择的记录排序顺序。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**相关文章您可能也喜欢**

[](/how-to-merge-pandas-dataframes-221e49c41bec)  [](/loc-vs-iloc-in-pandas-92fc125ed8eb) [## 熊猫中的 loc 与 iloc

towardsdatascience.com](/loc-vs-iloc-in-pandas-92fc125ed8eb) [](/how-to-efficiently-convert-a-pyspark-dataframe-to-pandas-8bda2c3875c3) 