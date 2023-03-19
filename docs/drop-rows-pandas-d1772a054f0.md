# 如何从 Pandas 数据框架中删除行列表

> 原文：<https://towardsdatascience.com/drop-rows-pandas-d1772a054f0>

## 用 Python 从 pandas 数据帧中删除多行

![](img/d176a6bdf984eff91bfd9fed0e6f5dd4.png)

[王禹](https://unsplash.com/@stanyw?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/pandas?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍照

当处理 pandas 数据帧时，我们通常希望丢弃原始结构中的行。Pandas 提供了一个直观的 API，我们可以利用它来执行这样的操作。

在今天的文章中，我们将通过提供相应记录的索引列表来演示如何从 pandas 数据帧中删除行。此外，在处理大量数据时，我们将探索一些替代方法。

首先，让我们创建一个示例数据框架，我们将在整个教程中引用它来演示一些概念，并帮助您逐步完成这个过程。

```
import pandas as pd df = pd.DataFrame(
    [
        (1, 125, True, 'A'),
        (2, 222, False, 'A'),
        (3, 871, False, 'C'),
        (4, 134, False, 'D'),
        (5, 908, True, 'D'),
        (6, 321, False, 'E'),
        (7, 434, False, 'B'),
        (8, 678, True, 'C'),
    ], 
    columns=['colA', 'colB', 'colC', 'colD']
)print(df)
 ***colA  colB   colC colD*** *0     1   125   True    A
1     2   222  False    A
2     3   871  False    C
3     4   134  False    D
4     5   908   True    D
5     6   321  False    E
6     7   434  False    B
7     8   678   True    C*
```

## 删除带有索引列表的记录

上面输出中最左边的列(0–7)对应于我们的 pandas 数据帧中每个记录的索引。然后，我们可以创建一个列表，包含我们希望删除的记录的索引

```
drop_idx = [1, 3, 6]
```

然后利用`[pandas.DataFrame.drop()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html)`方法直接指定要删除的索引或列名。最后，我们将索引列表分为一系列索引标签:

```
df = df.drop(df.index[drop_idx])print(df) ***colA  colB   colC colD*** *0     1   125   True    A
2     3   871  False    C
4     5   908   True    D
5     6   321  False    E
7     8   678   True    C*
```

请注意，如果您不执行就地更改，您将不得不将`drop()`方法的结果赋回您的`df`，如上例所示。要执行就地更改，您需要做的就是在调用`drop()`时传递相应的参数:

```
df.drop(df.index[drop_idx], axis=0, inplace=True)
```

## 处理大量数据

当处理大型数据帧时——或者甚至当需要删除大量记录时，上一节中介绍的`df.drop()`效率不会很高，并且可能会花费大量时间。

在按索引删除行时，处理大型数据帧的最有效方法是实际上反转问题。换句话说，我们不是找出需要删除的记录，而是处理我们实际上想要保留在结果数据帧中的行。

因此，我们从数据帧的所有索引中排除想要删除的索引，如下所示

```
drop_idx = [1, 3, 6]
keep_idx = list(set(range(df.shape[0])) - set(drop_idx))
```

最后，代替`drop()`，我们调用`[pandas.DataFrame.take()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.take.html)`方法，该方法可以通过提供沿轴的位置索引来检索元素(在我们的例子中，这将是行轴)。

```
df = df.take(keep_idx)print(df)
 ***colA  colB   colC colD*** *0     1   125   True    A
2     3   871  False    C
4     5   908   True    D
5     6   321  False    E
7     8   678   True    C*
```

## 最后的想法

通常，我们希望通过为想要删除的相应行提供一个索引列表来删除特定的记录。在今天的简短教程中，我们演示了几种不同的方法，具体取决于您正在处理的数据的大小。

请注意，您可能希望选择满足(或不满足)特定条件的记录，而不是删除特定的记录。如果这是你正在寻找的，那么请随意阅读下面的文章，它将带你完成这个过程。

</how-to-select-rows-from-pandas-dataframe-based-on-column-values-d3f5da421e93>  

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

<https://gmyrianthous.medium.com/membership>  

**相关文章你可能也喜欢**

</how-to-efficiently-convert-a-pyspark-dataframe-to-pandas-8bda2c3875c3>  </how-to-merge-pandas-dataframes-221e49c41bec>  </data-engineer-tools-c7e68eed28ad> 