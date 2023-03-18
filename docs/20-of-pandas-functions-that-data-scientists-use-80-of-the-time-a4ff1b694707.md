# 数据科学家 80%的时间使用熊猫 20%的功能

> 原文：<https://towardsdatascience.com/20-of-pandas-functions-that-data-scientists-use-80-of-the-time-a4ff1b694707>

## 将帕累托法则运用于熊猫图书馆

![](img/d05d6f8418d15e9641c2911e1c38ca0d.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Austin Distel](https://unsplash.com/@austindistel?utm_source=medium&utm_medium=referral) 拍摄的照片

掌握像*熊猫*这样的整个 Python 库对任何人来说都是具有挑战性的。然而，如果我们退一步思考，我们真的需要了解特定库的每一个细节吗，尤其是当我们生活在一个受帕累托法则支配的世界中时？对于那些不知道的人来说，帕累托法则(也称为 80-20 法则)说，你 20%的投入总会产生 80%的产出。

因此，这篇文章是我将帕累托法则应用于 Pandas 库的尝试，并向你介绍你可能用 80%的时间处理数据帧的 20%的 Pandas 函数。下面提到的方法是我发现自己在日常工作中反复使用的方法，我觉得对于任何开始接触熊猫的人来说，熟悉这些方法是必要和充分的。

# 1/n:读取 CSV 文件:

如果您想在 Pandas 中读取一个 CSV 文件，使用 *pd.read_csv()* 方法，如下所示:

![](img/d5efab0dd137d0e9e45798a5da7dcfb8.png)

读取 CSV 文件的代码片段(作者使用 snappify.io 创建的图片)

在这里阅读文档。

# 2/n:将数据帧保存到 CSV 文件:

如果您想将数据帧保存到 CSV 文件，使用如下所示的 *to_csv()* 方法:

![](img/eebf0c00392770e80b0dc2c520e7305b.png)

将数据帧保存到 CSV 文件的代码片段(作者使用 snappify.io 创建的图片)

在这里阅读文档。

# 3/n:从列表列表中创建数据帧:

如果你想从一系列列表中创建一个数据帧，使用 *pd。DataFrame()* 方法如下所示:

![](img/1f7c68100c2a634e13841b769965a431.png)

用于从列表的列表中创建 DataFrame 的代码片段(作者使用 snappify.io 创建的图片)

在这里阅读文档。

# 4/n:从字典创建数据帧:

如果你想从字典创建一个数据帧，使用 *pd。DataFrame()* 方法如下所示:

![](img/105ad14638a64e2dabda4aa024cfcd1b.png)

从字典创建 DataFrame 的代码片段(作者使用 snappify.io 创建的图片)

在此阅读文档[。](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

# 5/n:合并数据帧:

DataFrames 中的 Merge 操作与 SQL 中的 JOIN 操作相同。我们用它来连接一列或多列上的两个数据帧。如果您想要合并两个数据帧，使用如下所示的 *pd.merge()* 方法:

![](img/1126ffd38288984c6666010a0dd1e2b9.png)

用于合并数据帧的代码片段(作者使用 snappify.io 创建的图片)

在这里阅读文档。

# 6/n:对数据帧进行排序:

如果您想要根据特定列中的值对数据帧进行排序，请使用如下所示的 *sort_values()* 方法:

![](img/50ceea7850ee54e4960ed674b9588425.png)

用于对数据帧进行排序的代码片段(作者使用 snappify.io 创建的图片)

在这里阅读文档。

# 7/n:连接数据帧:

如果您想要连接数据帧，请使用 *pd.concat()* 方法，如下所示:

![](img/a3a51750bd65b1df76e04c117941db23.png)

用于连接数据帧的代码片段(作者使用 snappify.io 创建的图片)

在这里阅读文档。

*   *axis = 1 将列堆叠在一起。*
*   *轴= 0 将行堆叠在一起，前提是列标题匹配。*

# 8/n:重命名列名:

如果要重命名数据帧中的一列或多列，使用如下所示的 *rename()* 方法:

![](img/7838358b5e07418e48fdf0636013e301.png)

用于重命名 DataFrame 中的列的代码段(作者使用 snappify.io 创建的图片)

在这里阅读文档。

# 9/n:添加新列:

如果要向数据帧添加新列，可以使用通常的赋值操作，如下所示:

![](img/d49eab09228754b90e2af325b0a09f9f.png)

向 DataFrame 添加新列的代码段(作者使用 snappify.io 创建的图片)

# 10/n:根据条件过滤数据帧:

如果要根据条件从数据帧中筛选行，可以如下所示进行:

![](img/333cda0666cf25b9c009bb6a72adad87.png)![](img/9050da38850c9cf1dd935ba95fa554bc.png)

用于过滤数据帧的代码片段(作者使用 snappify.io 创建的图片)

# 11/n:删除列:

如果要从数据帧中删除一列或多列，使用如下所示的 *drop()* 方法:

![](img/85c13bb37e2a967e255d04752c7fde7d.png)

从 DataFrame 中删除列的代码段(作者使用 snappify.io 创建的图片)

在此阅读文档[。](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html?highlight=drop#pandas.DataFrame.drop)

# 12/n:分组依据:

如果要在分组后执行聚合操作，请使用如下所示的 *groupby()* 方法:

![](img/035684de043aafe4f49c1e15088dc78c.png)

用于对数据帧进行分组的代码片段(作者使用 snappify.io 创建的图片)

在这里阅读文档。

# 13/n:列中的唯一值:

如果您想计算或打印数据帧的一列中的唯一值，使用如下所示的 *unique()或 unique()* 方法:

![](img/b1eaaf571cc5c367a03e65a38a6ee6ea.png)

用于在 DataFrame 列中查找唯一值的代码段(作者使用 snappify.io 创建的图片)

在这里阅读文档。

# 14/n:填充 NaN 值:

如果您想用其他值替换列中的 NaN 值，请使用如下所示的 *fillna()* 方法:

![](img/61c9cbc85093edff4e3b0778fe480b38.png)

用于在 DataFrame 中填充 NaN 值的代码片段(作者使用 snappify.io 创建的图片)

在这里阅读文档。

# 15/n:对列应用函数:

如果您想将一个函数应用到一个列，使用如下所示的 *apply()* 方法:

![](img/00e9132a0f0250f6aa1084db37224a43.png)

对数据帧应用函数的代码片段(作者使用 snappify.io 创建的图片)

在这里阅读文档。

# 16/n:删除重复项:

如果要删除重复值，请使用 *drop_duplicates()* 方法，如下所示:

![](img/c22236889d5a3e6be4cd43f8e06ef5b5.png)

用于从数据帧中删除重复数据的代码片段(作者使用 snappify.io 创建的图片)

在这里阅读文档。

# 17/n:数值计数:

如果您想要查找列中每个值的频率，请使用如下所示的 *value_counts()* 方法:

![](img/5693ec28e05dd6ee283933a447e0dd6a.png)

计算列中值的频率的代码片段(作者使用 snappify.io 创建的图片)

# 18/n:数据帧的大小:

如果要查找数据帧的大小，请使用。*形状*属性如下图所示:

![](img/469d642126fcc7dae2ae2b9ff670452c.png)

最后，在这篇文章中，我介绍了 Pandas 中一些最常用的函数/方法，以帮助您开始使用这个库。虽然这篇文章将有助于你熟悉语法，但我强烈建议你创建一个虚拟的数据框架，并在 jupyter 笔记本上进行实验。

此外，没有比参考熊猫官方文件更好的地方了[这里](https://pandas.pydata.org/docs/)获得熊猫各种方法的基本和实用知识。panda 的官方文档提供了一个函数所接受的每个参数的详细解释，以及一个实际的例子，在我看来，这是获得 panda 专业知识的一个很好的方法。

感谢阅读。我希望这篇文章是有帮助的。