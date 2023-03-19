# 数据科学家的熊猫练习—第二部分

> 原文：<https://towardsdatascience.com/pandas-exercise-for-data-scientists-part-2-4d532cfc00bf>

## 一组具有挑战性的熊猫问题

![](img/6ce4ff7fa161f4c7219d12fa252fabc6.png)

艾伦·德·拉·克鲁兹在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

熊猫图书馆一直吸引着数据科学家用它来做令人惊奇的事情。毫无疑问，它是表格数据处理、操作和加工的首选工具。

因此，为了扩展您的专业知识，挑战您现有的知识，并向您介绍数据科学家中众多受欢迎的 Pandas 功能，我将介绍 Pandas 练习的第二部分**。你可以在这里找到熊猫练习的第一部分:**

</pandas-exercise-for-data-scientists-part-1-b601a97ee091>  

目标是增强您的逻辑能力，并帮助您用一个最好的 Python 包进行数据分析，将数据操作内在化。

在这里找到包含本次测验所有问题的笔记本: [GitHub](https://github.com/ChawlaAvi/Pandas-Quiz-P2) 。

**目录:**

[1。DataFrame](#54af)
[2 中一列的累计和。为每个组分配唯一的 id](#9c0c)
[3。检查一列是否有 NaN 值](#3872)
[4。将一个列表作为一行追加到数据帧](#689a)
[5。获取第一行中每一列的唯一值](#e829)
[6。识别熊猫合并](#e620)
[中每一行的来源 7。从数据帧](#5535)
[8 中过滤 n 个最大值和 n 个最小值。将分类数据映射到唯一的整数值](#c3bb)
[9。给每个列名添加前缀](#261e)
[10。将分类列转换为一个热点值](#4442)

作为练习，我建议你自己尝试这些问题，然后看看我提供的解决方案。

请注意，我在这里提供的解决方案可能不是解决问题的唯一方法。你可能会得出不同的结论，但仍然是正确的。然而，如果发生这种情况，请留下评论，我会很有兴趣知道你的方法。

我们开始吧！

# 1.数据帧中一列的累积和

**提示:**给你一个数据框。您的任务是从 integral 列生成一个新列，该列表示该列的累积和。

**投入和预期产出:**

**解决方案:**

在这里，我们可以对给定的序列使用 [*cumsum()*](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.cumsum.html) 方法，得到如下所示的累计和:

P.S .能不能也试试累计积，累计最大，累计最小？

# 2.为每个组分配唯一的 id

**提示:**接下来，您有一个数据帧，其中一列有重复值。你的任务是生成一个新的序列，以便每个组都有一个唯一的编号。

**输入和预期输出:**

下面， *col_A* 中的值*“A”*在新系列中被赋予值 *1* 。此外，对于每次出现的*“A”*,*group _ num*列中的值始终为 *1* 。

**解决方案:**

这里，在 *group_by* 之后，可以使用如下所示的[*grouper . group _ info*](https://pandas.pydata.org/docs/reference/api/pandas.Grouper.html)方法:

# 3.检查列是否有 NaN 值

**提示:**作为下一个问题，你的任务是判断一列中是否存在 NaN 值。不需要找出 NaN 值的个数什么的，只需要**真**或**假**列中是否有一个或多个 NaN 值。

**输入和预期输出:**

**解决方案:**

在这里，我们可以对序列使用 [*hasnans*](https://pandas.pydata.org/docs/reference/api/pandas.Series.hasnans.html) 方法来获得所需的结果，如下所示:

# 4.将列表作为一行追加到数据帧中

**提示:**大家都知道如何将元素推送到 python 列表中(在列表上使用 **append** 方法)。但是，您曾经向数据帧追加过新行吗？对于下一个任务，您将获得一个数据帧和一个列表，该列表应作为新行追加到数据帧中。

**输入和预期输出:**

**解决方案:**

这里，我们可以使用 [*loc*](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html) 并将新行分配给数据帧的新索引，如下所示:

# 5.获取列中每个唯一值的第一行

**提示:**给定一个数据帧，你的任务是获得列 *col_A* 中每个唯一元素第一次出现的整行。

**输入和预期输出:**

**解决方案:**

这里，我们将对给定的列使用 GroupBy，并获取第一行，如下所示:

# 6.识别熊猫合并中每一行的来源

**提示:**接下来，考虑你有两个数据帧。您的任务是将它们连接起来，使输出包含一个表示原始数据帧中行的来源的列。

**输入和预期输出:**

**解决方案:**

我们可以使用 [*合并*](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html?highlight=merge#pandas.DataFrame.merge) 的方法，将*指标*实参作为*真*传递，如下所示:

# 7.从数据帧中过滤 n 个最大值和 n 个最小值

**提示:**在这个练习中，给你一个数据框。您的任务是获取整个行，其在 *col_B* 中的值属于该列的 top- *k* 条目。

**投入和预期产出:**

**解决方案:**

我们可以使用 [*nlargest*](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nlargest.html) 方法，并从指定的列中传递我们需要的顶部值的数量:

与上面的方法类似，可以使用 [*nsmallest*](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nsmallest.html) 方法从列中获取前 k 个最小值。

# 8.将分类数据映射到唯一的整数值

**提示:**接下来，给定一个 DataFrame，您需要将列的每个惟一条目映射到一个惟一的整数标识符。

**输入和预期输出:**

**解决方案:**

使用 [*pd.factorize*](https://pandas.pydata.org/docs/reference/api/pandas.factorize.html) 方法，您可以生成表示给定列的基于整数的编码的新序列。

# 9.为每个列名添加前缀

**提示:**与前面的任务类似，您会得到相同的数据帧。您的工作是重命名所有的列，并添加*“pre _”*作为所有列的前缀。

**输入和预期输出:**

**解决方案:**

这里，我们可以使用 [*add_prefix*](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.add_prefix.html) 方法，将我们想要的字符串作为前缀传递给所有列名，如下所示:

# 10.将分类列转换为一个热点值

**提示:**最后，在数据帧中给出一个分类列。你需要把它转换成一个热点值。

**输入和预期输出:**

**解决方案:**

在这里，我们可以使用 [*get_dummies*](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html) 方法，并将序列作为参数传递，如下所示:

这个小测验到此结束，我希望你喜欢尝试这个。让我知道你答对了多少。另外，如果你没注意到，这个小测验可以在 Jupyter 笔记本上找到，你可以从[这里](https://github.com/ChawlaAvi/Pandas-Quiz-P2)下载。

另外，不要走开，因为我打算很快发布更多的练习。感谢阅读。