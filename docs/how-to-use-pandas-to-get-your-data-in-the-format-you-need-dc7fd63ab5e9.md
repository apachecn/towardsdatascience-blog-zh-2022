# 如何使用 Pandas 以您需要的格式获取数据

> 原文：<https://towardsdatascience.com/how-to-use-pandas-to-get-your-data-in-the-format-you-need-dc7fd63ab5e9>

## 了解长格式和宽格式数据之间的区别，以及如何在 Pandas 中进行转换。

![](img/ec9968af97868e21f9f11692717a2be5.png)

戴维·贝克尔在 Unsplash[上的照片](https://unsplash.com?utm_source=medium&utm_medium=referral)

数据科学家都知道这样一个事实:你的数据永远不会是你想要的样子。您可能会得到一个有点条理的电子表格或合理的合理表格，但是在您准备好进行分析之前，总会有一些清理工作要做。

因此，能够在不同形式的数据之间转换至关重要。有时，这只是可读性和易于解释的问题。其他时候，你会毫不夸张地发现，除非你的数据是特定的格式，否则你试图使用的软件包或模型根本无法工作。不管是什么情况，这都是一项很好的技能。

在本文中，我将讨论两种常见的数据形式:**长格式数据**和**宽格式数据**。这些是数据科学中广泛使用的范例，熟悉它们是有好处的。我们将通过一些例子来了解这两种数据格式到底是什么样子，然后我们将了解如何使用 Python(更具体地说，是 Pandas)在它们之间进行转换。

让我们开始吧。

## 长格式与宽格式数据

最简单的方法是从直接的定义开始[1]:

*   **宽格式数据**为自变量的每个可能值提供一行，所有因变量记录在列标签中。因此，每一行中的标签(对于自变量)将是唯一的。
*   **长格式数据**每个观察值占一行，每个因变量作为一个新值记录在多行中。因此，独立变量*的值在行内重复*。

好吧，酷——但这意味着什么？如果我们看一个例子会更容易理解。假设我们有一个学生的数据集，我们存储他们在期中考试、期末考试和课堂项目中的分数。概括地说，我们的数据如下所示:

![](img/92ce8ed5b7196e49176188a4d5b7c307.png)

作者图片

这里，每个学生都是自变量，每个分数都是各自的因变量(因为特定考试或项目的分数取决于学生)。我们可以看到,`Student`的值对于每一行都是唯一的，正如我们对宽格式数据的预期。

现在让我们看看完全相同的数据，但形式更长:

![](img/3182879b2d56a151cae3802fcf201124.png)

作者图片

这一次，我们为每个观察值安排了一行。在这种情况下，观察对应于特定作业的分数。在上面这个数据的宽格式版本中，我们在一行中记录了多个观察值(分数)，而这里每一行都有自己的分数。

此外，我们可以看到自变量`Student`的值以这种数据格式重复出现，这也是我们所期望的。

一会儿，我们将讨论为什么您应该关心这些不同的格式。但是首先，让我们快速看一下如何使用 Pandas 在这些不同的数据格式之间进行转换。

## 宽格式数据到长格式数据:Melt 函数

再一次，让我们看看上面的宽格式数据。这一次，我们将为数据帧起一个名字:`student_data`:

![](img/92ce8ed5b7196e49176188a4d5b7c307.png)

作者图片

为了将学生数据转换成长格式，我们使用下面一行代码:

```
student_data.melt('Student', var_name='Assignment', value_name='Score')
```

![](img/3182879b2d56a151cae3802fcf201124.png)

作者图片

以下是一步一步的解释:

*   `melt`函数被设计成将宽格式数据转换成长格式数据[2]。
*   `var_name`参数指定了我们想要命名的第二列——包含我们各自的因变量的那一列。
*   `value_name`参数指定了我们想要命名的第三列——包含我们正在观察的单个值的列(在本例中是分数)。

好了，现在我们有了长格式数据。但是，如果——不管出于什么原因——我们需要回到宽格式呢？

## 长格式数据到宽格式数据:Pivot 函数

现在，我们从上面数据的长版本开始，称为`student_data_long`。下面一行代码会将其转换回原始格式:

```
student_data_long.pivot(index='Student', columns='Assignment', values='Score')
```

![](img/3d64a2698f78f823de9849453c90cf3c.png)

作者图片

除了稍微更新的标签(`pivot`显示整个列标签`'Assignment'`，这正是我们从上面开始的数据。

以下是一步一步的解释:

*   `pivot`函数被设计成将宽格式数据转换成长格式数据[3]，但是[实际上可以完成比这里显示的更多的事情](https://medium.com/towards-data-science/a-comparison-of-groupby-and-pivot-in-pythons-pandas-module-527909e78d6b) [4]。
*   `index`参数指定我们希望将哪一列的值作为唯一的行(即独立变量)。
*   `columns`参数指定哪个列的唯一值(长格式)将成为唯一列标签。
*   `values`参数指定哪一列的标签将构成我们宽格式中的实际数据条目。

这就是全部了！

## 为什么重要？

最后，我想简单强调一下，虽然上面的乍一看似乎很肤浅，但实际上这是一项非常有用的技能。很多时候，你会发现以某种格式保存数据会让你的生活变得更加容易。

我用自己工作中的一个例子来说明。我经常需要用 Python 做数据可视化，我选择的模块是 Altair。这导致了一个意想不到的问题:大多数电子表格倾向于宽格式，但 Altair 的规范在长格式中使用起来要容易得多。

今年早些时候，我花了很长时间来开发一个特别的可视化。当我接受堆栈溢出时，我发现我需要做的就是将我的数据转换成长格式。如果你持怀疑态度，[你可以自己去看看这个帖子](https://stackoverflow.com/questions/71059865/is-there-a-way-to-transform-a-nominal-dataframe-into-a-bubble-plot-in-altair)。

现在，你可能不从事可视化工作，但是如果你正在阅读这篇文章，可以肯定你确实在从事数据工作。因此，你应该知道如何操作它，这只是你的工具箱中又一个有用的技能。

祝你的数据科学事业好运。

**想擅长 Python？** [**获取独家，免费获取我简单易懂的指南**](https://witty-speaker-6901.ck.page/0977670a91) **。想在介质上无限阅读故事？用我下面的推荐链接注册！**

[](https://murtaza5152-ali.medium.com/?source=entity_driven_subscription-607fa603b7ce---------------------------------------)  

## 参考

[1][https://altair-viz.github.io/user_guide/data.ht](https://altair-viz.github.io/user_guide/data.ht)
【2】[https://pandas . pydata . org/docs/reference/API/pandas . melt . html](https://pandas.pydata.org/docs/reference/api/pandas.melt.html)
【3】[https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot.html)
【4】[https://medium . com/forward-data-science/a-comparison-of-group by-and-pivot-in-python-pandas-module-527909 e78d6b](https://medium.com/towards-data-science/a-comparison-of-groupby-and-pivot-in-pythons-pandas-module-527909e78d6b)