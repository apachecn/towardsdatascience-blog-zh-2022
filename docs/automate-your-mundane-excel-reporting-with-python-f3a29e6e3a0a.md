# 使用 Python 自动生成普通的 Excel 报表

> 原文：<https://towardsdatascience.com/automate-your-mundane-excel-reporting-with-python-f3a29e6e3a0a>

## 了解如何使用 Excel 自动生成 Excel 报表

![](img/ef01b57074bd1a538dc35a9ee9b85fe0.png)

由[活动创作者](https://unsplash.com/@campaign_creators?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

Excel 功能强大，无处不在。但是也是重复的，手动的。当然，您可以使用像 VBA 这样的工具来自动化报告，但是使用像 Python 这样的通用语言允许您自动化报告过程的更广泛的方面(比如移动和用电子邮件发送文件)。

在这篇文章结束时，你将学会如何:

1.  如何将多个 Excel 文件合并成一个文件
2.  用熊猫数据透视表汇总 Excel 数据
3.  在 Excel 报表中添加标题行
4.  使用 Python 将动态图表添加到 Excel 文件中
5.  用 Python 设计 Excel 文件的样式

我们开始吧！

想把这个当视频教程看？查看我下面的完整视频教程:

# 设置您的环境

为了跟随本教程，我们将使用三个库:`os`、`pandas`和`openpyxl`。其中的第一个`os`与 Python 捆绑在一起，因此您不需要安装它。然而，另外两个需要使用`pip`或`conda`来安装。

让我们看看如何在 Python 中安装这些库:

在您的终端中使用上述任何一种方法都会安装所需的库。现在，让我们来看看您可以用来跟随本教程的数据。

现在我们已经安装了库，我们可以导入将要使用的库和类:

你可以[在这里](https://github.com/datagy/mediumdata/raw/master/Excel%20Files.zip)下载文件。zip 文件包含 3 个不同的 Excel 文件，每个文件包含不同月份的销售信息。虽然像这样存储数据是有意义的，但它会使分析数据变得相当困难。

**正因为如此，我们需要先把所有这些文件组合起来**。这是您将在下一节学到的内容！

# 用 Python 组合多个 Excel 文件

Pandas 库将数据存储在 DataFrame 对象中，这可以被认为是一个 Excel 表(尽管这有点简化)。让我们来分解一下我们想要做什么，然后我们来看看如何使用 Python 来做这件事:

1.  将我们所有的文件收集到一个 Python 列表中
2.  循环遍历每个文件，并将其附加到熊猫数据帧中

让我们写一些代码，看看如何在 Python 中实现这一点！

让我们来分析一下我们在这里做了什么:

1.  在第 1 节中，我们首先加载保存文件的路径，并使用`os.list()`函数获取该文件夹中包含的所有文件的列表
2.  在第 2 节中，我们首先创建了一个空的数据帧，然后遍历每个文件，将它加载到一个数据帧中，并将其附加到我们的`combined`数据帧中
3.  最后，我们将文件保存到它自己的 Excel 文件中

那很容易，不是吗？有许多不同的方法来完成这项任务，这就是 Python 的魅力所在！

现在我们的数据已经加载完毕，让我们学习如何用熊猫来总结我们的数据。

![](img/83d1f72d300311f6c4fcbfe6b1d1f059.png)

[斯科特·格雷厄姆](https://unsplash.com/@homajob?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

# 用熊猫汇总 Excel 数据

在本节中，您将学习如何使用 Pandas 创建一个汇总表！为此，我们将使用名副其实的熊猫`.pivot_table()`函数。

该函数旨在熟悉 Excel 中的数据透视表。有了这个，我们就可以知道如何总结我们的数据。假设我们想计算出每个销售人员的总销售额。我们可以这样写:

在上面的代码中，我们使用 Pandas `.pivot_table()`函数创建了一个数据透视表。其中大部分应该感觉类似于 Excel 数据透视表。然而，默认情况下，Pandas 将使用一个聚合函数`'mean'`，所以我们需要将其设置为`'sum'`。

# 使用 OpenPyxl 向 Excel 报表添加标题行

在本节中，您将学习如何向 Excel 报表添加描述性标题行，以使它们更易于打印。为此，我们将开始使用 Openpyxl 库。

Openpyxl 的工作原理是将工作簿加载到内存中。从那里，您可以访问其中的不同属性和对象，例如工作表和这些工作表中的单元格。

这里的关键区别在于，您是直接使用*工作簿。对于 Pandas，我们只需将工作簿的*保存到*。(我承认还有更多，但这是一个很好的思考方式。)*

在上面的代码中:

1.  我们加载了一个工作簿对象`wb`。从那里，我们访问工作表。
2.  我们能够通过使用`.insert_rows()`方法插入三行来操作工作表。
3.  这些行是在索引 0(第一行)处添加的，包括三行。我们给两个单元格分配了有用的值。
4.  最后，我们使用 Openpyxl `.save()`方法保存工作簿。

![](img/aaf57b7e8abd455bae56142e539dc923.png)

由 [Towfiqu barbhuiya](https://unsplash.com/@towfiqu999999?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

# 使用 Python 向 Excel 添加动态图表

在这一节中，我们将了解如何向 Excel 文件添加图表。Openpyxl 库的一大优点是我们可以创建基于 Excel 的图表，这些图表对数据保持动态。

为了使我们的图表动态化，我们需要创建`Reference()`对象，顾名思义，这些对象保存对工作簿中位置的引用。

我们将创建一个条形图，并将数据和类别粘贴到工作簿中。让我们来看看如何做到这一点:

让我们来分解一下上面的代码是做什么的:

1.  我们设置了两个`Reference`对象:一个用于我们的数据，一个用于我们的类别。这些引用对象将把我们的图表与特定的单元格联系起来，使它们保持动态。
2.  我们添加一个`BarChart`对象。对于图表，我们使用`.add_data()`和`.set_categories()`方法来传递我们的引用对象。
3.  最后，我们将图表添加到工作表上的特定锚点。

# 用 Python 设计 Excel 报表的样式

在这最后一节，我们将看看如何使用 OpenPyxl 来设计工作簿的样式。OpenPyxl 可以基于 Excel 中存在的样式向 Excel 工作簿中的单元格添加样式。

这意味着我们可以添加像`Title`和`currency`这样的样式。让我们使用`.style`属性来看看这是如何工作的:

在上面的例子中，我们使用了`.style`属性为不同的单元格分配不同的样式。我们在单元格 5–6 中使用 for 循环的原因是 OpenPyxl 不允许您为范围分配样式，所以我们一次分配一个样式。

# 下一步做什么？

太好了！您已经自动完成了枯燥的 Excel 报告！现在，你接下来应该看什么？我建议考虑您可能需要自动化的其他元素。例如:

*   如何向工作表中添加姓名？
*   你如何能自动地用电子邮件发送你的结果文件？
*   如何将值样式化为表格？

# 结论

感谢您一路阅读教程！我希望你喜欢它，并且你学到了一些东西。使用 Python 来自动化您的工作是一件非常值得学习的事情。如果你想了解更多，请点击[这里](http://www.youtube.com/channel/UCm7fT9_IdCDbeBXjAwL9IIg?sub_confirmation=1)订阅我的 YouTube 频道。