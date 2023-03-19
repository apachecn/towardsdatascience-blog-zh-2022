# 6 熊猫数据框架任务任何学习 Python 的人都应该知道

> 原文：<https://towardsdatascience.com/6-pandas-dataframe-tasks-anyone-learning-python-should-know-1aadce307d26>

## 确保你能够在 Pandas 中完成以下任务

![](img/aa0995d4df99e29de25db12bf80d41e0.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的 [ThisisEngineering RAEng](https://unsplash.com/@thisisengineering?utm_source=medium&utm_medium=referral)

当用 Python 编程时，尤其是在数据科学领域，可能没有人会避开库熊猫。在许多应用程序中，数据必须以表格的形式使用，在 Pandas DataFrames 的帮助下，这在 Python 中是最容易处理的。

确保你知道下面的命令，并且可以毫不费力地使用它们，这样你就可以节省时间，每天最大限度地利用熊猫。

# 例子

在本文中，我们使用以下数据作为示例来测试我们的命令:

对于我们的例子，我们简单地生成一个 DataFrame，其中有两列和日期作为索引。我们简单地使用 Numpy 随机填充数值。

# 1.添加新列

有几种方法可以将新列添加到现有的数据框架中。通过简单地用方括号定义新列，它将作为新列从右边添加到数据帧。

相反，如果我们希望在特定索引处插入新列，我们可以使用“df.insert()”来实现。

传递给该函数的第一个值是要插入的新列的索引，然后是列的名称，最后是要作为列插入的对象。最后一个参数指定是否允许该列重复。因此，如果具有相同名称和相同值的列已经存在，并且“allow_duplicates”设置为“False”，那么您将得到一条错误消息。

# 2.添加新行

如果以后要以新行的形式添加值，您可以简单地定义它并为它提供一个新的索引。传递的值列表必须与数据帧中的列数相匹配，这一点很重要。

# 3.删除列

与任何优秀的 Pandas 命令一样，删除列有几个选项。最简单的两种方法是使用函数“df.drop()”和列名，以及使用“axis=1”进行列选择。或者您可以使用标准 Python 函数“del”并定义相应的列:

# 4.删除空字段

当我们将空值传递给 DataFrame 或其他 Pandas 对象时，它们会自动被 Numpy NaNs(不是数字)替换。对于计算，如平均，这些字段不包括在内并被忽略。我们可以简单地用一个空的第三列来扩展现有的数据帧，该列只包含索引 01/01/2022 的值。然后，其余值自动设置为 NaN。

如果我们想要删除至少一列中有空值的所有行，我们可以使用下面的命令。

如果我们想删除缺少值的列，我们使用相同的命令并额外设置' axis = 1 '。否则，我们也可以用预定义的值填充空字段，例如用值 0。

在某些情况下，将缺失值显示为布尔值(真/假)也很有用。然而，在大多数情况下，DataFrame 对象太大，这不是一个有用的表示。

# 5.删除一行

如果我们不想只是从数据帧中删除空值，而是删除行，有两种方法可以做到。首先，我们可以通过使用我们想要删除的行的索引从数据帧中删除行。在我们的例子中，这是一个具体的日期，比如 01.01.2022:

通过这样做，我们删除了这个对象中的第一行。但是，在大多数情况下，我们还不知道要删除的具体行。然后，我们还可以将数据帧过滤为我们想要删除的行，然后输出相应行的索引。

在这种情况下，我们删除在“列 1”中检测到值大于 0.1 的所有行。这在“df”对象中总共留下了四行。

# 6.合并熊猫对象

Pandas 提供了几种连接系列或数据帧对象的方法。concat 命令通过第二个命名的对象来扩展第一个命名的对象(如果它们属于同一类型)。该命令当然可以用两个以上的数据结构来执行。

对于数据帧，代码行看起来是一样的。附加的“ignore_index”用于分配新的连续索引，而不是来自原始对象的索引。

Pandas 还允许使用“Merge”进行连接，大多数人可能对 SQL 很熟悉。

如果我们想执行一个内连接而不是左连接或右连接，我们再次使用 Concat 命令并加上' join = "inner " '。

# 这是你应该带走的东西

*   熊猫为处理缺失的价值观提供了许多可能性。您可以删除有问题的列/行，或者用值替换字段。
*   对于 Pandas，我们拥有与 SQL 相同的连接可能性。

*如果你喜欢我的作品，请在这里订阅*<https://medium.com/subscribe/@niklas_lang>**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，medium 允许你每月免费阅读* ***3 篇*** *。如果你希望有****无限制的*** *访问我的文章和数以千计的精彩文章，请不要犹豫，点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格**

*<https://medium.com/illumination/intuitive-guide-to-artificial-neural-networks-5a2925ea3fa2>  </4-basic-commands-when-working-with-python-tuples-8edd3787003f>  </an-introduction-to-tensorflow-fa5b17051f6b> *