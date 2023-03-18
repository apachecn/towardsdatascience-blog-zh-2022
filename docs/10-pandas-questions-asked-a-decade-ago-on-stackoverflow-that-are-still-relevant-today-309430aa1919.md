# 十年前在 StackOverflow 上问的 10 个熊猫问题在今天仍然适用

> 原文：<https://towardsdatascience.com/10-pandas-questions-asked-a-decade-ago-on-stackoverflow-that-are-still-relevant-today-309430aa1919>

## 重温熊猫的基本操作

![](img/7ce99b2c4c5f445cbe7d332febcb2f9d.png)

照片由[埃文·丹尼斯](https://unsplash.com/@evan__bray?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

随着越来越多的结构化表格数据的出现，对于这个时代的数据科学家来说，熟悉使用 Pandas 进行表格数据分析和管理是绝对必要的。虽然自学是获得专业知识的一个很好的方法，但有时，通过寻找他人面临问题的答案来跟随同伴学习也有助于自我成长。

因此，为了在这个方向上迈出一步并提高您的熊猫专业知识，在这篇博客文章中，我汇集并解决了 StackOverflow 上与熊猫相关的十大**投票最多的**问题，作为一名数据科学家，您必须知道如何解决这些问题。成为投票最多的问题之一凸显了数据科学界的普遍兴趣及其发展方向。

你可以在这里找到这篇文章[的代码。下面提到了这篇文章的亮点:](https://deepnote.com/workspace/avi-chawla-695b-aee6f4ef-2d50-4fb6-9ef2-20ee1022995a/project/StackOverflow-Questions-d5aaa600-f9de-4918-a993-ef3283cfd6c0/%2Fnotebook.ipynb)

```
[**Q1: How to iterate over rows in a DataFrame in Pandas?**](#59ff)[**Q2: How do I select rows from a DataFrame based on column values?**](#2b64)[**Q3: Renaming column names in Pandas**](#0c5a)[**Q4: Delete a column from a Pandas DataFrame**](#71bb)[**Q5: How do I get the row count of a Pandas DataFrame?**](#0330)[**Q6: Selecting multiple columns in a Pandas dataframe**](#0bf7)[**Q7: How to change the order of DataFrame columns?**](#9683)[**Q8: Change column type in pandas**](#23f2)[**Q9: Get a list from Pandas DataFrame column headers**](#7be5)[**Q10: Create a Pandas Dataframe by appending one row at a time**](#e389)
```

我们开始吧🚀！

> **注意:**下面提到的解决方案描述了我自己解决这些问题的方法，并没有从任何来源复制。而且 StackOverflow 上所有用户生成的内容都是授权商业使用的(CC BY 4.0)，可以免费使用。

# [Q1:如何在 Pandas 中迭代数据帧中的行？](https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas)

迭代(也称为循环)是单独访问数据帧中的每一行并执行一些操作。

考虑下面的数据框架:

在熊猫中，你可以用三种不同的方式迭代，使用`range(len(df))`、`iterrows()`和`itertuples()`。

我在下面的博文中详细讨论了迭代数据帧的不同方法:

[](/five-killer-optimization-techniques-every-pandas-user-should-know-266662bd1163) [## 每个熊猫用户应该知道的五个黑仔优化技术

### 数据分析运行时优化的一步

towardsdatascience.com](/five-killer-optimization-techniques-every-pandas-user-should-know-266662bd1163) 

# [Q2:我如何根据列值从数据框中选择行？](https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values)

此问题旨在了解如何根据条件过滤数据帧。要了解常用的过滤方法，请考虑下面的数据框架:

过滤数据帧的一些方法实现如下:

上面使用的`[isin()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isin.html)`方法接受一个过滤器值列表。另一方面， [query()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html) 方法计算一个字符串表达式来过滤数据帧中的行。

# [Q3:重命名 Pandas 中的列名](https://stackoverflow.com/questions/11346283/renaming-column-names-in-pandas)

这里的目标是更改列标题的名称。考虑与上面相同的数据帧。

我们可以使用`[rename()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html)`方法将`col1`的名称改为`col3`，如下所示:

这里，原始数据帧保持不变。如果不想创建新的数据帧，使用如下所示的`inplace=True`:

使用`rename()`方法时，您需要创建一个从`old-column-name`到`new-column-name`的映射作为字典。如果列名保持不变，则不需要在字典中指定它。

# [Q4:从熊猫数据帧中删除一列](https://stackoverflow.com/questions/13411544/delete-a-column-from-a-pandas-dataframe)

要从 DataFrame 中删除一列或多列，可以使用 drop()方法，并将要删除的列作为列表传递。这在下面的`Method 1`和`Method 2`中显示。或者，如`Method 3`所示，您可以选择想要在最终数据帧中保留的列子集。

`drop()`方法的语法类似于`rename()`的语法，唯一的区别是`columns`参数接受要删除的列的列表。

# 问题 5:我如何获得熊猫数据帧的行数？

这个问题围绕着了解熊猫数据帧的形状。要回答这个问题，请考虑以下三行两列的数据帧:

要找到形状，使用 DataFrame 的`shape`属性，如下所示:

属性返回一个`python`元组。第一个元素对应于行数，第二个元素表示列数。

# [Q6:在熊猫数据框架中选择多列](https://stackoverflow.com/questions/11285613/selecting-multiple-columns-in-a-pandas-dataframe)

这里要实现的目标是从数据帧中选择一个以上的列进行进一步处理。例如，如果原始数据帧包含三列，即`col1`、`col2`和`col3`，如何仅选择`col1`和`col3`。

有两种方法可以做到这一点:

`iloc`中的列表`[0,2]`解释为位于第 0(`col1`)和第 2(`col3`)索引的列。你可以在这里阅读更多关于`iloc` [的内容](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html)。

# [Q7:如何改变 DataFrame 列的顺序？](https://stackoverflow.com/questions/13148429/how-to-change-the-order-of-dataframe-columns)

改变数据帧中列的顺序指的是在不改变列的数量(或数据帧的形状)的情况下重新排列列。

考虑下面的数据框架。目标是将列排列为`col1` - `col2` - `col3`。

有两种方法可以做到:

`iloc`中的列表`[2,0,1]`解释为位于第 2(`col1`)、第 0(`col2`)和第 1(`col3`)索引的列。

# [Q8:更改熊猫中的列类型](https://stackoverflow.com/questions/15891038/change-column-type-in-pandas)

通过这个问题，目的是知道如何改变一个列的数据类型。考虑下面的数据帧，其中`col1`将整数值保存为字符串。

`col1`的当前数据类型为`object`(与`string`相同)。目的是将`col1`的数据类型从`string`更改为`integer`。您可以按如下方式更改数据类型:

如果不想创建列，可以将新值存储在同一列中，如下所示:

使用`astype()`方法时，必须确保从源数据类型到目标数据类型的类型转换是可行的。例如，您不能将字母的`string`列转换为`integer`或`float`数据类型。

# [Q9:从 Pandas DataFrame 列标题中获取列表](https://stackoverflow.com/questions/19482970/get-a-list-from-pandas-dataframe-column-headers)

这里的目标是以列表的形式获取数据帧中所有列的名称。考虑下面的数据框架:

要获得列的列表，使用如下所示的`columns`属性:

上面的代码将列作为索引对象返回。要以列表形式获取它，请将获得的结果转换为列表:

# [Q10:通过一次追加一行来创建一个熊猫数据帧](https://stackoverflow.com/questions/10715965/create-a-pandas-dataframe-by-appending-one-row-at-a-time)

在这个问题中，目标是一次将一行追加到原来为空的数据帧中。假设您有以下空数据帧和一个列表列表`data`，其中每个单独的子列表将作为一行添加到数据帧中。

为了一次追加一行，我们必须遍历`data`列表并添加一个新行，如下所示:

如上文 Q5 中所讨论的，由`shape`属性返回的元组的第一个元素表示数据帧中的行数。因此，添加到数据帧的每个新行都确保为下一行创建新索引。

总之，在这篇文章中，我解决了 StackOverflow 上熊猫类的十个投票最高的问题。如果你愿意进一步解决列表中的问题，我认为你应该这样做，你可以在这里找到它们。

或者，如果你想在熊猫中练习一些具有挑战性的问题，可以参考下面我之前的两篇博文来练习熊猫。

[](/pandas-exercise-for-data-scientists-part-1-b601a97ee091) [## 数据科学家的熊猫练习—第一部分

### 一组具有挑战性的熊猫问题。

towardsdatascience.com](/pandas-exercise-for-data-scientists-part-1-b601a97ee091) [](/pandas-exercise-for-data-scientists-part-2-4d532cfc00bf) [## 数据科学家的熊猫练习—第二部分

### 一组具有挑战性的熊猫问题

towardsdatascience.com](/pandas-exercise-for-data-scientists-part-2-4d532cfc00bf) 

> 有兴趣在媒体上阅读更多这样的故事吗？？

✉️ [**注册我的电子邮件列表**](https://medium.com/subscribe/@avi_chawla) 不要错过另一篇关于数据科学指南、技巧和提示、机器学习、SQL、Python 等的文章。Medium 会将我的下一篇文章直接发送到你的收件箱。

**感谢阅读！**