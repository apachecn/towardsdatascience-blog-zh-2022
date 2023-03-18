# 用这个窗口函数优化你的 SQL 代码

> 原文：<https://towardsdatascience.com/optimize-your-sql-code-with-this-window-function-409d3341cb20>

## 通过使用 FIRST_VALUE()来替换您痛苦的 cte

![](img/7cdfdea4ec74bd0e73b3f41c1aa49a24.png)

照片由 [Say Cheeze 工作室](https://unsplash.com/@saycheezestudios?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/first?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

我们都遇到过这种 SQL 代码块…它很容易阅读，但是它的某些部分就是，嗯，*烦人的*。通常，可以用一个查询编写的代码被视为 CTE。

在清理一些旧的数据模型时，我遇到过很多这样的代码。其他代码很容易修复，因为读起来很痛苦，而且如何改进也很简单。然而，这并不是板上钉钉的事。

我说的是什么 SQL 代码？

这个。

# 问题是

```
WITHorders AS (
   SELECT 
      name, 
      model,
      year, 
      date_at_lot
      row_number() over(partition by model, year order by date_at_lot asc) AS order
   FROM cars 
)SELECT
   name AS oldest_car_name,
   model, 
   year
FROM orders
where order = 1 
```

这是一个你非常想把它写得更好，但又不确定如何去做的 cte。这并不是说它超级低效，但是如果你不断地写这样的查询，读起来就变得多余了。

还有，可以的话为什么不优化一下呢？

# 解决方案

如果你熟悉[窗口功能](/how-to-use-sql-window-functions-5d297d29f810)，那么你可能已经知道我在这里暗示的功能。然而，这个窗口功能并没有得到应有的重视。

当我第一次学习窗口函数的时候，我从来没有接触过这个函数！你经常听说`[LEAD()](/how-to-use-sql-rank-and-dense-rank-functions-7c3ebf84b4e8?source=your_stories_page----------------------------------------)`、`[RANK()](/how-to-use-sql-rank-and-dense-rank-functions-7c3ebf84b4e8)`、`ROW_NUMBER()`，但从来没听说过`FIRST_VALUE()`。

## FIRST_VALUE()返回有序、分区数据输出中的第一个值。

它基本上取代了在一个查询中使用`ROW_NUMBER()`然后在下一个查询中使用`column = 1`进行过滤的需要。

它的工作原理与任何其他窗口函数相同，您可以在其中指定希望对数据进行分区的列以及希望对它们进行排序的顺序。

## 分割

如果您不熟悉的话，`PARTITION BY`帮助对数据进行分组，以便对每个分区重新开始计数。

例如，使用`FIRST_VALUE()`时按型号和年份划分会将数据分成不同的组，这些组具有相同的汽车型号和年份。然后，在这些组中，它将根据组的排序方式找到第一个值。

所有 2013 款福特 Escapes 将被分组在一起，然后所有 2013 款福特 Escapes 的 **first_value** 将作为输出，这取决于`ORDER BY`子句中指定的内容。

如果不包含`PARTITION BY`子句，该函数会将整个数据集视为一个单独的分区。

## 排序

有了 first_value 函数，`ORDER BY`就显得尤为重要。如果你用错了方法，你可能得到的是最后一个值而不是第一个值。在对数据分区进行排序时，确保使用`ASC`或`DESC`来正确指定排序。

例如，如果您想要某个型号的最新汽车，您想要`ORDER BY date_at_lot DESC` 。但是，如果您想要某个型号的批次中最老的汽车，您需要编码`ORDER BY date_at_lot ASC`。

您也可以选择使用`LAST_VALUE()`函数，其工作方式相同，但选择数据子集中的最后一个值，而不是第一个值。这完全取决于您如何对数据进行排序。

那么，使用`ROW_NUMBER()`的例子和使用`FIRST_VALUE()`的例子会是什么样子呢？让我展示给你看。

```
SELECT 
   FIRST_VALUE(name) OVER(PARTITION BY model, year ORDER BY date_at_lot ASC) AS oldest_car_name
   model,
   year
FROM cars
```

注意我是如何在函数的`FIRST_VALUE()`部分包含`name`的。这样做将导致该函数输出第一辆汽车的名称。您可以在这里指定任何列，但是这个名称最有意义，因为它在不同的型号和年份中都是唯一的。

# 结论

现在您已经准备好进一步优化您的 SQL 代码了！停止使用`ROW_NUMBER()`切换到`FIRST_VALUE()`对第一个输出进行滤波。这个功能的创建是有原因的，现在是我们都开始使用它的时候了！

不断改进你的代码是成为更好的程序员、分析师、工程师等的关键。当我们停止进步，我们就停止成长。我强烈建议对您的旧代码进行这样的小改动，不断优化它，并不断学习。

如果你想获得更多关于编写 SQL、成为一名[分析工程师](/analytics-engineer-the-newest-data-career-role-b312a73d57d7)或学习现代数据堆栈的技巧，请订阅我即将发布的[时事通讯](https://mailchi.mp/e04817c8e57e/learn-analytics-engineering)。