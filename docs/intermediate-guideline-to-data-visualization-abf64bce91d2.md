# 数据科学数据可视化中级指南(下)

> 原文：<https://towardsdatascience.com/intermediate-guideline-to-data-visualization-abf64bce91d2>

## Python 数据可视化:标准指南

![](img/83242fd6a4b0b73af2ac0ae97e202352.png)

[aisvri](https://unsplash.com/@aisvri?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

## 动机

> ***“数字有一个重要的故事要讲。他们依靠你给他们一个清晰而令人信服的声音****——斯蒂芬几*。

如果没有可视化，数据就是一些数值或分类值的集合。适当的可视化为找到有价值的信息铺平了道路。视觉化越舒缓，人们就越能掌握信息。

我们大多数人都熟悉像`*scatter plots, bar charts, pie charts, histograms, line plots, etc.*` *这样的普通图，除了*这样的普通图，许多其他图可以更精确地表示信息。

本文将向您介绍一些有趣的文章来可视化信息。

*【注意 Python 中有很多数据可视化库。为了使内容简单，我使用了 matplotlib、seaborn 和 plotly 库。】*

## 目录

1.  `[**Dataset Description**](#01e1)`
2.  `[**Stacked Bar Chart**](#6a06)`
3.  `[**Grouped Bar Chart**](#9f3d)`
4.  `[**Stacked Area Chart**](#9858)`
5.  `[**Pareto Diagram**](#8b5b)`
6.  `[**Donut chart**](#5710)`
7.  `[**Heatmap**](#0cd4)`
8.  `[**Radar Chart**](#83eb)`
9.  `[**Tree Map**](#1f8c)`

## 数据集描述

出于演示的目的，我创建了一个孟加拉蔬菜的合成数据集。数据集包含变量`**Vegetable, Season, Weight and Price**`，其中`Vegetable and Season, Weight` 和`Price` 是连续的分类和数值变量。

[*在这里找到数据集*](https://deepnote.com/workspace/zubair063-9767-9eddd116-b682-475a-9479-3807be1e71db/project/Data-Visualization-b20ec91c-3126-4ea1-9377-d7b7bddab634/%2FVegetable.csv) *。*一瞥数据集。

## **堆积条形图**

堆积条形图是一种特殊类型的条形图。我们可以在堆积条形图中集成比传统条形图更多的信息。

为了更好地理解，我们举一个用 python 编写的例子。

在上面堆叠的条形图中，三个条形代表每种类型`seasonal vegetable`的频率。这些频率也可以用一个简单的条形图来表示。但是我们发现从`stacked bar` 图中产生的额外信息是代表不同颜色的每个 v `egetable for each season`的频率计数。

`It is worth mentioning that stacked bar chat is applicable only for categorical variables.`

## 分组条形图

“分组条形图”这个名称表明——这是一种分成不同组的特殊类型的条形图。主要用于比较两个分类变量。

看上面的图表。`‘Vegetable’`和`‘Season’` 是两个分类变量。`‘x-axis’` 表示不同蔬菜的名称，`‘y-axis’`表示频率。如果你仔细看图表，你会发现不同蔬菜结合季节的频率是存在的。第一个组条显示`‘Bean.’`在组中出现的频率，有两个不同颜色的条。这两个条代表**【豆】**在`**winter**` 和`**summer**`两个季节出现的频率。其他成组的条形也表示相同的意思。

## **堆积面积图**

`***The Stacked Area Chart***`以多个区域系列叠加的形式绘制。每个系列的高度由每个数据点中的值决定[1]。

通过这个堆积面积图，我们可以比较数据点的两个或多个数据变量的值。它还为我们提供了两个或多个变量的值如何彼此不同的信息。

`*[The synthetic dataset I have created, mentioned at the beginning, is not fit for the graph. So, I have created another dataframe.]*`

*绘制堆积面积图。*

*图表显示了每个班级的“A 区”和“b 区”面积*

## **排列图**

帕累托图包含条形图和折线图，其中各个值由条形图以降序表示，而折线图表示累计总数。

在带条形图的帕累托图中，蔬菜的频率按降序分布。双轴被用来绘制图表。左边的`y-axis` 代表频率，右边的`y-axis`代表 [*累计频率*](https://www.cuemath.com/data/cumulative-frequency/) 。是发现数据趋势的最好方法。

## **圆环图**

甜甜圈图是一个简单的圆形中心切割图。虽然它代表了与饼图相同的含义，但它也有一些优点。在饼图中，我们经常会混淆每个类别所占的面积。由于饼状图的中心被从环形图中移除，这强调了读者要关心饼状图的外部弧[2]。内圈也可以用来显示附加信息。

用`matplotlib` 实现环形图如下，用于表示`**‘Season’**` 变量的频率。

## 热图

热图是一个矩形图，分为用不同颜色表示不同值/强度的子矩形。

这是相关性热图，其中`x-axis`和`y-axis`表示不同的变量，每个矩形代表两个变量的相关性。热图不仅可以表示相关值，还可以表示与变量相关的其他值。

## 雷达图

**雷达图**是一种以二维图表的形式显示多元数据的图形方法，该二维图表是从同一点[3]开始在轴上表示的三个或更多量化变量。来自中心的辐条，称为半径，代表变量[4]。半径之间的角度不包含任何信息[3]。

我已经使用了`**plotly**` 库来绘制雷达图。有了这个图，我们可以清楚的发现两点`([1,5,2,2,3] and [4,3,2,5,1,2])`的不同。数据点`[1,5,2,2,3]`表示— `processing cost:1, mechanical properties: 5, chemical stability: 2, thermal stability: 2, device integration: 3,` ，点【4，3，2，5，1，2】也是如此。

## 树形地图

树形图用于以嵌套的矩形形式显示`**hierarchical data**`[6]。

我已经使用了`**plotly**` 库来生成树形图。**库的内置数据集用于绘制**。看一下数据集就知道了。

`*[N.B. — I have used the* [*gapminder*](https://www.kaggle.com/datasets/tklimonova/gapminder-datacamp-2007) *dataset which is a public domain dataset under “*[*CC0: Public Domain*](https://creativecommons.org/publicdomain/zero/1.0/)*” license.*]`

*数据集的树形图。*

仔细看这张图表。它在分层嵌套的矩形中显示信息。对于树形图，`**world**` 位于层次结构的顶端。在 `**world, continents**`下被分配。在每一个`**continent**`，都有一些`**countries**`。群体的数量决定了矩形的大小。颜色因`**life expectations**`的刻度而异。

## 结论

作为一个人，我们希望获得精确的、有美感的信息。在策划之前，我们应该小心情节的性质。你对图表也有一个清晰的概念。

本文是数据可视化系列的延续。我也发表过关于 [***基础***](/basic-guide-to-data-visualization-for-data-science-8e7d966bf10a) 和 [***高级数据可视化***](/11-less-used-but-important-plots-for-data-science-dede3f9b7ebd) 的文章。

`***Keep an eye out for my next article.***`

`*[I like to write a series of articles on different data science topics. Please, keep your eyes on the next article.]*`

> *最后，如果你觉得这篇文章有帮助，别忘了* `[***follow***](https://medium.com/@mzh706)` *我。你也可以用我的* `[***referral link***](https://mzh706.medium.com/membership)` *加入 medium。通过电子邮件获取我所有的文章更新* `[***subscribe***](https://mzh706.medium.com/subscribe)` *。*

## 参考

1.  [堆积面积图| Chartopedia | AnyChart](https://www.anychart.com/chartopedia/chart-type/stacked-area-chart/#:~:text=Stacked%20Area%20Chart%20is%20plotted,vary%2C%20on%20the%20same%20graphic.)
2.  什么是圆环图？| TIBCO 软件
3.  [雷达图—维基百科](https://en.wikipedia.org/wiki/Radar_chart)
4.  [什么是雷达图？| TIBCO 软件](https://www.tibco.com/reference-center/what-is-a-radar-chart)
5.  [树形图—维基百科](https://en.wikipedia.org/wiki/Treemapping)

**此处提供了完整的数据可视化指南..**

[](https://medium.datadriveninvestor.com/ultimate-guide-to-data-visualization-for-data-science-90b0b13e72ab)  

我为你挑选了一些有趣的文章。别忘了朗读。

[](/basic-guide-to-data-visualization-for-data-science-8e7d966bf10a)  [](/11-less-used-but-important-plots-for-data-science-dede3f9b7ebd)  [](/spread-of-covid-19-with-interactive-data-visualization-521ac471226e)  [](/ultimate-guide-to-statistics-for-data-science-a3d8f1fd69a7) 