# 5 个窗口函数示例，让您的 SQL 技能更上一层楼

> 原文：<https://towardsdatascience.com/5-window-function-examples-to-take-your-sql-skills-to-the-next-level-2b3306650bb6>

## 数据科学

## 如果你想掌握 SQL 中的数据操作，你需要理解窗口函数

![](img/439e5ca16aa6debead467f92f87093b0.png)

照片由 [Pexels](https://www.pexels.com/photo/person-sitting-on-mountain-cliff-1659438/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) 的 [M Venter](https://www.pexels.com/@m-venter-792254?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) 拍摄

为了达到 SQL 的下一个专业水平，理解窗口函数是很重要的。窗口函数起初看起来令人生畏，但是一旦你理解了它们，你会发现它们有许多不同的应用。在我分享我的例子之前，我想为那些可能不熟悉的人解释一下它们是什么。我希望在这篇文章结束时，你将会探索在你自己的工作中使用窗口函数的方法。如果你已经熟悉了窗口函数，可以直接跳到第三个例子。

我分享的例子是我职业生涯中最常用的窗口函数，所以希望它们也适用于你的工作。

我将使用这个 SQLite 沙盒数据库作为我的代码示例:【https://www.sql-practice.com/ 

## **什么是窗口功能？**

窗口函数是对数据集的多行执行计算，同时保持原始表的相同行数和行数的函数。

## **1。你的第一个窗口函数**

在试图理解窗口函数时要记住的主要事情是，窗口函数维护表的原始行。对于我的第一个例子，我使用的查询与我在[上一篇文章](/10-quick-sql-tips-after-writing-daily-in-sql-for-3-years-37bdba0637d0)中写的关于在您的数据生涯中使用 SQL 技巧的查询相同。

这个例子是我想象的窗口函数的一个简单的用例。也就是说，如果你能掌握第一个例子，你就能把这个模式应用到很多不同的用例中。你掌握 SQL 中数据操作的能力一定会向前迈进一大步。

首先，我将解释上述窗口函数的输出是什么。输出将为您提供相同数量的行，并且您的最终列将是城市的最大权重。这意味着如果一个城市出现不止一次，那么权重值将会重复。这种重复经常发生在窗口函数中，因为我们在执行聚合时没有折叠表中的任何行。

下面是另一个查询，它给出了与上面完全相同的结果，只是时间更长，计算量更大。希望这能帮助你理解窗口函数的作用。

具体来说，一个典型的窗口子句将如下所示:

<window function="">结束(</window>

窗口函数可以是聚合窗口函数，也可以是内置窗口函数。下面是您可以使用的[聚合函数](https://www.sqlite.org/lang_aggfunc.html)的列表。你很可能已经熟悉这些。

下一部分是增加“T4”条款。 **如果它没有 over 子句，那么它就不是窗口函数。**

over 子句位于窗口定义之前。在这种情况下，我们使用 partition by。现在不要担心其他的窗口定义。最常见的两种是按划分的*和按*排序的*，这就是我在本文中要介绍的全部内容。您可能已经猜到了， *partition by* 指定了聚合的级别。在这种情况下，我们告诉查询我们想要城市的最大权重。*

就这样，这是你的第一个窗口函数🙌

## 2.**真正最简单的窗口功能**

好了，现在我们已经完成了你的第一个窗口函数，我想让你的大脑休息一下，让第二个例子变得非常简单。

当您不包括窗口定义时，您将对整个表进行聚合。这在特殊情况下会派上用场，所以我想我会快速展示一下，这样它就在你的脑海中，供你以后工作时使用。

下面是不带窗口函数的等效查询。

## 3.介绍“按窗口排序”定义

既然我们已经介绍了窗口函数的基本知识，我想介绍第二个窗口定义:ORDER BY。这正如它的名字所暗示的；它根据您选择的列对表格行中的数据进行排序。

在上面的例子中，这个窗口函数是按日期计算入院人数的累计。我不会为此包含一个等价的，因为我不认为不使用窗口函数就能做到这一点。这就是为什么我在文章的开头说窗口函数会让你在 SQL 中的数据操作技巧更上一层楼。

## 4.ROW_NUMBER 窗口函数

对于第四个例子，我想结合您到目前为止所学的一些内容，同时还添加了行号窗口功能。到目前为止，我只为第一个函数包含了聚合窗口函数，因为这是最简单的窗口函数子句类型，但是也有内置的窗口函数可用(确切地说有 11 个)。

本例中唯一的新概念是窗口函数开头的 row_number()函数。这将根据窗口定义分配一个连续的数字。在这种情况下，我们的定义是希望按城市分区，按出生日期降序排序。

如果你仍然不确定如何形象化这一点，这里是我在开始链接的 sql 实践网站的链接:[https://www.sql-practice.com/](https://www.sql-practice.com/)。在该网站中键入删除了 where 子句的 SQL 代码，以了解发生了什么。

该查询回答的问题是，“每个城市中第二年轻的患者是谁？”。您也许能够轻松地编写一个等价的查询来查找每个城市中最年轻的患者，但是第二个或第三个呢？窗口函数使得一些困难的问题变得简单得多。

## **5。滞后和超前窗口功能**

该窗口函数回答以下问题，“每个患者与下一个最高的人相比，身高有什么不同？”。

显然，这只是一个例子，寻找高度差异可能不是一个现实的问题，但有很多使用情况下，与数据集中的前一行进行比较是有用的-股票是一个常见的例子。

Lag 获取前一行的值并将其放在当前行上。默认的滞后是 1 行，但是您可以在 lag 函数中选择列之后立即指定该参数。

超前和滞后类似，只是作用方向相反。它获取下一行并将值放入当前行进行比较。

## 结论

这是窗口功能的概述。我希望它对您提高 SQL 技能有所帮助。如果你想要一个更高级的关于窗口功能或者我在这里没有提到的东西的教程，请在评论中告诉我。此外，如果你觉得有一个 SQL 主题你想了解更多，也请让我知道。

要阅读关于窗口函数的其他详细信息，请参考 SQLite 窗口函数文档。

[](https://medium.com/@andreasmartinson/membership) [## 加入 Medium 并通过我的推荐链接支持我

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

medium.com](https://medium.com/@andreasmartinson/membership) 

如果你喜欢这篇文章，请在 [LinkedIn](https://www.linkedin.com/in/aem193/) 上联系我，或者看看我的另一个故事:

[](/why-is-nobody-talking-about-sql-anti-joins-f970a5f6cb54) [## 如何在您的数据科学职业生涯中使用 SQL 反连接

### 为什么你应该知道他们

towardsdatascience.com](/why-is-nobody-talking-about-sql-anti-joins-f970a5f6cb54) [](/10-quick-sql-tips-after-writing-daily-in-sql-for-3-years-37bdba0637d0) [## 三年来每天用 SQL 写作的 10 个快速 SQL 技巧

### 这些技巧解决了您作为数据专业人员将会遇到的常见 SQL 问题

towardsdatascience.com](/10-quick-sql-tips-after-writing-daily-in-sql-for-3-years-37bdba0637d0) 

**参考文献**

1.  [sql-practice.com](https://www.sql-practice.com/)，SQL Practice.com
2.  [内置聚合函数](https://www.sqlite.org/lang_aggfunc.html)，SQLite 文档
3.  [窗口功能](https://www.sqlite.org/windowfunctions.html)，SQLite 文档