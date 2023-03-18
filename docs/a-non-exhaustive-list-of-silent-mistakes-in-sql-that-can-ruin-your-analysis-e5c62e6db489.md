# SQL 中可能会破坏您的分析的“无声”错误的非详尽列表

> 原文：<https://towardsdatascience.com/a-non-exhaustive-list-of-silent-mistakes-in-sql-that-can-ruin-your-analysis-e5c62e6db489>

## 从犯了单子上大多数错误的人那里

我的大部分数据项目通常以同样的方式开始:一堆 SQL 查询。经过几年的经验，我认识到——有时是艰难的——不要搞砸这一步是至关重要的，因为即使是一个小错误也会很快使你得到的结果无效。

![](img/0fe5515d8e5f01be884b1c466df0e9e0.png)

瓦尔瓦拉·格拉博瓦在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

犯这些小错误的方法有很多。我不是在说忘记“分组”或用零除某些东西——这些很容易发现，因为您的查询工具会返回错误。我说的是**那些无声的错误**，那些根本不会返回任何结果的错误，你根本没有任何警告。**我说的是这样的情况，你最终相信你得到了你想要的结果，但你实际上没有——现在你所做的查询不允许你正确地回答你为自己设置的问题。**

下面是这些小错误的不完全列表，根据我自己的经验按频率排序。为了说明它们，我将使用[我通常的例子](/what-to-do-when-your-experiment-returns-a-non-statistically-significant-result-81ecaf56fb32)——想象一下:

*   你在一家游戏公司工作，该公司发布了多个免费游戏(有可能删除付费用户的广告)。
*   您的角色是了解用户增长，为此，您可以访问以下 3 个数据库:

```
**user_status_per_country**
user_id 
country
is_paid_user**daily_country_revenue**
date
country
monetization_source
revenue**daily_country_users**
date
country
game_id
users
```

让我们开始吧！

# #1:假设数据库/字段的内容基于其名称或描述

让我们从一个例子开始:你如何使用**user _ status _ per _ country**来吸引所有的免费用户？

应该是 is _ paid _ user = FALSE 吧？

好吧，假设这个字段是主用户数据库和历史数据库之间的左连接的结果，历史数据库包含在任何时间点成为付费用户的所有用户及其当前状态，因此，它以这样一种方式实现，它实际上可以接受 3 个不同的值:TRUE(用户是付费用户)、FALSE(用户曾经是付费用户，现在不是了)和 NULL(用户不存在于正确的数据库中，即它以前从未是付费用户)。

那么 is _ paid _ user = FALSE 将不会给出您所需要的结果，因为这将返回过去至少支付过一次的所有用户。

一般来说，由于您的大部分工作和结果都取决于您使用的数据源，因此不要依赖于关于字段/数据库内容的假设(尤其是基于命名的假设)是很重要的，并且要有办法确保您的假设是正确的。

# #2:由于重复而导致的超额计数

这是一个普通的。当您的表中要连接的字段上的每个值有多行时，您可能会遇到这种情况。

例如，使用我在介绍中提到的情况，如果您尝试计算每个国家的每个用户的每日收入，您不能直接加入每日 _ 国家 _ 收入和每日 _ 国家 _ 用户，因为您在每日 _ 国家 _ 收入中有一个国家的每日多个条目(由于不同的货币化来源)，并且在每日 _ 国家 _ 用户中有一个国家的每日多个条目(由于不同的游戏 id)。

具体来说，如果你这样写:

```
SELECT
  date,
  country,
  SUM(revenue) AS revenue,
  SUM(users) AS users
FROM daily_country_users
JOIN daily_country_revenue
USING(date_id,country)
GROUP BY 
  date,
  country
```

你将最终过度计算你的用户和收入。

# #3:搞乱环境

## 逻辑错误

你知道一个程序员的笑话吗？他的搭档告诉他们“去商店，买一加仑牛奶，如果他们有鸡蛋，就买 6 加仑”，然后他们带着 6 加仑牛奶回来了。

这里的概念是一样的——你需要确保你给出的条件是非常明确的，这样你才能得到你想要的。

例如，如果您想获得法国或西班牙的付费用户，您应该编写如下查询

```
is_paid_user AND (country_code = "FR" OR country_code = "ES")
```

而不是

```
is_paid_user AND country_code = "FR" OR country_code = "ES"
```

因为这最后一个语句会给你所有法国付费用户和 ES 的所有用户。(旁注:IN 语句也可以用在这里——但这对我阐述我的观点没有帮助)

(无耻的自我推销:在这一点上，有些人可能会说，我们可以使用 UNION ALL 来提高性能，而不是使用“OR”。我们将在后续文章中讨论这个问题，敬请关注！).

## “疏忽”错误

这一条属于“关注细节”的范畴——确保你使用正确的条件是很重要的。尤其是在处理字符串的时候，比如字符串是大写的。使用前面的示例，下面的语句:

```
is_paid_user AND (country_code = "fr" OR country_code = "ES")
```

…只会返回西班牙语付费用户。

# #4:误解一些函数/运算符的工作方式

一个小测验:如果给定以下参数，你知道通常的聚合函数会做什么吗？

```
1\. AVG(1, 2, 3, 4, NULL, NULL)
2\. AVG(1, 2, 3, 4, 0, 0)
3\. COUNT("A", "B", "B", NULL, NULL)
4\. COUNT("A", "B", "B", "", "")
5\. STRING_AGG("A","B",NULL,"C",NULL)
```

一般来说，了解您使用的不同函数如何处理空值，以及这会如何影响您的结果(在某些情况下，您可能希望这些空值不被计算在内，而在其他情况下，您可能希望它们被视为 0)是很重要的。

同样的事情也适用于操作人员——充分了解他们的行为可以避免糟糕的意外。例如，BETWEEN 是包含性的，但是当您开始比较不同的日期格式时，这可能会有点误导(更多信息请参见[本文](https://sqlblog.org/2009/10/16/bad-habits-to-kick-mis-handling-date-range-queries))，所以确保您很好地理解条件应该是什么以及您将使用的操作符将如何表现是很重要的。

# #5:在第二个数据库上使用非空条件进行左/右连接

基本上，如果您执行左/右连接并在第二个表中的字段上添加一个非空条件，您就有点违背了使用左/右连接的目的，因为您将根据表 2 的条件筛选表 1(而最有可能的情况是，如果您使用左/右连接，您希望表 1 中的所有记录只应用于表 2)。

好吧，这可能很难理解—[stack overflow 上的这个问题](https://stackoverflow.com/questions/9160991/left-join-with-condition)很好地说明了我在这里提到的内容，以及可以使用的解决方案。

这是我在 SQL 中发现的前 5 个最好的无声错误——这些错误可能不会马上看到，但会严重影响您的工作。希望这个故事能起到警示作用，对一些人有所帮助！

还有第六个问题，我遇到过很多次，我犹豫着要不要加到这个列表中:窗口函数的错误实现。最后，我认为窗口函数应该有它们自己的独立文章，所以我将在后续的故事中回到这一篇。

希望你喜欢阅读这篇文章！**你有什么建议想要分享吗？在评论区让大家知道！**

如果你想更多地了解我，这里有一些你可能会喜欢的文章。

[](/7-tips-to-avoid-public-embarrassment-as-a-data-analyst-caec8f701e42) [## 让您的数据分析更加稳健的 7 个技巧

### 增强对结果的信心，建立更强大的个人品牌

towardsdatascience.com](/7-tips-to-avoid-public-embarrassment-as-a-data-analyst-caec8f701e42) [](/how-to-build-a-successful-dashboard-359c8cb0f610) [## 如何构建成功的仪表板

### 一份清单，来自某个制造了几个不成功产品的人

towardsdatascience.com](/how-to-build-a-successful-dashboard-359c8cb0f610) [](https://medium.com/@jolecoco/how-to-choose-which-data-projects-to-work-on-c6b8310ac04e) [## 如何…选择要处理的数据项目

### 如果你有合理利用时间的方法，你可以优化你创造的价值。

medium.com](https://medium.com/@jolecoco/how-to-choose-which-data-projects-to-work-on-c6b8310ac04e)