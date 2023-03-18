# 你为什么没有通过 SQL 面试？

> 原文：<https://towardsdatascience.com/why-are-you-failing-your-sql-interviews-3afe6c32ac2d>

![](img/c726b972997d79c32f39d6d2151cff78.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上 [NeONBRAND](https://unsplash.com/@neonbrand?utm_source=medium&utm_medium=referral) 拍摄的照片

## 避免这 5 个常见错误

随着数据素养越来越重要，我们将会看到越来越多的数据科学以外的职位的 SQL 面试。不幸的是，**这些面试可能会令人生畏、棘手、艰难**——尤其是如果你是第一次学习 SQL 的话。

根据我做数百次 SQL 访谈的经验，本文涵盖了我见过的 5 个最常见的错误。虽然有些似乎显而易见，但我已经采访了足够多来自不同背景的候选人，我认为每个人都值得单独讨论。我相信任何一个有扎实的 SQL 基础并且能够避免这 5 个错误的候选人都可以通过任何 SQL 面试。

# 什么是“SQL 面试”？

虽然“SQL 面试”可能有不同的定义，但一般来说，这是一个持续 30 分钟到 1 小时的**面试，主要目的是编写 SQL 代码，从数据库中查询数据**。像*“上个月优步司机的平均星级为 4.70+的百分比是多少”*或*“芝加哥哪 10 家商店的 Yelp 五星评级最高？”*常见。

虽然我见过许多人轻松通过这些面试，但我也采访过许多在面试中苦苦挣扎的求职者。虽然小错误看起来微不足道，但它们可能最终成为候选人是否继续前进的决定性因素。

# 错误 1:没有时间意识

这是目前为止候选人遇到的最常见的错误。你可以把 SQL 面试想象成一次考试，有两个很大的区别。

在考试中，你通常会把所有的问题都列在前面。这意味着你可以在考试开始时花几分钟来了解你可能如何分配你的时间，这可能会影响你如何回答某些问题或跳过问题以获得尽可能好的分数。

在 SQL 面试中，区别在于:

1.  只有面试官会知道问题，他们会一一揭晓
2.  没有蹦蹦跳跳

这使得时间变得尤为重要——因为你真的不知道会有多少问题，定步调可能很困难。我面试过许多看起来有望通过面试的候选人，但最终失败了，因为他们无法通过整个面试。

虽然这个错误可能是最难解决的，但是一个可以帮助你节省宝贵时间的通用规则是认识到**越简单的查询越好。**更复杂的答案不仅需要更长的打字时间，而且面试官解释和评估的时间也更长。例如，如果我们想回答*“芝加哥哪 10 家商店拥有最多的 Yelp 五星评级？”有两种方法可以做到这一点，两种方法都会得到相同的分数，尽管第二个答案要花 3 倍的时间来写。*

**更简单+更好:**

```
SELECT 
store_id,
SUM(CASE WHEN rating = 5 then 1 else 0 end) n_5star 
FROM ratings_db
WHERE market = 'Chicago'
GROUP BY 1
ORDER BY 2 DESC
LIMIT 10
```

**更复杂，可能会让面试官感到困惑:**

```
WITH temp_table as (
SELECT 
rating_id,
store_id,
rating
FROM ratings_db
WHERE market = 'Chicago'
AND rating = 5
)temp2 as (
SELECT 
store_id,
count(*) n_5star,
DENSE_RANK() OVER(ORDER BY n_5star DESC) rk
FROM temp_table
)SELECT
store_id
FROM temp2
WHERE rk <= 10
```

# 错误 2:没有事先澄清问题

![](img/f77056359f81b11569e4d0a72dec18d5.png)

照片由 [Neora Aylon](https://unsplash.com/@loveneora?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

如果你对被问到的问题不是 100%清楚，你需要做的第一件事就是要求面试官澄清。面试官希望你成功，应该随时愿意帮助进一步澄清问题。

为了了解为什么这个错误会代价高昂——假设你不太确定问题是什么，但你还是开始回答了。你可能会在面试官理解之前花几分钟回答错误的问题，这意味着之后你可能不得不从头开始。

# 错误 3:不知道如何处理时间戳

![](img/8c8da4f71233185b234943e249c36069.png)

卢卡斯·布拉塞克在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄的照片

不同的公司使用不同的指标。但是，**所有公司都用时间戳**。因此，了解如何处理时间戳对你面试的任何公司都是相关的。

参加面试时，了解与时间戳相关的基础知识总是一个好主意:

*   **如何获取当前时间戳:**这对于过滤类似于*“最近 30 天”*或*“昨天”*的内容很有用。`current_date`或`current_timestamp`应该就是你所需要的。
*   **加减时间戳:** `timestampdiff`是确定两个时间戳之间经过时间最清晰的方法。例如，`timestampdiff(seconds, ts1, ts2).`使用一个简单的减号可能有点含糊*(结果是秒吗？分钟？毫秒？)*
*   **比较时间戳:**类似于数字，可以用`<`和`>`来比较时间戳。如果时间戳更大，那么它就更近(即 2021 < 2022)
*   **提取组件:** `YYYY-MM-DD HH:MM:SS` `2022-01-01 23:59:59`是标准的时间戳格式。要知道，你也可以提取或截断时间戳来关注相关的时间单位(例如，对于一个小时，你可以做`DATE_TRUNC(hour, ts)`或使用`HOUR(ts)`)

# 错误 4:使用错误的连接

几乎在每一次 SQL 访问中，都应该有一个问题，你需要连接两个或更多的数据集。几乎总是，你在决定使用`LEFT JOIN`还是`INNER JOIN`。

有时，连接会给你相同的答案——然而也有一些时候**面试官在寻找一种特定类型的连接来回答问题**。如果我们再把 SQL 面试看作一次考试，使用错误的连接意味着你会被扣分——我已经在几次面试中看到这是决定性的因素。

在进行联接之前，您应该考虑内部联接是否会产生与左联接不同的结果。如果结果不同，那么确保你使用的是正确的。如果你不确定，你可以随时向面试官询问更多的上下文*(例如，内连接会丢失数据，而左连接会保留更多的数据，我们对哪一个感兴趣？)*

# 错误 5:语法和格式错误

最后，如果你是一个边缘候选人，你要避免可能让你脆弱的小错误。以下是一些我见过的求职者纠结的语法问题:

*   **运算符之间的顺序错误:**而不是`between 0 and 1`，他们可能会键入`between 1 and 0`。
*   **不缩进。**我见过一些应聘者在一行字里输入他们的全部答案。这使得作为一名面试官很难评估代码，我将不得不花更多的时间来检查他们的代码。
*   **不用化名。**特别是对于连接表的问题，如果你不为表取别名，你可能要花很多时间一遍又一遍地写表名:

**不推荐:**

```
SELECT
rides_database.ride_id,
driver_database.first_name,
driver_database.email
FROM rides_database 
JOIN driver_database
ON rides_database.driver_id = driver_database.driver_id
```

**改为:**

```
SELECT
r.ride_id,
d.first_name,
d.email
FROM rides_database r
JOIN driver_database d
ON r.driver_id = d.driver_id
```

# 总结想法

如果这篇文章能让你学到什么，那就是避免每一个错误都会增加你通过考试的机会。面试令人伤脑筋，我发现做好准备有助于缓解这种情况。希望我上面列出的五个错误是你在接下来的面试中能记住的！