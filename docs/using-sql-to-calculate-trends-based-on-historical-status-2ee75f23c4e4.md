# 使用 SQL 根据历史状态计算趋势

> 原文：<https://towardsdatascience.com/using-sql-to-calculate-trends-based-on-historical-status-2ee75f23c4e4>

## 我们怎样才能捕捉到存在但没有活动时间戳的事物的趋势呢？

说到捕获时间戳，我处理过的大多数数据集都非常健壮。无论是销售、产品还是人才获取数据，API 和数据仓库工具都需要能够为应用程序上发生的特定操作或事务提供时间戳。

![](img/e46e7cb6eaaaa6a943b848c5e2645572.png)

大多数 web 应用程序会为其网站上发生的事件生成一个时间戳，以及事件内容的记录。图片由 Malvestida 通过 [Unsplash](https://unsplash.com/photos/FfbVFLAVscw) 提供。

然而，我一直纠结的一个分析是基于某个时间点的某件事的状态的时间序列趋势。例如，假设我正在与人才获取团队合作，我们正在尝试了解 2022 年我们的职业页面上有多少工作岗位，按月分列。一个标准的数据集可能包含帖子的创建日期、帖子的发布日期以及帖子被删除的日期(基于操作的时间戳)。然而，可能不会有内置的时间戳显示“1 月 15 日，发布是实时的。1 月 16 日、17 日和 18 日也有直播，还有……”

如果我想找到 2022 年 1 月 1 日至 2022 年 1 月 31 日期间在我的工作网站上发布的所有帖子(但可能在这些日期或之前发布)，我过去使用的一种解决方法是:

```
if posting_publish_date ≤ 2022–01–01 and (posting_closed_date is null or posting_closed_date > 2022–01–31) then True
```

这将帮助我创建一个标志，以过滤在兴趣期当天或之前发布、在兴趣期之后关闭或尚未关闭的帖子(因此，在 2022 年 1 月 1 日至 1 月 31 日之间有效)。但是，如果我想做一个时间序列分析，需要一个月的细分，而不是一个“时间点”的数字呢？

对于那些不熟悉人才获取数据的人来说，另一个例子是，如果我想按月跟踪销售周期中处于潜在客户创造阶段的机会数量。

我也许能够捕捉到他们何时进入潜在客户创造阶段以及何时退出，但如果我只是想知道每个月潜在客户创造的机会数量，而不管他们何时进入该阶段，该怎么办？基于传统数据集，这些数据集仅在有行动时(即 opp 进入或退出阶段)捕获时间戳，我只能显示进入或退出阶段的机会数量，而不能显示刚刚进入阶段的机会数量。

如果这些例子中的任何一个引起了你的共鸣，就你想尝试的或者过去已经尝试过的分析而言，那么这个教程就是为你准备的！

# 步骤 1:为日历日期生成一列

该列是您的线图的隐喻 x 轴。您希望创建一个列来存储一年中的每个日期，即使当天没有任何活动。我通常会生成比一年更长的日期，所以数据集在未来几年仍然有效。

一些 SQL 风格有一个 [generate_series](https://www.postgresql.org/docs/current/functions-srf.html) 函数，它将为您创建这个日历列。如果你像我一样不幸，并且你正在使用的 SQL 风格还不支持 generate_series(像亚马逊雅典娜🥺)，你可以使用来自 [Looker](https://help.looker.com/hc/en-us/articles/4420211251987-Generate-Date-Series-Create-a-calendar-view-of-future-dates-Community-) 的这段代码，它对我有效(我在亚马逊雅典娜上为 PostgreSQL 稍微修改了一下):

```
SELECT parse_datetime('2020–01–01 08:00:00', 'yyyy-MM-dd H:m:s') + (interval '1' day * d) as cal_date from 
FROM ( SELECT
ROW_NUMBER() OVER () -1 as d
FROM
(SELECT 0 as n UNION SELECT 1) p0,
(SELECT 0 as n UNION SELECT 1) p1,
(SELECT 0 as n UNION SELECT 1) p2,
(SELECT 0 as n UNION SELECT 1) p3,
(SELECT 0 as n UNION SELECT 1) p4,
(SELECT 0 as n UNION SELECT 1) p5,
(SELECT 0 as n UNION SELECT 1) p6,
(SELECT 0 as n UNION SELECT 1) p7,
(SELECT 0 as n UNION SELECT 1) p8,
(SELECT 0 as n UNION SELECT 1) p9,
(SELECT 0 as n UNION SELECT 1) p10
)
```

您可以将“2020–01–01”替换为您计划开始历史分析的日期。

# 步骤 2:在日历列和感兴趣的表之间进行左连接

现在您已经有了 calendar 列，您需要通过 left join 添加活动表。为什么离开加入？如果我们继续上面的销售周期示例，可能会有没有阶段移动的日期(即周末、节假日、淡季)，如果我们依赖于阶段移动日期列，我们将会丢失该日期的一行。但是，如果我们将日历表作为主表，并通过 left join 添加 stage activity 表，那么所有的日历日期都将出现(不管那天是否发生了任何移动)。

下一部分有点棘手——决定加入哪个时间戳。这将取决于您试图跟踪的指标。如果我试图寻找所列日期的潜在机会，我会使用这样的语句:

```
Select c.cal_date, count(distinct opp_id) as "historical_prospects" from calendar c
left join opportunities o
on o.stage_entered ≤ c.cal_date 
and (o.stage_exited is null or o.stage_exited > c.cal_date)
```

这将找到在感兴趣的日期当天或之前进入该阶段的所有人，并且这些人或者仍然在该阶段中，或者在感兴趣的日期之后退出该阶段。该脚本的最后一个数据框将为我提供一列日历日期，以及在该日期处于潜在客户发现阶段的相应机会数。

# 总结和进一步应用

![](img/9b752f9b1ae1af7e4ac889348fefc4dc.png)

在查找和分析历史数据时，肯定有更简单的方法！照片由 Aditya via [Unsplash](https://unsplash.com/photos/dvPd91Pdh5c) 提供

这样，我关于如何基于时间点状态找到历史趋势的教程就结束了。希望我的解释和伪代码足够清晰，能让大家理解我的主旨和逻辑！可以想象，除了我在本文中分享的两个例子之外，还有更多应用。

您可以对上面的示例进行更复杂的分析，例如过滤来自特定来源(即 2019 年 1 月期间处于潜在客户阶段并以 LinkedIn 作为来源标签的每个人)的管道阶段中的机会。

请随时留下评论和反馈，让我知道你是否有不同的方法来实现这种类型的历史时序分析！我总是喜欢在 StackOverflow 上看到有人用 50 行代码，用 5 行代码就能完成我做的事情😅