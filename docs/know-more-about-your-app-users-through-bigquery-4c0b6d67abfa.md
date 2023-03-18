# 通过 BigQuery 了解更多关于您的应用用户的信息

> 原文：<https://towardsdatascience.com/know-more-about-your-app-users-through-bigquery-4c0b6d67abfa>

## 超越 Firebase 和 Google Analytics 的更加定制化的事件分析方法

![](img/12cc58c0ae88c2da4e29a7b96b5b4651.png)

罗宾·沃拉尔在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

如果你正在运营一个 Android 应用程序业务，或者是一个应用程序业务的分析师，你可能会经常使用谷歌分析和 Firebase。这些工具可帮助您监控应用性能，并让您深入了解应用用户。

然而，GA 和 Firebase 并不能回答所有的问题，即使回答了，它们也不能完美地捕捉到您想要的分析和呈现数据的方式。在这种情况下，将历史数据导出到数据仓库可能是值得的，然后您应该能够自己进行更高级的分析。由于 BigQuery 与 GA 和 Firebase 属于同一个谷歌生态系统，因此导出更加无缝。

本文讨论通过在 BigQuery 中运行简单查询来回答的示例业务问题。先决条件是一些基本的 SQL 技能，一个[谷歌云账户](https://cloud.google.com/)，和一个 [Firebase](https://firebase.google.com/) 连接的应用；然后，您就可以自己进行这些查询了！

第一步是将 Firebase 数据导出到 BigQuery 中。一旦您完成了这一步，事件数据就会自动流向您的数据仓库，为数据争论和分析做好准备。要了解更多关于如何启用 Firebase 到 BigQuery 导出的信息，您可以参考这个[文档](https://firebase.google.com/docs/projects/bigquery-export)。

# 数据集

对于这个演示，我将使用来自 Flood-it 的示例游戏应用程序数据！，一款在 Android 和 iOS 平台上都有售的益智游戏。它是一个模糊的数据集，模拟 GA 和 Firebase 中的实际实现。

您可以检查[这个链接](https://developers.google.com/analytics/bigquery/app-gaming-demo-dataset)以获得关于数据集及其模式的更多信息。

关于用户标识符，请注意:“user_pseudo_id”用于标识演示数据中的单个用户。如果您想要跨多个应用程序、设备和分析提供商关联用户数据，您应该设置一个“ [user_id](https://firebase.google.com/docs/analytics/userid) ”。

现在我们准备分析数据。

# Firebase 和 BigQuery 可以回答的用户体验相关问题示例

## 有多少用户安装了该应用程序？

类似这样的查询可以揭示在特定时间段内有多少用户安装了该应用程序:

```
SELECT COUNT(DISTINCT user_pseudo_id) as users_installed
FROM `firebase-public-project.analytics_153293282.events_*`
WHERE event_name = "first_open"
AND _TABLE_SUFFIX BETWEEN '20180927' and '20181003'
```

由于表是当天的格式(每天一个表)，所以可以使用通配符“*”在一个查询中组合几个表。使用这段代码时要非常小心，因为执行起来可能会非常昂贵，尤其是当你有很多数据的时候。切记始终使用“_TABLE_SUFFIX”来定义日期边界和限制数据。

通过运行这段代码，我们看到有 **321** 个用户在指定的时间内安装了应用程序。

要获得用户安装的每日明细，您还可以运行以下代码:

```
SELECT
  FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%Y%m%d', event_date)) AS date,
  COUNT(DISTINCT user_pseudo_id) as users_installed
FROM `firebase-public-project.analytics_153293282.events_*`
WHERE event_name = "first_open"
  AND _TABLE_SUFFIX BETWEEN '20180927' and '20181003'
GROUP BY date
ORDER BY date
```

该查询产生以下结果:

## 用户的人口统计特征是什么？

Firebase 中提供了一些人口统计数据，例如位置和设备信息。

例如，该查询给出了我们的用户所在的前 10 个国家，以及相应的百分比:

```
WITH
--Compute for the numerators
country_counts AS (
SELECT
  geo.country,
  COUNT(DISTINCT user_pseudo_id) AS users
FROM `firebase-public-project.analytics_153293282.events_*`
WHERE event_name = "first_open"
  AND _TABLE_SUFFIX BETWEEN '20180927' and '20181003'
  AND geo.country <> ""
GROUP BY geo.country
),
--Compute for the denominators
user_counts AS (
SELECT
  COUNT(DISTINCT user_pseudo_id)
FROM `firebase-public-project.analytics_153293282.events_*`
WHERE event_name = "first_open"
  AND _TABLE_SUFFIX BETWEEN '20180927' and '20181003'
),
--Compute for the percentages
percent AS (
SELECT
  country,
ROUND(users / (SELECT * FROM user_counts), 4) AS percent_users
FROM country_counts
)SELECT * FROM percent
ORDER BY percent_users DESC
LIMIT 10
```

我们通过运行代码得到以下结果:

由此，我们知道我们的用户群中有五分之二来自**美国**，而**印度**也有 15%的良好表现。

通过运行以下查询，我们还可以了解他们玩游戏时使用的设备类型(手机还是平板电脑):

```
WITH
device_counts AS (
SELECT
  device.category,
  COUNT(DISTINCT user_pseudo_id) AS users
FROM `firebase-public-project.analytics_153293282.events_*`
WHERE event_name = "first_open"
  AND _TABLE_SUFFIX BETWEEN '20180927' and '20181003'
  AND device.category <> ""
GROUP BY device.category
),user_counts AS (
SELECT
  COUNT(DISTINCT user_pseudo_id)
FROM `firebase-public-project.analytics_153293282.events_*`
WHERE event_name = "first_open"
  AND _TABLE_SUFFIX BETWEEN '20180927' and '20181003'
),percent AS (
SELECT
  category,
  ROUND(users / (SELECT * FROM user_counts), 4) AS percent_users
FROM device_counts
)SELECT * FROM percent
ORDER BY percent_users DESC
```

从这个例子中我们看到 **81%** 在用手机，而只有 **19%** 在用平板。

## 每天有多少用户在积极使用该应用？

每日活跃用户(DAU)指标让您了解当前应用程序用户的参与度。如何衡量取决于你如何定义活跃用户。在本例中，我们将活动用户定义为一天中执行了任何“user_engagement”操作的用户。

要计算指定时间段的 DAU，只需运行以下查询:

```
WITH
daily_user_count AS (
SELECT
  FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%Y%m%d', event_date)) AS date,
  COUNT(DISTINCT user_pseudo_id) AS active_users
FROM `firebase-public-project.analytics_153293282.events_*`
WHERE event_name = "user_engagement"
  AND _TABLE_SUFFIX BETWEEN '20180901' and '20180930'
GROUP BY date
)SELECT AVG(active_users) AS daily_active_users
FROM daily_user_count
```

平均而言，9 月份有 **496** 名用户使用该应用。

## 用户在该应用上的花费是多少？

我们还可以通过运行以下查询来了解用户在应用内购买上的花费:

```
SELECT SUM(user_ltv.revenue) AS revenue
FROM `firebase-public-project.analytics_153293282.events_*`
WHERE event_name = "in_app_purchase"
  AND geo.country = "United States"
  AND _TABLE_SUFFIX BETWEEN '20180901' and '20180930'
```

对美国用户来说，这只是 5.94 美元，并不多，但了解这一点后，我们可以制定更好的策略来追加销售应用内购买。

## 有多少用户遇到了应用崩溃？

为了防止流失和卸载，有必要监控不愉快的用户体验，如应用程序崩溃。

我们可以通过查询“app_exception”事件来深入了解这一点:

```
SELECT COUNT(DISTINCT user_pseudo_id) AS users
FROM `firebase-public-project.analytics_153293282.events_*`,
UNNEST(event_params) e
WHERE event_name = 'app_exception'
  AND _TABLE_SUFFIX BETWEEN '20180901' and '20180930'
  AND e.key = 'fatal' AND e.value.int_value = 1
```

注意 UNNEST()函数。事件参数在 BigQuery 中存储为数组，因此需要首先取消嵌套。查看[这篇文章](https://medium.com/firebase-developers/how-to-use-select-from-unnest-to-analyze-multiple-parameters-in-bigquery-for-analytics-5838f7a004c2)可以更深入地讨论 BigQuery 中的非嵌套事件参数。

根据这个查询，我们已经确定 **269** 个用户在九月份遇到过崩溃。

## 有多少用户在卸载应用？

App 删除是一种不良结果，因此应定期监控。这个在 Firebase 中对应的事件名是“app_remove”。

以下查询输出了在 9 月份安装该应用程序的人群中，一周后仍在使用该应用程序的用户的百分比:

```
WITH
--List of users who installed in Sept
sept_cohort AS (
SELECT DISTINCT user_pseudo_id,
FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%Y%m%d', event_date)) AS date_first_open,
FROM `firebase-public-project.analytics_153293282.events_*`
WHERE event_name = 'first_open'
AND _TABLE_SUFFIX BETWEEN '20180901' and '20180930'
),--Get the list of users who uninstalled
uninstallers AS (
SELECT DISTINCT user_pseudo_id,
FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%Y%m%d', event_date)) AS date_app_remove,
FROM `firebase-public-project.analytics_153293282.events_*`
WHERE event_name = 'app_remove'
AND _TABLE_SUFFIX BETWEEN '20180901' and '20181007'
),--Join the 2 tables and compute for # of days to uninstall
joined AS (
SELECT a.*,
b.date_app_remove,
DATE_DIFF(DATE(b.date_app_remove), DATE(a.date_first_open), DAY) AS days_to_uninstall
FROM sept_cohort a
LEFT JOIN uninstallers b
ON a.user_pseudo_id = b.user_pseudo_id
)--Compute for the percentage
SELECT
COUNT(DISTINCT
CASE WHEN days_to_uninstall > 7 OR days_to_uninstall IS NULL THEN user_pseudo_id END) /
COUNT(DISTINCT user_pseudo_id)
AS percent_users_7_days
FROM joined
```

这告诉我们，9 月份安装者的 7 天保留率在**为 76%** ，或者说**有 24%** 的 9 月份人群已经卸载了该应用。

这个数字有多好？这可以通过对行业基准进行一些研究来回答，或者通过监控趋势如何随着时间的推移而发展。

您也可以将保留率的时间范围更改为最有用的时间范围，方法是将最终的“CASE WHEN”条件编辑为您想要的天数，而不是 7 天。

## 崩溃是否可能影响用户体验，导致他们卸载？

我们可以进一步深入了解卸载该应用程序的用户，以确定删除它的可能原因。因为我们可以访问几个事件，所以我们可以查看它们并检查与卸载的相关性。

例如，我们可能希望在卸载之前检查有多少卸载程序遇到了崩溃。我们不能断定这导致了他们的沮丧并导致他们移除应用程序，但只看数字可能会有所帮助。可以运行这样的查询:

```
WITH
--List of users who installed in Sept
sept_cohort AS (
SELECT DISTINCT user_pseudo_id,
FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%Y%m%d', event_date)) AS date_first_open,
FROM `firebase-public-project.analytics_153293282.events_*`
WHERE event_name = 'first_open'
AND _TABLE_SUFFIX BETWEEN '20180901' and '20180930'
),--Get the list of users who uninstalled
uninstallers AS (
SELECT DISTINCT user_pseudo_id,
FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%Y%m%d', event_date)) AS date_app_remove,
FROM `firebase-public-project.analytics_153293282.events_*`
WHERE event_name = 'app_remove'
AND _TABLE_SUFFIX BETWEEN '20180901' and '20181007'
),--Get the list of users who experienced crashes
users_crashes AS (
SELECT DISTINCT user_pseudo_id,
FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%Y%m%d', event_date)) AS date_crash,
FROM `firebase-public-project.analytics_153293282.events_*`,
UNNEST(event_params) e
WHERE event_name = 'app_exception'
AND _TABLE_SUFFIX BETWEEN '20180901' and '20181007'
AND e.key = 'fatal' AND e.value.int_value = 1
),--Join the 3 tables
joined AS (
SELECT a.*,
b.date_app_remove,
DATE_DIFF(DATE(b.date_app_remove), DATE(a.date_first_open), DAY) AS days_to_uninstall,
c.date_crash
FROM sept_cohort a
LEFT JOIN uninstallers b
ON a.user_pseudo_id = b.user_pseudo_id
LEFT JOIN users_crashes c
ON a.user_pseudo_id = c.user_pseudo_id
)--Compute the percentage
SELECT
COUNT(DISTINCT
CASE WHEN days_to_uninstall <= 7 AND date_crash IS NOT NULL
THEN user_pseudo_id END)
/ COUNT(DISTINCT
CASE WHEN days_to_uninstall <= 7 THEN user_pseudo_id END)
AS percent_users_crashes
FROM joined
```

这表明只有 2.4%的卸载程序经历了崩溃。看起来数量很少，但与非安装者进行比较可能有助于得出更合理的结论。此外，我们可以创建一个应用程序删除率模型来确定应用程序删除的预测因素。我不会在本文中涉及这些内容，因为这超出了范围，但是我推荐它作为一个强大的下一步。

# 结论

在这篇文章中，我们发现了 BigQuery 可以为您的应用分析带来的巨大可能性。我们提到了一些简单的查询，但随着您对 Firebase 中可用数据的了解越来越多，我鼓励您更深入地挖掘，亲自看看这个强大的资源如何帮助您发展应用业务。

如果你喜欢这篇文章或者觉得它有用，请关注我的博客[获取更多关于营销分析和数据科学的资源。快乐学习！](https://medium.com/@noemiramiro)

参考资料:

Google Analytics 4 游戏应用实施的 BigQuery 样本数据集。*谷歌分析样本数据集。*网上有:[https://developers . Google . com/analytics/big query/app-gaming-demo-dataset](https://developers.google.com/analytics/bigquery/app-gaming-demo-dataset)。许可信息:[知识共享署名 4.0 许可](https://creativecommons.org/licenses/by/4.0/)