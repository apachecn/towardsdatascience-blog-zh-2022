# 6 种鲜为人知的 SQL 技术每月为您节省 100 个小时

> 原文：<https://towardsdatascience.com/6-lesser-known-sql-techniques-to-save-you-100-hours-a-month-10ceed47d3fe>

## 使用这些简单的技术使您的分析和数据提取更容易

在八年的数据职业生涯中，我依靠一些简单但鲜为人知的 SQL 技术为自己节省了无数的时间来执行分析和构建 ETL 管道。

在本文中，我将分享六个反复出现的问题:

## 从表中查找和删除重复记录

```
with x as (select *, row_number() over(partition by [key],[key],[key] order by [key]) as rowRank from {schema}.{table})
select * from x where rowRank > 1;
```

没有比复制品更糟糕的了。可怕的重复记录给我的数据生命周期带来了巨大的痛苦。重复会弄乱任何分析或仪表板——尤其是那些不会因为简单的 DISTINCT 子句而消失的分析或仪表板。有多种方法可以识别重复项，但我发现上面的例子是最简单的。

只需将主查询包装在一个 CTE 中，在您希望检查的所有变量之后，添加一个在所有表键上分区的 *row_number* 函数。该分区必须包含所有表键才能正常工作，否则您可能会将非重复项错误分类。 *row_number* 函数在这里做的是对您提供的所有键实例进行排序。在 CTE 之后，运行一个简单的选择和过滤，其中新的*row _ number*function*字段大于 1。输出将返回所有重复的记录——因为任何具有 *rowRank > 1* 的记录在表中都有重复的键。此外，您可以通过运行以下命令来查看有多少重复记录:*

```
*with x as (select *, row_number() over(partition by [key],[key],[key] order by [key]) as rowRank from {schema}.{table})
select [keys], max(rowRank) - 1 num_duplicates from x group by [keys];*
```

*最后，如果您想删除所有的重复项，您实际上可以在 CTE 中使用 delete 语句！*

```
*with x as (select *, row_number() over(partition by [key],[key],[key] order by [key]) as rowRank from {schema}.{table})
delete * from x where rowRank > 1;*
```

*注意:delete 将**永久地**从表中删除记录——所以应该非常小心地使用它。测试这种方法的一种方法是创建一个有问题的表的临时副本，并在第一个副本上运行删除操作。然后在主表上执行删除之前做一些质量保证。*

*![](img/d8e5ddfb3033f3f571da99e7c601ff6a.png)*

*凯文·Ku 从[派克斯](https://www.pexels.com/photo/data-codes-through-eyeglasses-577585/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)拍摄的照片*

## *从表中查询最近的一组记录*

```
*select a.*, a.[date] from {schema}.{table} a 
join (select max(date) maxDate from schema.table) b 
on a.date = b.maxDate*
```

*大多数数据专业人员都处理大量时间序列数据。然而，时间序列不仅仅是带有日期戳的值，它还可以是数据集的带有日期戳的版本。例如，在我目前的工作中，我们会定期“快照”数据集当天版本的副本，以便我们可以跟踪它如何随着时间的推移而变化。从表中获取最新的记录集(即最新的“版本”)变得很重要。上面的查询通过在最大日期字段上将有问题的表连接到自身来实现这一点。内部联接筛选出日期不等于最大日期的所有记录。或者，您可以使用左连接，然后使用 where 子句进行筛选:*

```
*select a.*, a.[date], b.maxDate from {schema}.{table} a 
left join (select max(date) maxDate from schema.table) b 
on a.date = b.maxDate
where date = maxDate*
```

## *每月或每周开始/周末汇总每日数据*

***每月***

```
*select [key], sum([field]),  DATEADD(month, DATEDIFF(month, 0, [date field]), 0) as month from {schema}.{table} group by [key]*
```

***周初***

```
*select [key], sum([field]),  DATEADD(wk, DATEDIFF(wk, 6, [date]), 6) as weekBeginning from {schema}.{table} group by [key]*
```

*以上两种技术将允许您快速、轻松地在月初或周初对每日级别的时间序列数据进行分组。SQL 中还有其他日期函数可以做到这一点，但我发现这些是最简单的。这种技术还有助于在仪表板工具或 excel 中更容易地显示时间序列。例如，我经常喜欢使用以“YYYY-MM”的形式显示时间序列中连续几个月的视觉效果，以这种方式设置查询将使这一工作变得更加容易。*

## *聚集自定义(案例时)类别的数据*

```
*select [key], sum([field]), 
CASE WHEN date between '2022-09-01' and '2022-12-31' then 'Fall'
WHEN date between '2022-01-01' and '2022-03-31' then 'Winter'
WHEN date between '2022-04-01' and '2022-06-30' then 'Spring'
WHEN date between '2022-07-01' and '2022-08-31' then 'Summer' end as Seasons from {schema}.{table} group by 
CASE WHEN date between '2022-09-01' and '2022-12-31' then 'Fall'
WHEN date between '2022-01-01' and '2022-03-31' then 'Winter'
WHEN date between '2022-04-01' and '2022-06-30' then 'Spring'
WHEN date between '2022-07-01' and '2022-08-31' then 'Summer' end*
```

*使用这种技术，您可以使用 CASE 语句和 GROUP BY 子句聚合自定义类别的数据。这可以在上面的一个语句中完成，或者如果你想避免使用长组，你可以使用 CTE。(注意:在 GROUP BY case 语句中，以“end”结尾，而不是像 SELECT 语句中那样以“end as”结尾)。*

```
*WITH X as (select [key], [field]), 
CASE WHEN date between '2022-09-01' and '2022-12-31' then 'Fall'
WHEN date between '2022-01-01' and '2022-03-31' then 'Winter'
WHEN date between '2022-04-01' and '2022-06-30' then 'Spring'
WHEN date between '2022-07-01' and '2022-08-31' then 'Summer' end as Seasons from {schema}.{table})
select [key], sum([field]), Seasons from X group by Seasons* 
```

*在这个例子中，我使用日期参数创建了一个“季节”字段，但是您可以做任何事情。*

## *在同一个表中找出今天和昨天(或任意两个日期)的区别*

```
*-- MS SQL SERVER 2016 or laterwith x as (
select *, row_number() over(partition by [keys] order by [date_field] desc) as dateOrder
from {schema}.{table}
where [date_field] >= dateadd(day,-2,getdate()))
,
x1 as (
select * from x where dateOrder = 1),
x2 as (select * from x where dateOrder = 2)
select [fields] from x1 
left join x2 on x1.key = x2.key (and x1.key = x2.key and x1.key = x2.key)
where x2.[key] is null -- POSTGRES SQL with x as (
select *, row_number() over(partition by [keys] order by [date_field] desc) as dateOrder
from {schema}.{table}
where [date_field] >= CURRENT_TIMESTAMP - interval '2 day'
,
x1 as (
select * from x where dateOrder = 1),
x2 as (select * from x where dateOrder = 2)
select [fields] from x1 
left join x2 on x1.key = x2.key (and x1.key = x2.key and x1.key = x2.key)
where x2.[key] is null*
```

*这一个看起来非常合适，但是它是一个经常出现的用例。它的一些用途:*

*   *监视每天有多少新记录被添加到表中。*
*   *识别在“快照”表中的两个日期之间添加的新记录(即，正如我上面描述的，这些表具有相同数据集/数据源的时间戳副本)。*

## *将一个表中的数据合并到另一个表中(最简单的方法)*

```
*delete from {schema}.{target_table} where exists (select 1 from {schema}.{source_table} where {schema}.{source_table}.[key] = {schema}.{target_table}.[key])*
```

*有许多方法可以将数据从一个表合并到另一个表。MS SQL 实际上有一个 MERGE 语句来做这样的事情。然而，我发现以上是在脚本化 ETL 管道中设置数据合并的最简单的方法。*

*我编写了大量代码来自动从 API 获取数据，然后例行公事地将新数据转储到数据库表中。通常，我的做法是让 Python 脚本拉回特定时间范围内的新数据(2 天-1 周或更长时间，具体取决于数据源)，然后将所有数据推送到临时表中。一旦新数据出现在临时表中，我就运行上面的 delete 语句，该语句将扫描生产表，查找新表中已经存在的记录。最后，一旦删除了目标表中所有已经存在的记录，我就运行一个简单的从临时表到目标表的插入操作。*

*我希望这些技术中至少有一种对您来说是新的，并且有助于简化您的查询和分析。如果你喜欢这篇文章，你可以在这里查看我的其他作品。如果您对从 REST APIs 获取数据感兴趣，那么[这篇文章](/how-to-pull-data-from-an-api-using-python-requests-edcc8d6441b1)也会很有帮助。*