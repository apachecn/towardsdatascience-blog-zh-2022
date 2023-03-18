# 你没有使用但应该使用的 5 个雪花查询技巧

> 原文：<https://towardsdatascience.com/5-snowflake-query-tricks-you-arent-using-but-should-be-7f264b2a72d8>

## 更少的行、更低的成本和更快的执行时间

![](img/59f0c32735d64c1d39e48c2b263e9b04.png)

凯利·西克玛在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

雪花是一个云计算数据解决方案，允许用户在他们的云平台上直接存储数据和运行查询，可以通过网络浏览器直接访问。它通常用于廉价的数据存储和自动伸缩能力，集群可以自动启动和停止来管理查询工作负载。

经常被忽视的是，雪花不仅仅使在数据库上设置和运行查询变得更容易。它还具有独特的查询语法，这是 PostgreSQL 或 MySQL 等其他数据库系统所没有的。在下面的这篇文章中，我们将介绍我最喜欢的这些强大的子句，以及如何使用它们来改进语法和可读性，但最重要的是减少计算成本和执行时间。

# 1.具有资格；合格；取得资格(或学历)

[qualify](https://docs.snowflake.com/en/sql-reference/constructs/qualify.html) 子句允许我们直接过滤窗口函数的结果，而不是先在 CTE 中创建结果，然后再过滤。窗口函数的一个非常常见的方法是使用子查询首先获得 row_number()

```
row_number() over (partition by email order by created_at desc) as date_ranking
```

然后在另一个 CTE 中对此进行过滤，以获得组中的第一行。

```
where date_ranking = 1
```

这种方法的问题是它需要一个额外的子查询。在 Snowflake 中，这可以通过使用 qualify 将窗口函数应用为 where 并同时执行这两个步骤来实现。

Qualify 还有另一个非常强大的用例。一个常见的即席或 QA 查询是检查重复项，以找出唯一性测试失败的原因，并避免无意中复制行的连接。这通常看起来像这样。

```
select
   product_id,
   count(*)
from product_sales
group by 1
having count(*) > 1
order by 2 desc
```

然而，这仅仅给了我们主键，并没有告诉我们副本出现在哪一列。要修复重复，我们需要知道是什么导致了它，因此最简单的方法就是能够看到所有的列。这可以通过使用上述查询的 CTE，然后执行另一个在 id 上过滤的选择(或者通过复制和粘贴主键值)来完成。

```
with base as (
  select
     product_id,
     count(*)
  from product_sales
  group by 1
  having count(*) > 1
  order by 2 desc
)select *
from product_sales
where product_id in (select product_id from base)
```

但是现在我们知道了 qualify 的存在，我们实际上可以在四分之一的行中完成这个查询，而不需要任何额外的步骤。

```
select *
from product_sales
qualify count(*) over (partition by product_id) > 1
```

# 2.敌我识别系统(Identification Friend or Foe)

[iff](https://docs.snowflake.com/en/sql-reference/functions/iff.html) 子句允许我们使用一个简单的例子，但是在语法上更漂亮。这样做的好处是可以替换单个比较的 CASE 子句(例如，创建一个 true/false 字段)。

```
case when col is null then true else false end
```

我们现在可以用更少的单词和更常用的语法(如 Excel 或 Python)来执行上述功能，这就是`if a then b else c`逻辑。

这比前一种方法更漂亮(我认为),也清楚地表明了哪些情况下只执行一次比较，而哪些情况下实际上需要 CASE 子句。当与其他子句链接时是否更容易理解，因为它是一个带有开始和结束括号的自包含函数。

# 3.在枢轴上转动

在对每个列执行相同的聚合时， [pivot](https://docs.snowflake.com/en/sql-reference/constructs/pivot.html) 子句用于将一列中的唯一值分散到多个列中。透视值是一种常见的技术，用于对总数进行分段以供进一步分析，例如在创建产品销售群组视图以查看逐月业绩时。像 sql 中的许多东西一样，这可以通过使用 CASE 语句来实现。

```
select
  product_id,
  sum(case when month = 'jan' then amount else 0 end) as amount_jan,
  sum(case when month = 'feb' then amount else 0 end) as amount_feb,
  sum(case when month = 'mar' then amount else 0 end) as amount_mar
from product_sales
group by 1
order by product_id
```

但是，这种方法要求我们为我们想要透视的每个月值重复 CASE 逻辑，随着月数的增加，这可能会变得很长(想象一下，如果我们想要透视 2 年的值)。值得庆幸的是，在 Snowflake 中这是不必要的，因为我们有 pivot 子句可用，但是要使用该子句，我们首先必须将表简化为只有行列(仍然是行)、pivot 列(不同的值分布在多个列中)和值列(填充单元格值)。

这里，透视列在 AS 子句中有别名，以便使列名更具信息性，并删除列名中出现的引号，以便将来更容易引用它们。

# 4.尝试 _ 到 _ 日期

[try_to_date](https://docs.snowflake.com/en/sql-reference/functions/try_to_date.html) 子句使我们能够尝试多种类型的日期转换而不抛出错误。如果日期存储为字符串(不要这样做)或通过某种自由流动的文本框收集(也不要这样做)，这将特别有用。理论上，您处理的所有日期都应该作为日期或时间戳类型存储在数据库中，但在实践中，您可能会遇到需要将多种类型的日期字符串转换为日期的情况。这就是该子句的亮点，因为您可以应用各种日期格式而不会出现错误。

假设我们将日期存储为文本列中的`14/12/2020`和`19 September 2020`。如果我们试图将该列转换为日期，如果有任何日期不能正确转换，我们将会得到一个错误。

```
Date '19 September 2020' is not recognized
Date '14/12/2020' is not recognized
```

通过返回 null 而不是错误，try_to_date 解决了我们之前的困境，它使我们能够将列转换为多种日期格式而不会引发错误，如果没有找到有效的日期转换，则最终返回 null。我们可以用一个 coalesce 子句将多种日期格式链接起来以实现这一点。

这也处理了雪花的假设，即日期是以`MM/DD/YYYY`格式表示的，即使对于像 `14/12/2020`这样的情况，这样的日期是不可能的，因为这将意味着一个月大于 12。

# 5.变量引用

可能是我们今天要讨论的最强大的技术。当执行 select 语句时，雪花实际上允许我们在查询的其他地方重用逻辑。这消除了对复制/粘贴业务逻辑的需要，这是在编写业务逻辑可能变得庞大和复杂的查询时的一个常见问题。在 select 和 where 中，有时甚至在 group 或 order by 子句中重复这样的逻辑既麻烦又不方便。

下面是一个简单的例子，我们重用了`month`别名，而不是重复最初构建它的查询。

```
select
   date_trunc('month', created_at) as month,
   count(*) as total_transactions
from product_sales
where month = '2022-01-01'
```

但是，如果我们使用的引用变成隐式的，我们需要小心(有两列引用)。在下面的例子中，雪花将使用第一个/已经存在的列`i.status`，而不是新创建的列。

```
select
  iff(p.status in ('open', 'active'), 'active', i.status) as status,
  iff(status = 'active', true, false) as is_active
from product_sales p
```

为了解决这个问题，我们可以简单地给中间列取不同的别名。这有助于减少成本和执行时间，因为我们只需要构建一次业务逻辑！

这并不总是我最喜欢的结果，因为我遇到过在引用它之前应用一些转换来实现结果的情况。正如我们之前看到的，这遇到了我们在`status`中遇到的重复别名的问题，所以如果有人设法找到了一个很酷的解决方案，请告诉我！

# 最后的想法

雪花是一个强大的数据库解决方案，它还具有一些非常有用的查询选项。我们看了一些 help，它们可以帮助我们绕过一些常见的查询障碍，减少相同输出所需的行数，最重要的是，改进语法和可读性，并减少成本和执行时间。

如果您喜欢这篇文章，您可以在我的上找到更多文章，并在我的[个人资料](https://medium.com/@anthonyli358)上关注我！