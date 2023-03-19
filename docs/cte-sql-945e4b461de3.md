# SQL 中的 cte 是什么

> 原文：<https://towardsdatascience.com/cte-sql-945e4b461de3>

## 了解 SQL 中的公用表表达式(CTE)

![](img/bf48c9bc557477057a30bf2d02fde831.png)

由 [Sunder Muthukumaran](https://unsplash.com/@sunder_2k25?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/sql?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄的照片

编写清晰、易读和高效的 SQL 查询是团队中任何工程或分析过程的一个重要方面。这种查询可以有效地维护，并且在适当的时候可以很好地扩展。

为了实现这一点，开发人员和分析人员都可以轻松采用的一个 SQL 结构是通用表表达式(CTE)。

## 常用表表达式

公用表表达式(CTE)是一种构造，用于临时存储指定查询的结果集，以便后续查询可以引用它。CTE 的结果**不是持久存储在磁盘**上，而是其生命周期持续到引用它的查询的执行。

用户可以利用 cte，将复杂的查询分解成更容易维护和阅读的子查询。此外，公共表表达式可以在一个查询中被多次引用，这意味着您不必重复。假定 cte 是命名的，这也意味着用户可以让读者清楚地知道一个特定的表达式应该返回什么结果。

## 构造公共表表达式

每个 CTE 都可以使用`WITH <cte-name> AS`子句来构造

```
WITH sessions_per_user_per_month AS (
    SELECT
      user_id,
      COUNT(*) AS no_of_sessions,
      EXTRACT (MONTH FROM session_datetime) AS session_month,
      EXTRACT (YEAR FROM session_datetime) AS session_year
    FROM user_sessions
    GROUP BY user_id
)
```

在一个查询中可以指定多个 cte，每个 cte 之间用逗号分隔。cte 也可以引用其他 cte:

```
WITH sessions_per_user_per_month AS (
    SELECT
      user_id,
      COUNT(*) AS no_of_sessions,
      EXTRACT (MONTH FROM session_datetime) AS session_month,
      EXTRACT (YEAR FROM session_datetime) AS session_year
    FROM user_sessions
    GROUP BY user_id
),
running_sessions_per_user_per_month AS (
    SELECT
      user_id, 
      SUM(no_of_sessions) OVER (
        PARTITION BY 
          user_id, 
          session_month, 
          session_year
      ) AS running_sessions
    FROM sessions_per_user_per_month
)
```

然后，后续查询可以像任何表或视图一样引用 cte:

```
WITH sessions_per_user_per_month AS (
    SELECT
      user_id,
      COUNT(*) AS no_of_sessions,
      EXTRACT (MONTH FROM session_datetime) AS session_month,
      EXTRACT (YEAR FROM session_datetime) AS session_year
    FROM user_sessions
    GROUP BY user_id
),
running_sessions_per_user_per_month AS (
    SELECT
      user_id, 
      SUM(no_of_sessions) OVER (
        PARTITION BY 
          user_id, 
          session_month, 
          session_year
      ) AS running_sessions
    FROM sessions_per_user_per_month
)

SELECT 
  u.username,
  u.email
  u.country,
  s.running_sessions
FROM users u
LEFT JOIN sessions_per_user_per_month s
  ON u.user_id = s.user_id
WHERE country = 'US';
```

## cte 与子查询

通常，使用子查询可以获得相同的结果。顾名思义，子查询是在另一个查询中定义的查询(也称为嵌套查询)。

有一种误解，认为 cte 往往比子查询执行得更好，但这不是真的。实际上， **CTE** 是一个**语法糖**，这意味着在后台，子查询仍然会被执行，但是在决定是否要编写一个公共表表达式或子查询时，您需要记住一些事情。

**cte 比嵌套查询更具可读性**。您需要做的不是在一个查询中包含两个或多个查询，而是定义一个 CTE 并在后续查询中引用它的名称。

这意味着**cte 还可以被后续查询多次**重用和引用。对于子查询，您必须一遍又一遍地重写相同的查询。

此外， **CTE 可以是递归的**，这意味着它**可以引用自身**。递归 cte 的语法与用于指定非递归 cte 的语法有些不同。您将需要使用`WITH RECURSIVE`来指定它，并使用`UNION ALL`来组合递归调用和基础用例(也称为锚)的结果:

```
-- Syntax used for recursive CTEs
WITH RECURSIVE <cte-name> AS (
  <anchor case>
  UNION ALL
  <recursive case>
)
```

我现在不打算谈论更多关于递归 cte 的细节，但是我计划在接下来的几天里专门为此写一篇文章，所以一定要订阅下面的内容，并且在它发布的时候得到通知！

## 最后的想法

公共表表达式提供了一种简单而强大的方式来编写干净、可读和可维护的 SQL 查询。用户可以利用这种结构来增强跨查询的可重用性，在某些情况下甚至可以提高性能，因为 CTE(临时结果集)可以被多次引用。

只要有可能，cte 应该优先于嵌套连接，因为后者会使您的代码混乱，如果需要多次，会使您的代码可读性更差。此外，cte 还可以是递归的，如果需要的话，这是一大优势。

尽管存在子查询可以提供比 cte 更大灵活性的用例，但是本文并不打算让您相信子查询是完全无用的！例如，因为 cte 必须在`SELECT`子句之前指定，这意味着它们不能在`WHERE`子句中使用。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

<https://gmyrianthous.medium.com/membership>  

**相关文章你可能也喜欢**

</visual-sql-joins-4e3899d9d46c>  </dbt-models-structure-c31c8977b5fc>  </sql-select-distinct-277c61012800> 