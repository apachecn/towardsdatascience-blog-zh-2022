# 数据科学家的 SQL 访谈研究计划

> 原文：<https://towardsdatascience.com/sql-interview-study-plan-for-data-scientists-4a64b132a95d>

## 带 LeetCode 问题的 SQL 学习计划

![](img/08da700af3ef852fcfb171fb299486a6.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由[Christina @ wocintechchat.com](https://unsplash.com/@wocintechchat?utm_source=medium&utm_medium=referral)拍摄的照片

# 学习计划介绍

大量的数据科学职位需要精通 SQL。因此，SQL 技术面试通常包含在数据科学面试流程中。

LeetCode 是练习面试的一个很好的资源，从数据结构和算法到 SQL。LeetCode 将他们的问题分为简单、中等和困难三个难度级别。LeetCode 也有自己的 SQL 学习计划；然而，SQL 主题没有被很好地分类(或者在某些情况下是正确的)，因此我发现 LeetCode 作为测试工具比作为学习工具更有帮助。

本学习指南将 SQL 问题分为不同的 SQL 主题，以便用户可以通过重点关注和重复来提高各个领域*的能力。你会发现 LeetCode 问题的精选链接，这些问题很好地代表了每个主题的面试问题。*

## 学习计划时间表

本学习指南的一个好节奏是每天尝试 2-4 个 SQL 问题。这使您可以选择一个在学习期间将重点关注的 SQL 主题，并巩固您对该领域的理解。如果你觉得你在基础领域已经很强了，你可以跳过它们，专注于更中级和高级的主题。

![](img/ba42ce854078af9c702639327898ae6c.png)

由[埃斯特·扬森斯](https://unsplash.com/@esteejanssens?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

第 9 天—第 10 天:这几天包括**窗口功能**。我发现一个很好的学习来源是 sqltutorial.org[SQL 窗口函数页面](https://www.sqltutorial.org/sql-window-functions/)。通读短文，然后浏览页面底部的每个窗口函数(值窗口函数和排名窗口函数)。关于窗口函数的非高级 LeetCode 问题不多，所以建议在这方面进行进一步的练习，尤其是排名函数。

# 学习计划

## 第一天:选择和过滤

SQL `SELECT`函数从一个或多个表中选择列。SQL `WHERE`子句允许您基于一个或多个条件过滤行。

*   [当列匹配时选择](https://leetcode.com/problems/article-views-i/)
*   [多条件选择](https://leetcode.com/problems/big-countries/)
*   [对包含空值的列进行过滤](https://leetcode.com/problems/find-customer-referee/)
*   [选择第二高的值](https://leetcode.com/problems/second-highest-salary/)

## 第二天:连接和联合

有 4 种主要的连接类型:`INNER JOIN`(或`JOIN`)、`LEFT JOIN` / `RIGHT JOIN`、`FULL OUTER JOIN`和`CROSS JOIN`。

SQL `UNION`函数将两个或多个 select 语句的结果集组合成一个结果。SQL `UNION ALL`函数保留了重复的行。

*   [一侧外连接](https://leetcode.com/problems/combine-two-tables/)
*   [全外连接](https://leetcode.com/problems/employees-with-missing-information/)
*   [选择重复行](https://leetcode.com/problems/duplicate-emails/)
*   [将列重新排列为行值](https://leetcode.com/problems/rearrange-products-table/)

## 第三天:分组依据

SQL `GROUP BY`子句根据一个或多个列的值对行进行分组，为每个组返回一行。您可以对每个组执行聚合功能，如`SUM`和`COUNT`。

*   [组和集合](https://leetcode.com/problems/daily-leads-and-partners/)
*   [加入并分组](https://leetcode.com/problems/customer-who-visited-but-did-not-make-any-transactions/)
*   [带条件的组](https://leetcode.com/problems/the-latest-login-in-2020/)

## 第 4 天:分组依据

SQL `HAVING`子句为`GROUP BY`中定义的组指定了一个条件。这通常用于筛选由 group by 和聚合产生的行。

*   [空值求和](https://leetcode.com/problems/top-travellers/)
*   [有条件的组](https://leetcode.com/problems/actors-and-directors-who-cooperated-at-least-three-times/)

## 第五天:情况何时

SQL `CASE`函数评估一个或多个条件，并返回由该条件定义的结果。这就像一个`if`语句。

*   [用条件计算值](https://leetcode.com/problems/calculate-special-bonus/)
*   [对数据关系进行分类](https://leetcode.com/problems/tree-node/)
*   [负数求和](https://leetcode.com/problems/capital-gainloss/)
*   [透视表](https://leetcode.com/problems/reformat-department-table/)

## 第 6 天:子查询

SQL 子查询是嵌套在另一个查询中的查询。您可以使用一个查询的结果来支持另一个查询。

*   [在子查询中加入](https://leetcode.com/problems/sales-person/)
*   [连接中的子查询](https://leetcode.com/problems/market-analysis-i/)

## 第 7 天:更新并从表中删除

SQL `UPDATE`函数改变表中的现有数据。SQL `DELETE`函数从表中删除一行或多行。

*   [反转一列的值](https://leetcode.com/problems/swap-salary/)
*   [删除重复行](https://leetcode.com/problems/delete-duplicate-emails/)

## 第八天:字符串处理

有`UPPER`、`LOWER`、`CONCAT`、`GROUP_CONCAT`、`TRIM`等多种字符串处理函数，以及正则表达式的利用。熟悉一些常见的 SQL 字符串函数[这里](https://www.w3schools.com/sql/sql_ref_sqlserver.asp)。

*   [只大写第一个字符](https://leetcode.com/problems/fix-names-in-a-table/)
*   [通过](https://leetcode.com/problems/group-sold-products-by-the-date/)与组连接
*   [过滤包含子串的字符串](https://leetcode.com/problems/patients-with-a-condition/)

## 第 9 天:价值窗口函数

`FIRST_VALUE()`和`LAST_VALUE()`窗口函数分别返回一组有序值中的第一个值和最后一个值。`LAG()`窗口功能提供对前一行或多行数据的访问。`LEAD()`窗口功能提供对下一行或多行数据的访问。

*   [选择连续值](https://leetcode.com/problems/consecutive-numbers/)
*   [获取分区顶部](https://leetcode.com/problems/department-highest-salary/)
*   [每两行交换一次](https://leetcode.com/problems/exchange-seats/)

## 第 10 天:排序窗口函数

值得注意的排名窗口功能有`ROW_NUMBER()`、`RANK()`、`DENSE_RANK()`和`NTILE()`。您可以在这里熟悉价值和排名窗口函数[。](https://www.sqltutorial.org/sql-window-functions/)

*   [连续得分排名](https://leetcode.com/problems/rank-scores/)
*   [获取分区的前 3 名](https://leetcode.com/problems/department-top-three-salaries/)

# 摘要

为了测试 SQL 熟练程度，数据科学面试流程中经常包括 SQL 技术面试。LeetCode 是练习面试的一个很好的资源，然而，问题的随机性产生了一个测试工具，而不是一个研究工具。为了一次将学习重点放在一个 SQL 主题上，本学习指南将 LeetCode SQL 问题归类为 SQL 访谈中出现的核心主题。