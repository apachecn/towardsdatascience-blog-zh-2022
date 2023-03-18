# 您应该避免的 5 个最常见的 SQL 错误

> 原文：<https://towardsdatascience.com/5-most-common-sql-mistakes-you-should-avoid-dd4eb4088f0c>

## 了解如何加快 SQL 查询的执行速度

![](img/f7594ac30ace5c0bd0658d3d17d80887.png)

你应该避免的 5 个最常见的 SQL 错误

SQL 是查询关系型大数据最常用的编程语言之一。在 SQL 中，可能有多种创建数据表的方法，但了解并遵循正确的(最佳)方法非常重要，这样您就不必为执行代码而等待数小时。

在这篇博客中，我将带您了解 5 个最常见的 SQL 编程错误，为了优化您的 SQL 查询，您应该避免这些错误。

1.  **不要使用 Select *** : Select *输出一个数据表的所有列，这是一个开销很大的操作，会增加查询的执行时间。理想的方法是只选择子查询或输出表中的相关列。

例如，假设我们想从 Orders 表中获取订单 id，那么我们应该只选择 order id 列，而不是使用 select *选择所有列。

2.**不要用‘Having’代替‘where’**:Having 子句用于对聚合列(sum、min、max 等)应用过滤器。)使用 group by 操作创建。但是有时，程序员也使用‘having’子句(而不是‘where’子句)来过滤非聚集列。

例如，为了查找雇员 id 为 3 的雇员执行的订单总数，使用' having '子句进行筛选将会得到意外的结果。

3.**不要使用‘where’进行连接**:有些人使用‘where’子句执行内部连接，而不是使用‘inner join’子句。尽管两种语法的性能相似(查询优化的结果)，但由于以下原因，不建议使用“where”子句进行联接:

I .在过滤和连接中使用“where”子句会影响可读性和理解。

二。用于连接的“where”子句的**用法**非常有限**，因为除了内部连接之外，它不能执行任何其他连接。**

4.**不使用 Distinct**:Distinct 子句用于通过删除重复行来查找与所选列对应的 Distinct 行。“Distinct”子句在 SQL 中是一个耗时的操作，替代方法是使用“group by”。例如，以下查询从 order details 表中查找订单数:

5.**避免过滤中的谓词**:谓词是一个等同于布尔值(即真或假)的表达式。使用谓词执行过滤操作会降低执行时间，因为谓词不使用索引(SQL 索引是一个用于快速检索的查找表)。因此，应该使用其他替代方法进行过滤。例如，如果我们要查找电话号码以(171)代码开头的供应商。

# **谢谢！**

*如果你觉得我的博客有用，那么你可以* [***关注我***](https://anmol3015.medium.com/subscribe) *每当我发布一个故事，你就可以直接得到通知。*

*如果你自己喜欢体验媒介，可以考虑通过* [***报名成为会员***](https://anmol3015.medium.com/membership) *来支持我和其他成千上万的作家。它每个月只需要 5 美元，它极大地支持了我们，作家，而且你也有机会通过你的写作赚钱。*

分享几个你可能感兴趣的故事:

[](https://anmol3015.medium.com/write-your-sql-queries-the-right-way-9c04dfbb6499) [## 以正确的方式编写您的 SQL 查询！

### 去掉你所有的语法错误！

anmol3015.medium.com](https://anmol3015.medium.com/write-your-sql-queries-the-right-way-9c04dfbb6499) [](/10-sql-operations-for-80-of-your-data-manipulation-7461c56e25f4) [## 80%的数据操作需要 10 次 SQL 操作

### 关系数据库(表格数据)是最常用的数据库之一，它约占全部数据的 70%

towardsdatascience.com](/10-sql-operations-for-80-of-your-data-manipulation-7461c56e25f4)