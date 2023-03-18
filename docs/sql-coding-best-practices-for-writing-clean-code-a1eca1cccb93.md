# 编写干净代码的 SQL 编码最佳实践

> 原文：<https://towardsdatascience.com/sql-coding-best-practices-for-writing-clean-code-a1eca1cccb93>

## 了解如何使您的 SQL 代码可解释

![](img/f7594ac30ace5c0bd0658d3d17d80887.png)

编写干净代码的 SQL 编码最佳实践(图片由作者提供)

# 为什么我们需要遵循 SQL 编码最佳实践？

获得好的分析结果的旅程充满了探索、实验和多种方法的测试。在这个过程中，我们编写了许多代码，我们和其他人在团队工作时会参考这些代码。如果您的代码不够干净，那么理解、重用或修改您的代码对其他人(您也一样)来说将是一种痛苦。

> 如果你写了一个整洁干净的代码，你会感谢自己，别人也会感谢你。

这篇博客将向您介绍编写整洁干净代码的 SQL 编码最佳实践。

# SQL 编码最佳实践

## 1.**子句使用大写字母**

您应该使用大写的 SQL 子句，因为它将帮助您或任何阅读您的代码的人容易地找到您正在执行的操作/操纵。

> **不好的做法**

```
**Select** order_id, prod_id, sales **from** db.Order
```

> **好做法**

```
**SELECT** order_id, prod_id, sales **FROM** db.Order
```

## **2。使用缩进/适当的间距**

让你的代码有一个合适的缩进是非常重要的，它让你的代码看起来干净，因此任何人都很容易理解。

*缩进提示:*

1.  每行选择一列。
2.  每行写一个从句。

> **不良做法**

```
SELECT columnB, AGG_FUNC(column_or_expression) FROM left_table    JOIN right_table ON left_table.columnA = right_table.columnA    WHERE filter_expression    GROUP BY columnB    
HAVING filter_expression    ORDER BY columnB ASC/DESC   LIMIT ROW_COUNT;
```

> **良好实践**

```
SELECT columnB, 
       AGG_FUNC(columnC) 
FROM left_table    
JOIN right_table 
ON left_table.columnA = right_table.columnA    
WHERE filter_expression    
GROUP BY columnB    
HAVING filter_expression    
ORDER BY columnB ASC/DESC    
LIMIT ROW_COUNT;
```

## **3。使用别名**

别名用于为列提供直观的用户定义名称。SQL 中的别名是使用“as”创建的，后跟用户定义的名称。

> **不良做法**

```
SELECT OrderID,
       SUM(Price), 
FROM table1 
INNER JOIN table 2 
ON table1.ProductID = table 2.ProductID
GROUP BY OrderID
```

> **良好实践**

```
SELECT **Products**.OrderID,
       SUM(**OrderDetails**.Price) as **Order_Amount**, 
FROM table1 as **OrderDetails**
INNER JOIN table 2 as **Products**
ON OrderDetails.ProductID = Products.ProductID
GROUP BY **Products**.OrderID
```

## **4。直观的表格名称:**

设置直观的表名使得开发人员很容易理解代码的目标。这些名字应该代表你想要达到的目标。

> **不好的做法**

```
CREATE TABLE ***db.table1*** as 
SELECT OrderDate, 
       count(OrderID) 
FROM Orders
WHERE EmployeeID = 3
GROUP BY OrderDate
```

> **良好实践**

```
CREATE TABLE **db.Orders_across_dates** as 
SELECT OrderDate, 
       count(OrderID) as Total_Orders
FROM Orders
WHERE EmployeeID = 3
GROUP BY OrderDate
```

## 5.注释代码

对代码进行注释是使代码可读和易于理解的最重要的步骤之一。

意见应涵盖以下方面:

1.  守则的目的。
2.  作者姓名。
3.  脚本日期。
4.  代码的描述。

> **不好的做法**

```
SELECT Products.OrderID,
       SUM(OrderDetails.Price) as Order_Amount, 
FROM table1 as OrderDetails
INNER JOIN table 2 as Products
ON OrderDetails.ProductID = Products.ProductID
GROUP BY Products.OrderID
```

> **良好做法**

```
**/*
Objective: Get the order amount corresponding to each order.
Author: Anmol Tomar
Script Date: May 05, 2022
Description: This code gives the order id and corresponding order amount by joining OrderDetails and Products table on ProductID (Primary Key)
*/** SELECT Products.OrderID,
       SUM(OrderDetails.Price) as Order_Amount, 
FROM table1 as OrderDetails
INNER JOIN table 2 as Products
ON OrderDetails.ProductID = Products.ProductID
GROUP BY Products.OrderID
```

## 6.在 SELECT 子句中提及列名

在 SELECT 子句中使用列名对于理解我们为输出选择哪些列(或信息)非常有用。使用 select *会使输出有点像黑盒，因为它隐藏了所选的信息/列。

> **不好的做法**

```
SELECT ***** 
FROM db.Order
```

> **良好实践**

```
SELECT **order_id,
       prod_id,
       sales** 
FROM db.Order
```

## 谢谢大家！

如果您觉得我的博客有用，那么您可以 [***关注我***](https://anmol3015.medium.com/subscribe) *每当我发布一个故事时，您都可以直接获得通知。*

*如果你自己喜欢体验媒介，可以考虑通过* [***注册会员***](https://anmol3015.medium.com/membership) *来支持我和其他几千位作家。它每个月只需要 5 美元，它极大地支持了我们，作家，而且你也有机会通过你的写作赚钱。*