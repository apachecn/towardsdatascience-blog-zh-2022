# MySQL 中如何用分组计算中位数

> 原文：<https://towardsdatascience.com/how-to-calculate-medians-with-grouping-in-mysql-abb22a3e5097>

## 学习高级 MySQL 查询来计算不同场合的中位数

![](img/66cd7a279d1ef31fbb7ec27dc6468896.png)

[图片由杰勒特拍摄于 Pixabay](https://pixabay.com/illustrations/entrepreneur-diagram-curve-4664726/)

在任何编程语言中，计算一组数据的中值都非常简单，即使在 Excel 中，也可以直接使用内置或第三方中值函数。但是，在 MySQL 中，median 函数不是原生支持的。为了得到中间值，我们需要编写一些带有子查询的智能查询。

在本帖中，我们将揭开 MySQL 中计算中位数的查询的神秘面纱。特别是，我们将演示如何计算分组字段的中位数。如果一个列有多个类别，这些类别的中位数应该单独计算，那么逐个计算每个类别的中位数就变得很麻烦。在这篇文章中，你可以通过一个简单的查询来计算所有类别的中位数。此外，您还将学习如何同时计算多个列的中位数。

## 准备

首先，我们需要一个 MySQL 服务器，创建一个数据库和表，并插入一些虚拟数据进行处理。我们将使用 Docker 在本地启动一个 MySQL 8 服务器:

```
# Create a volume to persist the data.
$ docker volume create mysql8-data

# Create the container for MySQL.
$ docker run --name mysql8 -d -e MYSQL_ROOT_PASSWORD=root -p 13306:3306 -v mysql8-data:/var/lib/mysql mysql:8

# Connect to the local MySQL server in Docker.
$ docker exec -it mysql8 mysql -u root -proot

mysql> SELECT VERSION();
+-----------+
| VERSION() |
+-----------+
| 8.0.31    |
+-----------+
1 row in set (0.00 sec)
```

要编写复杂的 SQL 查询，建议使用 IDE 而不是控制台。IDE 的好处是代码完成、代码格式化、语法突出显示等。在本文中，我们将使用 [DBeaver](/some-tips-for-using-dbeaver-a-univeral-database-tool-94af18d50671) ，这是一个通用的数据库管理工具，可用于管理各种关系数据库和 NoSQL 数据库。然而，任何 IDE 都可以工作。您甚至可以复制本文中演示的查询，并直接在 MySQL 控制台中运行它们。

对于 DBeaver，如果遇到“不允许检索公钥”错误，应该编辑连接并为驱动程序添加以下两个用户属性:

```
# User properties
useSSL: false
allowPublicKeyRetrieval: true
```

更多 DBeaver 的设置，请查看[本帖](/some-tips-for-using-dbeaver-a-univeral-database-tool-94af18d50671)。

然后运行以下查询来创建数据库、创建表并插入一些虚拟数据:

```
CREATE DATABASE products;

CREATE TABLE `products`.`prices` (
  `pid` int(11) NOT NULL AUTO_INCREMENT,
  `category` varchar(100) NOT NULL,
  `price` float NOT NULL,
  PRIMARY KEY (`pid`)
);

INSERT INTO products.prices
    (pid, category, price)
VALUES
    (1, 'A', 2),
    (2, 'A', 1),
    (3, 'A', 5),
    (4, 'A', 4),
    (5, 'A', 3),
    (6, 'B', 6),
    (7, 'B', 4),
    (8, 'B', 3),
    (9, 'B', 5),
    (10, 'B', 2),
    (11, 'B', 1)
;
```

## 以“经典”方式计算中位数

既然数据库和数据已经设置好了，我们可以开始计算中位数了。传统的解决方案是使用 SQL 变量:

```
SELECT AVG(sub.price) AS median
FROM ( 
    SELECT @row_index := @row_index + 1 AS row_index, p.price
    FROM products.prices p, (SELECT @row_index := -1) r
    WHERE p.category = 'A'
    ORDER BY p.price 
) AS sub
WHERE sub.row_index IN (FLOOR(@row_index / 2), CEIL(@row_index / 2))
;

median|
------+
   3.0|
```

此查询的要点:

*   `@row_index`是一个 SQL 变量，在`FROM`语句中初始化，并在`SELECT`语句中为每一行更新。
*   应该对将要计算中值的列(本例中的`price`列)进行排序。不管是升序还是降序排序都没关系。
*   根据[对中位数](https://en.wikipedia.org/wiki/Median)的定义，中位数是中间元素的值(总计数为奇数)或两个中间元素的平均值(总计数为偶数)。在本例中，类别 A 有 5 行，因此中位数是排序后第三行的值。`FLOOR(@row_index / 2)`和`CEIL(@row_index / 2)`的值都是 2，这是第三行。另一方面，对于具有 6 行的类别 B，中值是第三和第四行的平均值。

这个解决方案简单易懂。但是，如果表有许多类别，我们需要为每个类别运行查询，这很麻烦，并且结果不容易存储和比较。

为了解决这个问题，我们需要一个使用`GROUP BY`、`GROUP_CONCAT`和 `SUBSTRING_INDEX`的非经典解决方案。

## 灵活计算中位数

让我们一步一步来。最终的查询乍看起来可能很复杂。然而，一旦你知道它是如何工作的，它实际上更容易理解，你可以根据自己的用例自由地改变它。

让我们首先获得每个类别的所有排序价格:

```
SELECT
    category,
    GROUP_CONCAT(price ORDER BY p.price) AS prices,
    COUNT(*) AS total
FROM products.prices p
GROUP BY p.category
;

category|prices     |total|
--------+-----------+-----+
A       |1,2,3,4,5  |    5|
B       |1,2,3,4,5,6|    6|
```

注意，如果你的表有很多数据，`GROUP_CONCAT`不会包含所有的数据。在这种情况下，通过以下方式增加`GROUP_CONCAT`的限值:

```
SET GROUP_CONCAT_MAX_LEN = 100000;
```

您可以将限制设置为适合您的用例的数字。但是，如果您的表包含太多数据，您可能会遇到内存问题。在这种情况下，您需要编写一些脚本，以更智能的方式执行数据处理和计算。尽管如此，本文中提供的解决方案适用于大多数中小型表。

然后，让我们获得每个类别的中间元素，我们需要检查总计数是奇数还是偶数，并相应地处理它:

```
SELECT 
    sub1.category,
    sub1.total,
    CASE WHEN MOD(sub1.total, 2) = 1 THEN SUBSTRING_INDEX(SUBSTRING_INDEX(sub1.prices, ',', CEIL(sub1.total/2)), ',', '-1')
         WHEN MOD(sub1.total, 2) = 0 THEN SUBSTRING_INDEX(SUBSTRING_INDEX(sub1.prices, ',', sub1.total/2 + 1), ',', '-2')
    END AS mid_prices
FROM 
    (
        SELECT
            p.category,
            GROUP_CONCAT(p.price ORDER BY p.price) AS prices,
            COUNT(*) AS total
        FROM products.prices p
        GROUP BY p.category
    ) sub1
;

category|total|mid_prices|
--------+-----+----------+
A       |    5|3         |
B       |    6|3,4       |
```

我们使用`MOD`函数来检查总数是奇数还是偶数。使用了两次`SUBSTRING_INDEX`函数来提取中间元素。让我们更详细地演示一下它是如何工作的:

```
-- Categoy A, 5 rows:
SUBSTRING_INDEX('1,2,3,4,5', ',', CEIL(5/2)) => '1,2,3'
SUBSTRING_INDEX('1,2,3', ',', -1) => 3

-- Categoy B, 6 rows:
SUBSTRING_INDEX('1,2,3,4,5,6', ',', 6/2 + 1) => '1,2,3,4'
SUBSTRING_INDEX('1,2,3,4', ',', -2) => '3,4'
```

最后，让我们计算中间元素的平均值，以获得每个类别的中值:

```
SELECT
    sub2.category,
    CASE WHEN MOD(sub2.total, 2) = 1 THEN sub2.mid_prices
         WHEN MOD(sub2.total, 2) = 0 THEN (SUBSTRING_INDEX(sub2.mid_prices, ',', 1) + SUBSTRING_INDEX(sub2.mid_prices, ',', -1)) / 2
    END AS median    
FROM 
    (
        SELECT 
            sub1.category,
            sub1.total,
            CASE WHEN MOD(sub1.total, 2) = 1 THEN SUBSTRING_INDEX(SUBSTRING_INDEX(sub1.prices, ',', CEIL(sub1.total/2)), ',', '-1')
                 WHEN MOD(sub1.total, 2) = 0 THEN SUBSTRING_INDEX(SUBSTRING_INDEX(sub1.prices, ',', sub1.total/2 + 1), ',', '-2')
            END AS mid_prices
        FROM 
            (
                SELECT
                    p.category,
                    GROUP_CONCAT(p.price ORDER BY p.price) AS prices,
                    COUNT(*) AS total
                FROM products.prices p
                GROUP BY p.category
            ) sub1
    ) sub2
;

category|median|
--------+------+
A       |3     |
B       |3.5   |
```

干杯！两个类别的中值计算正确。这比传统的解决方案要多一点代码。然而，它更透明，因此更容易理解。此外，它更加灵活，您可以针对不同的用例轻松调整查询，而不仅仅是使用分组来获得不同类别的中值。

## 奖金-计算多列的中间值

在中间值以上，只计算一列。有了新的解决方案，我们可以很容易地计算多个列的中位数。让我们首先用一些虚拟数据创建一个新表:

```
CREATE TABLE `products`.`orders` (
  `order_id` int(11) NOT NULL AUTO_INCREMENT,
  `price` float NOT NULL,
  `quantity` float NOT NULL,
  PRIMARY KEY (`order_id`)
);

INSERT INTO products.orders
    (order_id, price, quantity)
VALUES
    (1, 2, 50),
    (2, 1, 40),
    (3, 5, 10),
    (4, 3, 30),
    (5, 4, 20)
;
```

请注意，这些数据是假的，只是为了演示，因此越简单越好。

用于计算多个列的中间值的查询可以很容易地从上面的查询修改而来。不同之处在于，我们不再需要按类别分组，而是需要对需要计算中值的每一列重复查询:

```
SELECT
    CASE WHEN MOD(sub2.total, 2) = 1 THEN sub2.mid_prices
         WHEN MOD(sub2.total, 2) = 0 THEN (SUBSTRING_INDEX(sub2.mid_prices, ',', 1) + SUBSTRING_INDEX(sub2.mid_prices, ',', -1)) / 2
    END AS median_of_price,
    CASE WHEN MOD(sub2.total, 2) = 1 THEN sub2.mid_quantities
         WHEN MOD(sub2.total, 2) = 0 THEN (SUBSTRING_INDEX(sub2.mid_quantities, ',', 1) + SUBSTRING_INDEX(sub2.mid_prices, ',', -1)) / 2
    END AS median_of_quantity
FROM 
    (
        SELECT 
            sub1.total,
            CASE WHEN MOD(sub1.total, 2) = 1 THEN SUBSTRING_INDEX(SUBSTRING_INDEX(sub1.prices, ',', CEIL(sub1.total/2)), ',', '-1')
                 WHEN MOD(sub1.total, 2) = 0 THEN SUBSTRING_INDEX(SUBSTRING_INDEX(sub1.prices, ',', sub1.total/2 + 1), ',', '-2')
            END AS mid_prices,
            CASE WHEN MOD(sub1.total, 2) = 1 THEN SUBSTRING_INDEX(SUBSTRING_INDEX(sub1.quantities, ',', CEIL(sub1.total/2)), ',', '-1')
                 WHEN MOD(sub1.total, 2) = 0 THEN SUBSTRING_INDEX(SUBSTRING_INDEX(sub1.quantities, ',', sub1.total/2 + 1), ',', '-2')                 
            END AS mid_quantities
        FROM 
            (
                SELECT
                    COUNT(*) AS total,
                    GROUP_CONCAT(o.price ORDER BY o.price) AS prices,
                    GROUP_CONCAT(o.quantity ORDER BY o.quantity) AS quantities
                FROM products.orders o
            ) sub1
    ) sub2
;

median_of_price|median_of_quantity|
---------------+------------------+
3              |30                |
```

再次欢呼，果然有效！

目前，计算中位数的功能还没有在 MySQL 中实现，因此我们需要自己编写一些查询来计算它们。本文介绍了两种解决方案。第一种是使用 SQL 变量的经典解决方案。第二个是新的，用`GROUP_CONCAT`和`SUBSTRING_INDEX`完成。第二个代码多一点，但是可扩展性更好。您可以使用它来计算不同类别的相同字段的中位数，也可以计算多个字段的中位数。

## 相关文章:

*   [使用通用数据库工具 DBeaver 的一些技巧](/some-tips-for-using-dbeaver-a-univeral-database-tool-94af18d50671)
*   [如何用 Python 中的 SQLAlchemy 执行普通 SQL 查询](https://betterprogramming.pub/how-to-execute-plain-sql-queries-with-sqlalchemy-627a3741fdb1)