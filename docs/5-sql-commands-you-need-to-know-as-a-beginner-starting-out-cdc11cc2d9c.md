# 作为初学者，你需要知道的 5 个 SQL 命令

> 原文：<https://towardsdatascience.com/5-sql-commands-you-need-to-know-as-a-beginner-starting-out-cdc11cc2d9c>

# 作为初学者，你需要知道的 5 个 SQL 命令

## 如果你想精通 SQL，你需要知道的 5 个基本命令

![](img/7c8839a0be3730e14ec4288a6bbe5e7f.png)

来自 Unsplash 的照片由 [Lala Azizli](https://unsplash.com/photos/8IJ5xNTv1QM) 拍摄

SQL 是一种强大的语言，可用于各种数据分析任务。

它也非常适合想从事编程的人，因为它与许多其他编程语言非常相似。

在本文中，我们将分解您需要知道的 5 个 SQL 命令，并提供示例，这样到最后，您将对 SQL 有足够的了解，可以开始在自己的项目中使用它了！

# 1.挑选

如果使用 SQL，您需要知道的第一个命令是 SELECT。这是 SQL 中最基本的命令，用于从表中获取数据。

SELECT 的一些用途包括:

*   从表中选择所有数据
*   从表中选择特定的列
*   基于特定标准选择数据(使用 WHERE)

# 示例:

## **从表名**中选择*

这将为您提供 tablename 表中的所有数据。您也可以通过在 select:

## **从表名中选择 id，名称**

这将为您提供 tablename 表中的 id 和 name 列。

## 选择不同

如果只想选择唯一值，可以使用 SELECT DISTINCT。此命令从结果中删除重复值:

## 从表名中选择不同的 id

这将为您提供 tablename 表中所有惟一 id 的列表。

## 选择计数

SELECT COUNT 命令返回表中的行数:

## SELECT COUNT(*) FROM tablename

这将返回 tablename 表中的总行数。您还可以对特定的列进行计数

# 2.在哪里

这里是 SQL 中另一个非常常见的命令。它用于过滤出现在 SELECT 语句中的数据:

WHERE 的一些用法包括:

*   按特定列过滤数据
*   按特定值过滤数据
*   按日期范围过滤数据

# 示例:

## SELECT * FROM tablename，其中 id = 100

这将只返回 tablename 表中 id 等于 100 的行。可以使用 AND 或 or 指定多个条件:

## SELECT * FROM tablename WHERE(id = 100)OR(name = ' John ')

这将返回 tablename 表中 id=100 或 name='John '的所有行。

## SELECT * FROM tablename，其中 id 介于 EN 100 和 200 之间

这将返回 tablename 表中 id 在 100 到 200 之间的所有行。

## SELECT * FROM 不在(100，200)中的表名

这将返回 tablename 表中 id 不等于 100 或 200 的所有行。

# 3.ORDERBY

ORDERBY 也常用在 SQL 中。它用于对 SELECT 语句的结果进行排序。这些结果可以按降序或升序排序。

ORDERBY 的一些用法包括:

*   按升序对结果排序:SELECT * FROM tablename ORDERBY id
*   按降序对结果排序:SELECT * FROM tablename order by id desc
*   按字母顺序对结果排序:SELECT * FROM tablename order by name
*   按日期对结果排序:SELECT * FROM tablename order by created _ at

# 示例:

## SELECT * FROM tablename ORDER BY name

这将返回 tablenname 表中的所有行，并按它们的名称排序。如果要使用多列进行排序，请在逗号分隔的列表中指定它们:

## SELECT * FROM tablename 其中 id > 100 ORDER BY age DESC，name ASC

这将给出 ID 大于 100 的所有行，并首先按年龄降序排列这些值，然后按姓名升序排列。

# 4.分组依据

GROUPBY 是 SQL 中的一条语句，用于按某一列对 SELECT 语句中的数据进行分组。

GROUPBY 的一些用途包括:

*   汇总数据
*   查找列的最大值或最小值
*   获取列的平均值、中值或标准差

# 示例:

## 从按 id 分组的表名中选择 id，name，SUM(age)作为“年龄”

这将返回一个包含三列的表:ID、姓名和年龄。Age 列将包含 tablename 表中按 ID 分组的所有年龄值的总和。

## 从按 id 分组的表名中选择 max(age)作为“最老的人”

这将返回一个包含一列的表:最老的人。“年龄最大的人”列将具有 tablename 表中按 ID 分组的最大年龄值。

## 从按 id 分组的表名中选择 avg(age)作为“平均年龄”

这将返回一个只有一列的表:平均年龄。“平均年龄”列将包含 tablename 表中所有行的平均年龄值，按 ID 分组。

# 5.喜欢

LIKE 运算符用于匹配字符串中的模式。百分号(%)用作通配符，这意味着它可以代表任意数量的字符。

LIKE 的一些用法包括:

*   匹配列中的模式
*   在列中查找特定值

# 示例:

## SELECT id，name FROM tablename WHERE name LIKE ' A % '

这将返回第一列(name)至少包含一次字母 A 的所有行。

## SELECT id，name FROM tablename WHERE name LIKE ' % end '

这将返回包含名为“end”的列的所有行。

## SELECT * FROM tablename，其中名称如“John%”

这将返回 tablename 表中的所有行，其中 name 列包含后跟任意数量字符(%)的字符串 John。可以在字符串的开头、结尾或任何地方使用%。

# 开始掌握 SQL

我们在这篇博文中讨论的 SQL 命令是强大的工具，可以帮助您充分利用数据。

使用这些命令来帮助您分析和优化数据，您将很快掌握 SQL。

[**与 2k+人一起加入我的电子邮件列表，免费获得“完整的 Python 数据科学备忘手册”**](https://dogged-trader-1732.ck.page/datascience)