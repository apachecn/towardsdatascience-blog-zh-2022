# 将 SQL 翻译成 Python —第 2 部分

> 原文：<https://towardsdatascience.com/translating-sql-into-python-part-2-5b69c571ddc3>

## 有点像谷歌翻译，但对于̶l̶o̶v̶e̶语言的数据

![](img/4f4c39ed4c9c8e44cf02424e6051c629.png)

作者图片

如果您对 Python 中 SQL 的相应语法感到厌倦，那么您就找对地方了。用两种语言写作的能力不仅能让你对客户或雇主更有吸引力，还能让你写出更高效、更精简的数据流程，从而节省你的大量时间。

在 python 中，您可以创建一个直接查询 SQL 数据库的脚本，然后在同一个脚本中编写后续数据流程。按 run，这就是你在一个命令中单独运行的所有步骤。相信我，对于您定期运行的流程来说，这是一个巨大的胜利，因为时间会越积越多。

本文建立在[第 1 部分](/translating-sql-into-python-part-1-92bf385d08f1#ca3b)的基础上，第 1 部分涵盖了“选择”、“哪里”、“更新”、“连接”和“排序”，可以通过下面的链接。

</translating-sql-into-python-part-1-92bf385d08f1>  

对于这一期，使用下面的内容跳到感兴趣的部分或 ctrl+f 记住这个函数。

**目录:**

1.  <#61b8>
2.  **[**【分组依据&聚合】**](#7735)**
3.  **<#a22c>**
4.  ****<#a552>****(尾随空格、串连、赶作等。)********

******对于这些例子，我们将使用 pandas、datetime 和 numpy 库。确保您已经安装了这些组件(在 Anaconda 包中它们都是默认的),并使用以下命令将其导入到您的脚本中:******

```
****import pandas as pd
import numpy as np
import datetime as dt****
```

******在这篇文章中，我使用了 postgres SQL 语法，MS SQL 和 Postgres SQL 之间存在微小的差异，但是这三者似乎都有些夸张(或者是另一篇文章的想法；)).******

## ******1.《约会时报》******

******确保您的日期时间在 python 中是这样格式化的(即不是字符串)。您可以通过以下方式实现这一点:******

```
****table['timestamp'] = pd.to_datetime(table['timestamp'])****
```

********a .日期部分********

****隔离日期时间的某个“部分”,例如日或年。****

****I .从时间戳中选择日期****

```
****-- SQL**
select date_part('day', timestamp) from table**# python**
table['day'] = table['timestamp'].dt.day**
```

****二。从时间戳中选择小时****

```
****-- SQL**
select date_part('h', timestamp) from table**# python**
table['hour'] = table['timestamp'].dt.hour**
```

****二。从时间戳中选择月份****

```
****-- SQL**
select date_part('m', timestamp) from table**# python**
table['month'] = table['timestamp'].dt.month**
```

******b. Date_trunc******

****将时间戳截断到相应的精度，例如小时、天或月(我相信这是 postgres SQL 特有的)。****

****I .从时间戳中选择精确到天****

```
****-- SQL**
select date_trunc('day',timestamp) as day from table
**# python**
table['day'] = pd.to_datetime(table['timestamp']).dt.to_period('D')**
```

****I .从时间戳选择精确到月****

```
****-- SQL**
select timestamp, date_trunc('month', timestamp) as month from table
**# python**
table['month'] = pd.to_datetime(table['timestamp']).dt.to_period('M')**
```

## ****2.“分组依据”和“聚合”****

****a.Groupby 单个列和聚合****

```
****-- SQL**
select name, count(*) from table group by name**group by # python**
table_grouped = table.groupby('name').count()**
```

****b.Groupby 多列和聚合单列****

```
****-- SQL**
select name, age, avg(height), max(height) from table group by name, age**-- python**
table_grouped =   table.groupby(['name','age']).agg({'height':   ['mean', 'max']})**
```

****b.按多列分组并在多列上聚合****

```
****-- SQL**
select name, age, avg(height), min(weight) from table group by name, age**-- python**
table_grouped =   table.groupby(['name','age']).agg({'height':   ['mean'], 'weight': ['min'})**
```

****c.按列分组、聚合和重命名聚合列****

```
****-- SQL**
select age, avg(height) as avg_height from table group by name, age**-- python** table_grouped =   table.groupby(['name','age']).agg({'height':   ['mean']}).rename(columns= {‘mean’: ‘avg_height’})**
```

****d.按年月分组****

```
****-- SQL**
select date_trunc('month', timestamp) as yearmonth, country, sum(rainfall) from table group by date_trunc('month', timestamp), country**-- python** table['month_year'] = pd.to_datetime(table['timestamp']).dt.to_period('M')table_grouped = table.groupby(['month_year', 'country']).agg({'rainfall': ['sum']})**
```

****e.按一年中的月份分组****

```
****-- SQL**
select date_part('day', timestamp) as month, country, std(rainfall) from table**-- python** table['month'] = table['timestamp'].dt.monthtable_grouped =   table.groupby(['month', 'country']).agg({'rainfall': ['std']})**
```

## ****3.“案件何时发生”****

****I .创建具有单一案例结果的新属性，否则没有变化****

```
****-- SQL**
select asset, age, 
case when age>= 20 or priority = 1 then 'To be replaced'
else status
end as status
from table**# python**
table.loc[(table.age > 20) | (priority == 1), ‘status’] = ‘To be replaced’
table[['asset', 'age', 'status']]**
```

****二。创建具有多个案例结果的新属性****

```
****-- SQL**
select Date, 
case when Date >= '2008-08-31' and Date <= '2009-05-30' then 'period1'
when Date >= '2009-08-31' and Date <= '2010-05-30' THEN 'period2'
when Date >= '2009-08-31' and Date <= '2010-05-30' THEN 'period3'
else 'N/A'
end as period from table**# python**
conditions = 
[table['Date'].between('2008-08-30', '2009-05-30', inclusive=True),
 table['Date'].between('2009-08-31', '2010-05-30', inclusive=True),
 table['Date'].between('2010-08-31', '2011-05-30', inclusive=True)]choices = ['period1', 'period2', 'period3']table['period'] = np.select(conditions, choices, default='N/A')table[['Date', 'period']]**
```

## ****4.**杂项******

****a.删除列中的尾随空格****

****I .删除前面的空格****

```
****-- SQL**
update table
set col1 = LTRIM(col1)**# python**
table['col1'] = table['col1'].str.lstrip()**
```

****二。删除后面的空格****

```
****-- SQL**
update table
set col1 = RTRIM(col1)**# python**
table['col1'] = table['col1'].str.rstrip()**
```

****三。删除前后的空格****

```
****-- SQL**
update table
set col1 = TRIM(col1)**# python**
table['col1'] = table['col1'].str.strip()**
```

****b.连接字符串****

```
****-- SQL**
select concat(col1, ' - ', col2) as newcol from table**# python**
table['newcol'] = table['col1'] + ' - ' + table['col2']**
```

****c.铸造柱组件****

****I .转换为字符串****

```
****-- SQL**
select cast (col1 as varchar) from table**# python**
table.['col1'] = table.[col1].astype(str)**
```

****二。转换为整数****

```
****-- SQL**
select cast (col1 as int) from table**# python**
table.['col1'] = table.[col1].astype(int)**
```

****三。转换为日期时间****

```
****-- SQL**
SELECT CAST(col1 AS datetime) from table**# python**
table['col1'] = pd.to_datetime(table['col1'])**
```

****这些不会涵盖所有的用例，但是应该提供一些不错的模板。更多详细信息，请参考[熊猫](https://pandas.pydata.org/docs/)、 [numpy](https://numpy.org/doc/) 和 [datetime](https://docs.python.org/3/library/datetime.html) 文档，这些文档也涵盖了输入参数，让您的代码更加高效。****

****<https://medium.com/@hollydalligan/membership> ****