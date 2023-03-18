# 编写优秀 SQL 代码的最佳实践

> 原文：<https://towardsdatascience.com/8-best-practices-for-writing-top-tier-sql-code-e1023996e459>

![](img/750be9af5ef5439b35726c53a5eebb53.png)

由 pikisuperstar 创建的 Css 矢量—[www.freepik.com](http://www.freepik.com)

# 介绍

磨练编码技能的最简单也是最有效的方法之一就是让你的代码更具**可读性。**让你的代码可读会让它更易理解，更易重现，更易调试。让你的代码更具可读性的最好方法是执行一些规则或标准，使它一致和干净。

说了这么多，我要和你分享编写顶层 SQL 代码的八个技巧:

> **请务必点击** [**订阅此处**](https://terenceshin.medium.com/membership) **千万不要错过另一篇关于数据科学指南、技巧和提示、生活经验等的文章！**

# 1.大写关键字和函数

我从一个你可能已经在做的简单提示开始，这很好，但你会惊讶于许多人不支持这个提示:当涉及到关键字和函数时，确保它们是大写字母！

虽然这看起来有点吹毛求疵，但像这样的技巧会让你的代码更加清晰。

*而不是:*

```
select name, date_trunc(register_date, week) from customers
```

*试试:*

```
SELECT name, DATE_TRUNC(register_date, WEEK) FROM customers
```

# 2.缩进和对齐

为了建立在前一个例子的基础上，下一个技巧将集中在代码的缩进和对齐上。考虑两个例子——哪个更清晰？

```
SELECT name, height, age, salary, CASE WHEN age<18 THEN "child" ELSE    
"adult" END AS child_or_adult
FROM People LEFT JOIN Income USING (name)
WHERE height<=200 and age<=65
```

或者

```
SELECT 
   name
 , height
 , age
 , salary
 , CASE WHEN age < 18 THEN "child"
        ELSE "adult"
   END AS child_or_adult
FROM 
   People
LEFT JOIN 
   Income USING (name)
WHERE 
   height <= 200
   AND age <= 65
```

很明显，第二个更容易阅读，这完全归功于它的编程风格！特别是，请注意第二个示例中的代码是如何缩进和垂直对齐的。这使得从未看过您的代码的人更容易浏览它。

虽然您不必遵循这种精确的缩进和格式样式，但在整个代码中应用它并开发一致的样式是很重要的。

> **请务必点击** [**订阅此处**](https://terenceshin.medium.com/membership) **千万不要错过另一篇关于数据科学指南、技巧和提示、生活经验等的文章！**

# 3.对模式、表、列应用一致的案例类型

在编程中，有几种类型的案例，仅举几个例子:

*   茶包
*   帕斯卡凯斯
*   蛇 _ 案例

不管你的偏好是什么，重要的是确保你在整个代码中保持一致。

*而不是:*

```
SELECT 
   firstName
 , LastName
 , child_or_adult 
FROM 
   customers
```

*试试:*

```
SELECT 
   first_name
 , last_name
 , child_or_adult 
FROM 
   customers
```

# 4.避免选择*

这是一个重要的技巧，不仅对于格式化，而且对于查询优化(我们将在另一篇文章中讨论！).

即使您发现自己几乎使用了表中的每一列，写出您将需要的列也是一个好习惯。为什么？随着表的发展和更多列的添加/更改，指定列名将使将来识别潜在的错误变得容易。

*而不是:*

```
SELECT 
   *
FROM 
   customers
```

*试试:*

```
SELECT 
   name
 , height
 , age
 , salary
FROM 
   customers
```

> **请务必点击** [**订阅此处**](https://terenceshin.medium.com/membership) **千万不要错过另一篇关于数据科学指南、技巧和提示、生活经验等的文章！**

# 5.用公共表表达式模块化代码

使用公共表表达式(或 cte)对于模块化和分解代码非常有用——就像你将一篇文章分解成几个段落一样。

如果您想更详细地了解 cte，请查看本文[,但是在其核心，cte 创建了一个临时表，允许您“查询一个查询”](https://www.essentialsql.com/introduction-common-table-expressions-ctes/)

*而不是:*

```
SELECT 
   name
 , salary
FROM 
   People
WHERE 
   name IN (SELECT DISTINCT 
              name 
           FROM 
              population 
           WHERE 
              country = "Canada"
              AND city = "Toronto")
   AND salary >= (SELECT 
                     AVG(salary)
                  FROM 
                     salaries
                  WHERE 
                     gender = "Female")
```

*试试:*

```
with toronto_ppl as (
   SELECT DISTINCT 
      name
   FROM 
      population
   WHERE 
      country = "Canada"
      AND city = "Toronto"
), avg_female_salary as (
   SELECT 
      AVG(salary) as avg_salary
   FROM 
      salaries
   WHERE 
      gender = "Female"
)SELECT 
   name
,  salary
FROM 
   People
WHERE 
   name IN(SELECT name FROM toronto_ppl)
   AND salary >= (SELECT avg_salary FROM avg_female_salary)
```

现在很容易看出 WHERE 子句正在过滤多伦多的名称。CTE 非常有用，不仅因为可以将代码分解成更小的块，还因为可以为每个 CTE 分配一个变量名(例如 toronto_ppl 和 avg_female_salary)。

说到变量名，这引出了我的下一个观点:

# 6.描述性变量名称

当您创建变量名时，您希望它们描述它们所代表的内容。考虑我之前的例子:

```
with toronto_ppl as (
   SELECT DISTINCT 
      name
   FROM 
      population
   WHERE 
      country = "Canada"
      AND city = "Toronto"
), avg_female_salary as (
   SELECT 
      AVG(salary) as avg_salary
   FROM 
      salaries
   WHERE 
      gender = "Female"
)SELECT 
   name
,  salary
FROM 
   People
WHERE 
   name IN(SELECT name FROM toronto_ppl)
   AND salary >= (SELECT avg_salary FROM avg_female_salary)
```

只需阅读变量名称本身，就可以清楚地看到，第一个 CTE 正在检索来自多伦多的人，第二个 CTE 正在获取女性的平均工资。

另一方面，这将是一个糟糕的命名约定的例子，(事实上我以前见过):

```
with **table_one** as (
   SELECT DISTINCT 
      name
   FROM 
      population
   WHERE 
      country = "Canada"
      AND city = "Toronto"
), **table_two** as (
   SELECT 
      AVG(salary) as **var_1**
   FROM 
      salaries
   WHERE 
      gender = "Female"
)SELECT 
   name
,  salary
FROM 
   People
WHERE 
   name IN(SELECT name FROM **table_one**)
   AND salary >= (SELECT **var_1** FROM **table_two**)
```

# 7.使用临时函数简化代码

临时函数是

1.  分解代码
2.  编写更干净的代码
3.  并且能够重用代码。

*如果你想了解更多关于临时函数的内容，* [*你可以阅读这篇文章*](https://cloud.google.com/bigquery/docs/reference/standard-sql/user-defined-functions) *。*

考虑下面的例子:

```
SELECT name
       , CASE WHEN tenure < 1 THEN "analyst"
              WHEN tenure BETWEEN 1 and 3 THEN "associate"
              WHEN tenure BETWEEN 3 and 5 THEN "senior"
              WHEN tenure > 5 THEN "vp"
              ELSE "n/a"
         END AS seniority 
FROM employees
```

相反，您可以利用一个临时函数来捕获 CASE 子句。

```
CREATE TEMPORARY FUNCTION seniority(tenure INT64) AS (
   CASE WHEN tenure < 1 THEN "analyst"
        WHEN tenure BETWEEN 1 and 3 THEN "associate"
        WHEN tenure BETWEEN 3 and 5 THEN "senior"
        WHEN tenure > 5 THEN "vp"
        ELSE "n/a"
   END
);SELECT name
       , seniority(tenure) as seniority
FROM employees
```

有了临时函数，查询本身就简单多了，可读性更好，还可以重用资历函数！

# 8.有意义的评论

下面是写评论时最重要的规则:**只在需要的时候写评论**。

通过遵循前面的七个技巧(使用描述性名称、模块化代码、编写干净的代码等。)应该不需要写很多评论。

也就是说，当代码本身不能解释你想要达到的目的时，注释是有用的，也许是必需的。

下面是一个差评的例子:

```
# Getting names of people in Toronto, Canada
with table1 as (
   SELECT DISTINCT name
   FROM population
   WHERE country = "Canada"
         AND city = "Toronto"
)# Getting the average salary of females
, table2 as (
   SELECT AVG(salary) as var1
   FROM salaries
   WHERE gender = "Female"
)
```

这些是糟糕的注释，因为它告诉我们通过阅读代码本身我们已经知道了什么。记住，评论通常会回答你为什么做某事，而不是你在做什么。

# 感谢阅读！

> ***务必*** [***订阅此处***](https://terenceshin.medium.com/membership) ***千万不要错过另一篇关于数据科学指南、诀窍和技巧、生活经验等的文章！***

不确定接下来要读什么？我为你挑选了另一篇文章:

[](/all-machine-learning-algorithms-you-should-know-in-2022-db5b4ccdf32f) [## 2022 年你应该知道的所有机器学习算法

### 最流行的机器学习模型的直观解释

towardsdatascience.com](/all-machine-learning-algorithms-you-should-know-in-2022-db5b4ccdf32f) 

**还有一个:**

[](/the-10-best-data-visualizations-of-2021-fec4c5cf6cdb) [## 2021 年 10 大最佳数据可视化

### 关于财富分配、环境、新冠肺炎等等的令人敬畏的可视化！

towardsdatascience.com](/the-10-best-data-visualizations-of-2021-fec4c5cf6cdb) 

# -特伦斯·申

*   ***如果您喜欢这个，*** [***订阅我的媒介***](https://terenceshin.medium.com/membership) ***获取独家内容！***
*   ***同样，你也可以*** [***关注我上媒***](https://medium.com/@terenceshin)
*   [***报名我的个人简讯***](https://terenceshin.substack.com/embed)
*   ***跟我上***[***LinkedIn***](https://www.linkedin.com/in/terenceshin/)***其他内容***