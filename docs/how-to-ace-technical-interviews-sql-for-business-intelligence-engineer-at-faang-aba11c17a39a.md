# FAANG 商业智能工程师如何通过技术面试(SQL)

> 原文：<https://towardsdatascience.com/how-to-ace-technical-interviews-sql-for-business-intelligence-engineer-at-faang-aba11c17a39a>

所有数据职业的一个共同技能是 SQL。学习它永远不会出错。无论是处理结构化数据还是非结构化数据，在这个过程中的某个时刻都需要 SQL。

![](img/42d8cefbec7a11ed0e7bab1967cc1111.png)

由[马库斯·斯皮斯克](https://unsplash.com/@markusspiske?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄的照片

在我最近在 FAANG 的面试中，SQL 是最先测试的技能之一。您可以在本文中了解整个过程。

[](/how-i-cracked-business-intelligence-engineer-interviews-at-faang-d08f64f7e748) [## 我是如何在 FAANG 破解商业智能工程师面试的

### 我最近面试了亚马逊的商业智能工程师，并成功通过了面试

towardsdatascience.com](/how-i-cracked-business-intelligence-engineer-interviews-at-faang-d08f64f7e748) 

如果你要参加 FAANG(或任何公司)的数据分析师面试，以下是你需要准备的五大 SQL 概念。

**1。常用表表达式—** 常用表表达式也称为 cte，是 SQL 中最被低估的概念之一。加入很重要，但是 cte 可以让你和你的继任者的生活变得非常容易。CTE 基本上是一个临时命名的数据集，可在进一步的 CTE/选择/插入/更新语句中使用。
它的引入是为了简化 SQL 查询。您可以将它们视为查询中的一个变量。你用`With <CTE Name> as(Select …)`来声明它

然后，可以在后续 SQL 代码中的任何地方调用这个 CTE，就像使用预声明的变量一样。要使用 CTE，您可以使用`join`、`on`关键字将它与其他表/视图/cte 连接起来。

> *请注意，cte 不仅仅是为了面试。它们也在我的日常工作中广泛使用。*

**2。row _ number()—** row _ number 函数与 rank()和 dense_rank()非常相似。基本的区别在于处理分区中有平局的情况。rank 函数将相同的序列号分配给所有的连接，并跳过下一个序列号。Dense_rank()复制所有绑定值的序列号，而不跳过序列中的下一个数字。而 Row_number()只是为所有绑定的行任意分配一个递增的序列号。对于像查找薪水最高的前 10 名员工这样的问题，row_number 非常有用。

**3。自我联接—** 在我参加过的所有 SQL 面试中，一个常见的问题是基于自我联接的。自连接最广泛地用于查找父子链接。我个人在工作中多次使用这个概念来显示客户经理及其主管、直接隶属于某个集团经理的所有员工等等。

**4。Case when 语句—** `Case When`语句主要用于根据不同的标准对一列进行分类/宁滨。比如，如果你想根据学生的考试成绩来显示他们的分组，你可以写一个这样的 case 语句

```
case
    when marks >= 60 then 'First'
    when marks >= 45 then 'Second'
    when marks >= 33 then 'Third'
    else 'Fail' 
end as Division
```

这使得 case 语句非常有用。它通常也用于创建不同类型的标志。

**5。窗口函数/聚合—** sum、min、max 和 average 等聚合是分析师工具箱中不可或缺的一部分。窗口功能可以显示每一行的计算结果。例如，显示累积和，显示前一个或下一个值。这些只是使用窗口函数可以做的几件事。我强烈建议你很好地理解这个概念。这将为您节省一些时间和冗长的代码。

要了解更多 SQL 概念，请参考这篇关于日期函数的文章。

[](/how-to-handle-dates-in-sql-using-in-built-functions-d6ca5a345e6d) [## 如何使用内置函数在 SQL 中处理日期

### 日期函数是 SQL 编程中谈论最多的领域之一

towardsdatascience.com](/how-to-handle-dates-in-sql-using-in-built-functions-d6ca5a345e6d) 

在我看来，你可以用这五个概念和内置的日期函数来清除所有的 SQL 访问。就 SQL 技能而言，你将是一个强有力的候选人。

我祝你旅途顺利。

感谢阅读！

*下次见……*

*附注——你们中的很多人都在 Linkedin 上寻找实际的问题。我在 NDA，不能分享实际的问题文件。我也不能单独回答你们所有人的问题。但是我保证上面阐述的概念涵盖了所有被问到的问题。*