# 3 次 SQL 交换以编写更好的代码

> 原文：<https://towardsdatascience.com/3-sql-swaps-to-write-better-code-f8d304699cde>

# 3 次 SQL 交换以编写更好的代码

## 如何确保您的代码是准确的、可读的和高效的

![](img/a62bc11a82cd4cabddab926c4756d6d1.png)

照片由[丹尼斯·莱昂](https://unsplash.com/@denisseleon?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/light-switch?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

无论你是数据分析师、数据工程师、数据科学家还是分析工程师，你都必须编写干净的代码。你不应该是唯一一个能理解你写的东西的人。它需要让团队中的其他人能够阅读，以便在未来几年中使用和改进。

我最大的烦恼之一是当我发现草率的代码，然后写它的人已经不在公司了！没有办法知道他们在想什么，为什么他们以某种方式写逻辑。在我目前的角色中，这是一个持续的斗争，我正在重写我们所有的核心数据模型。

信不信由你，写 SQL 代码有好有坏。仅仅因为它运行并得到您想要的结果，并不意味着它是编写良好的代码。要被认为是好的代码，它必须是准确的、可读的和高效的。

这里有一些你可以在代码中进行的 SQL 交换，以确保它更具可读性和效率，这是编写良好代码的两个核心要素。

# TOP →行号()

在重写旧的数据模型时，我经常遇到使用函数`TOP`的子查询。虽然这可能会让你得到你正在寻找的解决方案，但它很难阅读，并且可能需要很长时间来运行。

以下是我发现的一个例子:

```
SELECT
   customer_id,
   (SELECT
       TOP order_id 
    FROM orders
    WHERE orders.customer_id = customers.customer_id
    ORDER BY date DESC)
FROM customers 
```

作为工程师和分析师，我们希望确保我们正在编写尽可能干净和高效的代码。所以，如果你使用的是`TOP`，这里有一个函数可以代替。

`ROW_NUMBER()`是一个窗口功能，允许您为满足特定条件的每一行分配一个序号。使用`ORDER BY`和`PARTITION BY`指定这些条件。

`ORDER BY`指定您希望如何对表格中的值进行排序。日期通常用于此。

`PARTITION BY`指定作为值分组依据的列。这取决于您查询的目的。

在这个例子中，我们需要`ORDER BY` date 和`PARTITION BY` customer_id。因为我们试图为每个客户找到最近的订单，所以按客户 id 进行分区将为每个唯一的 id 计算一个新的序列。

```
SELECT 
   customer_id, 
   order_id,
   ROW_NUMBER() OVER(PARTITION BY customer_id ORDER BY date) AS order_sequence_number 
FROM orders
```

现在，这将产生一个与原始表具有相同行数的表，但是，这一次有一个新的列`order_sequence_number`来指示每个客户的订单顺序。为了获得客户的第一个订单，我们需要编写另一个查询，只选择等于 1 的序列号。

```
WITHorder_row_number_calculated AS (
   SELECT 
      customer_id, 
      order_id,
      ROW_NUMBER() OVER(PARTITION BY customer_id ORDER BY date) AS order_sequence_number 
   FROM orders
)SELECT 
   customer_id,
   order_id 
FROM order_row_number_calculated
WHERE order_sequence_number = 1
```

现在我们有了一个 CTE，它在第一个查询中查找每一行的行号，然后通过只查找分组中第一个行号来过滤查询。这对任何人来说都更容易阅读，并且通常运行速度更快。

# 子查询→cte

上面显示的最后一个例子也可以用于这种交换！如果你看上面，我们从查询中的子查询开始，然后使用 cte 结束。

CTE 到底是什么？CTE 代表普通餐桌用语。它创建了一组临时结果，您可以在后续查询中使用。它要求您以`WITH`开始，并在每个 CTE 前使用 `table_name AS`。序列中的最后一个 CTE 应该是一个简单的 SELECT 语句，既没有 table_name 也没有`AS`。

它应该是这样的:

```
WITH 
money_summed AS (
   SELECT 
      customer_id,
      date,
      SUM(balance) AS balance,
      SUM(debt) AS debt
   FROM bank_balance 
   GROUP BY customer_id, date
),money_available_calculated AS (
   SELECT 
      customer_id,
      date,
      (balance - debt) AS money_available 
   FROM money_summed 
   GROUP BY customer_id, date
)SELECT
   customer_id,
   money_available 
FROM money_available_calculated 
WHERE date = '2022-01-01' 
```

注意，在最后一个`SELECT`语句之前，我们没有使用表名或括号。前一组括号后面也没有逗号。一系列 cte 中的最后一个查询总是作为普通查询写入。请务必记住这一点，因为在使用 cte 时，由于格式化导致的错误是常见的！

# ≤和≥ →之间

使用日期列将两个表连接在一起是很常见的。大多数时候，当我看到这个的时候，他们是通过使用`≥`和`≤`来连接的。虽然这种方法有效，但看起来很混乱。幸运的是，有一个更好的方法来做到这一点！

`BETWEEN`是一个 SQL 函数，允许您选择两个给定值之间的范围。它返回在你给它的值之间的值*。它可以与数字、日期甚至字符串一起使用。*

但是，最需要注意的是，这个功能是*包含*的。这意味着指定的值也将包含在您的结果中。所以，它只能真正取代`≤`或`≥`而不是`<`或`>`。

首先，让我们看一个查询，它使用`≤`和`≥`符号连接两个使用日期的表。

```
SELECT
   customers.customer_id,
   customers.activated_at, 
   campaigns.campaign_id
FROM customers
LEFT JOIN campaigns 
ON customers.campaign_id = campaigns.campaign_id 
WHERE customers.activated_at >= campaigns.started_at AND customers.activated_at <= campaigns.ended_at
```

在`WHERE`子句中，我们需要使用`activated_at`指定两个不同的语句。一个是该列在`≥ started_at`处，另一个是`≤ ended_at`处。

现在，让我们使用`BETWEEN`。

```
SELECT
   customers.customer_id,
   customers.activated_at, 
   campaigns.campaign_id
FROM customers
LEFT JOIN campaigns 
ON customers.campaign_id = campaigns.campaign_id 
WHERE customers.activated_at BETWEEN campaigns.started_at AND campaigns.ended_at
```

这里，我们只需要指定`activated_at`一次，而不是两次。

哪个查询更容易阅读和理解？虽然第一个可能不太复杂，但是代码中的小差异会产生巨大的差异。这就是好的和伟大的*的区别。*

# 结论

刚开始学习 SQL 的时候，你希望专注于能够解决给你的问题，并得到正确的答案。随着你发展自己的技能，变得更加自信，你需要努力改进*如何*找到解决方案。

每个工程师和分析师都希望他们的队友能够阅读他们的代码，并且容易理解。你不仅会讨厌回答无数的问题，而且他们也会讨厌问你这些问题。努力使您的 SQL 代码高效易读。如果有一个功能可以作为快捷方式，那就使用它！

另外，不要忘记文档的重要性。给你的代码加注释可以让其他人的工作变得容易很多，也省去了提问的必要。如果你认为你写的东西可能很难理解，解释它的意思！我坚信代码永远不会有太多的注释。

有关提高 SQL 技能的更多提示，请查看您需要了解的 [8 个 SQL 日期函数](/8-sql-date-functions-you-need-to-know-c6c887a8394f)、[如何使用 SQL 交叉连接](/how-to-use-sql-cross-joins-5653fe7d353)，以及[如何使用 SQL 窗口函数](/how-to-use-sql-window-functions-5d297d29f810?source=your_stories_page----------------------------------------)。