# 节省使用 UDF 编写 SQL 的时间

> 原文：<https://towardsdatascience.com/save-time-writing-sql-with-udfs-24b002bf0192>

## SQL UDFs 介绍及示例

编写 SQL 是许多数据科学家和数据分析师角色的重要组成部分，但这通常是花费时间最不愉快的方式。获取运行分析所需的数据可能会变得繁琐而耗时。用户定义函数(UDF)是解决这个问题的好方法，它是查询编写过程的捷径。在本文中，我们将介绍什么是 UDF，为什么您可能会使用它们，它们在哪里被支持，以及一些基本的例子。

![](img/14f0744a68b0c2065c5cd8ca814c6f4e.png)

约翰·施诺布里奇在 [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

# 什么是 UDF？

UDF 是一个用户定义的函数，它在基本层面上非常类似于一个典型的 SQL 函数，比如 RIGHT()，但是它不是由一些标准定义的，任何人都可以定义它们。更专业地说，UDF 是一个函数，它接受一组类型化的参数，应用一些逻辑，然后返回一个类型化的值。下面是一个例子。

```
CREATE FUNCTION Sigmoid(@exponent FLOAT(10))
RETURNS FLOAT(10)
AS
   BEGIN
      1.0 / (1.0 + EXP(-@exponent))
      RETURN Data
   END
```

UDF 在 SQL 世界中广泛存在，它能够创建存在于数据库管理系统(如 SQL Server、Postgres、BigQuery 和 Presto)中的 UDF。注意一些数据库管理员会禁用这些，所以最好检查一下。

# UDF 的利与弊

使用 UDF 可能是有益的，但是它们也有缺点。我个人认为，拥有一个好的 UDF 集合，特别是在小的产品团队中，是非常有用的，但是我将分享其利弊，让您来决定。

**优点**:

*   **允许您用简单的一行程序替换部分复杂或重复的 SQL，使代码更具可读性**。例如，您可能有一个复杂的 CASE 语句，它占用了查询的许多行。有了 UDF，这可以减少到一行。
*   **鼓励集中定义流程**。想要对事物进行分类是很常见的，能够在一个人编写的查询中以及在多人之间保持一致是很有价值的。
*   **由于上述原因，它允许你更快地编码**。能够更快地准备好数据总是会让数据科学家感到高兴。

**缺点**:

*   虽然它可以使代码更具可读性，但是太多不清楚的 UDF 实际上会降低代码的可读性。如果函数被命名为类似于“func1”的名称，那么它们实际上在做什么就很不清楚了，这就需要读者查找函数定义，这通常比在代码中通读要花费更长的时间。
*   你需要记录你在记忆中创建的所有 UDFs】，这在你建立了一个像样的剧目之后会变得很困难。最坏的情况是，你开始创建更多的 UDF 做类似的事情，创建不必要的代码。为了避免这导致问题，建议保留一个包含您创建的 UDF 的字典，并确保与任何合作者共享它。您还可以确保尽可能创建更多的通用函数，以避免重复工作。
*   **UDF 可能运行效率低下，尤其是在大块数据上**。对于各种 SQL 函数，它们必须逐行计算，导致运行时间很长。这可以通过一些技术得到缓解，比如针对 [SQL Server](https://www.mssqltips.com/sqlservertip/5864/four-ways-to-improve-scalar-function-performance-in-sql-server/) 的技术。

如你所见，使用 UDF 有好的一面，也有不好的一面，当应用 UDF 时，确保使用常识，可以将这些缺点最小化。

# UDF 的例子

下面是一些有用的 UDF(小写输入参数)的例子。这些可以大致分为替代重复动作的和编码业务逻辑的。

**重复动作**:

*   **舍入器**:将一个数字舍入到最接近的指定因子，并带有下限和上限。

```
LEAST(GREATEST(ROUND(@number / @factor) * @factor, @lower), @upper)
```

*   **Sigmoid**:Sigmoid 函数[的 SQL 定义](https://www.notion.so/SQL-UDFs-95b6bb9567c04369992ef90391f2acc7)。

```
1.0 / (1.0 + EXP(-@exponent))
```

*   **采样**:给定一些值的数组，采样给定的数量。

```
SLICE(SHUFFLE(@input), 1, @value)
```

*   **X 中的 1**:在 X 语句中将一个数字从十进制转换为 1，例如 0.01 或 1%是 100 分之一。

```
‘1 in ‘ || CAST(CAST(POWER(10,CEIL(-LOG10(@percent))) AS INT) AS VARCHAR)
```

**业务逻辑**:

*   **年龄分类**:根据用户的年龄对他们进行分类。

```
CASE WHEN @age < 12 THEN ‘child’ WHEN @age < 20 THEN ‘teenager’ WHEN @age < 50 ‘adult’ ELSE ‘elderly’ END
```

*   **购买时段**:将用户购买的数量分组到特定的时段中(用于绘图)。

```
CASE WHEN @purchases = 0 THEN ‘0’ WHEN @purchases = 1 THEN ‘1’ WHEN @purchases < 5 THEN ‘2–4’ WHEN @purchases ≥ 5 THEN ‘5+’
```

*   **邮件域**:从邮件字符串中提取邮件域。

```
SPLIT_PART(@email, ‘@’, 2)
```

这些只是你如何应用 UDF 的一些例子；你可以用更多的方式来使用它们。唯一真正的限制是创造力！

# 结论

正如我们所看到的，UDF 有广泛的应用，包括优化重复代码和集中业务逻辑，这有助于您更有效地编写 SQL。虽然有缺点，但是可以用一点常识来适当地减轻它们，以确保您获得使用 UDF 的所有好处。鉴于它们在各种 SQL 系统中的广泛可用性，没有理由不给它们一个机会！