# 利用查询重构和基于函数的索引来提高性能

> 原文：<https://towardsdatascience.com/leveraging-query-restructuring-and-function-based-indexes-to-improve-performance-c50d50246964>

## 我们优化查询运行时间的旅程

![](img/dd3243e30f92813727a744f2309c08c7.png)

蒂姆·莫斯霍尔德在 [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

检查在线创建的无限数量的数字数据，即博客、网络研讨会等，需要一个[布鲁姆过滤器数据结构](https://medium.com/@sofi.vasserman/blooming-design-bloom-filter-and-its-complement-82b97056d6b0)来优化我们一些验证的运行时间，并最终在 [Brew](http://www.getbrew.com) 为我们的客户创造价值。

除了使用布隆过滤器，为了确保我们能够有效地验证数据，同时不断提高数据库中的搜索性能，我们需要将查询重构和基于函数的索引集成到我们的算法中。

注意:因为我们使用 Django 和 PostgreSQL，所以本博客中的所有代码示例都将使用 Django ORM 来转换原始 SQL 查询，但是我将讨论的原则和要点适用于任何使用 ORM 和关系数据库的开发人员。

Brew 平台处理由无数营销活动产生的无限量数据，每个营销活动都有一个唯一的 URL。
每个 URL 可以有多个合法版本，例如使用 HTTP 或 HTTPS 方案的 URL、以“www”开头的 URL 或不以“www”开头的 URL、路径中带有语言前缀的 URL(例如“http://example.com/en/page”)、以“html”后缀结尾的 URL 等。

由于我们可以接收这些变化中的任何一个作为潜在的新活动的链接，并且由于我们想要检查这是否是一个新活动，我们必须使用这些版本中的一个在我们的数据库中检查这个 URL 是否存在。为此，我们构建了一个包含所有常见变体的正则表达式的查询，以支持这些搜索。特别是在 Django 的 ORM 中，用正则表达式查询很简单，因为它有一个支持正则表达式的本地字段查找操作符。例如:

```
Entity.objects.filter(url__iregex="http(s)?:\/\/(www\.)?example\.com")
```

上面的查询获取所有具有 HTTP 或 HTTPS 模式的 URL，可以以“www”开头或者没有它。使用 *iregex* 操作符而不是 *regex* 使得搜索不区分大小写。

## **放大问题**

在查询中使用正则表达式在一段时间内效果很好，但是随着数据库的增长，我们注意到在我们的部分过程中有些缓慢。我们开始调查为什么会出现这种速度缓慢的情况，以及这些时候我们的服务器上发生了什么。我们注意到，在我们的一些搜索中，数据库的 CPU 使用率相当高。这令人惊讶，因为为了优化搜索，我们确保查询中经常使用的列都有索引。然而，在运行特定查询时，数据库中的高 CPU 意味着该查询中没有使用索引。为了验证这个假设，我们检查了可疑查询的执行计划。

可以使用 PostgreSQL 中的以下命令来检查执行计划:

```
EXPLAIN ANALYZE SELECT * FROM entity_table WHERE link = 'https://www.example.com'
```

它也可以通过类似 pgAdmin 的数据库管理工具来可视化。

我们正在检查的查询的可视化解释计划如下所示:

![](img/b417b60cff13e1017ebcadc86b03c020.png)

作者图片

用于 entity_table 的图标表示全表扫描，这意味着不使用索引。全表扫描是一个 O(N)操作，当试图在一个有数百万行的表中搜索一个数据点时，它可能是一个非常昂贵的操作。

那么，为什么这些查询中没有使用我们的索引呢？

当在 Django ORM 中使用 regex 的本地字段查找时，ORM 将这个操作符翻译成数据库中的 regex 操作符。因此，如果我们运行以下搜索:

```
Entity.objects.filter(
    url__iregex="http(s)?:\/\/(www\.)?example\.com"
)
```

在 PostgreSQL 中，它将转换为:

```
SELECT * 
FROM entity_table 
WHERE url ~* ‘http(s)?:\/\/(www\.)?example\.com’
```

数据库中的正则表达式运算符使查询评估每行中的字符串，以查看它是否与正则表达式匹配。该运算符不能使用常规 B 树索引(为该列构建的索引)，因为 B 树索引用于相等和简单比较，而在正则表达式中并非如此。

## 解决问题

在 PostgreSQL 的早期版本(9.4 之前)中，没有可用于正则表达式查询的索引。因此，有必要更改查询，以确保使用了索引并且优化了查询的运行时间。

为此，我们必须停止使用正则表达式的操作符，因为它阻止了 b 树索引的使用。然而，我们仍然想搜索所有的 URL 变体。为此，我们必须将查询拆分成多个条件，而不是将一个条件与一个正则表达式一起使用。我们之前的查询变成了这个查询:

```
Entity.objects.filter(
    Q(url__startswith="http://www.example.com") |
    Q(url__startswith="https://www.example.com") |
    Q(url__startswith="http://example.com") |
    Q(url__startswith="https://example.com")
)
```

ORM 中的这个查询转换成数据库中的以下查询:

```
SELECT * 
FROM entity_table 
WHERE url LIKE ‘http://www.example.com%' OR
      url LIKE 'https://www.example.com%' OR
      url LIKE 'http://example.com%' OR
      url LIKE 'https://example.com%'
```

这个查询现在使用索引，尽管我们有多个条件，而不是一个，但它比前面的查询优化得多，因为索引正在被使用。

*注意*:即使我们没有使用等式操作符，而是使用了`LIKE`操作符，它只匹配字符串的一部分，但是仍然可以使用索引，因为我们锚定了字符串的开头，在搜索索引时，它可以用作等式匹配。如果我们没有固定字符串的开头，索引就不能使用。
例如，下面的查询:

```
SELECT * 
FROM entity_table 
WHERE url LIKE ‘%://www.example.com%'
```

该查询必须运行全表扫描，因为字符串的开头是通配符，并且当字符串的开头不是特定字符串时，没有优化的方法在 b 树索引中搜索该字符串。

现在，我们的查询得到了优化，使用了 DB 索引，运行时间也大大减少了。然而，这个查询是不完整的，因为它遗漏了我们以前拥有的一个功能。在前面的查询中，我们使用了运算符`iregex` ，它被转换为不区分大小写的正则表达式。在新的查询中，我们使用操作符`startswith`，它区分大小写，并不完全匹配我们的用例。为了解决这个问题，我们有两个选择——一个是使用 ORM 的`LOWER`或`UPPER`函数操作符，另一个是使用`istartswith`操作符，它相当于`startswith`操作符，不区分大小写。吸取教训，我们不想在没有确保不会降低查询运行时间的情况下使用操作符。查看 ORM 中这个查询到数据库查询的转换，我们看到了下面的代码:

```
Entity.objects.filter(url__istartswith="http://example.com")
```

转换为查询:

```
SELECT *
FROM entity_table
WHERE UPPER(url) = UPPER("http://example.com")
```

*注意:*在这种情况下，两个可选解决方案之间没有区别，因为两者都使用了 LOWER 或 UPPER 数据库函数来使查询不区分大小写。

使用这些函数会引入以前遇到的相同问题，因为在我们转换要搜索的列时，不会使用常规的 b 树索引。

幸运的是，有一个简单的解决方案，那就是基于函数的索引。

基于函数的索引是根据函数或表达式的结果创建的索引，而不是根据此列中的原始值创建的索引。在我们的示例中，我们可以创建一个基于函数的索引，该索引基于数据库中“ *url* ”列中每个值的`UPPER`函数的结果。每当我们在查询中的“ *url* ”列上使用该函数时，都会使用该索引，这允许我们优化查询，即使不使用该列中的原始值。
大多数(如果不是全部的话)关系数据库都支持基于函数的索引，并且它们可以用于对表的常见查询在列上使用函数或某种简单表达式的任何场景，例如在浮点列中取整值、仅提取日期的月份部分、将日期截断到一天的开始或者甚至两列之和。这些示例中的每一个都将取消常规 b 树索引的使用，并可能导致查询运行时性能下降。因此，如果这些是您的表上的常见查询，您应该考虑对正在使用的常见表达式/函数使用基于函数的索引，这可以显著优化查询运行时间。

## 关键要点

这种由于错误使用 ORM 操作符而导致的运行时性能下降的经历向我们重申了了解我们的 ORM 以及它如何将我们的代码翻译成查询是多么重要。ORMs 是一个很好的工具，它增加了一个抽象层次，允许我们在与数据库交互时继续使用面向对象的范例。然而，我们总是需要记住这些抽象是有代价的。简单的查询在 ORM 中总是很好用，但是更复杂的查询需要更好地理解 ORM 如何翻译我们的代码。了解 ORM 是如何工作的可以帮助我们避免主要的陷阱，就像我们遇到的那样。

从我们的经验中可以学到的另一个教训是，了解数据库如何执行我们的查询是多么重要。了解这些知识以及优化查询的各种解决方案可以帮助我们在构建模型时提前计划，并提前防止优化问题，或者在遇到优化问题时轻松解决它们。

Django 开发者侧记。

对于 Django 开发人员来说，现在向模型中添加基于函数的索引是相对简单的，从 3.2 版本开始，可以在模型规范中声明它。例如:

```
class Entity(models.Model):
    url = models.URLField(max_length=2100)
    name = models.TextField()

    class Meta:
        indexes = [
            models.Index(Upper("url"))
       ]
```