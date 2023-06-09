# 编写良好的数据模型的 3 个关键组成部分

> 原文：<https://towardsdatascience.com/3-key-components-of-a-well-written-data-model-c426b1c1a293>

## 如何编写通过时间考验的高性能模型

![](img/84e69e3b7c8f087d5d1dc2f2fbb04113.png)

杰克·亨特在 [Unsplash](https://unsplash.com/s/photos/three?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

当我刚开始作为分析工程师的职业生涯时，我的任务是重写我们的商业智能工具中的数据模型。当我开始通读它们时，我震惊地发现它们是不可能理解的，非常有条理，并且使用了糟糕的代码。难怪他们中的一些人要花一天时间来跑步！第一次就构建好数据模型是很重要的，这样你就不必经历重写数据模型的过程。

当构建高质量的数据模型时，您需要记住**模块化**、**可读性**和**速度**。在优先考虑这三个品质的同时编写模型将有助于您尽可能编写最好的数据模型。我们的目标是让它们永远存在，或者只要业务逻辑保持不变，那么您就可以继续关注公司的其他领域，以提高数据的利用率。

# 模块性

当您的模型是模块化的时，它可以作为单独的代码段存在，可以在其他地方重用和引用。这意味着您必须编写更少的可重复代码，从而节省您的计算能力。

编写模块化模型时，你要遵循**干代码**的概念。这意味着您编写尽可能少的代码，同时仍然编写执行您需要的功能的代码。不要重复代码！你的函数简洁、清晰、易读。干代码和模块化数据模型齐头并进。当编写模块化模型时，您需要问自己三个不同的问题:

**什么代码在多个型号中重复？**

在重写数据模型之前，做一个完整的清单是很有帮助的——或者，如果从头开始，考虑一下重复出现的元素。这将帮助您确定将哪些代码转换成它们自己的数据模型。例如，如果您将用户映射到某个匿名浏览器 cookie，并在您的所有模型中使用这种映射，那么这应该作为一个模型存在，可以在其他模型中重复。

一旦你明白了这一点，你需要问自己， ***如何*将这些重复的代码拆分成它自己的模型，以便在多个模型中重用？**

您需要弄清楚一段代码如何能通用于多个用例。这可能意味着您需要删除特定于一个模型的任何筛选器，或者您需要包含更多在某些(但不是全部)模型中使用的列。

例如，如果订阅的数据模型只过滤用户映射代码中的活动用户，则需要从正在模块化的用户映射模型中删除它。然后，当您在订阅模型中直接引用用户映射模型时，您可以为活动用户添加过滤器。这样，一个同样使用“用户映射模型”，但所有类型的用户都需要它的营销模型，仍然可以引用模块化数据模型。

最后，您需要问问自己，这种模块化模型是否真的可以达到与已经直接构建到模型中的代码相同的目的。从一个更大的模型模块中生成一段代码会改变模型的结果吗？

目标是创建不影响代码输出的模块化数据模型。无论代码是如何分离的，结果数据集应该总是相同的。如果这改变了逻辑，也许它不应该成为自己的数据模型。您不需要强迫每一段代码都成为它自己的模型，只需要在您的所有业务逻辑中真正可重复的代码。

让我们看两个不同的数据模型。第一个映射用户第一次访问网站时的所有个人信息。

```
WITH users_mapped_to_cookies AS (
   SELECT
    sessions.User_id,
    browser_cookies.Cookie_id,
    Browser_cookies.first_url 
   FROM browser_cookies
   LEFT JOIN sessions 
   WHERE sessions.session_started_at <= browser_cookies.event_timestamp <=sessions.session_ended_at
),

mapped_users_joined_user_details AS (
 SELECT 
  Users_mapped_to_cookies.user_id, 
  Users_mapped_to_cookies.first_url, 
  Users.name,
  Users.email,
  Users.phone 
 FROM users_mapped_to_cookies 
 LEFT JOIN users 
 ON users_mapped_to_cookies.user_id = users.user_id 
)

SELECT * FROM mapped_users_joined_user_details
```

第二个数据模型将用户的首页访问映射到他们的订单。

```
WITH users_mapped_to_cookies AS (
 SELECT
  sessions.User_id,
  browser_cookies.Cookie_id,
  Browser_cookies.first_url 
 FROM browser_cookies
 LEFT JOIN sessions 
 WHERE sessions.session_started_at <= browser_cookies.event_timestamp <=sessions.session_ended_at
),

mapped_users_joined_orders AS (
 SELECT 
  Users_mapped_to_cookies.user_id, 
  Users_mapped_to_cookies.first_url, 
  Users.name,
  Users.email,
  Users.phone 
 FROM users_mapped_to_cookies 
 LEFT JOIN orders
 ON users_mapped_to_cookies.user_id = orders.user_id 
)

SELECT * FROM mapped_users_joined_orders
```

我们将把代码抽出来创建一个单独的模型，而不是在两个数据模型中都有这些映射子查询。然后，我们将在另一个模型的代码中引用这个数据模型，就像这样:

```
SELECT
  Users_mapped_to_cookies.user_id,
  Users_mapped_to_cookies.first_url,
  Users.name,
  Users.email,
  Users.phone
FROM users_mapped_to_cookies
LEFT JOIN users
  ON users_mapped_to_cookies.user_id = users.user_id
```

而且，别忘了，`the users_mapped_to_cookies`代码是作为自己的数据模型存在的！

# 可读性

如果代码是真正可读的，那么除了代码本身之外，其他人应该能够在没有任何资源的情况下阅读它并准确理解它做了什么。如果你的同事不得不不断地问你关于你写的代码的问题，你可以写得更简洁。如果你发现你的代码不可读，你很可能有一个很好的机会来优化它的性能和成本。

通过利用数据目录、描述、注释和沿袭等工具，代码也可以变得对非技术用户更具可读性。这些都有助于用户理解数据模型的完整上下文。

在为技术用户编写可读代码时，有三件事要记住:

## **总是评论那些不能直观理解的代码。**

如果读者需要更多的知识来理解你的代码在做什么，请在你的代码中注明！经常有大量的研究进入理解业务逻辑和如何正确地编码模型。确保您在代码的注释中捕捉到了这些部落知识。这样，你会记得*为什么*你做了某事，这将有助于向其他人解释你为什么这样写代码。

我个人喜欢评论我过滤掉的值的含义，为什么我使用了某个连接，甚至从模型的更大意义上来说查询在做什么。当有人审查代码时，这些都是很难理解的事情。通过添加这些注释，您还可以确保您的模型不依赖于构建它们的人的知识。当编写原始代码的人离开公司，你不能再问他们问题时，这特别有帮助。

## **使用 cte 代替子查询。**

子查询因使代码难以阅读而臭名昭著。在更大的模型中很难理解它们的上下文，因为它们使代码变得混乱，难以理解。通过使用 cte，您可以将代码分解成更小的步骤，这些步骤产生自己的输出。

较小的代码块使得在需要时更容易调试。调试子查询的唯一方法是将其转换为 CTE，或者将其完全从模型中取出。如果一段代码不能在不做修改的情况下自行调试，那么一开始就不应该使用它。

人们通常选择 cte 上的子查询，因为他们认为它们更复杂。事实并非如此。cte 和子查询有相似的运行时，但是子查询会无缘无故地使您的代码更加复杂。可读的代码总是优于不必要的复杂性。

## 使用描述性的名称。

最后，为了让您的模型可读，您需要在 cte 中为表、列和别名使用描述性的名称。那些审查你的代码的人应该确切地理解他们的意思，而不必搜索你的代码的其余部分或询问你。这将使您的模型更容易调试和理解。命名越具体，将来使用数据模型就越容易。

例如，如果您要在一个 CTE 中连接表`users`和`addresses`，您可能希望将其命名为`users_joined_addresses`而不是`user_addresses`。这告诉阅读您的代码的人，您正在连接两个表。`User_addresses` 告诉您用户和地址表正在被使用，但不告诉您它们是如何被使用的。

现在让我们看一个写得很好的数据模型的例子，由于别名、列名和代码注释，它非常易读。

```
WITH
Active_users AS (
  SELECT
    Name AS user_name,
    Email AS user_email,
    Phone AS user_phone,
    Subscription_id
  FROM users
  --- status of 1 means a subscription is active
  WHERE subscription_status = 1
),
Active_users_joined_subscriptions AS (
  SELECT
    Active_users.user_name,
    active_users.user_email,
    Subscriptions.subscription_id,
    subscriptions.start_date ,
    subscriptions.subscription_length
  FROM active_users
  LEFT JOIN subscriptions
    ON active_users.subscription_id = subscriptions.subscription_id
)
SELECT * FROM Active_users_joined_subscriptions
```

你可以阅读这个模型，因为它有明确的命名，所以你可以准确地理解它的作用。当命名不太清楚时，注释会准确地告诉审阅者 1 的状态是指什么。

# 速度

编写数据模型的主要目的之一是加速数据集向数据仓库的交付。通过自动化数据模型来生成可供业务使用的数据集，可以使您的数据在需要时更加可靠和可用。

如果数据模型很慢，需要几个小时才能运行，那么它们就没什么用了。这经常会在业务团队中造成瓶颈。如果业务团队试图根据由您的某个数据模型驱动的仪表板做出决策，他们的速度将会变慢，但这需要一整天的时间来运行。或者更糟的是，他们根本无法利用数据来做决定。

加快数据模型的速度可以简单到在模型开始时删除重复项或过滤掉空值。也可以更复杂，比如用窗口函数代替复杂代码。正如我在开始时提到的，将您的模型分解成更小的、模块化的模型也有助于这一点。

当我重写长时间运行的数据模型时，我看到两个函数被大量使用。首先，我看到了`TOP`函数与`GROUP BY`一起使用。这意味着代码必须对所有值进行分组、排序，然后选择每个有序组中的第一个值。这浪费了大量的计算能力。

相反，你可以使用`FIRST_VALUE()`窗口功能。这允许你`PARTITION`你的值，而不是使用`GROUP BY`，然后在每个组内排序。窗口功能选择第一个值要比顶部功能快得多。

以下函数可用于查找学生的最高考试分数:

`FIRST_VALUE(test_score) OVER(PARTITION BY student_name ORDER BY test_score DESC)`

我也看到有人使用子查询来帮助他们计算一个值的总和或平均值。正如我前面提到的，您总是希望使用 cte 而不是子查询，因为子查询会降低代码的速度。在这种情况下，您可以使用聚合窗口函数来替换子查询中使用的`SUM`或`AVERAGE`。只需在`PARTITION BY`后指定您希望分组的列，函数将计算每个组的聚合。

以下函数可用于计算每个学生的平均考试分数:

`AVG(test_score) OVER(PARTITION BY student_name)`

在这里，您不必包括`ORDER BY`，因为在查找总和或平均值时，顺序并不重要。

# 编写持久有效的数据模型

编写良好的数据模型有能力改变您的业务运营方式。它们将允许您的数据团队的工作随着业务的增长而增长，而不是随着业务的增长而被取代。直接在[**【Y42】**](https://www.y42.com/product/sql-model?utm_source=medium_madison&utm_medium=blog&utm_campaign=components_data_model)等平台中编写的 SQL 模型通过目录和沿袭特性得到了广泛的记录，使它们易于随业务一起扩展。当你用模块化、可读性和速度来构建你的模型时，它们将变得永恒和无价。它们不必每隔几个月就被替换，因为它们使得创建仪表板、报告和调试问题变得非常困难。

虽然牢记这些要点来编写未来的数据模型是很重要的，但是您也希望重新评估您当前已有的模型。他们缺少这些方面吗？你能抽出一些代码来创建一个可以在多个地方使用的模块化数据模型吗？需要给代码添加注释吗？这些都是你在回顾过去写过的东西时可以问自己的问题。现在关注这一点将有助于防止未来的技术债务，并优先考虑健康的数据生态系统。

欲了解更多关于分析工程、现代数据堆栈和 dbt 的信息，[订阅我的免费每周简讯](https://madisonmae.substack.com/)。

查看我的第一本电子书[《分析工程基础知识](https://madisonmae.gumroad.com/l/learnanalyticsengineering)，一本全方位的分析工程入门指南。