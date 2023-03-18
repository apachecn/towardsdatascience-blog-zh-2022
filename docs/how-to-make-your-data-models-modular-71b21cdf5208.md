# 如何使您的数据模型模块化

> 原文：<https://towardsdatascience.com/how-to-make-your-data-models-modular-71b21cdf5208>

## 通过这些步骤避免高度耦合的系统和意外的生产错误

![](img/f99214efb05e464aa4943ce01ab938b9.png)

照片由 [T.J. Breshears](https://unsplash.com/@tjbreshears?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/puzzle-piece?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄

发现生产中的某些东西坏了，这是每个工程师最可怕的噩梦。更糟糕的是，当一个简单的变化打破了一切。当这种情况发生时，很有可能你的系统是高度交织在一起的，这里的一个调整可能会引起那边的多米诺骨牌效应。

没有改变是安全的！你永远不知道它最终会影响到什么。

这正是我们在构建数据模型时想要避免的。您不希望它们紧密耦合，这样您就可以在不破坏整个管道的情况下，在必要时轻松地进行调试和更改。

在头脑中用**模块化**构建你的数据模型是这个问题的解决方案。模块化数据模型彼此独立存在。它们就像一个更大的拼图中的几块。这些部分共同创造了一些美丽的东西，并向您展示了全貌。然而，每个拼图块仍然可以被拉出并独立存在，而不会破坏整个图像。

我们希望我们的数据模型是拼图块，可以轻松地移除、更改和添加，而不会对数据管道产生任何性能影响。

我们如何做到这一点？它从重新评估您当前的数据模型以及如何重写它们以使其更加模块化开始。

# 为每个原始数据源创建基本模型。

任何数据模型都不应该直接引用原始数据。**您总是希望在您的数据仓库中保留一份原始数据的副本。这样，万一您的数据遭到破坏，您可以随时恢复到原始副本。**

基础模型的存在是为了保护您的原始数据不被以任何方式转换。它们通常是数据仓库中位于原始数据源之上的视图，因此它们不占用任何存储空间。他们引用原始数据，但也改变其基本特征，使其更清晰，便于分析工程师使用。

基本型号包括:

*   数据类型转换
*   列名更改
*   时区转换
*   简单 case when 语句

基本模型不应该包括任何花哨的计算、连接或聚合。它们只是原始数据的标准化版本，便于您在下游的数据模型中引用。

例如，假设您有一个原始数据源，如下所示:

```
select 
  id,
  user_id, 
  created_date,
  source
from web_traffic.first_visits
```

为了在我的下游模型中更容易理解和引用，我将使用 [dbt](/what-is-dbt-a0d91109f7d0) 为其编写一个基本模型，如下所示:

```
select 
  id AS first_visit_id,
  user_id, 
  CAST(created_date AS date) AS created_date,
  created_date::timestamp_ntz AS created_at,
  source AS referrer_source
from {{ source('web_traffic', 'first_visits') }}
```

我指定`id`也引用表名，以便将来的连接更容易阅读。

我将`created_date`转换为一个实际的日期，并使用该列将其转换为正确的时间戳类型并重命名。

最后，我更改了`source`列的名称，使其更具描述性。

# 识别数据模型之间的公共代码。

现在，您已经为每个原始数据源创建了一个基本模型，您希望更深入地研究已经存在的更复杂的数据模型。如果您没有使用像 [dbt](https://medium.com/geekculture/4-things-you-need-to-know-about-dbt-e54c016f338c) 这样的转换工具，那么您有可能为每个模型编写了很长的 SQL 代码文件。仔细阅读每一个文件，看看是否能找到在多个不同文件中重复的代码。

这种重复的代码可能是多次使用的货币换算计算、映射代码或频繁连接在一起的表。

我们来看两个独立的模型。

第一个是寻找 10 月份纽约州的所有促销订单。

```
# model looking for all promo orders to the state of New York in the month of October 
with order_information_joined AS (
  select
    orders.order_id,
    orders.ordered_at, 
    order_types.order_type_name, 
    order_addresses.city, 
    states.state_name
  from orders 
  left join order_types 
    on orders.order_type_id = order_types.order_type_id 
  left join order_addresses 
    on orders.order_address_id = order_addresses on order_address_id 
  left join states 
    on order_addresses.state_id = states.state_id 
) 

select 
  order_id 
from order_information_joined 
where '10-01-2022' <= CAST(ordered_at AS date) <= '10-31-2022' 
  and order_type_name = 'promo' 
  and state_name = 'New York'
```

第二个是查找 2022 年的订购单数量。

```
# model finding the number of subscription orders placed in 2022
with subscription_orders AS (
  select * 
  from orders 
  where order_type_id=1
),

order_information_joined AS (
  select
    orders.order_id,
    orders.ordered_at, 
    order_types.order_type_name, 
    order_addresses.city, 
    states.state_name
  from subscription_orders 
  left join order_types 
    on orders.order_type_id = order_types.order_type_id 
  left join order_addresses 
    on orders.order_address_id = order_addresses on order_address_id 
  left join states 
    on order_addresses.state_id = states.state_id 
)

select count(*) from order_information_joined where YEAR(ordered_at)=2022
```

这些模型有什么共同点？两者一起加入`orders`、`order_types`、`order_addresses`和`states`表。因为这段代码至少使用了两次，所以将它作为自己的模型编写可能是一个有用的查询。这样，无论何时需要它，它都可以简单地在某人正在编写的另一个模型中被引用。

另外，请注意第二个模型在将订单表*连接到其他表之前是如何过滤订单表*的。在我们的新模型中，我们不想这样做，因为这样会限制我们使用模型的范围。相反，当引用新模型时，您将能够过滤`order_type_id`。

# 将此代码编写为它自己的数据模型。

既然您已经确定了在多个数据模型中重复的代码，那么您想要将它转换成它自己的*数据模型。这就是为什么你的模型是模块化的！*

通过将重复的代码转换成自己的模型，您可以节省计算成本和运行时间。这样写的话，你的模型会运行得更快！您不再多次运行相同的代码，而是只运行一次，然后在不同的模型中引用它的输出。

**确保取出任何特定于某个数据模型的代码片段。**例如，您可能会为新用户筛选一个模型，但所有其他模型都会查看所有用户。不要在这个模块化数据模型中包含过滤器，而是将它添加到引用模块化模型的特定模型的代码中。

**我们的目标是让这些片段变得不特定和可重用。当你再次引用时，总会有空间添加细节。**

另一件需要注意的事情——确保给模型起一个描述性的名字。您希望其他人能够阅读模型的名称并知道它是做什么的。这样，他们也可以在他们的代码中引用它，而不是重新编写代码。编写模块化数据模型的一大好处是减少了数据团队工作之间的冗余。模块化模型使得使用你的队友已经写好的代码变得容易，节省了你的时间和精力！

如果我们看上面的例子，我会把数据模型叫做`order_details_joined_location`。这意味着我将不同的订单相关表连接到位置相关表。该模型将如下所示:

```
select
    orders.order_id,
    orders.user_id, 
    orders.product_quantity,
    orders.ordered_at, 
    order_types.order_type_id, 
    order_types.order_type_name, 
    order_address.street_line_1, 
    order_address.street_line_2, 
    order_addresses.city,
    order_address.zipcode,
    order_address.country,
    states.state_id,
    states.state_name
  from orders 
  left join order_types 
    on orders.order_type_id = order_types.order_type_id 
  left join order_addresses 
    on orders.order_address_id = order_addresses on order_address_id 
  left join states 
    on order_addresses.state_id = states.state_id
```

请注意，我是如何包含每个表中的所有列，而不仅仅是之前使用的特定模型中的列。这为模型的使用提供了更大的灵活性。

# 在其他模型中引用此模型。

现在，在您取出共享代码并使其成为自己的数据模型之后，您可以在原来的两个模型中引用这个新模型。 [dbt](https://medium.com/geekculture/the-ultimate-guide-to-using-dbt-with-snowflake-2d4bfc37b2fc) 通过允许您使用一个简单的`{{ ref() }}`函数来调用其他数据模型，使这变得很容易。

重写第一个模型以引用我们的新`order_details_joined_location` 模型，如下所示:

```
select 
  order_id 
from {{ ref('order_details_joined_location') }}
where '10-01-2022' <= CAST(ordered_at AS date) <= '10-31-2022' 
  and order_type_name = 'promo' 
  and state_name = 'New York'
```

第二个应该是这样的:

```
select 
  count(*)
from {{ ref('order_details_joined_location') }}
  where order_type_id=1
  and YEAR(ordered_at)=2022
```

现在我们有了两个简单的模型，它们产生与以前相同的输出，但是引用了另一个模型，该模型现在可以用于许多其他模型。

# 结论

模块化是一组能够经受时间考验的数据模型的关键特征。当数据模型模块化时，您可以降低计算成本，减少运行整个管道所需的时间，并通过创建协作编码流程使您的团队生活更轻松。数据建模的未来是模块化。 [dbt](/what-is-dbt-a0d91109f7d0) 成为如此流行的数据转换工具是有原因的！他们了解这种需求，并能很好地解决问题。

关于[分析工程](https://madisonmae.substack.com/)的更多信息，请订阅我的免费每周简讯，在那里我分享学习资源、教程、最佳实践等等。

看看我的第一本电子书，[分析工程基础知识](https://madisonmae.gumroad.com/l/learnanalyticsengineering)，一本全方位的分析工程入门指南。