# 要学习的 5 个 dbt 建模技巧

> 原文：<https://towardsdatascience.com/5-dbt-modelling-tricks-to-learn-2d4928cc7433>

## 充分利用 dbt(以及快速介绍)

![](img/ceed8655468c2f2355f141e75d4c0661.png)

由[凯文·Ku](https://unsplash.com/@ikukevk?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

作为“现代数据堆栈”的一部分，dbt 是一个最近广受欢迎的工具。这是一个非常棒的工具，它支持直接在 SQL 中执行数据转换，使分析师和分析工程师能够专注于将业务逻辑转换为模型，而不是数据工程师倾向于关注的管道、编排和读/写细微差别。

它还[支持 jinja 模板语言](https://docs.getdbt.com/docs/building-a-dbt-project/jinja-macros)，正如我们将在下面看到的，当创建我们的数据模型时，它产生了一大堆额外的选项。在本文中，我们将重点关注 dbt 特性，这些特性可用于简化和增强我们的数据建模项目，创建可用于整个项目的有用且可重复的逻辑。

# 刚接触 dbt？

如果您正在阅读这篇文章，那么您可能已经使用过或者至少听说过某种形式的 dbt，但是如果您是使用 dbt 的新手，我建议您首先回顾并理解他们常见的[最佳实践](https://docs.getdbt.com/guides/legacy/best-practices)。这些并不是 dbt 建模的全部，但是遵循这些将有助于您的项目结构，并减少未来重构的需要。

我会特别考虑:

1.  **文件夹管理** —这定义了文件树和数据库中数据模型的最终结构。您想要“dim”和“fact”文件前缀吗？基于模式或产品领域的文件夹？
2.  **SQL 和样式指南**—为构成模型的 SQL 创建一个约定(大小写、跳转、连接等)。)，以及 dbt 模型(如何使用 cte 和子查询)。如果对自动化这些检查感兴趣，你可以看看使用类似于 [sqlfluff](https://github.com/sqlfluff/sqlfluff) 的东西。
3.  **模型引用** —为了使 dbt 能够完成它的工作，您应该使用`{{ ref() }}` jinja 引用模型，以便 dbt 能够推断模型依赖关系并从上游表中正确构建。`{{ source() }}` jinja 应该只使用一次来从原始数据库中选择，并且永远不应该使用直接模型引用。
4.  **重用模型逻辑** —逻辑应尽可能向上游移动，以免重复(如列重命名)。如果您经常在多个数据模型中重用相同的逻辑，也许您可以将该逻辑本身转换成一个单独的数据模型，并将其移动到上游，这样就可以简单地用`{{ ref() }}`引用它。
5.  **分析层** —一旦开发了 dbt 项目，您可能会因为“循环依赖”而遇到错误，其中两个模型相互依赖。我发现解决这个问题的最好方法是拥有多个模型层，其中每一层只引用其下一层中的模型。有一些关于你如何解决这个问题的文件。

现在谈谈您可能还没有遇到过的主要 dbt 技巧…

![](img/8d61bc14dd4cb8ea2a5d40d5b5b71606.png)

塞缪尔·里甘-阿桑特在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

# 1.快照

[dbt 快照](https://docs.getdbt.com/docs/building-a-dbt-project/snapshots)是实现类型 2 渐变维度的一种简单方法，使分析师能够更轻松地执行数据检查，例如数据值或分析指标何时以及为何发生变化。

这些更改是通过添加了`dbt_valid_from`和`dbt_valid_to` 列的行快照实现的，当前有效的行有`dbt_valid_to=null`。但是，这些列仅在每次运行快照时更新，因此，如果快照每小时运行一次，并且在此期间状态发生多次更改，则只有最新的值会被捕获为更改。因此，这并不能取代实际的数据监控解决方案，如审计数据库或向数据库中写入产品列，如`status_value_at`。

这里从`product_sales`表中创建了一个快照。与常用的`updated_at`策略不同，`check`策略(包含所有列)使快照能够自动包含新的表列。相反，如果从源表中删除了列，快照将保留这些列。

# 2.代理键

dbt-utils 包有一个有用的代理键宏[,它实现了主键的生成。这对于为唯一主键创建 dbt 测试非常有用，对于在 Looker 等 BI 工具中使用该模型也非常有用，因为 Looker 需要专门定义主键才能正确执行聚合。](https://github.com/dbt-labs/dbt-utils#surrogate_key-source)

例如，如果我们为其创建快照的`product_sales`表没有`id`列，我们可以很容易地从两个分组列中创建这个主键。

```
select
{{ dbt_utils.surrogate_key(['product_id', 'month']) }} as product_sale_id
```

# 3.在枢轴上转动

再次回到 dbt-utils 包，这次是针对 [pivot 宏](https://github.com/dbt-labs/dbt-utils/blob/main/macros/sql/pivot.sql)。在 SQL 中，我们经常使用 CASE 语句来对总数进行分段，以便进行进一步的分析，例如在创建产品销售的群组视图来查看逐月业绩时。

```
select
  product_id,
  sum(case when month = 'jan' then amount else 0 end) as amount_jan,
  sum(case when month = 'feb' then amount else 0 end) as amount_feb,
  sum(case when month = 'mar' then amount else 0 end) as amount_mar
from product_sales
group by 1
order by product_id
```

但是随着我们希望透视的列数的增加，这种语法很快就会变得很麻烦。如果我们有 20 个月，我真的不想为每一列复制和粘贴相同的 SQL 逻辑，如果逻辑发生变化，那么我需要更新 20 行。

相反，我们可以使用 dbt pivot util 只用几行代码就可以做到这一点。缺点是这在 cte 上不起作用，只能在物化数据上起作用。因此，如果我们想要引用的模型不存在，我们需要创建它或使用另一个解决方案(如[雪花的枢纽函数](/5-snowflake-query-tricks-you-arent-using-but-should-be-7f264b2a72d8))。

这里选择了行(product_id ),然后`dbt_utils.pivot`函数接受要透视的列名的参数和要转换成列(month)的行值。这个宏是通过 CASE WHEN 语句 so 实现的，默认为`then 1 else 0 end`，但是这里我们设置了`then_value`，所以结果是`then amount else 0 end`。

# 4.试验

dbt 附带了四个通用测试，有助于涵盖您可能执行的一般数据完整性测试。这些是`unique`、`not_null`、`accepted_values`和`relationship`测试。通过项目 yaml 文件可以很容易地实现[通用测试](https://docs.getdbt.com/docs/building-a-dbt-project/tests)和文档，并且测试将在每次运行时与数据模型一起被检查。

这里有一个通用的主键检查，通常应该应用于每个数据模型。

```
version: 2models:
  - name: product_sales
    description: |
      Area: Product
      Table of product sales each month.
    columns:
      - name: product_sale_id
        description: "Primary key"
        tests:
          - unique
          - not_null
```

可能导致问题的是，如果 dbt 测试失败，将不会构建 dbt 模型。这可能意味着，如果不能设置测试的严格性，我们就不能在非主键列上添加健全性测试，比如`accepted_values`或`not_null`。

```
columns:
  - name: month
    description: "Month of sale."
    tests:
      - not_null
      - accepted_values:
          values: ['jan', 'feb', 'mar]
          config:
            severity: warn
```

# 5.宏指令

除了非常有用的 dbt-utils 包，我们还可以创建自己的宏。要使用它们，请确保[将它们添加到 dbt 项目路径](https://docs.getdbt.com/reference/project-configs/macro-paths)。这对于创建可以多次重用的代码片段非常有用，正如我们看到的 dbt-utils pivot 函数的实现一样。

这种用例的一个例子是获取表中所有未删除的行。可能是快照，原始数据是 Fivetran 同步的，有一个 deleted_at 列。最好用`{{ active_table(table) }}`来实现这一点，而不是在代码的多个地方添加 3 个不同的 WHERE 子句。

另一个例子是，尽管 [min/max](https://docs.snowflake.com/en/sql-reference/functions/min.html) 返回忽略雪花中的空值的最小值/最大值，但是对于日期[最大值](https://docs.snowflake.com/en/sql-reference/functions/greatest.html) / [最小值](https://docs.snowflake.com/en/sql-reference/functions/least.html)，如果任何参数为空，则两者都返回空值。例如，如果我们想要获得许多日期中的第一个日期，以获得客户的第一个活动日期，这种行为可能会令人讨厌。为了减轻这种情况，我们必须用一个虚拟值合并每个参数…或者我们可以创建一个宏来完成这个任务。

类似的`dbt_least`值的虚拟值类似于`'12/31/9999'`，所以在我写这篇文章的时候，它至少还能工作 7977 年！

# 最后的想法

dbt 是用于构建数据模型的强大分析工程解决方案，使用户能够专注于创建健壮的业务逻辑，而不是担心加载和写入数据的细节。我们快速介绍了 dbt 和一些重要的初始项目注意事项，然后介绍了 dbt-utils 包中一些有用的特性和宏。这有助于改进一般语法，避免重复代码块，并减少最终数据模型中可能出现的错误。

如果你喜欢这篇文章，你可以找到更多的文章，并在我的[个人资料](https://medium.com/@anthonyli358)上关注我！