# 使用雪花的原生数据分类之前要知道的 5 件事

> 原文：<https://towardsdatascience.com/5-things-to-know-before-using-snowflakes-native-data-classification-30f4e1d2ac06>

## 了解雪花的 PII 检测功能

![](img/cf3e54717b7b21be5cb5d2e3d6e12027.png)

由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的[absolute vision](https://unsplash.com/@freegraphictoday?utm_source=medium&utm_medium=referral)拍摄

在当今世界，数据收集和处理受到监管，组织别无选择，只能遵守这些法规。因此，公司开始重新思考他们设计信息系统、数据存储和业务流程的方式，同时考虑到隐私问题。

实施数据保护原则的一个基本要素是**数据分类。**

# 什么是数据分类？

数据分类通常被定义为**将** **数据**组织成**组类别**的过程，以帮助公司更有效地使用和保护数据。数据分类有助于我们了解我们在语义方面拥有什么，以便更好地保护它。

然而，这个组件通常是一个**难以解决的问题**…一些公司采用手工方式，另一些公司使用 ML 来自动分类他们的数据集。无论哪种方式，解决这个问题都是**昂贵的**并且**可能是无效的**取决于数据存储的方式和位置。

# 雪花拯救世界

如果您的数据堆栈中存在雪花，您可能希望利用其固有的[数据分类特性](https://docs.snowflake.com/en/user-guide/governance-classify-concepts.html)。在扫描和分析了数据仓库对象(表、视图等)的内容和元数据之后..)，该特征将确定适当的[语义和隐私类别](https://docs.snowflake.com/en/user-guide/governance-classify-concepts.html#classification-categories)。它将帮助您发现&标记 PII 数据，并显著降低管理和保护您的数据的复杂性和成本。

![](img/496c53a586c6f1860aeb40faf0e70825.png)

文本到图像使用[中途](https://www.midjourney.com/):北极熊在云端保护数据库

但是，在您决定使用雪花的原生数据分类功能之前，有几件重要的事情您应该考虑:

## **1。数据类型**

虽然您可以对半结构化数据(带有 JSON 对象的变量类型列)进行分类，但是该特性仅限于分析具有单一数据类型的变量，例如:varchar 或 number。如果您的表不包含任何 JSON 字段，这应该不是什么大问题。但是，如果您非常依赖雪花存储和查询半结构化数据的能力，您应该记住它不能与数据分类功能结合使用。您将需要考虑一个多步骤的过程，其中(1)您展平您的列，并确保它是受[支持的数据类型](https://docs.snowflake.com/en/user-guide/governance-classify-concepts.html#supported-objects-and-column-data-types)之一，然后(2)您运行分类。

## 2.综合

说到流程，第二点是关于找到您需要/想要执行数据分类的正确步骤。很可能，您已经建立了数据管道，为不同环境中的许多数据库提供数据。那么，具体到哪一点，您会对数据进行分类呢？也许，在将数据转储到数据仓库之后，您可能会想。

如果是，这个阶段的数据质量是否足够好，可以高置信度地进行可靠分类？数据量呢？也许在数据被清理和建模之后，在更下游的地方进行分类会更好，对吗？在这种情况下，您将如何处理法规遵从性、治理和安全性？永远不会到达业务/指标层的数据怎么办？在开始对数据进行分类之前，您需要彻底回答这些问题。

## 3.自动化和可扩展性

在他们的[博客](https://www.snowflake.com/blog/data-classification-available-in-public-preview/)中，Snowflake 描述了原生数据分类特性，好像它将移除所有手动过程。这可能是定制数据集的理想场景中的情况，然而，真实世界的用例却大不相同；数据仓库通常包含多个环境、数据库和数据共享。事实上，雪花提供了三个[存储过程](https://docs.snowflake.com/en/user-guide/governance-classify-using.html#use-stored-procedures-to-classify-all-tables-in-a-schema)；第一个用于对模式中的所有表进行分类，第二个用于对数据库中的所有表进行分类，第三个用于使用标记对分类的对象列应用分类结果。手动触发的(甚至是预定的)存储过程在自动化、可伸缩性和监控方面根本达不到预期。尤其是因为没有简单的方法来分类新的或改变的对象。

与上面提到的博客文章相比，Snowflake 的文档建议了一个[工作流](https://docs.snowflake.com/en/user-guide/governance-classify-using.html#classification-workflow)，用户可以选择手动检查分类输出，并根据需要进行修改。这种方法的问题是它很难扩展；这不仅是因为它需要人工关注，还因为缺少一个方便审查和批准过程的用户界面。您需要构建自己的工具来弥合这一差距。

## 4.表演

绩效评估是多方面的，但我将只讨论一个方面；全表扫描。

要分析表/视图中的列，需要运行以下函数:

```
EXTRACT_SEMANTIC_CATEGORIES('*<object_name>*' [,*<max_rows_to_scan>*])
```

除了对象名(如表名)，它还带有一个可选参数<***max _ rows _ to _ scan>***，代表样本大小。如果您没有明确地将它设置为 0 到 10000 之间的一个数字，它将默认为 10000 行。起初，我认为样本大小对性能(查询运行时间)有重要影响，但在对该特性进行实验后不久，我意识到无论我设置的样本大小是大是小，每次调用函数时，雪花都会执行全表扫描。样本大小将主要影响分类结果的准确性，而不是性能。如果您计划频繁运行分类过程，您应该评估性能。如果您发现分类很慢，您可以投入更多的计算能力来加快速度，或者使用像[基于分数的行采样](https://docs.snowflake.com/en/sql-reference/constructs/sample.html#sample-tablesample)这样的技术来绕过全表扫描。

## 5.展开性

一旦*EXTRACT _ SEMANTIC _ CATEGORIES*函数运行分类算法，下一步就是将生成的结果作为[标签](https://docs.snowflake.com/en/user-guide/governance-classify-sql.html#category-tag-values-and-mappings)应用于目标对象列。

截至本文发布之日，可用的分类标签如下所示:

```
{
  "name": [
    "PRIVACY_CATEGORY",
    "SEMANTIC_CATEGORY"
  ],
  "allowed_values": [
    [
      "IDENTIFIER",
      "QUASI_IDENTIFIER",
      "SENSITIVE",
      "INSENSITIVE"
    ],
    [
      "EMAIL",
      "GENDER",
      "PHONE_NUMBER",
      "IP_ADDRESS",
      "URL",
      "US_STATE_OR_TERRITORY",
      "PAYMENT_CARD",
      "US_SSN",
      "AGE",
      "LAT_LONG",
      "COUNTRY",
      "NAME",
      "US_POSTAL_CODE",
      "US_CITY",
      "US_COUNTY",
      "DATE_OF_BIRTH",
      "YEAR_OF_BIRTH",
      "IBAN",
      "US_PASSPORT",
      "MARITAL_STATUS",
      "LATITUDE",
      "LONGITUDE",
      "US_BANK_ACCOUNT",
      "VIN",
      "OCCUPATION",
      "ETHNICITY",
      "IMEI",
      "SALARY",
      "US_DRIVERS_LICENSE",
      "US_STREET_ADDRESS"
    ]
  ]
}
```

这些标签已经为您定义好了，并且存储在雪花型**只读**共享数据库的核心模式中。这意味着，如果您想要通过使用[*ASSOCIATE _ SEMANTIC _ CATEGORY _ TAGS*](https://docs.snowflake.com/en/sql-reference/stored-procedures/associate_semantic_category_tags.html)存储过程来自动应用标签，您将受限于这个可用标签列表。鉴于许多**标识符**和**准标识符**是**关注美国的**，你可能需要考虑定义你自己的标签列表。但是，真正的挑战是弄清楚这个新列表将如何与原列表一起工作。因此，您将经历额外的步骤，例如创建和设置标签:

```
CREATE [ OR REPLACE ] TAG [ IF NOT EXISTS ] ...
ALTER TABLE ... MODIFY COLUMN ... SET TAG
```

# 最后的想法

总而言之，设计和构建数据分类解决方案并不是一件容易的事情。Snowflake 提供了一个很好的起点，它已经通过调用一个函数抽象出了许多挑战。但是，不要指望它会自动扫描您的整个数据仓库，并使用标签显示任何 PII。数据工程师仍然需要设计端到端的流程；包括但不限于构建一些工具来促进手动审查过程以及对数据量、预算和使用模式的优化。上面列出的五点可能没有涵盖 Snowflake 中 PII 分类特性产品化的所有方面。所以，如果你有不同的东西要补充，或者如果你认为某些方面可以用更好的方法来解决，请写下评论并分享你的想法。