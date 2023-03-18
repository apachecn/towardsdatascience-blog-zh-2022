# 如何在 MongoDB 中进行基本的全文搜索

> 原文：<https://towardsdatascience.com/how-to-do-basic-full-text-searches-in-mongodb-48b17242676>

## 通过 MongoDB 中的文本索引搜索您的文本数据

由于 MongoDB 是一个面向文档的 NoSQL 数据库，所以在一些字段中存储纯文本是很常见的。为了搜索字符串字段，我们可以直接使用正则表达式操作符`[$regex](https://docs.mongodb.com/manual/reference/operator/query/regex/)`。然而，`$regex`只能用于简单的搜索查询，不能高效地使用索引。

![](img/ba74212ca51ad81589cc4b284fa698a8.png)

图片来自 Pixabay 的 DariuszSankowsk

在 MongoDB 中，有更好的方法来搜索字符串字段。一个经典的方法是创建一个`text`索引，并根据它进行搜索。尽管 MongoDB 现在支持一个[“高级”全文解决方案](https://lynn-kwong.medium.com/learn-powerful-full-text-searches-with-mongodb-atlas-search-e3fee4fcc324)，但是，它只有在您使用 [Atlas](https://lynn-kwong.medium.com/how-to-use-mongodb-atlas-to-manage-your-server-and-data-d97a6e7663c5) 托管数据时才有效。由于在我们的工作中经常使用自我管理的 MongoDB 服务器，特别是对于一些小而简单的项目，因此值得学习和使用经典的文本搜索解决方案，它可以通过简单的查询显著提高您的搜索效率。正如后面将要演示的，大多数常见的搜索问题都可以通过使用`text`索引以及经典的 MongoDB 搜索和聚合查询来解决。

为了进行演示，我们将搜索存储在 MongoDB 数据库中的笔记本电脑列表。请下载[这个 JSON 文件](https://gist.github.com/lynnkwong/86e314061919cccbc89aa7ee597ff61b)(由作者生成)包含一些虚构网店的笔记本电脑数据。请注意，数据是根据一些常见的笔记本电脑品牌随机生成的。它可以免费使用，不会有任何许可问题。然后使用以下命令导入数据:

当上面的代码运行时，我们将在`products`数据库中拥有一个包含 200 个笔记本电脑数据文档的`laptops`集合。这些文件的内容如下:

现在数据已经准备好了，我们可以开始创建一个`text`索引，并进行一些基本的全文搜索。

在本教程中，我们将使用`mongosh`直接运行查询。如果您需要编写一些复杂的查询，您可能会发现一个 [MongoDB IDE](https://lynn-kwong.medium.com/how-to-use-mongodb-with-graphical-ides-420597ede80e) 很有帮助，它提供了命令自动完成和错误突出显示。为了简单起见，我们将使用 Docker 容器附带的`mongosh`,因此我们不需要单独安装任何东西:

```
$ **docker exec -it mongo-server bash**$ **mongosh "mongodb://admin:pass@localhost:27017"**
test> **use products**
products > **show collections**
laptops
```

## **创建一个** `**text**` **指标**

在我们开始之前，有一些重要的事情我们应该记住，即**对于一个集合**只能有一个 `**text**` **索引。**

让我们在`name`字段上创建一个`text`索引，这是通过集合的`createIndex()`方法完成的:

```
products> db.laptops.**createIndex( { name: "text" } )** name_text
```

`name`是我们想要为其创建索引的字符串字段，而“`text`”值表示我们想要创建一个支持基本全文搜索的`text`索引。相比之下，要在 MongoDB 中创建一个[常规索引](https://medium.com/codex/how-to-use-indexes-properly-in-mongodb-ff4560dc67f5)，我们为一个字段指定 1 或-1，以指示该字段在索引中应该按升序还是降序排序。

在我们开始搜索`text`索引之前，我们应该知道尽管一个集合只能有一个`text`索引，但是这个索引可以覆盖多个字段。让我们删除上面创建的`text`索引，并创建一个包含`name`和`attributes`字段的新索引。

请注意，`text`索引可以有不同的名称，但是在一个集合中只能有一个`text`索引。

## **使用** `**text**` **索引**的基本全文搜索

现在让我们使用刚刚创建的`text`索引进行一些基本的全文搜索。我们将使用`$text`查询操作符来执行文本搜索。例如:

`$text`操作符使用接受字符串值的`$search`字段进行文本搜索。在底层，搜索字符串使用空格和标点符号作为分隔符。对于生成的令牌，它们中的每一个都被独立地搜索，并用一个逻辑`OR`操作符连接。此外，默认情况下，搜索不区分大小写。如果您想进行区分大小写的搜索，您可以为`$text`操作符指定`[$caseSensitive](https://docs.mongodb.com/manual/reference/operator/query/text/#case-and-diacritic-insensitive-search)`字段。

因此，使用上面的搜索查询，我们得到的文档包含“HP”或“ProBook ”,但不一定两者都包含。

```
[
  { _id: 19, name: 'HP ZBook Model 19' },
  { _id: 20, name: 'HP ZBook Model 20' },
  { _id: 3, name: 'HP EliteBook Model 3' },
  { _id: 18, name: 'HP ProBook Model 18' },
  ...
]
```

## 按文本分数排序

重要的是，使用`$text`操作符，会为每个文档分配一个分数，表明文档与搜索字符串的匹配程度。如果“HP”和“ProBook”都匹配某个文档，则该文档的得分高于只匹配“HP”或“ProBook”的文档。我们可以根据分数对文档进行排序，用`limit()`方法只能得到最上面的。

可能看起来很奇怪，分数是用`{$meta: "textScore"}`表达式返回的。此外，乍一看可能更奇怪的是:

*   给定的字段名称(`score`)并不重要。你可以给一个不同的名字，它仍然会工作。
*   按分数排序始终是降序。这是有意义的，因为通常我们希望找到最相关的匹配。

通过这个查询，我们可以得到我们想要的最相关的结果:

```
[
  { _id: 15, name: 'HP ProBook Model 15' },
  { _id: 16, name: 'HP ProBook Model 16' },
  { _id: 18, name: 'HP ProBook Model 18' }
]
```

## 按短语搜索

如果我们只想找到完全包含“HP ProBook”的文档，我们可以通过短语进行搜索，只需将搜索字符串放在一对嵌套的引号中。我们可以交替使用单引号和双引号，或者用反斜杠对引号进行转义。以下查询将给出相同的结果:

## 在搜索查询中使用否定

我们还可以在我们的搜索查询中使用否定，这要求文档不匹配某些标记。让我们搜索“惠普”但不是“ProBook”的笔记本电脑:

在结果列表中，我们再也看不到“ProBook”了:

```
[
  { _id: 19, name: 'HP ZBook Model 19' },
  { _id: 20, name: 'HP ZBook Model 20' },
  { _id: 3, name: 'HP EliteBook Model 3' },
  ...
]
```

## 嵌套文档中的文本搜索

现在让我们用一个属性进行搜索，看看`text`索引是否同时覆盖了`name`和`attributes`字段:

请注意，我们需要按分数排序，否则最高的结果可能不是你所期望的。这是因为“HP 1TB”不是一个短语。实际上，“HP”出现在`name`字段中，而“1TB”出现在`attributes.attribute_value`字段中。由于默认情况下使用了`OR`逻辑运算符，返回的文档将包含“HP”或“1TB ”,但不一定两者都包含。使用`sort()`和`limit()`方法，我们将返回最相关的结果，这些结果通常是我们想要的。

## 将$text 运算符与其他运算符结合使用

`$text`操作符可以和常规的 MongoDB 操作符一起使用。例如，让我们找到价格低于 10000 SEK 的 HP ProBooks:

这是我们得到的结果:

```
[
  { _id: 13, name: 'HP ProBook Model 13', price: 9994 },
  { _id: 9, name: 'HP ProBook Model 9', price: 9980 }
]
```

但是需要注意的是，搜索查询中应该只有一个`$text`操作符，否则只有最后一个有效。这是因为查询文档(Python 中的字典)不能有重复的键。

## 在聚合中使用$text 运算符

`$text`操作符也可以用在聚合管道中。但是，有三个主要限制:

*   `$text`操作器只能在`$match`阶段使用。
*   包含`$text`运算符的`$match`阶段必须是管道的第一个阶段。
*   `$text`操作符在`$match`阶段和整个流水线中只能出现一次。

让我们为“HP ProBook”编写一个聚合管道来计算按 RAM 大小分组的笔记本电脑数量:

```
[
  { _id: '16GB', count: 2 },
  { _id: '8GB', count: 4 },
  { _id: '4GB', count: 1 }
]
```

它显示了`$text`操作符就像聚合管道中的任何其他常规操作符一样工作。

在本文中，我们介绍了 MongoDB 中使用`text`索引的经典文本搜索。由`text`索引和相应的`$text`操作符提供的全文搜索解决方案很简单，但也非常强大。对于大多数只需要通过简单条件进行搜索的小项目来说应该足够了。如果您想进行更高级的搜索，需要有多个字符串字段的索引，并使用复杂的 ***【应该(不)***/**/*【必须(不)*** 条件，您可能想使用更高级的搜索引擎，如 [Elasticsearch](http://What is Elasticsearch and why is it so fast?) ，或“高级” [Atlas Search](https://betterprogramming.pub/learn-advanced-full-text-searches-with-mongodb-atlas-search-5e4b51719427) 。