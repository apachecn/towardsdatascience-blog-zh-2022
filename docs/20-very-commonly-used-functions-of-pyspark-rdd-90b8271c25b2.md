# PySpark RDD 的 20 个非常常用的功能

> 原文：<https://towardsdatascience.com/20-very-commonly-used-functions-of-pyspark-rdd-90b8271c25b2>

![](img/8f48266872aad10329d17792ec440ee4.png)

由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的[闪光灯 Dantz](https://unsplash.com/@flashdantz?utm_source=medium&utm_medium=referral) 拍摄

## 每个功能都用清晰的例子演示

Apache Spark 在大数据分析领域非常受欢迎。它使用分布式处理系统。PySpark 是 Python 中 Apache Spark 的接口。当你有一个万亿字节大小的巨大数据集时，常规的 python 代码会非常慢。但是 PySpark 算法要快得多。因为它将数据集分成更小的部分，将数据集分布到不同的处理器中，在每个处理器中分别执行操作，然后将它们重新组合在一起，得到总输出。

这是 PySpark 如何更快工作的高级概述。本文将重点介绍 PySpark 中一些非常常用的函数。

如果是初学者，可以用 google-colab 笔记本练习一下。您只需使用以下简单的命令行进行安装:

```
pip install pyspark
```

只需几分钟即可完成安装，笔记本电脑将准备好 PySpark 代码。

首先，需要创建一个 SparkContext，这是 Spark 功能的主要入口点。它代表与火花簇的连接。我在这里创建了一个 SparkContext:

```
from pyspark import SparkContext
sc = SparkContext.getOrCreate()
```

我将从最基本的函数开始，向更便于分析的函数发展。

> sc.parallelize()

在这里，我使用并行化方法，用这个 SparkContext 创建了一个非常简单的 RDD 对象。并行化方法创建一个并行化的集合，允许数据的分布。

```
rdd_small = sc.parallelize([3, 1, 12, 6, 8, 10, 14, 19])
```

您不能像在笔记本中打印常规列表或数组一样打印 RDD 对象。

> 。收集()

如果您简单地键入 rdd_small 并在笔记本中运行，输出将如下所示:

```
rdd_small
```

输出:

```
ParallelCollectionRDD[1] at readRDDFromFile at PythonRDD.scala:274
```

所以，它是一个 parallelCollectionRDD。因为这些数据在分布式系统中。你必须把它们收集到一起，才能作为一个列表使用。

```
rdd_small.collect()
```

输出:

```
[3, 1, 12, 6, 8, 10, 14, 19]
```

当数据集太大时，收集整个 RDD 对象可能并不总是有意义。您可能希望只获取数据的第一个元素或前几个元素来检查数据结构、类型或数据质量。

在这里，我做了一个更大的 RDD 物体:

```
rdd_set = sc.parallelize([[2, 12, 5, 19, 21], [10, 19, 5, 21, 8], [34, 21, 14, 8, 10], [110, 89, 90, 134, 24], [23, 119, 234, 34, 56]])
```

> 。首先()

仅获取 RDD 对象的第一个元素:

```
rdd_set.first()
```

输出:

```
[2, 12, 5, 19, 21]
```

> 。采取()

这里我取前三个元素:

```
rdd_set.take(3)
```

输出:

```
[[2, 12, 5, 19, 21], [10, 19, 5, 21, 8], [34, 21, 14, 8, 10]]
```

我们得到前三个元素作为输出。

> 。文本文件()

此时，我想引入一个文本文件来演示几个不同的功能。

我从美国的[维基百科页面复制了一些文本，用一个简单的记事本制作了一个文本文件。该文件保存为 usa.txt。您可以通过以下链接下载该文本文件:](https://en.wikipedia.org/wiki/United_States)

[](https://github.com/rashida048/Big-Data-Anlytics-Pyspark/blob/main/usa.txt)  

以下是如何使用文本文件创建 RDD:

```
lines = sc.textFile("usa.txt")
```

让我们使用。再次使用()函数查看文件的前 4 个元素:

```
lines.take(2)
```

输出:

```
["The United States of America (U.S.A. or USA), commonly known as the United States (U.S. or US) or America, is a country primarily located in North America. It consists of 50 states, a federal district, five major unincorporated territories, 326 Indian reservations, and nine minor outlying islands.[h] At nearly 3.8 million square miles (9.8 million square kilometers), it is the world's third- or fourth-largest country by geographic area.[c] The United States shares land borders with Canada to the north and Mexico to the south as well as maritime borders with the Bahamas, Cuba, Russia, and other countries.[i] With a population of more than 331 million people,[j] it is the third most populous country in the world. The national capital is Washington, D.C., and the most populous city and financial center is New York City.",  '']
```

> 。平面地图()

将文本内容分开进行分析是一种常见的做法。下面是使用 flatMap 函数按空间分割文本数据并生成一个大的字符串列表:

```
words = lines.flatMap(lambda x: x.split(' '))words.take(10)
```

输出:

```
['The',  'United',  'States',  'of',  'America',  '(U.S.A.',  'or',  'USA),',  'commonly',  'known']
```

前 10 个元素现在看起来像这样。

> 。地图()

如果要对 RDD 的每个元素应用某种变换或使用某个条件，该映射会很有用。在这种情况下，每个元素意味着每个单词。在这里，我将使每个单词小写，并将通过在每个单词上加 1 来将每个单词转换为一个元组。

```
wordsAsTuples = words.map(lambda x: (x.lower(), 1))wordsAsTuples.take(4)
```

输出:

```
[('the', 1), ('united', 1), ('states', 1), ('of', 1)]
```

下面是对发生的事情的一点解释。lambda 表达式中的“x”表示 RDD 的每个元素。你对 x 做的任何事情都适用于 RDD 中的每一个元素。

在这里，我们将“x”转换为(x，1)。所以，每个单词都是(word，1)。仔细查看输出。

> 。reduceByKey

如果有一个键-值对，并且您希望将同一个键的所有值相加，那么这个函数非常有用。例如，在上面的 wordsAsTuples 中，我们有键-值对，其中键是单词，值是 1。通常，元组的第一个元素被认为是键，第二个元素是值。

如果我们在 wordsAsTuples 上使用 reduceByKey，它会将我们为同一个键添加的 1 相加(这意味着相同的单词)。如果我们有 4 个“the”，它将加上 4 个 1，并将使它(' the '，4)

```
counts = wordsAsTuples.reduceByKey(lambda x, y: x+y)
counts.take(3)
```

输出:

```
[('united', 14), ('of', 20), ('america', 1)]
```

因此，在我们的文本数据中,“united”出现了 14 次,“of”出现了 20 次,“america”只出现了一次。

> 。顶部()

返回指定的顶部元素。在这个例子之后，我将进一步解释:

```
counts.top(20, lambda x: x[1])
```

输出:

```
[('the', 55),  
('and', 24),  
('of', 20),  
('united', 14),  
('is', 13),  
('in', 13),  
('a', 13),  
('states', 12),  
('it', 9),  
('to', 7),  
('as', 6),  
("world's", 6),  
('by', 6),  
('world', 5),  
('with', 5),  
('american', 5),  
('war', 5),  
('or', 4),  
('north', 4),  
('its', 4)]
```

这里发生了什么？在这个命令中，我们说我们想要前 20 个元素。然后 x[1]被指定为 lambda 表达式中的一个条件。在类似(' the '，55)的元组中，' the '是 x[0]，55 是 x[1]。在 lambda 中，指定 x[1]意味着我们想要基于每个元素的 x[1]的前 20 个元素。因此，它根据文本文件中的出现次数返回前 20 个单词。

如果使用 x[0]作为 lambda 的条件，它将根据字母顺序返回前 20 名，因为 x[0]是一个字符串。请随意尝试。

> 。过滤器()

在上面的前 20 个单词中，大部分单词都不是很显著。像“to”、“The”、“with”、“in”这样的词不能提供对文本的任何洞察。在处理文本数据时，通常会忽略那些无关紧要的单词。尽管这并不总是一个好主意。

如果我们能排除一些无关紧要的词，我们可能会看到一些更有意义的词出现在前 20 名中。

以下是我在选择前 20 个单词之前想从文本中排除的单词列表:

```
stop = ['', 'the', 'and', 'of', 'is', 'in', 'a', 'it', 'to', 'as', 'by', 'with', 'or', 'its', 'from', 'at']
```

现在，我们将过滤掉那些单词:

```
words_short = counts.filter(lambda x: x[0] not in stop)
```

这个新 RDD 单词 _short 没有我们在‘stop’中列出的那些单词。

以下是现在最热门的 20 个单词:

```
words_short.top(20, lambda x: x[1])
```

输出:

```
[('united', 14),
 ('states', 12),
 ("world's", 6),
 ('world', 5),
 ('american', 5),
 ('war', 5),
 ('north', 4),
 ('country', 3),
 ('population', 3),
 ('new', 3),
 ('established', 3),
 ('war,', 3),
 ('million', 3),
 ('military', 3),
 ('international', 3),
 ('largest', 3),
 ('america,', 2),
 ('states,', 2),
 ('square', 2),
 ('other', 2)]
```

我们在“禁止”列表中没有这些词。

> 。sortByKey()

我们可以用这个对整个 RDD 进行排序。sortByKey()函数。顾名思义，它通过键对 RDD 进行排序。在“计数”RDD 中，关键字是字符串。所以，它会按字母顺序排序。

```
counts.sortByKey().take(10)
```

输出:

```
[('', 3),  
('(1775–1783),', 1),  
('(9.8', 1),  
('(u.s.', 1),  
('(u.s.a.', 1),  
('12,000', 1),  
('16th', 1),  
('1848,', 1),  
('18th', 1),  
('1969', 1)]
```

如你所见，空字符串先出现，然后是数字字符串。因为数字键在字母顺序中排在字母之前。

默认情况下，排序以升序给出结果。但是如果在 sortByKey 函数中传递 False，它会按降序排序。这里我们按降序排序，取前 10 个元素:

```
counts.sortByKey(False).take(10)
```

输出:

```
[('york', 1),  
('years', 1),  
('world.', 1),  
('world,', 1),  
("world's", 6),  
('world', 5),  
('with', 5),  
('which', 1),  
('when', 1),  
('west.', 1)]
```

中排序之前应用函数或条件也是可能的。sortByKey 函数。这里有一个 RDD:

```
r1 = [('a', 1), ('B', 2), ('c', 3), ('D', 4), ('e', 5)]
```

在 RDD r1 中，一些键是小写的，一些键是大写的。如果我们按键排序，默认情况下大写字母会先出现，然后是小写字母。

```
r1.sortByKey().collect()
```

输出:

```
[('B', 2), ('D', 4), ('a', 1), ('c', 3), ('e', 5)]
```

但是，如果我们希望避免使用大小写部分，并且希望函数在排序时不区分大小写，我们可以在。sortByKey 函数。这里有一个例子:

```
r1.sortByKey(True, keyfunc=lambda k: k.upper()).collect()
```

输出:

```
[('a', 1), ('B', 2), ('c', 3), ('D', 4), ('e', 5)]
```

在上面的 lambda 表达式中，我们要求函数将所有键都视为大写，然后进行排序。它只将所有键视为大写字母，但不返回大写字母的键。

> 。groupByKey()

这个函数 groupByKey()根据键对所有值进行分组，并对它们进行聚合。**提醒一下，默认情况下，元组中的第一个元素是键，第二个元素是值。**在进一步讨论之前，我们先看一个例子:

```
numbers_only = wordsAsTuples.groupByKey().map(lambda x: sum(x[1]))
numbers_only.take(10)
```

输出:

```
[14, 20, 1, 1, 1, 6, 2, 13, 3, 1]
```

在这种情况下，关键词就是单词。假设“the”是一个键，当我们使用 groupByKey()时，它将这个键“the”的所有值分组，并按照指定的方式聚合它们。这里我使用 sum()作为聚合函数。所以，它总结了所有的值。我们得到了每个单词的出现次数。**但是这一次我们只得到出现次数的列表。**

> 。减少()

它用于减少 RDD 元素。减少数量 _ 只有我们从上一个例子中得到的:

```
total_words = numbers_only.reduce(lambda x, y: x+y)
total_words
```

输出:

```
575
```

我们得了 575 分。这意味着文本文件中总共有 575 个单词。

> 。地图值()

它可以用来对键值对的值进行某种转换。它返回键和转换后的值。下面是一个例子，其中键是字符串，值是整数。我将这些值除以 2:

```
rdd_1 = sc.parallelize([("a", 3), ("n", 10), ("s", 5), ("l", 12)])rdd_1.mapValues(lambda x: x/2).collect()
```

输出:

```
[('a', 1.5), ('n', 5.0), ('s', 2.5), ('l', 6.0)]
```

这里，lambda 表达式中的“x”表示值。所以，无论你对 x 做什么，都适用于 RDD 中的所有值。

再举一个例子会有助于更好地理解它。在本例中，使用了不同的 RDD，其中键是字符串，值是整数列表。我们将在列表中使用聚合函数。

```
rdd_map = sc.parallelize([("a", [1, 2, 3, 4]), ("b", [10, 2, 8, 1])])rdd_map.mapValues(lambda x: sum(x)).collect()
```

输出:

```
[('a', 10), ('b', 21)]
```

看看这里的输出。每个值都是值列表中整数的总和。

> 。countByValue()

以字典格式返回 RDD 中每个元素的出现次数。

```
sc.parallelize([1, 2, 1, 3, 2, 4, 1, 4, 4]).countByValue()
```

输出:

```
defaultdict(int, {1: 3, 2: 2, 3: 1, 4: 3})
```

输出显示了一个字典，其中键是 RDD 的不同元素，值是这些不同值的出现次数。

> 。getNumPartitions()

RDD 对象存储为元素簇。换句话说，一个 RDD 对象被分成许多分区。我们不做这些划分。这就是 RDDs 的本质。这是默认发生的。

集群可以为所有分区同时运行一个任务。顾名思义，这个功能。getNumPartitions()告诉你有多少分区。

```
data = sc.parallelize([("p",5),("q",0),("r", 10),("q",3)])data.getNumPartitions()
```

输出:

```
2
```

“数据”对象中有两个分区。

您可以使用. glom()函数来查看它们是如何划分的:

```
data.glom().collect()
```

输出:

```
[[('a', 1), ('b', 2)], [('a', 2), ('b', 3)]]
```

它显示了两个元素列表。因为有两个分区。

> 联盟

您可以使用 union 合并两个 rdd。例如，这里我制作了两个 rdd“rd1”和“rd2”。然后，我使用 union 将它们连接在一起，创建“rd3”。

```
rd1 = sc.parallelize([2, 4, 7, 9])rd2 = sc.parallelize([1, 4, 5, 8, 9])rd3 = rd1.union(rd2)rd3.collect()
```

输出:

```
[2, 4, 7, 9, 1, 4, 5, 8, 9]
```

新形成的 RDD“rd3”包括“rd1”和“rd2”的所有元素。

> 。独特()

它返回 RDD 的独特元素。

```
rd4 = sc.parallelize([1, 4, 2, 1, 5, 4])rd4.distinct().collect()
```

输出:

```
[4, 2, 1, 5]
```

我们只有“rd4”的独特元素。

> 。zip()

当我们在两个 rdd 上使用 zip 时，它们使用两个 rdd 的元素创建元组。一个例子将清楚地证明这一点:

```
rd11 = sc.parallelize(["a", "b", "c", "d", "e"])rdda = sc.parallelize([1, 2, 3, 4, 5])rda_11 = rdda.zip(rd11)
rda_11.collect()
```

输出:

```
[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')]
```

在 zip 操作中，我们首先提到了“rdda”。因此，在输出中，“rdda”元素排在第一位。

> 连接

本文最后要做的是连接。这个名字已经告诉你它连接了两个 rdd。让我们再做一个类似‘RDA _ 11’的 RDD，然后我们就加入。

```
rddb = sc.parallelize([1, 3, 4, 6])rd22 = sc.parallelize(["apple", "ball", "cal", "dog"])
rdb_22 = rddb.zip(rd22)rdb_22.collect()
```

输出:

```
[(1, 'apple'), (3, 'ball'), (4, 'cal'), (6, 'dog')]
```

我们现在有‘RDB _ 22’。让我们将上一个示例中的“rdda_11”和“rdb_22”连接在一起:

```
rda_11.join(rdb_22).collect()
```

输出:

```
[(4, ('d', 'cal')), (1, ('a', 'apple')), (3, ('c', 'ball'))]
```

默认情况下，join 操作连接键上的两个 rdd。再次提醒，每个元组的第一个元素被认为是键。

在基本的连接操作中，只有两个 rdd 中的公共键元素连接在一起。

还有其他种类的连接。以下是左外部联接的一个示例:

```
rda_11.leftOuterJoin(rdb_22).collect()
```

输出:

```
[(4, ('d', 'cal')),  
(1, ('a', 'apple')),  
(5, ('e', None)),  
(2, ('b', None)),  
(3, ('c', 'ball'))]
```

因为这是左外部连接，所以左边提到的 RDD，在这种情况下，“rda_11”将带来它的所有元素。但是元素的顺序可能和‘RDA _ 11’不一样。右侧的 RDD 只会带来与左侧的 RDD 相同的元素。

还有一个右外部联接的作用正好相反:

```
rda_11.rightOuterJoin(rdb_22).collect()
```

输出:

```
[(4, ('d', 'cal')),  
(1, ('a', 'apple')),  
(6, (None, 'dog')),  
(3, ('c', 'ball'))]
```

最后，有一个完整的外部连接，它从两个 rdd 返回每个元素。

```
rda_11.fullOuterJoin(rdb_22).collect()
```

输出:

```
[(4, ('d', 'cal')),
 (1, ('a', 'apple')),
 (5, ('e', None)),
 (2, ('b', None)),
 (6, (None, 'dog')),
 (3, ('c', 'ball'))]
```

如您所见，这包含了来自两个 rdd 的所有密钥。

## 结论

我想列出最常用和最简单的 RDD 操作，它们可以处理很多任务。还有很多 RDD 行动。稍后我可能会拿出更多的。希望这是有帮助的。

请随时在 [Twitter](https://twitter.com/rashida048) 、[脸书页面](https://www.facebook.com/Regenerative-149425692134498)上关注我，并查看我的 [YouTube 频道](https://www.youtube.com/channel/UCzJgOvsJJPCXWytXWuVSeXw)。

## 更多阅读

[](/regression-in-tensorflow-using-both-sequential-and-function-apis-314e74b537ca)  [](/simple-explanation-on-how-decision-tree-algorithm-makes-decisions-34f56be344e9)  [](https://pub.towardsai.net/data-analysis-91a38207c92b)  [](/understanding-regularization-in-plain-language-l1-and-l2-regularization-2991b9c54e9a)  [](/exploratory-data-analysis-with-some-cool-visualizations-in-pythons-matplotlib-and-seaborn-library-99dde20d98bf) 