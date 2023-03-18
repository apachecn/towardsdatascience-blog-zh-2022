# PySpark 配方:映射和取消透视

> 原文：<https://towardsdatascience.com/pyspark-recipes-map-and-unpivot-d724cca7d0d4>

## PySpark API 真的缺少关键功能吗？

![](img/ea0afe5c3784cea077230647b78fe92c.png)

照片由[威廉·布特](https://unsplash.com/@williambout?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

PySpark 提供了一个流畅的 API，可以满足大多数需求。尽管如此，有经验的 pandas 用户可能会发现一些数据转换并不那么简单。本文旨在提供少量的配方来涵盖一些用户可能认为 PySpark API 本身不支持的用例。实际上它们是被支持的，但是它们确实需要更多的努力(和想象力)。

我们从必要的导入和 spark 会话的创建开始

注意，出于本文的目的，我们使用一个本地 PySpark 环境，它有 4 个内核和 8gb 内存。我们将为每个配方创建简单的数据框。

**地图值**

第一个方法处理映射值，并基于创建一个映射列

映射键值对存储在字典中。构造`chain(*mapping.items())`返回一个键值对的链对象 as (key1，value1，key2，value2，...)，这些键值对用于创建映射[列](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.create_map.html)。这个映射列本质上是一个常量，因此，我们在数据框的每一行中都有相同的映射。映射是通过检索原始列中每个键的映射值来实现的。原始列可能包含在映射中作为键存在的值，这些值因此被映射为 null。原始列中的空值不会导致错误，并且在映射后保持为空。可以使用默认值代替空值，例如

为了完整起见，也可以在 spark 上使用 pandas API 来实现相同的结果

但是本文并没有深入研究这个新的 API，它是一个很大的发展，尤其是对于原型开发而言。我在结论中包含了更多关于这个问题的想法。

**融化(又称逆透视)**

使用过熊猫的用户可能想知道我们如何模仿熊猫 API 中的[融化](https://pandas.pydata.org/docs/reference/api/pandas.melt.html)功能。Melt 将数据帧转换为一种格式，其中一列或多列为标识符变量(id_vars)，而所有其他列(被视为测量变量(value_vars ))被“解投影”到行轴。换句话说，数据帧从宽格式转换为长格式。

演示该配方的起始数据框可以用以下内容构建

在 PySpark 中可能有几种方法来实现熔化函数。在本教程中，我们演示了两个，试图介绍一种通用的思维方式，这种方式在其他情况下也是有帮助的。第一个[解决方案](https://stackoverflow.com/questions/41670103/how-to-melt-spark-dataframe)构造了一个随后被分解的结构数组

为了增加行数和减少列数，我们需要以某种方式将值变量的内容打包到一个容器中，然后将容器分解成多行。使用一个理解，我们创建一个结构体数组，该数组被展开并存储在一个新创建的名为`_vars_and_vals`的列中。展开后，我们在每一行中都有一个结构，可以从中检索所有值变量的列名和列值。

举例来说，我们可以在起始数据框中取消透视一些或所有列

第二个[解决方案](https://stackoverflow.com/questions/70710359/unpivot-dataframe-in-pyspark-with-new-column)依赖于从数组中创建一个地图，然后对其进行分解。

地图的创建也可以通过其他方式来实现，例如`F.create_map(*chain(*((F.lit(x), F.col(x)) for x in value_vars))).` 使用地图作为容器可能不太明显，因为分解地图可能不会立即出现在脑海中，尽管[文档](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.explode.html)对此有明确说明。这两个解决方案提供了相同的结果，但是我不确定哪个解决方案的性能更好(欢迎评论)。就可读性而言，我个人认为第一个解决方案更好。

**结论**

希望这篇文章提供了关于如何使用生成和处理容器(如映射、数组和结构)的`pyspark.sql.functions`来模拟众所周知的 pandas 函数的见解。另一个选择是使用最近推出的 [PySpark pandas API](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html) ，在 Spark v3.2 之前它曾被称为[考拉](https://koalas.readthedocs.io/en/latest/whatsnew/v1.8.2.html)。在[文档](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/faq.html#should-i-use-pyspark-s-dataframe-api-or-pandas-api-on-spark)中的官方建议是，如果你已经熟悉 pandas，就使用 PySpark pandas API，如果你不熟悉，就直接使用 PySpark API。我认为自己是一个有经验的 pandas 用户，但是除了快速原型之外，我仍然更喜欢依赖 PySpark API。主要原因是，与考虑直接使用 PySpark API 的解决方案相比，我不需要担心可能需要更多时间才能发现的不兼容性。例如，Spark 提供了 null(在 SQL 意义上，作为缺失值)和 nan(数字而不是数字)，而 pandas 没有可用于表示缺失值的原生值。在使用 PySpark pandas API 时用 pandas 思考，在某种程度上就像试图通过翻译你的母语来说一种语言。从长远来看，直接用新语言思考可能更好。这只是个人观点，并不是对 PySpark 熊猫 API 质量的批评。我确信有不同的观点，意见的多样性对取得进展至关重要。