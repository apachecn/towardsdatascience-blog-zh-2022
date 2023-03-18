# PySpark 还是熊猫？为什么不两者都要

> 原文：<https://towardsdatascience.com/pyspark-or-pandas-why-not-both-95523946ec7c>

## 整体大于部分之和

![](img/5c97ee5d5f2aca6a2bd11dc057131018.png)

戴维·马尔库在 [Unsplash](https://unsplash.com/s/photos/nature?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

[导入并开始数据集](#f8c3)
[系列到系列和多个系列到系列](#e706)
[系列的迭代器到系列的迭代器和多个系列的迭代器到系列的迭代器](#a765)
[数据帧的迭代器到数据帧的迭代器](#22fd)
[系列到标量和多个系列到标量](#1331)
[组映射 UDF](#fdab)
[最终想法](#2952)

PySpark 允许许多开箱即用的数据转换。然而，在熊猫身上可以得到更多。Pandas 功能强大，但由于其内存处理特性，它无法处理非常大的数据集。另一方面，PySpark 是一个用于大数据工作负载的分布式处理系统，但不支持 pandas 提供的丰富的数据转换。随着 Spark 3.x 的发布，PySpark 和 pandas 可以通过利用多种方法创建 pandas 用户定义函数(UDF)来进行组合。本文的目的是展示一组使用 Spark 3.2.1 的熊猫 UDF 示例。在后台，我们使用 Apache Arrow，这是一种内存中的列数据格式，可以在 JVM 和 Python 进程之间高效地传输数据。更多信息可以在 PySpark 用户指南[中的官方 Apache Arrow](https://spark.apache.org/docs/3.2.1/api/python/user_guide/sql/arrow_pandas.html)中找到。

本文中的内容不要与官方用户指南[中描述的 Spark 上的最新熊猫 API 相混淆。这是 Spark 中利用熊猫表现力的另一种可能性，代价是一些不兼容性。](https://spark.apache.org/docs/3.2.1/api/python/user_guide/pandas_on_spark/pandas_pyspark.html)

# **导入和启动数据集**

对于本文中的例子，我们将依赖熊猫和 numpy。我们还使用(希望)常用的约定从`pyspark.sql`导入函数和类型模块:

```
import pandas as pd
import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.types as T
```

所有示例都适用于 20 行 4 列的小型数据集:

*   group，一个用作分组键的`T.StringType()`列
*   x，一个`T.DoubleType()`栏目
*   y_lin，一个`T.DoubleType()`列，它是 x 的两倍，带有一些噪声
*   y_qua，一个三倍于 x 的平方的`T.DoubleType()`列，带有一些噪声

火花数据帧可以通过以下方式构建

```
g = np.tile(['group a','group b'], 10)
x = np.linspace(0, 10., 20)
np.random.seed(3) # set seed for reproducibility
y_lin = 2*x + np.random.rand(len(x))/10.
y_qua = 3*x**2 + np.random.rand(len(x))
df = pd.DataFrame({'group': g, 'x': x, 'y_lin': y_lin, 'y_qua': y_qua})
schema = StructType([
    StructField('group', T.StringType(), nullable=False),
    StructField('x', T.DoubleType(), nullable=False),
    StructField('y_lin', T.DoubleType(), nullable=False),
    StructField('y_qua', T.DoubleType(), nullable=False),
])
df = spark.createDataFrame(df, schema=schema)
```

其中`spark`是产生的 spark 会话

```
spark = (
    SparkSession.builder
      .appName('learn pandas UDFs in Spark 3.2')
      .config('spark.sql.execution.arrow.pyspark.enabled', True) 
      .config('spark.sql.execution.arrow.pyspark.fallback.enabled', False)
      .getOrCreate()
)
```

数据帧可通过以下方式检查

```
def show_frame(df, n=5):
    df.select([F.format_number(F.col(col), 3).alias(col)
               if df.select(col).dtypes[0][1]=='double'
               else col
               for col in df.columns]).show(truncate=False, n=n)show_frame(df)
# +-------+-----+-----+------+
# |group  |x    |y_lin|y_qua |
# +-------+-----+-----+------+
# |group a|0.000|0.055|0.284 |
# |group b|0.526|1.123|1.524 |
# |group a|1.053|2.134|3.765 |
# |group b|1.579|3.209|7.636 |
# |group a|2.105|4.300|13.841|
# +-------+-----+-----+------+
# only showing top 5 rows
```

注意双栏的格式化/截断。仅显示了 20 行中的 5 行。

# **串联到串联和多个串联到串联**

最简单的熊猫 UDF 将一个熊猫系列转换为另一个熊猫系列，没有任何聚合。例如，通过减去平均值并除以标准偏差来标准化一个系列，我们可以使用

```
# series to series pandas UDF
@F.pandas_udf(T.DoubleType())
def standardise(col1: pd.Series) -> pd.Series:
    return (col1 - col1.mean())/col1.std()
res = df.select(standardise(F.col('y_lin')).alias('result'))
```

装饰者需要熊猫 UDF 的返回类型。还要注意函数定义中 python 类型的使用。结果可通过以下方式检查

```
print(f"mean and standard deviation (PYSpark with pandas UDF) are\n{res.toPandas().iloc[:,0].apply(['mean', 'std'])}")# mean and standard deviation (PYSpark with pandas UDF) are
# mean    6.661338e-17
# std     9.176629e-01
# Name: result, dtype: float64
```

正如我们在上面看到的，平均值在数字上等于零，但标准差不是。这是因为 PySpark 的分布式特性。PySpark 将执行 Pandas UDF，方法是将列分成批，并调用每个批的函数作为数据的子集，然后将结果连接在一起。因此，在上面的例子中，标准化应用于每个批次，而不是作为整体的数据帧。我们可以通过用熊猫本身测试熊猫 UDF 来验证这种说法的有效性:

```
res_pd = standardise.func(df.select(F.col('y_lin')).toPandas().iloc[:,0])print(f"mean and standard deviation (pandas) are\n{res_pd.apply(['mean', 'std'])}")# mean and standard deviation (pandas) are
# mean   -2.220446e-16
# std     1.000000e+00
# Name: y_lin, dtype: float64
```

在那里可以使用`standardise.func()`从装饰过的熊猫中找回原来的熊猫 UDF。验证语句有效性的另一种方法是使用重新分区

```
res = df.repartition(1).select(standardise(F.col('y_lin')).alias('result'))

print(f"mean and standard deviation (PYSpark with pandas UDF) are\n{res.toPandas().iloc[:,0].apply(['mean', 'std'])}")# mean and standard deviation (PYSpark with pandas UDF) are
# mean   -2.220446e-16
# std     1.000000e+00
# Name: result, dtype: float64
```

这当然不是现实生活中所希望的，但有助于在这个简单的例子中演示内部工作原理。

多个串联到串联的情况也很简单。作为一个简单的例子，我们添加两列:

```
# multiple series to series pandas UDF
@F.pandas_udf(T.DoubleType())
def add_cols(col1: pd.Series, col2: pd.Series) -> pd.Series:
    return col1 + col2
res = df.select(F.col('y_lin'), F.col('y_qua'), add_cols(F.col('y_lin'), F.col('y_qua')).alias('added columns'))show_frame(res)
# +-----+------+-------------+
# |y_lin|y_qua |added columns|
# +-----+------+-------------+
# |0.055|0.284 |0.339        |
# |1.123|1.524 |2.648        |
# |2.134|3.765 |5.899        |
# |3.209|7.636 |10.845       |
# |4.300|13.841|18.141       |
# +-----+------+-------------+
# only showing top 5 rows
```

返回的序列也可以是类型`T.StructType()`，在这种情况下，我们指示熊猫 UDF 返回一个数据帧。举个简单的例子，我们可以通过在数据框中组合两列来创建一个结构列

```
# series to series (struct) pandas UDF
schema = T.StructType([
    StructField('y_lin', T.DoubleType()), 
    StructField('y_qua', T.DoubleType()),
])
@F.pandas_udf(schema)
def create_struct(col1: pd.Series, col2: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({'y_lin': col1, 'y_qua': col2})res = df.select(F.col('y_lin'), F.col('y_qua'), create_struct(F.col('y_lin'), F.col('y_qua')).alias('created struct'))show_frame(res)
# +-----+------+------------------------------------------+
# |y_lin|y_qua |created struct                            |
# +-----+------+------------------------------------------+
# |0.055|0.284 |{0.05507979025745755, 0.28352508177131874}|
# |1.123|1.524 |{1.123446361209179, 1.5241628490609185}   |
# |2.134|3.765 |{2.134353631786031, 3.7645534406624286}   |
# |3.209|7.636 |{3.2089774973618717, 7.6360921152062655}  |
# |4.300|13.841|{4.299821011224239, 13.8410479099986}     |
# +-----+------+------------------------------------------+
# only showing top 5 rowsres.printSchema()
# root
#  |-- y_lin: double (nullable = false)
#  |-- y_qua: double (nullable = false)
#  |-- created struct: struct (nullable = true)
#  |    |-- y_lin: double (nullable = true)
#  |    |-- y_qua: double (nullable = true)
```

上面的一个小麻烦是列`y_lin`和`y_qua`被命名了两次。很高兴在评论中听到如果这可以避免的话！

# **系列迭代器到系列迭代器，多个系列迭代器到系列迭代器**

当我们想要为每一批执行一次昂贵的操作时，迭代器变体是很方便的，例如，通过初始化一个模型。在下一个例子中，我们通过简单地为每批生成一个随机倍数来模拟这一点

```
# iterator of series to iterator of series
from typing import Iterator
@F.pandas_udf(T.DoubleType())
def multiply_as_iterator(col1: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # the random multiple is generated once per batch
    random_multiple = np.random.randint(1,10,1)[0]
    for s in col1:
       yield random_multiple*sres = df.select(F.col('y_lin'), multiply_as_iterator(F.col('y_lin')).alias('multiple of y_lin'))show_frame(res)
# +-----+-----------------+
# |y_lin|multiple of y_lin|
# +-----+-----------------+
# |0.055|0.496            |
# |1.123|10.111           |
# |2.134|19.209           |
# |3.209|28.881           |
# |4.300|38.698           |
# +-----+-----------------+
# only showing top 5 rows
```

如果我们想要控制批处理大小，我们可以在创建 spark 会话时将配置参数 spark . SQL . execution . arrow . maxrecordsperbatch 设置为所需的值。这只影响像 pandas UDFs 这样的迭代器，即使我们使用一个分区，它也适用。

从多个系列的迭代器到系列的迭代器相当简单，如下图所示，我们在对两列求和后应用了倍数

```
# iterator of multiple series to iterator of series
from typing import Iterator, Tuple
@F.pandas_udf(T.DoubleType())
def multiply_as_iterator2(col1: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.Series]:
    # the random multiple is generated once per batch
    random_multiple = np.random.randint(1,10,1)[0]
    for s1, s2 in col1:
        yield random_multiple*(s1 + s2)

res = df.select(F.col('y_lin'), F.col('y_qua'), multiply_as_iterator2(F.col('y_lin'), F.col('y_qua')).alias('multiple of y_lin + y_qua'))show_frame(res)
# +-----+------+-------------------------+
# |y_lin|y_qua |multiple of y_lin + y_qua|
# +-----+------+-------------------------+
# |0.055|0.284 |1.693                    |
# |1.123|1.524 |13.238                   |
# |2.134|3.765 |29.495                   |
# |3.209|7.636 |54.225                   |
# |4.300|13.841|90.704                   |
# +-----+------+-------------------------+
# only showing top 5 rows
```

函数定义稍微复杂一些，因为我们需要构造一个包含熊猫系列的元组的迭代器。

# **数据帧的迭代器到数据帧的迭代器**

数据帧的迭代器到数据帧的迭代器的转换类似于多重序列的迭代器到序列的迭代器。当我们需要在完整的数据框上而不是在选定的列上执行 pandas 操作时，这是首选的方法。

作为一个简单的例子，考虑最小-最大归一化

```
# iterator of data frame to iterator of data frame
from typing import Iterator
def min_max_normalise(frames: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    for frame in frames:
        yield (frame-frame.mean())/(frame.max()-frame.min())
schema = T.StructType([
    StructField('y_lin', T.DoubleType()),
    StructField('y_qua', T.DoubleType()),
])

res = df.select(F.col('y_lin'), F.col('y_qua')).mapInPandas(min_max_normalise, schema=schema)show_frame(res)
# +------+------+
# |y_lin |y_qua |
# +------+------+
# |-0.497|-0.378|
# |-0.245|-0.287|
# |-0.007|-0.121|
# |0.246 |0.164 |
# |0.503 |0.622 |
# +------+------+
# only showing top 5 rows
```

首先要注意的是需要向`mapInPandas`方法提供一个模式，并且不需要装饰器。`mapInPandas`方法可以改变返回数据帧的长度。同样，迭代器模式意味着数据帧不是作为一个整体进行最小-最大归一化，而是分别对每一批进行最小-最大归一化。

# 序列到标量和多个序列到标量

可以使用或不使用拆分-应用-组合模式将一个系列聚合为标量。典型地，使用分组的分离-应用-组合被应用，否则整个列将被带到驱动程序，这首先违背了使用 Spark 的目的。作为一个简单的例子，我们使用另一列进行分组来计算一列的平均值

```
# series to scalar
@F.pandas_udf(T.DoubleType())
def average_column(col1: pd.Series) -> float:
    return col1.mean()

res = df.groupby('group').agg(average_column(F.col('y_lin')).alias('average of y_lin'))show_frame(res)
# |group  |average of y_lin|
# +-------+----------------+
# |group a|9.509           |
# |group b|10.577          |
# +-------+----------------+
```

这是一个人为的例子，因为没有必要用熊猫 UDF，而是用普通香草派斯帕克

```
res = df.groupby('group').agg(F.mean(F.col('y_lin')).alias('average of y_lin'))show_frame(res)
# |group  |average of y_lin|
# +-------+----------------+
# |group a|9.509           |
# |group b|10.577          |
# +-------+----------------+
```

也可以将一组列简化为标量，例如通过计算两列之和的平均值

```
# multiple series to scalar
@F.pandas_udf(T.DoubleType())
def average_column(col1: pd.Series, col2: pd.Series) -> float:
    return (col1 + col2).mean()

res = df.groupby('group').agg(average_column(F.col('y_lin'), F.col('y_qua')).alias('average of y_lin + y_qua'))show_frame(res)
# +-------+------------------------+
# |group  |average of y_lin + y_qua|
# +-------+------------------------+
# |group a|104.770                 |
# |group b|121.621                 |
# +-------+------------------------+
```

# **组图 UDF**

在迄今为止的例子中，除了(多个)系列到标量，我们无法控制批次组成。串行到串行 UDF 将在分区上操作，而串行到串行 UDF 的迭代器将在每个分区的批处理上操作。在本文使用的示例数据框中，我们包含了一个名为 group 的列，我们可以用它来控制批次的组成。在现实生活中，需要注意确保该批具有熊猫一样的大小，以避免出现内存不足的异常。

使用组图 UDF，我们可以输入熊猫数据框并生成熊猫数据框。一个简单的例子标准化了数据帧:

```
# group map UDF
def standardise_dataframe(df1: pd.DataFrame) -> pd.DataFrame:
    tmp = df1[['y_lin', 'y_qua']]
    return (tmp - tmp.mean())/tmp.std()
schema = T.StructType([
    T.StructField('y_lin', T.DoubleType()),
    T.StructField('y_qua', T.DoubleType()),
])res = df.groupby('group').applyInPandas(standardise_dataframe, schema=schema)show_frame(res)
# +------+------+
# |y_lin |y_qua |
# +------+------+
# |-1.485|-1.009|
# |-1.158|-0.972|
# |-0.818|-0.865|
# |-0.500|-0.691|
# |-0.170|-0.443|
# +------+------+
# only showing top 5 rows
```

默认情况下不包含组名，需要在返回的数据框和方案中明确添加，例如使用…

```
# group map UDF
def standardise_dataframe(df1: pd.DataFrame) -> pd.DataFrame:
    tmp = df1[['y_lin', 'y_qua']]
    return pd.concat([df1['group'], (tmp - tmp.mean())/tmp.std()], axis='columns')
schema = T.StructType([
    T.StructField('group', T.StringType()),
    T.StructField('y_lin', T.DoubleType()),
    T.StructField('y_qua', T.DoubleType()),
])res = df.groupby('group').applyInPandas(standardise_dataframe, schema=schema)show_frame(res)
# +-------+------+------+
# |group  |y_lin |y_qua |
# +-------+------+------+
# |group a|-1.485|-1.009|
# |group a|-1.158|-0.972|
# |group a|-0.818|-0.865|
# |group a|-0.500|-0.691|
# |group a|-0.170|-0.443|
# +-------+------+------+
# only showing top 5 rows
```

组图 UDF 可以更改返回的数据框的形状。例如，我们将通过对 y_lin 和 y_qua 列拟合二次多项式来计算系数

```
# group map UDF
def fit_polynomial(df1: pd.DataFrame) -> pd.DataFrame:
    tmp = df1[['x', 'y_lin', 'y_qua']]
    # see https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html
    poly_lin = np.polynomial.polynomial.Polynomial.fit(x=tmp['x'], y=tmp['y_lin'], deg=2).convert().coef
    poly_qua = np.polynomial.polynomial.Polynomial.fit(x=tmp['x'], y=tmp['y_qua'], deg=2).convert().coef
    df2 = pd.DataFrame({'group': df1['group'].iloc[0], 'y_lin fitted coffficients':  [poly_lin.tolist()], 'y_qua fitted coefficients': [poly_qua.tolist()]})
    return df2
schema = T.StructType([
    T.StructField('group', T.StringType()),
    T.StructField('y_lin fitted coefficients', T.ArrayType(T.DoubleType())),
    T.StructField('y_qua fitted coefficients', T.ArrayType(T.DoubleType())),
])

res = df.groupby('group').applyInPandas(fit_polynomial, schema=schema)
show_frame(res)show_frame(res)
# +-------+---------------------------------------------------------------+--------------------------------------------------------------+
# |group  |y_lin fitted coefficients                                      |y_qua fitted coefficients                                     |
# +-------+---------------------------------------------------------------+--------------------------------------------------------------+
# |group a|[0.05226283524780051, 1.9935642550858421, 4.346056274066657E-4]|[0.24210502752802654, 0.14937331848708446, 2.9865040888654355]|
# |group b|[0.07641111656270816, 1.9894934336694825, 8.012896992570311E-4]|[0.38970142737052527, 0.10989441330142924, 2.9877883688982467]|
```

返回的列是数组。我们可以看到，考虑到添加到原始数据帧的噪声并不过分，这些系数非常接近预期值。我们还看到两组给出了非常相似的系数。

# 最后的想法

Pandas UDFs 很好地补充了 PySpark API，并允许更具表现力的数据操作。PySpark 发展迅速，从 2.x 版到 3.x 版的变化非常显著。虽然本文涵盖了许多当前可用的 UDF 类型，但可以肯定的是，随着时间的推移，将会引入更多的可能性，因此在决定使用哪一种之前，最好先查阅文档。