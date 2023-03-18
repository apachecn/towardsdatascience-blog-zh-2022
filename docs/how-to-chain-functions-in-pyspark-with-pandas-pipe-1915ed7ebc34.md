# 如何用管道链接 Pyspark 中的函数

> 原文：<https://towardsdatascience.com/how-to-chain-functions-in-pyspark-with-pandas-pipe-1915ed7ebc34>

## 用熊猫管法的 pyspark 等效链多函数

# 动机

如果你是一个 Pandas 用户，你可能遇到过 Pandas DataFrame `.pipe()`方法，它允许用户对数据帧或系列应用可链接的函数。如果你对熊猫`.pipe()`不熟悉，这里有一个快速的例子和开始使用它的动机。

![](img/595c99e566328b8377612cc718e98197.png)

照片由 [Helio Dilolwa](https://unsplash.com/@dilolwa?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

**什么是熊猫** `**.pipe()**` **？**

**参数**

*   `func`:应用于系列/数据框的功能
*   `args`:位置参数传递给`func`
*   `kwargs`:传递给`func`的关键字参数

**退货**

*   `object`:返回类型`func`

在本例中，我们有 3 个函数对数据帧进行操作，`f1`、`f2`和`f3`，每个函数都需要一个数据帧和参数作为输入，并返回转换后的数据帧。

```
def f1(df, arg1):
	# do something	return # a dataframedef f2(df, arg2):
	# do something	return # a dataframedef f3(df, arg3):
	# do something	return # a dataframedf = pd.DataFrame(..) # some dataframe
```

如果不使用`.pipe()`，我们将以嵌套的方式应用函数，如果有多个函数，这可能看起来很难理解。

```
f1(f2(f3(df, arg3 = arg3), arg2 = arg2), arg1 = arg1)
```

为了遵循函数执行的顺序，必须从“由内向外”阅读。首先执行最内部的功能`f3`，然后执行`f2`，最后执行`f1`。

`.pipe()`避免嵌套，允许使用点符号(`.`)链接函数，使其更具可读性。`.pipe()`还允许传递位置参数和关键字参数，并假设函数的第一个参数指向输入数据帧/序列。

```
df.pipe(f3, arg3 = arg3).pipe(f2, arg2 = arg2).pipe(f1, arg1 = arg1)
```

遵循与`.pipe()`链接在一起的功能的执行顺序更直观；我们只是从左向右读。

Pyspark 是大数据处理中熊猫的流行替代品。在本文中，我们将研究如何使用以下方法链接 pyspark 函数:

1.  派斯帕克的`.transform()`方法
2.  创建一个 Pyspark，相当于熊猫的`.pipe()`方法

# 数据

这是我们将在示例中使用的合成数据。

```
df = spark.createDataFrame([('John', 30, 180, 90, 'm'), ('Karen', 27, 173, 60, 'f')], ['name', 'age', 'height', 'weight', 'gender'])
df.show()+-----+---+------+------+------+
| name|age|height|weight|gender|
+-----+---+------+------+------+
| John| 30|   180|    90|     m|
|Karen| 27|   173|    60|     f|
+-----+---+------+------+------+
```

该数据集包含两个人 John 和 Karen 的年龄、身高、体重和性别信息。任务是:

1.  找到他们的身体质量指数(身体质量指数)
2.  将`gender`列中的小写字符串转换为大写

我们创建以下函数来计算身体质量指数，它是以千克为单位的重量除以以米为单位的高度的平方。

```
def calculate_bmi(df):

  df = df.withColumn('bmi', f.round(f.col('weight')/(f.col('height')/100)**2,2))

  return df
```

# 改变

`pyspark.sql.DataFrame.transform`是链接自定义转换的简明语法。

**参数**

*   `func`:获取并返回数据帧的函数

以下是如何在数据帧上应用`calculate_bmi`功能。

```
df.transform(calculate_bmi).show()+-----+---+------+------+-----+
| name|age|height|weight|  bmi|
+-----+---+------+------+-----+
| John| 30|   180|    90|27.78|
|Karen| 27|   173|    60|20.05|
+-----+---+------+------+-----+
```

我们可以将多个转换和其他 DataFrame 方法链接在一起。在下面的例子中，我们计算了身体质量指数并将`gender`列大写。

```
(df
 .transform(calculate_bmi)
 .withColumn('gender', f.upper('gender'))
 .show())+-----+---+------+------+------+-----+
| name|age|height|weight|gender|  bmi|
+-----+---+------+------+------+-----+
| John| 30|   180|    90|     M|27.78|
|Karen| 27|   173|    60|     F|20.05|
+-----+---+------+------+------+-----+
```

与 Pandas 的`.pipe()`不同，Pyspark 的`.transform()`不向输入函数(`func`)传递参数。克服这一限制的一种方法是使用 python 的部分函数。

让我们扩展一下身体质量指数的例子。我们发现体重和身高机器没有很好地校准，因此我们必须对测量的身高和体重进行补偿，以获得正确的身体质量指数。下面是实现这一点的函数。

```
def offset(df, height_offset, weight_offset):

  df = (df
        .withColumn('height', f.col('height') + height_offset)
        .withColumn('weight', f.col('weight') + weight_offset)
       )

  return df
```

为了使用`.transform()`在数据帧上应用偏移函数，我们使用 python 的`functools.partial`方法来创建分部函数，该函数允许我们将参数传递给`offset`函数。

```
from functools import partial(df
 .transform(partial(offset, height_offset = 2, weight_offset = -0.5))
 .transform(calculate_bmi)
 .withColumn('gender', f.upper('gender'))
 .show())+-----+---+------+------+------+-----+
| name|age|height|weight|gender|  bmi|
+-----+---+------+------+------+-----+
| John| 30|   182|  89.5|     M|27.02|
|Karen| 27|   175|  59.5|     F|19.43|
+-----+---+------+------+------+-----+
```

或者，我们可以创建一个相当于 Pandas `.pipe()`的 pyspark，它接受位置和关键字参数。

# 自定义 Pyspark 管道方法

我们创建了一个`.pipe()`函数，并执行了一个 monkey 补丁，将它作为`pyspark.sql.DataFrame`类的一个方法包含进来。

```
from pyspark.sql import DataFramedef pipe(self, func, *args, **kwargs):
    return func(self, *args, **kwargs)DataFrame.pipe = pipe
```

**参数**

*   `func`:数据框上应用的输入功能
*   `args`:位置参数
*   `kwargs`:关键字参数

让我们来看一些例子。

**无参数函数**

与`.transform()`类似，`.pipe()`可以接受没有任何参数的输入函数。

```
df.pipe(calculate_bmi).show()+-----+---+------+------+------+-----+
| name|age|height|weight|gender|  bmi|
+-----+---+------+------+------+-----+
| John| 30|   180|    90|     m|27.78|
|Karen| 27|   173|    60|     f|20.05|
+-----+---+------+------+------+-----+
```

**λ函数**

`.pipe()`也可以接受 lambda 函数。

```
df.pipe(lambda df: df.withColumn('bmi', f.round(f.col('weight')/(f.col('height')/100)**2,2))).show()+-----+---+------+------+------+-----+
| name|age|height|weight|gender|  bmi|
+-----+---+------+------+------+-----+
| John| 30|   180|    90|     m|27.78|
|Karen| 27|   173|    60|     f|20.05|
+-----+---+------+------+------+-----+
```

**带关键字参数的函数**

由`.pipe()`接收的关键字参数将被传递给`offset`函数。我们还可以将多个`.pipe()`功能链接在一起。

```
(df
 .pipe(offset, height_offset = 2, weight_offset = -0.5)
 .pipe(calculate_bmi)
 .withColumn('gender', f.upper('gender'))
 .show())+-----+---+------+------+------+-----+
| name|age|height|weight|gender|  bmi|
+-----+---+------+------+------+-----+
| John| 30|   182|  89.5|     M|27.02|
|Karen| 27|   175|  59.5|     F|19.43|
+-----+---+------+------+------+-----+
```

**带有位置参数的函数**

`.pipe()`也适用于位置参数。

```
(df
 .pipe(offset, 2, -0.5)
 .pipe(calculate_bmi)
 .withColumn('gender', f.upper('gender'))
 .show())+-----+---+------+------+------+-----+
| name|age|height|weight|gender|  bmi|
+-----+---+------+------+------+-----+
| John| 30|   182|  89.5|     m|27.02|
|Karen| 27|   175|  59.5|     f|19.43|
+-----+---+------+------+------+-----+
```

# 摘要

在本文中，我们给出了链接函数的动机。在 Pyspark 中将定制函数链接在一起的本地方法是使用`pyspark.sql.DataFrame.transform`方法。这种方法的缺点是不允许向输入函数传递参数。为了克服这个限制，我们演示了如何将`.transform()`与`functools.partial`结合使用，或者在 Pyspark 中创建一个自定义的`.pipe()`方法。