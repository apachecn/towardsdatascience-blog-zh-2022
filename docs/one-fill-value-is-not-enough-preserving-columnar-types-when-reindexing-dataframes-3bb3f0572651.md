# 一个填充值是不够的:重新索引数据帧时保留列类型

> 原文：<https://towardsdatascience.com/one-fill-value-is-not-enough-preserving-columnar-types-when-reindexing-dataframes-3bb3f0572651>

![](img/359d47a9996add47fce92e6bd1d7210e.png)

当处理数据帧时，重新索引是很常见的。当数据帧被重新索引时，旧的索引(及其相关值)与新的索引相一致，可能重新排序、收缩或扩展行或列。当重新索引扩展数据帧时，需要新的值来填充新创建的行或列:这些是“填充值”

当使用 Pandas 重新索引时，通过`fill_value`参数，只允许一个值。如果`fill_value`的类型与一个或多个列的类型不兼容，则该列将被重新转换为不同的、可能不需要的类型。

例如，给定一个具有三列类型 object、integer 和 Boolean 的 DataFrame，缺省情况下，重新索引索引会用 NaN 填充新行，NaN 是一种 float 类型，它强制将 integer 列转换为 float，将 Boolean 列转换为 object。

```
>>> df = pd.DataFrame.from_records((('a', 1, True), ('b', 2, False)), columns=tuple('xyz'))
>>> df
   x  y      z
0  a  1   True
1  b  2  False>>> df.dtypes.tolist()
[dtype('O'), dtype('int64'), dtype('bool')]>>> df.reindex((1, 0, 2))
     x    y      z
1    b  2.0  False
0    a  1.0   True
2  NaN  NaN    NaN>>> df.reindex((1, 0, 2)).dtypes.tolist()
[dtype('O'), dtype('float64'), dtype('O')]
```

柱状退化通常是有害的。预先存在的柱状类型可能适合数据；仅仅因为重新索引而不必要地更改该类型通常是意想不到的。从一种 C 级 NumPy 类型转换到另一种类型，比如从 int 转换到 float，可能是可以忍受的。但是当从 C 级 NumPy 类型转换到 Python 对象数组(object dtypes)时，性能会下降。和熊猫重配的时候，这个问题是没有办法避免的。

StaticFrame 是一个不可变的数据框架库，为这类问题提供了解决方案。在 StaticFrame 中，替代填充值表示可用于在重新索引、移位和许多其他需要`fill_value`参数的操作中保留列类型。对于异构类型列数据的操作，一个填充值是远远不够的。

StaticFrame 支持将`fill_value`作为单个元素、一个行长度的值列表、一个列标签映射或者一个`FillValueAuto`，一个定义类型到值映射的新对象。

所有示例都使用 Pandas 1.4.3 和 StaticFrame 0.9.6。导入使用以下惯例:

```
>>> import pandas as pd
>>> import static_frame as sf
```

我们可以通过用一个填充值 NaN 重新索引同一个数据帧，在 StaticFrame 中重现熊猫的行为。这导致了和熊猫一样的柱状类型。注意，默认情况下，StaticFrame 显示每一列的 dtype，这使得列类型的退化变得非常明显。

```
>>> f = sf.Frame.from_records((('a', 1, True), ('b', 2, False)), columns=tuple('xyz'))
>>> f
<Frame>
<Index> x     y       z      <<U1>
<Index>
0       a     1       True
1       b     2       False
<int64> <<U1> <int64> <bool>>>> f.reindex((1, 0, 2), fill_value=np.nan)
<Frame>
<Index> x        y         z        <<U1>
<Index>
1       b        2.0       False
0       a        1.0       True
2       nan      nan       nan
<int64> <object> <float64> <object>
```

避免重新索引时类型退化的一种方法是为每列提供一个填充值。使用 StaticFrame，可以为填充值提供一个列表，为每列提供一个值:

```
>>> f.reindex((1, 0, 2), fill_value=['', 0, False])
<Frame>
<Index> x     y       z      <<U1>
<Index>
1       b     2       False
0       a     1       True
2             0       False
<int64> <<U1> <int64> <bool>
```

或者，可以使用字典来提供列标签到填充值的映射。如果未提供标签，将提供默认值(NaN)。

```
>>> f.reindex((1, 0, 2), fill_value={'z':False, 'x':''})
<Frame>
<Index> x     y         z      <<U1>
<Index>
1       b     2.0       False
0       a     1.0       True
2             nan       False
<int64> <<U1> <float64> <bool>
```

前面的例子都要求每列有一个显式值，以提供最大的特异性。在许多情况下(尤其是对于较大的数据帧)，需要一种更通用的方式来指定填充值。

一种选择是基于特定的 NumPy 数据类型映射填充值。这种方法被拒绝，因为 NumPy dtype 定义了一个以字节为单位的变量“itemsize ”,导致了大量可能的 NumPy dtype。更有可能的是，相同的填充值将用于独立于 itemsize 的 dtypes 族；例如，所有大小的整数(int8、int16、int32 和 int64)。

为了识别与大小无关的类型族，我们可以使用 dtype“kind”。NumPy dtypes 有一个独立于 dtype itemsize 的“kind”属性:例如，int8、int16、int32 和 int64 dtypes 都被标记为 kind“I”。如下所示，有 11 种数据类型，每种都有一个字符标签:

*   布尔
*   i: int
*   u: uint
*   外宾:浮动
*   丙:复杂
*   男:时差
*   m:日期时间
*   o:反对
*   s:字节
*   U: str
*   v:无效

为每种数据类型指定一个填充值提供了一种方便的方法来避免列类型强制，同时不需要为每列指定一个繁琐的规范。为此，StaticFrame 引入了一个新对象:`FileValueAuto`。

使用类`FillValueAuto`作为填充值为所有 dtype 类型提供了无类型强制的缺省值。如果需要不同的映射，可以创建一个`FillValueAuto`实例，为每种数据类型指定一个填充值。

回到前面的重新索引示例，我们看到了使用`FillValueAuto`类的便利，并且所有列类型都被保留:

```
>>> f
<Frame>
<Index> x     y       z      <<U1>
<Index>
0       a     1       True
1       b     2       False
<int64> <<U1> <int64> <bool>>>> f.reindex((1, 0, 2), fill_value=sf.FillValueAuto)
<Frame>
<Index> x     y       z      <<U1>
<Index>
1       b     2       False
0       a     1       True
2             0       False
<int64> <<U1> <int64> <bool>
```

如果我们需要偏离提供的`FillValueAuto`缺省值，可以创建一个实例，指定每种数据类型的填充值。初始化器的关键字参数是单字符数据类型种类标签。

```
>>> f.reindex((1, 0, 2), fill_value=sf.FillValueAuto(U='x', i=-1, b=None))
<Frame>
<Index> x     y       z        <<U1>
<Index>
1       b     2       False
0       a     1       True
2       x     -1      None
<int64> <<U1> <int64> <object>
```

在 StaticFrame 中，几乎在任何需要填充值的地方，都接受相同数量的填充值类型。例如，在移位数据中，必须提供填充值；但是当移动异构类型的整个数据帧时，一个填充值是不够的。如下所示，默认的`fill_value`，NaN，强制所有列类型要么是 object 要么是 float。

```
>>> f = sf.Frame.from_records((('a', 1, True, 'p', 23.2), ('b', 2, False, 'q', 85.1), ('c', 3, True, 'r', 1.23)), columns=tuple('abcde'))>>> f.shift(2)
<Frame>
<Index> a        b         c        d        e         <<U1>
<Index>
0       nan      nan       nan      nan      nan
1       nan      nan       nan      nan      nan
2       a        1.0       True     p        23.2
<int64> <object> <float64> <object> <object> <float64>
```

和以前一样，使用一个`FillValueAuto`实例允许一个通用的填充值规范，它完全避免了列类型的退化。

```
>>> f.shift(2, fill_value=sf.FillValueAuto(U='', b=False, f=0, i=0))
<Frame>
<Index> a     b       c      d     e         <<U1>
<Index>
0             0       False        0.0
1             0       False        0.0
2       a     1       True   p     23.2
<int64> <<U1> <int64> <bool> <<U1> <float64>
```

在二元运算符的许多应用中也需要填充值。一般来说，对带标签的数据进行二元运算会强制操作数重新索引到联合索引，这可能会引入缺失值。如果缺少的值仅为 NaN，则可能会重新转换生成的列类型。

例如，给定两个数据帧，每个数据帧都有一个 float 和一个 integer 列，二元运算将为重新索引的值引入 NaN，将 integer 列强制转换为 float。这可以通过使用`FillValueAuto`在 StaticFrame 中避免。

由于二元操作符不接受参数，StaticFrame 提供了`via_fill_value`接口，允许在二元操作中需要重新索引时指定要使用的填充值。这类似于熊猫`DataFrame.multiply()`和相关方法提供的功能。有了 StaticFrame 的`via_fill_value`，我们可以继续使用任意二元运算符的表达式。

当将两个数据帧相乘时，每个数据帧都有一列浮点数和一列整数，由于重新索引而引入的 nan 会将所有值强制转换为浮点数。

```
>>> f1 = sf.Frame.from_records(((10.2, 20), (2.4, 4)), index=('a', 'b'))
>>> f2 = sf.Frame.from_records(((3.4, 1), (8.2, 0)), index=('b', 'c'))>>> f1 * f2
<Frame>
<Index> 0         1         <int64>
<Index>
a       nan       nan
b       8.16      4.0
c       nan       nan
<<U1>   <float64> <float64>
```

通过使用`via_fill_value`和`FillValueAuto`，我们可以保留列类型，即使需要重新索引，并且继续在表达式中使用二元运算符。

```
>>> f1.via_fill_value(sf.FillValueAuto) * f2
<Frame>
<Index> 0         1       <int64>
<Index>
a       nan       0
b       8.16      4
c       nan       0
<<U1>   <float64> <int64>
```

上面使用的只有几列的例子并没有完全展示出`FillValueAuto`的威力:当处理成百上千列的异构类型数据帧时，规范的通用性提供了一个简洁而强大的工具。

由重新索引或其他转换导致的无意类型强制的成本可能会导致错误或性能下降。StaticFrame 灵活的填充值类型，以及新的`FillValueAuto`，为这些实际问题提供了解决方案。