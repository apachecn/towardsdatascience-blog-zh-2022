# 使用高阶容器高效处理 7，163(或更多)个数据帧

> 原文：<https://towardsdatascience.com/using-higher-order-containers-to-efficiently-process-7-163-or-more-dataframes-964da8b0c679>

## [理解大数据](https://towardsdatascience.com/tagged/making-sense-of-big-data)

# 使用高阶容器高效处理 7，163(或更多)个数据帧

## 介绍 StaticFrame 总线、批次、被子和纱线

![](img/d3ffc3db1d41683246163e552b1ee4ea.png)

作者图片

数据帧处理例程通常与表集合一起工作。这种集合的示例包括每年有一个表的多年数据集、每个股票行情表的历史股票数据，或者 XLSX 文件中多个工作表的数据。本文介绍了用于处理这种数据帧集合的新颖的“高阶”容器，这些容器在 Python StaticFrame 包中实现(一个提供不可变数据帧的 Pandas 替代品)。三个核心容器是`Bus`、`Batch`和`Quilt`。将简要介绍第四种容器`Yarn`。

这些高阶容器的替代方法是使用一个带有层次索引的大表。例如，多只股票的时间序列数据可能被编码在一个具有两层索引的表中，外层是股票代码，内层是日期。这种方法通常效率很低，因为即使只处理少量股票，也必须将整个表加载到内存中。

本文介绍了用于处理大型表集合的容器，从几千到几十万个表。通过延迟加载和可选的快速卸载提供了高效的内存使用。`Bus`(以电路中使用的总线命名)，为延迟加载存储在磁盘上的表集合提供了一个类似字典的接口；收藏可以存储在 SQLite、HDF5、XLSX 或压缩的拼花、NPZ、pickle 或分隔文本文件中。`Batch`(以批处理命名)是一个表的延迟处理器，提供了一个简洁的接口来延迟定义应用于所有表的操作。`Quilt`(以拼凑而成的纺织品命名)是所有表的一种惰性虚拟连接，允许对分区数据进行操作，就像它是一个统一的单个表一样。

所有这三个容器都提供了相同的接口，用于读取和写入上面提到的多表存储格式(SQLite、HDF5、XLSX 或拼花、NPZ、pickle 或分隔文本文件的压缩存档)。这种一致性允许在不同的上下文中重用相同的数据存储。

这些工具是从我的工作环境演变而来的:处理金融数据和建立投资系统模型。在那里，数据集自然地按日期或特征划分。对于历史模拟，所需的数据可能很大。`Bus`、`Batch`、`Quilt`、`Yarn`在这个领域提供了便捷高效的工具。像 Vaex 和 Dask 这样的核心外解决方案提供了处理大量数据的相关方法，尽管有不同的权衡。

虽然这些容器是在 StaticFrame 中实现的，但是这些抽象对于任何数据帧或表处理库中的应用程序都是有用的。StaticFrame 将数据帧简单地称为“帧”，这里将使用该约定。StaticFrame 按照以下约定导入:

```
>>> import static_frame as sf
```

# 容器概述

在演示使用这些容器处理成千上万个数据帧之前，我们将从处理两个数据帧开始。在创建了一个带有两个`Frame`的`Bus`之后，我们将使用同一个`Bus`来初始化一个`Batch`、`Quilt`和`Yarn`。通过这种介绍，可以观察到共同的和不同的特征。

## 公共汽车

两个简单的`Frame`可以用来演示初始化一个`Bus`。`Bus.from_items()`方法接受成对的标签和`Frame`；条目可以在元组中提供(如下所示),或者通过 Python 字典或相关容器上的`items()`方法提供。

```
>>> f1 = sf.Frame.from_element(0.5, index=('w', 'x'), 
    columns=('a', 'b'))
>>> f2 = sf.Frame.from_element(2, index=('y', 'z'), 
    columns=('a', 'b'))>>> bus = sf.Bus.from_items((('f1', f1), ('f2', f2)))>>> bus
<Bus>
<Index>
f1      Frame
f2      Frame
<<U2>   <object>
```

`Bus`可以被认为是`Frame`的`Series`(或有序字典)，允许访问给定标签的`Frame`。

```
>>> bus.loc['f2']
<Frame: f2>
<Index>     a       b       <<U1>
<Index>
y           2       2
z           2       2
<<U1>       <int64> <int64>
```

`Bus`的一个关键特性是，当从磁盘读取时，`Frame`被延迟加载:一个`Frame`仅在被访问时被加载到内存中，并且(通过`max_persist`参数)`Bus`可以被配置为仅保存对有限数量的`Frame`的强引用，急切地卸载超出该限制的最近最少使用的。这允许检查所有的`Frame`，同时限制`Bus`加载的总内存。

由于`Bus`支持读取和写入 XLSX 和 HDF5(以及许多其他格式)，它提供了熊猫`ExcelWriter`和`HDFStore`接口的功能，但具有更通用和一致的接口。同样的`Bus`可以用于编写 XLSX 工作簿(其中每一帧都是一张表)或 HDF5 数据存储，只需分别使用`Bus.to_xlsx()`或`Bus.to_hdf5()`。

此外，`Bus`也是创建`Batch`、`Quilt`或`Yarn`的便利资源。

## 一批

`Batch`可以被认为是一个 label 和`Frame`对的迭代器。除了迭代器之外，`Batch`还是一个在每个包含的`Frame`上组合延迟操作的工具。`Batch`几乎暴露了整个`Frame`接口；当被调用时，方法调用和操作符应用程序在新返回的`Batch`中被延迟，在存储的迭代器上组成延迟执行。只有在使用`Batch.to_frame()`方法创建复合`Frame`或使用类似字典的迭代器(如`Batch.keys()`、`Batch.items()`或`Batch.values`)时，才会执行操作和迭代对。

一个`Batch`可以用来自一个`Bus`或者任何一对标签的迭代器`Frame`的条目来初始化。从一个`Batch`中调用的方法或操作符只是返回一个新的`Batch`。调用`Batch.to_frame()`，如下图所示，是急切执行组合`sum()`操作所必需的。

```
>>> sf.Batch(bus.items()).sum()
<Batch at 0x7fabd09779a0>>>> sf.Batch(bus.items()).sum().to_frame()
<Frame>
<Index> a         b         <<U1>
<Index>
f1      1.0       1.0
f2      4.0       4.0
<<U2>   <float64> <float64>
```

除了`Frame`方法外，`Batch`还支持使用`Frame`选择接口和操作符。下面，每个`Frame`取二次幂，选择“b”列，并返回新的`Frame`(组合两个选择):

```
>>> (sf.Batch(bus.items()) ** 2)['b'].to_frame()
<Frame>
<Index> w         x         y         z         <<U1>
<Index>
f1      0.25      0.25      nan       nan
f2      nan       nan       4.0       4.0
<<U2>   <float64> <float64> <float64> <float64>
```

`Batch`与熊猫`DataFrameGroupBy`和`Rolling`对象相关，在配置分组或滚动窗口可迭代后，这些接口暴露这些组或窗口上的函数应用。`Batch`概括了这一功能，支持这些上下文，并提供标签和框架的任何迭代器的通用处理。

## 被子

一个`Quilt`用一个`Bus`(或`Yarn`)初始化，并要求指定虚拟连接哪个轴，垂直(轴 0)或水平(轴 1)。此外，`Quilt`必须为`retain_labels`定义一个布尔值:如果为真，`Frame`标签将作为外部标签保留在沿着连接轴的分层索引中。如果`retain_labels`为假，所有包含的`Frame`的串联轴上的所有标签必须是唯一的。以下示例使用先前创建的`Bus`来演示`retain_labels`参数。由于一个`Quilt`可能由数千个表组成，所以默认的表示形式会简化数据；`Quilt.to_frame()`可用于提供完全实现的表示。

```
>>> quilt = sf.Quilt(bus, axis=0, retain_labels=False)>>> quilt
<Quilt>
<Index: Aligned>      a b <<U1>
<Index: Concatenated>
w                     . .
x                     . .
y                     . .
z                     . .
<<U1>>>> quilt.to_frame()
<Frame>
<Index> a         b         <<U1>
<Index>
w       0.5       0.5
x       0.5       0.5
y       2.0       2.0
z       2.0       2.0
<<U1>   <float64> <float64>>>> quilt = sf.Quilt(bus, axis=0, retain_labels=True)>>> quilt.to_frame()
<Frame>
<Index>                a         b         <<U1>
<IndexHierarchy>
f1               w     0.5       0.5
f1               x     0.5       0.5
f2               y     2.0       2.0
f2               z     2.0       2.0
<<U2>            <<U1> <float64> <float64>
```

`Quilt`可以被认为是由许多较小的`Frame`组成的`Frame`，垂直或水平排列。重要的是，这个更大的`Frame`并没有急切地串联起来；更确切地说，根据需要从包含的`Bus`中访问`Frame`,提供了沿轴的表的惰性连接。

可以用`max_persist`参数配置`Quilt`中的`Bus`来限制保存在内存中的`Frame`的总数。这种显式内存管理允许在可能太大而无法加载到内存中的虚拟`Frame`上进行操作。

`Quilt`允许使用公共`Frame`接口的子集对这个虚拟连接的`Frame`进行选择、迭代和操作。例如，`Quilt`可用于迭代行和应用函数:

```
>>> quilt.iter_array(axis=1).apply(lambda a: a.sum())
<Series>
<Index>
w        1.0
x        1.0
y        4.0
z        4.0
<<U1>    <float64>
```

## 故事

这里仅简要描述的`Yarn`提供了一个或多个`Bus`的“虚级联”。与`Quilt`号一样，较大的容器不会被急切地连接起来。与`Quilt`的二维、单个`Frame`呈现不同，`Yarn`呈现了一个包含许多帧的一维容器，具有类似`Bus`的界面。与`Bus`或`Quilt`不同，`Yarn`的索引可以任意重新标记。这些功能允许异构的`Bus`在新标签下(如果需要)在单个容器中可用。

作为更高阶的容器，`Yarn`只能用一个或多个`Bus`或`Yarn`初始化。一个`Yarn`甚至可以从同一个`Bus`的多个实例中创建，如果每个实例都有一个唯一的`name`:

```
>>> sf.Yarn.from_buses((bus.rename('a'), bus.rename('b')),
    retain_labels=True)
<Yarn>
<IndexHierarchy>
a                f1    Frame
a                f2    Frame
b                f1    Frame
b                f2    Frame
<<U1>            <<U2> <object>
```

## 共同特征和区别特征

`Bus`、`Batch`和`Quilt`的一个共同特征是，它们都支持来自标签对和`Frame`的迭代器的实例化。当迭代器来自`Bus`时，`Bus`的延迟加载可以用来最小化内存开销。

这些容器都共享相同的基于文件的构造函数，如`from_zip_csv()`或`from_xlsx()`；每个构造器都有一个对应的导出器，例如分别是`to_zip_csv()`或`to_xlsx()`，允许往返读写，或者从一种格式转换成另一种格式。下面的列表总结了所有三个容器中可用的基于文件的构造函数和导出函数。(`Yarn`作为`Bus`的聚合，只支持出口商。)

*   `from_hdf5`，`to_hdf5`
*   `from_sqlite`，`to_sqlite`
*   `from_zip_csv`，`to_zip_csv`
*   `from_zip_npz`，`to_zip_npz`
*   `from_zip_pickle`，`to_zip_pickle`
*   `from_zip_parquet`，`to_zip_parquet`
*   `from_zip_tsv`，`to_zip_tsv`
*   `from_xlsx`，`to_xlsx`

这些容器可以通过维度、形状和界面来区分。`Bus`和`Yarn`是`Frame`的一维集合；`Batch`和`Quilt`呈现类似于`Frame`的二维界面。虽然`Bus`的形状等于`Frame`的数量(或者对于`Yarn`而言，等于所有包含的`Bus`中的`Frame`的数量)，但是`Quilt`的形状取决于其包含的`Frame`及其定向轴。像发电机一样，`Batch`的长度(或形状)在迭代之前是未知的。最后，当`Bus`和`Yarn`暴露一个类似于`Series`的接口时，`Batch`和`Quilt`暴露一个类似于`Frame`的接口，分别在单独的`Frame`或虚拟连接的`Frame`上操作。

如下表所示，对于形状( *x* ， *y* )的*m*n，这些容器填充了一系列维度和接口。

*   `Bus`
    呈现尺寸:1
    近似界面:`Series`
    组成: *n* `Frame`
    呈现形状:( *n* ，)
*   `Batch`
    呈现维度:2
    近似接口:`Frame`
    组成:标签对的迭代器，`Frame`
*   `Quilt`
    呈现尺寸:2
    近似界面:`Frame`
    组成:1*n*`Frame`的`Bus`或`Yarn`呈现形状:( *xn* ， *y* 或( *x* ， *yn* )
*   `Yarn`
    呈现尺寸:1
    近似界面:`Series`
    组成:*m*`Bus`of*n*`Frame`
    呈现形状:( *mn* ，)

# 处理 7163 个数据帧

“庞大的股票市场数据集”包含 7163 个 CSV 表的集合，每个表代表一只美国股票的时间序列特征。“archive.zip”文件可从[https://www . ka ggle . com/Boris marjanovic/price-volume-data-for-all-us-stocks-ETFs](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)获得

打开归档文件后，我们可以从包含的“股票”目录中读取，并使用`Batch`创建股票数据的 zip pickle，用 ticker 标记，以便在后面的示例中快速阅读。由于有些文件是空的，我们还必须过滤掉没有大小的文件。根据硬件的不同，这种初始转换可能需要一些时间。

```
>>> import os
>>> d = 'archive/Stocks'
>>> fps = ((fn, os.path.join(d, fn)) for fn in os.listdir(d))
>>> items = ((fn.replace('.us.txt', ''), sf.Frame.from_csv(fp, index_depth=1)) for fn, fp in fps if os.path.getsize(fp))
>>> sf.Batch(items).to_zip_pickle('stocks.zip')
```

由于`Bus`是懒惰的，从这个新的 zip 存档初始化加载零个`Frame`到内存中。仅当明确请求时，才提供对数据的快速访问。因此，当`Bus.shape`属性显示 7163 个包含的`Frame`时，`status`属性显示零个加载的`Frame` s。

```
>>> bus = sf.Bus.from_zip_pickle('stocks.zip')
>>> bus.shape
(7163,)
>>> bus.status['loaded'].sum()
0
```

访问单个`Frame`只会加载那个`Frame`。

```
>>> bus['ibm'].shape                                                                                                                
(14059, 6)>>> bus['ibm'].columns                                                                                                              
<Index>
Open
High
Low
Close
Volume
OpenInt
<<U7>
```

提取多个`Frame`会产生一个新的`Bus`从同一个存储中读取。

```
>>> bus[['aapl', 'msft', 'goog']]
<Bus>
<Index>
aapl    Frame
msft    Frame
goog    Frame
<<U9>   <object>
>>> bus.status['loaded'].sum()
4
```

用一个`Batch`我们可以对包含在`Bus`中的`Frame`执行操作，返回标记的结果。`Batch.apply()`方法可与`lambda`一起使用，将每个`Frame`的两列(“音量”和“关闭”)相乘；然后，我们用`iloc`提取最近的两个值，并产生一个复合的`Frame`，该索引来自原始的`Bus`标签:

```
>>> sf.Batch(bus[['aapl', 'msft', 'goog']].items()
    ).apply(lambda f: f['Close'] * f['Volume']).iloc[-2:].to_frame()
<Frame>
<Index> 2017-11-09         2017-11-10         <<U10>
<Index>
aapl    5175673321.5       4389543386.98
msft    1780638040.5600002 1626767764.8700001
goog    1283539710.3       740903319.18
<<U4>   <float64>          <float64>
```

为了对整个数据集进行观察，我们可以将`Bus`传递给`Quilt`。下面，一个空片用于强制一次加载所有的`Frame`以优化`Quilt`的性能。该形状显示了大约 1500 万行的`Quilt`。

```
>>> quilt = sf.Quilt(bus[:], retain_labels=True)
>>> quilt.shape
(14887665, 6)
```

使用`Quilt`我们可以计算单日 7000 只证券的总交易量，而无需显式连接所有的`Frame`。下面使用的 StaticFrame `HLoc`选择器允许在分层索引中进行每深度级别的选择。在这里，我们选择了 2017 年 11 月 10 日的所有证券记录，涵盖所有证券交易所，并对交易量求和。

```
>>> quilt.loc[sf.HLoc[:, '2017-11-10'], 'Volume'].sum()
5520175355
```

类似地，`iloc_max()`方法可用于查找所有证券中交易量最大的证券的代码和日期。股票代码和日期成为由`iloc_max()`选择的`Series`的`name`属性。

```
>>> quilt.iloc[quilt['Volume'].iloc_max()]
<Series: ('bac', '2012-03-07')>
<Index>
Open                            7.4073
High                            7.6065
Low                             7.3694
Close                           7.6065
Volume                          2423735131.0
OpenInt                         0.0
<<U7>                           <float64>
```

# 跨容器比较:相同的方法，不同的选择

前面的例子演示了用`Bus`、`Batch`和`Quilt`加载、处理和检查“巨大的股票市场数据集”。跨容器比较可用于进一步说明这些容器的特征。首先，我们可以通过对每个容器应用相同的方法来观察三个不同的选择是如何返回的。其次，我们可以观察如何对每个容器使用三种方法来返回相同的选择。

`head(2)`方法从任何容器中返回前两行(或元素)。理解方法的输出在`Bus`、`Batch`和`Quilt`之间的不同有助于说明它们的本质。

对`Bus`的`head(2)`方法调用返回一个新的由前两个元素组成的`Bus`，即“庞大的股市数据集”中的前两帧。

```
>>> bus.head(2)
<Bus>
<Index>
fljh    Frame
bgt     Frame
<<U9>   <object>
```

当`Batch`对`Bus`中的每个`Frame`进行操作时，调用`head(2)`会从“庞大的股票市场数据集中的每个`Frame`中提取前两行调用`to_frame()`将这些提取连接到一个新的`Frame`中，然后只从其中选择两列:

```
>>> sf.Batch(bus.items()).head(2).to_frame().shape
(14316, 6)
>>> sf.Batch(bus.items()).head(2).to_frame()[['Close', 'Volume']]
<Frame>
<Index>                     Close     Volume  <<U7>
<IndexHierarchy>
fljh             2017-11-07 26.189    1300
fljh             2017-11-08 26.3875   3600
bgt              2005-02-25 11.618    97637
bgt              2005-02-28 11.683    90037
angi             2011-11-21 15.4      469578
angi             2011-11-22 16.12     202970
ccj              2005-02-25 20.235    3830399
ccj              2005-02-28 19.501    3911079
uhs              2005-02-25 22.822    4700749
uhs              2005-02-28 23.056    1739084
eqfn             2015-07-09 8.68      489900
eqfn             2015-07-10 8.58      44100
ivfgc            2016-12-02 99.97     5005
ivfgc            2016-12-05 99.97     6002
achn             2006-10-25 11.5      0
achn             2006-10-26 12.39     361420
eurz             2015-08-19 24.75     200
...              ...        ...       ...
cai              2007-05-16 15.0      3960000
desc             2016-07-26 27.062    1015
desc             2016-07-27 27.15     193
swks             2005-02-25 7.0997    1838285
swks             2005-02-28 6.9653    2737207
hair             2017-10-12 9.92      2818561
hair             2017-10-13 9.6       294724
jnj              1970-01-02 0.5941    1468563
jnj              1970-01-05 0.5776    1185461
rosg             2011-08-05 181.8     183
rosg             2011-08-08 169.2     79
wbbw             2013-04-12 13.8      162747
wbbw             2013-04-15 13.67     126845
twow             2017-10-23 16.7      10045
twow             2017-10-24 16.682    850
gsjy             2016-03-07 25.238    14501
gsjy             2016-03-08 25.158    12457
<<U9>            <<U10>     <float64> <int64>
```

最后，`Quilt`表示所包含的`Frame`就好像它们是一个单独的、连续的`Frame`。调用`head(2)`返回虚拟`Frame`的前两行，用层次索引标记，其外部标签是`Frame`的标签(即 ticker)。

```
>>> quilt.head(2)[['Close', 'Volume']]
<Frame>
<Index>                     Close     Volume  <<U7>
<IndexHierarchy>
fljh             2017-11-07 26.189    1300
fljh             2017-11-08 26.3875   3600
<<U4>            <<U10>     <float64> <int64>
```

# 跨容器比较:相同的选择，不同的方法

接下来，我们将展示如何对每个容器使用三种方法来返回相同的选择。虽然上面使用的`head()`方法是一种预配置的选择器，但是所有容器都支持全系列的`loc`和`iloc`选择接口。以下示例提取了 1962 年 1 月 2 日以来的所有“打开”和“关闭”记录。

为了用`Bus`执行这个选择，我们可以遍历每个`Frame`并选择目标记录。

```
>>> for label, f in bus.items():
...     if '1962-01-02' in f.index:
...         print(f.loc['1962-01-02', ['Open', 'Close']].rename(label))
...
<Series: ge>
<Index>
Open         0.6277
Close        0.6201
<<U7>        <float64>
<Series: ibm>
<Index>
Open          6.413
Close         6.3378
<<U7>         <float64>
```

与使用`Bus`相比，`Batch`提供了更紧凑的界面来实现这种选择。在不编写循环的情况下，`Batch.apply_except()`方法可以从每个包含的`Frame`中选择行和列值，同时忽略从没有选定日期的`Frame`中产生的任何`KeyError`。调用`to_frame()`将结果和它们的`Frame`标签连接在一起。

```
>>> sf.Batch(bus.items()).apply_except(
    lambda f: f.loc[‘1962–01–02’, [‘Open’, ‘Close’]], 
    KeyError).to_frame()
<Frame>
<Index> Open      Close     <<U7>
<Index>
ge      0.6277    0.6201
ibm     6.413     6.3378
<<U3>   <float64> <float64>
```

最后，作为`Frame`的虚拟连接，`Quilt`允许选择，就像从单个`Frame`中选择一样。如下图所示，内层标签“1962–01–02”上的分层选择将所有报价机中该日期的所有记录汇集在一起。

```
>>> quilt.loc[sf.HLoc[:, '1962-01-02'], ['Open', 'Close']]
<Frame>
<Index>                     Open      Close     <<U7>
<IndexHierarchy>
ge               1962-01-02 0.6277    0.6201
ibm              1962-01-02 6.413     6.3378
<<U3>            <<U10>     <float64> <float64>
```

# 最小化内存使用

在前面的例子中，`Bus`被显示为在被访问时延迟加载数据。虽然这只允许加载需要的东西，但是对已加载的`Frame`的强引用保留在`Bus`中，将它们保存在内存中。对于大型数据集合，这可能会导致不希望的数据保留。

通过在`Bus`初始化中使用`max_persist`参数，我们可以确定`Bus`中保留的`Frame`的最大数量。如下图所示，通过将`max_persist`设置为 1，在加载每个`Frame`后，加载的`Frame`的数量保持为 1:

```
>>> bus = sf.Bus.from_zip_pickle(‘stocks.zip’, max_persist=1)
>>> bus[‘aapl’].shape
(8364, 6)
>>> bus.status[‘loaded’].sum()
1>>> bus[‘ibm’].shape
(14059, 6)
>>> bus.status[‘loaded’].sum()
1>>> bus[‘goog’].shape
(916, 6)
>>> bus.status[‘loaded’].sum()
1
```

使用这种配置，一个进程可以遍历所有 7，163 个`Frame`，在每个`Frame`上工作，但只会导致单个`Frame`的内存开销。虽然同样的例程可以在单个`Frame`上使用 group-by 来执行，但是这种方法明显倾向于在计算时间上最小化内存使用。下面的例子演示了这样一种方法，找出所有股票的收盘价之间的最大跨度。

```
>>> max_span = 0
>>> for label in bus.index:
...     max_span = max(bus[label]['Close'].max() 
            - bus[label]['Close'].min(), 
            max_span)
...
>>> max_span
1437986239.4042
>>> bus.status['loaded'].sum()
1
```

由于可以将`Bus`作为输入提供给`Batch`、`Quilt`和`Yarn`，所以整个系列的容器都可以从这种减少内存开销的方法中受益。

# 并行处理

独立处理大量的`Frame`是一个令人尴尬的并行问题。因此，这些高阶容器为并行处理提供了机会。

所有压缩文档的构造器和导出器，比如`from_zip_parquet()`或`to_zip_npz()`，都支持一个`config`参数，该参数允许在一个`StoreConfig`实例中指定多处理`Frame`反序列化或序列化的工作线程数量和块大小。`StoreConfig`的相关参数有`read_max_workers`、`read_chunksize`、`write_max_workers`和`write_chunksize`。

类似地，所有的`Batch`构造函数都公开了`max_workers`、`chunk_size`和`use_threads`参数，以允许并行处理`Frames`。只需启用这些参数，对大量`Frame`的操作就可以是多进程或多线程的，有可能带来显著的性能提升。虽然在 Python 中使用线程进行 CPU 受限的处理通常是低效的，但是使用线程池执行的一些基于 NumPy 的操作(在全局解释器锁之外)可以优于进程池。

# 结论

虽然存在处理数据帧集合的相关工具，但是`Bus`、`Batch`、`Quilt`和`Yarn`提供了定义良好的抽象，涵盖了处理潜在的大量表集合的常见需求。结合延迟加载、急切卸载和延迟执行，以及对多种多表存储格式的支持，这些工具为数据帧处理提供了宝贵的资源。