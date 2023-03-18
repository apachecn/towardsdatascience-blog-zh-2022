# 从数字到范畴

> 原文：<https://towardsdatascience.com/from-numerical-to-categorical-3252cf805ea2>

## 存储数字要素的三种方法

![](img/421e68c57ea2db4c049e2a5d5577b0e1.png)

弗兰克·麦凯纳在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

根据原始值所在的区间将宁滨数字要素分组可以提高模型性能。发生这种情况有几个原因。首先，可以基于领域知识来定义这些箱，以帮助模型更好地识别它正在寻找的模式。第二，数据总是有测量误差，宁滨可以减少这些误差的影响。

在基于领域知识的宁滨之外，有三种常见的方法:等宽度、等频率和 k 均值方法。(k-means 方法超出了本文的范围。)等宽取数值变量的范围，并将其分成大小相等的区间。这意味着组的大小是相同的，但是每个箱中的观测值的计数可以有很大的不同。这方面的一个例子是将个人的年龄分为五年或十年。等频箱该功能在每个箱中创建大致相等的计数。在年龄的情况下，如果大多数人都是二三十岁，宁滨十年甚至五年就能创造出无用的垃圾箱。宁滨通过频率，这些共同的年龄将更好地分开，更有利于模型。

## 熊猫中的宁滨

使用开源软件包 [RasgoQL](https://github.com/rasgointelligence/RasgoQL) 从数据库中提取的天气数据，

```
dataset = rql.dataset('Table Name')
df = dataset.to_df()
```

使用 pandas 的`cut`功能可以很容易地创建等宽箱。在这种情况下，创建了 4 个偶数大小的箱。

```
df['HIGH_TEMP_EQ_BINS'] = pd.cut(df.DAILY_HIGH_TEMP, bins=4,
                                 labels=False, include_lowest=True)
```

类似地，`qcut`可用于创建每个仓中计数大致相等的仓。

```
df['HIGH_TEMP_FQ_BINS'] = pd.qcut(df.DAILY_HIGH_TEMP, q=4, 
                                  precision=1, labels=False)
```

## 宁滨与 sci kit-学习

使用 scikit-learn 的预处理功能`KBinsDiscretizer`也可以创建相同的库。为了创建与箱相等，我们将策略设置为“统一”。

```
est = KBinsDiscretizer(n_bins=4, encode='ordinal', 
                       strategy='uniform')
df['HIGH_TEMP_SK_EQ_BINS'] = est.fit_transform(
                                         df[['DAILY_HIGH_TEMP']])
```

类似地，将策略设置为“分位数”将创建大致相等的频率区间

```
est = KBinsDiscretizer(n_bins=4, encode='ordinal', 
                       strategy='quantile')
df['HIGH_TEMP_SK_FQ_BINS'] = est.fit_transform(
                                         df[['DAILY_HIGH_TEMP']])
```

无论哪种情况，要将模型投入生产，都需要为生产环境重写这些转换。在熊猫版本的情况下，需要从`cut`或`qcut`中提取截止值，并将其硬编码到生产代码中。在 scikit-learn 的情况下，要么需要删除截止，并重写整个转换以使用这些截止，要么需要保存预处理对象并将其重新加载到生产环境中。

除了将建模代码重构为生产代码之外，当处理大量数据时，这种方法在最好的情况下会很慢，在最坏的情况下几乎不可能。对于大量的数据，仅仅是等待数据传输到建模环境就浪费了大量的时间。如果数据足够大，它可能不适合 pandas 的内存，并且不是所有的数据都可以用于这些转换。

## 现代数据堆栈中的宁滨

通过利用开源 Python 包 [RasgoQL](https://github.com/rasgointelligence/RasgoQL) ，这两个问题都可以避免。首先，因为 RasgoQL 直接在数据库中创建 bin，所以它可以处理任何大小的数据。其次，在创建这些 bin 并在 Python 中检查它们的过程中，底层 SQL 代码被保存在数据库中。这意味着，当新数据到达数据库时，它将随着应用的箱而自动可用。

为了创建等宽条，可以使用“等宽”类型调用 RasgoQL 函数`bin`。

```
eq_bin = dataset.bin(type='equalwidth',
                     bin_count=4,
                     column='DAILY_HIGH_TEMP')
eq_bin.save(table_name="HIGH_TEMP_EWB")
```

或者，将类型设置为“ntile”将创建相等数量的箱。

```
fq_bin = dataset.bin(type='ntile',
                     bin_count=4,
                     column='DAILY_HIGH_TEMP')
fq_bin.save(table_name="HIGH_TEMP_NTB")
```

此时，这些数据集被发布回数据库，并可在建模和生产环境中使用。

虽然直接在 pandas 中宁滨数据或使用 scikit-learn 的宁滨函数很容易在传统的 Python 机器学习工作流中实现，但当处理存储在数据库中的大数据时，转换为 SQL 方法在速度上具有优势，并且易于推广到生产中。基于 SQL 的特征工程允许更快的处理、更容易的生产路径以及跨多个项目的特征重用。然而，大多数数据科学家更喜欢 Python(或 R)而不是 SQL，而且，对于许多计算来说，SQL 是复杂的。开源包 RasgoQL 允许数据科学家继续在 Python 中工作，但在数据库中执行计算。

如果你想查看 RasgoQL，文档可以在这里找到[，在这里](https://docs.rasgoql.com/)找到库[。](https://github.com/rasgointelligence/RasgoQL)