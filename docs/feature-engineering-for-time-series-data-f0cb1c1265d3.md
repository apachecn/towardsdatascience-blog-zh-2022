# 时间序列数据的特征工程

> 原文：<https://towardsdatascience.com/feature-engineering-for-time-series-data-f0cb1c1265d3>

## 现代数据堆栈上的时间序列聚合

![](img/9a6d337ee183b0fa121b36dba4148c32.png)

图为[唐纳德·吴](https://unsplash.com/@donaldwuid?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

在许多商业数据科学问题中，具有时间序列特征的数据(例如，交易、传感器读数等)。)必须汇总到个人(例如，客户、设备等)。)水平。在现代机器学习环境中这样做可能会带来麻烦，因为传统的训练测试和交叉验证分割将会失败。这是因为在训练集和测试集之间随机拆分观察结果几乎总是会在这两个集之间拆分相关的(因此也是相关的)观察结果。更有问题的是，观察将被分割，以便一些训练观察可以在测试观察之后进行。换句话说，模型将学习从未来预测过去。这通常会导致模型在投入生产时失败。

有许多方法可以处理这个问题，我在这里讨论了我的首选方法。一种常用的方法是对每个人进行一次观察。这可以通过为每个人随机选择一个观察值或选择一个日期并仅使用该日期的观察值来实现。

我们将集中精力为给定日期的每个观察创建一个单独的记录。在本例中，我们将使用天气数据并对其进行汇总，以显示多个时间段内县级的平均高温。在熊猫身上做到这一点可能很有挑战性，但我们将展示在 Rasgo 的现代数据堆栈上做到这一点是多么容易。

## 熊猫的单矩聚集

给定包含县代码 *FIPS* 、日期*日期*和某个县在日期*高温*、`rolling`和`groupby`的数据帧`df`，可以使用它们来创建所有日期的滚动平均值。在这种情况下，将创建一周、两周和四周的滚动平均值。

首先，应该创建滚动窗口

```
df['HIGH_TEMP_AVG_7D'] = df.sort_values(
                              by=['FIPS', 
                                 'DATE']).set_index(
                                 'DATE').groupby(
                                 'FIPS')[
                                 "HIGH_TEMP"].rolling('7D',             
                                 closed='both').mean()
df['HIGH_TEMP_AVG_14D'] = df.sort_values(
                              by=['FIPS', 
                                 'DATE']).set_index(
                                 'DATE').groupby(
                                 'FIPS')[
                                 "HIGH_TEMP"].rolling('14D',             
                                 closed='both').mean()
df['HIGH_TEMP_AVG_28D'] = df.sort_values(
                              by=['FIPS', 
                                 'DATE']).set_index(
                                 'DATE').groupby(
                                 'FIPS')[
                                 "HIGH_TEMP"].rolling('28D',             
                                 closed='both').mean()
```

可以重置索引以将县代码和日期返回到数据帧中的简单列。

```
df.reset_index(inplace=True)
```

这种方法有几个问题。首先，如果数据量很大，将数据下载到工作站并进行处理会非常耗时。虽然这可以通过仅下载所讨论的日期所需的数据来加速，但是仍然要对数据中的所有日期执行额外的计算。更重要的是，生产工作流只需要聚合到当前日期，并且应该生成聚合。

## RasgoQL 和现代数据栈方法

在现代数据栈上，开源包 RasgoQL 以几种方式解决了这两个问题。首先，处理保存在云数据仓库中，因此没有数据被移动到工作站。其次，RasgoQL 转换`timeseries_agg`利用数据仓库的能力处理比工作站上可能处理的数据量大得多的数据。

首先，获取对数据仓库中保存天气数据的表的引用。

```
dataset = rql.dataset('WEATHER_TABLE_NAME')
```

每天的滚动平均值可以通过调用转换来生成。

```
tsagg = dataset.timeseries_agg(aggregations={
                                'DAILY_HIGH_TEMP': ['AVG']
                               },
                               group_by=['FIPS'],
                               date='DATE'
                               offsets=[7, 14, 28],
                               date_part='day')
```

这些数据可以通过调用`tsagg`上的`to_df`来下载，以获得熊猫数据帧进行进一步处理。

```
tsagg_df = tsagg.to_df()
```

或者，可以将这些数据发布回云数据仓库，以创建一个视图，其他人可以在 RasgoQL 中使用该视图，也可以通过 SQL 直接从数据仓库中使用该视图。

```
tsagg.save(table_name="ROLLING_WEATHER_SNAPSHOT")
```

在现代数据栈上使用开源 Python 包 RasgoQL 的转换使得创建这种滚动窗口比在 pandas 中创建要简单得多。该包创建一个在数据库中执行的 SQL 调用，并且可以将结果保存回数据库，以便将来在现有项目或需要相同数据的其他项目中引用。通过使用 SQL，处理的数据量可能会大得多。最后，使用 pandas，使用 aggregate 函数可以允许在多个列上同时进行多个聚合。除了在每个步骤中只能考虑一个时间窗口(例如七天或十四天)。这意味着要创建 7 天、14 天和 28 天的平均值，需要三种不同的 Python 语句。在 RasgoQL 中，一个 SQL 调用可以同时生成所有三个。

如果你想查看 RasgoQL，文档可以在这里找到[，在这里](https://docs.rasgoql.com/)找到库[。](https://github.com/rasgointelligence/RasgoQL)