# 时序数据的每周汇总

> 原文：<https://towardsdatascience.com/weekly-aggregation-of-time-series-data-f9bdcc495a58>

## 现代数据堆栈中的聚合

![](img/e8ee7d54c58e94b5a73c449c83d0dc91.png)

詹姆斯·a·莫尔纳尔在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

最常见的特征工程任务之一是将多个记录聚合成单个记录。这可以是每个客户汇总到一个记录中的客户交易。这些类型的聚合通常具有类似时间序列的特征。例如，通常只需要汇总某个日期或交易之前的交易，而不是汇总所有客户交易。或者，只需要合并给定交易之前特定时间段内的交易。其他聚合发生在时间以外的维度上。这些可以是患有某种疾病的所有患者的聚合生物特征、所有欺诈交易的交易细节等。

在本文中，我们将关注使用 Python 和使用开源包 [RasgoQL](https://github.com/rasgointelligence/RasgoQL) 的现代数据堆栈来聚合数据。特别是，我们将创建底层数据的每周汇总。

## 熊猫的聚集

在熊猫中，可以通过调用`aggregate`(或者`agg`作为别名来进行聚合。`agg`的参数可以是函数(max)、包含函数名的字符串(' max ')、函数或函数名的列表([max，' mean ')、标签(通常是列名)和函数、函数名或列表的字典:

```
df.agg({'column_A': max,
        'column_B': np.sum,
        'column_C': [max, 'mean', np.sum]})
```

聚合通常不直接应用于数据帧，而是应用于组。大多数情况下，这些组是通过调用`groupby`创建的

```
group = df.groupby(by=['column_A'])
```

接着是给`agg`的一个电话。

## 时间窗口内的聚合

在这种情况下，为了在一个时间窗口内进行聚合，使用函数`resample`而不是`groupby`。为了使用`resample`，数据帧的索引需要是日期或时间。使用`set_index`将索引设置为*日期*。

```
df.set_index('DATE', inplace=True)
```

然后创建每周小组

```
weekly_group = df.resample('7D')
```

最后，调用`agg`将特性汇总到每周级别

```
weekly_df = weekly_group.agg({'COLUMN_A': ['min', 'max'],
                              'COLUMN_B': ['mean', 'sum']})
```

这种方法在数据适合内存的单台机器上运行良好。然而，这通常需要重写以在生产环境中工作，或者创建一个公司其他人可以利用的共享版本。

## 使用 RasgoQL 聚合

利用 RasgoQL 将允许数据科学家从他们的笔记本或计算机创建相同的每周聚合，但将利用现代数据堆栈的能力直接在云数据仓库上创建这些功能。与 pandas 不同，我们需要创建想要分组的要素，而不是重新采样到每周级别。在这种情况下，*日期*需要转换成周变量。这可以通过`datetrunc`转换将日期截断为星期来完成。

```
dataset.datetrunc(dates={'DATE': 'week'})
```

这创建了一个新变量 *DATE_WEEK* 。为了更清楚，使用`rename`转换将这个新变量重命名为 *WEEK* 。

```
dataset.datetrunc(dates={'DATE': 'week'}).rename(
                  renames={'DATE_WEEK': 'WEEK})
```

现在，应用与 pandas `agg`函数中使用的相同字典的`aggregate`转换，将把数据聚集到每周级别，就像对 pandas 所做的那样。

```
dataset.datetrunc(dates={'DATE': 'week'}).rename(
                  renames={'DATE_WEEK': 'WEEK}).aggregate(
                  aggregations={'COLUMN_A': ['min', 'max'],
                                'COLUMN_B': ['mean', 'sum']})
```

这个转换的结果可以通过调用`save`函数发布回云数据仓库。

```
wkds = dataset.datetrunc(dates={'DATE': 'week'}).rename(
                         renames={'DATE_WEEK': 'WEEK}).aggregate(
                         aggregations={'COLUMN_A': ['min', 'max'],
                                       'COLUMN_B': ['mean', 'sum']})wkds.save(table_name='WEEKLYAGGDATA')
```

通过调用`to_df`，这些数据可以在 Python 中作为 pandas dataframe 获得。

```
weekly_df = wkds.to_df()
```

这些数据现在可以用于建模、即席分析或仪表板开发。此外，随着数据仓库中原始数据的更新，RasgoQL 将自动更新这个每周汇总。这意味着在这个项目开发过程中执行的聚合同时在生产环境中进行，并且可以立即用于生产系统。

在我职业生涯的大部分时间里，在用 Python 和 pandas 构建了我的特性之后，我会将代码交给软件工程，他们会在那里对代码进行重构。这个过程花了几个星期，如果不是几个月，完成。我见过模型无法执行的情况，因为当它们被投入生产时，建模数据不再反映当前的生产数据。有了 RasgoQL，我仍然可以使用 Python，但是可以将我的特征工程保存在我的数据库中，并且拥有软件工程可以更容易实现的 dbt 模型。这可以在部署模型之前节省几个月的开发工作。

如果你想查看 RasgoQL，可以在这里找到文档[，在这里](https://docs.rasgoql.com/)找到存储库[。](https://github.com/rasgointelligence/RasgoQL)