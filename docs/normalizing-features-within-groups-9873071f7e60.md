# 标准化组内的特征

> 原文：<https://towardsdatascience.com/normalizing-features-within-groups-9873071f7e60>

## 标准化的另一种方法

![](img/8fc573f840079eedd07e69887ba3ca67.png)

照片由[菲利普·姆罗兹](https://unsplash.com/@mroz?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

数字特征最常见的特征工程技术之一是标准化这些特征。这通常通过减去平均值并除以标准偏差或减去最小值并除以最大值和最小值之差来实现。这两种方法都有助于模型的训练和各种特征对模型的影响的评估。此外，无论是使用 scikit-learn 还是直接在 pandas 中执行这些操作都是微不足道的。然而，这些不包含新的信息。

## 在组内标准化

有一种类似的技术可以将信息添加到特征中，并帮助提高模型的性能。而不是使用全局统计(平均值、标准偏差、最小值、最大值等。)，使用较小组内的统计数据。例如，设想围绕客户的信用卡交易构建模型。这些可能是欺诈、信用风险或流失模型。想想单笔交易的金额。这可以基于所有事务的全局统计数据来标准化；然而，可能有更有趣的方法来标准化这些数量。

考虑仅针对客户的支出来规范这一交易。这是花费在平均值附近还是平均值的三个标准差？交易偏离均值三个标准差对欺诈或流失风险意味着什么？此外，考虑在该商家、该邮政编码或该商家类别的所有交易中标准化交易。在珠宝店，一笔交易与客户通常的模式有三个标准偏差，但接近平均值，这与在杂货店同样有三个标准偏差，但也离平均值很远的交易看起来非常不同。

## 熊猫的分组统计

有几种方法可以计算熊猫的群体内统计数据。最简单的方法之一是使用熊猫功能`pivot_table`。对于这个例子，使用天气数据，考虑确定对于给定的天气模式，在给定的区域中高温有多不寻常。除了位置( *FIPS* 联邦信息处理标准县代码)，我们的天气数据报告天气图标(是多云，部分多云，晴朗，等等。).要了解阴天的温度是否异常高，我们需要知道该地区阴天温度的平均值和标准差。这可以通过旋转天气图标(*列*)来完成，汇总每个位置的高温(*值*)(*指数*)。

```
pd.pivot_table(weather_df, 
               values='DAILY_HIGH_TEMP', 
               index=['FIPS'],
               columns=['WEATHER_ICON'],
               aggfunc=[np.mean, np.std)
```

该调用的输出可以加入到 *FIPS* 上的原始数据集，并使用适当的平均值和标准偏差来计算归一化温度。

## 云数据仓库中的枢纽

如果数据位于数据仓库中，这种透视和聚合也可以在 SQL 中执行，而无需将数据移动到您的 pandas 环境中。然而，pivot 的 SQL 语法很繁琐，需要识别列中需要提取的每个值。有了开源包 [RasgoQL](https://github.com/rasgointelligence/RasgoQL) [转换](https://docs.rasgoql.com/primitives/transform),[pivot 转换](https://docs.rasgoql.com/transforms/table-transforms/pivot)为你做所有这些。

处理每日天气数据时，可以通过调用`rql.dataset`来使用这些数据

```
dataset = rql.dataset('DAILY_WEATHER')
```

通过调用`pivot`可以调用 pivot 变换由 FIPS 和 WEATHER_ICON 创建平均高温。在这种情况下，pandas 索引、列和值分别对应于 RasgoQL `pivot`的维度 value_column 和 pivot_column。

```
t1 = dataset.pivot(dimensions=['FIPS'],
                   pivot_column='DAILY_HIGH_TEMP',
                   value_column='WEATHER_ICON',
                   agg_method='AVG')
```

可以通过检查`t1.sql()`来检查 SQL，并且可以通过调用`t1.preview()`来检查数据样本。一旦透视看起来正确，就可以通过调用`save`将其保存到数据仓库中

```
t1.save(table_name="WEATHER_AVG_HIGH_BY_FIPS")
```

类似地，可以通过运行另一个`transform`并将 *agg_method* 从 **AVG** 更改为 **STDDEV** 来计算标准差。

```
t2 = dataset.pivot(dimensions=['FIPS'],
                   pivot_column='DAILY_HIGH_TEMP',
                   value_column='WEATHER_ICON',
                   agg_method='STDDEV')
t2.save(table_name="WEATHER_STD_HIGH_BY_FIPS")
```

这些可以使用这里讨论的连接转换连接在一起。

通过在数据仓库中执行这些转换，避免了在发生特征工程的数据库和服务器之间移动数据所花费的时间。此外，通过处理数据仓库中的数据，可以使用所有数据，而不受服务器上内存量的限制。最后，通过将这些转换作为视图或表保存回数据库，它们可以立即在生产工作流中使用，而不需要数据工程团队对它们进行重构。

如果你想查看 RasgoQL，文档可以在[这里](https://docs.rasgoql.com/)和[这里](https://github.com/rasgointelligence/RasgoQL)找到。