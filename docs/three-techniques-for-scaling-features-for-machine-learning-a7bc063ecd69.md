# 用于机器学习的缩放特征的三种技术

> 原文：<https://towardsdatascience.com/three-techniques-for-scaling-features-for-machine-learning-a7bc063ecd69>

## 现代数据堆栈的扩展功能

![](img/7dca3b667aa53892faf138ef7b73dcbd.png)

照片由 [Julentto 摄影](https://unsplash.com/@julensan09?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

虽然数字特征的缩放并不总是需要像 [Praveen Thenraj](https://medium.com/@praveenmec67) 在他关于基于树的机器学习技术的[帖子](/do-decision-trees-need-feature-scaling-97809eaa60c6)中解释的那样进行，但它确实有利于线性和逻辑回归、支持向量机和神经网络。许多数据科学家已经创建了建模脚本，用于自动构建和测试许多不同类型的建模算法。这些脚本允许数据科学家为该数据选择性能最佳的模型。特征的缩放通常在这些脚本中执行。

使用了两种主要的缩放技术。第一种是标准缩放(或 z 缩放),计算方法是减去平均值并除以标准差。第二个是最小-最大缩放，计算方法是减去最小值，再除以最大值和最小值之差。

## sci kit-基于学习的缩放

通过从预处理模块导入`StandardScaler`并将其应用于数据帧，标准缩放器可用于缩放列`scale_columns`的列表

```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[scale_columns] = scaler.fit_transform(df[scale_columns])
```

类似地，最小-最大缩放器可以应用于与

```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[scale_columns] = scaler.fit_transform(df[scale_columns])
```

## 仅基于熊猫的缩放

由于 scikit-learn 模块相当大，如果它仅用于缩放特征，那么使用 pandas 进行缩放会更容易。要按平均值和标准差缩放每个要素，请调用

```
df[scale_columns] = (df[scale_columns] - df[scale_columns].mean()) /
                     df[scale_columns].std()
```

请注意，这并不重复来自`StandardScaler`的结果。这是因为 scikit-learn 使用了 numpy 的`std`函数。默认情况下，该函数使用 0 个自由度。另一方面，熊猫的`std`是默认的无偏估计量。要获得相同的值，请运行

```
df[scale_columns] = (df[scale_columns] - df[scale_columns].mean()) /
                     df[scale_columns].std(ddof=1)
```

类似地，要获得最小-最大缩放器，运行

```
df[scale_columns] = (df[scale_columns] - df[scale_columns].min()) /
                 (df[scale_columns].max() - df[scale_columns].min())
```

## 数据仓库中的 RasgoQL 伸缩

开源 Python 包 [RasgoQL](https://github.com/rasgointelligence/RasgoQL) 可以直接在数据仓库中创建缩放变量，而不是从数据仓库中提取数据。首先，这将节省提取数据并在缩放后将其推回数据仓库的时间。其次，通过利用仓库的能力，可以一次转换大得多的数据集。

标准定标器可用作

```
scaledset = dataset.standard_scaler(
                       columns_to_scale=['DS_DAILY_HIGH_TEMP', 
                                         'DS_DAILY_LOW_TEMP',
                                         'DS_DAILY_TEMP_VARIATION',
                                         'DS_DAILY_TOTAL_RAINFALL'])scaledset.save(table_name='DS_STANDARD_SCALED')
```

类似地，最小-最大缩放器可以应用为

```
scaledset = dataset.min_max_scaler(
                       columns_to_scale=['DS_DAILY_HIGH_TEMP', 
                                         'DS_DAILY_LOW_TEMP',
                                         'DS_DAILY_TEMP_VARIATION',
                                         'DS_DAILY_TOTAL_RAINFALL'])scaledset.save(table_name='DS_MIN_MAX_SCALED')
```

这些缩放后的要素现在可以通过连接到原始数据或任何其他数据来使用。它们还可以用于管道建模、数据可视化和生产管道预测。可以通过调用`to_df`将缩放后的数据下载到 pandas 中用于建模

```
df = scaledset.to_df()
```

如果您想检查 RasgoQL 用来创建这个表或视图的 SQL，运行`sql`:

```
print(scaledset.sql())
```

最后，如果您在生产中使用 dbt，并且希望将这项工作导出为 dbt 模型，供您的数据工程团队使用，请致电`to_dbt`:

```
scaledset.to_dbt(project_directory='<path to dbt project>'
```

当在单台机器上处理少量数据时，scikit-learn 和 pandas 方法将很好地用于在建模之前缩放特征。但是，如果数据很大或者已经存储在数据库中，在数据库中执行缩放(使用 RasgoQL 或 SQL)可以在数据准备期间节省大量时间。更重要的是，用于缩放数据的代码可以很容易地放入生产工作流中，潜在地节省了将模型投入生产的数周时间。

如果您想查看 RasgoQL，可以在这里找到文档[，在这里](https://docs.rasgoql.com/)找到存储库[。](https://github.com/rasgointelligence/RasgoQL)