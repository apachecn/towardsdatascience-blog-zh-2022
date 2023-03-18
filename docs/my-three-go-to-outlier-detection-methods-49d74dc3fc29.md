# 我的三种离群点检测方法

> 原文：<https://towardsdatascience.com/my-three-go-to-outlier-detection-methods-49d74dc3fc29>

## 异常值检测至关重要，尤其是对于数据质量评估

![](img/1ec71075deda10c5589f9754b30a3326.png)

来自[像素](https://www.pexels.com/ko-kr/photo/5935788/)的免费使用照片

# 介绍

离群点检测在许多不同的方面都是至关重要的。如果一家公司想要了解异常/非典型的客户行为，它需要首先识别这样的客户。离群点检测技术在这种情况下发挥了作用。离群点检测对于检查数据集的数据质量也非常有用。在这里，我们来看看常用于检测异常值的三种主流方法。

# IQR /标准差方法

```
Outliers are defined as:Points that are < mean/median - n * IQR/Std or > mean/median + n * IQR/Std
```

IQR 代表四分位距。要理解这个概念，首先需要知道四分位数是什么意思。在统计学中，四分位数是将数据分成四份的值，因此自然会有四个四分位数。[1]它们中的每一个通常被表示为 Q1、Q2、Q3 和 Q4。四分位范围(IQR)是数据集的中间五十，可表示为 Q3- Q1。[2]它经常被用来代替标准偏差(Std)来衡量分布和变化，因为后者更不稳定，对异常值更敏感。

使用 IQR 和标准偏差检测异常值的方法非常简单，位于 n 倍 IQR 或标准偏差定义的特定范围之外的数据点可以被视为异常值。但是，请注意，这种方法对于单变量类型的数据是有效的。

记住高斯分布的一个特性。偏离平均值 3 个标准偏差之外的点仅占分布的 1%。这意味着，与大多数其他点相比，构成 1%分布的那些点是非典型的，并且可能是异常值。当然，现实世界中的数据很少是完美的高斯分布，但是这个更大的概念仍然成立，即远离平均值或中值的点很可能是异常值。

没有用于设置阈值 n 设置规则。这取决于异常值检测的目的以及用户希望异常值检测是保守的还是全面的。因此，在一些测试数据上修改阈值将是一个好主意。

# k 表示聚类

k 表示聚类是数据科学领域中使用的最经典、最传统的聚类方法之一。在这里，我不会深入讨论聚类算法本身是如何工作的。请回顾[一篇](/k-means-clustering-explained-4528df86a120)解释这种算法如何工作的文章。[3]

尽管是一个聚类算法，它也可以用于离群点检测！

一种方法是将簇的数量设置为 K = 1。然后，质心将成为数据中所有点的平均值。然后，计算所有点的欧几里德距离。根据要标记为异常值的点的数量，可以选择距离质心最远的前 n 个点。k 意味着可以通过 Python 的 scikit-learn 库轻松实现集群。参考下面的示例代码，假设存储在变量 df 中的数据有两个数字列 V1 和 V2。(如果包括分类变量，记得对变量进行编码)。

```
**import** pandas **as** pd
**import** numpy **as** np
**from** sklearn.cluster **import** KMeans### Assume you already read in data in pandas in the variable called df (with two numerical columns V1 and V2)X = df.to_numpy() # change the dataframe to numpy matrixkmeans **=** KMeans(n_clusters**=**1)
kmeans**.**fit(X)
distances **=** kmeans**.**transform(X) # apply kmeans on data# If you want to flag 50 points as outliers, grab indexes flagged as outliers
sorted_idx **=** np**.**argsort(distances**.**ravel())[::**-**1][:50]
```

另一种方法是使用 k > 1 个聚类，并标记最小大小的聚类中的所有点。当然，您必须首先通过使用 elbow 方法或剪影评分方法来确定最佳聚类数，我不会在这里详细介绍这两种方法。

# 隔离森林

使用尽可能少的技术术语的隔离森林是一种算法，它不使用随机森林进行预测，而是如其名称所示“隔离”点。

主要思想如下-它试图“通过随机选择一个特征，然后随机选择所选特征的最大值和最小值之间的分割值来隔离观察值。”[4]我们把这种隔离过程称为“分割”。然后，每一轮划分可以被认为是“一棵随机树”，然后所有这些树的集合将是“随机森林”。分离一个样本所需的分裂数就是“树的深度”或“从根到终端节点的路径长度”。然后，在随机森林中的所有树上对该路径长度进行平均，并成为算法的度量。该算法假设使用相对较少数量的分区更容易隔离离群点(因此平均而言隔离在树的较浅深度)。

请参考再次使用 scikit 学习包的隔离森林的 Python 实现示例。

```
**import** pandas **as** pd
**import** numpy **as** np
**from** sklearn.ensemble **import** IsolationForestdata = df[['V1','V2','V3']] # assume there is data with three numerical columns V1, V2, and V3min_max_scaler = preprocessing.StandardScaler()data_scaled = min_max_scaler.fit_transform(data)*# train isolation forest
outliers_fraction = 0.2 # you can set how much would be flagged as outliers*model =  IsolationForest(contamination = outliers_fraction)model.fit(data)*# add the anomaly flags to the data*data['anomaly'] = pd.Series(model.predict(data_scaled))# Flagged outlier points are labelled as -1 and non-outlier points are labelled as 1 and so relabel them as binary outcomes (1 and 0) data['anomaly'] = data['anomaly'].map( {1: 0, -1: 1} )
```

还有其他多种异常检测算法，包括 DBSCAN、局部异常因子(LOF)等。我希望在其他一些帖子中讨论这个问题！如果你感兴趣，请关注我并订阅我的帖子: )

# 参考

[1]统计学如何，[什么是四分位数？定义](https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/what-are-quartiles/)

[2]统计学如何，[四分位距(IQR):它是什么以及如何找到它](https://www.statisticshowto.com/probability-and-statistics/interquartile-range/)

[3] S .耶尔德勒姆，[走向数据科学，K-均值聚类—解释(2020)](/k-means-clustering-explained-4528df86a120)

[4] [Python Scikit 学习文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

# 关于作者

*数据科学家。在密歇根大学刑事司法行政记录系统(CJARS)经济学实验室担任副研究员。Spotify 前数据科学实习生。Inc .(纽约市)。即将入学的信息学博士生。他喜欢运动，健身，烹饪美味的亚洲食物，看 kdramas 和制作/表演音乐，最重要的是崇拜耶稣基督。结账他的* [*网站*](http://seungjun-data-science.github.io) *！*