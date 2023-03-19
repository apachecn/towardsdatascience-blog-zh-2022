# sklearn 的最佳进口产品

> 原文：<https://towardsdatascience.com/the-best-imports-from-sklearn-b5b9c4f2c0e3>

## 了解从这个神奇的 Python 库中导入什么以及何时导入

![](img/bbca942482e4f9a11ebb6139fed3a86e.png)

[亚历山大·奈特](https://unsplash.com/@agk42?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/robots?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍照

# sci kit-学习

Science Kit Learn，或者对入门者来说简称为`sklearn`，可能是 Python 中建模数据的主要软件包。

Sklearn 是 2007 年由 *David Cournapeau 开发的。历史告诉我们，它始于谷歌的一个夏季项目。从那以后，发生了很多变化，信不信由你，1.0 版本直到 2021 年 12 月才发布！当然，早在那个日期之前，它就已经在各地提供了巨大的成果。*

总之， **sklearn** 是一个库，它不仅处理非监督学习(如*聚类*)和监督学习(如*回归*和*分类*),还处理围绕数据科学项目的所有其他组件。使用 sklearn，我们可以访问预处理工具，如缩放、归一化。我们可以看到模型选择工具，如 k-fold，网格搜索，交叉验证。当然，有创建模型的算法，也有检查指标的工具，比如混淆矩阵。

在这篇文章中，我想与您分享的是在您使用 Scikit Learn 时要导入的一些模块，因此您可以在构建模型时使用这些内容作为快速参考。让我们看看他们。

# 预处理数据

建模数据不仅仅是加载 sklearn 并通过算法运行数据。它需要更多的时间来处理数据，因此您可以为模型提供良好的输入。为此，您可以使用以下模块。

## 最小最大缩放器

此工具将根据变量的最大值和最小值对数据进行归一化。最大值变为 1，最小值为 0，两者之间的所有值都是最大值的百分比。

```
from sklearn.preprocessing import [**MinMaxScaler**](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler)
```

## 标准缩放器

Normalize 会将变量转换为均值= 0，标准差= 1。但是，它不会改变数据的形状，这意味着它不会将数据转换为正态分布。

```
from sklearn.preprocessing import [S](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)[tandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)
```

## 估算者

当缺失值过多，并且我们希望使用某种插补技术来估计当前为 NA 的值时，使用插补器。其主要模块如下。

```
from sklearn.impute import [**SimpleImputer**](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer)from sklearn.impute import [**KNNImputer**](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html#sklearn.impute.KNNImputer)
from sklearn.impute import [**IterativeImputer**](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer)
```

# 特征选择

如果挑战是为一个模型寻找最好的特征，有许多可能性。Scikit Learn 将提供这些以及更多。

```
from sklearn.feature_selection import [**SelectKBest**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest), [**f_classif**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif), [**r_regression**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.r_regression.html#sklearn.feature_selection.r_regression)from sklearn.feature_selection import [**RFE**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE)
```

总而言之:

**【In】数字；[Out]数值:**

`SelectKBest(score_func=f_regression, k=number best attributes)`

`SelectKBest(score_func=mutual_info_regression, k=number best attributes)`

**【In】数字；[Out]分类:**

`SelectKBest(score_func=f_classif, k=number best attributes)`

**【直言不讳】；[Out]分类:**

`SelectKBest(score_func=chi2, k=number best attributes)`

`SelectKBest(score_func=mutual_info_classif, k=number best attributes)`

# 型号选择

在经过预处理和特征选择之后，就到了选择模型的时候了。

## 列车测试分离

当然，我们需要将数据分为解释变量(X)和被解释变量(y)。为此，我们使用训练测试分割。

```
from sklearn.model_selection import **train_test_split**
```

## 交叉验证和折叠

有许多方法可以交叉验证数据。最常见的是使用 K-fold，将数据分成 K 个部分，每个部分都用作训练集和测试集。例如，如果我们将一个集合折叠成 3 个，则第 1 部分和第 2 部分是训练，第 3 部分是测试。然后下一次迭代使用 1 和 3 作为训练，使用 2 作为测试。最后，2 和 3 被训练，1 被测试。

```
from sklearn.model_selection import [**cross_validate**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate), [**cross_val_score**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score)
from sklearn.model_selection import [**KFold**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold)
```

在这个[链接](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)中，你可以看到更多的选项和文档，比如省去一个、分层折叠、洗牌拆分等等。

## 模型调整

为了调整模型，sklearn 将为我们提供这些令人惊叹的选项，即网格搜索或随机搜索。使用这种方法，可以测试模型参数的多种组合，将最佳结果作为继续预测的最佳估计值。

```
from sklearn.model_selection import[**GridSearchCV**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)from sklearn.model_selection import [**RandomizedSearchCV**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV)
```

# 评估者

估计量是可供我们使用的算法。Scikit learn 有很多这样的例子。

## 无监督:聚类

无监督学习是指我们不为预测提供标签。因此，该算法将在没有监督的情况下寻找模式并对数据点进行分类。意思是，没有“正确或错误的答案”。

```
# Clustering
from sklearn.cluster import [**KMeans**](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)from sklearn.cluster import [**DBSCAN**](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN)from sklearn.mixture import [**GaussianMixture**](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture)
```

## 分类

分类模型将理解来自数据集的模式以及什么是相关联的标签或组。然后，它可以根据这些模式对新数据进行分类。最常用的是集合模型，如随机森林或梯度推进。还有一些更简单的，像决策树，逻辑回归和 K 近邻。

```
from sklearn.ensemble import [**RandomForestClassifier**](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)from sklearn.ensemble import [**GradientBoostingClassifier**](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier)
from sklearn.tree import [**DecisionTreeClassifier**](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
from sklearn.neighbors import [**KNeighborsClassifier**](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)from sklearn.linear_model import[**LogisticRegression**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)
```

## 回归

回归问题是那些需要返回一个数字作为输出的问题。回归解决的经典问题是汽车和房价。在这种情况下，最常用的模型是线性模型。有一些正则化回归的选项，如山脊或套索。对于非线性关系，也可以使用基于树的模型。

```
from sklearn.linear_model import [**LinearRegression**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)
from sklearn.linear_model import [**Ridge**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge)from sklearn.ensemble import [**RandomForestRegressor**](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor)from sklearn.ensemble import [**GradientBoostingRegressor**](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor)
```

# 韵律学

最后，我们可以使用 sklearn 的度量组件来评估模型。接下来看看最常用的。

## 分类指标

```
from sklearn.metrics import [**f1_score**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)
from sklearn.metrics import [**confusion_matrix**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)
from sklearn.metrics import [**accuracy_score**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)
from sklearn.metrics import [**auc**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc)
```

## 回归度量

```
from sklearn.metrics import [**mean_squared_error**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error)
from sklearn.metrics import [**mean_absolute_error**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error)
from sklearn.metrics import [**r2_score**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score)
```

# 在你走之前

Scikit learn 是数据科学家的得力助手。在他们的文档中，您可以探索和学习更多的东西。

在这里，我们只介绍了一些常用于数据科学的导入，但是，如果您访问他们的页面，您可以看到更多。

他们的文档的优点是非常清晰和有条理。此外，很多时候它给你一个关于算法或函数所解决的问题的深度知识。

如果你喜欢这个内容，请关注我的博客。

[](http://gustavorsantos.medium.com/)  

# 参考

[](https://scikit-learn.org/stable/modules/classes.html#)  [](https://en.wikipedia.org/wiki/Scikit-learn) 