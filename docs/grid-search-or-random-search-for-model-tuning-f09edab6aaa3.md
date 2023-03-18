# 模型调整的网格搜索或随机搜索

> 原文：<https://towardsdatascience.com/grid-search-or-random-search-for-model-tuning-f09edab6aaa3>

## 了解如何在 SciKit-Learn 的 GridSearchCV 或 RandomizedSearchCV 之间进行选择

![](img/27f16dbd714f2d5ab33509e8cc236a12.png)

由[马库斯·温克勒](https://unsplash.com/@markuswinkler?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/search?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

# 为什么要微调模型？

机器学习并不总是像在 *Iris* 、*泰坦尼克号*或*波士顿房价*数据集那样简单明了。

但是，嘿，不要误会我。我从那些著名的玩具数据集中学到了很多东西(并且一直在学)。它们的最大优点是不需要太多的探索或预处理。很多时候，我们可以直接进入我们想要练习和学习的点，比如管道、建模、模型调整、可视化*等。*

我想我想说的是，当对数据建模时，它不会像我们用来研究的玩具数据集那样容易。真实数据需要调整、拟合，并对模型进行微调，因此我们可以从算法中获得最佳效果。为此，Scikit-Learn 的`GridSearchCV`和`RandomizedSearchCV`是两个不错的选择。

好吧，也许是因为你需要通过为你的模型选择正确的超参数来使你的预测更好。因此，本快速教程中介绍的两个选项将允许我们为建模算法提供一个超参数列表。它会把选项一个一个组合起来，测试很多不同的模型，然后呈现给我们最好的选项，性能最好的那个。

太棒了，不是吗？因此，让我们继续了解它们之间的区别。

# 差别

为了用一个简单的类比来说明这个概念，让我们想象一下，我们要去参加一个聚会，我们想要选择最佳的服装组合。我们带了几件衬衫、几条裤子和几套衣服。

> 如果我们是 **GridSearchCV** ，我们会尝试**每一种** **衬衫、裤子、鞋子的**组合，对着镜子照一张照片。最后，我们将审视一切，选择最佳方案。
> 
> 如果我们是 **RandomizedSearchCV** ，我们会尝试**一些**随机挑选**的组合**，拍照，最后选出表现最好的。

![](img/6ff8070d3e5ecf23d56192ef12e5d29e.png)

照片由[卢卡斯·黄](https://unsplash.com/@zuizuii?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/clothes?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

现在，用这个比喻，我相信你能感觉到网格搜索将花费更多的时间，因为我们增加了服装的数量来尝试。如果只是两件衬衫，一条裤子，一双鞋，用不了多久。但是如果有 10 件衬衫，5 条裤子和 4 双不同的鞋子，那么…你明白了。但是，从另一方面来说，它会有一张所有东西的图片，所以它有非常完整的选项集可供选择。

随机搜索不会花很长时间，因为它只会尝试一些随机选择的组合。因此，如果你的选择范围很小，那么使用它就没有意义。训练所有选项或其中几个选项的时间基本相同。但是当你有很多组合可以尝试的时候，可能更有意义。但是请记住，这个选项不会尝试所有选项，所以真正的“最佳估计者”甚至可能不会尝试。

现在让我们看看他们的行动。

# 编码

让我们进入编码部分。我们将开始导入本练习所需的模块。

```
# Imports
import pandas as pd
import numpy as np
import seaborn as sns# Dataset
from sklearn.datasets import make_regression# sklearn preprocess
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split# Search
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
```

接下来，我们可以创建一个回归数据集。

```
# Dataframe
df = make_regression(n_samples=2000, n_features=5,
                     n_informative=4, noise=1, random_state=12)# Split X and y
X= df[0]
y= df[1]
```

我们可以分开训练和测试。

```
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random)
```

让我们创建一个管道来扩展数据并适应决策树模型。

```
# Creating the steps for the pipeline
steps = [ ('scale', StandardScaler()),
          ('model', DecisionTreeRegressor())  ]# Creating pipeline for Decision Tree Regressor
pipe = Pipeline(steps)# Fit the model
pipe.fit(X_train, y_train)
```

下一步是创建一个要测试的超参数网格`params`，以微调模型。这里有(2 x 3 x 2 = 12)个选项需要测试。

```
%%timeit
# Creating dictionary of parameters to be tested
params= {'model__max_features': [2,5], 'model__min_samples_split':[2, 5, 10], 'model__criterion': ['friedman_mse', 'absolute_error']}# Applying the Grid Search
grid = GridSearchCV(pipe, param_grid=params, cv=5, scoring='neg_mean_squared_error')grid.fit(X_train, y_train)# Best model
grid.best_estimator_
```

时间结果如下。循环 2.37 秒。总时间约为 18 秒。那很好。

```
2.37 s ± 526 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

但是，如果我们增加要测试的选项的数量，会发生什么呢？让我们试试(4 x 6 x 2 = 48)个选项。

```
%%timeit# Creating dictionary of parameters to be tested
params= {'model__max_features': [2,3,4,5], 'model__min_samples_split':[2,5,6,7,8,10],'model__criterion': ['friedman_mse', 'absolute_error']}# Applying the Grid Search
grid = GridSearchCV(pipe, param_grid=params, cv=5, scoring='neg_mean_squared_error')grid.fit(X_train, y_train)# Best model
grid.best_estimator_
```

时间增加了很多。每圈 6.93 秒。这里的总时间超过 1 分钟。

```
6.93 s ± 505 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

现在让我们看看随机搜索。首先，我们将尝试与第一轮相同的 12 个选项。

```
%%timeit# Creating dictionary of parameters to be tested
params= {'model__max_features': [2,5],'model__min_samples_split':[2, 5, 10],'model__criterion': ['friedman_mse', 'absolute_error']}# Applying the Grid Search
randcv = RandomizedSearchCV(pipe, param_distributions=params, cv=5, scoring='neg_mean_squared_error')randcv.fit(X_train, y_train)# Best model
randcv.best_estimator_
```

时间低于网格搜索，不出所料。每个循环 1.47 秒，总共运行大约 10 秒。

```
1.47 s ± 140 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

如果我们增加网格中的选项数量，让我们看看会发生什么。

```
%%timeit# Creating dictionary of parameters to be tested
params= {'model__max_features': [2,3,4,5],
         'model__min_samples_split':[2,5,6,7,8,9,10],
         'model__criterion': ['friedman_mse', 'absolute_error']}# Applying the Grid Search
randcv = RandomizedSearchCV(pipe, param_distributions=params, cv=5, scoring='neg_mean_squared_error')randcv.fit(X_train, y_train)# Best model
randcv.best_estimator_
```

这是结果。哇，几乎同时！每圈 1.46 秒。

```
1.46 s ± 233 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

嗯，那很好。但是他们给了我们相似的结果吗？接下来看看。

# 结果

现在让我们来评估一下来自`GridSearchCV`和`RandomizedSearchCV`的结果。

计算网格搜索的 RMSE。

```
# Taking the best estimator
best_grid = grid.best_estimator_# Predict
preds_grid = best_grid.predict(X_test)# RMSE
np.sqrt( mean_squared_error(y_test, preds_grid) )**[OUT]:
53.70886778489411**
```

计算随机搜索的 RMSE。

```
# Taking the best estimator
best_rand = randcv.best_estimator_# Predict
preds_rand = best_rand.predict(X_test)# RMSE
np.sqrt( mean_squared_error(y_test, preds_rand) )**[OUT]:
55.35583215782757**
```

结果有 3%的差异。网格搜索得到了最好的结果，因为它训练每一个模型，因此，它会找到最适合的。当你尝试太多组合时，权衡就是训练的时间。在这种情况下，随机搜索可能是一个很好的选择。

# 在你走之前

在这篇文章中，我们想展示两个微调模型的好方法。

当你需要考虑每一个可能的优化时，你可以使用`GridSearchCV`。但是要考虑训练模型的时间。如果您知道应该选择哪个超参数，这可能是您的最佳选择。

当您有太多的超参数组合可供选择时，`RandomizedSearch`可能是最佳选择。例如，当使用网格搜索时，您可以运行它并获得最佳估计值，从而为您指出从哪个组合开始的正确方向。

如果你喜欢这篇文章，关注我的博客或者在 [Linkedin](https://www.linkedin.com/in/gurezende/) 上找到我。

[](http://gustavorsantos.medium.com/) [## 古斯塔沃·桑托斯-中等

### 阅读古斯塔夫·桑托斯在媒介上的作品。数据科学家。我从数据中提取见解，以帮助个人和公司…

gustavorsantos.medium.com](http://gustavorsantos.medium.com/) 

# 参考

[](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) [## sklearn.model_selection。GridSearchCV

### 对估计量的特定参数值进行穷举搜索。重要成员是适合的，预测。GridSearchCV…

scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) [](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV) [## sklearn.model_selection。随机搜索

### 超参数随机搜索。RandomizedSearchCV 实现了一个“fit”和一个“score”方法。它还实现了…

scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV)