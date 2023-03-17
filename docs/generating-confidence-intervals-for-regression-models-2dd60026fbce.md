# 为回归模型生成置信区间

> 原文：<https://towardsdatascience.com/generating-confidence-intervals-for-regression-models-2dd60026fbce>

## 几种模型无关方法的解释和研究

![](img/e7b745f61fff79e39ff0eda5390939b2.png)

由 [CardMapr](https://unsplash.com/@cardmapr?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

数据科学家开发回归模型来预测日常工作中的一些连续变量是很常见的。可能不太常见的是，尤其是当我们学习回归方法时，如何为我们的模型的给定预测定义一个置信区间。

当然，有一些模型具有定义此区间的内置方式，并且在处理分类时，大多数模型会输出一个概率，该概率可以帮助我们处理预测的不确定性。

但是如果我们有一个没有内置输出方式的回归模型呢？我们需要一种不可知论的方法来生成回归变量的置信区间，这就是我们在这篇文章中要探索的。

为此，我们将研究论文[使用刀切+](https://arxiv.org/abs/1905.02928) [1]的预测推理，该论文解释了生成这些区间的几种方法及其优缺点。我们将回顾主要的方法，并对它们进行编码，以更好地巩固概念。

这篇文章的代码也可以在 Kaggle 和 Github 上找到。

# 天真的方法

当我们试图生成置信区间时，最先想到的可能是天真的方法。这个想法是使用我们模型的残差来估计我们可以从新的预测中得到多少偏差。

算法如下:

*   在训练集上训练模型
*   计算训练集预测的残差
*   选择残差分布的(1-alpha)分位数
*   从该分位数中减去每个预测的总和，以获得置信区间的极限

人们期望，由于残差的分布是已知的，新的预测应该不会偏离它太多。

然而，这种天真的解决方案是有问题的，因为我们的模型可能会过拟合，即使没有过拟合，大多数时候训练集的误差也会小于测试集的误差，毕竟，这些点是模型已知的。

这可能会导致过于乐观的置信区间。因此，永远不要使用这种方法。

为了更好地理解它，让我们编写这个方法。对于这个方法和帖子上的其他方法，我们将使用糖尿病数据集[2]，可以在商业用途的 sklearn 包中免费获得，也可以在[这里](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html)找到。我们将应用随机森林回归来创建预测。

首先，让我们导入所需的库:

```
import numpy as np
import pandas as pdfrom sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
```

现在，让我们导入数据集，定义 alpha，并将其分成训练集和测试集:

```
X, y = load_diabetes(return_X_y=True)
alpha = 0.05
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
```

我们现在将定义一个函数，该函数将为我们生成结果数据集:

```
def generate_results_dataset(preds, ci):
    df = pd.DataFrame()
    df['prediction'] = preds
    if ci >= 0:
        df['upper'] = preds + ci
        df['lower'] = preds - ci
    else:
        df['upper'] = preds - ci
        df['lower'] = preds + ci

    return df
```

该函数接收预测和 CI(残差分布的 1-alpha 分位数),并生成预测的下限和上限。

现在让我们生成简单的方法结果。为此，我们将拟合回归变量并计算训练残差，然后我们将获得测试集的分位数和预测，以使用我们的函数:

```
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)residuals = y_train - rf.predict(X_train)ci = np.quantile(residuals, 1 - alpha)
preds = rf.predict(X_test)df = generate_results_dataset(preds, ci)
```

人们可以使用 Kaggle 笔记本上的一个功能来可视化预测区间，以及预测值和真实值。

# 折叠法

刀切法试图克服朴素方法的过度乐观结果。为此，它不使用来自训练集的残差，而是使用来自留一预测的残差。

对于那些不知道的人来说，留一法包括为我们数据集中的每个数据点训练一个模型，在整个数据集上训练它，但一次移除一个样本。

通过这样做，残差是根据留一预测来计算的。如您所见，这是对测试集上每个点的预测，因为每个模型在训练阶段都没有看到该点。

然后，为了预测结果(或置信区间的中间点)，我们在整个训练集上拟合最后一个模型，并使用它来进行预测。

让我们看看这在代码中是什么样子的:

```
kf = KFold(n_splits=len(y_train)-1, shuffle=True, random_state=42)
res = []
for train_index, test_index in kf.split(X_train):
    X_train_, X_test_ = X_train[train_index], X_train[test_index]
    y_train_, y_test_ = y_train[train_index], y_train[test_index]

    rf.fit(X_train_, y_train_)
    res.extend(list(y_test_ - rf.predict(X_test_)))
```

为此，我们将使用 sklearn 库中的 KFold 类，使用拆分数量等于实例数量减 1，这将为留一法过程生成拆分。

然后，对于每个训练好的模型，我们计算残值并将其保存在一个列表中。

现在，是时候在整个训练集上拟合模型并生成结果了:

```
rf.fit(X_train, y_train)
ci = np.quantile(res, 1 - alpha)preds = rf.predict(X_test)
df = generate_results_dataset(preds, ci)
```

这种方法比天真的方法更有效，但是，它仍然存在一些问题:

*   留一法成本很高，在许多应用中可能不可行
*   当我们的回归的维数接近实例数时，这种方法就失去了覆盖面。

这种覆盖范围的丢失很奇怪，不是吗？这种情况背后的直觉是，适合整个训练集的模型比用于生成残差的每个其他模型多使用了一个点。这使得它们不能直接比较。

# 折叠+方法

Jackknife+方法试图解决由于模型适合整个训练集而导致的 Jackknife 方法的覆盖范围损失。

为此，它将使用所有留一训练模型来生成预测。这样，对区间中点进行预测的模型将与残差相当。

但它将如何做到这一点呢？根据这篇论文，直觉是我们得到了所有模型预测的中间值。然而，如果希望对该预测或多或少地保守，可以使用预测的分位数。

然而，在实践中，请注意，通常情况下，模型不会随着单点的丢失而改变太多其行为，因此分布通常具有很小的方差，这使得该方法产生与重叠法非常相似的结果。这种方法的优点来自于退化的情况，其中一个点可能极大地改变模型。

让我们把它编码起来:

```
kf = KFold(n_splits=len(y_train)-1, shuffle=True, random_state=42)
res = []
estimators = []
for train_index, test_index in kf.split(X_train):
    X_train_, X_test_ = X_train[train_index], X_train[test_index]
    y_train_, y_test_ = y_train[train_index], y_train[test_index]

    rf.fit(X_train_, y_train_)
    estimators.append(rf)
    res.extend(list(y_test_ - rf.predict(X_test_)))
```

这里的代码与我们在刀切法中使用的代码非常相似，但是，这一次我们还保存了用于生成残差的模型。

现在，我们的预测将有所不同，因为我们需要对每个模型进行预测:

```
y_pred_multi = np.column_stack([e.predict(X_test) for e in estimators])
```

现在，让我们来计算下限和上限:

```
ci = np.quantile(res, 1 - alpha)
top = []
bottom = []for i in range(y_pred_multi.shape[0]):
    if ci > 0:
        top.append(np.quantile(y_pred_multi[i] + ci, 1 - alpha))
        bottom.append(np.quantile(y_pred_multi[i] - ci, 1 - alpha))
    else:
        top.append(np.quantile(y_pred_multi[i] - ci, 1 - alpha))
        bottom.append(np.quantile(y_pred_multi[i] + ci, 1 - alpha))
```

最后，让我们使用中值预测来生成结果:

```
preds = np.median(y_pred_multi, axis=1)
df = pd.DataFrame()
df['pred'] = preds
df['upper'] = top
df['lower'] = bottom
```

现在，这种方法没有解决产生置信区间所花费的时间的问题。它只减少了一个拟合过程，但增加了每个模型的预测开销。

# CV+方法

CV+方法试图解决生成间隔所需的时间问题。它与 Jackknife+完全相同，但是，它使用 K-Fold 交叉验证，而不是留一法。

论文模拟表明，这种方法比刀切法稍差，但速度更快。实际上，这可能是大多数情况下使用的方法。

它的实现与 Jackknife+相同，我们只需要更改 for 循环以使用小于数据集长度的折叠数:

```
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

这个简单的更改将实现 CV+方法。

# 结论

这个新工具可能对许多需要为回归模型生成预测区间的数据科学家有用。此外，这些方法在 [MAPIE 库](https://mapie.readthedocs.io/en/latest/)上是开源的。

折叠刀+太贵了，用不上。但是，如果您已经在方法上使用了交叉验证，那么您可以应用 CV+方法，而不会有开销或性能损失。

[1]巴伯、里娜&坎迪斯、埃马纽埃尔&拉姆达斯、阿迪亚&蒂布拉尼、瑞安。(2021).用刀切+进行预测推理。统计年鉴。49.486–507.10.1214/20-AOS1965。

[2] Bradley Efron，Trevor Hastie，Iain Johnstone 和 Robert Tibshirani (2004)“最小角度回归”，统计年鉴(附讨论)，407–499