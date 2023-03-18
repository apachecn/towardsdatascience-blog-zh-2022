# 你的模型超过基线了吗？

> 原文：<https://towardsdatascience.com/does-your-model-beat-the-baseline-3b9fb31cbe76>

## 让我们将我们的模型与一个普通的基线进行比较

![](img/7fae093a1a81cfc39d73e8fa4f527f2a.png)

*图片 by Pixabay:*[T3【https://www.pexels.com/it-it/foto/lampadina-chiara-355948/】](https://www.pexels.com/it-it/foto/lampadina-chiara-355948/)

每次我们训练一个模型时，我们都应该检查它的性能是否超过了某个基线，这是一个没有考虑输入的琐碎模型。将我们的模型与基线模型进行比较，我们实际上可以计算出它实际上是否学习了。

# 什么是基线模型？

基线模型是一种实际上不使用特征的模型，但对所有预测使用一个平凡的常量值。对于一个回归问题，这样的值通常是训练数据集中目标变量的平均值(10 年前，我曾经进行 ANOVA 测试来比较线性模型和这样一个微不足道的模型，这种模型被称为零模型)。对于分类任务，普通模型只返回训练数据集中最频繁的类。

因此，这是我们数据集的基线，一个经过适当训练的模型应该能够超越这种算法的性能。事实上，如果一个模型像基线一样执行，它实际上没有考虑特性，所以它不是学习。请记住，基线模型的给定定义根本不使用特性，它们只是以某种方式对目标值进行平均。

在这篇文章中，我们将会看到如何比较一个模型和一个基线模型。

# 战略

总的想法是用模型和基线模型在测试数据集上计算一些性能指标。然后，使用 [bootstrap](https://www.yourdatateacher.com/2021/04/19/the-bootstrap-the-swiss-army-knife-of-any-data-scientist/) ，我们计算这种测量的 [95%置信区间](https://www.yourdatateacher.com/2021/11/08/how-to-calculate-confidence-intervals-in-python/)。如果间隔不重叠，则模型不同于基线模型。

按照我们选择的性能指标，我们允许模型实际上具有可比性的概率高达 5%。不幸的是，只看指标的平均值是不够的。小数据集可能会引入有限大小的效应，使我们的分析变得不可靠。这就是为什么我更喜欢使用 bootstrap 来计算置信区间，这让我们更好地了解情况，从我们的数据集中提取尽可能多的信息。

# Python 中的一个例子

在 Python 中，基线模型由 DummyClassifier 和 DummyRegressor 对象表示。前者考虑训练数据集中最频繁的目标类，后者考虑目标变量的平均值。这些设置是可以改变的(例如，通过考虑中间值来代替平均值)，但我通常更喜欢使用默认设置，因为它们非常真实和有用。

对于回归问题，我们将使用“糖尿病”数据集和随机森林分类器。我们的绩效指标将是 r 平方分数。

让我们先导入一些库:

```
import numpy as np
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine, load_diabetes
from sklearn.dummy import DummyClassifier,DummyRegressor
from sklearn.metrics import accuracy_score, r2_score
```

然后，让我们导入数据集:

```
X,y = load_diabetes(return_X_y = True)X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.4,random_state=0)
```

我们现在可以训练随机森林和虚拟回归器。

```
model = RandomForestRegressor(random_state=0) model.fit(X_train,y_train) dummy = DummyRegressor() 
dummy.fit(X_train,y_train)
```

最后，通过 500 次迭代的 bootstrap，我们可以根据相同的测试数据集计算模型的 r 平方的置信区间。

```
scores_model = []
scores_dummy = []
for n in range(500):
  random_indices = np.random.choice(range(len(X_test)),size=len(X_test),replace=True)
  X_test_new = X_test[random_indices]
  y_test_new = y_test[random_indices] scores_model.append(r2_score(y_test_new,model.predict(X_test_new)))
 scores_dummy.append(r2_score(y_test_new,dummy.predict(X_test_new)))
```

最后，这些是模型和虚拟分类器的置信区间:

```
np.quantile(scores_model,[0.025,0.975]),np.quantile(scores_dummy,[0.025,0.975]) # (array([0.20883809, 0.48690673]), array([-3.03842778e-02, -7.59378357e-06]))
```

正如我们所看到的，区间是不相交的，并且与模型相关的区间的下限大于与虚拟模型相关的区间的上限。因此，我们可以说我们的模型在 95%的置信度下比基线表现得更好。

使用分类器可以遵循相同的方法。在本例中，数据集将是“葡萄酒”数据集。评分标准将是准确性得分。

```
X,y = load_wine(return_X_y = True) X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=0)
```

以下是模型:

```
model = RandomForestClassifier(random_state=0) 
dummy = DummyClassifier() 
model.fit(X_train,y_train) 
dummy.fit(X_train,y_train)
```

这是两个模型的准确性得分的自举:

```
scores_model = []
scores_dummy = []
for n in range(500):
  random_indices = np.random.choice(range(len(X_test)),size=len(X_test),replace=True)
  X_test_new = X_test[random_indices]
  y_test_new = y_test[random_indices] scores_model.append(accuracy_score(y_test_new, model.predict(X_test_new)))
  scores_dummy.append(accuracy_score(y_test_new, dummy.predict(X_test_new)))
```

最后，这些是置信区间:

```
np.quantile(scores_model,[0.025,0.975]),np.quantile(scores_dummy,[0.025,0.975]) # (array([0.91666667, 1\. ]), array([0.31215278, 0.54166667]))
```

同样，该模型以 95%的置信度比虚拟模型表现得更好。

# 结论

在本文中，我展示了一种技术来评估一个简单的基线模型的性能。尽管这经常被忽视，但这种比较很容易进行，而且必须经常进行，以便评估我们的模型的稳健性及其泛化能力。

*原载于 2022 年 7 月 3 日*[*【https://www.yourdatateacher.com】*](https://www.yourdatateacher.com/2022/07/04/does-your-model-beat-the-baseline/)*。*