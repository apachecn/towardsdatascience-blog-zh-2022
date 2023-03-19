# 如何修复值错误:未知标签类型:scikit-learn 中的“连续”

> 原文：<https://towardsdatascience.com/valueerror-sklearn-fix-4a5fe105536a>

## 了解是什么导致了 scikit 中连续变量的 value error-learn 以及如何消除它

![](img/dbaf924c42c2f11a608ef0f9ac8f766d.png)

照片由 [Federica Giusti](https://unsplash.com/@federicagiusti?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/numbers?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

## 介绍

刚接触机器学习和 Python 编程的人在尝试使用`scikit-learn`包训练模型时可能会遇到一些麻烦。

最常报告的问题之一与目标变量的类型有关，当观察到不适当的值时，可能会触发特定的错误。其中一个错误是`ValueError: Unknown label type: 'continuous'`,其完整的回溯共享如下。

```
Traceback (most recent call last):
  File "test.py", line 14, in <module>
    clf.fit(train_X, train_Y)
  File "/usr/local/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py", line 1347, in fit
    check_classification_targets(y)
  File "/usr/local/lib/python3.7/site-packages/sklearn/utils/multiclass.py", line 183, in check_classification_targets
    raise ValueError("Unknown label type: %r" % y_type)
ValueError: Unknown label type: 'continuous'
```

在今天的简短教程中，我们将尝试重现该错误，理解为什么会出现这个异常，并最终展示如何处理它。

## 重现错误

首先，让我们创建一些样本数据，我们将使用这些数据来训练一个带有`scikit-learn`的示例逻辑回归模型。注意**目标变量是连续的**。

```
import numpy as np
from sklearn.linear_model import LogisticRegression train_X = np.array([
    [100, 1.1, 0.8],  
    [200, 1.0, 6.5],  
    [150, 1.3, 7.1],  
    [120, 1.2, 3.0],  
    [100, 1.1, 4.0],  
    [150, 1.2, 6.8],
])train_Y = np.array([1.0, 2.1, 5.6, 7.8, 9.9, 4.5])clf = LogisticRegression()
clf.fit(train_X, train_Y)
```

现在，如果我们尝试执行上述代码片段，标准输出会报告以下错误:

```
Traceback (most recent call last):
  File "test.py", line 14, in <module>
    clf.fit(train_X, train_Y)
  File "/usr/local/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py", line 1347, in fit
    check_classification_targets(y)
  File "/usr/local/lib/python3.7/site-packages/sklearn/utils/multiclass.py", line 183, in check_classification_targets
    raise ValueError("Unknown label type: %r" % y_type)
ValueError: Unknown label type: 'continuous'
```

本质上，错误告诉我们目标变量的类型是连续的，这与我们试图拟合的特定模型不兼容(即`LogisticRegression`)。

## 消除错误

对于许多初学者来说，**逻辑回归实际上用于分类**而不是回归的事实是一个惊喜。我认为这是完全可以理解的，因为新来的人可能会对车型名称感到困惑。

分类器(即应该执行分类的模型)期望标签(也称为目标变量)是分类的而不是连续的(这是模型在执行回归时的期望)。

因此，在使用逻辑回归(或几乎任何分类器)时，您的第一个选择是执行**标签编码**。但是请注意，这可能不是最合适的解决方案，答案实际上取决于您的特定用例！所以对此要有所保留。

在标签编码之前，我们可以使用`utils.multiclass.type_of_target`方法验证目标变量的类型。

```
>>> import utils
>>> print(utils.multiclass.type_of_target(train_Y))
'continuous'
```

现在，我们可以使用`scikit-learn`中的`[LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)`来执行标签编码，以便规范化目标变量。

```
from sklearn import preprocessinglabel_encoder = preprocessing.LabelEncoder()
train_Y = label_encoder.fit_transform(train_Y)
```

现在我们可以验证新编码的目标变量是多类类型:

```
>>> import utils
>>> print(utils.multiclass.type_of_target(train_Y))
'multiclass'
```

现在，我们可以使用编码的目标变量来训练我们的逻辑回归模型，没有任何问题:

```
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression# Label encoding
label_encoder = preprocessing.LabelEncoder()
train_Y = label_encoder.fit_transform(train_Y)# Model training
clf = LogisticRegression()
clf.fit(train_X, train_Y)
```

但是在使用标签编码之前，您可能仍然需要考虑逻辑回归是否适合您的特定用例。如果您想要对数据执行回归，那么您可能必须探索一些替代选项。

要了解更多关于机器学习环境中分类和回归问题之间的区别，请务必阅读我在 Medium 上的一篇旧文章，分享如下。

[](https://medium.com/analytics-vidhya/regression-vs-classification-29ec592c7fea)  

## 最后的想法

在今天的简短教程中，我们讨论了由`scikit-learn`分类器引发的`ValueError: Unknown label type: 'continuous'`错误，这些分类器根据特定模型预期的标签类型，在提供的目标变量中观察到无效值。

我们重现了这个错误，讨论了最初为什么会出现这个错误以及如何消除它。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读媒介上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**相关文章你可能也喜欢**

[](/scikit-learn-vs-sklearn-6944b9dc1736)  [](/predict-vs-predict-proba-scikit-learn-bdc45daa5972)  [](/training-vs-testing-vs-validation-sets-a44bed52a0e1) 