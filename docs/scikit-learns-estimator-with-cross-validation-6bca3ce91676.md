# 交叉验证的 Scikit Learn 估计器

> 原文：<https://towardsdatascience.com/scikit-learns-estimator-with-cross-validation-6bca3ce91676>

## 理解常规估计量和 CV 估计量的区别

![](img/0041fbc77d8774965081c8c949ad9a21.png)

照片由 [Redd](https://unsplash.com/@reddalec?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/crossing?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄

# 介绍

Scikit Lean 是最完整的 Python 机器学习库之一。数据科学家在许多项目中使用它，他们中的许多人实际上通过编写或审查代码来帮助项目。这有助于库不断发展，并在编程新模型时使我们的生活更加轻松。

查看 Scikit Learn 的文档，可以看到许多评估者现在都有交叉验证(CV)选项。比如有`LogisticRegression` 、`LogisticRegressionCV`，还有很多其他的，像`LassoCV`、`RidgeCV`。

这些估值器与常规估值器的不同之处在于，算法中有一个内置的交叉验证参数，帮助您在创建新模型时节省时间。

# 交互效度分析

交叉验证可以帮助我们数据科学家确保我们的模型在面对新数据时能够很好地推广。我总是喜欢将有监督的机器学习模型与学生准备考试进行比较。学生从教授那里得到内容，然后必须完成一些练习。但是练习不能总是相同的方式。

有时教授会给学生一份数据，要求学生计算缺失的部分。其他时候，学生接收公式和数据来输入和计算。就这样，教授混合了内容提供的方式，所以学生可以概括这个概念。每次收到问题时，只要有不同的数据，学生现在就可以应用获得的公式和知识来解决问题。

> 交叉验证是一种重采样方法，它使用数据的不同部分在不同的迭代中测试和训练模型。

与学生的类比就像交叉验证。我们是教授，模型是学生，公式和内容是算法。如果我们不断混合数据并将其呈现给模型，它可以进行归纳，一旦收到从未见过的数据进行预测，就有更大的成功机会。

# CV 估计量

好了，现在让我们进入这篇文章的核心。让我们来看看如何对带有交叉验证(CV)的估计量进行编码，以及它们的行为。

让我们导入所需的模块。

```
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.datasets import load_breast_cancer
```

我选择了逻辑回归估计量来测试它。所以，让我们开始使用那些著名的玩具数据集来进行演示。我们将加载来自 **sklearn** 的乳腺癌数据进行预测。

*(边注:我们会跳过数据清理和探索，所以这篇帖子不会太长。但是我们仍然可以看到评估者在起作用)*

```
#Split explanatory (X) and explained (y)
X, y = load_breast_cancer(return_X_y=True)
```

首先，我们可以运行常规的`LogisticRegression()`。

```
# Regular Logistic Regression
log_reg = LogisticRegression().fit(X, y)
```

我们来看看比分。

```
log_reg.score(X,y)**[OUT]:
0.9472759226713533**
```

现在，让我们看看带有 CV 的估计量是如何表现的。代码差别不大。我们将使用超参数 cv=10，将交叉验证折叠的数量添加到训练中。

```
# Fit the model with CV
log_cv = LogisticRegressionCV(cv=10, random_state=0).fit(X, y)# Score
log_cv.score(X, y)**[OUT]:
0.961335676625659**
```

在这种情况下，输出提高了 2%。但这是否意味着它会一直那样呢？我们应该总是使用带有 CV 的估计量吗？

简单的答案是否定的。

例如，有时使用交叉验证有助于使模型过度适应训练数据。我能想到的另一个例子是时间问题。根据数据的大小，训练时间可能会随着 CV 而显著增加。

其他时候，如果这是我们在项目中使用的度量标准，这对于提高准确性来说是不够的。让我们看下一个例子。

```
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
```

我们将使用 sklearn `make_classification`创建一个数据集。

```
# Creating a sample dataset
data = make_classification(n_samples= 5000, n_features= 9,
                           n_classes=2,random_state=42) X = pd.DataFrame(data[0], columns=['V' + str(i) for i in range(1,10)])y= data[1]
```

我们可以在训练和测试中拆分它。

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

现在让我们试着拟合一个逻辑回归。

```
# Logistic Regression
model = LogisticRegression()# Fit
model.fit(X_train, y_train)#score
model.score(X_test, y_test)**[OUT]:
0.856**
```

如果我们使用 CV 选项，结果如下。

```
# Logistic Regression with CV
model_cv = LogisticRegressionCV(cv=10, random_state=42)# Fit
model_cv.fit(X_train, y_train)#score
model_cv.score(X_test, y_test)**[OUT]:
0.856**
```

同样的结果。准确率 85.6%。

# 在你走之前

在这篇文章中，我们了解到 **sklearn** 已经内置了带有交叉验证的估计器。一般来说，需要做的只是在实例化模型时使用超参数`cv`。

有时，结果会改善，有时则不会。对于大型数据集，训练时间也会增加，因此请记住这一点。

如果你喜欢这篇文章，请随时关注我

<http://gustavorsantos.medium.com/>  

如果你正在考虑中等会员资格，[这里有一个推荐代码](https://gustavorsantos.medium.com/membership)给你。它帮助并激励着我！

# 参考

<https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29>  <https://stats.stackexchange.com/questions/320154/when-not-to-use-cross-validation>  <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV> 