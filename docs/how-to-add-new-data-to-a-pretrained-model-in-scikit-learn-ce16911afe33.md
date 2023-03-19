# 如何在 Scikit-learn 中向预训练模型添加新数据

> 原文：<https://towardsdatascience.com/how-to-add-new-data-to-a-pretrained-model-in-scikit-learn-ce16911afe33>

## 机器学习

## 关于如何在 scikit-learn 中使用 warm_start=True 和 partial_fit()的分步教程

![](img/4058220778a58638e208f53bad3fa0f9.png)

照片由[h·海尔林](https://unsplash.com/@heyerlein?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

当您从头开始构建机器学习模型时，通常会将数据集分为训练集和测试集，然后在训练集上训练您的模型。然后，您在测试集上测试您的模型的性能，如果您得到一些不错的东西，您可以使用您的模型进行预测。

**但是，如果新数据在某个时候变得可用，该怎么办呢？**

换句话说，如何训练一个已经训练好的模型？还是那句话，如何给一个已经训练好的模型添加新的数据？

在本文中，我试图使用 scikit-learn 库来回答这个重要的问题。你可以查看[vid hi Chugh](/when-are-you-planning-to-retrain-your-machine-learning-model-5349eb0c4706)的这篇有趣的文章，了解你什么时候需要重新训练你的模型。

对于前一个问题，一个可能的(琐碎的)解决方案是通过使用新旧数据从头开始训练模型。然而，如果第一次训练需要很长时间，这种解决方案就不能扩展。

问题的解决方案是向已经训练好的模型中添加样本。这个 scikit-learn 允许你在某些情况下这样做。只要遵循一些预防措施。

Scikit-learn 提出了两种策略:

*   部分拟合
*   热启动

为了说明如何在 Scikit-learn 中向预训练模型添加新数据，我将使用一个实际的例子，使用由 Scikit-learn 库提供的众所周知的 [iris 数据集](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)。

# 热启动

热启动是一些 Scikit 型号提供的参数。如果设置为 True，则允许在随后的 f it 调用中使用现有的 fit 模型属性来初始化新模型。

例如，您可以在随机森林分类器中设置`warm_start = True`，然后您可以定期拟合模型。如果对新数据再次调用 fit 方法，新的估计器将被添加到现有的树中。这意味着使用 warm_start = True 不会改变现有的树。

`warm_start = True`是否应该**而不是**用于在可能存在**概念漂移**的新数据集上进行增量学习。概念漂移是数据模型中的一种漂移，它发生在输出和输入变量之间的基本关系发生变化时。

为了理解`warm_start = True`是如何工作的，我描述一个例子。这个想法是为了表明，如果我添加新数据，使用 warm_start = True 可以提高算法的性能，新数据与原始数据具有相同的分布，并且与输出变量保持相同的关系。

首先，我加载由 Scikit-learn 库提供的 iris 数据集:

```
from sklearn import datasetsiris = datasets.load_iris()
X = iris.data
y = iris.target
```

然后，我将数据集分成三部分:

*   `X_train`， `y_train` —40%数据的 80%训练集(48 个样本)
*   `X_test`， `y_test` —测试集 40 个数据的 20%(12 个样本)
*   `X2, y2` —新样本(60%的数据)(90 个样本)

```
from sklearn.model_selection import train_test_splitX1, X2, y1, y2 = train_test_split(X, y, test_size=0.60, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.20, random_state=42)
```

我会用`X2`和`y2`重新训练模型。

请注意，训练集非常小(48 个样本)。

我训练模型，用`warm_start = False`:

```
from sklearn.ensemble import RandomForestClassifiermodel = RandomForestClassifier(max_depth=2, random_state=0, warm_start=False, n_estimators=1)
model.fit(X_train, y_train)
```

我算了一下分数:

```
model.score(X_test, y_test)
```

它给出了以下输出:

```
0.75
```

现在，我用新数据来拟合这个模型:

```
model.fit(X2, y2)
```

先前的拟合删除了已经学习的模型。然后，我计算分数:

```
model.score(X_test, y_test)
```

它给出了以下输出:

```
0.8333333333333334
```

现在我用 warm_start = True 构建一个新模型，看看模型得分是否增加。

```
model = RandomForestClassifier(max_depth=2, random_state=0, warm_start=True, n_estimators=1)
model.fit(X_train, y_train)
model.score(X_test, y_test)
```

它给出了以下输出:

```
0.75
```

现在，我拟合模型并计算分数:

```
model.n_estimators+=1
model.fit(X2, y2)
model.score(X_test, y_test)
```

它给出了以下输出:

```
0.9166666666666666
```

增量学习提高了分数！

# 部分拟合

Scikit-learn 提供的向预训练模型添加新数据的第二个策略是使用`partial_fit()`方法。并非所有的模型都提供这种方法。

虽然`warm_start = True`参数不改变模型已经学习的属性参数，但是部分拟合可以改变它，因为它从新数据中学习。

我再次考虑虹膜数据集。

现在我用一个`SGDClassifier`:

```
from sklearn.linear_model import SGDClassifier
import numpy as npmodel = SGDClassifier() 
model.partial_fit(X_train, y_train, classes=np.unique(y))
```

第一次运行 partial_fit()方法时，我还必须将所有的类传递给该方法。在这。例如，我假设我知道 y 中包含的所有类，尽管我没有足够的样本来表示它们。

我算了一下分数:

```
model.score(X_test, y_test)
```

它给出了以下输出:

```
0.4166666666666667
```

现在，我将新样本添加到模型中:

```
model.partial_fit(X2, y2)
```

我计算分数:

```
model.score(X_test, y_test)
```

它给出了以下输出:

```
0.8333333333333334
```

添加新数据提高了算法的性能！

# 摘要

恭喜你！您刚刚学习了如何在 Scikit-learn 中向预训练模型添加新数据！您可以使用设置为`True`的`warm_start`参数或`partial_fit()` 方法。然而，并非 Scikit-learn 库中的所有模型都提供向预训练模型添加新数据的可能性。因此，我的建议是检查文档！

你可以从我的 [Github 库](https://github.com/alod83/data-science/blob/master/DataAnalysis/Add%20New%20Data%20to%20a%20Pretrained%20Model.ipynb)下载本教程中使用的代码。

如果你读到这里，对我来说，今天已经很多了。谢谢！你可以在[这个链接](https://alod83.medium.com/my-most-trending-articles-4fbfbe107fb)阅读我的趋势文章。

# 相关文章

[](/how-to-run-a-data-science-project-in-a-docker-container-2ab1a3baa889)  [](/an-overview-of-the-scikit-learn-clustering-package-d39a0499814)  [](/model-evaluation-in-scikit-learn-abce32ee4a99)  

# 保持联系！

*   在[媒体](https://medium.com/@alod83?source=about_page-------------------------------------)上跟随我
*   注册我的[简讯](https://medium.com/subscribe?source=about_page-------------------------------------)
*   在 [LinkedIn](https://www.linkedin.com/in/angelicaloduca/?source=about_page-------------------------------------) 上连接
*   在推特上关注我
*   跟着我上[脸书](https://www.facebook.com/alod83?source=about_page-------------------------------------)
*   在 Github 上关注我