# 如何用低代码解决每一个 ML 问题

> 原文：<https://towardsdatascience.com/how-to-solve-every-ml-problem-with-low-code-97550c712a68>

## Python 库 PyCaret 简介

![](img/c6a674a58638ef682191d757a9644361.png)

阿诺·弗朗西斯卡在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

# 介绍

根据《财富商业洞察》[1]，全球机器学习(ML)市场规模预计将从 2021 年的 150 亿美元增长到 2029 年的 2090 亿美元。这意味着对数据科学家的需求正在上升，而人才供应仍然稀缺。

因此，引入了一个新的角色。公民数据科学家的角色。

公民数据科学家的角色可以描述如下[2]:

> 根据 Gartner 的说法，公民数据科学家是指创建或生成利用预测性或规范性分析的模型的人，但其主要工作职能不属于统计和分析领域。

因此，一个公民数据科学家应该解决 ML 问题，而不需要在统计和分析领域有太多的知识。

为了使这些人能够解决 ML 问题，而不必运行探索性的数据分析、数据预处理、数据清洗、模型选择和所有其他需要的步骤，需要低代码和端到端的机器学习和模型管理工具。

这就是 Python 库 PyCaret 发挥作用的地方。

在本文中，我将向您介绍 Python 库 PyCaret。这个库允许你只用几行代码就可以很容易地训练一个 ML 模型！

我将通过解决一个具体的 ML 问题并为您提供代码片段，直接向您展示这个库的威力。

因此，如果你有兴趣成为一名公民数据科学家，或者你已经是一名数据科学家，但想学习如何简化培训过程，那么请继续阅读！

# PyCaret

让我们首先深入研究 PyCaret。下面引用的 PyCaret 官方文档很好地总结了 PyCaret 及其用途[3]:

> PyCaret 是一个用 Python 编写的开源、低代码的机器学习库，可以自动化机器学习工作流。这是一个端到端的机器学习和模型管理工具，可以成倍地加快实验周期，提高您的工作效率。

根据文档，PyCaret 可用于分类、回归、聚类、异常检测和自然语言处理。

他们还提供教程甚至数据集，让用户有可能快速测试这个库。

因为这是一个 Python 库，所以您可以使用以下命令通过 pip 轻松安装它:

```
pip install pycaret
```

PyCaret 本身是几个其他 ML 库的 Python 包装器，比如 scikit-learn、CatBoost、XGBoost 等等。

该库允许您解决 ML 问题，而不需要成为有经验的数据科学家，因此非常适合公民数据科学家。

但是它也帮助数据科学家快速解决 ML 问题，因为只有这个库的一部分可以用于解决 ML 问题。例如，数据科学家可以自己准备数据，然后利用 PyCaret 找到该数据的最佳模型。

# 用 PyCaret 解决分类问题

现在让我们展示一下使用 PyCaret 解决一个具体的 ML 问题的能力和简易性。

你可以在我的 Github 库[这里](https://github.com/patrickbrus/Medium_Notebooks/tree/master/Low_Code_ML)找到包含所有代码的完整笔记本。

PyCaret 提供了几个数据集，可以用来快速测试他们的库的能力。

您可以通过运行以下命令获得所有可用的数据集:

```
from pycaret.datasets import get_data

credit_data = get_data("index")
```

其中一个数据集是信用数据集，这是我在本文中使用的数据集。该数据集被称为“信用卡客户数据集的默认值”。你也可以在 [Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) 上找到更多关于那个数据集的信息。

基本上包含了 2005 年 4 月到 2005 年 9 月台湾地区信用卡客户拖欠还款的信息。

您可以使用以下命令加载数据集:

```
# load credit dataset
credit_data = get_data("credit")
```

该数据集总共包含 **24000** 个样本，每个样本有 **24** 个特征。

作为第一步，我创建了一个拒绝测试集。最后，这将用于查看最终模型在真实世界数据中的“真实”性能。

为此，我使用了 scikit-learn 的训练测试分割功能:

```
from sklearn.model_selection import train_test_split

# create a train and test split
data_train, data_test = train_test_split(credit_data, test_size=0.1)
```

## 设置数据管道和环境

解决 ML 问题的第一步总是建立你的数据管道和你的环境。设置数据管道通常包括运行一些探索性的数据分析以更好地理解数据、进行特征工程以充分利用数据、进行数据清理以移除缺失值等等。

使用 PyCaret，所有这些步骤都是 **setup()** 函数的一部分。这个函数基本上创建了为训练准备数据的所有转换。

它接受一个 Pandas 数据帧和目标列的名称。

除了这些值，您还可以添加更多的可选值来设置您的数据管道。例如，您可以对数据进行归一化处理，将其转换为更接近高斯分布的形式，忽略方差较小的要素，移除相关性较高的要素，选择用于输入缺失数据的数据插补类型，应用降维，移除异常值，计算用于创建新要素的组要素统计数据等等。您可以在这里找到分类用例[的所有输入选项。](https://pycaret.readthedocs.io/en/stable/api/classification.html)

现在，让我们使用用于训练的数据帧，并使用以下选项运行 **setup()** 函数:

```
# call the pycaret setup function to setup the transformation pipeline
setup = setup(data=data_train, 
              target="default",
              normalize=True,
              transformation=True, # make data more gaussian like
              ignore_low_variance=True, # ignore features with low variance
              remove_multicollinearity=True, # remove features with inter-correlations higher than 0.9 -> feature that is less correlated with target is removed
              group_features = [['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'], # features with related characteristics
                                ['PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']],
              use_gpu=True # use GPU when available, otherwise CPU                                               
              )
```

完成了。通过运行此功能，您将应用预处理、标准化、转换、一些特征工程和移除高度相关的特征。所有这些只需要一个函数调用和几行代码。太棒了。

## 比较所有模型

解决 ML 问题的下一步是比较几种不同的模型以找到最佳选择。

PyCaret 再次提供了一个函数，通过使用 k-fold 交叉验证来评估模型库中所有可用估计器的性能。

让我们现在运行这个函数，并返回 3 个最佳选项:

```
from pycaret.classification import *

best_3_models = compare_models(n_select=3)
```

然后，该功能将运行 10 重交叉验证，以评估几个分类模型的性能，并输出几个指标。

您可以在图 1 中找到对信用卡数据集运行该函数的输出。

![](img/4bb1b32f4270a0ee86760dd8910477a2.png)

图 1:所有可用分类模型的比较结果(图片由作者提供)。

具有良好 AUC 分数和 F1 分数的模型是梯度增强分类器。因此，我决定在本文的剩余部分继续讨论这个问题。

我没有根据准确性选择一个好的模型，因为数据集本身是不平衡的，准确性不会提供对性能的良好估计。但是我将在本文的后面回到这一点。

## 优化最终模型

现在我们有了一个可以继续进行的模型，是时候优化这个模型了。

当然，PyCaret 也提供了一个调整模型的函数。但是在调优之前，我们必须调用 **create_model()** 函数。此函数使用交叉验证来训练和评估模型。但是这个函数只是利用所选分类器的默认参数来训练它。

为了调优超参数，有一个名为 **tune_model()** 的函数。该函数接受一个训练模型，并基于一个选择的优化度量和几个其他选项来优化超参数。一个选项是设置搜索算法，可以是随机的、网格搜索、贝叶斯搜索等等。默认为随机网格搜索。

现在让我们训练一个具有 5 重交叉验证的梯度增强分类器，然后使用 F1 分数作为优化的度量来调整超参数:

```
final_classifier = create_model("gbc", fold=5)

tuned_classifier_F1 = tune_model(final_classifier, optimize="F1")
```

未调整的模型的 AUC 得分为 77.4%，F1 得分为 36.2%，而调整后的模型的 AUC 得分为 77.3%，F1 得分为 47%。

现在，我们还可以通过运行以下命令来获取导致最佳模型结果的所有超参数:

```
plot_model(tuned_classifier_F1, plot="parameter")
```

PyCaret 还提供了其他一些现成的绘图。

一个示例图是 ROC 曲线，它可以通过运行以下函数调用来绘制:

```
plot_model(tuned_classifier_F1, plot="auc")
```

图 2 显示了得到的 ROC 曲线。

![](img/92834222270a36ac6a813e6fb0b0360e.png)

图 PyCaret 函数调用返回的 ROC 曲线(图片作者提供)。

您还可以获得您的特征对预测目标变量的重要性(图 3):

```
plot_model(tuned_classifier_F1, plot="feature")
```

![](img/d78f39d4d9f59f262f75f70e3044a511.png)

图 PyCaret 函数调用返回的特性重要性图(图片由作者提供)。

或者我们也可以看看混淆矩阵，因为数据集是不平衡的(图 4):

```
plot_model(tuned_classifier_F1, plot="confusion_matrix")
```

![](img/06e21593b4cf708ee36c16e2bb2251b8.png)

图 PyCaret 函数调用返回的混淆矩阵(图片由作者提供)。

也有很多其他选项可用于绘图，可以在官方用户文档中找到。

正如您在混淆矩阵中看到的，分类器在多数类(类 0)上表现良好，而在少数类(类 1)上表现不佳。在现实世界中，这将是一个巨大的问题，因为得到正确的 1 类可能比 0 类更重要，因为预测客户将支付其违约，但实际上他没有支付将导致损失。

因此，我们可以尝试修复数据集中的不平衡，以检索更好的模型。为此，我们只需稍微修改 **setup()** 功能，将**fix _ unbalance**输入参数设置为 true:

```
# call the pycaret setup function to setup the transformation pipeline
setup = setup(data=data_train, 
              target="default",
              normalize=True,
              transformation=True, # make data more gaussian like
              ignore_low_variance=True, # ignore features with low variance
              remove_multicollinearity=True, # remove features with inter-correlations higher than 0.9 -> feature that is less correlated with target is removed
              group_features = [['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'], # features with related characteristics
                              ['PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']],
              fix_imbalance=True, # fix the imbalance in the dataset
              use_gpu=True # use GPU when available, otherwise CPU                                               
             )
```

然后，我们可以再次调整优化 F1 分数的模型:

```
# train baseline model
final_classifier = create_model("gbc", fold=5)

# optimize hyperparameters by optimizing F1-Score
tuned_classifier_F1 = tune_model(final_classifier, optimize="F1")

# plot new confusion matrix
plot_model(tuned_classifier_F1, plot="confusion_matrix")
```

![](img/0b43138db17e925404c836a4cc608f7d.png)

图 PyCaret 函数调用返回的不平衡优化模型的混淆矩阵(图片由作者提供)。

新模型现在的 AUC 分数为 77%，F1 分数为 53.4%。F1 分数从 47%提高到 53.4%！

混淆矩阵(图 5)现在也显示了类 1 的性能有所提高！太好了。但这是以类 0 的更差性能为代价的，这通常没有错误预测类 1 那么严重。

当比较两个模型的回忆分数时，这也变得清楚。老款召回 37%，新款召回差不多 60%。这意味着新模型发现了 60%的未偿付违约，而旧模型只发现了 37%。

## 在搁置测试集上测试最终模型

最后一步是在保留测试集上测试最终模型，看看它在真实世界和看不见的数据上表现如何。

为此，我们现在可以利用本文开头创建的拒绝测试集:

```
predict_model(tuned_classifier_F1, data=data_test)
```

该函数将采用调整后的梯度提升分类器，并对保留测试集进行预测。然后，它返回预测的指标。

AUC 得分为 77%，F1 得分为 54.3%。这向我们表明，该模型没有过度拟合我们的训练和验证数据，并且在真实世界数据上表现相似。

# 结论

在本文中，介绍了 PyCaret Python 库，并将其应用于现实世界的数据集，以解决分类问题。它清楚地显示了使用这个库可以多么容易地解决一个 ML 问题，我只是触及了表面。还有很多其他选项可供您探索和评估。

因此，这个库是公民数据科学家在没有太多统计和其他相关领域知识的情况下解决 ML 问题的完美工具。但它也可以提高数据科学家找到解决方案的效率和速度。

我也没有深入研究数据集本身，因为本文的目的是展示这个库的威力。当然，最终的估计值仍然可以优化，PyCaret 也提供了更多可以进一步研究的选项。

但是这超出了本文的范围。

谢谢你把我的文章看完！我希望你喜欢这篇文章。如果你想在未来阅读更多类似的文章，请关注我，保持更新。

# 接触

[**LinkedIn**](https://www.linkedin.com/in/patrick-brus/)|**[**GitHub**](https://github.com/patrickbrus)**

# **参考**

**[1]《财富商业洞察》，[机器学习市场规模](https://www.fortunebusinessinsights.com/machine-learning-market-102226) (2022 年)**

**[2] Manasi Sakpal，[公民数据科学家可以提升组织的商业价值和分析成熟度；然而，在大多数情况下，他们的能力仍未得到充分利用](https://www.gartner.com/smarterwithgartner/how-to-use-citizen-data-scientists-to-maximize-your-da-strategy) (2021)**

**[3] Ali Moez，[py caret:Python 中的开源、低代码机器学习库](https://www.pycaret.org) (2020)**