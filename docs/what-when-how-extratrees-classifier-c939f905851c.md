# 什么？什么时候？怎么会？:树外分类器

> 原文：<https://towardsdatascience.com/what-when-how-extratrees-classifier-c939f905851c>

## 什么是树外量词？什么时候用？如何实施？

![](img/18faa17df5b3d4a38ea7763910e6e131.png)

由 [Unsplash](https://unsplash.com/s/photos/questioning?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的[Eunice lituaas](https://unsplash.com/@euniveeerse?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)拍摄的照片

在过去的十年中，基于树的模型越来越受欢迎，主要是因为它们的鲁棒性。基于树的模型可以用于任何类型的数据(分类/连续)，可以用于非正态分布的数据，并且几乎不需要任何数据转换(可以处理缺失值/比例问题等)。)

虽然决策树和随机森林通常是基于树的模型，但不太为人所知的是 ExtraTrees。(如果你不熟悉基于树的模型，一定要看看下面的[帖子](/understanding-random-forest-58381e0602d2))。

# 什么是树外模型？

与随机森林类似，ExtraTrees 是一种集合 ML 方法，它训练大量决策树并聚集来自决策树组的结果以输出预测。然而，额外的树和随机森林之间几乎没有区别。

随机森林使用[打包](/understanding-random-forest-58381e0602d2)来选择训练数据的不同变化，以确保决策树足够不同。但是，Extra Trees 使用整个数据集来训练决策树。因此，为了确保各个决策树之间有足够的差异，它会随机选择分割特征和创建子节点的值。相比之下，在随机森林中，我们使用算法进行贪婪搜索，并选择分割要素的值。除了这两个区别，随机森林和额外的树基本上是相同的。那么这些变化有什么影响呢？

*   使用整个数据集(这是默认设置，可以更改)允许提取树减少模型的偏差。但是，分割时特征值的随机化会增加偏差和方差。介绍额外树模型的[论文对不同基于树的模型进行了偏差-方差分析。**从论文中我们看到，在大多数分类和回归任务(分析了六个)中，抽提树比随机森林有更高的偏差和更低的方差。**然而，论文继续说这是因为额外树中的随机化将无关的特征包括到模型中。这样，当不相关特征被排除时，比方说通过特征选择预建模步骤，额外的树得到类似于随机森林的偏差分数。](https://orbi.uliege.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf)
*   **就计算成本而言，额外的树比随机森林要快得多。**这是因为 Extra Trees 随机选择分割要素的值，而不是随机森林中使用的贪婪算法。

# 什么时候应该使用提取物？

![](img/45fc505fd01360e44e995f2ae15eb80a.png)

照片由 [Jens Lelie](https://unsplash.com/@madebyjens?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/decision?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

随机森林仍然是基于集成树的模型(最近有来自 [XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) 模型的竞争)。然而，从我们之前对随机森林和额外树之间的差异的讨论中，我们看到了 Extra Trees 的价值，尤其是当计算成本是一个问题时。**具体来说，在构建具有大量特征工程/特征选择预建模步骤的模型时，计算成本是一个问题，相比其他基于系综树的模型，ExtraTrees 是一个不错的选择。**

# 如何建立一个 ExtraTrees 模型？

ExtraTrees 可用于构建分类模型或回归模型，可通过 Scikit-learn 获得。在本教程中，我们将介绍分类模型，但是代码可以用于稍加调整的回归(例如，从 ExtraTreesClassifier 切换到 ExtraTreesRegressor)

**建立模型**

我们将使用 Scikit-learn 中的 make_classification 来创建虚拟分类数据集。为了评估该模型，我们将使用 10 重交叉验证，以准确性作为评估标准。

**超参数调谐**

额外树模型的详细参数列表可在 [Scikit-learn 页面](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)上找到。 [Extra Trees 研究论文](https://orbi.uliege.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf)明确提出了三个关键参数，陈述如下。

*“参数 K、nmin 和 M 具有不同的效果:K 确定属性选择过程的强度，nmin 确定平均输出噪声的强度，M 确定集合模型聚合的方差减少的强度。”*

让我们从实现的角度更仔细地看看这些参数。

*   **K** 是 Scikit-learn 文档中的 max_feature，指每个决策节点要考虑的特性数量。K 值越高，每个决策节点考虑的特征越多，因此模型的偏差越低。然而，过高的 K 值降低了随机化，否定了系综的效果。
*   **nmin** 映射到 min_sample_leaf，并且是在叶节点所需的最小样本数。其值越高，模型越不可能过度拟合。样本数量越少，分裂越多，树越深，越专门化。
*   **M** 映射到 n_estimators，是森林中树的数量。其值越高，模型的方差越低。

如下所示，可通过 [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 选择最佳参数集。

# 最终外卖

![](img/0404d12dfa4fe695274b9ce9ab0f45b3.png)

照片由 [Riccardo Annandale](https://unsplash.com/@pavement_special?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/result?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

*   ExtraTrees 分类器是一种基于系综树的机器学习方法，它使用依赖随机化来减少方差和计算成本(与随机森林相比)。
*   树外分类器可用于分类或回归，在这种情况下，计算成本是一个问题，特征已仔细选择和分析。
*   额外的树可以从 Scikit-learn 实现。对于调优很重要的三个超参数是 max_feature、min_samples_leaf 和 n_estimators。

就是这样！树外之物，何时，如何！

# 参考

  <https://machinelearningmastery.com/extra-trees-ensemble-with-python/>  <https://stats.stackexchange.com/questions/175523/difference-between-random-forest-and-extremely-randomized-trees> 