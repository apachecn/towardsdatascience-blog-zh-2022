# 让脚本找出优于您的 ML 模型

> 原文：<https://towardsdatascience.com/python-automl-sklearn-fd85d3b3c5e>

## 这个自动 ML Python 库让每个人都成为数据科学向导

![](img/913424f02870fbf7050e73a15a601dd3.png)

将数据科学交给 AutoML —照片由 [Aidan Roof](https://www.pexels.com/@rozegold?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) 从 [Pexels](https://www.pexels.com/photo/person-on-truck-s-roof-2449600/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) 拍摄

自动化是一个古老的概念，它改变了一切。我们发明的所有机器和技术都是一种方式，或者说是一种自动化形式。

数据科学领域是我们自动化思维的一种方式。我们不是手动给物体贴标签，而是训练计算机大规模地这样做。

但是教授计算机的过程在很长一段时间里是手工操作的；直到最近，科学家们还在考虑 AutoML。

构建预测系统是一个多阶段的过程。它可能包括

*   收集数据(或整合数据源；)
*   准备数据集；
*   训练机器学习模型和调整超参数；
*   在生产系统上部署模型，以及；
*   监控已部署模型的性能。

Auto ML 是一个术语，指的是在这些阶段中消除手工操作的技术。

[](/how-to-become-a-citizen-data-scientist-294660da0494)  

很多数据科学家用 Python，大部分用 scikit-learn。很难找到不是这样的人。

Scikit-learn 提供了优秀的高级 API 来构建 ML 模型。他们删除了大量样板代码，否则这些代码会让科学家们抓狂。

这篇文章是关于一个将 scikit-learn 推进了一步的工具——auto-sk learn。你可以从[高效稳健的自动化机器学习](https://papers.neurips.cc/paper/5872-efficient-and-robust-automated-machine-learning)，福雷尔*等人*，神经信息处理系统进展 28 (NIPS 2015)中了解更多。

# 为什么我们需要在 sklearn 之上安装 auto-sklearn？

除了数据清理和准备，数据科学家生活中最具挑战性的部分是找到正确的模型。

当 AutoML 还不流行的时候，我和我的团队有过几周这样做的经历。虽然如果你是初学者，我仍然推荐这种方法，但对于实际项目来说，它可能不是最佳的。

如果不充分利用你的脑力，你会陷入困境。反复改变超参数并等待结果并不能更好地利用你的智力技能。

[](/is-deep-learning-right-for-you-b59a2219c457)  

这就是 auto-sklearn 解决的问题。你可以让计算机改变超参数，评估它们的输出，然后**选择最佳模型**。另外，你可以**给培训设定一个时间限制**。除了这些好处，你可以在桌面分布式系统上运行并行的贝叶斯优化。

您可以从 PyPI 存储库中安装 auto-sklearn。下面的代码将安装这个库，并通过检查它的版本来确认。

```
pip install auto-sklearn
python -c "import autosklearn;print(autosklearn.__version__)"# 0.14.6
```

其他安装方法请参考 [auto-sklearn 文档](https://automl.github.io/auto-sklearn/master/installation.html)。

# 为我们的数据集构建(自动)ML 模型。

让我们开始使用自动学习建立一个实际的模型。

为了便于说明，我使用了 Kaggle 上的[葡萄酒质量数据集](https://www.kaggle.com/yasserh/wine-quality-dataset)。

该数据集包括 11 个预测变量和一个响应变量。响应变量葡萄酒质量有六个不同的等级。我们的任务是建立一个机器学习模型，在给定其他变量的情况下预测质量类别。

以下命令将加载数据集，并为模型定型做准备。它删除 id 列并分隔响应变量。

```
import pandas as pddf = pd.read_csv('./WineQT.csv')df.drop('Id', axis=1, inplace=True)y = df.quality.copy()
X = df.drop('quality', axis=1)
```

葡萄酒质量数据集的特征变量— [CC0:公共领域](https://www.kaggle.com/yasserh/wine-quality-dataset)

让我们也将数据集分成几组进行训练和测试。

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

到目前为止，我们还没有做任何与通常使用 scikit-learn 不同的事情。但是这里有一个 auto-sklearn API，它可以训练多个模型，并找到最适合我们的模型。

```
model = AutoSklearnClassifier(
    time_left_for_this_task=5*60,
    per_run_time_limit=30
)
```

在上面的代码中，我们给出了两个参数值。这些不是模型超参数。相反，我们限制了训练过程的时间。

在本例中，我们将 API 配置为每次运行 30 秒(对于一组特定的超参数)，整个过程不超过 5 分钟。

出于说明目的，这些数字是任意的。在决定这些值之前，您应该考虑真实情况下的数据集大小和计算资源。

让我们也根据我们的测试集来评估我们的结果模型。

```
y_hat = model.predict(X_test)
acc = accuracy_score(y_test, y_hat)
print(f"Model Accuracy:{acc}")Model Accuracy:0.6693121693121693
```

这是一个相当不错的葡萄酒质量分类的准确性。

如果我们自己找出最好的模型会是什么样的？

# 手工建立分类器的方法及其问题。

如果你想用 scikit-learn 来做这件事，你需要像下面这样的东西。这里的问题是你不知道`max_features`和`n_estimators.`的最佳值

```
RandomForestClassifier(max_features=15, n_estimators=512)
```

为了找到比其他组更好的一组，您可能需要为这些超参数尝试几个不同的数字。

但这只是在我们确信模型将是随机森林模型的时候。如果没有，你可能还得尝试其他算法。这给我们的流程增加了额外的复杂性。

另一方面，auto-sklearn 在很短的时间间隔内用许多超参数尝试了大约 15 个不同的分类器。

我们还可以看到 auto-sklearn 尝试的不同模型及其在此过程中的排名。下面的代码将打印所有跑步的排行榜。

```
model.leaderboard()
```

Auto-sklearn 排行榜告诉我们在我们的数据集上试用的不同算法及其统计数据。

如果我们需要更多地了解这些不同的模型，我们可以调用 show_models 方法来获取细节。

[](/become-a-data-scientist-or-data-engineer-without-coding-skills-fbf11ac8e278)  

# 用 AutoML 在大数据集上训练 ML 模型

当数据集很大时，自动 ML 技术将消耗更多的资源，并且运行更长的时间。

但是凭借我们的专业知识和偏好，我们可以大大缩短这个时间。以下代码将分类器搜索空间限制为高斯朴素贝叶斯和 k 个最近邻。

我们也可以关闭特征预处理。如果我们的任何特征是分类的，auto-sklearn 会自动进行一次性编码。在大型数据集上，这可能需要相当长的时间。

```
automl = autosklearn.classification.AutoSklearnClassifier(
    include = {
        'classifier': ["gaussian_nb", "k_nearest_neighbors"],
        'feature_preprocessor': ["no_preprocessing"]
    },
    exclude=**None**
)
```

include 和 exclude 配置使我们能够更好地控制 AutoML 搜索空间。然而，对于小数据集，我不会改变它们。这确保了我得到的模型是最好的，超出了我的启发式假设。

# 最后的想法

在我们结束之前，我有一些想法要分享。

auto-sklearn 等自动 ML 库解决了每个数据科学家面临的噩梦般的问题。如果没有这样的工具，熟练的专业人员的宝贵时间将被浪费。

但是不能保证 AutoML 模型永远是最好的。数据科学家可以通过领域知识和专业技能找到更好的模型。

但是 AutoML 会是未来吗？

我想是的。

当汽车上的自动变速器上市时，许多人批评它有一些小缺点。但是今天，自动挡的车比手动挡的车多。

就像汽车类比一样，Auto ML 在这一点上可能并不完美。但很快就能超越专业科学家。

> 数据集鸣谢:[葡萄酒质量数据集](https://www.kaggle.com/yasserh/wine-quality-dataset/metadata)作者: [M Yasser](https://www.kaggle.com/yasserh) 来自 [Kaggle](https://www.kaggle.com/) ，CC0:公共领域许可。

> 感谢阅读，在 [**LinkedIn**](https://www.linkedin.com/in/thuwarakesh/) ， [**Twitter**](https://twitter.com/Thuwarakesh) ， [**Medium**](https://thuwarakesh.medium.com/) 上向我问好。
> 
> 还不是中等会员？请使用此链接 [**成为会员**](https://thuwarakesh.medium.com/membership) 因为，不需要你额外付费，我为你引荐赚取一小笔佣金。