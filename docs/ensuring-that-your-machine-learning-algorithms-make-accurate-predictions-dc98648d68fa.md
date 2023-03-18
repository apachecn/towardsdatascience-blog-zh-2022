# 确保你的机器学习算法做出准确的预测

> 原文：<https://towardsdatascience.com/ensuring-that-your-machine-learning-algorithms-make-accurate-predictions-dc98648d68fa>

## scikit-learn 的 cross_val_score 如何简化这一关键流程

![](img/91ae917cc0c4887390b0d11b09a704d2.png)

图片来源:pexels 上的 RONDAE productions

我们之前已经[讨论过](/how-to-choose-between-multiple-models-a0c274b4228a)如何确保你的机器学习算法得到良好的训练，并在对给定数据集进行预测时表现良好。这是这个过程中非常重要的一部分:如果一个机器学习算法与数据集不匹配，那么预测很可能是垃圾，这意味着你的算法不会完成你需要它做的事情。

正如上一篇文章中所讨论的，验证模型的过程通常由以下步骤组成:

*   将数据集分成训练和测试数据集，
*   使用训练数据来训练算法，
*   使用算法预测测试数据的输出，以及
*   将预测值与实际测试数据进行比较。

这种方法有两个问题。一是这确保了模型对于特定的训练和测试数据集工作良好。这里的问题是，该模型在不同的训练/测试数据分割下可能不会工作得很好。第二个问题是，这不是解决问题的最简单的方法。

幸运的是，scikit-learn 包含一个交叉验证算法，这使得事情变得非常简单。

# 什么是交叉验证？

交叉验证本质上是一个多次执行上述步骤的过程。当它将数据分成测试和训练数据集时，它使用不同的随机化种子，这会返回不同的数据集以在该过程中使用。这样，每次执行训练和测试时，它都使用不同的数据集，并创建不同的分数。如果您使用交叉验证来多次执行该过程，您会比使用单个训练/测试实现更好地了解算法的执行情况。

为了让它更好，scikit-learn 提供了一个非常简单的算法，我们可以使用。该算法简称为 cross_val_score，它对用户指定的模型执行用户指定次数的验证。在执行验证时，它使用用户请求的评分方法计算分数。完成后，它返回每次运行的分数。

让我们看看如何执行交叉验证。

# 导入数据集

首先，我们需要导入一个数据集来执行验证。这类项目目前最受欢迎的数据集是加州住房数据集，它显示了加州房屋的价格和每个房屋的一些特征。由于这个数据集在 scikit-learn 中可用，我们可以很容易地导入和研究它。

```
from sklearn import datasetsca = datasets.fetch_california_housing()X = ca.datay = ca.targetcol_names = ca.feature_namesdescription = ca.DESCRprint(col_names)print(description)
```

运行该代码将提供一些输出，让您对该文件中包含的数据有所了解。具体来说，它告诉我们:

*   在任何分析中都有 20，640 个数据点可以使用，
*   数据集中有八个预测数字特征和一个目标值，以及
*   这八个特征包括诸如街区中值家庭收入、街区中每所房子的平均房间数量以及街区的纬度/纬度坐标的项目。

由于我们将特征存储在 **X** 变量中，将目标存储在 **y** 变量中，我们现在可以开发一个模型，并使用数据集进行验证。

# 执行交叉验证

在执行交叉验证之前，我们必须首先导入 scikit-learn 的交叉验证工具，并创建一个在交叉验证过程中使用的模型。我们可以这样做，使用随机森林回归(默认超参数。更多内容请见下文！)对于这个例子，使用下面的代码:

```
from sklearn.model_selection import cross_val_scorefrom sklearn.ensemble import RandomForestRegressormodel = RandomForestRegressor()
```

现在我们可以将模型和数据传递给 cross_val_score 来生成分数。cross_val_score 的一般语法(使用我认为最重要的输入。您可以在 [scikit-learn 的文档](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)中看到完整的输入列表，如下所示:

```
scores = cross_val_score(regression algorithm, X data, y data, scoring metric, number of iterations)
```

为了运行交叉验证，我们只需要用实际输入替换我上面使用的描述性术语。具体来说，我们需要更换如下物品:

*   回归算法需要用过程中使用的机器学习模型来代替。在这个例子中，我们在变量**模型**中保存了一个 RandomForestRegressor 实例，这样我们就可以在那里使用**模型**。
*   X 数据被替换为交叉验证过程中使用的 X 数据。由于我们之前已经在 **X** 中保存了加州住房数据集的特征，所以我们可以使用它。
*   除了目标数据之外，y 数据与 X 数据相同。类似地，我们可以使用来自加州住房数据集中的 **y** 变量。
*   评分标准指定您希望 scikit-learn 在评估预测器性能时使用的误差计算方法。您可以在 sklearn.metrics.SCORERS.keys()找到可用评分指标的完整列表。我最喜欢的指标是[均方根误差](https://en.wikipedia.org/wiki/Root-mean-square_deviation) (RMSE)。不幸的是，这在 scikit-learn 中不可用。幸运的是，我们可以使用 scikit-learn 的负均方误差( **neg_mean_squared_error** )度量，并由此计算 RMSE。
*   最后，我们需要说明我们想要执行验证过程的次数。次数越多，结果越好，但计算时间越长。你总是需要根据情况做出决定。这个模型极其精确有多关键？你有多少时间进行计算？对于这个例子，我们将使用 10 次迭代。记住:只要你使用不止一次迭代，交叉验证将会比单次训练/测试更好。

将所有这些放在一起，我们得到以下代码:

```
scores = cross_val_score(model, X, y, scoring = ‘neg_mean_squared_error’, cv = 10)
```

当我们运行这段代码时，我们将收到一个数组，其中包含 10 个验证过程中每一个验证过程的负均方误差值。由于我更喜欢使用 RMSE，我们可以将负均方误差值转换为 RMSE 值。我们还可以通过对结果进行后处理来计算平均 RMSE 和 RMSE 的平方根。所有这些步骤都可以用下面的代码来执行:

```
import numpy as npscores = np.sqrt(-scores)mean = scores.mean()stdev = scores.std()print(scores)print(mean)print(stdev)
```

这最后三行打印了 10 次运行的 RMSE、10 次得分的平均值和 10 次得分的标准偏差。这让我们看到模型平均表现如何，以及运行之间有多少可变性。我们得到的输出是:

```
[0.86971303 0.60083121 0.68912278 0.50477132 0.66335802 0.621953990.53314316 0.78384348 0.82483679 0.51839043]0.66099641905933050.12350211658614779
```

这意味着具有默认参数的 RandomForestRegressor 预测加利福尼亚州特定街区的房屋平均值，其平均 RMSE 为 66，100 美元，RMSE 的标准偏差为 12，350 美元。表现最好的案例显示平均 RMSE 为 51839 美元，表现最差的案例显示平均 RMSE 为 86971 美元。

这提供了一些在比较模型时使用的非常有价值的指标。例如，我们可以使用 ExtraTreesRegressor 或 MultiLayerPerceptronRegressor 尝试相同的过程，看看哪个模型表现最好。或者我们可以改变模型超参数，看看哪个参数集表现最好。最后，我们还可以对数据进行预处理，使其更易于机器学习算法的管理，并看看这是否能提高性能。

就是这样！现在你知道如何执行交叉验证，并评估不同机器学习算法的性能。