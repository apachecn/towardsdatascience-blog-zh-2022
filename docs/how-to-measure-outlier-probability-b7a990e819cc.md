# 如何度量异常值概率

> 原文：<https://towardsdatascience.com/how-to-measure-outlier-probability-b7a990e819cc>

## 一种计算点为异常值的概率的方法

![](img/5303ee7fa55c17ddad9e5a36f047afa5.png)

照片由 [***卢卡斯***T5*转自*](https://www.pexels.com/it-it/@goumbik?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) [***像素***](https://www.pexels.com/it-it/foto/persona-che-tiene-la-penna-che-punta-al-grafico-590020/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)

离群值对于数据科学家来说是个大问题。它们是数据集中的“奇怪点”，必须对其进行检查，以验证它们是错误还是真实现象。有几种算法可以帮助我们检测异常值，但我们有时需要测量某个点是异常值的概率，而不是 0-1 值。让我们看看一个可能的方法。

# 什么是离群值？

异常值实际上有几种定义。一般来说，它是一个离它所属的数据集的分布中心太远的点。这些点实际上是“奇怪的”,如果我们想用那个数据集填充我们的模型，就必须提前检查。机器学习模型可能不总是能够直接处理异常值，它们的存在可能会显著影响训练。

这就是为什么我们首先需要将一个点标记为离群点，然后检查它是否必须被删除。

# 如何检测异常值

有几种算法可以帮助我们检测异常值。四分位数范围是单变量数据集最常用的方法(例如，在箱线图的计算中使用)。多元数据集可能受益于高斯包络线或隔离森林。这些算法中的每一个都对数据集的每个点应用一个标签，如果该点不是异常值，则该标签为 0，否则为 1。

但是如果我们需要知道一个点是异常值的概率会发生什么呢？我们可以使用一个有用的工具来达到这个目的。

# 自举

关注我的出版物的人都知道我喜欢 bootstrap。这是数据科学家可以使用的最有用的统计工具之一。它允许我们计算测量的精度，并且异常值的检测实际上是一种测量。因此，让我们看看 bootstrap 的一种可能用途，根据给定的数据集计算某个点是异常值的概率。

这个想法是对我们的数据集进行重采样，对于每个生成的样本，检查每个点是否是异常值。您选择的算法无关紧要，所以您可以随意使用您喜欢的任何过程。然后使用从数据集生成的新样本再次重复该过程。由于数据集已经改变，一些原始点可能不再是异常值，而其他点可能由于同样的原因已经变成异常值。

根据需要多次重复该过程，并为每个原始点计算被标记为异常值的样本分数。这是你的概率。

那么，正式的程序是:

1.  对于数据集的每个点，如果该点不是异常值，则计算等于 0 的标签，否则计算等于 1 的标签
2.  使用替换对整个数据集进行重新采样，并创建一个相同大小的新数据集
3.  重复步骤 1 和 2 数百次
4.  计算生成的样本中标签的平均值

# Python 中的一个例子

现在，让我们看看如何使用 Python 编程语言来计算异常值的概率。对于这个例子，我将使用我在本文的[中谈到的 IQR 方法。](https://www.yourdatateacher.com/2021/11/01/outlier-identification-using-interquartile-range/)

首先，我们导入 NumPy。

```
import numpy as np
```

然后，让我们用一个正态分布生成的 30 个伪随机数来模拟一个数据集。

```
np.random.seed(0) 
x = np.random.normal(size=30)
```

众所周知，正态分布中出现大数(即大于 3)的概率很低。让我们添加一个人为的异常值，例如 5。

```
x = np.append(x,5)
```

正态随机变量具有大于 5 的值的概率是 0.000000287，因此我们可以说，这样的点，如果出现在从正态分布生成的数据集中，一定会引起一些怀疑。那么这是一个异常值。

现在让我们写一个函数，对于一个给定的点和数组，如果该点是一个离群点，则给我们 1，否则给我们 0。

```
def check_outlier(value, array): 
    q1 = np.quantile(array,0.25) 
    q3 = np.quantile(array,0.75) 
    iqr = q3-q1 
    return int(value > q3+1.5*iqr or value < q1-1.5*iqr)
```

让我们将此函数作为地图应用于原始数据集，并检查结果。

```
result = np.array(list(map(lambda t: check_outlier(t,x),x))).reshape(1,-1)
```

![](img/75c9e171c25cd1b69c42167b06ce3e66.png)

正如我们所看到的，所有的点都是“好”的，最后一点被正确地标记为异常值。

现在让我们应用 bootstrap 来计算概率。让我们记住这些步骤:对数据集重新采样，计算每个点现在是否是异常值，重复数百次(本例中为 500 次)，然后计算我们的点被标记为异常值的数据集的比例。

```
n = 500 
result = None 
for i in range(n): 
    new_x = np.random.choice(x,replace=True, size=len(x)) 
    outliers = np.array(list(map(
        lambda t: check_outlier(t,new_x),x))).reshape(1,-1)     if result is None: 
        result = outliers 
    else: 
        result = np.concatenate((result,outliers))
```

概率可以计算为 0-1 标签的平均值。

```
scores = np.apply_along_axis(np.mean,0,result)
```

现在让我们看看每个点及其分数:

```
np.concatenate((x.reshape(-1,1),scores.reshape(-1,1)),1)
```

![](img/57bb385552066f3e699728643f5e5e3c.png)

正如我们所见，-2.55298982 被认为是一个异常值，有 69.6%的概率。这很现实，因为这个数字对于正态分布来说很“奇怪”，尽管它不像 5 那样“太奇怪”，5 被认为是 98.6%概率的异常值。所以，我们可以说这些分数其实是有意义的。

总的想法是按照分数降序排列我们的值。得分最高的记录很可能是异常值。

# 结论

在本文中，我提出了一种算法，对数据集的每条记录应用一个分数，该分数表示它是异常值的概率。这个过程的主要成分是自举技术。该算法适用于您使用的任何异常值识别技术，甚至可用于多元数据集。由于离群值的定义相当主观，我认为计算分数而不是 0-1 标签可能有助于数据科学家从我们的数据中提取更多信息。

*原载于 2022 年 3 月 15 日*[*【https://www.yourdatateacher.com】*](https://www.yourdatateacher.com/2022/03/15/how-to-measure-outlier-probability/)*。*