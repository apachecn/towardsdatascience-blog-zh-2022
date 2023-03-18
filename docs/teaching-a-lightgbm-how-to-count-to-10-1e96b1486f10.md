# 教灯 GBM 如何数到 10

> 原文：<https://towardsdatascience.com/teaching-a-lightgbm-how-to-count-to-10-1e96b1486f10>

## 会有多难呢？

![](img/1fc73ce39e3f40f29dd1e9acd43bd941.png)

克里斯·贾维斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

**LightGBM 是一个强大的机器学习算法。但是能数到 10 吗？**

LightGBM 可以说是处理表格数据的最佳算法。这是一个聪明的方法，也是许多机器学习竞赛获奖解决方案的支柱。LightGBM 还提供时间序列数据。M 次预测竞赛(M5)最新一轮的[获胜者使用了 LightGBM 作为学习算法。](https://github.com/Mcompetitions/M5-methods/)

在摆弄它的时候，我决定用 LightGBM 来解决下面的问题:**预测序列 1，2，…，9 的下一个值。也就是数字 10** 。在这个任务中几乎没有不确定性，它只需要将 1 和前一个值相加。这么复杂的算法能有多难？

这是我第一次尝试的代码，我在下面解释。

我们首先创建一个可训练数据集，我们称之为 *sequence_df* (第 1–10 行)*。*下面是它的样子:

表 1:准备用于训练的数据集

以第一行为例，4 为目标值，[1，2，3]为解释变量的向量。因此，这遵循一个自回归模型——使用过去的观察值作为当前的解释变量。

在构建数据集之后，我们用它来装配 LightGBM(第 11–18 行)。通常，您应该优化模型的超参数。为了简单起见，这里我只使用默认配置。

最后，我们得到对输入向量[7，8，9]的预测。

我们得到 6.5 分。

![](img/96a15fe16678e9413f3a809339cd10a2.png)

由 [Sarah Kilian](https://unsplash.com/@rojekilian?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

为什么我们得到 6.5？ **LightGBM 使用基于树的方法构建。这些不能在观察(训练)范围之外外推。因此，LightGBM 也不能做到这一点。**6.5 值是训练目标变量的平均值。

**我们可以通过引入差分预处理步骤来解决这个问题，该步骤去除了序列的*趋势*。** [正如我在之前的一篇文章](/12-things-you-should-know-about-time-series-975a185f4eb2)中所描述的，求差是对时间序列的连续观测值取差的过程。其目的是去除趋势，稳定数据均值。下面是修改后的片段:

因此，**第 8 行中的差分步骤将序列[1，2，…，9]转换为 1 的常量序列**。除了预测阶段之外，其余过程与之前相同。

首先，我们得到对差异数据的预测( *prediction_raw* )。然后，我们通过添加最后一个已知的观察值(即 9)来恢复差分操作。

瞧，我们得到了 10。

这个故事是关于 LightGBM 行为的轶事，但也是一个警示故事。LightGBM 是一个强大的算法。但是时间序列的预处理在预测系统的开发中是至关重要的。因此，请注意这一趋势！

感谢阅读！

**附言**我们本可以应用线性模型(或助推器)。那么，差分预处理将不是必需的。不过，去除趋势也有助于时间序列建模时的线性模型。