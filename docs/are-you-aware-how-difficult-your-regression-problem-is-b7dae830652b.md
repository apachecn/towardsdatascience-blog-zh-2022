# 你知道你的回归问题有多难吗？

> 原文：<https://towardsdatascience.com/are-you-aware-how-difficult-your-regression-problem-is-b7dae830652b>

# 你知道你的回归问题有多难吗？

## 一半的解决方案是理解你的问题，但一半的理解是知道它有多复杂

![](img/896e7afcf9a4fc49cb3219cb98ee65d3.png)

Jaromír Kavan 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

很多 ML 的帖子都讨论过分类问题的复杂性，比如他们的[类敏感度](https://www.analyticsvidhya.com/blog/2020/01/class-sensitivity-machine-learning-classification-problems/)和[不平衡标签](https://machinelearningmastery.com/imbalanced-classification-is-hard/)。即使回归问题同样普遍，也没有通用的初步分析来评估给定回归问题的复杂性。幸运的是，ML 研究人员[1，2]已经采用了众所周知的分类复杂性度量，以数字量化基于特征相关性、线性或平滑度度量拟合回归数据集的固有困难。

我将系统地描述每个复杂性度量的直觉、定义和属性，以及我如何用 Python 实现它们。这篇文章旨在通过方便的方法帮助 ML 从业者区分简单的线性问题和更复杂的变化。出于可读性的考虑，我偶尔会使用我在 [GitHub](https://gist.github.com/hoss-bb/bcbae7a3315d10b6727e98baaf1fb843) 上分享的助手模块中的函数。

# 特征相关性度量(C)

## C1/C2:最大/平均特征相关性

**直觉:**一个特征和一个目标之间的相关度的绝对值反映了它们之间关系的强度，它表明了该特征能够在目标上提供多少信息(从 0 到 1 的标度)。

**定义:***C1*和 *C1* 度量分别表示与目标输出相关的特征的最大值和平均值。通过 Spearman 相关性的绝对值来估计特征相关性[ [3](/clearly-explained-pearson-v-s-spearman-correlation-coefficient-ada2f473b8) ]，这是一种非参数度量。

**性质:** *C1* 和 *C2* 在[0，1]的范围内，它们的高值暗示更简单的回归问题。

## C3:个体特征效率

**直觉:**数据中包含不同的对象或样本，因此特征和目标之间的关系在某些情况下可能很强，而在其他情况下可能很弱。在大多数情况下，当特征与目标高度相关时，回归被简化。

**定义:***C3*度量估计为了产生一个子数据集(其中一个特征与目标输出高度相关)而应该移除的样本的最小比例。因此，有必要固定一个相关阈值来确定某个特征是否可以被视为与输出高度相关。对于每个特征，可以迭代地推断出在与输出的相关性超过预定阈值之前必须移除的样本的比例。因此， *C3* 由所有计算比例的最小值组成。

**属性:** *C3* 产生[0，1]内的值，更简单的回归问题会有低值。

## C4:集体特征效率

**直觉:**以前，当应该移除较少的行时，我们查看与目标高度相关的特征。在这里，我们考虑特征的贡献来共同解释目标方差。

**定义:***C4*度量由可以用任何特征解释的剩余例子的比例组成。首先，根据要素与输出的相关性，从高值开始迭代要素。然后，排除使用所选特征的线性模型[ [4](/understanding-linear-regression-94a6ab9595de) ]可以解释其输出的所有示例。事实上，解释示例输出的能力是由小于预定阈值的残差值决定的。因此，当所有特征都已被分析或没有样本留下时，迭代过程结束，并且 *C4* 度量等于剩余样本的比率。

**属性:** *C4* 返回值在[0，1]以内 *C4* 值较低的回归问题更简单。

# 线性测量(L)

## L1/L2:线性模型的平均/绝对误差

**直觉:**如果一个线性函数可以近似拟合数据，那么回归问题可以看做简单的涉及线性分布的数据点。

**定义:** *L1* 或 *L2* 测度分别代表多元线性回归模型的绝对残差[ [5](https://medium.com/analytics-vidhya/mae-mse-rmse-coefficient-of-determination-adjusted-r-squared-which-metric-is-better-cd0326a5697e) ]或平方残差[ [5](https://medium.com/analytics-vidhya/mae-mse-rmse-coefficient-of-determination-adjusted-r-squared-which-metric-is-better-cd0326a5697e) ]的和。

**属性:***L1*和 *L2* 是简单回归问题得分较低的正指标，这些问题几乎可以用线性函数来解决。

## L3:线性模型的非线性

**直觉:**如果原始数据点是线性分布的，那么一个拟合的线性模型可以预测它们在分布内的插值变量。

**定义:**原始数据分布中的随机插值可用于生成合成数据。具有接近输出(低`|y_i-y_j|`)的一对数据点`x_i`和`x_j`，产生新的测试输入:`(x_k,y_k)=(alpha*x_i + (1 — alpha)*x_j, alpha*y_i + (1 — alpha)*y_j)`，其中`alpha`是在[0，1]内随机采样的实数。使用在原始数据上训练的线性模型，预测导出的随机输入的输出。因此， *L3* 等于所有合成输入的均方误差。

**性质:** *L3* 总是正的，只要回归问题简单就保持低。

# 平滑度测量值

## S1:产量分布

**直觉:**如果输入空间中相邻的数据点也具有接近的输出，那么基于输入特征预测目标将会更简单。

**定义:** *S1* 衡量一对相邻的数据点在多大程度上会有接近的输出值。首先，从数据中生成最小生成树(MST) [ [6](https://medium.com/@pp7954296/minimum-spanning-tree-940b80568ecb) 。因此，每个数据点对应于图形的一个顶点。根据数据点之间的欧几里德距离对边进行加权。MST 技术包括贪婪地将更近的顶点连接在一起。因此， *S1* 测量在构建的 MST 中连接的数据点之间的绝对输出差异的平均值。

**属性:** *S1* 始终为正，对于相邻输入要素的输出也很接近的简单回归问题，该值趋于零。

## S2:投入分配

**直觉:**如果输出接近的数据点在输入空间中也接近，则根据输入特征预测目标会更容易。

**定义:**首先，根据输出值对数据点进行排序，然后估计每对相邻行之间的距离。此后， *S2* 返回输出关闭输入之间的平均距离。

**属性:** *S2* 始终为正，对于输入空间中输入要素接近的输入要素也是相邻要素的简单回归问题，该值趋于零。

## S3:最近邻模型的平均误差

**直觉:**在一个简单的高密度回归问题中，一个例子的最近邻可以告诉我们很多。

**定义:** *S3* 使用留一法计算最近邻回归量(kNN [ [7](/intro-to-scikit-learns-k-nearest-neighbors-classifier-and-regressor-4228d8d1cba6) ] with `k=1`)的均方误差。换句话说，当数据点的预测仅仅是基于欧几里德距离的最近邻的输出时，S3 返回平均误差。

**属性:** *S3* 是一个正测度，其值越低，表示回归越简单。

**S4:最近邻模型的非线性**

**直觉:**对于具有高密度的简单回归问题，数据分布内的插值合成数据点将从原始数据中找到有信息的最近邻。

**定义:** *S4* 表示原始最近邻相对于线性导出的合成输入的信息量。可以通过在原始数据的分布内随机插值来生成一组合成的输入。具有接近输出(低`|y_i-y_j|`)的每一对数据点`x_i`和`x_j`，产生新的测试输入`(x_k,y_k)=(alpha*x_i + (1-alpha)*x_j, alpha*y_i + (1 — alpha)*y_j)`，其中`alpha`是在[0，1]内随机采样的实数。基于最近邻回归模型，预测导出的随机输入的输出。因此，S4 指数等于所有合成输入的均方误差。

**属性:** *S4* 返回正值，但对于较简单的回归问题，它们较低。

> 太棒了，你坚持到了最后。请随意使用这些指标来衡量您下一个回归问题的内在复杂性，并在评论中告诉我们您的想法。

# 参考

[1] Maciel 等，测量回归问题的复杂性(2016)，神经网络国际联合会议(温哥华)。

[2] Lorena 等人，《回归问题的数据复杂性元特征》(2018)，《机器学习》(第 107 卷第 1 期，第 209–246 页)。

[3] Juhi Ramzai，明确解释:Pearson V/S Spearman 相关系数(2020)， [TDS](/clearly-explained-pearson-v-s-spearman-correlation-coefficient-ada2f473b8) 。

[4]饶彤彤，了解线性回归(2020)， [TDS](/understanding-linear-regression-94a6ab9595de) 。

[5] Akshita Chugh，MAE，MSE，RMSE，决定系数，调整后的 R 平方——哪种度量更好？(2020)，[中](https://medium.com/analytics-vidhya/mae-mse-rmse-coefficient-of-determination-adjusted-r-squared-which-metric-is-better-cd0326a5697e)。

[6] Payal Patel，最小生成树(2019)，[中](https://medium.com/@pp7954296/minimum-spanning-tree-940b80568ecb)。

[7] Bex T .，Scikit-learn 的 k 最近邻分类器和回归器介绍(2021)， [TDS](/intro-to-scikit-learns-k-nearest-neighbors-classifier-and-regressor-4228d8d1cba6) 。