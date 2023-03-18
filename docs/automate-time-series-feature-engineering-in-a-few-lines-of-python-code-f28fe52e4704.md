# 用几行 Python 代码实现时间序列特征工程的自动化

> 原文：<https://towardsdatascience.com/automate-time-series-feature-engineering-in-a-few-lines-of-python-code-f28fe52e4704>

## 为您的时间序列用例提取数百个相关特征

![](img/fa306f20d4434decd1f19fcb0069a322.png)

图片来自[皮克斯拜](https://pixabay.com//?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=2980690)的[扬·瓦塞克](https://pixabay.com/users/jeshoots-com-264599/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=2980690)

时间序列数据随着时间的推移重复捕获变量的值，从而产生一系列按时间顺序索引的数据点。在时间序列中，数据具有自然的时间顺序，即特定时间的变量值取决于过去的值。

传统的机器学习算法不是为了捕捉时间序列数据的时间顺序而设计的。数据科学家需要执行功能工程，将数据的重要特征捕获到一些指标中。生成大量的时间序列特征并从中提取相关特征是一项耗时且乏味的任务。

在这里 **tsfresh** 包发挥了作用，它可以为您的时间序列数据生成标准的数百个通用特征。在本文中，我们将深入讨论 tsfresh 包的用法和实现。

# tsfresh:

**tsfresh** 是一个开源包，可以生成数百个相关的时间序列特征，适合训练机器学习模型。从 tsfresh 生成的特征可用于解决分类、预测和异常值检测用例。

## 开始使用:

tsfresh 包提供了对时间序列数据执行特征工程的各种功能，包括:

*   特征生成
*   特征选择
*   与大数据的兼容性

## 安装和使用:

tsfresh 是一个开源 Python 包，可以通过以下方式安装:

```
**pip install -U tsfresh**
# or
**conda install -c conda-forge tsfresh**
```

## 1)特征生成:

tsfresh 包提供了一个自动特性生成 API，可以从 1 个时间序列变量生成 750 多个相关特性。生成的特征包括宽范围的光谱，包括:

*   描述性统计(平均值、最大值、相关性等)
*   基于物理学的非线性和复杂性指标
*   数字信号处理相关特性
*   历史压缩特征

**用法:**

一个数据科学家不需要在特征工程上浪费时间。`**tsfresh.extract_features()**` 函数为 1 个时序变量从多个域生成了 789 个特征。

(作者代码)

> 人们可以浏览 [tsfresh 文档](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html)来获得提取特征的[概述。](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html)

## 2)特征选择:

tsfresh 包还提供了基于假设检验的特征选择实现，可以识别目标变量的相关特征。为了限制不相关特征的数量，tsfresh 部署了 fresh 算法(fresh 代表基于可伸缩假设测试的特征提取)。

`**tsfresh.select_features()**` 功能用户可以实现特征选择。

## 3)与大数据的兼容性:

当我们有大量的多时间序列数据时。tsfresh 还提供 API 来扩展大量数据的特征生成/提取和特征选择实施:

*   **多处理**:默认情况下，tsfresh 包可以在多个内核上并行执行特征生成/提取和特征选择实现。
*   ***tsfresh 自己的分布式框架*** 用于扩展适合单台机器的数据的实施，并将特性计算分布在多台机器上以加快计算速度。
*   ***Apache spark 或 Dask*** 对于不适合单机的数据。

> 这里有一篇 Nils Braun 的精彩文章，解释了使用 Dask 实现 ts fresh([文章第 1 部分](https://nils-braun.github.io/tsfresh-on-cluster-1/)，[文章第 2 部分](https://nils-braun.github.io/tsfresh-on-cluster-2/))。

[](https://nils-braun.github.io/tsfresh-on-cluster-1/) [## (真正)大数据样本上的时间序列特征提取

### 如今时间序列数据无处不在。从股市趋势到脑电图测量，从工业 4.0 生产线…

尼尔斯-布劳恩](https://nils-braun.github.io/tsfresh-on-cluster-1/) 

# 结论:

tsfresh 是一个方便的包，可以在几行 Python 代码中为时间序列特性生成和选择相关特性。它自动从基于时间的数据样本的多个域中提取和选择 750 多个经过实地测试的要素。它减少了数据科学家浪费在特性工程上的大量工作时间。

通常，时间序列数据是相当大的，tsfresh 包也可以解决同样的问题。tsfresh APIs 可以应用于使用多处理、dask 或 spark 的大型数据样本。

# 参考资料:

[1]t 新鲜文档:【https://tsfresh.readthedocs.io/en/latest/ 

[2]尼尔斯·博朗 GitHub 文章:[https://nils-braun.github.io/tsfresh-on-cluster-1/](https://nils-braun.github.io/tsfresh-on-cluster-1/)

> 感谢您的阅读