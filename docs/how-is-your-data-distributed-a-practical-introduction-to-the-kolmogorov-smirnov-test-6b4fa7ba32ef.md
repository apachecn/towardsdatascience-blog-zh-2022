# 您的数据是如何分布的？Kolmogorov-Smirnov 检验的实用介绍

> 原文：<https://towardsdatascience.com/how-is-your-data-distributed-a-practical-introduction-to-the-kolmogorov-smirnov-test-6b4fa7ba32ef>

## 初学者 KS 测试介绍

![](img/6fb31a298e09ae2931c43413743ba47d.png)

*照片由*[*papazachariasa*](https://pixabay.com/it/users/papazachariasa-12696704/)*上* [*Pixabay*](https://pixabay.com/it/photos/dado-gioco-d-azzardo-gioco-fortuna-5029548/)

数据科学家通常需要评估数据的适当分布。我们已经看到了正态分布的夏皮罗-维尔克检验，但是非正态分布呢？还有一个测试可以帮助我们，那就是科尔莫戈罗夫-斯米尔诺夫测试。

# 需要检查分布情况

数据科学家通常面临检查他们的数据分布的问题。他们处理样本，需要检查样本是否来自正态分布、对数正态分布，或者两个数据集是否来自同一分布。当你在机器学习中执行[训练测试分割](https://www.yourdatateacher.com/2022/05/02/are-your-training-and-test-sets-comparable/)时，这很常见。

例如，您可能希望查看从总体中抽取的样本在统计上是否与总体本身相似。或者从同一人群中抽取几个样本，想看看彼此是否相似。

所有这些问题都有一个共同的因素:将样本的分布与另一个样本的分布或已知的概率分布进行比较。

接下来是 Kolmogorov-Smirnov 测试。

# 科尔莫戈罗夫-斯米尔诺夫试验

KS 测试有两个版本，每个版本都有自己的零假设:

*   样本是根据给定的概率分布生成的
*   从相同的概率分布中生成了两个样本

前者是单样本 KS 检验，后者是双样本 KS 检验。

两种测试都将样本的累积分布函数与给定的累积分布函数进行比较。从数学上来说，计算这种分布之间的距离，并将其用作可用于计算 p 值的统计值。

这些测试非常强大，尽管它们在计算 p 值时存在一些近似值，并且存在异常值。但是，是一个非常有用的工具，必须放在数据科学家的工具箱中。

让我们看看它们在 Python 中是如何工作的。

# Python 中的一个例子

让我们分别从正态分布和均匀分布创建两个数据集。它们不需要具有相同的尺寸。

```
import numpy as np 

x = np.random.normal(size=100) 
y = np.random.uniform(size=200)
```

现在，我们可以执行 KS 检验来评估第一个样本是来自正态分布还是均匀分布。为了执行这个测试，我们需要从 SciPy 导入我们想要检查的分布的累积分布函数和一个执行测试的适当函数(“ks_1samp”函数)。

```
from scipy.stats import ks_1samp,norm,uniform
```

现在，我们可以运行测试，将“x”数据集的分布与正态累积分布进行比较。

```
ks_1samp(x,norm.cdf) 
# KstestResult(statistic=0.05164007841056789, pvalue=0.9398483559210086)
```

正如所料，p 值相当大，我们不能拒绝零假设，即数据集是由正态分布生成的。

如果我们对均匀分布进行同样的检查，结果会非常不同:

```
ks_1samp(x,uniform.cdf) 
# KstestResult(statistic=0.5340516556530323, pvalue=3.580965283851709e-27)
```

一个非常小的 p 值可以让我们拒绝零假设，即样本是由均匀分布产生的，这实际上是我们所期望的。

双样本测试非常简单。我们必须从 SciPy 导入“ks_2samp”函数，并将两个样本作为参数传递。

```
from scipy.stats import ks_2samp 

ks_2samp(x,y) 
# KstestResult(statistic=0.53, pvalue=9.992007221626409e-16)
```

正如预期的那样，p 值非常低，因为我们从非常不同的分布中人工构建了这些数据集。

因此，Kolmogorov-Smirnov 测试证实了我们的假设。

# 结论

Kolmogorov-Smirnov 测试是数据科学家工具箱中非常强大的工具，每次我们想要查看我们的数据是否来自给定的分布或者两个数据集是否共享相同的分布时，都必须正确使用。然而，p 值的计算可能会受到一些近似值的影响，必须正确处理，就像任何 p 值一样。

***我是 Gianluca Malato，意大利数据科学家、作家和企业家。我通过我的在线课程和书籍帮助人们成为专业的数据科学家和分析师。看看我的在线课程*** [***这里***](https://www.yourdatateacher.com/online-courses/?utm_source=medium&utm_medium=post&utm_campaign=How%20is%20your%20data%20distributed%3F) ***。***

*原载于 2022 年 11 月 14 日 https://www.yourdatateacher.com*<https://www.yourdatateacher.com/2022/11/14/how-is-your-data-distributed-a-practical-introduction-to-the-kolmogorov-smirnov-test/>**。**