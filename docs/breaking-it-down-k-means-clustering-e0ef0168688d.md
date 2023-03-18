# 分解它:K-均值聚类

> 原文：<https://towardsdatascience.com/breaking-it-down-k-means-clustering-e0ef0168688d>

## 使用 NumPy 和 scikit-learn 探索和可视化 K-means 聚类的基本原理。

```
**Outline:**
[1\. What is K-Means Clustering?](#1791)
[2\. Implementing K-means from Scratch with NumPy](#f947)
   [1\. K-means++ Cluster Initialization](#96e1)
   [2\. K-Means Function Differentiation](#d0b4)
   [3\. Data Labeling and Centroid Updates](#ea21)
   [4\. Fitting it Together](#9b4e)
[3\. K-Means for Video Keyframe Extraction: Bee Pose Estimation](#e5af)
[4\. Implementing K-means with](#aabf) [scikit-learn](#aabf)
[5\. Summary](#8571)
[6\. Resources](#f720)
```

![](img/c51967791e75b1f510f67d91eea0eb37.png)

文章概述

参见我的 GitHub [learning-repo](https://github.com/JacobBumgarner/learning-repo) 获取这篇文章背后的所有代码。

# 1.什么是 K-Means 聚类？

k 均值聚类是一种算法，用于将数据分类到用户定义数量的组中， *k* 。K-means 是一种无监督的机器学习形式，这意味着在运行算法之前，输入数据没有标签。

由于各种原因，使用 k-means 等算法对数据进行聚类是有价值的。首先，聚类用于在构建数据分析管道时识别未标记数据集中的独特组。这些标签对于数据检查、数据解释和训练 AI 模型是有用的。K-means 及其变体在各种上下文中使用，包括:

*   **研究。**例如，对单细胞 RNA 测序结果进行分类[](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008625)
*   **计算机科学。**例如，对电子邮件进行聚类以检测和过滤垃圾邮件[](https://www.semanticscholar.org/paper/Spam-Filtering-using-K-mean-Clustering-with-Local-Sharma-Rastogi/901af90a3bf03f34064f22e3c5e39bbe6a5cf661?p2df)
*   **营销。**例如，针对信用卡广告的客户群细分[](https://www.kaggle.com/code/muhammadshahzadkhan/bank-customer-segmentation-pca-kmeans)

# 2.用 NumPy 从头开始实现 K-Means

为了获得对 k-means 如何工作的基本理解，我们将检查算法的每个步骤。我们将通过可视化的解释和用 NumPy 从头构建一个模型来实现这一点。

k-means 背后的算法和数学函数很漂亮，但相对简单。让我们从一个概述开始:

K-Means 算法简介

总之，k-means 算法有三个步骤:

1.  指定初始聚类中心(质心)位置
2.  基于最近的质心标注数据
3.  将质心移动到新标记的数据点的平均位置。返回步骤 2，直到聚类中心收敛。

让我们继续构建模型。为了使用该算法，我们需要编写以下函数:

K-Means 类函数概述

## 2.1.集群初始化

k-means 算法的第一步是让用户选择数据应该被聚类到的组的数量， *k* 。

在算法的原始实现中，一旦选择了 *k* ，将通过随机选择输入数据点的 *k* 作为质心起始位置来初始化聚类中心(或*质心*)的初始位置。

这种方法被证明是非常低效的，因为开始的质心位置可能最终彼此随机接近。2006 年，亚瑟和瓦西维茨基 [⁴](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf) 开发了一种新的更有效的质心初始化方法。他们在 2007 年发表了他们的方法，称之为 **k-means++** 。

**k-means++** 不是随机选择初始质心，而是基于距离分布有效地选择位置。让我们想象一下它是如何工作的:

K-Means ++质心初始化

既然 k-means++背后的直觉已经暴露出来，让我们为它实现函数:

k-Means**_ init _ centroids _ plus plus**函数

值得注意的是，除了必须手动选择 *k* 之外，还可以使用几种无偏技术来确定最佳数字。 [Khyati Mahendru](https://medium.com/u/78433997a4a?source=post_page-----e0ef0168688d--------------------------------) 在她的文章中解释了其中两种方法，即**肘**和**剪影方法** [。值得一读！](https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb)

## 2.2.数据标注和质心更新

质心初始化之后，算法进入数据标记和质心位置更新的迭代过程。

在每次迭代中，输入数据将首先根据其与质心的接近程度进行标注。在此之后，每个质心的位置将被更新为其聚类中数据的平均位置。

这两个步骤将重复进行，直到标签分配/质心位置不再改变(或*收敛*)。让我们想象一下这个过程:

现在，让我们实现数据标签代码:

k-Means**_ compute _ labels**函数

最后，我们将实现质心位置更新功能:

k-Means**_ 更新 _ 质心 _ 位置**功能

## 2.3.k-均值函数微分

k-means 算法的第三步是更新质心的位置。我们看到这些质心被更新到所有聚类的标记点的平均位置。

将质心更新到平均聚类位置似乎很直观，但是这一步背后的数学原理是什么？基本原理在于 k-means 方程的微分。

让我们通过探索 k-means 函数微分的动画证明来展示这种直觉。该证明表明位置更新是旨在最小化组内*方差*的 k 均值方程的结果。

k-均值函数微分

## 2.4 装配在一起

现在我们已经为 k-means 模型构建了主干函数，让我们将它结合到一个单一的`fit`函数中，该函数将使我们的模型适合输入数据。我们还将在这里定义`__init__`函数:

k-表示**拟合**函数

K-Means **__init__** 函数

现在我们可以把这个模型和我的漫游笔记本[一起使用了。这个笔记本使用合成生成的数据(如上面的视频所示)来演示我们新编写的](https://github.com/JacobBumgarner/learning-repo/blob/main/k_means/k_means_walkthrough.ipynb) [k_means.py](https://github.com/JacobBumgarner/learning-repo/blob/main/k_means/k_means.py) 代码的功能。

# 3.视频关键帧提取的 k-均值算法:蜜蜂姿态估计

太棒了——我们已经完全从零开始构建了 k-means 模型。与其将这些代码扔到一边，不如让我们在一个示例场景中使用它们。

在过去的几年里，神经科学和 DL 研究社区取得了令人印象深刻的进步，实现了高度准确和自动化的动物行为跟踪和分析*。该研究领域中使用的框架实现了各种卷积神经网络架构。这些模型还严重依赖迁移学习来减少研究人员需要生成的训练数据量。这些框架的两个流行例子包括 [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) 和 [SLEAP](https://github.com/talmolab/sleap) 。

> * *边注:这个子域通常被称为* **计算神经行为学**

为了训练模型来自动跟踪动物身上的特定点，研究人员通常必须从他们的行为视频中手动标记 100-150 个独特的帧。考虑到所有的事情，这是一个非常小的数字，可以无限期地自动跟踪**长的行为视频！**

**然而，当标记这些训练帧时，研究人员必须考虑的一个重要方面是，它们应该尽可能地互不相同。如果存在数小时的记录，给一个视频的前 5 秒贴标签是非常没有目的的。这是因为动物在前 5 秒的行为和身体状态可能不会准确地代表整个视频数据集的特征。因此，该模型将不会被训练成有效地识别各种特征。**

**那么这和 k 均值有什么关系呢？无需从视频中手动识别唯一的关键帧，可以实现 k-means 等算法来自动将视频帧聚类到唯一的组中。让我们想象一下这是如何工作的:**

**视频关键帧提取的 k-均值算法**

**为了获得对这个过程的实际理解，您可以使用[我的漫游笔记本](https://github.com/JacobBumgarner/learning-repo/blob/main/k_means/kmeans_frame_selection_walkthrough.ipynb)跟随用于隔离这些帧的代码。**

# **4.用 scikit-learn 实现 K-means**

**在现实世界中，除非必要，否则通常应该避免实现自己构造的算法。相反，我们应该依靠由专家付费和志愿者贡献者维护的精心有效设计的框架。**

**在这个实例中，让我们看看用 scikit-learn 实现 k-means 有多容易。这个类的文档可以在[这里](http://4\. Implementing K-means with scikit-learn)找到。**

**模型初始化和拟合的 scikit-learn 实现与我们的非常相似(不是巧合！)，但是我们必须跳过大约 250 行的代码。此外，scikit-learn 框架为 k-means 实现了[优化的 BLAS 例程](https://github.com/scikit-learn/scikit-learn/blob/60f16feaadaca28f9a1cc68d2f406201860d27e8/sklearn/cluster/_k_means_lloyd.pyx#L186-L190)，使得它们的实现*比我们的*快得多。**

**长话短说——从零开始学习是无价的，但从零开始工作却不是。**

# **5.摘要**

**在这篇文章中，我们探索 k-means 算法背后的数学和直觉的基础。我们使用 NumPy 从头构建了一个 k-means 模型，使用它从一个动物行为视频中提取唯一的关键帧，并学习了如何使用 scikit-learn 实现 k-means。**

**我希望这篇文章对你有价值！如有任何意见、想法或问题，请随时联系我。**

# **6.资源**

```
****References:** [1\. Hicks SC, Liu R, Ni Y, Purdom E, Risso D (2021). mbkmeans: Fast clustering for single cell data using mini-batch *k*-means. PLoS Comput Biol 17(1): e1008625.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008625)
[2\. Sharma A, Rastogi V (2014). Spam Filtering using K mean Clustering with Local Feature Selection Classifier. Int J Comput ApplMB means108: 35-39.](https://www.semanticscholar.org/paper/Spam-Filtering-using-K-mean-Clustering-with-Local-Sharma-Rastogi/901af90a3bf03f34064f22e3c5e39bbe6a5cf661?p2df)
[3\. Muhammad Shahzad, Bank Customer Segmentation (PCA-KMeans)](https://www.kaggle.com/code/muhammadshahzadkhan/bank-customer-segmentation-pca-kmeans)
[4\. Arthur D, Vassilvitskii S (2006). k-means++: The Advantages of Careful Seeding. *Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms*. Society for Industrial and Applied Mathematics Philadelphia, PA, USA. pp. 1027–1035](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)**Educational Resources:**
- [Google Machine Learning: Clustering](https://developers.google.com/machine-learning/clustering)
- [Andrew Ng, CS229 Lecture Notes, K-Means](http://cs229.stanford.edu/notes2020spring/cs229-notes7a.pdf)
- [Chris Piech, K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html)**
```