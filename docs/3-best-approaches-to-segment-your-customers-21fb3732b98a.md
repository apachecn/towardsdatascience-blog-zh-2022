# 细分客户的 3 种最佳方法

> 原文：<https://towardsdatascience.com/3-best-approaches-to-segment-your-customers-21fb3732b98a>

## 学习层次聚类、k-均值聚类和 RFM 分割的用法

![](img/dfc4501e4250bd84e4dbf33d89fe5cf5.png)

Avinash Kumar 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

# 客户细分

今天，客户是一切的中心。但是，你不可能让所有人都满意。现实就是这样。你越早了解这一点，它将更好地为你和你的企业服务。这就是为什么业务分析师要做的第一件事是根据客户的需求、愿望和共同特征将客户(现有的和潜在的)分成不同的组。了解客户的偏好可以让你设计量身定制的策略来赢得他们，并提供最好的产品和服务。这对于软件即服务(SaaS)企业来说更为重要，因为客户保持率(RR)是关键 KPI 之一。

有几种方法来细分客户。分层聚类、最近频率和货币(RFM)分割和 K-均值聚类是流行的方法。

> 最近，我写了一个由 3 部分组成的系列文章，详细描述了如何进行客户细分。查看它们:
> 
> [客户细分介绍](/customer-segmentation-with-python-implementing-stp-framework-part-1-5c2d93066f82)
> 
> [使用 k 均值聚类进行客户细分](/customer-segmentation-with-python-implementing-stp-framework-part-2-689b81a7e86d)
> 
> [用 PCA 改进 k-means 客户细分模型](/customer-segmentation-with-python-implementing-stp-framework-part-3-e81a79181d07)

# 分层分段

> 根据维基百科，层次聚类是一种聚类分析方法，旨在建立聚类的层次结构。

<https://en.wikipedia.org/wiki/Hierarchical_clustering>  

在层次聚类中，成对样本基于相似性被分组在一起，然后它们被合并以形成下一层次。最后形成一个树状图(一个树形结构)。要形成的簇的数量由专家分析师基于该树状图来确定。

下面的代码基于标准化的客户数据集生成一个树图。完整的代码可以在 [Deepnote 笔记本](https://deepnote.com/workspace/asish-biswas-a599-b6cca607-3c12-4ae6-b54d-32861e7e9438/project/Analytic-School-8e6c85bd-e8c9-4387-ba40-0b94fb791066/%2Fnotebooks%2Fnotebooks-customer_segmentation.ipynb)中找到。

# k-均值算法

K-means 聚类是一种无监督聚类算法。它试图根据最接近的平均值对观察值进行分组。下面是实现 K-means 算法所需的步骤。

1.  选择聚类的数量(k)
2.  为每个簇分配初始质心(在 scikit-learn 的 KMeans 类对象的 init 参数中传递`kmeans++`)。
3.  根据距离测量值，将每个观测值分配给质心最近的聚类。
4.  计算修改后的质心。
5.  只要质心保持变化，重复步骤`3`和`4`

<https://en.wikipedia.org/wiki/K-means_clustering>  

一旦我们知道了最佳的集群数量，我们就可以通过`n_clusters`参数对我们的客户进行细分。

一旦用数据训练了模型，我们就可以从 kmeans 对象的`label_`参数中获得分配的分类标签。

# RFM 分割

**R** 频率、 **F** 频率和货币细分(RFM)是一个管理型的客户细分过程，适应性强，易于理解。关键实体是:

*   **新近度**:客户互动的新近度。
*   **频率**:客户购买的频率。
*   **货币**:顾客消费的总金额。

RFM 细分的关键要素是将静态管理权重分配给客户的 3 个因素，并计算每个客户的最终等级，从而确定客户群。

客户细分是第一步。下一步是制定强有力的战略。在实施你的策略时，保持专注并定期检查以确保你在正确的轨道上。

万事如意。

*感谢阅读！如果你喜欢这篇文章一定要* ***鼓掌(最多 50！)*** *和让我们* ***连接上****[***LinkedIn***](https://www.linkedin.com/in/asish-biswas/)*和* ***在 Medium*** *上关注我，随时更新我的新文章。**

**请通过此* [*推荐链接*](https://analyticsoul.medium.com/membership) *加入 Medium，免费支持我。**

*<https://analyticsoul.medium.com/membership> *