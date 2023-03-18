# 聚类算法的详细情况

> 原文：<https://towardsdatascience.com/the-ins-and-outs-of-clustering-algorithms-acd4e33290c5>

解决一个数据科学问题通常从一遍又一遍地问同样的简单问题开始，偶尔会有变化:这里有关系吗？这些数据点属于同一个吗？那边的那些人呢？前者与后者有什么关系？

事情可能(也确实)变得非常复杂——尤其是当我们在处理大型数据集时试图检测微妙的模式和关系时。这就是聚类算法派上用场的地方，它有能力将一堆杂乱的数据分成不同的、有意义的组，然后我们可以在分析中利用这些组。

为了帮助您踏上集群学习之旅，我们选择了关于该主题的最佳近期文章，它们涵盖了从基本概念到更专业的用例的许多领域。尽情享受吧！

*   [**k 均值聚类的基本原理**](/breaking-it-down-k-means-clustering-e0ef0168688d) 。无论你是机器学习的新手，还是需要可靠指导的老手， [Jacob Bumgarner](https://medium.com/u/e1f3762eb90c?source=post_page-----acd4e33290c5--------------------------------) 对最广泛使用的基于质心的聚类方法的介绍是一个很好的起点。
*   [**基于密度聚类的易理解指南**](/dbscan-clustering-break-it-down-for-me-859650a723af) 。一旦您掌握了 k-means 聚类，并准备扩展一下， [Shreya Rao](https://medium.com/u/99b63de2f2c3?source=post_page-----acd4e33290c5--------------------------------) 将在这里为您提供清晰的 DBSCAN(基于密度的带噪声应用程序空间聚类)指南，这是一种“需要最少的领域知识，可以发现任意形状的聚类，并且对大型数据库有效”的算法。
*   [**如何把一个算法用好**](/image-color-segmentation-by-k-means-clustering-algorithm-5792e563f26e) 。感到有灵感将您的集群知识应用到一个具体的问题上？ [Lihi Gur Arie，PhD](https://medium.com/u/418175cbf131?source=post_page-----acd4e33290c5--------------------------------) 的新教程是基于 k-means 聚类；它耐心地引导读者根据颜色识别和量化图像中的对象。

![](img/34ed2cc13cc3d2e15a7df1cdc1eb1527.png)

安妮·斯普拉特在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

*   [**聚类方法满足可再生能源**](/evaluation-of-clustering-methods-for-discovering-wind-turbine-neighbors-27849dea14a2) 。在现实世界中，对于给定的用例，哪种集群方法最有效并不总是显而易见的。 [Abiodun Olaoye](https://medium.com/u/ec14ee65bc42?source=post_page-----acd4e33290c5--------------------------------) 研究了多种算法——k-means、凝聚聚类(AGC)、高斯混合模型(GMM)和相似性传播(AP)——以确定哪种算法在发现风力涡轮机邻居方面最有效。
*   [**如何选择正确的基于密度的算法**](/density-based-clustering-dbscan-vs-hdbscan-39e02af990c7) 。决定哪种模型最适合您的数据集，有时取决于细微的差别。[Thomas A . dor fer](https://medium.com/u/7c54f9b62b90?source=post_page-----acd4e33290c5--------------------------------)通过比较 DBSCAN 与其较新的同类产品 HDBSCAN 的性能，展示了一个这样的例子，并向我们展示了如何看待不同集群选项的优缺点。

准备好阅读更多关于其他主题的优秀文章了吗？这里有一个我们很高兴分享的人造集群:

*   艾德丽安·克莱恩的统计训练营系列的每一个新的部分都值得庆祝，最新的部分——关于第一类和第二类错误——也不例外。
*   我们很兴奋地发布了约瑟夫·罗卡和巴普蒂斯特·罗卡的新深度剖析:他们耐心地解释了扩散概率模型的内部运作。
*   “如果任其自生自灭，人们会找到不同寻常的方法来挫败你的数据收集意图。”Cassie Kozyrkov 强调了掌握数据设计艺术对 T2 的重要性。
*   我们如何[解决深度学习架构和关系数据库之间的差距](/towards-deep-learning-for-relational-databases-de9adce5bb00)？Gustavír 分享了一个发人深省的建议，这个建议可能会产生深远的影响。
*   不要屈服于沉没成本谬论——Sarah kras Nik 的最新文章提供了一个有用的路线图，来反对和告别未充分使用和未使用的仪表板。

随着这一年接近尾声，我们一如既往地感谢读者的支持。如果你想产生最大的影响，考虑[成为中级会员](https://bit.ly/tds-membership)。

直到下一个变量，

TDS 编辑