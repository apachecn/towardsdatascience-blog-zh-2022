# 用 Python 预测需求的价格弹性(实现 STP 框架-第 5/5 部分)

> 原文：<https://towardsdatascience.com/predicting-price-elasticity-of-demand-with-python-implementing-stp-framework-part-5-5-8383ecc4ae68>

## 实施逻辑回归预测需求价格弹性

![](img/2c3f09b84e038645058641c4b934e989.png)

[活动发起人](https://unsplash.com/@campaign_creators?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

在本系列中，我们将学习 STP 框架(细分、定位和营销)，这是最流行的营销方法之一。在前三篇文章([第一部分](/customer-segmentation-with-python-implementing-stp-framework-part-1-5c2d93066f82)、[第二部分](/customer-segmentation-with-python-implementing-stp-framework-part-2-689b81a7e86d)和[第三部分](/customer-segmentation-with-python-implementing-stp-framework-part-3-e81a79181d07))中，我们已经学会了使用细分来了解我们的客户，并根据他们的人口统计、心理特征和行为特征将我们的客户群分为四个部分。这在本系列的第一个主要部分中已经讨论过了。

我们的四类客户是

1.  标准
2.  以职业为中心
3.  机会更少
4.  富裕的

在[上一篇文章(第 4 部分)](/predicting-price-elasticity-of-demand-with-python-implementing-stp-framework-part-4-646b025b8b34)中，我们开始了实现 STP 框架的第二个主要部分。在那里，我们训练了一个逻辑回归模型，发现*平均价格*和*购买概率*之间存在反比关系。这意味着随着平均价格的下降，客户购买的可能性增加。

现在，在此基础上，我们试图确定需求的实际价格弹性。我们将努力确定在不损害需求的情况下我们可以提高的价格，并检查价格上涨在什么时候开始影响市场。

**如果想刷新本系列之前的文章:**

1.  [STP 框架介绍及利用分层聚类实现客户细分。](/customer-segmentation-with-python-implementing-stp-framework-part-1-5c2d93066f82)
2.  [使用 k 均值聚类进行客户细分(基础模型)](/customer-segmentation-with-python-implementing-stp-framework-part-2-689b81a7e86d)
3.  [改进的 k-means + PCA 分割模型](/customer-segmentation-with-python-implementing-stp-framework-part-3-e81a79181d07)
4.  [用逻辑回归预测需求的价格弹性](/predicting-price-elasticity-of-demand-with-python-implementing-stp-framework-part-4-646b025b8b34)

和往常一样，[笔记本](https://deepnote.com/workspace/asish-biswas-a599-b6cca607-3c12-4ae6-b54d-32861e7e9438/project/Analytic-School-8e6c85bd-e8c9-4387-ba40-0b94fb791066/%2Fnotebooks%2FPositioning.ipynb)和[数据集](https://deepnote.com/workspace/asish-biswas-a599-b6cca607-3c12-4ae6-b54d-32861e7e9438/project/Analytic-School-8e6c85bd-e8c9-4387-ba40-0b94fb791066/%2Fdata%2Fpurchase_data.csv)在 [Deepnote 工作区](https://deepnote.com/workspace/asish-biswas-a599-b6cca607-3c12-4ae6-b54d-32861e7e9438/project/Analytic-School-8e6c85bd-e8c9-4387-ba40-0b94fb791066/%2Fnotebooks%2FPositioning.ipynb)中可用。

## 需求价格弹性

*商品的需求价格弹性(PED)是衡量需求数量对价格的敏感程度。当价格上涨时，几乎所有商品的需求量都会下降，但对某些商品来说下降的幅度比其他商品更大。*

如果我们把价格提高百分之一，价格弹性预测的是需求量的百分比变化，假设其他一切都不变。在现实生活中，需求量可能会因为其他变量而发生变化，但在计算 PED 时，我们必须假设其他变量保持不变。

> 需求的价格弹性=需求量的百分比变化/价格的百分比变化

1.  如果 PED 大于 1(PED > 1)，它被称为“弹性”，这意味着价格的变化会导致需求的显著变化。
2.  如果 PED 等于 1 (PED = 1)，那么这意味着价格的任何变化都会引起需求的同等变化。
3.  如果 PED 小于 1(PED < 1)，它被称为“非弹性的”。这意味着价格变化不会对需求产生太大影响。
4.  如果 PED 等于 0 (PED = 0)，这被称为“完全无弹性”，这意味着价格的任何变化都不会导致需求的变化。

经济学家应用这一点来理解当一种产品的价格变化时，供给和需求是如何变化的。

如果我们在 3 类或 4 类，我们可以在不伤害需求的情况下提高价格。但是如果我们属于第 1 类或第 2 类，我们就有价格弹性，我们必须分析价格变化对需求的影响。

让我们简化我们场景的等式。

我们可以用代码实现的简化版本如下:

> PED =贝塔*价格* (1 -购买概率)

这里，β是逻辑回归模型的系数。

## 履行

简单回顾一下我们正在使用的代码。

*price_range* 是一个数组，它保存的糖果价格略高于(两边)任何品牌糖果的最低和最高价格。我们准备了这个数组来计算每个价格点的 PED，并确定价格弹性边界的确切点。

*purchase_proba* 是在相应的平均价格( *price_range* )上发生购买行为的概率。

我们在这张图表中看到了完整的弹性曲线及其与价格区间的关系。价格弹性随着价格的上升而下降。这意味着产品价格越高，对需求的影响就越小。由于价格和需求之间存在反比关系，随着价格的上涨，顾客购买的可能性越小。

从定义中我们知道，如果弹性的绝对值小于 1，则为非弹性，如果大于 1，则为弹性。

查看图表(也是数据框架)，我们看到平均价格为 1.25 时，价格弹性为-1.0。这意味着，在这个价格点之前，如果我们将价格提高 1 %,需求将减少不到 1%。因此，购买概率将是无弹性的。另一方面，在该价格点(1.25)之后，如果我们将价格提高 1%，需求将减少 1%以上。这使得情况缺乏弹性。

这使得 1.25 成为转变点，因为对于非弹性值，一般建议提高价格，因为它不会导致购买概率的显著下降。另一方面，如果我们有弹性，我们应该降低价格来增加需求。

## 按细分市场比较价格弹性

这是我们的基线，整个数据集平均价格的价格弹性。现在让我们比较一下每个细分市场的价格弹性，因为这将让我们深入了解每个客户细分市场的价格弹性。有些部分可能比其他部分更灵活。

我们将采用逻辑回归模型的新实例，然后用特定的数据段对其进行训练，并像以前一样计算弹性。让我们看一下图表。

## 观察

我们在图表中(也在数据框架中)看到，富裕的*群体*和以职业为中心的*群体*的平均无弹性价格比机会较少的*群体*和*标准群体*高出约 14%。只要 PED 没有弹性，我们就可以提高早期产品的价格。

我们可以做出的另一个推论是，线的陡度在图形的右侧变化(超过价格点 1.5)。它表明了每个细分市场的不同弹性水平，并告诉需求将如何随着价格的上涨而变化。我们必须小心调整价格，以保持健康的需求水平，而 PED 是有弹性的。似乎机会较少的群体比其他群体对价格更敏感。

至此，我结束了我们用 Python 实现了 STP 框架的这个博客系列。

如果你有任何问题，请写在评论里，我会尽我所能尽快回答。

整个[笔记本](https://deepnote.com/workspace/asish-biswas-a599-b6cca607-3c12-4ae6-b54d-32861e7e9438/project/Analytic-School-8e6c85bd-e8c9-4387-ba40-0b94fb791066/%2Fnotebooks%2FPositioning.ipynb)和[数据集](https://deepnote.com/workspace/asish-biswas-a599-b6cca607-3c12-4ae6-b54d-32861e7e9438/project/Analytic-School-8e6c85bd-e8c9-4387-ba40-0b94fb791066/%2Fdata%2Fpurchase_data.csv)都可以在 [Deepnote 工作区](https://deepnote.com/workspace/asish-biswas-a599-b6cca607-3c12-4ae6-b54d-32861e7e9438/project/Analytic-School-8e6c85bd-e8c9-4387-ba40-0b94fb791066/%2Fnotebooks%2FPositioning.ipynb)中获得。

*感谢阅读！如果你喜欢这篇文章一定要* ***鼓掌(最多 50！)*** *让我们* ***连接上****[***LinkedIn***](https://www.linkedin.com/in/asish-biswas/)*和* ***在 Medium*** *上关注我，随时更新我的新文章。**

**通过此* [*推荐链接*](https://analyticsoul.medium.com/membership) *加入 Medium，免费支持我。**

*[](https://analyticsoul.medium.com/membership) [## 通过我的推荐链接加入媒体

### 阅读阿西什·比斯瓦斯(以及媒体上成千上万的其他作家)的每一个故事。您的会员费直接支持…

analyticsoul.medium.com](https://analyticsoul.medium.com/membership)*