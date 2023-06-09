# 使用大查询 ML 预测您网站上的交易

> 原文：<https://towardsdatascience.com/predict-transactions-on-your-website-using-big-query-ml-c365f58d29ca>

## 基于谷歌分析数据训练模型

![](img/908ea3e28b4befffff844f3970ae6456.png)

[皮卡伍德](https://unsplash.com/@pickawood?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄的照片

如果你在网上销售产品，预测哪些用户会转化可以帮助你增加收入。例如，它可以让您:

*   创建定制广告受众
*   基于用户行为生成动态产品推荐
*   自定义现有用户的电子邮件工作流

对于这个分析，我们将使用`google_analytics_sample`公共数据集，您可以在这里找到[。](https://support.google.com/analytics/answer/7586738#zippy=%2Cin-this-article)

大查询公共数据集是实践数据科学的惊人资源。然而，如果你有自己的网站，用你的网站数据训练你的模型会更有趣！

## 步骤 1:了解数据集

当我们研究这个模式时，我们可以看到我们有一个列 **visitId** ，这意味着我们每个会话有一行。

> ***会话***
> 
> 用户在您的网站或应用上活跃的时间段。默认情况下，如果用户处于非活动状态达 30 分钟或更长时间，任何未来的活动都将归于新的会话。离开您的站点并在 30 分钟内返回的用户被算作原始会话的一部分。

来源:[谷歌分析文档](https://support.google.com/analytics/answer/6086069?hl=en)

我们还有一个列 **transactions，**，它是一个整数列，对应于会话中的事务数量。我们并不真正关心实际的数字，我们只是想检查用户是否转换。

然后，我们有我们的一系列功能:

*   **totals** 包含一些有趣的指标，如页面浏览量、网站停留时间…
*   **trafficSource** 包含关于源(用户来自哪里)的数据
*   **设备**包含所用设备的信息
*   **地理网络**包含地理信息
*   **hits** 包含事件信息，甚至是交易信息。

这里我们有一个小问题:hit 包含关于事务的数据，这可能是数据泄漏。ML 模型应该只访问在事务之前**发生的事情。**

为了避免数据泄露，我们将去掉 hits 列。当然，对于包含在其他指标中的事务数据，我们可能仍然有一些问题。例如，会话的持续时间包括结帐过程的持续时间，这需要一些时间。页面浏览量也是如此。理想情况下，我们需要从这些指标中减去结帐页面发生的事情，但是我们现在忽略这一点。

## 步骤 2:将数据分为训练集和测试集

大查询允许我们在创建模型时分割数据，但我们也希望避免查看测试数据，因此我们将把数据集分割成一个训练和测试表。

我们的表有 2556 行，因此 80–20%的分割将得到:

*   列车拆分的 2044 行
*   512 行用于测试分割

为此，我们需要生成随机数。SQL 没有本地随机函数，所以我们将使用 Javascript 函数。这是我喜欢大查询的原因之一，你可以对任何列应用 JavaScript 函数！

这给了我们:

是的，有效！现在让我们创建我们的表。首先，让我们创建保存随机数的表:

然后，我们将创建我们的训练和测试集。

## **步骤 3:探索 Data Studio 中的训练集**

让我们在 Data Studio 上创建一个新报告，并选择 Big Query 作为源。

我们需要知道的第一件事是交易的访问份额。

作者的数据工作室报告

这显然是一个不平衡的数据集，我们必须记住这一点！

现在让我们试着看看购买者和非购买者的页面浏览量和网站停留时间的影响。

作者的数据工作室报告

这里显然有一个模式:似乎买家比非买家在网站上花费更多的时间，浏览更多的页面，这是有道理的。

最后，让我们看看来源和转换之间的关系。这次我们将使用频道分组。

> ***频道分组***
> 
> 频道分组是基于规则的流量来源分组。在整个分析报告中，你可以看到你的数据根据默认的渠道分组进行组织，这是一个最常见的流量来源分组，如*付费搜索*和*直接*。这使您可以快速检查每个流量通道的性能。

来源:[谷歌分析文档](https://support.google.com/analytics/answer/6086078?hl=en&ref_topic=6083659)

作者的数据工作室报告

频道分组似乎有影响，但是我们必须小心，因为我们没有很多数据。

## 步骤 4:构建我们的 ML 模型

让我们试试逻辑回归，这是最常见的分类模型之一。如果您不知道逻辑回归，我在参考资料中提供了一些 StatQuest 视频的链接。

在大查询上创建模型的语法很简单:

## 步骤 5:评估我们的模型

现在我们已经创建了模型，是时候看看它在我们的测试集上是否表现良好了！

由于我们的数据是不平衡的，准确性将不是一个评估模型的好指标:仅仅通过每次预测错误，我们将获得超过 98%的准确性。

因此，我们将使用 ROC 曲线来预测我们的模型。

让我们编写一个查询来评估我们的逻辑回归:

这说明用途:

*   我们识别 50%的交易。
*   在我们确定为事务的会话中，实际上只有 22%是事务。

即使这些数字看起来不怎么样，但当你想到一个用户只有 1.7%的转化机会时，也没那么糟糕。这给了我们几乎 95%的 ROC_AUC。

同样，我们可能没有足够的数据来构建一个强大的模型，但是想象一下用成千上万的会话来做这件事！

## 步骤 6:使用模型

在现实生活中，我们可以通过多种方式使用这个模型。

一种方法是实时使用 ML 模型来确定用户将要购买的概率:

*   如果概率很高，我们什么都不做。
*   如果概率为中等，我们会显示一个弹出窗口，显示只在今天有效的折扣代码，试图说服用户购买。
*   如果概率低，我们什么都不做。

这种模式将帮助我们说服犹豫不决的用户。

使用这种模式的另一种方法是建立观众群；我们把概率最高但是还没买或者很长时间没买的用户拿过来，我们用他们来创造一个定制的广告受众，或者我们给他们发邮件。

## 资源

*   [大查询公共数据集](https://cloud.google.com/bigquery/public-data)
*   [谷歌分析文档](https://developers.google.com/analytics)
*   [大查询 ML 文档](https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create#data_split_method)
*   [大查询用户自定义函数文档](https://cloud.google.com/bigquery/docs/reference/standard-sql/user-defined-functions?hl=en#javascript-udf-structure)
*   [Data Studio 文档](https://support.google.com/datastudio/answer/6283323?hl=en&ref_topic=6267740)
*   [统计任务:逻辑回归](https://www.youtube.com/watch?v=yIYKR4sgzI8&list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe)
*   [StatQuest: ROC 和 AUC，解释清楚！](https://www.youtube.com/watch?v=4jRBRDbJemM)