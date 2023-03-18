# 当你的实验返回一个无统计学意义的结果时，该怎么办

> 原文：<https://towardsdatascience.com/what-to-do-when-your-experiment-returns-a-non-statistically-significant-result-81ecaf56fb32>

## 从无统计学意义的结果中可以学到很多东西——当你报告结果时需要非常小心

![](img/274078af3b8850ed145a4e2352a186b0.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Ameer Basheer](https://unsplash.com/@24ameer?utm_source=medium&utm_medium=referral) 拍摄的照片

# **TL；博士**

*   当你得到一个没有统计学意义的结果时，有两种主要的解释:(1)影响(如果有的话)太小而不能被捕捉到(2)你的研究动力不足
*   深入研究结果，了解为什么它在统计上不显著，可以获得一些有助于改进实验设计的知识
*   报告时——重要的是报告所有的结果以及它们的置信水平(并让你的听众理解置信水平的含义),以确保根据你的结果做出合理的商业决策

# 形势

想象一下:

*   你正在为一家游戏公司工作，该公司发布了一款免费游戏(有可能删除付费用户的广告)。
*   你支持的收购团队的目标是增加付费用户的数量。他们有针对这一指标的季度 OKR。
*   他们开展了一项新的电子邮件活动，邀请免费用户转向付费用户，他们想了解这项活动的影响

您通常会与他们讨论为这种活动进行实验是否有意义，但团队担心发送这些电子邮件的潜在负面影响(例如，用户选择退出电子邮件通信)。

*   您与团队一起定义一个[总体实验标准](https://www.analytics-toolkit.com/glossary/overall-evaluation-criterion/)(即一个考虑了不同因素的综合指标，它将作为决定实验成功与否的标准)
*   您可以定义您希望在实验后运行的后分析(例如，确保您的研究有足够的能力对数据进行切片)
*   您与团队讨论，以了解哪个时间段最适合该实验，以及他们如何定义该实验的“成功”
*   根据之前定义的 OEC 和你认为你能影响它的程度，你可以计算出一个好的实验样本量
*   你实际上是在进行实验，A 组是邮件活动的目标，B 组没有收到邮件。

你选择的时间框架是“几天后”:结果出来了——但不幸的是，它们在统计学上并不显著。

# 深入研究结果

您的结果没有统计学意义有两个潜在原因:

*   你的活动没有影响或者影响很小
*   你的学习动力不足

重要的是要很好地理解是什么真正导致你的实验没有统计学意义。这将让你确保在实验后做出的决定是正确的。

## **你的活动影响极小**

想象两个相同的淡水湖，你在其中一个湖里倒入一杯盐。你对你治疗组的湖水含盐量有影响吗？从技术上来说，你确实往湖里加盐了——但是这种影响很难测量，如果不是不可能的话。

在我上面提到的情况下，你给你的免费用户发了一封邮件。很可能不是治疗组的每个人都收到了邮件并打开了它——你的目的是治疗整个治疗组，但你只治疗了一个小组。也许你的电子邮件产生了影响，但是由于你的治疗组中实际上没有得到治疗的用户数量，记录这种影响可能会变得棘手。

在这种特定情况下，根据你的研究问题和你试图证明的东西，你可能想使用一种方法，如[编者平均因果效应](https://en.wikipedia.org/wiki/Local_average_treatment_effect) (CACE)，这将允许你估计你的电子邮件对实际收到/打开它的群体的影响。

但有时，即使有这些方法，效果大小仍然太小，以至于没有统计学意义。在这种情况下——问题是:如果变更影响很小或没有影响，您是否希望继续进行变更？

## **你的学习动力不足**

简而言之，这意味着你没有足够的观察结果来自信地说你观察到的变化(如果有的话)是由于你的治疗或者仅仅是由于随机的机会。

这在很大程度上取决于你想要什么样的信心水平，以及你正在寻找什么样的指标来跟踪“变化”。这就是为什么在运行您的实验之前，总是建议进行功效分析，以了解在您想要的显著性水平上，需要多大的样本量才能获得具有统计显著性的结果，以及您为您的实验选择的成功度量。

不得不报告“由于样本量太小而没有统计上的显著影响”并不是一个理想的情况。因为这给解释和讨论留下了很大的空间，而这并不是你设计实验的真正目的。然而，一线希望是，如果你真的处于那种情况下，由于这个实验，你现在对你的治疗的影响程度(及其方向)有了更好的理解，你可以用这个更新的数据重新进行功效分析，这有望为潜在的下一次实验提供更合适的样本量。

(无耻的自我推销:在后续文章中，我将深入研究计算功率时的不同参数，以及如何使用它们来确保您能够回答您想要回答的问题)。

# 举报什么？

报告有统计学意义的结果肯定是有偏见的。但是随着时间的推移，我们似乎失去了这个词的全部含义。

> 你一直在用那个词，我认为它的意思不是你想的那样

“统计显著”意味着我们看到的结果不太可能是偶然的。但是“这种可能性”是(或者至少应该是)由你/你的利益相关者设定的，基于你对出错的风险以及结果的偶然性有多满意。

通常，选择约 95%的置信度作为阈值(著名的 p <0.05). But this is simply a convention — it doesn’t have to be what you use for your project. In the case of the email campaign presented above, if the campaign had an impact of +20% adoption, but you are only sure about this result with ~90% confidence — most likely you’ll still proceed with launching this campaign (while technically not having a statistically significant result at 95%).

Generally speaking — to avoid this binary approach, [文献](https://onlinelibrary.wiley.com/doi/pdf/10.1111/jan.14283) (1)建议报告观察到的差异(效应大小)以及 p 值(可能会突出显示“统计信号”的结果)。这样你就可以对效果的大小有一个全面的了解，并对效果有信心。

# 结论

在商业世界中，我们喜欢 stat sig 与非 stat sig 的二元方法，因为这感觉像是“科学支持”的绿灯，让我们继续做出决定。

但这可能会产生破坏性影响，从激励人们不报告非 stat sig 影响到扼杀没有显示 stat sig 积极影响的项目(尽管它们确实产生了积极影响)。

最终，很好地理解实验实际显示的东西和一些常识可以帮助你最大限度地利用这些结果。

希望你喜欢阅读这篇文章！**你有什么建议想要分享吗？在评论区让大家知道！**

**如果你想更多地了解我，这里有几篇你可能会喜欢的文章**:

[](/7-tips-to-avoid-public-embarrassment-as-a-data-analyst-caec8f701e42) [## 让您的数据分析更加稳健的 7 个技巧

### 增强对结果的信心，建立更强大的个人品牌

towardsdatascience.com](/7-tips-to-avoid-public-embarrassment-as-a-data-analyst-caec8f701e42) [](/how-to-build-a-successful-dashboard-359c8cb0f610) [## 如何构建成功的仪表板

### 一份清单，来自某个制造了几个不成功产品的人

towardsdatascience.com](/how-to-build-a-successful-dashboard-359c8cb0f610) [](https://medium.com/@jolecoco/how-to-choose-which-data-projects-to-work-on-c6b8310ac04e) [## 如何…选择要处理的数据项目

### 如果你有合理利用时间的方法，你可以优化你创造的价值。

medium.com](https://medium.com/@jolecoco/how-to-choose-which-data-projects-to-work-on-c6b8310ac04e) 

[(1)重要的真诚:报告非显著性统计结果](https://onlinelibrary.wiley.com/doi/pdf/10.1111/jan.14283)，Denis C. Visentin，Michelle Cleary，Glenn E. Hunt