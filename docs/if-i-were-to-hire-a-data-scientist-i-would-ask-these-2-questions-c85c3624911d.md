# 如果我要雇用一名数据科学家，我会问这两个问题

> 原文：<https://towardsdatascience.com/if-i-were-to-hire-a-data-scientist-i-would-ask-these-2-questions-c85c3624911d>

## 它们会引发一场揭示许多重要事情的对话

![](img/8a158afaffa021f971e655365101b422.png)

文森特·范·扎林格在 [Unsplash](https://unsplash.com/s/photos/two?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

我目前没有能力聘请数据科学家，但我希望有一天会。话虽如此，我已经好几次站在面试台的另一边了。

在数据科学家的访谈中，我回答了各种主题的问题，从 SQL 到 Python，从机器学习到贝叶斯定理。有些真的很有挑战性，但有些对我来说似乎没什么用。

现在我是一名数据科学家，我有时会想，如果我要雇用一名数据科学家，我会提出什么要求。

我这样做的主要目标不是找到最好的问题来提问，也不是在候选人中找到最好的数据科学家。相反，我的目标是成为一名更好的数据科学家。这样的思考过程帮助我提高技能，激励我学习新东西。由于数据科学仍在发展，持续学习至关重要。

先说第一个问题。

> **问题 1** :你被指派创建一个模型来解决一个监督学习问题，要么是回归，要么是分类。你会选择哪种算法，为什么？

你首先想到的大概是，这取决于很多东西。你完全正确。因此，我们并不试图找到一个单一的算法作为答案。

这个问题的目的是开始一个关于机器学习算法的讨论。但是，在讨论过程中的某个时候，我想让候选人解释以下内容:

*   预测准确性和模型可解释性之间存在权衡。

我们来详细阐述一下这个。一般来说，随着算法灵活性的增加，它往往会给出更准确的结果。例如，梯度推进决策树比线性回归灵活得多，并且在准确性和性能方面优于线性回归。

然而，我们用可解释性的代价换来了 GBDT 极其出色的表现。这意味着我们对决策是如何做出的了解有限。该算法可以计算特征重要性值，但我们对哪些特征起关键作用有模糊的理解。

另一方面，线性回归具有非常有限的灵活性，但提供了高度的可解释性。通过线性回归模型，我们可以全面了解每个特征对预测的影响。

所以算法的选择取决于我们想要达到的目的。如果我们不关心可解释性，只想获得好的结果，我们可以使用灵活的算法。股票价格预测就是一个例子。我们通常只对获得高精度感兴趣。

当我们在一项任务中工作时，重点是“为什么”要做出预测，那么我们应该选择可解释的模型。

机器学习不仅用于推荐系统等低风险环境，还用于癌症预测、药物测试等关键任务。在这些情况下，我们肯定想知道为什么一个决定是错误的。

这个问题本质上和可解释的机器学习或者可解释的 AI 有关。如果你想了解更多关于可解释机器学习的知识，这里有一本由 Cristoph Molnar 写的很棒的书。

<https://christophm.github.io/interpretable-ml-book/>  

> 问题 2:什么是机器学习中的偏差和方差？

这也是一个开放式问题，目标是看候选人是否知道偏差、方差、它们对机器学习模型的意义以及它们之间的权衡。

方差是模型对训练数据的敏感程度的度量。由于训练集中的微小变化，具有高方差的模型的预测可能会发生显著变化。然而，这不是所期望的。我们希望我们的预测在不同的训练集之间不会有太大的差异。因此，我们尽量避免高方差的模型。

偏差是指用一个非常简单的模型来近似一个复杂的问题。例如，使用线性回归模型，非线性关系可能导致高偏差。我们无法通过使用高偏差的模型来获得对目标变量的良好估计。因此，我们也尽量避免具有高偏差的模型。

但是，我们如何才能实现低偏差和低方差的模型呢？

预测模型的误差基本上是预测值和实际值之间的差异。这些误差由可约误差和不可约误差两个主要部分组成。

我们可以通过关注可减少的误差部分来改进模型，误差部分可以表示为预测的方差和方差。方差和方差都是非负值，因此，在最佳情况下，我们的目标是低偏差和低方差的模型。然而，这通常是一项非常具有挑战性的任务，并且在它们之间有一个权衡。

一般来说，随着方法灵活性的增加，方差趋于增加，偏差趋于减少。例如，基于 GBDT 的算法，如 XGBoost 和 LightGBM，可能具有非常低的偏差，但具有高的方差。

随着灵活性的增加，偏差下降的速度往往会快于方差增加到某一点的速度。在此之后，如果我们继续增加灵活性，我们不会在偏差方面取得很大成就，但方差会显著增加。

创建一个健壮而准确的模型的关键是找到最佳点。

还会有特征和目标变量之间的关系是线性的情况。在这些情况下，线性模型将没有偏见，因此它们优于更先进和复杂的模型。

这些是我肯定会问数据科学家候选人的问题。他们引导对话，揭示候选人对机器学习和统计学中许多重要概念的知识和理解。

我强烈建议思考你自己的问题，因为这既是一个很好的思维锻炼，也是一个学习的机会。

*你可以成为* [*媒介会员*](https://sonery.medium.com/membership) *解锁我的全部写作权限，外加其余媒介。如果你已经是了，别忘了订阅*<https://sonery.medium.com/subscribe>**如果你想在我发表新文章时收到电子邮件。**

*<https://sonery.medium.com/membership>  

感谢您的阅读。如果您有任何反馈，请告诉我。*