# 对 P 值的 5 种常见误解

> 原文：<https://towardsdatascience.com/5-common-misinterpretations-of-the-p-value-aacaa6dcfddd>

## 在解释和报告您的结果时避免这些

![](img/be17120e611b3944f71ca41b9d5a57ca.png)

照片由[奎特拉·登特](https://unsplash.com/@kitera?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

令人惊讶的是，我经常看到对 p 值的度量以及从中可以得出的结论的误解。对 p 值的不正确理解会使您对结果的描述不精确。更糟糕的是，它会导致对你的结果的误导，甚至是错误的结论。在本文中，我将讨论对 p 值最常见的 5 种误解，并尝试解释思考 p 值的正确方法。

**1。p 值为 0.05 意味着零假设为真的概率为 5%。或者，p 值为 0.05 意味着替代假设有 95%的可能性为真。**

这是对假设检验实际检验的内容和 p 值测量的内容的误解。p 值并不衡量零假设为真的概率，事实上，p 值*假设*零假设为真。

p 值表示您的数据与统计模型的假设(包括零假设)的符合程度。在进行假设检验时，我们不是在建立“真理”,我们检验的是我们能为替代假设找到多少支持，根据结果，我们*决定*我们认为合理的相信。

作为对 p 值的实际解释< 0.05, one can think *“假设零假设为真，我们将得到与我们观察到的结果相似的结果，或者更极端(远离零)，平均不超过 5%的时间”*。

**2。p 值小于 0.05 表示您发现了您的研究问题的重要科学结果。**

人们很容易认为低 p 值意味着你有值得分享的重大成果。虽然它可能具有统计学意义，但影响/关联的大小可能非常小，以至于缺乏任何实际或临床意义。

如果样本很大，即使微小的影响也会产生具有统计意义的结果。例如，你对一种新的减肥药的效果进行了一项大型研究。结果显示 p 值较低，但测试组和安慰剂组之间体重减轻的实际差异仅为 100 克。在这种情况下，即使结果具有统计学意义，质疑减肥药的有效性也是合理的。(卖药丸的公司大概会宣称这种效果已经被科学证明了！)

**3。p 值大于 0.05 意味着效果尺寸很小**

这与上面的误解有关，误解也是反方向的。p 值大于 0.05 并不意味着效果大小很小。如果数据中有大量噪声，统计测试可能无法检测出显著的影响，甚至是很大的影响。如果样本很小，这种情况尤其常见。

**4。p 值大于 0.05 意味着没有关系或影响**

在报告 p 值大于 0.05 的结果时，您应该避免使用类似“没有关联”或“没有证据”的短语。除非你的点估计值等于零假设值，否则声称“没有证据”是不正确的。这可能看起来令人困惑，但如果 p 值小于 1，则数据中存在某种关系。您还需要查看点估计值和置信区间，以评估它是否包括重要性的影响大小。

你可能会问，0.05 的临界值是什么，你应该不使用它吗？是的，p 值 0.05 是将 p 值描述为“显著”或“不显著”的常见临界值，您应该使用它。(这个特定的限制及其含义本身就是一个讨论的话题，在这里我只试图解释 p 值实际测量的是什么。)在解释 p 值时，需要记住的重要一点是，当使用“统计显著性”这一术语时，我们讨论的是结果的性质，而不是我们正在研究的总体或效应的性质。

**5。p 值为 0.05 意味着有 5%的假阳性风险(I 类错误意味着当空值为真时拒绝空值)。**

人们混淆了个别测试的 p 值与测试的显著性水平或 alpha 水平。这也被称为测试的 I 型误差或大小。这度量了 p 值被拒绝的频率(p < 0.05) over *重复*测试，假设所有假设和零假设为真。换句话说，它给出了误报的比例。你可以做一个模拟，指定零假设为真。运行模拟 1，000，000 次，如果测试的大小为 0.05，我们预计拒绝空的 50，000 次(测试的 5%)。这与在单独测试中拒绝空值不是一回事。假设零假设成立，如果你拒绝零假设，出错的几率是 100%，而不是 5%。

你可以在本文的[中找到关于如何估计显著性水平的更详细的解释。](/evaluating-the-performance-of-the-t-test-1c2a4895020c)

</evaluating-the-performance-of-the-t-test-1c2a4895020c>  

我意识到很容易误解 p 值的含义。我还可以看到，有人在解释假设检验的更广泛的概念时，可能会以一种易于理解的方式表述 p 值和显著性，但不幸的是，这是不正确的。上面的例子可能看起来是无关紧要的(双关语)细节，但它实际上对科学结果如何呈现以及观众如何理解它们有影响。

我希望这能提醒你在报告结果时注意措辞是多么重要。通常，你可以在新闻中听到一项新的科学研究在某个主题上发现了显著的结果，但实际的效果大小却没有报道。还记得公司卖减肥药的例子吗？他们可能声称他们的减肥药有一种已经被科学证明的效果。嗯，这在技术上是正确的，但我要声明这种说法是严重误导。

我希望这篇文章对你有用。如果您能想到其他对 p 值和统计显著性的常见误解，请在评论中告诉我。

</the-confusion-matrix-explained-part-1-5513c6f659c1>  

如果你喜欢阅读这样的故事，并想支持我成为一名作家，考虑注册成为一名媒体会员。每月 5 美元，你可以无限制地阅读媒体上的故事。如果你注册使用我的链接，我会赚一小笔佣金。

<https://medium.com/@andreagustafsen/membership> 