# ML 团队避免构建公平模型的 3 个(坏)原因

> 原文：<https://towardsdatascience.com/3-bad-reasons-ml-teams-avoid-building-fair-models-d8f85a1edc23>

## 浅谈机器学习模型中的社会公平

ML 有公关问题。

尽管我们对更智能、更细致、更强大的移动营销的能力突飞猛进，产生了巨大的积极社会价值，但消费者对移动营销的担忧并没有消退。

**事实上，它似乎在增长。**

这种趋势对于任何公司、团队或专业人士(像我一样)来说都是令人担忧的，他们的主要功能是为消费者应用构建 ML。

我认为这种趋势正在增长的原因是因为 **ML 团队在构建模型的时候主动避免了“公平”**。我将讲述我见过的团队避免这种情况的 3 个主要原因，以及当我们进一步进入 2020 年代时，为什么这些原因不足以忽视公平。

# 现状

解决 ML 应用中的“公平”是例外而不是规则。几乎所有的 ML 应用程序都是在没有充分考虑“公平”概念的情况下推出的。我认为这有三个原因:

## (1)定义不清:公平对于机器学习意味着什么？

在机器学习之外，判断一件事是否“公平”是一项困难的工作。添加数学、统计和 B]你的团队通常只理解的缺乏盒子模型—

*什么是地面真理？我们测试哪些群体/身份？模型性能的差异有多大值得偏见？我们能确定这不会与其他真正提升性能的特性混淆吗？*

—使得这项任务几乎不可能完成。一个团队如何测试社会偏见？

![](img/b90da5841d32f3bf648a140e2014f879.png)

硅谷的幕后人员发现讨论公平是徒劳的。[cherydeck](https://unsplash.com/@cherrydeck?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

***左右目前 ML 队伍争…***

由于定义和量化最大似然公平是一项棘手的任务，ata 的科学家们通常会放弃试图在他们的模型中明确解决社会偏见的想法。

对于数据科学中难以定义/量化的指标，这是一种常见的反应，通常也是一种好的反应。另一种选择是尝试将一个模糊的术语翻译成一个可量化的度量标准，[这对于 ML 项目来说常常以灾难告终](/ml-product-management-has-a-translation-problem-65e87df655b1)。

虽然公平肯定是一个模糊的概念，但有一些清晰的、几乎普遍接受的定义:

> **ML 公平性:**如果结果独立于给定变量，特别是那些被认为敏感的变量，例如不应与结果相关的个体特征(即[性别](https://en.wikipedia.org/wiki/Gender)、[种族](https://en.wikipedia.org/wiki/Ethnicity)、[性取向](https://en.wikipedia.org/wiki/Sexual_orientation)、[残疾](https://en.wikipedia.org/wiki/Disability)等)，则给定的 ML 模型是公平的。).[ [链接](https://en.wikipedia.org/wiki/Fairness_(machine_learning))

![](img/3c6d985fb1b06b6a71d3c8a4dfd0944a.png)

定义大联盟公平并不像大联盟团队想象的那么难

在模型开发阶段，上述定义可以很容易地转化为几个 ML 测试用例。

例如，一个团队可以通过简单的皮尔逊测试来检查模型的预测与性别的相关程度:

```
import numpy as np
from scipy.stats import pearsonrp = pearsonr(gender_values, labels)# If there is modest correlation between label and gender,# we should attempt to mitigate thisif np.abs(p) >= 0.5:
   print("Potential Gender Bias")
```

或者使用模型的特征重要性分数+与年龄相关的特征，以确保模型不会隐含地发现有偏差的信号:

```
# Using SKLearn Tree model + panda dataset
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressortree_model = GradientBoostingRegressor()
tree_model.fit(X_train, y_train)top_ten_imp_thresh = sorted(tree_model.feature_importances_)[-10]for i in range(len(X.columns)): col = X.columns[i] imp_score = tree_model.feature_importances_[i]
    fair_score = pearsonr(X[col].values, bias_labels) # Check each column for importance + bias
    print(col) if np.abs(fair_score) >= 0.5:
       print("Potential Bias")
    if imp_score >= top_ten_imp_thresh:
       print("Important Feature")
    print()
```

这些当然是不可思议的简单例子，但即使是简单的方法也能给人启发。

您永远无法真正知道您是否已经从您的模型中消除了所有偏见，但是简单地尝试解决它将确保社区发布的模型对我们的最终用户更加公平。

## (2)默许:垃圾进，垃圾出

解决公平性的另一个反对意见是将对公平性缓解的控制从建模团队移交给底层数据。短语:

> 垃圾进，垃圾出

是 DS 社区的主要内容。而在工业界，这是绝对正确的。我们把大部分时间花在清理数据上，而不是创造任何东西。数据科学家可以构建模型来发现数据中的复杂模式…但是如果数据不够干净，无法发现这些模式，我们就无能为力了！

![](img/ce9383b132e385d522651232718108b7.png)

有时候，数据科学感觉就像在垃圾堆里找到金子。埃文·德米科利在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

将这与 ML 公平性联系起来，如果数据本身包含模型拾取的有偏见的模式，我们就什么也做不了。或者我们甚至没有数据来检验我们的模型是否有偏差。

简而言之，有两个与数据相关的异议:

1.  **我们没有足够或正确的数据来验证我们的模型是否公平**
2.  **算法偏差是由于我们的数据和社会固有的偏差**

虽然两个不同的论点，都含蓄地得出结论 ML 公平是“不可能的”给定的情况。

**To (#1)** ，如果你有足够的数据来验证你的模型是准确的，你就有足够的数据来验证你的模型是公平的。如果您的应用程序是面向客户的，您绝对有足够的客户信息来围绕算法偏差构建测试。

即使没有明确的人口统计/身份信息，散列的电子邮件或邮政编码也可以为您的团队提供足够的果汁来解决 ML 公平性问题。你甚至可以使用像 IBM 的 fairness 360 这样的开源工具来完成这个任务。

[](http://aif360.mybluemix.net/)  

**To (#2)** ， ***仅仅因为你的数据/社会中存在偏见，并不意味着你注定要延续这种偏见。***

对算法偏见的测试肯定会揭示内在的数据/社会偏见，但一旦你意识到这些问题，你就可以解决它们，就像你会解决任何其他非公平相关的偏见一样，如不平衡的班级。

## (3)金钱万能:为了商业价值而忽视公平

在我看来，这是算法社会偏见在模型开发中被忽略的最常见原因，即使大多数数据科学家不会明确地说出来:

> **ML 团队需要不惜一切代价交付商业价值。如今，这种成本包括忽略社会偏见缓解，以加快模型部署。**

ML 产品开发仍处于初级阶段。85%的人工智能项目失败。

即使有正确的数据、人员和策略，ML 项目也只能在 20%的时间里实现 ROI。

从公司的角度来看(今天的*)，一个 ML 团队的唯一目标是尽快交付商业价值。为了完成这项任务，我们的工作岌岌可危。*

*![](img/1439ab8b6f6145736d129dad904adafa.png)*

*正如气候变化已经进入主流辩论，ML Fairness 也是如此，由 [Clem Onojeghuo](https://unsplash.com/@clemono?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄*

*就像几代人都忽略了环境后果，因为金钱激励压倒了任何反对意见，有偏见的模型在规模决策时的广泛社会反响被忽略，以确保 ML 团队的项目属于推动 ROI 价值的 15%的计划。*

*而且要快。*

*“*快速移动和打破东西”*后脸书工程开发的咒语很难转化为 ML 任务，但特别是使解决像算法社会偏见这样的外部性成为一个额外的不必要的步骤。*

*正如一位 DS 的知己告诉我的:*

> *我们已经有了构建和发布模型的紧迫期限。【他老板】不问公平，我就不考了。很糟糕，但事实就是如此。*

*但就像工业公司进入公共话语的环境后果一样，科技公司的社会后果也会如此。它将开始影响底线。*

# *结论*

*公众对公平 ML 模型的期望[已经改变](https://www.fintechnews.org/report-shows-consumers-dont-trust-artificial-intelligence/)。围绕 ML 公平发展的现状将需要向紧张的消费者保证，他们可以信任公司如何管理他们的个人数据，并且他们的模型的预测是同样准确和公平的。*

*“人工智能冬天”一词通常用来描述对人工智能研究的兴趣急剧下降的时期。我假设*

> ***除非 ML 团队承认并解决消费者对人工智能日益增长的不满，否则我们可能会进入一个奇怪的“消费者人工智能冬天”:***
> 
> ***大多数消费者非常不信任和不喜欢使用 ML 的产品，因此消费者对 ML 的应用不再流行。***

*避免这种命运将涉及添加护栏和功能，以确保我们的模型满足消费级技术所需的期望。*

*正如所讨论的，ML 社区能够并且应该解决这些模型的缺点。如果做不到这一点，将会(1)直接导致野生环境中更糟糕的结果，以及(2)从根本上削弱公众对人工智能产品的信任和渴望。*