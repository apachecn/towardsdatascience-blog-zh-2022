# 处理不完美实验设计的三条规则

> 原文：<https://towardsdatascience.com/three-rules-to-deal-with-imperfect-experimental-designs-830c92f60590>

## 一个现实生活中的咨询例子

![](img/3c914453581f577ff135be7668e7d8f4.png)

Irene B .的照片:[https://www . pexels . com/photo/sand-dirty-breaked-ground-12496974/](https://www.pexels.com/photo/sand-dirty-broken-ground-12496974/)

在现实世界中，在学习了数据科学或/和统计学之后，这可能是最重要的:**如何处理不完美的现实生活数据？**作为活体实验室研究人员的统计顾问和统计教师，我经常需要处理收集的数据。事实是，在统计学家设计的理想的纸上实验和人们已经做的或能够做的之间存在差距。

本文将首先介绍主要观点(三个规则)，然后用我最近实践中的一个真实例子来说明这个观点。

## **我处理不完美实验设计的三条规则**

从 A/B 测试的企业到科学研究:

1.**接受不完美:**定义最佳研究设计，并尽可能接近它。通常，我们无法实现我们在纸上定义的理想实验。然而，将您实际做的与您最初的理想实验设计进行比较可以帮助您识别和评估潜在的限制。

2.现实一点:客观地评估你所拥有的可能是最困难的任务之一，但它也能为你节省最多的时间。试图得到一个不存在的东西往往代价很高。无论如何，你以后可能会放弃，你可能会得出错误的结论/建议，或者其他人可能会发现(你甚至可能会对其他工作失去信誉)。对我来说，我和我的合作者已经放弃了五年的研究。然后，我们不得不从不同的角度从头开始。我仍然相信这是最好的决定，我们这样做节省了很多时间。另一方面，正如我们将在下面的例子中看到的，客户认为他们有完美的体验，并且确实有一个非常有前途的想法。然而，一个重大缺陷使得这项研究的一个关键特征过时了，从而降低了研究计划的等级。尽管在我们的缩放交流中，客户几次试图恢复最初的想法，但我建议采取另一种(不那么“性感”)的方法，并客观地评估其局限性。

3.**保守一点:**统计模型依赖于假设，如果机理是基于理论模型，在得出结果之前还要做更多的假设。因此，即使结果在统计上是显著的，现实和模型估计之间也可能有差距。评估局限性通常不会取消作者的资格，反而会增强他或她的可信度。我举两个例子。首先，我看到学术报告中作者从“声称”因果关系开始。通常在这个早期的断言之后，听众会在接下来的演讲中用(有时非常不现实的)故事和例子来攻击演讲者，说明为什么在这种情况下因果关系可能无法实现。在这种情况下，一种更保守的方法是解释你如何解决阻止因果关系被测量的问题，同时让读者/观众自己来判断这是否足够。其次，想一想在任何问题上都过于自信的人。如果你已经在统计领域工作了一段时间，你就会知道一切都是复杂的，很少有简单的无条件的答案。

# 最近的一个咨询问题说明:

一个客户联系我，问了我一个简单的统计问题，是关于他们最近做的一个实验。为了保护隐私，我不会透露姓名或问题的确切内容。

**研究问题:**衡量绩效的方法 B 会产生与方法 A 相似的结果吗？(方法 A 是目前最先进的方法，但非常昂贵，而方法 B 要便宜得多。)

**要检验的统计假设:**方法 B 与方法 A 相关吗？(有几种方法可以回答这个问题，我将集中讨论一种简约的方法)。

**设置:**客户用每种方法进行了八次绩效评估，历时八天，共有 27 名参与者。这个想法是进行成对比较(不是比较个体，而是比较每个个体内部的度量)。

**关键特征:**该设计的主要目的之一是观察新的性能测试是否足够灵敏，以捕捉个体差异。这些是身体测试。因此，如果你将一名运动员和一名普通跑步者进行比较，这两种测试可能很容易发现差异，但测试的主要目的是测量恢复时间(以及其他)。所以更小的差异(个体内部而不是个体之间)应该被捕捉。

**客户提问:**客户问我，在这种情况下，他们应该执行什么测试。我建议用重复观测值进行两两相关检验(c.f. Bakdash 和 Marusich (2017))。于是那个人递给我数据，在做测试之前，我用我通常的方法看了看(在[这篇 TDS 文章](/a-recipe-to-empirically-answer-any-question-quickly-22e48c867dd5)中有完整的描述)。

我通常的方法遵循 5 个步骤(变量选择、样本选择、单变量分析、双变量分析和结论)。我将跳过第 4 部分——双变量分析——直奔主题。不过，如果你也对第一步感兴趣，你可以在这里公开访问我的 Deepnote 文件[包括所有的步骤(笔记本末尾的第一步)。](https://deepnote.com/workspace/statswithquentin-9de199f7-1b70-481e-a6e2-df6c97c779f4/project/04sport-cd10f099-4552-4a5b-92a6-2d3669280431/%2F04_sport.ipynb)

## 4.双变量分析:

我们对两个变量之间的关系感兴趣。最初的想法是使用重复的配对相关性测试。相关系数是线性关系的量度。因此，我想先看看数据，看看线性假设是否可信。

**观察:**有正相关。总体关系似乎可以用线性趋势很好地近似。个体之间的关系是明确的(使用方法 A 得分较高的人倾向于使用方法 B 得分较高)，而在个体内部则不明确(每组相同颜色的点没有明确的趋势)。

**免责声明:**这是我将在几分钟内完成的第一个快速分析。当然，我可以探索不同的函数形式，对极值进行适当的测试，等等。

由于线性假设是合理的，并且两个变量是连续的，我确实会遵循我最初的建议:匹配测量的相关性。

**观察:**正如预期的那样，相关性与统计意义上的显著性相差甚远(p 值=0.52)，甚至为负(相关性=-0.048)。

**问题的早期迹象:**看着这个散点图，我认为与受试者之间的差异相比，个体内部的差异确实很低。这让我想到了一个潜在的问题。

**找出问题:**所以在与客户的下一次通话中，我问我们是否真的期望个人之间的表现有所不同(即使使用相同的方法)。基本上，如果你衡量自己在一周内每天都习惯做的一项任务的能力，你会期待一个显著的自发变化吗？我们的结论是事实并非如此。如果发生了重大的事情(例如，作为训练的结果)，就会观察到一个人的变化。事实上，在关于方法 A 的文献中，当实际存在治疗(例如，训练、比赛等)时，这种性能测试捕捉(内部)差异。).所以，设计漏掉了一个很重要的东西:一种治疗。这将允许对同一个人进行治疗前和治疗后观察，从而验证两种表现测试是否能够捕捉到差异。

## 5.结论:

**应用上述三个规则:**

1.**理想与现实:**在理想的设置中，个人将接受允许前后测量的治疗。考虑到这一点，让我们看看下面的 2 条规则。

2.现实一点:在当前的体制下，利用个人之间的差异是行不通的。如果你去掉个体间的差异，剩下的就是噪音。所以……那真令人失望。然而，仍然有可能利用个体间的差异。客户已经可以看到两种测试是否捕捉到受试者之间的差异。客户提出的问题还没有在文献中探讨过，因此，尽管它有一些缺陷，还是值得发表这些结果。如果这是关于这个主题的第二十篇论文，就更难“推销”了。此外，客户有一组重要的控制变量(性别、年龄、身高、体重)。因此，我建议采用包含控制变量的线性回归(以及由于重复值而聚集在个体水平上的误差)。线性回归是一种非常直观和简单的方法，可以根据控制变量捕捉两种方法之间的关联。

**3。最后，我的建议是清楚地解释研究的局限性以及如何改进研究。**