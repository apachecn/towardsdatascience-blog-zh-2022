# 实用主义者线性回归假设指南

> 原文：<https://towardsdatascience.com/the-pragmatists-guide-to-assumptions-in-linear-regression-fbb65482724b>

## 忘记假设的清单:你需要什么取决于你的用例

![](img/1960fc320f7508a6729e11358c349b82.png)

图片由作者用 AI 模型 DALL E mini 生成。提示:“数学公式旁边的指南针”。学分:craiyon.com

许多关于线性回归及其假设的文章看起来像长长的洗衣单。而这些列表似乎与数据科学家在实践中使用线性回归的方式联系不大。此外，他们给出了错误的想法，除非你的数据满足所有的假设，否则你不应该使用线性模型。

这里我想改变通常的方法:我想从用例开始，而不是列出假设。你要线性回归做什么？你需要它来做预测吗？或者你需要它来理解因果关系？如果有，要不要量化它的不确定性？这些问题的答案告诉你你真正需要的假设是什么。

这种方法更有用，因为它反映了数据科学家的实际日常工作(我们从要解决的问题开始，然后考虑适当的统计工具箱，而不是相反！).

## TL；速度三角形定位法(dead reckoning)

对于数据科学中的许多用例(侧重于预测)，您只需要几个假设(随机抽样，没有完美的多重共线性)。如果你想要一个因果解释，那么你应该检查更复杂的假设(严格的外生性)，特别是如果你想要通过一个标准的统计包量化因果效应的不确定性(同质性和正态性假设)。

# 用例 1:闭嘴，预测

假设您是一名数据科学家，为一家房地产中介工作。你一直在收集数据，现在你有很多关于市场上待售房屋的信息(大小、卧室数量、位置等。).您的老板现在希望您使用该数据集来构建一个潜在客户生成模型:即，一个可以找到最高预期售价的房屋的模型，以便您的公司可以联系业主并宣传其服务。

![](img/6b0ed72f9e5991f6afc34ec0c25900dd.png)

图片由作者用 AI 模型 DALL E mini 生成。提示:“一个房产中介机器人”。学分:craiyon.com

在这种情况下，您只需要找到一个具有良好预测能力的模型(特别是，善于预测房屋销售价格的模型)。如果我们使用线性回归模型，我们需要什么样的假设呢？因为我们只关心预测能力，所以假设列表减少到一个:

> *假设# 1——随机抽样:数据代表总体(又名无选择偏差)*

对，就是这样！如果你只想最大化线性模型的预测能力，你只需要担心你的数据的代表性和无偏性。毕竟，您可能会沿着其他模型(包括参数模型和非参数模型)测试线性回归，并且您通常没有一系列假设来检查其他模型，所以您为什么要担心线性模型呢？

## 为什么我们需要这个假设？

假设#1 只不过是标准 ML 最佳实践的一种重述，著名的“确保你的训练集来自与测试集相同的分布”。因此，这种假设并不是线性回归所独有的。换句话说，没有必要记住假设 1，因为它可能已经是你的工具箱的一部分了。

# 用例 2:具有可解释性的预测

然而在实践中，我们很少把预测作为我们唯一的目标。我们经常想看一眼模型的系数并获得一些见解——如果不是关于世界，至少是为了检查模型的健全性。例如，我们可能希望利用销售团队中同事的专业知识来改进我们对房屋数据进行特征工程的方式。为此，我们首先需要了解哪些特征在模型预测中的权重最大(即测量其系数的绝对值)，以及它们与目标变量是正相关还是负相关(即系数的符号)。简而言之，我们需要了解我们的模型是如何做出预测的。

我们只需要一个假设来确保我们可以明智地检查我们的线性模型:

> 假设# 2——没有完美的多重共线性:独立变量之间没有精确的线性关系。

## 为什么我们需要这个假设？

事实是，如果您的模型表现出完美的多重共线性，那么最小化残差的最小平方和的问题就有多个(实际上是无限个)解决方案。虽然这对于进行预测来说不一定是个问题，但它使得查看系数变得毫无意义:它们的值与特征和目标之间的真实相关性没有太大关系(与它们的估计量相关联的方差无限大)。

# 用例 3:带有因果解释的预测

现在让我们想象一个非常不同的场景。比方说，你对你的模型的准确性印象深刻，以至于你决定离开房地产经纪人，自己创业。你心目中的商业模式相当冒险:你想实际建造房屋，然后出售并获利。为了帮助你做到这一点，你希望你的模型告诉你每所房子应该有什么特征(大小，房间数量，位置等。)为了最大化售价。

就线性回归所需的假设而言，这意味着什么？这个场景与前两个场景非常不同。在没有意识到这一点的情况下，我们的新用例迫使我们离开相关性的世界，并把我们带到因果关系的狂野世界。这为我们的线性模型带来了新的假设。

让我们想象一下，在拟合我们的线性模型之前，我们只是检查上面的假设 1 和 2。让我们想象一下，我们的拟合模型告诉我们，在所有条件相同的情况下，一栋带游泳池的房子拥有迄今为止最高的预测市场价格。我们搓搓手，开始建一个带泳池的房子。当房子最终完工时，我们把它放在市场上……我们努力想卖掉它。在与几个潜在买家进行了几个月累人的谈判后，我们终于设法以远低于模型预测的市场价格卖掉了房子。尽管如此，该模型在用于挖掘潜在客户时仍有很强的预测能力(你以前的雇主，房地产经纪人，仍在利用同样的模型赚钱)。这怎么可能呢？

![](img/7ee840029b2e3c8685ac6cdaf8acc532.png)

图片由作者用 AI 模型 DALL E mini 生成。提示:“多重共线性和外生性”。学分:craiyon.com

## 为什么事情会偏离正轨

事情是这样的，在这里你试图做一些与引导一代非常不同的事情:你通过建造一座新房子来影响这个世界，你要求模型预测你的行动会产生什么结果。对于一个模型来说，这是一个更具挑战性的问题，因为答案更依赖于你忽略的潜在预测因素。

让我们想象一下，例如，有游泳池的房子一般都建在山顶上(这样游泳者可以从高处欣赏风景)。让我们想象一下，实际上是山顶的位置(而不是游泳池的存在)吸引了潜在的买家，推高了这些房子的价格。如果你在你的模型中包括变量 *pool* ，但是你忘记了包括一个 *hilltop_location* 二进制变量，那么模型将为 *pool* 分配一个大的(正)系数:这是因为 *pool* 与 *hilltop_location* 正相关，这反过来对*价格*有很大的影响。变量 *pool* 实际上是从 *hilltop_location* 中“借用”预测能力，这种预测能力在模型中是不存在的，只能通过 *pool* 来体现:这种影响称为混杂偏倚，而 *hilltop_location* 就是混杂变量。

如果我们使用该模型作为预测工具来进行潜在客户生成，那么 *pool* 只是借用预测能力这一事实并不是问题:是的，该模型将推荐带泳池的房子，但这些房子往往建在山顶上，因此它们可能对买家有吸引力。然而，如果我们将*池*的大系数视为我们应该建造一个池来提高我们新房子的价格的建议，而不管它是否在山顶上，这确实成为一个问题:在这种情况下，我们真的想确定是*池*真的*导致了*更高的市场价格，因为这是我们在现实世界中想要摆弄的唯一变量。

有没有办法从数据中发现我们的模型存在混杂偏倚？是的，这就是假设 3 派上用场的地方:

> 假设# 3——严格外生性——独立变量与误差项不相关。

## 为什么我们需要这个假设？

假设# 3 和混杂问题之间的关系可能不会立即显现出来，但是直观的想法是混杂变量的作用使得自变量和残差“一起移动” [2](#0cc1) 。

如果我们检查假设# 3，我们会意识到变量*池*实际上与误差项相关。这将告诉我们，一些其他变量混淆了*池*和*价格*之间的关系，为我们节省了很多钱…

# 用例 4:带有置信区间的因果预测

让我们回到我们假设的故事:在包含了 *hilltop_location* 之后，我们最终得到了一个适合因果解释的模型。在几年的时间里，我们成功地用这个模型建造了几栋房子，然后以预测的价格出售。我们赚了很多钱，我们很快乐。但我们也有科学抱负，这促使我们在同行评审的统计期刊上发表我们的模型。

该杂志的审稿人要求我们纳入与我们的模型系数相关的置信区间(CIs)。这些很重要，因为它们让读者了解我们的系数估计值有多不确定(或“嘈杂”)。我们以前没有计算 CIs，但是我们知道有一个简单的方法:如果我们的模型的残差遵循高斯分布，我们可以计算 CIs。这是需要检查线性回归的最后一个假设的时候:

> 假设# 4-同质性:给定任何特征值，残差具有相同的方差
> 
> 假设# 5——正态假设:残差服从(相同的)高斯分布，均值为零，标准差为σ

## 为什么我们需要这些假设？

这两个假设都与模型错误的“行为”有关，因此它们与 CIs 的联系并不令人惊讶。如果我们满足假设 4。和 5。然后我们就可以开始了:我们可以通过使用任何假设高斯分布的标准统计包来计算 CIs。

# 结论

正如我们所见，线性回归中并非所有假设都是平等的:您需要哪一个取决于您的用例[[3](#97c8)]——特别是，它取决于您想要的可解释性水平以及您是否想要对您的模型进行因果解释。对于数据科学中的许多用例(侧重于预测)，您只需要几个假设(随机抽样，没有完美的多重共线性)。如果你想要一个因果解释，那么你应该检查更复杂的假设(严格的外生性)，特别是如果你想要通过一个标准的统计包量化因果效应的不确定性(同质性和正态性假设)。

话虽如此，如果你真的想做因果分析，我强烈建议你检查一下在因果推理背景下开发的所有工具和技术(因果图、结构方程模型、工具变量……)[[4](#eb89)]。与线性回归假设# 3–5 相比，这些工具提供了一种更加完整和可靠的方法来估计因果关系，而线性回归假设# 3–5 只是真实交易的近似值。

最后，看起来线性回归假设的标准清单要么太重(当我们只需要做出预测时)，要么太轻(当我们需要实际的因果解释时)。也许没有洗衣单我们会做得更好…

[1]这并不意味着其他假设不能提高预测能力，只是意味着它们是不需要的。

[2]关于严格外生性和混杂变量问题之间关系的更深入的解释，请看吉姆·弗罗斯特的[这篇博文](https://statisticsbyjim.com/regression/confounding-variables-bias/)。

[3]Jeffrey m . Wooldridge 所著的《计量经济学导论——现代方法》是对线性回归及其假设(以及更多内容)的一个很好的介绍。

[4]关于因果推理，我推荐朱迪亚·珀尔的《为什么之书》。如果你正在寻找更短的/博客风格的内容，我真的很喜欢肯·阿克夸的《因果流》