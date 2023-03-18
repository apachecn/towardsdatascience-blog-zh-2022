# A/B 测试快速参考清单

> 原文：<https://towardsdatascience.com/a-quick-reference-checklist-for-a-b-testing-40f533cfb523>

## 不要错过你下一个实验的任何关键步骤

![](img/75869f2ce9d36fd683bca2b35d37f2ae.png)

托马斯·博尔曼斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

所以你想做个 A/B 测试？

如果您以前从未运行过测试，或者您已经有一段时间没有运行过实验了，那么您很容易陷入细节中。这就是为什么我作为一名数据科学家在[十一月](http://movember.com)设置实验时使用这个清单。

这个清单并不是 A/B 测试的全面指南(如果这是你想要的，请查看参考资料[这里](/ab-testing-with-python-e5964dd66143)，这里[这里](/a-practical-guide-to-a-b-tests-in-python-66666f5c3b02)和[这里](/how-to-conduct-a-b-testing-3076074a8458))。相反，您可以将清单加入书签，并在需要时引用它。

A/B 测试清单分为 12 个步骤:

1.  阐明你的假设
2.  选择您将跟踪的指标
3.  决定是否要测试一个以上的变体
4.  选择您的取样流程
5.  确定每组的样本量(功效分析)
6.  计算你的实验将持续多长时间
7.  确保你可以复制你的实验
8.  不要过早停止测试
9.  监控你实验的质量
10.  分析 A/B 测试结果
11.  检查你的结果的有效性
12.  考虑统计意义和实际意义之间的差异

我提供了快速复习概念的参考资料链接，并提出了最佳实践。

# 测试前

## 1.阐明你的假设

*   你的零假设和替代假设是什么？
*   您正在运行单面还是双面[测试](https://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/one-tailed-test-or-two/#:~:text=A%20one%2Dtailed%20test%20has,the%20image%20to%20the%20left).&text=Very%20simply%2C%20the%20hypothesis%20test,state%20that%20the%20mean%20%3D%20x.)？

## 2.选择您将跟踪的指标

*   你的[总体评价标准(OEC)](https://www.linkedin.com/pulse/overall-evaluation-criterion-oec-ronny-kohavi/) 是什么？OEC 是主要的决策变量，可以结合多种指标。
*   您的利益相关者批准了您将跟踪的 OEC 和指标了吗？

## 3.决定是否要测试一个以上的变体

*   您将使用多变量 [A/B/C 测试](/a-b-c-tests-how-to-analyze-results-from-multi-group-experiments-ad5d5cee0b05)分析多个变量吗？小心[多重比较问题](https://egap.org/resource/10-things-to-know-about-multiple-comparisons/)。
*   您是否会运行 [A/A 测试](/an-a-b-test-loses-its-luster-if-a-a-tests-fail-2dd11fa6d241)来检查采样问题，包括采样比率不匹配？

## 4.选择您的取样流程

*   是否会违反稳定单位治疗值假设(SUTVA)(如[网络数据](/ab-testing-challenges-in-social-networks-e67611c92916))？
*   您的样本中是否存在需要使用[分层抽样](https://en.wikipedia.org/wiki/Stratified_sampling)进行控制的不同亚人群(例如重度使用者和轻度使用者)？

## 5.确定每组的样本量(功效分析)

*   测试的[显著性水平](/finding-the-right-significance-level-for-an-ab-test-26d907ca91c9) ( *α* )是多少？显著性水平是当零假设为真(I 型错误，假阳性)时拒绝零假设的概率，通常设置为 0.05。
*   测试的[功率](/finding-the-right-significance-level-for-an-ab-test-26d907ca91c9) (1- *β* 是多少？功效是当备选项为真时，正确拒绝零假设的概率，通常设置为 0.8。
*   测试的最小可检测效应 ( *δ* 是多少？最小可检测效果是实验被视为成功的最小改进，通常由利益相关者设定。
*   您的测试和控制组需要多大的样本量？尺寸可以用 [Python](/introduction-to-power-analysis-in-python-e7b748dfa26) 计算。

## 6.计算你的实验将持续多长时间

*   满足上面计算的最低样品要求需要多少天？例如，如果每组需要 n=10，000 名参与者，并且您的网站每天接收 1，000 名独立访问者，则需要 20 天来填充这两个样本。
*   你会在多个季节、假期或[促销周期](https://www.widerfunnel.com/blog/ab-testing-during-the-holidays/)进行你的实验吗？一个经常被引用的经验法则是，让你的实验至少运行[两周](https://cxl.com/blog/12-ab-split-testing-mistakes-i-see-businesses-make-all-the-time/#:~:text=You must run tests for,minimum of two weeks anyway.)。

## 7.确保你可以复制你的实验

*   你为你的代码建立了一个仓库吗(例如 [GitHub](https://resources.github.com/webcasts/GitHub-for-data-scientists-thankyou/#:~:text=Data%20science%20and%20machine%20learning,team%20and%20across%20the%20organization.)) )？
*   你将如何记录你的过程和结果(如维基)？
*   你设置了随机种子吗？

# 在测试期间

## 8.不要过早停止测试

*   你是否因为观察到了积极的效果而提前停止了实验，从而成为了黑客？相反，你应该等到你已经收集了你的全部样本，或者之前已经就[提前停止标准](https://netflixtechblog.com/improving-experimentation-efficiency-at-netflix-with-meta-analysis-and-optimal-stopping-d8ec290ae5be)达成一致。

## 9.监控你实验的质量

*   您是否检查过[样本比例不匹配](/the-essential-guide-to-sample-ratio-mismatch-for-your-a-b-tests-96a4db81d7a4)(即样本大小不匹配的地方)？可以对您的样本进行卡方独立性检验，以检查是否存在不匹配。
*   还有其他可能威胁你实验有效性的质量问题吗？例如[闪烁效果](https://www.kameleoon.com/en/blog/ab-testing-flicker-effect#:~:text=During A%2FB tests%2C the,and overall engagement with consumers.)、[仪表效果](https://web.pdx.edu/~stipakb/download/PA555/ResearchDesign.html)。

# 测试后

## 10.分析 A/B 测试结果

*   什么是合适的统计检验？您对测试的选择将取决于几个因素，包括您的主要指标的概率分布函数(连续、二元等)。)，样本的大小，以及您使用的是单边测试还是双边测试。例如，在比较每个用户的平均收入时，您可以使用双样本 T 检验，或者使用卡方检验来比较转换数量。
*   检验统计量和 p 值的值是什么？关于统计测试的备忘单以及如何在 Python 中应用它们，请参见本文。
*   基于你选择的显著性水平，你是否拒绝或未能拒绝统计假设？
*   标准误差和置信区间是多少？对于非参数统计测试，您将需要使用重采样技术，例如[引导](https://elizavetalebedeva.com/bootstrapping-confidence-intervals-the-basics/)。

## 11.检查你的结果的有效性

*   你检查过实验是否[内部有效](https://www.verywellmind.com/internal-and-external-validity-4584479)(即调查在多大程度上建立了可信的因果关系)？例如，您应该检查测试组和对照组之间的[混杂变量](https://odsc.medium.com/mastering-a-b-testing-from-design-to-analysis-f0ab3974aee0)、[交叉污染](/online-controlled-experiment-8-common-pitfalls-and-solutions-ea4488e5a82e)等。
*   你是否检查过实验是否[在外部有效](https://www.verywellmind.com/internal-and-external-validity-4584479)(即，调查在多大程度上可以推广到更广泛的人群)？例如，您应该检查任何[选择偏差](https://medium.com/airbnb-engineering/selection-bias-in-online-experimentation-c3d67795cceb)，通过您的实验的意外事件，以及您的测试可以被复制的容易程度。

## 12.考虑统计意义和实际意义之间的差异

*   测得的效应是否大于功效分析中设定的最小可检测效应(即实际显著)？这不同于获得低于显著性水平(即统计显著性)的 p 值，后者甚至可以通过低效应大小来实现。
*   你是在结果出来后才假设的吗？

# 结论

本文描述了在 A/B 测试之前、之中和之后要检查的各种项目。该清单是根据我在 Movember 进行实验的经验编写的快速参考指南。

我错过了什么吗？请让我知道，我会更新清单。

喜欢你读的书吗？跟我上 [**中**](https://medium.com/@rtkilian) 。否则， [**推我**](https://twitter.com/rtkilian) 或者在 [**LinkedIn**](https://www.linkedin.com/in/rtkilian/) **上加我。**