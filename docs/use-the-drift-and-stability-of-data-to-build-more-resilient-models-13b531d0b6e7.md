# 利用数据的漂移和稳定性建立更具弹性的模型

> 原文：<https://towardsdatascience.com/use-the-drift-and-stability-of-data-to-build-more-resilient-models-13b531d0b6e7>

# 利用数据的漂移和稳定性建立更具弹性的模型

## 预测模型的好坏取决于为其提供动力的数据

![](img/6e1f2e718db8ef86c3277628a28b2e29.png)

图片由 iStock 授权的 bor chee 拍摄

在构建预测模型时，通过精度、召回率和曲线下面积(AUC)等指标衡量的模型准确性，传统上是模型设计和可操作化的主要驱动因素。虽然这导致了在训练和测试时的高保真模型构造，但是生产中的性能通常会降低，产生比预期更差的结果。

随着机器学习(ML)在组织内的成熟， ***弹性*** 往往会超越原始的预测准确性，成为操作化模型的定义标准。越来越多的 ML 实践者倾向于操作性能良好、可预测的生产模型，而不是那些在测试中表现出高性能但在部署时不太兑现承诺的模型。这种对弹性模型的偏好可以从来自[unite . ai](https://www.unite.ai/resilience-accuracy-why-model-resilience-should-be-the-true-metric-for-operationalizing-models/)[【1】](#_ftn1)[rapid miner](https://rapidminer.com/blog/model-accuracy-you-need-resilient-models/)[【2】](#_ftn2)和卡内基梅隆大学[软件工程学院的文章中得到证明【3】，](https://insights.sei.cmu.edu/blog/measuring-resilience-in-artificial-intelligence-and-machine-learning-systems/)但是我们如何到达那里呢？在我工作的 Mobilewalla，我们亲身经历了这个问题。在过去 3 年中，我们在生产中部署了 100 多个模型，产生了各种各样的结果:从匿名数字用户的年龄和性别预测，到没有信用记录的申请人的消费贷款违约倾向。在内部，我们跟踪一个名为*不稳定时间* (TTI)的指标，该指标测量模型表现出不稳定所需的时间，即，与预生产基准相比，它低于阈值精度。我们模型的平均 TTI 约为 90 天，方差在 3-250 天之间，明显偏向该范围的较低等级。

**数据漂移和稳定性** 模型在生产中的表现与训练和测试时不同的原因有很多，其中之一就是锚定它们的数据属性的变化。用于创建模型训练特征的原始数据与部署中支持模型的数据不同，这种现象被称为*数据漂移*。当现实世界中的环境以意想不到和计划不到的方式变化时，就会发生数据漂移，这可以说是非弹性模型的主要原因。
实际上，今天的每个企业 ML 软件工具包都包括确定数据漂移的机制，表现为如下漂移函数:

*drift_pkg(distribution1, distribution2) — >{Drift_Metric}*

Large cloud providers have also provided for data drift reporting within their ML suite of offerings; for example, Microsoft Azure Machine Learning dataset monitor, and Amazon Web Services SageMaker Model Monitor.

These are useful tools, but their biggest drawback is that they are reactive. When a deployed model misbehaves, these tools are invoked to check drift, revealing how the data fed into the underperforming model differs from the data used to train it. If drift is detected, the model is corrected (primarily through retraining).

**Towards Resilient Models, Not Reactive Tooling** Correction is fine and necessary, but it doesn’t address the most critical ML engineering problem of all — *how do we build resilient ML models from scratch*? Achieving resiliency means building models that have predictable behavior, and seldom misbehave. Without resiliency, operationalizing ML models will remain a major challenge — modelers will continue to build models that underperform in production, requiring frequent correction. The continual need to re-engineer these models will raise organizational questions over the operational utility of ML/AI.

We have come to believe that it is time to reconsider how drift is used in ML workflows and envision novel ways to incorporate drift into the model-building workflow to prevent, not react to, model misbehavior. To do this, we have formulated an artifact called *data stability*.

**Prioritizing Data Stability** Data drift represents how a target data set is different from a source data set. For time-series data (the most common form of data powering ML models), drift is a measure of the “distance” of data at two different instances in time. The key takeaway is that drift is a *singular*, or *point*, measure of the distance between two different data distributions.

While drift is a point measure, stability is a longitudinal metric. We believe resilient models should be powered by data attributes that exhibit *low drift over time* — such models, by definition, would exhibit less drift-induced misbehavior. In order to manifest this property, drift over time, we introduce the notion of data stability. Stable data attributes drift little over time, whereas unstable data is the opposite. We provide additional details below.

Consider two different attributes: the daily temperature distribution in NYC in November (TEMPNovNYC) and the distribution of the tare weights of aircraft at public airports (AIRKG). It is easy to see that TEMPNovNYC has lower drift than AIRKG; one would expect lesser variation between November temperatures at NYC across various years, than between the weights of aircrafts at two airports (compare the large aircrafts at, say JFK, to ones in a smaller airport, like Montgomery, Alabama). In this example, TEMPNovNYC is more stable as an attribute than AIRKG.

Data stability, though conceptually simple, can serve as a powerful tool to build resilient models. To see this, consider the following simple expression of data flow in ML:

*data — > features — > model*

Base data ingested into the modeling workflow is transformed to produce features, which are then fed into models. Stable data attributes are likely to lead to stable features, which are likely to power resilient models. We use a simple scale of stability values — 0, 1, 2, 3, 4, where 0 indicates “unstable,” and 4 denotes “highly stable” and each data attribute is assigned a stability value. These assignments, provided to our modelers during the feature engineering process, help them build features, and in turn models, that have known stability characteristics. Knowing how stable each input attribute is, modelers attempt to choose stable data elements, while avoiding unstable ones, to build stable features, which, in turn, will power resilient models. Understanding the stability value of data used to construct a feature would alert MLOPs that downstream models (i.e., those using this feature as input) need to be monitored more closely than others. MLOps can now anticipate rather than react.

**A Change in Mindset, a Change in Methodology** We realize that this is a substantive departure from extant methodologies. The current data pipeline for feature construction and model building doesn’t incorporate how “drifty” specific data items are or factor in the notion of data stability. Instead, it’s an ad hoc process, driven primarily by modeler intuition and expertise, incorporating analytic procedures (such as exploratory data analysis or EDA) whose primary objective is to provide the modeler insights into the predictive power of individual data and features. The primary reason for constructing and eventually using a feature is its contribution to the model’s accuracy. The significant drawback of this approach, as we know now, is that high predictivity shown at model testing doesn’t always translate in production, frequently driven by different properties of data than was determined at training and testing. Thus, unstable models.

Based on our experience, it is time for an approach informed by data stability. The modeler doesn’t need to sacrifice predictive power (i.e., model accuracy) but should be able to trade-off model accuracy and stability, building resilient, “accurate-enough” models with predictable behavior.

**Computing Stability** While the notion of using data stability to build resilient models makes intuitive sense, the actual computation of the stability artifact turned out to be quite challenging. Without going into the complete mathematical details, we provide the solution intuition below, followed by links to the source code that implement stability.

概括地说，属性的稳定性估计了它在未来经历漂移的可能性——在某种程度上，我们可以认为稳定性是漂移的*预测器。我们的第一个直觉(有点明显)是，该属性过去的行为将是这个预测的合理代理——来自过去的漂移属性将保持如此向前移动。更难的问题是如何表示数据属性的本质——应该跟踪它的哪些属性来模拟它的行为。通过对我们存储的大量历史消费者数据进行大量实验，我们发现以下四个宽泛的属性最适合描述一个属性并衡量其“漂移度”:***偏斜度*和*形状*。最终的稳定性度量是一个封闭形式的数学表达式，它卷积了代表上述四个属性中每一个属性的特定统计度量。最后一个挑战是规模—稳定性指标在计算上相当广泛—重要的是设计一个可扩展的稳定性实施方案，以便可以在数据摄取时作为核心分析的一部分进行高效计算。为了简洁起见，我们不会在本文中深入讨论细节，但对于感兴趣的读者，我们邀请他们在这里(链接到 anovos.ai)访问稳定性的精确公式的细节，在这里(链接到 Github)访问实现和源代码的细节。在这里(Github 的链接)，我们还提供了如何用样本数据集[计算和使用稳定性的例子。](https://github.com/anovos/anovos/blob/main/examples/notebooks/data_drift.ipynb)***

***数据稳定性的影响***

*我们已经在生产中使用稳定性来设计功能 6 个多月了，超过 60%的模型都配备了稳定性工具。我们已经看到了模型弹性方面有意义的改进。平均 TTI 值(我们在前面描述过)已经从 90 天上升到接近 130 天，并且随着我们对更多模型进行测试，这一数值还在不断提高。此外，由于稳定性是一个纵向度量，它本身在预测漂移方面越来越好，因为它的计算是基于更长的历史。虽然我们无法预测最终的改善情况，但我们的目标是将平均 TTI 提高 2 倍(即 180 天的目标平均 TTI)，这似乎并非遥不可及。有趣的是，除了平均 TTI 的提高，TTI 范围也从[3，250]提高到[10，250]，这表明对于我们最具弹性挑战的模型，我们已经成功地将 TTI 提高了 3+倍。虽然我们还没有，但我们正计划通过增加弹性来衡量模型 ROI 驱动的增加。我们期待着继续分享我们的故事。*

*[【1】](#_ftnref1)Ingo miers wa 博士，弹性>准确性:为什么“模型弹性”应该是可操作化模型的真实指标(2020 年 10 月)[https://www . unite . ai/Resilience-Accuracy-Why-model-Resilience-should-be-the-true-metric-for-operationalization-models/](https://www.unite.ai/resilience-accuracy-why-model-resilience-should-be-the-true-metric-for-operationalizing-models/)*

*[【2】](#_ftnref2)英戈·米尔斯瓦博士，模型精度不够:你需要弹性模型(2020 年 3 月)[https://rapid miner . com/blog/Model-Accuracy-You-Need-Resilient-Models/](https://rapidminer.com/blog/model-accuracy-you-need-resilient-models/)*

*[【3】](#_ftnref3)Alexander Petrilli 和 Shing-Han Lau，人工智能和机器学习系统中的弹性(2019 年 12 月)[://insights . sei . CMU . edu/blog/measuring-Resilience-in-Artificial-Intelligence-and-Machine-Learning-Systems/](https://insights.sei.cmu.edu/blog/measuring-resilience-in-artificial-intelligence-and-machine-learning-systems/)*