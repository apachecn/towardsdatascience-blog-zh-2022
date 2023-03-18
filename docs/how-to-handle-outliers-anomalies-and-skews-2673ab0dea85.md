# 如何处理异常值、异常值和偏差

> 原文：<https://towardsdatascience.com/how-to-handle-outliers-anomalies-and-skews-2673ab0dea85>

数据科学是关于发现模式并从分析中提取有意义的见解。然而，正如任何从业者都知道的，数据喜欢偶尔扔给我们一个曲线球:一个奇怪的峰值，一个意外的下降，或者(喘息！)一个形状奇特的星团。

本周，我们将注意力转向那些不和谐的时刻，当事情(和我们的图表)没有我们希望的那么顺利。我们精选的亮点涵盖了处理不规则性和应对不可预测性的不同方法。

*   [**寻找离群值，正确的方法**](/are-you-using-feature-distributions-to-detect-outliers-48e2ae3309) 。正如[亨尼·德·哈德](https://medium.com/u/fb96be98b7b9?source=post_page-----2673ab0dea85--------------------------------)所观察到的，“在大多数项目中，数据将包含多个维度，这使得肉眼很难发现异常值。”Hennie 展示了如何利用 Cook 的距离、DBSCAN 和隔离森林来识别需要额外审查的数据点，而不是依赖我们容易出错的观察能力。
*   [**如何将数据偏斜的潜在危险降到最低**](/3-common-strategies-to-measure-bias-in-nlp-models-2022-b948a671d257) 。在过去的几年里，偏见一直是数据和 ML 专业人士的热门词汇。 [Adam Brownell](https://medium.com/u/2479b1fc8999?source=post_page-----2673ab0dea85--------------------------------) 邀请我们将偏见视为“产生一种伤害的偏差”，并带领我们通过三种策略在自然语言处理模型的背景下有效地测量它。

![](img/ceb12cecd1e79c295218a79dd24f33fc.png)

詹妮弗·博伊尔在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

*   [**对抗性训练来救援？**](/ganomaly-paper-review-semi-supervised-anomaly-detection-via-adversarial-training-a6f7a64a265f) 异常检测在计算机视觉中尤其困难，在这种情况下，小数据量和通常有限的图像种类使得模型训练成为一种挑战。尤金妮亚·阿内洛(Eugenia Anello)的有益解释者带领我们通过一种新颖的方法 GANomaly，它利用生成对抗网络的力量来解决以前方法的缺点。
*   [](/dealing-with-outliers-using-three-robust-linear-regression-models-544cfbd00767)**保持线性回归的异常值。要实际演示稳健的线性算法，以及如何使用它们处理数据中潜伏的异常值，你应该看看[埃里克·莱文森](https://medium.com/u/44bc27317e6b?source=post_page-----2673ab0dea85--------------------------------)最近的教程。它涵盖了 Huber 回归、随机样本一致性(RANSAC)回归和 Theil-Sen 回归，并在同一数据集上对它们的性能进行了基准测试。**

**对于其他主题的顶级文章，我们在下面收集了一些我们最近喜欢的文章。不要害怕:这里没有局外人，只有持续启发性的讨论。**

*   **[Ari Joury 博士](https://medium.com/u/593908e0206?source=post_page-----2673ab0dea85--------------------------------)的新文章已经引起了轰动，他认为在花费大量时间寻找正确的算法之前，你应该[确保你知道你实际上在解决什么问题](/forget-about-algorithms-and-models-learn-how-to-solve-problems-first-c791fde5842e)。**
*   **在超市挑选移动最快的结账队伍是一个古老的难题，但是 [LeAnne Chan](https://medium.com/u/3984a193c444?source=post_page-----2673ab0dea85--------------------------------) 在这里帮助进行[基于博弈论的分析](/the-game-theory-of-queuing-bd1095998c42)。**
*   **如果你想[使用你的数据集来训练一个关于拥抱脸的深度学习模型](/how-to-turn-your-local-zip-data-into-a-huggingface-dataset-43f754c68f82)，你很幸运:[Varshita Sher 博士的最新教程展示了如何以一种平稳、无痛的过程移植你的数据。](https://medium.com/u/f8ca36def59?source=post_page-----2673ab0dea85--------------------------------)**
*   **正在蓬勃发展的合成数据生成领域还没有对表格数据给予足够的重视；Javier Marin 的新深度探索涵盖了一个新的开源项目，旨在纠正 T21 的这个问题。**
*   **是时候找到你的耳机了:我们很高兴[分享 TDS 播客](/can-the-u-s-and-china-collaborate-on-ai-safety-f066731975d1)的新一集，由[杰瑞米·哈里斯](https://medium.com/u/59564831d1eb?source=post_page-----2673ab0dea85--------------------------------)和分析师瑞安·费达修克主演；他们的谈话围绕着美国和中国在人工智能安全方面合作的潜力。**

**我们喜欢与您分享伟大的数据科学成果，您的支持——[包括您的媒体会员](https://bit.ly/tds-membership)——让这一切成为可能。谢谢大家！**

**直到下一个变量，**

**TDS 编辑**