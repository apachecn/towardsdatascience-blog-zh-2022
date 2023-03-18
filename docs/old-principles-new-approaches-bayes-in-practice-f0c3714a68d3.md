# 旧原则，新方法:贝叶斯在实践中

> 原文：<https://towardsdatascience.com/old-principles-new-approaches-bayes-in-practice-f0c3714a68d3>

在像数据科学这样以创新为中心的学科中，仅仅几年前还是尖端的方法今天可能会感觉过时。这使得贝叶斯统计——一套有近三个世纪历史的原则——享有如此长的保质期变得更加不可思议。

贝叶斯定理及其衍生应用不是你在大学统计学课程中学到的东西，只会被迅速存档在你记忆的遥远边缘。每天，数据科学和机器学习从业者都在很好地利用这些概念，并找到新的方法在他们的项目中利用它们。

本周，我们来看几个展示贝叶斯方法持久性的当代用例。让我们开始吧。

*   [**用贝叶斯扭曲进行 A/B 测试**](/bayesian-a-b-testing-in-r-4c6471e2e10e) 。 [Hannah Roos](https://medium.com/u/45a9e3b70a2?source=post_page-----f0c3714a68d3--------------------------------) 精彩的深度剖析清晰地解释了贝叶斯统计和频率统计之间的差异，并展示了如何用每种方法进行 A/B 测试。然后，它通过一个真实的例子来衡量他们各自的表现:衡量社交媒体内容的参与度。
*   [**如何用贝叶斯优化让你的模型更好的工作**](/mango-a-new-way-to-make-bayesian-optimisation-in-python-a1a09989c6d8) 。超参数调整是训练机器学习算法并最小化其损失函数的关键步骤。[Carmen Adriana Martinez barb OSA](https://medium.com/u/a0526bfe8d0e?source=post_page-----f0c3714a68d3--------------------------------)解开贝叶斯优化如何改进以前的方法，并带我们通过芒果包在 Python 中实现它。

![](img/9f0ca61269b0824f651c1b5d33b43762.png)

由 [Tara B](https://unsplash.com/@tarabear?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

*   [**给你的分类任务一个贝叶斯提升**](/not-so-naive-bayes-eb0936fa8b4a) 。在他的新解释器中，[micha oles zak](https://medium.com/u/c58320fab2a8?source=post_page-----f0c3714a68d3--------------------------------)涵盖了朴素贝叶斯分类算法的基础知识(如果你对这个主题不熟悉，这是一个很好的起点！).他接着指出，在某些情况下，去除算法天真的独立性假设有助于提高模型的准确性。
*   [**重新看待排名问题**](/learning-to-rank-the-bayesian-way-29af4c61939b) 。部分统计演练，部分实践教程，[Robert kübler](https://medium.com/u/6d6b5fb431bf?source=post_page-----f0c3714a68d3--------------------------------)博士的文章演示了如何构建一个模型，让您对一组球员进行排名(包括您需要的所有 Python 代码)，并阐明了为什么整合先验信念(贝叶斯技术的核心方面)会导致更稳健的排名。

虽然我们中的许多人可以连续几天钻研贝叶斯理论，但你也可能会阅读一些其他主题的优秀读物。以下是我们最近最喜欢的几个:

*   你能使用一个机器学习模型来增强另一个机器学习模型吗？Ria Cheruvu 为复合人工智能系统提供了案例。
*   Erin Wilson 的新帖使复杂的工作流程对初学者变得容易理解:学习如何用 PyTorch 建模 DNA 序列。
*   [Derrick Mwiti](https://medium.com/u/4b814c3bfc04?source=post_page-----f0c3714a68d3--------------------------------) 为任何想要使用 TensorFlow 2 对象检测 API 进行图像分割(当然还有对象检测)的人提供了一个[全面的介绍。](/object-detection-with-tensorflow-2-object-detection-api-3f89da0f1045)
*   新的在线书籍提醒:我们很高兴分享来自[Mathias grnne](https://medium.com/u/1379b0fd8db9?source=post_page-----f0c3714a68d3--------------------------------)的[自动编码器](/introduction-to-embedding-clustering-and-similarity-11dd80b00061)广泛介绍的第一章。
*   免安装交互式 Python 应用？！是的，你也可以通过跟随萨姆·迈诺特的 TDS 处女作来构建它们，这是一个有用的、基于 Streamlit 的教程。
*   [了解如何使用 Apache Spark](/serving-ml-models-with-apache-spark-3adc278f7a78) — [帕纳尔·埃尔索伊](https://medium.com/u/5411ba755d50?source=post_page-----f0c3714a68d3--------------------------------)分享一份耐心的端到端指南。
*   不要错过 [Lynn Kwong](https://medium.com/u/f649eccbbc3d?source=post_page-----f0c3714a68d3--------------------------------) 的最新贡献，它关注于[有效地将大量记录](/how-to-perform-bulk-inserts-with-sqlalchemy-efficiently-in-python-23044656b97d)插入数据库的不同方法。

我们喜欢与您分享伟大的数据科学成果，而您的支持— [包括您的中级会员资格](https://bit.ly/tds-membership) —让这一切成为可能。谢谢大家！

直到下一个变量，

TDS 编辑