# 数据:工程和科学相遇的地方

> 原文：<https://towardsdatascience.com/data-where-engineering-and-science-meet-296f54131752>

数据工程和数据科学之间的分歧仍然存在吗？

在某些地方，当然是这样:负责移动、存储和维护原始数据的工程师和处理、可视化和分析数据的专家呆在各自不同的领域。然而，在许多环境和组织中，事情变得更加模糊。一个人(或一个团队)可能承担一系列介于传统数据工程和数据科学工作之间的职责。

本周，我们从数据工程的角度挑选了一些优秀的帖子。有些处理您可能非常熟悉的工作流，而有些处理您尚未探索的流程。我们认为无论如何你都会发现它们是有用的:增加我们工具包的深度，或者更具体地了解我们同事正在做的工作，这从来都不是一个坏主意。开始了。

*   [**建设高性价比管道**](/how-i-build-a-real-time-bigquery-pipeline-for-cost-saving-and-capacity-planning-15712c97f058) 。我们都应该注意我们的数据和计算繁重的工作消耗的资源，[的高](https://medium.com/u/2adc5a07e772?source=post_page-----296f54131752--------------------------------)说，“不管你是管理资源的工程师还是收账单的经理。”本文指导我们做出明智的决策，同时控制与云相关的费用。
*   [**如何避免最坏的数据迁移陷阱**](/top-25-painful-data-migration-pitfalls-you-wish-you-had-learned-sooner-a949c3e3b80a) 。在现实世界中，搬到一个新地方很少是有趣的，而且——唉！—在数字领域，这也可能是一种痛苦。 [Hanzala Qureshi](https://medium.com/u/467270b83111?source=post_page-----296f54131752--------------------------------) 为规划和执行平稳的数据迁移提供了有用的资源，明确关注于维护数据质量和“测试、测试和更多测试”
*   [**一个精简的数据库工作流程？是的，请**](/pymysql-connecting-python-and-sql-for-data-science-91e7582d21d7) 。如果您是一名数据科学家，但还不太精通 SQL，这不应该成为您与数据工程朋友在重要数据库项目上合作的障碍。 [Kenneth Leung](https://medium.com/u/dcd08e36f2d0?source=post_page-----296f54131752--------------------------------) 带我们领略 PyMySQL 的魔力，它使用 Python 访问和查询 MySQL 数据库成为可能。
*   [**有效管道事在毫升，太**](/from-ml-model-to-ml-pipeline-9f95c32c6512) 。机器学习从业者投入如此多的精力在他们的模型的预测能力上是有道理的；这是他们工作的核心。然而，正如 [Zolzaya Luvsandorj](https://medium.com/u/5bca2b935223?source=post_page-----296f54131752--------------------------------) 所强调的，确保你的模型能够顺利接收原始数据、对其进行预处理并产生输出也同样重要。

![](img/c0f9f5ede8062a88fac1c7ded22e269d.png)

照片由[斯维特拉娜 B](https://unsplash.com/es/@svebar15?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

*   [**掌握编排容器化应用的艺术**](/learn-kubernetes-the-easy-way-d1cfa460c013) 。Kubernetes 是许多数据基础设施中的一个关键构建块，但是学习如何使用它似乎令人望而生畏。Percy Bolmér 在这里提供帮助，他提供了一个深入代码的完整教程，但保持了事情的可管理性和可访问性。
*   [**熟悉数据存储的来龙去脉**](/aws-essentials-for-data-science-storage-5755afc3cb4a) 。云存储是一项几乎每个人都在使用的技术，但很少有人关心了解它的内部工作原理。如果你想更好地理解它在数据科学工作流环境中的功能，马特·索斯纳的最新贡献非常详细地涵盖了 AWS 的本质。
*   [**设计 GitHub 和 Docker**](/create-robust-data-pipelines-with-prefect-docker-and-github-12b231ca6ed2) 之间的健壮桥梁。Khuyen Tran 最近的教程清楚地表明，一点点数据工程知识可以在非常局部的个人层面上简化过程。它利用开源工具在不同的位置存储和执行您的代码，同时自动化该过程的主要循环块。

等等，还有呢！(不是一直都有吗？)如果你正在寻找其他主题的引人入胜的读物，我们推荐以下几个选项:

*   [玛丽·勒费夫尔](https://medium.com/u/2a04bf49928f?source=post_page-----296f54131752--------------------------------)分享了关于[帮助一篇数据科学文章形成的要素的实用见解](/writing-a-data-article-is-like-building-a-house-755fc267760a)。
*   怀特的异方差一致性估计量听起来可能有点拗口(好吧，它*就是*有点拗口)，但是[萨钦日期](https://medium.com/u/b75b5b1730f3?source=post_page-----296f54131752--------------------------------)是让复杂话题变得平易近人的专家[。](/introducing-the-whites-heteroskedasticity-consistent-estimator-821beee28516)
*   要获得清晰、生动的正态分布介绍，只需看看艾德丽安·克莱恩在广受欢迎的统计训练营系列中的最新文章。
*   [wero nika Gawarska-Tywonek](https://medium.com/u/28e24868993e?source=post_page-----296f54131752--------------------------------)对数据可视化中颜色的探索汇集了[设计理论和实际操作的见解](/the-function-of-color-in-data-viz-a-simple-but-complete-guide-c324ca4c71d0)。
*   通过收听最新的 TDS 播客来了解 ML 传感器的新兴世界，主持人 [Jeremie Harris](https://medium.com/u/59564831d1eb?source=post_page-----296f54131752--------------------------------) 和 TinyML 研究员(和 TDS 撰稿人) [Matthew Stewart](https://medium.com/u/b89dbc0712c4?source=post_page-----296f54131752--------------------------------) 。

感谢您支持我们发表的作品！如果你想产生最大的影响，考虑成为中层成员。

直到下一个变量，

TDS 编辑