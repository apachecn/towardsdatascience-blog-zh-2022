# 非工程师的坏数据指南

> 原文：<https://towardsdatascience.com/the-non-engineers-guide-to-bad-data-c5325d2b1458>

## 为什么你最喜欢的仪表板坏了，以及如何修理它们。

![](img/90b94b36992359b17a4073376ec7bcf6.png)

图片由 [Shutterstock](http://www.shutterstock.com) 上的 [fizkes](https://www.shutterstock.com/g/fizkes) 提供，购买使用标准许可。

*根据 HFS* 、[、**、*最近的一项研究，75%的高管不相信他们的数据*、**、*。以下是为什么以及依赖数据的团队可以做些什么。*](https://info.syniti.com/hfs-report)

现在是周一早上 8 点 45 分——还有 15 分钟，你就要和你的营销主管进行电话讨论上个月广告活动的表现了。

为了准备会议，您在 Tableau 中刷新了一个仪表板(就是您每个月用于这个会议的那个仪表板！)来可视化一个新的推广对你网站访客数量的影响。

浏览了一下报告，你愣住了。

数据是完全错误的——11 月份的 50，000 次独立月访问量怎么会下降到 12 月份的 500 次？

呀。你在检查错误的仪表板吗？

不，是同一份(*营销广告支出报告——很好——用这份——V2*)。你快速侦查的尝试让你回到了起点。

你的下一个策略是什么？给数据部的吉尔发几条宽松的信息。

*“为什么这个报道是错的？”*

“我看到的是上个月的数据吗？”

“我们的营销主管会不高兴的。”

**没反应。**

时钟在上午 9 点敲响。你别无选择，只能两手空空地登录 Zoom，却一无所知。谁需要数据呢？

如果这种情况似曾相识，请放心:你并不孤单。

对于很多数据驱动的专业人士来说，这个问题， [**数据宕机**](https://www.montecarlodata.com/the-rise-of-data-downtime/) ，是经常发生的事情，导致失去信任，浪费时间，处处让老板沮丧。数据宕机指的是数据丢失、不准确或出现其他错误的时间段，并且很少得到应有的关注。

您没有相信您的数据能够为决策提供信息，而是被迫质疑您团队的报告和仪表板是否可靠，以及从长远来看，您公司的任何数据是否可信。事实上，根据 HFS 研究公司 2021 年的一项研究，75%的高管不相信他们的数据。

但是事情并不一定是那样的。

在本文中，我们将介绍贵公司数据宕机的主要症状，强调数据中断的首要原因，并分享将**信任和可靠性**带入仪表板的更好方法。

# 但是首先:关于无畏的数据工程师的一句话。

在我们深入探讨之前，我要感谢现代数据团队的无名英雄: [**数据工程师**](https://www.coursera.org/articles/what-does-a-data-engineer-do-and-how-do-i-become-one) **，**往往是抵御破碎仪表盘和报表的第一道防线。

每当你向你公司的数据工程师询问丢失的数据时，你可能会得到这样的回答(或者类似的话):*“让我调查一下。”*

如果你的 CMO 的主要目的是修改你的报告，那么每月与他们的会面会容易得多，但他们有很多事情要做，然而他们为组织带来的价值常常被忽视。

从回应来自分析团队的特别请求到构建您组织的数据基础设施(这样您就可以了解您的新登录页面有多粘，或者新功能在减少客户流失方面有多有效)，[成为一名数据工程师](https://www.montecarlodata.com/the-future-of-the-data-engineer/)极具挑战性。数据工程师还确保数据在正确的时间流向正确的地方，将新的数据源纳入数据生态系统，构建新的[数据产品](https://www.montecarlodata.com/how-to-build-your-data-platform-like-a-product/)，并设计更具弹性的数据系统。

在处理数据停机时，这些责任甚至还不算什么。

当您的数据出现问题时，他们会立即找出发生了什么(通常称为 [**根本原因分析**](https://www.montecarlodata.com/the-data-engineers-guide-to-root-cause-analysis/) )、对业务的影响以及如何修复它。

为了把握这个责任的轻重，了解破碎数据对你公司的影响是有帮助的。

# 不良数据对您业务的影响

![](img/937964ac1ffd1e1f243669b6dc59ac24.png)

一个糟糕的仪表板就能毁掉一次完美的会议。图片由[卢克·切瑟](https://unsplash.com/photos/JKUTrJ4vK00)在 [Unsplash](http://www.unsplash.com) 上提供。

不准确的报告、在会议中感到措手不及，或者缺少工作所需的重要信息，这些都是糟糕数据的常见后果，但现实对公司的底线有着更大的影响。

以下是一些与数据质量差相关的切实的业务问题，也许它们会引起共鸣:

*   **收入损失:**这是显而易见的:数据支持您的业务目标，从帮助您重新锁定潜在客户到优化客户体验。如果你的营销团队使用活动数据来预测广告支出，你的营销团队可能会在这些工作上超支或支出不足。
*   **资源分配不当:**数据宕机的最大后果之一是预算分配不当，这是由陈旧或不准确的数据导致的，这些数据为推动决策制定的关键报告提供了动力。如果你的季度预测显示，在密苏里州的堪萨斯城开一家新餐馆的最佳市场是，在你花费数百万美元开一家新餐馆之前，你最好确定数据是正确的。
*   **用于创新的时间更少:**平均而言，数据工程师每周花费超过 [**20 个小时**](https://www.montecarlodata.com/data-observability-how-blinkist-increases-trust-in-data-with-monte-carlo/) 来调查和解决数据火灾演习。由于数据工程师一半以上的时间都用于解决数据质量问题，因此他们从事增值活动的能力有限，例如构建实验平台或其他产品来改进公司的产品和服务。

最终，坏数据会耗费你公司每个人的时间和金钱，而不仅仅是数据工程师。

在这种背景下，让我们回到最初的问题:为什么数据会损坏，以及如何帮助您的数据工程师修复它？

## 那么，数据为什么会断裂呢？

![](img/750e8760011012502d584fbc75de70a1.png)

数据损坏可能有数百万种原因，由数据工程师来确定根本原因。图片由 [Shutterstock](http://www.shutterstock.com) 上的[彼得拜尔](https://www.shutterstock.com/image-photo/support-colleague-you-can-find-your-1812761575)提供。

数据中断的原因有很多，通常归结为不止一个根本原因。(我们就不说技术术语了，但好奇的读者应该去看看奥莱利关于数据质量的新书<https://www.oreilly.com/library/view/data-quality-fundamentals/9781098112035/>****，了解数据可靠性和可信度的来龙去脉)。****

****当你和你的数据工程朋友一起修理你的仪表板时，它有助于理解幕后发生的事情。以下是即使是最懂数据的公司也面临的一些常见数据质量问题:****

*   ******应用业务逻辑的系统出现故障:**通常情况下，数据工程师会运行定制代码，将业务逻辑应用于数据(即特定范围内的值)。有时，SQL(或 Python 或 Apache Spark)命令完全失败，有时它不能正常运行，这两种情况都会导致坏数据从裂缝中溜走，破坏下游的仪表板。****
*   ******坏源数据:**当一个数据集或表格包含重复记录、不完整数据、过期列或格式不一致的数据时，我们称之为*坏源数据*。错误的源数据可能会中断进入报表、仪表板和产品的数据流。****
*   ******未能验证您的数据:**在数据被清理并采用适合您业务的格式后，需要对其进行验证，或者换句话说，根据业务要求进行检查，以确保数据有意义且正确。如果工程师无法验证他们的数据，坏数据进入有价值的仪表板和报告的可能性就会增加。****
*   ******模式的意外变化:**如果你不熟悉[模式](/understanding-data-engineering-jargon-schema-and-master-branch-525dff66fcb8)，它基本上是数据的 DNA。模式构成数据库和数据仓库，并包含如何设计数据库的说明。有时，当将代码推向生产时，工程师可能会在代码中出现一个小错误(例如将表名从“customer”更改为“customers”)，这反过来会导致一系列问题。这些变化通常会被忽视，直到事实发生，损害已经造成。****
*   ******bug 被引入代码:**把 bug 想象成代码的错误或意外行为。随着公司的成长，预测和捕捉每一个可能发生的错误会变得非常困难，尤其是当代码变得越来越复杂的时候。错误代码是每个行业数据停机的主要原因，甚至导致[一个价值 1 . 25 亿美元的 NASA 探测器崩溃](https://www.latimes.com/archives/la-xpm-1999-oct-01-mn-17288-story.html#:~:text=NASA%20lost%20its%20%24125%2Dmillion,space%20agency%20officials%20said%20Thursday.&text=In%20a%20sense%2C%20the%20spacecraft%20was%20lost%20in%20translation.)。****
*   ******贵组织运营环境的更新/变更:**数据问题的另一个常见原因是运行[数据作业](https://www.ibm.com/docs/SSQNUZ_2.5.0/cpd/svc/datastage/t_run_jobs.html)的运营环境的变更，换句话说，是数据生命周期中的一个阶段。这背后的原因各不相同，从长时间运行的查询到意外丢失或放错数据的作业计划的更改。(通过我的同事弗朗西斯科了解更多关于这个话题的信息。****

****既然我们已经知道了损坏数据的症状、影响和原因，那么实际上我们该如何修复它呢？****

## ****如何用可观测性防止不良数据****

****![](img/b1bbf68c3c9d0887574293403b65f076.png)****

****数据可观察性为公司提供了一个 10，000 英尺的数据健康视图，帮助他们利用更准确的数据做出更好的决策。图片由[哑光贵族](https://unsplash.com/photos/BpTMNN9JSmQ)在 [Unsplash](http://www.unsplash.com) 上提供。****

****幸运的是，我们可以超越疯狂的懈怠信息和烦扰的文本，向我们在数据工程领域的朋友保持领先，以打破仪表板和报告。然而，要实现这一点，我们首先需要了解数据的健康状况。****

****一种越来越受公司欢迎的防止数据停机的方法是通过 [**数据可观测性**](https://www.montecarlodata.com/data-observability-the-next-frontier-of-data-engineering/) ，这是一种以协作和整体方式大规模处理数据质量的新方法。通过确保数据工程师在坏数据到达您的仪表板之前首先了解坏数据，数据可观察性有助于公司了解其数据在任何给定时间点的可靠性和可信度。****

****通过数据可观察性，即使是分析师和其他了解数据的业务用户也可以获得关键数据问题的警报，从而避免因凭直觉做出决策和丢失数据而导致的沮丧、尴尬和收入损失。现在，数据工程师可以在公司其他任何人意识到有问题之前排除故障并解决这些问题。****

****如果您的数据损坏并导致下游问题，可观察性使您的数据团队能够了解损坏的报告的影响，以便您可以相应地纠正过程，并防止丢失的值在周一上午的会议中造成影响。****

****在此之前，祝您拥有完美的仪表盘、准确的预测，最重要的是，没有数据宕机。****

*******想了解*** [***数据可观察性***](https://www.montecarlodata.com/data-observability-the-next-frontier-of-data-engineering/) ***如何修复您组织的破损仪表板和报表？*******

*******向巴尔伸出援手，团队在*** [***蒙特卡洛***](http://www.montecarlodata.com/request-a-demo) ***进行更多了解。*******

****[*斯科特·奥利里*](https://www.linkedin.com/in/scott-o-leary-78000a43/) *对本文有贡献。*****