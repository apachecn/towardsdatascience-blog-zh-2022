# 数据科学家每天面临的挑战

> 原文：<https://towardsdatascience.com/challenges-data-scientists-face-everyday-344bb18fba84>

![](img/ce46ee19cb4ab6ab2e1ec1ceb4150bac.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Boitumelo Phetla](https://unsplash.com/@writecodenow?utm_source=medium&utm_medium=referral) 拍摄的照片

**数据科学**和**机器学习**现在是互联网上的热门词汇，而且这种趋势还在增长。随着各种格式的大量数据，公司越来越依赖于数据科学家、机器学习工程师和软件开发人员来自动化各种日常任务的流程，并提高短期和长期运营的**生产率**和**效率**。此外，数据科学家和 ML 工程师的工资也随着良好的薪酬和股票福利进一步增加。

![](img/d585f30fb572d05149f995631e1b0b3a.png)

由 [Markus Spiske](https://unsplash.com/@markusspiske?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

然而，也应该考虑到数据科学家在工作中经常面临许多挑战，从数据提取到大规模部署最佳超参数调整模型。因此，意识到这些挑战并学习如何应对它们会对轻松完成工作产生重大影响。下面重点介绍了数据科学家在工作中面临的一些挑战，以及应对这些挑战的一些技巧和策略。

数据在任何地方都可以以各种格式获得，例如以文本、视频、音频、图片和网站的形式。根据 seedscientific.com 提供的估计，在 2020 年的黎明，世界上可用的数据量是惊人的 440 亿字节。今年的数字甚至更高，而且在未来也会有增长的趋势。有了这些庞大的信息，通过分析趋势和了解预测来充分利用这些信息对公司来说是很方便的，这样他们就可以采取适当的措施来确保他们朝着正确的方向前进并获得利润。

了解了下面详细介绍的挑战后，数据科学家可以收集应对挑战所需的所有工具和资源，并为公司做出有益的贡献。

## 寻找正确的数据

![](img/ee49c92e9653069bac8ad4222ced3b90.png)

安德烈·德·森蒂斯峰在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

拥有大量数据的挑战是找到团队可以使用的正确数据，以便他们从中产生有价值的模式和见解。重要的是要问一些问题，如谁应该得到什么数据，是否应该有持续的数据流用于分析，或者数据是否是固定的。问这些有趣的问题可以使数据科学工作流的任务变得轻松，同时使系统设计变得不那么繁琐和易于遵循。

可能存在包含大量影响机器学习模型性能的**异常值**、**缺失值**或**不准确信息**的数据。因此，对数据进行预处理也很重要，这样模型就能以最佳方式高效运行，同时性能也有很大提高。

## **数据准备**

![](img/a7046af1a36636b912ed09c0f8e78fb8.png)

由 [Towfiqu barbhuiya](https://unsplash.com/@towfiqu999999?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

数据科学家必须考虑的挑战之一是准备大量的数据，并使其可供团队的其他成员访问和解释，同时提供他们自己有用的见解和模式。预处理数据也有助于增加其可读性，以便团队中的其他成员可以查看数据中的特征。有些情况下，数据中的各种特征可能存在异常值，必须将其视为异常值，因为并非所有的机器学习模型都对这些异常值具有鲁棒性。除此之外，还可能存在包含缺失值或不正确值的特征，这些特征必须被识别，以便它们不会降低准备在生产中部署的 ML 模型的性能。所有这些事情都可以在**探索性数据分析** (EDA)的帮助下确定，这通常是处理大量数据时机器学习的第一步。因此，最初必须遵循这一步骤，以确保我们分别从我们的模型中获得最佳结果。

## 选择正确的绩效指标

![](img/7b4ac7b53216120847c03252aa5462c2.png)

安德烈·德·森蒂斯峰在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

由于机器学习中有大量可用的指标，因此可能会陷入循环，而无法决定可用于评估的最佳工具或指标。对于分类问题，我们有流行的度量标准，如**准确度**、**精确度**、**召回**和**f1-分数**以及其他。

对于回归任务，我们必须考虑其他指标，例如**均方误差**或**平均绝对误差**。在时间序列问题的情况下，这也主要是一个回归任务，我们采取其他指标，如**平均绝对百分比误差** (MAPE)或**均方根误差**。因此，选择正确的指标可能是数据科学家或机器学习工程师必须应对的挑战，以提高工作效率，并确保公司通过这种分析获得最佳结果。

## 部署

![](img/d030fe1e5a0251977fe80595815c49cc.png)

照片由[戴恩·托普金](https://unsplash.com/@dtopkin1?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

在获取数据并对其进行预处理并确保它在交叉验证数据上表现良好之后，现在是时候部署它并将其投入生产了。毕竟，如果模型只是给出正确的预测，而没有显示测试数据或以前没有见过的数据的结果，这将是没有用的。因此，还应该考虑在生产中部署模型。

有时，在尝试实时部署模型时，还应该考虑用于运行这些模型的基础设施。如果我们想要一个在互联网应用中广泛使用的低延迟系统，选择能够快速给出结果的 ML 模型可能是一个值得考虑的好事情。还有其他系统对延迟的要求可能没有这么严格。一些应用涉及电影的网飞推荐系统。在这个系统中，并不总是需要在很短的时间内给出建议。该模型可能需要一两天的时间从感兴趣的特定用户以及其他用户那里收集更多的见解，然后提出可靠的推荐。因此，在部署之前考虑手头的业务问题是必要的。

## 性能监控

![](img/3d6bb99053de94a2a3dab3bb86473264.png)

照片由 [Ahmed Zayan](https://unsplash.com/@zayyerrn?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

作为一名机器学习工程师，监控模型在生产中的性能非常重要。就项目的延迟、效率和范围而言，总有改进的余地。还可能出现这样的情况:模型变得不正常，或者基于新数据产生扭曲的结果。因此，对模型的不断监控和再训练可能是机器学习工程师必须处理的挑战之一。

减少数据的维度也可以是监控系统性能的好步骤，并且根据 ML 问题是分类问题还是回归问题来查看准确性或均方误差是否有大的降低。

## 结论

总而言之，我们已经看到了如何使用机器学习以及与机器学习工作流相关的挑战。看一看这些挑战，数据科学家可以确保他们拥有正确的工具和资源来应对这些挑战，并为公司提供有价值的见解。

如果你想进一步了解我的工作，下面是我们可以联系的细节，你也可以查看我的工作。谢了。

GitHub:[https://github.com/suhasmaddali](https://github.com/suhasmaddali)

**领英:**[https://www.linkedin.com/in/suhas-maddali/](https://www.linkedin.com/in/suhas-maddali/)

**https://www.facebook.com/suhas.maddali**[脸书](https://www.facebook.com/suhas.maddali)