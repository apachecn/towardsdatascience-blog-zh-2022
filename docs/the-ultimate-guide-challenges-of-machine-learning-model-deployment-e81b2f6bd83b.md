# 终极指南:机器学习模型部署的挑战

> 原文：<https://towardsdatascience.com/the-ultimate-guide-challenges-of-machine-learning-model-deployment-e81b2f6bd83b>

# 动机

> “机器学习模型部署很容易”

这是一个我听过很多次的神话。作为一个有工程背景的数据科学家，我也有这个观点，直到实际开发了一个机器学习部署(或 [MLOps](https://aipaca.ai/) )项目。从技术上讲，部署机器学习(ML)模型可能非常简单:启动一台服务器，创建一个 ML [推理](https://hazelcast.com/glossary/machine-learning-inference/#:~:text=Machine%20learning%20(ML)%20inference%20is,as%20a%20single%20numerical%20score.&text=ML%20inference%20is%20the%20second,data%20to%20produce%20actionable%20output.) API，并将该 API 应用于一个现有的应用程序。不幸的是，这个工作流程太容易出现了，以至于人们往往低估了它的复杂性。事实上，我的一些 ML 工程师朋友抱怨说，他们的工作不被这么多人理解，例如来自不同团队的工程师、产品经理、执行团队，甚至客户。

![](img/12506b63d5d8097c87eaeb0c31e0fc3b.png)

不要根据提示来判断 MLOps 项目的复杂性。西蒙·李在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

通过这个故事，我希望更多的人能够理解 MLOps 背后的困难。我想穿上工程师裤，与你分享 ML 模型部署挑战的终极指南。

*背景:ML 工程师与数据科学家紧密合作。例如，数据科学家构建 ML 模型，ML 工程师实现模型。*

# 第一阶段:当一个模型刚刚交给 ML 工程师时

![](img/c951040ff9cfc29e70e0592b67300016.png)

照片由 [ETA+](https://unsplash.com/@etaplus?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

> **“该模型实际上无法在生产服务器上运行”**

当数据科学家将他们的模型传递给 ML 工程师时，该模型可能无法在不同的机器上工作。这个问题通常是由**软件** **环境变化**或者**代码质量差**造成的。

像 [Docker](https://www.docker.com/) 和 [K8s](https://kubernetes.io/) 这样的容器能够通过跨机器调整软件环境来解决大部分的复制问题。然而，模型容器化并不是每个数据科学家(DS)都具备的技能。如果这种情况发生，DS 和 ML 工程师将需要额外的时间来交流知识。

另一方面，服务器和 ML 框架之间的编译也会导致系统错误。例如，尽管 Flask + Tensorflow 是许多教程中使用的组合，但有一段时间我们发现，随着环境变得越来越复杂，Flask 服务器环境对 Tensorflow 2 并不友好。我们花了一段时间才找到解决办法。

数据科学家不是程序员。写代码时遵循 [PEP 8 指南](https://www.python.org/dev/peps/pep-0008/)对于数据科学来说并不是必须的。一个 ML 工程师声称的“糟糕的代码质量”可能来自于科学家和工程师之间不同的编码习惯。Jupyter Notebook 取代了 VS Code 等传统 IDE，是一款更受数据科学家欢迎的代码编辑工具。笔记本中的编程逻辑与普通的软件开发非常不同。因此，当代码模型从 Jupyter Notebook 迁移出来时，它可能会出错。

如果生产服务器使用与开发服务器不同规格(例如，操作系统、CPU 类型、GPU)的机器，那么 MLOps 项目将会上升到更高的复杂性水平。

# 阶段 2:当团队开始合作时

假设来自数据科学团队的 ML 模型现在可以在生产环境中成功运行，那么是时候将它迁移到现有的应用程序中了。然而，在哪里以及如何在应用中使用模型来解决实际的业务问题是一个新的课题，需要跨团队的协作。

![](img/55cc760c45c89a9135b8bc7b44e69ed6.png)

[杰森·古德曼](https://unsplash.com/@jasongoodman_youxventures?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

> **“我们为什么要关心……”**

由于分散的责任和优先权，跨团队的沟通面临许多挑战。**工程师们关心软件的效率、系统的稳定性和易维护性**。然而，大多数决策支持系统更关心 ML 模型的性能和严密性。为了最大化模型性能，他们总是利用各种数据科学工具。我见过 DS 的同事用 SQL 预处理数据，用 R 启动模型管道，然后是 Sklearn，最后是 Pytorch。当然，这种结构是不会被工程师欣赏的。

当 DS 和 ML 工程师争论什么应该更优先时，产品经理(pm)进入舞台并要求两个团队关注路线图，因为**pm 的责任是确保产品交付按时发布**。

***“有意思，部署 ML 车型的门票正在引发团队辩论……”***

没有固定的解决方案来避免这种纠结。软件效率，ML 模型性能，路线图，哪个更有意义？答案是企业与企业之间的转移，而且永远不会完美。

# 阶段 3:当模型即将发布时

团队最终为了彼此的需要而妥协。工程团队还成功地将模型推理功能添加到应用程序中。一切看起来都很好，不是吗？

> **“等一下，模型托管服务器应该支持多少流量？”**

![](img/2150881a82ee661ef6a06355771bd705.png)

压倒性的系统日志。由 [Markus Spiske](https://unsplash.com/@markusspiske?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

当涉及到用户吞吐量问题时，如果一家公司资源丰富，最值得推荐的解决方案是购买一组功能强大的服务器，即使在高峰时也足以处理所有流量负载。然而，用于容纳 ML 模型的机器是稀缺和昂贵的。8 个 V100 核心的按需 p 3.16 x 大型 AWS 服务器的定价为每小时 24.48 美元，每月 17625.6 美元。

可悲的是，上述解决方案只有少数公司负担得起。对于其他公司来说，根据需要扩展计算能力更实际，但即使对于高级 ML 工程师来说也很有挑战性。数据搜索、并发、一致性和速度是伸缩系统中的四个常见问题。更糟糕的是，由于服务器容量的不足，ML 的可扩展性更加困难:假设你的项目中最常用的云服务器叫做服务器 A，在传统的扩展系统中，你只需要考虑你应该扩展到的服务器 A 的数量。但是在机器学习中，即使在 AWS 这样的大型云平台中，服务器 A 也并不总是具备容量，因为它是稀缺的。您的扩展策略还应该包括具有更高容量的其他类型的服务器。[负载测试](https://en.wikipedia.org/wiki/Load_testing)需要对所有种类的组合进行。还可以添加新的云平台，这样，如果服务器 A 在一个平台上不可用，您仍然有机会通过查找其他平台来获得一个。因此，很少有 ML 项目最终开发出成熟的缩放系统。

# 阶段 4:当模型被部署时

恭喜你！您最终部署了模型，但现在还不是离开的时候。

> **“什么？挑战还没结束？”**

即使你是一个在这个行业工作了 10 年的有经验的 ML 工程师，你的 ML 基础设施也不会一直在运转。其实如果你是有经验的，你应该比我更担心 ML 系统变质。

![](img/1c87f4fdd89d74e2f220cbdecc2ed7f9.png)

不稳定堆积的石头。照片由[科尔顿鲟鱼](https://unsplash.com/@coltonsturgeon?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

恶化来自两个方面:工程方面和数据科学方面。

在工程方面，内部软件迭代可能是关闭机器的主要原因，特别是当模型部署模块与应用程序的其余部分高度集成时。当一个软件更新时，它可能会破坏其他的连接部分。隔离模块可能是一个解决方案，但是缺点是开发速度变慢，因为重用的工作减少了。同样，引入和升级外部软件包也会对系统稳定性产生负面影响。例如，当版本升级时，R 包以破坏模型脚本而闻名。

在最坏的情况下，工程师可能会犯错误。曾经有一段时间 [Google Photo 工程师不小心部署了一个性能很差的模型](https://www.theverge.com/2018/1/12/16882408/google-racist-gorillas-photo-recognition-algorithm-ai)，它把黑人朋友认成了“大猩猩”。

在数据科学方面，随着时间的推移，[数据转移](https://www.section.io/engineering-education/correcting-data-shift/)是模型性能的一大杀手。数据偏移被定义为来自 ML 模型的输入和输出数据之间的潜在关系的改变。如果发生数据转移，数据科学家将需要重新训练旧模型。[反馈回路](https://www.clarifai.com/blog/closing-the-loop-how-feedback-loops-help-to-maintain-quality-long-term-ai-results)是克服数据偏移的解决方案之一。它检测性能变化，并通过新收集的数据重新训练部署的模型。是的，你是对的。这种解决方案也有不利的一面。模型可能存在严重偏差，偏差问题难以识别。

*“假设一家杂货店使用 ML 模型来预测下个月的库存变化。该模型预测瓶装水是下个月最受欢迎的商品，因此店主采纳了它的建议，储备了更多的瓶装水。因为有更多的瓶装水，下个月最畅销的商品确实是瓶装水，这个数据作为新收集的数据被再次输入到 ML 模型中。结果，反馈回路使模型非常偏向瓶装水，总是要求所有者获得更多的瓶装水……当然，这种预测是不恰当的。”*

为了检测退化，监控系统在模型部署中是必不可少的，这也是最后一个挑战点。监视器需要是实时的，检测异常事件，发送警报，收集 ML 度量，跟踪模型性能，等等。

# 结束了

这篇博客描述了工程团队在部署 ML 模型时可能面临的挑战。我描述了时间序列中的挑战。总结一下:

1.  阶段 1 中的挑战:当从开发环境迁移到生产环境时，模型可能会有不同的行为。
2.  阶段 2 中的挑战:当在生产中将 ML 模型添加到现有的应用程序中时，很难满足所有团队的需求。
3.  第 3 阶段的挑战:构建可扩展的计算能力来服务模型是必要的，但也是艰难的。
4.  第四阶段的挑战:ML 系统总是随着时间而恶化；应该建立一个监测系统。

[一个常见的团队配置是每个数据科学家有 2-3 名数据工程师](https://www.oreilly.com/radar/data-engineers-vs-data-scientists/)，在一些具有更复杂数据工程任务的组织中，这个数字可能会超过 5 名。这种说法与我的经验相关，即 ML 模型部署总是比模型开发花费更长的时间(除了由学术界领导的旨在给整个 ML 世界带来翻天覆地变化的研究)。为了保持故事的简洁，我在解释一些挑战时仍然保持高水平。如果你有兴趣了解更多的细节，请加入我的不和谐社区来 DM 我:【https://discord.gg/vUzAUj7V。

# 关于我们

我们来自 [Aipaca](https://aipaca.ai) 团队，构建一个无服务器的 MLOps 工具 Aibro，帮助数据科学家训练&在 2 分钟内在云平台上部署 AI 模型。与此同时，Aibro 采用了专为机器学习打造的成本节约战略，将云成本降低了 85%。

![](img/8b0a2de8c2a76d6caa6e735f9d83f9ca.png)

AIpaca Inc .图片作者