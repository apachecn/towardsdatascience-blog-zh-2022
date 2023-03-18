# 如何像管理产品团队一样管理您的数据团队

> 原文：<https://towardsdatascience.com/how-to-run-your-data-team-like-a-product-team-7efd8a0fd423>

## 伟大的产品团队是以用户为中心的，积极主动的。优秀的数据团队也会从这样做中受益。

![](img/0802322d37f9010e1fe3e9c7c1ab651f.png)

一群太空人互相跟随。我想，这是对我们的数据团队如何需要更好的领导实践的模糊比喻。🤷作者的‍♀️形象。

去年，Emilie Schario 和 Taylor Murphy 提出了这个美妙的想法[“像运营产品团队一样运营您的数据团队”](https://locallyoptimistic.com/post/run-your-data-team-like-a-product-team/)。这篇文章的关键前提是这样的:产品团队有许多伟大的实践，数据团队将从采用这些实践中受益。但是在这个过程中，我们忽略了这一点，并愉快地用稻草人来代替它:[为我们的数据资产维护生产级系统](https://www.mckinsey.com/capabilities/quantumblack/our-insights/how-to-unlock-the-full-value-of-data-manage-it-like-a-product) , [构建更多的数据产品](https://readtechnically.medium.com/data-as-a-product-vs-data-as-a-service-d9f7e622dc55)，或者煞费苦心地定义[生产](https://benn.substack.com/p/what-is-production)在强化[数据契约](/data-contracts-ensure-robustness-in-your-data-mesh-architecture-69a3c38f07db)中的含义。所有这些当然都值得考虑，但他们更关心的是数据和数据资产的正确处理，而不是实际推动影响的数据团队。

这里的中心思想是永远不要对“数据产品”的定义和边界吹毛求疵，或者为数据生产者设置 SLA，而是迫使我们通过使用产品团队作为模型来重新考虑数据团队如何运作。

我想花一些时间来讨论如何像管理产品团队一样管理您的数据团队。

# 两个核心理念:以用户为中心和主动性

产品团队体现了两个核心原则——以用户为中心和主动性。让我们依次讨论每一个。

## 以用户为中心

最好的产品团队是以用户为中心的。他们定期与客户交流，让直接的用户反馈直接影响他们的路线图。这个飞轮是任何好产品的生命线——它确保它们不仅仅是运输功能，而是解决问题。

数据团队需要以同样的方式运作。我们太沉迷于我们的工作在技术上可以有多有趣，而忘记了我们不是科学/工程追求的独立避风港——我们是被雇佣来提供商业价值的业务单位。如果我们像产品团队一样，不使用数据解决业务问题，我们隐喻的“数据产品”(我们所做的所有数据工作)就会失败。

这并不意味着被动地响应临时请求。这也不意味着完全回避科学努力。它只是意味着保持与业务需求的协调，并为此寻求机会。虽然泰勒和艾米莉认为你的同事是你的客户，但我认为这还不够——公司是你的客户。你需要了解它，理解它，并围绕它来指导你所做的一切。

## 积极主动性

第二，最好的产品团队有主动的过程来支持产品构建过程。他们为自己提供有意的空间来设定愿景、集思广益、追求超出直接回应客户要求范围的激情项目。

另一方面，分析团队很少这样运作。至少，我们应该花一些时间探索入站请求之外的数据。在团队层面上，我们应该寻找模式，这样我们就可以有意识地设计我们的路线图，并做高杠杆的工作。

也就是说，反应式工作当然仍然有它的位置——分析师是企业探索数据的主要手段，因此我们经常会发现自己必然处于辅助角色。但关键是不断推动理解这项工作背后的背景，并让这种背景激励战略性的高杠杆项目。

![](img/a985ad71da56b6dc91bf442a4d3cfb3f.png)

图片作者。

# 但是如何开始呢？

最初的 LO 文章有一些很好的组织层面的建议来使这一切成为可能:有足够的人员，这样你就有足够的带宽来进行战略部署；召集一个多学科团队，从中汲取灵感。除此之外，这里有几个具体的*流程*级别的变化，您可以立即 make:‍

## 1.建立巩固知识的过程。‍

将团队的工作放在一个地方是以用户为中心的先决条件。为了让您的工作以用户为中心，您需要了解您被要求做的所有工作。在一个共享的空间中组织你的工作可以使你的团队工作模式匹配——相当于一个产品团队在头脑风暴之前研究 UX 的研究结果。

这将是你最大的障碍，因为不合规是一个大问题。人们努力保持现状，我经常看到文档/知识共享计划失败。在 wiki 工作空间中发布作品，如 idea、Confluence 或 Dropbox paper(对于特定于分析的解决方案， [hyperquery](https://hyperquery.ai/?utm_source=medium&utm_medium=organic-content&utm_campaign=2022-10-18-data-team-product-team) )可以打破这一障碍。

确保广泛采用的关键要素是:

*   **降低使用摩擦。尽管使用 git 和建立同行评审流程可能很有吸引力，但层层流程并不能确保严格性——它们会降低采用率。做一些简单、轻松的事情，并专注于确保您的团队将他们的工作放入工具中。**
*   **搭建组织脚手架**。同样，使用类似于 [hyperquery](https://hyperquery.ai/?utm_source=medium&utm_medium=organic-content&utm_campaign=2022-10-18-data-team-product-team) 、conception 或 Confluence 的工具，通过 wiki 结构，使你的团队不仅能够围绕集中化，而且能够围绕*组织*建立实践。就合理的、功能性的类别达成一致，并创建一个集中的“从这里开始”的文档，让新的分析师加入到您的实践中。

![](img/554cac0893efec56ae6a7ca8d71107b2.png)

在 [hyperquery](https://hyperquery.ai/?utm_source=medium&utm_medium=organic-content&utm_campaign=2022-10-18-data-team-product-team) 中组织的工作。图片作者。

## 2.深入了解业务需求。

我们不仅仅是技术工人。我们是数据和业务其余部分之间的桥梁。如果我们只是沉浸在数据中——这只是对话的一个方面——我们将无法达到应有的效率。

我们为自己的技术实力感到自豪，但是我们只有在知道我们为什么要工作的情况下才是有效的。如果没有敏锐的商业头脑，我们会写下一个又一个毫无意义的分析，直到我们被转移到[存储 B](https://www.youtube.com/watch?v=Vsayg_S4pJg) ，我们的交互被降级为数据拉取。

实际上，这看起来像什么:

*   [**总是问为什么**](/why-youre-doing-ad-hoc-analytics-wrong-49d177202c7a) **。**在你深入 SQL 之前，确保你和你的利益相关者在业务目标上保持一致。把这些写下来，商定一个方法，然后开始工作。这开创了一个先例，即你在决策过程中的参与不仅限于技术工作——至少，你至少会被视为一名翻译，充其量是一名思想伙伴。
*   **关心业务。**这听起来显而易见，但我经常看到分析师和数据科学家无视业务，沉浸在他们工作的技术方面。这种行为预示着一个真正功能失调、影响力低下的分析机构。更高的影响通常不是来自更好的分析，而是来自更高水平的战略执行的数据驱动的影响。

# 结论

在过去的十年中，数据分析的本质发生了巨大的变化。我们比以往任何时候都可以访问更多的数据、更多的计算能力和更多的工具。但是我们还没有想出我们应该在一个拥有新力量的组织中运作。从其他领域借鉴成功的实践可能会有所帮助。特别是对于产品团队来说，对以用户为中心和主动性的关注可能意味着帮助台分析团队和真正推动战略的团队之间的差异。以用户为中心和主动性来自对业务需求的敏锐认识和更好的知识共享实践。

*推文*[*@ imrobertyi*](https://twitter.com/imrobertyi)*/*[*@*hyperquery](http://twitter.com/hyperquery)*来问好。👋*
*关注我们上*[*LinkedIn*](https://www.linkedin.com/company/hyperquery/)*。🙂*

*要了解更多关于 hyperquery 的帮助，请查看*[*hyperquery . ai*](https://hyperquery.ai/?utm_source=medium&utm_medium=organic-content&utm_campaign=2022-10-18-data-team-product-team)*。原博文* [*在 hyperquery 上发布*](https://www.hyperquery.ai/blog/how-to-run-your-data-team-like-a-product-team) *。*