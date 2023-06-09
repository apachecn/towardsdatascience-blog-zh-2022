# 想要为您的投资组合创建更有影响力的数据科学项目吗？

> 原文：<https://towardsdatascience.com/want-to-create-more-impactful-data-science-projects-for-your-portfolio-4e4b0d9099c8>

## 构建热数据科学，而不是冷数据科学

![](img/5f1e0b874ebc07e525888636dca58938.png)

图片来自 thommas68 的 [Pixabay](https://pixabay.com/illustrations/fire-and-water-hands-fight-fire-2354583/)

在数据科学和生活中，热通常比冷更受欢迎。毕竟，冷与旧事物、死亡联系在一起，并且从字面上定义为缺乏(分子)运动。另一方面，热让人联想到火、汗水和动力。

然而，尽管非常清楚地认识到热对冷的吸引力，但大多数数据科学项目最终都是冷的、静态的项目，在这些项目中，模型被训练和评估，然后被报告和保存。只是被遗忘在互联网或我们个人硬盘的深处。

Kaggle 竞赛是这样工作的。我们在主办比赛的任何人(如网飞、脸书等)提供的数据集上训练模型。然后，根据结果度量，将这些模型与其他模型进行比较。最佳结果指标=获胜者。就是这样。完成了。

数据依然冷淡，因此项目也冷淡。

好吧，我想在某些情况下，模型获胜者可能能够为赞助企业部署他们的模型，但这些情况很少，并且发生在竞争之外。因此，我们从来没有对新的和输入的数据进行模型性能评估。我们永远无法确定这些模型的真正普遍性。

考虑这个场景:构建了两个模型来预测业务结果。一个模型是使用训练数据的训练-测试-验证分割来构建的。仅基于特征在模型中的统计显著性来挑选特征(或者更糟糕的是，不使用特征选择过程)。

第二个模型也建立在训练数据的分割以及包括来自科学文献的特征的基础上，尽管在训练数据中没有明显的统计意义。

第一个模型在训练数据上优于第二个模型。应该的。毕竟是用训练数据优化的。

然而，第二个模型在未来数据上表现更好。为什么？因为它包含了第一个模型没有发现的特征，但根据科学研究，这些特征可以预测结果。因此，尽管所述文献衍生的特征在训练数据中未被发现(例如，统计显著性，由于缺乏特征工程而不可用等)，但随着时间的推移，并且随着新数据的包含，这些特征在准确预测未来结果中仍然重要。

无论是哪种情况，学习如何在冷数据上建立模型是一回事，但学习如何创建能够响应传入的热数据的解决方案将为您的工具包添加一套全新的技能，可以为发展中的数据科学家项目组合做出巨大贡献。

在本文中，我分析了冷门和热门数据科学项目之间的区别，并揭示了将您的数据科学项目转变为热门投资组合商品的一些有用步骤和注意事项。感兴趣吗？接着读下去！

# **热门数据让数据科学项目变得更酷**

热数据和冷数据之间的区别对于技术领域[来说并不陌生。事实上，亚马逊的 AWS 提供了一项名为“冰川”的 S3 存储服务我是说，你能有多冷？现在推出…ZeroK 存储！但是我跑题了。](https://www.backblaze.com/blog/whats-the-diff-hot-and-cold-data-storage/#:~:text=Generally%2C%20though%2C%20hot%20data%20requires,and%20consequently%2C%20less%20expensive%20media)

就数据而言，热数据是经常使用、定期刷新和/或快速需要的数据。因此，热数据的存储成本通常也更高，因为计算要求不同于冷数据。

至少，热数据科学需要建立一个响应热数据的流程。也就是说，让我们的数据科学投资组合变得更酷(俚语的本意)意味着建立热门数据存储，让我们能够对新数据的模型进行评分。

但是，开发热数据科学解决方案需要的不仅仅是热数据存储。热门数据科学需要使用不同的方法，这些方法通常由 MLOps 团队决定，但仍然会受益于数据科学家在设置方面的专业知识。

例如，热数据科学需要包含评估指标，以持续监控新数据和传入数据的模型性能。当我们观察到度量中的模型漂移时，我们该怎么办？事实上，监控模型性能和寻找模型漂移只是热门数据科学问题的一部分。

真正热门的数据科学解决方案是允许随着新数据和新功能的出现而构建和训练新模型的解决方案。精明的数据科学家可能认识到这种方法是冠军挑战者方法的基础。Challenger 模型是用新的和输入的数据构建的，确保我们不仅解决了模型漂移问题，还拥有一个持续改进模型的机制。

随着您建立热门数据科学的技能的提高，您甚至可以找到机会从客户、用户或任何可以用来进一步训练新模型的人那里收集数据。随着可用训练数据量的增加，这种场景非常适合建立再训练协议，或者使用强化算法建立在线学习。

可能性是无限的，但是在他们自己的项目组合中，从哪里开始展示这种技能呢？

在下一节中，我们将更深入地探讨一些想法，以建立您自己的热门数据科学项目，并将其添加到您的投资组合中。

# 使用热数据创建数据管道

如前所述，热数据科学项目至少需要建立一个热数据存储管道。作为项目组合的一部分，在演示中有很多方法可以做到这一点。首先，您需要确定一个您想要解决的问题，该问题可以通过热数据来解决。还有什么比互联网更重要的热门数据来源呢？

比方说，你是一个内容创建者，你想看看每周新闻都有哪些话题，以便为你提供一个策略，让你知道自己的内容应该关注什么。一种方法可能是从 Google Alerts 设置一些电子邮件提醒，通过电子邮件向您发送与您的专业主题相关的热门新闻。通过对电子邮件服务进行一些调整，可以每周部署一次 python 代码来提取结果，并将它们存储在本地机器上的简单 SQLite 数据库中。

建立这样一个服务还需要您确定如何使代码自动化。设置和调度代码的一些想法包括在 Windows 机器上使用窗口调度器，比如这里的，或者在 Linux 机器上使用 Cron，如这里的[所解释的](/how-to-schedule-python-scripts-with-cron-the-only-guide-youll-ever-need-deea2df63b4e)。对于更高级的编排，请查看[气流](https://airflow.apache.org/)。

一旦有了用新数据存储和更新数据存储的方法，下一步就是构建用于对数据进行评分的数据科学。您可以使用预先训练的模型来标记具有已知感兴趣标签的新闻源，或者您可以使用无监督的主题建模方法来发现每周的关键主题。

# **利用输入数据构建冠军挑战者场景**

一旦我们建立了一个按计划收集数据的管道，使用监督或非监督模型对数据进行评分，并通过某种机制(例如，一组保存的图像、数据集、仪表板)向我们提供这些结果，下一步就是建立一个可以从新数据中学习的流程。

在无监督模型的情况下，比如主题模型，我们可能想要识别是否出现了新的主题。或者，在预训练监督模型的情况下，我们可能希望确保我们的模型仍然能够准确地对新数据进行评分。在这两种情况下，我们都需要在新数据到来时建立训练新模型的能力，并将其与现有模型进行比较。

# **结论**

在这篇简短的文章中，我研究了一些想法，通过关注热数据的使用来激发创建更令人兴奋和更有影响力的数据科学项目。值得注意的是，热数据科学不仅仅是热数据，因为它涉及到在整个解决方案管道中包含额外的方法和考虑事项。此外，热门数据科学更令人兴奋，因为它展示了我们如何将数据科学视为一个活生生的过程，可以使用源自其自身学科的方法来创建更智能的数据输出，从而满足我们自己的生活和呼吸需求。

比如参与学习数据科学、职业发展、生活或糟糕的商业决策？[加入我](https://www.facebook.com/groups/thinkdatascience)。