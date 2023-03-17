# 强化学习初学者指南

> 原文：<https://towardsdatascience.com/beginners-guide-to-reinforcement-learning-f296e8dd8260>

## 强化学习模型的高级概述

![](img/dddcdad28f32392a1a93447d79411272.png)

凯利·西克玛在 [Unsplash](https://unsplash.com/s/photos/reinforcement-learning?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

强化学习是[机器学习](https://databasecamp.de/en/machine-learning)中的第四种主要学习方法，与[监督](https://databasecamp.de/en/ml/supervised-learning-models)，无监督，半监督学习并列。主要区别在于模型不需要任何[数据](https://databasecamp.de/en/data)来训练。它通过奖励想要的行为和惩罚坏的行为来学习结构。

# 强化学习的例子

在我们可以详细了解这种模型的训练过程之前，我们应该了解这些算法在哪些情况下可以有所帮助:

*   当教计算机玩游戏时，使用强化学习。目的是学习哪些策略会导致胜利，哪些不会。
*   在自动驾驶中，也使用这些学习算法，以便车辆可以自行决定哪种行动是最好的。
*   对于服务器机房的空调，强化学习模型决定何时以及在多大程度上冷却房间以有效地使用能量。

强化学习的应用通常以必须做出大量连续决策的事实为特征。程序员也可以向计算机具体说明这些(例如室温:“如果温度上升到 24°C 以上，则冷却到 20°C”)。

然而，在强化学习的帮助下，人们希望避免形成一连串的如果-那么条件。一方面，这在许多用例中可能根本不可能，比如自动驾驶，因为程序员无法预见所有的可能性。另一方面，人们希望这些模型也能为复杂问题开发新的策略，而这可能是人类根本做不到的。

# 强化学习如何工作

强化学习模型应该被训练成能够独立做出一系列决策。假设我们要训练这样一个算法，代理人，尽可能成功地玩游戏吃豆人。代理从游戏领域中的任意位置开始，并且它可以执行有限数量的可能动作。在我们的例子中，这将是四个方向(上、下、右或左),它可以在运动场上前进。

算法在这个游戏中找到自己的环境是游戏场和幽灵的移动，这是一定不能遇到的。在每一个动作之后，比如上升，代理会收到一个直接的反馈，即奖励。在《吃豆人》中，这些要么是获得积分，要么是遭遇幽灵。也有可能在一次行动后没有直接的回报，但它会在未来发生，例如在一两次进一步的行动中。对代理人来说，未来的奖励不如眼前的奖励有价值。

随着时间的推移，代理人制定了所谓的政策，即承诺最高长期回报的行动策略。在第一轮中，算法选择完全随机的动作，因为它还不能获得任何经验。然而，随着时间的推移，一个有希望的策略出现了。

# 机器学习方法之间的差异

在[机器学习](https://databasecamp.de/en/machine-learning)领域，总共有四种不同的学习方法:

1.  [**监督学习**](https://databasecamp.de/en/ml/supervised-learning-models) 算法使用已经包含模型应该预测的标签的数据集来学习关系。但是，它们只能识别和学习包含在训练数据中的结构。例如，监督模型用于图像的分类。使用已经分配到一个类的图像，他们学习识别关系，然后他们可以应用到新的图像。
2.  **无监督学习**算法从一个数据集学习，但是这个数据集还没有这些标签。它们试图识别自己的规则和结构，以便能够将数据分类到尽可能具有相同属性的组中。例如，当您想要根据共同特征将客户分组时，可以使用无监督学习。例如，订单频率或订单金额可用于此目的。然而，由模型本身来决定它使用哪些特征。
3.  **半监督学习**是[监督学习](https://databasecamp.de/en/ml/supervised-learning-models)和非监督学习的混合。该模型具有相对较小的带有标签的数据集和较大的带有未标签数据的数据集。目标是从少量已标记的信息中学习关系，并在未标记的数据集中测试这些关系以从中学习。
4.  强化学习不同于以前的方法，因为它不需要训练数据，而是通过所描述的奖励系统简单地工作和学习。

# 强化学习需要训练数据吗？

是的，强化学习模型也需要数据来进行训练。但与其他机器学习方法相比，这些信息不必在外部数据集中给出，而是可以在训练过程中创建。

在[机器学习](https://databasecamp.de/en/machine-learning)的世界里，数据对于训练好的、健壮的模型是必不可少的。[监督学习](https://databasecamp.de/en/ml/supervised-learning-models)为此使用人类标记的数据，这种数据最好在大量情况下可用。这通常很昂贵，而且数据集很难获得或创建。[无监督学习](https://databasecamp.de/en/ml/unsupervised-learnings)另一方面，也需要大量数据，但不需要有标签。这使得获取信息更加便宜和容易。

正如我们已经看到的，强化学习与监督学习和非监督学习完全相反。然而，大量的数据通常会导致更好的训练结果的原则在这里也适用。然而，与其他类型的机器学习模型的不同之处在于，这些数据不一定是外部提供的，而是由模型本身生成的。

让我们以学习一个游戏为例:一个机器学习模型被训练来赢得游戏，即导致胜利的移动被认为是积极的，导致失败的移动被认为是消极的。在这种情况下，模型可以使用许多游戏运行作为训练数据，因为目标“赢得游戏”是明确定义的。随着每一个新的游戏，模型学习，新的训练数据产生，模型变得更好。

# 强化学习是深度学习的未来吗？

强化学习在未来无法取代[深度学习](https://databasecamp.de/en/ml/deep-learning-en)。这两个子领域是紧密相连的，但它们并不相同。[深度学习](https://databasecamp.de/en/ml/deep-learning-en)算法非常擅长识别大型数据集中的结构，并将其应用于新的未知数据。另一方面，强化学习模型即使没有训练数据集也能做出决策。

在许多领域，[机器学习](https://databasecamp.de/en/machine-learning)和[深度学习](https://databasecamp.de/en/ml/deep-learning-en)模型将继续足以取得良好的效果。另一方面，强化学习的成功意味着现在可以开辟以前不可想象的人工智能新领域。然而，也有一些应用，如股票交易，强化学习将取代[深度学习](https://databasecamp.de/en/ml/deep-learning-en)模型，因为它提供了更好的结果。

在这个领域中，已经尝试学习如何从过去的市场数据中识别和交易新股票。然而，对于股票业务来说，训练一种强化学习算法来制定具体的策略，而不依赖于过去的数据，可能更有前途。

# 这是你应该带走的东西

*   强化学习是机器学习领域的一种学习方法。
*   它指的是被训练来预测一系列决策的模型，这些决策承诺最高的可能成功率。
*   例如，强化学习用于教计算机玩游戏或在自动驾驶中做出正确的决定。

*如果你喜欢我的作品，请在这里订阅*[](https://medium.com/subscribe/@niklas_lang)**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，媒体允许你每月免费阅读* ***3 篇*** *。如果你想让***无限制地访问我的文章和数以千计的精彩文章，不要犹豫，通过点击我的推荐链接:*[https://medium.com/@niklas_lang/membership](https://medium.com/@niklas_lang/membership)获得会员资格，每个月只需支付 ***5*****

***[](https://medium.com/@niklas_lang/basics-of-ai-deep-learning-vs-machine-learning-93a8499d5679) [## 人工智能基础:深度学习与机器学习

### 深度学习是一种来自信息处理的方法，使用神经网络分析大量数据。这个…

medium.com](https://medium.com/@niklas_lang/basics-of-ai-deep-learning-vs-machine-learning-93a8499d5679) [](https://medium.com/@niklas_lang/basics-of-ai-supervised-learning-8505219f07cf) [## 人工智能基础:监督学习

### 监督学习是人工智能和机器学习的一个子类。它的特点是…

medium.com](https://medium.com/@niklas_lang/basics-of-ai-supervised-learning-8505219f07cf) [](https://medium.com/@niklas_lang/what-are-recurrent-neural-networks-5c48f4908e34) [## 理解递归神经网络

### 递归神经网络(RNNs)是第三种主要类型的神经网络，前向网络和神经网络

medium.com](https://medium.com/@niklas_lang/what-are-recurrent-neural-networks-5c48f4908e34)***