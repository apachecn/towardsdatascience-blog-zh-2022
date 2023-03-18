# 数据科学中的强化学习

> 原文：<https://towardsdatascience.com/reinforcement-learning-in-data-science-e7805f133d7>

## 机器学习中使用的另一种技术

![](img/2c4824792f2587947a6ed2e6d2db2d15.png)

照片由[麦克多比胡](https://unsplash.com/@hjx518756?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

在过去的几周里，我一直在研究数据科学中的[线性回归。然而，这个星期，我想改变一些事情。我们对监督学习方法和非监督学习方法有所了解，但我们还没有谈到一种不同类型的学习:强化学习。这是一种不需要监督的学习，就像无监督学习一样，但也有独特的品质。在我们开始之前，有一点需要注意的是，强化学习不像其他模型那样被广泛使用，比如监督学习方法。到目前为止，许多例子都是理论上的或研究驱动的。因此，当我们讨论强化学习时，我们将讨论一些用例，但请记住，许多技术要么正在兴起，要么仍处于理论测试阶段。像往常一样，快速浏览一下我们的主题。首先，我们将讨论什么是数据科学中的强化学习。接下来，我们将看几个使用案例。最后，我们将讨论使用这种机器学习方法的一些好处和缺点。所以，事不宜迟，让我们开始研究吧。](/linear-regression-in-data-science-b9a9d2aacc7c?source=your_stories_page-------------------------------------)

# 什么是强化学习

强化学习是机器学习的一种数据科学方法。这是一种无监督的学习方法，因为您不需要提供标记数据。然而，它不同于典型的无监督学习方法，因为尽管数据是无标记的，但需要显式编程。开发人员必须创建算法来确定目标以及将要使用的奖励和惩罚。强化学习不同于监督学习，因为一旦那些初始参数被写入，开发者就不再需要任何中断。相反，机器将根据设定的目标和参数来解释数据。

用奖惩来决定行为？养狗时，这些信息是有意义的。当狗坐下的时候，给狗一个奖励。否则，不请客。但是你如何对待你的机器呢？老实说，这需要一点研究才能理解。我找到了一个来源，确定一台机器可以用状态来惩罚。如果机器做出了错误的决定，你就向机器返回一个错误。一个错误的状态作为惩罚。老实说，我理解错误是一种“惩罚”，但什么是奖励呢？这让我花了更多的时间去理解。我不认为我完全理解这一点，所以如果你能想到一个更好的方式来描述它，请在评论中留下它。否则，下面是我发现的一个例子:

对于一个学习玩 Pacman 的机器，它被给予了一个基于点数的奖励系统。移动到空白空间没有分数，移动到小球有一分，收集水果有两分，获得能量球有三分，当能量球被激活时击中鬼魂有四分，一旦通关有五分。作为对每一次与鬼魂碰撞的惩罚，如果没有来自能量球的不可战胜性，将会被扣除 5 分。因为初始参数的目的是收集尽可能多的点，所以机器将扣分理解为一种惩罚，因为它现在已经最小化了它可以收集的最大点数。这也鼓励机器尝试新的方法来优化它可以收集的点数。

强化学习使用一个代理，其目标是找到达到奖励的最佳可能方式。这可以通过正强化或负强化来实现。有了积极的强化，事件会因为特定的行为而发生，这就像是一种奖励，这增加了所述行为的强度和频率。我们的目标是使用积极的效果来最大化性能，在更长的时间内维持变化，并且不允许过多的强化产生过载的状态。使用负强化，一个负面的条件实际上是为了停止或避免另一种行为。负面条件增加了行为，提供了对最低绩效标准的挑战，并且仅提供了足以满足最低行为的能力。

现在，我们已经谈了一点什么是强化学习，让我们来看几个用例。

# 强化学习的用例

正如前面提到的，一个例子是教机器玩视频游戏。因为这一点已经解释过了，所以我可以把这一部分讲得简短一些。选择一个视频游戏，写下你的参数(目标、奖励和惩罚)，让机器用不同的方法漫游，直到它完美地以最多的分数运行游戏(如果你选择了基于分数的奖励系统)。

另一个更真实的例子是自动驾驶汽车。如你所知，在旅途中没有办法预测所有的可能性。这一点，以及冗长的 If-Then 块可能需要时间，并且缺乏完整的逻辑。但是因为开发者不能总是监控算法，所以学习方法需要无人监督。通过制造强化学习的方法，你允许代理获得复杂性，也变得更加智能久而久之。您还可以允许代理关注速度等特征，以便汽车可以确定最快的路径，并在某些情况下改变所选路径的速度时对路线进行更改。

# 不同的强化学习技术

我想简单提一下你可以使用的三种不同的强化学习算法可能会有所帮助:

1.  状态-动作-奖励-状态-动作(SARSA)-SARSA 通过将策略传递给代理开始。政策本质上是告诉代理人某个行为产生回报的可能性。
2.  Q-Learning——在 Q-Learning 中，没有政策。这意味着代理必须在没有指导的情况下在环境中导航，使得学习是自我驱动的。
3.  深度 Q 学习-深度 Q 学习使用自我驱动导航，但也利用神经网络。神经网络会记住过去的行为及其回报，这有助于影响未来的决策。

# 强化学习的利与弊

现在我们已经了解了更多关于强化学习的知识，让我们来回顾一下其中的利弊:

赞成的意见

*   该模型可以纠正在训练过程中发生的错误(例如错误分类或决策)
*   如果出现错误，重复同样错误的机会就会大大减少
*   当只想了解环境并与之互动时，这个模型是有用的
*   目标是最大化性能并达到给定环境下模型的理想行为
*   保持探索(测试不同的方法以查看其结果)和开发(使用学到的方法以产生最佳结果)之间的平衡
*   这是实现长期结果的首选方法，而长期结果是很难实现的

这些只是一些优点，所以让我们看看一些缺点:

骗局

*   过多的强化会导致状态过载，从而降低价值
*   简单解决方案不推荐
*   真实物理系统的学习受到维数灾难的限制(在高维空间中分析和组织数据时出现的现象，这些现象不会在低维空间中出现，如日常经验的三维物理空间)
*   真实世界样本的诅咒(例如维护使用这种学习的机器人，因为硬件昂贵并且会随着时间而磨损，这需要对系统进行昂贵的维护和修理)
*   数据饥渴(意味着它需要尽可能多的数据和计算)
*   强化学习的大多数问题需要学习技术的组合来纠正，例如因为与深度学习配对

# 结论

在今天的文章中，我们谈到了强化学习。我们了解到，这是数据科学中使用的另一种机器学习技术，在为代理创建一组指令(策略)以供其遵循后，该技术在无人监督的情况下工作。我们学习了代理如何探索不同的可能性，并面临相关的奖励或惩罚，因此它可以学习哪些决策产生最佳结果。然后，它可以在探索新决策和利用已经发现的最佳结果之间取得平衡。我认为理解如何奖励模型是理解强化学习最困难的部分。尽管如此，看完这些例子后，我对它的含义更有信心了。我希望你也发现这是有用的和有趣的。一旦弄清楚了这一点，我们回顾了学习方法的一些优点和缺点。现在我们对强化学习有了更多的了解，我希望你对这种技术有更好的理解，如果你曾经尝试过使用强化学习，请在评论中告诉我。感谢您的阅读，一如既往，我们将在下一篇文章中再见。干杯！

***用我的*** [***每周简讯***](https://crafty-leader-2062.ck.page/8f8bcfb181) ***免费阅读我的所有文章，谢谢！***

***想阅读介质上的所有文章？成为中等*** [***成员***](https://miketechgame.medium.com/membership) ***今天！***

查看我最近的一些文章

[](https://medium.com/codex/pointless-automation-dropbox-file-downloads-with-python-e1cb26a41fff) [## 无意义的自动化:用 Python 下载 Dropbox 文件

### 即使是简单的东西也可以自动化。

medium.com](https://medium.com/codex/pointless-automation-dropbox-file-downloads-with-python-e1cb26a41fff) [](/linear-regression-what-is-the-sum-of-squares-3746db90a05d) [## 线性回归:平方和是多少？

### 深入了解平方和对数据科学的重要性

towardsdatascience.com](/linear-regression-what-is-the-sum-of-squares-3746db90a05d) [](/linear-regression-in-data-science-b9a9d2aacc7c) [## 数据科学中的线性回归

### 机器学习的数学技术

towardsdatascience.com](/linear-regression-in-data-science-b9a9d2aacc7c) [](https://python.plainenglish.io/getting-started-with-seq-in-python-4f5fde688364) [## Python 中的 Seq 入门

### 向脚本中快速添加日志记录的简单方法。

python .平原英语. io](https://python.plainenglish.io/getting-started-with-seq-in-python-4f5fde688364) [](https://medium.com/codex/javascript-cdns-and-how-to-use-them-offline-e6e6333491a3) [## Javascript CDNs 以及如何离线使用它们

### 它们是什么？它们如何帮助我们？

medium.com](https://medium.com/codex/javascript-cdns-and-how-to-use-them-offline-e6e6333491a3) 

参考资料:

[](https://www.techtarget.com/searchenterpriseai/definition/reinforcement-learning) [## 什么是强化学习？全面的概述

### 强化学习是一种基于奖励期望行为和/或惩罚的机器学习训练方法

www.techtarget.com](https://www.techtarget.com/searchenterpriseai/definition/reinforcement-learning) [](https://deepsense.ai/what-is-reinforcement-learning-the-complete-guide/) [## 什么是强化学习？完全指南- deepsense.ai

### 人工智能预计市场规模为 73.5 亿美元，正在突飞猛进地发展…

deepsense.ai](https://deepsense.ai/what-is-reinforcement-learning-the-complete-guide/) [](https://www.geeksforgeeks.org/what-is-reinforcement-learning/) [## 强化学习-极客论坛

### 强化学习是机器学习的一个领域。这是关于采取适当的行动，以最大限度地提高回报…

www.geeksforgeeks.org](https://www.geeksforgeeks.org/what-is-reinforcement-learning/) [](https://www.simplilearn.com/tutorials/machine-learning-tutorial/reinforcement-learning) [## 什么是强化学习？

### 训练你的狗的最好方法是使用奖励系统。当狗表现好的时候，你给它一点奖励，你…

www.simplilearn.com](https://www.simplilearn.com/tutorials/machine-learning-tutorial/reinforcement-learning) [](https://pythonistaplanet.com/pros-and-cons-of-reinforcement-learning/) [## 强化学习的利与弊|皮托尼斯塔星球

### 我们可以使用许多机器学习策略，每一种都有优点和缺点…

pythonistaplanet.com](https://pythonistaplanet.com/pros-and-cons-of-reinforcement-learning/)