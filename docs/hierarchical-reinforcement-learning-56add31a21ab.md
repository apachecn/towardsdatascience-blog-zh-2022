# 分层强化学习

> 原文：<https://towardsdatascience.com/hierarchical-reinforcement-learning-56add31a21ab>

## 使用表格 Q-Learning 的 Options-Critic 框架

![](img/abc08d9e99c1bb2fa1b965b2a941ec25.png)

[unsplash.com](https://unsplash.com/photos/Ya2j-l92uXQ)

分层强化学习将长期决策过程分解成更简单的子任务。这个想法非常类似于将大量代码行分解成更小的函数，每个函数执行一个非常具体的任务。

我们来看一个例子，假设代理人要清理或摆放餐桌。这包括伸手拿盘子的任务。这些都是高水平的任务。在较低的水平上，它需要控制和移动四肢，然后手指伸出并抓住物体，然后将它们放在适当的位置。分层强化学习是用同样的逻辑设计的。有多个策略级别，每个策略处理较低级别的任务，如移动手指，而较高级别的策略处理任务，如抓取对象。

HRL 让我们在训练和探索中受益匪浅:

1.  训练:由于高等级对应多个环境步骤，因此情节相对较短，从而更快地传播奖励并提高学习。
2.  由于探索发生在更高的层次，它能够学习更有意义的策略，从而采取比原子层次更有意义的行动。举例来说，与理解手指的关节运动相比，智能体在抓握物体的层次上会学习到更好的策略来达到目标。

HRL 的一些常见架构是-

1.  [选项—评论家框架](https://arxiv.org/abs/1609.05140)
2.  [封建强化学习](https://proceedings.neurips.cc/paper/1992/file/d14220ee66aeec73c49038385428ec4c-Paper.pdf)

让我们看看如何使用 Q-Learning 在一个简单的四个房间的环境中建立你自己的选择-评论框架。你可以看看这个博客来了解更多关于期权批判框架是如何工作的。

我们将在这里使用 2D 四室环境。环境有 4 个房间。env.reset()重置环境并返回随机开始状态。我们可以使用 env.goal()来更改目标，并将目标设置为从 4 个房间的 1 个角落开始随机选择。在这篇博客中，我们将在一千集之后改变一次目标，这类似于选择批评论文。-[https://github . com/anki 08/Option-Critic/blob/main/four rooms . py](https://github.com/anki08/Option-Critic/blob/main/fourrooms.py)。

![](img/3f74175e0059a328dcde7297e1b72d3f.png)

四个房间的环境——作者图片

现在让我们创建我们的策略。

Q_Omega 是通知较低层策略的较高层元策略。

让我们将 Q_Omega 策略定义为 2D Q 表，其中每个状态都有一组选项。这些选项指导较低级别的策略采取何种行动来最大化其回报。选项的数量由选项定义。我们以ε贪婪的方式从 Q_Omega 表中采样。

选项是动作的概括，让我们定义宏观动作。在 [Sutton et al. (1999)](https://pdf.sciencedirectassets.com/271585/1-s2.0-S0004370200X00549/1-s2.0-S0004370299000521/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEIP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIESLGtl5GrhhtZzdj5yqOgLrGDYmpVYXrKly71u0oGyFAiBv%2FCP50jyEYbf%2F%2BE2HfV6EW7K5zBrv5JIzFhq9uCH6Eyq9Awi8%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAMaDDA1OTAwMzU0Njg2NSIMNYyrwvB33DZ1Kvq2KpED2FGegHwZmRk4sG7jfW1H78WdCz2G5%2BpEUrTz7aVuPOz0Zq0uPgCvB%2FKt%2FnIs0nxyfjamVx%2BTiEg7mXd8Kx7LV1EuFqSrFAIZnICMr5wmQowQ5D1XY%2BGCRRCgAsSi2BYqPtVpR6taAMgOwvaotyUXU0l6%2BVWE%2F5WJJEYkKdSYNhFGTLGwt0q7KtzO4B8n1BydcHuoeUmAfXptKjAQ8UFMEC2UnQNcBEz3tsN1SnC7%2BGvMi79YSxGypwzqa6SjZK1AyUQMRi%2F8whfXBereqN3ambFtnW0uGU9iuZi3MXWC3oZVQkrEmvXrqFeb21V0XZhprRG9NzDdpDmCrtuWWJsDu8XJTR8978wWKmHFPfnOMdQ6DvJ12vfy%2BwxNktNdVe8Tv61CwpYaHVU1n0D9h7VnGYjrA00XwBbdSd1HWv%2BTf88TMfRfHWi1emigaIsCFO8roZA8NOtsagNpLnKUuTrh89c1SeNGHrkyp9bKiEAe4Lztujruty1HSYm%2Bel0wUKdfpL4F%2FH7BtEHgq9bEwSiYrGYw3MzP9QU67AGxunsKSQrJkq6nGSSjWdrUrWMcvcBN6YtyOo7S5Ou%2F5s4WeZltWSkYMKa%2FbrbnLbNwvk%2BmMYUEfroS%2FNN51aaY03WfncGwo6DvX6S3llWno0E5mtLiF7x3JYXkilFnXnpcleWERzSP6ubqMLFYxyvsFAdgImdAK4DXxV89kmyht0tG%2FxQeQ1BULxqwbg8RlTIL0TTusUvCDy8Wu0zaRJpwxvwYZAWukK7VbnrGEkMZkaJ7iVs9EZBgMmw72HW7sRViY2X3qdfbJCI4Twzbx1T%2FXwhJ8RH50e0LpE%2B5CXwZIc5iHqee3pw5E7mWfA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200507T112434Z&X-Amz-SignedHeaders=host&X-Amz-Expires=299&X-Amz-Credential=ASIAQ3PHCVTY2WMI6KVY%2F20200507%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=fbeea06d2fffc09501375dfd3c8ccb28725cbffa5f2776ac31e8fb35278d5d0a&hash=e3cf3d3e59d1e679b1771c52c7242cfd07f5faa8a499adb416f226a1b1297e89&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0004370299000521&tid=spdf-d8bd98f6-5625-46bc-b78e-9f168d0a531b&sid=2d141b7d833611469889b85080d0b045cbe8gxrqb&type=client) 中，它被定义为:

> O *选项由三个部分组成:策略π : S × A → [0，1],终止条件β : S+ → [0，1],初始集 I ⊆ S。当且仅当 st∈ I 时，选项⟨I,π,β⟩在状态 st 可用。如果选择了选项，则根据π选择动作，直到选项根据β随机终止。*

让我们将较低级别的策略创建为名为 Q_U 的 softmax 策略。我们的 Q_U 表存储了较低级别的策略采取的操作。这里，让我们用一个 softmax 策略来举例说明我们的操作。我们的更新是行李员更新。

最后，让我们定义我们的终止策略。这将用于更改较低级别一直遵循的选项或较高级别的策略。

我们已经准备好了两层。现在我们将定义我们的批评家。这是演员-批评家框架的延伸。在这里，批评家评估选项，并告诉更高层次的选项有多好。Q_Omega，即通知选项内策略的元策略，Q_U 和终止策略形成选项部分的一部分，而 Q_U，即在状态和选项的上下文中执行动作的值，形成批评的一部分。

现在我们有了所有的部件，让我们训练和测试代理。我为培训和测试部分创建了一个 colab 笔记本。[*https://colab . research . Google . com/drive/1 q 5j 0 ceahgp 2m _ bgjm 3 nhfpkvqgzwzj _ M？usp =分享*](https://colab.research.google.com/drive/1q5J0CeAhGP2M_bgJM3NhFpKVqgzWzJ_M?usp=sharing)

在训练期间，元策略学习选项和终止策略，然后在测试期间使用这些选项和终止策略来通知较低级别的策略要采取的动作。我们每 1000 集改变一次目标，以表明随着目标的每一次改变，代理花费更少的时间，因为它不必从头开始学习策略，并使用先前的知识来通知行动。

![](img/c24f791eec8e1dcbd1a4c61afbf786b3.png)

作者图片

正如我们所看到的，代理只需 6 个步骤就可以学习并达到目标，而 Q-learning 需要 50 个步骤。

你可以在这里找到完整的代码→【https://github.com/anki08/Option-Critic 

参考资料:

[](https://www.sciencedirect.com/science/article/pii/S0004370299000521)  [](https://arxiv.org/abs/1609.05140)  [](/hierarchical-reinforcement-learning-a2cca9b76097) 