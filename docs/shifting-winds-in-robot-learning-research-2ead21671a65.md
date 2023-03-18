# 机器人学习研究中的风向转变

> 原文：<https://towardsdatascience.com/shifting-winds-in-robot-learning-research-2ead21671a65>

## 强化学习已死，强化学习万岁！

![](img/fc35c4e4c6961f67939cfada8f37b7d6.png)

鸣谢:拜伦·大卫(经许可)

当我告诉科技圈的人我从事机器人的机器学习时，他们的第一反应是‘哦，强化学习……’这并不罕见，我过去从不考虑这种说法。毕竟，在过去几年中，我们看到的很大一部分成功都是关于将机器人操纵框定为[大规模学习](https://ai.googleblog.com/2016/03/deep-learning-for-robots-learning-from.html)，将问题转化为[强化学习自我改进循环](https://ai.googleblog.com/2018/06/scalable-deep-reinforcement-learning.html)，[大规模扩展飞轮](https://ai.googleblog.com/2021/04/multi-task-robotic-reinforcement.html)，[在此过程中学习大量课程](https://journals.sagepub.com/doi/full/10.1177/0278364920987859)，瞧！用伊利亚·苏茨基弗的不朽名言来说:“成功是有保证的。”

除了…这很复杂。强化学习(RL)可以说是[一头难以驯服的野兽](https://www.alexirpan.com/2018/02/14/rl-hard.html)。这导致了有趣的研究动态，如果你的主要目标不是关注学习循环，而是关注表示或模型架构，有监督的学习就更容易工作。因此，[许多](https://arxiv.org/abs/1903.01973) [研究](https://transporternets.github.io/) [线程](https://ai.googleblog.com/2022/10/table-tennis-research-platform-for.html)专注于监督学习——在机器人术语中又称为行为克隆(BC)——而将 RL 留给读者作为练习。即使在 RL 应该大放异彩的地方，围绕[随机搜索](https://ai.googleblog.com/2022/10/pi-ars-accelerating-evolution-learned.html?m=1)和[黑盒方法](http://proceedings.mlr.press/v80/choromanski18a.html)的各种变化也让“经典”RL 算法大放异彩。

然后… BC 方法开始[变好](https://sites.google.com/view/bc-z/home)。[真的](https://arxiv.org/abs/2109.00137) [好的](https://interactive-language.github.io)。如此之好，以至于我们今天的[最佳操纵系统](https://say-can.github.io/)大多使用 BC，上面还撒了一点 Q 学习来执行高级动作选择。今天，不到 20%的研究投资是在 RL 上，基于 BC 的方法的研究跑道感觉更健壮。机器人学习研究几乎是 RL 代名词的日子是不是已经过去了？

虽然听起来很诱人，但我认为今天就放弃是非常有问题的。RL 的主要承诺是自主探索:随着经验扩展，没有任何人类保姆。这有两个主要后果:在模拟中执行大量经验收集的机会，以及在现实世界中自主数据收集的可能性。

有了 RL，你就有了一个机器人学习过程，它需要在模拟基础设施上进行固定投资，然后随着 CPU 和现场部署的机器人的数量而扩展——如果你能访问大量计算，这是一个很好的制度。但在以 BC 为中心的世界中，从可扩展性的角度来看，我们最终反而陷入了最差的局部最优:我们仍然需要投资于模拟，即使只是为了进行快速实验和模型选择，但当涉及到经验收集时，我们基本上只能随着在监督环境中控制机器人的人类数量而扩展。然后，当你自主部署机器人时，不仅人类启发的行为是你的天花板，而且封闭探索和持续学习的循环变得极其困难。谢尔盖·莱文雄辩地阐述了这代表的长期机会成本。

但是，很难突破 BC 的吸引力:押注于大规模模型很少是一个好主意，如果这些模型要求监督而不是强化，那么我们有什么理由争论呢？“巨型语言模型”革命应该让任何人暂停专注于设计复杂的训练循环，而不是一头扎进收集大量数据的问题中。也不是不可能想象，一旦我们接受了监督机器人的巨大固定成本，我们就可以让它们一直达到“足够好”的性能水平以取得成功——毕竟，这是自动驾驶汽车行业的整个战略。也不是不可能想象，一旦我们找到了更多可扩展的方法来在现实世界的机器人环境中释放自我监督学习，扬恩蛋糕上的[樱桃](https://twitter.com/ylecun/status/1396238451176099842?lang=en)开始尝起来有点酸了。

我不是唯一一个[注意到研发领域风向变化的人。该领域的许多人已经将目光投向离线 RL，作为突破自主馆藏天花板的途径。最近的一些焦点是让 BC 和 RL](https://www.reddit.com/r/MachineLearning/comments/xfmqny/d_what_happened_to_reinforcement_learning/) [相互配合](https://awopt.github.io)，将[可扩展的探索带到监督设置](https://language-play.github.io/)，或者让 RL 假装它是一个[监督顺序决策问题](https://arxiv.org/abs/2106.01345)，以便保留大型变压器的理想缩放属性。这是对稳定的 MuJoCo 研究的一个令人耳目一新的突破，误差棒大到几乎无法在页面上显示(哈哈！).我预计未来几个月将会有更多关于这一健康的自我反省过程的具体表述出现，并有望出现新的见解，说明如何最好地驾驭 BC 的近期回报与 RL 的长期承诺之间的紧张关系。

*感谢凯罗尔·豪斯曼对本文草稿的反馈。观点都是我的。*