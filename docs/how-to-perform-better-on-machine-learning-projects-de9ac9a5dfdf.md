# 如何在机器学习项目上表现得更好

> 原文：<https://towardsdatascience.com/how-to-perform-better-on-machine-learning-projects-de9ac9a5dfdf>

## 问三个问题，下次会更好

![](img/0eb2ff9c54d097b90e79c5f092b86d63.png)

[马特·霍华德](https://unsplash.com/@thematthoward?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

喜剧演员克里斯·洛克一小步一小步地提出新颖的想法。在获得巨大成功的演出之前，他尽可能多的晚上呆在小型喜剧俱乐部，尝试新的笑话。当然大部分都是不了了之，观众嘲笑他。但在这一百个失败的想法中，不可避免地会有一小撮火花。这个过程就是克里斯·洛克如何为他的新节目做准备的，长达一年的时间每天苦读一小时的(高超的)娱乐节目。因为:戏后是戏前。

我们可以从 Rock 的创作过程中学到很多东西:纪律、谦逊，以及接触挑剔甚至敌对的观众。然而，我想挑出三个问题，我们可以在结束一个(机器学习)项目后使用。这三个问题是*我学到了什么？*、*下次要避免什么？*，还有*我下一次又该怎么做？回答这三个问题可以让我们在下一个项目中表现得更好。而且，因为即使是很小的改进也是复合的，习惯性地检查它们会给我们很大的动力。唉，对于第一个问题。*

# 我学到了什么？

当克里斯·洛克测试他的新笑话时，经常看到他拿着一个记事本，潦草地记下观众的反应。Rock 的仔细分析让他看到了自己需要修改的地方。同样，我们问*我学到了什么？*首先是因为答案构成了后续步骤的基础。我们正在寻找一个诚实的评估我们所有的经验和行动，作为该项目的一部分。为了更具体，让我参考[上一篇文章](/how-to-not-fail-a-machine-learning-project-bc35a473ee1e)中的一个例子。在一个 [TFRecord 数据集](/a-practical-guide-to-tfrecords-584536bc786c)的基础上实现了一个增强例程之后，我的模型的性能有了很大的提高，正如损失和准确性分数所衡量的那样。满足于这种提升，我没有进一步考虑输入管道，而是将注意力集中在其他特性上。后来，我注意到增强步骤引入了一个严重的瓶颈，将每个时期的时间从 1 分钟增加到了惊人的 15 分钟

让我们一起来分析一下这个事件，看看我们能从中学到什么:

1.  对模型预测能力的单一关注掩盖了严重的瓶颈。因此，我们得到的教训是，我们不仅要监控主要的性能指标(损失、准确性等)。)还要检查次要指标(管道吞吐量、硬件利用率等)。).
2.  我用的增强包明确说明了所有操作都是在 CPU 上执行的。虽然我记得读过这篇文章，但我认为我没有考虑到这个事实的影响。因此，第二个教训是，我们不应该盲目地添加酷的东西。
3.  我们还应该关注我们在执行增强程序时所学到的东西。在这种情况下，[从 TFRecord 格式](/a-practical-guide-to-tfrecords-584536bc786c)中读取数据，并通过定制的预处理管道传递它。因此，第三个经验是如何从这种特定格式读取数据并对其进行预处理。

从这个事件中，我们至少可以学到三点:检查二级指标，添加新特性时要小心，以及预处理来自 TFRecords 的数据。用我们所有的经验重复这种细致的分析，我们会为下一个项目找到许多有用的指示。

然后，在编制了一份(冗长的)清单后，我们可以回答第二个问题。

# 下次应该避免什么？

在记录了我们在前面步骤中的失误和正确决策后，我们将列表分为两类。第一类，*避免*，包含我们下次应该避免的所有事件/教训。

从这个例子中，我们已经可以将两个项目归入这个类别。第一点是，我们不应该盲目地关注单个指标的值。第二点是，我们不应该仅仅因为我们认为功能很酷就添加它们。

为了澄清这一类别，让我再举一个例子[，来自之前的帖子](/how-to-not-fail-a-machine-learning-project-bc35a473ee1e)。对于神经网络，我们可以优化无数的参数。学习率就是其中之一。因为我试着变聪明，我认为让学习率在整个训练过程中变化会很聪明。在搜索相关研究后，我找到了莱斯利·史密斯关于循环学习率的论文。这个很酷的特性让我很兴奋，我马上开始在培训中实现它。然而，经过几天的编码和参数优化，我意识到我并不真正需要这个项目的这个特性。此外，让学习率变化引入了额外的超参数，增加了项目的复杂性。最后，我完全放弃了调度策略。

通过分析这一事件，我们可以重新发现以前的教训:不要盲目地添加酷的东西。这是很自然的事情；许多事件可以提供类似的教训。除此之外，我们还可以找到第二个相关的提示:不要在错误的时间和地点迷失在细节中。我小心翼翼地调整了学习速度，并重构了示例中的大部分底层代码。不幸的是，由于所有的劳动都是徒劳的，投入的时间都花在了错误的地方(次要特征)，最终都浪费了。因此，这一课符合*避免*的类别。

在整个项目中，我们可能做了一些对整体进度没有好处的事情。然而，通过把它们都归入*避免*类别，我们为我们未来的职业生涯建立了一个强大的资源。更好的是，在下一步中，我们编制了一个推荐行动的列表，作为指导方针。

# 下一次我应该做什么？

对于喜剧演员克里斯·洛克来说，在挑剔的观众面前尝试他的笑话也有积极的一面，告诉他应该重复哪一个。他可能会认为他们是他的一些草图的绝对烟花，只看到他们被忽视。但是，他的一些看似二流的笑话让观众捧腹大笑；他可以把这些笑话蚀刻出来。

在我们的案例中，对项目的仔细分析可以让我们找到相似的见解。因此，通过回答前一个问题填充的*避免*类别有一个兄弟，即*重复*类别。此类别包含所有对项目进展有积极贡献或可用作未来项目蓝图的经验和行动。从给出的两个例子中，我们也可以得出积极的指示。

一个是，在 TFRecord 管道上实现了增强技术之后，我们可以重用我们的知识，并在下一次构建类似的设置。换句话说，我们可以写下:使用 TFRecord 格式存储数据，并在其上构建一个管道。第二个指示是将我们的注意力、精力和时间——这些都是非常重要的资源——集中在核心特征上，就像学习率的例子告诉我们的那样。只有在建立了一个坚实的基础之后，我们才能进入细节。

和以前一样,*重复*类别给了我们一个方便的想法列表，我们可以遵循它来提高我们在项目中的表现。

# TL；速度三角形定位法(dead reckoning)

完成一个项目后，分析它并收集所有的经验教训。把它们分成避免下一次的和重复的。

作为结束语，我想强调的是，我们已经可以在项目期间实施这个拟议的问答计划，而不仅仅是在项目结束之后。通过这样做，我们可以在旅途中优化我们的性能，避免重复错误，将它们转化为可操作的指令。