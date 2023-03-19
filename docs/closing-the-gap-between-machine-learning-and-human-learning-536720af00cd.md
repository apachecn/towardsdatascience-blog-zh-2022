# 缩小机器学习和人类学习之间的差距

> 原文：<https://towardsdatascience.com/closing-the-gap-between-machine-learning-and-human-learning-536720af00cd>

## 大型语言建模的进展

![](img/64bdebd12827519910a8e4bfb2235be7.png)

由 [Lukas](https://unsplash.com/@hauntedeyes?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄的照片

人类拥有强大的推理能力。他们能理解他人提出的问题，并给出最恰当的答案。人脑可以通过快速的数学运算来回答一个微不足道的问题，比如“如果我有 10 个球，买了两个罐头，每个罐头有 5 个球，我会有多少个球？”人类可以进行常识性的推理，比如“如果一个司机在交叉路口看到一个行人，他会怎么做？”人类有智能理解某人是否在开玩笑，并可能更深刻地理解说话者真正想说什么？

问题是，我们能训练机器获得我们人类拥有的这种智能吗？近年来，科学家们在这一领域进行了大量的研究。随着深度神经网络(DNN)和大型语言模型(LLM)的发明，我们在实现这一目标方面取得了良好的进展。在本文中，我将向您介绍通过使用 LLM 和 Google 最新的 PaLM[ ] (Pathways 语言模型)所取得的成就。

首先，让我们考虑一下我们正在努力完成的任务。

# NLU 任务

在大型神经网络在自然语言处理(NLP)中取得巨大成功之后，研究人员将注意力集中在语言理解(NLU)和生成上，而不是简单的文本处理任务。那么，我们试图用这些巨大的网络来解决什么问题呢？我在下面给出了我们寻求解决的 NLU 任务的简短列表。尽管下面的列表并不详尽，但它会让您对我们的目标有所了解。

*   语言翻译
*   聊天机器人(问答)
*   文本摘要
*   语言生成
*   论证

# 语言翻译

从英语翻译到德语或反之亦然，更确切地说，从任何语言翻译到任何语言，一直是我们的需要。今天，有几个 ML 模型甚至移动应用程序使用这样的预训练模型来以非常高的准确度完成这项任务。

# 聊天机器人(问答)

用自动化系统取代庞大的客户服务代表队伍一直是企业的梦想。这一点现在可以通过近乎完美的聊天机器人来实现。聊天机器人需要自然语言理解和问答能力。尽管特定领域的问答系统已经非常完善，但是开发一个面向开放领域的问答系统仍然是一个挑战。人类很快理解问题的上下文(领域)来回答问题。这就需要我们所知的 LLM 的少量学习[ ]。GPT-3[ ]是第一个应用少镜头学习的。最近的法学硕士如 GLaM[⁴]，LaMDA[⁵]，Gopher[⁶]和威震天-图灵·nlg[⁷]都采用了少击学习。

# 文本摘要

很多时候，我们需要创建一个长文档的摘要。虽然这是一项 NLP 任务，但语言理解在创建有意义的摘要时也起着有效的作用。具有*注意力*的编码器-解码器架构和基于*转换器*的架构在创建抽象和提取摘要方面都表现出了突出的成功。

# 语言生成

像莎士比亚那样写作是许多人的梦想。从 RNN(循环神经网络)、LSTM(长短期记忆)和最新的 Transformer 开始的神经网络架构允许创作可以模仿阿加莎·克里斯蒂和许多著名作家作品的小说。有许多工具可用，如 Arria NLG PLC[⁸]，AX Semantics[⁹]，Yseop[ ⁰]，Wordsmith[ ]，SimpleNLG[ ]，NaturalOWL[ ]和其他自然语言生成(NLG)[ ⁴].

# 论证

人类有很强的常识推理能力、概念理解能力、玩琐事、同义词游戏的能力，以及根据反事实做出反应的能力。

这些仅仅是 NLP/NLU 研究进展强劲的几个领域。上述目标可以通过创建大型语言模型来实现。

# 创建 LLM

创建 LLM 的主要挑战是训练一个具有数百万和数十亿参数的超大型深度神经网络。像 GLaM 和 LaMDA 这样的模型是在一个单独的 TPU v3 吊舱上训练的。威震天-图灵 NLG 使用流水线并行在 GPU 集群中扩展到 2240 - A100 个 GPU。使用多个 TPU v3 pod 的 Gopher 实现了 4096 个 TPU v3 芯片的规模。他们观察到，具有更多训练参数的更大模型改善了 NLG 结果。PaLM 是这一类别中最新的一个，它使用谷歌的*路径*系统将训练扩展到 6144 个芯片，并创建了 5400 亿参数语言模型。它实现了 57.8%的硬件 FLOPs 利用率的训练效率，这是迄今为止 LLM 实现的最高效率。Google 重新设计了 *Transformer* 模块，允许并行计算*注意力*和*前馈*层。这有助于为训练网络创建更好的并行策略。

<https://medium.com/@profsarang/membership>  

# 手掌训练

他们在英语和多语言数据集的组合上训练 PaLM。其中包括维基百科文章、书籍、网络文档、对话甚至 GitHub 代码。注 PaLM 也能写计算机代码，那大概是因为它在 GitHub 上的训练。在代码生成中，空格很重要。因此，培训师创造了一个保留空白的“无损”词汇。他们还负责处理词汇表之外的 Unicode 字符，并将大数字拆分成单个数字。所有这些都有助于有效的代码生成。

现在，我将讨论 PaLM 的一些成就。

# 棕榈成就

研究人员在 29 个广泛使用的 NLP 任务上评估了 PaLM。在这 29 个任务中的 28 个上，它超过了先前语言模型的少数表现。它还在多语言 NLP 基准测试中显示了强大的性能，包括翻译。BIG[ ⁵](超越模仿游戏)是最新的基准，包含了 150 多个新的语言建模任务。与地鼠、龙猫和人类相比，PaLM 在 58/150 任务的子集上表现更好。它有效地区分了原因和结果。对于特定的上下文，它会为您的问题提供合适的答案。它可以玩同义词游戏。它可以从一篇文章中推导出相反的事实。

在我之前提出的问题中——“如果我有 10 个球，买了两个罐头，每个罐头有 5 个球，我会有多少个球？”—单纯的答案可能不容易以其准确性说服读者。你需要给出多步算术，推理出结论性的答案是如何推导出来的。使用思维链提示，PaLM 将通过为所有中间步骤生成文本来推理出答案。Google 博客[1]提供了一些例子来展示这些能力。

PaLM 可以生成一个明确的解释，甚至是一个笑话。生成这样的解释需要多步逻辑推理、世界知识和深刻的语言理解。博客[ ]中提供了一个很好的例子来说明这一点。

除了自然语言文本生成，代码生成是我们期望 LLMs 执行的另一个重要任务。代码生成可能意味着文本到代码，从一种编程语言翻译到另一种，甚至修复编译错误(代码到代码)。PaLM 训练数据集包括编码示例，尽管只有 5%。它已经显示出与诸如 Codex 12B[ ⁶].我将再次向您推荐 Google 博客[ ]中的优秀示例。

# 结论

在查看了 LLM 的最新发展，尤其是 PaLM 之后，人们可以观察到机器在学习自然语言和我们人类之间的差距正在迅速缩小。最近，Meta AI 也向研究社区提供了它的 OPT-175[ ⁷]十亿参数模型。我们可以希望看到机器学习和人类能力之间的差距很快缩小。

# 参考资料:

1.  [通路语言模型(PaLM)](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html)
2.  [少投学习](https://arxiv.org/abs/2005.14165)
3.  [GPT-3](https://arxiv.org/abs/2005.14165)
4.  [多面手语言模型(GLaM)](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html)
5.  [LaMDA](https://ai.googleblog.com/2022/01/lamda-towards-safe-grounded-and-high.html)
6.  [地鼠](https://arxiv.org/abs/2112.11446)
7.  [威震天-图灵 NLG](https://arxiv.org/abs/2201.11990)
8.  [阿里亚 NLG 公司](https://www.arria.com/)
9.  [AX 语义](https://cloud.ax-semantics.com/)
10.  [Yseop](https://yseop.com/)
11.  语言大师
12.  简单明了
13.  [自然猫头鹰](https://sourceforge.net/projects/naturalowl/)
14.  [自然语言生成(NLG)](https://medium.com/sciforce/a-comprehensive-guide-to-natural-language-generation-dd63a4b6e548)
15.  [大板凳](https://github.com/google/BIG-bench)
16.  [抄本 12B](https://arxiv.org/abs/2107.03374.pdf)
17.  [OPT-175](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/)