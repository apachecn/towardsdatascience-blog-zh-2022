# 长短期记忆网络导论(LSTM)

> 原文：<https://towardsdatascience.com/an-introduction-to-long-short-term-memory-networks-lstm-27af36dde85d>

## 理解长短期记忆的概念和问题

![](img/fac647f8d0ce6e3718d159202ff929a2.png)

照片由 [Soragrit Wongsa](https://unsplash.com/@invictar1997?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

长短期记忆(简称:LSTM)模型是[循环神经网络](https://databasecamp.de/en/ml/recurrent-neural-network) (RNN)的一个亚型。它用于识别数据序列中的模式，例如出现在传感器数据、股票价格或自然语言中的模式。rnn 之所以能够做到这一点，是因为除了实际值之外，它们还在预测中包括其在序列中的位置。

# 什么是递归神经网络？

为了理解递归神经网络如何工作，我们必须再看一下常规的[前馈神经网络](https://databasecamp.de/en/ml/artificial-neural-networks)是如何构造的。其中，隐藏层的神经元与前一层的神经元和后一层的神经元相连。在这样的网络中，一个神经元的输出只能向前传递，而绝不会传递给同一层甚至上一层的神经元，因此得名“前馈”。

这对于递归神经网络是不同的。神经元的输出可以很好地用作前一层或当前层的输入。这比前馈神经网络的构建方式更接近我们大脑的工作方式。在许多应用中，我们还需要在改善整体结果之前立即了解计算的步骤。

# RNNs 面临哪些问题？

递归神经网络是[深度学习](https://databasecamp.de/en/ml/deep-learning-en)领域的一个真正突破，因为第一次，最近过去的计算也包括在当前计算中，显著提高了语言处理的结果。尽管如此，在训练过程中，他们也带来了一些需要考虑的问题。

正如我们在关于[梯度法](https://databasecamp.de/en/ml/gradient-descent)的文章中已经解释过的，当用梯度法训练神经网络时，梯度可能会呈现非常小的接近 0 的值或者非常大的接近无穷大的值。在这两种情况下，我们都不能在[反向传播](https://databasecamp.de/en/ml/backpropagation-basics)过程中改变神经元的权重，因为权重要么根本不变，要么我们不能用这么大的值乘以数字。由于递归神经网络中的许多互连以及用于它的反向传播算法的稍微修改的形式，这些问题发生的概率比正常的前馈网络高得多。

常规 rnn 非常擅长记忆上下文，并将其纳入预测。例如，这允许 RNN 认识到在句子“clouds are at the _ _”中，需要单词“sky”来在该上下文中正确完成句子。另一方面，在一个较长的句子中，保持上下文变得困难得多。在稍加修改的句子“部分流入彼此并低悬的云在 __”中，一个递归神经网络推断“天空”这个词变得困难得多。

# 长短期记忆模型是如何工作的？

递归神经网络的问题在于，它们有一个短期记忆来保留当前神经元中以前的信息。然而，对于较长的序列，这种能力下降得非常快。作为对这一点的补救，LSTM 模型被引入，以便能够更长时间地保留过去的信息。

递归神经网络的问题在于，它们只是将之前的数据存储在它们的“短期记忆”中。一旦其中的内存耗尽，它就简单地删除保留时间最长的信息，并用新数据替换它。LSTM 模型试图通过在短期记忆中只保留选定的信息来避免这个问题。

为此，LSTM 架构总共包括三个不同的阶段:

1.  在所谓的**遗忘门**中，决定哪些当前和先前的信息被保留，哪些被丢弃。这包括上次运行的隐藏状态和当前状态。这些值被传递到一个 sigmoid 函数中，该函数只能输出 0 到 1 之间的值。值 0 意味着所有以前的信息都被遗忘，1 意味着所有以前的信息都被保留。
2.  在**输入门**中，决定当前输入对解决任务有多大价值。为此，当前输入乘以隐藏状态和上次运行的权重矩阵。
3.  在**输出门**中，计算 LSTM 模型的输出。取决于应用，例如，它可以是补充句子意思的单词。

# 这是你应该带走的东西

*   LSTM 模型是递归神经网络的一个亚型。
*   它们用于识别数据序列中的模式，例如出现在传感器数据、股票价格或自然语言中的模式。
*   一种特殊的结构允许 LSTM 模型决定是在短期记忆中保留先前的信息还是丢弃它。因此，序列中更长的依赖性也被识别。

*如果你喜欢我的作品，请在这里订阅*<https://medium.com/subscribe/@niklas_lang>**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，medium 允许你每月免费阅读* ***3 篇*** *。如果你想让***无限制地访问我的文章和数以千计的精彩文章，不要犹豫，通过点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格***

**</comprehensive-guide-to-principal-component-analysis-bb4458fff9e2>  </why-you-should-know-big-data-3c0c161b9e14>  </what-are-deepfakes-and-how-do-you-recognize-them-f9ab1a143456> **