# 何时使用量子计算机

> 原文：<https://towardsdatascience.com/when-to-use-a-quantum-computer-830367322b8f>

## 量子计算很奇怪

量子机器学习要不要入门？看看 [**动手量子机器学习用 Python**](https://www.pyqml.com/page?ref=medium_when&dest=/) **。**

不，我不是指控制量子计算机运行的量子力学的奇怪现象。

我指的是媒体报道中的差异。在大众媒体中，量子计算被吹捧为一种神奇的设备，可以解决我们所有的问题。相比之下，批评性文献怀疑量子计算是否会有任何用处。

如果我们相信媒体所写的关于量子计算的一切，这就像患有人格障碍，你在快乐和毁灭之间摇摆不定。

![](img/1d579ade89566a08197c58e759131323.png)

作者图片

我也有罪。我已经在我的每周帖子里提出了这两个问题，我已经写了将近两年了。我强调了量子计算机能做的所有不可思议的事情。但我也澄清了由于对量子计算机的严重误解而导致的过高期望。

虽然我试图通过不仅仅依赖数学公式来使整个主题尽可能容易理解，但我尽最大努力保持尽可能的脚踏实地。

因此，在我之前的帖子中，我指出了量子计算机的作用:

> 量子计算所做的只是让实现非确定性算法成为可能。

当然，如果没有进一步的澄清和解释，这个定义是没有用的。

所以让我们看看这意味着什么和它的含义，因为这是识别我们应该用量子计算机解决的问题的关键。

但是在我们进入这个话题之前，让我简单解释一下为什么我强调“应该”而不写“可以”。原因很简单，量子计算机可以做数字计算机能做的一切。但是量子计算机是昂贵的设备，它的少数量子位容易受到噪声的影响。

IBM 目前的 QPU(量子处理单元)Eagle 有 127 个量子位。所以它能做 127 位 CPU 能做的一切。因此，如果我们有数百万(1 兆量子位)或数十亿(1 千兆量子位)的量子位，量子计算机可以做目前 CPU 能做的一切。

但是，即使是拥有数十亿量子位的量子计算机，也比同等的经典芯片速度更慢、更容易出错、更昂贵。所以我肯定会坚持使用数字电脑…除非有使用量子计算机的理由。

问题不在于我们是否可以使用量子计算机。相反，问题是我们是否应该这样做。那么，我们为什么还要为微小的处理器、嘈杂的结果和成本而烦恼呢？

> 我们应该使用量子计算机来解决非确定性多项式(NP)问题。根据定义，这些是非确定性算法可以在多项式时间内解决的决策问题，而确定性算法则不能。这听起来像一个同义反复，但它有意义。

为了理解这意味着什么，我们需要谈谈复杂性理论。这是一个关于解决一个问题所需的时间或步骤数量如何随着问题规模的增加而增加的理论。

我们来考虑两个问题。首先，两个数相乘，例如，“三乘以七等于几？”第二，两个数的因式分解，例如，“21 的质因数是多少？”这两个问题都很容易解决。原因是问题规模很小。

这些问题怎么样:

*   101 乘以 223 是多少？
*   22523 的质因数是什么？

虽然你可以用纸和笔解决第一个问题，但你可能很难解决第二个问题。对了，这些也还是小问题。

n 位数相乘的复杂度增加了大约 n2 倍。我们称之为多项式问题解。相比之下，寻找一个 n 位数的质因数的复杂性是 e^{n^{1/3}}.这意味着努力随着位数的增加而呈指数增长。

不要低估多项式复杂度和指数复杂度之间的差异，这一点很重要。虽然你的智能手机在几秒钟内将 800 位数字相乘，但在超级计算机上对这些数字进行因式分解需要大约 2000 年。

我们认为可以在多项式时间内解决的问题是易处理的，而指数增长的问题是难处理的。对于 n 的大值，具有指数增长复杂性的问题变得如此困难，以至于即使是最好的超级计算机也要花费太长时间来解决它们——在某些情况下，这意味着数百万甚至数十亿年。

以上任务你都费心解决了吗？如果我告诉你 22523 的质因数是 101 和 223 呢？

这个解是否正确可以在多项式时间内检查，因为检查是乘法。这使得数的因式分解成为一个不确定的多项式决策问题。

决策问题是一个我们可以回答是或不是的问题。此外，这个答案必须通过简洁的证明来验证。如果一个证明的复杂性呈多项式增长，那么它就是简洁的。

此外，如果我们可以猜测它的解决方案，一个问题被称为是一个非确定性多项式(NP)问题。这意味着我们没有特定的规则可以遵循来推断解决方案，但如果我们幸运的话，我们仍然可以在第一次尝试中找到正确的解决方案。

因此，如果一个非确定性算法可以在多项式时间内解决一个 NP 问题，这意味着它使这样一个问题易于处理。

量子计算的力量源于它实现非确定性算法的能力。因此，我们可以为这种 NP 问题创建算法。

这正是我们**应该**使用量子计算机的时候。我们**应该**在面对经典计算机无法有效解决的 NP 决策问题(即多项式复杂性)时使用量子计算机。

以乘法为例。这是一个我们已经可以有效解决的决策问题。因此，我们**不应该**使用量子计算机，即使我们可以。

另一方面，因式分解是一个我们无法有效解决的决策问题。因此，我们应该为它使用一台量子计算机。

优化呢？假设您想要在一系列城市之间找到最短的可能路线。这是一个我们还不能有效解决的问题。然而，这不是一个决策问题，因为即使有人告诉你，你也无法证实一个解决方案是最好的。在考虑所有可能的路线之前，你永远不会知道是否有更好的解决方案。

但这并不意味着量子计算不能帮助优化。例如，它可以用来首先找到可能的路线。当然，有时候寻找有效路线本身就是一个 NP 问题。但是一旦我们有了一条路线，我们可以快速检查它是否正确，即使我们无法判断它是否是最优的。

在这种情况下，我们应该使用量子计算机来寻找路径。然后，传统的优化器可以选择最佳路线。这就是我们所知的[变分量子经典算法](/anatomy-of-a-quantum-machine-learning-algorithm-24d97dfd388d)。

显然，你需要在问题领域有深厚的专业知识来判断你是否应该使用量子计算机来解决它。

使用这些知识来识别你的领域中应该用量子计算机解决的问题。它会把你带到杆位。

量子机器学习要不要入门？看看 [**动手用 Python**](https://www.pyqml.com/page?ref=medium_when&dest=/) **学习量子机器。**

![](img/c3892c668b9d47f57e47f1e6d80af7b6.png)

在这里免费获得前三章。