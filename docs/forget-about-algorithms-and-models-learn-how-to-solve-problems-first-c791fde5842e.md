# 忘记算法和模型——首先学习如何解决问题

> 原文：<https://towardsdatascience.com/forget-about-algorithms-and-models-learn-how-to-solve-problems-first-c791fde5842e>

## 意见

## 有抱负的开发人员和数据科学家经常本末倒置

![](img/09cc53f2cb7f3c52530193c5b5385351.png)

解决问题是真正的脑力劳动。作者图片

![A](img/905c9dac3e604fd9ae883e6c79475524.png)  A 几乎每周都有朋友或熟人问我，*我想学编码；我应该从哪种语言开始？*差不多两周一次，我在 LinkedIn 上收到一条 DM，开头是*我儿子应该开始编程了；对他来说最好的语言是什么？*

不仅仅是以前没编过代码的人。我经常从有几年编码经验的人那里得到这些信息。

我说这些不是为了抱怨。

我在 Medium 上研究不同编程语言、框架和人工智能模型的优缺点，以此谋生。人们提出这样的问题让我受益匪浅。

问题很直观。毕竟，每个人都希望用最好的工具工作，并尽可能快地建立他们的软件技能。

当你观察到每个开发人员似乎使用不同的技术堆栈时，想知道哪一个是正确的是完全有意义的。

问题是，这完全取决于手头的问题。

没有技术本身是好的或坏的；这取决于你想解决什么类型的问题。归根结底，编程就是:通过使用计算机来解决问题。

所以，对于想开始编程或者想提升软件开发或数据科学技能的人来说，问题不应该是*我该用什么，* [*Python 还是 Julia*](/bye-bye-python-hello-julia-9230bff0df62) *？*问题应该是*我怎样才能更好地解决软件问题？*

# 如何解决问题

完全公开，我的职业不是计算机科学家。我是一名粒子物理学家，碰巧使用编程和数据科学的概念，因为我处理来自粒子对撞机的海量数据。

也就是说，物理学家和计算机科学家一样受欢迎。这不是因为他们对中微子或黑洞的了解；是因为他们解决问题的能力。

《T21》引用亚伯拉罕·林肯的话说，“给我六个小时去砍树，我会用前四个小时去磨斧子。”。

对于程序员和数据科学家来说，这意味着在开始编码之前，要花时间理解问题并找到高层次的解决方案。在一般的编码面试中，候选人预计花在实际编写代码上的时间不到一半，其余时间用于理解问题。

## 1-理解问题

永远不要跳过这一步！

知道你是否理解一个问题的关键是你是否能向不熟悉它的人解释它。尽量用通俗易懂的英语或者母语写下来；画一个小图；或者告诉一个朋友。如果你的朋友不明白你在说什么，你需要回到问题陈述。

要问的关键问题是:

*   **什么是输入？期望的输出是什么？**
    例如，输入可能是一组数据，输出可能是对数据的线性回归。
*   **问题背后的假设是什么？**
    例如，你可能假设你的数据中(几乎)没有测量误差。
*   是什么让这个问题变得复杂？
    例如，您拥有的数据可能不完整，或者数据集可能太小，无法得出明确的结论。

## 2-分解问题

每个大问题都由许多小问题组成。鉴于我们之前的线性回归示例，您可能需要考虑以下子问题:

*   清理数据
*   找出数据中哪些变量对回归有意义，哪些变量可以安全地忽略
*   寻找合适的工具来进行回归(这就是关于编程语言和框架的老问题发挥作用的地方)
*   评估您的结果并检查错误

把问题分解有助于你为工作制定一个合适的计划。

这也更有激励性，因为你会在前进的道路上实现小而重要的里程碑。这比坐在堆积如山的工作面前感觉自己没有前进要令人满意得多。

## 3-从一个例子开始

魔鬼总是在细节中。

不要从整个项目开始，取其中的一小部分。试试你的计划是否可行，或者你是否因为不可预见的困难而不得不修改它。

这有助于你理解困难的部分。许多问题听起来很简单，但是当你开始构建它们时，就会遇到一个接一个的障碍。

在我们的例子中，我们可以先对几个变量进行线性回归，而不是使用所有相关变量。这不会给你任何项目完成的分数；然而，当您仍在处理少量数据时，发现脚本中的错误可以挽救生命。

当你把所有的数据都扔进机器，运行几个小时，然后*然后*回来意识到脚本中途挂起，你会非常沮丧。

相信我，这种事经常发生！

首先运行小测试，并确保您的解决方案如您所设想的那样工作。

## 4-执行

这是肉多的部分。现在，您可以为您的大问题构建解决方案了。

把你所有的数据都扔给代码。运行一个奇特的模型。想做什么就做什么。

完成前面的三个步骤后，这应该会非常顺利！

如果有错误，您可能必须返回到步骤 1-3，看看您是否已经理解了所有内容，并且没有忽略任何错误。

## 5-反射

仅仅因为你找到了一个解决方案，并不意味着你找到了 T2 最好的解决方案。不要跑掉，收工；思考如何优化您的解决方案，以及如何以不同的方式实现它。

你可能想和你的同事交流，问他们如何解决这个问题。他们的方法和你的不同吗？

您还可以尝试确定解决方案中最大的瓶颈，即执行时花费最多时间和资源的部分。你如何改进它们？

最后，思考您的解决方案在未来可能会如何发展。新的软件框架或人工智能的使用会让你的解决方案更好吗？你的解决方案如何有助于解决其他更复杂的问题？

# 吹牛

包括我自己在内的人们，倾向于沉迷于不同的编程语言和最新的框架，这些语言和框架可能会使所有事情的效率提高 1000 倍。

值得提醒自己的是，这还不到成为一名优秀程序员所需的一半。另一半是解决问题。

你不会在一夜之间获得解决问题的技能。

但是如果你采用这些步骤，问正确的问题，并且经常这样做，你就走在了让你的职业生涯从优秀走向卓越的正确道路上。

*成为* [*中等会员*](https://arijoury.medium.com/membership) *可完全访问我的内容。*