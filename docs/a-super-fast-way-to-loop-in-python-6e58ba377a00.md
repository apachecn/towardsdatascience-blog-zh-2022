# 一种超快速的 Python 循环方式

> 原文：<https://towardsdatascience.com/a-super-fast-way-to-loop-in-python-6e58ba377a00>

## 你觉得 Python 慢吗？这里有一个在 Python 中快速循环的方法

![](img/9c6df219f5883acd436e8d2eefb3f609.png)

图片来自 Shutterstock，授权给 Frank Andrade

众所周知，Python 是一种缓慢的编程语言。虽然 Python 比其他语言慢是事实，但是有一些方法可以加速我们的 Python 代码。

怎么会？简单，优化你的代码。

如果我们编写消耗很少内存和存储的代码，我们不仅可以完成工作，还可以让我们的 Python 代码运行得更快。

这里有一个快速也是超快速的 Python 循环方法，这是我在[上的一门 Python 课程](https://medium.com/p/38c7866f5cb5)中学到的(我们从未停止学习！).

# 平均循环

假设我们想对从 1 到 100000000 的数字求和(我们可能永远不会这样做，但这个大数字将有助于我阐明我的观点)。

典型的方法是创建一个变量`total_sum=0`，在一个范围内循环，并在每次迭代中用`i`增加`total_sum`的值。

这就完成了工作，但是需要大约 6.58 秒。

虽然现在看起来没那么慢，但是当你在范围内的数字上加更多的 0 时，它会变得更慢。

让我们加速吧！

# 使用内置函数进行循环的更快方法

在 Python 中循环的更快方法是使用内置函数。

在我们的例子中，我们可以用`sum`函数替换 for 循环。此函数将对数字范围内的值求和。

上面的代码耗时 0.84 秒。这比我们之前使用的循环快多了！这就是为什么我们应该选择内置函数而不是循环。

但仍有改进的空间。

# 使用 Numpy 的超快速循环方式

几周前，在我参加的[数据科学课程中，我了解到要成为一名更好的数据科学家，我应该遵循的软件工程实践之一就是优化我的代码。](https://medium.com/p/ca54b6619e68)

我们可以通过向量化操作来优化循环。这比它们的纯 Python 对等物快一/两个数量级(特别是在数值计算中)。

矢量化是我们可以通过 NumPy 得到的东西。Numpy 是一个具有高效数据结构的库，用于保存矩阵数据。它主要是用 C 写的，所以速度是你可以信赖的。

让我们尝试使用 Numpy 方法`.sum`和`.arange`来代替 Python 函数。

这可以在 0.22 秒内完成工作。这比以前的方法要快得多。

这就是为什么应该尽可能在循环中使用向量运算的原因。

# 用更多的计算来测试循环和数字

到目前为止，我们已经看到了 Numpy 的一个简单应用，但是如果我们不仅有一个 for 循环，还有一个 if 条件和更多的计算要做呢？

Numpy 的表现明显优于 loops。

假设我们有一组随机的考试分数(从 1 到 100)，我们想要获得那些没有通过考试的人的平均分数(score <70).

Here’s how we’d do this with a for loop.

That takes approximately 15.7 seconds. Not bad, but we can get faster results with Numpy.

Here’s how we’d do this with Numpy.

The code above takes about 0.78 seconds. That’s way faster and the code is straightforward!

Learning Data Science with Python? [**)通过加入我的 10k+人的电子邮件列表来获得我的免费 Python for Data Science 备忘单。**](https://frankandrade.ck.page/26b76e9130)

如果你喜欢阅读这样的故事，并想支持我成为一名作家，可以考虑报名成为一名媒体成员。每月 5 美元，让您可以无限制地访问数以千计的 Python 指南和数据科学文章。如果你用[我的链接](https://frank-andrade.medium.com/membership)注册，我会赚一小笔佣金，不需要你额外付费。

[](https://frank-andrade.medium.com/membership) 