# 使用 SciPy 和 NumPy 的随机抽样:第二部分

> 原文：<https://towardsdatascience.com/random-sampling-with-scipy-and-numpy-part-ii-234c2385828a>

![](img/f41cd3a6597b4aa46ff2aecf94bf207d.png)

照片由[андрейсизов](https://unsplash.com/@alpridephoto?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/algorithm?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

## 奇特的算法，源代码演练和潜在的改进

[在第一部分](/random-sampling-using-scipy-and-numpy-part-i-f3ce8c78812e)中，我们讲述了逆变换采样(ITS)的基础知识，并创建了我们自己的 ITS 纯 python 实现来从标准正态分布中采样数字。然后，我们比较了我们稍微优化的功能和内置的 SciPy 功能的速度，发现我们有些欠缺——慢了`40x`倍。

在这一部分中，我们的目的是通过挖掘 SciPy 和 NumPy 代码库的相关部分来解释为什么会出现这种情况，看看这些速度改进在哪里表现出来。总的来说，我们会发现它由以下几个部分组成:

*   由于是用 Cython 或直接用 C 编写的，所以函数速度更快
*   与我们屡试不爽的逆变换采样相比，更新的采样算法速度更快

## 我们如何在 SciPy 中生成正态分布的随机样本？

下面是从标准正态分布生成随机数`1,000,000`的代码。

```
43.5 ms ± 1.2 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

所以函数`rvs`在刚刚超过`40ms`的时间内生成`1,000,000`个样本。为了进行比较，我们使用基于逆变换采样原理的算法在`2.3s`中平均实现了这一点。为了理解速度差异，我们将不得不深入到那个`rvs`方法中。

值得注意的是，*(一般而言)*使用 SciPy，逻辑的核心包含在下划线方法中——所以当我们想要查看`rvs`时，我们真的想要查看`_rvs`的代码。非下划线方法通常在传递给下划线方法之前执行一些参数类型检查或默认设置。

在我们开始之前，让我们先简要概述一下 SciPy 在库中组织分发功能的方式。

## RV _ 一般和 RV _ 连续

SciPy 发行版是从一个简洁的继承结构中创建的，它具有:

*   `rv_generic`作为顶层类，提供`get_support`和`mean`等方法
*   `rv_continuous`和`rv_discrete`用更具体的方法继承它

所以在上面的例子中，我们将我们的正态分布类`snorm`初始化为`stats.norm()`，这实际上是创建了一个`rv_continuous`的实例，它继承了`rv_generic`的很多功能。更具体地说，我们实际上创建了一个`rv_frozen`实例，它是`rv_continuous`的一个版本，但是分布的参数是固定的(例如，平均值和方差)。记住这一点，现在让我们看看`rvs`方法的内部。

## 房车

当我们在`snorm.dist._rvs`上运行`??`魔术时，我们看到下面的代码片段:

```
# ?? snorm.dist._rvs
def _rvs(self, size=None, random_state=None):
    return random_state.standard_normal(size)
```

因此，似乎在我们创建的 distribution 类中的某个地方，我们已经在某个地方分配了一个`random_state`对象，该`random_state`对象包含一个方法，该方法可以返回根据标准正态分布分布的数字。

**原来，吐出这些随机数的** `**random_state**` **物体其实来自 NumPy。**我们通过查看 [rv_generic](https://github.com/scipy/scipy/blob/b5d8bab88af61d61de09641243848df63380a67f/scipy/stats/_distn_infrastructure.py#L627) 的源代码来了解这一点，它在其`__init__`方法中包含对一个名为 [check_random_state](https://github.com/scipy/scipy/blob/e3cd846ef353b10cc66972a5c7718e80948362ac/scipy/_lib/_util.py#L209) 的 SciPy util 方法的调用，如果还没有传递种子，该方法将把`random_state`设置为`np.random.RandomState`的实例。下面是这段代码:

## 交给 NumPy

因此，似乎提供如此高速采样的“魔法”实际上存在于 NumPy 而非 SciPy 中。这不应该太令人震惊，因为 SciPy 是故意构建在 NumPy 之上的，以防止两个库可能提供相同特性的重复和不一致。这在 SciPy 简介文档的第一行[中有明确说明，这里是](https://docs.scipy.org/doc/scipy/tutorial/general.html):

SciPy 是建立在 Python 的 NumPy 扩展之上的数学算法和便利函数的集合

要了解这是怎么回事，我们可以看看这里的`np.random.RandomState`类。从使用中我们可以看出:

*   用`cdef`代替`def`进行函数声明
*   一个`.pyx`文件扩展名代替。巴拉圭

这两者都表明该函数是使用 [Cython](https://cython.readthedocs.io/en/latest/index.html) 编写的——这是一种非常类似于 python 的语言，允许以几乎 Python 的语法编写函数，然后编译成优化的 C/C++代码以提高效率。正如他们自己在文档中所说的[:](https://cython.readthedocs.io/en/latest/src/quickstart/overview.html)

*“源代码被翻译成优化的 C/C++代码，并被编译成 Python 扩展模块。这允许非常快速的程序执行和与外部 C 库的紧密集成，同时保持 Python 语言众所周知的高程序员生产率。”*

在本课程中，我们需要了解两件事来理解采样过程:

*   它是如何生成均匀分布的随机数(PRNG)的
*   它使用什么算法将这些均匀分布的数字转换成正态分布的数字

## PRNG

正如在第一部分中提到的，生成随机样本需要某种形式的随机性。几乎总是这不是真正的随机，而是由“伪随机数发生器”(PRNG)产生的一系列数字。正如采样算法一样，有多种 PRNGs 可用，这里使用的具体实现在`np.random.RandomState`的 `[__init__](https://github.com/numpy/numpy/blob/b991d0992a56272531e18613cc26b0ba085459ef/numpy/random/mtrand.pyx#L180)` [方法](https://github.com/numpy/numpy/blob/b991d0992a56272531e18613cc26b0ba085459ef/numpy/random/mtrand.pyx#L180)中详细介绍:

如上所示，当该类被初始化时，默认的 PRNG 被设置为[梅森扭结](https://en.wikipedia.org/wiki/Mersenne_Twister)算法的一个实现——如此命名是因为它的周期长度为[梅森素数](https://en.wikipedia.org/wiki/Mersenne_prime)(它在开始重复自身之前可以生成的随机数的数量)。

## 取样过程

沿着类`np.random.RandomState`的代码往下，我们看到`[standard_normal](https://github.com/numpy/numpy/blob/b991d0992a56272531e18613cc26b0ba085459ef/numpy/random/mtrand.pyx#L1344)`的定义调用了一个叫做`legacy_gauss`的东西。`legacy_gauss`函数的 C 代码在这里是，为了便于查看，我们将在这里显示它:

正如在 Wiki 上的[实现部分](https://en.wikipedia.org/wiki/Marsaglia_polar_method#Implementation)中所看到的，这正是 [Marsaglia 极坐标方法](https://en.wikipedia.org/wiki/Marsaglia_polar_method#Implementation)的 C 实现，用于在给定一串均匀分布的输入数的情况下，从正态分布中生成随机样本。

## 概述

我们已经经历了很多，所以有必要回顾一下，确保一切都非常清楚。我们已经从:

*   一个用 python 写的名为`_rvs`的函数启动
*   一个 NumPy 类`np.random.RandomState`，用 Cython 写的，它
*   使用 Mersenne Twister 算法生成均匀分布的数字，然后
*   将这些数字输入用 C 编写的函数`legacy_gauss`，该函数使用 Marsaglia 极坐标法生成正态分布的样本

以上强调了构建 SciPy 和 NumPy 的聪明人为了生成高效代码所付出的努力。在基础设施的更深层尽可能接近 C 语言(为了速度)之前，我们有一个可以被用户(比如你和我)调用的顶层，它是用 python 编写的(为了 python 的“程序员生产力”)。

## 为什么 SciPy 调用 NumPy 函数被视为“遗留”？

因为采样是数学/计算机科学的一个分支，仍然在向前发展。与其他领域不同，在这些领域中，某些原则在几个世纪前就已达成一致，并且从那以后没有发生变化，有效地对各种分布进行采样仍然有新的发展。随着新的开发得到测试，我们希望更新我们的默认流程，以纳入这些进步。

这正是 2019 年 7 月 NumPy 1.17.0 所发生的事情，当时[他们引入了 2 个影响采样的新功能](https://numpy.org/devdocs/release/1.17.0-notes.html):

*   一种新的默认伪随机数发生器(PRNG)的实现:[麦丽莎·奥尼尔的 PCG 算法家族](https://www.pcg-random.org/index.html)
*   一个新的采样过程的实现:金字形神塔算法

然而，由于对 prng 向后兼容性的期望，他们没有创建突破性的改变，而是引入了一种新的方式来启动 prng，并将旧的方式切换到引用“遗留”代码。

这里提到的向后兼容性是指在给定相同种子的情况下，希望 PRNG 函数生成相同的随机数字符串。两种不同的算法不会产生相同的随机数，即使它们被给予相同的种子。这种再现性对于测试尤其重要。

看来 SciPy 还没有升级来利用这些新的发展。

## 我们能打败西皮吗？

鉴于我们现在所知道的关于在 SciPy 中如何实现正态分布抽样的知识，我们能战胜它吗？

答案是肯定的——通过利用 NumPy 为我们实现的最新采样技术。下面是采样的一个实现，其中我们:

*   使用最新的 PRNG
*   使用新的金字形神塔算法将这些数字转换成正态分布的样本

```
# test scipy speed
%timeit snorm.rvs(size=n)51 ms ± 5.08 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)# test numpy speed
%timeit nnorm.normal(size=n)24.3 ms ± 1.84 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

所以看起来我们现在已经和 SciPy 一样快了 NumPy 在他们的发布中强调了这是预计的 2-10 倍。

## 结论:这个有多大用处？

说到实现定制分布抽样:非常有用。我们现在完全理解了追求 SciPy 式采样速度的决定，并且可以适当地实现定制分布采样。我们可以:

*   坚持使用第一部分中的纯 python 逆采样转换实现(毕竟，在大多数上下文中，`2s`对于`1,000,000`的示例来说并不坏)
*   编写我们自己的采样程序——最好用 C 或 Cython 编写这个采样程序——这不是一个小问题

在下一部分中，我们将研究如何做到这一点——在 SciPy 基础结构中实现一个高效的定制分布采样函数。这给了我们两全其美的东西——既可以灵活地实现我们选择的精确分布，又可以利用我们从`rv_generic`和`rv_continuous` SciPy 类继承的高效且编写良好的方法。