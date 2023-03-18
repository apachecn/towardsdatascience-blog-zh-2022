# pur——保持您的需求文件最新的最简单的方法

> 原文：<https://towardsdatascience.com/pur-the-easiest-way-to-keep-your-requirements-file-up-to-date-22d835279348>

![](img/e1674cb3736f34a1818aac137b81c530.png)

米哈伊尔·瓦西里耶夫在 [Unsplash](https://unsplash.com/s/photos/cat?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

## 用一行代码更新您的`requirements.txt`中的所有库

我不认为我需要让你相信保持你的 Python 库(或者事实上其他软件)最新的好处:随着时间的推移，错误被修复，潜在的安全漏洞被修补，兼容性问题可能出现，等等。这样的例子不胜枚举。

在 Python 项目中，我们经常使用一个`requirements.txt`文件作为在我们的 Python 环境中应该使用哪些库(以及它们的哪个版本)的信息来源。当我们想要更新库的时候，我们经常在我们的环境中更新它们，然后相应地修改需求文件。

但是，有一个更简单的方法。我最近发现了一个小的 Python 库，它对于保持一个干净的`requirements.txt`文件和加速更新过程非常有帮助，尤其是对于有大量依赖项的项目。

![](img/f5047ab799d40c5f3249417969a84287.png)

照片由 [Tran Mau Tri Tam](https://unsplash.com/@tranmautritam?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/kitten?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄

# `pur`在行动

`pur`代表 *pip 更新需求*，是一个小的 Python 库，可以用一个命令更新需求文件。它的基本用法非常简单，我们将用一个例子来说明它。

首先，我们需要安装`pur`:

```
pip install pur
```

让我们假设我们在一个虚拟环境中，我们有项目的`requirements.txt`文件，它包含以下内容:

```
pandas==1.2.4
yfinance==0.1.63
```

这是一个非常简单的例子，但它符合我们的目的。此外，我们知道这两个库都过时了。在这种情况下，我们可以通过在终端中运行以下命令来使用`pur`库:

```
pur -r requirements.txt
```

它用文件中列出的库的最新版本修改了`requirements.txt`。执行此操作时，它会打印以下内容:

```
Updated pandas: 1.2.4 -> 1.4.1
Updated yfinance: 0.1.63 -> 0.1.70
All requirements up-to-date
```

在这一点上，强调一个关键的事情是很重要的:库从不修改环境本身，也就是说，它既不安装也不更新任何库。它只做一件事——更新需求文件。话虽如此，现在我们实际上需要安装库的更新，例如，通过运行以下命令:

```
pip install -r requirements.txt
```

它安装了最新版本的库——`yfinance`的 0.1.70 和`pandas`的 1.4.1。此时，如果依赖项的版本发生冲突，安装将会失败。

`pur`还提供了额外的选项，例如，迭代地询问需求文件中每个库的更新。在我们知道一些库实际上需要被固定到某些版本的情况下，这可能是有用的，否则事情将会崩溃。

# 外卖食品

*   `pur`是一个方便的轻量级库，它负责用库的最新版本更新`requirements.txt`文件，
*   库从不安装/更新库，它只修改需求文件，
*   当一些版本需要保持固定以使我们的代码正常工作时，应该谨慎使用这个库。

此外，欢迎任何建设性的反馈。你可以在推特[或评论中联系我。](https://twitter.com/erykml1?source=post_page---------------------------)

喜欢这篇文章吗？成为一个媒介成员，通过无限制的阅读继续学习。如果你使用[这个链接](https://eryk-lewinson.medium.com/membership)成为会员，你将支持我，不需要你额外付费。提前感谢，再见！

您可能还会对以下内容感兴趣:

[](https://eryk-lewinson.medium.com/my-2021-medium-recap-650326b2832a) [## 我的 2021 年中期回顾

### 2021 年的简要回顾和 2022 年的计划

eryk-lewinson.medium.com](https://eryk-lewinson.medium.com/my-2021-medium-recap-650326b2832a) [](/a-step-by-step-guide-to-calculating-autocorrelation-and-partial-autocorrelation-8c4342b784e8) [## 计算自相关和偏自相关的分步指南

### 如何在 Python 中从头开始计算 ACF 和 PACF 值

towardsdatascience.com](/a-step-by-step-guide-to-calculating-autocorrelation-and-partial-autocorrelation-8c4342b784e8) [](/5-free-tools-that-increase-my-productivity-c0fafbbbdd42) [## 5 个提高我工作效率的免费工具

### 这不是一个好的 IDE，尽管它很有帮助！

towardsdatascience.com](/5-free-tools-that-increase-my-productivity-c0fafbbbdd42) 

# 参考

*   [https://github.com/alanhamlett/pip-update-requirements](https://github.com/alanhamlett/pip-update-requirements)