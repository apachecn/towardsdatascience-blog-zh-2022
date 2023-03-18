# 编写其他人可以使用的 Python 函数的综合指南

> 原文：<https://towardsdatascience.com/comprehensive-guide-to-writing-python-functions-others-can-use-2fa186c6be71>

## 函数编写是一项技能——迈出掌握它的第一步

![](img/72cb75ee660e42915cf7d52a23ac571f.png)

**照片由**[**Dziana Hasanbekova**](https://www.pexels.com/@dziana-hasanbekava?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)**[**Unsplash**](https://www.pexels.com/photo/unrecognizable-man-relaxing-on-hammock-5480702/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)**

## **介绍**

**让我们搞清楚:每一行代码都是写给其他人阅读的。句号。**

**想知道为什么大多数人用英语编码吗？为什么不用汉语、俄语、克林贡语或古波斯语？其实编码语言重要吗？嗯，没有。**

**每一个源代码，不管是什么语言，都被转换成只有计算机才能使用的机器语言。**

**因此，大多数编程语言使用英语关键字的根本原因是，英语是全球通用的语言，被数十亿人所理解。**

**用英语编写源代码使人类更容易创建计算机程序，并与世界各地的其他程序员合作。这一切都归结于编写可理解的代码。**

**你写的代码是你的脸面，是其他程序员评判你的第一件事。这就是为什么你越早把这个真理灌输给自己越好。**

**这篇文章将讲述如何编写干净的、文档记录良好的函数，这些函数遵循最佳实践，并且是 ide 所乐于见到的。**

**[](https://ibexorigin.medium.com/membership) [## 通过我的推荐链接加入 Medium-BEXGBoost

### 获得独家访问我的所有⚡premium⚡内容和所有媒体没有限制。支持我的工作，给我买一个…

ibexorigin.medium.com](https://ibexorigin.medium.com/membership) 

获得由强大的 AI-Alpha 信号选择和总结的最佳和最新的 ML 和 AI 论文:

[](https://alphasignal.ai/?referrer=Bex) [## 阿尔法信号|机器学习的极品。艾总结的。

### 留在循环中，不用花无数时间浏览下一个突破；我们的算法识别…

alphasignal.ai](https://alphasignal.ai/?referrer=Bex) 

## 文档字符串

在您编写了一个函数之后，让它变得可理解的第一步是添加一个 docstring。下面是对一个好的 docstring 的剖析:

所有记录良好且受欢迎的库都以不同的格式遵循这种剖析。有 4 种主要的文档字符串格式:

*   谷歌风格
*   Numpydoc
*   重组后的文本
*   EpyTex

我们只关注前两个，因为它们是最受欢迎的。

## Google 风格的文档字符串

先说 Google Style 的功能描述部分:

第一句话应该包含函数的目的，就像文章中的主题句一样。它应该在打开三重引号后立即开始。可选的解释应作为单独的、不缩进的段落给出:

接下来是论据部分:

以`Args:`开始新段落表示您正在定义参数。参数在新的一行给出，缩进两个空格。在参数名之后，参数的数据类型应该用括号括起来。对于可选参数，应该添加一个额外的*‘可选’*键。

最后，定义返回值:

如果您的函数引发了任何有意的错误，您也可以通过 errors 部分:

有时，您可能需要在结尾处包含示例或额外注释:

## Numpydoc 格式文档字符串

这种格式是数据科学社区中最流行的。以下是完整格式:

`error` s 和`notes`部分遵循相同的模式。尽管它需要更多的台词，但我更喜欢这部。

以下是两种风格的示例函数:

> 如果功能使用`*yield*`关键字，可以用`*Yields*`改变`*Returns*`段。

如果用户使用像 PyCharm 这样的现代 ide，添加类型提示会非常有帮助。

## 不用谷歌就能访问函数的文档字符串

您也可以通过调用函数名上的`.__doc__`来访问任何函数的 docstring:

使用`__doc__`可以很好地处理小函数，但是对于像`numpy.ndarray`这样具有大文档字符串的大函数，您可以使用`inspect`模块的`.getdoc`函数:

它以一种易于阅读的方式显示函数的文档。

## 一次做一件事

许多初学者常犯的一个错误是编写太长太复杂的函数。

总是建议将函数设计为只执行一个特定的任务。小而精确的函数更容易用现代 ide 测试和调试

现在，你可能会想:‘如果我的代码产生了一个错误，我从来不需要花超过 2-3 分钟来解决它。“我总是可以用简单的`print`语句来解决它们，然后玩一玩……”这是典型的初学者误解。

当我刚开始学习 Python 时，我也认为我所学的书籍和课程在代码中制造了大量的 bug。因为那时，我写的代码还不够复杂，不足以产生“令人头疼的”错误。

如果你仍然这样想，试着写一个实际上有几百行的脚本/程序。你会明白我的意思。

同时，考虑这个函数:

数据集链接:[https://www.kaggle.com/datasets/juanmah/world-cities](https://www.kaggle.com/datasets/juanmah/world-cities)

首先，docstring 没有很好地描述函数。如果我们花一些时间阅读代码，我们会意识到它的主要目的是从`path`参数中读取一个`csv`文件，并使用`country`参数对其进行子集化，返回该国人口最多的前 25 个城市。

如果你注意的话，这个函数的主要目的是在一行中完成的(就在第二个注释之后)。其他生产线正在执行不太清楚的清洁任务。

最理想的情况是将这一功能拆分开来，这样所有的清洁工作都在一个区块中完成，而 25 个城市的子集则在另一个区块中完成。让我们从清洁开始:

![](img/7bc747196e17a999c6d887c01d3f367f.png)

在上面的函数中，我使用了`.rename`方法将`lng`列重命名为`lon`。在最初的脏函数中，创建了一个新列，删除了不必要的旧列。

下一步是创建另一个函数，它是给定国家顶级城市的子集:

这个函数比原来的要好，因为我还插入了异常处理逻辑，用于当数据中没有匹配的国家时。

这两个新功能包含更好的文档并遵循最佳实践。

## 结论

在本文中，您了解了编写清晰易读的函数的最佳实践。现在，您可以用两种流行的惯例编写函数文档字符串，并将冗长而混乱的函数分解成更容易调试的模块化函数。

编写好的函数本身就是一种技能，就像任何技能一样，需要时间和练习来掌握它。感谢您的阅读！

**您可以使用下面的链接成为高级媒体会员，并访问我的所有故事和数以千计的其他故事:**

[](https://ibexorigin.medium.com/membership) [## 通过我的推荐链接加入 Medium。

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

ibexorigin.medium.com](https://ibexorigin.medium.com/membership) 

**或者订阅我的邮件列表:**

[](https://ibexorigin.medium.com/subscribe) [## 每当 Bex T .发布时收到电子邮件。

### 每当 Bex T .发布时收到电子邮件。注册后，如果您还没有中型帐户，您将创建一个…

ibexorigin.medium.com](https://ibexorigin.medium.com/subscribe) 

## **阅读更多我的故事:**

[](/complete-guide-to-experiment-tracking-with-mlflow-and-dagshub-a0439479e0b9) [## 使用 MLFlow 和 DagsHub 进行实验跟踪的完整指南

### 创建可重复且灵活的 ML 项目

towardsdatascience.com](/complete-guide-to-experiment-tracking-with-mlflow-and-dagshub-a0439479e0b9) [](https://ibexorigin.medium.com/6-sklearn-mistakes-that-silently-tell-you-are-rookie-f1fe44779a4d) [## 6 Sklearn 默默告诉你是菜鸟的错误

### 没有错误消息——这就是它们的微妙之处

ibexorigin.medium.com](https://ibexorigin.medium.com/6-sklearn-mistakes-that-silently-tell-you-are-rookie-f1fe44779a4d) [](/3-best-often-better-alternatives-to-histograms-61ddaec05305) [## 直方图的 3 个最佳(通常更好)替代方案

### 避免直方图最危险的陷阱

towardsdatascience.com](/3-best-often-better-alternatives-to-histograms-61ddaec05305)**