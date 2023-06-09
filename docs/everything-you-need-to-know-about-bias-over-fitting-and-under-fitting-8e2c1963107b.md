# 关于偏差、过度拟合和欠拟合，你需要知道的一切

> 原文：<https://towardsdatascience.com/everything-you-need-to-know-about-bias-over-fitting-and-under-fitting-8e2c1963107b>

## 偏见的详细描述以及它是如何融入机器学习模型的

![](img/b8bd445b8f89a56f498a0f807ca9b128.png)

(图片由[马库斯·斯皮斯克](https://unsplash.com/photos/8CWoXxaqGrs)在 [Unsplash](http://unsplash.com) 上拍摄)

# 介绍

数据科学领域的一个关键特征是将人工智能和统计学应用于现实世界的数据。数据科学的这一部分的伟大之处在于，它非常强大，几乎可以应用于任何可以成功争论数据的地方。实际上，编写一个机械地适应特征的算法是相当困难的，因此一个非机械的、基于损失的算法可以允许人们使用统计数据来做出决策。

虽然在许多应用程序中使用机器学习可能是有益的，但在每种情况下使用机器学习也有一些缺点。首先是性能，因为通常神经网络的性能不会超过简单的条件和传统算法。虽然性能是模型对话中的一个重要话题，但我想把今天的焦点更多地转向在不同的环境中使用机器学习的不同缺点，

> 机器学习很难。

请允许我详细说明，预测建模是**而不是**难。我认为，大多数(如果不是所有的话)有编程经验的人可能会查看 Python 中基本网络的一些源代码，并快速掌握代码中发生的事情。然而，困难的是预测建模之前的一切。总的来说，我认为预测建模确实包括处理数据，但是我想说的是，完美地处理数据以处理您的模型比仅仅将模型拟合到所述数据要困难得多。今天，我想回顾建模和模型偏差的基础知识，所有这些如何影响你的预测，以及一些可以用来缓解这些问题的潜在途径。

# Part №1:什么是偏见？

在我们继续讨论偏见之前，我们也许应该谈一谈偏见本身究竟是什么。在常规的人类定义中，偏见意味着对某一特定类别或观点存在某种程度的偏见。例如，一个飞机制造商可能有偏见地说飞机是最好的运输方式。这个定义如何应用于数据？

那么，在机器学习环境中，什么是偏见呢？所有的机器学习模型都有偏差，偏差并不一定需要低或高。偏差只是模型基于特征得出结论的倾向。偏见的重要之处在于，随着模型信息过载，它会不断增长。基本上，每当解算器和权重组合达到峰值时，就会发生这种情况。这意味着模型以最高效率**预测我们提供给它的**数据。第二部分非常重要，在我们讨论完“完美契合”之后，我们会明白为什么这是正确的。

# 第二部分:“完美契合”

T he perfect fit 只是我想出来的一个玩笑名字，用来描述一个在偏差上完全平衡的模型。在大多数情况下，很难得到完全完美的输入数据，因此很可能**大多数**模型可以在数据上稍作调整，以创建更完美的拟合。也就是说，找到至少一个可以接受的合适人选并不容易。现在，请允许我以三个月前的精神，十月，给你讲一个恐怖的故事。

> “偏见的平衡”

偏差的平衡是一个必须用他们的模型来玩的游戏，以确保模型有效地预测。有两种情况可能对与偏差相关的模型的性能完全有害:

*   过度拟合——参数估计值几乎没有偏差。
*   欠拟合-参数估计值中有太多的偏差。

## 过度拟合

过度拟合通常意味着您的模型在评估您的训练数据时基本上已经超出了偏差。换句话说，这些数据中包含了如此多的特征和观察结果，以至于模型在所有事物之间画出了太多的联系。总的来说，您的模型有太多的数据，或者经过劣质预处理的数据。

## 欠拟合

适配不足与适配过度正好相反。数据太多，偏差变为零，而偏差却增加到天文数字。发生这种情况是因为模型只有几个例子可以使用。总而言之，模型没有足够的数据。判断模型是否欠拟合的一个关键方法是检查偏差和方差。如果偏差较高，但方差较低，则很可能您正在处理一个急需更多数据的模型。

# 第 3 部分:减轻您的模型

我想说的最后一件事是如何修复一个表现不佳的模型。我有一整篇文章讨论如何通过处理输入数据来改进模型，你可能会在这里读到，因为在这篇文章中，我只打算介绍一些技术来帮助平衡偏差。如果你对这样一篇文章感兴趣，你可以在这里查阅:

</mitigate-your-model-with-data-3c4216580dae>  

## 过度拟合

现在让我们转到一些技术上来，这些技术可以用来帮助你的模型停止过度拟合。首先必须使用一个验证集。验证集通常要小得多，尽管更少的观察看起来可能是一件坏事，但最终模型将从这种变化中受益。另一个防止过度拟合的好方法是使用分解。如果您不熟悉分解及其用途，我有一整篇关于最流行的分解类型——奇异值分解的文章，其中我编写了一个 SVD 方法，您可以在这里深入研究:

</deep-in-singular-value-decomposition-98cfd9532241> [## 深入奇异值分解

towardsdatascience.com](/deep-in-singular-value-decomposition-98cfd9532241) 

无论如何，让一个模型停止过度拟合的最好方法可能是最小化你的特征。有很多很好的方法可以做到这一点，其中很多都在减轻您的模型文章中，但是最明显的方法就是去掉这些特性。或许，做一些基本假设，进行假设检验，然后**花点时间探索你的特征**，这样你就能确定哪些特征相关，哪些不相关。另一个好主意是设计一些特性，在这篇文章中，我也用 Python 演示了这些特性的代码绑定。

## 欠拟合

你的模型不合适的原因可能归结为两个简单的事情。第一种情况是你没有足够的数据。这通常意味着数据中有很高的方差，这使得诊断真正的拟合不足非常困难。如果是这种情况，除了简单地添加更多数据之外，您没有太多的方法来修复一个没有数据的模型。没有办法绕过它，你需要数据来拥有机器学习模型。

在其他情况下，您的模型可能不适合，或者可能看起来不适合，因为数据有很大的变化。减轻这种情况的一个好方法是对连续值做基本处理。评估方差，可能基于 Z 分数对数据进行归一化，等等。对于分类值，您可能获得的最佳结果是查看一个集合，也许是集合计数，这样您就可以看到哪个集合出现的次数最多，并且您将知道该特征的每个类别。然后应用正确的蒙版，使你的特征具有更重要的价值。

# 结论:所有数据都是不同的

既然我们知道了什么是偏差，以及它与过度拟合和欠拟合的关系，我们就有可能看到平衡偏差对于构建有效模型是多么重要。然而，关于这些技巧，要记住的一件关键事情是，所有的数据都是不同的。作为一名数据科学家，部分乐趣在于我们可以处理各种不同的数据。然而，这也提出了一个挑战，因为所有的数据都是不同的，需要解决问题的技巧，有时还需要巧妙的技巧，才能让您的数据完全符合您的需要。

在这条数据科学的道路上，你可能会遇到许多不同类型的数据，拥有一个平衡的模型的绝对关键是**仔细探索和选择你的特性**。你不会在一个不牢固的基础上建造一个房子，那么你为什么要在一个不牢固的基础上建造一个模型呢？这当然是要考虑的事情，我希望这一信息能引起共鸣，因为我发现通过一个我赞同的过程节省了很多时间。如果你想用我的数据了解更多我的过程，我有一整篇文章，其中我反复讨论了每个步骤，你可以在这里阅读:

</my-predictive-modeling-and-learning-step-process-technique-f0521ee76d90>  

> 是的，我喜欢我写了这么多文章，现在我可以提供我的旧文章作为阐述。

在开始学习数据科学时，偏见是一件需要理解的重要事情，一般来说，数据处理团队也是如此。我希望这篇文章足够吸引人，向我的读者揭示避免过度适应和适应不足的秘密。感谢您阅读我的故事，并祝您在数据科学探险中好运！