# 数据科学如何学习 Python

> 原文：<https://towardsdatascience.com/how-to-study-python-for-data-science-888a1ad649ae>

## 在用 Python 学习和实践数据科学一年后，我的建议是

![](img/20771f9efb56427358c4958d17697a78.png)

在 [Unsplash](https://unsplash.com/s/photos/python?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上由 [Hitesh Choudhary](https://unsplash.com/@hiteshchoudhary?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄的照片

众所周知，数据是新的石油，鉴于数据科学领域的大肆宣传，有很多人正在研究数据科学并试图向它过渡(我也是:)。

因此，有太多的课程可供选择:我们如何选择正确的课程？更多:“Python 对于数据科学”是什么意思？因为似乎每门课程都会教你数据科学的 Python 语言；但真的是这样吗？

在本文中，我将根据我学习和实践 Python 数据科学的第一年，给出我的答案。

# 建议 1:如何学习基础知识

我告诉你真相:真正需要关注的问题不是“正确”的课程。寻找“正确的课程”——也许是承诺你在课程结束后找到工作的课程——不是你应该关注的(嗯，我向你保证，确实有帮助你找到工作的课程；但这不是重点)。

如果你想学习“数据科学的 Python”几乎任何课程都会教你同样的东西(几乎，我们说的是基础)。

因此，选择一门课程并:

*   **多多练习**。实践让你成为开发者，而不是看视频。所以，看视频，读课文，做大量练习【额外提示:**选一门有项目**的课程，这样你在练习的时候就有了一些指导方针】。
*   **掌握基础知识**。基础是任何事情的基础；你不能从屋顶开始盖房子。所以，学得很好:列表，元组，字典，循环，之类的东西。当然，还有:用这些概念进行练习。
*   **将函数和类留待以后使用。**我知道，有人可能会嗤之以鼻，但请听我说。在我看来，Python 课程中的人让函数和类变得容易，但事实并非如此。当你真的编程了很多的时候，你会需要函数和类；明确地说(简化):当你复制和粘贴相同的代码行时，你需要一个函数，当你复制和粘贴相同的函数时，你需要类；因此，为了简化，当您想要“自动化”您的代码时，您需要函数和类，这不是您在开始时所做的。如果你学习数据科学的 Python，你想分析数据；所以我的建议是:*理解函数和类，但只是为了理解*；不要太关注这些。当您分析完数据后，定义您的标准并创建您自己的类和函数来自动化您的代码。例如，[那些](https://github.com/federico-trotta/plots_custom_functions)是我的一些定制功能。

# 建议 2:统计学、Matplotlib 和 Seaborn

在您掌握了 Python 的基础知识之后，您将开始分析数据，主要是使用 Numpy 和 panasus，并且——因为您喜欢数据——您想要制作一些情节；因此，您需要了解一些统计数据和一些数据可视化库:Matplotlib 和 Seaborn。

让我用统计学非常明确地说:*你不需要成为一个统计学家。*请再读一遍。

尤其是在开始时，只需掌握以下基础知识:

*   **正态分布**
*   **什么是平均值、模式和中位值**
*   条形图和直方图之间的***区别是什么(是！有区别的！)***
*   *什么是**方框图**，我们可以从中获得什么信息*
*   *什么是**相关性**以及如何处理。[在这里](/the-difference-between-correlation-and-regression-134a5b367f7c)你会发现我写的一篇关于这个话题的文章可能对你有帮助*

*等等。*

*您不需要了解高级统计主题来分析数据和进行一些数据科学研究。从基础开始，理解它们，并通过实践来掌握它们。*

*然后，在策划时，我的建议是:*

*   ***从 Matplotlib** 开始。Matplotlib 是一个非常强大的数据可视化库；在我看来，它的缺点是你有时需要过多的代码来创建一个情节；但是这种努力是值得的，因为与此同时，您正在掌握您的编程技能*
*   *过一段时间，开始用 **Seaborn** 。当你开始使用 Seaborn 时，你将面对它的综合性和简单性，同时有可能绘制复杂的情节。你甚至可能会问自己为什么之前没有学过，原因很简单:Matplotlib 更容易理解；而且，事实是，当你甚至已经学会了 Seabors，你将同时使用 Matplotlib 和 Seaborn，有时甚至在同一个情节中。*
*   *其他可视化工具和库(像 **Plotly** )。在您掌握了 Matplotlib 和 Seaborn 之后，您可能会喜欢尝试其他工具和可视化。这完全没问题，但我的建议是不要进入兔子洞；所有这些图书馆和信息可能会让人不知所措；事实是，你通常会使用专门的软件进行数据可视化(比如 Power BI 或 Tableau)。因此，探索你感兴趣的一切，但我在这里的最后建议是，了解你在真实的工作环境中可能需要什么，并使用这些工具和软件。*

# *建议 3:机器学习*

*毫无疑问:使用 scikit-learn 作为 ML 的库。正如你将看到的，sk-learn 是一个具有许多功能的巨大库，但是——你可能知道——机器学习是一个广阔的领域，所以它的主库必须是。*

*在深入研究 ML 模型之前，我的建议是:*

*   *理解回归和分类之间的**差异**，因为这是你在 ML 中将要面对的两种主要问题*
*   *理解**将数据帧**分割成训练、验证和测试集的重要性，并开始练习 sk-learn 中的“train_test_split()”函数*
*   *了解您可以用来验证模型的**指标**。例如，从一个简单的线性回归问题开始，开始使用 MSE 和 RMSE 来验证你的模型。这是我给你做的一个项目，作为初学者，你可以作为一个指南。*

# *结论*

*我想强调的重要一点是，学习路径不是线性的，正如你可能认为的那样；但这是普遍真理。如果你正纠结于一个概念，没有必要为之疯狂:把它留在那里没关系，过一会儿再回来。*

*举个例子，正如我所说的:如果你正在纠结于函数和类，试着去理解它们，如何使用它们，何时使用它们；然后，当你明白你需要它们时，再回到它们身上(并深化概念，在你的代码中使用它们)。*

*此外，如果你全职工作或学习，并且想找时间学习和实践数据科学，在本文[中，你可以找到我的建议。](https://federicotrotta.medium.com/how-to-study-data-science-even-if-you-work-or-study-full-time-b52ace31edac)*

*需要 Python 和数据科学方面的内容来开始或促进您的职业生涯？下面是我的一些文章，可以帮到你:*

***巨蟒:***

*   *[Python 中的循环和语句:深入理解(附示例)](/loops-and-statements-in-python-a-deep-understanding-with-examples-2099fc6e37d7?source=your_stories_page-------------------------------------)*
*   *Python 循环:如何在 Python 中迭代的完整指南*
*   *[学习 5 个 Python 库，开始你的数据科学生涯](/5-python-libraries-to-learn-to-start-your-data-science-career-2cd24a223431)*

***数据科学:***

*   *[即使全职工作(或学习)也要如何学习数据科学](/how-to-study-data-science-even-if-you-work-or-study-full-time-b52ace31edac)*
*   *[如何处理数据科学中的缺失值](/how-to-deal-with-missing-values-in-data-science-9e5a56fbe928)*
*   *[如何在数据科学项目中进行特征选择](/how-to-perform-feature-selection-in-a-data-science-project-591ba96f86eb)*
*   *[如何检测数据科学项目中的异常值](/how-to-detect-outliers-in-a-data-science-project-17f39653fb17?source=your_stories_page-------------------------------------)*
*   *[进行图形残差分析的两种方法](/two-methods-for-performing-graphical-residuals-analysis-6899fd4c78e5)*
*   *[如何利用学习曲线轻松验证您的 ML 模型](https://medium.com/mlearning-ai/how-to-easily-validate-your-ml-models-with-learning-curves-21cc01636083)*
*   *条形图和柱状图有什么区别？*
*   *[相关和回归的区别](/the-difference-between-correlation-and-regression-134a5b367f7c?source=your_stories_page-------------------------------------)*
*   *[了解 l1 和 l2 正规化](/understanding-l1-and-l2-regularization-93918a5ac8d0?source=your_stories_page-------------------------------------)*
*   *[逻辑回归:我们来清理一下！](https://medium.com/mlearning-ai/logistic-regression-lets-clear-it-up-8bf20e9b328a?source=your_stories_page-------------------------------------)*
*   *什么是训练有素的模特？*

**考虑成为会员:你可以支持我和其他像我一样的作家，不需要额外的费用。点击* [*这里*](https://federicotrotta.medium.com/membership) *成为会员。**