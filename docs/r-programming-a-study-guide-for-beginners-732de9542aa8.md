# r 程序设计——初学者学习指南

> 原文：<https://towardsdatascience.com/r-programming-a-study-guide-for-beginners-732de9542aa8>

## 使用本指南来规划您的 R 编程学习之旅，并了解如何准备构建生产级 R 代码

![](img/1df5a89c435de97011babf1dafa3b6e2.png)

照片由[弗雷迪婚姻](https://unsplash.com/@fredmarriage) @Unsplash.com 拍摄

D 在过去的几年里，我已经培训了数千名学生学习 R 编程，并与许多试图将 R 作为他们第一编程语言的人进行了交流。

作为一种开源语言，R 有如此多的特性，以至于在围绕它制定学习计划时会迷失方向。你从哪里开始？学习对象？学习模型？处理数据框架？这些概念之间的纠缠让一切变得更加复杂。

对于初学者来说，R 有优势。可以说，它比 Python 简单一点，因为大多数人学习这门语言是为了做以下三件事之一:

*   数据分析
*   数据科学
*   数据可视化

有了 Python，你可以拿起这门语言去做其他事情(更多地与软件工程联系在一起),比如后端开发或一些自动化项目——这使得这门语言比 r 更复杂一些。

这篇文章应该可以帮助你围绕 R 计划你的学习之旅。我在对我的绝对初学者课程的 [R 进行了大量迭代并结合了我的学生的大量反馈后，设计了这个学习流程(向他们致敬！).](https://www.udemy.com/course/r-for-absolute-beginners/?couponCode=IVOSTUDENTSJULY)

这是我推荐你学习的六大主题，依次是:

*   r 基本对象
*   r 数据帧
*   系统模型化
*   功能
*   图书馆
*   测绘

让我们更详细地了解一下吧！

# r 基本对象

R 对象是理解整个 R 语言的基础。基本的是:

*   向量
*   列表
*   数组
*   矩阵

在学习它们的过程中，你会接触到两个重要的特征，它们将决定你如何与它们互动，即:

*   单一类型与多类型对象
*   一维对象与多维对象。

为什么不立即跳到主 R 对象，即数据帧？因为它们是重要的操作，只有通过控制其他对象才可能实现——两个例子:

*   向量可以和`%in%`命令一起使用，在一个过滤器中子集化多个实例。
*   列表是 R 中唯一允许嵌套对象的对象。

R 的基本编程逻辑是围绕这些基本对象的。你应该在处理其他事情之前先研究它们，以增加成为一名优秀 R 开发人员的可能性。你可以在这里查看更多关于对象[的信息，或者直接进入我的](/5-r-objects-you-should-learn-to-master-r-programming-685341ce6661) [R 编程课程](https://www.udemy.com/course/r-for-absolute-beginners/?couponCode=IVOSTUDENTSJULY)的第一部分。

# 钉住数据框对象

如果您从事数据分析或科学工作，操作数据框将是最重要的技能。

**如果你习惯于用其他二维格式工作，比如 SQL 表，这个对象将是一个范式转变。**要掌握数据框架，您需要深入了解:

*   在 R 中索引行和列；
*   排序对象；
*   通过特定的键聚集；
*   过滤；

这些操作在为分析或建模构建数据时非常常见。如果您真正了解如何来回处理数据框，您将能够加快代码开发。

W3 学校包含一个关于他们的[不错的指南](https://www.w3schools.com/r/r_data_frames.asp)！

# 功能

**函数让你的代码变得可重用和干净。它们是正确的 R 脚本的支柱，没有它们，我们就不能在不同的对象上调用几个方法。**

**你知道你一启动 R 就和函数交互吗？**例如，当你使用`c()`创建一个 vector 时，你正在与一个名为`c`的函数交互，这个函数将你输入的对象组合成参数。不相信我？就在你的 R 控制台上呼叫`help(c)`！

没有功能，每个人都将编写枯燥重复的代码，无法维护和调试。

第一次构建自己的函数时，您可能会有点困惑，因为我们大多数人都习惯于编写脚本(特别是，如果您不是软件工程背景)。学习如何编写它们将会提高你的编码技能。使您准备好处理其他编程语言和编码范例。

你可以在我的 [R 编程课程](https://www.udemy.com/course/r-for-absolute-beginners/?couponCode=IVOSTUDENTSJULY)上了解更多，或者通过查看这个[博客上的一些最佳实践。](/writing-better-r-functions-best-practices-and-tips-d48ef0691c24)

# 走进图书馆

只有当你与别人的代码打交道时，你才是真正的 R 开发人员。怎么做呢？使用库！

库(或包)是使用 R 的主要优势(与其他非开源语言相比)。学习如何安装、加载和调试软件包的代码将让你接触到社区开发的数百万行代码。

你可以开始调查哪些图书馆？以下是一些建议:

*   训练决策树的内置`[rpart](https://cran.r-project.org/web/packages/rpart/rpart.pdf)`库。
*   一个非常酷的数据辩论库。
*   `[ggplot2](https://ggplot2.tidyverse.org/)`r 国最著名的绘图库。你应该把它留到以后学习的时候。

当你觉得合适的时候，不妨去看看这篇[博文](/top-r-libraries-for-data-scientists-16f064151c04)，看看一些你可以探索的图书馆的推荐。

# 系统模型化

当人们一头扎进 R 时，他们犯的主要错误是直接进入模特行业。

如果您在没有理解基本对象和功能的情况下从这里开始，您可能会有令人沮丧的经历。**为什么？**

**首先，您将无法很好地操作您的模型的输出**，因为多个模型需要不同的对象，甚至可能输出不同的格式。

其次，你将很难理解建模函数的参数是如何工作的。你不希望只在基础 R 中与建模捆绑在一起——你希望能够使用`caret`、`h2o`或其他独立库如`ranger`来训练你自己的高级模型。所有这些图书馆都有自己的琐碎和特色。它们都需要不同类型的参数、对象和细节。

**最后，每个模型本身都是一个函数，有自己的一组自变量和参数。**此外，有三件重要的事情你必须知道，以便与他们无缝合作:

*   如何操作函数。
*   参数期望什么类型的对象。
*   如何通过使用具有更快或更准确模型的外部库来改进您的培训过程。

当你准备好解决建模问题时，请随意进入我的 [R 数据科学训练营](https://www.udemy.com/course/r-for-data-science-first-step-data-scientist/?couponCode=IVOSTUDENTSJULY)，在那里你将学习用 R 构建机器学习模型的理论和实践

# 测绘

这个列表的最后一个元素是绘图。有了`ggplot2`、`plotly`或`altair`，当谈到可视化库时，你有很多选择。它们都适合构建非常有趣的图表，可以讲述关于您的数据的故事。

成为数据可视化专家绝非易事。我提到的库有数百个参数和设置，人们可以通过调整来改进。我建议您从建立以下地块的基线开始:

*   [一个简单的散点图。](https://r-graph-gallery.com/272-basic-scatterplot-with-ggplot2.html)
*   [历史数据折线图。](http://www.sthda.com/english/wiki/ggplot2-line-plot-quick-start-guide-r-software-and-data-visualization)
*   [一个方框图。](https://plotly.com/python/box-plots/)
*   饼图。

你可以在我上面详述的每个库中做其中的一个图。

理解它们的主要差异和复杂性将使您在围绕数据构建自己的故事时更加灵活。另一个重要的细节——我建议您跳过 base R plotting，因为与上面提到的任何包相比，它都非常有限。

就是这样！我希望你喜欢这篇文章，你可以更好地规划你的学习之旅。

做这样的学习流程帮助我培训了世界上成千上万想要学习 R 的人——当人们跟随这个旅程时，我看到了他们编码能力的重大突破。

**当然，这并不意味着你要在完全掌握了之前的概念之后，才可以跳到下一个概念。**首先，很好地掌握基础知识，巩固基础知识，在做了一些实践练习和构建了一些代码之后，您就可以开始下一个组件了。

重要的是，在进行下一项技能之前，你对每一项技能都感到满意。

如果你想去我的 R 课程，请随时加入这里( [R 编程绝对初学者](https://www.udemy.com/course/r-for-absolute-beginners/?couponCode=IVOSTUDENTSJULY))或这里([数据科学训练营](https://www.udemy.com/course/r-for-data-science-first-step-data-scientist/?couponCode=IVOSTUDENTSJULY))。我的课程大致遵循这种结构，我希望有你在身边！

![](img/2702d465b5b32684ef0f97483ae65c31.png)

[数据科学训练营:你成为数据科学家的第一步](https://www.udemy.com/course/r-for-data-science-first-step-data-scientist/?couponCode=IVOSTUDENTSJULY) —图片由作者提供

[](https://ivopbernardo.medium.com/membership) [## 通过我的推荐链接加入 Medium-Ivo Bernardo

### 阅读我在 Medium 上的所有故事，了解更多关于数据科学和分析的信息。加入中级会员，您将…

ivopbernardo.medium.com](https://ivopbernardo.medium.com/membership)