# 学习 Tableau 你必须知道的五件事

> 原文：<https://towardsdatascience.com/five-essential-things-you-must-know-to-learn-tableau-2b899403bfcb>

## *免费资源指南，帮助你培养技能*

![](img/6fe89e2f238263be397771dbcfc47883.png)

秘鲁的因卡小径。图片由作者提供。

Tableau 是可视化数据的优秀工具。然而，这不是你可以通过发现来学习的直观软件。如果您在没有扎实的数据和可视化原理背景的情况下就开始点击界面，那么您会遇到很多挫折。

本文基于我最近的一篇文章，其中包含了学习 Tableau 的免费资源。

</the-best-free-resources-for-learning-tableau-skills-5f0e5bfaa87e>  

在这里，我通过总结学习者在学习场景之前*必须知道的五件重要事情来帮助你最大限度地利用这些和其他资源。我的建议是基于我自己教授新学员数据分析和可视化的经验。*

这个列表应该被视为帮助你确定学习活动方向的起点。请小心，认为这是一个详尽的建议列表或一个简短的清单，您可以在使用该软件之前完成。

# 1.数据类型

![](img/2e1cb1ce72635a631aad2c701c78d8be.png)

中国昆明的街头艺人。图片由作者提供。

确保您对数据类型之间的差异有一个清晰而深刻的理解，尤其是*度量*(或*度量*)和*维度*，以及*连续*与*离散之间的差异。*Tableau 中的每个决策都取决于对数据的清晰理解。请记住，这些概念广泛适用于数据分析和可视化，而不仅仅是 Tableau。

*   intrincity 101[YouTube]:[什么是维度和度量？](https://www.youtube.com/watch?v=qkJOace9FZg&t=61s)
*   sqlbelle 的 Tableau 教程[YouTube]: [维度与度量，蓝色与绿色，离散与连续](https://www.youtube.com/watch?v=LaDOkX1iWp8&t=24s)
*   LearnBI。在线[YouTube]: [什么是度量和维度](https://www.youtube.com/watch?v=j7HuAWl4VW8)
*   Tableau 文档:[尺寸和尺寸介绍](https://www.tableau.com/drive/dimensions-and-measures-intro)
*   表格文档:[尺寸和度量，蓝色对绿色](https://help.tableau.com/current/pro/desktop/en-us/datafields_typesandroles.htm)

# 2.数据结构

![](img/8da2e674463eb24d4b5e28dd1ec9db61.png)

一些电的东西。图片由作者提供。

数据可视化中最耗时的工作是准备数据。大多数数据集都是宽*长*的格式，但是 Tableau 更喜欢长长*的格式。Tableau 提供了*旋转*数据的功能，但永远不会告诉你*什么时候*该是时候了。在学习 Tableau 之前，请确保您对宽而长的数据有坚实的理解。*

我强烈建议每个学生花大量的时间来复习 Hadley Wickham 的论文。您可以放心地忽略关于 R 编码的内容，但是关于结构化数据的所有其他内容都是必不可少的。这篇论文对初学者来说是技术性的和复杂的，但对任何严肃的数据工作来说是至关重要的。花点时间通读这篇论文，并在积累数据经验的同时定期重温它。

*   哈德利·韦翰:[整理数据](https://www.jstatsoft.org/article/view/v059i10)
*   静态学:[长数据与宽数据](https://www.statology.org/long-vs-wide-data/#:~:text=A%20dataset%20can%20be%20written,repeat%20in%20the%20first%20column.&text=Notice%20that%20in%20the%20wide,the%20first%20column%20is%20unique.)
*   Udacity [YouTube]: [宽长格式](https://www.youtube.com/watch?v=zlaeISxRESQ)

我还推荐 Jonathan Serrano 在《走向数据科学》中的这篇文章:

[https://towards data science . com/long-and-wide-formats-in-data-explained-e48d 7 c 9 a 06 CB](/long-and-wide-formats-in-data-explained-e48d7c9a06cb)

# 3.电子表格技能

![](img/d577ea08ccb189e161d79a55620862e2.png)

灭火系统。图片作者。

不要低估电子表格技能的价值。出于几个原因，我教授 Tableau 入门课程，并确保我的学生牢牢掌握电子表格技能。电子表格允许您查看您的数据，这对于理解其内容和结构至关重要。处理数据时，您应该不断地查看原始数据和汇总表格。电子表格非常适合查看您的数据。

你有时需要在原始数据中进行修正，这在 Tableau 中是不可能的。此外，用于操作数据的电子表格函数类似于 Tableau 用于创建计算字段的函数。有许多优秀的免费电子表格学习资源。如果你想要一套高质量的免费资源，直接去 Leila Gharani 的 YouTube 频道。她收集了很多精彩的视频，涵盖了你需要了解的关于电子表格的一切。

# 4.数据透视表

![](img/90c39f9eca5cb206226847d7229c0f8a.png)

中国重庆交通茶馆。图片由作者提供。

可视化数据类似于使用电子表格数据透视表汇总数据。在建立我的可视化之前，我经常在 Tableau 中创建汇总表，以确保我所有的计算都是正确的。虽然在 Tableau 中通常被称为*文本表*，但它们本质上与电子表格数据透视表相同。数据透视表将迫使您理解度量和维度之间的关键差异以及不同类型的聚合。由于数据透视表是电子表格技能的一部分，我建议您参考 Leila Gharani 的教程:

*   Leila ghar ani[YouTube]:[Excel 数据透视表 10 分钟讲解](https://www.youtube.com/watch?v=UsdedFoTA68)
*   凯文·斯特拉特维特[YouTube]: [数据透视表 Excel 教程](https://www.youtube.com/watch?v=m0wI61ahfLc&t=113s)
*   科迪·鲍德温[YouTube]: [在 6 分钟内学会数据透视表](https://www.youtube.com/watch?v=qu-AK0Hv0b4)
*   多产橡树[YouTube]: [谷歌数据表](https://www.youtube.com/watch?v=Tty0RyD1KLw)

# 5.视觉编码

![](img/bfaedaf2b7c87cf30000a363ce6d2290.png)

工具照片。图片由作者提供。

Tableau 是一个可视化数据的工具。不管你对画面的技术熟练程度如何，你的视觉传达知识将是你的限制因素。您需要对定量数据的可视化表示有很好的理解，尤其是度量和维度之间的差异，以及连续值和离散值之间的差异。花时间建立对*标志*和*渠道的概念性理解。这里有几个视频可以帮你入门。*

*   Udacity:可视化编码— [数据可视化和 D3.js](https://www.youtube.com/watch?v=14FJU1kP6-M)
*   Curran Kelleher: [标记数据可视化中的&通道](https://www.youtube.com/watch?v=KGUxDlZ6OFQ)
*   Tamara Munzer: [标记和通道](https://www.youtube.com/watch?v=xplSAMwlTmY&t=141s)
*   塔玛拉·曼泽:[商标和渠道(视频 2)](https://www.youtube.com/watch?v=Oz0Zs-R9USE&t=5s)

还是那句话，记住 Tableau 只是一个工具，学习 Tableau 并不等同于学习数据可视化。我可以教你如何使用锤子，但这并不意味着你可以*设计*和*建造*房子。出于这个原因，我将冒昧地向您推荐我的一篇早期文章，这篇文章强调了在进行数据可视化时软件和设计角色之间的重要区别。

</less-software-more-design-449175a34e59>  

# 后续步骤

![](img/9b75562fa18ad3dfd2c46aca4fa56875.png)

秘鲁的因卡徒步旅行路线。图片由作者提供。

同样，这篇文章并不是详尽的，而是一个简短的列表，列出了学习 Tableau 的基本要点。请随意在评论区张贴其他资源和对你的学习有帮助的东西。如果您对构建数据技能和学习 Tableau 感兴趣，请务必关注我。