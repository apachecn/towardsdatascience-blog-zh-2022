# 机器人过程自动化——没有麻烦的自动化

> 原文：<https://towardsdatascience.com/robotic-process-automation-automation-without-the-hassle-ac857e5b847a>

## 自动化枯燥和重复的过程，而不触及底层系统。深入了解企业软件中发展最快的分支。

![](img/ffea5a1ce67f4a442deecd740cd8bec5.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上[尔汗·阿斯塔姆](https://unsplash.com/@vaultzero?utm_source=medium&utm_medium=referral)的照片

人类总是被让事情变得更简单、更舒适的愿望所驱使。从第一把斧子到沟渠，从网页脚本到机器学习算法——我们总是努力**为重复性任务**找到解决方案。

从这个角度来看，电脑在工作场所的兴起对推动人类进步不可或缺，极大地提高了我们的生产力。尽管计算机简化了许多任务，但人类干预在数字化运营中仍然至关重要，不断增长的数字化劳动力就是证明。

尽管计算机拥有巨大的能力，但我们在计算机上执行的许多任务都相当普通。例如，当雇用一名新员工时，我们经常需要处理他们的姓名、出生日期、社会保险号、职称、访问权限等。在各种系统和账户中。像这样的重复性任务**会浪费时间和认知技能**，而这些可以被更有效地利用。

“自动化”可能是答案，但是传统的流程自动化——在系统的后端运行——并不像看起来那么简单。在系统成功通信和编写准确的脚本之前，我们通常必须克服许多障碍。尽管技术上可行，但这并不总是最明智的商业决策。

此外，优秀的工程师供不应求。如今的企业宁愿将它们部署在实现人工智能能力和创造新的见解上，而不是煞费苦心地复制琐碎的任务。更不用说，数据科学家自己也倾向于在重复任务(如数据清理)上花费大量时间，而不总是有时间自动完成这些任务。简而言之，大家都吃亏。

幸运的是，还有一种选择——机器人！

> 本文将主要讨论软件机器人的简单化变体。正如我们将在最后看到的，更智能的(决策)变体正在崛起。这就是数据科学社区将发挥推动作用的地方，**结合基于规则的自动化和人工智能**。

# RPA 是什么？

![](img/049ec6259b84d54b1966ce901224fcc6.png)

图片由 [Taras Shypka](https://unsplash.com/@bugsster?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

与传统的过程自动化相比，机器人过程自动化(RPA) **在系统**的前端运行，即图形用户界面(GUI)，或系统 API。RPA 是一种软件技术，它部署软件代理，在数字系统上模拟人类操作。由于 RPA 不会侵入现有系统，因此通常可以快速集成以推动数字化转型。

与传统自动化相比，RPA 的主要优势在于它**不会干扰现有系统**。代理可以直接模仿通常由人类执行的操作。

RPA 超越了传统的屏幕捕捉，因为它可以**在系统和桌面之间切换**。例如，典型的工作流程可能是选择一封电子邮件，打开附件 pdf，复制相关字段，然后将它们粘贴到计费系统中。此类任务通常跨越多个业务部门，但可以表示为一系列操作。

# 何时使用 RPA？

在确定是否使用 RPA 来自动化特定工作流时，必须回答一些问题:

*   **自动化成本？** —完整的后端自动化通常成本高昂且复杂。完整处理信息需要多少时间？
*   **自动化的好处？—** 节省了多少工时？释放出来的人力资本将如何配置？自动化会增加工作满意度吗？
*   **流程是否基于规则？** —流程可以分解为简单的、基于规则的任务吗？需要任何(人类)判断吗？
*   **正确输出的数量？** —流程是否只有一个正确的输出？
*   **任务的重复性如何？** —很低的频率可能更适合手动控制，很高的频率将受益于完全自动化。
*   **流程有多成熟？—**GUI 是否有望在更长时间内保持不变？从长远来看，业务流程会保持不变吗？

回答这一系列问题对于**决定自动化哪些流程**是必要的，但是有点无聊。本质上，它归结为(I)过程足够简单(基于规则，没有例外或决策，等等。)和(ii)产生足够价值的自动化。让我们用一个简洁的行动优先级矩阵来想象一下。

![](img/9809fd879b0c5ed76f04dedbb6b6ef30.png)

行动优先矩阵。RPA 的理想候选是一个只需有限的努力即可实现自动化并具有重大业务影响的流程[图片由作者提供]

在自动化领域定位 RPA 的另一种方式是在**自动化连续体**上。RPA 将处于光谱的低端(严格基于规则的自动化)，而更复杂的任务可能涉及机器学习来处理决策。

有大量任务可能适合 RPA。完全(后端)自动化既费时又费钱，而且日常运营中仍有许多任务需要很少甚至不需要认知努力。

# 谁应该使用 RPA？

从零售到保险，从制造到医疗保健，各个领域都有 RPA 的身影。也就是说，RPA 似乎特别适合使用大量**遗留系统**的机构，这些系统很难更换或自动化。

如前所述，选择正确的自动化流程至关重要。虽然是一个轻量级、非侵入性的解决方案， **RPA 也有责任**，例如，考虑商业许可、详细的流程步骤、长期维护。大约一半的 RPA 项目会失败，所以愿意应用它的公司必须确保投入一些资源和严格性。半心半意的承诺不起作用。

潜在的用例种类繁多。RPA 可用于全自动化(由系统本身触发)，也可用于**人在回路**解决方案(手动激活机器人)。RPA 还可以自动化流程的各个部分，在谨慎的情况下使用人工判断和监控。

总而言之，RPA 可能适合具有以下特征的组织:

*   将大量人力资本花在重复性的、基于规则的任务上，禁止认知的、创造价值的活动；
*   拥有不容易自动化或更新的遗留系统；
*   对 RPA 的(不)功能有扎实的理解，但不一定有很强的内部软件技能；
*   能够并愿意严格处理自动化轨迹并投入适当的资源。

# RPA 用在哪里？

原则上，RPA 适用于任何具有简单、基于规则的任务的重复流程。在实践中，RPA 应用程序的常见示例包括:

*   **入职** —创建账户、访问权限、系统注册等。对于新员工。
*   **发票处理** —将外部来源的发票明细填入系统。
*   **提取-转换-加载(ETL)过程** —重复的数据操作
*   **付款提醒** —从以前的发票中复制粘贴信息
*   **银行流程自动化** —为涉及抵押、贷款、支付等流程填写客户详细信息。
*   **客户服务自动化** —对客户查询的自动化(非智能)响应

有大量可以应用 RPA 的示例流程。然而，关于自动化的好处和复杂性的问题应该始终是中心问题。

# RPA 怎么用？

RPA 领域目前由商业解决方案主导。也有开源的替代方案，但这些并不普遍，通常需要一些软件专业知识才能运行。当然，雇佣顾问来支持自动化轨迹是可能的；成功需要一定水平的技术能力和抽象过程思维。

一个关键的挑战是确定哪些过程应该自动化。既需要****对单个流程** **的深入了解，又需要对可用流程**的概述，才能准确确定最大的自动化潜力。为此，访谈、调查、文档分析、流程挖掘或头脑风暴研讨会可能是可行的选择。**

**重要的是**流程是基于规则的**。因此，应该可以将这个过程分解成一系列具体的任务，让软件机器人来执行。业务流程建模语言(BPML)是一种可用于构建业务流程的语言。无法对流程进行全面建模意味着它不适合 RPA。**

**![](img/a8f5413e168d9331b5cbf4f01d1f78ec.png)**

**使用 BPML 建模的简单流程示例。应该可以用具体的规则(如果…那么)和步骤来完整地描述一个适合 RPA 的流程。[图片来自维基媒体，作者是 [CLMann](https://commons.wikimedia.org/w/index.php?title=User:CLMann&action=edit&redlink=1) ]**

**最后，由于业务流程和 GUI 往往会随着时间的推移而变化，因此**灌输监控程序**以验证 RPA 随着时间推移的正确运行是非常重要的。自然地，过程的选择应该倾向于成熟和稳定的过程，但是即使这样 RPA 也不会在它的实现中结束。有必要定期检查忙碌的数字工作者的功能。**

# **不利因素和挑战**

**RPA 是一种轻量级解决方案，实施和加速自动化通常既快速又便宜。但是它也有自己的局限性和缺点。**

****更新**——使用图形用户界面的风险之一是它们容易改变。一旦系统布局发生变化，我们也必须更新软件机器人。这同样适用于频繁变化的业务流程。尤其是对于不成熟的系统和流程，更新 RPA 可能是一件痛苦的事情。**

****可扩展性** —尽管机器人可以比人更快地执行 GUI 任务，但我们不会部署 RPA 来执行一百万次拖放操作。工作负载可能会随着时间的推移而增长，在这种情况下，RPA 无法很好地扩展。最终，我们可能不得不切换到后端自动化来实现期望的性能。**

****基于规则的** — *完全基于规则的*程序比较少见。尽管在例外情况下可以将控制权交给人，但是 RPA 仍然针对有限范围的业务流程。缺乏情报和决策能力是爱国军应该考虑的一个缺点。**

# **现状与展望**

**几年来，RPA 的业务一直在蓬勃发展。2019 年，Gartner 将其确定为**增长最快的企业软件**，并且一直保持两位数的增长数字。现代商业仍然严重依赖重复任务的手动执行。RPA 提供了自动化任务的轻量级解决方案，释放人力资本来处理更具吸引力和认知性的问题。**

**RPA 的一个有趣分支被称为'**认知 RPA'** 。许多预期的自动化任务并不完全基于规则，然而一个智能层足以做出正确的决定。因此，RPA 和 AI 的融合开启了一个新机遇的世界。**

**从更广泛的意义上来说，RPA 是**“超自动化”**背后的驱动力，推动可以自动化的一切。为此，RPA 本身是不够的，但与事件处理和人工智能的结合可以释放其全部潜力。通过选择与 RPA 互补的技术，可以自动化的流程范围大大增加。**

**正是在与人工智能的融合中，RPA 也引起了数据科学界的兴趣。我们喜欢创建和应用机器学习算法，但我们只能在正确的生态系统中获得最佳效果。在未来的几年里，为 RPA 注入健康的智能是高度自动化领域的一项艰巨任务，这也是我们数据科学家可以发挥关键作用的地方！**

**简而言之，数字世界中 RPA 的持续激增是有原因的。RPA 可能不像成熟的自动化那样令人满意，也不像尖端的人工智能那样令人兴奋，但它在实现速赢方面非常出色。最终，这些胜利会永久改变人类和系统的互动方式。**

**![](img/cb5dbe8024df232dd9f799d711e171a6.png)**

**RPA:当软件代理处理你的无聊任务时，你可以坐下来放松一下**

# **进一步阅读**

**对于那些对 RPA 感兴趣的人来说，有许多关于这个问题的网站。精选:**

**[https://www . Cai . io/resources/thought-leadership/what-is RPA-and-why-is every one-that-talking-it](https://www.cai.io/resources/thought-leadership/what-is-rpa-and-why-is-everyone-still-talking-about-it)**

**[https://www . Gartner . com/reviews/market/robotic-process-automation-software](https://www.gartner.com/reviews/market/robotic-process-automation-software)**

**[https://www . help systems . com/blog/robotic-desktop-automation-RDA-vs-robotic-process-automation-RPA](https://www.helpsystems.com/blog/robotic-desktop-automation-rda-vs-robotic-process-automation-rpa#:~:text=The%20main%20difference%20between%20RDA,users%2C%20departments%2C%20and%20applications)**

**[https://www . robo motion . io/blog/RPA-成长最快-企业-软件/](https://www.robomotion.io/blog/rpa-fastest-growing-enterprise-software/)**

**[https://TechCrunch . com/2019/06/24/Gartner-finds-RPA-is-fast-growing-market-in-enterprise-software/](https://techcrunch.com/2019/06/24/gartner-finds-rpa-is-fastest-growing-market-in-enterprise-software/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuYWxpcnJpdW0uY29tLw&guce_referrer_sig=AQAAACqfplUUMKNgQ2XKY8OwDTx2k8bvUxbXXX1ug7J1fY48O1zOGabBUQu-YyGzlEwOqlGNHjrdZKwH6PaItadIHg1N_5Z_Jc5Ts5fFnIWoFDG-RD-b_rHP5WCJGOW3eOkrvPzM5SVh0rbVjGYoWNk1iQVzfP1G628XiOr3Wj6fg_Uk)**

**[https://www.uipath.com/rpa/robotic-process-automation](https://www.uipath.com/rpa/robotic-process-automation)**