# 首先，我们必须发现。然后，我们可以探索。

> 原文：<https://towardsdatascience.com/first-we-must-discover-only-then-we-can-explore-8334b764535e>

## 结构化数据发现方法的案例

![](img/9a76b3adb8fe842f74de759fc056da11.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Clarisse Meyer](https://unsplash.com/@clarissemeyer?utm_source=medium&utm_medium=referral) 拍照

B ack 在 20 世纪 70 年代，约翰·图基发表了**探索性数据分析**，通过这本书，他支持在进行假设检验之前先对我们的数据集进行试验的想法。Tukey 认为，这样做，人们可以发现关于数据和主题的新信息，并开发可能导致一些有趣结果的新假设。他写道，

> 在你学会衡量你做得有多好之前，衡量你能做什么是很重要的。

从那时起，探索性数据分析(EDA)越来越受欢迎，今天，很难找到一个不以 EDA 开始的 Kaggle 挑战提交笔记本。

如果你有足够的好奇心去阅读**探索性数据分析**(你最近已经这样做了)，你可能会发现它充满了许多过时的技术——比如如何轻松地找到数字的日志，以及如何手工绘制数据点。但是如果你勇敢地阅读这本书，或者翻阅或者思考我们已经从这些史前教程走了多远，你会发现很多有用的例子和想法。首先，你会看到 EDA 不是一个人可以执行的特定指令集——它是*思考*数据的一种方式，也是练习好奇心的一种方式。

然而，虽然图基出色地描述了 EDA 及其技术，但他的书忽略了数据分析中经常被忽略的第一步:理解主题。虽然这种想法对一些人来说是直观的，但并不是每个人在实践中都这么做。虽然跳入编程或制作美丽的可视化效果可能很有趣，但如果我们不理解数据集代表什么，EDA 可能会误导我们。因此，在开始我们的数据探索之前，我们也应该对我们的数据试图描述的主题或过程感到好奇。

**具体来说，我们应该问自己两个问题:**

1.  我们知道什么？
2.  什么是我们不知道的？

在试图回答这些问题时，我们应该能够建立一个参照系来进行我们的分析。

# 知识就是力量

当试图解决一个数学问题时，一个好的策略是*首先并且最重要的是*写下关于这个问题的所有已知信息。类似地，在数据分析中，如果我们已经有了一个计划分析的数据集，很自然地想要知道数据代表什么。如果我们还没有数据集，为了收集数据集的适当需求和理解最终目标，询问关于我们主题的问题是很自然的。在这一节中，我提出了一种结构化的方法来收集有关我们分析的事实。事实上，问题*“我们知道什么？”*可以分为三个独立的“什么”问题。

**是什么题材？**

虽然主题专业知识可以留给专家，但一个熟练的数据分析师应该调查主题，并尽可能了解关于该主题的一切。这样做的原因超出了纯粹的好奇心。理解主题有助于确定分析需要什么信息，并有助于收集特定的需求。使用现有数据集时，它在 EDA 过程中会有所帮助。此外，它可以帮助分析师避免做多余的工作。

例如，如果我们知道一家公司向公众公布季度收益，那么它可以帮助解释为什么股票价格在季度基础上经历突然的变化。当分析公司的股票价格波动时，分析师可以将此添加到已知事实的列表中，并在 EDA 过程中节省一些挖掘信息的时间。此外，分析师可以要求季度财务报表作为额外的数据要求。

**有哪些定义？**

在进行分析之前，重要的是整理一个定义和已知术语的字典。拥有一本可用的字典可以帮助发现分析中的某些细微差别，理解各种计算中涉及的逻辑，并与利益相关者进行交流。编纂一本字典也可以提出一些额外的问题和假设来帮助分析。

如果给你一个葡萄酒质量数据集([就像这个](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset))并要求你预测葡萄酒的质量，你只需要导入数据集，导入 scikit-learn，然后运行一个模型。但是如果你花时间建立一个术语词典，你就会明白，例如，“挥发性酸度”被定义为葡萄酒中低分子量脂肪酸的量度，并且与葡萄酒腐败有关。因此，如果你的模型预测它对预测的葡萄酒质量有积极的贡献，可能是时候重新审视你的模型，或者准备向你的利益相关者证明这个结果。

**底层流程是什么？**

试图收集知识的最后一步是理解什么潜在的过程控制着你的分析主题。通常，这应该使用系统分析来完成，系统分析可以识别过程，并且应该有助于提出和回答关于数据的各种问题。它也可以作为指导分析的工具。

假设你的任务是确定有助于公司发展的因素。在理解应该收集哪些数据和测试各种假设时，绘制增长路径的系统图是一个很好的起点。例如，一个树形图可以强调你的公司有三种方式可以增加其感知收入:向新客户销售，增加现有客户的收入，或者减少流失客户的数量。从这些因素，你可以开始建立一个详细的图片。首先，你可以确定你的公司获取客户的各种方式，并确定所需的数据，以验证这些因素是否重要。

# 无知是福

一旦我们对我们所讨论的主题有了更好的理解，是时候问问我们自己了"*什么是我们不知道的？*“虽然在某些情况下我们可以收集额外的数据，但这个问题的真正目的是了解我们的局限性和我们无法满足的要求。也有必要了解我们的偏见和假设，在此基础上我们将提出建议。

**信息完整吗？**

为了回答这个问题，让我们看看我们能够获得的数据或我们已经拥有的数据，并评估以下标准:

1.  **数据描绘的是整个画面还是仅仅一部分？**例如，如果我们的任务是分析来自电子病历(EPR)的中风数据集，我们只关注有记录的个人，而不是每个可能患有中风的人。这对于我们的分析和我们计划提出的任何建议意味着什么？
2.  **是否有缺失的数据点？数据集通常会有缺失的记录或数据列。通常，分析师必须设计一个策略来处理丢失的数据点。但是，该策略可能会因数据点丢失的原因而有所不同。**
3.  **数据质量好吗？**在许多情况下，数据转换、数据收集中的错误或手动输入都可能导致较差的数据质量。在使用给定的信息之前，分析师应该首先确认数据是否正确，如果可能的话，与现有来源进行交叉引用，其次，制定一个策略来降低质量差和不确定性。

比方说，你的任务是分析网站上的产品评论(像这些一样的[，以便得出趋势和见解，并影响未来的产品库存。这样的数据集依赖于已经输入其产品评论的个人。然而，并不是所有的人都这样做。那么，这个数据集能代表整个人口吗？如果我们对它执行自然语言处理，我们是期望评论的质量是好的还是应该做一些预处理以使一些条目更易读？如果不做预处理，是不是遗漏了某些信息？我们能得到任何额外的数据或结果来改进我们的分析吗？](https://www.kaggle.com/datasets/linzey/amazon-echo-dot-2-reviews)

**我们在做什么假设？**

在数据不完整或质量差且不可能获得额外信息的情况下，为了继续分析，分析师可能不得不接受它们的命运和设计策略。但是，重要的是要注意必须做出的假设，因为它们将设定分析结果解释的界限和范围。做出假设伴随着风险，决策者必须被告知他们愿意承担的风险，以便利用分析的结果。

例如，如果我们要分析一个全球范围内的幸福得分数据集([就像这个](https://www.kaggle.com/datasets/mathurinache/world-happiness-report))，我们无法获得任何额外的信息，我们必须在几个假设下工作。首先，如果我们想对所代表的国家的总体人口做出推断，我们必须假设各个国家的结果代表了它们各自的人口。这意味着参与调查的个人代表了他们各自国家的其他人(这是一个重大假设)。我们还必须假设进行调查的盖洛普世界民意测验所采用的方法不偏向特定人群或方法(这也是一个重大假设)。我们必须做出的另一个假设是，将调查翻译成不同的语言不会妨碍结果。

我们有偏见吗？

决定分析和解释分析结果的策略的最后一个问题涉及偏见——对某一观点的倾向或偏见。偏见产生于人们感知世界的方式，以及他们的经历和与他人的互动。如果不了解偏见的类型，不通过不带偏见的视角检查我们对信息的理解，我们得出的见解可能不代表现实。相反，他们可能对某个特定的观点有偏见。有偏见的分析不仅不能充分反映现实，还可能不公平和不道德。

在最后这个例子中，让我们来看看美国总统辩论记录的数据集([就像这个](https://www.kaggle.com/datasets/arenagrenade/us-presidential-debate-transcripts-19602020))。对这些数据的分析可能会受到新近偏见的影响，在新近偏见中，个人可能会更重视最近的辩论，而放弃考虑文化在历史上的变化和发展。此外，持有强烈政治观点的分析师可能会面临确认偏差，他们可能会忽略不支持其观点的证据。

这篇文章为数据发现的结构化方法提供了一个案例——一个在探索性数据分析发生之前必须发生的过程。这个过程的目标是双重的:**理解什么信息是已知的**和**评估什么信息是未知的**。回答问题有助于达到发现的目标，这使我们能够发展一种视角和透镜，通过它来解释结果。这也有助于我们理解我们的洞察力是否可以用来做决定，以及我们局限于什么样的决定。

在开始探索未知的土地之前，我们应该发现这些土地，了解它们，调查它们，并为旅程做好准备。只有这样，我们才能安全而自信地探索。

我希望你喜欢这本书！我迫不及待地想在评论区看到你的想法。

# 参考

[1]图基，J. (1977) *探索性数据分析*。爱迪生-韦斯利出版公司。