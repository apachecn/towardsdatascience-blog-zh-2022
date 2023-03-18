# 利用人工智能打造更好的团队

> 原文：<https://towardsdatascience.com/using-ai-to-make-better-teams-85084be0ecfb>

## 使用 BERT 的简单演练

![](img/31ae27700f66524993e9bc597006d034.png)

安德鲁·莫卡在 [Unsplash](https://unsplash.com/?utm_source=medium&utm_medium=referral) 上拍摄的照片

“我精心决定了你们的搭配，这样你和你的伴侣的技能可以互补，”吉尔·赫尔姆斯博士说，他是我在斯坦福大学选修的一门写作课的教授。起初，我不相信这样的事情是可能的:即使她完全理解我们所有的技能，她怎么能优化每一个单独的配对呢？但后来我开始思考这个问题，并提出了一个极端过度设计，但有趣的解决方案。

如果你可以把每个人的技能，以及他们希望变得更好的东西，表示为空间中的向量，我们需要做的就是根据这些向量之间的相似性来匹配人们。信不信由你，使用几个标准的 python 包，这实际上并不太难。我们开始吧！

为了明确我们在做什么，我们将编写代码，要求一组学生给出他们的名字，一句关于他们技能的话，以及一句关于他们想学什么的话。然后，代码将为学生输出理想的匹配，这样学生就可以与拥有他们想要的技能的其他人相匹配。

# 组织项目

为了可读性和可重用性，我选择将函数分成两个文件(例如，如果我想将其作为 Flask API)。我们将处理数据检索的那个称为`main.py`，进行配对的那个称为`get_pairings.py`。我们还需要使用`pip`安装一些包来让代码工作。具体来说，我们将依赖三个主要包:

*   `sentence-transformers`提供最先进的预训练转换器架构，用于生成句子嵌入。通俗地说，这个包给了我们将句子转化为向量的功能。
*   `sklearn`是一个广泛有用的数据分析库。在这个项目中，我们将使用它来获得计算向量之间相似性的函数。
*   `numpy`是一个非常流行的框架，帮助 python 处理长数组数字。我们将用它来配对。

这些包都可以通过在终端中运行行`pip install sentence-transformers sklearn numpy`来安装。一旦我们完成了这些，我们就可以开始写代码了！

# 获取数据

我们的第一步是询问每个学生的名字和句子。就我们的目的而言，存储一组姓名、一组“技能集”句子和一组“所需技能集”句子效果很好。这将由`main.py`文件处理。为了让你的生活更容易，我将粘贴所有的代码在这里，并解释后。

首先，我们从`get_pairings.py`导入代码，我将在下面详细解释。然后，我们编写一个名为`pair`的函数，它使用 python 的内置`input`函数从命令行获取字符串形式的用户输入。我们最终遍历每个学生，建立上面描述的数组(我们称之为`knows`、`wants`和`names`)。然后，我们将这些数组传递给我们导入的`run`函数，以获取学生姓名列表。这个返回值中的每个内部列表都包含一对学生。然后我们拿着这个列表，把每一对都打印出来。我们现在已经有了程序的主要结构，剩下唯一要做的事情就是弄清楚如何进行配对！

# 配对

我们需要编写一个函数，接受上面收到的三个列表，并将它们转换为优化的配对。如果我们从头开始做这些，可能需要成百上千行代码。幸运的是，我们能够使用上面描述的包来构建其他人的代码。我希望你会发现结果相对简单。我的`get_pairings.py`文件的代码粘贴在这里:

我们有四个功能要分解。让我们从最后一个(也是最重要的)开始:`run`函数。它接受代表学生的三个数组，并返回包含最佳配对的列表列表。它首先加载一个模型(特别是 BERT，由 Google 的研究人员创建的最先进的 transformer 模型)。然后，它运行句子列表，通过模型进行分析，以接收“嵌入”，这是每个句子的功能向量表示。然后，它计算一个包含这些向量的余弦相似性的表。这可以被认为是每个向量之间的角度，如果你在空间中画出它们的话。

其他三个函数各自处理这个表，以便将这些相似之处转化为良好的配对。首先，表格被`avg_scores`函数“平均”。假设比利的技能与鲍勃想要的技能完全匹配，但比利想要的技能与鲍勃的实际技能有很大差异。这个平均函数将用两者的平均值替换这些相似性的相应表条目。因此，如果 Billy-desires 和 Bob-skill 的匹配是 1，但是 Bob-desires 和 Billy-skills 的相似性是 0，那么每个都将变成 0.5。

接下来，`get_pairs`函数成对读取表格。它通过查找表中的最高平均相似性得分，将它们存储到一个数组中，然后将该配对中每个人的所有相似性得分设置为负无穷大，这样该组的成员就不会被放入另一组。它一直这样做，直到找到所有可以配对的线对。最后，它获取这些对的索引，并将它们转换成由 names 数组给出的名称，并将结果返回给`main.py`或使用它的任何其他客户端进行处理。

# 结果

最终，工作函数应该能够做到这一点。

```
$ python3 main.py
Enter how many people you would like to pair off: 4
Enter the name of student 1: Daniel
Enter one or more sentences describing Daniel’s skillset: I like art.
Enter one or more sentences describing Daniel’s desired skillset: Something relating to water, boats, or swimming.
Enter the name of student 2: Gaby
Enter one or more sentences describing Gaby’s skillset: I know a lot about the ocean and am a diver.
Enter one or more sentences describing Gaby’s desired skillset: I would like to know about history.
Enter the name of student 3: Jade
Enter one or more sentences describing Jade’s skillset: Computer Science and math.
Enter one or more sentences describing Jade’s desired skillset: Economics is very cool to me.
Enter the name of student 4: Ali
Enter one or more sentences describing Ali’s skillset: I love studying the economy and monetary policy.
Enter one or more sentences describing Ali’s desired skillset: Quantitative fields like science and math.
Here are the pairings:
1: Jade and Ali
2: Daniel and Gaby
```

我喜欢这样的东西，因为它显示了将最先进的人工智能用于日常项目和用例是多么容易。我希望一些人可以从中吸取教训，开始将 AI 带给世界各地更多的人。