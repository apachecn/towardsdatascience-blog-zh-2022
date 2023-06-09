# 如何让你的数据科学代码评审更好

> 原文：<https://towardsdatascience.com/how-to-make-your-data-science-code-reviews-better-13ac090560d2>

## 从主持代码评审中吸取的经验教训

![](img/c13bee6bc92f9129eeaec182509275f3.png)

Artem Sapegin 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

对于许多开发人员来说，代码评审可能是有压力的，尤其是如果你以前没有做过的话。反过来，它们在数据科学中越来越常见，尤其是当我们开始与更多面向软件的个人合作时。在不同的数据科学和工程团队中工作，让我在审查其他人的工作时，看到了哪些工作得好，哪些工作得不好。当然，并不是每个团队都以同样的方式进行代码评审，但是我想分享一些共同的经验以及我如何从这些评审中获得价值。

# 开发标准实践

无论是编程风格指南还是命名约定，您都需要为您的团队开发一套标准的编码实践。制定这一标准并经常引用它将有助于您的数据科学家符合预期。

我在我工作过的团队中发现，如果没有为人们制定的实践标准，每个人都会创建自己的标准，每个人的编码风格都是不同的。我不是说我们不能有自己的风格；看到人们编码方式的差异是很好的，但是命名约定和变量定义等项目可能会有很大的不同。因此，制定一个标准可以提供一些一致性，使代码更具可读性和可维护性。

当我作为团队领导专注于工具和 ML 实践时，我花了一个月的时间为我们的团队开发标准实践。这些实践包括所有我们期望的命名约定，创建功能性和面向对象代码的编码风格指南，以及代码评审的过程。这组文档被我们的团队很好地使用，并且经常被新的数据科学家引用。这是如何发展的基本原则。将这些文档视为您入职的起点和经验丰富的成员的参考。

查看编码标准的另一个原因包括对您将使用的编码范例的类型进行描述。我经常看到数据科学家更倾向于脚本或函数式编程。函数式编程有助于数据科学，也是我经常看到的团队采用的范例。对于我合作过的数据科学团队来说尤其如此，他们知道编码的基础，但不是软件开发人员。

# 总结您的更改

在数据科学家入职并开始开发他们的代码后，他们打开拉请求以批准他们的更改和添加。当我在以前的角色中主持代码审查时，我首先要求数据科学家总结他们的变化。这听起来可能有点奇怪，因为不是每个团队都这样做，但是概述你的变化是一个很好的反思工具。

当你坐下来打开你的拉请求时，花几分钟写一下你的代码。首先，您应该记录您对代码的添加、删除和更改。这个拉取请求解决了什么问题，你是如何解决的？如果您自己都不能回答这些问题，那么您的代码审查人员应该如何回答呢？

给自己一些思考的时间来总结你的变更，这有助于评审者理解你的代码，并给你时间来理解你在这个拉请求中所做的事情。通常，花几分钟时间反思我的工作可以让我在进行代码评审之前看到我可以做出的错误或更改。

这种类型的文档有帮助的另一个原因是为了可追溯性。例如，如果一位经理或客户问你六个月以后，你什么时候以及为什么做出了改变，你能回答吗？大概不会。彻底跟踪您的存储库中已经发生的变更，并记录这些变更为什么会发生，这为您回顾过去提供了一个很好的方法。当然，你永远不知道接下来谁会需要查看这些变化，或者为什么。因此，请准备好手头的文档。

# 分享知识和提出问题

您的代码审查可能脱机或正在开会；这取决于球队如何运行他们的。但无论形式如何，这都是分享知识和提问的时间。这是您向其他数据科学家、软件工程师和其他人学习的时间。当有人对你的代码发表评论时，确保你理解他们为什么留下评论。

*   他们会推荐一个 bug 补丁吗？
*   他们是否陈述了要遵循的最佳实践？
*   他们有没有指出一种更好的方式来格式化你的代码或者重新设计一个函数？
*   你的工作中有新的算法、技术或数据需要考虑吗？

不管他们提出了什么，让他们澄清他们的评论，这样你就可以从中学习。一旦你理解了你的评审者在要求什么，你需要知道这是否是批准拉取请求所必需的，或者是不需要实现的“思考的食粮”类型的变更。我经常看到代码评审者分享一些想法，这些想法不需要被实现，但是在某种程度上是有教育意义的，可以和你分享技术和新的想法。

不要浪费时间。相反，从中学到尽可能多的东西，并在你的下一个任务中使用这些知识。

# 不要独占所有权

我喜欢代码审查以及与其他数据科学家和软件工程师合作的部分是当人们愿意接受其他人添加到他们工作中的时候。我不认为代码应该由公司中开发它的人“拥有”。代码不是你的；这是公司，所以不要霸占所有权或领土。

每个人都应该觉得他们可以在代码库的不同部分工作，并互相学习。总有一些东西可以从研究你的代码的其他人那里学到，并且让更多的人熟悉代码库。例如，如果只有一个人拥有代码，然后他们离开了，现在你要花更多的时间去理解他们离开后的工作。当有人在那里和你一起工作，并教你他们当时为什么做出编码决策时，学习起来更容易。

当讨论 ML 算法的设计决策时，尽可能从你的队友那里学习代码是至关重要的。当您查看这些代码时，您可以向他们询问他们所创建的特性，他们使用什么数据分析来得出结论，他们如何检查输出的指标，等等。举行这些知识传授会议并向他人学习是有益的。所以不要强占所有权；相反，注重协作和创新。

# 成为代码评审员

在我工作过的团队中，并不是每个人都是代码审查者。通常，这是由高级数据科学家和软件工程师来完成的。虽然我可以理解为什么让一个更资深的成员来评审你的代码是有帮助的，但是我认为至少有一个中级或初级人员来评审代码也是很好的。让他们审查其他人的代码，并学会给出可靠的反馈。

在以前的工作中，我经常邀请一两个低年级学生加入我们的评审。我们的评审是在一个会议上进行的，个人可以亲自讨论他们的结果和代码。在一次会议结束时，一名初级数据科学家找到我，告诉我他们看到其他人编写代码感到很欣慰。这让他们意识到他们知道的比他们想象的要多，并让他们看到其他人的代码是如何实现的。结果，随着时间的推移，这个人开始对自己的编码能力更加自信，并开始根据他的学习给出反馈。

向从未做过评论的人公开你的评论是一次很好的学习经历。不要觉得他们不能做出贡献，因为他们比你新。每个人都需要从某个地方开始，越快越好。

# 最后的想法

代码审查可能是一种紧张的，有时甚至是可怕的经历，但是它们不一定是这样的。学会与你的团队合作开发一个安全开放的环境，允许协作和知识共享。

1.  为你的团队开发标准实践。这为经验丰富的团队成员提供了很好的参考，并允许新人更快入职。
2.  **总结你的变化。花时间反思你的工作可以帮助你发现错误或你可以做出的改变。这也让其他人更容易知道从你的代码中能得到什么。**
3.  分享知识，提出问题。这是你学习的时间！与你的队友分享知识和想法，并乐于提问。确保你澄清了你不清楚的评论。
4.  **不要霸占所有权。**代码不应该属于一个人。相反，每个人都应该感觉到他们可以学习并为代码库的不同部分做出贡献。培养创新和协作的文化。
5.  **成为一名代码评审员。**无论你是初级、中级还是高级，你都应该参与点评。代码评审是建立信心、学习和发展反馈能力的好方法。

什么帮助你改进了代码评审？你有什么经验教训想要分享吗？

感谢阅读！我希望你喜欢阅读我所学到的东西。如果你愿意，你可以通过使用这个链接成为一个媒介会员来支持我的写作。