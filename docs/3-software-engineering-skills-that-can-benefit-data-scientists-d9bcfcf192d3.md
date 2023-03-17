# 3 种软件工程技能可以让数据科学家受益

> 原文：<https://towardsdatascience.com/3-software-engineering-skills-that-can-benefit-data-scientists-d9bcfcf192d3>

## 意见

# 3 种软件工程技能可以让数据科学家受益

## 软件工程和其他有助于简化数据科学工作的非数据科学技术

![](img/802d1f9079a9a1d7606e8c38da4092bd.png)

彼得·冈博斯在[Unsplash](https://unsplash.com/s/photos/software-engineering?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【1】上的照片。

# 目录

1.  介绍
2.  DevOps 部署脚本
3.  Python 中的 SQL 查询
4.  模型间的预处理自动化
5.  摘要
6.  参考

# 介绍

数据科学需要广泛的技能，大多数人在学术界遵循传统路线，如研究生学位，因此可能不会太重视软件工程。本文面向从非技术背景进入工作岗位的数据科学家，或者只是对软件工程有所帮助的其他方式感兴趣的人。对我来说，我在本科学位时没有学习工程/技术，所以我很惊讶在我的第一个数据科学角色中，我必须执行多少软件工程，以及它能有多大的帮助。数据科学完全是关于算法的，但是为了使用它们，你需要对软件工程有所精通。因此，下面，让我们讨论一下软件工程中可以帮助您改进数据科学流程的 3 个领域。

# DevOps 部署脚本

![](img/b07de4add0841fe6a238e1307cbf9e4b.png)

由 [Philippe Oursel](https://unsplash.com/@ourselp?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在[Unsplash](https://unsplash.com/s/photos/docker?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【2】上拍摄的照片。

数据科学教育中最容易被忽视的话题可能是 DevOps 和机器学习操作，这两者都使用了某种形式的软件工程。其中一个原因是因为类不会有可以将算法部署到 AWS ( *Amazon Web Services)之类的系统，可能是因为它很贵，对于较短的类风格来说太复杂了*，而且还有很多不同的工具可以使用，所以重点往往是放在你的 Jupyter 笔记本上，或者只是本地预测。

> 以下是一些软件工程技能和场景的示例，它们对一些数据科学过程至关重要:

*   当使用一个模型时，您会希望对代码进行修改，并在生产中更新模型。例如，其中一种方法是使用`Docker`，在这里您将构建并推动代码变更。虽然这听起来很简单，而且您不一定需要在本地完成，但是熟悉 Docker 文件语言，使用像`FROM`、`ENV`、`COPY`和`RUN`这样的命令是非常有益的。
*   除了脚本本身之外，您还想知道如何构建和推动代码变更。这个过程可以通过终端或 GitHub 库中的命令来完成，例如 `docker image build [OPTIONS] PATH | URL | -`。

虽然 DevOps 不完全是软件工程，但它是与软件工程齐头并进的领域之一。例如，DevOps 中不一定有本科学位，但通常是 SE 中的某个人执行这些过程，这些技能对于希望更新其模型的数据科学家来说非常有价值(在这个用例中是*)。当然，在其他情况下，数据科学家可能会在没有 docker 或 DevOps 的情况下执行不同的流程。*

# *Python 中的 SQL 查询*

*![](img/4b61bf40b62a931a795c90ea20ba9988.png)*

*[David Clode](https://unsplash.com/@davidclode?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在[Unsplash](https://unsplash.com/s/photos/python?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【3】上的照片。*

*虽然 SQL 不仅仅是一种软件工程技能，但它也被许多人使用，比如数据工程师、数据科学家、数据分析师、业务分析师和产品经理。这里的区别在于，有时您会将 SQL 和 Python 结合起来创建一个脚本来自动运行您的 SQL 查询。*

> *这一过程在以下情况下发生:*

*   *从 SQL 创建一般查询*
*   *创建从 SQL 启动任务的 Python 脚本*
*   *添加 Python 来确保任务以特定的频率运行，比如说，每周一次，或者每小时一次*
*   *添加 Python 来创建非静态值，例如，如果您想要查询一个月前的开始日期，您可以插入一个从另一个 Python 文件填充的参数，该参数指示多少个月前，比如说 1 或 12，以获取去年的数据。这样，您就不必编辑查询本身。*

*最终，您在这里所做的是使用软件工程技术来自动化和扩展您对数据科学模型数据的查询。*

# *模型间的预处理自动化*

*![](img/4b3aa8e47ef4aec4e3b84e68202b6b7f.png)*

*[Joshua Sortino](https://unsplash.com/@sortino?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在[Unsplash](https://unsplash.com/s/photos/data?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【4】上拍摄的照片。*

*最后一项技能是在面向对象编程方面纯粹使用软件工程。虽然您可以在您的本地笔记本中有更多的静态代码用于测试，这仍然是有益的，但是将运行您的模型的整个过程的不同的、有序的任务组合起来可能更简单。您甚至可以使用这种类型的编程来自动测试您的模型，以及测试准确性、错误等。)。*

*使用这种类型的编程的另一种方式是，您可以使用您的生产代码，但是在开发环境中运行它，因此您知道您的模型将如何以 1:1 的方式运行到生产中，但是不需要在生产中执行更改。*

> *以下是如何自动化数据科学建模的预处理:*

*   *如果你有不止一个模型，你可以有一个其他模型继承的基类，这个预处理文件可以用来过滤掉某些数据并在`Pandas`中创建新的特性，而不是从你的 SQL 代码中，这有时是数据科学家特别喜欢的。*
*   *当您拥有预处理文件时，您的训练任务(建模过程中的下一个任务)可以读入预处理 Python 文件中生成的训练和测试数据。*
*   *您可以创建一个流程来简化您的模型构建过程，从查询到预处理，再到训练，然后部署和预测。*

*当您的文件和模型数量开始增长时，这是非常有用的；它本质上允许您在不增加代码量的情况下提高生产率。*

# *摘要*

*如果在您的数据科学之旅中，您还没有专注于软件工程，那么现在是开始学习的时候了。如上所述，本文主要面向初学数据的科学家，并告诉您不仅需要学习数据科学本身，如算法和统计，还需要学习软件工程技术，以便使您的数据科学工作更加容易和高效。是的，一些数据科学教育、训练营和认证计划确实包含了一些这方面的内容，但是拥有一个 3 天的课程，例如 SQL 介绍和 Python 介绍，不足以成为一个更全面的数据科学家。*

> *总而言之，这里有一些非数据科学和软件工程技能和场景可以帮助您的数据科学过程:*

```
** DevOps Deployment Scripts* SQL Querying Within Python* Preprocessing Automation Between Models*
```

*我希望你觉得我的文章既有趣又有用。如果您同意或不同意这些提到的技能和用例，请随时在下面发表评论。为什么或为什么不？关于软件工程和数据科学，你认为还有哪些技能是重要的？这些当然可以进一步澄清，但我希望我能够阐明掌握软件工程如何在数据科学的许多方面帮助你。*

****我不属于这些公司中的任何一家。****

**请随时查看我的个人资料、* [Matt Przybyla](https://medium.com/u/abe5272eafd9?source=post_page-----d9bcfcf192d3--------------------------------) 、*和其他文章，并通过以下链接订阅接收我的博客的电子邮件通知，或通过点击屏幕顶部的订阅图标* *的 ***点击订阅图标，如果您有任何问题或意见，请在 LinkedIn 上联系我。*****

***订阅链接:**[https://datascience2.medium.com/subscribe](https://datascience2.medium.com/subscribe)*

*【https://datascience2.medium.com/membership】引荐链接:*

*(*如果你在 Medium* 上注册会员，我会收到一笔佣金)*

# *参考*

*[1]由[彼得·冈博斯](https://unsplash.com/@pepegombos?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/software-engineering?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片，(2019)*

*[2]照片由[菲利普·欧塞尔](https://unsplash.com/@ourselp?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在[Unsplash](https://unsplash.com/s/photos/docker?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)(2021)拍摄*

*[3]照片由 [David Clode](https://unsplash.com/@davidclode?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/python?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄，(2018)*

*[4]Joshua Sortino 在 [Unsplash](https://unsplash.com/s/photos/data?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片，(2017)*