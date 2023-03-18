# 数据科学家应该知道的 3 种独特工具

> 原文：<https://towardsdatascience.com/3-unique-tools-data-scientists-should-know-7713d2da8a81>

## 意见

## 如何将 DevOp 工具用于您的数据科学和机器学习操作

![](img/6fe1318c76b9bef666fcbb0674010397.png)

托尼·汉德在[un splash](https://unsplash.com/s/photos/three?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【1】上拍摄的照片。

# 目录

1.  介绍
2.  为什么？
3.  邮递员
4.  大牧场主
5.  詹金斯
6.  摘要
7.  参考

# 介绍

数据科学家的职位可以有很大的不同，无论是从 *only* 专注于 Jupyter 笔记本中的算法，还是成为利用数据科学模型的全栈软件工程师。在学校或在你的早期职业生涯中，可以预计你会更多地关注不同类型的算法及其适用的用例。随着你的职位越来越高，你会越来越熟悉机器学习操作和 DevOps 工具，我们将在本文中讨论。

# 谁为什么？

有一些更大的公司，如果你只在算法领域工作，实际上会更受欢迎，而且你永远不会接触某些工具。虽然还有其他规模较小的公司希望你成为全栈数据科学家，但不是从全栈软件工程师的角度，而是从能够剖析问题、找到数据、聚合数据、自动化数据收集、特征工程师、模型构建、部署以及监控和警报的角度。总而言之，不管是什么情况，了解和掌握数据科学的全部管道都是令人满足的。也就是说，这些工具不仅对所有级别的数据科学家和在面试中保持领先非常重要，而且对希望在工作中合作和完全自主的数据科学家也非常重要。

# 邮递员

![](img/9e4e454ee78511452cda4d122a1edca6.png)

Joanna Kosinska 在[Unsplash](https://unsplash.com/s/photos/mail?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【2】上的照片。

我们将讨论的第一个工具叫做[Postman](https://www.postman.com/)【3】，我称这个工具更独特，因为与其他更广为人知的工具相比，我看到的关于它和接下来两个工具的讨论和期望要少得多。

这个工具被描述为一个 API 平台。它可以用于各种用例，但由于我们希望专注于数据科学，我们可以强调您希望使用它的原因。

> 以下是您应该使用 Postman 作为数据科学家的原因:

*   发送 API 请求
*   测试您的 Python 任务
*   确保您的新模型代码将在生产中工作，而不会扰乱生产
*   检查 Python 任务是否返回预期结果
*   GitHub pull 请求的最后检查( *PR* )

数据科学家的邮差请求测试的一个例子是从 PR 测试您的生产管道代码变更。例如，如果您更新了某种类型的`preprocess.py`文件，并且您希望确保它在生产环境中能够正确工作，那么您可以发送 API 请求，并利用文件中包含的所有其他代码(比如特定的 Python 库导入)来检查您用新代码预处理的数据是否能够工作。

# 大牧场主

![](img/6cc5807bbc14804a54b07b9ab3828ac1.png)

[雅各布棉](https://unsplash.com/@jakobcotton?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在[Unsplash](https://unsplash.com/s/photos/ranch?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【4】上拍照。

在 Postman 的基础上，你可以从这个工具中获益。它被描述为一个软件栈，用于与采用容器和交付 Kubernetes 作为服务进行协作。

> 以下是您应该使用 Rancher 作为数据科学家的原因:

*   现在您已经使用了 Postman，您可以在 Rancher 中看到您的请求日志
*   希望您的新代码更改不会返回错误，但是如果返回错误，您也可以在 Pod 日志中看到它们
*   如果您有任何错误，您将能够通过错误消息来解决它们，而不会影响当前的生产过程

对于正在寻找更加自动化和简化的工作检查方式的数据科学家来说，这是一个非常好的工具。

# 詹金斯

![](img/a74bfeee0a0e0e607332bc87a8fd03ec.png)

帕特里克·托马索在[Unsplash](https://unsplash.com/s/photos/plane-taking-off?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【6】上拍摄的照片。

第三个工具也将在这个更加面向 DevOps 和机器学习操作的过程中最后使用。这个工具，[Jenkins](https://www.jenkins.io/)【7】，被描述为一个自动化的开源服务器，在这个用例中，它允许数据科学家构建、测试和部署他们的模型更改或只是一般的模型。

一旦你用 Postman 和 Rancher 测试了你的改变，你的下一步可以是使用 Jenkins。

> 以下是您应该使用 Jenkins 作为数据科学家的原因:

*   确保您可以部署 docker PROD 映像
*   构建和测试变更
*   这也可以在您将代码变更合并到 GitHub 之后自动发生

# 摘要

总之，这三个独特的工具对于数据科学家来说非常有帮助，可以确保和验证您的代码按照预期的方式工作。了解这一点也是有益的，这样你就可以更加自律。最后，它可以让你成为一个更有竞争力的申请人。

> 总而言之，这里有三个你应该知道的对数据科学家来说独特而有益的工具:

```
* Postman* Rancher* Jenkins
```

我希望你觉得我的文章既有趣又有用。如果您同意或不同意这些特定的工具，请随时在下面发表评论。为什么或为什么不？关于数据科学开发操作和机器学习操作(*或 MLOps* )，您认为还有哪些工具值得一提？这些当然可以进一步澄清，但我希望我能够为数据科学家提供一些更独特的工具、技能和平台。

***我不属于这些公司中的任何一家。***

*请随时查看我的个人资料、* [Matt Przybyla](https://medium.com/u/abe5272eafd9?source=post_page-----7713d2da8a81--------------------------------) 、*和其他文章，并通过以下链接订阅接收我的博客的电子邮件通知，或通过点击屏幕顶部的订阅图标* ***点击关注图标*** *的订阅图标，如果您有任何问题或意见，请在 LinkedIn 上联系我。*

**订阅链接:**[https://datascience2.medium.com/subscribe](https://datascience2.medium.com/subscribe)

**引荐链接:**[https://datascience2.medium.com/membership](https://datascience2.medium.com/membership)

(*如果你在 Medium* 上注册会员，我会收到一笔佣金)

# 参考

[1]照片由[托尼·汉德](https://unsplash.com/@mr_t55?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/three?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄，(2019)

[2]Joanna Kosinska 在 [Unsplash](https://unsplash.com/s/photos/mail?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片，(2018)

[3] 2022 邮差公司，[邮差主页](https://www.postman.com/)，(2022)

[4]照片由[雅各布棉花](https://unsplash.com/@jakobcotton?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在[Unsplash](https://unsplash.com/s/photos/ranch?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)(2019)上拍摄

[5]牧场主，[牧场主主页](https://rancher.com/)，(2022)

[6]帕特里克·托马索在 [Unsplash](https://unsplash.com/s/photos/plane-taking-off?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片，(2018)

[7]詹金斯，[詹金斯主页](https://www.jenkins.io/)，(2022)