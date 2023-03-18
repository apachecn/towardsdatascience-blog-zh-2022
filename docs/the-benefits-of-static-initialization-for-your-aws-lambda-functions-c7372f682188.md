# AWS Lambda 函数静态初始化的好处

> 原文：<https://towardsdatascience.com/the-benefits-of-static-initialization-for-your-aws-lambda-functions-c7372f682188>

## 如何有效地编写和优化无服务器函数

![](img/7a638050ab2c128502cfe1daaf2af1a2.png)

约尔格·安格利在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

无服务器并不是一个流行词。这是用词不当。没有服务器也能运行应用程序的想法听起来像是魔术。如何才能避免让服务器全天候运行的高昂成本呢？

事实是你不能。无服务器只不过是一个服务器。它会在 5 分钟不活动后自动关闭。当需要再次调用它时，它会自动重启服务器。您只需在一个文件中指定依赖项，无服务器功能将从那些已安装的依赖项中创建一个容器映像，然后在重新启动时部署该映像和那些依赖项。

这就是为什么无服务器是一个误称。顾名思义，当在后台使用服务器时，不使用任何服务器。

但是无服务器功能仍然比实际的服务器便宜吗？这取决于你的商业案例。如果您的逻辑是轻量级的，可以放在一个函数中，那么无服务器更便宜。如果你的逻辑很重，依赖于巨大的软件包/定制操作系统/大存储，那么一个服务器就比较便宜。

> 等等，为什么我们的功能必须是轻量级的？

这个想法是，如果你的函数很小，当你需要在几分钟不活动后访问这个函数时，自动重新安装所有的包和依赖项会更容易。不活动后重新安装软件包和设置该功能的容器所需的时间称为**冷启动。**

容器将运行并在 5 分钟后关闭。如果在这 5 分钟内调用该函数，它将比在容器不活动期间调用该函数更快地返回结果。容器运行后获得函数结果所需的时间称为**热启动**。

目标是尽量减少冷启动和热启动。从用户体验来说，调用一个函数，1 分钟后等待一个结果，会很烦。

> 好吧，但是如果在大小和逻辑上受到限制，那么无服务器有什么好处呢？

你肯定不想使用无服务器功能…

*   托管网站
*   高性能计算
*   托管大型机器学习模型(神经网络、基于树的模型)
*   灾难恢复

所有这些都可以在 EC2 上托管。机器学习模型可以托管在 AWS SageMaker 上。

您可以将无服务器功能用于…

*   自动化任务
*   触发器(处理上传到 AWS S3 的对象，发送到 AWS SQS 的事件)
*   实时过滤和转换数据

因此，无服务器函数非常适合于获取数据、转换数据并将结果发送到另一个端点进行处理。它甚至可以用于将转换后的数据发送到 AWS SageMaker 上托管的机器学习模型，并从该模型中获得预测。

在我之前的一个教程中，我设计了一个 lambda 函数，将上传到 S3 桶的 PDF 文件发送到一个托管 AWS 服务的端点:Textract。

以下是该功能的要点

下面是详细介绍教程的文章。你有空的时候可以去看看。

[](https://medium.com/codex/improving-ocr-recognition-through-human-review-via-amazon-augmented-ai-a9c5f6d51c04) [## 通过亚马逊增强人工智能的人工审查来改善 OCR 识别

### 创建一个管道，在 13 分钟内通过 OCR 和人工审查从 PDF 文档中提取数据

medium.com](https://medium.com/codex/improving-ocr-recognition-through-human-review-via-amazon-augmented-ai-a9c5f6d51c04) 

> 等一下。你提到无服务器应该是轻量级的。但是在你的 *send_from_s3_to_textract* 函数中，你使用的是 python 包 *boto3。*那不是超过 50 MB，会大幅度影响冷启动吗？

正确。为此函数初始化容器所需的时间会很慢。在处理多个请求时，有一个简单的技巧可以优化函数。

> 真的吗？这是什么？

我们在函数中初始化了两个不同的 boto3 客户端**:一个 textract 客户端和一个 s3 客户端。让我们把它从**函数中移出**。参见下面的要点。**

> 等等，你刚刚把第 15-17 行的 *textract* 和 *s3 boto3* 客户端移到了第 10 -12 行的函数之外？这如何优化功能？

初始化 boto3 客户端已经是一项耗时的任务。如果它在函数处理程序内部(在本例中，在*send _ from _ S3 _ to _ text ract*函数内部)，那么客户端将在每次调用时被初始化。从冷启动的角度来看，这并不重要，因为函数和客户端都已初始化和安装。从热启动的角度来看，在容器运行时，每次调用函数都要初始化两个巨大的客户端，这是非常耗时的。

如果我们想改进，我们希望在函数处理程序之外初始化客户端。这样，该函数将在多次调用中重用客户端。虽然这不会影响冷启动，但它会大大减少热启动，因为它会获取在冷启动/环境初始化阶段已经实例化的两个客户端。

这个过程称为**静态初始化。**静态初始化是在函数处理程序中代码开始运行之前，在处理程序之外运行逻辑的过程。开发人员使用静态初始化来导入库和依赖项，初始化与其他服务的连接，并在多个 lambda 函数处理程序中重用变量。

> 好吧，但我如何优化冷启动？

这是另一个超出本文范围的话题。最简单的解释是确保提前创建 lambda 函数容器，远在按需调用它们之前。AWS Lambda 允许**提供并发**，它在可预测的时间初始化 Lambda 函数。

这对于适应突发流量和重大扩展事件非常有用。启用预配并发后，用户可以避免在 lambda 调用时遭遇冷启动。

现在您知道了如何编写和优化无服务器函数。如果有人大胆地宣称无服务器将永远取代服务器，我们是否应该一笑置之。

感谢阅读！如果你想阅读更多我的作品，查看我的[目录](https://hd2zm.medium.com/table-of-contents-read-this-first-a124146f566c)。

如果你不是一个中等收入的会员，但对订阅《走向数据科学》感兴趣，只是为了阅读像这样的教程和文章，[点击这里](https://hd2zm.medium.com/membership)注册成为会员。注册这个链接意味着我可以通过向你推荐 Medium 来获得报酬。

**参考文献:**

[](https://lumigo.io/blog/provisioned-concurrency-the-end-of-cold-starts/) [## AWS Lambda 提供的并发性:冷启动的终结

### 了解如何使用 AWS Lambda 调配的并发性来防止无服务器冷启动-了解调配的…

lumigo.io](https://lumigo.io/blog/provisioned-concurrency-the-end-of-cold-starts/) [](https://aws.amazon.com/blogs/compute/operating-lambda-performance-optimization-part-1/) [## 运行 Lambda:性能优化-第 1 部分| Amazon Web Services

### 在操作 Lambda 系列中，我为开发人员、架构师和系统管理员介绍了一些重要的主题，他们是…

aws.amazon.com](https://aws.amazon.com/blogs/compute/operating-lambda-performance-optimization-part-1/) [](https://aws.amazon.com/blogs/compute/operating-lambda-performance-optimization-part-2/) [## 运行 Lambda:性能优化-第 2 部分| Amazon Web Services

### 这篇文章是关于 Lambda 性能优化的 3 部分系列文章的第二部分。它解释了记忆的作用…

aws.amazon.com](https://aws.amazon.com/blogs/compute/operating-lambda-performance-optimization-part-2/)