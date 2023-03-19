# 为什么打包你的 ML 代码没有你想象的那么痛苦

> 原文：<https://towardsdatascience.com/why-packaging-your-ml-code-is-not-as-painful-as-you-might-expect-d3583439ec1d>

## 如何将您的机器学习代码及其依赖项转移到您的生产环境中。

![](img/4d58dc1677f95665673f981c2aef2374.png)

包装您的 ML 模型—用稳定扩散创建的图像

Jane 在一家成功的初创公司担任机器学习(ML)工程师。他们即将发布其产品的第一个版本，该版本严重依赖于她正在研究的 ML 算法的性能。经过几次迭代之后，她所训练的模型在一个坚持的测试集上表现得相当好，她已经准备好进行下一步了。

首先，她使用 [Gradio](https://gradio.app/) 快速开发出一个原型，并将其放在拥抱脸[空间](https://huggingface.co/spaces)上。现在，她有了一个运行中的 ML 驱动的应用程序，它有一个简单的 UI，可以与信任的朋友和同事分享。Jane 利用这一点获得反馈，并验证模型在真实场景中的性能。

</keep-your-ml-models-out-of-your-application-servers-9fe58f9c91a5>  

她现在准备开始开发过程的下一阶段；构建一个合适的推理服务器。她确信模型即服务(MaaS)方法是前进的方向，因此在她的任务列表中添加了一些新项目:

*   打包模型
*   构建一个 REST API
*   优化服务的性能
*   解决缩放问题
*   简化版本推出

</the-unnerving-sweet-spot-for-ml-powered-products-c34b54e17179>  

今天是星期一。她拿起单子上的第一项:包装模型。在这个故事中，我们研究了她的两种选择，以及为什么走得最多的路——尽管是一条崎岖的路——是甜蜜点路线。

> [Learning Rate](https://www.dimpo.me/newsletter?utm_source=medium&utm_medium=article&utm_campaign=model-packaging) 是一份时事通讯，面向那些对 AI 和 MLOps 世界感到好奇的人。你会在每个月的第一个星期六收到我关于最新人工智能新闻和文章的更新和想法。订阅[这里](https://www.dimpo.me/newsletter?utm_source=medium&utm_medium=article&utm_campaign=model-packaging)！

# 模型打包和依赖管理

ML 模型是一个数学函数，它接受一个明确定义的输入并产生一个输出。但这也是 Jane 必须打包的一段代码，然后加载到一个与她工作的环境完全不同的环境中。

模型预测依赖于她编写的代码、它的依赖关系和模型权重。所有这些工件都需要出现在推理服务器上。不幸的是，这不是一件容易的事情。您可能认为，对于 Jane 来说，启动一个 VM，在其中使用 ssh，安装所有的依赖项，复制模型的代码和权重，运行一个 flask 应用程序，并将其称为推理服务器就足够了。它不是；那么，有什么问题呢？

依赖性是很多令人头疼的问题的根源。很难在不同的环境中保持它们的一致性，甚至其中一个环境中的微小升级都会产生难以追踪的问题。

您的模型代码可能依赖于数十或数百个其他库(例如，`pytorch`、`torchvision`、`torchmetrics`、`numpy`、`scipy`等)。).您的应用服务器也有自己的依赖项(例如`flaks`、`gunicorn`等)。).众所周知，跟踪一切并维护独特的环境非常困难。Jane 必须同步她的开发、测试、试运行和生产环境。这是一场噩梦。

Jane 以前亲身经历过这些困难，所以她知道有两种选择:以不可知的形式保存模型以约束模型的依赖关系，或者将整个推理应用程序包装在一个容器中。

## 约束模型依赖关系

Jane 的选择之一是约束她的模型的依赖关系。今天，完成这项工作的主要方法是使用一个名为 ONNX(开放式神经网络交换)的库。

ONNX 的目标是成为 ML 模型的互操作性标准。使用 ONNX，Jane 可以用任何语言定义网络，并在任何地方一致地运行它，不管 ML 框架、硬件等等。

这看起来很理想，但简以前也曾上当。由于 ML 和数值库迭代速度很快，所以翻译层经常会出现难以追踪的 bug。ONNX，不管它的意图是好是坏，仍然会导致比它试图解决的问题更多的问题。

此外，正如我们所看到的，Jane 必须考虑与她的模型无关的部分。数据处理部分或者服务器的代码呢？

## 将推理代码打包到容器中

当我们谈论集装箱时，十有八九是指码头集装箱。虽然 Docker 有一点学习曲线，但入门并不困难，是 Jane 的最佳选择。

Jane 必须编写 DockerHub 文件，构建映像，并将其推送到映像注册中心，比如 DockerHub。在稍后阶段，这可以通过 CI/CD 管道实现自动化，因此在这方面投入时间会让她走得很远。Jane 现在可以将这个映像放入任何运行容器运行时的计算机中，并毫无困难地执行它。

容器和 Docker 彻底改变了我们打包应用程序的方式。它们无处不在，即使你不想深入细节，一些服务也可以帮助你将你的 ML 代码打包成标准的、生产就绪的容器。 [Cog](https://github.com/replicate/cog) 、 [BentoML](https://www.bentoml.com/) 和 [Truss](https://truss.baseten.co/) 是提供更快速运送模型的项目。

然而，我的观点是，既然您无论如何都要编写一个 YAML 文件，那么最好学习 Docker 和 Kubernetes 的语法。您也可以将这些知识转移到其他领域，帮助您对部署过程中的许多步骤有一个总体的了解。

# 结论

开发您的 ML 模型只是构建 ML 驱动的应用程序的一步。事实上，它往往是前进的许多小步中最小的一步。

当您准备打包它并构建推理服务器时，您实际上有两个选择:约束它的依赖关系或将其打包到一个容器中。

在这个故事中，我们看到，虽然 ONNX 等工具的承诺是雄心勃勃的，但将模型代码及其依赖项打包到一个容器中仍然是一条路要走。接下来，我们将看到如何使用 Docker、KServe 和 Kubernetes 来实现这一点。

# 关于作者

我叫 [Dimitris Poulopoulos](https://www.dimpo.me/?utm_source=medium&utm_medium=article&utm_campaign=model-packaging) ，我是一名为 [Arrikto](https://www.arrikto.com/) 工作的机器学习工程师。我曾为欧洲委员会、欧盟统计局、国际货币基金组织、欧洲央行、经合组织和宜家等主要客户设计和实施过人工智能和软件解决方案。

如果你有兴趣阅读更多关于机器学习、深度学习、数据科学和数据操作的帖子，请关注我的 [Medium](https://towardsdatascience.com/medium.com/@dpoulopoulos/follow) 、 [LinkedIn](https://www.linkedin.com/in/dpoulopoulos/) 或 Twitter 上的 [@james2pl](https://twitter.com/james2pl) 。

所表达的观点仅代表我个人，并不代表我的雇主的观点或意见。