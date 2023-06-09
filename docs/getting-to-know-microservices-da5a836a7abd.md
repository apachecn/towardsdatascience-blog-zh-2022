# 了解微服务

> 原文：<https://towardsdatascience.com/getting-to-know-microservices-da5a836a7abd>

## 将复杂的软件拆分成单独的服务的原因

![](img/b90c1f5924a9bce41bc0c6320f54997b.png)

照片由 [@charlybron](https://unsplash.com/@charlybron?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

微服务起源于软件架构领域，描述了一种结构，在这种结构中，软件由许多可以通过接口进行通信的独立服务组成。

# 微服务是如何工作的？

在微服务架构中，一个程序被分割成许多单独的服务，这些服务承担预定义的子任务，并且可以相互通信。这些服务是独立的，具有以下特征:

*   这些服务很少甚至没有相互依赖。
*   这些服务可以单独部署，彼此独立。这意味着技术基础设施也可以适应服务的负载分布。
*   在许多情况下，责任在于明确定义的团队。
*   可以构建微服务来解决特定的业务问题。

微服务中的这种结构通过将复杂的程序分割成较小的、可管理的服务来实现它们。此外，程序还可以通过添加新服务或删除现有服务(因为不再需要它们)来轻松地进行回溯性扩展。

# 单片和微服务架构

基本上有两种不同的方法来开发大型软件。在所谓的整体架构中，完整的功能被打包到单个服务中。代码中的各个元素紧密交织，有时甚至无法区分。例如，在 [Python](https://databasecamp.de/en/python-coding) 编程中，一个有许多函数的类可以代表一个整体。

另一方面，微服务架构使用许多代表独立功能的独立服务。为了使它成为一致的软件，函数通过 API 接口进行通信，从而交换它们的中间结果。

在汽车行业可以找到微维修和整体维修之间区别的实际例子。一家大型的结构化汽车制造商将汽车的制造分成许多不同的部门。有油漆车身的特殊团队和制造发动机的其他团队。生产的中间阶段通过装配线到达相应的部门。在软件开发中，部门会是许多不同的微服务。

另一方面，一个想修理或修复自己汽车的私人机械师表现得像一块巨石。如果发动机需要修理，他可以检查并更换损坏的零件。同样的，他也会给有划痕的部位喷漆。

# 微服务有什么特点？

当仔细观察软件时，可以通过两个基本特征来识别微服务架构。

1.  服务是**自给自足的**:各个组件完全相互独立，也就是说，它们可以毫无问题地扩展或部署，并且不会影响整个软件。自给自足的第一个标志是使用 API 进行服务间的通信。
2.  服务是**专门化的**:所有的组件都有一个他们解决的特定问题。如果问题太复杂，可以在函数中调用其他服务来帮助解决。

# 微服务相对于单片有什么优缺点？

近年来，微服务架构相对于整体架构已经获得认可，主要是因为以下优势:

*   **独立性**:开发团队可以彼此独立工作，只需要交换关于 API 接口格式的信息。所有其他决策都可以以对服务最优的方式做出。例如，这包括编程语言，因此最终的软件可能包含用不同语言编程的服务。同时，部分软件也可以由外部团队开发。
*   敏捷:服务的创建可以非常快，因为单个问题并不复杂，因此团队可以快速而简单地工作。这也意味着团队之间的协调相对较少，因为服务是相互独立的。
*   **扩展**:通过拆分成服务，计算密集型功能可以有针对性地进行扩展。另一方面，在单片架构中，整个软件必须按比例增加，这反过来会大大增加成本。例如，可以为使用[神经网络](https://databasecamp.de/en/ml/artificial-neural-networks)的个别服务配备 GPU。
*   **兼容性**:由于通过接口的简单通信，接口之间的依赖性相对较低。这意味着服务也可以用于全新的项目。
*   弹性:一旦一个服务失败，其他服务仍然可以响应，整个软件仍然可以访问。另一方面，对于一个整体来说，单一功能的故障或错误会导致整个系统的故障。因此，维护和测试的成本明显较高。

尽管微服务提供了大量优势，但在引入这种架构时，也需要考虑一些挑战。

monoliths 的优点是可以进行集中记录，即系统的状态通过一个中心点返回。在微服务架构中，必须首先建立这样的日志记录，否则只能接收关于单个服务的信息。这也使得对软件的监控变得复杂，这在中央是不可能的。

由于分布式架构，测试系统变得更加复杂，因为大量的服务是互连的。同时，服务之间的通信可能会出现延迟，从而降低整个系统的速度。如果软件响应时间是一个关键问题，微服务架构可能已经过时了。

现有系统很难迁移到这样的架构。因此，如果可能的话，应该尝试在新的开发中引入该架构，因为它会显著增加现有软件所需的工作量。

# 微服务和 Docker 有什么关系？

由于微服务是相互独立的，所以它们也可以在自己的环境中运行，只要这些环境可以通过接口相互通信。虚拟化平台 [Docker](https://databasecamp.de/en/data/docker-basics) 经常用于此目的。它使得在一台机器上启动几个彼此独立运行的所谓容器成为可能。

</beginners-guide-to-kubernetes-and-docker-c07655e43ebb>  

单个服务所需的所有信息和资源都包含在这个容器中。此外，如果服务需要，容器还可以单独缩放，即变得更强大。

# 为什么微服务在机器学习中很重要？

机器学习，尤其是深度学习，通常需要相对较大的计算能力来提供可以在生产环境中使用的快速结果。同时，ML 服务应该是可伸缩的，以处理峰值负载。出于这些原因，在云中部署经过训练的机器学习模型是有意义的。

然而，将完成的模型和机器学习管道划分到不同的微服务中是有意义的，以便尽可能地节约成本和资源。特别是神经网络和变压器可以用 GPU 相对快速地计算。但是在云中使用 GPU 在很多情况下是很昂贵的，所以只有模型计算的微服务才应该运行在 GPU 上。另一方面，预处理和后处理也可以在没有 GPU 的情况下进行。

<https://medium.com/nerd-for-tech/easy-guide-to-transformer-models-6b15c103bfcf>  

因此，在许多用例中，将复杂的模型划分为微服务以提高部署效率是有意义的。然而，具体的优势取决于许多因素，应该与每个个案中重新编程的额外努力进行权衡。

# 这是你应该带走的东西

*   微服务架构描述了一个软件概念，其中一个系统被分成许多解决子问题的独立服务。
*   与此相反的是整体，其中软件由一个结合了所有功能的大型结构组成。
*   微服务的优点是独立于单独的服务，增加了兼容性，并且易于扩展。

*如果你喜欢我的作品，请在这里订阅*<https://medium.com/subscribe/@niklas_lang>**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，medium 允许你每月免费阅读* ***3 篇*** *。如果你希望有****无限制的*** *访问我的文章和数以千计的精彩文章，不要犹豫，点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格**

*</beginners-guide-extract-transform-load-etl-49104a8f9294>  </why-you-should-know-big-data-3c0c161b9e14>  </introduction-to-apache-hadoop-distributed-file-system-99cb98d175c> *