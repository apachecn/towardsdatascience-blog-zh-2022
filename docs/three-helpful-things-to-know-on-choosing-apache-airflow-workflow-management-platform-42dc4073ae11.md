# 在选择 Apache Airflow 作为工作流管理平台之前，需要了解三个有用的技巧

> 原文：<https://towardsdatascience.com/three-helpful-things-to-know-on-choosing-apache-airflow-workflow-management-platform-42dc4073ae11>

## 帮助您决定选择 Apache Airflow 的建议

![](img/b5d94186d4d620241ff3e5b12cf055c9.png)

由 [Unsplash](https://unsplash.com/s/photos/pinwheel?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的 [Yaakov Winiarz](https://unsplash.com/@ywiniarz?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄的照片

[Apache Airflow](https://airflow.apache.org/) 是工作流管理平台的绝佳选择。然而，这并不意味着气流可以成为盲目的选择。在 StackOverflow 中有许多讨论，工程师提出了超出气流设计用途的问题。总的来说，气流对所有用例来说都更好。在决定最终采用气流之前，需要评估一些注意事项。本文将深入探讨在选择 Apache Airflow 工作流管理平台时需要了解的三件有用的事情。即使 Airflow 已经在您的项目中作为工作流管理工具被采用，这篇文章仍然可以帮助改进用例，以进行下一步工作或者提供一个替代的解决方案。

## 气流不是拖拽工具，气流需要编码

传统上，工作流管理工具不要求人们具有深入的编程知识。SQL 通常足以在 2000 年(2000-2010 年)的这些角色中取得成功。像微软(SSIS)和 Informatica 这样的公司提供拖放工具来满足市场这样的需求。

拖放式工作流管理工具非常易于安装。拖放工具的开发通常包括提供给用户的 ETL(提取、转换和加载)功能组件列表。

功能组件是 ETL 步骤的抽象，底层执行仍然由代码完成，但是用户不需要知道详细的实现。例如，我想将列类型从字符串转换为整数。我们将“Type Convert”组件拖放到画布上，然后将上游数据源移到这个组件框中。一旦打开“Type Convert”组件的配置窗口，我们就提供列名和数据类型，以便进行数据类型转换。然后，用户通过在功能元素之间画线来定义工作流。

在使用这个工作流管理工具时，用户通常会戴上一顶“**设计师**的帽子。拖拽工具类似于一个流程图绘制 app，不可避免的会给用户留下设计的印象。

Airflow 采用了**另一种方法，这种方法需要比拖放工具更好的编码**。

*   代码更加灵活。Airflow 也有由 Airflow 社区或第三方供应商完成的类似功能组件。然而，更强大的是，您可以编写自定义 Python 代码作为操作符/传感器，以更灵活地完成缺失的任务。虽然拖放工具允许您在一定程度上做到这一点，但它限制了您能做的事情。由于最初可以与代码中的整个管道进行交互，因此可以在气流中实现更多。
*   **代码更容易共享和协作。**代码天生就易于共享和阅读，开源的代码共享库使它比以往任何时候都不那么痛苦。然而，共享真正的可编辑拖放项目是一个不同的故事。它需要你有一个兼容的版本来打开，在适当的级别放大/缩小，有时它需要用其他工程师最初设计的工作流程的优秀布局来修复。
*   **代码不那么含糊。**阅读 Python 代码的行与观看画布上连接的多个盒子是不同的。我们可以将清晰的模式、表达性的注释和易于理解的抽象放到代码中，以减少歧义。一旦涉及到视觉效果，就变得更加困难；还有更多类似哈佛商业的例子在“[糟糕的数据可视化:误导数据的 5 个例子](https://online.hbs.edu/blog/post/bad-data-visualization)”一个人工设计的工作流可以更加模糊和误导。

如果代码完美，我们是否应该立即跳入气流？

![](img/e02dd667718554d5c5723a77f4f88638.png)

图片由 [giphy](https://giphy.com/gifs/Bounce-TV-hmm-wait-what-tilting-head-nTfdeBvfgzV26zjoFP) 提供

看你需要什么，看你团队的技术。拖放工具已经有几十年的历史了，它被广泛采用是有充分理由的。拥有一个能用 Python 读写有效代码的人需要软件工程师技能，这不同于一个精通 SQL 的人。使用拖放工具开发中小型工作流的工作量与 Airflow 相同，但是更复杂的工作流将更容易构建和调试代码。

**替代方案**:现在市面上还有很多拖拽工具。如果你使用微软，[SQL Server Integration Services(SSIS)](https://learn.microsoft.com/en-us/sql/integration-services/sql-server-integration-services)仍然是一个很好的选择。对于开源软件来说， [Apache Hop](https://hop.apache.org/) 是另一个很好的选择。它从 Pentaho 数据集成开始，并成为 Apache 基金会的数据项目之一。Talend 和 Pentaho 也是基于 GUI 的 ETL 解决方案的早期参与者。它们都为复杂的数据管道提供了丰富的特性。

## 气流不是用来流动的

“ **Airflow 不是数据流解决方案**”([此处](https://airflow.apache.org/docs/apache-airflow/stable/index.html#beyond-the-horizon)在《地平线之外》一节)从一开始就出现在 Airflow 官方文档的首页。需要用关于气流文档的两个简短段落来澄清这意味着什么，除非你已经在气流方面工作了一段时间，并且理解气流是如何工作的。

什么是数据流解决方案？ [Tyler Akidau](https://www.oreilly.com/people/tyler-akidau/) 在他著名的 [Streaming 101](https://www.oreilly.com/radar/the-world-beyond-batch-streaming-101/) 博客文章中提到了流媒体系统的如下定义:

> “一种数据处理引擎，设计时考虑了无限数据集。仅此而已。(为了完整性，也许值得指出的是，这个定义包括真正的流和微批处理实现。)."

为什么气流不适合串流？提到气流的一个原因是**任务不会将数据从一个移动到另一个**(虽然收费可以交换元数据！).为了将数据从一个作业转移到另一个作业，Airflow 不会存储这些数据，而是依赖 S3 和 GCS 等外部存储来支持这种操作。与气流不同，数据可以从一个任务中流出，而无需像 Flink、Storm 和 Spark Streaming 这样的现代流包中的强制暂存区。

另一个较少讨论的原因是 Airflow 的气流调度器的**设计。airflow scheduler 最初是按照以 ETL 为中心的思想设计的，该架构侧重于触发一个数据窗口(通常是一天，但也可能是一个小时)。气流的最小单位是 1 分钟，因为它使用 cron。然而，每 1 分钟执行一次工作流会给气流调度器带来很大压力。它需要 Airflow 用新的状态持续更新其后端数据库，并触发新的作业。在生产环境中，如果您的工作流需要在不到 1 分钟的时间内启动一个小批量，这就成了一个问题。在这种情况下，气流表现不佳；工作可以快速排队，需要赶上。**

**Airflow 更适合批处理风格的任务**，需要一个预定的间隔，不一定要 24/7 运行。如果一个工作流项目的 SLA 被限制为小规模，那么一个替代的流解决方案可能更适合这里。

**替代解决方案**:如果您正在寻找作为流式解决方案处理的数据。最好能看看[阿帕奇 Flink](https://flink.apache.org/) 、[阿帕奇 Spark 结构化流](https://spark.apache.org/streaming/)、[阿帕奇光束](https://beam.apache.org/)，或者[卡夫卡流](https://kafka.apache.org/documentation/streams/)。这些是从一开始就以流为核心概念设计的框架，适合 24/7 流用例。如果你有兴趣了解更多关于流媒体系统的工作原理，我推荐[泰勒·阿基多](https://www.oreilly.com/people/tyler-akidau/)的博客文章。

*   博文[流媒体 101:批量之外的世界](https://www.oreilly.com/radar/the-world-beyond-batch-streaming-101/)
*   博客帖子[流媒体 102:批量之外的世界](https://www.oreilly.com/radar/the-world-beyond-batch-streaming-102/)
*   《流式系统:大规模数据处理的内容、地点、时间和方式》一书

## 气流需要定期维护

虽然 Airflow 中的工作流不是流式的，但 Airflow 调度程序和 web 服务器是全天候运行的。Airflow scheduler 是一个守护进程，它不断检查新作业是否需要触发或更新。Airflow web 服务器确保 Airflow 正常运行，以允许用户查看作业状态。更值得关注的是调度器，因为它与作业调度相关。如果 Airflow 调度程序终止或停滞，将不会调度新的作业，并且已启动的作业不会更新最新状态。如果 Airflow web 服务器不正常，所有工作流仍将由 air flow 调度程序运行。唯一的缺点是用户从 UI 中看不到任何东西。这意味着你需要一个**随叫随到的时间表**，让某人看看为什么气流调度器和网络服务器在半夜不健康。

此外，将开源气流部署到生产中是一项复杂的任务。这与在数据基础设施端部署其他工具需要付出同样的努力。作为第一步，**适当的资源规划**对于长期健康的气流是必要的。

一旦您部署了 Airflow，Airflow 就应该包括在随叫随到的调度堆栈中，以避免用户的调度停机。随着更多的工作在船上，气流后端数据库也在增长。任务状态的一行不会花费一分钱，但是当长期累加所有行时，后端数据库可能会是指数级的。由于许多查询不是很有效，例如，在 Xcoms 中的`include_prior_dates`，我在另一篇文章“关于气流 Xcoms 你应该知道的 5 件事”中更详细地提到了它通过清除旧状态并在后端数据库中保留更少的记录来进行定期维护，可以帮助减少查询开销，使气流调度器更少被烧毁。

**替代解决方案**:如果你没有资源来维护团队内部的气流基础设施，领先的云提供商有一些托管解决方案:谷歌提供 [Cloud Composer](https://cloud.google.com/composer/docs/how-to/accessing/airflow-web-interface) ， [AWS 为 Apache Airflow(MWAA)](https://aws.amazon.com/cn/managed-workflows-for-apache-airflow/) 提供亚马逊托管工作流，以及公司支持的 Airflow [天文学家. io](https://www.astronomer.io/) 。因此，您可以更多地关注 ETL 管道和业务逻辑，而不是担心整个流程的停机时间。

## 最后的想法

如果您希望使用 Airflow 作为您的工作流管理平台，上面提到的关于 Apache Airflow 工作流管理平台的三个有用的东西已经通过 Airflow 为您预览了数据工程堆栈。本文将帮助您做出决定，并帮助您了解气流以及如何使用它。

请在下面评论您认为应该添加到 Airflow 上的项目，或者对该工作流管理平台有更多了解的项目。

希望这个故事对你有帮助。本文是我的工程&数据科学系列的**部分，目前包括以下内容:**

![Chengzhi Zhao](img/51b8d26809e870b4733e4e5b6d982a9f.png)

[赵承志](https://chengzhizhao.medium.com/?source=post_page-----42dc4073ae11--------------------------------)

## 数据工程和数据科学故事

[View list](https://chengzhizhao.medium.com/list/data-engineering-data-science-stories-ddab37f718e7?source=post_page-----42dc4073ae11--------------------------------)47 stories![](img/4ba0357365403d685da42e6cb90b2b7e.png)![](img/89f9120acd7a8b3a37b09537198b34ca.png)![](img/1f0d78b4e466ea21d6a5bd7a9f97c526.png)

你也可以 [**订阅我的新文章**](https://chengzhizhao.medium.com/subscribe) 或者成为 [**推荐媒体会员**](https://chengzhizhao.medium.com/membership) 也可以获得媒体上的所有报道。

如果有问题/评论，**请不要犹豫，写下这个故事的评论**或者通过 [Linkedin](https://www.linkedin.com/in/chengzhizhao/) 或 [Twitter](https://twitter.com/ChengzhiZhao) 直接**联系我。**