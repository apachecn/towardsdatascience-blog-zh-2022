# 在 Docker 容器上运行 Jupyter 笔记本

> 原文：<https://towardsdatascience.com/running-jupyter-notebooks-on-docker-containers-47862f6b96a0>

## 与 SageMaker Studio Lab 和 Docker 合作的项目

![](img/8b7a06ba4561d325e125a5b36d944229.png)

[伊恩·泰勒](https://unsplash.com/es/@carrier_lost?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

这篇文章的目标是在 AWS 上运行数据科学工作流，然后使用 Docker 发布它，从而创建一个端到端的机器学习任务。

此外，我将更多地关注“如何”对数据科学项目进行分类，而不是“为什么这个项目很酷”。也就是说，使用 Docker 有很多[的好处:](https://www.microfocus.com/documentation/enterprise-developer/ed40pu5/ETS-help/GUID-F5BDACC7-6F0E-4EBB-9F62-E0046D8CCF1B.html)

*   轻便
*   表演
*   灵活
*   隔离
*   可量测性

另一方面，AWS [SageMaker Studio Lab](https://studiolab.sagemaker.aws/) 提供了 SageMaker 的强大功能，无需显式定义每个子流程。

# 挑选数据集

对于这个项目，我们将使用来自斯坦福网络分析项目( [SNAP](https://snap.stanford.edu/data/web-Amazon.html) )的 2，441，053 个产品的 6，643，669 个用户的 34，686，770 条亚马逊评论。这些数据是在 BSD 许可下发布的，学术和商业用途都是免费的。

使用的数据是在 Kaggle 上找到的数据的子集。它包含 1，800，000 个训练样本，200，000 个测试样本，因此每个评论要么是“正面的”，要么是“负面的”。

[](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)  

由于这是一个大型数据集，并且我们正在使用 AWS，因此数据存储在 S3 存储桶中。此外，确保您正在使用的 bucket 和 AWS 服务在同一个[区域](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html)中。对于这个项目，选择的地区是**美国东部-2** (美国东俄亥俄州)。

# IAM:了解你的角色

使用 AWS，一个常见的缺陷是没有所需的权限，或者不知道您的权限范围。转到[身份和访问管理](https://aws.amazon.com/iam/) (IAM)，创建一个角色，并为其附加以下策略:

*   亚马逊 3FullAcess
*   AmazonSageMakerFullAccess
*   awsglueconsolesagemakernotebook full access
*   AWSImageBuilderFullAccess
*   AmazonSageMakerPipelinesIntegrations
*   amazonseagemakergroundtrutheexecution

# 为什么选择 SageMaker Studio Lab？

SageMaker Studio Lab 提供了一个轻量级环境，可以比在 SageMaker 上更快地执行任务。[它也有一些缺点](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html)，例如无法访问 AWS Data Wrangler，也没有大规模的分布式培训。

要开始使用 SageMaker Studio Lab，您只需要一个 AWS 帐户，然后等待大约 24 小时，让您的帐户获得批准。一旦进入实验室，环境看起来就像一个 Jupyter 笔记本。

你可以在这里申请一个免费账户:【https://studiolab.sagemaker.aws/ 

# 开始会话

下面的代码简介是连接到您的 AWS 资源的标准方式，并定义了基本资源。可以把它看作是开始使用 AWS 的一种[方式。](https://www.in-ulm.de/~mascheck/various/shebang/)

# 预处理步骤

在 SageMaker Studio 中，第一步是编写一个预处理脚本，如下所示。这个脚本和所有后续脚本需要包含所有必需的库，并且能够独立运行。

此外，在 ML 工作流中采用的思维模式是“t *his 将在其他人的机器上运行*”。因此，在 Jupyter 笔记本环境中编写的内容能够包含在一个简洁的 python 脚本中是至关重要的。

对于 SageMaker 的入门者，记得坚持规定的目录格式:“ ***/opt/ml/…*** ”。事实上，“/opt/ml”及其所有子目录都由 SageMaker [根据文档](https://docs.aws.amazon.com/sagemaker/latest/dg/build-your-own-processing-container.html)保留。例如，“ [/opt/ml/model/](https://sagemaker-examples.readthedocs.io/en/latest/advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.html#:~:text=%2Fopt%2Fml%2Fmodel%2F,a%20compressed%20tar%20archive%20file.) ”是您编写算法生成的模型的目录。

# 培训脚本

在预处理之后是训练，就像前一部分一样包含在脚本中。特定的模型没有建立正确的工作流重要。一旦模型按预期运行，请随意插入您选择的算法。

# 模型评估

数据完全平衡:50%正，50%负。因此，精确度是一个足够的度量。精确度和召回率也包括在内，以发现预测中的任何问题。

在任何参数优化之前，准确率徘徊在 80%,精度/召回率也是如此。

# 使用 FrameworkProcessor 处理作业

SageMaker 处理模块允许我们使用一个[框架](https://sagemaker.readthedocs.io/en/stable/api/training/processing.html#sagemaker.processing.FrameworkProcessor)处理器来添加依赖项，该处理器允许一组 Python 脚本作为处理作业的一部分运行。此外，它允许处理器访问一个目录中的多个脚本，而不是只指定一个脚本。

# 运送到码头

现在我们的工作流程运行顺利，是时候发货了！首先在实验室中创建一个名为 docker 的目录，并编写 docker 文件来创建处理容器:

下一步是使用 docker 命令构建容器，这会创建一个弹性注册容器(ECR)并推送 docker 映像:

**附言**如果你使用的是苹果 M1 电脑或更新的电脑，确保在构建 docker 镜像时明确地调用你的平台。

# 运行 Docker 容器

首先我们调用 ScriptProcessor 类，它允许您在容器中运行命令，现在我们可以运行以前的相同脚本，只是它们在 docker 容器中。

就这样，我们成功地在 docker 容器上运行了一个处理作业。为了简洁起见，没有显示训练和评估容器，因为它们遵循相似的步骤。

# 结论

借助 AWS 服务和 Docker，我们能够将 python 代码推入容器，并将数据科学工作流产品化。

如果您有兴趣查看完整的代码和输出来验证您自己的工作，请查看下面的 Github repo:

[](https://github.com/NadimKawwa/amazon-review-docker/blob/main/README.md) 