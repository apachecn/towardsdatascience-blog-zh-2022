# 2022 年导航 MLOps

> 原文：<https://towardsdatascience.com/navigating-mlops-dc2a242ef7ed>

## 生产中的数据科学:经验、工具和框架

![](img/95cf2d3e4430b01c8795f7d911bed615.png)

劳拉·奥克尔在 [Unsplash](https://unsplash.com/) 上的照片

[MLOps](https://ml-ops.org/) 已经在机器学习、数据科学、软件工程和(云)基础设施的交叉领域确立了自己的独立地位。在这篇文章中，我想看看机器学习/数据科学在生产中的现代方法、经验教训和实践经验。

# 数据环境变化很快

我刚开始做机器学习的时候，2014 年我还是大数据工程师的时候，大部分是在大数据的背景下应用的。机器学习或数据科学并不新鲜，但随着 Hadoop 中的 MapReduce 和后来的内存引擎(如 Apache Spark)将机器学习的能力与分布式计算和海量(web)数据的能力联系起来。我们已经看到了许多重大转变，从(本地)Hadoop 生态系统的兴衰开始(并非最后是因为巨大的管理成本),以及云中数据处理的持续趋势，我们可以肯定只有一件事是一致的:变化。

不仅基础设施变了，框架和方法也变了。从 TensorFlow 到 Pytorch，从 CRISP-DM 到 Agile，从 SOAP 到无服务器。很难跟上潮流。

# MLOps Buzz

如果你看看人工智能初创公司和咨询公司做出的承诺，以及真正适用的承诺，两者之间存在很大差距。虽然他们中的一些人几乎没有应用任何人工智能，但他们中的大多数人都因为坏的、丢失的或无用的数据而失败。然而，近年来许多公司已经开始成功地应用数据科学，并且有许多模型等待投入生产。这带来了新一波的 MLOps 创业公司，他们现在再次承诺提供一个适合所有人的解决方案。

# 没有人能统治他们所有人

当从定制的 MySQL 数据仓库迁移到云并评估许多不同的产品时，只有一个结论:没有适用于所有东西的平台，其中大多数都在为非常狭窄的用例工作。对于 MLOps 也是如此。因此，不要认为没有 ML 基础设施团队就可以将模型投入生产。

但是当然，有一些优秀的工具可以解决特定的用例。而且你不需要在投产前建立完整的基础设施，你也不应该。但是您需要相应地扩展您的基础架构。

因此，我将在这里详细说明，并直接说出供应商或开源框架的名称，而不涉及其中任何一个。

# 语言

只是一个简短的提醒，每一个代码都应该遵循软件工程的原则。这意味着测试 CI/CD、项目结构和编码最佳实践。我在这里不关注这个，因为它与软件工程没有什么不同。

## 计算机编程语言

当然，我推荐使用 pandas、scikit-learn、seaborn、tensorflow、pytorch、 [huggingface](https://huggingface.co/) 、jupyter 的 Python 堆栈——凡是你能想到的。然而，从 MLOps 的角度来看，更重要的是该堆栈的部署。如果你有轻量级的脚本，我建议使用 AWS Lambda，如果它变得更高级，我更喜欢运行云服务器(如 AWS EC2)并在 [Docker](https://www.docker.com/) 中隔离一切。将[地形](https://www.terraform.io/)用于基础设施是明智之举。我也推荐使用 anaconda，因为他们的库是有管理的，对于专业人员来说非常便宜。

## 稀有

有时你可能需要生产 R 脚本。在这种情况下，要么选择托管服务，要么使用 Docker 来隔离环境。不要在裸机服务器上安装 R 并试图在那里运行生产脚本，你会经历地狱，因为 R 肯定不是生产级语言。并了解 R 库的安全含义。

## Java 语言(一种计算机语言，尤用于创建网站)

如果你身边有 Java，就像许多公司一样，使用 java ml 框架可能更容易。在许多情况下，将您的模型直接集成到您的软件环境中是非常有意义的。但是你需要会用 Java 编码的人，或者需要会“移植”代码的工程师。虽然我个人不喜欢 [WEKA](https://www.cs.waikato.ac.nz/ml/weka/) 并会避免 [deeplearning4j](https://github.com/eclipse/deeplearning4j) ，但我真的很喜欢使用 [Smile-ML](https://haifengl.github.io/) 。

# 框架和产品

## 实验跟踪

周围有很多不同的工具，但我个人喜欢 [Neptune.ai](https://neptune.ai/) 。他们称自己为 ML-Ops 的元数据存储库，这相当准确。你可以免费使用它们，甚至在一个小团队中，并跟踪你的实验和保存你的模型，支持许多不同的框架，如 scikit-learn，TensorFlow，Pytorch，R…

## 部署脚本与部署深度学习模型

如果你部署深度学习模型，我真的会建议你使用模型服务器。可以是 [Triton](https://developer.nvidia.com/nvidia-triton-inference-server) ， [tensorflow serve](https://www.tensorflow.org/tfx/guide/serving) ，或者你喜欢的任何东西，但是要用模型服务器！如果你没有定制层，使用 [ONNX](https://onnx.ai/) 作为格式，如果你有定制层，使用框架的模型服务器来避免不必要的工作。[这里](/how-to-not-deploy-keras-tensorflow-models-4fa60b487682)我写的更详细，重点是 TensorFlow。

如果您为批量预测部署脚本，那么简单的方法就是在 Docker 中运行它们。对于 API，你可以像之前说的那样在 Docker 容器中运行无服务器产品，如 [AWS Lambda](https://aws.amazon.com/de/lambda/) 或 [FastAPI](https://fastapi.tiangolo.com/) 。

## 模特培训

有了 terraform，你还可以在 GPU 实例(AWS、GCP、Azure)上的云中自动进行模型训练。然而，我个人喜欢在我的开发机器上有一个强大的 GPU 来快速试验东西和评估模型。在本地有一个 GPU，这对于原型来说要快得多。还要注意，有特定的 GPU 实例用于推理。

## 管弦乐编曲

对于编排来说， [AirFlow](https://airflow.apache.org/) 或 [Prefect](https://www.prefect.io/) 似乎是不错的选择，但是 Prefect 要求你在每台服务器上安装一个代理(管理员通常不喜欢这样)，在 AirFlow 中你可以使用 SSH。

## 监视

根据我的经验，监控是高度定制的，但我个人喜欢的是用于时间序列监控的 [Kibana](https://www.elastic.co/de/kibana/) ，它在付费版本中提供了开箱即用的异常检测。一般来说，监控模型预测(计数、分布)、训练结果和特征特性之类的东西是明智的。NeptuneAI 还涵盖了与模型训练相关的指标。

## SQL 和矢量数据库

我很喜欢使用 MySQL 或 PostgeSQL 这样的纯 SQL 数据库。然而，在某种程度上，使用数据仓库是明智的，因为在 2022 年没有人会构建 Hadoop 集群，所以你最好看看像 [Snowflake](https://www.snowflake.com/) 这样的云数据仓库，它提供了很多高级功能，特别是针对 ML 和 DS ( [Snowpark](https://www.snowflake.com/snowpark) )。

但还有更多，如果你使用一些相似性搜索引擎(如视觉相似性)，你可能要调查向量数据库。直截了当的方法是使用 [HNSWlib](https://github.com/nmslib/hnswlib) (支持包括 Java 在内的许多语言)或[惹恼来自 Spotify 的](https://github.com/spotify/annoy)，其中的索引只是一个文件。但是还有更高级的选项，比如 [Milvus.io](https://milvus.io/) 。

## 项目管理

不要用 Scrum，用看板，但这是我的看法。详见[此处](https://medium.com/towards-data-science/machine-learning-projects-and-scrum-a-major-mismatch-c155ad8e2eee)。

# 推理设置

最后，让我们快速讨论一下推理设置。通常，很多模型不需要提供实时预测。如果不需要，引入额外的堆栈是没有意义的。以我的经验来看，有三种推理模式。**批量预测**、**实时预测**和**在线学习系统**，其中最后一个是一个例外。我从未真正将在线学习系统投入生产，所以我不能真正谈论它，但至少我可以说这是最复杂的场景，因为它是完全自主的，你需要大量的监控来确保模型不会失败，就像[微软向我们展示的](https://www.theverge.com/2016/3/24/11297050/tay-microsoft-chatbot-racist)。

与批处理相比，实时推理增加了许多复杂性。您需要确保应用所有的软件工程实践来构建一个可靠的、可伸缩的系统。这就是为什么我建议使用模型服务器，因为它们是为这种用例而设计的。

批处理要简单得多，因为如果它失败了，可以很容易地重复，你不需要考虑延迟，通常你只需要输入->输出，不需要太多的网络参与。

我希望这能给你一些启发，帮助你理解 MLOps 是什么。一如既往，这里有很多观点，有其他经历也没关系，但我很高兴在评论中听到你的。