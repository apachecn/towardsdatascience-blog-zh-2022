# SageMaker Python SDK vs Boto3 SDK

> 原文：<https://towardsdatascience.com/sagemaker-python-sdk-vs-boto3-sdk-45c424e8e250>

## 在 Amazon SageMaker 中使用哪个 SDK 以及在哪里使用

![](img/79e047e2ad2c3ebeaf73d4f8ad85e820.png)

图片来自[的](https://unsplash.com/@hiteshchoudhary)[Unsplash](https://unsplash.com/photos/D9Zow2REm8U)Hitesh Choudhary

[SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/) 和 [Boto3 Python SDK](https://aws.amazon.com/sdk-for-python/) 经常导致与亚马逊 SageMaker 的许多混淆。至少在我开始使用的时候是这样，直到我获得了更多使用这项服务的经验。

让我们快速地在两者之间建立一个清晰的区别，然后深入一些例子。SageMaker Python SDK 是一个[开源的](https://github.com/aws/sagemaker-python-sdk/blob/24a4bc96c49dc85d7a44aff5c0ca175ec75ad025/doc/index.rst)库，专门用于在 SageMaker 上训练和部署模型。该软件包旨在简化 Amazon SageMaker 上不同的 ML 流程。

另一方面， **Boto3 SDK** 是用于 AWS 的**通用 Python SDK** 。您也可以使用这个 SDK 与 SageMaker 之外的任何 AWS 服务进行交互。如果您更喜欢使用 [Python](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html) 之外的语言，那么在流行的语言中有许多具有相同 API 调用的相应 SDK，比如 [Java](https://docs.aws.amazon.com/AWSJavaSDK/latest/javadoc/com/amazonaws/services/sagemaker/AmazonSageMaker.html#createTrainingJob-com.amazonaws.services.sagemaker.model.CreateTrainingJobRequest-) 、 [Go](https://docs.aws.amazon.com/sdk-for-go/api/service/sagemaker/) 等等。

# 什么时候用什么

## SageMaker SDK

这两个 SDK 可以用于相同的任务，但在某些情况下，使用其中一个比使用另一个更直观。例如，SageMaker as a service 为 Sklearn、PyTorch 和 TensorFlow 等流行框架提供预构建和维护的映像。您可以使用 SageMaker SDK[检索这些图像](https://aws.plainenglish.io/how-to-retrieve-amazon-sagemaker-deep-learning-images-ff4a5866299e)，而不必担心使用 Docker。它们还通过在这些框架内为训练提供现成的评估器来帮助去除任何图像处理或维护。请看下面的例子，了解使用 SageMaker SDK 和支持的框架进行训练是多么容易。

使用为 TensorFlow 提供的 SageMaker 估算器进行培训

从您用于培训的评估器直接部署也非常容易。SageMaker SDK 通过从您传递给评估器的参数中进行推断，负责模型和端点配置的创建。您可以使用简单的“部署”API 调用直接部署到端点。

直接部署到端点

总之，当您有 SageMaker 支持的框架时，使用 SageMaker SDK 是一个很好的实践。除此之外，如果您在 SageMaker 上使用这个框架进行培训和部署，那么通过使用 SageMaker SDK 就非常简单了。

## [Boto3 SDK](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html)

另一方面，有时你有预先训练好的模型或者你可能使用的不同框架。这需要更多的定制，而 SageMaker SDK 并不总是提供这种功能。为了演示这一点，我们来看一个[自带容器(BYOC)](/bring-your-own-container-with-amazon-sagemaker-37211d8412f4) 的例子，其中有一个预先训练好的 NLP 模型，您希望部署它来进行推理。在这种情况下，您有一个自己提供的图像和自己的定制框架，而不是 SageMaker 自带的。

我们有三个重要的步骤和相应的 boto3 API 调用，我们需要执行它们来部署端点:[模型创建](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.create_model)、[端点配置创建](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.create_endpoint_config)和[端点创建](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.create_endpoint)。前两个实体是通过 SageMaker SDK 和我们支持的框架抽象出来的，但是我们可以通过 Boto3 SDK 看到这些细节。

首先，我们实例化将要使用的客户端。sm_client 对象是我们将在我们详述的三个步骤中使用的对象。

模型创建

端点配置创建

端点创建

使用这个客户端，我们可以用 SageMaker SDK 提供的相同细节来配置我们的三个 API 调用。这里的主要区别是**我们可以灵活地用环境变量或不同的特性来检查和定制三个步骤**中的每一个。除此之外，我们有能力带来我们自己的框架或预先训练的模型，或者两者兼而有之。

另一个要考虑的主要因素是，当您使用更高级的 SageMaker 产品时，如[多模型端点或多容器端点](/sagemaker-multi-model-vs-multi-container-endpoints-304f4c151540)，当您处理增加的定制时，通过 Boto3 获得完全的灵活性和控制变得更容易。

总之，当您处理具有更大定制的用例时，请使用 Boto3 SDK。这个选项有更大的灵活性，随着你的 ML 平台变得更加复杂，你可以利用 Boto3 SDK

# 结论和附加资源

我希望这篇文章对这两个 SDK 之间的用法差异有所澄清。同样，两者都可以用于相同的目的，或者在很多时候结合使用，但是在特定的用例中使用一个会容易得多。我在下面附上了 BYOC 和 SageMaker SDK 示例的完整代码。

*   [BYOC 示例代码](https://github.com/RamVegiraju/SageMaker-Deployment/blob/master/RealTime/BYOC/PreTrained-Examples/SpacyNER/spacy-NER.ipynb)
*   [SageMaker SDK TensorFlow 培训&部署](https://github.com/RamVegiraju/SageMaker-Deployment/tree/master/RealTime/Script-Mode/TensorFlow/Classification)

*如果你喜欢这篇文章，请在* [*LinkedIn*](https://www.linkedin.com/in/ram-vegiraju-81272b162/) *上与我联系，并订阅我的媒体* [*简讯*](https://ram-vegiraju.medium.com/subscribe) *。如果你是新手，使用我的* [*会员推荐*](https://ram-vegiraju.medium.com/membership) *报名。*