# 拥抱面孔推断终点

> 原文：<https://towardsdatascience.com/huggingface-inference-endpoints-8984e7d8d8d4>

## 变压器模型的快速生产级部署

![](img/d08f3b7bd2e6649954883507eb7d22ee.png)

图片来自 [Unsplash](https://unsplash.com/photos/5u6bz2tYhX8) 由 [Towfiqu barbhuiya](https://unsplash.com/@towfiqu999999)

我的文章中一个不变的主题是你的机器学习模型的部署。随着机器学习越来越受欢迎，用户的模型部署选项也越来越多。[特别是 HuggingFace](https://huggingface.co/) 已经成为机器学习领域的领导者，对于数据科学从业者来说，你很可能在过去使用过[变形金刚模型](https://huggingface.co/docs/transformers/index)。

HuggingFace 与 AWS 和 Azure 都有合作关系，并提供了跨云提供商的部署选项。虽然在这些云提供商上部署 Transformers 模型是一个相对容易的过程，但它确实需要一些关于他们的生态系统的知识。HuggingFace 如何为模型托管提供生产级基础设施，同时让用户专注于他们的模型？

引入 [HuggingFace 推理端点](https://huggingface.co/docs/inference-endpoints/index)。这个托管选项仍然集成了两家云提供商提供的基础设施，但抽象出了他们的 ML 服务所需的工作，如 [Amazon SageMaker](https://aws.amazon.com/sagemaker/) 和 Azure ML 端点。

在这篇文章中，我们将看看如何旋转你的第一个 HuggingFace 推断端点。我们将设置一个示例端点，展示如何调用端点，以及如何监控端点的性能。

**注意**:对于这篇文章，我们将假设对 HuggingFace/Transformers 和 Python 有基本的了解。对于本文，您还需要创建一个 [HuggingFace 帐户](https://huggingface.co/welcome)并添加您的账单信息。确保删除您的终端，以免产生更多费用。

# 目录

1.  设置/端点创建
2.  端点调用/监控
3.  其他部署选项
4.  其他资源和结论

# 设置/端点创建

如前所述，确保创建一个 HuggingFace 帐户，您将需要添加您的账单信息，因为您将创建一个由专用计算基础架构支持的端点。我们可以到推理端点[主页](https://ui.endpoints.huggingface.co/new)开始部署模型。

使用推理端点创建，需要考虑三个主要步骤:

1.  型号选择
2.  云提供商/基础设施选择
3.  端点安全级别

要创建一个端点，您需要从[拥抱面部中枢](https://huggingface.co/docs/hub/index)中选择一个模型。对于这个用例，我们将采用一个 [Roberta 模型](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)，它已经在 Twitter 数据集上进行了调整，用于情感分析。

![](img/de7c99d5388c07eac271b33112fa2576.png)

型号选择(作者截图)

选择了端点部署模型后，您需要选择一个云提供商。对于这个实例，我们将选择 AWS 作为我们的提供商，然后我们可以看到哪些硬件选项可用于 CPU 和 GPU。

![](img/8d4ab7911a72789ecead2e6a9f6b35b8.png)

更高级的功能是设置自动缩放配置。您可以设置最小和最大实例数，以便根据流量负载和硬件利用率进行伸缩。

除此之外，在高级配置中，您还可以控制您的模型的[任务](https://huggingface.co/tasks)，源框架，以及一个[定制容器映像](https://huggingface.co/docs/inference-endpoints/guides/custom_container)。此映像可以包含您可能安装的其他依赖项或您在映像上安装的其他脚本。您可以指向 [Docker Hub](https://hub.docker.com/) 上的图像，也可以指向您的云提供商图像注册表，如 [AWS ECR。](/pushing-docker-images-to-amazon-elastic-container-registry-830c301b8971)

![](img/bdf81f0aa3b497a2cd36dc9334a84094.png)

高级配置(作者截图)

最后，您还可以定义端点背后的安全级别。对于私有端点，您必须使用 [AWS PrivateLink](https://huggingface.co/docs/inference-endpoints/guides/private_link) ，对于端到端指南，请遵循朱利安·西蒙的示例[此处](https://www.youtube.com/watch?v=ZQPm2-uR9zA)。为了简单起见，在这个例子中，我们将创建一个公共端点。

![](img/760a876a65a931c2419b82cbd62b5e97.png)

端点的安全级别(作者截图)

现在，您可以创建端点了，应该在几分钟内就可以完成配置。

![](img/0be5efd0a66ce14c5c2b5c256f3b87d8.png)

端点运行(作者截图)

## 端点调用/监控

为了调用我们的端点，推理端点 UI 通过提供一个自动化的 curl 命令使之变得简单。

![](img/a1e8d39192cc469aa1ae9952e23fa0aa.png)

测试终点(作者截图)

```
curl https://ddciyc4dikwsl6kg.us-east-1.aws.endpoints.huggingface.cloud \
-X POST \
-d '{"inputs": "I like you. I love you"}' \
-H "Authorization: Bearer PYVevWdShZXpmWWixcYZtxsZRzCDNVaLillyyxeclCIlvNxCnyYhDwNQGtfmyQfciOhYpXRxcEFyiRppXAurMLafbPLroPrGUCmLsqAauOVhvMVbukAqJQYtKBrltUix" \
-H "Content-Type: application/json"
```

使用 [curl 命令转换器](https://curlconverter.com/),我们可以获得等效的 Python 代码来测试本地开发环境中的端点。

```
import requests
import time

headers = {
    'Authorization': 'Bearer PYVevWdShZXpmWWixcYZtxsZRzCDNVaLillyyxeclCIlvNxCnyYhDwNQGtfmyQfciOhYpXRxcEFyiRppXAurMLafbPLroPrGUCmLsqAauOVhvMVbukAqJQYtKBrltUix',
    # Already added when you pass json=
    # 'Content-Type': 'application/json',
}

json_data = {
    'inputs': 'I like you. I love you',
}

def invoke_ep(headers, json_data):
    response = requests.post('https://ddciyc4dikwsl6kg.us-east-1.aws.endpoints.huggingface.cloud', headers=headers, json=json_data)
    return response.text
```

我们可以通过长时间发送请求来进一步对端点进行压力测试。

```
request_duration = 100 #adjust for length of test
end_time = time.time() + request_duration
print(f"test will run for {request_duration} seconds")
while time.time() < end_time:
    invoke_ep(headers, json_data)
```

我们可以使用[推理端点分析](https://huggingface.co/docs/inference-endpoints/guides/metrics) UI 来观察这些请求和端点性能。在这里，分析仪表板为我们提供了请求计数和延迟指标，以便我们了解我们的流量和相应的端点性能。

![](img/5c635b0e2a73d76cd425d6ff24c03cfb.png)

如果您需要调试您的端点，您也可以在 UI 上查看[容器日志](https://huggingface.co/docs/inference-endpoints/guides/logs)。在这里，我们还可以跟踪单个请求的持续时间，您在[自定义推理处理程序](https://huggingface.co/docs/inference-endpoints/guides/custom_handler)或[自定义容器映像](https://huggingface.co/docs/inference-endpoints/guides/custom_container)中添加的任何日志记录都会在这里得到反映。

![](img/391436ad461b860ff39c54baa45e5868.png)

容器日志(作者截图)

要更新或删除端点，请根据需要转到“设置”选项卡来管理您的资源。

![](img/1fdf0ed5f78fd1417670681aa0910554.png)

## 其他部署选项

在 HuggingFace 中，您也可以实现不同的托管选项。有免费的[托管推理 API](https://huggingface.co/docs/api-inference/index) ，在采用推理端点之前，您可以用它来测试您的模型。此外，还有一个 SageMaker，HuggingFace 与它紧密集成。HuggingFace 支持[容器图片](https://huggingface.co/docs/sagemaker/index)，你可以在 Amazon SageMaker 上使用它们进行训练和推理。除此之外，还有 [HuggingFace Spaces](https://huggingface.co/spaces) ，你可以利用它通过 [Streamlit](/building-web-applications-with-streamlit-for-nlp-projects-cdc1cf0b38db) 和 [Gradio](https://www.gradio.app/docs/) 框架为你的 ML 模型构建快速 UI。

## 其他资源和结论

[](https://github.com/RamVegiraju/HuggingFace-Examples) [## GitHub-RamVegiraju/hugging face-Examples:hugging face 示例/特性库/

### 此时您不能执行该操作。您已使用另一个标签页或窗口登录。您已在另一个选项卡中注销，或者…

github.com](https://github.com/RamVegiraju/HuggingFace-Examples) 

有关示例的代码，请单击上面的链接。如需进一步了解与 HuggingFace 相关的内容，请点击此处的列表。要开始使用 HuggingFace 推断端点，请遵循[官方文档](https://huggingface.co/docs/inference-endpoints/index)。我希望这篇文章对那些开始接触 HuggingFace 推断端点的人来说是一个有用的指南，请继续关注这个领域的更多内容。

*如果你喜欢这篇文章，请在*[*LinkedIn*](https://www.linkedin.com/in/ram-vegiraju-81272b162/)*上联系我，订阅我的媒体* [*简讯*](https://ram-vegiraju.medium.com/subscribe) *。如果你是新手，使用我的* [*会员推荐*](https://ram-vegiraju.medium.com/membership) *报名。*