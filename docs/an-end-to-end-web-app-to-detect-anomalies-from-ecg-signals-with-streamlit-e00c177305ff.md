# 利用 Streamlit 从 ECG 信号中检测异常的端到端 Web 应用

> 原文：<https://towardsdatascience.com/an-end-to-end-web-app-to-detect-anomalies-from-ecg-signals-with-streamlit-e00c177305ff>

## 本教程着重于用 MLflow、Sagemaker 和 Streamlit 构建一个 web 应用程序

![](img/efc0609e552af5e775207f931a52d93b.png)

迈克尔·芬顿在 [Unsplash](https://unsplash.com/photos/0ZQ8vojHNuc) 上的照片

这篇文章是关于如何使用 DagsHub+ML flow+AWS Lambda 部署你的 ML 模型的故事[。在前面的故事中，我展示了如何训练和部署一个模型来从 ECG 信号中检测不规则的心律。基准数据集是](/how-to-deploy-your-ml-model-using-dagshub-mlflow-aws-lambda-c85e07b06ef6?sk=30e3b1a636acd59d51f03cc7fc9e9459) [ECG5000 数据集](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000)，它包含从一名心力衰竭患者身上随机选取的 5000 次心跳。它经常被用在研究论文和教程中。

这一次，我将使用另一个真实世界的数据集，它更嘈杂，因此也更具挑战性。该数据包含时间戳、患者 id 和心率等字段。除了这些特征，还有一个标签告诉你心律是否异常，是通过人工标注得到的。

在本教程中，我们将构建一个检测心脏异常的 web 应用程序。在创建应用程序之前，我们将进行一些探索性分析，建立和比较不同的机器学习模型。我们开始吧！

# 我们试图解决的问题是什么？

![](img/4cf44fac9397d10ce96aec1f46a68853.png)

作者插图。

我们将再次关注从 ECG 信号中检测异常。上面是患者的 ECG 信号，其中 x 标记代表异常。从这个例子中，你可以猜到它不仅是异常检测，而且是峰值检测。因此，我们需要识别峰值，并确定这些峰值是否异常。

为了遵守异常检测公式，训练集仅包含正常 ECG 信号，而测试包含正常和异常信号。还有一个重要的考虑因素:异常构成少数类，并对应于峰值。事实上，测试集包含不到 1%的异常

基于这些原因，单靠心率不足以解决这个问题。除了心率，我们还需要创建两个新特性。首先，我们构建一个变量来计算心率的当前值和先前值之间的差值。另一个至关重要的特性是峰值标签，如果有峰值，它的值等于 1，否则返回 0。

**目录**

*   **第 1 部分:模型训练和物流跟踪**
*   **第 2 部分:使用 Amazon SageMaker 部署 MLflow 模型**
*   **第 3 部分:用 Streamlit 创建一个 Web 应用程序**

## **主要要求**

*   创建 GitHub 存储库。
*   登录 DagsHub，点击“+ Create”按钮，选择“Connect A Repo”选项。它允许你通过 GitHub 和 DagsHub 来管理你的库。
*   这创建了一个 DagsHub 存储库，需要它来跟踪 ML 实验和版本化数据。
*   在本地 PC 上克隆存储库。建议将 Visual Studio 代码作为 IDE。
*   使用`pip install dvc`安装 DVC。
*   在我们的项目中运行`dvc init`。

## 第 1 部分:模型训练和 MLflow 跟踪

就像在之前的教程中一样，我们将使用一个令人敬畏的开源平台来跟踪实验，打包机器学习模型，并在生产中部署它们，称为 MLflow。它与 DagsHub 一起使用，dag shub 允许您在存储库上找到所有结果实验，并有效地对数据和代码进行版本控制。除了这些特性，您还可以在 DasgHub repository 上可视化整个管道的交互式图形。

![](img/83917a9a2589bcbbe77ebbddc630d888.png)

这次，我考虑了两种检测心脏异常的模型:自动编码器和隔离森林。这一选择是因为这项任务非常具有挑战性，而且与正常观测相比，异常现象的数量非常少。由于其特有的假设:异常代表少数类，并且在隔离树上具有短的平均路径长度，隔离林被证明比自动编码器具有更好的性能。

此外，在训练算法之前，它设置超参数的值，该值对应于数据集中异常的比例，也称为污染。隔离林表现最好，污染率小于 1%，如 0.4%和 0.5%。autoencoder 发现这项任务更成问题，这并不奇怪，因为它没有这些类型的假设。

在脚本中，我们训练两个可用模型中的一个，并记录超参数和指标。我们还对将模型记录为工件并在模型被训练后注册它感兴趣。这两个操作可以使用`mlflow.sklearn.log_model(sk_model=,artifact_path,registered_model_name)`合并。如果还是不想注册模型，可以避免指定`registered_model_name`参数。

*你可以在这里* *找到 train.py* [*的完整代码。*](https://dagshub.com/eugenia.anello/anomaly_detection_ecg_part2/src/ecg_ad/src/train.py)

我们可以运行代码并尝试超参数的不同组合，例如 model_name (if 或 Autoencoder)、使用隔离林时的污染率、epochs 数以及切换到 Autoencoder 时的批量。

```
python src/train.py
```

我们可以在实验框中访问实验的结果，它存在于 DagsHub 存储库中。从测试集上的模型评估中获得的结果，您可以注意到自动编码器获得了非常小的精度和 f1 分数值，而召回率很高，等于 88%。这意味着假阳性的数量增加，然后，自动编码器发现异常甚至正常的模式。

与 autoencoder 不同，Isolation Forest 获得了更好的分数，63%的准确率和 85%的召回率，导致 f1 分数等于 73%。即使仍有一些误报，但考虑到这个问题具有挑战性，结果超出了预期。

![](img/c37d5600919bf5501bf74929f97be371.png)

为了更好地理解所获得的评估方法，让我们也将真实值与预测值进行对比。这是使用 Autoencoder 对患者的 ECG 信号进行编码后得到的曲线:

![](img/173cbb9f122cf713e8e23c2200263567.png)

这与隔离林获得的相同地块相比较:

![](img/0eec2932e6da25312eabc7f54013d7bf.png)

绿点代表预测标签，而红叉代表地面真相。很明显，自动编码器似乎发现了异常的大量观察值，而隔离森林考虑了最异常的样本。

## 第 2 部分:使用 Amazon SageMaker 部署 MLflow 模型

由于隔离林实现了最佳性能，我们将只关注此算法。它可以分为两步:

*   MLflow 模型的更新阶段
*   创建 AWS 帐户并设置 IAM 角色
*   将 MLflow 模型部署到 Sagemaker 端点

## 1.MLflow 模型的更新阶段

从 DagsHub 页面，可以访问 MLflow 服务器用户界面。我们只需点击右上角的远程按钮，然后选择“转到 MLflow UI”。之后，我们可以按下菜单中的型号选项，进入注册型号页面。然后，我们单击您注册的模型的最新版本，并将 Stage 参数从 None 设置为 Staging。

![](img/270d1512400b4349070e0983111fe3be.png)

除了手动执行此操作，您还可以直接使用 python 脚本:

现在，我们更新了模型的阶段，我们准备切换到下一步！

## 2.创建 AWS 帐户并设置 IAM 角色

在部署模型之前，需要考虑一些要求:

*   在 [AWS](https://aws.amazon.com/free/?trk=74082305-452d-479a-b9dd-603d703a3cd2&sc_channel=ps&s_kwcid=AL!4422!3!455721528479!e!!g!!aws&ef_id=Cj0KCQjwqoibBhDUARIsAH2OpWhC0b_QeLOL9BiMIoAI3s19JsmgH_jC97bLobePgIbXffUQzOvbd7AaAuslEALw_wcB:G:s&s_kwcid=AL!4422!3!455721528479!e!!g!!aws&all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=*all) 中注册账户
*   进入 IAM →用户→ **添加用户**。选择用户名，并选择访问密钥-编程访问作为 AWS 凭据类型。
*   如果您没有 SageMaker 组，请点击**创建组**。选择组名，添加两个策略:`AmazonSageMakerFullAccess`和`AmazonEC2ContainerRegistryFullAccess`。
*   之后，我们可以进入角色→ **创建角色**。选择 AWS 服务和 SageMaker。Sagemaker 策略将自动附加到角色。

![](img/0c43dee71800dd584194b99527f19d13.png)

作者截图

*   设置 AWS CLI 界面。在终端上运行`aws configure`,它会询问您 AWS 访问密钥 ID、AWS 秘密访问密钥、默认区域名称和默认输出格式。欲了解更多信息，请点击查看[。这是 AWS 文档中显示的一个示例:](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html)

```
$ **aws configure** 
AWS Access Key ID [None]: ***AKIAIOSFODNN7EXAMPLE*** 
AWS Secret Access Key [None]: ***wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY*** 
Default region name [None]: ***us-west-2*** 
Default output format [None]: ***json***
```

我想强调的是，这些已配置的凭据对于下一步至关重要。

## 3.将 MLflow 模型部署到 Sagemaker 端点

第一步是构建`mlflow-pyfunc`映像，并使用 MLflow CLI 将其推送到 SageMaker:

```
mlflow sagemaker build-and-push-container
```

![](img/37bd6716fcb9a10918462f0cf541cdf9.png)

你可以在 AWS 账户和 Docker 桌面上找到生成的图片。

![](img/1bd0e5f54bcae5e633b4cb5faadad072.png)

之后，我们可以使用 python 脚本通过`mlflow.sagemaker.deploy`将 MLflow 模型部署到 Sagemaker。您可以在 [mlflow 文档](https://www.mlflow.org/docs/latest/python_api/mlflow.sagemaker.html)中找到更多信息。

这将导致创建一个 Sagemaker 端点。

![](img/da84e8e8254338497c62c959c1e6e3aa.png)

您应该获得这个最终输出。如果出现错误，请检查是否正确指定了 image_url、model_uri 和 arn。

我们可以快速检查端点是否在工作。选择患者的 ECG 信号，并将其传递到部署的模型，该模型会将观察结果分类为正常或异常。

需要注意的是，当观察异常时，隔离林返回-1，否则返回 1。为了与地面实况进行比较，我们需要映射这些值，1 表示异常，0 表示正常。如果您运行该脚本，您应该获得如下输出:

## 第 3 部分:创建 Web 应用程序

![](img/f3cef6669ecf1576ba0d533c09f03367.png)

让我们最终构建一个 web 应用程序来检测 ECG 信号中的异常！我们将使用 Streamlit，这是一个免费的开源框架，允许使用 Python 用几行代码构建应用程序:

*   我们建立了到通过 AWS Lambda 获得的 API 服务的链接，其中已经部署了模型。
*   使用`st.markdown`显示网络应用的主标题。
*   在上传一个 CSV 文件来评估模型的性能时，使用`st.sidebar`创建一个左侧面板侧栏。
*   按钮“检查异常！”需要单击以显示部署模型的结果。

如果上传文件并选择按钮，将出现一个代表患者 ECG 信号的散点图。像以前一样，它将真实值与预测值进行比较。除了这些特性之外，您还可以通过以下代码更改患者 ID 的值

在 GitHub 中推送所有更改后，我们可以使用 Streamlit 部署应用程序。这非常简单明了。如果你想了解更多信息，点击这个 [YouTube 视频](https://www.youtube.com/watch?v=kXvmqg8hc70&ab_channel=DataProfessor)的链接。我部署的应用程序的链接是[这里](https://eugeniaring-anomaly-detection-ecg-part2-srcecg-app-xwxtyw.streamlit.app/)。

## 最终想法:

恭喜你！您已经完成了这个专注于检测 ECG 信号异常的项目。当从一个工具切换到另一个工具时，一开始可能会感到不知所措，但是当你达到结果时，它会给你带来满足感！特别是，一个网络应用程序可以是一个可爱和直观的方式来与其他人分享你的工作。感谢阅读。祝您愉快！

查看我的 DagsHub 存储库中的代码:

<https://dagshub.com/eugenia.anello/anomaly_detection_ecg_part2>  

你喜欢我的文章吗？ [*成为会员*](https://eugenia-anello.medium.com/membership) *每天无限获取数据科学新帖！这是一种间接的支持我的方式，不会给你带来任何额外的费用。如果您已经是会员，* [*订阅*](https://eugenia-anello.medium.com/subscribe) *每当我发布新的数据科学和 python 指南时，您都会收到电子邮件！*