# 使用 Terraform 在 GCP 上部署云功能

> 原文：<https://towardsdatascience.com/deploy-cloud-functions-on-gcp-with-terraform-111a1c4a9a88>

# 使用 Terraform 在 GCP 上部署云功能

## 在本教程中，您将使用 Terraform 部署一个由云存储事件触发的简单云功能

![](img/169b1e95c99705a9e4a046ce39e55708.png)

图片由[geralt](https://pixabay.com/users/geralt-9301/)**|**[Pixabay](https://pixabay.com/illustrations/smartphone-control-city-industry-4-6265047/)提供

[**云功能**](https://cloud.google.com/functions/docs/writing) 是可扩展的“现收现付”功能即服务(FaaS)来自[谷歌云平台](https://cloud.google.com/) (GCP)运行您的代码，无需服务器管理。

云函数可以用 [Node.js](https://nodejs.org/) ， [Python](https://python.org/) ， [Go](https://golang.org/) ， [Java](https://www.java.com/) ，[编写。NET](https://dotnet.microsoft.com/languages) 、 [Ruby](https://www.ruby-lang.org/en/) 和 [PHP](https://www.php.net/) 编程语言。为了可读性，本教程将包含一个用 **Python** 编写的云函数。但是请随意选择您喜欢的编程语言。

[**Terraform**](https://www.terraform.io/) 是一款基础设施即代码(IaC)开发工具，允许您安全、可重复且高效地构建、更改和版本化基础设施。它将云 API(GCP、AWS、Azure 等)编译成声明性的配置文件。

在本教程结束时，你将有一个云功能，一旦一个文件被上传到谷歌云存储桶触发。

# 先决条件

要学习本教程，您需要:

*   在你的本地机器上安装了。
*   [Google Cloud SDK 在你的本地机器上安装了](https://cloud.google.com/sdk/docs/install)。
*   一个[谷歌云平台项目](https://cloud.google.com/resource-manager/docs/creating-managing-projects)建立并附属于一个[计费账户](https://cloud.google.com/billing/docs/how-to/modify-project)。确保云函数 [API 已启用](https://cloud.google.com/service-usage/docs/enable-disable)。

然后，您可以在您的终端中运行`gcloud auth application-default login`的本地机器上通过 GCP 认证。

# 目标

本教程的目标是使用 Terraform 在 GCP 项目中进行部署:

*   上传文件的桶。
*   一个存储云函数源代码的桶。
*   每次将文件上传到第一个存储桶时触发的云函数，其源代码在第二个存储桶中。

# 项目结构

您可以创建一个新文件夹，我将它命名为`cloud_function_project`，但是请随意选择一个对您来说方便的名称。

然后创建如下定义的文件。暂时将它们留空，随着教程的继续，您将完成它们。

```
.cloud_function_project/
│ 
├── terraform/
│    │
│    ├── backend.tf
│    ├── function.tf
│    ├── main.tf
│    ├── storage.tf
│    └── variables.tf
│
└── src/
     │
     ├── main.py 
     └── requirements.txt
```

在您开始归档不同的文件之前，我们将快速浏览一下每个文件的作用。

`src`文件夹包含云函数的源代码。它是 Python 特有的结构。

*   `main.py`:云函数的源代码。
*   `requirements.txt`:运行`main.py`需要的 python 库列表。(在本教程中您不需要它)

`terraform`文件夹包含要部署的环境的配置文件。

*   `backend.tf`:声明[地形后端](https://www.terraform.io/language/settings/backends)。
*   `main.tf`:环境的主要声明。
*   `variables.tf`:变量的定义。
*   `storage.tf`:Google 云存储桶的声明。
*   `function.tf`:云函数声明。

# 创建云函数

编写和运行云函数不是本教程的重要部分。您将部署一个函数，该函数将记录一些关于已经上传到 bucket 中的文件的有用信息，以便进行跟踪。

> **注意:**该功能没有需求，所以`requirements.txt`为空。
> **注意:**这是一个 Python 云函数，所以你可以根据你选择的编程语言改变代码源。

# 创建地形基础设施

## 后端

通过指定 Terraform 后端开始声明您的环境。您可以选择`[local](https://www.terraform.io/language/settings/backends/local)`，这意味着所有的状态文件都将存储在本地目录中。你也可以选择一个`[gcs](https://www.terraform.io/language/settings/backends/gcs)`后端，但是让我们保持简单。

## 变量

声明 Terraform 文件中使用的变量。您需要根据您想要在其中部署资源的项目的 ID 来更改`project_id`变量。当然，你也可以更换`region`。

## 主要的

声明与 google [提供者](https://www.terraform.io/language/providers)的连接。

## 谷歌云存储

声明两个 Google 云存储桶，分别存储云函数的代码和上传文件。

> **注意:**桶名以`project_id`为前缀，因为桶必须有唯一的名称。

## 谷歌云功能

最后一步:声明云函数。这需要将源代码压缩成 zip 文件，并上传到 bucket 中进行存储。然后在用 Terraform 创建云函数时可以访问源代码。

> **注意:**文件有点长，所以可以随意查看注释来理解逻辑。

# 部署环境

一切准备就绪，可以部署了。在您的终端中找到项目的根目录，然后找到`terraform`文件夹的根目录。

```
$ cd ./terraform
```

初始化您的代码以下载代码中提到的需求。

```
$ terraform init
```

查看更改。

```
$ terraform plan
```

接受变更并将其应用于实际的基础设施。

```
$ terraform apply
```

# 测试您的云功能

测试一切工作正常:

*   打开[谷歌云控制台](https://console.cloud.google.com/)并连接到你的项目。
*   进入[谷歌云存储浏览器](https://console.cloud.google.com/storage/browser)。
*   您可以看到`<YOUR-PROJECT-ID>-function`和`<YOUR-PROJECT-ID>-input`铲斗。点击名为`<YOUR-PROJECT-ID>-input`的桶，将任意文件上传到其中，触发云功能。
*   要验证它是否有效，请访问您的[云功能列表](https://console.cloud.google.com/functions/list)。这里应该有一个名为`function-trigger-on-gcs`的云函数。点击它的名称并转到`LOGS`选项卡，查看它是由您上传的文件触发的。

![](img/a41213d2ce618721d92c099fba464fc2.png)

> **知识就是分享。**
> **支持**我，一键获取 [**中我所有文章的**访问**。**](https://axel-thevenot.medium.com/membership)

![](img/94e903d7d66ff043ca9645dbc42c33bb.png)

# 更进一步

我希望你喜欢这篇部署云功能的简短教程。可以应用许多改进。您可以:

*   设置一个 [IAM 策略](https://cloud.google.com/iam/docs/policies)。
*   添加变量并将该代码转换成可重复使用的[地形模块](https://learn.hashicorp.com/tutorials/terraform/module)。
*   将部署集成到[云构建](https://cloud.google.com/build)流程中，以实现持续部署。
*   还有很多…