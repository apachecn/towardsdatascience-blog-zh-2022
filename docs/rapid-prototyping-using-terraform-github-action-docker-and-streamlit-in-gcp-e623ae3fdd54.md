# 在 GCP 使用 Terraform、GitHub Action、Docker 和 Streamlit 进行快速原型制作

> 原文：<https://towardsdatascience.com/rapid-prototyping-using-terraform-github-action-docker-and-streamlit-in-gcp-e623ae3fdd54>

## 使用 CI/CD 工具加速见解共享

![](img/cda8df6e081557aa2a49cf09891371bf.png)

马修·布罗德在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

# 介绍

数据科学家和分析师是许多组织的宝贵资源。他们将大部分时间用于理解业务需求，并通过数据收集、争论、清理和建模来寻找见解。支持这些数据专家的工具非常丰富，例如，像 Jupyter 和 Colab 这样的笔记本、Python 或 R 库以及其他数据技术。

另一方面，当涉及到与利益相关者分享重要的发现和有价值的见解时，选择是有限的。对于非技术人员来说，笔记本可能不是最佳选择；在像 Word 和 PowerPoint 这样的办公文档中分享你的发现通常不能描述你的发现的动态和交互性质。对于数据专家来说，找到以合适的格式快速有效地向决策者展示调查结果的最佳方式通常是一件痛苦的事情。

在本文中，我们将展示一种使用 CI/CD(持续集成和持续交付)工具(如 Terraform、GitHub Actions 和 Docker with Streamlit applications)与利益相关方共享数据科学发现的方法，这些工具可以加速原型开发。我们将代码部署到谷歌云平台(GCP)的虚拟机上，但是你也可以将这些技术应用到其他云服务提供商。

## 工作流程概述

本文将指导您创建以下 CI/CD 管道。

1.在本地写 Terraform 文件，推送到 GitHub。
2。GitHub Action 使用 Terraform 实现 GCP 供应自动化。
3。在本地编写 Streamlit app 代码和 Dockerfile，并将代码推送到 GitHub。
4。通过 GitHub Action 将代码部署到 GCP 的虚拟机实例中。

![](img/66adfca33e2e13620f6441ff7fbab445.png)

图一。工作流程概述(作者图片)

## 先决条件

实施这些解决方案有四个要求。

● GitHub 账号
● Terraform 安装
● GCP 服务账号
● Python 库

## 开源代码库

GitHub 是一个使用 Git 的基于 web 的版本控制服务。它还提供一项名为 GitHub Actions 的服务。您不需要安装第三方应用程序来自动化测试、发布和将代码部署到生产中的工作流。您可以在 [GitHub 网站](https://github.com/)上创建一个帐户。

## 将（行星）地球化（以适合人类居住）

Terraform 是一个开源的 CI/CD 工具，它将基础设施作为代码(IaC)进行管理，由 HashiCorp 提供。IaC 使用代码来管理和配置基础设施，消除了通过云供应商提供的 UI 手动配置物理硬件的需要。

使用 Terraform 进行原型开发有一些优点。例如:

1.  避免手动调配基础架构。
2.  数据科学家和软件工程师可以自助。
3.  对当前设置的改变变得简单明了。
4.  易于传递知识和保持版本。

不需要在本地安装 Terraform，就可以通过 GitHub Action 实现流程自动化，因为 GitHub 在云中执行工作流。然而，在测试新资源或刚刚开始使用 Terraform 时，在本地安装软件可能是个好主意。在这种情况下，您需要向 Terraform 用户提供访问云资源的凭据，这在一些组织中可能是一个安全问题。你可以在 [Terraform 网站](https://www.terraform.io/downloads)上找到安装说明。

## GCP

服务账户(SA)代表谷歌云服务身份，例如，在计算引擎上运行的代码、在云存储中创建桶等。可以从 *IAM 和管理*控制台创建 sa。出于原型制作目的，请为帐户选择编辑者角色，以查看、创建、更新和删除资源。你可以参考谷歌云的[官方指南](https://cloud.google.com/iam/docs/creating-managing-service-accounts)。本文使用了计算引擎、云存储和 VPC 网络。

## 计算机编程语言

我们只用了三个库:Pandas、Yfinance 和 Streamlit。Pandas 是一个开源的数据操作工具。yfinance 是一个 python API，用于从 Yahoo！金融。Streamlit 是一个基于 Python 的 web 应用框架；与 Flask 或 Django 等其他框架相比，您可以用更少的代码开发 web 应用程序。

# 1.地形文件

Terraform 是一个 IaC 工具。您可以通过编写代码来管理云基础架构。Terraform 可以通过各种云提供商提供服务和资源。HashiCorp 和 Terraform 社区编写和维护代码。你可以在 Terraform Registry 网站上找到你可以使用的提供商列表，包括 AWS、Azure 和 GCP。

Terraform 的工作流程由三个步骤组成:定义、计划和应用。在定义步骤中，您将编写配置文件来声明提供者、服务和资源。配置文件可以由 Terraform 使用 *terraform fmt* 和 *terraform validate* 命令进行格式化和验证。在计划步骤中，Terraform 根据配置文件和现有基础设施的状态创建一个执行计划。它可以创建新的服务或更新/删除现有的服务。最后，应用步骤执行计划的操作，并记录 Terraform 提供的基础设施的状态。

我们将创建三个 Terraform 文件:main.tf、variables.tf 和 terraform.tfvars。

## main.tf

main.tf 是一个包含主要配置集的文件。它定义了所需的提供者、提供者变量和资源。该文件的内容如下:

首先，定义提供者。在本文中，我们将使用 google。

定义了提供者的变量。var。*将从单独的文件 variables.tf 中导入值。

credentials_file 应该是从 GCP 下载的 json 中服务帐户密钥的路径。

接下来的脚本创建了一个 VPC 网络。

然后，我们配置一个虚拟机实例。

我们指定虚拟机实例名称、机器类型、磁盘映像和网络接口。我们还添加标签并定义启动脚本。标记允许您将防火墙规则应用于 VM 实例。我们创建了三个标记，分别对应于下面定义的三个防火墙规则。metadata_startup_script 在启动时运行 shell 脚本。在我们的例子中，我们将运行脚本在虚拟机中预安装 Docker 引擎。

下一个特定的 http 访问防火墙规则。

您可以通过 CIDAR 格式的 source_ranges 来限制访问。

我们可能希望通过 SSH 访问 VM 实例。

第三个防火墙规则是针对 Streamlit web 应用程序的。端口 8501 将被打开。

最终的资源定义是针对云存储的。它使用从 variables.tf 文件中读取的变量创建新的 bucket。

## 变量. tf

var 的值。*由 variables.tf 文件提供。

每个变量都可以有一个默认值。在我们的例子中，一些变量没有默认值。因为它们是必需的参数，所以 Terraform 需要提供它们。提供变量的方法之一是通过*。tfvar 文件，减少了一些敏感信息的暴露。

## terraform.tfvars

我们在 terraform.tfvars 文件中有三行。请用您的值替换“***”。

## GCP 机器类型和图像列表

在 main.tf 中，我们指定了机器类型和磁盘映像。接受的值可能不总是与您在 GCP 控制台上看到的描述相同。您可以使用以下 gcloud 命令检索机器类型列表和计算引擎映像。(必须安装谷歌云 SDK。如果没有，请参考[官方指南](https://cloud.google.com/sdk/docs/install)

g 云计算映像列表

# Docker 安装脚本

我们将 docket 安装到虚拟机实例中。在 main.tf 文件所在的目录中找到安装脚本 install_docker.sh。Ubuntu 的安装脚本复制自官方 [Docker Doc 网站](https://docs.docker.com/engine/install/ubuntu/)。

# 2.GitHub 行动——GCP

GitHub 操作允许您构建自动化的 CI/CD 管道。公共存储库免费，私人存储库包含 2000 分钟[【1】](https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions)。

GitHub 动作读取了[下保存的 YAML](https://en.wikipedia.org/wiki/YAML) 文件。github/workflows/repository 中的目录并执行文件中定义的工作流。一个工作流可以有多个作业，每个作业都有一个或多个步骤。一个步骤可以有动作，但不是所有的步骤都可以执行一个动作。

文件内容如下:

首先，我们定义工作流名称和触发器，即 *push* 到主分支。接下来，我们添加关于作业的信息

上面的作业在服务器端的 ubuntu 机器上运行。每个作业在一个新的实例上运行；因此，每个作业下定义的环境变量不能用于另一个作业。

接下来，我们指定步骤和动作。

第二步，我们将使用秘密。GCP 萨基。这些秘密存储在存储库的动作秘密中。GitHub 支持 base64 编码的秘密。您可以使用 Python 将 GCP 服务帐户密钥编码为 base64 格式的 JSON:

Terraform init 步骤通过下载提供者插件启动工作目录的准备。我们还在后端配置中指定了一个 GCS bucket 名称。

如果您将目前准备好的文件从本地驱动器提交到主驱动器，GitHub 工作流会自动运行并提供服务。你可以在 GitHub Action 页面查看执行日志。

![](img/e594617f5bc316a8e162305813a22914.png)

图二。GitHub 动作工作流程日志(图片由作者提供)

# 3.Streamlit 和 Dockerfile

在前面的章节中，我们使用 Terraform 和 GitHub 操作准备了基础设施。我们现在用 Streamlit 和 Docker 准备运行在 GCP 虚拟机实例上的 web 应用程序代码。

作为一个简单的例子，我们将为 Yahoo Finance 的时间序列数据创建一个仪表板，显示每日回报率，以便用户可以比较不同的指数或外汇汇率。仪表板用户也可以修改日期范围。

我们将要创建的仪表板图像如下所示:

![](img/0c22d781cbb495f7645a33b9d05ac69a.png)

图 3。Streamlit Web 应用程序用户界面(图片由作者提供)

## Python 脚本

上面显示的应用程序可以使用 Streamlit 创建。

首先，我们导入三个库。如果没有安装，请使用 pip(或 pip3) install 命令安装它们。

我们在 st.title('text ')中指定 web 应用程序的名称。

然后，我们准备选择题。st.multiselect()创建一个下拉选择。

Streamlit 提供了 date_input 选项，可以从日历中选择日期。这里我们创建两个变量:from_date 和 to_date。

最后，我们用计算出的返回值设计图表。

Streamlit 动态读取选择器变量中的值，然后从 yfinance API 中检索收盘价数据。增长率是使用 pandas pct_change()计算的。最后一步是使用 st.line_chart()函数在折线图中表示数据框。

我们将脚本保存在 app.py 文件中。

## Dockerfile 文件

dockerfile 是一个文本文件，包含创建图像文件的所有命令。您可以使用 docker build 构建映像文件，并通过 docker run 命令将映像作为容器运行。

在上面的 docker 文件中，我们在最新的官方 Python docker 发行版上构建了自己的映像。因为我们需要三个库来运行 Streamlit 应用程序，所以我们在文本文件中指定依赖关系，然后运行 pip install 命令。Streamlit 的默认端口是 8501。然后，我们复制容器的/app 目录中的 app/py 文件，并运行 Streamlit web 应用程序。

# 4.GitHub Action — Web 应用程序

在前面的小节中，我们为 GitHub 存储库中的基础设施供应准备了一个 YAML 文件。我们还必须创建一个存储库并定义一个工作流来部署 docker 容器，该容器在 GCP 的已调配虚拟机实例上运行 Streamlit web 应用程序。

首先，我们为 web 应用程序代码创建一个新的私有 GitHub 存储库，然后重复相同的步骤，在 GitHub Action secrets 中添加 GCP 服务帐户凭证。

接下来，我们在 GitHub 中准备一个私有访问令牌。该令牌用于从 GCP 的 VM 实例克隆 GitHub 中的这个存储库。

在你的 GitHub 页面，进入*设置*->-*开发者设置*->-*个人访问令牌*，然后点击*生成新令牌*。在*新增个人访问令牌*页面:*回购*和*工作流*必须打勾。*工作流程*选项允许您更新 GitHub 动作工作流程。您只会看到一次生成的令牌，所以请将您的令牌复制到编辑器中。

![](img/b6341c189dfd99fda7d39f706226a206.png)

图 4。新的个人访问令牌(图片由作者提供)

我们需要在存储库 URL 中插入您的用户名和个人访问令牌。比如[*【https://github.com/your-user-name/your-repository.git】*](https://github.com/your-user-name/your-repository.git)会是[*https://****your-user-name:your-access-token @****github.com/your-user-name*](https://your-user-name:your-access-token@github.com/your-user-name)*/your-repository . git*。将完整的 URL 保存在 GitHub Action secrets 中，这样我们就可以在工作流 YAML 文件中调用它。

在下面。github/workflows 目录，我们创建 deploy_dokcer.yaml 文件。该工作流从 Dockerfile 构建 docker 映像，并将其部署在 GCP 虚拟机实例中。

在 jobs 部分，我们设置了一些变量和权限。

在“步骤”部分，我们定义了操作。为了通过 ssh 在 VM 实例中运行 bash，我们设置了 gcloud CLI。在命令中，我们克隆 GitHub 存储库并创建 docker 映像，然后从映像运行容器。

当您将文件推送到 GitHub repo 时，工作流运行并将容器部署到 VM。您可以从 GitHub Actions UI 查看工作流日志。

![](img/8e01ea21cf265f8f64775685e47e9f40.png)

图 5。GitHub 动作工作流程日志(图片由作者提供)

现在，您可以修改 app.py，将修改后的代码推送到存储库，并确认更改已应用到虚拟机。

# 结论

本文解决了数据专家在分享来自数据分析阶段的有价值的见解方面的难题，并提出了一种使用 Streamlit web 应用程序呈现发现的方法。为了部署应用程序，CI/CD 工具和服务(如 Terraform 和 GitHub Actions)通过自动化工作流来帮助数据专家加速原型开发。

我们使用的例子是一个简单的用例；然而，Streamlit 可以做得更多。我们建议您访问 [Streamlit 网站](https://streamlit.io/)了解它能提供什么。类似地， [Terraform Registry](https://registry.terraform.io/) 也有很多有用的资源，并且由 HashiCorp 和提供商积极更新。值得查看您感兴趣的提供商的文档，以找到工作流自动化的其他机会。最后，GitHub 动作允许你设计更复杂的工作流程。如果你想在原型之外使用 GitHub 动作，强烈推荐阅读官方文档。

# 参考

[1]“关于 GitHub 操作的计费”，GitHub 文档。[https://docs . github . com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions](https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions)【访问时间:2022 年 8 月 7 日】。

‌