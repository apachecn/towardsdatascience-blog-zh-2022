# 在五分钟内为 AWS Panorama 部署您的第一个计算机视觉应用

> 原文：<https://towardsdatascience.com/deploy-your-first-computer-vision-app-for-aws-panorama-in-five-minutes-221e974606e3>

## 了解如何以简单的方式设置您的第一个 Panorama 应用程序

![](img/9af97370ecf8914ad0495394ba83770a.png)

瑞秋·麦克德莫特在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

大约一年前，我在这里发表了[关于在 AWS Panorama 上部署物体探测器应用程序的数据科学分步指南](/deploy-an-object-detector-model-at-the-edge-on-aws-panorama-9b80ea1dd03a)。文章挺长的(17 分钟看完！)并涉及到许多微妙的步骤。使用官方工具创建 Panorama 应用程序并不容易:您应该使用固定且复杂的项目结构，记住许多关于项目的参数，并在每次调用它们时将它们传递给工具。然后，文件和文件夹必须以特定的方式命名，并且您必须在项目的不同部分维护相同信息(例如，深度学习模型属性)的一致副本。所有这些“内务”任务并没有交付真正的商业价值:它们仅仅是为了保持项目和样板代码的更新。

在过去的几个月里，我构建并部署了十几个 Panorama 应用程序，亲身经历了这种负担。显然，一些重复的任务和模式促使我尽可能自动化 Panorama 应用程序的设置和部署过程。我很高兴介绍 [cookiecutter-panorama](https://github.com/mrtj/cookiecutter-panorama) ，一个 AWS Panorama 项目生成器工具和构建系统。让我们深入探讨一下，从 AWS Panorama 应用程序的定义开始。

# 全景应用程序的剖析

如果你错过了[之前的](/deploy-an-object-detector-model-at-the-edge-on-aws-panorama-9b80ea1dd03a) [集](/remote-view-your-computer-vision-models-running-on-aws-panorama-32922369ecf)，AWS Panorama 是一个[硬件和软件平台](https://aws.amazon.com/panorama/)，可以运行计算机视觉(CV)应用程序，分析来自内部互联网协议(IP)摄像机的视频流。

*Panorama 应用*是打包的深度学习模型、业务逻辑代码和清单结构(也称为“应用图”)的集合，清单结构定义了这些组件之间的数据管道。

在“经典”全景应用程序的情况下，深度学习模型(在 Pytorch、Tensorflow 或 MXNet 中训练)由应用程序的构建过程打包在一个称为“模型资产”的工件中。类似地，您的业务逻辑代码(用 Python 编写，使用 Panorama 应用程序 SDK)被打包在“代码资产”工件中。这些工件被注册并上传到您的 AWS 帐户。每当您想要将 CV 应用程序部署到 Panorama 设备时，您必须提供引用这些工件的清单结构，并描述设备应该如何连接它们。

因此，Panorama 应用程序松散地耦合到一组深度学习和业务逻辑包，这些包通过清单文件绑定在一起。开发人员必须不断地同步应用程序清单文件和包中的引用。这是 cookiecutter-panorama 可以自动化的最大维护任务之一。

# 烹饪刀具

下面就来说说怎么做饼干吧！

![](img/61570de7e134a7fe2f6e36855cf59274.png)

[安舒阿](https://unsplash.com/@anshu18?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

[Cookiecutter](https://github.com/cookiecutter/cookiecutter) 是一个命令行实用程序，它从 Cookiecutter(项目模板)创建项目。它成为在 Python 生态系统中生成项目的事实上的标准，因为它漂亮简单的设计与引人注目的特性相匹配。目前，GitHub 上发布了超过 [7000 个 cookiecutter 项目模板](https://github.com/search?q=cookiecutter&type=Repositories)。Cookiecutter-panorama 将 panorama 应用程序模板添加到这个 pantheon 中。

# 开始吧！

首先，[用 pip ( `**pip install cookiecutter**`)、conda ( `**conda install -c conda-forge cookiecutter**`)或](https://cookiecutter.readthedocs.io/en/stable/installation.html) [pipx](https://pypa.github.io/pipx/) ( `**pipx install cookiecutter**`，推荐方式)安装 cookiecutter 命令行工具。

![](img/0a88775da4a39329e4c251854d4cb553.png)

由 [SpaceX](https://unsplash.com/@spacex?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

创建 Panorama 应用程序项目非常简单:

```
**$ cookiecutter** [**https://github.com/mrtj/cookiecutter-panorama.git**](https://github.com/mrtj/cookiecutter-panorama.git)
```

模板引擎会问你一堆参数的值。但是，它也提供了合理的默认值，您可以在大多数情况下保留这些值并确认选项。

你只需要设置两个参数:`project_name`和`s3_working_bucket`。后者应该是您拥有读/写权限的帐户中现有 S3 存储桶的名称。你可以在 cookiecutter-panorama 的[自述文件中找到参数的详细描述。](https://github.com/mrtj/cookiecutter-panorama/blob/main/README.md)

```
**$ cookiecutter** [**https://github.com/mrtj/cookiecutter-panorama.git**](https://github.com/mrtj/cookiecutter-panorama.git)
project_name [Panorama Video Processor]: ↲
project_slug [panorama_video_processor]: ↲
s3_working_bucket [my_bucket]: **my_real_bucket**
s3_working_path [s3://my_real_bucket/panorama_projects/panorama_video_processor]: ↲
camera_node_name [camera_input]: ↲
display_node_name [display_output]: ↲
code_package_name [panorama_video_processor_logic]: ↲
code_package_version [1.0]: ↲
code_asset_name [panorama_video_processor_logic_asset]: ↲
code_node_name [panorama_video_processor_logic_node]: ↲
model_package_name [panorama_video_processor_model]: ↲
model_package_version [1.0]: ↲
model_asset_name [panorama_video_processor_model_asset]: ↲
model_node_name [panorama_video_processor_model_node]: ↲
model_input_name [input0]: ↲
model_processing_width [224]: ↲
model_processing_height [224]: ↲
```

Cookiecutter 将在名为`project_slug`参数的目录中生成您的项目，在上面的例子中是`panorama_video_processor`:

```
**$ cd** **panorama_video_processor**
```

# 包含电池:构建系统

一旦生成了项目，您就可以开始与 Makefile 进行交互了。构建系统提供了一组丰富的功能:配置 git 存储库，构建应用程序容器和深度学习模型，将包上传到您的 AWS 帐户以准备部署到 Panorama 设备，在您的开发工作站上使用[测试实用程序](https://github.com/aws-samples/aws-panorama-samples)运行您的应用程序，防止您的 AWS 帐户 id 在公共 git 存储库中泄露，等等。有关功能的完整列表，请参考 Panorama 项目中生成的自述文件。[这里](https://github.com/mrtj/cookiecutter-panorama/blob/main/%7B%7Bcookiecutter.project_slug%7D%7D/README.md)是预览。

![](img/d16b78024b15e93e632e6cf5a85330ba.png)

[姚](https://unsplash.com/@hojipago?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

cookiecutter-panorama 提供的一些任务您可以从中受益:

*   `**make init-repo**`:初始化一个 git 仓库
*   安装所需的构建工具: [aws-cli](https://aws.amazon.com/cli/) 、 [panorama-cli](https://github.com/aws/aws-panorama-cli) 和 [docker](https://docs.docker.com/get-docker/) 。[将 aws-cli 配置为注册 Panorama 设备的帐户的凭据。](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)
*   `**make import**`:将项目导入您的 AWS 账户
*   `**make build**`:构建项目，创建一个虚拟的深度学习模型，以后可以用实际的模型替换。虚拟模型将使用模型包基础架构和 Panorama 设备的真实 GPU 计算输入视频帧的 RGB 通道的平均值。
*   `**make package**`:将编译好的应用容器和打包好的深度学习模型上传到你的 AWS 账号。这个脚本还将输出编译后的清单 JSON 的路径。

然后您可以拿起清单 JSON，前往 [AWS Panorama 控制台](https://console.aws.amazon.com/panorama/home)并部署您的应用程序！

有问题吗？问题？建议？[在](https://github.com/mrtj/cookiecutter-panorama/issues/new/choose) [cookiecutter-panorama github 项目](https://github.com/mrtj/cookiecutter-panorama)中打开一个问题，或使用以下联系人联系！

# 关于作者

Janos Tolgyesi 是 Neosperience 的 AWS 社区构建者和机器学习团队负责人，拥有 5 年多的 ML 技术专业知识和 8 年多的 AWS 经验。他喜欢构建东西，让它成为边缘的[视频分析应用](https://www.neosperience.com/solutions/people-analytics/)，或者基于点击流事件的[用户分析器](https://www.neosperience.com/solutions/user-insight/)。你可以在 [Twitter](https://twitter.com/jtolgyesi) 、 [Medium](https://towardsdatascience.com/@janos.tolgyesi) 和 [LinkedIn](http://linkedin.com/in/janostolgyesi) 上找到我。

cookiecutter-panorama 开源项目得到了 [Neosperience](https://www.neosperience.com/) 的支持。

我要特别感谢[卢卡·比安奇](https://medium.com/u/6550450171ac)校对了这篇文章。