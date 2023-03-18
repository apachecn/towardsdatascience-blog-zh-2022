# 如何在 Docker 容器中运行数据科学项目

> 原文：<https://towardsdatascience.com/how-to-run-a-data-science-project-in-a-docker-container-2ab1a3baa889>

## 环境设置

## 关于如何将您的数据科学项目包装在 Docker 映像中并在 Docker 容器中运行它的一些技巧和实际示例

![](img/aee32e0527e896d5e18dd541bec161a5.png)

[伊恩·泰勒](https://unsplash.com/@carrier_lost?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

为数据科学项目编写代码时，它应该尽可能具有可移植性，也就是说，即使在不同的机器上，它也可以被执行任意多次。

**在建模和测试阶段，数据科学项目在您的机器上正常工作，但在运行时出错的情况经常发生。**

这可能是由于例如主机上安装的不同版本的库和软件。

为了应对这个问题，你可以使用**虚拟机**，它们构成了一个隔离独立的环境，允许你执行代码。然而，**虚拟机有一个问题，它们托管整个操作系统**，这通常会占用几千兆字节的存储空间。

因此，最好使用更轻便的系统来构建独立的应用程序。

在本文中，我描述了 Docker 平台的概况，它允许您通过一点点努力来构建独立的应用程序。另外我给你提供一个实际的例子，把一个数据科学项目变成 Docker 镜像。

文章组织如下:

*   Docker 入门
*   Docker 项目的结构
*   构建和运行 Docker 映像
*   实际例子。

# Docker 入门

Docker 是一个允许你构建、运行和管理独立应用的平台。其思想是构建一个应用程序，不仅包含编写的代码，还包含执行代码的所有上下文，比如库、环境变量等等。

当您用所有上下文包装您的应用程序时，您构建了一个 **Docker 映像**，它可以保存在您的本地存储库中或公共 Docker 存储库中，也称为 [Docker Hub](https://hub.docker.com/) 。

如果你想运行你的 Docker 镜像，你需要为它创建一个 **Docker 容器**。如果可以为同一个映像创建多个容器，那么可以多次运行同一个 Docker 映像。

要开始使用 Docker，可以从[这个链接](https://www.docker.com/products/docker-desktop/)下载。根据您的操作系统，您将下载不同的安装程序。

安装后，您可以运行应用程序或使用命令行实用程序。

Docker 由以下组件组成:

*   **Docker 引擎** —用于构建、运行和管理 Docker 映像和容器的本地平台
*   **Docker Hub** —所有公共和私有 Docker 映像的存储库
*   **Docker 客户端** —与 Docker 引擎交互的客户端。它可以是命令行工具，也可以是使用 Docker 引擎提供的 REST APIs 的对象。

# 2 Docker 项目的结构

要构建 Docker 映像，您至少应该定义两个元素:

*   你的申请
*   一份文件

该应用程序可以是您的数据科学项目。在本文中，您将看到 Python 中的一个例子，因此您应该添加另一个名为`requirements.txt`的文件，它包含您的应用程序所使用的 Python 包的列表。

## 2.1 文档文件

Docker 文件是 Docker 用来构建独立应用程序的文件。你可以在[这个链接](https://docs.docker.com/engine/reference/builder/)找到 Dockerfile 的完整文档。

下面这段代码显示了 docker 文件的基本结构:

```
FROM <BASE_IMAGE>
COPY <SRC> <DTS
RUN <COMMANDS_AT_BUILD_TIME>
CMD <ENTRY_COMMANDS_AT_RUNTIME>
```

`FROM`指令包含 Docker 构建应用程序的基础映像。例如，基本映像可以是特定的操作系统。要选择最适合您的观测范围的基础图像，您应该注意图像大小以及您的应用程序所需的软件包和软件。比如你只需要 Python，你可以选择一个 slim 版本的 Python 作为基础镜像。你可以在[这个链接](https://hub.docker.com/_/python)找到官方 Python 图片列表。

`COPY`指令复制 Docker 映像中的源文件。典型的`COPY`语句是:

```
COPY . .
```

它将文件系统的当前目录中包含的所有文件复制到 Docker 映像的当前目录中。

`RUN`指令在构建过程中运行指定为参数的命令。例如，在`RUN`指令下，您可以安装项目所需的所有 Python 库，如下所示:

```
RUN pip install --no-cache-dir -r requirements.txt
```

在前面的代码中，`requirements.txt`文件包含要安装的所有包的列表，每行一个。

`CMD`指令指定运行应用程序时要执行的命令。在 Python 项目中，CMD 指令应该指定用于运行主应用程序的命令:

```
CMD ["python", "./app.py"]
```

## 2.2 要求. txt

如前所述，该文件包含要安装的 Python 包的列表，每行一个，如下面的代码所示:

```
numpy
pandas
scikit-learn
```

# 3.构建和运行 Docker 映像

在本节中，您将看到:

*   如何建立码头工人形象
*   如何运行 Docker 映像
*   如何将 Docker 图像保存到 Docker Hub
*   从 Docker Hub 获取 Docker 图像

## 3.1 建立码头工人形象

要为您的数据科学项目构建 Docker 映像，您应该输入包含您的项目的目录，然后从终端运行以下命令:

```
docker build -t <IMAGE_NAME> .
```

最后一个点(`.`)表示包含您的应用程序的当前目录。

一旦构建了 Docker 映像，就可以通过编写以下命令来检查它的可用性:

```
docker images
```

作为输出，前面的命令显示了本地存储库中所有可用的 Docker 图像。

## 3.2 运行 Docker 映像

要运行 Docker 映像，您应该告诉 Docker 引擎为它构建一个容器。您可以从文件系统中的任何地方运行以下命令:

```
docker run -rm --name <CONTAINER_NAME> <IMAGE_NAME>
```

`-rm`参数告诉 Docker 引擎在停止运行时删除容器。

## 3.3 将 Docker 映像保存到 Docker Hub

Docker Hub 是 Docker 图像的注册表。它包含私有和公共图像。要将 Docker 图像保存到 Docker Hub，您首先需要在[这个链接](https://hub.docker.com/)创建一个账户。

创建帐户后，您可以从命令行登录，如下所示:

```
docker login
```

系统要求您输入用户名和密码。输入密码后，您就可以登录到您的远程存储库。

要将 Docker 映像添加到 Docker Hub，您需要将至少一个标记添加到您的 Docker 映像，如下面这段代码所示:

```
docker -tag <MY_LOCAL_IMAGE> <MY_LOCAL_IMAGE>:<MY_TAG>
```

以及您的远程映像:

```
docker tag <MY_LOCAL_IMAGE>:<MY_TAG> <MY_ACCOUNT>/<MY_LOCAL_IMAGE>:<MY_TAG>
```

然后，您可以将本地映像推送到 Docker Hub:

```
docker push <MY_ACCOUNT>/<MY_LOCAL_IMAGE>:<MY_TAG>
```

## 3.4 从 Docker Hub 获取 Docker 映像

要从 Docker Hub 获取 Docker 映像，只需运行以下命令:

```
docker pull <MY_ACCOUNT>/<MY_LOCAL_IMAGE>
```

# 4 一个实际例子

作为一个实际的例子，我们可以为我在[上一篇文章](/a-complete-data-analysis-workflow-in-python-and-scikit-learn-9a77f7c283d3)中描述的例子构建一个 Docker 映像。该项目在三种不同的条件下计算 K-最近邻分类器的 ROC 和精确召回曲线:

*   不平衡数据集
*   过采样数据集
*   欠采样数据集。

这个例子的目的是将描述的应用程序包装到 Docker 映像中，并将生成的图形保存在当前目录中。关于项目代码的更多细节，可以参考[我之前的文章](/a-complete-data-analysis-workflow-in-python-and-scikit-learn-9a77f7c283d3)。

你可以从我的 [Github 库](https://github.com/alod83/data-science/blob/master/DataAnalysis/Data%20Analysis.ipynb)下载该项目的原始代码。

## 4.1 设置

您可以按照下面描述的步骤将示例转换为 Docker 图像:

*   由于原始代码是用 Jupyter 编写的，首先，您需要**将其下载为 Python 脚本**。简单地说，您可以访问 Jupyter 中的文件菜单，并选择下载 Python 脚本。
*   然后，您可以创建一个新目录，其中应该包含所有项目，包括源数据集。在我们的示例中，源数据集位于`source`目录下，因此您可以将其复制到当前目录。
*   将下载的脚本重命名为 app.py，打开，将所有出现的`plt.show()`替换为`plt.savefig(f”{images_dir}/roc.png”)`，其中`image_dir`是包含输出目录名的变量，比如`images`。
*   现在，您可以创建`requirements.tx` t 文件，它应该包含所有需要的库:

```
numpy
pandas
scikit-learn
imblearn
matplotlib
scikit-plot
```

*   您可以按如下方式创建 Dockerfile 文件:

```
FROM python:slim
WORKDIR /app_home
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "./app.py"]
```

`FROM`指令指定使用 Python 的精简版本，因此生成的图像不会占用太多存储空间。

`WORKDIR`指令定义了 Docker 映像中的当前目录。

`COPY`指令将文件系统中我们目录的内容复制到 Docker 映像中的工作目录。

`RUN`指令安装所有的 Python 库。

`CMD`指令定义当 Docker 运行容器中的图像时运行哪个应用程序。

## 4.2 构建和运行 Docker 映像

现在，从当前目录，您可以通过以下命令构建映像:

```
docker build -t test-model .
```

您可以通过以下命令运行它:

```
docker run --rm -v `pwd`:/app_home test-model
```

注意，我们已经使用了 `-v`参数来映射文件系统中的当前目录(由`pwd`指定)和 Docker 映像中的`app_home`目录。这允许我们访问应用程序生成的图表。

# 摘要

恭喜你！您刚刚学习了如何将您的数据科学项目转换为 Docker 映像！该程序非常简单，并且产生可移植的软件，该软件也可以在不同于你的机器上使用！

您可以从我的 [Github 资源库](https://github.com/alod83/data-science/tree/master/EnvironmentSetup/Docker)下载本教程中使用的代码。

如果你读到这里，对我来说，今天已经很多了。谢谢！你可以在[这个链接](https://alod83.medium.com/my-most-trending-articles-4fbfbe107fb)阅读我的趋势文章。

# 相关文章

[](/how-to-convert-your-python-project-into-a-package-installable-through-pip-a2b36e8ace10) [## 如何将您的 Python 项目转换成可通过 pip 安装的包

### 一个带有现成模板的教程，描述了如何将 Python 项目转换成包，该包可在…

towardsdatascience.com](/how-to-convert-your-python-project-into-a-package-installable-through-pip-a2b36e8ace10) [](/three-tricks-on-python-functions-that-you-should-know-e6badb26aac2) [## 你应该知道的关于 Python 函数的三个技巧

### 快速概述一些可以提高你编程技能的技巧:嵌套函数、可变参数和

towardsdatascience.com](/three-tricks-on-python-functions-that-you-should-know-e6badb26aac2) [](/understanding-the-n-jobs-parameter-to-speedup-scikit-learn-classification-26e3d1220c28) [## 了解 n_jobs 参数以加速 scikit-learn 分类

### 一个现成的代码，演示了如何使用 n_jobs 参数来减少训练时间

towardsdatascience.com](/understanding-the-n-jobs-parameter-to-speedup-scikit-learn-classification-26e3d1220c28) 

# 你认为使用 Docker 太复杂了吗？请改用 virtualenv

Python virtualenv 是一个独立的 Python 环境，您可以在其中仅安装项目所需的包。

此处继续阅读[。](/have-you-ever-thought-about-using-python-virtualenv-fc419d8b0785)

# 保持联系！

*   跟着我上[媒体](https://medium.com/@alod83?source=about_page-------------------------------------)
*   注册我的[简讯](https://medium.com/subscribe?source=about_page-------------------------------------)
*   在 [LinkedIn](https://www.linkedin.com/in/angelicaloduca/?source=about_page-------------------------------------) 上连接
*   在推特上关注我
*   跟着我上[脸书](https://www.facebook.com/alod83?source=about_page-------------------------------------)
*   在 Github 上关注我