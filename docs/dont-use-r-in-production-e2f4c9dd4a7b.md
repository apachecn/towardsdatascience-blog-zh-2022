# 不要在生产中使用 R

> 原文：<https://towardsdatascience.com/dont-use-r-in-production-e2f4c9dd4a7b>

## 但是，如果你这样做，这里是如何

![](img/b2f62e795578e382d61faa7a0fceab8e.png)

DALL-E 生成:计算机科学可再生技术环境的说明

我们已经在生产中运行 R 好几年了，这篇文章是关于如何让它发生的。这篇文章也是关于为什么我仍然不认为你应该在生产中运行 R，如果你真的不需要它。我是从一个工程师的角度来看这个问题的。我们将讨论许可、R 限制以及如何解决主要障碍，以达到这样一个目的，即您可以在生产环境中运行 Rscripts，而不那么痛苦( [Github link](https://github.com/digital-thinking/reproducible-r) )。这篇文章不是关于付费或托管服务的。

# 生产中的研发

## 使用检查点解决版本控制问题

R 的最大问题是可重复性。与 Python 相比，R 没有高级版本和包管理。数据科学家在他们的机器上使用 R，安装所有的包并使用当时发布的版本。从用户的角度来看，这很容易，不需要考虑版本控制。但是如果您现在想在其他机器上运行这个脚本，那么它很有可能不起作用，尤其是在编写脚本和部署之间有延迟情况下。安装一个旧版本的 R 包并不像你想象的那样简单，有很多方法，但是很麻烦。

微软承认这一点，并建立了[检查点](https://mran.microsoft.com/package/checkpoint)库，它这样描述自己:

> checkpoint 的目标是解决 r 中的包可复制性问题。具体来说，checkpoint 允许您在特定的快照日期安装 CRAN 上存在的包，就像您有一台 CRAN 时间机器一样。

因此，有了 checkpoint，我们可以选择按日期进行版本化，而且到目前为止已经有了可复制的 R 环境。这个帖子在使用 checkpoint 的同时，还有 [renv](https://rstudio.github.io/renv/articles/renv.html) ，解决了同样的问题。

使用它很简单，我们只需将它添加到我们的*脚本*中:

```
library(checkpoint)
checkpoint(“2022–07–17”, r_version=”4.2.1")
```

Checkpoint 将扫描脚本中的导入并管理它们，但是如果您有自定义包，您需要在包含以下内容的 *checkpoint.yml* 中排除这些包:

```
exclude:
   - myRPackage
```

## 使用 Docker 隔离环境

通常，我们有多个脚本，使用不同的库和版本。为了使它们相互独立，我们使用 Docker。有了 Docker，我们可以构建可靠的环境，并将它们部署到我们想要的任何地方。Docker 与检查点库相结合，解决了可再现性问题。

[rocker](https://rocker-project.org/images/versioned/r-ver.html) 项目提供了预构建的版本化 R 容器，我们可以通过扩展版本化映像来使用这些容器:

```
ARG VERSION=”4.1.0"
FROM rocker/r-ver:$VERSION as base
# Install needed packages and git
RUN apt-get update \
&& apt-get -y install \
libgit2-dev \
libcurl4-openssl-dev \
libssl-dev (...)# checkout our git repository
WORKDIR /workdir
RUN git clone [https://{secret_key}@github.com/username/repo.git](https://{secret_key}@github.com/username/repo.git) .# install checkpoint library
RUN Rscript -e “install.packages(‘checkpoint’)”#start R script
CMD git pull && Rscript my_r_script.R
```

有了这个小 docker 文件，你已经能够用一个特定的 R 版本来启动一个 R 脚本了( **docker run** ),而且你还可以确保所使用的库的版本是由检查点库修复的。但是，您会注意到执行花费了相当长的时间，因为检查点库第一次管理指定日期的脚本依赖项时，需要花几分钟来下载和设置它们，并且每次都是这样。

## 确定开始时间

像这样使用 Docker 和预构建的 R-images 是一个很好的开端，但是我们必须解决另一个问题。如果我们在每次代码更改时都从头构建映像，我们会有巨大的开销，因为检查点库最初需要花费很多时间来构建。但是，如果 R 代码发生变化，我们不需要构建映像，也不需要每次都清除检查点。

为了避免这种情况，我们将 R 脚本的执行移出了 docker 文件。为了避免在设置 R 之后关闭容器，我们在最后调用**sleep infinity**将控制权交给我们的 bash 脚本。

现在 bash 脚本必须自己启动和停止 docker 容器。然而，现在我们可以在启动 R 脚本之前执行 git pull，而不必在每次代码更改时都重新构建容器。

在 Dockerfile 文件中，我们通过调用以下命令来替换最后一个命令:

```
CMD sleep infinity
```

我们最初构建容器(或者如果我们需要更新依赖项),使用:

```
docker build --no-cache -t foo_image . #build the image
docker create --name foo_container foo_image #create container
```

我们的 bash 脚本完成了以下工作

```
docker start foo_container
docker exec foo_container git pull origin master
docker exec foo_container Rscript my_r_script.R
docker stop foo_container
```

有了这些简单的脚本( [github](https://github.com/digital-thinking/reproducible-r) )我们现在可以快速、可伸缩和可复制地运行 R 脚本。

# 为什么 R 还不用于生产

## 许可证

R 使用多个许可证，大多数核心 R 库都是在 copyright license 下发布的，比如 GPL-2 和 GPL-3。因此，在某些情况下，这已经是一个问题。如果您计划编写 R 代码并将其作为产品的一部分发布，您将被迫开源您的代码。这已经是一个原因，为什么没有很多产品允许在云环境中执行 R。然而，如果您的产品没有附带 R，那么对于核心库来说，您很可能是安全的。但由于我不是律师，请询问你们的法律部门。

## r 不适合软件工程师

大多数 R 用户不会发现这些问题，直到他们必须反复修改他们的脚本。这很好，因为大多数 R-coder 不是软件工程师，不需要知道产品软件应该是什么样子。然而，经常有人试图争辩，因为它对他们来说很容易使用，它在其他地方也一定很容易使用。以下是我个人在生产中遇到的一些实际问题:

*   R **缺乏 OOP 支持**，R 是一种函数式语言是有充分理由的。这对小脚本来说很好，但对大项目来说是个问题，因此不建议构建更大、更复杂的应用程序。r 更像是编写狭窄任务的脚本，而不是管理大型工作流。
*   与编译语言相比，r 本身很慢。因此，很多 R 库要么依赖于**预编译的二进制文件**，要么需要安装 g++工具链**从源代码**构建它们。该来源依赖于平台，可能会导致许多问题。此外，出于安全原因，您(或您的 IT)不希望在生产服务器上使用 g++。
*   您可以生成日志，但这很糟糕，因为您**不能轻松地将所有相同格式的日志重定向到 **std:out 和 std:err** ，所以如果您将日志推入 ELK(或类似的)并处理来自 R print 语句的多行日志，您必须做更多的工作。**
*   在 python 中，你有 requirements.txt/setup.py 来定义需求，处理依赖和版本，在 R 中你没有。r 易于使用，但这是有代价的。这表明 R 正在**关注特定的分析任务**。
*   当您在自己的机器上编写代码时，它可能无法在另一台机器上运行，尤其是如果中间有一段时间。变化很大，一些库被修改了，代码不再工作了。r 缺乏非常基本的**版本控制**。
*   **CI/CDs 工具通常不支持 R** ，你必须定制一切。这包括单元测试、集成测试、静态代码检查、包部署和版本控制。
*   r 没有成熟的生态系统来部署 web 服务器和 API。有用于简单可视化的[闪亮的](https://shiny.rstudio.com/)和用于 API 的 [OpenCPU](https://www.opencpu.org/) 和[水管工](https://www.rplumber.io/)，但正如其名称所示，它笨重且感觉有限，与 Python 等语言中的类似工具相去甚远。特别是 R 用户经常提倡的 plumber，缺少 HTTPs、OAUTH 和其他大多数更高级的特性。此外，在 R 中，一切都是单线程的。
*   社区中没有安全问题的意识，人们经常用 devtools 安装未经检查的 github 库。这是生产服务器上的一个主要安全问题(不仅仅是那里)。
*   r 确实比 python 更容易崩溃，因为它的易用性和函数式编程风格让用户不必考虑极限情况。

# 结论

在这篇文章中，我展示了为什么我认为 R 不是生产用例的好选择。然而，有时候无论如何都是有意义的。原因可能是上市时间、团队组成或技术基础设施。对于这些情况，我展示了如何使用 Docker 和 checkpoint 以可重复的方式部署 R。代码可以在[这里](https://github.com/digital-thinking/reproducible-r)找到。