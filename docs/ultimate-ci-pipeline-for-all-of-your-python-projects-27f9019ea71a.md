# 所有 Python 项目的终极 CI 管道

> 原文：<https://towardsdatascience.com/ultimate-ci-pipeline-for-all-of-your-python-projects-27f9019ea71a>

## Python 项目的持续集成管道拥有您想要的一切——在几分钟内启动并运行

![](img/80cf4d4421c7c1593d936319ce82cecc.png)

照片由[约书亚·沃罗尼耶基](https://unsplash.com/@joshua_j_woroniecki?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄

每个项目都可以从一个强大的持续集成管道中受益，该管道可以构建您的应用程序、运行测试、链接代码、验证代码质量、运行漏洞分析等等。然而，构建这样的管道需要大量的时间，这本身并不能带来任何好处。因此，如果您想要为您的 Python 项目准备一个基于 *GitHub Actions* 的全功能、可定制的 CI 管道，并准备好您能想到的所有工具和集成，只需大约 5 分钟，那么这篇文章就能满足您！

# 快速启动

如果您不是很有耐心，或者只想马上开始，那么下面是启动和运行管道所需的最少设置:

上面的 YAML 配置了一个 GitHub Actions 工作流，该工作流引用了我的存储库[中的](https://github.com/MartinHeinz/workflows/blob/v1.0.0/.github/workflows/python-container-ci.yml)[可重用工作流](https://docs.github.com/en/actions/using-workflows/reusing-workflows)。这样，您就不需要复制(并在以后维护)包含所有动作的大型 YAML。您所需要做的就是将这个 YAML 包含在您的存储库的`.github/workflows/`目录中，并根据您的喜好配置`with:`节中列出的参数。

所有这些参数(配置选项)都有相同的默认值，它们都不是必需的，所以如果你相信我的判断，你可以省略整个`with`节。如果没有，那么您可以如上所示调整它们，以及工作流定义[这里](https://github.com/MartinHeinz/workflows/blob/v1.0.0/.github/workflows/python-container-ci.yml)中显示的其余选项。关于如何找到值和配置例如 Sonar 或 CodeClimate 集成所需的秘密的解释，请参见存储库中的[自述文件](https://github.com/MartinHeinz/workflows/blob/v1.0.0/README.md)，或者通读以下详细解释的部分。

工作流显然必须对您的存储库的内容做出一些假设，所以它期望有一个您的应用程序和`Dockerfile`的源代码目录，其他的都是可选的。关于存储库布局的示例，请参见测试存储库[这里的](https://github.com/MartinHeinz/pipeline-tester)。

您可能已经注意到，上面的代码片段引用了使用`@v1.0.0`的工作流的一个特定版本。这是为了避免引入任何您可能不想要的潜在更改。也就是说，不时地签出存储库，因为可能会有一些额外的更改和改进，因此会有新的版本。

# 基础知识

现在您可能已经配置好了管道，但是让我们看看它在内部做了什么，以及如何进一步定制它。我们从最基本的开始——检查存储库、安装代码和运行测试:

为了清楚起见，上面的代码片段被删减了，但是如果你熟悉 GitHub 的动作，所有的步骤应该是相当明显的。如果没有，不要担心，你实际上不需要理解这一点，重要的是要知道，管道与所有主要的 Python 依赖管理器——即`pip` `poetry`和`pipenv`——一起工作，你需要做的就是设置`DEPENDENCY_MANAGER`,管道会处理剩下的事情。这显然假设您的存储库中有`requirements.txt`、`poetry.lock`或`Pipfile.lock`。

上面的步骤还创建了 Python 的虚拟环境，在整个管道中使用它来创建隔离的构建/测试环境。这也允许我们缓存所有依赖项，以节省管道运行时间。额外的好处是，只要锁文件不变，依赖项就会跨分支缓存。

至于测试步骤— `pytest`用于运行您的测试套件。Pytest 将自动获取您在存储库中可能拥有的任何配置(如果有的话)，更具体地说是按照优先顺序:`pytest.ini`、`pyproject.toml`、`tox.ini`或`setup.cfg`。

# 代码质量

除了基础知识，我们还想加强一些代码质量度量。有很多代码质量工具可以用来确保你的 Python 代码是干净的和可维护的，这个管道包括了所有这些工具:

我们首先运行*Black*——Python 代码格式化程序。使用 Black 作为预提交钩子或者在每次本地保存文件时运行它是一个最佳实践，因此这一步应该只作为验证没有任何东西从缝隙中溜走。

接下来，我们运行 *Flake8* 和 *Pylint* ，它们应用了比 Black 更多的样式和林挺规则。这两者都可以通过各自的配置文件进行配置，配置文件会被自动识别。在 Flake8 的情况下，选项有:`setup.cfg`、`tox.ini`或`.flake8`，对于 Pylint 则有:`pylintrc`、`.pylintrc`、`pyproject.toml`。

上述所有工具都可以设置为强制模式，如果发现问题，这将使管道失败。这些的配置选项有:`ENFORCE_PYLINT`、`ENFORCE_BLACK`和`ENFORCE_FLAKE8`。

除了 Python 特有的工具，该管道还包括两个流行的外部工具——分别是 *SonarCloud* 和 *CodeClimate* 。两者都是可选的，但是考虑到它们可用于任何公共存储库，我建议使用它们。如果打开(使用`ENABLE_SONAR`和/或`ENABLE_CODE_CLIMATE`，Sonar scanner 将对您的代码运行代码分析，并将其发送给 SonarCloud，CodeClimate 将获取`pytest`调用期间生成的代码覆盖报告，并为您生成覆盖报告。

要配置这些工具，请在管道配置中包含它们各自的配置字段:

并按照存储库 [README](https://github.com/MartinHeinz/workflows/tree/v1.0.0#configure-sonar) 中概述的步骤来生成和设置这些秘密的值。

# 包裹

当我们确信我们的代码符合标准时，是时候打包它了——在这种情况下是以容器映像的形式:

这个管道默认使用 *GitHub 容器注册表*，它是你的仓库的一部分。如果这是您想要使用的，除了在存储库中提供`Dockerfile`之外，您不需要配置任何东西。

如果您更喜欢使用 *Docker Hub* 或任何其他注册表，您可以在`CONTAINER_REGISTRY_USERNAME`和`CONTAINER_REGISTRY_PASSWORD`(在管道配置的`secrets`部分)中提供`CONTAINER_REGISTRY`和`CONTAINER_REPOSITORY`以及凭证，管道会处理其余的。

除了基本的登录、构建和推送序列，该管道还生成附加到映像的附加元数据信息。这包括用 commit SHA 标记图像，如果有标签的话，还包括`git`标签。

为了提高流水线的速度，在`docker`构建过程中还使用了缓存来避免创建不需要重建的图像层。最后，为了提高效率，还对图像运行*驱动*工具，以评估图像本身的效率。它还为您提供了一个在`.dive-ci`中提供配置文件的选项，并为 Dive 的[指标](https://github.com/wagoodman/dive#ci-integration)设置阈值。与该管道中的所有其他工具一样，Dive 也可以使用`ENFORCE_DIVE`设置为强制/非强制模式。

# 安全性

我们不要忘记 CI 管道应该确保我们的代码不包含任何漏洞。为此，该工作流程包括额外的几个工具:

首先是一个叫做 *Bandit* 的 Python 工具，它在 Python 代码中寻找常见的安全问题。该工具有默认的规则集，但可以使用工作流程的`BANDIT_CONFIG`选项中指定的配置文件进行调整。正如前面提到的其他工具一样，Bandit 在默认情况下也运行在非强制模式下，但是可以使用`ENFORCE_BANDIT`选项切换到强制模式。

这个管道中包含的另一个检查漏洞的工具是 Aqua Security 的 *Trivy* ，它扫描容器图像并生成图像本身中发现的漏洞列表，这超越了仅限于您的 Python 代码的问题。这份报告随后被上传到 *GitHub 代码扫描*，然后会出现在你的存储库的*安全标签*:

![](img/6b9a44b0ab1e473f66ca1eaf1e470644.png)

繁琐的代码扫描报告—图片由作者提供

上述工具保证了我们构建的应用程序的安全性，这很好，但是我们还应该提供最终容器图像的真实性证明，以避免供应链攻击。为此，我们使用 tool 用 GitHub 的 OIDC 令牌对图像进行签名，该令牌将图像与将代码推送到存储库的用户的身份联系起来。这个工具不需要任何密钥来生成签名，所以它可以开箱即用。然后，该签名将与您的图像一起被推送到容器注册表中，例如在 Docker Hub 中:

![](img/0d0ff38727a1ccf2e4c8243ad80cef3f.png)

Docker Hub 中的连署签名—图片由作者提供

然后可以使用以下方法验证上述签名:

有关`cosign`和容器图像签名的更多信息，请参见 GitHub 博客上的[文章](https://github.blog/2021-12-06-safeguard-container-signing-capability-actions/)。

# 通知

这条管道的最后一个小特性是 *Slack* 通知，它对成功和失败的构建都运行——假设你用`ENABLE_SLACK`打开它。你需要提供的只是使用`SLACK_WEBHOOK`库秘密的 Slack channel webhook。

要生成上述 webhook，请遵循 [README](https://github.com/MartinHeinz/workflows/tree/v1.0.0#configure-slack-notification) 中的注释。

![](img/566c998b2fce8e08eaa5b313eebb8871.png)

宽限通知-按作者排序的图像

# 结束语

这应该是让您的端到端完全配置的管道启动并运行所需的一切。

如果定制选项或特性不完全符合您的需求，请随意派生[存储库](https://github.com/MartinHeinz/workflows)或提交一个带有特性请求的问题。

除了额外的特性，将来这个库中可能会出现更多 Python(或其他语言)的管道选项，因为这里介绍的管道并不适合所有类型的应用程序。因此，如果您对这些或任何新的开发感兴趣，请确保不时地查看存储库以了解任何新的发布。

*本文最初发布于*[*martinheinz . dev*](https://martinheinz.dev/blog/69?utm_source=medium&utm_medium=referral&utm_campaign=blog_post_69)

[](/optimizing-memory-usage-in-python-applications-f591fc914df5)  [](/speeding-up-container-image-builds-with-remote-cache-c72577317886)  [](/all-the-things-you-can-do-with-github-api-and-python-f01790fca131) 