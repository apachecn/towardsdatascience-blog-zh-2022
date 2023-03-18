# 实现 DBT 的持续集成

> 原文：<https://towardsdatascience.com/enabling-continuous-integration-for-dbt-f77a5b8ce994>

## 使用 Github Actions 和 Docker 自动化测试、宏、种子文件、完整 dbt 加载等等

![](img/94c71cd9eaa900b643f909af5c6c21f5.png)

马库斯·温克勒在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

作为一名数据工程师，我认为工作要求的一句俏皮话是“能够解决问题”。我喜欢我工作的这一部分。对我来说，处理一个任务或问题并为它建立一个解决方案是非常值得的。

虽然我喜欢在工作中解决问题和开发解决方案，但我*不*特别是喜欢重复的任务。当我发现自己一遍又一遍地做同样的事情时，我经常问自己“这里缺少了什么？我能做些什么来简化或自动化这一过程？”

这个问题是最近用 [dbt](https://docs.getdbt.com/docs/introduction) (数据构建工具)提出来的。我使用 dbt 来构建数据仓库和模型，供业务分析师和执行报告在下游使用。dbt 是一个非常棒的工具，它有很多我不会在本文中介绍的好处。尽管对于 dbt 的所有优点来说，我目前并不喜欢为生产准备 dbt 代码的过程。

我们有一个标准的 dbt 命令链，希望在打开 PR 进行审查之前运行。该链通常如下所示，可能需要大约 20-30 分钟来运行，具体取决于我们拥有的模型和测试的数量:

```
dbt deps
dbt seed
dbt snapshot
dbt run
dbt run-operation {{ macro_name }}
dbt test
```

即使错过这些命令中的一个，也可能导致生产管道中的故障，从而导致下游消费者的大规模中断。

虽然我知道开发和测试干净的代码有多重要，但我不想等着输入每一个命令并希望它们成功。我更愿意提交我的提交，打开一个 PR，让一个自动化的工作流为我处理这个过程，同时我开始一个新的任务。然后我可以稍后回来检查运行的结果。

为了实现这一点，我构建了一个自动化的 CI 工作流，它利用了一个 Docker 容器，该容器可以启动、安装 dbt 及其依赖项，并通过 GitHub 操作运行我的存储库中可用的模型和测试。

## 要求

1.  用模型和数据源配置的现有 dbt 项目(我使用的是雪花)
2.  GitHub 回购
3.  Dockerfile 文件

## 码头集装箱

我将`python:3.9-bullseye`作为 Docker 容器的基础[图像](https://hub.docker.com/layers/library/python/3.9-bullseye/images/sha256-a8bb865d30b5eb878f26d19479fe8ec258efe410a459271476e07eef854e9d66?context=explore)。从那里，我将 dbt 目录复制到容器中，安装所需的 dbt 包，并添加构建参数来处理 dbt 的环境雪花凭证*。

在文件的后半部分，我将目标环境切换到`test`并执行 PR 批准所需的必要 dbt 命令。

传入构建参数，然后将这些值设置为 DBT 的容器环境变量，这似乎有些重复，但是我无法找到一种方法来使用 docker 构建命令设置环境变量，就像您可以使用构建参数一样。如果你对此有更好的想法，那么我们来聊聊吧！

## 工作流文件

接下来，应该构建 docker 容器，并在打开 pull 请求和添加其他提交时运行列出的 dbt 命令。然而，我只希望这个动作在`/dbt`目录中发生变化时执行。

为了执行这个工作流文件，我建议对您的 dbt 数据源凭证使用类似的存储库机密。这个文件应该放在项目根目录下的`.github/workflows`目录中。

有了这两个文件，您的项目应该可以开始了！每当打开包含对`/dbt`目录所做更改的 PR 时，就会触发一个工作流来启动 Docker 容器并运行 Docker 文件中列出的指定 dbt 命令。

这是为数据仓库构建可靠的 CI(持续集成)管道的第一步。CI 的[好处](https://about.gitlab.com/topics/ci-cd/benefits-continuous-integration/#:~:text=Continuous%20integration%20(CI)%20makes%20software,the%20overall%20pace%20of%20innovation.)可以包括使开发更容易、更快、风险更小。在将变更推向生产之前，对其进行彻底的测试，这可以建立信任，并获得涉众和开发团队的支持。我强烈建议在您的数据仓库生命周期中实现和自动化 CI 流程(如果还没有实现的话)。

如果你对这篇文章的内容有其他问题，请联系我，如果我的其他一些帖子引起了你的注意，请查看一下！

*注意:我只是在一个私有的 org 存储库中构建带有 GitHub 动作的 Docker 容器。我能够管理该存储库的适当机密，并使用这些机密来存储该配置项的凭据。我不建议将你的容器推送到注册中心，因为在存储库机密和凭证方面存在潜在的安全隐患。