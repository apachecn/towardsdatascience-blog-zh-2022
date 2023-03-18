# 使用 Crontab 和 HTTP 请求按照自定义计划运行 Airbyte 同步

> 原文：<https://towardsdatascience.com/run-airbyte-syncs-on-a-custom-schedule-with-crontab-and-http-requests-8d6fbb3ea112>

## 利用 Airbyte 的简单 REST API 进行复杂配置

![](img/fa9f2b6ce8f56215a4cc99b8cdf2a533.png)

照片由[尹英松](https://unsplash.com/@insungyoon?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

Airbyte 是一个神奇的工具，最近在数据工程社区掀起了波澜。这是一个用于运行 ELT(提取、加载和转换)过程的开源工具。它是 Fivetran 等昂贵软件的绝佳替代品，每月可为您的公司节省数千美元。

从现在开始，我将假设您对 Airbyte 的结构以及如何在生产环境中使用它有一定的了解。

Airbyte 很棒，但是，就目前的情况来看，Airbyte 对连接同步的 cron 式调度没有一流的支持。这很快就会到来，但还没有完全准备好，我的公司运行的是旧版本的 Airbyte，还不需要升级。

当谈到目前的日程安排，有几个选项，如让您的同步运行每分钟，每小时，每天等。，但是如果你想要更具体的东西，或者如果你想自己编排你的同步，在 UI 中是不可能的，这是大多数用户在大多数情况下与 Airbyte 交互的方式。

但是不要害怕！Airbyte 不仅提供了一个很好的 UI 来管理您的 ELT 过程，而且还提供了一个 HTTP REST API，允许您使用 POST 请求来触发来自外部源的连接。

完整的 API 文档在这里是，但是如果你继续读下去，我将向你展示如何使用 Crontab 和 Python 来按照你自己的计划运行你的同步。当然，您可以使用任何您喜欢的工具和语言。

# ⚙️连接到 API

Airbyte 公开的所有端点都是 POST 端点，这使得它非常容易交互。因为 Airbyte 是容器化的，所以我将假设您正在使用 Airbyte 附带的所有默认端口，并且您将在同一台机器上运行您的 Python 脚本。

下面的代码非常简单，将使用 API 同时触发每个启用的连接，但是它应该让您对需要访问的端点有一定的了解。

如果您使用 Python，您需要确保安装了`requests`模块，然后开始编写一个函数来获取与您的安装相关联的 Airbyte 工作空间:

Airbyte 中的每个工作区都有多个源、多个目的地和多个连接，这些连接决定了数据从源到目的地的流动。为了触发同步，我们首先必须获得每个工作区的连接列表，然后单独触发每个连接。我们可以通过以下两个函数来实现:

然后，我们要做的就是把它打包成一个整洁的小包，这样整个程序看起来就像这样:

这就是 Python 的一面。剩下的就是安排这个脚本运行了！

# 🕒计划 Airbyte 同步

同样，您可以使用任何您喜欢的工具，但是我喜欢使用 Crontab，因为它简单快捷。将上面的脚本调整为使用 Python `schedule`包并一直在后台运行也是可行的。

如果您决定只使用 Crontab，请确保它安装在您的机器上。我喜欢使用 Crontab.guru 来确保我会得到我想要的时间表。我不会在这里给出完整的教程，但是如果您想在每小时的早上 6 点到下午 6 点之间运行 Airbyte 同步，您可以使用`crontab -e`命令将下面一行插入 Crontab:

`00 6-18 * * * python3 force_airbyte_sync.py`

保存您的 Crontab，并确保使用`crontab -l`将其正确插入。然后坐下来，放松，享受在你的英语教学过程中少花 50%的钱！

我希望这篇文章至少对你们有些人有用！如果你有任何问题，欢迎评论这篇文章或发邮件给我，地址是 isaac@isaacharrisholt.com。我喜欢谈论数据和所有工程方面的东西，所以为什么不开始一个对话呢！

也可以在 [**GitHub**](https://github.com/isaacharrisholt/) 上找我或者在 [**LinkedIn**](https://www.linkedin.com/in/isaac-harris-holt/) 上联系我！我喜欢写我的项目，所以你可能也想在 [**Medium**](https://isaacharrisholt.medium.com/) 上关注我。

[](/parsing-qif-files-to-retrieve-financial-data-with-python-f599cc0d8c03) [## 使用 Python 解析 QIF 文件以检索金融数据

### Quiffen 包的基本概述以及它如此有用的原因

towardsdatascience.com](/parsing-qif-files-to-retrieve-financial-data-with-python-f599cc0d8c03) 

***这一条就这么多了！我希望你喜欢它，任何反馈都非常感谢！***