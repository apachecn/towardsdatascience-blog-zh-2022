# 使用 inotifywait 实现数据处理自动化

> 原文：<https://towardsdatascience.com/data-processing-automation-with-inotifywait-663aba0c560a>

## 如何在拥有生产就绪的 MLOps 平台之前实现自动化

![](img/fcb703de2552865a4b870def03c08680.png)

由[活动创作者](https://unsplash.com/@campaign_creators?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

当在集群或任何其他类型的共享处理系统中工作时，一个很快出现的问题是如何容易地共享数据处理工具。所有用户都有自己的环境、库，并且软件的功能可以秘密地依赖于. bashrc 中定义的一些路径。如果编写一些有价值的数据处理工具却不能与真正需要它们的人共享，那就太遗憾了。

此外，你真的不希望人们每次需要时都要求你为他们处理，是吗？

很明显，答案是使用 MLOps 平台，以及像特性库这样的工具。然而，当一个还没有到位时，建立一个需要时间，甚至更糟的是，从零开始建立它。在这种情况下，我们在等待的同时有什么可以做的吗？

<https://www.featurestore.org/what-is-a-feature-store>  

任何事情都比花大量时间运行来自公司的所有请求的流程，而不是进一步开发您的数据处理或新功能/项目要好。

节省时间的解决方案显然是自动化，但您不必运行完整的 MLOps 管道就可以开始享受自动化的好处。

# 什么是 inotifywait？

inotifywa 是一个用于 linux 的 cli 工具，它只做一件事:监听目录中发生的事件，并在一个事件发生时打印一条消息。它基于一个名字相似的工具:`inotify`，但是`inotifywait`可以递归地监听子文件夹，而前者则不能。

就是这样。倾听事件，并在事件发生时写下一些东西。然而，这足以开启有用的自动化。

事实上，每次它监听一个目录或其中一个子目录上的新事件时，都会打印一条消息，其中包含事件发生的路径、触发事件的文件的文件名以及事件类型。然后，在 bash 中，您可以读取这样的消息，并根据事件类型决定一个动作。例如，您可以运行一个数据处理脚本，该脚本在目录中新创建的文件上运行。

# 安装 inotifywait

inotifywait 通常不是发行版默认设置的一部分，而是 inotify-tools 的一部分。它的安装依赖于 linux 发行版，可以从[源](https://docs.rockylinux.org/books/learning_rsync/06_rsync_inotify/)或[包](https://ubuntu.pkgs.org/20.04/ubuntu-universe-amd64/inotify-tools_3.14-8_amd64.deb.html) [管理器](https://src.fedoraproject.org/rpms/inotify-tools)中完成。只需选择最适合您的使用案例和发行版的选项。

请注意，它总是需要管理员权限才能安装。

# 一个例子

让我们看看如何使用 inotifywait 递归地监视目录，并发出文件创建的所有事件。换句话说，每当文件夹中的文件创建完成时，它都会打印一些内容。 *Complete* 在这里很重要，因为我们不希望事件在开始写文件时被触发，而是在结束时被触发。

```
$ inotifywait -m /path/to/target/folder -e close_write -r |
    while read dir action file; do
        python data_processing.py ${dir}/${file}
    done
```

让我们一点一点来看。首先，我们启动 inotifywait 来监控在`-m`之后指定的目录，由于`-r`的原因，递归地进行监控。`-e close_write`表示发出所有且仅发出文件创建类型的事件。

然后，`inotifywait`监听该文件夹，每次在其中或其子文件夹中创建新文件时，它都会写入 3 条信息:事件发生的目录路径、事件类型和文件名。本例中的事件类型只有`close_write`，但是我们可以指定多个参数到-e，用逗号分隔。查看 [inotifywait 手册](https://linux.die.net/man/1/inotifywait)了解更多关于事件类型的信息。我们可以用文件名来处理文件。

脚本的第二部分，`while read dir action file`读取来自 inotifywait 的行(因为它在管道之前)，并将每行的三个部分分配给 bash 变量`$dir, $action, $file`。最后，我们可以使用这些变量来启动我们的数据处理工具。

让我们假设我们有一个新工具，它运行一些文本规范化规则，然后用附加信息(如词性标记)注释文本。这些转换对于团队中的其他人或者其他团队执行的一些下游任务是必需的。

</implementing-part-of-speech-tagging-for-english-words-using-viterbi-algorithm-from-scratch-9ded56b29133>  

然后，我们可以要求有新文本数据的人将它们添加到受监控的目录中，这个简单的脚本会在它们被添加到文件夹后立即处理它们。

与目前市场上提供的相比，它看起来像是自动化的原始形式，但是如果团队还没有进行 MLOps，那么它可能是一个合理的选择。

# 结论

在我们快节奏的工作环境中，做得好的自动化是生产力的关键，你的工作是(一种)软件工程师或数据科学家。当它起作用时，它完成工作，而你可以把你宝贵的精神能量花在新的挑战性任务上。

然而，正确地完成自动化本身需要大量的工作，如果没有合适的自动化框架就更是如此。

幸运的是，几十年来在操作系统工具方面的工作给了我们大量的工具来设置简单的自动化，这些工具可以在准备“真正的东西”时使用。是这些工具中的一种，你对它的使用只受你想象力的限制。

*感谢您的阅读！*

# 更多来自我

</3-common-bug-sources-and-how-to-avoid-them-182f9974d2ab>  </parse-dont-validate-f559372cca45>  </machine-translation-evaluation-with-cometinho-c89880731409> 