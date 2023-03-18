# 本地开发时如何安全存储和检索敏感数据的快速指南

> 原文：<https://towardsdatascience.com/a-quick-guide-on-how-to-safely-store-and-retrieve-sensitive-data-when-developing-locally-cd766d2ea1c5>

## 永远不要在代码中放入用户名、密码或 API 密钥/秘密。利用环境变量保护您的数据安全。

![](img/c3fffbfa549aa1af58ab7e667e5a30ea.png)

[regularguy.eth](https://unsplash.com/@moneyphotos?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

**注意:** *本文建议在本地开发时，使用一种安全的方法来存储和检索环境变量中的敏感数据。在公司环境中工作时，请遵循公司推荐的最佳实践。*

如果您曾经使用过任何 API(比如 Twitter)或者试图连接到任何远程数据源，您可能会遇到需要传递敏感参数的情况，比如 **API 密钥/秘密**、**用户名**和**密码**。你用 GitHub 展示你的作品吗？如果您这样做了，并且您已经在笔记本或代码中以纯文本的形式输入了这些值，那么您肯定会收到电子邮件警告，您的敏感信息是公开的！

```
twitter_api_key = 'frBtFyG7JefJcY76nTlyOuT2iAUg457ndbhKpj9vERw'
```

好消息是，你可以养成一个简单的习惯，让你的钥匙、秘密和敏感信息只属于你一个人。我们将利用**环境变量、**和一个名为`os`的简单 Python 库，它将允许我们从本地机器中检索这些值。

# 什么是环境变量？

**环境变量**存储在您机器本地的用户配置文件中。当执行代码时，它们可以在 python 中动态地准备好，并使敏感数据远离人类可读的代码。

# 添加环境变量

让我们从添加环境变量开始。**。zshrc** 是一个配置文件，包含运行 zsh shell 的命令，就像**一样。包含 bash shell 命令的 bashrc** 文件。您可以运行下面的命令来查看您的**。zshrc** 文件。请注意，这是一个隐藏文件，所以您必须将`-a`参数添加到`ls`命令中。

```
ls -a
```

# 是时候简单介绍一下 VIM 了

现在我们将进入一个小小的 **VIM** 。VIM 是一个开源的基于屏幕的文本编辑器，内置于 MacOS 中。如果你从未接触过 VIM，要习惯它可能有点复杂，但它非常强大。如果你不熟悉 VIM，我建议你看看这个 [VIM 教程](https://www.openvim.com/)。我将运行您将环境变量添加到**所需的基本命令。zshrc** 文件。先来打开我们的**。VIM 中的 zshrc** 文件。

```
vim .zshrc
```

你的**。zshrc** 文件将会打开，并根据写入的其他解决方案进行不同的设置。在我已经安装了 Anaconda 的地方，它可能看起来像这样。

```
# If you come from bash you might have to change your $PATH.
# export PATH=$HOME/bin:/usr/local/bin:$PATH

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/brianroepke/miniforge3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/brianroepke/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/Users/brianroepke/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/Users/brianroepke/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

接下来，我们可以四处导航。VIM 中有两种模式。编辑器将在**正常**模式下打开，在这里您可以浏览文件。您的触控板/鼠标应该可以使用箭头键。滚动到文件底部，按下`i`进入**插入**模式。现在您可以添加您的环境变量。我将在下面补充几个例子。

```
export USER="test_user"
export PW="some_very_hard_to_crack_password"
export HOST="foo.bar.net"
```

按下`esc`退出**插入**模式。现在，您可以在文件中导航。按下`:`进入**命令**模式。键入`wq`并按下`enter`保存并退出。

当你加到**的时候。zshrc** 文件或进行更改，需要重新加载文件才能使更改生效。在 zsh shell 中运行以下命令来重新加载配置文件并使您的更改生效。

```
source ~/.zshrc
```

**注:**我有时发现，即使我这样做了，我也需要重启我的计算机，让 python 能够用`os.environ`命令读取它们。

使用以下命令检查新的**环境变量**:

```
export
```

每当您需要在项目中使用一组新的敏感信息时，请重复此过程。对我来说很有效的一件事是使用描述性的名称，比如`TWITTER_API_KEY`或`TWITTER_API_SECRET`。这样，你就可以很容易地记住它们的用途，并把它们添加到你的**中。zshrc** 文件。

# 在 Python 中使用环境变量

剩下的就简单了！现在我们可以在 **Python** 中使用我们的环境变量了。让我们从导入`os`库开始。

```
import os
```

然后，我们使用`os.environ`命令获取环境变量的值，并将它们存储在内存中。

```
USER = os.environ.get("USER")
PW = os.environ.get("PASS")
HOST = os.environ.get("HOST")
```

像任何其他变量一样，您可以将它们传递给函数、连接字符串或任何您喜欢的东西。下面是一个在连接字符串中使用它们的例子。

```
uri = f"mongodb+srv://{USER}:{PW}@{HOST}"
client = MongoClient(uri)
```

查看我的文章[关于用 Python 从 MongoDB 中提取数据的](/quick-start-from-mongodb-to-pandas-3f777dcbfb6e),了解更多关于使用这个连接字符串的细节。

# 结论

了解如何正确存储和检索敏感数据对于任何数据科学家和分析师都至关重要。这是一个非常简单的习惯，可以确保你不会将敏感信息暴露给外界，或者在你的第一个职业角色中看起来很傻！我们从向**添加环境变量开始。zshrc** 文件。然后我们使用`os`库在 Python 中检索它们。现在你已经准备好了！

如果你喜欢阅读这样的故事，并想支持我成为一名作家，考虑注册成为一名媒体成员。一个月 5 美元，无限量访问数千篇文章。如果你使用 [*我的链接*](https://medium.com/@broepke/membership) *注册，我将免费获得一小笔佣金。*