# Python 中的环境变量

> 原文：<https://towardsdatascience.com/environment-variables-in-python-a37cab6d2530>

## 在本文中，我们将探索如何在 Python 中使用环境变量

![](img/9ea399b7f660f688121e6a14a68d9ce4.png)

由 [Unsplash](https://unsplash.com/s/photos/environment?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的[卡斯滕·沃思](https://unsplash.com/@karsten_wuerth?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)拍摄的照片

**目录**

*   介绍
*   如何使用 Python 获取环境变量
*   如何使用 Python 设置环境变量
*   结论

# 介绍

环境变量是存储在程序之外(在操作系统级别)的变量，它们会影响程序的运行方式，并有助于存储敏感信息。

例如，你的程序可能正在使用一些有密钥的 API。这个密钥应该作为环境变量存储在您正在运行的代码之外。

但我们为什么要这么做？

使用环境变量有两个主要优点:

1.  **提高敏感数据的安全性**——当多个开发人员在一个代码库上工作时，您可能希望将您的密钥放在代码库之外，以提高安全性。或者，如果您在 GitHub 上共享某个程序的源代码，您希望将您的 API 密匙保密，不要让任何下载代码的人都可以访问。
2.  **自动化** —将秘密存储在环境变量中有助于避免手动更新代码库，例如，当您在其他人的机器上运行相同的代码时。因此，您可以使用环境变量自动更新源代码，而不是根据运行代码的不同“用户”来更改源代码。

接下来，让我们探索如何在 Python 中使用环境变量！

在本教程中，我们将使用 [os](https://docs.python.org/3/library/datetime.html) 模块，该模块提供了一种使用操作系统相关功能的可移植方式，因此不需要额外安装。

# 如何使用 Python 获取环境变量

在 Python 中，环境变量是通过 os 模块实现的，并且可以使用 **os.environ** 进行检索，它提供了表示流程环境的键值对的映射。

要使用 Python 获取所有环境变量，只需运行:

```
#Import the required dependency
import os

#Get all environment variables
env_vars = os.environ

#Print environment variables
for key, value in dict(env_vars).items():
    print(f'{key} : {value}')
```

您应该可以打印出环境变量的完整映射:

```
TERM_PROGRAM : vscode
SHELL : /bin/bash
TERM_PROGRAM_VERSION : 1.72.2
ORIGINAL_XDG_CURRENT_DESKTOP : undefined
USER : datas
PATH : /Library/Frameworks/Python.framework/Versions/3.7
__CFBundleIdentifier : com.microsoft.VSCode
LANG : en_US.UTF-8
HOME : /Users/datas
COLORTERM : truecolor
_ : /usr/local/bin/python3
__PYVENV_LAUNCHER__ : /usr/local/bin/python3
```

为了获得特定环境变量的值，我们将使用 **os.environ.get()** 。它将检索指定的环境变量的值。

例如，我想检索用户环境变量:

```
#Import the required dependency
import os

#Get specific environment variable
user_var = os.environ.get('USER')

#Print environment variables
print(user_var)
```

您应该可以打印出系统上设置的特定用户变量。

我的情况是:

```
datas
```

如果你要查找一个没有匹配关键字的环境变量，代码应该简单地返回“None”。

让我们尝试查找一些不存在的环境变量，比如“TEST_PASSWORD”:

```
#Import the required dependency
import os

#Get specific environment variable
pwd_var = os.environ.get('TEST_PASSWORD')

#Print environment variables
print(pwd_var)
```

您应该得到:

```
None
```

# 如何使用 Python 设置环境变量

在 Python 中，环境变量是通过 os 模块实现的，并且可以使用 **os.environ** 进行设置，这与将键值对插入字典是一样的。

环境变量 key 和值都必须是字符串格式。

在上一节中，我们尝试检索一个不存在的环境变量“TEST_PASSWORD”，代码返回“None”。

现在让我们用一个自定义值设置这个环境变量，并将其打印出来:

```
#Import the required dependency
import os

#Set specific environment variable
os.environ['TEST_PASSWORD'] = '12345'

#Get specific environment variable
pwd_var = os.environ.get('TEST_PASSWORD')

#Print environment variables
print(pwd_var)
```

您应该得到:

```
12345
```

**注意:**在运行代码时，对环境变量(在我们的例子中是添加的环境变量)的更改只对这一个会话有效。它不会影响系统环境变量或其他会话中的环境变量。

# 结论

在本文中，我们将探索如何在 Python 中使用环境变量。

我们重点讨论了如何使用 Python 获取现有的环境变量，以及如何在 Python 中为会话设置环境变量。

如果你有任何问题或对编辑有任何建议，请随时在下面留下评论，并查看我的更多 [Python 编程](https://pyshark.com/category/python-programming/)教程。

*原载于 2022 年 12 月 20 日 https://pyshark.com**[*。*](https://pyshark.com/environment-variables-in-python/)*