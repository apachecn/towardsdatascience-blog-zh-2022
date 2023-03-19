# Python 中的 requirements.txt 与 setup.py

> 原文：<https://towardsdatascience.com/requirements-vs-setuptools-python-ae3ee66e28af>

## 在开发和分发包时，理解 Python 中 requirements.txt、setup.py 和 setup.cfg 的用途

![](img/b48bcbebc23576044f82b515279135db.png)

照片由 [Eugen Str](https://unsplash.com/@eugen1980?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/tool?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

## 介绍

管理 Python 项目中的依赖关系可能相当具有挑战性，尤其是对于语言新手来说。当开发一个新的 Python 包时，您可能还需要使用一些其他的包，这些包最终会帮助您编写更少的代码(用更少的时间)，这样您就不必重新发明轮子了。此外，您的 Python 包也可能在未来的项目中用作依赖项。

在今天的文章中，我们将讨论如何正确管理 Python 项目的依赖性。更具体地说，我们将讨论`**requirements.txt**`文件的用途以及如何使用`**setuptools**`来分发您的 Python 包并让其他用户进一步开发它。因此，我们还将讨论设置文件(即`**setup.cfg**`和`**setup.py**`)的目的，以及如何将它们与需求文件一起使用，以使包的开发和再分发更加容易。

## Python 项目的依赖关系是什么

首先，让我们从包依赖关系开始讨论；它们到底是什么，以及为什么为了让 Python 项目更容易维护而正确管理它们很重要。

简单来说，依赖项是您自己的项目所依赖的外部 Python 包，以便完成工作。在 Python 环境中，这些依赖关系通常可以在 Python 包索引(PyPI)或其他存储库管理工具中找到，比如 Nexus。

作为一个例子，让我们考虑一个使用熊猫数据帧的 Python 项目。在这种情况下，这个项目依赖于`pandas`包，因为没有预装 pandas 它就不能正常工作。

每个依赖项——它本身又是一个 Python 包——也可能有其他依赖项。因此，依赖性管理有时会变得非常棘手或具有挑战性，需要正确处理，以避免在安装甚至增强软件包时出现问题。

现在，我们自己的 Python 项目可能依赖于第三方包的特定版本。因为这是一种情况，我们也可能以依赖冲突结束，其中(至少)两个依赖可能依赖于另一个包，但是每一个都需要那个外部包的特定版本。这些情况都是可以处理的(嗯，不一定！)通过`pip`等包管理工具。不过，通常我们应该指导`pip`它需要如何处理依赖关系，以及我们需要什么特定的版本。

处理依赖关系和指导包管理工具在我们自己的项目中需要什么特定版本的最常见方法是通过一个需求文本文件。

## requirements.txt 文件

`requirements.txt`是一个文件，列出了特定 Python 项目的所有依赖项。如前所述，它还可能包含依赖关系的依赖关系。列出的条目可以是固定的，也可以是非固定的。如果使用 pin，那么您可以指定一个特定的包版本(使用`==`)、一个上限或下限或者两者都指定。

**例题** `**requirements.txt**` **文件**

```
matplotlib>=2.2
numpy>=1.15.0, <1.21.0
pandas
pytest==4.0.1
```

最后，您可以使用以下命令通过`pip`安装这些依赖项(通常在虚拟环境中):

```
pip install -r requirements.txt
```

在上面的示例需求文件中，我们使用不同的引脚指定了一些依赖关系。例如，对于没有关联 pin 的`pandas`包，`pip`通常会安装最新版本，除非其他依赖项之一与此有任何冲突(在这种情况下，`pip`通常会安装满足其余依赖项指定条件的最新`pandas`版本)。现在，对于`pytest`，包管理器将安装特定版本(即`4.0.1`)，而对于 matplotlib，将安装最新版本，该版本至少大于或等于`2.2`(同样，这取决于另一个依赖项是否指定了其他版本)。最后，对于`numpy`包，`pip`将尝试安装版本`1.15.0`(含)和`1.21.0`(不含)之间的最新版本。

一旦安装了所有的依赖项，就可以通过运行`pip freeze`来查看虚拟环境中安装的每个依赖项的精确版本。该命令将列出所有封装及其特定引脚(即`==`)。

需求文件非常有用，但是在大多数情况下，它必须用于开发目的。如果您计划分发您的包以便广泛使用(比如在 PyPI 上)，您可能需要比这个文件更多的东西。

## Python 中的 setuptools

`[setuptools](https://setuptools.pypa.io/en/latest/)`是建立在`distutils`之上的一个包，允许开发者开发和分发 Python 包。它还提供了使依赖性管理更容易的功能。

当你想发布一个包时，你通常需要一些**元数据**，包括包名、版本、依赖项、入口点等等。而`setuptools`恰恰提供了这样的功能。

项目元数据和选项在一个`setup.py`文件中定义，如下所示。

```
from setuptools import setup setup(     
    name='mypackage',
    author='Giorgos Myrianthous',     
    version='0.1',     
    install_requires=[         
        'pandas',         
        'numpy',
        'matplotlib',
    ],
    # ... more options/metadata
)
```

事实上，考虑到文件是纯声明性的，这被认为是一个有点糟糕的设计。因此，更好的方法是在名为`setup.cfg`的文件中定义这些选项和元数据，然后在`setup.py`文件中简单地调用`setup()`。一个示例`setup.cfg`文件如下所示:

```
[metadata]
name = mypackage
author = Giorgos Myrianthous
version = 0.1[options]
install_requires =
    pandas
    numpy
    matplotlib
```

最后，您可以拥有一个最小的`setup.py`文件:

```
from setuptools import setup if __name__ == "__main__":
    setup()
```

请注意，`install_requires`参数可以接受一个依赖项列表以及它们的说明符(包括操作符`<`、`>`、`<=`、`>=`、`==`或`!=`)，后跟一个版本标识符。因此，当安装项目时，环境中尚未满足的每个依赖项都将位于 PyPI 上(默认情况下)，并被下载和安装。

关于`setup.py`和`setup.cfg`文件之间的区别，你可以阅读这篇文章

</setuptools-python-571e7d5500f2>  

## 我们需要 requirements.txt 和 setup.py/setup.cfg 文件吗？

良好的..看情况！首先，我想澄清的是，`requirements.txt`与`setup.py`之间的矛盾并不完全对应于苹果与苹果之间的比较，因为它们通常被用于实现不同的事情。

这是一个关于这个主题的常见误解，因为人们通常认为他们只是在这些文件中复制信息，他们应该使用其中的一个。然而实际情况并非如此。

首先，让我们了解一下，通常情况下，您是拥有这两个文件，还是只有其中一个文件。**作为一般经验法则**:

*   如果你的包主要是用于开发目的，但是你不打算重新发布它，`**requirements.txt**` **应该足够了**(即使当包在多台机器上开发时)。
*   如果你的包只由你自己开发(即在一台机器上)但你打算重新发布它，那么`**setup.py**` **/** `**setup.cfg**` **应该足够了**。
*   如果您的包是在多台机器上开发的，并且您还需要重新发布它，那么**需要** `**requirements.txt**` **和** `**setup.py**` **/** `**setup.cfg**` **文件**。

现在，如果你需要使用这两种符号(老实说，几乎总是如此)，那么你需要确保你不会重复自己。

如果您同时使用两者，您的`**setup.py**` **(和/或** `**setup.cfg**` **)文件应该包括抽象依赖项列表，而** `**requirements.txt**` **文件必须包含具体的依赖项**，并带有每个包版本的特定管脚(使用`==`管脚)。

> 鉴于`install_requires`(即`setup.py`)定义了单个项目的依赖关系，[需求文件](https://pip.pypa.io/en/latest/user_guide/#requirements-files)通常用于定义完整 Python 环境的需求。
> 
> 鉴于`install_requires`需求是最小的，需求文件通常包含一个详尽的固定版本列表，目的是实现一个完整环境的[可重复安装](https://pip.pypa.io/en/latest/user_guide/#repeatability)。
> 
> - [Python 文档](https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/#requirements-files)

## 最后的想法

在今天的教程中，我们讨论了在开发 Python 项目和应用程序时，正确的依赖性管理的重要性。我们讨论了`requirements.txt`文件的用途，以及如何与`setuptools`(即`setup.py`和`setup.cfg`)的安装文件一起使用，以确保其他开发人员可以安装、运行、开发甚至测试 Python 包的源代码。

正如已经强调的，`setuptools`并不能完全取代`requirements.txt`文件。在大多数情况下，您将需要这两个文件都存在，以便正确地管理您的包依赖关系，以便以后可以重新分发。

如果您现在想知道如何分发您的 Python 包并使其在 PyPI 上可用，以便它也可以通过`pip`包管理器安装，请务必阅读下面的文章。

</how-to-upload-your-python-package-to-pypi-de1b363a1b3>  **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

<https://gmyrianthous.medium.com/membership>  

**相关文章你可能也喜欢**

</8-must-know-venv-commands-for-data-scientists-and-engineers-dd81fbac0b38>  </pycache-python-991424aabad8>  <https://betterprogramming.pub/11-python-one-liners-for-everyday-programming-f346a0a73f39> 