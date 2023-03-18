# Python 中的 setup.py 与 setup.cfg

> 原文：<https://towardsdatascience.com/setuptools-python-571e7d5500f2>

## 使用`setuptools`来管理依赖项和分发 Python 包

![](img/3aa1f6119abd3058b27b7157489ea601.png)

弗勒在 [Unsplash](https://unsplash.com/s/photos/tools?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

## 介绍

在我最近的一篇文章中，我讨论了`[requirements.txt](/requirements-vs-setuptools-python-ae3ee66e28af)`[和](/requirements-vs-setuptools-python-ae3ee66e28af) `[setup.py](/requirements-vs-setuptools-python-ae3ee66e28af)` [文件](/requirements-vs-setuptools-python-ae3ee66e28af)之间的区别，这些文件最终可以帮助开发人员管理他们的包的依赖关系，以一种他们也很容易重新分发它们的方式。

在今天的文章中，我将集中讨论`setuptools`包，并讨论`setup.py`和`setup.cfg`文件之间的区别。此外，我们还将讨论这些文件如何理解应该列出依赖关系的`requirements.txt`文件。

最后，我们还将讨论`pyproject.toml`文件的用途，以及在接受 PEP-517 后，包分发格局是如何改变的。我们还将演示如何以符合 PEP-517 和 PEP-518 的方式过渡到使用`pyproject.toml`和`setuptools`。

## Python 中的 setuptools

`[setuptools](https://setuptools.pypa.io/en/latest/)`是一个构建在`[distutils](https://docs.python.org/3/library/distutils.html)`之上的库，该库已被弃用(从 Python 3.12 开始将被移除)。这个包提供了广泛的功能，**方便了 Python 项目的打包**，这是一个不断发展的过程，有时会变得相当具有挑战性。

所谓打包，我们通常是指**依赖** **管理**和**打包** **分发**。换句话说，打包就是处理你的项目的依赖关系(甚至依赖关系的依赖关系等等)，以及如何分发你的包，以使它可以被其他项目广泛使用。

## setup.py 文件

`setup.py`文件可能是应该包含在**根 Python 项目目录**中的最重要的文件，它主要有两个主要用途:

1.  它包含了与你的包相关的各种信息，包括**选项和元数据**，比如包名、版本、作者、许可证、最小依赖、入口点、数据文件等等。
2.  用作允许执行打包命令的命令行界面。

**示例 setup.py 文件**

```
# setup.py placed at root directoryfrom setuptools import setupsetup(
    **name**='examplepackage'
    **version**='1.0.1',
    **author**='Giorgos Myrianthous',
    **description**='This is an example project',
    **long_description**='This is a longer description for the project',
    **url**='https://medium.com/@gmyrianthous',
    **keywords**='sample, example, setuptools',
    **python_requires**='>=3.7, <4',
    **install_requires**=['pandas'],
    **extras_require**={
        'test': ['pytest', 'coverage'],
    },
    **package_data**={
        'sample': ['example_data.csv'],
    },
    **entry_points**={
        'runners': [
            'sample=sample:main',
        ]
    }
)
```

下面是对调用`setup()`方法时使用的关键字的解释:

*   `**name**`:这是一个对应于包名的字符串
*   `**version**`:对应包版本号的字符串
*   `**author**`:表示包的作者
*   `**description**`:这是包的简短描述的字符串(通常是单行描述)
*   `**long_description**`:包的更长描述的字符串
*   `**url**`:表示包的 URL 的字符串(通常是 GitHub 存储库或 PyPI 页面)。
*   `**keywords**`:这是一个逗号分隔的字符串(也可以是字符串列表)，包含一些与包相关的关键字
*   `**python_requires**`:这是一个逗号分隔的字符串，包含包支持的 Python 版本的版本说明符。
*   `**install_requires**`:包含包成功运行所需的最小依赖项的字符串列表。
*   `**extras_require**`:字典，其中键对应于额外模式和值的名称，是包含所需最小依赖性的列表。例如，一个额外的模式可以是`test`，其中依赖项列表应该包括所有需要的附加包，以便执行包中定义的测试。
*   `**package_data**`:这是一个字典，其中键是包名，值是 glob 模式列表。
*   `**entry_points**`:这是一个字典，其中键对应于入口点名称，值对应于源代码中定义的实际入口点。

有关元数据和选项的可用关键字的更全面列表，您可以参考官方文档中的[相关章节。](https://setuptools.pypa.io/en/latest/references/keywords.html)

## setup.cfg 文件

传统上，`setup.py`文件用于构建包，例如通过命令行界面使用`build`命令。

```
$ python setup.py build
```

在上面演示的示例`setup.py`文件中，我们看到大部分代码只是列出了一些关于 Python 项目的选项和元数据。事实上，就代码质量和设计而言，这可能不是最好的方法。因此，在`setup.cfg`文件中指定这些包细节可能更合适。

`setup.cfg`是一个 ini 文件，包含`setup.py`命令的选项默认值。您几乎可以在新的`setup.cfg`文件中指定我们在`setup.py`文件中使用的每个关键字，并简单地使用`setup.py`文件作为命令行界面。

**示例 setup.cfg 文件**

```
# setup.cfg file at the root directory**[metadata]**
name = examplepackage
version = 1.0.1
author = Giorgos Myrianthous
description = This is an example project
long_description = This is a longer description for the project
url = [https://medium.com/@gmyrianthous](https://medium.com/@gmyrianthous)
keywords = sample, example, setuptools**[options]**
python_requires = >=3.7, <4
install_requires = 
    pandas**[options.extras_require]**
test = 
    pytest
    coverage**[options.package_data]**
sample = 
    example_data.csv'
```

## 带有 setup.cfg 的 setup.py

现在假设您已经将所有选项移动到一个`setup.cfg`文件中，如前一节所述，您现在可以创建一个虚拟的`setup.py`，它将简单地调用`setup()`方法:

```
from setuptools import setupif __name__ == '__main__':
    setup()
```

关于如何使用`setup.py`和`setup.cfg`文件的更全面的例子，你可以参考 GitHub 上的 [PyPA 示例项目](https://github.com/pypa/sampleproject)。

## pyproject.toml 文件和向 PEP-517 的过渡

截至 [PEP-517](https://www.python.org/dev/peps/pep-0517) 和 [PEP-518](https://www.python.org/dev/peps/pep-0518/) `setuptools`不再是打包 Python 项目时应该使用的事实上的工具。根据 PEP-518 的规范，Python 项目的构建系统依赖项应该包含在遵循 TOML 格式的名为`pyproject.toml`的文件中。

随着时间的推移，`setup.py`在 Python 社区中越来越受欢迎，但是`setuptools`的一个最大问题是，如果不知道可执行文件(即`setup.py`)的依赖关系，就无法执行它。除非您实际执行包含与包依赖关系相关的信息的文件，否则没有办法知道这些依赖关系是什么。

`pyproject.toml`文件应该解决构建工具依赖鸡和蛋的问题，因为`pip`本身可以读取`pyproject.yoml`以及项目所需的`setuptools`或`wheel`的版本。该文件需要一个用于存储构建相关信息的`[build-system]`表。

现在让我们把拼图拼在一起。如果您想使用`setuptools`，您现在必须指定一个`pyproject.toml`文件，其内容如下所示:

```
# pyproject.toml file specified at the root of the directory**[build-system]**
requires **=** **[**"setuptools>=42", "wheel"**]** # PEP 508 specifications.
build-backend **=** "setuptools.build_meta"
```

然后，您还需要指定一个`setup.py`或`setup.cfg`文件，就像我们在本教程的前几节中展示的那样。注意，我个人更喜欢后一种符号。

最后，您可以使用一个构建器来构建您的项目，比如 PyPA build，您可以通过 pip ( `pip install build`)来检索它，并最终调用构建器

```
$ python3 -m build
```

瞧啊。您的发行版现在可以上传到 PyPI 上了，这样它就可以被广泛访问了。

注意，如果你想在可编辑模式下安装软件包(即通过运行`pip install -e .`)，除了`setup.cfg`和`pyproject.toml`之外，你必须有一个有效的`setup.py`文件。您可以使用我在上一节中共享的同一个虚拟设置文件，该文件只对`setup()`方法进行一次调用。

## 那么我还需要 requirements.txt 文件吗？

简短的回答是，**最** **大概是的**。`setuptools`文件应该保存**抽象依赖关系**，而`requirements.txt`应该列出**具体依赖关系**。

为了更全面地了解`requirements.txt`和`setuptools`的目的，你可以阅读我最近分享的一篇文章。

[](/requirements-vs-setuptools-python-ae3ee66e28af) [## Python 中的 requirements.txt 与 setup.py

### 了解 Python 中 requirements.txt、setup.py 和 setup.cfg 在开发和分发时的用途…

towardsdatascience.com](/requirements-vs-setuptools-python-ae3ee66e28af) 

## 最后的想法

在今天的文章中，我们讨论了`setuptools`包，以及如何利用`setup.py`和`setup.cfg`文件来管理包的依赖关系，使包在 Python 中的分发更加容易。

此外，我们讨论了可以指定的各种选项和元数据，以及`setup.py`和`setup.cfg`文件之间的差异。我们还讨论了关于`requirements.txt`文件以及如何使用它来完成 Python 包中的依赖管理难题。最后，我们介绍了 PEP517 和一个名为`pyproject.toml`的文件的用途。

最后，请记住 Python 中的包分发指南是不断发展的——尤其是在过去几年中——所以请确保随时更新并遵循最佳实践。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读媒体上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership) [## 通过我的推荐链接加入 Medium-Giorgos Myrianthous

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

gmyrianthous.medium.com](https://gmyrianthous.medium.com/membership) 

**相关文章你可能也喜欢**

[](/pycache-python-991424aabad8) [## Python 中 __pycache__ 是什么？

### 了解运行 Python 代码时创建的 __pycache__ 文件夹

towardsdatascience.com](/pycache-python-991424aabad8) [](/how-to-upload-your-python-package-to-pypi-de1b363a1b3) [## 如何将 Python 包上传到 PyPI

towardsdatascience.com](/how-to-upload-your-python-package-to-pypi-de1b363a1b3)