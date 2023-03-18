# 用诗歌管理 Python 依赖关系

> 原文：<https://towardsdatascience.com/python-poetry-83f184ac9ed1>

## 依赖性管理和用诗歌打包

![](img/2e9d07596c569118537205d65bee3dce.png)

[张彦宏](https://unsplash.com/@danielkcheung?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/lego?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

依赖性管理是 Python 项目最基本的方面之一。这很快就会变成一场噩梦，尤其是当一个包由许多开发者维护的时候。因此，使用正确的工具非常重要，这些工具最终可以帮助项目的维护者以正确的方式处理依赖关系。

依赖性管理的概念也涉及到包升级——仅仅因为您指定了一个 pin 并不意味着您将忽略更新。

在今天的文章中，我们将讨论诗歌——一种可以帮助你处理依赖性的依赖性管理工具。此外，我们还将讨论诸如*dependent bot*之类的工具，这些工具用于自动化依赖性更新。

## 为什么固定依赖关系很重要

首先，让我强调一个事实，即**依赖项必须被钉住**。一直都是。是的，甚至致力于概念验证。是的，即使只有你一个人在做一个项目。

钉住的依赖项将允许维护者(或者甚至其他可能使用你的包作为依赖项的包)复制一个环境，该环境具有应该被安装的包依赖项的精确版本，以使你的包完全起作用。

有了固定的依赖项，并不意味着您不应该关心 PyPI 上发布的更新版本。确保固定的软件包是最新的非常重要，因为更新的版本通常会修复错误并减少安全漏洞。

这敲响了良好和可靠的单元测试的警钟。如果您未能创建一个可靠的测试套件，最终覆盖您源代码的大部分方面，那么您将无法判断依赖升级是否会破坏您的代码。

## 什么是诗歌

诗歌，是 Python 项目的依赖管理和打包工具。换句话说，诗歌将处理您在`pyproject.toml`文件中定义的依赖关系。

如果您的机器上尚未安装诗歌，您可以按照官方文档[中提供的指南进行安装。](https://python-poetry.org/docs/#installation)

现在让我们假设你即将开始一个全新的项目。在这种情况下，您可以通过运行`poetry new`命令来创建一个:

```
$ poetry new myproject
```

该命令将在名为`myproject`的目录下创建一个新项目，包含以下结构:

```
poetry-demo
├── pyproject.toml
├── README.md
├── myproject
│   └── __init__.py
└── tests
    └── __init__.py
```

## pyproject.toml 文件

很长一段时间以来，`setuptools`是用于管理 Python 项目中依赖关系的“事实上的”工具。然而，自从 [PEP-518](https://peps.python.org/pep-0518/) 提出后，这种情况发生了变化。这个 Python 增强提案引入了一个名为`pyproject.toml`的 TOML 文件，它基本上包含了每个项目的所有构建系统依赖项。

现在让我们快速浏览一下默认的由诗歌创造的`pyproject.toml`:

```
**[tool.poetry]**
name = "myproject"
version = "0.1.0"
description = ""
readme = "README.md"
packages = [{include = "myproject"}]

**[tool.poetry.dependencies]**
python = "^3.7"

**[build-system]**
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

第一部分(`[tool.poetry]`)包含一些关于你的项目的一般信息(如果`poetry`被用来在 PyPI 上发布你的包，这些细节也会被用到。

第二部分`[tool.poetry.dependencies]`用于指定 Python 版本以及包的依赖关系，无论是强制的还是可选的依赖关系。举个例子，

```
**[tool.poetry.dependencies]**
python = "^3.7"
pandas = "==1.1.1"# Optional dependencies
dbt-core = { version = "==1.1.1", optional = true }**[tool.poetry.extras]** dbt_core = ["dbt-core"]
```

最后，`poetry`生成的`pyproject.toml`文件的最后一段叫做`[build-system]`。PEP 与 PEP-517 兼容，PEP-517 引入了一种新的标准方法，用于在维护项目时指定构建系统。因此，本节用于指定具体的构建系统(例如，这可能是`setuptools`或`poetry`):

```
**[build-system]**
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

有关部分和有效字段的完整列表，请随意参考官方[文档](https://python-poetry.org/docs/pyproject/)。在下面的一节中，我们还将讨论一些常用于操作`pyproject.toml`文件的诗歌命令。

## 用诗歌安装依赖项

现在我们已经在`pyproject.toml`中定义了依赖项，我们可以继续安装它们了。为此，只需运行`install`命令:

```
$ poetry install
```

该命令将读取 TOML 文件，解析所有的依赖项，并最终将它们安装在一个虚拟环境中，默认情况下，poems 将在`{cache-dir}/virtualenvs/`下创建这个虚拟环境。注意`cache-dir`目录可以作为[诗歌配置](https://python-poetry.org/docs/configuration/#cache-dir)的一部分进行修改。

如果出于某种原因想要激活环境，只需运行以下命令创建一个新的 shell

```
$ poetry shell
```

现在，如果您想停用虚拟环境，同时退出 shell，运行`exit`。或者，要停用虚拟环境但保持 shell 活动，运行`deactivate`。

## 锁定文件

一旦你运行`poetry install`命令，两件事之一就会发生。如果这是你第一次运行`poetry install`，那么一个名为`poetry.lock`的文件将不会出现。因此，诗歌将读取依赖项，并下载您的`pyproject.toml`文件中指定的所有最新版本。一旦安装了这些依赖项，这些包及其特定版本将被写入一个名为`poetry.lock`的文件中。

现在，如果`poetry.lock`文件已经存在，那么这是因为你之前已经运行过`poetry install`，或者`poetry.lock`已经被维护代码库的其他人推送了。不管是哪种情况，只要存在一个`poetry.lock`文件并且执行了`poetry install`，那么依赖项将直接从锁文件中安装。

每当您手动将依赖项添加到`pyproject.toml`文件中时，您可能还希望`poetry.lock`文件反映这些更改。为此，你必须跑步

```
$ poetry lock --no-update
```

即使您的锁文件由于任何原因已经过期，也可以使用相同的命令。

请注意，您应该始终对`poetry.lock`文件进行版本控制，因为确保所有贡献者使用相同版本的依赖项非常重要。每当用户添加或删除一个依赖项时，他们必须确保他们也更新了`poetry.lock`并将其包含在提交中。

## 为开发指定和安装依赖项

现在，除了项目运行所需的标准依赖项之外，您可能还需要定义额外的依赖项，作为测试的一部分。此类依赖关系可在`[tool.poetry.dev-dependencies]`一节中定义:

```
**[tool.poetry.dev-dependencies]** pytest = "==3.4"
```

现在每次运行`poetry install`时，开发依赖项也会被安装。如果不希望在环境中安装`dev-dependencies`部分列出的包，那么您可以提供`--no-dev`标志:

```
$ poetry install --no-dev
```

请注意，在较新的版本中，上述符号已被否决。您可以改为使用`--without`选项:

```
$ poetry install --without dev
```

## 有用的诗歌命令

您甚至可以使用`poetry add`命令在`pyproject.toml`中插入额外的依赖项，而不是在`pyproject.toml`文件中指定依赖项:

```
$ poetry add pytest
```

上面的命令将执行两个动作——它将把`pytest`包添加到`pyproject.toml`的 dependencies 部分，并安装它(及其子依赖项)。

同样，您甚至可以从依赖关系中删除一个包:

```
$ poetry remove pytest
```

注意，您甚至可以通过传递`--dev|-D`标志来添加或删除开发依赖关系:

```
$ poetry add pytest -D
```

现在让我们假设您想要使用`pytest`来运行项目的单元测试。因为您的测试套件需要已经安装在诗歌环境中的依赖项，所以您可以使用`poetry run`命令在该环境中运行`pytest`:

```
$ poetry run pytest
```

事实上，您可以运行任何您喜欢的 Python 脚本

```
$ poetry run python my_script.py
```

诗歌还支持`extras`，它本质上对应于我们之前讨论过的可选依赖项。

```
[tool.poetry.extras]
dbt_core = ["dbt-core"]
pandas = ["pandas"]
```

如果你想安装`extras`，你只需要使用`-E|--extras`选项:

```
poetry install --extras "dbt_core pandas"
poetry install -E dbt_core -E pandas
```

## 保持依赖关系最新

既然我们已经学习了诗歌，以及如何利用诗歌来有效地管理包的依赖关系，那么强调你应该确保这些依赖关系不时地被更新是很重要的。

这将确保您的源代码是安全的(因为许多包升级将减少安全漏洞)，具有尽可能少的错误，并且它还可以访问包的最新特性。

由于维护这样的过程相当困难——尤其是如果您的 Python 包有大量的依赖项——您可能不得不依赖能够自动更新依赖项的外部工具。

如果您在 GitHub 上托管您的存储库，一个选择是[dependent bot](https://github.com/dependabot)，它可以自动搜索依赖项的最新更新并创建 Pull 请求。你所需要做的就是准备好良好的单元测试，以便能够捕获软件包更新可能引入的任何 bug。

## 最后的想法

无论如何，您都必须为依赖项(也称为 pin)指定特定的包版本。鉴于您的 Python 项目将有一组特定的需求，需要在运行它之前安装这些需求，因此使用能够简化依赖关系管理的工具非常重要。

其中一个工具是 poem——您可以使用它来安装、添加和删除依赖项，以及打包您的 Python 项目，以便它可以在 PyPI 上发布和分发，从而可以被更广泛的社区访问。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership) [## 通过我的推荐链接加入 Medium-Giorgos Myrianthous

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

gmyrianthous.medium.com](https://gmyrianthous.medium.com/membership) 

**相关文章你可能也喜欢**

[](/requirements-vs-setuptools-python-ae3ee66e28af) [## Python 中的 requirements.txt 与 setup.py

### 了解 Python 中 requirements.txt、setup.py 和 setup.cfg 在开发和分发时的用途…

towardsdatascience.com](/requirements-vs-setuptools-python-ae3ee66e28af) [](/setuptools-python-571e7d5500f2) [## Python 中的 setup.py 与 setup.cfg

### 使用 setuptools 管理依赖项和分发 Python 包

towardsdatascience.com](/setuptools-python-571e7d5500f2) [](/how-to-upload-your-python-package-to-pypi-de1b363a1b3) [## 如何将 Python 包上传到 PyPI

towardsdatascience.com](/how-to-upload-your-python-package-to-pypi-de1b363a1b3)