# 如何在苹果硅 M1 Mac 上管理 Conda 环境

> 原文：<https://towardsdatascience.com/how-to-manage-conda-environments-on-an-apple-silicon-m1-mac-1e29cb3bad12>

## 使用 conda 管理 ARM64 和 x86 Python 环境

![](img/5b0c4e81bbc3c06ff016476ccba52d81.png)

简·kopřiva 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

*本文描述了如何使用 conda 管理 ARM64 和 x86 Python 环境。要使用 pyenv(我的首选工具)和虚拟环境完成同样的事情，请参见我的文章* [*。*](/how-to-use-manage-multiple-python-versions-on-an-apple-silicon-m1-mac-d69ee6ed0250)

如果您使用 Python 的时间足够长，您最终将需要使用不同的 Python 版本来管理环境。当开始新项目时，你会想要利用最新的特性( [Python 3.10](https://docs.python.org/3/whatsnew/3.10.html) 最近发布了！)，但是您还需要维护使用以前版本的旧项目。作为一名数据科学家，我经常遇到这个问题——我经常回到旧的项目和代码，所以我需要一种简单的方法来管理使用不同 Python 版本的多个环境。

这就是康达的用武之地。Conda 是一个非常流行的包和环境管理工具(我们将讨论环境管理方面)。它允许您为不同的项目维护单独的环境——每个环境可以包含不同的包、不同的包版本，甚至不同的 Python 版本——并且可以快速轻松地在它们之间切换。

*注意:本文面向 Mac 用户，尤其是 Apple Silicon Mac 用户，但基本的 conda 指令将在所有平台上工作。*

# 使用 conda 创建 Python 环境

按照这些说明安装 conda 并开始使用环境。

## 1.安装康达

有几种口味的康达可用，其中一些描述如下。按照超链接查找安装说明。每个安装程序都将为您提供相同的环境管理工具。**如果你不确定要安装哪个，就选 Miniforge。**

*   [**Miniforge**](https://github.com/conda-forge/miniforge)***(我的推荐)* : 一个社区驱动的项目，支持多种系统架构。它创建了类似于 Miniconda 的最小 conda 环境(见下一个要点)。**
*   **[**Miniconda**](https://docs.conda.io/en/latest/miniconda.html):conda 官方最小安装程序。它创建了最小的 conda 环境，而没有自动安装任何软件包。**
*   **[**Anaconda**](https://docs.anaconda.com/anaconda/install/index.html) :原康达分布。它会自动将大量的 Python 包安装到新的 conda 环境中，因此它倾向于创建大型且臃肿的环境。我不推荐蟒蛇。**

## **2.创造康达环境**

**创造康达环境非常容易。要使用特定版本的 Python(在本例中为 Python 3.9)创建一个新的 conda 环境，请从您的终端运行以下代码:**

```
conda create -n myenv python=3.9
```

**这将创建一个名为`myenv`的新 conda 环境。要激活 conda 环境，运行`conda activate myenv`(您可以通过`conda env list`查看您所有 conda 环境的列表)。一旦环境被激活，你可以使用`conda install`或`pip install`安装 Python 包——`conda install`更彻底地管理包版本的依赖关系，你必须使用`pip install`来安装不能通过 conda 获得的包。**

**管理 conda 环境的完整文档链接[此处](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)。**

# **苹果硅苹果电脑的其他挑战**

**上述步骤几乎总是足够了。然而，在较新的苹果电脑上(就我个人而言，我用的是 M1 MacBook Air )，你可能会在安装某些软件包时遇到问题。当苹果从英特尔芯片转到他们内部的苹果硅芯片时，他们从 x86 架构变成了 ARM64 架构。这在很大程度上是一件好事——你在日常使用中会注意到的唯一区别是，新芯片比旧芯片更快、更高效。**

**不幸的是，您可能偶尔会遇到包兼容性问题。苹果的 ARM64 架构尚不支持一些 Python 包——例如，我在使用 ortools 包时就遇到了这个问题。您将在安装期间得到错误，并且您将不能在您的代码中使用这个包。最终，当开发人员为他们的包添加 ARM64 支持时，这个问题应该会消失，但与此同时，您必须找到另一个解决方案。**

**幸运的是，conda 提供了一个简单的短期解决方案:**您可以在 Apple Silicon Mac 上轻松创建 x86 架构的 conda 环境。**很简单，不需要安装任何额外的软件。**

# **使用 x86 架构创建 conda 环境**

*****如果您需要使用只在 x86 架构上工作的包，只需遵循这些步骤。首先尝试前面的步骤，只有在遇到软件包安装问题时才到这里来。否则，您会牺牲性能而没有任何好处。*****

## **1.安装 conda(如果您还没有安装的话)**

**遵循与上一节中步骤 1 相同的说明。您*不需要*安装一个单独版本的 conda 来处理 x86 环境。**

## **2.创造康达环境**

**这是指令有点不同的地方，但只是一点点不同。我们只需要告诉 conda，我们希望我们的环境使用 x86 架构，而不是原生的 ARM64 架构。我们可以通过下面几行代码来实现，这些代码创建并激活一个名为`myenv_x86`的新 conda 环境:**

```
CONDA_SUBDIR=osx-64 conda create -n myenv_x86 python=3.9
conda activate myenv_x86
conda config --env --set subdir osx-64
```

**第一行创建环境。我们简单地设置`CONDA_SUBDIR`环境变量来指示 conda 应该使用 x86 Python 可执行文件创建 en environment。第二行激活环境，第三行确保 conda 将 x86 版本的 Python 包安装到环境中。**

**我在我的`~/.zshrc`文件中创建了一个简单的 shell 函数，这样我就不需要在每次创建新的 x86 环境时记住确切的命令。**

```
### add this to ~/.zshrc (or ~/.bashrc if you're using Bash)create_x86_conda_environment () {
  # create a conda environment using x86 architecture
  # first argument is environment name, all subsequent arguments will be passed to `conda create`
  # example usage: create_x86_conda_environment myenv_x86 python=3.9

  CONDA_SUBDIR=osx-64 conda create -n $@
  conda activate $1
  conda config --env --set subdir osx-64
}
```

**就是这样！一旦您创建了 x86 环境，它的工作方式与使用默认系统架构的常规 conda 环境完全相同。只需开始安装软件包，您就可以开始了！**

# **结论**

**如果你是 Apple Silicon Mac 用户，你可能偶尔会遇到软件包安装问题。最终，在开发人员有足够的时间将 ARM64 支持添加到他们的包中后，这些问题应该会消失，但对于一些利基包来说，这可能需要几年的时间。同时，本文演示了如何使用 conda 创建与不支持 ARM64 的包兼容的环境。希望这篇文章能帮你省去一些故障排除！**

**Python 版本管理可能很棘手，但 conda 提供了一个很好的解决方案。我喜欢的另一个工具是 pyenv(与虚拟环境相结合)——详情见我的文章[这里](/how-to-use-manage-multiple-python-versions-on-an-apple-silicon-m1-mac-d69ee6ed0250)。如果你有另一个你更喜欢的工具，我很乐意听听它！**

**[成为媒体会员](https://medium.com/@djcunningham0/membership)访问成千上万作家的故事！**