# Python Conda 环境，适用于 M1 苹果芯片上的 arm64 和 x86_64

> 原文：<https://towardsdatascience.com/python-conda-environments-for-both-arm64-and-x86-64-on-m1-apple-silicon-147b943ffa55>

## 在 M1 苹果芯片上的 arm64 和 x86_64 Python 依赖项之间无缝切换

![](img/4ff03e893386d2d2f41b277e8f60ec7a.png)

由[劳拉·奥克](https://unsplash.com/@viazavier?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

# 介绍

新款 MacBook Pros 中 M1 苹果芯片的发布可以被认为是芯片技术最大的时代飞跃之一。新的 M1 芯片具有更高的功率和计算效率，基于 arm64 架构，不同于前几代的英特尔 x86_64 芯片。

虽然这一变化引发了许多兼容性问题，但 M1 芯片的单线程和多线程性能明显优于其英特尔前代产品。这使得 M1 MacBook Pro 成为数据科学相关工作负载的最佳笔记本电脑之一。

像苹果发布的许多新技术一样，行业往往需要一段时间才能赶上。当我们等待开发者发布对我们所依赖的许多依赖项的原生支持时，苹果已经发布了 Rosetta 来支持仍然基于 x86_64 架构的软件。尽管如此，对于像我一样的许多 Python 开发人员来说，仍然经常会遇到与我们的 Python 环境的兼容性问题。

为此，这篇博客详细介绍了我在 M1 苹果芯片上设置 Python 环境的配置步骤。在这篇博客的结尾，你将学会如何配置 Python 来使用基于 arm64 或 x86_64 的依赖项。

出于演示的目的，我们将为`tensorflow-macos`设置我们的 Python 环境，因为 tensorflow-macos 只能在 arm64 Python 环境中执行。然而，这里详述的过程将允许您为 arm64 和 x86_64 环境安装 Python 包。

# 安装依赖项

我们需要首先安装两个特定的依赖项。

1.  Xcode
2.  小型锻造 3

## 1.安装 Xcode:

若要安装 Xcode，请先打开终端会话，方法是前往“应用程序”>“实用工具”>“终端”。然后键入命令:

```
xcode-select --install
```

请注意 Xcode 的安装可能需要一段时间。

## 2.安装小型锻造 3

接下来，我们将安装 Miniforge3。Miniforge3 是社区(conda-forge)驱动的极简主义`conda`安装程序。此外，Miniforge3 具有对基于 arm64 的架构的原生支持，这无疑是有利的。

要安装 Miniforge3，请从以下网址下载 shell 脚本:

**确保选择 arm64(苹果芯片)架构。**

[](https://github.com/conda-forge/miniforge)  

下载完 shell 脚本后，您可能需要使用以下命令来启用 shell 脚本的执行:

```
chmod +x Miniforge3-MacOSX-arm64.sh
```

之后，使用以下命令执行脚本:

```
sh Miniforge3-MacOSX-arm64.sh
```

按照提示进行操作，这将在您的机器上安装`conda`。我们可以使用以下命令来确认是否安装了`conda`:

```
conda --version
```

您应该得到如下所示的输出。

![](img/efc5714004a3fcfcb2509031a58ba990.png)

图片来自作者

# Conda 设置

一旦安装了所有的依赖项，我们就可以继续为 arm64 或 x86_64 Python 环境配置`conda`。

我们将从在当前 shell 中添加一些快捷方式开始，以便于安装不同的`conda`环境。下面的代码片段将添加两个快捷函数，这两个函数将创建一个`osx-64`或`osx-arm64` conda 环境，我们可以在其中安装 Python 包。

为此，将以下代码添加到`~/.zshrc`或`~/.bashrc`。

```
# Create x86 conda environment
create_x86_conda_environment () {# example usage: create_x86_conda_environment myenv_x86 python=3.9
 CONDA_SUBDIR=osx-64 conda create -n $@
 conda activate $1}# Create ARM conda environment
create_ARM_conda_environment () {# example usage: create_ARM_conda_environment myenv_x86 python=3.9
 CONDA_SUBDIR=osx-arm64 conda create -n $@
 conda activate $1}
```

# 创造康达环境

Miniforge3 的特性之一是能够为特定的 Python 环境定义特定于处理器的子目录。例如，通过设置`CONDA_SUBDIR==osx-64`，将指示`conda`从 x86_64 (osx-64)特定子目录安装软件包。

这将使用户能够根据`CONDA_SUBDIR`定义的值创建安装 arm64 或 x86_64 (osx-64) Python 包的环境。

让我们使用之前创建的快捷方式安装两个不同的环境，一个基于 x86_64 (osx-64)，另一个基于 arm64 (osx-arm64)。

## x86_64 (osx-64)

创建一个名为 env_x86 的 Python 3.9.13 `osx-64` (x86_64)环境:

```
create_x86_conda_environment env_x86 python=3.9.13
```

或者，如果您选择不使用快捷方式，您可以使用命令获得与上面相同的结果:

```
CONDA_SUBDIR=osx-64 conda create -n env_x86 python=3.9.13
conda activate env_x86
```

## arm64 (osx-arm64)

使用快捷方式创建一个名为 tensorflow_ARM 的 Python 3.9.13 `osx-arm64` (arm64)环境:

```
create_ARM_conda_environment tensorflow_ARM python=3.9.13
```

或者，如果您选择不使用快捷方式，您可以使用命令获得与上面相同的结果:

```
CONDA_SUBDIR=osx-arm64 conda create -n tensorflow_ARM python=3.9.13
conda activate tensorflow_ARM
```

执行上述步骤将安装两个使用 x86_64 (osx-64)或 arm64 (osx-arm64)的 Python 环境。通过使用以下命令激活特定环境，可以在这两种环境之间无缝切换:

```
conda activate <NAME_OF_ENVIRONMENT>
```

# 张量流装置

让我们验证是否安装了 arm64 环境。为此，我们将安装`tensorflow-macos`。如前所述，`tensorflow-macos`只能在 arm64 Python 环境中执行。

**注意:** `**tensorflow**` **在 x86_64 Python 环境下无法工作**

1.  使用以下命令激活`tensoflow_ARM`环境:

```
conda activate tensorflow_ARM
```

2.安装 arm64 特定 tensorflow 依赖项。

**注:** `**-c**` **=频道**

```
conda install -c apple tensorflow-deps
```

3.安装 tensorflow 库:

```
pip install tensorflow-macos tensorflow-metal
```

## 张量流验证

让我们验证 tensorflow 安装是否有效。

1.  首先激活适当的环境并启动 python 环境。

```
conda activate tensorflow_ARM && python
```

2.导入张量流并创建常数:

```
import tensorflow as tf
tf.constant([1,2,3])
```

![](img/91004592d38c89e8dd68e7be41dad2ca.png)

作者图片

3.让我们也检查一下 tensorflow 是否在您的机器上使用了 GPU 加速:

```
tf.config.list_physical_devices('GPU')
```

## 解决纷争

安装 tensorflow 时，您可能会遇到的一个最常见的问题是错误:

```
[1]    21781 illegal hardware instruction  python
```

这个错误是由于将`tensorflow-macos`安装在错误的`conda`环境中造成的。确保您已经在 arm64 `conda`环境中安装了`tensoflow-macos`。

# 结论

管理 Python 依赖关系一直很困难。M1 芯片的发布只是增加了一层额外的复杂性。在行业赶上来之前，像这篇博客中详细介绍的方法这样的变通方法就足够了。

希望这篇博客中的详细说明能让你在新的 M1 MacBook 上更容易地管理 Python 的依赖性。

如果您有任何问题或疑问，请留下您的评论，我将非常乐意帮助您。