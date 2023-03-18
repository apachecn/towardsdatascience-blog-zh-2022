# 为数据科学设置苹果定制芯片设备

> 原文：<https://towardsdatascience.com/setting-up-apples-custom-silicon-devices-for-data-science-c70bf671b901>

## 两分钟内准备好任何 M1 和 M2 机器

随着定制片上系统第二代产品的发布，苹果再次提高了计算能力。新的 M2 芯片支持高达 24 GB 的主内存，并可以提供多达 10 个 GPU 核心。虽然与 M1 Pro、Ultra 和 Max 型号相比，这款产品目前还很小，但预计第二代产品将在以后推出更高性能的版本。

![](img/76d75e71739cfe484cea469967583d97.png)

由 [Ales Nesetril](https://unsplash.com/@alesnesetril?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

在此之前，我们可以最大限度地利用当前的计算能力。然而，从基于英特尔的系统转换到定制芯片后，设置 Mac 电脑变得有点复杂。在[这个指南的早期版本](/setting-up-apples-new-m1-macbooks-for-machine-learning-f9c6d67d2c0f)中，我手动指导你一步一步地完成这个过程。自从这篇文章发表后，我收到了反馈，并对设置有了更多的了解。更多的知识带来更多的可能性，在这篇文章中，我将向您展示一个自动化安装过程的脚本。该脚本可以在任何 M*设备上运行，无论是最初的 M1 芯片，它的兄弟，还是全新的 M2 版本。

## 运行完整的脚本

该脚本让您在不到两分钟的时间内解决我的设置中的机器学习问题。它有 20 行代码，其中 9 行(约 50%)是注释，让我们了解安装过程的最新情况。

**如果你只有两分钟的时间，下面是方法:**要运行这个脚本，下载源代码。为此，打开一个新的终端窗口(Spotlight →终端)并输入

> *wget*[*https://github . com/phrasenmaeher/apple _ _ ml _ setup/blob/main/apple _ setup . sh*](https://github.com/phrasenmaeher/apple__ml_setup/blob/main/apple_setup.sh)

这会将安装脚本下载到您的机器上。接下来，键入 *chmod +x apple_setup.sh* ，这使得脚本可执行。最后，键入*。/apple_setup.sh* 开始安装。

整个脚本如下所示。每当我在这篇文章的剩余部分提到行号时，我指的是这个完整脚本的行:

## 安装 MiniForge

在前四行中，我们下载并安装了 MiniForge 的最新版本，这是一个适用于 Apple 系统的最小 Anaconda 版本。最小化意味着它不臃肿，只包含基本的特性。这对于我们的用例来说已经足够了；我们将手动安装所有需要的东西。通常，我们需要包管理器软件来处理我们机器学习任务所需的所有不同工具。简而言之，该软件允许我们安装所有需要开始使用的软件包，例如 TensorFlow 和 PyTorch。

## 创建新的虚拟环境

在第五行(上面代码片段中的一行)，我们激活了新的 MiniForge 包管理器。

这随后让我们创建一个专门为机器学习任务设计的新环境。我们将这个环境命名为*机器学习*。每个虚拟环境可以看成一个盒子，每个项目通常使用自己的盒子。通过为每个项目使用单独的盒子，我们可以极大地减少项目之间不必要的干扰。这是 Anaconda 和 MiniForge 的主要工作。

## 安装 TensorFlow

从第九行开始(在上面的代码片段中有一行)，我们激活我们的新环境，并继续安装 TensorFlow。为此，我们需要一个特定的包， *tensorflow-deps* ，它是我们从苹果的软件库(也就是第 14 行的*-c Apple*；上面第一行)。之后，我们安装实际的 TensorFlow 包(第 15 行；上面第二行)。

## 安装 PyTorch 和 Scikit-学习

安装 TensorFlow 之后，我们也是碰运气，安装了 scikit-learn(通常叫做 *sklearn* )和 PyTorch。

就是这样；您现在已经为机器学习准备好了一个环境！要在终端应用程序中激活它，请键入

> *source ~/miniforge 3/bin/activate*

然后输入

> *康达激活机器学习*

来激活新的环境。在你选择的 IDE 中，你可能需要将解释器(让你运行代码的东西)指向*/Users/<replace _ with _ your _ username>/*miniforge 3*/envs/machine _ learning*

对于这个过程的可视化，参见[我之前的帖子](/setting-up-apples-new-m1-macbooks-for-machine-learning-f9c6d67d2c0f)并适当地插入上面的路径。

## 摘要

在这篇博客中，我指导您完成了一个简短的安装脚本，让您在两分钟内为机器学习做好准备。它使用 MiniForge 来管理所需的包，并自动安装 TensorFlow 和 PyTorch。重复我在开头写的内容:要运行脚本，通过打开一个新的终端窗口(Spotlight->Terminal)并输入

> *wget https://github . com/phrasenmaeher/apple _ _ ml _ setup/blob/main/apple _ setup . sh*

这会将安装脚本下载到您的机器上。接下来，键入 *chmod +x apple_setup.sh* ，这使得脚本可执行。最后，输入*。/apple_setup.sh* 开始安装。