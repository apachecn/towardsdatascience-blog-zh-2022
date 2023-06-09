# 什么是计算机内核？

> 原文：<https://towardsdatascience.com/what-is-a-computer-kernel-2320b9330eef>

## 了解 Jupyter 笔记本电脑内核的概念

![](img/f3209dd956958c50a5720818c80ddb91.png)

安基特·辛格在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

内核，或称系统核心，确保操作系统在计算机中平稳运行，是软件和硬件之间的接口。它用于所有带有操作系统的设备，例如电脑、笔记本电脑、智能手机、智能手表等。

# 内核执行哪些任务？

内核主要是操作系统(即软件)和设备中的硬件组件之间的接口。这导致必须完成各种任务。然而，对于最终用户来说，这项工作是不可见的，只能通过所有程序是否正确运行来表示。

当我们在计算机上使用程序时，比如 Excel，我们在所谓的图形用户界面(GUI)上处理它。该程序将每一次按钮点击或其他动作转换成机器代码，并发送给操作系统内核。如果我们想在 Excel 表格中添加一个新列，这个调用会到达系统核心。这又将调用传递给计算机处理单元(CPU ), CPU 执行该动作。

如果几个程序同时运行，系统核心还确保程序的请求被一个接一个地处理，并且给用户进程同时运行的感觉。因此，在我们的示例中，如果浏览器窗口与 Excel 同时打开，系统内核会确保程序对 CPU 和其他硬件的需求以协调的方式得到处理。

此外，还有其他更详细的任务，例如:

*   管理外部连接的设备，如键盘、鼠标或显示器及其正常运行。
*   解决内存使用错误，例如，当程序的内存使用增加过多时。
*   优化 CPU 的使用

# 内核由哪几层组成？

内核在几个相互构建的层中工作。其中包括:

*   **硬件**:最底层形成了操作系统可以访问的系统硬件的接口。这包括例如 PCI Express 控制器或存储器。
*   **内存管理**:在下一层，专用的可用内存量被分配给正在运行的进程。此外，虚拟主存储器也可以分布在这里。
*   **流程管理**:在这一层，程序的时间被管理，传入的请求被定时，这样它们对我们这些用户来说是并行的。
*   **设备管理**:在这一层，进行设备管理。与硬件层相反，这涉及外部连接的设备，如显示器或打印机，它们的通信通过特殊的驱动程序来保证。
*   **文件系统**:在最顶层，进程被分配到硬盘上的指定区域，即 HDD 或 SSD。

这些层的运作方式是，每一个更高的层都依赖并建立在它下面的层之上。例如，进程管理层也访问硬件层中的进程。然而，反之则不然。

# 内核类型有哪些？

通常，根据可以同时管理的进程和用户的数量来区分三种不同的内核类型。

## 微内核

微内核只执行最少的任务，比如内存管理和 CPU 进程管理。其他功能，如文件系统或设备管理，被外包给操作系统本身。优点是它不会像 Windows 那样导致整个系统的故障。

例如，苹果电脑的操作系统 macOS 就是基于微内核架构的。

## 整体内核

单片内核集中处理已经描述过的所有任务，负责所有内存和进程管理，还处理外部连接设备的硬件管理。Windows 操作系统基于单一内核。

由于采用中央设计，单片内核通常比微内核快得多，但如果单个进程运行不正常，这也会导致系统崩溃。

## 混合内核

顾名思义，混合内核是这两个概念的混合。它基本上也是一个大的内核，可以接管所有的任务，但是更加紧凑，可以分成不同的模块。

# Jupyter 笔记本的内核是如何工作的？

在处理[数据](https://databasecamp.de/en/data)和[机器学习](https://databasecamp.de/en/machine-learning)时，人们经常会求助于 Jupyter 笔记本。它是一个基于网络的平台，用于创建和共享编程代码。它经常用于数据科学应用，因为可以执行单独的代码块，并且它们的结果(例如，图形)是直接可见的。当接下来的编程步骤依赖于先前的结果时，这对于模型创建或数据集分析尤其有利。

当使用 Jupyter Notebook 时，还会启动一个内核，这有时会导致问题，例如在下面的示例中建立连接时。然而，这与本文中所描述的操作系统完全不同。

![](img/c141b6131764828f0736b2c19d5dd2e3.png)

Jupyter Notebook 内核是一个执行笔记本代码的引擎，专用于特定的编程语言，比如 Python。然而，它不执行迄今为止描述的全面的接口功能。

在处理 Jupyter 笔记本内核时，以下命令特别有用:

*   **中断**:该命令停止单元中当前正在运行的进程。例如，这可用于停止模型的训练，即使尚未到达所有训练时期。
*   **重启&运行全部**:该命令可以再次执行所有单元，删除之前的变量。如果您想将一个较新的数据集读入现有程序，这可能会很有用。
*   **重启**:单一的“重启”命令导致相同的结果，但并不是所有的单元都被再次执行。
    重新连接:在训练大型模型时，内核会因为内存已满而“死亡”。那么重新联系是有意义的。
*   **关机**:只要内核还在运行，它也会占用内存。如果您要并行运行其他程序来释放内存，那么“Shutdown”命令是有意义的。

# 这是你应该带走的东西

*   内核确保操作系统在计算机中平稳运行，是软件和硬件之间的接口。
*   它被分成不同的层，这些层相互构建。
*   这些任务包括管理并行运行的进程或外部连接设备的正常运行。
*   Jupyter Notebook 内核不是所描述的系统内核，因为它只用于执行编程代码。

*如果你喜欢我的作品，请在这里订阅*<https://medium.com/subscribe/@niklas_lang>**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，medium 允许你每月免费阅读* ***3 篇*** *。如果你想让***无限制地访问我的文章和数以千计的精彩文章，请不要犹豫，通过点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格***

**</introducing-nosql-databases-with-mongodb-d46c976da5bf>  <https://medium.com/codex/understanding-the-backpropagation-algorithm-7a2e3cb4a69c>  </beginners-guide-to-gradient-descent-47f8d0f4ce3b> **