# 如何安装 dbt(数据构建工具)

> 原文：<https://towardsdatascience.com/install-dbt-1bd6a4259a14>

## 为您的特定数据仓库安装数据构建工具

![](img/838b8ae808b104e42c3963f4cf29a988.png)

由[马库斯·斯皮斯克](https://unsplash.com/@markusspiske?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/lego?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄的照片

**数据构建工具** (dbt)无疑是现代数据堆栈中最强大的工具之一，因为它允许团队和组织以可伸缩、高效和有效的方式管理和转换数据模型。dbt 将处理所有数据模型的相互依赖性，并为您提供您需要的一切，以便对您的数据执行测试，并提高您的数据资产的数据质量。

根据您使用的数据平台，您必须安装一些额外的适配器来使 dbt 工作并与该平台正确通信。在接下来的几节中，我们将演示如何在虚拟环境中安装`dbt`和所需的适配器，以便开始使用数据构建工具。

## 创建虚拟环境

首先，我们需要创建一个虚拟环境，该环境独立于主机上安装的任何设备:

> 虚拟环境是在现有 Python 安装的基础上创建的，称为虚拟环境的“基础”Python，并且可以选择与基础环境中的包隔离，因此只有明确安装在虚拟环境中的包才可用。— [Python 文档](https://docs.python.org/3/library/venv.html)

```
python3 -m vevn dbt-venv
```

然后激活新创建的 venv:

```
source dbt-venv/bin/activate
```

如果一切顺利执行，您应该能够在终端的每一行看到一个`(dbt-venv)`前缀。

## 安装 dbt-core

dbt 提供了两种与工具本身交互和运行项目的可能方式——一种是在云上，另一种是通过命令行界面(cli)。在本教程中，我们将演示如何安装所需的软件包，让您可以从本地机器上使用 dbt。

因此，您需要安装的第一个依赖项是`dbt-core`。以下命令将安装 PyPI 上可用的最新版本:

```
pip install dbt-core
```

如果您希望安装一个特定的版本，那么您必须在安装命令中指定它:

```
pip install dbt-core==1.3.0
```

安装完成后，您可以通过运行以下命令来确保安装成功，该命令将简单地在终端上打印出安装在本地计算机上的 dbt 版本:

```
dbt --version
```

## 为您的数据平台安装 dbt 插件

现在，为了让 dbt 成功运行，它需要与您(或您的团队)使用的数据平台建立连接。使用一个**适配器插件**，数据构建工具可以扩展到任何平台。你可以把这些插件想象成我们在上一步中安装的`dbt-core`正在使用的 Python 模块。

dbt 实验室维护他们自己的一些适配器，而其他一些适配器最初是由社区创建的(并且正在被积极地维护)。你可以在这里找到可用插件的完整列表。下面我将分享其中一些的安装说明:

**BigQuery(谷歌云平台)**

```
pip install dbt-bigquery
```

**雅典娜**

```
pip install dbt-athena-adapter
```

**Postgres 和 AlloyDB**

```
pip install dbt-postgres
```

**天蓝色突触**

```
pip install dbt-synapse
```

**数据块**

```
pip install dbt-databricks
```

**红移**

```
pip install dbt-redshift
```

**雪花**

```
pip install dbt-snowflake
```

**火花**

```
pip install dbt-spark
```

## 后续步骤

现在，您已经成功地安装了`dbt-core`和基于您正在使用的数据平台的所需适配器，您已经准备好创建您的第一个 dbt 项目和与目标数据平台交互所需的概要文件。在接下来的几天里，我将分享更多关于如何做到这一点的教程，所以请确保订阅并在这些文章发布时收到通知！

## 最后的想法

如果您还没有尝试过数据构建工具，我强烈建议您尝试一下——您可能会惊讶地发现它是如何帮助您的团队最小化构建、管理和维护数据模型的工作的。

在今天的简短教程中，我们介绍了在本地机器上安装 dbt 所需的步骤。本指南将帮助您安装 dbt CLI 以及创建、管理、运行和测试数据模型所需的适配器(基于您的首选数据平台)。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**相关文章你可能也喜欢**

[](/visual-sql-joins-4e3899d9d46c)  [](/2-rules-groupby-sql-6ff20b22fd2c)  [](/diagrams-as-code-python-d9cbaa959ed5) 