# 如何用 Python 中的 Bulk API 索引 Elasticsearch 文档

> 原文：<https://towardsdatascience.com/how-to-index-elasticsearch-documents-with-the-bulk-api-in-python-b5bb01ed3824>

## 了解有效地对大量文档进行索引的不同方法

![](img/ab2c877ee5a4cbd61713bd12836a779c.png)

图片由 [PublicDomainPictures](https://pixabay.com/photos/freighter-cargo-ship-industry-port-315201/) 在 Pixabay 拍摄

当我们需要创建一个 Elasticsearch 索引时，数据源通常是不规范的，不能直接导入。原始数据可以存储在数据库、原始 CSV/XML 文件中，甚至可以从第三方 API 获得。在这种情况下，我们需要对数据进行预处理，使其能够与批量 API 一起工作。在本教程中，我们将演示如何用简单的 Python 代码从 CSV 文件中索引 Elasticsearch 文档。本地的 Elasticsearch `bulk` API 和来自`helpers`模块的 API 都将被使用。您将学习如何在不同场合使用合适的工具来索引 Elasticsearch 文档。

## 准备

像往常一样，我想提供建立一个演示系统和环境的所有技术细节，您可以在其中直接运行代码片段。自己运行代码是理解代码和逻辑的最佳方式。

请使用此`[docker-compose.yaml](https://gist.github.com/lynnkwong/a1e3bfdd6b25c98baab525afb25ac2f6#file-401fd3fff829-docker-compose-1-yaml)`通过 Docker 设置一个本地的 Elasticsearch 服务器。要了解如何在 Docker 上运行 Elasticsearch 和 Kibana，请查看[这篇文章](https://levelup.gitconnected.com/how-to-run-elasticsearch-8-on-docker-for-local-development-401fd3fff829)。

然后我们需要创建一个[虚拟环境](https://lynn-kwong.medium.com/how-to-create-virtual-environments-with-venv-and-conda-in-python-31814c0a8ec2)并安装 Elasticsearch Python 客户端库，其版本与 Elasticsearch Docker 映像的版本兼容。我们将安装最新的版本 8 客户端。

入门 Elasticsearch 最好用最新版本。另一方面，如果你想将你的 Elasticsearch 库从版本 7 升级到版本 8，请看看[这篇文章](https://lynn-kwong.medium.com/important-syntax-updates-of-elasticsearch-8-in-python-4423c5938b17)，这很可能会为你节省很多代码更新的精力。

## 用 Python 创建索引

我们将创建相同的`latops-demo`索引，如本文中的[所示。然而，语法会有所不同，因为我们在这个例子中使用的是 Elasticsearch 8。首先，我们将使用 Elasticsearch 客户端直接创建一个索引。此外，`settings`和`mappings`将作为顶级参数传递，而不是通过`body`参数传递，如本文](https://lynn-kwong.medium.com/all-you-need-to-know-about-using-elasticsearch-in-python-b9ed00e0fdf0)中[所解释的。](https://lynn-kwong.medium.com/important-syntax-updates-of-elasticsearch-8-in-python-4423c5938b17)

配置可以在[这里](https://gist.github.com/lynnkwong/3c5ed5b3225a1e4e56e9bc6b739881e2#file-elasticsearch-index-configurations-py)找到，创建索引的命令是:

现在索引已经创建好了，我们可以开始向它添加文档了。

## 使用原生 Elasticsearch 批量 API

当你有一个小数据集要加载时，使用原生 Elasticsearch `[bulk](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-bulk.html#docs-bulk-api-request)` [API](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-bulk.html#docs-bulk-api-request) 会很方便，因为语法与原生 Elasticsearch 查询相同，后者可以直接在开发控制台中运行。你不需要学习任何新的东西。

将要加载的数据文件(作者创建的虚拟数据)可以从[此链接](https://gist.githubusercontent.com/lynnkwong/c5ee4a0f4963d8c2c3281fecf32b5dae/raw/e3e1a243c69bc9653cb020360b86af4f9b5ba04b/b9ed00e0fdf0-laptops-data.csv)下载。将其另存为`laptops_demo.csv`，将在下面的 Python 代码中使用:

注意，我们使用`csv`库方便地从 CSV 文件中读取数据。可以看出，原生批量 API 的语法非常简单，可以跨不同语言使用(包括开发工具控制台)，如[官方文档](https://gist.github.com/lynnkwong/5650dc12b8243a2c243dd07df28d5402)所示。

## 使用批量助手

如上所述，原生`bulk` API 的一个问题是，所有的数据在被索引之前都需要被加载到内存中。当我们有一个大的数据集时，这可能是有问题的和非常低效的。为了解决这个问题，我们可以使用 bulk helper，它可以从迭代器或生成器中索引 Elasticsearch 文档。因此，它不需要先将所有数据加载到内存中，这在内存方面非常有效。然而，语法有点不同，我们很快就会看到。

在我们使用 bulk helper 索引文档之前，我们应该删除索引中的文档，以确认 bulk helper 确实能够成功工作:

然后，我们可以运行以下代码，用 bulk helper 将数据加载到 Elasticsearch:

事实上，代码比使用原生的`bulk` API 更简单。我们只需要指定要索引的文档，而不是要执行的操作。从技术上来说，你可以用`_op_type`参数指定其他动作，比如`delete`、`update`等，虽然这些并不常用。

特别是对于 bulk helper，会创建一个生成器来生成要索引的文档。请注意，我们应该调用该函数来实际创建一个生成器对象。使用生成器，我们不需要首先将所有文档加载到内存中。相反，它们是动态生成的，因此不会消耗太多内存。如果您需要更细粒度的控制，您可以使用`[parallel_bulk](https://elasticsearch-py.readthedocs.io/en/master/helpers.html#elasticsearch.helpers.parallel_bulk)`，它使用线程来加速索引过程。

在这篇文章中，介绍了用 Python 批量索引文档的两种不同方法。分别使用本机批量 API 和批量助手。前者适用于不消耗大量内存的小数据集，而后者适用于加载量很大的大数据集。有了这两个工具，您可以方便地为各种数据集在 Python 中索引文档。

相关文章:

*   [Python 中 Elasticsearch 8 的重要语法更新](https://lynn-kwong.medium.com/important-syntax-updates-of-elasticsearch-8-in-python-4423c5938b17)
*   [关于在 Python 中使用 Elasticsearch 你需要知道的一切](https://lynn-kwong.medium.com/all-you-need-to-know-about-using-elasticsearch-in-python-b9ed00e0fdf0)