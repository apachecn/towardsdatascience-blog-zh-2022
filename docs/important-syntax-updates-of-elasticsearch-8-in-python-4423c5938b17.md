# Python 中 Elasticsearch 8 的重要语法更新

> 原文：<https://towardsdatascience.com/important-syntax-updates-of-elasticsearch-8-in-python-4423c5938b17>

## 一些帮助你应对弹性搜索突变的建议

![](img/2c92537fe9fae493f40a6058c354593c.png)

图片由皮沙贝的[杰拉德](https://pixabay.com/illustrations/update-upgrade-to-update-board-1672385/)拍摄

Elasticsearch Python 客户端库第 8 版有不少突破性的改动，在你从 7 版更新到 8 版的时候会给你带来很多麻烦。尽管这可能是一项痛苦的任务，但仍然建议将库更新到最新版本，因为已经添加了许多新功能，并且应该更加用户友好。在这篇文章中，我们将概述 Elasticsearch 8 的重要语法更新，这将有助于您重构代码，使其与最新版本的库一起工作。

## 准备

如果您想测试本文中演示的代码片段，最好在本地运行一个 Elasticsearch 服务器。我们将[使用 Docker 启动 Elasticsearch 和 Kibana containers](https://levelup.gitconnected.com/how-to-run-elasticsearch-8-on-docker-for-local-development-401fd3fff829) ，使用 Docker 图像的最新版本，在撰写本文时是 8.2.2 版，但应该也适用于 8 版的其他图像。

此外，我们需要安装 Elasticsearch Python 客户端库的版本 8。最好将它安装在一个虚拟环境中，这样它就不会影响你系统中现有的库。

## 对连接使用严格的客户端配置

在 Elasticsearch 版本 8 中，删除了*方案*、*主机*和*端口*的默认值，现在我们需要明确指定它们。否则会有一个`ValueError`:

有关客户端配置的更多示例，请查看此 [GitHub 问题](https://github.com/elastic/elasticsearch-py/issues/1690)。

## 使用 Elasticsearch 而不是 IndicesClient 来管理指数

我们曾经使用`[IndicesClient](https://elasticsearch-py.readthedocs.io/en/v7.12.0/api.html#indices)`类来管理索引。然而，它已经过时了，我们现在应该使用`Elasticsearch`类来直接管理索引。实际上，在本文中，我们将使用上面创建的`es_client`对象来执行所有与索引相关的操作。

## API 只允许关键字参数

在 Elasticsearch 版本 8 中，我们只能对所有 API 使用关键字参数。现在使用关键字参数将引发一个`TypeError`:

## 使用 client.options()指定传输参数

像`ignore`这样的每请求选项现在应该用`client.options()`来指定，而不是在 API 方法中。另外，`ignore`现在改名为`ignore_status`。

更多用`client.options()`指定的选项可在[这里](https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/migration.html#migration-options)找到。

## 使用顶级参数而不是 body 字段

这可能是影响最大的变化。我们曾经使用`body`字段来指定所有与 Elasticsearch 相关的设置或搜索查询，如本文中的[所示。现在我们需要将它们全部改为顶级参数。它不仅仅影响`search` API，而是影响所有的 API。例如，`create` API 需要更新如下:](https://lynn-kwong.medium.com/all-you-need-to-know-about-using-elasticsearch-in-python-b9ed00e0fdf0)

如您所见，我们只需要将`body`参数扩展到顶级参数。如果你不想一个接一个地明确指定顶级参数，你可以使用字典扩展语法`**[configuaration](https://gist.github.com/lynnkwong/3c5ed5b3225a1e4e56e9bc6b739881e2#file-elasticsearch-index-configurations-py)`。这一变化同样适用于所有的 Elasticsearch APIs。

## 批量 API 的更新

当我们有大量文档需要索引时，我们可以使用`bulk` API 来批量加载它们。现在语法也不同了。首先，我们需要明确地指定索引。此外，我们应该使用`operations`而不是`body`参数来指定动作:

还要注意，在旧语法中，`actions`是一个长字符串，每个动作由一个换行符分隔。然而，在新的语法中，`actions`可以是作为字典的操作列表，这意味着我们现在不需要将每个操作作为 JSON 字符串转储。我们还可以使用带有新语法的`filter_path`参数来指定在响应中显示哪些字段。

## API 响应现在是对象而不是字典

在以前的版本中，Elasticsearch APIs 的响应是字典，你通常通过`resp['hits']['hits']`得到结果。然而，在 Elasticsearch 8 中，响应不再是字典，而是类`ObjectApiResponse`的实例。它有两个属性，`meta`和`body`，分别用于访问请求元数据和结果。

神奇的是，我们仍然像以前一样直接访问`hits`键。但是，我们只能像访问字典键一样访问它，而不能访问对象属性。

这是为了与旧的行为保持一致。我认为这也可能会被否决，因此，更安全的方法是从`body`访问`hits`，如上所示。

## 使用更细粒度的错误类

在以前的版本中，`TransportError`既包括传输错误(如连接超时)也包括 API 错误(如未找到索引)。但在 Elasticsearch 版中，`TransportError`只覆盖了传输错误，新的`ApiError`需要用于 API 错误，对调试更有帮助。

在本帖中，我们介绍了在 Python 中使用 Elasticsearch 8 的一些重要语法更新。它们中的许多都是突破性的改变，这意味着你需要更新你以前的 Python 代码，以使它们适用于最新的版本。有了本文中展示的列表，代码的重构应该会简单得多。您也可以参考[官方文档](https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/migration.html#migration-options)和 [GitHub 问题](https://github.com/elastic/elasticsearch-py/issues/1696)来更好地了解变化的原因以及更多技术细节。

相关文章:

*   [关于在 Python 中使用 Elasticsearch 你需要知道的一切](https://lynn-kwong.medium.com/all-you-need-to-know-about-using-elasticsearch-in-python-b9ed00e0fdf0)