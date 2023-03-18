# 使用 Hasura 向 shell 脚本管道添加事件通知和结果持久性

> 原文：<https://towardsdatascience.com/adding-event-notifications-and-result-persistence-to-shell-scripted-pipelines-with-hasura-91b31e12e64a>

![](img/0dc4e8c3375a72bd4fb9f60514c89fc9.png)

原始图像从 [iStockphoto](https://www.istockphoto.com) 获得许可，Hasura 徽标经许可使用

在生物信息学中，作为一堆外壳脚本开发的计算管道无处不在。尽管像 [Apache Airflow](https://airflow.apache.org/) 这样的现代应用和像 [Prefect](https://www.prefect.io/) 这样的云服务提供了更优雅、功能更丰富的解决方案，但现实是，在一个全新的平台上重新实现整个 shell 脚本管道对许多人来说往往是一个太大的步骤。当管道是从很久以前的博士后传下来的，或者当开发人员更多地来自科学背景(例如计算生物学)而不是软件工程背景时，情况就更是如此。

像 [Nextflow](https://www.nextflow.io/) 这样的工具通过提供一个 DSL 来运行 shell 脚本，从而允许一个更加渐进的方法，这样命令就可以逐渐从 shell 脚本中移出并迁移到 DSL 上。问题是这些工具正在老化，并且缺少许多对今天的工作流来说必不可少的功能。以事件通知为例，Nextflow 只为电子邮件和电子邮件提供配置。计算管道本质上是事件驱动的，因此在处理完成时，我可能想要一封电子邮件，但我也想通知一个松弛的通道，我想启动一两个 webhook，我想广播到多个上游应用程序，如 LIMS 和文件服务器。

shell 脚本管道的另一个痛处是保存结果。写出数据文件很容易，但是捕获和持久化管道运行的关键结果和度量又如何呢？这些数据当然也可以被写入某种机器可读的中间文件，但是这个文件需要在下游的某个地方被解析，这是低效的，并且会产生打字和后续逻辑的问题(例如舍入精度差异)。

或者，可以调用最终的 shell 脚本来运行一组直接的 SQL `UPDATE/INSERT`语句，这些语句用结果数据填充数据库。假设您可以获得一个配置了适当身份验证和受限角色的数据库用户帐户(防止[小鲍比表](https://bobby-tables.com))，那么这是一个可行的解决方案，前提是您只需要记录一些值。一旦你有了更多不同格式的不同类型的值(例如`csv` vs `json`，那么试图在一个 shell 脚本中将数据转换成 SQL 语句不仅是糟糕的做法，而且很快变成了转义符的噩梦。例如，双引号(`“`)在 JSON 和 Bash 中用于分隔字符串，但在 Postgresql 中用于分隔标识符(例如列名)，因此值必须始终用单引号(`‘`)替换，同时在 shell 中进行双转义。

最后一种选择是开发某种定制 API，通常通过 HTTP，并从 cURL 或 wget 发送一个请求，将数据打包到一个不可读的 URL 编码的查询字符串或一个又大又长又丑的 JSON 参数中。这是可行的，就像在 90 年代一样，但是有更快、更容易和更健壮的选择，不会招致同样的技术债务和维护开销。

# 带有 Hasura 的即时自动 GraphQL

如果您是 GraphQL 的新手，在被仅仅为了改进管道而不得不学习一种全新的查询语言的前景吓退之前，您必须知道关于 Hasura 的两件最重要的事情:

1.  你不需要了解任何 GraphQL 来开始使用 Hasura——它为你编写了 GraphQL，所以你可以一步一步地学习。
2.  **你不需要用 Hasura 替换任何东西** — 它和你正在运行的任何东西一起安静地工作，你可以慢慢地开始在有意义的地方和时间使用功能。

![](img/c78ec1d5203f9ae42619d2e3de84b74c.png)

图片由作者提供，Hasura & GraphQL 徽标经许可使用

Hasura 封装了一个现有的数据库(PostgreSQL，MySQL，MSSQL ),但是除非你要求，否则不会接触你的数据。对于这个例子，让我们使用一个非常简单的演示数据库(让 [SQL](https://gist.github.com/s1monj/05e374c6865a4095e314a6fc07257721) 创建它)。

![](img/971ac483fe564707ead008277b3b566b.png)

作者图片

现在有了您的 DB 凭证，我们可以用下面的命令在 Docker 上下载并运行 Hasura。

```
docker run -d -p 8080:8080 \
  -e HASURA_GRAPHQL_DATABASE_URL=
postgres://hasurauser:hasurapwd@host.docker.internal:5432/pipelinedemo \
  -e HASURA_GRAPHQL_ENABLE_CONSOLE=true \
  -e HASURA_GRAPHQL_ADMIN_SECRET=MyHasuraSecret \
  hasura/graphql-engine:v2.1.1# replace HASURA_GRAPHQL_DATABASE_URL and MyHasuraSecret as required
```

接下来转到 [http://localhost:8080/](http://localhost:8080/) ，在密码提示中输入上面`HASURA_GRAPHQL_ADMIN_SECRET`的值，然后点击顶部的“数据”导航菜单选项卡。首先要注意的是，Hasura 免费为您提供了一个整洁的 GUI 数据库浏览器。您可以插入行、配置新的表和列，甚至直接从控制台运行任意 SQL。连接好数据库后，我们现在希望*跟踪*我们的表和关系，这样 Hasura 就可以分析模式并创建 API。只需点击“public”模式，然后点击*未跟踪的表或视图*旁边的“Track All”即可看到下面的屏幕。

![](img/e01f4992fb6c07f843f36c3fec6133d2.png)

作者图片

我们还想告诉 Hasura 有关关系的信息，所以我们单击*未跟踪的外键关系旁边的“Track All”。*最后，通过从左侧选择*运行*和*样品*表，点击*插入行*选项卡并输入以下值，将一些测试值添加到这些表中。

![](img/97e2ac1a8eddbe38a88232f92830f362.png)

作者图片

现在 Hasura 知道了我们的 DB，我们可以通过点击顶部的“API”导航菜单标签来尝试一些 GraphQL。当我们的管道完成时，我们希望在数据库中添加或更新一条新记录 GraphQL 中的任何写操作都被称为*突变*。让我们创建一个变异，将一个示例记录添加到我们的`sample_runs`表中。

1.  向右滚动到左边的*浏览器*框的底部，在那里显示*添加新的*选择“突变”并点击“+”按钮。
2.  从左侧的*浏览器*框中展开 *insert_sample_runs* ，点击所有复选框(`sample_id`、`run_id`、`qc_score`、`duplication_rate`，为每个字段添加一个示例值。
3.  从 *insert_sample_runs* 下方展开*返回*，点击`sample_id` 和`run_id`复选框
4.  点击播放按钮，`run_id`和`sample_id`值应返回如下。您可以通过返回到*数据*导航菜单选项卡并查看*样本 _ 运行*表来检查嵌件是否工作。

![](img/d822be5f6ec3012a2b7e9bdf17e24285.png)

作者图片

不把这篇文章作为关于 [GraphQL](https://medium.com/towards-data-science/search?q=graphql) 的文章，另一个值得注意的关键特性是，与*相关的*表也可以在同一个请求中被查询和变异——上面的那些`> run:`和`> sample:`节点可以被扩展以插入/返回相应表中的列。这大大优于众所周知的因[提取不足/提取过多而导致](https://www.howtographql.com/basics/1-graphql-is-the-better-rest/)效率低下的 REST。

# graphqurl 是 GraphQL 的命令行 cURL

从 web GUI 中改变数据对于学习和测试来说非常好，但是对于我们的管道来说，我们需要从我们的 shell 脚本中调用更新，这就是 graphqurl 的用武之地——这是一个强大的 CLI，也是由 Hasura 开发的。

```
npm install -g graphqurl
```

现在我们可以调用相同的 GraphQL 变体，方法是将上面的代码复制并粘贴到一个纯文本文件中，我们称之为`mutation.graphql`，并至少更改`sample_id`或`run_id`值，因为它们用于主键。现在，当我们的 shell 脚本管道想要在`mutation.graphql`中报告数据时，它只需运行下面的命令。

```
gq http://localhost:8080/v1/graphql \
  -H 'X-Hasura-Admin-Secret: MyHasuraSecret' \
  -q "$(cat mutation.graphql)"
```

最后一步是将我们的静态值换成变量并绑定它们(见下文)。与发送 SQL 插入或 HTTP/REST 请求相比，GraphQL 真正的优点在于，GraphQL 是强类型的，因此类型检查和验证是该语言的一部分。这对于精确度至关重要的科学领域尤其有价值。例如， *n* 小数点的值应完全按照该值进行通信，而不是作为字符串发送，因为存在根据目的地类型解析为较低精度的风险。类型检查还使得 shell 脚本开发变得更加高效和轻松，因为客户端可以在发送请求之前对其进行验证。

```
# mutation.graphqlmutation ($sample_id: Int!, $run_id: Int!, $qc_score: Int!, $duplication_rate: Float!){
  insert_sample_runs(objects: {
    sample_id: $sample_id,
    run_id: $run_id,
    qc_score: $qc_score,
    duplication_rate: $duplication_rate}) {
      returning {
        run_id
        sample_id
      }
  }
}
```

我们现在可以使用`-v`参数运行 graphqurl 来处理单个管道结果，或者使用`-variablesJSON`来处理一组已经格式化的结果，如下例所示。

```
gq http://localhost:8080/v1/graphql \
  -H 'X-Hasura-Admin-Secret: MyHasuraSecret' \
  -q "$(cat mutation.graphql)" \
  --variablesJSON '{"sample_id": 1003, "run_id": 502, "qc_score": 88, "duplication_rate": 0.5678}'# or -v for individual key-value pairs (to avoid JSON formatting)
```

通过这一条命令，我们的管道现在可以直接从 shell 脚本中保存结果，并免费进行类型检查和验证——这多好啊！添加新的数据点非常简单，只需在`mutation.graphql`中添加一行来定义要保存到哪个表和列，然后将新值作为参数传递。调试轻而易举，因为您可以从 GraphQL 获得有用的错误，其他任何人都可以过来打开`mutation.graphql`来准确理解正在发生的事情，而无需关注一个转义字符——天堂！

# 灵活的无代码事件触发器

既然我们已经保存了数据，那么触发一些事件就好了，而不仅仅是一封来自 Sendmail 旧版本的电子邮件，你知道它随时都会关闭。同样，您可以创建自己的 webhook 微服务，测试、部署、监控和维护它，但当 Hasura 包含一个易于配置并可以从友好的 Web UI 监控的可靠解决方案时，为什么要这么麻烦呢？

对于本例，让我们向 Slack 发送一条消息，提醒我们管道结果。首先，你需要按照 Slack 的[这些步骤](https://api.slack.com/messaging/webhooks)来获得你唯一的秘密 webhook URL。现在转到顶部的“事件”导航菜单选项卡，点击左侧*事件触发器*旁边的“创建”，给它命名，然后将你的 Slack URL 粘贴到 *Webhook 处理程序*框中。然后，我们希望选择 *sample_runs* 表，并选中 *Insert* 旁边的框，以便在我们从管道添加新记录时触发它。

![](img/799bd699047e3330006414cedc1e1f83.png)

作者图片

从[规格](https://api.slack.com/messaging/webhooks)中，我们还可以看到 Slack 解释消息所需的请求格式，因此我们需要相应地讨论我们的管道结果，这就是 Hasura(再次)为我们提供帮助的地方。只需创建一个新的*有效负载转换*，然后我们就可以配置请求体来匹配 Slack 规范。我们甚至得到了自动完成，请求的预览和我们输入时的错误检查——你还能要求什么呢！

![](img/a202ce7867696b6a8033dbf19cf81ecc.png)

作者图片

保存触发器后，向 *sample_runs* 添加新记录，然后返回 Hasura 控制台的 *Events* 选项卡。单击左侧的事件( *slack_notification* )，您应该会在 *Processed Events* 选项卡下看到最近的调用，以及在 *Invocation Logs* 选项卡下的完整请求和响应，这对调试很有用。

如果您有一个支持 webhook 的接收器/处理器，那么 web hook 就很棒，但是如果您有一个客户端，比如说一个独立的实验室仪器，它需要从一个事件中触发，但是不能运行一个专用的服务器来持续监听 web hook 调用，那该怎么办呢？Hasura 为您提供了*订阅—* 另一个宝贵的 GraphQL 特性，它允许客户端通过 websocket 监听实时变化，而无需轮询。客户端可能是一个简单的 web 应用程序，或者在这种情况下是 graphqurl，它可以订阅一个查询，然后在数据库值发生变化时触发仪器上的一个功能——查看 graphqurl [文档](https://github.com/hasura/graphqurl#readme)以获取示例。

希望这篇文章已经证明了，使用像 Hasura 这样的现代工具可以让你在为 shell 脚本管道添加新特性方面立竿见影。请记住，我们只是触及了 Hasura 的皮毛——使用 JWTs 进行认证、使用 DB 查询进行授权、模式拼接、数据库触发器、可定制的外部*操作*——对于您现有的数据库来说，它真的是一把无代码的瑞士军刀，值得一试。

在 [Whitebrick](https://hello.whitebrick.com/) 我们是 Hasura 的忠实粉丝，我们总是有兴趣了解您如何使用它，我们还提供咨询服务，所以不要犹豫[联系](https://hello.whitebrick.com/consulting)。