# 使用 Hasura、哈比神和无服务器发送 HL7 v2 消息

> 原文：<https://towardsdatascience.com/sending-hl7-v2-messages-with-hasura-hapi-and-serverless-9cd1630876a4>

![](img/2fc03def2c35f77ec75da076031e1860.png)

作者图片

健康级别 7 (HL7)版本 2 于 80 年代末开发，多年来一直是世界上最流行的数据交换标准。该标准最近被 HL7 快速医疗保健互操作性资源(FHIR)取代，但采用率比预期的要[慢，所以即使在 2022 年，使用 HL7 v2 仍然令人惊讶地普遍。](https://www.fedscoop.com/fhir-standard-adoption-stagnant-onc/)

在这篇文章中，我将概述一个典型用例的解决方案:假设我们有一个传统的电子病历(EMR)系统，在一个诊所中有一个支持关系数据库，每当一个新病人注册时，它需要向另一个机构发送一条消息(例如 ADT_A04)。下面的步骤可以作为演示在开发机器上运行—不需要云平台。虽然这篇文章关注的是 HL7 v2，但我们目前正在研究为 Hasura 设置的 GraphQL & REST FHIR，所以如果你对此感兴趣，请去投票支持我们的 [GitHub 讨论](https://github.com/whitebrick/hasura-fhir/discussions/1)。

解决这个问题最常见的方法是使用一个名为*happy Connect*的集成工具，这是一个很好的基于 Java 的服务器应用程序，最初是在 15 年前开发的，主要面向医疗保健行业。尽管 happy 的核心是开源的，但是像 FHIR 支持这样的扩展需要订阅，截止到本文撰写时，订阅费用大约为每年 2 万美元。

相比之下，Hasura 虽然才出现了几年，但已经被从沃尔玛到空中客车等各行各业的组织所使用，并支持现代微服务架构和设计模式。Hasura 是完全开源的(所有特性都可以免费获得)，用 Haskell 开发，可以作为 Docker 镜像快速部署。

# 为什么是哈苏拉？

选择 Hasura 而不是 happy Connect 的几个原因包括:

*   **instant API**:如果您正在通过支持 RDBMS 将 HL7 v2 添加到传统 EMR，这可能只是许多(当前或未来)集成中的一个，因此在您现有的数据库上免费获得一个 Instant API 肯定会派上用场。此外，为了支持这一特性，Hasura *跟踪*DB 模式，该模式具有自动生成表列映射的额外好处，便于分配给 HL7 消息模板字段，而不必手动编写 SQL 查询。
*   **GraphQL** :如果你还在考虑 REST 和 GraphQL，那么你就错过了联邦、模式拼接和远程连接的大画面，我鼓励你去看看由 Uri Goldshtein 为 Firely DevDays 做的[演讲。GraphQL 还支持现代客户端框架(如 Gatsby、Next.js、Gridsome ),可以快速开发定制屏幕和仪表盘(在医疗保健领域无处不在),无需开发定制端点即可读取/写入所需数据。](https://www.youtube.com/watch?v=i3Sj8JTp4MU)
*   **同步事件:**Hasura 在后台创建并管理主机数据库中的触发器，以便同步触发相应的动作。另一方面，joyful 的方法是定期轮询数据库，这种方法效率不高，并且与事件驱动的体系结构相反。
*   **管理控制台:**这种类型的集成可能是一个复杂的过程，因此任何有助于调试的东西通常都是必不可少的。joyful Connect 确实有一个 Java GUI，但是与 Hasura 相比，它缺少了一个对这个场景特别有用的特性:一个可视化的数据库浏览器。Hasura 基于 web 的管理控制台允许我们在不离开 web 浏览器的情况下，可视化我们的 DB 以及完成对行、表和关系的 CRUD 操作。

# 演示概述

*要求:Git、Docker、Node.js、NPM、Java ≥8*

![](img/71a9c3a07180e753d69e70fd1fe75982.png)

图片由作者提供，徽标经许可使用

这个解决方案采用微服务方法，从 Hasura 连接到现有的 EMR DB 开始，并监视 Patients 表中的新插入。当一个新病人被注册时，Hasura 将插入的数据转换成 JSON 格式的模板 HL7 消息，并将其作为 HTTP 请求发送给运行在无服务器框架上的 [Java HAPI](https://github.com/hapifhir/hapi-hl7v2) 服务(在本演示中使用无服务器离线)。Java HAPI 是 HL7 处理的黄金标准——该服务解析 JSON HL7 消息，将其转换为 ER7 格式(管道分隔),并将其作为 HTTP 请求转发给 HTTP-MLLP 网关。网关是一个简单的 Node.js 应用程序，它接收 ER7 格式的 HTTP 请求体，并通过 MLLP 将其发送到最终目的地，接收 ACK 响应并将其返回给 Java HAPI 服务，该服务再将响应返回给 Hasura。理想情况下，MLLP 网关也将作为无服务器功能运行，但不幸的是，AWS API 网关和应用程序负载平衡器目前只允许基于 HTTP 的调用。

# Postgres 和 Hasura 入门

Hasura 支持 Postgres、MS SQL、Citus 和 BigQuery，并且刚刚发布了 MySQL 的早期版本。在这个演示中，我们将使用 Postgres 和来自 [OpenEMR](https://www.open-emr.org/) 的稍微重构的 *patient_data* 表。

按照下面的命令进行设置。

```
$ git clone [git@github.com](mailto:git@github.com):whitebrick/hl7v2-hasura-hapi-serverless.git
$ cd hl7v2-hasura-hapi-serverless
$ psql# Create a new user, DB and add the pgcrypto extension
postgres=# CREATE USER myemrusr WITH password 'myemrpwd';
postgres=# CREATE DATABASE myemr OWNED BY myemrusr;
postgres=# CREATE DATABASE myemr WITH OWNER = myemrusr;
postgres=# \c myemr
postgres=# CREATE EXTENSION pgcrypto;
postgres=# \q# Test new user and DB and load data
$ psql -U myemrusr myemr
myemr=> \i ./sql/openemr_patient_data_table_postgres.sql
myemr=> \i ./sql/openemr_patient_data_rows_postgres.sql
```

一旦我们设置好了`patient_data`表，我们就可以从 Docker 启动 Hasura，并配置它使用我们的新数据库，如下所示(注意，数据库主机是用于 Mac 的，参见 Linux 或 Windows 的[指南](https://hasura.io/docs/latest/graphql/core/guides/docker-networking/))。Hasura 在 DB 中创建了一个新的模式`hdb_catalog`来保存所有的元数据，这样它就不会接触到我们在 *public* 模式中的数据，除非我们告诉它这样做。

```
# Edit as required
$ vi ./hasura_docker_run.bashdocker run -d -p 8080:8080 \
-e HASURA_GRAPHQL_DATABASE_URL= postgres://myemrusr:myemrpwd@host.docker.internal:5432/myemr \
-e HASURA_GRAPHQL_ENABLE_CONSOLE=true \
-e HASURA_GRAPHQL_DEV_MODE=true \
-e HASURA_GRAPHQL_ADMIN_SECRET=admin \
hasura/graphql-engine:latest$ bash ./hasura_docker_run.bash
```

现在转到 [http://localhost:8080](http://localhost:8080/console) ，当提示输入管理员密码时，输入上面的值(本例中为“admin”)。一旦登录，我们需要做的第一件事就是*跟踪*T2 表，这样 Hasura 就可以分析和监控它。点击*数据*导航选项卡，点击`patient_data`工作台旁边的 *e 轨迹*按钮。一旦被跟踪，我们现在可以从上面描述的方便的管理控制台可视化和管理数据库。

![](img/79cc04de94c500ab884e429a7a9dfa0c.png)

作者图片

现在，Hasura 理解了我们的表，这是使用即时 API 有多简单的快速示例，如果我们单击 *API* 选项卡，我们可以简单地选择左侧的几个复选框，一个 GraphQL 查询就写在我们眼前——包括语法突出显示、代码完成和错误检查。然后，我们可以点击播放按钮来查看我们的查询结果。

![](img/a7d936793e5abfc3a96842ec2fb30068.png)

作者图片

# 无服务器运行 HAPI

我们将离开 Hasura 一会儿，通过安装[无服务器框架](https://www.serverless.com/framework/docs/getting-started)并使用下面的命令构建和测试[hapi-无服务器](https://github.com/whitebrick/hapi-serverless)来运行我们的 HAPI 微服务。[无服务器离线](https://github.com/dherault/serverless-offline)插件允许我们在本地运行 lambda，就像在云中一样，方法是在幕后使用 docker，这就是为什么我们要提取`lambda:java8`图像。

**注意:由于无服务器离线使用 Docker，初始请求可能需要一分钟左右的时间来处理。**

```
# Build
$ git clone [git@github.com](mailto:git@github.com):whitebrick/hapi-serverless.git
$ cd hapi-serverless
$ npm install
$ mvn package
$ docker pull lambci/lambda:java8# Start
$ npm start
...
Server ready: [http://localhost:3000](http://localhost:3000)# Test
$ cd test/functional
$ bash ./test_parsing.bash
$ bash ./test_forwarding.bash
```

# 从 Hasura 创建 HL7 v2 消息

下一步是获取模板消息。尽管我们可以从头开始构建消息，但通常最好的做法是向消费者索要一个模板，这样任何与标准不符的地方都会被考虑在内。对于这个例子，我在谷歌上搜索 ADT_04 样本，北达科他州卫生部[消息指南](http://ndhealth.gov/disease/SS/Docs/NDImplementationGuide_v2.pdf)首先出现。我从指南中复制了 ER7 格式的示例消息，并将其粘贴到文件`./hl7/NDDH_ADT_A04.er7`中，然后将其发布到 HAPI 无服务器服务器，以获得`./hl7/NDDH_ADT_A04.json`中的 JSON 表示。

现在回到 Hasura，我们将点击 *Events* 选项卡，创建一个名为`HL7-Send_ADT`的新事件，并将其附加到`public`模式中的`patient_data`表。我们只想触发新的`Insert`操作，我们将从上面放入本地 HAPI 无服务器 URL，但针对 Docker 网络进行了调整(`[http://host.docker.internal:3000/dev](http://host.docker.internal:3000/dev)`这是针对 Mac 的，参见[指南](https://hasura.io/docs/latest/graphql/core/guides/docker-networking/)针对 Linux 或 Windows)。

![](img/2ebbab3ea2a2c92fec7d686662ed2180.png)

作者图片

我们还想将超时时间增加到`120`秒，以便让我们的 HAPI 无服务器 Docker 映像有足够的时间启动，将请求方法设置为`POST`，将`Content-Type`头设置为`application/json`。

![](img/73a558b7b74e6847ac9cd596ed176594.png)

作者图片

接下来，我们要单击*添加有效负载转换*，在这里我们可以将插入的 DB 值映射到 HL7 JSON 模板。因为 Hasura 已经知道了我们的表，所以它可以自动创建预期的插入输入。然后我们粘贴来自`./hl7/NDDH_ADT_A04.json`的模板，并开始将值映射到我们的列——对于这个演示，我们将只映射名字、姓氏和中间名。正如我们在下面看到的，transform editor 还提供了语法突出显示、代码完成和验证，以及一个示例预览(一旦代码完成/有效就会显示)。使用 Hasura 自己的 [Kriti Lang](https://github.com/hasura/kriti-lang) 将`{{$body.event.data.new.*}}`路径映射到输入的 JSON 结构，这看起来很熟悉，是受 Go 的模板语言的启发。

![](img/2fbc7f8630c1e1648f3dd892c16f680d.png)

作者图片

最后点击页面最底部的*保存*按钮，现在让我们测试一下。点击*数据*导航选项卡上的返回，选择`patient_data`表，然后选择*插入行*菜单选项卡，然后输入名称值(见下文)并点击*保存*按钮。

![](img/ae10478630af8640f40f537b0ef4cc64.png)

作者图片

假设我们的 hapi-serverless 仍然在运行，并且我们有正确的 Docker 网络 URL，我们现在应该在终端日志上看到一个点击。如果我们单击返回到*事件*导航选项卡，选择`HL7-Send_ADT`事件，然后选择*未决事件*子选项卡，我们应该会看到一个新事件在等待，因为我们的 HAPI 无服务器需要一些时间来启动和响应。最终，该事件将被移动到 *Processed Events* 选项卡，并允许我们查看请求和响应，在那里我们可以看到我们的 ER7 转换。

![](img/70507183c45377b3ef4587e91aa1f6a0.png)

作者图片

# HTTP 到 MLLP

不幸的是，HTTP 在 80 年代末不像在 90 年代末那样普及，所以我们只能用 MLLP，但至少它仍然基于 TCP/IP。为了通过 MLLP 转发 ER7 消息，我们使用 Express 和 [mllp-node](https://github.com/amida-tech/mllp) 包组装了一个非常[基本的网关](https://github.com/whitebrick/http-mllp-node)在 Node 上运行。网关寻找一个头部，例如`Forward-To: mllp://ack.whitebrick.com:2575`，并使用 mllp-node 发送带有相应控制字符的主体，然后返回一个 HTTP 响应。如果您有自己的端点来测试接收 ACK 消息，您可以在`Forward-To`头中使用它，否则欢迎您使用我们上面的 Apache Camel 测试端点——详情[此处](https://hl7v2-test.whitebrick.com/)。

```
# Build
$ git clone [git@github.com](mailto:git@github.com):whitebrick/http-mllp-node.git
$ cd http-mllp-node
$ npm install

# Start
$ node gateway.js
...
HTTP server listening on port 3030# Test
$ cd test/gateway-functional
$ bash ./functional.bah
```

# 从 Hasura 发送 HL7 v2 消息

既然我们已经运行了 MLLP 网关，我们就可以使用 HAPI 无服务器的转发特性，不仅可以转换来自 JSON 的消息，还可以通过网关转发 ER7 数据并接收返回的响应。为了做到这一点，我们只需将下面的标题(根据您自己的 Docker 网络进行调整)添加到我们的事件配置中。

*   `MLLP-Gateway` : `[http://host.docker.internal:3030](http://host.docker.internal:3030)`
*   `Forward-To` : `mllp://ack.whitebrick.com:2575`

![](img/c68701e44a2fcce5bbe7be1452077f16.png)

作者图片

有了附加的标题，我们可以返回到*数据*页面，插入另一条记录来测试完整的发送和响应。如果一切顺利，我们现在应该看到`retEr7`和`retJson`值以及来自下游 MLLP 消费者的 ACK 数据。

![](img/8fb536f5fbed7b652b2fe1ef04e9d1db.png)

作者图片

就是这样！我们现在有一个完整的端到端解决方案，因此当 EMR 创建新的患者记录时，Hasura 从一个模板构建一个 HL7 v2 JSON 消息，通过 MLLP 网关发送它，捕获 ACK 响应，并在管理控制台中很好地显示结果。

# 访问响应数据并发出错误警报

在发送了一些消息后，下一个明显的问题是:如果我想对响应数据做些什么，而不仅仅是在管理控制台中显示它，该怎么办？因为 Hasura 元数据存储在同一个 DB 中(但是是独立的模式),所以我们可以创建一个 function+触发器，将感兴趣的数据复制到我们自己的表中，然后在新表上设置附加事件。例如，假设当 HL7 消息有错误响应时，我们希望通过发送电子邮件或 Slack 消息来发出警报。

让我们首先返回到*数据*导航选项卡，单击左侧的 *SQL* 菜单，粘贴下面的代码并点击*运行*按钮，为我们的错误消息创建一个新表。

```
CREATE table hl7_error_messages(
  id serial PRIMARY KEY NOT NULL,
  created_at timestamp without time zone DEFAULT now(),
  request json,
  response json
)
```

如果我们使用 psql 查看 Hasura 表`hdb_catalog.event_log`和`hdb_catalog.event_invocation_logs`，我们将从管理控制台视图中看到相同的数据。当 Hasura 运行任何事件(例如发送 HL7 消息)时，下面的触发器调用`copy_hl7_error_messages`函数。然后，该函数检查事件名称是否与`HL7-Send_ADT`匹配，并且在将它复制到我们新创建的`hl7_error_messages`表之前，状态是否为不成功。额外的一行是取消转义并解析响应 JSON，因为它将进入 Postgres JSON 字段。

```
CREATE OR REPLACE FUNCTION copy_hl7_error_messages()
RETURNS trigger AS
$$
DECLARE
  event_name text;
  response_json json;
BEGIN
  SELECT trigger_name INTO event_name FROM hdb_catalog.event_log WHERE id = NEW.event_id;
  IF (event_name = 'HL7-Send_ADT') AND (NEW.status != 200 OR NEW.status IS NULL) THEN
    -- unescape JSON
    SELECT cast('{"data":{"message":' || cast(((NEW.response->'data'->'message') #>> '{}' )::jsonb as text) || '}}' as json) INTO response_json;
    INSERT INTO hl7_error_messages (request, response)
    VALUES (NEW.request, response_json);
    RETURN NEW;
  ELSE
    RETURN NULL;
  END IF;
END;
$$
LANGUAGE plpgsql;CREATE TRIGGER copy_hl7_error_messages_trigger
AFTER INSERT
ON hdb_catalog.event_invocation_logs
FOR EACH ROW
EXECUTE PROCEDURE copy_hl7_error_messages();
```

接下来点击数据库下的*默认*菜单链接，然后点击 *hl7_error_messages* 旁边的*跟踪*按钮。现在，让我们通过终止网关节点进程并插入一个新的患者记录来测试这个功能。幸运的话，我们现在应该在`hl7_error_messages`表中有一条记录，因为我们没有对消息进行转义，所以我们可以用 JSONPath 直接从 API 中查询它(见下文)。

![](img/610629fcae1e85500d8bcf28cdb62910.png)

作者图片

既然错误响应存在于一个*跟踪的*表中，我们可以继续通过 Hasura 创建任意数量的新事件，遵循我们之前所做的相同步骤，在新记录插入到`hl7_error_messages`表*中时触发电子邮件/松弛通知警报挂钩。*

我希望这个演示为您下次使用 HL7 v2 提供了另一个选择，如果您正在采用微服务方法，我们很想知道如何实现。如果你需要帮助或想要更多关于医疗保健互操作性和集成的阅读，请前往 whitebrick.com，不要犹豫。