# 借助 DuckDB 和 Iceberg API 提升您的云数据应用

> 原文：<https://towardsdatascience.com/boost-your-cloud-data-applications-with-duckdb-and-iceberg-api-67677666fbd3>

## 使用 Iceberg API 和 DuckDB 优化云存储中大量 Iceberg 表的分析查询

![](img/965b6cdf5885cbd84327f2157e84503f.png)

休伯特·纽菲尔德在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

# 1.介绍

Apache Iceberg 最为人所知的是，它使 Spark、Dremio 和 Trino 等流行的查询引擎能够可靠地查询和操作存储在数据湖中的巨大表中的记录，并在确保安全的并发读写的同时大规模地这样做。因此，它解决了现代数据湖平台的一些主要问题，如数据完整性、性能、维护和成本。

Iceberg 之所以神奇，很大程度上是因为它有效地组织了表的元数据，并跟踪其数据文件、统计数据和版本历史。这也是 Iceberg 表上的查询速度更快、数据扫描更少的原因。对不使用或保存文件级元数据(例如 Hive)的表进行查询通常会涉及代价高昂的列表和扫描操作，只是为了找出查询需要运行的相关数据文件。相比之下，当我们为了满足查询而需要的数据位于一组特定的文件中时，Iceberg 原则上可以立即告诉我们这一点，甚至不需要扫描一个数据文件。

Iceberg 是一种表格格式，但它也是一个库，公开了一个强大的 API 和一组实现其主要功能的过程。Iceberg 的 API 使查询引擎能够利用其强大的元数据结构，并运行扫描较少文件和避免昂贵的列表操作的查询。

然而，并不是所有的用例都需要或证明查询引擎的重要性，比如 Spark 或 Trino，它们可能并不总是可用的。幸运的是，我们不必使用庞大的计算引擎来利用 Iceberg 的 API 所提供的功能。任何可以处理 parquet 文件的应用程序都可以使用 Iceberg 表及其 API，以便更有效地进行查询。这篇简短的实践文章的目的是演示如何操作。

DuckDB 是一个流行的进程内数据库，在读取和运行对 parquet 文件的 SQL 查询方面表现出色。因此，它为云数据应用程序提供了一个重要的功能，可以使它们更加健壮，减少对庞大查询引擎的依赖。然而，它处理大量数据的能力是有限的，主要是受运行它的机器或进程的内存大小的限制。这正是冰山 API 发挥作用的地方。

这篇文章将如下进行:第 2 节展示并解释了相关的用例，以及 Iceberg API + DuckDB 如何有效地解决这个问题。接下来，第 3 节演示如何使用 Iceberg API，第 4 节将 DuckDB 添加到图片中。第 5 节解释了两者如何一起发挥作用并解决上述问题。第 6 节总结。

(您可以在下面找到源代码的链接。我在代码示例中使用了 Scala，尽管它依赖于 Iceberg 的 Java API，并且很容易移植)

# 2.简而言之，这个问题(和解决办法)

我们有一个巨大的冰山表，按*日期*划分，包含来自我们客户的活动数据。每个事件由一个名为 *account_id* 的字段标识，包含一组计数器和一个 *event_type* 字段。因为大多数数据消费者通常将 *account_id* 添加到他们的 WHERE 子句中，所以我们决定按照 *account_id* 对每个分区中的文件进行排序(这是这里很重要的一点。我稍后会详细解释)。

要求是创建一个服务，告诉我们一个给定的客户在特定的一天发生了多少事件。该服务将接收某个日期作为参数，以及一个帐户 id，并将返回一个按 *event_type* 的聚合(以 JSON 格式)。简而言之，服务需要运行类似下面的查询:

```
SELECT date, account_id, event_type, count(*) as events
FROM customer_events
WHERE date = date '2022-01-02' AND account_id = '2500'
GROUP BY 1,2,3
```

听起来够简单吧？嗯，不一定。这项服务将不得不每天回答来自 10K 的大约 100 名客户的几十个请求。因此，按需运行或仅在需要特定客户的数据时运行是有意义的。这种用例似乎无法证明旋转 Spark 或 Trino 集群的合理性，但它必须扫描的数据似乎相当庞大。即使我们知道所需的分区，列出和扫描分区也可能是内存和计算密集型的。

这就是冰山 API 大放异彩的地方。如果表按*日期*分区，文件按 *customer_id* 排序(如下图所示)，那么表的元数据将显示，上面的查询实际上只需要扫描一个数据文件，而不是扫描整个分区及其文件——数据文件#2 ( `WHERE date=2022-01-02 AND account_id=2500).`。实际上，对 Iceberg 表运行这个查询将导致只扫描这个文件，这就是 Iceberg API 将告诉我们的。

![](img/4997e9a809559558d1a9e75e8b895ba4.png)

假设每个数据文件重约 250MB，那么单个文件的查询和扫描操作似乎更适合数据应用程序处理。

至于我们将在拼花文件上运行这些查询的方式，我已经提到过，我们将使用越来越流行的 DuckDB，因为它能够高效地在位于 S3 的拼花文件上运行 SQL 查询，此外它还易于嵌入到数据应用程序中。

现在我们可以结束我们的服务流程了(见下图):服务将通过一个 *date* 变量和一个 *account_id* 被调用。接下来，我们调用 Iceberg API 来查询与我们的查询相关的数据文件的表元数据。最后，在获得文件列表后，我们创建并执行一个 DuckDB 查询语句，该语句也将结果集格式化为 JSON 格式(一个简洁的 DuckDB 特性)并将它们返回给调用者。

![](img/9cf796ebaa1de1b310b72a86c03b8cbf.png)

# 3.使用 Iceberg API 过滤

如果你需要对 Iceberg API 有一个很好的介绍，那么请随意查看帖子底部的一些链接，尽管我相信代码足够简单，即使你对 Iceberg 知之甚少。

我们做的第一件事是创建一个过滤*表达式*对象，Iceberg 将使用该对象来查询包含与我们的过滤器匹配的数据的文件的表元数据。你可以链接表达式，也可以像我下面做的那样组合它们。

```
 val partitionPredicate = Expressions.and(
      Expressions.equal("date", "2022-11-05"),
      Expressions.equal("accountID", "0012345000"))
```

接下来，我们创建一个 Catalog 对象，这是 Iceberg 的 API 的入口点。我们使用 AWS Glue 作为我们的目录，但是您也可以使用 JDBC 和 Hadoop 目录。(注意，要实现这一点，在您的 env 或 path 中需要 AWS 凭证)。下面代码块中的第二个函数简单地返回一个 Iceberg 表对象，该对象将用于调用所需的操作，最后一个函数使用我们上面定义的表达式对象的可选变量执行一个*表扫描*过程。基于 Iceberg 的元数据，最后一个函数将告诉我们哪些文件与查询相关。

```
 private def getGlueCatalog(): Try[GlueCatalog] = Try{
  val catalog =  new GlueCatalog
  val props = Map(
    CatalogProperties.CATALOG_IMPL -> classOf[GlueCatalog].getName,
    CatalogProperties.WAREHOUSE_LOCATION -> "s3://Doesnt_Matter_In_This_Context",
    CatalogProperties.FILE_IO_IMPL -> classOf[S3FileIO].getName
  ).asJava
  catalog.initialize("myCatalog", props)
  catalog
}

private def getIcebergTableByName(namespace: String, tableName: String, catalog: GlueCatalog): Try[Table] = 
Try{
    val tableID = TableIdentifier.of(namespace, tableName)
    catalog.loadTable(tableID)
  }

private def scanTableWithPartitionPredicate(table:Table, partitionPredicate:Expression):Try[TableScan] =
  Try(table.newScan.filter(partitionPredicate))
```

在我们创建了 Iceberg *目录*，加载了我们的*表*，并使用过滤*表达式*执行了新的扫描之后，Iceberg 返回了一个*表扫描*对象。 *TableScan* 对象将有望包含符合我们过滤表达式的计划数据文件列表。下面的函数只是从表扫描中提取文件名，并将它们链接成一个长字符串，这样我们就可以使用 DuckDB 来查询它们。

```
 private def getDataFilesLocations(tableScan:TableScan): Try[String] = Try {
    // chain all files to scan in a single string => "'file1', 'file2'"
    tableScan.planFiles().asScala
      .map(f => "'" + f.file.path.toString + "'")
      .mkString(",")
  }
```

这总结了我们的服务对 Iceberg API 的使用。我们从一个目录开始，使用一个表达式过滤器启动一个表扫描，该表扫描使用我们的表的元数据来确定与我们的查询相关的文件。

# 4.使用 DuckDB 查询

DuckDB 是一个越来越受欢迎的进程内 OLAP 数据库，擅长在各种数据源上运行聚合查询。DuckDB 与类似产品(如 SQLite)的不同之处在于它为 OLAP 查询提供的性能，以及它提供的灵活性。简而言之，它本质上是一个进程内的小 DWH，使我们能够在相对较大的数据集上运行聚集密集型查询。

然而，有些数据集对于我们现有的或我们想要使用的机器或设备来说太大了。这正是 DuckDB 和 Iceberg 的结合非常强大的原因——有了 Iceberg API，我们可以在更少的时间内对更少的数据进行查询。

DuckDB 允许使用关键字`parquet_scan()`查询 parquet 文件，该关键字可用于如下查询:

```
SELECT date, account_id, event_type, count(*) as events
FROM parquet_scan([ <FILES_LIST>])
WHERE date = date '2022-01-02' AND account_id = '2500'
GROUP BY 1,2,3
```

因此，在使用 Iceberg 的 API 获得数据文件列表后，我们可以简单地用从 Iceberg 收到的文件列表替换上面查询中的字符串“<files_list>”，并对过滤后的文件执行查询。</files_list>

如下面的代码块所示，我们首先在内存中初始化 DuckDB。因为我们想在 S3 处理表，所以我们首先需要安装并加载 *httpfs* 模块(以及默认情况下应该已经加载的 *parquet* 模块)。模块通过执行语句加载到 DuckDB 中，这些语句也包括变量 setters，我们在这里使用它们来设置 AWS 凭证。下面代码中的第二个函数只是在查询中注入文件列表，正如我们上面看到的，最后一个函数执行语句。

```
 private def initDuckDBConnection: Try[Connection] = Try {
    val con = DriverManager.getConnection("jdbc:duckdb:")
    val init_statement =
      s"""
         |INSTALL httpfs;
         |LOAD httpfs;
         |SET s3_region='eu-west-1';
         |SET s3_access_key_id='${sys.env.get("AWS_ACCESS_KEY_ID").get}';
         |SET s3_secret_access_key='${sys.env.get("AWS_SECRET_ACCESS_KEY").get}';
         |SET s3_session_token='${sys.env.get("AWS_SESSION_TOKEN").get}';
         |""".stripMargin
    con.createStatement().execute(init_statement)
    con
  }

private def formatQuery(query:String, dataFilesStr:String):Try[String]  = Try {
    query.replaceAll("<FILES_LIST>", dataFilesStr)
  }

  private def executQuery(connection: Connection, query:String):Try[ResultSet] = Try{
     connection.createStatement.executeQuery(query)
  } 
```

这解决了我们在查询端的核心服务。在我们获得列表文件之后，我们只需要确保 DuckDB 被正确初始化，格式化查询并执行它。

# 5.把所有的放在一起

如前所述，我们的服务的处理逻辑包括 2 个主要阶段，在下面的代码块中详细描述了 7 个步骤(Scala 的理解实际上使这变得非常简单)。

```
 val queryStatement = """
     |SELECT row_to_json(resultsDF)
     |FROM (
     |   SELECT date, account_id, event_type, count(*) as events
     |   FROM parquet_scan([ <FILES_LIST>])
     |   WHERE acc_id = '2500' AND date = '2022-01-01'
     |   GROUP BY 1,2,3
     |) resultsDF
     |""".stripMargin

val filterExpr = Expressions.and(
      Expressions.equal("date", "2022-11-05"),
      Expressions.equal("accountID", "0012345000"))

val jsonDataRows = for {
  catalog         <- getGlueCatalog
  table           <- getIcebergTableByName("db_name", "table_name", catalog)
  tableScan       <- scanTableWithPartitionPredicate(table, filterExpr)
  dataFilesString <- getDataFilesLocations(tableScan)
  queryStatement  <- formatQuery(query, dataFilesString)
  dbConnection    <- initDuckDBConnection
  resultSet       <- executQuery(dbConnection, queryStatement)
} yield resultSet.toStringList
```

Iceberg API 首先用于获取表引用并执行表扫描，获取包含我们感兴趣的*日期*以及 *customer_id* 的数据文件的名称和位置。一旦我们得到过滤后的文件列表和它们在 S3 的位置，我们就把它们链接成一个长字符串，并注入到查询中。最后，使用 DuckDB 仅对相关文件执行查询，并获得结果。

如您所见，上面的查询模板被包装在一个`row_to_json`函数中，该函数将结果集转换成一个 JSON 文档，这正是我们想要实现的。

# 6.结论

Apache Iceberg 正迅速成为元数据管理和组织海量数据湖表的标准。因此，毫不奇怪，流行的查询引擎，如 Impala、Spark 和 Trino，以及公共云提供商很快宣布支持 Iceberg table 格式和 API。

这篇文章的目的是展示一种强大的方法，数据应用程序可以独立于强大的查询引擎利用 Iceberg 表提供的好处。它展示了我们如何将 Iceberg 的 API 与 DuckDB 结合使用，以创建轻量级但功能强大的数据应用程序，这些应用程序可以在大规模表上运行高效的查询。我们看到，通过使用 Iceberg 公开的 API，我们可以创建一个“优化的”查询，只扫描与查询相关的文件。DuckDB 可以轻松地对存储在云存储中的 parquet 文件进行查询，这使得两者的结合非常强大，并突出了 Iceberg 表格式及其特性的潜力。

完整的源代码可以在[这里](https://github.com/a-agmon/icebergapi/blob/main/main.scala)找到

希望这个有用！

* **资源**

一个非常好的冰山介绍可以在[这里](https://www.dremio.com/subsurface/apache-iceberg-101-your-guide-to-learning-apache-iceberg-concepts-and-practices/)和[这里](https://medium.com/expedia-group-tech/a-short-introduction-to-apache-iceberg-d34f628b6799)找到

更多关于 Java API 的信息可以在这里找到

** *所有图片，除非特别注明，均为作者所有*