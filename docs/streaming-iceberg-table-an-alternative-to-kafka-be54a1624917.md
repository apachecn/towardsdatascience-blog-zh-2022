# 流式冰山桌，卡夫卡的替代品？

> 原文：<https://towardsdatascience.com/streaming-iceberg-table-an-alternative-to-kafka-be54a1624917>

## Spark 结构化流支持 Kafka 源和文件源，这意味着它可以将文件夹视为流消息的源。完全基于文件的解决方案真的能与 Kafka 这样的流媒体平台相比吗？

![](img/2a4754c5a3b430b6caf6866dafe00c71.png)

爱德华·库雷在 Unsplash 上的照片

在本文中，我们探索使用冰山表作为流消息的来源。为此，我们创建了一个将消息写入 Iceberg 表的 Java 程序和一个读取这些消息的 pySpark 结构化流作业。

# 动机

Azure Event Hubs 是一个大数据流平台，每秒能够处理数百万个事件。但是，它缺乏消息压缩，并且消息被限制为 1 MiB。当处理非常高吞吐量的数据流时，事件中心可能成为昂贵的选择。Apache Kafka 支持消息压缩。也可以编写大于 1 MiB 的消息，但不建议这样做，因为非常大的消息被认为是低效的。在虚拟机上运行 Kafka 流媒体平台在人力和物理资源方面也可能成本高昂。有没有一个替代方案，Azure blob 存储可以被用作一个排队系统？

# 文件源

Spark 结构化流能够从各种来源读取流数据；卡夫卡，事件中心，文件。但是，内置文件源的伸缩性不好，因为在每个微批处理中，文件源都会列出源目录下的所有文件。这样做是为了确定哪些文件是新添加的，需要在下一个微批处理中处理。随着文件数量的增长，列表的成本会越来越高。

尽管有这个缺点，FileSource 还是有锦囊妙计的。它还检查是否存在一个`_spark_metadata`文件夹。当 Spark 结构化流用于使用`writeStream`创建文件目录时。FileSource 创建一个包含“事务日志”的`_spark_metadata`,记录在每个微批处理中哪些文件被添加到目录中。这里有一个使用`writeStream`写入目录的例子

```
eventsDf.writeStream
    .outputMode("append")
    .format("json") # spark tables can be in any format supported by spark
    .trigger(processingTime='5 seconds')
    .option("checkpointLocation", "/tmp/stream_eval/checkpoint")
    .option("path", "/tmp/stream_eval/output/")
    .start()
```

文件源`readStream`利用了`_spark_metadata` 文件夹，因此避免了昂贵的列表。

但是，这不是唯一支持元数据的表格格式。Apache Iceberg 也是一种表格格式，支持时间旅行和回滚。它通过提交历史记录跟踪添加/删除的文件。这张图[展示了 Iceberg 提交的结构。简而言之，每次提交都会记录表中添加或删除的数据文件。因此，Iceberg `readStream`利用它的元数据来避免昂贵的列表。](https://iceberg.apache.org/spec/#overview)

Apache Iceberg 可以在 Spark、Trino、Flink、Presto 和 Dremio 等许多大数据引擎中使用。有趣的是，Iceberg 的 Java API 也可以用于独立的 Java 程序。

# 作家们

我们将使用 Iceberg Java API 创建并写入 Iceberg 表。我们用一个类似于卡夫卡主题的图式来说明这个概念；时间戳、密钥和值。

```
Schema schema =
  new Schema(
    Types.NestedField.optional(1, "timestamp", Types.TimestampType.withZone()),
    Types.NestedField.required(2, "key", Types.LongType.get()),
    Types.NestedField.optional(3, "value", Types.BinaryType.get())); 
```

创建冰山表非常简单

```
PartitionSpec spec = PartitionSpec.builderFor(SCHEMA).build();
HadoopTables tables = new HadoopTables(hadooConf);
Map<String,String> properties = new HashMap();
properties.put("commit.retry.num-retries", "200");
Table table = tables.create(schema, spec, properties, tableLocation); 
```

为了为我们的表创建拼花文件，我们使用 Iceberg 的 DataWriter

```
OutputFile file = HadoopOutputFile.fromLocation(fullPath, hadooConf);
DataWriter<Record> dataWriter =
    Parquet.writeData(file)
        .forTable(table)
        .createWriterFunc(GenericParquetWriter::buildWriter)
        .build();

try {
  for (Record record : createRecords()) {
    dataWriter.write(record);
  }
} finally {
  dataWriter.close();
}
```

这段代码在数据湖中创建物理拼花文件。然而，这些新文件对 Iceberg 表的客户机来说还不可见。最后一次表提交不包括对这些新数据文件的任何引用。为了使这些文件对客户机可用，我们必须将这些文件附加到表中并提交事务。

```
DataFile dataFile = dataWriter.toDataFile();
AppendFiles append = table.newFastAppend();
append.appendFile(dataFile);
append.commit();
```

DataFile 对象是真正的 Parquet 文件的名字。DataFile 对象捕获诸如文件大小、列的上限/下限、Parquet 文件路径和记录数量之类的信息。可以多次调用`appendFile`方法，在一次提交中向表中添加多个数据文件。

# 并发性

Iceberg 使用乐观并发策略。一个编写器假定没有其他编写器同时在表上操作。在提交操作期间，编写器创建并写出新的元数据文件。然后，编写器尝试将这个新的元数据文件重命名为下一个可用的版本化元数据文件。如果重命名因另一个编写器同时提交而失败，失败的编写器将重试提交操作。失败的写入程序重建新的元数据文件，并尝试再次提交。这个过程在[这里](https://iceberg.apache.org/docs/latest/reliability/#concurrent-write-operations)有更详细的解释。

当作者创建大的 Parquet 文件并偶尔提交时，这种策略非常有效。在这种情况下，写入者花费大部分时间创建数据文件，只有一小部分时间用于创建提交。提交阶段的争用很少。

尽管如此，如果许多作者频繁地提交少量数据，乐观锁定机制可能会成为瓶颈。在这种情况下，作家花大部分时间试图提交。

# 簿记员

为了实现频繁的提交，有必要将提交阶段集中在一个提交者中，我们称之为簿记员。编写器仍然创建拼花文件，但不会将它们附加到表中。这项任务委托给簿记员。只有簿记员将数据文件追加到表中。

然而，这需要写入者以某种方式通知簿记员它需要注册的数据文件是什么。有几种方法可以做到这一点。一种简单的方法是利用数据湖。

当编写器创建一个或多个 Parquet 文件时，它会将数据文件对象序列化到一个“事务日志”中，该日志位于编写器和簿记员已知的位置。

```
// create parquet file
...

// obtain DataFile moniker objects
List<DataFile> dataFiles = ...
dataFiles.append(dataWriter.toDataFile());

// serialize DataFile monikers
Path txLogDir = new Path("abfss://.../transactionLogs/"
Path txLogTemp = new Path(txLogDir, uuid);
Path txLog = new Path(txLogDir, uuid + ".txlog.ser");
try (FSDataOutputStream fout = fs.create(txLogTemp);
        ObjectOutputStream out = new ObjectOutputStream(fout) ) {
    out.writeObject(dataFiles);
}

// make transaction log visible to single committer
fs.rename(txLogTemp, txLog);
```

簿记员不断列出“事务日志”文件夹，发现需要追加到表中的新数据文件。

```
List<DataFile> dataFiles = …
Path txLogDir = new Path("abfss://…/transactionLogs/"
FileStatus[] txLogFiles = listTransactionLogs(txLogDir);
for (FileStatus txLog: txLogFiles) {
  try (FSDataInputStream fin = getFileSystem().open(txLog.getPath());
        ObjectInputStream in = new ObjectInputStream(fin)) {
    dataFiles.appendAll(in.readObject());
  }
}
```

簿记员然后将数据文件对象的列表附加到表中并提交。

```
AppendFiles append = table.newFastAppend();
for(DataFile dataFile : dataFiles) {
  append.appendFile(dataFile);
}
append.commit();
```

簿记员然后可以安全地丢弃交易日志。

```
for (FileStatus txLog: txLogFiles) {
  getFileSystem().delete(txLog.getPath());
}
```

为了实现具有一定保留期的有限队列，簿记员将旧数据文件标记为删除。

```
import org.apache.iceberg.expressions.Expressions;
// keep 7 days
long nowMicros = System.currentTimeMillis() * 1000;
long watermarkMicros = nowMicros - (7 * 24 * 60 * 60 * 1000 * 1000);
table
  .newDelete()
  .deleteFromRowFilter(
      Expressions.lessThan("timestamp", watermarkMicros))
  .commit();
```

一旦数据文件被标记为删除，它就不再对该表的客户机可用。但是，它尚未被物理删除。

# 死神

难题的最后一部分是物理删除未引用的元数据和数据文件。为此，我们让旧快照过期。当 Iceberg 删除一个快照时，它也物理地删除了被簿记员标记为`to-be-deleted`的元数据文件和数据文件。

```
long nowMillis = System.currentTimeMillis();
long watermarkMillis = nowMicros - (10 * 60 * 1000); // 10 minutes
table
  .expireSnapshots()
  .expireOlderThan(watermarkMillis)
  .retainLast(100)
  .commit()
```

作者是完全独立的，我们可以随心所欲地发表他们的作品。簿记员的工作是轻量级的；它读取事务日志，并向 Iceberg 表提交追加/删除数据文件请求。

在后台，Reaper 定期终止快照，并以物理方式删除旧数据文件。

# 火花写作流

当然，如果使用 Spark 是一种选择，那么创建一个流冰山表是非常容易的。你所要做的就是运行一个带有`iceberg`格式的 Spark `writeStream`查询。Iceberg 被集成到 Spark 中，处理作者和簿记员之间的协调。

```
query = ( df
        .writeStream
        .outputMode("append")
        .format("iceberg")
        .trigger(processingTime="5 seconds")
        .option("checkpointLocation", "abfss://.../checkpoint/")
        .toTable("icebergcatalog.dev.events_table")
    )
```

然而，您仍然需要运行一个 Reaper 进程。Iceberg 通过一个 SQL 程序解决了这个问题。更多详情请点击[这里](https://iceberg.apache.org/docs/latest/spark-procedures/#expire_snapshots)。

```
spark.sql("""
  CALL icebergcatalog.system.expire_snapshots(
    'dev.events_table', TIMESTAMP '2021-06-30 00:00:00.000', 100)
""") 
```

# **读者**

Iceberg 表与 Spark 结构化流深度集成，因此支持`readStream`和`writeStream`功能。得益于 Iceberg 的提交机制，`readStream`可以在每个微批处理中高效地发现哪些文件是新的。下面是一个 pySpark 流查询示例。

```
# current time in milliseconds
ts = int(time.time() * 1000)
# create a streaming dataframe for an iceberg table
streamingDf = (
    spark.readStream
    .format("iceberg")
    .option("stream-from-timestamp", ts)
    .option("streaming-skip-delete-snapshots", True)
    .load("icebergcatalog.dev.events_table")
)
# start a streaming query printing results to the console
query = (
    streamingDf.writeStream
    .outputMode("append")
    .format("console")
    .trigger(processingTime="15 seconds")
    .start()
)
```

`stream-from-timestamp`用于在特定时间点定位第一个微批次。在每个微批处理之后，偏移量在检查点期间被存储到数据湖中(容错支持)。

Iceberg 无法限制每个微批处理的行数或文件数。然而，添加此功能的 [git pull 请求](https://github.com/apache/iceberg/pull/4479)当前处于打开状态。

# 结果呢

在我们的第一个实验中，我们模拟了一个保留时间为 1 小时的队列。我们使用 5 个写入器，它们不断地创建一个 20k 记录的拼花文件。每个记录大约为 1700 字节，因此拼花文件为 20MiB。一个写入程序创建一个拼花地板数据文件平均需要 7 秒钟。

我们使用一个簿记员，它不断地读取事务日志并提交附加/删除文件请求。列出事务日志文件夹的内容平均需要 8 秒，读取和提交附加文件平均需要 2 秒。标记要删除的文件只需不到 2 秒钟。簿记员花大部分时间在数据湖中列出文件。不再需要列出文件将减少延迟。例如，Writer-1 可以将其数据文件对象序列化到一个众所周知的文件 txlog1.ser 中。簿记员随后将检查该文件是否存在并加载它。检查文件的存在只需要大约 300 毫秒。

我们安排收割者每两分钟运行一次。物理删除快照和旧数据文件需要 10 到 70 秒。幸运的是，这是一项后台工作，不会影响读者的数据可用性。

我们使用配置为使用 5 个线程(5 个 CPU)的 pySpark 结构化流作业读取队列。处理一个 20 万条记录的小批量的平均时间是 11 秒。

![](img/526251e02a61afc13b5f3736ec9632b7.png)

查看细分，我们看到 Iceberg 用了不到 500 毫秒的时间来决定在微批处理中消耗哪些文件。微批执行时间相对稳定，大部分时间花在处理记录上。

我们还尝试使用 Avro 文件的冰山表，而不是拼花文件。结果是可比较的，除了作者花了 6 秒来创建一个 Avro 文件。

在我们的第二个实验中，我们使用 24 小时的保留时间。因此，Iceberg 记录了 24 倍数量的数据文件。我们没有看到簿记员或读者的表现下降。

# **结论**

如果不需要亚秒级的延迟，那么在 Azure 数据湖上使用 Iceberg 表可能是一个不错的选择。在本文中，我们展示了小于 15 秒的延迟是可以实现的。这对于基于微批处理架构的 Sparks 结构化流查询来说是完全可以接受的。这是批量处理大容量流的一个很好的用例。然而，当延迟很关键时，例如在高频交易中，这种选择并不理想。

这个设计很简单。流式作业使用与批处理作业相同的技术；数据湖和冰山。不需要维护额外的基础设施或支付额外的服务协议费用。在[的后续文章](https://medium.com/towards-data-science/leveraging-azure-event-grid-to-create-a-java-iceberg-table-d419da06dbc6)中，我们将这个解决方案与使用 Azure Event Grid 进行了对比。

不需要外部模式管理系统。该模式构建在 Iceberg 表中。该模式也是开放式的，允许您根据自己的需要丰富模式。你不局限于卡夫卡的时间戳、键和值。一个额外的好处是客户端不需要读取所有的列。流读取与批读取一样利用 Parquet 列格式。这可能会减少流式作业所需的 I/O。