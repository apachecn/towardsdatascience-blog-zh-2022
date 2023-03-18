# Gallia 的 Scala 中的数据转换:版本 0.4.0 已经过时

> 原文：<https://towardsdatascience.com/data-transformations-in-scala-with-gallia-version-0-4-0-is-out-f0b8df3e48f3>

## 介绍这个增压版本的新功能

![](img/4a3dc2882598ab760b84772fea5bf967.png)

由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的 [Shubham Dhage](https://unsplash.com/@theshubhamdhage?utm_source=medium&utm_medium=referral) 拍摄的照片

这是我之前对 Gallia 的[介绍的后续文章，Gallia 是 Scala 中的模式感知数据转换库。本文将关注最新版本中包含的最重要的变化:](/gallia-a-library-for-data-transformation-3fafaaa2d8b9) [0.4.0](https://github.com/galliaproject/gallia-core/releases/tag/v0.4.0) (适用于 Scala [2.12](https://search.maven.org/artifact/io.github.galliaproject/gallia-core_2.12/0.4.0/jar) 和 [2.13](https://search.maven.org/artifact/io.github.galliaproject/gallia-core_2.13/0.4.0/jar) )。

***目录***

*   [*读写自阿帕奇 Avro/拼花*](#166d)***–****[*阿帕奇 Avro*](#ff5d)***–****[*阿帕奇拼花*](#beec)**
*   **[*从案例类中读取*](#d0c0)**
*   **[*活接头类型*](#102c)**
*   **[*元模式*](#d600)**
*   **[*复杂聚合*](#a75a)**
*   **[*复杂转换/共转换*](#e3eb)***–****[*通过数据类转换(用于嵌套实体)*](#d204)***–****[*通过数据类共转换(用于当前级别)*](#9a3b)****
*   **[*新支持的类型*](#069f)***–****[*枚举*](#4987)***–****[*二进制数据*](#8a0a)****
*   ***[*结论*](#e9d4)***

# **从 Apache Avro/Parquet 读取/写入**

**回想一下上一篇文章，Gallia 中的典型处理过程如下:**

**它产生:**

```
**<root>
 title            _String
 air_date         _String
 doctor           _Inttitle                   | air_date         | doctor
----------------------- | ---------------- | ------
THE ELEVENTH HOUR       | 3 April 2010     | 11    
THE DOCTOR'S WIFE       | 14 May 2011      | 11 
...**
```

****注:**一个 *TSV* 版本的剧集数据集可作为要点[在此](https://gist.github.com/anthony-cros/3db86cca82cf27d0ea42a724bd78555c)。**

## **阿帕奇 Avro**

**类似地，Avro 文件现在可以通过以下方式使用:**

**这产生了完全相同的结果。**

**相反，可以用`.writeAvro(“./mydata.avro”)`将结果写入 Avro 文件**

****备注:****

*   **原始文件本身是否是 Avro 文件并不重要，因为 Gallia 中的输入和输出是完全独立的**
*   **上述说明的一个直接后果是，人们可以纯粹出于转换目的使用 Gallia:*Avro->JSON，Avro- > TSV，Avro- > Parquet，Parquet- > Avro* 等。但是有一点需要注意:数据必须符合 Gallia 的[数据模型](https://github.com/galliaproject/gallia-docs/blob/master/modeling.md)(例如，不使用 Avro 的[映射](https://avro.apache.org/docs/1.10.2/spec.html#Maps)，不使用 2+D 数组等)。**
*   **将来，`streamAvro/writeAvro`方法将被通用的`stream/write`方法所取代，扩展作为参考(参见 [I/O](https://github.com/galliaproject/gallia-core/blob/v0.4.0/README.md#io) 部分)**

## **阿帕奇拼花地板**

**要处理拼花文件而不是 Avro，代码完全相同，除了:**

*   **模块名称是`gallia-**parquet**`而不是`gallia-avro`**
*   **进口是`import gallia.**parquet**._`而不是`import gallia.avro._`**
*   **方法名是`stream**Parquet**/write**Parquet**`而不是`streamAvro/writeAvro`**

****备注:****

*   **Gallia 使用罩下的 Avro 转换器，即[*AvroParquetReader*](https://github.com/apache/parquet-mr/blob/apache-parquet-1.12.3/parquet-avro/src/main/java/org/apache/parquet/avro/AvroParquetReader.java)和[*AvroParquetWriter*](https://github.com/apache/parquet-mr/blob/apache-parquet-1.12.3/parquet-avro/src/main/java/org/apache/parquet/avro/AvroParquetWriter.java)**
*   **在未来的版本中，我们将改为提供直接处理，通过 Parquet 提供的自定义[*read support*](https://github.com/apache/parquet-mr/blob/apache-parquet-1.12.3/parquet-hadoop/src/main/java/org/apache/parquet/hadoop/api/ReadSupport.java)/[/*write support*](https://github.com/apache/parquet-mr/blob/apache-parquet-1.12.3/parquet-hadoop/src/main/java/org/apache/parquet/hadoop/api/WriteSupport.java)抽象**

# **从案例类中阅读**

**让我们考虑下面的案例类:**

```
 **case class **Person**(**name**: String, **age**: Int, **income**: Int)**
```

**它现在可以作为一个单一的实体被摄取:**

```
 **val peter = gallia.***aobjFromDataClass***(
      **Person**(**name** = "Peter" , **age** = 29, **income** = 80000))**
```

**同时，其集合可以如下摄取:**

**举个例子，如下:**

**会产生预期的:**

```
**<root>
  name             _String
  age              _Int
  hourly_rate      _Doublename   | age | hourly_rate      
------ | --- | ------------------
Peter  | 29  | 38.31417624521073
Joanna | 29  | 14.367816091954023
Samir  | 28  | 38.31417624521073**
```

****注意**:虽然 Gallia 还不允许**将**写入 case 类，但它将在下一个版本中发布——参见 [NEXT_RELEASES](https://github.com/galliaproject/gallia-core/blob/2a5c81f/NEXT_RELEASES.md) (将通过反射和宏允许)**

# **工会类型**

**在 [0.4.0](https://github.com/galliaproject/gallia-core/releases/tag/v0.4.0) 中也增加了对联合类型的部分支持。**

**用法示例如下:**

**这会产生以下输出:**

```
**<root> [... schema unchanged]name  | age
----- | ------------
Peter | 29
Samir | TWENTY-EIGHT**
```

**因为`toUpperCase`是一个纯字符串操作，所以具有整数值的实体 *age* 保持不变。**

**上面的例子给出了一个简单的例子，但是当然还有更复杂的例子。例如，当多个嵌套实体是一个联合的一部分时:**

**它产生以下输出:**

```
**<root> [... schema unchanged]{ "data":           "Michael (27yo)" }
{ "data": { "name": "**PETER**", "age": 29 }}
{ "data":           "Joanna (29 years old)" }
{ "data": { "name": "Samir", "dob": "March 14, 1971" } }**
```

**只有值`"Peter"`是大写的，因为它是唯一具有 *age* 条目的嵌套实体。**

****备注:****

*   **参见 [union_types.md](https://github.com/galliaproject/gallia-docs/blob/v0.4.0/union_types.md)**
*   **更多的例子可以在 [UnionTypeTest.scala](https://github.com/galliaproject/gallia-testing/blob/v0.4.0/src/main/scala/galliatest/suites/UnionTypeTest.scala) 中看到**
*   **Gallia 中的联合类型在这一点上仍然被认为是试验性的，并不是所有的操作都支持它们(但是基本的操作支持它们)。**
*   **支持联合类型的主要原因之一是帮助数据清理/特性工程工作。事实上，在遗留数据集中，用不同类型捕获字段的情况非常普遍(想想`true`和`"Yes"`之类的)**

# **元架构**

**增加对联合类型支持的结果是 Gallia 能够提供自己的[元模式](https://github.com/galliaproject/gallia-core/blob/v0.4.0/src/main/scala/gallia/MetaSchema.scala):**

**这意味着 Gallia 可以自己进行模式转换。例如，下面是嵌套字段重命名的样子(这里从 *f* 到 *F* ):**

****备注:****

*   **Gallia 实际上并没有在内部以这种方式使用元模式**
*   ***模式*是数据的特例，即*元数据*(或“关于数据的数据”)，因此 Gallia 的*元模式*也是*元元数据*。而且既然 Gallia 的 *metaschema* 也可以用来自己建模，那么它也是自己的*meta schema*。因此它也是*元元元数据*。很明显。**

# ****复杂聚合****

**让我们重复使用早先中[的`people`手柄。汇总数据的一个非常简单的方法是:](https://gist.github.com/anthony-cros/2760c3dbb6f5d9d2f8c3cd41edbb0ba6)**

```
**people.**group**("name" ~> "names").**by**("age")
  .printJsonl()**
```

**它产生以下内容:**

```
**{ "age": 29, "names": [ "Peter", "Joanna" ] }
{ "age": 28, "names": [ "Samir" ] }**
```

**也可以(不一定)通过以下方式实现:**

```
**people.**aggregateBy**("age").as("names")
    .**using** { _.strings("name") }
  .printJsonl()**
```

**这产生了与上面相同的结果，但是显示了对`aggregateBy`的简单使用。**

**虽然 Gallia 中有更多的内置聚合器可用( *SUM BY* ， *MEAN BY* ，…)，但是要在单个操作中实现这样的处理，就需要使用`aggregateBy`构造:**

```
**people.aggregateBy("age").as("names")
    .using { _.strings("name").**concatenateStrings** }
  .printJsonl()**
```

**它产生:**

```
**{ "age": 29, "names": "**PeterJoanna**" }
{ "age": 28, "names": "**Samir**" }**
```

**实际上是:**

```
**people.aggregateBy("age").as("names")
    .using { _.strings("name")
      .**mapV** { _.reduceLeft(_ + _) } }
  .printJsonl()**
```

**因此，它可以根据需要进行定制，例如:**

```
**people.aggregateBy("age").as("names")
    .using { _.strings("name")
      .mapV { _.reduceLeft(
        **_.toUpperCase + "|" + _.toUpperCase**) } }
  .printJsonl()**
```

**生产:**

```
**{ "age": 29, "names": "**PETER|JOANNA**" }
{ "age": 28, "names": "**SAMIR**" }**
```

**但是,`aggregateBy`构造的真正威力在于能够实现以下更具定制性的聚合类型:**

```
**people.aggregateBy("age").using { **group** =>
    **(**"names"       **->** **group**.strings("name")**,**
     "mean_income" **->** **group**.ints   ("income").mean**)** }
  .printJsonl()**
```

**这导致:**

```
**{"age": 29, "names": [ "Peter", "Joanna" ], "mean_income": 55000.0}
{"age": 28, "names": [ "Samir" ],           "mean_income": 80000.0}**
```

****注意**:上面使用的基于元组的实体创建只是更显式的`gallia.headO("names" -> ...)`的简写，最多可用于 5 个条目**

# ****复杂的转换/协同转换****

**让我们切换到一个更严肃的领域来突出这些新特性。考虑以下数据集:**

****注意**:提醒一下`bobj`和`bobjs`构造是一种方便的机制，允许构造那些模式很容易推断的实体。因此`bobj("f" -> 1)`相当于更显式的`aobj("f".int)(obj("f" -> 1))`。**

## ****通过数据类的转换(用于嵌套实体)****

**Gallia 现在提供了通过 case 类(“数据类”)转换嵌套实体的能力。例如，考虑:**

```
**case class **Change**(
   **chromosome**: String,
   **position**  : Int   ,
   **from**      : String,
   **to**        : String) {

  def **shorthand**: String=
    s"${**chromosome**}:${**position**};${**from**}>${**to**}"

}**
```

**它对上面的`mutations`数据集中的*变化*嵌套实体进行建模，并封装一个操作，该操作为遗传变化产生一个简写符号(例如`"3:14532127;C>GG"`)。**

**下面的代码将把*变更*实体转换成它的速记副本:**

```
**mutations
  .transformDataClass[**Change**]("**change**")
    .using(_.**shorthand**)
  .display()**
```

**它产生:**

```
**[...]patient_id | age | change
---------- | --- | ----------------
p1         | 23  | 3:14532127;C>GG
p2         | 22  | 4:1554138;C>T
p3         | 21  | Y:16552149;AA>GT**
```

**注意事项:**

*   **这将通过`.transformDataClass[**Option**[Change]]`、 `.transformDataClass[**Seq**[Change]]`、和 `.transformDataClass[**Option**[**Seq**[Change]]`应用于嵌套实体的可选/必需和单个/多个的任何其他组合**
*   **Gallia 的后续版本将利用宏使这种机制更加有效(目前依赖于反射)**

## ****通过数据类的协同转换(针对当前级别)****

**现在让我们考虑下面的 case 类，它模拟了当前级别**的字段子集**(这次与嵌套实体相反):**

```
**import java.time.Yearcase class **Demographics**(
    **age**: Int,
    **sex**: String) {

  deftoNewDemographics =
    **NewDemographics**(
      **year_of_birth**    = Year.*now*().getValue - age,
      **has_Y_chromosome** = sex == "male")

}**
```

**以及对期望的模型变化建模的以下 case 类:**

```
**case class **NewDemographics**(
    **year_of_birth**   : Int,
    **has_Y_chromosome**: Boolean)**
```

**以下代码通过使用 origin case 类中的封装方法来共同转换这两个字段:**

```
**mutations
  .cotransformViaDataClass[**Demographics**]
    .usingWithErasing(_.**toNewDemographics**)
  .display()**
```

**这导致了预期的结果:**

```
**<root>
 patient_id       _String
 year_of_birth    **_Int**
 has_Y_chromosome **_Boolean**
 change ...{ "patient_id"      : "p1",
  "change"          : { ... },
  "year_of_birth"   : **1999**,
  "has_Y_chromosome": **true** }
...**
```

****备注:****

*   **按照现在的情况，分别转换年龄和性别更有意义，但是这里的目标是强调一个联合转换，其中的字段可以任意混合和匹配**
*   **`.usingWithErasing`删除原始条目，而`.using`会保留它们**

## ****通过定制处理进行联合转换****

**Gallia 现在还为真正的定制处理提供了改进的机制，尽管这通常不是一个好主意(因为我们失去了模式/数据将相应地共同发展的保证)。例如，为了重现上面的共同转换，我们可以创建下面的对象，扩展`gallia.ObjToObj`:**

**下面的代码将利用它:**

```
**mutations
  .custom(**CustomDemographicsTransformation**)
  .display()**
```

**这将产生相同的输出。**

# ****新支持的类型****

**Gallia 现在支持其他类型:**

*   **枚举**
*   **二进制数据**
*   **时间类型**

## **枚举**

**枚举可以这样创建/使用:**

**它产生:**

```
**<root>
  choice           _Enm(List(rock, paper, scissors, **spock**))choice
------
**paper****
```

## **二进制数据**

**可以像这样创建/使用二进制数据:**

**它产生:**

```
**bin
-----------
base64:**Ym9v****
```

****备注:****

*   **`"Ym9v"`将`"boo"`编码到 *Base64* 中，因为我们在`'f'`字节(`0x66`)的地方放了一个`'b'`字节(`0x62`)。**
*   ***Base64* 输出仅用于序列化，内存中的表示在整个处理过程中保持不变**

## **时间类型**

**Gallia 中支持的时态类型与它们的 Java 对应物相匹配:`java.time.{LocalDate, LocalTime, LocalDateTime, OffsetDateTime, ZonedDateTime, Instant}`**

**它们可以这样创建/使用(例如使用`LocalDateTime`):**

**它产生:**

```
**<root>
   hungover         _LocalDateTime**hungover**
-------------------
**2022-01-01**T00:00:00**
```

# **结论**

**这就结束了我们对 [0.4.0](https://github.com/galliaproject/gallia-core/releases/tag/v0.4.0) 带来的主要变化的浏览，至少是那些将改善客户端代码体验的变化。**

**其他值得注意的新增内容包括:**

*   **与 **Python** 生态系统的实验性整合:
    –**熊猫**:参见 [ScalaPyPandasTest.scala](https://github.com/galliaproject/gallia-python-integration/blob/v0.4.0/gallia-pandas/src/main/scala/gallia/pandas/ScalaPyPandasTest.scala#L10) 和[gallapandas test . Scala](https://github.com/galliaproject/gallia-python-integration/blob/v0.4.0/gallia-python-viz/src/main/scala/gallia/pyviz/GalliaPandasTests.scala#L29)–**Seaborn**:参见 [GalliaVizTest.scala](https://github.com/galliaproject/gallia-python-integration/blob/v0.4.0/gallia-python-viz/src/main/scala/gallia/pyviz/GalliaVizTest.scala#L37-L40)**
*   **对**密集数据**进行实验性内存优化，捕获为`(size: Int, data: Array[Any])`而不是`Array((Symbol, Any))`，并进行一些相应的操作。参见 [Obg9.scala](https://github.com/galliaproject/gallia-core/blob/v0.4.0/src/main/scala/gallia/obg9/Obg9.scala) 和 [Obg9Test.scala](https://github.com/galliaproject/gallia-testing/blob/v0.4.0/src/main/scala/galliatest/suites/Obg9Test.scala) 的示例用法**

**有关此版本带来的变更和改进的更完整列表，请参见 [CHANGELOG.md](https://github.com/galliaproject/gallia-core/blob/2a5c81f/CHANGELOG.md) 。**

**随时欢迎反馈！**