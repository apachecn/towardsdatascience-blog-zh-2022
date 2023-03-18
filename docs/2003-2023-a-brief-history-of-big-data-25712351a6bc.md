# 2003–2023:大数据简史

> 原文：<https://towardsdatascience.com/2003-2023-a-brief-history-of-big-data-25712351a6bc>

## 总结 Hadoop 20 年的历史以及相关的一切

![](img/692ba92c99692187609069fa58ddd6c0.png)

每当我进入一个 RPG 电子游戏的图书馆，我都会忍不住看看每个书架，以便更好地了解游戏的宇宙。有人记得《上古卷轴》里的“[帝国简史](https://en.uesp.net/wiki/Morrowind:Brief_History_of_the_Empire)”吗？

大数据，尤其是 Hadoop 生态系统，诞生于 15 年多一点的时间前，并以很少有人能预料到的方式发展。

自其诞生和开源以来，Hadoop 已经成为存储和操作 Pb 级数据的首选武器。围绕它形成了一个包含数百个项目的广泛而充满活力的生态系统，许多大公司仍在使用它，即使其他几个基于云的专有解决方案正在与之竞争。通过这篇文章，我旨在快速追溯 Hadoop 生态系统这 15 年的发展历程，解释它在过去十年中是如何成长和成熟的，以及大数据生态系统在过去几年中是如何不断发展的。

所以，系好安全带，开始 20 年的时间旅行，我们的故事从 2003 年开始，在旧金山南部的一个小镇...

*免责声明:我最初的计划是用提到的公司和软件的徽标来说明这篇文章，但是在 TDS 上广泛使用徽标是被禁止的，我决定用随机的图像和无用的琐事来保持娱乐性。努力回忆当时我们在哪里，做了什么是很有趣的。*

# 2003 年至 2006 年:开始

![](img/ec07b7cbe9d6aec49ee568eb3fe518ed.png)

[始于 2003 年](https://www.computerhope.com/history/2003.htm) : iTunes，Android，Steam，Skype，Tesla。[2004 年开始](https://www.computerhope.com/history/2004.htm) : Thefacebook，Gmail，Ubuntu，魔兽世界。[始于 2005 年](https://www.computerhope.com/history/2005.htm) : Youtube，Reddit。[始于 2006 年](https://www.computerhope.com/history/2006.htm) : Twitter，蓝光。Waze，遗忘。(图片由[罗伯特·安德森](https://unsplash.com/@robanderson72?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄)

这一切都始于千禧年之初，当时山景城一家名为 **Google** 的已经不算小的初创公司正试图索引整个已经不算小的互联网。他们必须面对两大挑战，但这两大挑战都没有得到解决:

> 如何在数千个磁盘上存储数百 TB 的数据，
> 跨越一千多台机器，而不会出现停机、数据丢失、
> 甚至数据不可用的情况？
> 
> 如何以一种高效且有弹性的方式并行计算，让
> 在所有这些机器上处理所有这些数据？

为了更好地理解为什么这是一个困难的问题，考虑当您有一个有一千台机器的集群时，总有*平均至少有*一台机器停机。

从 2003 年到 2006 年，谷歌发布了三篇研究论文，解释了他们的内部数据架构，这将永远改变大数据行业。第一篇论文发表于 2003 年，题目是“[谷歌文件系统](https://dl.acm.org/doi/abs/10.1145/945445.945450)”。第二篇论文发表于 2004 年，标题为“ [MapReduce:大型集群上的简化数据处理](https://dl.acm.org/doi/abs/10.1145/1327452.1327492)”，据谷歌学术称，从那以后，该论文被引用了 21 000 多次。第三个版本于 2006 年发布，标题为“ [Bigtable:一个用于结构化数据的分布式存储系统](https://dl.acm.org/doi/abs/10.1145/1365815.1365816)”。即使这些论文对 Hadoop 的诞生至关重要，谷歌也没有参与其诞生，因为他们保留了自己的源代码。然而，这个故事背后的故事非常有趣，如果你没有听说过杰夫·迪恩和桑杰·格玛瓦特，那么你绝对应该读一读《纽约客》的这篇文章。

与此同时，Hadoop 之父，雅虎！员工名叫*道格·卡丁*，他已经是 [Apache Lucene](https://lucene.apache.org/) (位于 [Apache Solr](https://solr.apache.org/) 和 [ElasticSearch](https://www.elastic.co/elasticsearch/) 核心的搜索引擎库)的创建者，正在从事一个名为 [Apache Nutch](https://nutch.apache.org/) 的高度分布式网络爬虫项目。像谷歌一样，这个项目需要分布式存储和计算能力来实现大规模。在阅读了谷歌关于谷歌文件系统和 MapReduce 的论文后，Doug Cutting 意识到他目前的方法是错误的，并从谷歌的架构中获得灵感，于 2005 年为 Nutch 创建了一个新的子项目，[，他以他儿子的玩具](https://www.cnbc.com/id/100769719)(一只黄色的大象)命名:[***Hadoop***](https://hadoop.apache.org/)。这个项目从两个关键组件开始:Hadoop 分布式文件系统(*)和一个 ***MapReduce*** 框架的实现。不像谷歌，雅虎！决定将该项目作为 Apache 软件基金会的一部分进行开源，从而邀请所有其他主要的技术公司使用并为该项目做出贡献，并帮助他们缩小与邻居的技术差距(雅虎位于山景城旁边的桑尼维尔)。正如我们将看到的，接下来的几年超出了预期。当然，谷歌也做得很好。*

# *2007–2008:Hadoop 的早期采用者和贡献者*

*![](img/fdc66e1348602b4f9164bf5eb36ab7b9.png)*

*[始于 2007 年](https://www.computerhope.com/history/2007.htm) : iPhone、Fitbit、传送门、质量效应、生化奇兵、巫师。[始于 2008 年](https://www.computerhope.com/history/2008.htm):苹果应用商店、安卓市场、Dropbox、Airbnb、Spotify、谷歌 Chrome。(照片由[莱昂纳多·拉莫斯](https://unsplash.com/es/@leonardoeron?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄)*

*很快，面临类似容量分析问题的其他公司开始使用 Hadoop。在过去，这意味着巨大的承诺，因为他们必须自己安装和管理集群，并且编写 MapReduce 作业不是在公园散步(相信我)。雅虎！为了降低编写 MapReduce 作业的复杂性，微软推出了一款名为 [***的 Apache Pig***](https://pig.apache.org/) 的 ETL 工具，它能够将自己的语言 Pig Latin 翻译成 MapReduce 步骤。但是很快其他人也开始为这个新的生态系统做贡献。*

*2007 年，一家名为**脸书**的年轻但发展迅速的公司，由 23 岁的马克·扎克伯格领导，在 Apache 许可下开源了两个新项目:[***Apache Hive***](https://hive.apache.org/)，以及一年后的[***Apache Cassandra***](https://cassandra.apache.org/_/index.html)。Apache Hive 是一个框架，能够将 SQL 查询转换为 Hadoop 上的 Map-Reduce 作业，而 Cassandra 是一个宽列存储，旨在以分布式方式大规模访问和更新内容。Cassandra 并不需要 Hadoop 来运行，但随着 MapReduce 连接器的创建，它迅速成为了 Hadoop 生态系统的一部分。*

*与此同时，一家名为 **Powerset** 的不太知名的公司正在开发一个搜索引擎，他们从谷歌的 Bigtable paper 中获得灵感，开发了[***Apache h base***](https://hbase.apache.org/)，这是另一个依靠 HDFS 进行存储的宽列商店。Powerset 很快被**微软**收购，来自举一个新项目叫做[***Bing***](https://www.bing.com/)***。****

*最后但同样重要的是，另一家公司在 Hadoop 的快速采用中发挥了决定性作用: **Amazon** 。通过启动第一个按需云 ***亚马逊网络服务*** ，并通过 ***弹性 MapReduce*** 服务快速添加对 MapReduce 的支持，亚马逊允许初创公司轻松地在 s3(亚马逊的分布式文件系统)上存储他们的数据，并在其上部署和运行 MapReduce 作业，而没有管理 Hadoop 集群的麻烦。*

# *2008–2012:Hadoop 供应商的崛起*

*![](img/d728f97fdaefcf1f0fe552c692363266.png)*

*[2009 年开始](https://www.computerhope.com/history/2009.htm):比特币、Whatsapp、Kickstarter、优步、USB 3.0。[2010 年开始](https://www.computerhope.com/history/2010.htm) : iPad，Kindle，Instagram。[2011 年开始](https://www.computerhope.com/history/2011.htm) : Stripe，Twitch，Docker，《我的世界》，Skyrim，Chromebook。(照片由[斯潘塞·戴维斯](https://unsplash.com/@spencerdavis?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄)*

*使用 Hadoop 的主要难点是设置、监控和维护 Hadoop 集群需要大量的工作。很快，第一家 Hadoop 厂商 **Cloudera** 于 2008 年成立，Hadoop 之父 Doug Cutting 很快加入其中。Cloudera 提出了 Hadoop 的预打包发行版，名为 ***CDH*** ，以及集群监控接口***cloud era Manager***，最终使得安装和维护 Hadoop 集群以及 Hive 和 HBase 等配套软件变得非常容易。不久之后，出于同样的目的，Hortonworks 和 MapR 也成立了。2010 年 Datastax 成立时，卡珊德拉也有了自己的供应商。*

*很快，每个人都同意，尽管 Hive 是处理大量 ETL 批处理的很好的 SQL 工具，但它不适合交互式分析和 BI。任何习惯于标准 SQL 数据库的人都希望它们能够在不到几毫秒的时间内扫描一个有 1000 行的表，而 Hive 需要几分钟(这是当你让大象做鼠标的工作时得到的结果)。这是一场新的 SQL 战争开始的时候，这场战争至今仍在肆虐(尽管我们将看到其他人从那时起也进入了舞台)。谷歌又一次间接地对大数据世界产生了巨大影响，它在 2010 年发布了第四篇研究论文，名为“ [Dremel:网络规模数据集的交互式分析](https://research.google/pubs/pub36632/)”。本文描述了两个主要的创新:一个分布式交互式查询架构，它将启发我们下面将要提到的大多数 interactive SQL 一个面向列的存储格式，它将启发几种新的数据存储格式，如由 Cloudera 和 **Twitter** 联合开发的[***Apache Parquet***](https://parquet.apache.org/)，以及由 Hortonworks 和脸书联合开发的[***Apache ORC***](https://orc.apache.org/)。*

*受 Dremel 的启发，Cloudera 试图解决 Hive 的高延迟问题，并将其与竞争对手区分开来，于 2012 年决定启动一个新的开源 SQL 引擎，用于交互式查询，名为 [***阿帕奇黑斑羚***](https://impala.apache.org/) ***。*** 类似地，MapR 启动了自己的开源交互式 SQL 引擎，名为[***Apache Drill***](https://drill.apache.org/)，而 Hortonworks 决定他们宁愿致力于使 Hive 更快，而不是从头开始创建一个新的引擎，并启动了[***Apache Tez***](https://tez.apache.org/)，这是一种类似于 MapReduce 的*版本 2* ，并对 Hive 进行了调整，以在 Tez 而不是 MapReduce 上执行两个原因可能推动了这一决定:第一，由于比 Cloudera 小，他们缺乏人力来采用与他们相同的方法，第二，他们的大多数客户已经在使用 Hive，他们宁愿让它工作得更快，也不愿切换到另一个 SQL 引擎。正如我们将看到的，很快许多其他分布式 SQL 引擎出现了，并且“*每个人都比 Hive* 快”成为了新的座右铭。*

# *2010–2014:Hadoop 2.0 和火花革命*

*![](img/3bd4cb2dcb55f47222e3caa84586772f.png)*

*[始于 2012 年](https://www.computerhope.com/history/2012.htm) : UHDTV、Pinterest、脸书活跃用户达到 10 亿，加南风格视频在 Youtube 上的浏览量达到 10 亿。[2013 年开始](https://www.computerhope.com/history/2013.htm):爱德华·斯诺登泄露 NSA 文件，React，Chromecast，谷歌眼镜，Telegram，Slack。(照片由 [Lisa Yount](https://unsplash.com/@lisaleo?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄)*

*当 Hadoop 正在整合和添加一个新的关键组件时，[](https://hadoop.apache.org/docs/stable/hadoop-yarn/hadoop-yarn-site/YARN.html)*(另一个资源管理器)作为其官方资源管理器，这个角色以前由 MapReduce 笨拙地完成，当开源项目[***Apache Spark***](https://spark.apache.org/)开始以前所未有的速度获得牵引力时，一场小革命开始了。很快就清楚了，Spark 将成为 MapReduce 的一个很好的替代品，因为它有更好的功能，更简单的语法，并且在许多情况下比 MapReduce 快得多，特别是由于它能够在 RAM 中缓存数据。与 MapReduce 相比，唯一的缺点是它最初的不稳定性，这个问题随着项目的成熟而逐渐消失。它还与 Hive 有很好的互操作性，因为 SparkSQL 是基于 Hive 的语法(实际上，他们*一开始借用了* Hive 的 lexer/parser)，这使得从 Hive 迁移到 SparkSQL 相当容易。它在机器学习领域也获得了巨大的牵引力，因为以前在 MapReduce 上编写机器学习算法的尝试，如[***Apache Mahout***](https://mahout.apache.org//)*(现已退休)很快被 Spark 实现超越。为了支持 Spark 的快速增长并从中获利，其创始人于 2013 年创立了 Databricks。从那以后，它的目标是通过提供多种语言(Java、Scala、Python、R、SQL，甚至是 Java)的简单而丰富的 API，让每个人都能大规模地操作数据。NET)和许多数据源和格式(csv、json、parquet、jdbc、avro 等)的本地连接器。).值得注意的一件有趣的事情是，Databricks 采取了与其前辈不同的市场战略:不是提议对 Spark 进行内部部署(Cloudera 和 Hortonworks 很快将其添加到自己的平台上)，Databricks 选择了纯云平台，从 AWS(当时最受欢迎的云)开始，然后是 Azure 和 GCP。九年后，我们可以有把握地说这是一个明智之举。***

**与此同时，其他新兴科技公司开源了处理实时事件的新项目，如 [***阿帕奇卡夫卡***](https://kafka.apache.org/) ，由 **LinkedIn** 制作的分布式消息队列，以及 [***阿帕奇风暴***](https://storm.apache.org/) ，由 **Twitter** 制作的分布式实时计算引擎。两者都是 2011 年开源的。此外，在此期间，亚马逊网络服务变得和以往一样受欢迎和成功:网飞在 2010 年令人难以置信的增长，主要是由亚马逊的云实现的，这本身就说明了这一点。云竞争对手终于开始出现，2010 年 ***微软 Azure*** 开始全面上市，2011 年 ***谷歌云平台*** (GCP)。**

# **2014-2016 年到达 Apex⁴**

**![](img/a627969d66de85b38aaad26385d701d0.png)**

**[2014 年开始](https://www.computerhope.com/history/2014.htm) : Terraform，Gitlab，炉石。2015 年开始:Alphabet，Discord，Visual Studio Code。(照片由[威尔弗里德·桑特](https://unsplash.com/@wsanter?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄)**

**从那时起，属于 Hadoop 生态系统的项目数量持续呈指数级增长。它们中的大多数在 2014 年之前就开始开发，其中一些在那之前就已经开源了。项目的数量开始变得令人困惑，因为我们到达了这样一个点，对于每一个需求，都存在多个软件解决方案。更高级别的项目也开始涌现，如，[***Apache Apex***](https://github.com/apache/apex-core)(现已退役)或[***Apache Beam***](https://github.com/apache/beam)(大多由 Google 推动)，旨在提供一个统一的接口，在各种分布式后端(如 Apache Spark、Apache Flink 或 Google 的数据流)之上处理批处理和流处理。**

**我们还可以提到，由于 Airbnb 和 T2 的 Spotify，我们终于开始看到优秀的开源调度程序出现在市场上。调度器的使用通常与使用它的企业的业务逻辑联系在一起，它也是一个非常自然和简单的软件，至少在开始时是这样。然后你意识到，保持它的简单和便于他人使用是一项非常艰巨的任务。这就是为什么几乎每一家大型科技公司都编写并(有时)开源了自己的软件:Yahoo！s[***Apache oo zie***](https://oozie.apache.org/)，Linkedin 的 [***阿兹卡班***](https://azkaban.github.io/) ，Pinterest 的[***Pinball***](https://github.com/pinterest/pinball)*(现已退役)，还有很多。然而，从来没有一个广泛的共识，其中之一是一个非常好的选择，大多数公司坚持自己的。好在 2015 年前后， **Airbnb** 开源[***Apache air flow***](https://airflow.apache.org/)，而 **Spotify** 开源[***Luigi***](https://github.com/spotify/luigi)⁵，这两个调度器迅速达到了跨其他公司的高采用率。特别是，气流现在可以在[谷歌云平台](https://cloud.google.com/composer/docs/composer-2/run-apache-airflow-dag?hl=en)和[亚马逊网络服务](https://aws.amazon.com/managed-workflows-for-apache-airflow/?nc1=h_ls)上以 SaaS 模式使用。***

**在 SQL 方面，出现了其他几个分布式数据仓库，旨在提供比 Apache Hive 更快的交互式查询功能。我们已经谈到了 Spark-SQL 和 Impala，但我们还应该提到 [***Presto***](https://prestodb.io/) ，2013 年由脸书开源，2016 年已被亚马逊更名为 [***雅典娜***](https://aws.amazon.com/athena) 用于他们的 SaaS 发售，并在他们离开脸书后被其原开发者分叉为[***Trino***](https://trino.io/)。在专有方面，也发布了几个分布式 SQL 分析仓库，如谷歌的[***big query***](https://cloud.google.com/bigquery)，2011 年首次发布，亚马逊的 [***红移***](https://aws.amazon.com/redshift/)*，2012 年创立的 [**雪花**](https://www.snowflake.com/en/) 。***

***要获得作为 Hadoop 生态系统的一部分被引用的所有项目的列表，请查看此页面的[，其中引用了超过 150 个项目](https://hadoopecosystemtable.github.io/)。***

# ***2016–2020 集装箱化和深度学习的兴起，以及 Hadoop 的衰落***

***![](img/99d276d2540a300099fa91a5bdac504b.png)***

***[2016 年开始](https://www.computerhope.com/history/2016.htm):奥库斯裂谷，Airpods，Tiktok。[2017 年开始](https://www.computerhope.com/history/2017.htm):微软团队，堡垒之夜。[2018 年开始](https://www.computerhope.com/history/2018.htm) : GDPR，剑桥分析公司丑闻，在我们中间。[2019 年开始](https://www.computerhope.com/history/2019.htm):迪士尼+、三星 Galaxy Fold、谷歌 Stadia(图片由 [Jan Canty](https://unsplash.com/@jancanty?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄)***

***接下来的几年，一切都保持着加速和互联。跟上大数据市场中的新技术和公司变得越来越困难，因此长话短说，我将谈谈我认为对大数据生态系统影响最大的四个趋势。***

***第一个趋势是数据基础设施向云的大规模迁移，HDFS 被云存储取代，如*亚马逊 S3* 、*谷歌存储*或 *Azure Blob 存储*。***

**第二个趋势是集装箱化。你可能已经听说过 Docker 和 Kubernetes。 [***Docker***](https://www.docker.com/) 是 2011 年推出的一个容器化框架，从 2013 年开始迅速流行起来。2014 年 6 月，Google 开源了其内部容器编排系统[***Kubernetes***](https://kubernetes.io/)***(***又名***【K8s】***)，该系统立即被许多公司采用，以构建其新的分布式/可扩展架构的基础。Docker 和 Kubernetes 允许公司为包括基于事件的实时转换在内的许多用例部署新型的分布式架构，更加稳定和可伸缩。Hadoop 花了一些时间赶上 docker，因为在 2018 年 3.0 版本中支持在 Hadoop 中启动 Docker 容器。**

**如前所述，第三个趋势是用于分析的完全托管的大规模并行 SQL 数据仓库的兴起。“现代数据栈”和 2016 年首次开源的 dbt 的崛起很好地说明了这一点。**

**最后，影响 Hadoop 的第四个趋势是深度学习的出现。在 2010 年的后半年，每个人都听说过深度学习和人工智能: **AlphaGo** 在围棋比赛中击败了世界冠军柯洁，这是一个里程碑，就像 20 年前 IBM 的深蓝与卡斯帕罗夫在国际象棋中的表现一样。这一技术飞跃已经实现了奇迹，并带来了更多希望，就像自动驾驶汽车一样，它经常与大数据联系在一起，因为它需要处理大量信息才能训练自己。然而，Hadoop 和机器学习是两个非常不同的世界，他们很难一起工作。事实上，深度学习推动了对大数据新方法的需求，并证明了 Hadoop 并不是万能的工具。**

**长话短说:从事深度学习的数据科学家需要两件 Hadoop 当时无法提供的东西。他们需要 GPU，这是 Hadoop 集群节点通常没有的，他们需要安装最新版本的深度学习库，如 ***Tensorflow*** 或 ***Keras*** ，这在整个集群上很难做到，特别是当多个用户要求同一库的不同版本时。Docker 很好地解决了这个问题，但是 Docker 对 Hadoop 的集成花了很长时间才变得可用，数据科学家现在就需要它。因此，与使用集群相比，他们通常更喜欢使用 8 个 GPU 来生成一个虚拟机。**

**这就是为什么当 Cloudera 在 2017 年首次公开募股时，他们已经将开发和营销重点放在了他们的最新软件 [***数据科学工作台***](https://www.cloudera.com/products/data-science-and-engineering/data-science-workbench.html) 上，该软件不是基于 Hadoop 或 YARN，而是基于 Docker 和 Kubernetes 的容器化，并允许数据科学家将他们的模型作为容器化的应用程序部署到他们自己的环境中，而没有安全或稳定性问题的风险。**

**这不足以阻止他们的衰落。2018 年 10 月，Hortonworks 和 Cloudera 合并，仅保留 Cloudera 品牌。2019 年，MapR 被惠普企业(HPE)收购。2021 年 10 月，一家名为 CD&R 的私人投资公司以低于最初价格的股价收购了 Cloudera。**

**然而，Hadoop 的衰落并不意味着它的死亡，因为许多大公司仍在使用它，尤其是在内部部署中，并且围绕它构建的所有技术都在继续使用它，或者至少是它的一部分。创新也仍在进行。例如，新的存储格式是开源的，如最初在 2016 年**优步**开发的 [***阿帕奇胡迪***](https://hudi.apache.org/)*；2017 年**网飞**开始的 [***三角洲湖***](https://delta.io/)***[有趣的是，这些新文件格式背后的主要目标之一是规避我提到的第一种趋势的后果:Hive 和 Spark 最初是为 HDFS 构建的，HDFS 保证的一些性能属性在迁移到 S3 这样的云存储中丢失了，这导致了效率低下](https://delta.io/)。但是我不会在这里深入讨论细节，因为这个特定的主题需要另一篇完整的文章。******

# **2020–2023 现代**

**![](img/ac5d9d26a725834458fb346584417635.png)**

**[2020 年开始](https://www.computerhope.com/history/2020.htm):新冠肺炎疫情。[2021 年开始](https://www.computerhope.com/history/2021.htm) : Log4Shell 漏洞，Meta，Dall-e .[2022 年开始](https://www.computerhope.com/history/2022.htm):乌克兰战争，中途，稳定扩散。(照片由[乔纳森·罗杰](https://unsplash.com/@jonathanroger?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄)**

**如今，Hadoop 在云中的部署大多被 Apache Spark 或 Apache Beam⁶应用程序(大多在 GCP 上)所取代，从而有利于 *Databricks* 、*亚马逊的 Elastic Map Reduce (EMR)* 、*谷歌*、 Dataproc/Dataflow 或 *Azure Synapse。*我也看到许多年轻公司直接瞄准“现代数据堆栈”方法，围绕 SQL 分析仓库构建，如 *BigQuery、Databricks-SQL、Athena 或 Snowflake，*由无代码(或低代码)数据摄取工具提供，并使用 dbt 进行组织，[这些工具似乎根本不需要 Spark 等分布式计算工具](/modern-data-stack-which-place-for-spark-8e10365a8772)。当然，仍然倾向于本地部署的公司仍然在使用 Hadoop 和其他开源项目，如 Spark 和 Presto，但移动到云的数据比例每年都在增加，我认为目前没有理由改变这一点。**

**随着数据行业的不断成熟，我们也看到了更多的元数据管理和目录工具被构建和采用。在那个范围内，我们可以提到 2015 年由 Hortonworks 开始的[***Apache Atlas***](https://atlas.apache.org/#/)、2020 年由[***Amundsen***](https://www.amundsen.io/)、2019 年由 **Lyft** 开源、2020 年由[***data hub***](https://datahubproject.io/docs/)、Linkedin 开源。许多民营科技创业公司也出现在这个领域。**

**我们也看到了围绕新的调度器技术建立的创业公司，如 [***提督***](https://www.prefect.io/)[***达格斯特***](https://dagster.io/) 和[***Flyte***](https://flyte.org/)，它们的开源库分别于 2017 年、2018 年和 2019 年开始，它们正在挑战 Airflow 目前的霸权。**

**最后，*湖畔小屋*的概念已经开始浮现。湖库是一个结合了数据湖和数据 warehouse⁷.优势的平台这允许数据科学家和 BI 用户在同一个数据平台内工作，从而使治理、安全性和知识共享变得更加容易。由于 Spark 在 SQL 和 DataFrames 之间的多功能性，Databricks 是第一个创造该术语并将其定位于该产品的公司。紧随其后的是雪花与[](https://docs.snowflake.com/en/developer-guide/snowpark/index.html)*[*Azure Synapse*](https://learn.microsoft.com/en-us/azure/architecture/example-scenario/analytics/secure-data-lakehouse-synapse)以及最近的谷歌与 [*BigLake*](https://cloud.google.com/biglake) *。*开源方面，[***dre mio***](https://www.dremio.com/)*提供了 2017 年以来的 lakehouse 架构。****

# ***谁能说出未来会是什么样子？***

***![](img/ebb8768e1181047257974927abfc731f.png)***

***2023 年开始:谁知道呢？(安妮·斯普拉特在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄)***

***自从这一切开始以来，大数据世界中的开源项目和创业公司的数量逐年增加(只要看一看 [2021 年的前景](https://mattturck.com/data2021/)就知道它已经变得多么巨大)。我记得在 2012 年左右，有人预测新的 SQL 战争将会结束，真正的胜利者将最终出现。这还没有发生。所有这些在未来将如何发展很难预测。尘埃落定还需要几年时间。但如果让我做一些王尔德的猜测，我会做出以下预测。***

1.  ***[正如其他人已经注意到的](https://thesequel.substack.com/p/snowbricks-and-dataflake)，现有的主要数据平台(Databricks、Snowflake、BigQuery、Azure Synapse)将继续改进并添加新功能，以缩小彼此之间的差距。我希望看到每个组件之间越来越多的连接，以及像 SQL 和 Python 这样的数据语言之间的连接。***
2.  ***未来几年，我们可能会看到新项目和公司的数量放缓，尽管这更多是因为新的互联网泡沫破裂后缺乏资金(如果这种情况真的发生的话)，而不是缺乏意愿或想法。***
3.  ***从一开始，主要缺乏的资源就是熟练劳动力。这意味着，对大多数 companies⁸来说，投入更多资金解决性能问题，或者迁移到更具成本效益的解决方案，比花更多时间优化它们更简单。尤其是现在主要分布式仓库的存储成本变得如此低廉。但也许在某个时候，供应商之间的价格竞争将变得更加难以维持，价格将会上涨。即使价格不上涨，企业存储的数据量也在逐年增加，[以及与之相关的低效率成本](https://medium.com/@laurengreerbalik/how-fivetran-dbt-actually-fail-3a20083b2506)。也许在某个时候我们会看到一个新的趋势，人们开始寻找新的、更便宜的开源替代品，一个新的类似 Hadoop 的循环将再次开始。***
4.  ***从长远来看，我认为真正的赢家将是云提供商，谷歌，亚马逊和微软。他们所要做的就是等待和观察风向，等待时机，然后获得(或简单地复制)最有效的技术。集成到他们的云中的每个工具都让用户的工作变得更加简单和无缝，尤其是在安全性、治理、访问控制和成本管理方面。只要他们不犯重大的组织错误，我看现在没人能赶上他们。***

# ***结论***

***我希望你喜欢这次和我一起的回忆之旅，它能帮助你更好地理解(或者简单地回忆)这一切是从哪里以及如何开始的。我试图让这篇文章便于所有人理解，包括非技术人员，所以请不要犹豫，与您的同事分享，他们有兴趣了解大数据来自哪里。***

***最后，我想强调的是，如果没有开源和知识共享的神奇力量，人类在人工智能和大数据方面的知识和技术永远不会发展得如此之快。我们应该感谢最初通过学术研究论文分享知识的谷歌，我们应该感谢所有开源项目的公司。在过去的 20 年里，开源和免费(或至少廉价)获取技术一直是互联网经济创新的最大驱动力。20 世纪 80 年代，一旦人们买得起家用电脑，软件创新就真正开始了。3D 打印也是如此，[它已经存在了几十年，在 21 世纪初随着自我复制机器的出现而腾飞](https://all3dp.com/2/history-of-3d-printing-when-was-3d-printing-invented/)，或者是推动 DYI 运动的树莓派的出现。***

***开源和轻松获取知识应该永远被鼓励和争取，甚至比现在更多。这是一场永无止境的战斗。一场这样的战斗，也许是最重要的一场，[这些天正在与 AI](/bloom-is-the-most-important-ai-model-of-the-decade-97f0f861e29f) 进行较量。大公司确实为开源做出了贡献(例如谷歌和 TensorFlow)，但[他们也学会了如何使用开源软件如 venus flytraps](https://ghuntley.com/fracture/) 来吸引用户进入他们的专有生态系统，同时保留专利背后最关键的(也是最难复制的)功能。***

***我们继续尽最大努力支持开源和知识共享(如维基百科)对人类和世界经济至关重要。政府、公民、公司以及最重要的投资者必须明白这一点:增长可能由创新驱动，但创新是由与大众分享知识和技术驱动的。***

***![](img/7b611c444c45b39a539d0ea91b3c4035.png)***

***“做你必须做的。无论发生什么”(写在巴黎人文小教堂墙上的句子)***

## *****脚注*****

***:如果算上来自谷歌的前传，甚至是 20 年，因此得名。***

***:也许在 2022 年，我们在硬件可靠性方面取得了足够的进步，使这一点不那么真实，但 20 年前肯定是这样。***

***:2016 年， [Twitter 开源 Apache Heron](https://medium.com/@kramasamy/introduction-to-apache-heron-c64f8c7c0956) (好像还在 Apache 孵化阶段)取代 Apache Storm。***

***⁴:一语双关。***

***⁵:2022 年， [Spotify 决定停止使用 Luigi，转而使用 Flyte](https://engineering.atspotify.com/2022/03/why-we-switched-our-data-orchestration-service/)***

***⁶::我怀疑 Apache Beam 主要用在有数据流的 GCP 上。***

***⁷: [正如 Databricks 所说的](https://www.databricks.com/glossary/data-lakehouse)，lakehouse 将数据湖的灵活性、成本效益和规模与数据仓库的数据管理和 ACID 事务结合在一起。***

***⁸:当然，我这里说的不是像优步的网飞那么大的公司。***