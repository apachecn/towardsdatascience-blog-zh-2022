# 用于处理大型数据集的熊猫的 8 种替代方案

> 原文：<https://towardsdatascience.com/8-alternatives-to-pandas-for-processing-large-datasets-928fc927b08c>

## 停止使用熊猫

![](img/5a89ec1c4f6092425f1636d1eb3eb989.png)

埃里克·麦克林在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

Pandas 库已经成为 python 中数据操作的事实库，并被数据科学家和分析师广泛使用。然而，有时数据集太大，熊猫可能会遇到内存错误。以下是熊猫处理大型数据集的 8 个替代方案。对于每个备选库，我们将研究如何从 CSV 加载数据并执行一个简单的`groupby`操作。幸运的是，这些库中有许多与 Pandas 相似的语法，因此学习曲线不会太陡。

# 达斯克

> [*Dask*](https://docs.dask.org/) *在大于内存的数据集上提供多核分布式并行执行。*

Dask 数据帧是一个大型并行数据帧，由许多较小的 Pandas 数据帧组成，沿索引拆分。这些 Pandas 数据帧可能存在于磁盘上，用于单台机器或集群中许多不同机器上的大内存计算。一个 Dask 数据帧操作触发对组成 Pandas 数据帧的许多操作。

```
pip install daskimport dask.dataframe as dd
df = dd.read_csv('../input/yellow-new-york-taxi/yellow_tripdata_2009-01.csv', dtype={'Tolls_Amt': 'float64'})
df2 = df.groupby('vendor_name').agg({'Passenger_Count':'mean'})
```

到目前为止，还没有执行任何计算。计算`df2`的结果

```
df2.compute()
```

Dask 不仅仅是一个数据操作库。 [Dask-ML](https://ml.dask.org/) 提供可扩展的机器学习，可以与 Scikit-learn、Xgboost 和 LightBGM 等流行的机器学习库一起使用。

# 摩丁

> [*Modin*](https://modin.readthedocs.io/en/stable/index.html) *使用 Ray 或 Dask 提供一种毫不费力的方式来加速你的熊猫笔记本、脚本和库。*

```
pip install modinimport modin.pandas as pd
df = pd.read_csv('../input/yellow-new-york-taxi/yellow_tripdata_2009-01.csv')
df2 = df.groupby('vendor_name').agg({'Passenger_Count':'mean'})
```

# 数据表

> [*Datatable*](https://datatable.readthedocs.io/en/latest/start/quick-start.html)*是一个用于操作表格数据的 python 库。它支持内存不足的数据集、多线程数据处理和灵活的 API。*

```
pip install datatablefrom datatable import dt, f, by
df = dt.fread('../input/yellow-new-york-taxi/yellow_tripdata_2009-01.csv', skip_blank_lines = True)
df2 = df[:, dt.mean(f.Passenger_Count), by('vendor_name')]
```

# 极地

> [*Polars*](https://pola-rs.github.io/polars-book/user-guide/index.html) *是 Rust 中实现的一个速度惊人的数据帧库，使用 Apache Arrow Columnar 格式作为内存模型。*

```
pip install polarsimport polars as pldf = pl.read_csv('../input/yellow-new-york-taxi/yellow_tripdata_2009-01.csv')
df2 = df.groupby('vendor_name').agg([pl.mean('Passenger_Count')])
```

# Vaex

> [*Vaex*](https://vaex.io/docs/index.html) *是一个用于懒惰的核外数据框架(类似于熊猫)的 python 库，用于可视化和探索大型表格数据集*

```
pip install vaexdf = vaex.read_csv('../input/yellow-new-york-taxi/yellow_tripdata_2009-01.csv')
df2 = df.groupby('vendor_name').agg({'Passenger_Count':'mean'})
```

# Pyspark

> [*Pyspark*](https://spark.apache.org/docs/latest/api/python/index.html#) *是 Apache Spark 的 Python API，用于通过分布式计算处理大型数据集。*

```
pip install pysparkfrom pyspark.sql import SparkSession, functions as f
spark = SparkSession.builder.appName("SimpleApp").getOrCreate()df = spark.read.option('header', True).csv('../input/yellow-new-york-taxi/yellow_tripdata_2009-01.csv')
df2 = df.groupby('vendor_name').agg(f.mean('Passenger_Count'))
```

到目前为止，还没有执行任何计算。计算`df2`的结果

```
df2.show()
```

# 树袋熊

> *[*考拉*](https://koalas.readthedocs.io/en/latest/index.html) *项目通过在 Apache Spark 之上实现 pandas DataFrame API，使数据科学家在与大数据交互时更有效率。**

*由于考拉运行在 Apache Spark 之上，Spark 也必须安装。*

```
*pip install pyspark
pip install koalasimport databricks.koalas as ks
from pyspark.sql import SparkSession
df = ks.read_csv('../input/yellow-new-york-taxi/yellow_tripdata_2009-01.csv')
df2 = df.groupby('vendor_name').agg({'Passenger_Count':'mean'})*
```

# *cuDF*

> *[*cuDF*](https://docs.rapids.ai/api/cudf/stable/user_guide/10min.html) *是一个 Python GPU DataFrame 库，构建在 Apache Arrow columnar 内存格式上，用于数据操作。cuDF 还提供了一个类似熊猫的 API，数据工程师&数据科学家会很熟悉，所以他们可以使用它来轻松地加速他们的工作流程，而无需进入 CUDA 编程的细节。**

*cuDF 可以通过 conda 安装，看看[这个指南](https://github.com/rapidsai/cudf)。*

```
*import cudf
df = cudf.read_csv('../input/yellow-new-york-taxi/yellow_tripdata_2009-01.csv')
df2 = df.groupby('vendor_name').agg({'Passenger_Count':'mean'})*
```

# *结论*

*在本文中，我们研究了 8 个用于操作大型数据集的 python 库。它们中的许多都有与 Pandas 相似的语法，这使得学习曲线不那么陡峭。如果你想知道不同包的执行速度， [H2O.ai](http://H2O.ai) 已经在其中一些包上创建了一个有用的 [ops 基准](https://h2oai.github.io/db-benchmark/)。它比较了不同大小的数据集上各种包的执行速度，以及不同常见操作(如连接和分组)的执行速度。*

*我希望这是一个有用的开始，请在您处理 python 大型数据集的常用库下面留下评论。*

*[加入 Medium](https://medium.com/@edwin.tan/membership) 阅读更多这样的故事。*