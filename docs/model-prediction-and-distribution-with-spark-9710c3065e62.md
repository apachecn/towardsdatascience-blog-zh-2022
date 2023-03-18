# 用火花进行模型预测和分配

> 原文：<https://towardsdatascience.com/model-prediction-and-distribution-with-spark-9710c3065e62>

## 如何用 Spark 实现和分发机器学习模型——py Spark 实现

![](img/e81c23ea92a83e267375725749462a6b.png)

[**来源**](https://pixabay.com/illustrations/big-data-binary-code-background-7134400/)

由于 Apache Spark，任务处理可能是分散的。通过将内存中的属性与 Spark SQL 功能结合使用，它增强了这一过程。Spark 的分布式数据记录和处理方法是通过包括分布式脚本、数据处理、数据工作流的创建和具有 MLlib 函数的机器学习技术在内的功能实现的。

Spark 可以根据平台以不同的方式安装。在本节中，我们将重点关注本地安装。

Apache Spark 可以在任何安装了 Python、Scala 或 Java 的环境中运行。本文将关注 Python 语言。紧凑快速地安装所需的 Python 包和 Jupyter Notebook 的最简单方法是使用 [Anaconda](https://www.anaconda.com/products/individual#Downloads) 。

## 探索性数据分析

*“ny T2”*数据集将在整篇文章中使用。你可以通过 Kaggle 网站从[这个链接](https://www.kaggle.com/cmenca/new-york-times-hardcover-fiction-best-sellers)下载。

数据集的格式具有使用“JSON”函数的基本要求。

```
tr_df = spark.read**.**json('dataset/nyt2.json')ts_df = spark.read**.**json('dataset/nyt2.json')
```

# 新特征生成:特征工程

通过使用特征工程，可以从数据集的当前变量中收集更多的信息数据。

个人的头衔也包含在 Titanic 数据集的“*姓名*”列中。该模型可以从该信息中受益。然后创建一个新的变量。可以使用' *withColumn* '方法添加新的标题列。

```
tr_data = tr_df.withColumn("writer",   regexp_extract(col("author"),"([A-Za-z]+)\.", 1))
```

可能有一些重复的作者姓名。“替换”功能可用于替换它们。

```
feature_dataframe =   tr_data.\
   replace(["Jane Greenn", 
            "Stepheniei Meyer",
            "Jimmy Buffett"],
           ["Jane Green", 
            "Stephenie Meyer",
            "Jimmy Buffett"]) 
```

标题的分布看起来比先前遵循替换过程更精确。

# 用 Spark MLlib 建模

最终的建模数据集必须将所有字符串格式的列转换为正确的数字类型，因为预测算法需要数字变量。

```
from pyspark.ml.feature import StringIndexerwriterIndexer = StringIndexer(inputCol="writer", outputCol="writer_Ind").fit(feature_dataframe)descriptionIndexer = StringIndexer(inputCol="published_date", outputCol="published_ind").fit(feature_dataframe)
```

数据帧中包含所有数字变量，因为以前的字符串格式操作已被删除并编入索引。我们可以使用数据帧中的列来创建特征向量，因为每一列都具有非字符串格式。

```
from pyspark.ml.feature import VectorAssemblerassembler = VectorAssembler(
   inputCols = ["writer","price","published_ind"],
   outputCol = "features")
```

当创建流水线时，分类器的参数可以在` *ParamGridBuilder* 的帮助下进行优化。网格搜索后将创建相应的参数。

```
from pyspark.ml.tuning import ParamGridBuilder

 pg = ParamGridBuilder().build()
```

# 评估指标

精确度可用作模型评估的统计数据。“精确度”数学公式。

## **带 MLFlow 的 MLOps】**

对于 PySpark 模型，MLFlow 可用作模型服务库。根据官方文档中的说明，可以为 PySpark 编程安装该库。

```
pip install mlflow
```

“mlflow.pyfunc”函数可用于填充相关的模型推断。为此，独立分配模型和数据集路径至关重要。然后，模型路线可用于创建火花 UDF。接下来是读取它们并将其注册到数据帧中。先前建立的火花 UDF 在最后阶段用于构造新特征。

在[我的 GitHub repo](https://github.com/pinarersoy/PySpark_SparkSQL_MLib/blob/master/RDD%20Basics%20and%20PySpark%20ML%20Model%20Serving.ipynb) 中可以找到这个模型预测和分发脚本的完整实现示例版本！

非常感谢您的提问和评论！

# 参考

1.  [火花 MLlib](https://spark.apache.org/mllib/)
2.  [阿帕奇火花](https://spark.apache.org/)
3.  [Python API 文档](https://www.python.org/)
4.  [Scala API 文档](https://www.scala-lang.org/)
5.  [Java API 文档](https://www.java.com/)
6.  [MLFlow 文档](https://mlflow.org/)
7.  [MLFlow 安装](https://pypi.org/project/mlflow/)