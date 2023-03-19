# 如何在 Python 中为大数据工作流构建强大的气流 Dag

> 原文：<https://towardsdatascience.com/how-to-build-powerful-airflow-dags-for-big-data-workflows-in-python-14eb2b66f280>

## 将您的气流管道扩展到云

![](img/d0dd05c3dbdac623d152eac2406d52de.png)

图片由 Solen Feyissa 通过 [Unsplash](http://unsplash.com) 拍摄

# 气流 DAGs for(真的！)大数据

Apache Airflow 是用于编排数据工程、机器学习和 DevOps 工作流的最流行的工具之一。但是它有一个重要的缺点。开箱即用，Airflow 将在本地运行您的计算*，*这意味着您只能处理适合您的机器资源的数据集。

要在大于内存的数据集上使用 Airflow 进行计算，您可以将包含繁重工作负载的特定 Airflow 任务扩展到 Dask 集群。这篇博客将向您展示如何为大于内存的数据集构建气流 Dag，而只需对您现有的 Python 代码进行最小的更改。

我们将详细介绍一个例子。你可以在[这个专门的库](https://github.com/coiled/coiled-resources/tree/main/airflow-with-coiled)中找到其他气流 DAG 的例子。

# 如何使用 Dask 扩展气流 ETL 任务

气流工作流程由`Tasks`和`DAGs`定义，并由`Executors`协调。为了将繁重的工作流委托给 Dask，我们将在一个包含繁重计算的`Task` 中旋转一个盘绕的集群，并返回结果，在本例中是一个**。** `value_counts()`掠过一列兴趣。因为这个结果很容易被存储到内存中，所以我们可以立即关闭集群以限制成本，并在本地的下一个任务中继续使用这个结果。

*免责声明:我在 Coiled 工作，是一名数据科学传播者。*[*Coiled*](http://coiled.io/)*由*[*Dask*](https://dask.org/)*的最初作者马修·洛克林(Matthew Rocklin)创立，是一个面向分布式计算的开源 Python 库。*

# 定义您的气流 ETL DAG

`DAG`将包含以下 3 个`Tasks`:

1.  旋转盘绕的集群，对整个数据集执行繁重的计算，然后关闭集群；
2.  使用结果计算汇总统计数据，并将这些数据保存到 CSV 文件中；
3.  使用 result 查找前 100 名最活跃的 Github 用户，并将它们保存到一个 CSV 文件中。

让我们从使用`@dag`装饰器定义一个气流`DAG`开始，向它传递脚本中前面定义的`default_args`以及一些您可以调整的其他参数。

```
# define DAG as a function with the @dag decorator 
@dag( 
    default_args=default_args, 
    schedule_interval=None, 
    start_date=datetime(2021, 1, 1), 
    catchup=False, 
    tags=['coiled-demo'], 
) 
def airflow_on_coiled():
```

# 启动你的 Dask 集群

先来定义一下我们的第一个 `Task`。

这就形成了一个名为“气流-任务”的盘绕式集群，由 20 个 Dask 工人组成，每个工人运行一个指定的[盘绕式软件环境](https://docs.coiled.io/user_guide/software_environment.html)，以确保他们拥有所有正确的依赖关系。

```
# define Airflow task that runs a computation on a Coiled cluster @task() 
def transform():     # Create and connect to Coiled cluster
    cluster = coiled.Cluster( 
        name="airflow-task", 
        n_workers=20, 
        software="rrpelgrim/airflow", 
    )     client = Client(cluster)
```

然后，我们可以将存储在 S3 上的数据集读入 Dask 数据帧，并计算出我们感兴趣的结果。在这里，我们加载 2015 年的 Github 存档数据(子集仅包括 PushEvents ),并通过调用`.value_counts()`计算全年每个用户的 PushEvents 数量。

```
# Read CSV data from S3 
ddf = dd.read_parquet( 
    's3://coiled-datasets/github-archive/github-archive-2015.parq/',    
    storage_options={"anon": True, 'use_ssl': True}, 
    blocksize="16 MiB", 
) # Compute result number of entries (PushEvents) per user 
result = ddf.user.value_counts().compute()
```

因为我们现在在本地得到了结果，所以我们可以关闭集群来限制我们的成本。请注意，这实际上只是一种形式，因为 Coiled 会在 20 分钟不活动后自动关闭您的集群。

```
# Shutdown Coiled cluster 
cluster.close() 
return result
```

转到卷云仪表板，我们可以看到这个计算花费了我们 5 美分。不，这不是打印错误😉这意味着您可以使用[盘绕自由层](https://cloud.coiled.io/)每月免费运行这个气流 DAG 示例多达 200 次。

# 本地使用结果

我们已经利用云资源获得了我们感兴趣的结果，现在我们可以在本地继续执行以下任务。因为 Coiled 在您自己的机器上本地运行，所以读写本地磁盘很简单。

我们将生成`result`熊猫系列的汇总统计数据，并将其保存到一个 CSV 文件中:

```
@task() 
def summarize(series): 
    # Get summary statistics 
    sum_stats = series.describe()     # Save to CSV   
    sum_stats.to_csv(
         f'{storage_directory}usercounts_summary_statistics.csv'
    ) 

    return sum_stats
```

并获取前 100 名最活跃用户的用户名和推送事件数量:

```
@task() 
def get_top_users(series): 
    # Get top 100 most active users 
    top_100 = series.head(100)     # Store user + number of events to CSV 
    top_100.to_csv(f'{storage_directory}top_100_users.csv')     return top_100
```

最后但同样重要的是，我们将指定我们希望气流`Tasks`运行的顺序，并实际调用`DAG`函数来触发工作流。

```
# Call task functions in order 
series = transform() 
sum_stats = summarize(series) 
top_100 = get_top_users(series) # Call taskflow 
airflow_on_coiled()
```

干得好，都准备好了！现在，您可以将 Airflow DAG Python 脚本添加到您的`dags`文件夹(默认为:`~/airflow/dags`)中，并根据需要运行或调度它。

*重要提示:默认情况下，气流禁用* `*pickling*` *。您必须启用它才能运行 Dask 任务。你可以通过编辑你的* `*airflow.cfg*` *文件或者通过使用* `*export AIRFLOW__CORE__ENABLE_XCOM_PICKLING = True*` *设置相应的环境变量来实现。在启动 Airflow 服务器之前，请执行此操作。* *如果你在苹果 M1 机器上工作，你可能想看看这篇关于使用 conda* *安装 PyData 库的博客。具体来说，确保在您的本地和集群软件环境中既没有安装* `*blosc*` *也没有安装* `*python-blosc*` *库。*

# 更多气流 DAG 示例

在专用的 [**带盘管的气流**库](https://github.com/coiled/coiled-resources/tree/main/airflow-with-coiled)中，您将找到另外两个使用 Dask 的气流 DAG 示例。示例包括常见的气流 ETL 操作。

请注意:

*   JSON-to-Parquet 转换 DAG 示例要求您将 Airflow 连接到亚马逊 S3。您可以在气流文件[中找到操作说明，点击](https://airflow.apache.org/docs/apache-airflow-providers-amazon/stable/connections/aws.html)。您还需要使用`storage_options`关键字参数将您的 AWS 秘密传递给`to_parquet()`调用。
*   XGBoost DAG 示例仅适用于> 20GB ARCOS 数据集的约 250MB 子集。要在整个数据集上运行它，请查看本教程。

[](https://coiled.io/blog/common-dask-mistakes/)  

# 使用 DaskExecutor 运行所有气流 ETL 任务

上面报告中的气流 DAG 示例从气流任务的中的*启动盘绕簇。您还可以选择不同的架构，并在盘绕式集群上的 Airflow DAG 中运行所有任务。然后，您可以使用 Coiled 的[自适应伸缩](https://docs.coiled.io/user_guide/cluster_scaling.html#using-the-adapt-method)功能，根据工作负载来增减工作人员的数量。*

为此，从使用气流的默认`SequentialExecutor`切换到`DaskExecutor`。使用除默认 SequentialExecutor 之外的任何 Airflow executor 还需要[设置一个专用的数据库后端](https://airflow.apache.org/docs/apache-airflow/stable/howto/set-up-database.html)，Airflow 可以在那里存储与您的工作流相关的元数据。一旦完成，将`DaskExecutor`指向一个已经运行的盘绕星团。

您可以通过在您的`airflow.cfg`文件中进行以下更改来做到这一点，该文件默认存储在~/airflow/中。

1.  设定`executor = DaskExecutor`
2.  设置`cluster_address = <cluster_IP_address/cluster_port>`。您可以使用`cluster.scheduler_address`访问该地址
3.  设置集群的 TLS 设置:`tls_cert`、`tls_key`和`tls_ca`。您可以使用`client.security.tls_key`和`client.security.tls_cert`来访问它们。注意`tls_ca`的值与`tls_cert`相同。

然后，您可以在 Coiled 上运行整个气流 DAG。

在启动盘绕式集群的脚本中包含一个`cluster.adapt(minimum=1, maximum=20)`将确保集群根据工作负载在设定的最小和最大工作数量(在本例中为 1 到 20)之间自适应地伸缩。

# 取得联系

在这里关注我，在 Twitter 关注[了解更多类似的内容。](https://twitter.com/richardpelgrim)

*原载于 2022 年 1 月 7 日*[*https://coiled . io*](https://coiled.io/blog/3-airflow-dag-examples-with-dask/)*。*