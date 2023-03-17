# 数据管道中的超驱动步骤

> 原文：<https://towardsdatascience.com/hyperdrivestep-in-data-pipelines-fe48a69ec7c7>

# 数据管道中的超驱动步骤

## 使用 tqdm 和 Azure 机器学习扩展 Python 流程。

![](img/b24965993c8fac5dfea63ac5ff4e3630.png)

托马斯·凯利在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

欠开发和过度工程之间的平衡是如此之弱，以至于工程经常在这两个对立面之间摇摆。在本文中，我将解释几种数据管道扩展技术，我们可以应用这些技术来应对业务需求，并帮助您避免越界。

# 方案

你是一名数据科学家，在一家制作商店销售预测的新公司工作。在建立机器学习模型的过程中，需要对输入数据进行预处理。这个数据以这样一种方式建模，即有一个国家的集合，每个国家有许多商店，每个商店需要以自己的方式处理。

以下代码显示了处理所有商店的简单方法:

```
def etl_preprocess(int shop_id) -> pd.DataFrame:
    # do your work 
    pass
​
country_ids = get_countries()
for country_id in country_ids:
    shop_ids = get_country_shops(country_id)
    for shop_id in shop_ids:
        etl_preprocess(shop_id)
```

开始时一切都很好，但是随着公司因最终的成功而不断发展，这个过程在开始时只需要 30 分钟就可以为数百家商店执行，现在随着商店数量达到数百万，这个过程在几天内就会停滞不前。

你知道“死于成功”吗？当一个企业无法处理过度增长的需求时，就会出现这种情况，因此它会屈服于这种需求。

# 如何扩大规模

我们将从简单快速的胜利开始。所以首先要检查的是程序本身。回到理论上来，看看是否可以使用数据类型和/或不同的算法来重构这个过程。想的更聪明，不要更努力。

一旦达到算法性能的峰值，我们就可以开始考虑增加更多的*金属*(硬件资源)来提高程序的吞吐量。默认情况下，Python 阻止多个线程在同一个进程中同时执行。这是使用异步函数获得性能提升的基础，因为当当前线程被阻塞时，大量 IO 进程可以使用额外的线程。但是对于 CPU 密集型操作，您的代码可能会在一个强大的多核 CPU 中执行，然而由于 [GIL](https://python.land/python-concurrency/the-python-gil) 的原因，Python 解释器被限制在一个 CPU 内核中执行(要更好地理解这个问题，请查看[Python 中的线程和进程简介](https://medium.com/@bfortuner/python-multithreading-vs-multiprocessing-73072ce5600b))。

如果你使用云基础设施，你要为每一个 CPU 支付全价，所以很可能你会送钱。除非你有充分的理由，否则不要向你的经理提及此事。也不要责怪开发人员，因为 Python 通过设计(通过前面提到的 GIL)阻止了多个线程访问同一个进程。但是我们可以通过使用用于多处理的核心 python API 来创建多个进程，幸运的是还有第三方库提供更好的支持。

# 挤出所有的核

我发现(到目前为止)Python 中多处理的最佳方法是使用 [tqdm](https://github.com/tqdm/tqdm) 。在大多数情况下，由于这个库在 Python 社区中的成功，这个库很有可能已经是项目需求的一部分。

当安装了`tqdm`之后，您就可以访问这个隐藏在其 contrib 名称空间中的小东西了:

```
from tqdm.contrib.concurrent import process_map
```

从那里，您可以执行:

```
country_ids = get_countries()
for country_id in country_ids:
    shop_ids = get_country_shops(country_id)
    results = process_map(etl_preprocess, shop_ids)
```

这段代码按照您的设想工作:它将为每个调用创建一个新的进程(每个调用使用不同的参数执行名为`etl_preprocess`的函数，这些参数取自`shop_ids`变量)。默认情况下，它会使用所有可用的内核，`process_map`会一直等到所有进程执行完毕。在这种情况下，对`etl_preprocess`的每个调用都返回一个单独的`pandas`数据帧，因此`process_map`将收集这些函数的返回，并返回一个包含所有熊猫数据帧的列表(返回数据的顺序与调用的执行顺序无关)。

使用这种方法的一些缺点:

*   考虑到如果由`process_map`调用的函数运行另一个多进程函数，它可能会以死锁场景结束。当使用`xgboost`或`TensorFlow`训练一个模型时，我发现自己陷入了这个陷阱(但是你的运气可能会因所用库的版本和特性而异)
*   在发生异常的情况下，这些可以被`process_map`屏蔽，所以我建议小心处理您的日志记录
*   尽量避免依赖父执行(例如，避免使用*共享*变量)。`process_map`执行的方法应该是完全自治的(使用*依赖注入*设计模式)，或者公共数据应该被酸洗；否则，您可能会遇到竞态条件和意外的错误
*   有时这种方法对测试库来说并不好用。解决这种情况的一种方法是使用`process_map(...,max_workers=1)`参数或[模仿](https://docs.python.org/3/library/unittest.mock.html#patch) `process_map`方法

## 可供选择的事物

如果你的项目中不使用 tqdm，可以给 [verstack](https://github.com/DanilZherebtsov/verstack) 或者 [mpire](https://github.com/Slimmer-AI/mpire) 一个机会。我没有测试过这些库，但是 API 是相似的，除此之外，`verstack`包含了常见机器学习任务的助手。

# 分配你的工作

增强您的过程的另一个选择是将它变成一个完全分布式的作业，并在集群中启动它。在我们深入探讨之前，我先简单定义一些基本概念:**集群**是计算机的集合。在集群世界里，计算机被称为**节点**；大多数时候，存在一个名为*驱动*的节点，它管理其余节点(名为*工作节点*)的生命周期(节点初始化/关闭/监控)，驱动还负责向每个工作节点发送工作负载。在我们的上下文中，一个**作业**包括整个程序的执行，因此对所有国家/商店执行`etl_preprocess`。

综上所述，之前我们使用 CPU 内核来倍增性能，现在我们将使用集群(反过来可以由 *p* 节点和 *m* 内核组成)。我希望你能了解这个规模。

Azure Machine Learning 为设置集群提供了一个优秀的 API(甚至你可以使用 UI 来完成)。复杂的部分是你如何访问集群的被管理部分，并让它为你的分布式目的服务。有两种方式: [ParallelRunStep](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-steps/azureml.pipeline.steps.parallelrunstep?view=azure-ml-py) 和 [HyperDriveStep](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-steps/azureml.pipeline.steps.hyper_drive_step.hyperdrivestep?view=azure-ml-py) 。在我的例子中，基于`HyperDriveStep`的流程更容易实现(尽管我建议你自己测试`ParallelRunStep`并检查，DYOR 原则总是适用的)。

`HyperDriveStep`通常用于超参数优化，可配置为简单的网格搜索。这是*间接路径*到达*管理部分*并使其执行我们的工作负载。

最终的代码比之前稍微复杂一点，因为实现需要嵌入到 Azure 机器学习中。幸运的是，基本实现很容易:

```
run_config = ScriptRunConfig(source_directory=source_directory,
                             script="./etl_pre_process.py",
                             arguments=[],
                             compute_target=build_compute_target(ws),
                             environment=build_environment())
​
country_ids = get_countries()
param_sampling = GridParameterSampling({'country_id': choice(*country_ids)})
​
hyperdrive_config = HyperDriveConfig(run_config=run_config,                                                hyperparameter_sampling=param_sampling,
                                     primary_metric_name='count',                                   primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                     max_total_runs=len(ids))
​
step = HyperDriveStep("etl_pre_process", hyperdrive_config, allow_reuse=False)
​
```

现在，etl_pre_process 模块将被执行，传递参数`country_id`:

```
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--country_id', type=str, dest='country_id')
    args = parser.parse_args()
​
    shop_ids = get_country_shops(args.country_id)
    results = process_map(etl_preprocess, shop_ids)
```

在这种情况下，我们需要使用一个可执行的 Python 模块作为基本的执行功能，Azure 机器学习将负责繁重的工作:

*   将代码库签出到工作节点中
*   安装依赖项，并将所需的 Python 模块复制到工作节点。一旦节点准备就绪，节点将开始执行指定的 Python 模块
*   集群初始化需要一些时间，所以使用不同的集群大小进行一些测试，以减少作业的总执行时间
*   集群管理:例如，如果您的集群设置为最多 10 个节点，但是当前的执行只使用了 5 个国家，那么它将只启动 5 个节点。集群本身将*在所有节点之间协调/划分*作业，并在作业完成时逐渐关闭节点

这种解决方案的缺点是:

*   *侵入集群*(例如，在配置`HyperDriveConfig`时，我们必须使用`primary_metric_name`和`primary_metric_goal`等参数，否则这些参数对于简单的数据争论过程毫无意义)
*   从单台机器到集群范例的转换需要额外的工作，这些工作并不总是*直接/琐碎的*并且应该提前计划:如何*注入*每次执行所需的参数？如何平衡均匀的工作量？或者最终如何收集数据？

# 摘要

在 Python 中扩展流程不是一件容易的事情。大多数时候，扩展至少需要对现有代码进行一些重构，甚至从头开始开发代码。有许多选项唾手可得:核心 Python 多线程/多处理函数，像 [dask](https://dask.org/) 或 [ray](https://www.ray.io/) 这样的库，基于 Spark 的完全分布式解决方案(如 [Databricks](https://databricks.com/) 或 [Snowflake](https://www.snowflake.com/) )，最后，还有像 Azure Machine Learning 这样的 IaaS 解决方案，你可以使用本文中解释的*过程*来轻松扩展你的过程。