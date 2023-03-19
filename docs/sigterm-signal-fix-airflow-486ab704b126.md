# 如何确定气流中接收到的信号

> 原文：<https://towardsdatascience.com/sigterm-signal-fix-airflow-486ab704b126>

## 修复阿帕奇气流任务中的信号术语

![](img/e120c443af565e47a7446172d31bd99e.png)

杰里米·珀金斯在 [Unsplash](https://unsplash.com/s/photos/error?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

## 介绍

虽然我最近一直在致力于将 Dag 从气流 1 ( `v1.10.15`)迁移到气流 2 ( `v2.2.5`)，但我花了很多时间试图找出我在一些 Dag 中遇到的一个错误，这个错误根本不能提供信息。

```
WARNING airflow.exceptions.AirflowException: Task received SIGTERM signal
INFO - Marking task as FAILED.
```

尽管我花了一些时间尝试我在网上找到的可能的解决方案，但似乎没有一个对我有效。

在今天的文章中，我将介绍一些针对发送到任务的`SIGTERM signal`的潜在解决方案，它会导致气流 Dag 失败。根据您的配置和具体使用情况，可能会有不同的解决方案适合您，因此请确保仔细阅读并尝试每个建议的解决方案。

## DAG 运行超时

您的任务接收到一个`SIGTERM`信号的原因之一是由于一个短的`dagrun_timeout`值。DAG 类接受此参数，该参数用于指定 DagRun 在超时/失败之前应该运行多长时间，以便可以创建新的 DAG run。请注意，**超时仅针对计划的** DagRuns。

对于包含许多长时间运行任务的 DAG，有可能超过`dagrun_timeout`，因此活动运行的任务将接收到一个`SIGTERM`信号，以便 DAG 可以失败，并执行新的 DagRun。

您可以在 Airflow UI 上检查 DagRun 的持续时间，如果您观察到该持续时间大于创建 DAG 实例时指定的`dagrun_timeout`值，那么您可以根据您的特定用例将其增加到一个合理的时间量。

请注意，此配置适用于 DAG，因此您需要提供一个值，以便有足够的时间来运行 DAG 中包含的所有任务。

```
from datetime import datetime, timedeltafrom airflow.models.dag import DAG with DAG(
    'my_dag', 
    start_date=datetime(2016, 1, 1),
    schedule_interval='0 * * * *',
    dagrun_timeout=timedelta(minutes=60),
) as dag:
    ...
```

## 内存不足

另一种可能是，当前正在运行气流任务的机器内存不足。根据您部署 Airflow 的方式，您可能需要检查工作线程的内存使用情况，并确保它们有足够的内存。

例如，如果您的部署在云上，您可能必须检查是否有任何 Kubernetes pods 被驱逐。吊舱通常由于资源匮乏的节点而被驱逐，因此这可能是你的气流任务接收到`SIGTERM`信号的原因。

## 元数据数据库耗尽了 CPU

另一个常见的可能导致 Airflow 任务接收到`SIGTERM`信号的问题是元数据数据库上的 CPU 使用率。

默认情况下，Airflow 使用 SQLite，它仅用于开发目的，但它旨在支持 PostgreSQL、MySQL 或 MSSQL 的数据库后端。

数据库上的 CPU 使用率有可能达到 100%,这可能是您的气流任务接收到`SIGTERM`信号的原因。如果是这种情况，那么您应该考虑增加默认设置为 5 秒的`job_heartbeat_sec`配置(或`AIRFLOW__SCHEDULER__JOB_HEARTBEAT_SEC`环境变量)的值。

> `job_heartbeat_sec`
> 
> 任务实例监听外部终止信号(当您从 CLI 或 UI 清除任务时)，这定义了它们应该监听的频率(以秒为单位)。
> 
> - [气流文件](https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html#job-heartbeat-sec)

在气流配置文件`airflow.cfg`中，确保在`scheduler`部分指定该配置，如下图所示。

```
[scheduler]
job_heartbeat_sec = 20
```

或者，您可以通过相应的环境变量修改此配置的值:

```
export AIRFLOW__SCHEDULER__JOB_HEARTBEAT_SEC=20
```

如果数据库级别的 CPU 消耗是一个问题，那么增加上述配置现在应该可以显著减少 CPU 的使用。

## 禁用“迷你计划程序”

默认情况下，任务管理器进程会尝试调度同一气流 DAG 的更多任务，以提高性能并最终帮助 DAG 在更短的时间内执行。

该行为通过默认为`True`的`schedule_after_task_execution`进行配置。

> `schedule_after_task_execution`
> 
> 任务管理器进程是否应该执行“迷你调度器”来尝试调度同一 DAG 的更多任务。启用此选项将意味着同一 DAG 中的任务执行得更快，但在某些情况下可能会耗尽其他 DAG。
> 
> - [气流文件](https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html#schedule-after-task-execution)

由于气流中的一个 [bug](https://github.com/apache/airflow/pull/16289) ，任务被`LocalTaskJob`心跳杀死的几率相当高。因此，一个可能的解决方案是简单地禁用迷你调度程序。

在您的气流配置文件`airflow.cfg`中，您需要将`*schedule_after_task_execution*` 设置为 False。

```
[scheduler]
schedule_after_task_execution = False
```

或者，可以通过`AIRFLOW__SCHEDULER__SCHEDULE_AFTER_TASK_EXECUTION`环境变量覆盖该配置:

```
export AIRFLOW__SCHEDULER__SCHEDULE_AFTER_TASK_EXECUTION=False
```

如果这是你的问题所在，那么你可能也想考虑将 Airflow 升级到一个修复了这个 bug 的版本。

## 最后的想法

在今天的教程中，我们讨论了`SIGTERM`信号的含义，该信号有时会发送到气流任务，导致 Dag 失败。我们讨论了发生这种情况的几个潜在原因，并展示了如何根据您的具体用例来克服这个问题。

请注意，您的配置也有可能遇到本教程中讨论的不止一个问题，因此您可能需要应用我们今天讨论的解决方案组合来消除`SIGTERM`信号。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**相关文章你可能也喜欢**

[](/run-airflow-docker-1b83a57616fb)  [](/hashicorp-vault-airflow-cfdddab31ea)  [](/connect-airflow-worker-gcp-e79690f3ecea) 