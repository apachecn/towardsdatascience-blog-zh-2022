# 使用气流装饰器创作 Dag

> 原文：<https://towardsdatascience.com/airflow-dags-decorators-b5dc03c76f07>

## 用 Python decorators 创作 Apache Airflow DAGs 和任务

![](img/09010d401053cd21056d20cad6e95274.png)

由[柴坦尼亚电视台](https://unsplash.com/@tvschaitanya?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/circle?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄的照片

## 介绍

在气流中编写管道的最常见方式是使用 DAG 上下文管理器自动为该 DAG 分配新的运算符。从 Airflow 2 开始，您现在可以使用 decorators 来创作 Airflow DAGs 和任务。

在今天的教程中，我们将介绍这些装饰器，并展示如何使用它们来编写更简洁的代码。此外，我们还将演示如何使用更传统的方法编写相同的 DAG。

## 传统的方式

在 Airflow 中创作数据管道最常用的方法是使用 DAG 作为上下文管理器，自动为该 DAG 分配新的操作符。

例如，假设我们想要构建一个 ETL 管道，在这个管道中，我们将从外部 API 获取一些数据，对其进行转换，并最终将结果报告为标准输出。

为此，我们将创建三个简单的任务，如下所示。

```
import logging
import requests
from datetime import timedeltafrom airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator ENDPOINT = 'https://some-api.com/data'default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}def _extract_data():
    return requests.get(ENDPOINT).json()['data']def _transform_data(data):
    return {'no_records': len(data)}def _load_results(no_records):
    logging.info(
        f'No. of records fetched by {ENDPOINT}: {no_records}'
    )with DAG(
    default_args=default_args,
    schedule_interval='@hourly', 
    start_date=days_ago(1),
):
    extract = PythonOperator(
        task_id='extract_from_api',
        python_callable=_extract_data,
    ) transform = PythonOperator(
        task_id='transform_data',
        python_callable=_transform_data,
        op_kwargs=[extract]
    ) load = PythonOperator(
        task_id='load_data',
        python_callable=_load_results,
        op_kwargs=[transform['no_records']]
    ) extract >> transform >> load
```

## 使用气流装饰器创作 Dag

从 Airflow 2.0 开始，你也可以使用 decorators 从一个函数创建 Dag。Python 中的 decorator 是一个函数，它接受另一个函数作为参数，修饰它(即丰富它的功能)并最终返回它。

DAG 装饰器创建一个 DAG 生成器函数，每个用`@dag`装饰器装饰的函数都将返回一个 DAG 对象。但是请注意，到目前为止，您只能对 Python 函数使用 Airflow 装饰器，通常您必须通过`PythonOperator`调用这些函数。

```
from datetime import timedeltafrom airflow.decorators import dag, task
from airflow.utils.dates import days_ago ENDPOINT = 'https://some-api.com/data'default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}@dag(
    default_args=default_args,
    schedule_interval='@hourly', 
    start_date=days_ago(1),
)
def example_dag(): @task
    def extract():
        return requests.get(ENDPOINT).json()['data'] @task
    def transform(data):
        return {'no_records': len(data)} @task
    def load(no_records):
        logging.info(
            f'No. of records fetched by {ENDPOINT}: {no_records}'
        ) load(transform(extract()['no_records'])) dag = example_data()
```

正如你所看到的，decorators 帮助我们写了一个更干净、更简洁的代码。无需为每个单独的任务显式指定 PythonOperators，我们通过利用新的气流装饰器，仅用几行代码就创建了一个相当简单的 DAG。

## 最后的想法

在今天的文章中，我们展示了如何使用新的气流装饰器来创作任务和 Dag。这无疑是一个非常有前途的特性，但还不完全，因为它只能用于非常小的可用操作符子集。

此外，我对我们需要使用新符号来指定任务依赖关系的方式有点怀疑。对于复杂的依赖关系，语法会变得非常混乱，可读性也不好。总的来说，我认为这是一个很好的补充，至少可以用于一些基本的 Dag。

请注意，您甚至可以创建定制的装饰器，以满足您特定项目的需求。您可以在官方文档的[相关章节中找到更多关于如何操作的信息。](https://airflow.apache.org/docs/apache-airflow/2.0.2/howto/custom-operator.html)

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**相关文章你可能也喜欢**

[](/run-airflow-docker-1b83a57616fb)  [](/connect-airflow-worker-gcp-e79690f3ecea)  [](/hashicorp-vault-airflow-cfdddab31ea) 