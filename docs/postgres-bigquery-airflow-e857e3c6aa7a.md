# 用 Airflow 将数据从 Postgres 加载到 BigQuery

> 原文：<https://towardsdatascience.com/postgres-bigquery-airflow-e857e3c6aa7a>

## 使用 Apache Airflow 从 PostgreSQL 到 BigQuery 数据摄取—分步指南

![](img/d92c07bf30e309672adcc74eda6bff10.png)

在 [Unsplash](https://unsplash.com/s/photos/transfer?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上 [Venti Views](https://unsplash.com/es/@ventiviews?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄的照片

将数据从 Postgres 数据库(本地托管)接收到 Google Cloud BigQuery 的一种方法是使用 Airflow，它提供了大量可用于数据接收和集成过程的操作符。

支持使用这些操作符也很重要，因为它们可以帮助我们编写更少更简单的代码，以便在更广泛的数据工程环境中执行基本操作。

现在，为了从 Postgres 表中提取数据并将其加载到 BigQuery 中，我们可以编写一个管道，首先使用正确的操作符将 Postgres 中的数据以 csv 格式提取到 Google 云存储中，然后将 GCS 中的 csv 文件加载到 BigQuery 中。

在今天的教程中，我们将一步一步地介绍如何使用 Apache Airflow 将数据从 Postgres 数据库导入 Google Cloud BigQuery。我们将遵循三个单独的步骤，最终让我们从 Postgres 执行*完全加载*到 Google 云平台上的数据仓库，该数据仓库每天都会执行。

## 步骤 1:将 Postgres 表提取到 Google 云存储中

因此，第一步涉及一个过程，我们将数据从源 Postgres 表转移到 Google 云存储中。换句话说，我们将以 csv 格式将数据从源表导出到对象存储中。

在深入研究细节之前，首先需要确保已经创建了用于连接 Postgres 数据库的连接。这可以通过 Airflow UI(管理->连接)来实现。或者，为了更安全和可扩展地管理连接，您可以参考我在 Medium 上的一篇最新文章，在这篇文章中，我讨论了如何使用 Airflow 设置 HashiCorp Vault。

[](/hashicorp-vault-airflow-cfdddab31ea)  

现在，假设我们有一个有效的 Postgres 连接，我们现在可以利用包含在`apache-airflow-providers-google`包中的`[PostgresToGCSOperator](https://airflow.apache.org/docs/apache-airflow-providers-google/stable/_api/airflow/providers/google/cloud/transfers/postgres_to_gcs/index.html)`,将数据从源数据库传输到 Google 云存储中。

```
from airflow.providers.google.cloud.transfers.postgres_to_gcs import PostgresToGCSOperatorGCS_BUCKET = 'my-bucket'
GCS_OBJECT_PATH = 'postgres-test'
SOURCE_TABLE_NAME = 'mytable'
POSTGRESS_CONNECTION_ID = 'postgres' postgres_to_gcs_task = PostgresToGCSOperator(
    task_id=f'postgres_to_gcs',
    postgres_conn_id=POSTGRES_CONNECTION_ID,
    sql=f'SELECT * FROM {SOURCE_TABLE_NAME};',
    bucket=GCS_BUCKET,
    filename=f'{GCS_OBJECT_PATH}/{SOURCE_TABLE_NAME}.{FILE_FORMAT}',
    export_format='csv',
    gzip=False,
    use_server_side_cursor=False,
)
```

本质上，这个任务将把数据从名为`'mytable'`的源 Postgres 表传输到 Google 云存储中，更具体地说是在`gs://my-bucket/postgres-test/mytable.csv`下。

## 步骤 2:将数据从 GCS 加载到 BigQuery

既然我们已经将数据从 Postgres 复制到 Google 云存储中，我们就可以利用另一个名为`[GCSToBigQueryOperator](https://airflow.apache.org/docs/apache-airflow-providers-google/stable/_api/airflow/providers/google/cloud/transfers/gcs_to_bigquery/index.html)`的操作符将数据从云存储转移到 BigQuery。

```
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import GCSToBigQueryOperator BQ_DS = 'my_dataset'
BQ_PROJECT = 'my-project'schema = [
    {
        'name': 'id',
        'type': 'STRING',
        'mode': 'NULLABLE',
    },
    {
        'name': 'name',
        'type': 'STRING',
        'mode': 'NULLABLE',
    },
    {
        'name': 'age',
        'type': 'INTEGER',
        'mode': 'NULLABLE',
    },
    {
        'name': 'is_active',
        'type': 'BOOLEAN',
        'mode': 'NULLABLE',
    },
]gcs_to_bq_task = GCSToBigQueryOperator(
    task_id=f'gcs_to_bq',
    bucket=GCS_BUCKET,
    source_objects=[f'{GCS_OBJECT_PATH}/{SOURCE_TABLE_NAME}.csv'],
    destination_project_dataset_table='.'.join(
        [BQ_PROJECT, BQ_DS, SOURCE_TABLE_NAME]
    ),
    schema_fields=schema,
    create_disposition='CREATE_IF_NEEDED',
    write_disposition='WRITE_TRUNCATE',
    skip_leading_rows=1,
    allow_quoted_newlines=True,
)
```

上面的代码将把 bucket `gs://my-bucket/postgres-test/mytable.csv`上的 Google 云存储中的数据加载到 BigQuery 表`my-project.my_dataset.mytable`中。

请注意，我们还必须指定 BigQuery 表的所需模式——您可能希望对此进行调整，以适应您的特定用例。还要注意，我们指示操作符跳过 GCS 上源文件中的第一行(`skip_leading_rows=1`)，因为该行对应于我们并不真正需要的表列，假设我们已经在 BigQuery 上为目标表指定了所需的模式。

## 第三步:清理混乱

现在，下一步涉及删除我们存储在 Google 云存储中的中间文件，这些文件是为了将数据从 Postgres 转移到 BigQuery 而创建的。

为此，我们可以再一次使用一个名为`GCSDeleteObjectsOperator`的操作符，该操作符用于从 Google 云存储桶中删除对象，或者从对象名称的显式列表中删除对象，或者从匹配前缀的所有对象中删除对象。

```
from airflow.providers.google.cloud.operators.gcs import GCSDeleteObjectsOperator cleanup_task = GCSDeleteObjectsOperator(
    task_id='cleanup',
    bucket_name=GCS_BUCKET,
    objects=[f'{GCS_OBJECT_PATH}/{SOURCE_TABLE_NAME}.csv'],
)
```

## 步骤 4:编写最终气流 DAG

现在，我们需要做的就是创建一个 DAG，它将包含我们在本教程的上一部分中创建的三个任务。请注意，在最后，我们还指定了任务之间的依赖关系，以便它们可以按顺序执行。

```
from datetime import timedeltawith DAG(
    dag_id='load_postgres_into_bq',
    start_date=days_ago(1),
    default_args={
        'owner': 'airflow',
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
    },
    schedule_interval='0 9 * * *',
    max_active_runs=1,
) as dag:
    postgres_to_gcs_task = PostgresToGCSOperator(
        task_id=f'postgres_to_gcs',
        postgres_conn_id=POSTGRES_CONNECTION_ID,
        sql=f'SELECT * FROM {SOURCE_TABLE_NAME};',
        bucket=GCS_BUCKET,
        filename=f'{GCS_OBJECT_PATH}/{SOURCE_TABLE_NAME}.{FILE_FORMAT}',
        export_format='csv',
        gzip=False,
        use_server_side_cursor=False,
    ) gcs_to_bq_task = return GCSToBigQueryOperator(
        task_id=f'gcs_to_bq',
        bucket=GCS_BUCKET,
        source_objects=[f'{GCS_OBJECT_PATH}/{SOURCE_TABLE_NAME}.csv'],
        destination_project_dataset_table='.'.join(
            [BQ_PROJECT, BQ_DS, SOURCE_TABLE_NAME]
        ),
        schema_fields=schema,
        create_disposition='CREATE_IF_NEEDED',
        write_disposition='WRITE_TRUNCATE',
        skip_leading_rows=1,
        allow_quoted_newlines=True,
    ) cleanup_task = GCSDeleteObjectsOperator(
        task_id='cleanup',
        bucket_name=GCS_BUCKET,
        objects=[f'{GCS_OBJECT_PATH}/{SOURCE_TABLE_NAME}.csv'],
    ) postgres_to_gcs_task >> gcs_to_bq_task >> cleanup_task
```

## 完整代码

你可以在下面分享的 GitHub Gist 上找到这个教程的完整代码。

本教程中使用的完整代码——如何将数据从 Postgres 数据移动到 Google Cloud BigQuery

## 最后的想法

在今天的教程中，我们演示了一个将数据从 Postgres 数据库摄取到 Google Cloud BigQuery 的逐步方法。因为我们不能将 Postgres 上的源表中的数据加载到 BigQuery 中，所以我们使用 Google 云存储作为中间对象存储，Postgres 表将被提取到 csv 文件中，然后该文件将被加载到 Google Cloud BigQuery 上的目标表中。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**相关文章你可能也喜欢**

[](/airflow-dags-decorators-b5dc03c76f07)  [](/run-airflow-docker-1b83a57616fb)  [](/hashicorp-vault-airflow-cfdddab31ea) 