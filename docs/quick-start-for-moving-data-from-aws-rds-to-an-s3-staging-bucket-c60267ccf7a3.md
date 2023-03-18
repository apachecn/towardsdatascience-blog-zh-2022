# 将数据从 AWS RDS 移动到 S3 暂存区的快速入门

> 原文：<https://towardsdatascience.com/quick-start-for-moving-data-from-aws-rds-to-an-s3-staging-bucket-c60267ccf7a3>

## *利用 Python 和 Boto3 作为构建数据湖的第一步*

![](img/4b4caed5772537c287e8cabd4c3c80b8.png)

考特尼·摩尔在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

数据仓库架构中的一个常见元素是数据湖。数据湖是所有结构化和非结构化数据源的中央存储库。最近，我一直在使用雪花，一个流行的基于云的数据存储解决方案，作为一个数据湖和数据仓库。

获取源数据是构建数据湖的第一步。在本文中，我们将建立到 AWS RDS Postgres 实例的连接，并在将数据复制到 Snowflake 之前将数据传输到 S3 分段存储桶。本文利用 Python 和 Boto3 库来访问 AWS 资源。

## **要求**

1.  计算机编程语言
2.  Boto3 库
3.  Psycopg2 库
4.  IO 库
5.  AWS 帐户访问
6.  AWS 帐户访问 ID 和秘密访问密钥
7.  RDS Postgres 数据库和用户访问/凭证
8.  S3 桶访问

# 入门指南

[Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) 是 Python 的 AWS SDK。它提供了面向对象的 API 和对 AWS 服务的底层访问。为了建立到 RDS 的连接，我们可以利用`Boto3 Session`对象来生成一个`db_authentication_token`，我们将在稍后使用`psycopg2`连接到 Postgres 时使用它。

有了这个令牌(密码)，一旦我们有了连接对象，就可以用`psycopg2`连接并查询我们的 RDS 实例。

有了这个连接对象，我们可以向 RDS 数据库发送查询来执行必要的提取。举个例子，

```
import pandas as pd
sql = f'''
    select table_name
    from information_schema.tables
    where table_schema like 'public'
       and table_type like 'base table'
'''
results = pd.read_sql(sql, conn)
df = pd.DataFrame(results)
```

使用同样的逻辑，我们可以遍历表名的结果，并从每个表中提取行上传到雪花。

# S3 暂存桶

一旦我们有了包含源数据的可用数据帧，我们就可以开始设置另一个 python 会话资源来用于 S3 分段存储桶。一旦分配了资源，我们就可以利用`io.StringIO`库将数据帧作为 csv 文件上传到 S3 存储桶。

[StringIO](https://docs.python.org/3/library/io.html?highlight=stringio#io.StringIO) 是一个文本流对象，当我们将 csv 写入 S3 存储桶时，它使用内存中的文本缓冲区来保存数据帧的内容。

`StringIO`的好处是它可以很好地与 Apache Airflow 这样的 orchestrator 一起工作，在上传文件之前，我们不需要担心将查询的输出编写为本地 csv 文件(这是另一种选择)。然而，我们也需要考虑这种选择的局限性。随着表大小的增加，有一个明显的内存限制，这将需要增量提取或多部分上传。这可以很容易地用 pandas 数据帧批处理来处理，从而减少内存负载。

这些数据现在可以在 S3 以 csv 格式获得！将数据放入雪花数据湖的下一步是利用 [S3 存储桶作为中转站点](https://docs.snowflake.com/en/user-guide/data-load-s3-create-stage.html)，这样数据就可以直接加载到雪花中。期待在我的下一篇文章中更深入地探究这个过程。

概括地说，在本文中，我们探索了 Boto3 库，用 Python 建立了到 AWS 资源的会话和连接，并从 RDS 中提取数据，以`.csv`格式上传到 S3 桶。在我作为数据工程师的工作中，这是一种常见的做法。请伸出手来，与我联系，谈论更多关于这个或任何问题。