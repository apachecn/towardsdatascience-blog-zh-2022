# 如何用气流设置 HashiCorp 保险库

> 原文：<https://towardsdatascience.com/hashicorp-vault-airflow-cfdddab31ea>

## 将 HashiCorp Vault 与 Apache Airflow 集成

![](img/96875f324d1aa23fd52e0ec25a2577a0.png)

照片由[克里斯蒂娜·戈塔迪](https://unsplash.com/@cristina_gottardi?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/lock?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

## 介绍

默认情况下，Apache Airflow 从元数据数据库中读取连接和变量，该数据库实际上存储了 Airflow UI 的相应选项卡上可见的所有内容。

尽管通过 UI 添加(或删除)连接和变量(并因此将它们存储在也提供静态加密的元数据数据库中)绝对没有什么特别的问题，但有时将 Airflow 连接到一个中央秘密管理工具可能更容易管理，该工具也被组织中的其他工具使用。

这种方法将从本质上帮助您以更有效的方式管理您的秘密，这样无论何时发生变化，它都会反映在实际读取该秘密的每个工具上，因此您不必手动更新相应系统上的每个秘密(例如 Airflow)。

在今天的教程中，我们将展示如何将 HashiCorp Vault(最常用的秘密管理工具之一)与 Apache Airflow 连接。

## 步骤 1:更新 airflow.cfg 文件

为了将 HashiCorp Vault 与 Airflow 集成，以便后者从前者检索连接和变量，我们需要将`[**VaultBackend**](https://airflow.apache.org/docs/apache-airflow-providers-hashicorp/stable/_api/airflow/providers/hashicorp/secrets/vault/index.html#airflow.providers.hashicorp.secrets.vault.VaultBackend)`指定为`airflow.cfg`的`[secrets]`部分中的`backend`。

```
**[secrets]**
backend = airflow.providers.hashicorp.secrets.vault.VaultBackend
backend_kwargs = {
    "connections_path": "connections",
    "variables_path": "variables",
    "url": "http://127.0.0.1:8200",
    "mount_point": "airflow",
}
```

上述配置假设您的气流连接作为秘密存储在`airflow` mount_path 和路径`connections`(即`airflow/connections`)下。同样，您的变量存储在`airflow/variables`下。

您还应该确保指定允许您在 Airflow 和 Vault 之间执行身份验证的附加参数。例如，您可能需要传递`approle`、`role_id`和`secret_id`参数(或者可能是`token`参数)。你可以在这里看到可用参数的完整列表。

## 步骤 2:将连接作为机密添加到 Vault

连接库应存放在气流配置中指定的`connections_path`中，如上一步所示。

对于每个连接，您必须创建一个至少包含以下密钥之一的存储库密码:

*   `**conn_id**` ( `str` ) -连接 ID。
*   `**conn_type**` ( `str` ) -连接类型。
*   `**description**` ( `str` ) -连接描述。
*   `**host**`(`str`)——主持人。
*   `**login**` ( `str` ) -登录。
*   `**password**`(`str`)——密码。
*   `**schema**` ( `str` ) -模式。
*   `**port**` ( `int` ) -端口号。
*   `**extra**` ( `Union[str, dict]` ) -额外元数据。私有/SSH 密钥等非标准数据可以保存在这里。JSON 编码的对象。
*   `**uri**` ( `str` ) -描述连接参数的 URI 地址。

这是因为机密的内容应该与提供的应该与`[airflow.models.connections.Connection](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/models/connection/index.html#airflow.models.connection.Connection)`类的预期参数对齐。

## 步骤 3:添加变量作为 Vault 机密

现在进入变量，当你把它们作为秘密添加到 vault 中时，你应该小心一点，因为预期的格式可能不像你通常预期的那样直接。

让我们假设在`airflow.cfg`中，你已经将`variables_path`指定为`variables`，将`mount_point`指定为`airflow`。如果您希望存储一个名为`my_var`的变量，其值为`hello`，那么您必须将秘密存储为:

```
vault kv put airflow/variables/my_var value=hello
```

注意，秘密`Key`是`value`，秘密`Value`是`hello`！

## 步骤 4:访问气流 Dag 中的连接

您可以使用以下代码来访问作为 Vault 机密存储的 Airflow 连接，并观察其详细信息:

```
import json 
import logging from airflow.hooks.base_hook import BaseHook conn = BaseHook.get_connection('secret_name') 
logging.info(     
    f'Login: {conn.login}'     
    f'Password: {conn.password}'     
    f'URI: {conn.get_uri()}'     
    f'Host: {conn.host}'     
    f'Extra: " {json.loads(conn.get_extra())}'   
    # ... 
)
```

请注意，将首先搜索存储库机密，然后是环境变量，最后是 metastore(即通过 Airflow UI 添加的连接)。这种搜索顺序是不可配置的。

## 步骤 5:访问气流 Dag 中的变量

同样，以下代码片段将帮助您检索存储在 Vault 中的变量值:

```
import logging from airflow.models import Variable my_var = Variable.get('var_name') 
logging.info(f'var_name value: {my_var}')
```

## 步骤 6:使用 DAG 测试存储集成

在下面的代码片段中，您可以找到一个示例 Airflow DAG，您可以使用它来测试 Airflow 是否可以从`airflow.cfg`文件中指定的 Vault 中正确读取变量和连接。

```
import logging 
from datetime import datetime from airflow import DAG 
from airflow.models import Variable 
from airflow.hooks.base_hook import BaseHook 
from airflow.operators.python_operator import PythonOperator def get_secrets(**kwargs):

    # Test connections   
    conn = BaseHook.get_connection(kwargs['my_conn_id'])
    logging.info(
        f"Password: {conn.password}, Login: {conn.login}, "
        f"URI: {conn.get_uri()}, Host: {conn.host}"
    )            # Test variables     
    test_var = Variable.get(kwargs['var_name'])
    logging.info(f'my_var_name: {test_var}')with DAG(   
    'test_vault_connection',    
    start_date=datetime(2020, 1, 1),    
    schedule_interval=None
) as dag:      
    test_task = PythonOperator(         
        task_id='test-task',         
        python_callable=get_secrets,         
        op_kwargs={
            'my_conn_id': 'connection_to_test',
            'var_name': 'my_test_var',
        },     
    )
```

## 最后的想法

在今天的教程中，我们展示了如何配置 HashiCorp Vault，以便被 Apache Airflow 用作主要的后端秘密存储。然后，我们演示了如何向保险库添加连接和密码。

最后，我们通过几个实例演示了如何以编程方式从 Vault 中检索连接和变量，并在 Airflow DAGs 中使用它们。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**相关文章你可能也喜欢**

[](/run-airflow-docker-1b83a57616fb)  [](/connect-airflow-worker-gcp-e79690f3ecea)  [](/environment-variables-python-aecb9bf01b85) 