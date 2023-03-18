# Elasticsearch Python 研讨会#1

> 原文：<https://towardsdatascience.com/elasticsearch-python-workshop-1-741fa8b9aaf1>

## 基础知识

![](img/24a1a1a77f09efee566277c69814bd32.png)

戴维·克洛德在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

欢迎来到 Elasticsearch Python 研讨会的第一部分。本系列将从 Python 程序员的角度来关注这个问题，我希望我能对这个小型生态系统有所了解，这个生态系统是用 Python 客户端和工具从 Elasticsearch 集群中获取额外数据而构建起来的。特别是当你是一个数据科学家时，这个系列可能会节省你一些时间。像所有的研讨会一样:为了保持文章简洁，我把代码剪成了片段。但是你可以从 [GitHub repo 为这个工作坊](https://github.com/PascalThalmann/ElasticPythonWorkshop/tree/master/1_the_basics)下载完整的代码。话虽如此，让我们直入主题吧。

我在 [Elasticsearch 迁移 6.22 到 7.8.0](https://cdax.ch/2020/10/10/elasticsearch-migration-6-2-2-nach-7-8-0/) (抱歉，这篇文章只有德语版)中使用了 python 库 [Python Elasticsearch 客户端](https://elasticsearch-py.readthedocs.io/en/7.10.0/index.html)用于 reindex 任务。为了节省您的时间，我们从安装和一些基本的配置设置开始。这可能会更方便，因为 Elasticsearch 8.0.0 增强了安全性——在没有任何安全性的情况下连接到本地主机是默认不启用的。但是首先，让我们安装 Elasticsearch 客户端。下面的例子是使用 Ubuntu 18.04 LTS。

# 安装库

使用 pip3 安装最新的 Elasticsearch 客户端:

```
sudo apt update && sudo apt upgrade
sudo apt install python3-pip
sudo python3 -m pip install 'elasticsearch>=7.0.0,<8.0.0'
sudo python3 -m pip install elasticsearch_dsl
```

# 创建您的第一个连接

让我们从简单开始，假设您的集群是 pre-Elasticsearch=8.0.0，并且您没有实现安全性。如果没有将节点暴露给网络，请连接到“localhost”而不是主机名。

```
es = Elasticsearch(["[http://srvelk:9200](http://srvelk:9200)"])
es.cat.nodes()
```

就是这样。没有安全措施很简单，对吧？好了，让我们看看在启用安全性时需要做些什么。

# 创建启用安全性的连接

为了通过安全集群建立连接，我们需要再安装一个 pip 模块(可能已经安装了，只要确保您已经安装了):

```
sudo python3 -m pip install urllib3
```

您可能还没有证书，但您至少需要一个用户名/密码。您可以像这样设置连接:

```
from ssl import create_default_context
from elasticsearch import Elasticsearches = Elasticsearch(["[https://username:password@srvelk:9200](https://username:password@srvelk:9200)"], verify_certs=False)
es.cat.nodes()
```

这将是可行的，但会产生一些令人不快的警告。我们最好使用证书，所以如果您没有证书，让我们创建一个 pem-certificate。

# 生成 pem 证书

如果您使用默认设置安装了 Elasticsearch 8，您可能不知道集群的 SSL-Keystore 密码。但是温和地说，Elastic 提供了直接从密钥库中读取密码的工具，但是您需要 root 权限:

```
/usr/share/elasticsearch/bin/elasticsearch-keystore \
   show xpack.security.http.ssl.keystore.secure_password
```

这是将证书添加到密钥库中所需的“导入密码”。我们现在使用 OpenSSL 创建证书 python_es_client.pem:

```
openssl pkcs12 -in /etc/elasticsearch/certs/http.p12 \
   -cacerts -out /etc/elasticsearch/certs/python_es_client.pem
```

出现提示时，输入密码。选择您选择的 PEM 密码。之后，将 pem 文件复制到 python 脚本可以访问的位置。将您的证书存储在/tmp 中并不是一个好的选择，这只是出于演示的目的。

```
chmod 666 /etc/elasticsearch/certs/python_es_client.pem
cp /etc/elasticsearch/certs/python_es_client.pem /tmp
```

让我们用 Python 创建另一个连接:

```
from ssl import create_default_context
from elasticsearch import Elasticsearchcontext = create_default_context(cafile='/tmp/python_es_client.pem')
es = Elasticsearch(["[https://username:password@srvelk:9200](https://username:password@srvelk:9200)"], ssl_context=context)
es.cat.nodes()
```

最好不要在脚本中包含您的用户名和密码。我们需要的是一个 API 密钥，并在没有凭据的情况下连接到集群。

# 使用 API 密钥连接

我们需要首先创建一个角色。转到堆栈管理->角色并创建一个角色。该角色必须至少包含“管理 api 密钥”权限

![](img/5b423afccf2ab1a062e4425cb3097c32.png)

作者图片

现在向用户添加或创建一个角色。转到堆栈管理->用户并添加角色:

![](img/01ba227cfe540b70eadc087f6a0f1ba7.png)

作者图片

现在生成 API 密钥。转到堆栈管理-> API 密钥，并为需要通过 API 密钥访问的用户创建 API 密钥。请在创建后将 API 密钥保存在安全的地方:

![](img/bcfab55550fbcb6e2f433ee70adb2d83.png)

作者图片

让我们用 Python 创建另一个连接:

```
from ssl import create_default_context
from elasticsearch import Elasticsearchapi_key='aC1wNUYzOEJCWV...RSjJMaEhvbDMyWElvZw=='
context = create_default_context(cafile='/tmp/python_es_client.pem')
es = Elasticsearch(["[https://srvelk:9200](https://srvelk:9200)"], ssl_context=context, api_key=api_key)
es.cat.nodes()
```

这个看起来更好🙂

# 结论

如果你成功了:祝贺你！现在，您应该能够创建与 Python Elasticsearch 客户端的连接了。如有问题，请查阅官方文档或给我留言。你也可以和我联系或者关注[链接。](https://www.linkedin.com/in/pascal-thalmann/)

如果谷歌把你带到这里，你可能还会检查这个系列的其他[部分](https://pascalth.medium.com/list/python-elasticsearch-workshop-947054f07bd9)

*原发布于*[*https://cdax . ch*](https://cdax.ch/2022/02/20/elasticsearch-python-workshop-1-the-basics/)*。*