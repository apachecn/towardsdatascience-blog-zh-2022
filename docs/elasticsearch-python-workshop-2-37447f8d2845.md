# Elasticsearch Python 研讨会#2

> 原文：<https://towardsdatascience.com/elasticsearch-python-workshop-2-37447f8d2845>

## API、类以及如何调用它们

![](img/daa065c05cae1f10e1fa1075684ca145.png)

戴维·克洛德在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

欢迎来到 Elasticsearch Python 研讨会的第二部分。本系列将从 Python 程序员的角度来关注这个问题，我希望我能对这个小型生态系统有所了解，这个生态系统是用 Python 客户端和工具从 Elasticsearch 集群中获取额外数据而构建起来的。特别是当你是一个数据科学家时，这个系列可能会节省你一些时间。像所有的研讨会一样:为了保持文章简洁，我把代码剪成了片段。但是你可以从 [GitHub repo 下载这个工作坊的完整代码。话虽如此，我们还是开始吧。](https://github.com/PascalThalmann/ElasticPythonWorkshop/tree/master/2_classes_and_api)

让我们创建一个简单的连接。如果您使用的是 Elasticsearch 版本 8.0.0 及以上，请检查[首次研讨会](https://cdax.ch/2022/02/20/elasticsearch-python-workshop-1-the-basics/)以创建安全连接。

```
from elasticsearch import Elasticsearch
es = Elasticsearch(["[https://username:password@srvelk:9200](https://username:password@srvelk:9200)"], verify_certs=False)
```

# 弹性搜索类

因为我们的连接是用 Elasticsearch 类创建的，所以最基本的 API 调用可以作为方法直接从这个类中获得。这些方法直接列在模块[弹性搜索](https://elasticsearch-py.readthedocs.io/en/v8.0.0/api.html#module-elasticsearch)下:

![](img/748ec13eda657ca31bd12c677d19fde8.png)

作者图片

# 创建索引和文档

让我们用“create”方法创建一个索引和一个文档。来自[官方文件](https://elasticsearch-py.readthedocs.io/en/v8.0.0/api.html#module-elasticsearch):

![](img/283586e35aaaaaae5d8ce25492e80e56.png)

弹性搜索图片

这个调用看起来像这样:

```
response = es.create(index='testidx', id=1, document='{"item": "content"}')
```

让我们看看 API 调用的响应:

```
print(json.dumps(response, indent=4))
{
    "_index": "testidx",
    "_id": "1",
    "_version": 1,
    "result": "created",
    "_shards": {
        "total": 2,
        "successful": 1,
        "failed": 0
    },
    "_seq_no": 0,
    "_primary_term": 1
}
```

我们可以测试状态:

```
if response['result'] == 'created':
   print("document created")
```

# 得到一份文件

让我们使用 get 方法来读取文档:

```
response = es.get(index='testidx', id=1)
print(json.dumps(response, indent=4))
{
    "_index": "testidx",
    "_id": "1",
    "_version": 1,
    "_seq_no": 0,
    "_primary_term": 1,
    "found": true,
    "_source": {
        "item": "content"
    }
}
```

这与你在 curl 或 Kibana 中得到的结果是一样的:

![](img/37978be046cc5b22e5d87f0ce5979763.png)

作者图片

# 搜索文档

搜索 API 的调用如下所示:

```
es.search(index="testidx", body={
  "query": {
     "match_all": {}
   }
 })
```

# 使用 CatClient 类(和任何其他类)

当你看一看官方文档时，你会发现这个图书馆有丰富的课程。然而，对于初学者来说，如何调用它们可能会有点困惑。

例如，CatClient(或弹性术语中的“_ cat”API)的调用是:

![](img/0c77f6afa348ff4a28e4a3668b03f59e.png)

作者图片

![](img/c173f152baef9e2d2cc26de5738e61ea.png)

您可以导入整个 elasticsearch 库，并使用现有的“es”对象(您已经建立的连接)作为参数。然后调用方法，在本例中是 node()方法:

```
>>> elasticsearch.client.cat.CatClient(es).nodes()
'192.168.1.68 44 85 1 0.07 0.04 0.01 cdfhilmrstw * srvelk8\n'
```

看起来很奇怪，对吧？还有一个更好的方法:导入类“elasticsearch。Elasticsearch”包括 elasticsearch-library 的所有类作为对象。让我们看看在该类的 __dict__ 属性中发现了什么:

```
print(es.__dict__['cat'])
<elasticsearch.client.cat.CatClient object at 0x7ff874e24160>
```

如果用参数“cat”调用 Elasticsearch 类，基本上就是调用 CatClient 类。这样，您可以用相当短的语法调用 CatClient 类的 nodes 方法。

```
>>> es.cat.nodes()
'192.168.1.68 44 85 1 0.49 0.22 0.08 cdfhilmrstw * srvelk\n'
```

# 代表弹性搜索 API 的弹性搜索属性表

以下是作为对象存储在 __dict__ 中的属性(左)和类(右)的汇编:

![](img/349b4899576176f5528ba04c6061629c.png)

作者图片

# 结论

如果你成功了，恭喜你！现在，您应该能够使用 Python Elasticsearch 客户端的所有类，因此也能够使用最常用的 Elasticsearch APIs。如有问题，请查阅官方文档或给我留言。你也可以和我联系或者关注[链接。](https://www.linkedin.com/in/pascal-thalmann/)

如果谷歌把你带到这里，你可能也会检查[系列](https://pascalth.medium.com/list/python-elasticsearch-workshop-947054f07bd9)的其他部分。下一部分再见！

*原发布于*[*https://cdax . ch*](https://cdax.ch/2022/02/24/elasticsearch-python-workshop-2-objects-classes-and-how-to-call-them/)*。*