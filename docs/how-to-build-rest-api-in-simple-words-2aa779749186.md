# 如何用简单的话构建 REST API

> 原文：<https://towardsdatascience.com/how-to-build-rest-api-in-simple-words-2aa779749186>

## [行业笔记](https://pedram-ataee.medium.com/list/notes-from-industry-265207a5d024)

## 作为数据科学家使用 Flask 和 Docker 创建 REST API 的行动手册

![](img/63e0fd4dd1f55ebcb1b54fe996627b91.png)

约书亚·厄尔在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的[胡佩尔的照片](https://unsplash.com/@huper?utm_source=medium&utm_medium=referral)

我最近开发了一个名为 [OWL](https://rapidapi.com/pedram.ataee/api/word-similarity/) 的**上下文化的最相似单词** API 服务。OWL API 是使用 Flask 和 Docker 开发的高级 NLP 服务。该服务通过托管在 RapidAPI 市场上的一系列 REST APIs 与社区共享。这对我来说是一个伟大的旅程，所以我决定与你分享它发展的各个阶段。

本文描述了如何使用 Flask 和 Docker 构建 REST API 的分步指南。您可以在您的数据科学项目以及您正在开发的任何其他 web 应用程序中使用该指南。另外，这篇文章是**“…用简单的话”**系列的一部分，在这里我与你分享我的经验。这个系列之前帮助了很多数据科学爱好者。希望对你也有帮助。

*   [如何在 Kubernetes 上简单部署](https://medium.com/p/aba6f42cc888)
*   [如何用简单的话学习 Git](https://medium.com/p/263618071dd8)
*   [如何用简单的词语学习 circle ci](https://medium.com/p/2275e4299628)
*   [用简单的话深度学习](https://medium.com/p/448e2c7f6ebe)

如果您想阅读更多关于 OWL API 的内容，请查看下面的文章。

</how-to-compute-word-similarity-a-comparative-analysis-e9d9d3cb3080>  

## 步骤 1 —创建 app.py

首先，您必须编写一个 Python 脚本，将 RESTful 端点映射到一个函数。我们给剧本取名`app.py`。在这个例子中，RESTful 端点是`/general/<N>/<WORD>` ，其中`general`是一个常量，`N`和`WORD`是两个变量。通过这些变量，用户可以将数据发送到 RESTful 端点的相应函数，即`similar_words(N, WORD)`。在服务器上部署后，您在 Python 脚本中定义的 RESTful 端点将被添加到主机的 URL 中。

```
import os
import sys
from flask import Flask, jsonify, request
from flask_httpauth import HTTPBasicAuthimport gensim.downloader as api
sys.path += [os.getcwd()]
from src.core.word_clusterizer import WordClusterizer
from src.backend.wrapper import GensimWrapperauth = HTTPBasicAuth()
app = Flask(__name__)MODEL = api.load('glove-wiki-gigaword-300')**@app.route('/general/<N>/<WORD>', methods=['GET'])**
**def similar_words(N, WORD):** model = GensimWrapper(MODEL)
    try:
        wc = WordClusterizer(model=model, word=WORD, top_n=int(N)
        wc.fit()
        results = wc.result
    except KeyError:
        results = 'Nothing is found!'
    **return jsonify(results)**if __name__ == '__main__':
    port = os.environ.get("PORT", 8080)
    **app.run(host='0.0.0.0', port=port, debug=True)**
```

不用说，RESTful 端点的相应功能是您的数据科学项目的核心。在这种情况下，它是一个单词相似性服务，与当前的解决方案相比，它以更大的粒度提取最相似的单词。

## 步骤 2 —创建 Dockerfile 文件

您需要一个 Docker 映像，在启动服务器时运行`app.py`。要构建这样的 Docker 映像，您需要一个如下所示的 Docker 文件。

```
# MAINTAINER YOUR_NAMEFROM **pdrm83/nlp-essentials:latest**RUN pip install --upgrade pip
COPY . /project/
RUN pip install .**WORKDIR /project/src/.**
ENV PYTHONPATH /project
ENV PORT 8080**EXPOSE 8080**
EXPOSE 80**CMD ["python3", "app/app.py"]**
```

在 docker 文件中，您必须打开/暴露设置`app.py`运行的端口。正如您在步骤 1 中看到的，我将`app.py`脚本设置为在定义为`port = os.environ.get("PORT", 8080)`的端口上运行。这一行意思是，从环境中把`port`变量设置为`PORT`的值，如果设置了，否则设置为 8080。

在这种情况下，`PORT`没有预先设置。因此，我必须确保在 docker 文件中打开端口 8080。我通过 Dockerfile 文件中的`EXPOSE 8080`做到了这一点。如果`PORT`先前在环境中被设置为除 8080 之外的任何值，您必须通过添加`EXPOSE XXXX`在 docker 文件中打开该端口。

</how-to-create-an-ubuntu-server-to-build-an-ai-product-using-docker-a2414aa09f59>  

注意，我在名为`nlp-essentials`的 Docker 映像创建的环境中运行`CMD ["python3", "app/app.py"]`。这是我之前通过安装所有需要的库构建的 Docker 映像。这张 Docker 图片保存在我的 Dockerhub 账户`pdrm83`中，需要时使用。每当我想更新 API 服务时，就会拉出`nlp-essentials`映像。这有助于在我每次想要更新 API 时不安装所有需要的库。最后，但同样重要的是，由于我将`app.py`存储在`ROOT/src/app/.`中，所以我将`WORKDIR /project/src/.`添加到 Dockerfile 中。你不能复制这一行。你只需要根据你的代码库的文件结构在 Dockerfile 中正确地配置它。

</how-to-build-an-automated-development-pipeline-d0b9820a2f3d>  

## 步骤 3 —启动服务器

简而言之，您必须使用`docker build -f <<DOCKERFILE_NAME>> -t <<IMAGE_NAME>>`构建 docker 映像，然后使用`docker run -i -p 80:8080 <<IMAGE_ID>>`运行它。`IMAGE_ID`是您在上一步中创建的 Docker 图像的 id。注意，我将端口 8080 映射到 80，因为`app.py`在端口 8080 上运行，我想在端口 80 上为用户提供 API 服务。最后，您必须确保端口 80 在您使用的任何机器上都是开放的。

有很多方法可以让用户访问你的服务。不过，我想给你介绍一个名为 [**ngrok**](https://ngrok.com/) 的很酷的服务，它可以让你通过三个简单的步骤让你的电脑上网**😮。**这 3 个步骤简单如下:

1.  下载`ngrok.zip`和`unzip /path/to/ngrok.zip`。
2.  运行`ngrok authtoken YOUR_OWN_TOKEN`
3.  运行`ngrok http 80`

运行`ngrok http 80`之后，ngrok 服务在`ngrok.io`域上为您的项目创建一个专用的 URL，看起来像下面的链接。

```
[http://YOUR_DEDICATED_SUBDOMAIN.ngrok.io/](http://d9500fdc2530.ngrok.io/)
```

你的服务现在在互联网上对每个人都是可利用的。你不需要任何花哨的云服务就能把你的 API 服务推向市场。这不是很神奇吗？你可以在下面的文章中读到为什么我认为 ngrok 对你向社区提供 API 服务非常有用。

</how-to-avoid-cloud-bill-shock-in-data-science-services-e2ade5fae2a8>  

## 遗言

软件产品最值得推荐的架构之一是 **API 架构**。这个架构帮助你设计**模块化**和**可重用**服务，这些服务也**易于维护**。例如，有一些有效的解决方案来测试和记录一系列 REST APIs，帮助您轻松地维护它们。在本文中，我描述了如何为您的数据科学项目构建一个 **RESTful API** 。除了 RESTful 之外，还有其他类型的 API 协议，这不是本文的重点。

## 感谢阅读！

如果你喜欢这个帖子，想支持我…

*   *跟我上* [*中*](https://medium.com/@pedram-ataee) *！*
*   *在* [*亚马逊*](https://www.amazon.com/Pedram-Ataee/e/B08D6J3WNW) *上查看我的书！*
*   *成为* [*中的一员*](https://pedram-ataee.medium.com/membership) *！*
*   *连接上*[*Linkedin*](https://www.linkedin.com/in/pedrama/)*！*
*   *关注我* [*推特*](https://twitter.com/pedram_ataee) *！*

<https://pedram-ataee.medium.com/membership> 