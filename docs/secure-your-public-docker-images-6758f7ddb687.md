# 保护您的公共 Docker 图像

> 原文：<https://towardsdatascience.com/secure-your-public-docker-images-6758f7ddb687>

## 如果计算机病毒应该算作生命，那么我们已经按照自己的形象创造了生命。

![](img/a110e93d741b8311e5200e8325130a70.png)

照片由[通风视图](https://unsplash.com/@ventiviews?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

您刚刚完成了新的闪亮工具的实现，并准备将其打包到 Docker 容器中，并将其推送到公共存储库中供他人查找。你会怎么做？好吧，如果你的大脑说执行`docker push`命令，那你就错了！

你如何公证你的身份和你的 docker 图片的完整性？谁能证明他们下载的图片是你的呢？为什么有人要相信你？

当把你的图片推送到一个注册中心，比如 Docker Hub，你的信息会通过一个不可信的媒介:互联网。因此，以某种方式验证 Docker 映像的完整性是至关重要的。

Docker 内容信任(DCT)使您能够验证您正在使用的图像的完整性以及该图像发布者的完整性。因此，DCT 充当反欺骗控制。

[](https://medium.com/geekculture/the-docker-attack-surface-5184a36a23ca) [## 码头工人攻击面

### 什么会出错？

medium.com](https://medium.com/geekculture/the-docker-attack-surface-5184a36a23ca) 

> [学习率](https://www.dimpo.me/newsletter?utm_source=medium&utm_medium=article&utm_campaign=docker-notary)是为那些对 AI 和 MLOps 的世界感到好奇的人准备的时事通讯。你会在每个月的第一个星期六收到我关于最新人工智能新闻和文章的更新和想法。在这里订阅！

# Docker 公证服务器

DCT 的第一个组件是 Docker 公证服务器。这个组件就像一个人类公证人，一个被授权在法律事务中执行行为的独立的人。在我们的案例中，Docker 图像就像一个需要公证的法律文件。

为此，您应该获得一个签名密钥，并将其委托给公证服务器来实施图像签名。让我们先把理论搞清楚，看看如何准确地公证任何 Docker 图像。

## 用 DCT 签名图像

要在 Docker Hub 上推送 Docker 图像之前对其进行签名，您需要部署您的公证服务器并获得签名密钥。

首先，您可以使用两种方法获得签名密钥:

*   使用命令`docker trust key generate NAME`生成它
*   从证书颁发机构(CA)获取它

一旦您有了签名密钥，我们需要部署公证服务器。Docker 公司提供了 Docker 公证服务器，您可以使用`docker-compose`快速部署它。

接下来，我们需要将密钥委托给 Docker 公证服务器来签署图像。在这一步中，您需要将公钥和私钥都添加到公证服务器。

最后，我们需要强制客户端使用 DCT。这意味着当您通过 Docker CLI 使用`push`和`pull`命令时，您只能从注册表中推送和提取已经使用 DCT 签名的映像。

## 例子

在本演示中，您将设置 DCT 公证服务器，生成签名密钥，并在将 Docker 映像推送到 Docker Hub 之前对其进行签名。让我们首先生成签名密钥。要获取密钥，请运行以下命令:

```
docker trust key generate NAME — dir ~/.docker/trust
```

`NAME`变量可以是你想要的任何东西。这个我会用`medium_demo`。因此，该命令变成了:

```
docker trust key generate medium_demo — dir ~/.docker/trust 
```

如果`~/.docker/trust`路径不存在，您可以通过从您的`/home`目录运行以下命令来创建它:

```
mkdir -p .docker/trust
```

CLI 将要求您提供密码。你想用什么就用什么，但要确保你会记住它。如果一切顺利，您应该会看到以下消息:

```
Successfully generated and loaded private key. Corresponding public key available: /home/vagrant/.docker/trust/medium_demo.pub
```

接下来，您将打开公证服务器。为此，您需要克隆公证服务器 repo:

```
git clone [https://github.com/theupdateframework/notary.git](https://github.com/theupdateframework/notary.git)
```

Docker 公证人服务器是一个容器化的应用程序，您可以使用`docker-compose`来部署它。将您的工作目录更改为您克隆的`notary`目录，并运行以下命令:

```
docker-compose up -d
```

如果您的系统中没有安装“docker-compose ”,请按照[文档](https://docs.docker.com/compose/install/#install-compose-on-linux-systems)中的说明进行安装。

接下来，您将添加在本示例开始时生成的密钥。为此，您需要运行以下命令:

```
docker trust signer add — key PATH NAME ACCOUNT/REPO
```

让我们分解这个命令:

- `PATH`:存储签名密钥的目录。
- `NAME`:签名密钥名称
- `ACCOUNT`:您的 Docker Hub 账户名称
- `REPO`:您的 Docker Hub repo

在我的例子中，命令应该是:

```
docker trust signer add — key ~/.docker/trust/medium_demo.pub medium_demo dpoulopoulos/ubuntu-hardened
```

为新的根密钥和存储库密钥提供密码后，您应该会看到以下结果:

```
Successfully initialized “dpoulopoulos/ubuntu-hardened”
Successfully added signer: medium_demo to dpoulopoulos/ubuntu-hardened
```

接下来，您需要使用以下命令确保内容信任服务器指向公证服务器:

```
export DOCKER_CONTENT_TRUST_SERVER=[https://notary.docker.io](https://notary.docker.io)
```

我们现在准备签署一个图像，并将其推送到 Docker Hub。所以，首先，让我们得到一个图像。为此，您将使用官方的 Ubuntu 映像:

```
docker pull ubuntu
```

接下来，相应地标记图像。对我来说，这是:

```
docker tag ubuntu:latest dpoulopoulos/ubuntu-hardened:v1.0.0
```

通常，您应该遵循以下格式:

```
docker tag ubuntu:latest ACCOUNT/REPO:TAG
```

`ACCOUNT`和`REPO`变量应该与您之前使用的相同。`TAG`可以是你想要的任何东西，但是你必须指定一个。

接下来，在我的例子中，为了对图像进行签名，我需要运行以下命令:

```
docker trust sign dpoulopoulos/ubuntu-hardened:v1.0.0
```

最后，当我们推进和推送映像时，您需要在 Docker 守护进程上对所有映像操作强制使用 DCT。对于第一次运行:

```
export DOCKER_CONTENT_TRUST=1
```

对我来说，推动这个形象:

```
docker push dpoulopoulos/ubuntu-hardened:v1.0.0
```

恭喜您，您已经通过本地公证服务器签署了您的 Docker 图像，并将其推送到 Docker Hub！如果您想要检查图像中的 DCT 设置，您可以运行:

```
docker trust inspect — pretty ACCOUNT/REPO:TAG
```

# 结论

保护您的公共 Docker 映像是一项简单的任务，如果您要将这些映像投入生产，这也是一项必要的任务。当然，当您进入生产阶段时，您会希望从 CA 获得您的签名密钥，但是在此之后，过程仍然是相同的。

看起来可能涉及到很多步骤，但是大多数步骤只需执行一次。因此，没有什么可以阻止你签署你的图像，并告诉你的用户，他们可以信任你！

# 关于作者

我的名字是[迪米特里斯·波罗普洛斯](https://www.dimpo.me/?utm_source=medium&utm_medium=article&utm_campaign=docker-notary)，我是一名为[阿里克托](https://www.arrikto.com/)工作的机器学习工程师。我曾为欧洲委员会、欧盟统计局、国际货币基金组织、欧洲央行、经合组织和宜家等主要客户设计和实施过人工智能和软件解决方案。

如果你有兴趣阅读更多关于机器学习、深度学习、数据科学和数据操作的帖子，请关注我的 [Medium](https://towardsdatascience.com/medium.com/@dpoulopoulos/follow) 、 [LinkedIn](https://www.linkedin.com/in/dpoulopoulos/) 或 Twitter 上的 [@james2pl](https://twitter.com/james2pl) 。

所表达的观点仅代表我个人，并不代表我的雇主的观点或意见。