# 结合联邦学习和联邦平均算法的隐私保护深度学习

> 原文：<https://towardsdatascience.com/privacy-preserving-deep-learning-with-federated-learning-and-federated-averaging-algorithm-221588478759>

## 利用保护隐私的人工智能构建您的竞争优势——在不了解任何人的情况下向任何人学习——联合学习——数据科学

![](img/f70aea4ec72d376a562e19057c7f5728.png)

照片由[戴恩·托普金](https://unsplash.com/@dtopkin1?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/privacy?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

# 介绍

苹果最近宣布其 iPhone 收入同比增长 47%，2021 年第二季度售出 4926 万部智能手机[1]。对于 AAPL 持有者来说，这绝对是个好消息。但是对于我们这些数据科学和 AI 爱好者来说，这意味着什么呢？

好吧，让我们在这里做一些计算。平均而言，每个人每秒产生大约 1.7 兆字节的数据[2]，人们通常每天花大约 3 个小时在手机上。这意味着 iPhones 每天产生 848.46 PB(PB)的数据，比脸书每天产生的数据量(约 4 PB)大 200 倍。请记住，这只是仅来自 iPhones 的数据量。

虽然这些数字并不精确，但您现在会了解这些数据有多大，以及为什么这些数据会成为大数据的新来源。正如您可能看到的，这种丰富的数据源超过了任何一个组织所能容纳的信息量。这些数据可用于创建个性化体验，有助于提高用户对服务或应用的满意度。因此，世界各地的大组织都在试图利用这种新资源来增加现有产品的附加值，或者围绕它们创建新的商业模式。

![](img/56c8c4f9572f03209f2a9982e1ab32d6.png)

照片由[丹·尼尔森](https://unsplash.com/@danny144)在 [Unsplash](https://unsplash.com/photos/ah-HeguOe9k) 上拍摄

然而，使用这种类型的信息会带来许多挑战。与由组织集中存储和管理的传统大数据源不同，移动设备上的数据通常是大规模分布的，由用户而不是服务提供商拥有。由于用户拥有这些数据，出于隐私考虑，他们有权在自己的设备上保持这些数据的私密性，因此，这是希望利用这些数据的服务提供商面临的最大挑战。一方面，企业需要更多数据来更好地为客户服务，但另一方面，私有数据无法共享。

拉斯维加斯 CES 2019 上的苹果广告牌

处理隐私和敏感信息并不容易，解决这个问题最简单的方法就是不要接触它。你宁愿不使用这些数据在服务中添加新功能，也不愿侵犯用户隐私，这绝对是一个有效的观点。然而，如果你的竞争对手能够以某种方式利用这些数据构建人工智能应用，在不违反任何隐私限制的情况下提供更好的个性化服务，那么他们将在你失去竞争优势的同时获得竞争优势。实现这种能力的技术将是一种颠覆性的技术。

> 现在觉得服气了？让我们继续读下去！

好吧，那么现在你想在这个数据金矿的基础上建立一个人工智能模型，而不把它们从用户的设备中取出来。当您考虑数据隐私和相关保护时，加密是最流行的方法之一，其中数据可以用用户的私钥加密，然后发送到服务器。然而，加密也意味着你不能再访问信息，因为你没有用户的私钥。如果你不能把金子放进石头里，石头就只是石头。

解决方案确实比你想象的要简单。您可以将模型训练发送到用户的设备上，而不是将所有数据加载到服务器上进行模型训练，这些数据属于用户的设备，并且用户可以完全控制这些数据。这是联邦学习的核心思想。

# 联合学习

联合学习是人工智能系统架构的一种新范式，它本质上是在客户端设备上训练模型，因此不会发送任何私人数据。联合学习的训练过程通常包括四个步骤:

*   **步骤 1:** 开始在参与联合学习的所有设备上使用它们的本地数据训练模型。
*   **步骤 2:** 设备将它们本地训练的模型发送到服务器。请注意，上传的是模型，而不是私有数据。没有私人数据会离开设备。
*   **步骤 3:** 服务器通过使用聚合算法来组合从设备接收的所有模型。研究人员开发了许多聚合算法，但最受欢迎的算法是 2017 年由 **Google** 发布的联邦平均算法。
*   **步骤 4:** 服务器将聚合模型分发给所有设备。然后，聚集的模型可以用于提供 AI 功能，或者为下一轮训练做好准备(如果需要)。

![](img/aabec8421f6585732dbc19ce7b54ee82.png)

联合学习过程(由我创建的图)

> 在不了解任何人的情况下向每个人学习

聚集算法在联合学习中起着重要的作用，因为它负责组合来自所有设备的知识，而不知道用户的私人数据。在本文中，我将只解释联邦平均(Federated Averaging，FedAvg ),因为它是一种简单而有效的算法，正被用于 Google 产品的生产中(尽管我正在研究一种可能击败 FedAvg 性能的新算法)。

联邦平均算法通过对所有模型更新进行*加权*平均来生成聚合模型。为了帮助您更好地理解这一点，我将提供一个忽略算法的“加权”部分的过于简化的示例。给定一个有三个设备的系统，下面是它们的模型是如何聚合的。

```
- Device A sends model A with value **0.6** to the sever
- Device B sends model B with value **0.1** to the sever
- Device C sends model C with value **0.2** to the sever
- Server takes the average of these three values: **(0.6 + 0.1 + 0.2)/3 =** **0.3** - Server sends aggregated model with value **0.3** to A, B and C
```

就是这样！算法很简单，但把它们放在一起，这个想法已经被谷歌证明是“可生产的”，当他们在数十亿台 Android 设备上使用它作为他们的谷歌键盘产品的下一个单词预测功能时。他们的自然语言处理(NLP)模型超级准确，因为它从来自多种语言和文化的数十亿用户那里学习。但是，这样做的好处是不会收集任何输入数据，并且尊重用户隐私。这是您可能想要的数据产品的竞争优势。

你可以看看谷歌首席执行官桑德尔·皮帅在这段视频中对联合学习的介绍:

谷歌主题演讲—联合学习—谷歌 I/O 2019

# 应用和挑战

联合学习是一种新兴技术，由于其巨大的潜力，被世界各地的许多组织采用、研究和开发。人们可以使用联合学习为医院建立一个超级强大的诊断人工智能模型，同时保留患者的隐私。人们还可以利用街道上真实司机的驾驶行为来训练自动驾驶汽车。还可以建立基于个人数据的个性化推荐系统，以推荐与客户需求精确匹配的产品或服务(目前大多数推荐系统都是以产品为中心，而不是以客户为中心)。

就这项技术的成熟度而言，谷歌已经在他们的产品中使用了一段时间。此外，有许多由大公司支持的联邦学习框架，以便您在采用这项技术时不会从头开始。我自己试过的两个框架是 Intel 的 OpenFL 和 Google 的 TensorFlow Federated。这些框架提供了易于使用的编程接口，因此您的数据科学家只需一小段代码就可以构建和部署他们的第一个联邦学习应用程序。

> 每一个伟大的机遇都伴随着巨大的挑战。

然而，最大的挑战通常不是来自技术本身，而是来自实现技术解决方案的产品的总体设计。

当谷歌和苹果在 Android 和 iOS 中应用一项新技术时，默认情况下，你是作为用户选择加入的。他们从来不会问你这样的问题:“你能让我们访问你的隐私和敏感数据以帮助改善你的体验吗？我们保证不会将您的数据发送出去”。当你看到这个弹出窗口的时候，我猜你会立刻按下“不，谢谢”。

![](img/364d6f21663086b89de9dd2b831205ea.png)

我的 iPhone 截图

由于谷歌和苹果拥有操作系统，用户没有选择，除非他们停止使用设备。然而，作为一家在操作系统上开发应用程序的公司，你没有这种奢侈。每次应用程序需要访问用户的数据时，即使数据不会上传到其他地方，它仍然需要特别请求许可。想象一下，如果你必须向你的客户解释“我们只发送模型”部分以获得他们的许可，你会意识到这不仅是不可能的，而且你的组织的声誉可能会受到影响，因为你听起来像一个间谍公司，除非你像脸书那样做(你知道我的意思)。否则，你的产品或服务必须有足够的价值，让用户愿意“妥协”他们的数据，即使你的意图是将数据保存在他们的设备上。当我使用“妥协”这个词时，我表达的是用户的想法，而不是服务提供商的意图。例如，每个人都在使用脸书时做出妥协，因为他们都知道它正在收集他们的数据，但他们仍然继续使用它。

虽然客户的同意对 B2C 来说是一个挑战，但对 B2B 来说是一个更容易解决的问题。两家或两家以上的公司可以就共享数据达成协议，这对各方都有利，事实上，这种方法目前相当流行。现在有了联合学习，他们甚至可以共享数据，而不用实际发送出去。例如，互联网服务提供商(ISP)可以与银行交换其用户的互联网流量数据，以构建一些高级人工智能应用程序，为双方带来价值。

ISP 可以使用自己的数据构建需求预测模型，这些数据由来自银行的交易数据丰富。一个家庭的支出和他们的互联网使用之间可能有联系。这方面的一个例子可能是，一个家庭刚刚访问了一家家庭娱乐零售店(销售平板电脑、笔记本电脑和游戏机的商店)，并支付了价值约 3000 美元的款项。这也可能意味着家庭现在有更多的智能设备，因此他们将需要更多的互联网带宽。预测模型可以帮助 ISP 向客户提供更好的计划，并在基础设施规划中更加主动。

另一方面，银行可以利用 ISP 提供的互联网流量数据丰富的交易数据，为其机构客户(例如零售店)建立更好的现金流预测模型。特定网站的流量和拥有该网站的公司下周的收入之间可能存在相关性。例如，ISP 报告本周地址[www.theretailstore.com](http://www.theretailstore.com)的流量增加，银行可以将此信息与该零售店的历史交易数据结合起来，预测下周的收入，因为客户可能会在购买产品之前在网站上寻找信息。

我可以在这里列出更多的应用和想法，潜力是无限的。因此，无论是 B2B 还是 B2C，您可能都需要考虑您将从以前由于隐私问题而无法访问的数据中获得的价值，并投资于这项技术，以便获得与竞争对手相比的竞争优势。每一个巨大的机会都伴随着巨大的挑战，那些能够管理这些挑战所带来的风险的人将会获得机会并扰乱市场。

# 参考

1.  [苹果销量不及预期，蒂姆·库克称供应问题让公司损失 60 亿美元](https://www.cnbc.com/2021/10/28/apple-aapl-q4-2021-earnings.html)
2.  [2021 年每天创造多少数据？](https://techjury.net/blog/how-much-data-is-created-every-day/#gref)
3.  [联合学习:没有集中训练数据的协作机器学习](http://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
4.  [与谷歌的联合学习](https://federated.withgoogle.com/)
5.  [英特尔开放式联合学习](https://github.com/intel/openfl)
6.  [TensorFlow Federated:对分散数据的机器学习](https://www.tensorflow.org/federated)
7.  [从分散数据进行深度网络的通信高效学习](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)