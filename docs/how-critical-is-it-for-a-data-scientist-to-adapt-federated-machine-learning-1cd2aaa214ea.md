# 数据科学家适应联邦机器学习有多重要？

> 原文：<https://towardsdatascience.com/how-critical-is-it-for-a-data-scientist-to-adapt-federated-machine-learning-1cd2aaa214ea>

## 意见

## 联邦机器学习:一个新的机器学习婴儿

![](img/055bf3e50ef9c3fcb64e95cd603f29e5.png)

照片由 [DeepMind](https://unsplash.com/@deepmind?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/federated-learning?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

谷歌在 2016 年引入了联合学习这个术语，以标志着 ML 范式中新的机器学习方法的开始。联合学习解决了集中式和分布式训练方法的许多缺点。如果没有联邦学习的使用，我们就不会看到像谷歌助手中的“嘿谷歌”这样的高度改进的设备上机器学习模型。为了理解联合学习及其在当今物联网世界中的重要性，让我首先描述现有模型的缺点。

# 集中式/分布式培训

机器学习的概念始于集中训练。你将整个数据集上传到中央服务器，使用中央服务器训练机器学习模型，进行评估/测试，最后将其部署在中央服务器上。然后，客户端需要与远程部署的模型进行通信，以便对其数据进行推断。这种方法有严重的缺点。您需要将全部数据上传到服务器，即使对于中等规模的企业来说，目前的数据量也可能达到数 TB。这导致参与者和云服务器之间的高通信开销。将数据上传到云服务器还会损害数据隐私，从而违反 GDPR(一般数据保护法规)。另一方面，这种方法的优点是参与者从需要大量资源的计算责任中解脱出来。

分布式训练使用专门设计的机器学习算法来解决在大型数据集上操作的复杂算法中的计算问题。与集中训练中使用的算法相比，这些算法是高效的和可扩展的。多个参与者进行独立训练。中央服务器将这些训练集合起来，创建一个最终模型。我们在一个中央服务器上做收敛测试，如果准确度差，我们重复整个过程，直到我们达到可接受的准确度。您会看到整个方法需要大量的云资源。我们在云上部署最后一个模型，客户端必须连接到这个远程模型，以便对其数据进行推断。这种培训方法仍然没有解决数据隐私问题。

所以，联邦学习来了。

# 联合学习

联合学习采用了一种与我之前描述的传统培训有很大不同的培训方法。在这种方法中，通常从在集中式服务器上创建通用机器学习模型开始。这个初始模型充当所有参与进一步培训的参与者的基线。参与者从中央服务器下载基本模型。每个参与者通过使用其本地数据来改进该模型。作为一个小的集中更新，参与者将更改发送到服务器。通信使用同态加密或任何其他安全传输协议，确保数据隐私。我们不向服务器发送用户数据，我们只发送模型更新(神经网络中的权重)。

然后，服务器汇总所有参与者的学习成果，以改进基本模型。我们立即丢弃更新，以确保绝对的数据隐私。服务器将改进的模型发送给参与者用于进一步的个性化。我们反复运行这个过程来改进模型。在每次迭代中，服务器模型变得更好，对每个参与者来说更加个性化。

现在，让我们试着分析这种方法是如何克服早期模型的缺点的。

# 联合学习的好处

正如您从联合学习的工作中看到的，所有的训练数据都保留在您(参与者)的设备上。我们不会在云中存储单独的更新。事实上，谷歌为他们的联合学习使用了一个[安全聚合](http://eprint.iacr.org/2017/281)协议。它使用加密技术，允许协调服务器解密 100 或 1000 个用户的平均更新。该协议不允许检查单个更新。这为参与者提供了完全的隐私。

显然，该过程本身允许创建动态更新的更智能的模型。随着我们不断与用户分享更新的模型，您将在使用您的设备(例如手机)时看到个性化。传统的分布式算法要求低延迟和高吞吐量。联合学习的本质不会导致这些需求。在服务器上聚集模型所需的功耗和计算资源也大大减少。

# 一些缺点

仅提一个缺点，在联邦学习中，数据分布高度不均匀。请注意，数百万台设备处理自己的个人数据。这些设备具有高延迟和低吞吐量的连接。不仅如此，这些设备只能间歇性地用于训练。为了解决这些问题，谷歌开发了[联邦平均](https://arxiv.org/abs/1602.05629)算法，这一领域的其他公司也是如此。另一个挑战是将这项技术部署到数百万台异构设备上。你需要在每台设备上安装一个缩小版的 TensorFlow 或任何类似的机器学习框架。

科技巨头们已经成功地解决了这些问题，这种技术有许多实际的使用案例，我将在下面讨论。

# FL 使用案例

正如我前面提到的，谷歌助手使用联邦学习来不断改进“嘿谷歌”。设备上的培训通过不向云共享个性化语音消息来确保数据隐私。雅虎。百度、谷歌和 DuckDuckGo 都使用联合搜索索引来获得更准确的结果。苹果还使用联邦机器学习来改进他们移动设备上的模型，从而保护你的数据隐私。微软研究院推出了 [FLUTE](https://github.com/microsoft/msrflute) (联邦学习实用工具和实验工具)——一个运行大规模离线联邦学习模拟的框架。谷歌在 Android 上的[Gboard](https://blog.google/products/search/gboard-now-on-android/)使用 FL 作为下一个单词预测器。

# 数据科学家有什么？

跟随这些科技巨头的脚步，数据科学家现在可以将联合学习应用于许多潜在的行业。

例如，医疗保健和健康保险行业肯定可以从联合学习中受益。在这些行业中，数据隐私要求更加严格。联合学习保护敏感数据，提供更好的数据多样性——这通常在这些行业中观察到。数据来自医院、健康监测设备等等。数据多样性有助于诊断罕见疾病，并提供早期检测。

想象一家电子商务公司，如果他们有来自银行的客户消费能力和模式，他们将受益于更好的产品购买预测。当然，银行和电子商务是两个永远不会共享数据的垂直行业。然而，使用联合学习，两者都可以基于他们偏好的特征训练他们自己的 ML 模型；然后，我们可以由第三方管理人汇总模型参数，这最终将使双方消费者受益。

对于自动驾驶汽车来说，联合学习可以为乘客提供更好、更安全的体验。该模型可以利用交通和道路状况的实时数据进行学习。

还有许多行业，如金融科技(金融和技术)、保险以及物联网，都可以从 FL 中受益。数据科学家需要深入这些领域，为自己寻找更多业务。

有几个联合学习平台可供您开始您的 FL 之旅。Google 提供了一个基于 Tensorflow 的联邦学习平台——它被称为 [Tensorflow federated](https://www.tensorflow.org/federated) 。英特尔发布了一个用于联合学习的开源框架，名为 [openfl](https://pypi.org/project/openfl/) 。然后是 IBM [FL 平台](https://www.ibm.com/docs/en/cloud-paks/cp-data/4.5.x?topic=models-federated-learning)。英伟达[克拉拉](https://developer.nvidia.com/blog/federated-learning-clara/)助力联邦学习。你可以搜索到更多。

# 结束语

如您所见，联邦学习克服了传统和分布式机器学习的许多缺点。重要的是，它提供用户数据保护，这是当今技术消费者的需求。FL 可以轻松处理数据多样性。它允许实时连续学习，并且对硬件资源不太费力。FL 开辟了许多跨领域的新模型开发机会。那么，你还在等什么？

<https://medium.com/@profsarang/membership>  

# 信用

[Pooja Gramopadhye](https://www.linkedin.com/in/pooja-gramopadhye-31aa6b1a0/) —文字编辑