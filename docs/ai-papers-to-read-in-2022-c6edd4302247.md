# 2022 年要读的 AI 论文

> 原文：<https://towardsdatascience.com/ai-papers-to-read-in-2022-c6edd4302247>

## 阅读建议让你了解人工智能和数据科学的最新和经典突破。

![](img/1eab1a258eaef89617f9ca0a3e2a3db3.png)

照片由[阿尔方斯·莫拉莱斯](https://unsplash.com/@alfonsmc10?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com/?utm_source=medium&utm_medium=referral) 拍摄

今天我回到我的人工智能论文推荐系列，带给你十个新的人工智能论文建议(和更多链接)。我的老读者应该记得第一个、[、第二个](/ten-more-ai-papers-to-read-in-2020-8c6fb4650a9b)和第三个列表。现在，近两年后，第四部分来了。对于本系列的新手来说，这是一个非常固执己见的列表，你可能不同意所有的观点。有鉴于此，我尽力概述为什么你应该关心每一篇论文，以及它们在大计划中的位置。

在论文之前，我需要为所有阅读这篇文章的研究人员做出一个关于 AI 的声明:*我们不需要更大的模型；我们需要解决方案。*虽然公司已经达到万亿参数大关，但在医疗人工智能或缓解解决新问题所需的数据要求等实际问题上，几乎没有任何进展。出于这个原因，不要指望我在这里建议 GPT 废话。

最后但并非最不重要的是，作为一个小小的免责声明，我的大部分工作都围绕着计算机视觉，所以可能有很多关于强化学习、图形和音频等主题的优秀论文，而我并不知道。如果有任何你认为我应该知道的论文，请在评论中告诉我，❤.

我们走吧:

# #1 深度学习不是你需要的全部(2021)

> 施瓦兹-齐夫、拉维德和阿米泰·艾蒙。[“表格数据:深度学习不是你需要的全部。”](https://arxiv.org/abs/2106.03253) *信息融合*81(2022):84–90。

今年是 AlexNet 十周年。从那以后，深度学习变得比人工智能本身更加突出，机器学习现在听起来很过时，对 A*搜索没有概念的数据专业人士的数量不断增长。尽管如此，老式的机器学习技术仍然和以往一样与许多任务相关。

在这篇论文中，作者展示了 XGBoost 如何在各种表格数据集上匹配或超越深度学习解决方案，无论有无调整。另外，它显示了自动调优的 XGBoost 分类器比未经调优的分类器好得多。

**理由一:** AI 不是深度学习。远不止如此。特别是如果你是一个初学数据的科学家，要对经典技术给予应有的尊重，比如线性和逻辑回归、决策树、支持向量机和助推器。

**原因 2:** 在现实生活中，人们很容易忽略自动化调优方法如何在高效模型上创造奇迹，比如 XGBoost。在本文中，作者使用了 HyperOpt 的贝叶斯优化，在基线的基础上提高了约 30%。也许你也该学点[远视](http://hyperopt.github.io/hyperopt/)了:)

**花絮:**你知道 XGBoost 是 2014 年发布的吗？它几乎和 TensorFlow 一样古老，但比神经网络要新得多。

**延伸阅读:**关于助推器，一个很好的读物是最初的 [AdaBoost](https://www.sciencedirect.com/science/article/pii/S002200009791504X) 论文(1997)，它为后来的大多数集成方法奠定了基础。是我[第二张单子](/ten-more-ai-papers-to-read-in-2020-8c6fb4650a9b)的第二次建议阅读。关于简单模型击败复杂模型，另一个令人兴奋的阅读是[快速文本线性文本分类器的分析和优化](https://arxiv.org/abs/1702.05531)

# #2 面向 21 世纪 20 年代(2022 年)的 ConvNet

> 刘、庄等[“面向 21 世纪 20 年代的网络”](https://arxiv.org/abs/2201.03545) *arXiv 预印本 arXiv:2201.03545* (2022)。

虽然卷积神经网络(CNN)超越了以前的许多计算机视觉文献，但视觉变形金刚(vit)正在努力超越由 CNN 征服的空间。在这方面，ViTs 还没有这样做的普遍同意的原因是它们的计算成本，这仍然是一个公开的问题。

本文表明，经过仔细调整和训练的 ResNet 模型可以在 ImageNet、COCO 和 ADE20k 上匹配或超越 Transformers。换句话说，CNN 可能会存在更长时间。作者将其现代化的 ResNet 命名为“ConvNeXt”。

理由 1: 这是一篇非常实用的论文。几乎所有对 ResNet 的更改都可以扩展到其他模型。尤其是第 2.6 节，非常具有可操作性，今天就能给你结果。

**原因二:**对变形金刚颇有炒作。然而，这些论文不仅仅是关注。本文展示了如何将这些元素移植到乏味的旧模型中。

**原因 3:** 跟#1 一样，时髦的模型可能不是你任务的最佳模型。事实上，关于计算机视觉，ResNet 可能仍然是最安全的赌注。

**花絮:**如果你曾经想知道在 CNN 之前哪些算法是流行的，那么 [ILSVRC 2012](https://image-net.org/challenges/LSVRC/2012/results.html) 第二名使用了 [SIFT](en.wikipedia.org/wiki/Scale-invariant_feature_transform) 等等。

**延伸阅读:**尽管 ConvNeXt 可以说更好，但关于 [Vision](https://arxiv.org/abs/2010.11929) 和 [Swin Transformers](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html) 还是值得一读的。其他有趣的阅读肯定会出现在未来的列表中的是 [MLP](https://arxiv.org/abs/2105.01601) 和[康米克斯](https://arxiv.org/abs/2201.09792)。顺便说一句，这并不是第一家尝试翻拍 ResNet 的报纸。这里是[另一个例子](https://arxiv.org/abs/2110.00476)。

# #3 变压器调查(2021 年)

> 林，天阳，等[《变形金刚概览》](https://arxiv.org/abs/2106.04554) *arXiv 预印本 arXiv:2106.04554* (2021)。

从 2020 年到 2022 年，比以往任何时候都有更多的地球资源被塑造成人工智能突破。如今，说“全球变暖需要关注”是一种奇怪的讽刺。尽管如此，在这个不断变化的领域，通过及物性，对最热门话题的调查是最热门的论文。

**理由 1:** 尽管有 N 个复杂性，变形金刚还是在这里呆着。对于任何数据专业人员来说，了解自然语言处理(NLP)的最新进展都是非常重要的。

**原因 2:** 截至发稿时，还没有 X-former 被广泛采用作为原始 Transformer 的继任者，尽管有人声称[线性复杂度](https://arxiv.org/abs/2006.04768)。今天，任何设计神经网络的人都有兴趣看看作者迄今为止是如何试图提高注意力的。

**花絮:** [为什么这么多 AI 系统都以布偶命名？](https://www.theverge.com/2019/12/11/20993407/ai-language-models-muppets-sesame-street-muppetware-elmo-bert-ernie)

**延伸阅读:**这篇论文之后，一个自然的后续就是 2022 ICLR 的[视觉变形金刚是如何工作的？](https://paperswithcode.com/paper/how-do-vision-transformers-work-1?from=n26)然而，如果你想处于新闻的边缘，我强烈推荐阅读带有代码简讯的[论文。感谢](https://paperswithcode.com/newsletter)[大卫·佐丹奴](https://medium.com/u/25b62fff1eb0?source=post_page-----c6edd4302247--------------------------------)在一年多前的评论中向我推荐这份时事通讯[。从那以后，它成了我最喜欢的新闻来源之一。](/how-i-stay-updated-on-the-latest-ai-research-b81203155551)

# 第四名 SimCLR (2020 年)

> 陈，丁等.[“视觉表征对比学习的一个简单框架”](https://arxiv.org/abs/2002.05709) *机器学习国际会议*。PMLR，2020 年。

到目前为止，所有提到的论文都解决了监督学习:学习将 X 映射到 y。然而，整个世界都致力于一个“无 y”的世界:无监督学习。更详细地说，这个领域处理没有明确答案的问题，但是可以获得有用的答案。例如，我们可以通过几种方式对一组客户进行聚类:性别、年龄、购买习惯等。，我们可以根据这些集群制定有利可图的营销策略。

在这篇论文中，作者简化了现有的关于对比学习的文献来创建 SimCLR。该方法显示出产生更好的下游结果，同时比竞争方法简单得多。从某种意义上来说，你可以把这项工作理解为相当于 [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) 的视觉——一种从大型图像语料库中提取有用特征的系统化方法。

**理由 1:** 如果你仔细想想，大部分人类学习都是无人监督的。我们不断地观察这个世界，并根据我们所看到的画出图案。我坚信任何对普通智力的突破都将带有相当大的无人监管的成分。因此，关注这一领域与人工智能领域的几乎所有人都有关系。

**原因二:**过去十几年 NLP 的突破都来自于无监督的预训练。到目前为止，类似的革命还没有出现在图像中。关注这个话题的另一个原因。

**琐事:**尽管非监督算法没有“y”，但大多数技术都像监督方法一样优化成本函数。例如，超分辨率模型最小化了原始图像和它们的缩减采样重建之间的重建误差。

**延伸阅读:**有许多有趣的无人监管的问题。下面是一个不详尽的列表: [GANs](https://paperswithcode.com/method/gan) ，[风格转移](https://paperswithcode.com/task/style-transfer)，[图像超分辨率](https://paperswithcode.com/task/image-super-resolution)，[聚类](https://en.wikipedia.org/wiki/Cluster_analysis)，[异常检测](https://en.wikipedia.org/wiki/Anomaly_detection)，[关联规则挖掘](https://en.wikipedia.org/wiki/Association_rule_learning)等。继续对比学习主题， [SimCLRv2](https://arxiv.org/abs/2006.10029) 是一个自然的后续。

# 第五名效率网(2019)

> 谭，明星和郭乐。“效率网络:反思卷积神经网络的模型缩放” *机器学习国际会议*。PMLR，2019。

手动调整神经网络经常感觉像玩乐高。你可以添加/删除层和神经元，玩激活功能，调整训练计划等。大多数情况下，我们的行为是任意的，比如将事物翻倍或减半，或者坚持 2 和 10 的幂。

在这项工作中，Tan 和 Quoc 研究了一种更具原则性的方法，即使用神经结构搜索(NAS)来放大和缩小网络。他们的研究发现，当深度、宽度和分辨率一起缩放时，可以获得最佳结果。此外，他们发布了一组预训练模型，从微型到超大，实现了最先进的结果。

**理由 1:** 本文是一个很好的例子，说明自动化调优策略(如 NAS 和贝叶斯优化)如何比手动调优模型更具成本效益。此外，您可以轻松控制您的调优预算。

**理由二:**谨防只有“畸形”设定的机型。稳健模型总是可以从小到大扩展，同时保持最先进的重要性。EfficientNet 就是一个很好的例子。

**原因 3:** 最近，无论是在视觉还是 NLP 任务上，主干架构的重要性都在持续增长。虽然我们仍然看到头部网络[ [1](https://github.com/ultralytics/yolov5) 、 [2](https://arxiv.org/abs/2004.01177) 、 [3](https://paperswithcode.com/task/object-detection) ]的进步，但很难说真正的收益来自哪里。

**花絮:**供参考，根据代码为的[论文，ImageNet 上当前](https://paperswithcode.com/sota/image-classification-on-imagenet) [Top-1 最先进的](https://paperswithcode.com/paper/coatnet-marrying-convolution-and-attention)在约 24 亿个参数下的准确率为 90.88%。至于 Top-5，[目前为止最好的模型](https://paperswithcode.com/paper/florence-a-new-foundation-model-for-computer)排名 99.02%。

**延伸阅读:**虽然对最大的语言模型 IMHO 存在激烈的竞争，但对高效但强大的模型的讨论要有趣得多(也更具包容性)。早期的名字包括 [MobileNet](https://arxiv.org/abs/1704.04861) 、 [ShuffleNet](https://arxiv.org/abs/1807.11164) 和 [SqueezeDet](https://arxiv.org/abs/1612.01051) ，而最近的冠军是 [Conv 混合器](https://arxiv.org/abs/2201.09792)。

# #6 推动窄精度的极限(2020 年)

> Darvish Rouhani，Bita，et al. [“用微软浮点运算推动云级窄精度推理的极限”](https://proceedings.neurips.cc/paper/2020/hash/747e32ab0fea7fbd2ad9ec03daa3f840-Abstract.html) *神经信息处理系统进展*33(2020):10271–10281。

继续速度的话题，在本文中，微软的研究人员展示了微软浮点格式(MSFP)的有效性，这是一种用于高效大规模浮点数学的共享指数方法。虽然仅限于推理工作负载，但所呈现的架构清楚地显示了专用单元如何能够显著扩展我们当前的容量。

然而，我并不是因为这篇论文的优点*才把它加入列表的*。相反，*我的意思是提高对正在进行的窄精度革命的认识*。市场上有[谷歌 TPU](https://cloud.google.com/tpu)、 [AWS 推理](https://aws.amazon.com/machine-learning/inferentia/)、 [Nvidea TensorCores](https://www.nvidia.com/en-us/data-center/tensor-cores/) ，还有无数创业公司在定制 AI 芯片上工作。虽然科学进步很大，但我预计未来五年的成果将完全由硬件驱动。

**原因 1:** MLOPs 是当今商业 AI 最关键的方面之一。实现高精度固然不错，但是在保持低成本的同时无缝部署模型却是另一个挑战。

**理由二:**今天的研究就是明天的现实。我们现在都使用混合精度训练。我不知道接下来我们会使用哪种 floats，但是我建议留意所有的候选人。

**花絮:**与此同时，Meta 正在搭建[六千 GPU 集群。](https://ai.facebook.com/blog/ai-rsc/)

**延伸阅读:**本次讨论感兴趣的是模型[量化和修剪](https://arxiv.org/abs/2101.09671)的主题。总的来说，这些技术旨在减少模型的大小和开销，以提高效率和降低成本。边缘设备(例如，移动电话)对这些技术特别感兴趣。

# #7 我们真的需要乘法吗？(2020)

> 陈、韩婷等[“AdderNet:深度学习真的需要乘法吗？."](https://arxiv.org/abs/1912.13200)*IEEE/CVF 计算机视觉和模式识别会议论文集*。2020.

在这篇论文中，Chen *等人*表明，标准卷积层可以由更简单的基于加法的方法来代替，只有边际精度损失(< 2%)。当卷积将特征与滤波器相乘时，所提出的方法计算绝对差。实际上，这相当于激活是特征和过滤器之间的 L1 距离。

直观上，卷积计算特征与过滤器的相关程度(即，它们是否共享相同的符号)。当我们改变绝对差的乘法时，我们有效地计算特征和过滤器之间的相似性。如 3.1 节末尾所述，这可以理解为执行模板匹配。

理由 1: 如果无乘法网络不能激发你的兴趣，我不知道还有什么能激发你的兴趣。像这样激进的想法总是值得一读。

**原因 2:** 也许所有投入到更快的浮点数学和硬件矩阵乘法上的努力可能都是徒劳的。

**花絮:**CNN 背后的原始灵感可以追溯到 1979 年，并被命名为 [NeoCognitron](https://en.wikipedia.org/wiki/Neocognitron) 。然而，CNN 直到 2012 年[才在它的许多](https://en.wikipedia.org/wiki/AlexNet)[理论](https://en.wikipedia.org/wiki/Backpropagation)和[实践](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units)问题得到解决之后才开始崭露头角。

**延伸阅读:**虽然有大量工作在寻找卷积替代方案，但也有大量人致力于用[替代反向传播](https://proceedings.neurips.cc/paper/2021/hash/feade1d2047977cd0cefdafc40175a99-Abstract.html)。简而言之，反向传播本质上是顺序的——我们一次训练一层。如果我们有一种算法可以并行训练所有层，我们会更好地使用当前的计算集群。

# 第八名 StyleGAN 3 (2021)

> Karras，Tero，et al. [《无别名生成对抗网络》](https://proceedings.neurips.cc/paper/2021/hash/076ccd93ad68be51f23707988e934906-Abstract.html) *神经信息处理系统进展* 34 (2021)。

贯穿本文的一个不变主题是深度学习中的任何事情都可能改变或可能是错误的——甚至乘法也不安全。最初的 StyleGAN 论文从根本上改变了我们在 GAN 上使用噪声的方式。在第三次迭代中，作者邀请我们重新解释数据如何通过网络流动，而不是作为一个离散的实体，而是作为一个连续的信号。

简而言之，他们认为边界填充、上采样和 ReLU 等操作要么会泄漏位置数据，要么会给信号带来不必要的高频。实际上，这使得生成器学习不必要的偏差，例如始终将眼睛和鼻子放在相同的坐标上。通过改造这些元素，作者改进了平移和旋转的网络均衡性。

**原因 1:** 我们知道 CNN 是空间不变的，因为平面卷积是空间不变的。有时候事情比我们意识到的要复杂。这提醒我们应该如何灵活地对待我们所有的信仰。

**理由 2:** 所有这三篇 StyleGAN 论文都是关于数据可视化和表示的生动的一课。单从视觉效果来看，它们是值得打开的。

**原因 3:** 最近的生成模型，如 StyleGAN，显示了人工智能对于新颖的图形应用程序是多么成熟。也许这篇论文能启发你想出一些创业点子。

**琐事:**你可以使用[外伏发生器](https://waifulabs.com/generate)来体验甘斯的强大力量(或者浏览[此人不存在](https://this-person-does-not-exist.com/en))

**延伸阅读:**几年前，只有 GANs 能做出好看的作品。如今，变分自动编码器(VAEs)赶上来了，但其他方法也出现了。下面是对[深度生成模型(2021)](https://arxiv.org/abs/2103.04922) 的最新综述。

# #9 人工智能中的透明度和再现性(2020 年)

> 本杰明·海贝-卡恩斯等人[《人工智能中的透明性和再现性》](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8144864/) *性质* 586.7829 (2020): E14-E16。

这篇文章不是一般的研究论文。相反，这是一封公开信，谴责谷歌的乳腺癌人工智能团队，比如说，不完整的论文。总之，谷歌声称它开发了一种用于乳腺癌筛查的人工智能，明显优于最先进的*和人类放射科医生*。然而，无论是代码还是数据集都没有发布，这让国际社会对这项工作产生了怀疑。此外，乳腺癌是女性死亡的主要原因之一，因此这篇论文引起了媒体的广泛关注。

由于这篇论文，原作者最终发布了文章的[附录。他们进一步描述了技术方面和](https://www.nature.com/articles/s41586-020-2767-x) [OPTIMAM 数据集](http://commercial.cancerresearchuk.org/optimam-mammography-image-database-and-viewing-software)，该数据集包括研究中使用的英国数据集。然而，他们的答复证明大部分代码严重依赖内部基础设施，不能公开。

**原因 1:** 光有名气和大成绩是不够的。透明度和再现性很重要。确保你比谷歌做得更好。

**原因 2:** 得知谷歌同意发表原始论文，让人们对导致 [Timnit Gebru](https://en.wikipedia.org/wiki/Timnit_Gebru) 被谷歌解雇的所谓内部审查过程有了一些了解。

**理由三:**这篇论文是关于礼貌地批评别人工作的一课。引用摘要，“方法细节和算法代码的缺乏破坏了它的科学价值。"

**延伸阅读:**关于这个讨论，阅读[原论文](https://www.nature.com/articles/s41586-019-1799-6)和作者随后的[回复](https://www.nature.com/articles/s41586-020-2767-x)可以感兴趣。快进到 2022 年，虽然作者纠正了大多数问题，但最初的考虑不应被忘记:*透明度和再现性是最重要的。*

# 智力指标排名第十(2019)

> *弗朗索瓦，Chollet。* [*“论智力的衡量”*](https://arxiv.org/abs/1911.01547) *arXiv 预印本 arXiv:1911.01547 (2019)。*

虽然我之前列表中的大多数建议仍然适用，但这篇论文尤其值得重复，不是因为它具体的名气或结果，而是因为它带来的问题。在我们努力获得更精确的模型的同时，整个 AI 场景从更广泛的追求开始:*什么是智能，我们如何人工创造它？*

我相信我们还没有触及这个问题的表面。我们连智力是什么、如何测量都无法达成一致，更不用说意识等更深刻的概念了。在这篇论文中，弗朗索瓦·乔莱(Franç ois Chollet)尽力开发出可操作的定义，并为我们提供了 [ARC](https://github.com/fchollet/ARC) ，一种学习算法的智商测试。

引用论文中的话，*“为了朝着更智能、更像人类的人工系统稳步前进，我们需要遵循适当的反馈信号。”*

**理由#1:** 虽然数据科学很酷也很时髦，但人工智能才是真正的东西。如果没有人工智能，就不会有数据科学。而它的终极目标并不是去发现对数据的洞察。取而代之的是制造能有自己想法的机器。所以，花点时间思考一些基本问题:什么是智力，我们如何衡量它？这篇论文是一个良好的开端。

**原因#2:** 在过去的几十年里，IA 社区被来自数理逻辑和演绎推理的思想所主导。然而，在没有任何形式的显式推理的情况下，支持向量机和神经网络在该领域的发展远远超过了基于逻辑的方法。ARC 会引发经典技术的复兴吗？

**原因 3:** 如果 Chollet 是正确的，我们仍然需要几年的时间来创建算法来解决 ARC 数据集。如果你正在寻找一个可以在业余时间玩的数据集，[这里有一个](https://github.com/fchollet/ARC)可以让你忙起来:)

**花絮:**以防你不知道，[Fran ois Chollet](https://fchollet.com/)是 Keras 和神奇的[深度学习用 Python](https://www.manning.com/books/deep-learning-with-python) 书背后的人。

**延伸阅读:**2018 年，Geoffrey Hinton、Yosha Bengio 和 Yan LeCun 因其在深度学习基础方面的开创性工作获得了图灵奖。2020 年，他们在 AAAI 会议上分享了他们对人工智能未来的看法。[你可以在这里看](https://www.youtube.com/watch?v=UX8OubxsY8w)。从深度学习的角度来看，它已经有将近两年的历史了，就像十年前一样。然而，看到他们当时的想法并将其与今天已经做的事情联系起来是很有趣的。

T21:他的一切都是为了现在。如果你对这篇文章或报纸有任何问题，请随时评论或与我联系。你也可以订阅我在这里发布[的时候通知你](https://ygorserpa.medium.com/subscribe)。

写这样的清单是一项很大的工作，所以，如果这是一篇值得阅读的文章，请与你的同龄人分享，或者在评论中给我一个阅读建议。谢谢大家！

如果你是中新，我强烈推荐[订阅](https://ygorserpa.medium.com/membership)。对于数据和 IT 专业人士来说，中型文章是 [StackOverflow](https://stackoverflow.com/) 的完美搭档，对于新手来说更是如此。注册时请考虑使用[我的会员链接。](https://ygorserpa.medium.com/membership)你也可以通过[请我喝杯咖啡](https://www.buymeacoffee.com/ygorreboucas):)来直接支持我

感谢阅读:)