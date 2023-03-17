# 知识图嵌入——一个简化版本

> 原文：<https://towardsdatascience.com/knowledge-graph-embedding-a-simplified-version-e6b0a03d373d>

## 解释什么是知识图嵌入以及如何计算它们

![](img/a76677475717a914a6dc676dbd8d7595.png)

纳斯蒂亚·杜尔希尔在 [Unsplash](https://unsplash.com/s/photos/network?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

raphs 是我最喜欢使用的数据结构之一。它们使我们能够表示复杂的现实世界网络，如快速交通系统(如巴黎地铁、纽约地铁等)、地区或全球航空交通，或类似个人社交网络的相关事物。它们非常灵活，很容易被人类理解。但是，为了让计算机“理解”它们并从中“学习”，我们还额外增加了一步(称为矢量化)。这个解释可能太简单了，但是对于理解和理解这篇文章的其余部分来说已经足够了。

# 知识图的特别之处是什么？

![](img/b4f7f5f03ea34c31c8086571316fd59f.png)

照片由[乌列尔 SC](https://unsplash.com/@urielsc26?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/network?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

为了容易理解知识图与其他图有何不同，我们来讨论一个类比。想象一个有不同关卡的电子游戏，每一关都随着你的进展变得越来越难。

**一级**:可以是一个简单的无向图，像大学里的朋友群，朋友是节点，友谊是边。这里我们只有节点和边，没有什么太花哨的。

**第二层**:在前一层的基础上增加一层信息，如方向感，得到有向图。一个简单的例子是城市范围的公共汽车网络。将公共汽车站视为节点，将公共汽车的路线视为边。在这里，每辆公共汽车沿着特定的方向从一个车站开往另一个车站。这增加了方向信息。

第三级:我们取一个有向图，给节点和边添加多种风格。为了容易理解这一点，想象一下互联网上的社交网络。节点上的多种口味是用户所基于的社交网络的类型。例如，它可以是推特、脸书或 YouTube。边上的味道可以是不同用户之间的交互类型，即关注(在 Twitter 的情况下)，朋友或关注(在脸书的情况下)，以及订阅(在 YouTube 的情况下)。图形的有向性开始发挥作用，因为以下内容只能是单向的。例如，你可以关注埃隆·马斯克，但他可能不会在 Twitter 上关注你，这就是优势的方向性。

**第 4 层**:考虑前一层中的图，但是你不用节点和边，而是用三元组的语言来说话。三元组是知识图的构建块，其中它是由 3 个元素组成的元组，即:源节点(头)、关系和目标节点(尾)。

> 注意:源节点和目标节点有时被称为实体。

请注意术语“知识图”的使用有点模糊。我想说的是，知识图没有固定的定义，从广义上讲，我们可以将任何包含一些知识/重要信息的相当大的图命名为知识图。

> 这里的关键要点是，我们将知识图的元素视为三元组。

# 知识图嵌入方法

![](img/f661cdd44091ab102fe8417155a5068f.png)

照片由[像素](https://www.pexels.com/photo/close-up-photography-of-yellow-green-red-and-brown-plastic-cones-on-white-lined-surface-163064/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)的 [Pixabay](https://www.pexels.com/@pixabay?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) 拍摄

> 概述:矢量化或嵌入(图形的实体和关系的数字表示)对于将图形用作机器学习算法的输入是必要的。

因为我们对待知识图的方式不同于其他的，所以我们需要不同的技术来学习它们的数字表示(或嵌入)。

有几种方法可以生成知识图嵌入(KGE)，我们可以将其大致分为 3 个部分:

1.  **基于翻译的方法:** 这里，基于距离的函数(在欧几里德空间中)用于生成嵌入。我们可以建立一个简单的算法，即
    使头部和关系向量的组合等于尾部向量。
    可以表示为 ***h + r ≈ t*** 。这种算法叫做 TransE。同样的算法还有其他版本，但是对它的修改很少。一些例子包括 TransH、TransR、TransD、TranSparse 和 TransM。
2.  **基于因式分解的方法:** 这种基于张量因式分解的思想，使用这种技术提出的初始算法被重新标度。三向张量以`n x n x m`的形式定义，其中 n 是实体的数量，m 是关系的数量。张量的值为 1，表示实体之间存在关系，如果没有关系，则为 0。
    嵌入通过分解该张量来计算。对于大型图形来说，这通常计算量很大。这个问题是由 DistMult、HolE、ComplEx 和 QuatE 等算法解决的，这些算法都是建立在 RESCAL 思想的基础上的。
3.  **基于神经网络的方法:** 神经网络现在在许多领域都很流行，它们被用于发现知识图嵌入并不令人惊讶。语义匹配能量是一种算法，该算法通过使用神经网络来定义用于将值分配给三元组的能量函数。神经张量网络使用能量函数，但是它用双线性张量层代替了神经网络的标准线性层。【ConvE 等卷积神经网络以“图像”的形式重塑实体和关系的数字表示，然后应用卷积滤波器来提取特征，从而学习最终的嵌入。我们还可以找到 GAN 启发的模型，如 KBGAN 和基于 Transformer 的模型，如 HittER，来计算知识图嵌入。

为了实现这些算法，我们有多个 python 库，例如:

1.  [LibKGE](https://github.com/uma-pi1/kge)
2.  [PyKEEN](https://pykeen.readthedocs.io/en/stable/)
3.  [石墨](https://github.com/DeepGraphLearning/graphvite)
4.  [放大图](https://docs.ampligraph.org/en/1.4.0/)

# 一种 KGE 算法的结构

![](img/7f2fae616fb72ebec5b1a6d47471edcd.png)

照片由 [Pixabay](https://www.pexels.com/@pixabay?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) 从[像素](https://www.pexels.com/photo/abstract-business-code-coder-270348/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)拍摄

构建计算知识图嵌入的算法有一些共同的基本思想。下面列出了其中的一些想法:

1.  负生成:
    这是一个在知识图中生成负的或损坏的三元组的概念。负三元组是不属于原始图的三元组。这些可以随机生成，也可以通过使用一些策略(如伯努利负采样)来生成。
2.  评分函数:
    包装三元组的函数，吐出一个值或一个分数。我们可以定义，如果分数高，则三元组有效，如果分数低，则为负三元组。评分函数是构造 KGE 算法的重要组成部分之一。
3.  损失函数:
    由于该算法是根据优化问题建模的，所以我们在训练过程中使用损失函数。这个损失函数使用正三元组和负三元组的分数来计算损失。我们的目标是最小化损失，我们也采用了优化器。
    这里使用的损失函数的一些例子是——交叉熵损失、基于成对余量的铰链损失等。

# 生成嵌入后的下一步是什么？

![](img/562e880fc603ceff6e3bea1cbc0eda3d.png)

照片由 [Ann H](https://www.pexels.com/@ann-h-45017?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) 从 [Pexels](https://www.pexels.com/photo/pink-jigsaw-puzzle-piece-3482441/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) 拍摄

学习 KGE 算法并应用它们来寻找嵌入是很有趣的。现在，接下来是什么？嵌入的用途是什么？

有一些图表下游任务可以应用于知识图表，例如:

**知识图完成:**
这也被称为链接预测，我们的目标是预测知识图中缺失的关系或潜在可能的关系。它也可以被称为知识图扩充。
这个任务可以归结为找到一个实体，这个实体可以与一个给定的关系和一个实体一起被最好地表示为一个事实。简单来说，就是现在猜测`(?, r, t)`或`(h, r, ?)`中缺失部分的任务也可以分别称为头部预测或尾部预测。我们使用基于等级的评估技术来发现我们的知识图嵌入的性能。

**三元组分类:** 这是一个识别给定三元组是否有效的问题，即它是正三元组还是负三元组。该任务的输出仅仅是是或否。
采用评分函数，并设置阈值以将正三元组与负三元组分开。基本上，现在这是一个二元分类的问题。

推荐系统是一个可以使用知识图嵌入的领域。嵌入的质量对于上述任务的性能和准确性很重要。这些任务的结果告诉我们是否能够生成高质量的嵌入。

> 注意:如果我在这里解释的任何概念对您来说有点沉重，那么我建议您查看我写的关于图形的其他文章。

如果你看到了这篇文章的这一部分，感谢你的阅读和关注。

```
**Want to Connect?**Reach me at [LinkedIn](https://www.linkedin.com/in/rohithteja/), [Twitter](https://twitter.com/rohithtejam), [GitHub](https://github.com/rohithteja) or just [Buy Me A Coffee](https://www.buymeacoffee.com/rohithteja)!
```