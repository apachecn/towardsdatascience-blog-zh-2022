# ChatGPT 的碳足迹

> 原文：<https://towardsdatascience.com/the-carbon-footprint-of-chatgpt-66932314627d>

## 意见

## 本文试图估算名为 ChatGPT 的流行开放式聊天机器人的碳足迹

![](img/08ba0d9932d57b2045b26ff15d6bd8b3.png)

沃洛季米尔·赫里先科在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

最近有很多关于 ChatGPT 的讨论，有些人谈论运行该模型的货币成本，但没有多少人谈论该模型的环境成本。

人类活动导致的大气中温室气体含量的增加是气候变化的主要驱动力[8] **。**信息和通信技术(ICT)行业和数据中心行业在全球温室气体排放中占有相对较大的份额[9]。

因此，我们——运行在数据中心的数字工具的用户和开发者——需要尽自己的一份力量来减少数字活动的碳足迹，从而缓解气候变化。

为此，最重要的是，我们要意识到，即使是数字产品也需要能源来开发和消费，因此它们会产生碳足迹。这篇文章有助于实现这一目标。此外，在我们讨论如何减少我们的碳足迹时，获取事实信息也很重要，这样我们就可以优先解决那些产生最大碳节约的问题。

最后，我希望这篇文章将激励机器学习模型的开发人员公开他们模型的能耗和/或碳足迹，以便这些信息可以与模型准确性度量一起用于评估模型的性能。

[](/chatgpts-electricity-consumption-7873483feac4) [## ChatGPT 的电力消耗

### ChatGPT 可能在 2023 年 1 月消耗了多达 17.5 万人的电力。

towardsdatascience.com](/chatgpts-electricity-consumption-7873483feac4) [](https://kaspergroesludvigsen.medium.com/chatgpts-electricity-consumption-pt-ii-225e7e43f22b) [## ChatGPT 的用电量，PT。二

### 对 ChatGPT 成本的估计支持了 ChatGPT 每月耗电量为 110 万至 2300 万千瓦时的估计

kaspergroesludvigsen.medium.com](https://kaspergroesludvigsen.medium.com/chatgpts-electricity-consumption-pt-ii-225e7e43f22b) 

# 大规模机器学习的环境成本

环境成本可能以各种形式出现，例如用水、土壤污染、空气污染。在本帖中，我们将从碳足迹的角度来看看 ChatGPT 对环境的影响。

当确定机器学习模型的碳足迹时，人们可以区分 a)来自训练模型的碳足迹，b)来自一旦被部署就用模型运行推理的碳足迹，c)模型的总生命周期碳足迹。要深入了解如何估计和减少机器学习模型的碳足迹，请参见[9]。

不管范围如何，计算任何模型的碳足迹都需要知道两件事:

1.  它消耗的电量
2.  这种电力的碳密度

# 1 在很大程度上取决于运行它的硬件以及硬件的利用率。

# 2 很大程度上取决于电力是如何产生的，例如太阳能和风能显然比煤更环保。为了量化这一点，我们通常使用硬件所在电网的平均碳强度。

[](/how-to-estimate-and-reduce-the-carbon-footprint-of-machine-learning-models-49f24510880) [## 如何估计和减少机器学习模型的碳足迹

### 轻松估算机器学习模型碳足迹的两种方法和如何减少碳足迹的 17 个想法

towardsdatascience.com](/how-to-estimate-and-reduce-the-carbon-footprint-of-machine-learning-models-49f24510880) 

# 来自培训聊天的碳足迹

如果我理解正确的话，ChatGPT 是基于 GPT-3 的一个版本。据估计，训练 GPT-3 消耗 1287 兆瓦时，排放 552 吨二氧化碳当量[1]。

然而，这些排放量不应仅归因于 ChatGPT，我不清楚如何将这些排放量的一部分分配给 ChatGPT。此外，ChatGPT 已经使用强化学习[2]进行了训练，这应该被添加，但是关于这个训练过程的相关信息是不可用的，并且我不知道任何合理的代理。如果是的话，请给我指出正确的方向。

# 运行 ChatGPT 的碳足迹

现在，让我们看看使用 ChatGPT 运行推理可能会产生多少 CO2e。我没有遇到任何关于# 1 和# 2 wrt 的信息。运行 ChatGPT。所以让我们来做一些猜测。

大型语言模型 BLOOM 曾经在一个使用 16 个 Nvidia A100 40GB GPUs 的 Google 云平台实例上部署了 18 天[3]。

让我们假设 ChatGPT 使用相同的硬件。由于模型的大小大致相同——GPT-3 和 BLOOM 的参数分别为 175b 和 176b 我们假设 ChatGPT 也运行在 16 个 Nvidia A100 40GB GPUs 上，但运行在 Azure 实例上，而不是 GCP 实例上，因为 Open AI 与微软有合作关系[4]。

由于 OpenAI 的总部位于旧金山，所以我们进一步假设 ChatGPT 运行在美国西海岸的一个 Azure 区域。由于美国西部的电力比美国西部 2 的碳强度低，所以让我们使用前者。

使用 [ML CO2 影响计算器](https://mlco2.github.io/impact/)，我们可以估计 ChatGPT 每天的碳足迹为 23.04 kgCO2e。平均每个丹麦人每年排放 11 吨二氧化碳当量[7]，所以 ChatGPT 每天的碳足迹大约是一个丹麦人每年碳足迹的 0.2%。如果 ChatGPT 运行一年，其碳足迹将为 365 * 23.04 千克= 8.4 吨，约为一个丹麦人年碳足迹的 76%。

每天 23.04 千克二氧化碳当量的估算值是通过假设每天 16 个 GPUs * 24 小时= 384 个 GPU 小时得出的。目前还不清楚，但我认为 ML CO2 Impact 假设硬件利用率始终为 100 %,这在这种情况下可能是一个合理的假设，因为据报道该服务正在经历沉重的负载。

我们应该在多大程度上相信这个猜测？

为了得到一个概念，让我们看看它是如何与布鲁姆的碳足迹进行比较的。

在 18 天的时间里，23.04 千克二氧化碳当量的日排放量将使 ChatGPT 的排放量达到 414 千克二氧化碳当量。相比之下，布鲁姆在 18 天内排放了 360 公斤。事实上，这两个估计值相差不远，表明 23.04 千克二氧化碳当量可能不是一个糟糕的猜测值。

两种排放估计之间的差异可以归结为许多事情，例如布鲁姆和查特 GPT 的电力的碳强度的差异。

同样值得注意的是，BLOOM 在 18 天内处理了 230，768 个请求[3]，相当于平均每天 12，820 个请求。如果 ChatGPT 的 100 万用户中的 1.2 %每天发送一个请求，ChatGPT 将会产生与 BLOOM 同期相同数量的每日请求。如果社交媒体和传统媒体上对 ChatGPT 的所有讨论都表明了它的用途，那么 ChatGPT 可能会处理更多的日常请求，因此它可能会有更大的碳足迹。

另一方面，如果 OpenAI 的工程师已经找到一些更有效地处理所有请求的聪明方法，我对 ChatGPT 碳足迹的估计可能会太高。

[](/8-podcast-episodes-on-the-climate-impact-of-machine-learning-54f1c19f52d) [## 8 集关于机器学习对气候影响的播客

### 这里有一个精心策划的列表，列出了 8 个关于机器学习的环境足迹以及如何…

towardsdatascience.com](/8-podcast-episodes-on-the-climate-impact-of-machine-learning-54f1c19f52d) 

# ChatGPT 全生命周期碳足迹

为了计算 ChatGPT 的总生命周期碳足迹，我们需要考虑训练过程中的排放。可以获得这方面的一些信息，但很难确定 GPT-3 训练排放中有多少份额应归于 ChatGPT。

我们还需要考虑训练数据预处理的排放。此信息不可用。

此外，我们还需要获得生产硬件的具体排放量的估计值。这是一个相当复杂的任务，留给读者来做练习。可以在[3]和[5]中找到有用的信息，前者估计了布鲁姆的总生命周期碳足迹，后者估计了脸书一些大型模型的总生命周期碳足迹。

[](https://kaspergroesludvigsen.medium.com/the-10-most-energy-efficient-programming-languages-6a4165126670) [## 10 种最节能的编程语言

### 在一项对 27 种编程语言的能效调查中，C 高居榜首，Python 位居第二…

kaspergroesludvigsen.medium.com](https://kaspergroesludvigsen.medium.com/the-10-most-energy-efficient-programming-languages-6a4165126670) 

# 结论

这篇文章估计运行 ChatGPT 每天的碳足迹为 23.04 千克二氧化碳当量。这一估计是基于一些粗略的假设，因此存在很多不确定性，但与一个名为 BLOOM 的可比语言模型对碳足迹的全面估计相比，这似乎是合理的。

通过提供基于事实的 ChatGPT 碳足迹估计，这篇文章使得关于 ChatGPT 的成本和收益的辩论变得有根据。

最后，本文只关注 ChatGPT 的 CO2 排放量。除了二氧化碳排放，其他类型的环境影响，包括用水、空气污染、土壤污染等。，也是重要的考虑因素。

就是这样！我希望你喜欢这篇文章🤞

请留下评论让我知道你的想法🙌

关注更多与可持续数据科学相关的帖子。我也写时间序列预测，比如这里的或者这里的。

此外，请务必查看[丹麦数据科学社区](https://ddsc.io/)的[可持续数据科学](https://github.com/Dansk-Data-Science-Community/sustainable-data-science)指南，了解更多关于可持续数据科学和机器学习的环境影响的资源。

并随时在 [LinkedIn](https://www.linkedin.com/in/kaspergroesludvigsen/) 上与我联系。

# 参考

[1][https://arxiv.org/ftp/arxiv/papers/2204/2204.05149.pdf](https://arxiv.org/ftp/arxiv/papers/2204/2204.05149.pdf)

[2][https://openai.com/blog/chatgpt/](https://openai.com/blog/chatgpt/)

[https://arxiv.org/pdf/2211.02001.pdf](https://arxiv.org/pdf/2211.02001.pdf)

[4][https://news . Microsoft . com/2019/07/22/open ai-forms-exclusive-computing-partnership-with-Microsoft-to-build-new-azure-ai-super computing-technologies/](https://news.microsoft.com/2019/07/22/openai-forms-exclusive-computing-partnership-with-microsoft-to-build-new-azure-ai-supercomputing-technologies/)

[https://arxiv.org/pdf/2111.00364.pdf](https://arxiv.org/pdf/2111.00364.pdf)

[6][https://twitter.com/sama/status/1599668808285028353](https://twitter.com/sama/status/1599668808285028353)

[7][https://kefm . dk/aktuelt/nyheder/2021/apr/foerste-officielle-vurdering-af-danmarks-globale-klimaaftryk](https://kefm.dk/aktuelt/nyheder/2021/apr/foerste-officielle-vurdering-af-danmarks-globale-klimaaftryk)

[8][https://story maps . ArcGIS . com/stories/5417 CD 9148 c 248 c 0985 a5b 6d 028 b 0277](https://storymaps.arcgis.com/stories/5417cd9148c248c0985a5b6d028b0277)

[9][https://medium . com/forward-data-science/how-to-estimate-and-reduce-the-carbon-footprint-of-machine-learning-models-49f 24510880](https://medium.com/towards-data-science/how-to-estimate-and-reduce-the-carbon-footprint-of-machine-learning-models-49f24510880)