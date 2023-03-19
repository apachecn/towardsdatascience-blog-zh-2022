# 具有多阶段预训练的多语言语音翻译

> 原文：<https://towardsdatascience.com/multilingual-speech-translation-with-multi-phase-pretraining-305d642b8a66>

## 如何将不同模态的预训练模型与单语数据混合成强大的多语言翻译模型

![](img/fb78c1e9980afac1e7402ac2a5887af9.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Austin Distel](https://unsplash.com/@austindistel?utm_source=medium&utm_medium=referral) 拍摄的照片

注意:这是本系列的第三篇文章。之前的文章解释了如何在双语和多语言环境下将海量预训练用于机器翻译:
第 1 部分:[用于双语机器翻译的海量预训练](/massive-pretraining-for-bilingual-machine-translation-3e26bfd85432)
第 2 部分: [mBART50:可扩展的多语言预训练的多语言微调](/mbart50-multilingual-fine-tuning-of-extensible-multilingual-pretraining-70a7305d4838)
**第 3 部分:具有多阶段预训练的多语言语音翻译**

# IWSLT

自动语音翻译是从音频形式的语音输入开始，研究从一种人类语言到另一种语言的翻译的领域。它的研究跨越了几十年，多年来提出了许多不同的方法。

国际口语翻译会议(IWSLT)已经到了第 19 届，其目标是跟踪这一领域的进展。他们从两方面着手，一方面接受科学贡献，另一方面组织共同的任务，在不同的场景中，根据共同的基准比较真实的系统。

我在以前的一篇文章中谈到过 IWSLT:

[](https://medium.com/machine-translation-fbk/machines-can-learn-to-translate-directly-your-voice-fbk-iwslt18-bb284ccae8bc)  

# 多语言语音翻译

去年的一个场景是多语言语音翻译，包括从西班牙语(es)、法语(fr)、葡萄牙语(pt)和意大利语(it)到英语(en)和西班牙语(es)的翻译。首先要注意的是，所有涉及到的语言都有些相似，英语是这个群体中唯一的非浪漫语言。这使得多语言翻译更加有效，因为对于模型来说，使用许多类似语言的数据进行改进相对容易。
其次，英语只出现在目标端，而很多研究和很多数据集都集中在源端的英语上。

[](https://medium.com/machine-translation-fbk/must-c-a-large-corpus-for-speech-translation-8e2350d01ea3)  

该任务的三个平移方向(It-En、It-Es 和 Pt-Es)被认为是相对于受约束轨迹的零炮，这意味着没有专门针对这些方向的训练数据。在不受约束的赛道上，参与者可以使用他们想要的所有额外的训练数据，因此没有任何控制。

# 纸

这里描述的论文是去年(2021 年)IWSLT 多语言赛道的[获奖作品](https://scontent-dus1-1.xx.fbcdn.net/v/t39.8562-6/246840550_351924723290671_1968728059268811349_n.pdf?_nc_cat=100&ccb=1-5&_nc_sid=ad8a9d&_nc_ohc=LVLpkxx-5iQAX8qnfjp&_nc_ht=scontent-dus1-1.xx&oh=00_AT9rzfTqP2h69MMUnnnlHMNdz4oeBAtCyZyR-Dwt6utSdw&oe=624E6380)，由脸书 AI Research (FAIR)开发。他们提交的目标是探索用大量并行和未标记的数据预训练多模态翻译模型的解决方案。通过训练语音翻译、文本机器翻译和语音识别的模型，利用来自不同任务的数据。

# 数据

共享任务提供的训练集是 TEDx、CoVoST 和 EuroParlST。

TEDx 是一个在 TEDx 上发表的演讲的集合，共有 13 个语言方向的翻译。这里，只考虑了任务的 7 个方向。 [CoVoST](https://github.com/facebookresearch/covost) 是 Mozilla Common Voice 的扩展，提供大型数据集的翻译，包括 11 个翻译成英语的方向。[europarlist](https://arxiv.org/abs/1911.03167)是一个相对较小的多语言数据集，包含欧洲议会 6 种语言的翻译演讲，共 11 个翻译方向。

此外，作者使用两个已与单语文本对齐的多语转录音频数据集挖掘了并行数据。

转录音频的两个多语种语料库分别是 [CommonVoice](https://commonvoice.mozilla.org/it) (29 种语言)和[Multilingual LibriSpeech](http://www.openslr.org/94/)(8 种语言的有声读物)，而 [CCNet](https://github.com/facebookresearch/cc_net) 则作为多种语言的高质量单语文本大集合。

给定这些数据，通过使用[激光器](https://engineering.fb.com/2019/01/22/ai-research/laser-multilingual-sentence-embeddings/)从 CCNet 中的源音频文本的抄本中提取句子嵌入以对齐具有相似语义的句子，如嵌入相似度所给出的，来获得另外的语音翻译数据。因为在源语言中音频和文本是对齐的，所以这个过程导致源音频与目标语言中的文本对齐。对于零炮点方向，得到的对准数据相当于几十个对准音频。

# 文本数据

5 种语言的单语数据用于训练 [mBART](https://medium.com/towards-data-science/massive-pretraining-for-bilingual-machine-translation-3e26bfd85432) 。单语数据来自 CC100。然后，mBART 在从 OPUS 下载的 7 种语言的并行数据上进行微调。所得的微调模型将在以后用于初始化语音翻译模型。

# 方法

模型培训遵循基于 3 个连续步骤的迁移学习方法:

1.  经过自我监督学习预处理的单模态模块
2.  多任务联合训练
3.  特定于任务的微调

他们的目标分别是:

1.  从大量未标记数据中训练
2.  从文本到文本再到语音到文本
3.  对最终任务进行微调以获得更好的结果

## 单一模态预训练

Wav2vec 2.0 用大量未标记音频数据训练，mBART 用不同语言的大量单语文本训练。 [Wav2vec 2.0](https://arxiv.org/abs/2006.11477) 然后用于初始化第二训练阶段的语音编码器。mBART 的编码器和解码器用于初始化下一阶段模型的编码器和解码器。

## 多任务联合训练

在第二阶段，语音到文本模型与文本到文本模型被联合学习。因此，整个模型由两个编码器和一个解码器组成。这两个编码器共享文本编码器的权重，但在处理音频输入时会使用附加层。适配器层用于促进在两个编码器之间共享的纯语音编码器和文本编码器权重之间的连接。一些培训[技巧](https://arxiv.org/abs/2107.05782)像交叉注意规则(CAR)和在线知识提炼(online KD)已经被用来促进任务间的知识转移。

## 特定于任务的微调

在最后阶段，文本编码器被丢弃，并且剩余的语音到文本翻译模型使用简单的交叉熵在并行音频翻译上被微调。

# 结果

本文的主要结果是使用所描述的过程训练的三个系统的集合，但是具有稍微不同的编码器，比强语音翻译基线高出 8.6 个 BLEU 点。基线是使用上述所有数据建立的，但没有第二阶段的联合训练。

此外，该集合仅比翻译正确音频转录本的强 MT 模型弱 3 个 BLEU 点(在语言方向上平均),这代表了语音翻译性能的巨大进步。

# 结论和意见

这篇论文展示的结果无疑是惊人的，但不幸的是，并不是所有的小组都能够训练这种类型的模型。的确，虽然第 2 期和第 3 期相对便宜(5 + 2 天 8 个 NVidia V100 GPUs)，但像 wav2vec 和 mBART 这样从零开始的训练模型确实很昂贵，所需资源甚至没有在论文中提及。尽管如此，尽管共享任务的范围有限，但结果是显著的，并且清楚地表明大型预训练模型在跨模态设置中也是使能器。

这种强大的模型在该领域开辟了新的可能性，现在下一个前沿是在实时设置中也获得良好的结果，而该系统仅在批处理模式下工作。对于进一步的发展，我们只需要等待第 19 版 IWS lt T1 的结果，其中包括 8 个评估语音翻译不同方面的共享任务。

# 中等会员

你喜欢我的文章吗？你是否正在考虑申请一个中级会员来无限制地阅读我的文章？

如果您通过此链接订阅，您将通过您的订阅支持我，无需为您支付额外费用【https://medium.com/@mattiadigangi/membership