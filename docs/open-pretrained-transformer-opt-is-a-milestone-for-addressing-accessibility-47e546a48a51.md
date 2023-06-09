# 开放式预训练转换器(OPT)是解决可访问性的一个里程碑

> 原文：<https://towardsdatascience.com/open-pretrained-transformer-opt-is-a-milestone-for-addressing-accessibility-47e546a48a51>

## 选择加入 GPT-3 人出局

![](img/0bb7b37698d94f2530a553ad1c944e7d.png)

图片来自[皮克斯拜](https://pixabay.com/)的格尔德·奥特曼

2022 年 5 月 3 日，Meta AI 公布了新的大型语言模型(LLM)开放式预训练转换器(OPT-175B)。在这篇文章中，我们将谈论 OPT 如何在机器学习领域，特别是自然语言处理(NLP)领域，建立了一个可重复性的基准。

**再现性是怎么回事？**

可访问性与再现性问题密切相关。如果你能获得有关该方法的信息，你可以复制这个实验。为什么再现性如此重要？让我们从更广阔的角度来看待这个问题，并回到过去。大约在 16 世纪，*智人*对他们获取知识的方式做出了重大改变。*智人*不再假设信息是正确的，而是开始使用科学的方法来确定假设，进行实验，分析结果，并得出结论。在过去的几个世纪里，科学家们利用这一过程来建立我们对自然世界及其规律的集体理解。通过关注科学发现的透明度和再现性，我们在技术上取得了巨大进步。(必须注意，定性方法不一定要产生可重复的结果。是的，定性的方法仍然很强)。

尽管可重复性是定量科学方法的一个基本考虑因素，但 2016 年《自然》*杂志上的一项调查显示，超过 70%的研究人员在试图重现另一位研究人员的实验时失败了，超过 50%的人在重现自己的实验时失败了(Pineau 等人，2021；贝克，2016)。*

*这是一个严重的问题。评估研究主张的可信度是科学过程中一个核心的、持续的、费力的部分(Alipourfard et al .，2021)。如果一个科学发现是不可复制的，它就违反了科学方法的一个基本前提。Joelle Pineau 等人(2021)指出，机器学习研究的挑战之一是确保呈现和发布的结果是合理和可靠的。(注:Joelle Pineau 是脸书人工智能研究所的联合董事总经理，麦吉尔大学副教授。她在让巴勒斯坦被占领土变得可及方面发挥了作用。)*

*不幸的是，学术论文并不总是提供可重复的结果，比如缺少步骤或缺乏关于其方法的信息。作为一名数据科学家，我在阅读 ML 论文时也多次遇到可重复性问题。*

***GPT-3 和再现性问题***

*当我们谈论再现性问题时，我们有一只大象在房间里，GPT-3。在将近两年的时间里，OpenAI 对不公开该模型给出了粗略的解释。关于 GPT-3，OpenAI 曾表示“公开太危险了”Meta AI 显然认为安全不应该是一个让公众无法接触到的模型。在阅读了[梅塔关于 OPT-175B](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/) 的博客文章后，我们可以看到，如果你做了严谨的功课，在负责任的同时公开一个 LLM 是可能的。*

*【Meta AI 对 OPT 的可访问性持什么态度？*

*   *Meta AI 团队已经注意使 OPT 模型可以公开访问。他们使用了负责任人工智能的指导方针。我知道脸书和负责任相处不好，但我们在这里。欢迎来到 2022 年！*
*   *OPT 团队与拥抱脸紧密合作。OPT 于 5 月 3 日公布。目前[抱脸](https://huggingface.co/patrickvonplaten)上有 6 款:125M、350M、1.3B、2.7B、6.7B、30B 参数截止 5 月 11 日。175B 参数可通过应用程序访问。 [Stephen Roller](https://github.com/facebookresearch/metaseq/issues/88) ，OPT 论文的第二作者，正与拥抱脸团队合作，使各种各样的 OPT 模型易于使用。*
*   *OPT 团队(包括 [OPT 论文](https://arxiv.org/abs/2205.01068)作者)非常活跃，并对 [Github 问题](https://github.com/facebookresearch/metaseq/issues)做出快速回复。*
*   *OPT 在公开可用的数据集上接受培训，以允许更多的社区参与了解这一基础新技术。*

***当前 OPT 的可访问性挑战***

*   *根据官方[指南](https://github.com/facebookresearch/metaseq/blob/main/docs/api.md#:~:text=Right%20now%20only%20on%20Azure%2C%20as%20it%20requires%20the%2080GB%20A100s)，OPT 需要一个 A100 80GB 的 GPU。这对用户来说是一个巨大的可访问性障碍。*
*   *目前，它只运行在 Azure 云服务上(基于官方指南)。在我的本地机器上安装 OPT 时，我看到 OPT 有 AWS 的基础设施。我相信我们会看到 OPT 与其他云计算平台的集成。*
*   *各种安装问题。例如，它不适用于 Python 3.10.2，因为 Python 3.10.2 不支持所需的 torch 版本(1.10.2)。*
*   *Metaseq 是使用 OPT 的基本代码。不幸的是，[斯蒂芬·罗拉](https://github.com/facebookresearch/metaseq/issues/88#:~:text=Metaseq%20is%20notoriously%20unfriendly)说，*梅塔塞克是出了名的不友好。**

*OPT 是一个令人兴奋的大型语言模型。一旦它变得更加用户友好，它将是 NLP 领域的游戏规则改变者。在这篇文章中，我们想分享我们对 OPT 语言模型可访问性方面的第一印象。在 GPT-3 炒作和无法访问它之后，我们希望 OPT 将带来对大型语言模型开发的新理解。拥抱脸和变形金刚库集成完成后，我们将有机会尝试它，并在这里再次分享我们的经验！
编辑:5 月 12 日，各种 OPT 模型已经可以通过变形金刚图书馆访问)*

*埃尼斯·格克切——NLP 数据科学家*

*穆罕默德·埃姆雷·塞内尔 —博阿济奇大学计算机科学*

*感谢[梅尔·梅德](https://www.linkedin.com/in/mel-meder/)校对文章*

***参考文献**:*

*阿里普法德，n .，阿伦特，b .，本杰明，D. M .，本克勒，n .，毕晓普，m .，伯斯坦，m，…和吴，J. (2021)。对公开研究和证据的系统化信心。*

*贝克，M. (2016)。1500 名科学家揭开再现性的盖子。自然，533(7604)。*

*皮诺，j .，文森特-拉马尔，p .，辛哈，k .，拉里维埃，v .，贝格尔齐默，a .，阿尔凯-Buc，f .，… &拉罗歇尔，H. (2021)。提高机器学习研究的再现性:来自 NeurIPS 2019 再现性计划的报告。*机器学习研究杂志*， *22* 。*