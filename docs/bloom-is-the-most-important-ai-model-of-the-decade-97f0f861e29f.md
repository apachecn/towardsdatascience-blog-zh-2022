# 布鲁姆是十年来最重要的人工智能模型

> 原文：<https://towardsdatascience.com/bloom-is-the-most-important-ai-model-of-the-decade-97f0f861e29f>

## 意见

## 不是戴尔 2 号，不是 PaLM，不是 AlphaZero，甚至不是 GPT 3 号。

![](img/0630e0c43322b9f22744aaaa5eb4fc7e.png)

鸣谢:[大科学研究工作坊](https://twitter.com/BigscienceW)

你可能想知道这样一个醒目的标题是不是真的。答案是肯定的。我来解释一下原因。

[GPT-3](/gpt-3-a-complete-overview-190232eb25fd) 于 2020 年问世，开创了一条全新的道路，此后整个 AI 行业都在有意和关注着这条道路。科技公司一次又一次地制造更好、更大的模型。但是，尽管他们已经投入了数百万来完成这项任务，他们中没有一个人从根本上改变了领先的范式或两年前 GPT-3 制定的游戏规则。

[地鼠](/deepmind-is-now-the-undisputed-leader-in-language-ai-with-gopher-280b-79363106011f)、[龙猫](/a-new-ai-trend-chinchilla-70b-greatly-outperforms-gpt-3-175b-and-gopher-280b-408b9b4510)和[棕榈](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html)(可以说是目前大型语言模型的领奖台)明显比 GPT-3 好，但本质上，它们更多的是一回事。Chinchilla 已经证明了略有不同的缩放定律的成功，但它仍然是一个大型的基于变压器的模型，像其他人一样使用大量的数据和计算。

[DALL E 2](/dall-e-2-explained-the-promise-and-limitations-of-a-revolutionary-ai-3faf691be220) 、 [Imagen](https://imagen.research.google/) 和 [Parti](https://parti.research.google/) ，尽管他们做的事情不同——文本到图像模型增加了变形金刚之外的技术——但他们基本上基于相同的趋势。甚至[火烈鸟](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model)和[加托](https://www.deepmind.com/publications/a-generalist-agent)，稍微偏离了《GPT 3》对人工智能的更一般化、多模态的方法，也只是应用于小说任务的相同想法的混合。

但是，最重要的是，所有这些人工智能模型都源于私营科技公司的巨大资源。这是共同的因素。不仅仅是它们的技术规格使它们属于同一个包。这是因为少数富有的盈利性研究实验室对它们施加了绝对的控制。

这种情况即将改变。

# BLOOM 和 BigScience 标志着人工智能社区的一个转折点

BLOOM(big science Language Open-science Open-access Multilingual)是独一无二的，不是因为它在架构上与 GPT 3 不同——它实际上是上述所有模型中最相似的，也是一个基于变压器的模型，具有 176B 个参数(GPT 3 有 175 b)——而是因为它是人工智能社会政治范式转变的起点，这将定义该领域未来几年的发展——并将打破 big tech 对大型语言模型(LLM)的研发的束缚。

公平地说， [Meta](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/) 、 [Google](https://github.com/google-research/t5x) 和 [OpenAI](https://openai.com/blog/vpt/) 最近开源了他们的一些基于大型变压器的模型(分别是 OPT、Switch Transformers 和 VPT)。是因为他们突然喜欢上了开源吗？我相信那些公司的大多数工程师和研究人员一直都有。他们知道开源的价值，因为他们每天都在使用基于开源的库和工具。但是，作为无道德的赚钱实体，这些公司不会在更广泛的人工智能社区的偏好面前如此轻易地低头。

如果不是因为一些机构和研究实验室已经开始向这个方向施加难以置信的压力，这些公司不会开源他们的模型。

[BigScience](https://twitter.com/BigscienceW) 、[抱脸](https://huggingface.co/)、 [EleutherAI](https://www.eleuther.ai/) 等不喜欢大 tech 对领域的所作所为。垄断一项可能——也有希望——让很多人受益的技术在道德上是不对的。但他们不能简单地要求谷歌或 OpenAI 分享他们的研究，并期待积极的回应。这就是为什么他们决定建造并资助他们自己的实验室——并向想探索其奇迹的研究人员免费开放。最先进的人工智能不再是口袋鼓鼓的大公司的专利。

布鲁姆是这些努力的顶点。经过 2021 年 1 月开始的一年多的集体工作，以及在 Jean Zay 公共法国超级计算机上进行的 3 个多月的训练，BLOOM 终于准备好了。它是[大科学研究研讨会](https://bigscience.notion.site/bigscience/BigScience-214dc9a8c1434d7bbcddb391c383922a)的成果，包括来自世界各地的+1000 名研究人员的工作，并依靠 250+个机构的合作和支持，包括拥抱脸、[、IDRIS](http://www.idris.fr/eng/info/missions-eng.html) 、 [GENCI](https://www.genci.fr/en) 和[蒙特利尔人工智能伦理研究所](https://twitter.com/mtlaiethics)等。

他们的共同点是，他们认为技术——尤其是人工智能——应该是开放的、多样的、包容的、负责任的和可访问的，以造福人类。

他们令人印象深刻的集体努力和他们在人工智能行业中的独特立场只能与他们对社会、文化、政治和环境背景的关注相提并论，这些背景是人工智能模型设计(特别是 BLOOM)以及数据选择、管理和治理过程的基础。

BigScience 的成员发布了[道德宪章](https://bigscience.huggingface.co/blog/bigscience-ethical-charter)，确立了他们在这些技术的开发和部署方面坚持的价值观。他们把这些分为两类——内在的、[有价值的……作为目的](https://www.oxfordhandbooks.com/view/10.1093/oxfordhb/9780199959303.001.0001/oxfordhb-9780199959303-e-9)、外在的、[有价值的作为手段](https://www.oxfordhandbooks.com/view/10.1093/oxfordhb/9780199959303.001.0001/oxfordhb-9780199959303-e-9)。我将在这里引用宪章来总结这些价值观，因为我认为它们对于理解大科学和布鲁姆的前所未有的重要性至关重要。(我还是推荐通读整篇章程。它很短。)

## **内在价值**

*   **包容性:**“…平等获得大科学的文物…不仅仅是不歧视，还有一种归属感…”
*   **多样性:**“…来自 50 个国家的 900 多名研究人员和社区…涵盖 20 多种语言…”
*   **再现性:**...大科学旨在确保研究实验和科学结论的再现……”
*   **开放性:**“…来自世界各地的人工智能相关研究人员可以贡献并加入该倡议…[并且]成果…将在开放的基础上共享…”
*   **责任:**“每个贡献者对他们在大科学项目中的工作负有个人和集体(社会和环境)责任……”

## **外在价值**

*   **可达性:**“作为实现开放性的手段。BigScience 尽最大努力使我们的研究和技术成果易于向更广泛的公众解释和说明……”
*   **透明:**“作为实现再现性的手段。大科学工作在各种会议、网络研讨会、学术研究和科学普及中得到积极推广，以便其他人可以看到我们的工作……”
*   **跨学科:**“作为实现包容性的手段。我们不断在计算机科学、语言学、法律、社会学、哲学和其他相关学科之间搭建桥梁，以便在开发大科学人工制品时采用整体方法。”
*   **多语制:**“作为实现多样性的一种手段。拥有一个从概念上讲是多语种的系统，其近期目标是覆盖世界上 20 种最常用的语言……”

毫无疑问，BigScience 和 BLOOM 是过去十年人工智能领域最引人注目的尝试，旨在消除大技术在人工智能领域建立的所有障碍——不管是愿意还是不愿意。和最真诚和诚实的事业，以建立人工智能(特别是 LLMs)造福每个人。

如果你想更多地了解大科学方法，请阅读这个关于 LLM 研究中社会背景的三篇文章的伟大系列。可以通过[拥抱脸](https://huggingface.co/bigscience/bloom)进入布鲁姆。

# 是什么让布鲁姆与众不同

正如我在开始时提到的，BLOOM 不是第一个如此大规模的开源语言模型。Meta、Google 和其他公司已经开源了一些模型。但是，正如预期的那样，这些并不是这些公司能提供的最好的。赚钱是他们的主要目标，所以分享他们最先进的研究是不可能的。这就是为什么仅仅通过这些战略性的公关行动来表明他们参与开放科学的意图是不够的。

大科学(BigScience)和布鲁姆(BLOOM)是一套伦理价值观的体现，这些价值观是公司无法用定义来代表的。无论哪种情况，可见的结果都是一个开源的 LLM。然而，指导大科学的隐藏的——也是极其必要的——基础强调了这些集体倡议和强大的大技术之间不可调和的差异。

被环境所迫而不情愿地采用开源实践和因为你坚信这是正确的方法而这样做是不一样的。BigScience 成员的信念是，我们应该使人工智能民主化，并致力于让尽可能多的人受益——通过开放访问和结果或解决伦理问题——这是 BLOOM 的独特之处。我承认，这也是它成为十年来最重要的人工智能模型的原因。

布鲁姆是一个领域的先锋，这个领域正处于彻底变革的边缘。这是一个超越当前研究趋势的运动的旗帜。这是人工智能新时代的终结，它不仅将推动该领域更快地向前发展，还将迫使那些喜欢以其他方式前进的人接受现在管理该领域的新规则。

这不是开源第一次赢得隐私和控制权。我们在计算机、操作系统、浏览器和搜索引擎中都有例子。最近的历史充满了那些想为自己保留利益的人和那些代表其他人战斗并取得胜利的人之间的冲突。开源和开放科学是技术的终极阶段。我们即将进入人工智能的新时代。

*订阅* [**算法桥**](https://thealgorithmicbridge.substack.com/) *。弥合算法和人之间的鸿沟。关于与你生活相关的人工智能的时事通讯。*

*您也可以直接支持我在 Medium 上的工作，并通过使用我的推荐链接* [**这里**](https://albertoromgar.medium.com/membership) 成为会员来获得无限制的访问权限！ *:)*