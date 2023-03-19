# 定义 2021 年的 10 个人工智能模型

> 原文：<https://towardsdatascience.com/10-ai-models-that-have-defined-2021-7d804b87b10>

## 未来将建立在这些基础上

![](img/80b40575cad14e271b08d244fee38d60.png)

[Kirichay D](https://www.shutterstock.com/es/g/Kirichay+D) 在 [Shutterstock](https://www.shutterstock.com/es/image-photo/beautiful-woman-purple-hair-futuristic-costume-1747573019) 上拍照

人工智能不断加速。每年我们都会读到大量的新闻，谈论新的人工智能模型，这些模型将彻底改变 X 行业或将人工智能推向新的高度。但是，信噪比很低。只有一堆令人印象深刻的突破值得铭记。

2021 年没有什么不同，这就是为什么我按时间顺序列出了今年最相关和最有影响力的车型。尽管研究和开发的路线多种多样，但有几个明显的趋势:大型语言模型变得更大，未来的顶级模型很可能是多模态的，对效率的追求正在得到有趣的结果。我用一小段描述了每个模型对该领域的主要影响。尽情享受吧！

# 开关变压器:第一个+1T 参数模型

**影响:**稀疏性支持将模型扩展到巨大的规模，这对于密集架构是不可行的，同时保持计算负担不变。

2021 年 1 月，谷歌发表了论文“[开关变压器:通过简单有效的稀疏性扩展到万亿参数模型](https://arxiv.org/abs/2101.03961)他们提出了一种新的语言模型架构，以证明模型可以增长到超过 1T 参数(开关变压器具有 1.7T 参数)，同时保持计算成本稳定。

为了保持较低的计算要求，switch transformer 使用了由深度学习先驱 Geoffrey Hinton 共同发明的[专家混合](https://en.wikipedia.org/wiki/Mixture_of_experts)范式的变体。模型的参数被分成 2048 个专家，使得输入仅由一个专家处理。在任何给定时间，只有一小部分参数是活动的，即稀疏模型。

GPT-3 和大多数其他语言模型是密集的——它们使用整个模型来处理每个输入。通过利用稀疏性，开关变压器降低了计算成本，同时提高了精度和功耗。

# 一张图片胜过千言万语

影响: DALL E 利用丰富的自然语言创造出各种各样的图像。这是最早流行的多模式人工智能之一。

OpenAI 于 2021 年 2 月建造了 [DALL E](https://openai.com/blog/dall-e/) 。该模型以西班牙著名画家萨瓦尔多·达利和可爱的皮克斯机器人瓦力命名，是 GPT 3(120 亿参数)的小型版本，并在文本-图像对上接受训练，以“通过语言操纵视觉概念”

DALL E 采用自然语言书写的句子，并使用预期的含义生成图像。它的力量依赖于它的零射击能力；它可以执行没有经过训练的生成任务，而不需要例子。DALL E 创造性地利用了语言和图像相结合的可能性——例如，将高层次的概念融合到一幅图像中。用“竖琴做的蜗牛”或“鳄梨形状的扶手椅”来提示它，会给出你所期待的确切的[，尽管世界上不存在任何类似的东西。](https://openai.com/blog/dall-e/#:~:text=navigatedownwide-,Combining%20Unrelated%20Concepts,-The%20compositional%20nature)

DALL E 加入了人工智能画家的行列——谷歌的 [DeepDream](https://deepdreamgenerator.com/) 、 [Ai-Da](https://www.ai-darobot.com/) 、[显见](https://obvious-art.com/)等等。创造力曾经是我们的专利，但现在不再是了。DALL E 已经证明，人工智能更接近于赋予“一张图胜过千言万语”这句话新的含义。

# 谷歌输入/输出大会:妈妈和 LaMDA

影响:类似妈妈的模特会简化在互联网上搜索的过程。LaMDA 是创造类似人类的对话式人工智能的最新一步。

谷歌一年一度的输入输出事件发生在 2021 年 5 月。技术高管展示了最新的产品和研究，其中 [MUM](https://blog.google/products/search/introducing-mum/) (多任务统一模型)和 [LaMDA](https://blog.google/technology/ai/lamda/) 被描绘成明星。两种语言模型，没有什么好羡慕流行的 GPT-3 的。MUM 是搜索引擎的未来，而 LaMDA 是一个能够进行有趣对话的聊天机器人。

基于 Transformer 架构和以前的系统，如 BERT 和 Meena，谷歌建立了 MUM 和 LaMDA 其技术规格仍未披露。但是我们知道一些关于他们的事情。MUM 的使命是进一步增强谷歌的搜索引擎——可能会让 SEO 过时，正如我在之前的一篇文章中所说的。MUM 比 BERT(谷歌目前支持搜索引擎的力量)强大 1000 倍，可以处理跨语言、跨任务、最重要的是跨模式的自然语言复杂查询——它理解文本和图像。

LaMDA 以与人交流为导向。作为一个明智、具体、有趣和实事求是的聊天机器人，LaMDA 可以管理开放式对话——正如首席执行官[桑德尔·皮帅在演示中通过让 LaMDA 扮演冥王星和纸飞机的角色向](https://www.dailymotion.com/video/x81d4rm)展示的那样。人类可以从一个句子中创造出一千条独特的路径。我们只需要做出选择，整个世界就会从那里出现。LaMDA 更接近于能够做同样的事情。

# 中国的语言人工智能市场:悟道 2.0，M6

**影响力:**这些模型为中国在人工智能研究、开发和效率方面达到第一做出了巨大贡献。

中国一直在努力在技术开发和研究方面赶上美国，人工智能是一个特别热门的领域。去年有两款车型引起了分析师的关注:[武道 2.0](/gpt-3-scared-you-meet-wu-dao-2-0-a-monster-of-1-75-trillion-parameters-832cd83db484)(1.75 吨)和[M6](/meet-m6-10-trillion-parameters-at-1-gpt-3s-energy-cost-997092cbe5e8)(10 吨)。它们基于在性能、计算成本降低和污染减少方面利用稀疏性的承诺。

2021 年 6 月，Wu Dao 2.0 害羞地亮相，声称它是有史以来最大的神经网络，规模是 GPT-3 的 10 倍。两者的主要区别是吴导的多模态性；作为一个通用的语言模型，它是第一个像 DALL E 或 MUM 那样利用多模态的语言。然而，性能方面没有太多的信息，因为北京人工智能研究院(BAAI)没有给出任何结果。(我的猜测是，该模型并没有被训练成收敛，而是作为一个实验来分析专家混合范式的力量)。

由阿里巴巴 DAMO 学院建设的 M6 经历了一个阶段性的发展过程。它于 6 月首次以 1T 参数作为多模式和多任务模型推出。然后，在 2021 年底，研究人员发表了一篇文章，概述了一个惊人的结果:在 10 万亿个参数下，新版本的 M6 在训练期间消耗的计算成本只有 GPT-3 的 1%——在效率和减少碳足迹方面达到了一个新的里程碑。

# 微软和英伟达联手:MT-NLG

影响:这是最大的密集语言模型，也是 Nvidia 和微软合作的第一个潜在的其他发展。

10 月，Nvidia [在其博客](https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/)中发布消息称，他们已经与微软联手打造了(至今)最大的密集语言模型。MT-NLG 在 530B 参数上大于 GPT-3、J1-Jumbo 和 Gopher(尽管明显小于稀疏模型)。在大多数基准测试中，MT-NLG 比 GPT-3 具有更好的性能，直到最近，它仍然是我们拥有数据的顶级语言模型。

我把它列入这个名单的另一个原因是 Nvidia 和微软的联盟。两家大型科技公司都是人工智能领域的佼佼者。英伟达是图形处理器的头号制造商和提供商，微软在云服务和人工智能软件方面有着非常强大的影响力。这种伙伴关系的未来努力将值得关注。

# GitHub Copilot:你的互惠生程序员

**影响:** GitHub Copilot 是程序员迈向自动化最枯燥重复任务的第一步。它最终也会成为一种威胁。

GitHub 的母公司微软和 OpenAI 联手创造了 [GitHub Copilot](/github-copilot-a-new-generation-of-ai-programmers-327e3c7ef3ae) ，许多开发者已经在日常工作中使用它。copilot——扩展了 GPT-3 强大的语言技能，并作为 OpenAI 编码模型 Codex 的基础——擅长编程。它可以跨语言工作，可以完成代码行，编写完整的函数，或者将注释转换成代码，还有其他功能。

一些用户指出由于版权和许可的潜在法律问题。副驾驶从哪里学的？使用它所有的完成是否安全？如果最终触犯了法律，该怪谁？但是不管它的大图限制，它是编码未来的一个重要里程碑。它很可能成为开发人员工具箱中的必备工具。

# DeepMind 进入语言 AI 的入口:地鼠，复古

影响: Gopher 是目前排名第一的语言模型。复古证明，新技术可以提高效率(降低成本和碳足迹)数量级超过以前的模式。

DeepMind 这些年一直保持沉默，而全球的公司都参与了人工智能语言的爆炸。自从谷歌发明了 Transformer 架构以来，语言人工智能已经成为研究的主要焦点，超过了视觉、游戏和其他领域。作为主要的人工智能公司之一，DeepMind 的缺席令人惊讶。

2021 年 12 月，情况发生了变化。DeepMind 发表了三篇关于语言 AI 的论文。首先，他们提出了 [Gopher](/deepmind-is-now-the-undisputed-leader-in-language-ai-with-gopher-280b-79363106011f) ，一个 280B 参数密集模型，在 124 个任务中的 100 个任务中抹杀了能力——包括 GPT-3、MT-NLG 和 J1-Jumbo。一夜之间，DeepMind 不仅成为了强化学习的领导者，也成为了语言人工智能的领导者。Gopher 现在是有史以来无可争议的最好的语言模型。

第三篇论文更令人印象深刻。在 7B 参数下， [RETRO](https://deepmind.com/research/publications/2021/improving-language-models-by-retrieving-from-trillions-of-tokens) (检索增强的 Transformer)是一个小一点的语言模型。然而，尽管它比 GPT-3 小 25 倍，但在各项基准测试中，它的性能不相上下。与 GPT-3 相比，DeepMind 通过该模型实现了 10 倍的计算成本降低。RETRO 使用一种检索机制，允许它实时访问大型数据库，避免模型必须记住所有的训练集。

这具有重要的后果。首先，它证明了新技术可以显著提高语言模型的效率。随着成本变得更可承受，这也有利于较小的公司参与进来。最后，它有助于减少人工智能对环境的影响。

*如果你喜欢这篇文章，可以考虑订阅我的免费周报*<https://mindsoftomorrow.ck.page/>**！每周都有关于人工智能和技术的新闻、研究和见解！**

**您也可以使用我的推荐链接* [***这里***](https://albertoromgar.medium.com/membership) *直接支持我的工作，成为中级会员，获得无限权限！:)**