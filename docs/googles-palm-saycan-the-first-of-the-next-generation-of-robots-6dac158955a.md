# 谷歌的 PaLM-SayCan:第一代下一代机器人

> 原文：<https://towardsdatascience.com/googles-palm-saycan-the-first-of-the-next-generation-of-robots-6dac158955a>

## 谷歌进入了一条新的道路:融合人工智能和机器人技术。

![](img/7ac86d021c94e4769bbeda300777fbba.png)

棕榈说可以拿起一个苹果。鸣谢:[谷歌研究](https://youtu.be/E2R1D8RzOlM)

不管谷歌搜索怎么说，从历史上来说，人工智能与具有人形的闪亮金属机器人几乎没有什么关系。现在似乎不再是这样了。在过去的几年里，科技公司在人工智能机器人上下了很大的赌注。不是任何类型的(Roomba 是一个有用的工具，但远非机器人的原型)。不。公司正在制造人形机器人。

波士顿动力公司是该集团在机器人领域经验最丰富的公司，于 2021 年展示了最新版本的 [Atlas](https://youtu.be/tF4DML7FIWk) 。三十年后，他们得到了一个具有相当不错的运动和本体感受技能的模型(它可以进行致命的跳跃)。现在由亚马逊支持的 Agility Robotics 生产 Digit，这是一种通用机器人，可以可靠地以人类的速度做仓库工作[三星(Samsung)的 Handy](https://techcrunch.com/2022/07/21/agilitys-next-digit-robot-will-have-a-face-and-hands/) 似乎能够做一些需要一些手工技巧的家务。小米最近加入了这个群体，并在两周后的特斯拉人工智能日推出了会说话的机器人 [CyberOne](https://youtu.be/iRKiXCASoDs) ，它类似于[特斯拉的 Optimus](https://youtu.be/2RS28Fpp1UM) 。

许多高知名度的科技和机器人公司正在押注人形机器人，这一事实本身就很有趣。我以前曾提出过[制造具有人类特征的机器人有很好的理由:世界在高度、形状、动作方面适应了我们……这些项目揭示了该行业对制造机器人的兴趣，正如马斯克去年在 2021 年人工智能日期间所说，“消除危险、重复和无聊的任务”，或在家里帮助我们。](/tesla-ai-day-2021-review-part-4-why-tesla-wont-have-an-autonomous-humanoid-robot-in-2022-76dff743f885#:~:text=One%20key%20detail,as%20we%20can.)

但是这篇文章不是关于人形机器人的。至少不仅仅是。这是一种机器人学的新方法，我上面提到的例子中没有一个是遵循的。我说的是将最先进的人工智能系统——特别是语言模型——与可以在世界上导航的全身机器人融合在一起。大脑和身体。一些人工智能公司专注于构建下一个超大型语言模型，而机器人公司想要最灵巧的机器人，但似乎没有重叠——尽管这似乎是显而易见的前进道路。

## 莫拉维克悖论和人工智能与机器人融合的复杂性

为什么大多数人工智能公司不进入机器人领域(OpenAI [去年解散了其机器人分支](https://venturebeat.com/business/openai-disbands-its-robotics-research-team/))以及为什么大多数机器人公司将其机器人的范围限制在简单的任务或简单的环境(或两者都有)是有充分理由的。其中一个主要原因是所谓的[莫拉维克悖论](https://en.wikipedia.org/wiki/Moravec%27s_paradox)。它说，与直觉相反，很难让机器人足够好地执行感觉运动和感知任务(例如，拿起一个苹果)，而创造能够解决困难的认知问题(例如，玩棋盘游戏或通过智商测试)的人工智能相对容易。

对人类来说，微积分显然比在空中接球更难。但那只是因为从进化论的角度来说，微积分是相对较新的。我们还没来得及掌握。正如人工智能的创始人之一马文·明斯基所说:“我们更了解工作不好的简单过程，而不是工作完美的复杂过程。”简而言之，制造可以四处移动并与环境完美互动的机器人极其困难(在过去几十年中进展甚微)。

但现在，有一家公司试图克服莫拉维克悖论的明显局限性(我想强调“努力”，我们会看到为什么)。我指的是谷歌。在与[日常机器人](https://everydayrobots.com/)的合作中，这个科技巨头创造了很可能是机器人领域的下一个突破: [PaLM-SayCan](https://ai.googleblog.com/2022/08/towards-helpful-robots-grounding.html) (PSC)，一个(没有那么多)人形机器人，拥有上面其他人只能梦想的混合能力。

我对谷歌的方法特别感兴趣，因为我是人工智能虚拟系统和现实世界机器人融合的倡导者。不管我们是否想要建立一个人工通用智能，这是两个学科的自然道路。一些研究人员和公司认为[缩放假说](https://www.gwern.net/Scaling-hypothesis)是人类智能人工智能的关键。相反，我认为让人工智能在现实世界中扎根是至关重要的，这既是为了解决当前的缺点(如人工智能对世界如何工作的普遍无知或互联网数据集的偏见)，也是为了将其提升到下一个水平(推理和理解需要只有通过探索世界才能获得的隐性知识)。

(注:如果你想进一步了解这个话题，我推荐我这个大多被遗忘的帖子“[人工智能和机器人技术将不可避免地融合](/artificial-intelligence-and-robotics-will-inevitably-merge-4d4cd64c3b02?sk=674a99d59fc01413f58786d024ed8bb3)”)

谷歌的 PSC 显示，该公司最终接受了这是前进的方向，并决定不放弃纯粹的人工智能，而是重新关注人工智能+机器人，作为实现更有能力的智能系统的一种手段。最后，这与训练多模态模型没有什么不同(通常被认为是深度学习模型的自然下一步)。同样，能够“看”和“读”的人工智能比那些只能感知一种信息模式的人工智能更强大，能够行动和感知的人工智能或机器人将在我们的物理世界中表现得更好。

让我们看看谷歌的 PSC 有什么能力，以及它如何设法将大型语言模型的能力与物理机器人的灵活性和动作能力结合起来。

## PaLM-SayCan:第一代新一代机器人

在高层次上，我们可以将 PSC 理解为一个结合了 PaLM 对自然语言的掌握(PaLM 是一种语言模型，很像 GPT-3 或 LaMDA——尽管略胜一筹)和机器人与世界互动和导航的能力的系统。手掌充当人类和机器人之间的中介，机器人充当语言模型的“手和眼”。

用更专业的术语来说，PaLM 允许机器人执行高级复杂任务(记住 Moravec 悖论指出，随着任务变得更加复杂，对于一个没有享受数千年进化进步的机器人来说，正确完成任务变得更加困难)。例如，“给我拿一份点心”虽然看似简单的任务，但包含许多不同的基本动作(表达本身包含某种程度的省略和暗示；“哪种零食？”).

PaLM 为机器人提供了任务基础:它可以将自然语言请求转化为精确(尽管复杂)的任务，并将其分解为有助于完成任务的基本动作。像 Atlas 或 Digit 这样的机器人可以非常好地完成简单的任务，但如果没有显式编程，它们无法解决 15 步请求。PSC 可以。

作为回报，机器人向 PaLM 提供关于环境和自身的上下文知识。它给出了基于世界的信息，这些信息可以告诉语言模型，在给定外部真实世界条件的情况下，哪些基本动作是可能的——它能够做什么。

手掌陈述什么是有用的，机器人陈述什么是可能的。这是谷歌创新设计的关键，也是这种方法让该公司处于领先地位的原因(尽管不一定是在成就方面——PSC 仍然是一个研究原型，而 Atlas 和 Digit 是完整的产品)。PSC 结合了任务基础(给定请求的意义)和世界基础(给定环境的意义)。PaLM 和机器人都无法独自解决这些问题。

现在，让我们来看一个 PSC 能做什么的例子，它是如何做到的，以及它与替代方案相比有多好。

## PaLM-SayCan 在行动:利用 NLP 导航世界

谷歌研究人员在他们的实验中使用的一个例子(发表在论文“[尽我所能，不要像我说的那样:机器人启示中的基础语言](https://arxiv.org/pdf/2204.01691.pdf)”中)以人类的请求开始，自然地表达:“我刚刚健身，请给我带点小吃和饮料来恢复。”

这对于一个人来说是一件容易的事情，但是传统设计的机器人对如何完成请愿毫无头绪。为了使请求有意义，PSC 利用 PaLM 的语言能力(特别是，[思维链提示](https://arxiv.org/abs/2201.11903)，它只是使用中间推理步骤来得出结论)来将请求重新定义为可以分解成步骤的高级任务。PaLM 可以总结道:“我会给这个人带一个苹果和一个水瓶。”

PaLM 充当了人类语言的微妙和含蓄与机器人能够理解的精确、僵硬的语言之间的中介。既然 PaLM 已经定义了一个任务来满足用户的请求，它就可以提出一系列有用的步骤来完成这个任务。然而，由于 PaLM 是一个与世界没有联系的虚拟 AI，它不一定会提出最佳方法，只会提出对任务有意义的想法——而不会考虑实际设置。

这就是机器人启示发挥作用的地方。经过训练的机器人“知道”在现实世界中什么是可行的，什么是不可行的，它可以与 PaLM 合作，为那些可能的动作赋予更高的价值，而不是那些更难或不可能的动作。PaLM 给有用的动作打高分，机器人给可能的动作打高分。这种方法允许 PSC 最终找到给定任务和环境的最佳行动计划。PSC 集两者之长。

回到零食的例子。PaLM 已经决定它应该“给这个人一个苹果和一个水瓶”然后它可能会提议去商店买一个苹果(有用)。然而，机器人会给这一步打很低的分，因为它不会爬楼梯(不可能)。另一方面，机器人可能会建议拿一个空杯子(可能)，PaLM 会说完成任务没有用，因为这个人想要的是水，而不是杯子(没用)。通过从有用的和可能的提议中获得最高分，PSC 将最终决定去厨房找一个苹果和水(有用的和可能的)。一旦该步骤完成，该过程将重复，PSC 将决定在新状态下应该采取的下一个基本行动，即在每个后续步骤中更接近任务的完成。

谷歌的研究人员在 101 个教学任务中测试了 PSC 与两种选择的对比。一个使用了一个较小的模型，该模型在指令回答方面进行了明确的微调( [FLAN](https://arxiv.org/abs/2109.01652) )，另一个没有使用将语言模型融入现实世界所必需的机器人启示。他们的发现很清楚:

> “结果显示，使用具有启示基础的 PaLM(PaLM-say can)的系统在 84%的情况下选择正确的技能序列，在 74%的情况下成功执行它们，与 FLAN 和没有机器人基础的 PaLM 相比，错误减少了 50%。”

这些结果揭示了一种很有前途的方法，可以将最先进的语言人工智能模型与机器人结合起来，形成更完整的系统，可以更好地理解我们，同时更好地导航世界。

不过，这种方法也有缺点。有些在 PSC 中是显而易见的，而其他的将在公司探索了问题的整个范围后显现出来。

## PaLM-SayCan 做不到的:很难打败进化

这里我将忽略外围模块的有效性(例如，语音识别、语音转文本、检测和识别对象的视觉传感器等。)尽管这些必须完美地工作以使机器人发挥作用(例如，照明的变化可能使物体检测软件变得无用，从而使 PSC 不能完成任务)。

我想到的第一个问题——也是我在文章中反复提到的一个问题——是语言模型无法从人类的角度理解。我举了一个例子，一个人想要一份点心和饮料，手掌正确地解释了一个苹果和一个水瓶就可以了。然而，这里有一个隐含的问题，即使是最好的语言模型，如 PaLM，也可能无法解决更复杂的情况。

PaLM 是一个非常强大的自动完成程序。它被训练成在给定令牌历史的情况下准确预测下一个令牌。虽然这个训练目标已经被证明对于令人满意地解决大量的语言任务非常有用，但是它没有为 AI 提供理解人类或者有意图地生成话语的能力。PaLM 输出单词，但它不知道为什么，它们意味着什么，或者它们可能产生的后果。

PaLM 可以正确地解释这个请求，并指示机器人拿一个苹果和水来，但这将是一个无脑的解释。如果它猜错了，就不会有自我评估的内部机制让模型知道它得出了错误的解释。即使 PaLM(或者更聪明的人工智能)可以处理大多数请求，也没有办法确保 100%的请求可以得到解决——也没有办法知道人工智能可以解决哪些请求，不能解决哪些请求。

PSC 很可能遇到的另一个问题是机器人的动作出错。假设 PaLM 已经正确地解释了这个人的请求，并且提出了一个合理的任务。PSC 已经决定了一系列有用的和可能的步骤，并且正在采取相应的行动。如果其中一个动作没有正确完成或者机器人犯了一个错误怎么办？比方说，它去摘苹果，苹果掉在地上，滚到墙角。PSC 是否有反馈机制来重新评估其状态和世界状态，以提出一套新的行动来解决新情况下的请求？答案是否定的。

谷歌在一个非常受限的实验室环境中进行了实验。如果 PSC 在世界上运行，它将遇到无数不断变化的条件(移动的物体和人，地面的不规则性，意外事件，阴影，风等)。).它几乎什么也做不了。现实世界中的变量数量几乎是无限的，但 PSC 是用在受控环境中行动的其他机器人的数据集来训练的。当然，PSC 是一个概念证明，所以这不是判断其性能的最公平的镜头，但谷歌应该记住，从这一点到现实世界工作机器人的飞跃不仅仅是数量上的。

这些是主要的语言和动作问题。但还有许多其他的与此有些关联:PaLM 提出的任务可能需要比机器人的上限更高的步骤。失败的概率随着完成任务所需的步骤数呈指数增长。机器人可以找到它不熟悉的地形或物体。由于缺乏常识，它可能会发现自己处于一种无法随机应变的新局面。

最后一个缺点是，PaLM 和所有其他语言模型一样，容易使它在训练中看到的偏见永久化，对此我将用一整段话来说明。有趣的是，来自约翰·霍普金斯大学的研究人员最近分析了一个机器人的行为，在它通过互联网数据得到增强后，发现偏见超越了语言:[机器人是种族主义者和性别歧视者](https://research.gatech.edu/flawed-ai-makes-robots-racist-sexist)——机器人的行为和数据一样有偏见。这是极有问题的。语言中存在的偏见可能是显而易见的(大部分时间都是如此)，但行动中的偏见更加微妙，这使得它们更难定位、分析和消除。

最后，这总是谷歌人工智能博客帖子的补充，该公司以优先考虑安全和负责任的开发而自豪。PSC 带有一系列机制来确保程序的安全性:PaLM 不应该产生不安全或有偏见的建议，机器人不应该采取潜在的危险行动。尽管如此，这些问题普遍存在，公司没有一个标准的解决方案。虽然 PSC 似乎是新一代最先进的人工智能机器人的第一款，但在这方面没有什么不同。

*订阅* [**算法桥**](https://thealgorithmicbridge.substack.com/) *。弥合算法和人之间的鸿沟。关于与你生活相关的人工智能的时事通讯。*

*您也可以直接支持我在 Medium 上的工作，并通过使用我的推荐链接* [**这里**](https://albertoromgar.medium.com/membership) 成为会员来获得无限制的访问权限！ *:)*