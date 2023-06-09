# 第一幕:用动作变形金刚构建人工智能的未来有多熟练

> 原文：<https://towardsdatascience.com/act-1-how-adept-is-building-the-future-of-ai-with-action-transformers-4ed6e2007aa5>

## 人工智能的未来是数字和物理代理，它们可以在人类命令的指导下在世界上行动。

![](img/1b398a0481f3f7a992c2fbb7a09361cb.png)

鸣谢:作者 via midway

人工智能最雄心勃勃的目标之一是建立能够做人类所能做的一切的系统。GPT-3 可以写，稳定扩散可以画，但都不能直接与世界互动。10 年来，人工智能公司一直试图以这种方式创造智能代理。现在这种情况似乎正在改变。

我最近的一篇文章报道了谷歌的 PaLM-say can(PSC)，这是一个由 PaLM 驱动的机器人，是迄今为止最好的大型语言模型。PSC 的语言模块可以解释用自然语言表达的人类请求，并将其转换为高级任务，这些任务可以进一步分解为基本动作。然后，机器人模块可以在现实世界中执行这些动作，以满足用户的请求。虽然它有重要的局限性(我评论过)，但谷歌的 PSC 是上一代人工智能与机器人集成的首批例子之一——数字连接到物理。

但是，物理机器人并不是唯一可以直接影响世界的人工智能代理。另一个有希望的研究方向是可以通过开放式动作与软件交互的数字代理。我这里说的不是 GPT-3 或 DALL E，它们只有极其有限的行动空间(它们缺乏任何类型的运动系统，无论是物理的还是数字的)，因此只能间接影响世界(在人类阅读或看到它们的后代之后)。例如，我指的是 OpenAI 的视频预训练模型(VPT。我在这里介绍了，它通过观察人类演奏来学习演奏《我的世界》——在某种程度上模仿他们的行为。

到目前为止，像 VPT 这样的系统并不多，因为这是一项非常新的技术。训练和建造这些系统很难:GPT-3 只能修改世界的信息，而 VPT 也可以修改世界的状态——即使是在数字层面。VPT 享有更高程度的代理，并代表着一个更接近一般的智能。

但今天我不会谈论谷歌的 PSC 或 OpenAI 的 VPT。

## Adept 制造了第一个动作变形金刚(ACT-1)

今天的文章介绍了一家新的人工智能初创公司， [Adept](https://www.adept.ai/) ，它于今年早些时候推出，旨在构建“有用的通用智能”Adept 汇集了来自谷歌、DeepMind 和 OpenAI 的优秀人才——包括 2017 年发明变压器的家伙。像大多数其他人工智能公司一样，他们的主要目标是构建智能代理，但他们的愿景相当独特。

他们不是要建立一个可以做我们所能做的一切的 AGI——这将最终导致人类劳动力的大规模替代——而是要建立一个智能界面，可以充当人类和数字世界之间的自然翻译器——人工智能和人类合作而不是竞争。正如研究员[大卫·栾(David Luan)所说](https://twitter.com/jluan/status/1519035169537093632)(现在在 Adept 工作，之前在谷歌和 OpenAI 工作)，“我们想建立一个自然语言界面……——一个你的计算机的自然语言前端。”

Adept 现在[宣布](https://twitter.com/AdeptAILabs/status/1570144499187453952)它的第一个人工智能模型:动作变形金刚(ACT-1)。他们没有透露任何技术细节，只是说这是“一个经过训练使用数字工具的大规模变压器。”(他们很快会发表一篇论文。我会更新这篇文章以包含新的信息。Adept 的 ACT-1 是一个数字代理，旨在与其他程序和应用程序进行通信，并作为我们和数字世界之间的接口，一个自然的人机界面(HCI)。

Adept 发布了几个简短的[演示视频](https://www.adept.ai/act)，在视频中你可以看到第一幕的表演。它可以接受用自然语言表达的高级请求并执行它们——非常像谷歌的 PSC。这些任务可能需要跨软件工具和网站的几个步骤，复杂程度各不相同。ACT-1 可以在流程的不同点执行涉及各种工具的任务，并可以接受用户反馈以进行改进。

最重要的是，ACT-1 可以执行我们不知道如何做的动作。这就是 ACT-1 的用处变得明显的地方。ACT-1 可以充当多任务元学习者，能够处理各种软件应用。为了让它工作，我们只需要知道如何与第一幕沟通，以及我们想要的结果。如果 ACT-1 运行良好，我们就不必学习使用 Excel、Photoshop 或 Salesforce。我们会简单地将工作委派给第一幕，并专注于更具认知挑战性的问题。

如果你重读最后一段，你可以瞥见 ACT-1(和一般的数字代理)的两个关键方面，我将在文章的其余部分讨论这两个方面。首先，我们有一个很大的“如果”在“如果第一幕完美运作。”如果没有，当我们面临一项我们缺乏能力或知识的任务时(例如，您必须组织一些数据，但不知道如何使用 Excel)，我们如何知道何时应该信任它呢？第二，正确使用 ACT-1 要求我们知道如何与它沟通——这强调了提示在未来将有多重要(如果从 GPT-3 或稳定扩散中还不清楚的话)。

让我们从第一点开始。

## 艰巨挑战背后的大承诺

GPT-3 等基于变压器的模型的一个主要限制是，它们太不可靠，无法用于高风险环境(例如，心理健康治疗)。原因是这些模型(GPT-3 以及 VPT 或 ACT-1)是在互联网数据上训练的，并被优化以学习下一个令牌/动作，给定先前令牌/动作的历史。这意味着他们缺乏常识，表达意图的能力，或者对世界如何运作的深刻理解。这些人工智能系统在它们能做什么以及它们工作得有多好方面存在固有的局限性。一些限制可能是规模问题(更多的数据和更多的参数将解决它们)，但其他限制似乎是它们如何设计和开发的内在问题。

尽管 ACT-1 的目的与 GPT-3 不同，尽管它有更大的行动空间，但它属于同一类模型，因此由于同样的原因而受到限制。它解释人类请求的能力不包括理解意图的能力。在其中一个演示示例中，用户要求 ACT-1 在休斯顿寻找一栋价格低于 60 万美元的房子。ACT-1 去寻找一个符合标准的房子，但是它不知道——不像任何人会立即推断出它——用户想要房子是为了*别的事情*。请求背后有一些意图，做出正确决策需要一些现实环境。第一幕无法获取这些信息。

现在，让我们来看看我上面提到的那个大“如果”。在房屋搜索的情况下，用户应该知道如何在没有 ACT-1 的情况下进行搜索。但是如果用户提出了一个他们不知道如何完成的请求呢？一个合理的可能性是，用户会盲目地相信人工智能系统正在做正确的动作(我们，愚蠢地，总是这样)。这是对我们无法信任的人工智能产生不健康依赖的完美配方。

你可能会说，我们已经依赖于许多许多抽象层，如果这些抽象层被打破，我们和其他人将完全失去防御能力。这是真的，但是我们倾向于用足够的可信度来构建这些层(例如，飞机似乎工作得很好，尽管你很可能不知道它们如何工作或者它们是否可靠。你相信社会有动力去建造它们，所以它们不会崩溃)。相比之下，基于深度学习的系统——即使是最先进的系统——也没有这种可靠性。

另一种可能性是，如果用户意识到人工智能的工作方式并了解其风险，决定不盲目信任系统。但是，即使在这种情况下，他们也没有办法评估 ACT-1 是否做对了。当我们问 GPT 3 号一些我们不知道答案的问题时，也会发生同样的情况。在某些情况下，我们可以简单地自己检查它(这破坏了系统的实用性)，但在其他情况下，我们不能。

如果用户的不信任足够强烈，可能会导致他们不使用该系统。但另一个问题出现了:如果社会开始严重依赖这些类型的自然语言数字界面(就像它在社交媒体或智能手机上所做的那样)，会怎么样？大麻烦。

在我们能够建立我们可以信任的人工智能之前(正如加里·马库斯教授所说)，像 ACT-1 这样的系统的承诺只是承诺。如果它不能可靠地工作，ACT-1 只是一个非常昂贵的工具，只能完成我们也能完成的任务——我们经常不得不重新做它做错的事情。

ACT-1 的最终目标是雄心勃勃的，但在 Adept(或 OpenAI 或 DeepMind)这样的公司能够实现这一目标之前，前面还有重要的挑战。

## 提示是人机交互的未来

现在说第二点:提示的重要性(我之前在“[软件 3.0——提示将如何改变游戏规则](/software-3-0-how-prompting-will-change-the-rules-of-the-game-a982fbfe1e0?sk=844fc3b9f6d22e0a0dfbf0df1230a855)”中写过这个问题)。

你可能已经非常熟悉激励的概念(最先进的生成模型，如 GPT-3、LaMDA、DALL E、稳定扩散等。所有工作都有提示)。如果你不是，提示只是一种使用人类自然语言(例如英语)与人工智能系统(更一般地说，与计算机)交流的方式，让它们做一些特定的动作或任务。

提示是我们如何让生成式人工智能模型做我们想做的事情。如果你想让 GPT-3 写一篇文章，你可以说:“写一篇关于人工智能风险的 5 段文章。”那是一个提示。如果你想让 DALL E 创造一个美丽的形象，你可以说:“一只猫和一只狗在夏天玩球，充满活力和色彩的风格，高清。”那是另一个提示。谷歌的 PSC 和 Adept 的 ACT-1 工作方式相同。

提示与编程语言不同，它对我们来说非常直观。像 Python 或 C 这样的编程语言是当今最常见的 HCI。计算机天生就能理解这些语言，但我们学习它们会更加困难(它们可能需要多年的练习才能掌握)。因为提示不是别的，就是自然语言，我们马上就能学会。

有些人在提示工具和无代码工具之间做了类比，但是有一个重要的区别。尽管无代码软件消除了学习编码的需要，但它仍然需要用户单独学习每个特定的工具——无代码工具不是元工具。要让 ACT-1——一个元工具——做一些你只需要一项技能的事情；提示。诚然，这是一种无代码的形式，但它也是一种*横向*，自然技能——非技术通人士的终极梦想。

把提示放在历史的背景下，我们可以把它看作是 HCI(穿孔卡、机器码、汇编、低级编程语言和高级编程语言)漫长历史的最后一步。我们一直在爬楼梯，从用机器语言交谈到用人类语言交谈。我们在以前的基础上建立了越来越抽象的层，目的是隐藏人机交流背后的复杂性，使我们更容易。

提示是让计算机做某事的最新、最简单、最直观的方式。这是最强大的交流人机界面，因为它让我们在自己的领域感到舒适。它将数字用户的障碍降至最低。出于这些原因，我认为提示 HCI 在几年后会像今天的智能手机一样无处不在。这是一个我们每天都会使用的工具，可以处理任何与数字世界有关的事情。

提示是直观的，但也是一种技能

但是，即使提示是与计算机交流的最自然的方式，它也不是一种天生的能力。这是一项需要实践才能掌握的技能(即使它需要的数量比不上学习编程)。你可以把它想象成一种新的话语模式——相当于我们在和一个孩子说话时如何改变我们的语气、风格和词汇，或者政治家在和我们说话时如何使用修辞。提示是以特定形式针对特定目标的自然语言交流。从这个意义上来说，掌握它需要时间。

[科技博客作者 Gwern](https://www.gwern.net/GPT-3#prompts-as-programming) 建议将框架提示作为一种新的编程范式。这个定义可能会疏远非编码人员，但它有助于他们理解这不是天生的。这是一项技能，虽然非常直观，但也需要实践(例如，让 GPT-3 输出你想要的东西可能需要几次尝试，才能得到像样的东西)。

正如 Gwern 解释的那样，如果我们把提示想象成一种编程语言，那么每个提示都可以理解为一个程序。当你用英语向 GPT-3 输入一个请求时，你就是在给它“编程”，让它去做一件其他版本不知道该怎么做的任务。你正在创造一个稍微不同的版本。因此，提示不仅仅是一种向计算机传达我们的需求的方式。这也是教他们用自然语言完成新任务的一种方式。

Gwern 以 GPT-3 为例强调提示是一种技能。他说，GPT-3 早期受到的主要批评之一是它无法正确执行一些基本的语言任务。他设法通过找到更好的提示来证明批评者是错误的。他证明了并不是所有的提示对于达到某种结果都是同样好或者同样有效的——就像与人类交谈可以被视为一种技能，可以表现得更好或者更差一样(如果我们将这一论点扩展到无限，我们会发现一切都是一种技能)。

即使像 GPT-3 或 ACT-1 这样的人工智能系统被证明非常有用，人们仍然需要学习创建良好的提示(类似于我们现在对 GPT-3 或稳定扩散所做的，这些工具并不是每个人都掌握到相同的程度)。

无论如何，虽然提示不是万灵药，但它绝对是人机交互的一大飞跃——将使人们能够利用计算机、程序、应用程序和其他工具的能力民主化，否则人们将无法这样做。

**歧义和语境:提示的致命弱点**

然而，尽管与以前的 HCI 相比，提示带来了巨大的优势和好处，但它并不完美。它有一个重要的缺点:人类语言固有的模糊性加上缺乏语境。

如果你考虑编程语言(甚至无代码工具)，没有解释的余地。语法是严格而清晰的。如果你用 Python 键入一个句子，它只能表示一件事，计算机不需要“推理”或“理解”它的意思。它可以根据请求立即行动。因为提示存在于自然语言的领域中，所以它们失去了传统代码的刚性和明确性。如果我们想让 HCI 和提示一起工作，这是一个关键问题。

我们人类相互理解(尽管不一定总是如此),因为我们既可以访问我们认为是共同的共享知识库，也可以访问围绕任何给定交互的上下文信息。这是语言的实用主义的一面，它不能被整合到 GPT-3 或 ACT-1 的请求中。

一方面，这些系统缺乏常识，无法获取我们对世界的共享知识。另一方面，他们缺乏任何给定交互的特定上下文，因为该上下文通常通过显式方式(即，书面或口头语言)是不可传递的。这意味着，当有歧义时，第一幕要么猜一猜，要么就此打住，不完成任务。

我[以前写过](/ai-wont-master-human-language-anytime-soon-3e7e3561f943?sk=72474faf6975d37b22a919e9517e0819)关于大型语言模型的这个关键限制(现在也存在于这些数字代理中，甚至在 PSC 或即将到来的 Tesla bot 这样的物理机器人中)，我不知道我们如何克服它。

我看到的唯一解决方案是设计和开发这些人工智能系统，就像我们教和教育孩子一样。他们需要在这个世界中成长，并与这个世界互动，以内化他们错过的所有知识。另一种选择是将它们的范围限制在任何不需要上下文的任务上，但这可能对它们来说太过严格而没有用。我们必须等待，看看我是否错了，还有另一种方法。

## 结论

Adept 已经开始雄心勃勃地寻求建立可以在世界上行动的人工智能代理。像 OpenAI、DeepMind 或谷歌一样，它面临着艰巨的挑战，要开发不仅非常强大，而且可靠的人工智能。

Adept 的愿景和目标强调了即时编程作为一种新的软件范式与人工智能系统和计算机进行通信的重要性。它们也揭示了传统人机界面的优势，如编程语言，以及看似不可逾越的限制。

总而言之，Adept 绝对是一家值得关注的公司。第一个动作转换器 ACT-1 开启了一条充满希望的研究路线，在未来的几个月/几年里会有很多值得谈论的内容。

*订阅* [**算法桥**](https://thealgorithmicbridge.substack.com/) *。弥合算法和人之间的鸿沟。关于与你生活相关的人工智能的时事通讯。*

*您也可以直接支持我在 Medium 上的工作，并通过使用我的推荐链接* [**这里**](https://albertoromgar.medium.com/membership) 成为会员来获得无限制的访问权限！ *:)*