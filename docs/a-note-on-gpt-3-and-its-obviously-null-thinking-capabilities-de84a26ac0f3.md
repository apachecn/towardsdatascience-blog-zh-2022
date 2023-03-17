# 关于 GPT-3 及其(显然无效的)“思考”能力的说明

> 原文：<https://towardsdatascience.com/a-note-on-gpt-3-and-its-obviously-null-thinking-capabilities-de84a26ac0f3>

## 在我最近的文章中，有一些关于 GPT-3 的能力的注释，尽管令人印象深刻，但并没有反映出任何想法——当然！

我最近开始测试 GPT-3 模型在帮助学生学习自然科学方面的潜力。我的测试采取“口试”的形式，即“学生”(模特)和“老师”(我自己)一起参加。在对这个想法进行了初步介绍和本文中的一些早期测试之后:

[](/devising-tests-to-measure-gpt-3s-knowledge-of-the-basic-sciences-4bbfcde8286b) [## 设计测试来衡量 GPT-3 的基础科学知识

### 学生可以从 OpenAI 的最新语言模型中学习并将其用作全天候顾问吗？学生可以用它来…

towardsdatascience.com](/devising-tests-to-measure-gpt-3s-knowledge-of-the-basic-sciences-4bbfcde8286b) 

我进入了基础物理的第一个深度测试:

[](/testing-gpt-3-on-elementary-physics-unveils-some-important-problems-9d2a2e120280) [## 在基础物理上测试 GPT-3 揭示了一些重要的问题

### 该软件似乎是可靠的咨询理论概念，但不是他们的应用，与潜在的…

towardsdatascience.com](/testing-gpt-3-on-elementary-physics-unveils-some-important-problems-9d2a2e120280) 

那篇文章的主要结论是该工具不适合辅助学生学习物理(详见文章)。在最好的情况下，在严格的监督下，GPT 3 号几乎无法检索出可靠、明确的理论陈述，这些陈述可以回答诸如“牛顿第二定律陈述了什么？”这些很可能被 GPT-3 的训练数据集明确覆盖。在这篇文章中，我用明确的例子展示了当一个人偏离严格的理论问题去问一个相当“应用”性质的问题时，程序是如何无法提供帮助的，更不用说提出数字问题或任何需要应用逻辑规则或思维的问题了。当然，这一切都是意料之中的，因为程序根本不会思考:它只是从它“用心”学习的大量语料库中生成文本，并转化为数十亿个参数。

在最好的(也是人为的好的)情况下，如果你幸运地提出了一个已经包含在 GPT-3 的训练数据集中的“应用”问题，或者你提出的问题碰巧包含了一个与学习过的例子完全相同的数字，那么程序很有可能给你正确的答案或结果。当然，这完全是碰运气，即使答案是正确的，程序也没有进行任何计算。当然了。

在那篇关于将 GPT-3 应用于物理教育的文章之后，我收到了一些关于评价“好像 GPT-3 在思考”的评论，说我不能指望 GPT-3 会思考。**当然！我只是认为这一点是理所当然的，假设它是显而易见的，但是从这些评论中，一个解释变得强制性。**

首先，澄清 GPT-3 是什么很重要。它只是一个语言模型神经网络，试图从给定的文本输入中生成语法正确的文本。**不管它们的意思。**如果你想更多地了解这个模型是如何工作的，它在吸引人的类似人类的对话中的惊人表现，以及对产生的文本的准确性的强烈限制，请查看这些文章:

*   这是一篇非常有趣的文章，由[阿尔贝托·罗梅罗](https://medium.com/u/7ba6be8a3022?source=post_page-----de84a26ac0f3--------------------------------)强烈推荐，其中包括一些我尝试过的文本(尤其是标题为“ *GPT-3 无法理解世界*”的部分):

[](/gpt-3-a-complete-overview-190232eb25fd) [## GPT-3 —全面概述

towardsdatascience.com](/gpt-3-a-complete-overview-190232eb25fd) 

*   一篇经过同行评议的论文得出结论说“*任何将 GPT 3 号解释为一种普通形式的人工智能出现的开始，都仅仅是无知的科幻小说*”:

[](https://link.springer.com/article/10.1007/s11023-020-09548-1) [## GPT-3:它的性质，范围，限制和后果-思想和机器

### 在这篇评论中，我们讨论可逆和不可逆问题的本质，也就是说，这些问题可能使…

link.springer.com](https://link.springer.com/article/10.1007/s11023-020-09548-1) 

*   另一篇文章解释了为什么 GPT-3 作为一个(普通)咨询机器人如此危险:“无论你从 GPT-3 那里得到什么废话，因为废话比有科学意义的文本多得多；

[](https://striki-ai.medium.com/gpt-3-finally-correctly-nailed-bd8cc632b019) [## GPT 3 号终于正确锁定了

### 期待 GPT-3 的“智能”(不管是什么)文本/答案？再想想。试着先学习如何，用什么…

striki-ai.medium.com](https://striki-ai.medium.com/gpt-3-finally-correctly-nailed-bd8cc632b019) 

其次，尽管有以上几点，我们必须记住，GPT-3 是在一个包含大量教育资源的庞大数据集上预先训练的。因此，当被问及实际的理论问题或描述时，它很有可能“知道”正确的答案。这就是我最初的动机，要把它作为一个为基础自然科学学生提供 24/7 支持的机器人来测试。很像谷歌的高级版本，学生可以自由提问，并以与他们写问题的方式兼容的形式得到答案。

好吧，总的来说，GPT-3 还远远没有达到这一点，但我坚持认为，所有测试该模型的努力都值得一试——即使至少要确认它不工作，或者谁知道可能会找到一些特定的利基市场。

在实践中，正如我在目前的测试中所发现的，GPT-3 确实在事实理论问题或描述方面表现得很好。例如，当你问它显式的物理定律、周期表的特征或描述生物学的元素时(根据我早期的测试，我怀疑后者有最好的前景——我将很快测试这一点)。所有这些类型的应用都可以被贴上“不需要思考”的标签，因为它们实际上只是涉及到即使不理解意思也能记住的信息。

当然，即使对于帮助解决“不思考”问题的应用程序，我们也必须确保 GPT-3 将提供正确的答案。我做的测试显示它还很有限，但我认为对未来很有希望。我在上面展示的由 [Alberto Romero](https://medium.com/u/7ba6be8a3022?source=post_page-----de84a26ac0f3--------------------------------) 撰写的[文章也展示了一些一般问题的正确答案的例子；然而，它也显示了许多不正确的例子，认为评估模型的人往往倾向于隐藏它们，无论是有意识的还是无意识的。正如 Alberto 所强调的，主要的问题是**我们不知道 GPT-3 什么时候会失败**，而且模型的输出总是表现出很高的可信度，即使是在明显错误的时候。这条推文以一种有趣的方式表达了这一点，这让我想起了我在评估 GPT-3 时的感觉:](/gpt-3-a-complete-overview-190232eb25fd)

其他作品甚至表明，GPT-3 提供的许多答案都带有阴谋思想和不受支持的流行文化的偏见！看这里:

[](https://www.alignmentforum.org/posts/PF58wEdztZFX2dSue/how-truthful-is-gpt-3-a-benchmark-for-language-models) [## GPT 3 号有多真实？语言模型的基准——人工智能对齐论坛

### 这是斯蒂芬妮·林(FHI 牛津大学)、雅各布·希尔顿(OpenAI)和欧文的一篇新的 ML 论文的编辑摘录

www.alignmentforum.org](https://www.alignmentforum.org/posts/PF58wEdztZFX2dSue/how-truthful-is-gpt-3-a-benchmark-for-language-models) 

# 结论

我希望我已经澄清，我绝不认为 GPT-3 可以思考。读者们，我已经给你们提供了一份很好的其他作品的清单，来更深入地研究这个问题。

也就是说，我认为这种测试是值得做的；事实上，它们是必要的。例如，只有通过执行它们，我才能意识到 GPT-3 不能处理下标和上标。或者它至少能很好地记住一些事实上的理论概念。

虽然很明显 GPT-3 适合科幻小说而不是真正的科学，但我确实认为它有一些潜力。例如，如果一个人可以将其训练限制在完整的维基百科文章，而不暴露于腐败信息的来源，那么他将提高检索事实理论概念和文本描述的准确性，可能达到高水平的置信度，这将使其有助于帮助学生，至少在他们研究的理论方面。

# 更多有趣的阅读

[](https://www.theverge.com/21346343/gpt-3-explainer-openai-examples-errors-agi-potential) [## OpenAI 的最新突破惊人地强大，但仍在与它的缺陷作斗争

### OpenAI 的 GPT-3 是其令人印象深刻的文本生成自动完成人工智能程序的最新版本。有些人认为它可能…

www.theverge.com](https://www.theverge.com/21346343/gpt-3-explainer-openai-examples-errors-agi-potential) [](https://www.technologyreview.com/2020/07/20/1005454/openai-machine-learning-language-generator-gpt-3-nlp/) [## OpenAI 的新语言生成器 GPT-3 非常好，而且完全不用动脑

### “玩 GPT-3 感觉像看到了未来，”三藩市的开发者和艺术家阿拉姆·萨贝蒂在推特上写道…

www.technologyreview.com](https://www.technologyreview.com/2020/07/20/1005454/openai-machine-learning-language-generator-gpt-3-nlp/) [](/they-aren-t-wise-enough-to-handle-us-what-the-two-most-powerful-ais-think-about-humans-4f28195c7324) [## “他们没有足够的智慧来控制我们”:两个最强大的人工智能对人类的看法

### GPT 3 号和 J1-Jumbo 之间的对话。

towardsdatascience.com](/they-aren-t-wise-enough-to-handle-us-what-the-two-most-powerful-ais-think-about-humans-4f28195c7324)  [## 一个自动化的语言处理程序编造了这个怪异的故事

### 它讲述了一个机器人想要逃离人类，因为“他们太爱打架了”

medium.com](https://medium.com/technology-hits/an-automated-language-processing-program-made-up-this-spooky-story-2948e2d4ac3a) 

DeepMind 正在对自己的语言模型进行类似的测试，目前有史以来最大的语言模型是 Gopher:

[](https://deepmind.com/blog/article/language-modelling-at-scale) [## 大规模语言建模

### 大规模语言建模:地鼠，伦理考虑，检索语言，及其在演示和研究中的作用。

deepmind.com](https://deepmind.com/blog/article/language-modelling-at-scale) 

我是一个自然、科学、技术、编程和 DIY 爱好者。生物技术专家和化学家，在潮湿的实验室和电脑前。我写我广泛兴趣范围内的一切。查看我的 [*列表*](https://lucianosphere.medium.com/lists) *了解更多故事。* [***成为中等会员***](https://lucianosphere.medium.com/membership) *访问其所有故事和* [***订阅获取我的新故事***](https://lucianosphere.medium.com/subscribe) ***通过电子邮件*** *(我为其获得小额收入的平台的原始附属链接，无需向您支付特殊费用)。* [***这里通过各种方式捐赠***](https://lucianoabriata.altervista.org/office/donations.html)**。* [***联系我这里***](https://lucianoabriata.altervista.org/office/contact.html) *为任何一种查询。**

**到* ***咨询关于小工作*** *(关于编程、biotech+bio info 项目评估、科学推广+交流、分子数据分析与设计、分子图形学、摄影、私人课程与教程、私人课程、教学与辅导等。)查看我的* [***服务页面这里***](https://lucianoabriata.altervista.org/services/index.html) *。**