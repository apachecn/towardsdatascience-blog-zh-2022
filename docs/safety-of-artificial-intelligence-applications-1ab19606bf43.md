# 人工智能应用的安全性

> 原文：<https://towardsdatascience.com/safety-of-artificial-intelligence-applications-1ab19606bf43>

## **对使用人工智能可能引发的问题的探讨**

![](img/407363663dbe1f188dd99ddb58cb70e2.png)

[阿瑟尼·托古列夫](https://unsplash.com/@tetrakiss?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/terminator?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

人工智能(AI)不是上图中的机器人，它可能与电视上看到的任何东西都不相似。它是关于使用技术来帮助完成特定的任务，从接下来看什么电影，你可能有兴趣购买什么商品，你可能会点击什么广告，或者将智能手机摄像头对准你的脸以拍摄完美的自拍。

对于 AI 的未来，不同的人有不同的看法和猜想。但是，这些仅仅是这些；猜测。是的，有担忧，应该有关于我们如何在道德上使用人工智能的担忧，但是，我们不知道我们不知道什么。要更深入地了解什么是人工智能，你可以在这里阅读。

真正的担忧不应该是人工智能变得邪恶，而是它的目标是否会恶意或意外地偏离最初的意图，从而导致各种不同的问题。

在下面的段落中，我们提出了在开发或使用人工智能系统时可能出错的关键问题。这包括从高层次的哲学甚至观点到更多的技术考虑。

## 目录

–[偏见和公平](#9204)
–[道德和责任](#952b)
–[可解释性和可解释性](#8b04)
–[数据隐私](#d015)
–[健壮性](#3bec)–[再现性和一致性](#d841)
–[评估](#c358)
–[局部最小值/最大值](#328c)
–[类别不平衡](#9d7e)

## **偏见和公平**

偏见可以通过各种方式进入人工智能模型。通常，根本原因在于数据。由于人工智能模型从数据中学习，如果数据是以引入偏见的特定方式收集的(例如，它们包括有偏见的人类决策或反映历史或社会不平等)，这将影响从这些数据中训练的模型。另一个原因可能是人工智能模型没有在足够的数据上进行适当的训练(也称为欠拟合)，并且正在犯不可接受的错误，或者相反，它已经在太多的数据上进行训练，以至于它失去了概括(也称为过拟合)的能力，而不是产生有偏见的预测。

为了更具体，这里有三个例子:

*   预测未来的罪犯:

2016 年 ProPublica 的一项调查得出结论，法官用来确定被定罪的罪犯是否有可能犯下更多罪行的人工智能系统似乎对少数族裔有偏见。[【1】](#_ftn4)有趣的是，2019 年，英国政府发布了解决警务偏见问题的指导方针和法规。[【2】](#_ftn5)

*   谷歌图像识别:

谷歌的人工智能图像识别系统混淆了动物和少数民族的人。[【3】](#_ftn6)该系统没有经过足够的正确数据训练，无法做出更准确的预测，它的一些错误令人不快。

*   流氓 Twitter 聊天机器人:

早在 2016 年，一个微软聊天机器人在推特上耍流氓，骂人，发表种族主义言论和煽动性政治言论。聊天机器人接受了包括这种语言和评论的数据训练。

## 道德和责任:是否有与道德、角色和责任相关的标准？

道德和责任是人工智能领域的热门话题，一整篇论文都可以专门讨论这个问题。在这里，我们将简要解释我们的意思，并给出一些例子来更详细地理解由于更广泛地采用人工智能，我们作为人类将面临的困境。具体来说，这个领域是关于什么是人工智能系统的道德部署，以及如果出现问题，谁应该负责。

道德问题的一个例子可能与自动驾驶汽车(无人驾驶汽车)有关。这与电车问题类似。[【5】](#_ftn8)具体来说，想一想一辆汽车 2 秒内即将撞车，车上乘客全部遇难的不幸事件。2 秒钟的时间滞后可以被看作是一个小的机会窗口来操纵和拯救乘客。问题是，另一种选择是轧死一些行人，而不是杀死他们。车到底该不该做？

在伦理人工智能下可以调查的其他问题有:

1.如果人工智能系统的部署导致失业或错失机会怎么办？例如，一个简历被自动系统筛选的求职者可能会因为某些甚至不是工作要求的特征而被拒绝。

2.应该允许人工智能系统杀人吗？

在问责制方面，问题是谁对人工智能系统的给定行为负责？是建造系统的 AI 工程师吗？是 AI 系统本身还是别人？

我们再讲一个例子来更好的理解这个问题。

一个病人去医院，由人工智能系统检查。系统检查肺部图像并输出结果。病人是阴性的，意味着图像是清晰的。几个月后，病人再次接受测试，结果是毁灭性的。病人处于肺癌晚期。不幸的是，人工智能系统错了。在这种情况下会发生什么？人工智能系统能对预测误差负责吗？是医生从系统中得到的结果吗？是最初建造它的工程师吗？

本节要提出的最后一点是关于人工智能系统反人类的问题。即使围绕这一点有很多炒作，但大多数也只是一个神话。任何人工智能系统都不存在固有的恶意，除非它被明确编程或在数据中训练成这样。当然，人工智能可以是一种工具或武器，就像刀可以是工具和武器一样。AI 依然可以被恶意使用；你可以将人工智能技术武器化，就像任何其他新技术一样。简而言之，这个话题目前不构成风险，但它肯定是我们应该继续讨论的问题，以避免这成为一个巨大的风险。

一份由著名的英国研究所撰写的更详细的报告更详细地描述了这一部分的许多方面，并提供了一个在其中运作的道德框架。此外，英国政府出于同样的目的发布了数据伦理框架，该框架与上述报告相关联。[【7】](#_ftn10)

## **人工智能系统的可解释性/可解释性:**

人工智能系统的可解释性和可解释性是经常被问到的问题，因为知道为什么和如何做出某些决定对于审计目的是重要的，而不仅仅是。

模型如何决定这是对我的公司最好的行动？它如何知道这个客户比另一个客户更有可能购买我的产品？业内许多高管向数据科学家和人工智能专家提出了这个问题，因为他们不想将他们的决策建立在他们不理解他们最初是如何得出结果的模型上。这可能是一个困难的问题，也是数据科学家和人工智能专家可能无法始终回答的问题。例如，在机器学习领域内使用特定技术，即深度学习，利用人工神经网络来进行预测。这个系统类似于大脑，其中每个神经元都以这样的方式与其他神经元相连，即信息可以在神经元之间传递，直到返回结果。[【8】](#_ftn11)这种类型的人工智能系统在某些任务(图像识别、自然语言处理)中表现优于其他机器学习算法，但更具挑战性的是询问和理解结论是如何得出的。

近年来，随着美国和欧盟引入解释权，这个问题变得更加紧迫。[【9】](#_ftn12)鉴于算法的输出与个人相关并对个人产生重大影响，特别是在法律或财务方面，这是对算法输出做出解释的权利。例如，某人可以申请贷款，而 AI 系统由于此人不可信而拒绝了该申请。关键问题是，系统是如何得出这个结论的。

一些说明人工智能系统中可解释性重要性的例子是图像识别软件学会作弊而不是识别图像。特别是，一个被训练来识别杂志中的马图像的系统实际上并没有识别马，但事实证明，它学会了检测马图像上可见的版权标签。[【10】](#_ftn13)

## **数据隐私:**

隐私是一个与人工智能和一般数据相关的大问题。在欧洲，2018 年生效的《一般数据保护条例》( GDPR)规范了个人数据的收集和使用。[【11】](#_ftn14)数据保护法并未明确提及人工智能或机器学习，但对个人数据的大规模自动化处理和自动化决策有着重要的关注。这意味着，如果人工智能使用个人数据，它就属于该条例的范围，并适用 GDPR 原则。这可以通过使用个人数据来训练、测试或部署人工智能系统。不遵守 GDPR 可能会导致相关公司受到巨额处罚。

个人数据的例子包括出生日期、邮政编码、性别，甚至是用户的 IP 地址。

GDPR 教赋予个人不受完全自动化的决定支配的权利。对人工智能专家来说，关键问题是:当你在人工智能的帮助下对一个人做出决定时，你如何证明你公平、透明地对待了他，或者给他们机会质疑这些决定？

## **不利条件下的鲁棒性:**

当进行对抗性攻击时，人工智能系统可能会遇到不利条件。对抗性攻击就是对人工智能系统的任何恶意干扰，目的是迫使它做出错误的预测。在这一节中，我们讨论两种情况，在这两种情况下，对人工智能系统的对抗性攻击会使系统完全混乱。

*   图像识别:图像识别是指能够识别图像中的对象的人工智能系统，即能够对图像进行分类，例如，能够区分猫图像和狗图像。这种人工智能系统被证明有弱点或者容易受到敌对攻击。研究人员已经表明，即使人工智能系统在数以千计的图像上进行训练，图像中精心放置的像素也可以从根本上改变人工智能系统对它的感知，从而导致错误的预测。[【12】](#_ftn15)
*   自然语言处理(NLP):当人工智能系统处理自然语言时，自然语言也容易受到对抗性攻击。例如，研究表明，在电影评论中使用特定的同义词会改变人工智能系统对该评论的看法。[【13】](#_ftn16)例如，句子/评论“在不可思议的人为情况下塑造的人物，完全脱离现实”就有负面情绪。一篇类似的评论仔细选择了如下词汇:“在不可能的工程环境中塑造的角色，完全脱离现实”，人工智能系统在将其归类为正面时感到困惑。这种攻击的主要目的是集中攻击那些根据上下文可能会产生歧义的词。NLP 环境中的对抗性攻击的其他示例包括特定单词中的字符交换或在句子中添加常见的人类打字错误。[【14】](#_ftn17)

## **再现性和一致性**

人工智能中的一个常见问题是，复制我们获得的结果或生成的模型有多容易。许多算法在训练它们的模型时具有随机元素，这意味着不同的训练尝试产生不同的模型，并且不同的模型具有不同的预测结果。此外，在本地机器上使用我们的数据表现良好的系统，在现场测试时可能表现不佳。我们如何确保我们最初拥有的性能传播到部署的应用程序？我们如何确保系统的性能不会随着时间的推移而下降？

## **评估**

构建人工智能系统的关键问题是“我们如何评估系统？”“准确性是一个好的衡量标准吗？”想想一个问题，你有 100 个女人，其中 10 个怀孕了。你有一些关于这些妇女的信息，你试图建立一个模型来预测谁怀孕了，谁没怀孕。你的模型有 80%的准确率。这是否意味着你有一个好的模型？另一方面，让我们假设你没有模型，你更愿意做的是预测所有没有怀孕的女性。这有 90%的准确率，因为你 100 次中有 90 次是正确的。在这种情况下，准确性不是一个好的衡量标准，因为它很少告诉我们模型有多好。那么，我们使用什么指标，我们如何评估模型的性能？

聊天机器人在过去几年中非常受欢迎，特别是随着自然语言处理(NLP)领域中人工智能模型的改进。

但是，你会如何评价一个聊天机器人呢？想想临床环境中的聊天机器人，我们如何确保它不会推荐错误的治疗方法或提供错误的建议？

## **局部最小值/最大值**

许多算法解决优化问题来训练模型或使用优化方法来寻找模型的最佳超参数。然而，存在优化器落入一些局部最小值/最大值的情况，这导致超参数的次优选择。简而言之，算法表现不佳是因为可能错误地选择了某些参数。

## **阶层失衡**

当你训练一个人工智能系统来识别图像时，你可能会面临一些我们之前讨论过的潜在问题。另一个更技术性的问题可以用下面的例子来说明:考虑一组 100，000 幅图像，其中只有 100 幅是猫的图像，99，900 幅是狗的图像。人工智能系统更有可能预测一只狗，因为它被训练得更频繁；它没有足够的反面案例来准确区分这两种类型的图像。

## **黑天鹅**

依靠历史数据来预测未来并不总是可行的。一个很好的例子是试图预测股票市场。由于多种原因，这在本质上是困难的。利用长期以来具有某种结果的数据，可以创造出在其历史范围内有效的模型。这意味着，如果你在一个没有市场崩盘的时期训练一个模型，这个模型就不可能预测到崩盘。即使你在市场崩溃期间训练了它，由于事件的罕见性，模型仍然不太可能知道什么时候会发生。想想在全球疫情时代预测未来的模型；因为所有的模型在过去都没有类似的数据，所以不可能准确预测未来。[【15】](#_ftn18)

## **优化正确的奖励函数**

通过反复试验来学习的人工智能系统通常被称为强化学习。更正式地说，强化学习(RL)是机器学习的一个领域，处理软件代理应该如何在环境中采取行动，以最大化累积回报的概念。[【16】](#_ftn19)

作为一名人工智能专家，你需要创造一个环境，让软件代理能够“生活”，定义代理可能采取的潜在行动，并设计一个与代理的目标直接相关的奖励函数。例如，这种类型的学习被成功地用在电子游戏中，在这种游戏中，环境通常是明确定义的，代理人有非常明确的目标。一些例子包括 RL 在“超级马里奥”[【17】](#_ftn20)和“俄罗斯方块”中的应用。[【18】](#_ftn21)然而，当环境没有被很好地定义，或者更经常地，当奖励函数不合适时，会发生什么？人工智能系统可以通过寻找捷径或在游戏中“作弊”来学习获胜。[【19】](#_ftn22)

## **结论和未来工作**

很明显，我们需要一个人工智能框架，相关的应用程序可以在其中运行。该框架应讨论上述所有考虑事项以及如何缓解其中每一项的建议。

人工智能系统的系统测试和验证方法应该是该框架的中心主题，但是，这应该进行调整，并详细描述如何实施以解决上述问题。

## 参考资料:

[【1】](#_ftnref4)[https://www . propublica . org/article/machine-bias-risk-assessments-in-criminal-pending](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)

[【2】](#_ftnref5)[https://assets . publishing . service . gov . uk/government/uploads/system/uploads/attachment _ data/file/831750/RUSI _ Report _-_ Algorithms _ and _ Bias _ in _ policing . pdf](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/831750/RUSI_Report_-_Algorithms_and_Bias_in_Policing.pdf)

[【3】](#_ftnref6)[https://www . the verge . com/2018/1/12/16882408/Google-种族主义者-大猩猩-照片-识别-算法-ai](https://www.theverge.com/2018/1/12/16882408/google-racist-gorillas-photo-recognition-algorithm-ai)

[【4】](#_ftnref7)[https://www.bbc.co.uk/news/technology-35890188](https://www.bbc.co.uk/news/technology-35890188)

[【5】](#_ftnref8)[https://en.wikipedia.org/wiki/Trolley_problem](https://en.wikipedia.org/wiki/Trolley_problem)

[【6】](#_ftnref9)[https://www . turing . AC . uk/sites/default/files/2019-06/understanding _ artificial _ intelligence _ ethics _ and _ safety . pdf](https://www.turing.ac.uk/sites/default/files/2019-06/understanding_artificial_intelligence_ethics_and_safety.pdf)

[【7】](#_ftnref10)[https://www . gov . uk/government/publications/data-ethics-framework/data-ethics-framework](https://www.gov.uk/government/publications/data-ethics-framework/data-ethics-framework)

[【8】](#_ftnref11)[https://en.wikipedia.org/wiki/Artificial_neural_network](https://en.wikipedia.org/wiki/Artificial_neural_network)

[【9】](#_ftnref12)[https://en.wikipedia.org/wiki/Right_to_explanation](https://en.wikipedia.org/wiki/Right_to_explanation)。同样参见[https://www.privacy-regulation.eu/en/r71.htm](https://www.privacy-regulation.eu/en/r71.htm)关于欧盟的方法。

[【10】](#_ftnref13)[https://www . the guardian . com/science/2017/nov/05/computer-says-no-why-making-ais-fair-accountable-and-transparent-is-critical](https://www.theguardian.com/science/2017/nov/05/computer-says-no-why-making-ais-fair-accountable-and-transparent-is-crucial)

【https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?】[uri = CELEX:32016 r 0679&from = EN](#_ftnref14)

[【12】](#_ftnref15)[https://arxiv.org/pdf/1710.08864.pdf](https://arxiv.org/pdf/1710.08864.pdf)

[【13】](#_ftnref16)[http://groups . csail . MIT . edu/medg/FTP/PSZ-papers/2019% 20Di % 20 Jin . pdf](http://groups.csail.mit.edu/medg/ftp/psz-papers/2019%20Di%20Jin.pdf)

[【14】](#_ftnref17)[https://medium . com/forward-artificial-intelligence/adversarial-attacks-in-textual-deep-neural-networks-245 DC 90029 df](https://medium.com/towards-artificial-intelligence/adversarial-attacks-in-textual-deep-neural-networks-245dc90029df)

这并不意味着模型是无用的，而是说你在构建这些模型时应该小心谨慎。

[【16】](#_ftnref19)[https://en.wikipedia.org/wiki/Reinforcement_learning](https://en.wikipedia.org/wiki/Reinforcement_learning)

[【17】](#_ftnref20)[https://medium . com/datadriveninvestor/super-Mario-bros-reinforcement-learning-77d 6615 a805e](https://medium.com/datadriveninvestor/super-mario-bros-reinforcement-learning-77d6615a805e)

[https://github.com/nuno-faria/tetris-ai](https://github.com/nuno-faria/tetris-ai)

[【19】](#_ftnref22)[https://www . tomeveritt . se/paper/2017/05/29/reinforcement-learning-with-corrupted-reward-channel . html](https://www.tomeveritt.se/paper/2017/05/29/reinforcement-learning-with-corrupted-reward-channel.html)