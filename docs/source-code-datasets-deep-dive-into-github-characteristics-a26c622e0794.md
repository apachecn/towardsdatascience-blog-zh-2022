# 源代码数据集——深入探究 GitHub 特征

> 原文：<https://towardsdatascience.com/source-code-datasets-deep-dive-into-github-characteristics-a26c622e0794>

## 采样 GitHub 时要考虑的亚人群

![](img/da7677463c403a5e62d6a6192eb9fd9f.png)

照片由来自 [Pexels](https://www.pexels.com/photo/different-types-of-sauce-3622479/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) 的 Jonathan Borba 拍摄

# **Github 隐性偏见——为什么它很重要**

CodeAI 最近成为了 AI 界的共同话题；从 [Github 的副驾驶](https://copilot.github.com/)，通过[微软的 CodeBert](https://github.com/microsoft/CodeBERT) 到 [VsCode 的 GuessLang](https://guesslang.readthedocs.io/en/latest/) (PS，你能找到这三者之间的共同联系吗？).机器学习比以往任何时候都更多地应用于源代码领域。推动它的是 NLP 世界最近的重大改进，代码库的高可用性(无论是其 [Github](https://github.com/) 、 [Gitlab](https://about.gitlab.com/) 还是甚至像 [Rosetta Code](http://www.rosettacode.org/wiki/Rosetta_Code) 这样的开源项目)，以及源代码是技术世界中几乎每个人的重要部分这一事实。让我们每个人都成为该领域的专家。但是这也是一个主要的缺点，因为获得源代码数据集的简单性可能会掩盖它所包含的隐含的复杂性。当我们对 Github 的机器学习应用程序进行采样时，我们基本上会搜索类似于[推理时间](https://blogs.gartner.com/paul-debeasi/2019/02/14/training-versus-inference/)的代码，类似于它将应用到的代码，最有可能是公司内部的代码。但问题是，我们可以假设这些代码与占 GitHub 较高份额的开源和虚拟个人项目有很大不同。天真地没有考虑到这一点，按原样对 Github 进行采样，将最有可能[导致我们的模型过度适应](/how-to-generate-code-dataset-for-machine-learning-applications-fbc1b888cc84)手头有偏见的人群，而不是对目标人群进行足够的概括(值得一提的是一个类似的案例，其中许多当局决定[禁止面部识别服务](https://www.nature.com/articles/d41586-020-03186-4)，因为训练人群太有偏见)。这就是为什么深刻理解 GitHub 的内部特性如此重要。Github 的主要子群体详述如下。

# **源代码语言**

第一个也是最明显的地方是 Github 的源代码语言发行版。众所周知，Github 有一个超级长尾语言版本(一些语言如 Javascript 非常常见，而其他语言如 C 非常罕见)。考虑到 Github 的社交方面，这并不奇怪。由于同样的现象可以在其他基于社交的技术网站上看到(如 StackOverflow [最常见的问题标签](https://stackoverflow.com/tags))，Github 拥有类似的特征(更常见和流行的源代码语言和技术的比率更高)是有道理的。考虑到天真的 Github 采样将以代理长尾分布结束，这会严重影响机器学习模型，这可能是有问题的。应该强制执行诸如[分层抽样](https://en.wikipedia.org/wiki/Stratified_sampling)的规范化政策，以避免生成的模型过度拟合这种倾斜的语言分布。

# **账户类型**

Github 有两种[账户类型](https://docs.github.com/en/get-started/learning-about-github/types-of-github-accounts) —用户(个人)和组织(公司)。理论上我们会假设它反映了两种不同的使用模式；个人(用户)存储库是辅助/虚拟项目，公司(组织、公共)存储库是开源/文档(鉴于 Github 启用了仅由公司员工访问的私有模式，公共模式应该包括我们希望与世界共享的内容——开源、示例和文档)。事实是，我们可以在这里观察到一个有趣的组合:一些公司向他们的公共帐户发布看似私人的代码，一些用户向他们的个人帐户发布看似他们公司相关的代码。这有许多可能的原因；公司可能会错误地发布代码。用户，尤其是在 covid 期间在家工作的用户，可能缺乏对 Github 上私人用户(公共范围)和公司(私人范围)帐户之间的区别的理解。这种现象没有什么有趣的应用；首先，它增加了[在公司的公共存储库和员工的存储库中找到内部机密的可能性。第二，当我们对 Github 进行采样时，我们可能会面对个人和(公共)公司账户的内部代码。虽然 Github 的较高份额仍然是个人账户(在随机抽样测试中，我们面临大约 9:1](https://medium.com/@oriabramovski/stop-searching-for-aws-secrets-in-code-f5cfda9431a9) 的[比率)，因此大部分抽样代码很可能是私有的(与公司无关)，但如果我们想要将 Github 上的内部专业级存储库作为目标，应该将它考虑在内。](https://medium.com/@oriabramovski/stop-searching-for-aws-secrets-in-code-f5cfda9431a9)

# **组织账户范围**

如前所述，公司账户可以是[公共账户或私人账户](https://github.com/pricing)。考虑到不同的共享范围，假设私有存储库包含更多的公司关键代码是有意义的，这些代码很可能会受到较低标准的影响(因为代码在公司外部是不可见的，缺乏清晰的文档和代码样式不太重要)。开源项目中高标准的一个例子可以在像 [Meta](https://github.com/facebook) 官方账户这样的地方找到..通过开源技术构建社区)。Github 中有许多开源项目类型，其中包括员工仅公司所有(如 Meta，其中“成员必须有两个因素授权”)、一般开放公司所有(如 [SalesForce](https://github.com/salesforce) ，其中贡献不限于 SalesForce 员工)，以及一般开放，如 [VueJs](https://github.com/vuejs) (非公司所有)。这样的账户应该被抽样，以防我们寻找高代码标准或者成为数据集负总体，在那里错误不太可能出现。以防我们寻找更多的内部知识库；其他不太面向开源的公司应该成为目标。Github 启用了[高级搜索标准](https://github.com/search/advanced)，这对此很有用——通过考虑关注者的数量、存储库的数量甚至账户位置，我们可以有意地将目标锁定在规模更适中的账户上(这里应该出现较低的标准，因此代码应该更类似于公司内部)。

# **正在使用的技术**

正如我们到目前为止所看到的，依赖 Github 作为主要数据源来训练我们的模型可能会产生一种隐含的偏见——虽然我们最有可能针对的是**内部**代码，但天真随机的抽样群体最有可能由**开源和私有项目**组成。这可能会影响正在使用的代码标准和技术(例如，看到环境变量或日志记录等专业代码模式的可能性)。反映这一点的一个简单直观的练习可以是，在搜索像谷歌地图这样的超级通用技术[时，比较结果的数量(大约 200 万个结果，其中许多似乎是次要项目)与搜索像 Okta](https://github.com/search?q=maps.googleapis.com&type=code) 这样的更小众的企业相关技术[(大约 9 万个结果，其中许多似乎是内部公司相关的)。另一个相关的证词可以在 Meli at el](https://github.com/search?q=okta.com&type=Code) [Github 的泄露秘密分析](https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_04B-3_Meli_paper.pdf)中找到，他们发现 Google api key 的泄露频率几乎是 Twilio 的 1000 倍(与 Okta 一样，它可以被认为是更专业的代码)。考虑使用中的技术可以突出其他特征，比如我们发现的项目的代码标准(内部代码应该比虚拟项目的质量更高)。因此，依靠这种技术的存在来定位更有可能与内部公司相关的存储库是有意义的。

# **主题**

代码项目可以满足各种需求，从服务器客户端到 Android 原生应用。Github 的存储库目标可以通过查看存储库[标记的主题](https://github.com/topics)来查看。同时值得一提的是，根据 [Izadi 等人](https://arxiv.org/abs/2010.09116)的分析，大多数 Github 的存储库都没有主题/正确的主题。如何评估存储库主题的另一个选择是查看源代码语言和正在使用的技术——使用 Angular 的 Javascript 最有可能与客户端相关，而使用 Node.js 的 Javascript 最有可能与服务器相关。C、C++和 C#更有可能是服务器端而不是客户端相关的。正如我们在前面的分析中看到的。搭载谷歌地图的安卓，可以增加成为副业的可能性。作为一个例子，Python 比 Javascript 在数据从业者中更常见，因此它更有可能包含相关的数据科学/数据工程代码。因此，确保我们的数据集跨语言均匀采样不仅对于避免语言过度拟合很重要，而且确保它将跨不同的代码主题通用化。例如，虽然我们的目标用例与 C++的相关性可能不如 Javascript，但确保我们的数据集包含 C++项目仍然很重要，以便获得子样本揭示的相关代码模式的可见性。

# **用户特征分析**

私人帐户可以是不同的成熟级别(虚拟项目对面向开源)和不同的目标范围(附带项目对内部公司相关)。相关的指示可以是专业代码模式的存在，如[许可证](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)文件、[自述文件](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)文件、专业技术(如 Okta)或者甚至只是测试文件。例如在 Python 上，没有一个 [requirements.txt](https://pip.pypa.io/en/stable/reference/requirements-file-format/) 文件可以表明回购标准较低。考虑到特定的源代码语言特征可以使目标语言具有特定的代码味道(就像 Python 数据从业者一样，[使用雪花连接器](https://github.com/search?l=python&p=7&q=%22snowflake.connector.connect%22&type=Code)并将凭证保持为明文代码)。根据 Meli 等人 [Github 的秘密分析](https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_04B-3_Meli_paper.pdf)发现泄露的秘密是一个重复的问题(泄露的秘密更有可能同时出现)，我们可以假设我们在一个回购中发现的特征也适用于用户的其他回购。使我们能够通过在一些用户帐户的存储库中观察特定代码模式(如代码味道)来锁定用户。

# 智能瞄准

Github 有很多子种群。这里介绍了最重要的几个，但是还有很多其他的。我们看到，用户帐户在保密率和代码实践方面更类似于私有(内部)公司，而公司(公共)帐户在使用的技术类型方面更类似于私有公司。在天真地对 Github 进行抽样之前，我们应该首先了解我们寻找的目标项目特征是什么(以及为了产生丰富的负面群体，什么不是)。只有这样，我们才能找到针对感兴趣人群的方法，并真正估计我们的数据集代表它的程度。但无论如何，不要天真地试用 Github，就认为它会工作得很好。在大多数情况下，更聪明的目标是必须的。