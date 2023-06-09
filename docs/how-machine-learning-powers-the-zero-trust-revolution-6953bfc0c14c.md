# 机器学习如何推动零信任革命

> 原文：<https://towardsdatascience.com/how-machine-learning-powers-the-zero-trust-revolution-6953bfc0c14c>

![](img/233430f5c39963334f91cefcd21e15a2.png)

图片来源: [Pixabay](https://cdn.pixabay.com/photo/2016/03/25/02/13/abstract-1278077_960_720.jpg)

零信任正在安全领域掀起风暴。这是一种新的模式，在提高现代分布式 It 环境的安全性方面具有巨大的潜力。然而，零信任需要先进的技术来评估计算机网络上的活动，动态确定风险，并自动分配策略。

在本文中，我将解释零信任系统的这些复杂要求，以及机器学习如何使它们成为可能。

# 什么是零信任？

[零信任是一种安全模型](https://www.hysolate.com/learn/zero-trust/zero-trust-security/)，旨在保护网络免受内部威胁。它包括实施安全措施，执行“永远不信任，永远验证”的方法。常见的零信任安全措施包括微分段、粒度访问控制策略和第七层威胁防御。

[Forrester Research 的 John kinder vag](https://www.forrester.com/blogs/category/zero-trust-security-framework-ztx/)构思了这个安全模型。他意识到传统的安全模型天真地假设网络中的一切都是可信的。因此，这些网络不会怀疑用户身份遭到破坏，假设所有用户都负责任地行动。

零信任模型将信任视为一个漏洞，因为它使内部人员(无论恶意与否)能够在网络中横向移动，访问用户允许的任何数据和操作。

# 机器学习如何推动零信任安全

# 启用基于风险的安全策略

安全领导者正在从传统的以合规为中心的安全方法转向风险驱动的方法。不同之处在于，在传统方法中，组织对已知的法规遵从性要求做出反应，而在基于风险的方法中，组织不断评估威胁形势并采取主动措施来防止威胁。

组织可以依靠机器学习技术来实时评估用户请求，评估设备、网络和相关行为数据等安全环境，并生成风险评分。访问控制策略可以包括这种动态风险分值，以允许访问、拒绝访问或要求更严格的身份验证。

机器学习驱动的风险评分可以考虑各种因素来识别和应对风险:

*   试图访问的位置。
*   设备特征，包括固件类型、浏览器类型和操作系统。
*   设备健康，包括安全补丁、软件更新和反恶意软件保护。
*   角色、资历和历史访问日志等用户属性。
*   日、月、年的时间。
*   最近对权限或权限请求的异常更改。
*   异常命令或对资源的访问。

# 大规模实施安全策略

在大型组织中，可能有数百个应用程序和云服务以及数千个用户。需要在这个复杂的环境中实施标准化的策略，并可靠地检测违规行为。

机器学习可以通过根据行为模式的实时分析自动调整访问策略来提供帮助。这可以根据组织定义的中央策略自动完成，而不是让 IT 和安全团队不断审查访问请求并手动授予访问权限。

从最终用户的角度来看，这也提供了更好的体验，因为如果用户的请求是合法的，他们将获得快速访问，而无需等待手动批准。

# 改善用户体验

传统上，在改进的安全性和积极的用户体验之间有一个缺点。更安全的系统通常令人沮丧且难以使用。组织现在认识到安全性应该是透明的，对用户来说是非侵入性的，这将鼓励用户采用安全的身份验证机制，并将提高整体安全性。

早期的多因素身份认证(MFA)对用户体验有负面影响，并受到用户和员工的反对。许多组织倾向于不采用 MFA，或使其成为自愿的，以避免吓跑用户。然而，新一代零信任技术正在改变这一点。

基于机器学习技术的现代认证将用户的安全环境考虑在内，可以实现更简化的用户体验。一个例子是[无密码认证](https://frontegg.com/blog/what-is-passwordless-authentication)系统，它可以通过组合动态认证因素来认证用户。另一个例子是，从办公室使用公司工作站登录的用户可能使用 SSO 身份验证令牌登录，甚至不需要提供密码。相反，如果用户试图远程登录或在不正常的时间登录，身份验证系统可以传达登录尝试不正常，需要额外的身份验证。

这种类型的差别认证是有意义的，并且将会得到用户的支持，尤其是当他们的日常登录过程是积极的时候。基于 ML 风险评分的差异认证可以将用户转变为组织安全工作的合作伙伴。

# AI/ML 在零信任技术中是如何使用的？

几项安全技术在零信任部署中起着关键作用:

*   **下一代防病毒(NGAV)** —用于验证最终用户设备的健康状况，并使零信任系统能够阻止被入侵设备的访问。
*   **扩展检测和响应(XDR)** —用于跨越孤岛，从混合环境中收集数据，以检测和响应复杂的威胁。
*   **用户和事件行为分析(UEBA)** —用户和服务帐户行为分析背后的引擎，它是零信任访问方法的基础。

# 人工智能在下一代反病毒中的应用

NGAV 通过检测与已知文件签名不匹配的新攻击来改进传统防病毒软件。它可以抵御零日攻击、无文件攻击和改变源代码以避免被传统防病毒软件检测到的规避恶意软件。

NGAV 利用几种机器学习技术来检测未知威胁:

*   **静态特征** —将单个二进制代码或汇编文件与已知恶意软件的代码片段进行比较。
*   **字符串分析** —分析来自可执行程序的可打印字符串，这些字符串可能出现在 URL、文件路径、菜单、API 命令、配置文件或命令行输出中。
*   **N-grams 分析** —分析从可疑二进制文件中提取的字节序列。
*   **熵** —分析代码的统计方差，识别加密或混淆代码。
*   **可视化二进制内容** —将恶意软件的二进制代码转换为图像格式(其中每个字节都是图像中的一个像素)，并使用计算机视觉技术对其进行分析。
*   **控制流图(CFG)** —使用控制流图来比较可疑恶意软件和已知恶意软件。

# 扩展检测和响应(XDR)中的人工智能

[XDR 是一种新的安全技术](https://www.cynet.com/xdr-security/understanding-xdr-security-concepts-features-and-use-cases/)，它关联和分析来自各种来源的数据，以识别规避的威胁。这有助于在零信任环境中检测和主动搜寻高级威胁。

XDR 解决方案利用机器学习技术将来自网络日志、端点和云日志的数据点缝合在一起，以检测攻击并实现更简单的调查。该数据具有很高的维度(通常一次攻击的行为维度超过 1000)。对组合攻击数据的分析利用了:

*   **自动化数据预处理** — XDR 解决方案从许多来源聚合数据，将其标准化并自动清理，以便在机器学习算法中使用。从某种意义上来说，XDR 解决方案是一个“盒子里的数据科学家”，从 IT 系统中提取实时数据，并将其转化为标准化的数据集，可以馈入无监督和有监督的 ML 算法。
*   **无监督机器学习** — XDR 解决方案使用无监督算法来创建用户和设备的基线行为，并识别对等组，这有助于确定特定实体的正常行为。然后，它可以比较过去的行为、当前的行为和对等行为，以检测恶意软件、命令和控制(C & C)通信、横向移动和数据泄漏。
*   **监督机器学习** — XDR 解决方案识别网络上的各种实体，如 Windows 电脑、电子邮件服务器或 Android 手机。然后，它可以使用已知的可疑事件作为训练集，对与一类实体相关的流量和事件进行大规模监督算法训练。然后，该算法执行推理以在运行时识别可疑事件。这种方法可以极大地减少误报。

# 用户和实体行为分析中的人工智能(UEBA)

UEBA 是许多安全系统的基础，这些系统试图通过将当前行为与已知的行为基线进行比较来识别未知的威胁。

UEBA 系统的输入是日志数据。这些数据被处理成事件，被摄取到机器学习算法中，输出是每个事件的风险分数。商业 UEBA 系统在非常大的规模上这样做，以低延迟推理对实时大数据进行操作。

自动 UEBA 的一个重要部分是其自动化数据预处理。它执行统计分析以查看哪些数据字段适合进行分析，提取对安全分析有用的变量或特征，并确定哪些可用算法最适合数据。

UEBA 使用非监督和监督方法来分析安全数据，动态决定哪种算法最适合每种类型的数据:

*   **监督算法** —当 UEBA 拥有预先标记的数据时，它会利用监督算法。例如，如果安全系统提供已确认事件的历史日志，则该日志被标记为恶意活动数据。UEBA 系统可以根据这些数据进行训练，并分析任何类似的数据来检测相同类型的恶意活动。UEBA 系统中常用的算法有逻辑回归、线性回归和深度神经网络。
*   **无监督算法** —在许多情况下，UEBA 没有带标签的数据集可供使用。在这种情况下，它使用像 K-means 聚类这样的算法，可以学习数据点之间的相似性。例如，聚类算法可以查看访问日志的大型数据集，识别统计属性，如频率、直方图和时序结构，并将它们分组为具有明显相似性的聚类。通常，异常行为将被分组到最小的群集中，这是系统可以发现可疑或恶意访问尝试的地方。

# 结论

在这篇文章中，我解释了零信任的基础，以及机器学习如何影响组织中零信任的采用。我介绍了三个案例研究，展示了机器学习是如何在零信任技术中使用的:

*   **下一代防病毒(NGAV)** —利用各种技术来分析被怀疑为不安全的二进制文件，并识别新的和未知的恶意软件中的恶意属性。
*   **AI in eXtended Detection and Response(XDR)**—执行自动化数据预处理，创建与多个 IT 系统相关的海量数据集，并使用监督和非监督方法将相关事件缝合在一起并识别攻击。
*   **AI in User and Entity Behavioral Analytics(UEBA)**——收集网络中实体的数据，动态决定对其应用哪种机器学习算法，并基于标记或未标记的数据集计算风险分值。

在未来几年，我们可以期待看到人工智能和机器学习在安全行业变得更加普遍。很快，强大的 ML 基础将成为任何零信任安全技术的关键部分。