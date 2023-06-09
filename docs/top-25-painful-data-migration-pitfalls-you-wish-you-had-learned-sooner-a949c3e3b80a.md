# 25 个令人头疼的数据迁移陷阱，您希望自己能早点知道

> 原文：<https://towardsdatascience.com/top-25-painful-data-migration-pitfalls-you-wish-you-had-learned-sooner-a949c3e3b80a>

## 我希望在开始数据迁移之旅之前了解的经验教训

![](img/964fa36ff2f2e44b03d0f26b544bcddc.png)

照片由[西格蒙德](https://unsplash.com/@sigmund?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

“生活就像一盒巧克力；你永远不知道你会得到什么”——要是汤姆·汉克斯知道数据迁移(DM)的危险就好了。在周五下午晚些时候，收到一个可怕的“和解又失败了”的紧急 ping 会让你的心跳比早上的有氧运动还要快。

如果你参与过 DM，你会知道这是数据团队承担的最复杂的项目之一。DM 本身很简单，即我将数据从一个地方移动到另一个地方。但是执行有许多活动的部分，从技术映射到数据质量和协调数字到目标架构。让所有的元素都到位是一项艰巨的工作。

> 您不是在迁移数据，而是在为多年的技术债务买单

本文旨在通过真实世界的例子，帮助你列出一个你应该积极避免的陷阱清单，根据我过去十年的经验，帮助你在你的 DM 之旅中取得成功。

它分为五个小节:

*   **数据迁移规划**
*   **源和目标架构**
*   **数据质量**
*   **测试、测试和更多测试**
*   **不要让这些成为事后的想法**

让我们开始吧！

# ***数据迁移规划***

## 1.计划失败，计划失败

一个成功的 DM 有一个复杂的计划做后盾；有许多移动的部分和许多源和目标依赖关系。拥有一个突出风险、假设、问题和依赖性的计划是你在执行 DM 的成功结果时可以瞄准的最好的开始。

*例如:在将数据从商店 A 迁移到 B 的计划中，我将包括这些核心部分；利益相关者参与图、源和目标数据架构、数据分析、测试数据覆盖、数据映射文档、测试策略和执行计划、迁移迭代/完整加载活动。*

## 2.没有经验的团队

低估面前的艰巨任务肯定会让你的 DM 之旅失败。不要通过让初级/无经验的团队领导核心工作流来节省成本。投资一个有经验的团队，这个团队以前经历过 DM，知道这些陷阱。

例如:在采访项目团队成员时，我会问一些与数据挖掘相关的具体问题，比如“数据挖掘成功的核心风险是什么？”或者“迭代 vs 满负荷 DM 的利弊是什么？”或者“你如何确定好的测试数据覆盖率？”

## 3.低估成本

如果任何项目需要一个应急基金，那就是这个。许多参与者必须得到补偿:源系统、基础设施、数据测试、架构师、工程师、项目经理、scrum masters 等。用一个复杂的计划来确保你已经考虑了所有这些任务的成本。

*例如:当制定成本和预算计划时，我会避免使用基于直觉的成本估算。对于每个参与者，我会要求根据计划中的已知活动进行粗略的影响评估，添加适当的延误风险，然后提交预算以供批准。*

## 4.寻找独角兽

你不会找到一个能够回答所有项目问题的主题专家(SME)。然而，你会有许多拥有各种技能的中小企业。如果你能确保正确的 SME 相互沟通，帮助定义详细的任务和具有巨大影响的风险，那将是最好的。

*举个例子:面试时，专注于这个人的核心技能，我会避免让同一个人在团队中身兼数职，因为你会希望他们精通各自的工作流程。*

## 5.缺乏自主权

计划将 DM 作为一个大项目来运营，所有的道路都通向一个决策者，这将会导致挫败感，并使你慢下来。组织一个合适的团队结构，把自主权交给有经验的人。

例如:在管理资源时，我会在需要时为他们的工作提供指导和支持；然而，我会避免一直盯着他们。通常，我会让他们坚持自己的决定，让他们建立自信。

# 源和目标架构

## 6.没有人知道源系统问题

在大型迁移中，源系统是陈旧过时的。他们缺乏文件；他们没有明确的血统；他们没有明确的过程；没有字典。少数人知道系统的复杂性，有时他们坐在官僚程序的后面。在你的计划中考虑这种细微差别(见 11。)

*例如:我见过客户个人信息不正确的客户核心系统。在迁移之前的许多年里，这些都是不正确的，但是从来没有注意到，因为没有执行基本的概要分析。电子邮件和姓氏等不正确的元素导致了许多客户投诉；然而，由于缺乏文档和明确的补救流程，这些问题持续存在。*

## 7.基础设施不容易获得

我已经记不清为数据传输和测试建立一个源和目标环境需要几个月的时间了。作为计划的一部分，拥有一个开放的环境应该是一个关键的依赖和风险。您可能需要不止一个环境来进行不同类型的测试。可悲的是，即使在更新的云环境中，这种痛苦依然存在。

*例如:在计划迁移活动时，我会向相关团队提交一个请求，以尽快建立一个环境。IT 团队有时可以使用沙盒环境；但是，请确保这符合您需要的规格。*

## 8.没有意识到技术债务的程度

技术债是最好的一种债；你不用偿还，也没有任何后果！当然是讽刺。

几十年来，技术债务一直在源系统中积累；作为规划的一部分，有必要花一些时间来了解这种影响在您进行的 DM 活动中的作用。记住(3。)你可能不得不拿出一些预算来解决这些债务项目。

*例如:大多数组织都有一个技术性债务登记册；我通常以此开始，然后将已知的问题与计划中的活动联系起来，并建立一个依赖/滑动技术债务成本的图片。*

## 9.过于雄心勃勃的数据之旅

将数据从源迁移到目标会非常困难；一路上不能改正自己的错误，就会变得不可能。确保数据旅程的设置经过各种暂存环境，使您能够在不影响最终目的地的情况下协调数据并解决问题。

*例如:采购基础设施时(在 7)。我还将提供一个高级逻辑架构图，展示在使用新数据填充目标系统之前所需的多个登台环境。*

## 10.决定迭代和交付大爆炸

决定您打算如何迁移数据，并相应地设置基础架构。如果环境是迭代地迁移数据，确保你真正地迭代，而不是收集你的发布来制造一个大爆炸。

*举个例子:我见过无数次这种情况，早期的版本由于各种问题而延迟，后期的版本“迭代”超载。环境(7)不是为 big bang DM 设置的；我们就是这样做的，伪装成一个大的迭代。*

# 数据质量

## 11.期待高质量的数据

数据质量会一直很差。我从来没有开始过一个具有高质量数据的迁移项目，我可以提升和转移这些数据。这样的事件类似于猪飞。相反，在迁移数据之前，与源系统 SME 一起投入时间和精力来理解和提高数据质量。

*例如:我会从已知的关键数据表开始，并进行简单的分析，以查看大量的空值、格式、重复等。，以帮助衡量整体质量。*

## 12.没有对足够大的数据样本进行分析

数据有趋势；更多的数据有更多的趋势。使用大量数据样本来了解隐藏在明处的数据质量问题。您总会遇到一些边缘案例，这些案例无法在剖析练习中全部捕获；因此，在选择数据样本时要有意识。

*例如:我以前接触过中小企业，了解到企业何时对源系统进行了重大变更；我将那个特定时期的数据纳入了剖析工作，这导致更多的 DQ 问题浮出水面。*

## 13.不知道您的关键数据属性

您的目标结构可能只有有限数量的属性值得迁移。如果你明白什么是最重要的，这会有所帮助。100%的数据质量不是一个东西，所以你必须在什么是关键的和值得修复的，什么不是之间找到一个平衡。

*例如:我确保关键属性以两种方式定义；首先，目标系统 SME 从技术上确认哪些属性是关键的，其次，业务最终用户确定哪些属性满足最终业务目标。*

## 14.期望数据只有一个定义

当执行数据映射时，我们需要知道源数据属性的定义，包括它的质量度量。组织通常在数据治理方面投资不足；因此，这一信息不容易获得。

*例如:通常，我会对定义进行最好的猜测，并确保业务用户验证它。*

## 15.没有数据补救流程

发现数据质量问题是一场战斗；补救是另一个问题。特别是如果您有一个陈旧的源系统，其中的更改需要很长时间，那么系统故障转移的风险就很高。拥有合适的团队会带来回报(参见 2。)和流程，以找到一种在源中补救数据质量问题的快速方法。

例如:在项目开始时，我会为整个数据挖掘过程中出现的问题制定一个端到端的补救流程。这个过程将有助于解决 DQ 和其他技术问题，如批处理失败和数据损坏。

# 测试，测试和更多的测试

## 16.没有在验证测试上花费足够的时间

还记得被问到计划有没有懈怠吗？然后，奇迹般地，测试工作减少了，以使项目上线并承担可接受的风险水平。我不知道是否有人定义过什么是可以接受的。测试应该是数据挖掘之旅的核心，而验证是这个核心的中心。

*例如:我项目中的测试团队将执行两种类型的验证，数据的技术验证，确保它符合数据标准，以及功能验证，确保它符合最终业务需求。*

## 17.没有明确的验收标准

你可以继续测试直到母牛回家，但这可能是浪费时间。定义验收标准可以确保最终目标不会不断变化。此外，拥有验收标准有助于创建边界，这有助于规划核心任务。

例如:我使用一套标准，并随着项目的进展对其进行调整。这些标准包括“数据质量的可接受百分比”、“用户能否以相同的方式在源和目标中访问报告/数据”，或者“数据能否在源中加载的同时在目标中加载。”

## 18.缺乏早期性能考虑

性能测试通常是项目的最后活动之一，立即成为下一组技术债务。在一个漫长而艰巨的项目之后，很少有兴趣重新审视项目早期做出的设计和编码决策。更改它们既费钱又费时，将最重要的“上线”日期置于风险之中。通过确保从一开始就考虑性能来避免这种情况。

*例如:性能测试团队将从项目开始就参与到数据迁移的短期性能测试中。这提供了有意义的结果和性能开始恶化的阈值。*

## 19.跳过测试阶段

每个测试阶段都有一个目的；跳过或缩短它们会增加 DM 输送失败的风险。系统集成和回归测试因为项目期限紧张而大打折扣。确保就强大的测试团队达成一致，并且所有相关团队都了解其影响。

*举个例子:每个测试阶段都会被定义并且有一个明确的目标。我避免为了增加测试阶段而增加测试阶段，因为这会导致在交付时间紧迫时跳过这些阶段。*

## 20.缺乏明确的协调框架

游戏的目标是在目标中拥有与源中相同的数据集。最好的方法是建立一个协调框架，在整个数据之旅中检查数据是否丢失。

*例如:我不会只在源端和目标端进行对账；中间的每个检查点都将进行协调检查，以尽早发现问题并进行补救。*

# **不要让这些成为事后的想法**

## 21.不了解数据迁移的含义

处理数据和创建新的服务或架构需要咨询组织的核心团队。与安全团队合作对于避免未知的网络风险至关重要。与数据隐私团队合作对于正确处理个人身份数据至关重要。有必要与数据治理团队合作，以确保遵循正确的策略。

*例如:在迁移开始时，我会召开动员会，邀请所有受影响的/已知的各方讨论整个计划。此外，我会与相关团队澄清，以最好的方式让他们支持这一 DM 活动。*

## 22.缺乏环境容量规划

环境能经受住时间的考验吗？甚至在此之前，环境能经受住 DM 事件的考验吗？值得花时间与相关基础架构团队一起创建容量规划，以管理数据大小随时间的波动。当然，有了云技术，大部分工作都可以自动化。

*例如:创建一个简单的容量规划，添加所有当前正在迁移的数据和属性及其数据大小。然后，我会根据业务/数据战略增加%的数据，并确保在这些计算中包括删除冗余数据的容量增强措施。最后，您会得到一个合理的容量案例，可用于规划。*

## 23.不投资自动化

与其手动执行每个迁移事件，不如投资于自动化整个迁移序列；这包括自动化数据捕获、摄取、测试、传输和转换。

*例如:即使迁移是一次性事件，我也会确保开发一个自动化代码来帮助运行多个迁移事件，包括彩排。这也使得将来的执行变得简单。*

## 24.单人依赖

执行迁移需要很长时间；在整个阶段，关键的超级明星开始发光。由于缺乏知识共享文化，这些人变成了一个人的依赖者。

例如，创建一个知识共享门户，每个团队成员都可以在其中添加和标记信息。创建一个包含这些信息的新的加入者培训包，以便他们可以快速上手。

## 25.不断变化的变量

DM 旅途中有足够多的动人片段；尽量减少不断变化的变量。

例如:减少源系统中的新变化，尤其是当目标系统应该是你的最终目标时。创建灵活的目标模型以适应源测试和数据质量变化:对有限范围的数据属性进行 DQ 和验证测试。尝试一次限制太多的更改，以避免产生新的依赖关系，并使整体目标处于风险之中。

## 结论

哇——这是一个很长的列表。数据挖掘是任何人都可以从事的最具挑战性但也是最有回报的项目之一。本文应该涵盖了 80%的已知数据挖掘场景；当然，这里不会捕获边缘案例，所以请在下面的评论中分享它们。

如果你喜欢这类内容，可以看看我的其他帖子:

</top-5-data-architecture-trends-and-what-they-mean-for-you-ef7c07bfa755>  

如果您没有订阅 Medium，请考虑使用我的推荐链接订阅[。它比网飞便宜，而且客观上能更好地利用你的时间。](https://hanzalaqureshi.medium.com/membership)如果你使用我的链接，我会获得一小笔佣金，而你可以在 Medium 上获得无限的故事。

我也定期在推特上写东西；跟着[我这里](https://twitter.com/hanzalaqureshi_)。