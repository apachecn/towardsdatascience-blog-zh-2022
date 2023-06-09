# 硅谷数据工程师的一天

> 原文：<https://towardsdatascience.com/a-day-in-the-life-of-a-google-data-engineer-722f1b2206cc>

数据工程师在过去的 10 年里越来越受欢迎，但是数据工程师到底是做什么的呢？在我的经验中，数据工程师身兼多职，经常处于商业智能、软件工程和数据科学三角的中间。数据工程师的一个主要角色是与下游团队合作，如商业智能和数据科学，以了解业务的数据需求，并构建数据集成来提供这些数据。另一个角色可以是与软件工程师合作消费应用程序数据；典型的新软件开发工作，或“0 到 1”项目。数据工程师往往隐藏在暗处；监控数据质量仪表板、倾听工程冲刺以及在分析会议中偷听。一个好的数据工程师是你不会经常想到的；您的数据是按照 SLA 到达的，值是干净和有用的，并且您总是能够在任何生产发布中找到新数据。正因为如此，数据工程并不是像数据科学家那样性感的职业，但是如果没有数据工程师为他们提供新鲜干净的数据，数据科学家就无法创造价值。在许多较小的组织中，这对于软件工程师、分析工程师来说是典型的，很少；数据科学家做数据工程师的工作。

数据工程师使用 Java 等工具构建 API，使用 Python 编写分布式 ETL 管道，使用 SQL 访问源系统中的数据并将其移动到目标位置。数据工程师是数据领域的物流专家。

![](img/99578b630acfba22d09c47c0cf38609f.png)

资料来源:www.unsplash.com

**数据工程师需要什么技能才能成功？**

为了回答这个问题，我最近在[做了大量研究](https://blog.devgenius.io/become-a-data-engineer-in-2022-analysis-of-over-1-000-faang-job-postings-38784fa727a8)，分析了 1000 多份招聘信息。由于数据工程师涉及许多技术领域，技能也各不相同。常见的思路是编程语言:SQL、Python，偶尔还有 Java 极其突出。数据工程师的常用工具包括用于分布式数据处理的 PySpark、Redshift 或 Azure 等数据库以及 Kafka 或 Flink 等数据流技术。随着这些基础技术的建立，需求通常会向一个或多个方向倾斜。一些公司喜欢看到数据工程师配备像 Tableau 或 PowerBI 这样的数据可视化工具。许多其他公司更喜欢精通软件部署和使用 Docker 和 Glue 等工具的数据工程师。

这些技术是现代数据工程师的重要基础，但软技能很少被提及，并且是成功的数据工程师的关键组成部分。数据工程师必须是专业的沟通者，因为我的经验是，我们经常被夹在不同需求的不同团队之间。我们可能正在形成来自商业智能工程师团队的需求，并将它们转化为软件工程师的需求。能够熟练地驾驭这些相互竞争的需求、消除通信中的模糊性以及交付满足跨职能需求的数据管道是普遍面临的挑战。除了交流，数据工程师绝对不能停滞不前。技术在快速发展，今天相关的工具在 5 年前没有这么流行。随着工具的快速发展，数据工程师也必须如此。

数据工程师是创造性的问题解决者；通常打造新的路径来铺设基础设施和支撑架构，以领先于组织的需求。一个优秀的数据工程师还能够预见未来，并规划可扩展的系统，以满足不断发展的业务需求。

**硅谷数据工程师的一天**

**8:00**

我的一天从比利时华夫饼开始；这是成功的秘方。鲜切草莓和一杯咖啡让生活变得值得。

我登录查看邮件，希望没有收到任何关于管道故障的邮件通知。不可避免地，我有，这可能是最优先解决的事情，然后再做其他事情。坦率地说，这可以是从 15 分钟的修复到一整天的寻找 bug 的冒险。我个人的原则是:我想在业务之前找到失败的管道。如果企业忘记了我的存在，这又是成功的一天。不要误解我——我喜欢合作，但我更愿意谈论新的开发工作，而不是失败的管道。

**九点**

希望我能够快速回复电子邮件，快速诊断任何管道故障，然后继续前进。我在早上写代码效率最高，所以我尽量不把会议安排在 1:00 之前。今天，我答应了一个同事一个技术设计文档(TDD ),所以我将把我的注意力集中在那里。在 TDD 中，我填写关于新特性或项目的完整信息，最终在开始大规模开发工作之前进行同行评审，以确保我们在应该如何完成它上保持一致。

**12:00**

到了中午，我通常会开始对我的关注时间感到疲劳，我会在问题跟踪工具中检查其他承诺。我想确保我及时兑现了我的承诺，所以我确保添加更新，添加到待办事项列表中，创建早上可能发现的任何新的 bug 或功能请求，并将现有项目推向完成。

**1:00**

每天早上我都向自己承诺“今天我要吃一顿丰盛的午餐”——但这从未发生过。我更喜欢快速地吃我的午餐，以免分散我的注意力；这实际上让我保持高效。我已经学会避免吃高碳水化合物的午餐，更喜欢吃清淡的零食，而不是把自己吃得昏迷不醒。虽然；这种事情发生的次数比我愿意承认的还要多。

**1:30**

下午，我通常要参加两三个会议。我的第一次会议是我管理的一个多年项目的非技术利益相关者的签到。每周，我通常会研究新的工作成果，将它们转化为技术需求，或者自己完成新的开发，或者经常将它交给另一个数据工程师。我的第二次会议是与另一位工程师的工作会议。我们正在努力了解如何最好地部署管道，以便我们消耗负责任的资源量，并以这样一种方式集成不同的工具，以便更容易地扩展新的数据管道。我今天的最后一次会议是交接会议；我正在将一个项目移交给另一个数据工程师，希望确保我的知识不会在移交过程中丢失。我让他们了解我的代码的位置，我的设计原理图，以及项目的历史或发展。

**4:00**

我通常会留出一些管理时间，这些时间我会用在很多不同的地方。有时候，我努力指导新的团队成员。其他日子，我可能会利用这一天结束的时间来计划我的第二天早上。这段时间也偶尔用来构思新项目，寻找机会改进现有产品或功能，并寻找机会创造价值。发现机会的一个例子；我注意到一种趋势，即一组特定的文件经常导致违反类型的错误。我利用这一天结束的时间研究解决这个问题的不同方法，给不同的团队发电子邮件，告诉他们处理这个问题的方法，编写新的技术设计，并最终实现这个设计。这个项目节省了我团队中许多工程师的时间，因为它减少了每次出现这种类型违反错误时所造成的技术损失。

**6:00**

我结束一天的时间变化很大。当我付出 100%时，我的一天就结束了，油箱里什么都没有了。有时候，我 4 点就到了这里。就像通常一样，我会被一些发现所吸引，并一直工作到 8:00 或 9:00。它出来了，我总是可以解决故障或回答问题，这才是最重要的。