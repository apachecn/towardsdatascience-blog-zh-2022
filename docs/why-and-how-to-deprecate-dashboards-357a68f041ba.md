# 为什么以及如何弃用仪表板

> 原文：<https://towardsdatascience.com/why-and-how-to-deprecate-dashboards-357a68f041ba>

## 当有太多的仪表板需要筛选时，建立一个自动化的弃用流程来保持 BI 实例的整洁。

![](img/1f0509e4df29b82a160c4a1be5bd25bb.png)

你的仪表盘没这么重要。图片来自 [Unsplash](https://unsplash.com/s/photos/siren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

一个人可以从多少仪表板中获得合理的洞察力并采取行动？虽然我不知道答案(这实际上会是一个很好的调查)，但我希望我们都同意极限确实存在。当我们先跑后走，不分析问题本身就为每个问题创建一个仪表板时，这就是仪表板膨胀的时候。

> **定义*:仪表板膨胀*** *是当浪费时间寻找相关的视觉来回答问题或重新创建已经存在的视觉时，对组织的影响。*

这种情况下的成本是时间(因此也是效率)。当 holy grail Looker 实例组织得很差时，增长主管花更多的时间在 CAC 上寻找一个有用的图表，按付费用户和免费用户进行细分，而不是实际改变增长策略。如果清洁度和组织性已经荡然无存，分析团队会重新创建已经存在的可视化效果。

仪表板膨胀可能是实施不当的[自助服务](https://sarahsnewsletter.substack.com/p/the-froyo-data-shop)策略的结果，也可能是自助服务的对立面，这取决于您的观点。自 2013 年以来，一些人一直说仪表板不是答案[，但 10 年后，它们会一如既往地被使用。虽然肯定会出现数据应用、自动化洞察和可操作化数据等新趋势，但仪表板在很长一段时间内不会有任何发展。](https://robbieallen.medium.com/dashboards-aren-t-the-answer-f591d5c5e3dc)

在他们已经存在的前提下，我们如何组织他们？

# 战略铺平了前进的道路

他们说创业公司赢是因为他们行动快，创业公司的分析团队也是如此。通常有这么多事情要做，所以在技术债务已经存在之前，很难将战略优先于每件事情的持续执行。

如果有人刚刚开始考虑 BI 工具策略，这一节是为你准备的。通常，战略会谈是在走得太远太快，需要往回走的时候进行的。在这种情况下，回答这些问题是关于你希望世界是什么样子，而不是今天是什么样子。

只有回答一些高层次的问题，如作为数据领导者，您如何展望您组织内部的商业智能，并积极地进行修正以尽可能与愿景保持一致，才能避免仪表板膨胀。

**💡思考整体分析策略。**

*   您如何决定优先考虑哪些数据产品，以及如何交付它们？
*   谁负责建模数据和构建可视化？

**💡具体说明商业智能的游戏规则。**

*   实例中的文件夹是如何组织的？
*   组织中谁拥有查看、编辑和管理权限？
*   可视化什么时候达到寿命终止状态并且不再被支持？

这些问题的答案因公司规模而异(50 人对 500 人？)、文化(集中式还是分布式分析团队？)，业务类型(B2B 还是 B2C？)，还有更多。然而，最后一个问题是最重要的。

让我们从头开始:这里是我开始的基本策略。

**分析策略**

1.  确保每个数据产品都有一个足够大的用途，以优先考虑数据产品的持续维护。目标应该与更大的公司目标相联系。
2.  确定并记录每个数据产品的 SLA。每个用户的属性来源真的需要通过 API 公开吗？它真的会被近乎实时地使用吗？(不，绝对不会。)
3.  每个数据产品应该回答一个问题，而不是一个陈述(即而不是“免费 vs 付费用户的 CAC”，做“免费和付费用户之间 CAC 什么时候最低？”).

**BI 实例组织**

1.  按团队组织文件夹中的实例。“常规”文件夹可以包含公司范围的指标，但不应过度使用。
2.  每个团队应该只对他们的团队文件夹有编辑权限，对其他所有内容都有查看权限(团队绩效或 PII 除外)。
3.  自动清理您的 BI 实例——这非常重要，值得单独列出一节。

# 清理是维持战略的关键

作为数据专业人员，我们都有过被称为任务关键型的问题。人们说情人眼里出西施；批判性也是。当我们发现每 15 分钟更新一次的仪表板实际上并没有被使用，当然，它可以被删除一次。但是，是什么防止了同样的问题再次发生呢？

**策略只和它的维护一样好，维护的一个很关键的部分就是弃用。**

虽然人工 QA 可能是一家公司内最政治正确的方法，但它是最手工的，也是最难优先考虑的。我提出了一个自动化的仪表板弃用策略，只需构建一次，需要最少的人工支持。

**自动化仪表板弃用策略会获取所有 BI 元数据，并自动删除一段时间内未使用的视觉效果。**

我将使用 Looker 浏览 Python 中的伪代码示例，因为它很受欢迎并且容易膨胀，同时还会给出其他企业工具的指南(因为几乎任何 BI 工具都可以做到这一点)。

让我们来谈谈技术。

## 1.编写一个脚本，将所有 BI 元数据转储到仓库中。

**初始化对实例的访问。对于 Looker，这是通过一个 [Python SDK](https://developers.looker.com/api/getting-started) 实现的。为实例 URL、客户机 ID 和客户机机密设置环境变量。其他 BI 工具可能也有官方 SDK(例如 [Tableau](https://tableau.github.io/server-client-python/docs/) ，其他的有非官方 SDK(例如 [Domo](https://github.com/domoinc/domo-python-sdk) )，或者你可能会发现直接调用 rest API 很方便(例如 [PowerBI](https://learn.microsoft.com/en-us/rest/api/power-bi/apps/get-dashboards?source=recommendations) )。**

```
import looker_sdk
sdk = looker_sdk.init31()
```

**直接通过 SDK 获取所有可以获取的数据。**对于 Looker 来说，最有用的信息是获取所有仪表盘、外观和用户。对于 [Tableau 服务器](https://tableau.github.io/server-client-python/docs/api-ref#workbooksget)，获取工作簿、视图和用户。无论是哪种 BI 工具，您都需要清理响应，要么将它转换为 JSON，要么只提取相关的特定字段(如 ID、名称、创建日期、用户)。

```
dashboards = sdk.all_dashboards()
looks = sdk.all_looks()
users = sdk.all_users()
```

**从内部使用统计报告中获取数据。**许多工具不直接通过它们的 API 公开用法和历史。然而，它们确实公开了原始数据集，比如在 [Looker](https://cloud.google.com/looker/docs/creating-usage-and-metadata-reports-with-i__looker) (i__looker 元数据)和 [PowerBI](https://learn.microsoft.com/en-us/power-bi/admin/service-admin-auditing) (原始审计日志)的情况下，或者带有 [Tableau](https://help.tableau.com/current/server/en-us/adminview_postgres.htm) 和 [Domo](https://domo-support.domo.com/s/article/360042934594?language=en_US) 的预建报告。任何任意 BI 报告本身都可以导出为 Python 中的数据集。我们真正需要的是一份报告，上面有每个视频的最新访问日期。

```
history_data = sdk.run_look(my_admin_look_id, result_format="json")
```

将所有这些输出写入仓库表。我不会用这个的代码片段来打扰你。用数据转储覆盖表(像所有视觉效果一样)，并追加随着时间推移而构建的数据(像历史访问)。这个过程应该是等幂的，并按计划(我建议每天)运行，以实现完全自动化。

## 2.对数据进行建模，以了解使用情况。

无论您选择如何转换数据，原始表都需要连接、透视和聚合才能有意义。

让我们回顾一下现有的来源:

*   每个视觉效果的表格(仪表板和 Looker 示例的外观)。称之为 *`looker_dashboard`* 和 *`looker_look`*
*   用户表。称之为 *`looker_user`*
*   历史访问表(原始或汇总至最近访问日期，根据视觉效果)。称之为*` looker _ historical _ access `*

我们需要的结果是一个表，其中每个视图都有一行，分别是创建它的时间、创建它的用户以及最后一次查看或编辑它的日期。粗略的查询可能如下所示:

```
with history as (
    select visual_id,
           max(access_date) as latest_access_date
    from looker_historical_access
    group by visual_id
), dashboards as (
    select
        id as visual_id,
        name as visual_name,
        user_id as visual_created_user_id,
        created_at as visual_created_at,
        'dashboard' as type
    from dashboard
), looks as (
    select
        id as visual_id,
        name as visual_name,
        user_id as visual_created_user_id,
        created_at as visual_created_at,
        'look' as type
    from look
), visuals as(
    select * from dashboards union all select * from looks
)
select
     v.*,
     coalesce(h.latest_access_date, v.visual_created_at) as latest_access_date,
     u.email
from visuals as v
left join history as h on h.visual_id = 
left join user as u on v.visual_created_user_id;
```

我忽略了几件事:

*   历史访问表可能已经聚合，也可能尚未聚合，这取决于您的源报表。它也可能不包含一个干净的“*visual _ id”,*因此必须派生它。
*   你必须联合不同的视觉信息(无论是仪表板和 Looker 的外观还是 Tableau 的工作簿和视图)。
*   有时，创建视觉效果并不算访问它，所以你需要确保最近创建的视觉效果没有被标记为删除。
*   当您开始引入用户访问数据、文件夹结构等等时，数据会变得更加复杂。如果基于这个过程，您可以在不同的 dbt 模型中更严格地构建它，但是我已经为初学者采用了最简单的方法。
*   要通过 Slack 提醒用户，您需要将他们的电子邮件映射到 Slack 用户名。
*   如果它是一个表而不是一个视图，请按计划更新它。

## 3.弃用前自动警告用户，然后删除视觉效果。

因此，我们已经获得了仓库中的所有数据，并且我们知道哪些视觉效果最近没有被使用(我通常建议将 60 或 90 天作为“不是最近”的阈值)。BI 工具通常在数据团队之外大量使用，那么应该如何沟通这种努力呢？

**传达努力的原因。**组织沟通永远是最难的一步。在开始弃用之前，记录并向更广泛的组织传达保持一个干净的 BI 实例的好处(…或者如果您愿意的话，可以传阅这篇文章)。目的不是删除别人的作品；这是为了让公司的每个人都能更快地从数据中获得洞察力。

**为自动化沟通创建一个反对松弛渠道。**BI 工具的任何用户都应被添加到此渠道。

**查询 X-7 天内未被访问的视觉效果，并发送一条 Slack 消息**。如果在 60 天的空闲时间删除，包含的视觉效果应该是 53 天未使用的，或者如果在 90 天的空闲时间删除，应该是 83 天未使用的。为每个视频发送一个 Slack，标记创建它的用户。

```
# Everything below is pseudo-code, with utility methods abstracted away
deprecation_days = 60
warn_visuals = get_warehouse_data( # Pseudo method
    f'''
    select visual_name, created_by_user
    from modeled_looker_data
    where current_date - last_accessed_date = {deprecation_days - 7}
    ''')
slack_message_template = '''
    Visual {{visual_name}} created by @{{slack_username}} will be 
    deprecated in 7 days. If this is incorrect, please contact the 
    analytics team.
'''
for visual in warn_visuals:
    send_slack_message(slack_message_template, visual) # Pseudo method
```

**查询准备删除的图像，并通过程序删除它们。在准备好要删除的视觉效果列表后，你必须遍历它们并删除每一个。在迭代中，可能会有不同的方法与不同类型的视觉效果相关。在对该步骤中使用的数据建模时，可以存储该类型。这些图像中的每一个之前都应该有警告信息。**

```
deprecation_days = 60
delete_visuals = get_warehouse_data( # Pseudo method
    f'''
    select visual_id
    from modeled_looker_data
    where current_date - last_accessed_date >= {deprecation_days}
    ''')
for visual in delete_visuals:
    visual_id = visual['visual_id']
    if visual['type'] == 'look':
         sdk.delete_look(visual_id)
    else:
         sdk.delete_dashboard(visual_id)
```

就像木工里说的:量两次，切一次。在删除东西的时候，通过注释掉实际的删除来运行自动化过程几个星期，以确保逻辑是合理的。

# 最后的想法

这篇文章的标题是有目的的:我发现清理令人兴奋，这包括自动清理 BI 实例。他们说，当你的工作空间整洁时，工作效率就会提高，那么为什么你的 BI 实例不能同样整洁呢？

最后，如果我不在下面补充几点意见，那将是我的失职。

大多数企业工具没有自由层，这就是为什么我包括了一个粗略的代码大纲，而不是难以持续测试的特定代码片段。我也没有提到更新的或者代码繁重的 BI 工具，比如超集、Lightdash、Metabase 等等。尽管我建议无论使用何种工具都使用这种方法，但是 API 中公开的特定端点可能会有所不同。

提到数据目录在元数据工作中的作用是很重要的。虽然大多数[现代数据目录](https://sarahsnewsletter.substack.com/p/choosing-a-data-catalog)连接到 BI 工具并为您收集元数据，但它们(还)没有完全建立为主动和删除视觉效果。然而，中间的一个折中方案是直接从一个集中的数据目录中导出 BI 元数据，然后自己编写弃用逻辑。这种方法仍然需要处理 API 之类的东西。

【sarahsnewsletter.substack.com】原载于<https://sarahsnewsletter.substack.com/p/the-thrill-of-deprecating-dashboards>**。**