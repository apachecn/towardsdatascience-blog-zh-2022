# 众所周知，这是一个改变数据验证的开源工具

> 原文：<https://towardsdatascience.com/an-open-source-tool-to-change-data-validation-as-you-know-it-7d6fcb5e9301>

## data-diff 如何节省您验证数据模型的时间

![](img/9bc003dacb1ce1215b6052a4a2f6fa4b.png)

莱娜·科勒在 [Unsplash](https://unsplash.com/s/photos/barrier?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

如果您曾经不得不将您的数据仓库迁移到一个新的平台或帐户，那么您就会知道这是多么耗时和痛苦。最近，我不得不将我的数据从一个雪花帐户迁移到另一个帐户，重新配置所有的数据接收管道和编排。您很快就会意识到一个小小的变化可能(并且将会)破坏您的数据管道中的一切。

随着整个管道的迁移，需要检查迁移是否正确。

*   您如何知道所有的源数据都作为原始数据被接收到您的数据仓库中呢？
*   您如何知道您的数据模型配置正确？
*   你的表格中的数据看起来和其他仓库中的一样吗？

问题层出不穷。回答这些问题的唯一方法是通过数据验证。

如果你曾经验证过任何东西，那么你应该很清楚要花多少时间。这是乏味的，不用动脑筋的，而且相当烦人。作为分析工程师，验证是我们最不想做的事情。我们希望把时间花在解决棘手的问题、构建有洞察力的数据模型以及优化我们的数据管道上。没有验证一切看起来像预期的那样。

验证数据通常看起来像编写一堆计算行数和某些值的基本查询。有时它甚至意味着用相同的主键比较特定的行。我不知道你怎么样，但我只能坚持几个小时，然后我的眼睛就会疲劳，数字也会变得模糊。那么，如果有更好的方法呢？一个既能节省您宝贵时间，又能完成您需要的验证的解决方案？终于有了！而且，完全免费。

# Datafold 的开源 data-diff 是什么？

作为比较数据模型中代码变化的 Datafold 的超级粉丝，我非常兴奋地了解到这个团队正在创建的开源工具。data-diff 是一个开源命令行工具和 Python 库，用于比较两个不同数据库中的行。它允许您在同一个数据仓库平台内或跨两个不同平台比较数据库，使其成为用于任何类型迁移的理想工具。

以下是您可以比较的几个数据库:

*   一种数据库系统
*   关系型数据库
*   雪花
*   神谕
*   BigQuery
*   红移
*   数据砖

如果你想看完整的名单，你可以在这里查看[。](https://github.com/datafold/data-diff)

这个工具的工作原理是将表分成更小的块，然后对正在比较的两个数据库中的每个块进行校验和检查。它在表级别工作，允许您比较相同或不同数据库中的两个表。校验和检查会继续，直到工具找到不同的行。然后，它将输出两个数据库之间不同的行，节省您手动查找它们的时间。

该工具不仅可以用于在数据迁移过程中比较数据库，还可以用于几乎所有的数据验证场景。在使用 Airbyte 和 Fivetran 等工具时，您可以使用它来比较从关系数据库和数据仓库(如 Snowflake)获取的数据。当数据丢失时，您可以使用它来创建数据质量检查。您甚至可以使用它来帮助跟踪使用硬删除删除的数据行。

# 如何使用数据差异

Datafold 有一个 Github [repo](https://github.com/datafold/data-diff) ，里面有如何使用这个命令行工具的所有说明。我将向您介绍我做了什么，在这个过程中我遇到了一些问题，以及我是如何解决这些问题的。请记住，我是在比较两个不同雪花账户的表格。幸运的是，我在每个帐户中都设置了相同的用户、角色和仓库，这使得配置变得更加容易。

## 创造 URI

最大的障碍是编写在命令行工具中使用的连接字符串。这些连接字符串包含关键信息，如数据库用户、密码、帐户、数据库、模式、仓库和角色。在 Snowflake 以外的数据库中，您只需要您的用户名、密码和主机名。

雪花 URI 看起来像这样:

`snowflake://user:password@account/database/SCHEMA?warehouse=WAREHOUSE&role=role`

您将运行的命令是:

`data-diff DB1_URI TABLE1_NAME DB2_URI TABLE2_NAME [OPTIONS]`

如您所见，这非常简单。您只需要安装 data-diff 包以及适用于您的数据库类型的相应驱动程序。

`pip install ‘data-diff[snowflake]’`

完成后，您就可以为每个数据库编写 URI 了。

以下是我用在自己身上的:

**用户名和密码**

我使用我的 dbt 用户和密码进行测试，但是我强烈建议只为 data-diff 包或任何类型的开源工具创建一个用户。

**账户名称**

当你点击雪花账户右上角的用户名时，账户应该是下拉列表中粗体显示的 id 或昵称。确保包含您所在的地区和提供商。我没有包括这一点，我调试 URI 的时间太长了。因此，如果您的帐户是 madisonprod.central-us.azure，您可以使用**madisonprod . central-us . azure**作为您的帐户。

**数据库和模式**

这只是您要比较的表所在的数据库和模式的名称。请记住，这些可能会因帐户或数据库而异。在我的例子中，我的两个表都位于数据库 base 和模式 WEB_DBO 中。

**仓库和角色**

最后，您需要指定希望 Datafold 使用的仓库和角色。请记住，必须将角色分配给用于连接的用户。我还建议选择该用户和角色的默认仓库。在我的例子中，我使用的是 TRANSFORMER_WH 和 TRANSFORMER 角色，我也用它来运行我的 dbt 模型。

当我把这些放在一起时，我的 URI 看起来会像这样，但有两个不同的帐户:

`snowflake://dbt_user:password@madisonprod-central-us.azure/BASE/MADISON_DBO?warehouse=TRANSFORMER_WH&role=TRANSFORMER`

# 策划整个命令

一旦你创建了你的 URI，你就快完成了！您只需要指定想要比较的两个表的名称。

如前所述，data-diff 使用以下命令格式:

`data-diff DB1_URI TABLE1_NAME DB2_URI TABLE2_NAME`

用你上面写的 URIs 代替 URI DB1 和 URI DB2。在我的例子中，我想比较来自两个不同雪花帐户的订单表。我将这样写命令:

`data-diff “snowflake://dbt_user:password@madisonprod-central-us.azure/BASE/MADISON_DBO?warehouse=TRANSFORMER_WH&role=TRANSFORMER” ORDERS “snowflake://dbt_user:password@madisondev-central-us.azure/BASE/MADISON_DBO?warehouse=TRANSFORMER_WH&role=TRANSFORMER” ORDERS`

需要注意两件事:

*   请务必大写您的表名
*   包括你的两个 URIs 的引用

当我没有做这些事情时，我得到了一个帐户错误，然后是一个表错误。一定要注意小细节！

**指定主键**

如果您正在比较的表的主键与`id`的名称不同，那么您需要指定一个“键列”。这将告诉 data-diff 哪个列充当表的主键，并允许它准确地比较两个表。我的表的主键是`order_id`，所以我这样指定:

`data-diff “snowflake://dbt_user:password@madisonprod-central-us.azure/BASE/MADISON_DBO?warehouse=TRANSFORMER_WH&role=TRANSFORMER” ORDERS “snowflake://dbt_user:password@madisondev-central-us.azure/BASE/MADISON_DBO?warehouse=TRANSFORMER_WH&role=TRANSFORMER” ORDERS — key-column ORDER_ID`

**其他选项**

还有一些类似于`key-column`标志的选项，您可以添加到 data-diff 命令中以优化它。下面是一些我觉得有用的例子:

—列(允许您指定要比较的特定列，从而加快查询速度)

—交互式(向您展示幕后发生的每一步)

—二等分因子(增加数据集分割成的线段数)

尝试这些不同的选项，看看什么最适合你！根据您要比较的内容和数据表的大小，您可能需要使用更多的选项。

运行该命令后，data-diff 将打印出两个不同数据库中的主键。它应该是这样的:

![](img/8aac6f668cc4d0f8cae820bf425514b7.png)

这里 data-diff 打印存在于一个表中但不存在于另一个表中的主键。当键前面有一个+时，这表示这些行存在于第二个表中，而不存在于第一个表中。如果有-符号，则表示该行存在于第一个表中，而不存在于第二个表中。这里，正在打印的订单 id 出现在我的 URI 的第二个表中，而不是第一个表中。

# 调试您的 URIs

如果您收到与您的帐户相关的错误，建议您使用诗歌运行 data-diff。这将允许您打印您正在向 data-diff 提供的凭据，确保凭据输入正确。

为了做到这一点，您首先要克隆数据差异报告。

接下来，您可以安装诗歌及其依赖项:

`pip install poetry`

`poetry install`

在您克隆了 repo 并安装了 poem 之后，导航到 [snowflake.py](https://github.com/datafold/data-diff/blob/master/data_diff/databases/snowflake.py#L40) 文件并打印您的凭证。您的 init 函数应该如下所示:

```
def __init__(self,account: str,_port: int,user: str,password: str,*,warehouse: str,schema: str,database: str,role: str = None,**kw,):snowflake = import_snowflake()logging.getLogger(“snowflake.connector”).setLevel(logging.WARNING)print(account)print(user)print(password)print(warehouse)print(schema)print(database)print(role)logging.getLogger(“snowflake.connector.network”).disabled = Trueassert ‘“‘ not in schema, “Schema name should not contain quotes!”self._conn = snowflake.connector.connect(user=user,password=password,account=account,role=role,database=database,warehouse=warehouse,schema=f’”{schema}”’,**kw,)self.default_schema = schema
```

现在，您可以像前面一样运行相同的 data-diff 命令，但要以`poetry run python3 -m`开始。这将打印出 data-diff 在产生错误之前从您的 URI 注册的值。

让我们将其与典型的验证过程进行比较。

以前，当比较两个数据集时，您需要做这样的事情:

*   求 2021 年订单数:

`select count( distinct order_id) from orders where year(created_at)=2021;`

*   求 2021 年 5 月订单数:

`select count( distinct order_id) from orders where year(created_at)=2021 and month(created_at)=5;`

将订单差异缩小到 5 月份后，您可能需要下载数据集并在 Excel 中比较这两个数据集(除非您碰巧在同一个雪花帐户中有这两个数据集)。Excel 加载时间很长，如果导入大量数据，速度会明显变慢。在这里，您可以使用函数来比较哪些 order _ ids 在一个数据集中不存在，而在另一个数据集中不存在。

我这样做了太多次，这是一种痛苦。分析工程师有更好的事情要做！我们不想花时间去大海捞针。data-diff 消除了寻找数据集之间差异的整个手动过程。现在，您有了额外的时间来研究*为什么*订单 id 在一个表中，而不在另一个表中。这是我们喜欢做的调查工作，实际上有*的影响*。

# 结论

data-diff 是一个强大的验证数据的开源解决方案。它消除了一部分可能需要几个小时的手动验证。无论您是像我一样在帐户迁移后使用它来比较帐户间的表，还是使用它来验证您的编排管道，它都将为您腾出时间来完成真正重要的任务。data-diff 使我能够成功地验证我的帐户迁移，所用的时间只是比较每个模式中每个表所需时间的一小部分。

开源工具是现代数据栈的未来。但是，如果你想使用复杂的用户界面比较数据模型，一定要看看 datafold 的付费[工具](https://www.datafold.com/)。

更多关于分析工程和现代数据堆栈的信息，[订阅](https://madisonmae.substack.com/)我的免费每周时事通讯。