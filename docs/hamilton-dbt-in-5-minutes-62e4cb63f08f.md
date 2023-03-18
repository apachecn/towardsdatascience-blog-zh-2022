# 汉密尔顿+ DBT 在 5 分钟内

> 原文：<https://towardsdatascience.com/hamilton-dbt-in-5-minutes-62e4cb63f08f>

## 一起使用这两个开源项目的快速演练

![](img/e19a005cad4bd9f6f20f1dff09798508.png)

配合得很好的东西。图片来自 Pixabay。

在这篇文章中，我们将向你展示在一个 [DBT](https://github.com/dbt-labs/dbt-core/) 任务中运行[汉密尔顿](https://github.com/dagworks-inc/hamilton)是多么容易。利用 DBT 令人兴奋的新 [python API](https://docs.getdbt.com/docs/building-a-dbt-project/building-models/python-models) ，我们可以无缝融合这两个框架。

Hamilton 是用 python 描述[数据流](https://en.wikipedia.org/wiki/Dataflow)的声明性微框架。例如，它非常适合表达特征转换的流程及其与拟合机器学习模型的联系。

DBT 是一个帮助人们描述由 SQL 组成的数据流的框架，现在有了最新的功能，甚至是 python！

虽然这两个框架乍看起来可能不兼容，甚至是竞争的，但 DBT 和汉密尔顿实际上是互补的。

*   DBT 最擅长管理 SQL 逻辑和处理物化，而汉密尔顿擅长管理 python 转换；有些人甚至会说 Hamilton 是 python 函数的“*DBT”*。
*   DBT 包含自己的编排功能，而 Hamilton 依赖运行 python 代码的外部框架来执行您定义的内容。
*   DBT 没有在“列”的层次上建模转换，而是在“表”的层次上建模。Hamilton 致力于让用户能够以一种可读、可维护、python 优先的方式描述“柱状”转换和“表格”转换。
*   DBT 专注于分析/仓库级别的转换，而 Hamilton 擅长表达 ML 相关的转换。

在高层次上，DBT 可以帮助你获得数据/在你的仓库中运行大规模操作，而汉密尔顿可以帮助你从中制作一个机器学习模型。DBT 可以在 SQL 的测试和文档方面提供帮助。汉密尔顿可以为您的 python 代码提供软件工程最佳实践、测试和文档故事方面的帮助(例如，汉密尔顿的[整洁的生产熊猫](/tidy-production-pandas-with-hamilton-3b759a2bf562))！

为了证明这一点，我们从 [xLaszlo 的 DS 教程](https://github.com/xLaszlo/CQ4DS-notebook-sklearn-refactoring-exercise)的代码质量中获得了灵感，并使用 DBT +汉密尔顿的组合重新编写了它。这与通过 [scikit-learn 的 openml 数据集功能](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html#sklearn.datasets.fetch_openml)获得的 Titanic 数据集一起玩。

虽然我们只指定了一个由 DBT 执行的带有汉密尔顿的 python 任务，但这足以让您开始自己的旅程，将汉密尔顿与 DBT 结合起来！

# 不熟悉 DBT 或汉密尔顿？

为了使这篇文章简短，我们假设对这两者都很熟悉。然而，如果你想了解更多关于这两个项目的信息，这里有一些链接:

对于 DBT:

*   [什么是 DBT](https://www.getdbt.com/product/what-is-dbt/) ？
*   github—[https://github.com/dbt-labs/dbt-core](https://github.com/dbt-labs/dbt-core)

对于汉密尔顿:

*   [介绍汉密尔顿](/functions-dags-introducing-hamilton-a-microframework-for-dataframe-generation-more-8e34b84efc1d)(背景故事和简介)
*   [汉密尔顿+熊猫 5 分钟](/how-to-use-hamilton-with-pandas-in-5-minutes-89f63e5af8f5)
*   [在笔记本上与汉密尔顿迭代](/how-to-iterate-with-hamilton-in-a-notebook-8ec0f85851ed)
*   [与汉密尔顿一起整理生产熊猫](/tidy-production-pandas-with-hamilton-3b759a2bf562)
*   github—[https://github.com/dagworks-inc/hamilton](https://github.com/dagworks-inc/hamilton)

# 给汉密尔顿一个 DBT 任务

**先决条件**:开发好基于汉密尔顿的代码，并准备投入使用。

**集成概述:**

1.  您将您的 DBT 项目定义为正常项目。
2.  创建一个表，作为 python `dbt`模型的输入。
3.  编写`dbt` python 模型，并将 Hamilton“driver”代码放入其中。
4.  像往常一样继续使用`dbt`。

## DBT 运行汉密尔顿的 python 代码:

关于所有代码，包括自述文件和说明，请参见汉密尔顿资源库的示例文件夹中的 [DBT 示例。请注意，我们在这里选择了](https://github.com/DAGWorks-Inc/hamilton/tree/main/examples/dbt) [duckdb](https://duckdb.org/) 作为`dbt`后端，因为它很容易在本地运行；对于其他后端，情况可能会有所不同——如果我们所拥有的最终没有为您工作，请让我们知道。在这篇文章的剩余部分，我们总结了一些要点来理解这两者之间的集成。

使用 DBT，你可以定义一个“dbt 模型”，如 python 函数所描述的，它接受`dbt`和`session`对象(更多细节见 [dbt 文档](https://docs.getdbt.com/docs/building-a-dbt-project/building-models/python-models))。这些能让你接触到你对 DBT 的定义。

在查看代码之前，先简单说明一下，我们正在使用`dbt-fal`适配器([链接](https://github.com/fal-ai/fal/tree/main/adapter#4--cool-feature-alert--environment-management-with-dbt-fal))来帮助管理 python 依赖关系。随着 DBT 对 python 支持的发展，我们预计 DBT 的官方支持在未来会有所改变(例如，参见本[讨论](https://github.com/dbt-labs/dbt-core/discussions/5741))；我们将更新这个例子，使它总是正确的。

在 DBT 运行 Hamilton 的代码如下所示:

```
import pandas as pd
# import our Hamilton related functions that will define a DAG.
from python_transforms import data_loader, feature_transforms, model_pipeline
# import Hamilton modules for making it all run.
from hamilton import base, driver

def model(dbt, session):
    """A DBT model that does a lot -- it's all delegated to the hamilton framework though.
    The goal of this is to show how DBT can work for SQL/orchestration, while Hamilton can
    work for workflow modeling (in both the micro/macro sense) and help integrate python in.
    :param dbt: DBT object to get refs/whatnot
    :param session: duckdb session info (as needed)
    :return: A dataframe containing predictions corresponding to the input data
    """
    raw_passengers_df = dbt.ref("raw_passengers")
    # Instantiate a simple graph adapter to get the base result
    adapter = base.SimplePythonGraphAdapter(base.DictResult())
    # DAG for training/inferring on titanic data
    titanic_dag = driver.Driver(
        {
            "random_state": 5,
            "test_size": 0.2,
            "model_to_use": "create_new",
        },
        data_loader,
        feature_transforms,
        model_pipeline,
        adapter=adapter,
    )
    # gather results
    results = titanic_dag.execute(
        final_vars=["model_predict"], inputs={"raw_passengers_df": raw_passengers_df}
    )
    # Take the "predictions" result, which is an np array
    predictions = results["model_predict"]
    # Return a dataframe!
    return pd.DataFrame(predictions, columns=["prediction"])
```

就函数中的代码而言——它看起来非常像标准的 Hamilton“驱动程序”代码。

我们:

1.  导入正确的 python 模块(详见注释)。
2.  创建一个“驱动程序”，传入正确的配置、模块和适配器。
3.  执行代码，将 DBT 提供的数据作为熊猫数据帧传入。代码特征化，符合机器学习模型，然后创建一些预测。
4.  返回“预测”的数据帧，这正是模型对整个数据集的预测。

作为一名从业者，你需要考虑的是将汉密尔顿与 DBT 整合:

1.  使用 DBT，您可以定义输入数据集。
2.  使用 Hamilton，您可以定义转换来表征输入，创建模型，然后使用它来预测相同的数据集。对这段代码的一个简单扩展是预测由`dbt`提供的不同数据集，并返回该数据集。

# 和汉密尔顿一起跑 DBT

要运行这个示例，您需要做三件事:

(1)检查汉密尔顿储存库。

```
$ git clone git@github.com:stitchfix/hamilton.git
$ cd hamilton
```

(2)转到 dbt 示例目录并安装依赖项(为此，我们鼓励使用[一个新的 python 虚拟环境](https://realpython.com/python-virtual-environments-a-primer/))。

```
 $ cd examples/dbt
$ pip install - r requirements.txt
```

(3)执行`dbt`！

```
# Currently this has to be run from within the directory
$ dbt run
00:53:20  Running with dbt=1.3.1
00:53:20  Found 2 models, 0 tests, 0 snapshots, 0 analyses, 292 macros, 0 operations, 0 seed files, 0 sources, 0 exposures, 0 metrics
00:53:20
00:53:20  Concurrency: 1 threads (target='dev')
00:53:20
00:53:20  1 of 2 START sql table model main.raw_passengers ............................... [RUN]
00:53:20  1 of 2 OK created sql table model main.raw_passengers .......................... [OK in 0.06s]
00:53:20  2 of 2 START python table model main.predict ................................... [RUN]
00:53:21  2 of 2 OK created python table model main.predict .............................. [OK in 0.73s]
00:53:21
00:53:21  Finished running 2 table models in 0 hours 0 minutes and 0.84 seconds (0.84s).
00:53:21
00:53:21  Completed successfully
00:53:21
00:53:21  Done. PASS=2 WARN=0 ERROR=0 SKIP=0 TOTAL=2
```

这将修改一个代表我们数据库的 [duckdb 文件](https://github.com/stitchfix/hamilton/blob/096132f7703d87e9958128b372266014d994cd95/examples/dbt/data/database.duckdb)。您可以使用 python 或您喜欢的 duckdb 接口来检查结果。将 duckdb 替换为您在现实生活中选择的 db。

**恭喜你！**你刚刚和汉密尔顿一起跑过 DBT！

# 更多的细节

为了帮助您了解 DBT 示例的代码并根据您的需要进行修改，我们将代码组织成两个独立的 DBT 模型:

1.  [raw_passengers](https://github.com/stitchfix/hamilton/blob/main/examples/dbt/models/raw_passengers.sql) :这是一个简单的选择和连接，使用了 SQL 中定义的 duckdb 和 DBT。
2.  [train_and_infer](https://github.com/stitchfix/hamilton/blob/main/examples/dbt/models/train_and_infer.py) :第 10 行`dbt.ref("raw_passengers").df()`是这个`dbt`模型与(1)的链接。有了提供的数据，代码会:
    —特征工程提取测试/训练集
    —使用训练集训练模型
    —在推理集上运行推理
    要查看详细信息，请查看 [python_transforms](https://github.com/stitchfix/hamilton/tree/main/examples/dbt/python_transforms) 包中定义的转换。

注意(1):根据规定，在`train_and_infer`步骤中可以计算的内容中，Hamilton 只运行定义的转换的子集——我们可以很容易地请求它输出指标、返回拟合模型等。对于这篇文章，我们只想保持简单。

注意(2):再次强调前面的观点，我们使用`dbt-fal` [适配器](https://github.com/fal-ai/fal/tree/main/adapter#4--cool-feature-alert--environment-management-with-dbt-fal)来帮助管理这个例子中的 python 依赖关系。python 中的 DBT 仍处于测试阶段，我们将开放问题/做出贡献以使其更先进，并随着其 python 支持的发展更新此示例！

注意(3):如果你想使用[训练和推断](https://github.com/stitchfix/hamilton/blob/main/examples/dbt/models/train_and_infer.py)的输出，你可以像引用其他下游`dbt`模型一样引用它。

# 未来方向

我们认为汉密尔顿和 DBT 在一起会有一个漫长/令人兴奋的未来。特别是，我们可以做更多来改善这种体验:

1.  将 Hamilton 编译成 DBT 进行编排——我们正在开发的新的 [SQL 适配器](https://github.com/stitchfix/hamilton/issues/197)将很好地编译成 dbt 任务。
2.  添加更多自然的集成——包括汉密尔顿任务的`dbt`插件。
3.  添加更多不同 SQL 方言/不同 python 方言的例子。*提示* : *我们正在寻找贡献者……*

如果你对这些感到兴奋，就来看看吧！获得帮助的一些资源:

*   📣[加入我们的 slack 社区](https://join.slack.com/t/hamilton-opensource/shared_invite/zt-1bjs72asx-wcUTgH7q7QX1igiQ5bbdcg) —我们非常乐意帮助回答您可能有的问题或帮助您起步。
*   [DBT 支持](https://docs.getdbt.com/docs/dbt-support)页面。
*   [xLaszlo 的 CQ4DS 不和](https://discord.gg/8uUZNMCad2)
*   github 上的⭐️美国
*   📝如果你有所发现，请给我们留下一个问题

# 您可能感兴趣的其他汉密尔顿帖子:

*   [如何在 5 分钟内将汉密尔顿与熊猫配合使用](/how-to-use-hamilton-with-pandas-in-5-minutes-89f63e5af8f5)
*   [如何在 5 分钟内将 Hamitlon 与 Ray 配合使用](/scaling-hamilton-with-ray-in-5-minutes-3beb1755fc09)
*   [如何在笔记本环境中使用 Hamilton](/how-to-iterate-with-hamilton-in-a-notebook-8ec0f85851ed)
*   [一般背景故事&汉密尔顿简介](/functions-dags-introducing-hamilton-a-microframework-for-dataframe-generation-more-8e34b84efc1d)
*   [开发可扩展的特征工程 DAGs](https://outerbounds.com/blog/developing-scalable-feature-engineering-dags) (Hamilton with Metaflow)
*   [与汉密尔顿一起创建数据流的好处](https://medium.com/@thijean/the-perks-of-creating-dataflows-with-hamilton-36e8c56dd2a)(汉密尔顿上的有机用户帖子！)