# 增强您的 ML 工作流程的五个 MLOps 技巧

> 原文：<https://towardsdatascience.com/five-mlops-tips-to-enhance-your-ml-workflows-d9689ba7c60f>

![](img/0ab67a1a47fbd1ea9f4e447e0e74674e.png)

图片来自 [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=1662204) 的[张秀坤·卡什](https://pixabay.com/users/domkarch-3283484/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=1662204)

组织中的机器学习成熟度和流畅性已经达到令人印象深刻的水平，并且在不断上升。一段时间以前，维护一个 ML 决策支持系统可能被认为是一个充满曲折的危险旅程，你每隔几个月左右就要开始一次。如今，它几乎渗透到了大多数组织的敏捷软件开发周期中。

这意味着，作为一个 MLE，你不能再忽视组织的其他部分而使用 d-word；“哦！但是数据科学与软件开发不同。ML 不再孤立于组织的其他部分，而是需要与您的软件一起发展。这意味着，你必须运营并积极维护你的机器学习管道。要做到这一点，你需要应用行业最佳实践，编写好的代码，等等。在您的组织中建立稳固的 MLOps 文化有助于您为客户提供最佳价值，并做出基于数据的准确决策。

这篇文章将讨论几个你可以关注的技巧，以提升 ML 管道的可操作性。这将是对这些技巧的一次高级演练。只要有可能，我会提供关于工具/库的建议。但是这取决于读者去评估它们的适用性和可行性。

# 1.可观察性:在训练和服务期间记录

![](img/e4210b5f56bdb632f1d79a954fca5fbd.png)

图片来自 [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=886462) 的 [Simon](https://pixabay.com/users/simon-3/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=886462)

你最不希望看到的事情就是在离开你的模型一夜训练，却不知道哪里出了问题之后，丢失了训练过的工件。为了避免这种不愉快的情况，在模型训练和服务过程中引入持久日志记录非常重要。一些需要小心的事情，

*   如果您经常训练许多模型，那么在开发和生产过程中考虑不同级别的日志记录(例如`DEBUG` 与`INFO`)，以避免日志在文件大小上膨胀。
*   考虑使用循环、压缩来管理写入日志的存储要求
*   在服务期间记录时要小心，在服务期间记录大量数据会导致更高的延迟

在 Python 中，你可以合并像 Python `logging`或`loguru`这样的库(标准`logging`库的简化版本)。如果您在大规模运行，每天记录数 GB 的数据，那么您可以使用[弹性搜索和一个日志检查仪表板，如 Kibana](https://aws.amazon.com/blogs/database/analyze-user-behavior-using-amazon-elasticsearch-service-amazon-kinesis-data-firehose-and-kibana/) 来有效地聚合和搜索大量日志。您还可以使用基于云的一体化日志工具，如 [loggly](https://www.loggly.com/) 、 [papertrail](https://www.papertrail.com/) 等。

# 2.避免生产中的流氓模型

![](img/11310e60956ebf015244dcfec336d53a.png)

图片由来自 [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=2581314) 的 [FrankGeorg](https://pixabay.com/users/frankgeorg-4935546/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=2581314) 拍摄

渗透到生产环境中的流氓模型会给公司带来严重的后果。这是一个潜伏很长时间的问题，会慢慢侵蚀性能。它们可能相当难以捉摸，因为模型仍将起作用，但会产生质量差/不准确的预测。因此，尽早采取安全措施来避免/检测问题非常重要。这里有一些你可以做的事情。

## 数据新鲜度检查

数据新鲜度检查确保您的模型是基于最近的数据而不是几个月或几年前的数据进行训练的。它们可以捕获数据仓库或数据接收管道中的问题，并通知您陈旧的数据。

这些检查可以像检查加载的数据集的最大时间戳一样简单。如果您正在使用像`dbt`这样的工具来转换数据和构建表，它提供了开箱即用的[新鲜度检查](https://docs.getdbt.com/reference/resource-properties/freshness)。

## 数据验证检查

数据验证和新鲜度检查一样重要。在数据验证中，

*   空值
*   分类特征的分布
*   数字特征的范围

这些检查中的一些(例如[检查空值](https://docs.getdbt.com/docs/building-a-dbt-project/tests#generic-tests))可以在`dbt`中实现。您还可以使用 Python 库，如`ydata — quality`(【https://github.com/ydataai/ydata-quality】)来简化 ML 工作流中的数据质量检查。

## 受祝福的模特

最后，在盲目地将一个新版本的模型投入生产之前，进行一些模型验证检查。支票可以是这样的，

*   新模型的验证精度必须大于或等于以前的模型
*   新模型的验证精度需要在[x，y]范围内

注意不要对你的祝福政策过于严格，因为这会导致生产中的陈旧模型。

# 3.自动化培训工作

![](img/88088900bcad250099de182599100ff5.png)

图片来自 [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=2917047) 的 [Gerd Altmann](https://pixabay.com/users/geralt-9301/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=2917047)

与其手动训练模型，不如为模型设置自动训练计划。不仅如此，它将为您省去手动触发培训任务的麻烦，而且还会迫使您编写生产就绪的可重用代码。但真正的好处是，你的模型总是最新的，最少的人工干预，不会随着时间的推移而变得陈旧。

您可以设置各种调度程序，从使用 bash 脚本的简单 [cron 作业到像](https://www.geeksforgeeks.org/how-to-schedule-python-scripts-as-cron-jobs-with-crontab/) [Apache airflow](https://airflow.apache.org/) 和 [Argo](https://argoproj.github.io/) 这样的工具。使用成熟的 too 而不是 bash 脚本的好处如下。

*   更容易定义复杂的工作流程(例如，引入分支、条件、循环等。)
*   更容易监控/跟踪工作流程的进度/错误
*   代码更具表现力，可读性更强，并且更少需要从头开始编写

# 4.跟踪实验

![](img/95faab267dd7cc7dc31fbcc4e9856ad5.png)

图片来自 [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=1201014) 的[翻拍工作室](https://pixabay.com/users/remazteredstudio-1714780/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=1201014)

> 如果你不能衡量它，你就不能改进它——彼得·德鲁克

随着您组织的 ML 用例及工作流的发展，很自然地会更频繁地产生更多的模型和相同模型的版本。随着这一点变得越来越明显，最好投资一种工具，将您的所有实验结果以及其他相关元数据聚合并集中起来，以实现您的 ML 实验的可见性/透明度。最重要的是，它们将帮助您基于模型性能做出特定于 ML 的决策(例如，选择最佳超参数，将模型发布到生产环境中，等等)。).

像 [MLFlow](https://mlflow.org/) (自托管)和[Weights&bias](https://wandb.ai/site)(基于云的 w 私有托管选项)这样的工具就是围绕这种需要而设计的。它们提供了方便的 API，可以帮助你组织实验，并以一种容易找到结果的方式呈现出来。该工具可以帮助您的一些示例场景如下:

*   搜索在特定或相对日期或时间发生的实验的日志
*   从超参数搜索中识别最佳超参数
*   日志记录/版本跟踪训练(例如，训练模型)/评估(例如，图、图表)工件

# 5.检测并避免培训服务偏差

![](img/5ad86857303705c405acdf7b830ddb45.png)

图片来自 [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=2315559) 的 [Kawita Chitprathak](https://pixabay.com/users/iamnotperfect-95839/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=2315559)

众所周知，很难做到的一件事是验证你的模型在服务期间的表现是否符合预期。如果没有，这通常被称为*训练-发球偏斜*。对于实时(即在线)提供机器学习模型的公司/用例来说，这通常是一个问题。训练和服役期间的主要区别，

*   在培训期间，我们进行*批处理*，而在服务期间，我们进行*在线处理*。
*   在服务期间，需要从持久存储中加载存储的已训练工件

由于这种差异，训练模块和服务模块之间的代码不可避免地会重复。培训和服务代码分离的需求加剧了这种情况。这种分离有助于我们维护*职责分离*，其中服务代码只负责服务。这种重复意味着什么？这意味着，作为一个人，你可能会在实现功能时无意中引入错误，

*   输入验证检查
*   处理缺失数据
*   特征工程

[谷歌机器学习规则的第 32 条规则](https://developers.google.com/machine-learning/guides/rules-of-ml#rule_32_re-use_code_between_your_training_pipeline_and_your_serving_pipeline_whenever_possible)建议在训练和服务管道之间重用代码。但是要正确地做到这一点，是一项费时费钱的工程工作。如果操作不当，会导致代码不可读和不可维护。这里有几个更实用的解决方案，直到您弄清楚如何在训练和服务管道之间重用代码。

*   为评估保留一个纵向(第二天)验证集。我的意思是，按时间分割数据。例如，前 3 个工作日生成训练数据，后 2 个工作日生成验证数据— [规则#33](https://developers.google.com/machine-learning/guides/rules-of-ml#rule_33_if_you_produce_a_model_based_on_the_data_until_january_5th_test_the_model_on_the_data_from_january_6th_and_after) 。进行这种分割时，要注意数据的季节性(例如工作日与周末)
*   记录培训/服务输入/预测。一些云服务提供了开箱即用(例如 [Google cloud 请求-响应日志](https://cloud.google.com/architecture/ml-modeling-monitoring-logging-serving-requests-using-ai-platform-prediction))。您甚至可以编写一个自定义管道，1)在训练时，简单地将原始输入、特征和预测记录到 SQL 表中，2)在服务时，记录到缓存中，然后缓存将数据卸载到 SQL 表中以保持低延迟
*   保持用同一种语言训练/服务代码，这对代码重用非常重要。

# 好处:标准化你的模型的 IO

![](img/5d8048778898288f02cfecc574778f71.png)

图片来自 [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=3176115) 的 [_Alicja_](https://pixabay.com/users/_alicja_-5975425/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=3176115)

我们刚刚谈到在模型训练和服务中使用相同的语言。但实际可行吗？试着说服后端工程师用 Python 写后端。现在是 2022 年，很有可能，你还是会输掉这场辩论。Python 正在增加其在后端开发中的主导地位。但是在大多数地方，由于性能(例如处理并发请求)和成熟度，Java 仍然比 Python 更受青睐。

这与在训练和服务逻辑之间重用代码的能力严重矛盾。幸运的是，不完全是！与语言无关的数据序列化框架，如 [Google 的](https://developers.google.com/protocol-buffers) `[protobuf](https://developers.google.com/protocol-buffers)`和 [Apache](https://avro.apache.org/docs/current/) `[arvo](https://avro.apache.org/docs/current/)`促进了语言之间的代码重用[。](https://developers.google.com/protocol-buffers/docs/tutorials)

`protobuf`指定各种对象的消息格式(如输入数据——批量或单个记录、预测等。).该消息使用提供的语法写成一个`.proto`文件。然后，您可以使用为您选择的语言提供的编译器来生成您指定的实体的类。

这意味着您不用手动编写模型的输入/输出模式，而是自动生成代码。这降低了将错误引入代码的风险。

# 结论

让我们总结一下我们的讨论。

*   随着组织拥抱机器学习，它不再是孤立的，而是与周围的软件融为一体
*   良好的 MLOps 实践，为您节省大量时间和金钱
*   在培训和服务期间记录数据，同时注意非功能性需求(例如存储、服务延迟)
*   尽早检测并防止流氓模型。数据可能是罪魁祸首。有保障措施来确保数据质量
*   自动化培训工作以避免生产中的陈旧模型
*   跟踪实验以做出基于证据的决策，并维护 ML 实验的详细历史记录
*   当心训练-发球倾斜。如果不采取适当的措施，可能很难检测到
*   使用数据序列化技术，如`protobuf`或`arvo`，标准化模型的 IO

# 参考

[谷歌的机器学习规则](https://developers.google.com/machine-learning/guides/rules-of-ml)