# 我的预测建模和学习“分步”过程技术

> 原文：<https://towardsdatascience.com/my-predictive-modeling-and-learning-step-process-technique-f0521ee76d90>

## 概述了我将数据转化为机器学习模型的不同步骤

![](img/74bfcf6bb69766fbe9f13cca73440f75.png)

(图片由 [geralt](https://pixabay.com/images/id-6655274/) 在 [Pixabay](http://pixabay.com) 上提供)

# 介绍

从表面价值来看，预测建模和数据科学可能会令人望而生畏。当谈到预测建模时，为了有效，可能需要学习很多词汇和基础知识。随之而来的是一些统计知识，数据处理，还有很多。那么，一个实质性的问题是，我们刚刚讨论的所有内容需要协同工作，以创建一个单一的结果、一个准确度分数及其关联模型。如果不调查返回后数据可能有什么问题，所有这些事情也很少或没有任何推论。

数据科学领域提供的所有这些知识的复杂性和多样性肯定不会马上下载到某人的大脑中。人们不可能列出一堆方法名，然后期望有人马上永远记住它们。也就是说，一个人学习数据科学的任何部分或任何东西的一个很好的方法是，通过将每个大的任务分成小的任务来完成。今天，我想把我通常用数据和建模完成的任务进行划分，然后一步一步地安排它们，并解释我通常如何完成每个目标的细节。

# 第一步:数据争论

数据科学过程的第一站是让我们争论数据。“争吵”只是一个词，在这种情况下意味着收集。在这一步中，正如标题所暗示的，我们从各种来源收集数据。尽管一个重要的预防措施是确保您的数据是您的软件实际上可以读取的格式，但是所有这些数据通常都不是非常组织良好的。

数据争论通常非常简单，但肯定有一些细微差别和需要了解的事情可以节省大量时间和精力。我的第一条建议是，只争论你需要的数据，并且总是将数据输出为某种非专有的传统数据格式，比如。比如 CSV。这将允许你的项目中的数据是可复制的，如果你想拒绝或接受无效假设并做其他科学的事情，这肯定是合适的。在某些情况下，甚至可能不需要这一步，因为您可能正在处理已经存在于这些类型的文件中的数据集。

一般来说，每当我们争论数据时，我们的目标并不是用任何方法来保持数据的干净，而是仅仅将数据转换成某种可读的输入，然后我们可以对其进行处理。然而，您可以节省一些时间，潜在的存储空间，或潜在的内存，在这个过程中花一点额外的时间来清理您的数据。虽然这并不是每个人都做的事情，也不是必须的，但是它可以为你节省很多时间，包括整个下一步。

对于使用 Python 的人来说，这一步我要做的是数据收集的方法，比如 requests 模块或 ScraPy。如果您完全是新手，我建议您跳过这一部分，下载一个. CSV 或。网上的 JSON 文件。

# 第二步:预处理

数据的初始预处理不要太多。如果有像“日期”、“姓名”、“id”或类似的完全没有用的特性，那么最好也把它们去掉。您使用的功能越少，您需要执行的步骤就越少。然而，您拥有的具有统计学意义的特性越多，您就越有可能从您的项目中获得一个优秀的模型。

可能在预处理阶段需要做的最大的事情是从数据帧中删除丢失的值。如果我们在其他任何时候这样做，很可能我们所有的函数在遇到缺失值时都会返回错误，同样，在我们分割我们的特征后，我们会有多个名字充满了这样的缺失值。

对于 Python，您可能希望熟悉 Pandas 的 df.drop()函数和 df.dropna()函数。这两个函数可以分别用于删除列和丢失值的观察值。

# 第三步:分析

我通常采取的第三步是分析。既然数据至少能够被查看而不抛出错误，我们应该深入研究每个特性。如果我们的数据在头脑中已经有了一个目标，这是通常的情况，那么尝试分析可能更相关的特征——或者非常有效地证明与您的目标更相关。找出其中的古怪之处，找出平均值，最常见的值，有多少个类别，诸如此类的东西。在分析完所有特征后，您应该有一些您认为与值非常相关的特征，这将有助于下一步。

对于 Python 程序员来说，在分析任何数据之前熟悉 numpy 和 scipy.stats 可能是个好主意。通过 matplotlib、seaborn、plot.ly 等实现可视化。也是快速了解更多特性的好方法。在此期间，继续进行并拟合基线模型也是一个好主意。一个基线模型将会让你更容易知道这个特性有多难生产。更重要的是，它会给你一个坚实的起点。

# 步骤 4:特征选择

下一步是特征选择。特征选择可能是整个建模过程中最重要的步骤之一。这是因为这些特征是建立一个能成功预测你的目标的模型的绝对关键。不良要素会导致不良预测，因此要素选择和要素处理可能是最难的两个部分，也是对模型影响最大的两个部分。

在此期间，从数据分析中得到的测试可以用来提供最有价值的特性。这也是您可以设计功能的地方。为什么您应该设计功能？工程特征是降低输入数据维数的一种好方法。如果你有两个特征，例如我们正在研究松果，我们有三个特征，宽度，高度和深度。我们可以将它们一起乘以体积，体积可能会比单独的特征更好地累积这些特征的强度。我的意思是这样想。

一个高 10 厘米，宽 6 米的松果有多大？相比之下松果的体积是 60 立方厘米。第二个是我们可以立即评估和比较的一个值。这样的例子在机器学习中比比皆是，特征选择很重要，因为它创造了这些价值。所有这些通常都是手工完成的，或者通过索引来完成。过滤值可以通过 pd 完成。DataFrame[BitArray]。获取数据帧上 BitArray 的索引只会删除基于条件的值。您还可以在这里使用映射函数将掩码映射到值。掩码只需要返回 0、1 或真/假值。

# 步骤 5:特征处理

我的数据科学过程的下一步是特征处理。特性通常需要编码、标准化和类似的重要步骤。在某些情况下，如果我们没有对输入数据进行处理，模型将无法预测出我们的输入数据。

在 Python 中，你可能想看看 SkLearn，以及 Tensorflow 中的一些其他工具，用于批处理和诸如此类的事情。编码器和定标器可能是这些操作最流行的选择，但实际上你的处理器可以是任何东西。通常，这些对象以某种管道包装器或 Python 文件的形式聚集在一起，因为我们通常会序列化该模型并自动进行特征处理。这是在一些测试中投入更多精力的另一个好理由。我们还需要再做一部分特征处理，因为这些线有些模糊。现在我们最有可能使用 train_test_split()方法进行一次 test/train/val 分割。这种子抽样随机观察，然后将它们分成两组不同的相同特征。我们之所以要在处理完数据后再这样做，而不是在

# 第六步:建模

可能感觉最大和最吸引人的一步是建模。在建模步骤中，我们将把这些数据仔细整合到输入数据中。然后，这些数据将作为输入提供给我们的机器学习模型。根据您的模型，超参数可能也需要在过程的这一部分进行调整。

这部分相对简单，因为库的输入通常被映射到两个简单的位置参数。确保尺寸正确，并将特征发送到模型中。得到一个预测，并在你的验证集上检查它，然后回去看看是否还可以做更多的事情来获得更好的准确性。最终，通过足够的工作，您将获得一个相当有信心的模型，可以准备好进行流水线操作。

# 第七步:流水线作业

最后一步是把你的东西用管道连接起来。您将希望在此管道中包含任何用于预处理的方法。重要的是处理是相同的，以便最终模型的输入保持相同的格式，因此每个特征集的输出也保持相同。

在大多数 ML 模块内部，通常有一个相当健壮的流水线接口。在 SkLearn 的例子中，您可能会在第一对模型中使用它，您可以使用 Pipeline 构造函数来创建一个新的管道。

# 结论

数据科学过程可能看起来令人生畏。仅仅浏览这些标题可能会让人不知所措。然而，用一种系统的方法一步一步地做事情，就像你在计算中经常做的那样，将极大地帮助你创建一个有效的工作模型。我希望这个模型能够成功地展示出一个好的模型需要做些什么。这项技术的伟大之处在于，当涉及到声明式编程时，将事情分解成步骤是非常有效的。话虽如此，但我认为它可以应用于生活和软件中不断学习不同的东西。祝你好运利用这项技术，我知道它一定会在这个应用程序中派上用场！感谢您的阅读！