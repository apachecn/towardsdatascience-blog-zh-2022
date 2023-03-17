# 机器学习项目中要避免的一个常见错误

> 原文：<https://towardsdatascience.com/a-common-mistake-to-avoid-in-machine-learning-projects-b2f32c1a029d>

![](img/c1e061488c0e61b9baccd4caaab44817.png)

马文·埃斯特夫在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

机器学习项目通常非常令人兴奋，如果开发得当，可能会给公司带来很多价值。然而，即使有些项目看起来相似，但每个项目都有其独特的特点，必须谨慎开发以避免错误。

急于解决项目并开始开发模型以获得结果可能会导致一些常见的错误。其中一些错误很容易被检测到，一些库甚至被编程来捕捉这些错误(比如当您向模型输入空值时)。

其他的，更加微妙，你的模型可能运行得很好，但是得到的结果计算不正确，可能不具有代表性。

# **场景**

Y 你是一名数据科学家，任务是开发一个模型来帮助医生分类患者是否有更高的几率患心脏病。

让我们来看看为这个项目开发机器学习模型所采取的步骤，该模型使用在医疗检查中获得的医疗历史信息来分类给定患者在未来是否有机会患心脏病。

您希望看到，使用所有这些信息，您的模型是否能够检测出某个特定患者是否应该得到额外的关注，因为他/她患心脏病的可能性更大。

![](img/27df0835ca9855494ff397061d142183.png)

照片由[马库斯·温克勒](https://unsplash.com/@markuswinkler?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

# 逐步地

*   **您读取数据:
    -** 信息包含在不同的表中，因此您读取所有的表并执行必要的连接以获得单个数据集。
*   **执行探索性数据分析**:
    ——检测数据集中是否存在空值。
    -对于某些特性，您看到移除具有空值的样本是有意义的，因为它们代表可用数据的一小部分，所以您不会丢失太多信息；
    -对于布尔特征，你决定使用列的模式来替换空值；
    -对于一些数字特征，用列的平均值替换空值。
*   **执行一些特征工程**:
    ——你正在使用的数据有一个时间戳列，所以你创建一些新的特征比如考试的日、月、年，使用这个特征；
    -您正在使用的数据有一列是受试者的年龄，因此您创建了一个新列，将该患者归类到您创建的几个年龄范围之一。
*   **选择一个机器学习模型**:
    ——你把你的数据分成训练集和测试集。然后，使用训练集训练一些模型，并使用测试集评估它们，保存每个模型的结果并选择最佳模型。
    -在评估过程中，您已经使用贝叶斯搜索或 Optuna 优化了每个模型的超参数。
*   **创建一个报告:**
    ——你创建一个有几个图形的报告，计算你认为必要的所有指标，以显示这个模型是一个好的工具，应该使用；你的结果很有希望，所以你跑去见你的老板，他已经梦想着新的晋升。

![](img/35319a0728278ebb14f4a12149bc26d3.png)

照片由 [krakenimages](https://unsplash.com/@krakenimages?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

# **都好吧？**

这种循序渐进的方法看起来是开发所提议模型的一个很好的第一步，但是它包含了一个微妙而关键的错误。你能察觉到吗？

![](img/1b0b3044e1bca6ab6346dac5ebc56004.png)

[Luis Tosta](https://unsplash.com/@luis_tosta?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

# 它所包含的问题有一个很有名的名字:**数据泄露。**

D 数据泄漏是指在训练阶段，您的模型可以访问测试集的信息，而这些信息在这个阶段是不可用的。

这意味着从测试阶段获得的结果很可能被高估，而你的模型在投入生产时很可能表现不佳。

为了检验该模型是否能够很好地概括，必须用以前从未见过的数据对其进行测试。但是，在处理空值时，您会意外地将测试样本中的信息输入到训练样本中。

![](img/1a97b06e5b014665d4ca1a2d76b4989e.png)

[Jelleke Vanooteghem](https://unsplash.com/@ilumire?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

W 当您用列的均值或 de 模式替换空值时，您使用了所有数据集来计算这些指标。因为当您计算均值/模式时，训练集和测试集仍然在一起，所以来自测试数据的信息会泄漏到训练数据中。

稍后，当您的模型使用训练集进行训练时，它可以访问有关测试集的一些信息。这使得模型“知道”一点测试数据，当它后来被测试时，测试集对它来说并不是一个全新的东西。

这似乎是一个小问题，但由于机器学习模型应该能够很好地概括，它们必须用全新的信息进行测试，以模拟生产环境可能的样子。

# 好了，现在我们如何解决这个问题？

![](img/5087ce04bb2ce22cb8ea578b6f6d58c6.png)

肯·苏亚雷斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

到避免数据泄露在这种情况下，在替换空值之前，首先你必须把数据分成训练集和测试集。然后，使用训练数据，计算列的平均值和模式。

这些度量将用于替换训练数据中的空值，以及测试数据中的空值。其余的步骤保持不变。

这样做，您只是避免了将有关测试数据的信息输入到训练数据中，而现在，当您的模型被测试时，它会被赋予全新的信息，并且可以对其进行适当的评估。

必须使用训练数据来计算像归一化和空替换这样的预处理步骤。然后，测试数据必须通过相同的步骤，但是使用通过训练数据计算的信息。来自测试数据的信息绝不能与训练数据混合。

![](img/a09a1b202f6999d7e964dfeeefb7cd46.png)

照片由 [Felicia Buitenwerf](https://unsplash.com/@iamfelicia?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

这种事可能发生在任何地方，任何人身上，所以如果这种事已经发生在你身上，或者你没有马上发现问题，不要难过。

这个特殊的问题不久前发生在我身上，我想就这个问题阐明一下。

感谢您的阅读，希望对您有所帮助。

欢迎任何意见和建议。

请随时通过我的 Linkedin 联系我，并查看我的 GitHub。

[领英](https://www.linkedin.com/in/alexandre-rosseto-lemos/)

[Github](https://github.com/alerlemos)