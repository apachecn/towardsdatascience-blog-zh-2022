# 基于张量流的 NYT 情感分析

> 原文：<https://towardsdatascience.com/nyt-sentiment-analysis-with-tensorflow-7156d77e385e>

![](img/a334f87d7d9905618eb9657f26ca99ac.png)

乔恩·泰森在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

现在我们进入拜登政府一年了，我开始想知道与前几任总统的头几年相比，他执政第一年的新闻报道有多积极。为了找到答案，我决定对过去四位总统执政第一年的每个月的 NYT 文章摘要进行情感分析。下面，我讨论/显示以下步骤的代码:

1.  使用 NYT API [纽约时报](https://github.com/michadenheijer/pynytimes#archive-metadata)
2.  使用[文本 Blob](https://textblob.readthedocs.io/en/dev/) 进行情感分析
3.  用 TensorFlow 训练我自己的情感分析模型
4.  比较模型
5.  可视化结果

完整代码，请点击这里下载 Jupyter 笔记本[。](https://github.com/abode118/NYT-Sentiment-Analysis)

## 第一步。使用 NYT API [纽约时报](https://github.com/michadenheijer/pynytimes#archive-metadata)

我们将导入所需的包，连接到 API，创建一个字典来保存我们的结果，提取相关的数据，将我们的字典保存到一个 JSON 文件中，这样我们就不必再次提取数据，并关闭我们与 API 的连接。

## 步骤 2:使用[文本块](https://textblob.readthedocs.io/en/dev/)进行情感分析

我们将再次导入必要的包。然后，我们将在一些随机选择的抽象上测试该模型，以检查其合理性。

当我运行这个抽查时，我注意到 TextBlob 相当不准确。例如，下面的摘要被标为“积极的”:*警方表示，午夜过后不久，曼谷一家高端夜总会发生火灾，造成至少 59 人死亡，200 多人受伤，当时狂欢者正在庆祝新年。*

## 第三步。用 TensorFlow 训练我自己的情感分析模型

因为 TextBlob 似乎做得不太好，所以我决定练习我的 ML 技能，并使用 TensorFlow 建立一个情感分析模型([本教程](https://jovian.ai/outlink?url=https%3A%2F%2Fwww.tensorflow.org%2Fhub%2Ftutorials%2Ftf2_text_classification)非常有帮助)。首先，我们将导入所需的包，并加载我们将用于训练/测试的数据集。我在一个 1.6 毫米的标注 tweets (标注为正面或负面)的数据集上训练了我的模型。然后，我们将以 80/20 的比例随机拆分用于训练/测试的数据，并将 tweets 及其标签重新格式化为 numpy 数组，这样我们就可以在训练模型时将它们作为输入进行加载。

接下来，我们将使用 keras 创建一个序列模型。我们模型的第一层会把句子作为输入，转换成数值的向量(这叫做“单词嵌入”)。幸运的是，有人创建了一个可以做到这一点的模型，可以从 tensorflow-hub 下载。我在这一层使用的模型可以在这里找到。然后我们将添加两个隐藏层和一个输出层。我们的隐藏层分别有 16 个和 8 个节点，并且都使用 ReLU 激活函数。我们的输出层有 1 个节点，因为这是一个二元分类问题，我们使用 sigmoid 激活函数。最后，我们将使用 Adam 优化器编译我们的模型，使用 BinaryCrossentropy 计算损失，并使用阈值为 0.5 的 BinaryAccuracy 计算准确性(如果我们的模型预测句子为肯定的可能性≥0.5，我们将把句子分类为肯定的)。

接下来，我们将留出一些训练数据用于训练过程中的验证。然后，我们将训练模型，评估结果，并使用混淆矩阵可视化我们的模型在测试数据上的表现。

因此，使用我们的测试数据集，我们有 79%的准确率。从我们的混淆矩阵中，我们可以看到，当推文是正面的，但我们预测它是负面的时，我们的大多数错误都会发生。所以，我们对模型有一点负面的偏见。但是，让我们看看这 79%是否比 TextBlob 做得更好。

## 第四步。比较模型

如果我们使用 TextBlob 对相同的测试数据集进行分类，我们只能达到 62%的准确率。注意:TextBlob 预测“中性”情绪以及积极和消极情绪。因此，这不是一个直接的比较，但仍然是有帮助的。在下面的测试中，我们随机将 TextBlob 的“中性”预测重新分类为“正面”或“负面”【TensorFlow 的 79%准确率明显优于 TextBlob 的 62%准确率。

两个模型在摘要样本上的性能的直接比较可以在下面找到。幸运的是，通过 TensorFlow 模型，我们现在可以准确地将关于火灾、死亡和伤害的标题归类为“负面”总的来说，它在对我们的摘要进行分类时似乎更加准确，尽管仍然不完善。

## 第五步。可视化结果

首先，我们将使用我们的模型来预测我们在步骤 1 中提取的所有摘要的情感。然后，我们将计算每个月积极/消极情绪的百分比，并将其添加到我们的字典中。

然后，我们将把数据重新格式化成只包含我们想要可视化的关键统计数据的数据帧。然后我们将创建一些图表来更好地理解结果！*(下载笔记本看看我是如何用 seaborn 创建这些图表的)*

## 结论

这个项目对我更加熟悉 TensorFlow 中的建筑模型非常有帮助。TextBlob 就是没剪！我在 TensorFlow 中构建的模型要精确得多，尽管它明显有一点负面偏差(正如我们从混淆矩阵中了解到的)。

我用张量流模型得到的结果非常有趣。基本上，在布什执政的第一年，新闻是最负面的。就所有新闻(只有 25%的正面)和直接提到“布什”的新闻摘要(只有 28%的正面)而言，都是如此。另一方面，在特朗普的第一年，新闻通常是最积极的(34%积极)，直接新闻报道对奥巴马最积极(63%积极)。有趣的是，直接提到拜登的摘要(57%)比直接提到特朗普的摘要(52%)更负面。这有点令人惊讶，因为 NYT 公开反对川普。

我打算对我提取和分类的数据进行额外的分析，包括查看最常用的词，以更好地理解这些意想不到的结果。例如，在特朗普执政的第一年，总体新闻报道最为积极，这可能是因为 2017 年我们的危机比 2001 年(互联网泡沫破裂、9/11 袭击)、2009 年(大衰退)和 2021 年(挥之不去的新冠肺炎疫情)更少。我期待着进一步探索这些数据！