# 变形金刚的概念和训练变形金刚模型

> 原文：<https://towardsdatascience.com/the-concept-of-transformers-and-training-a-transformers-model-45a09ae7fb50>

## 变压器网络如何工作的逐步指南

![](img/f41e76209b1eca0a253c5f948b977b11.png)

[来源](https://pixabay.com/illustrations/artificial-intelligence-brain-think-3382507/)

## **什么是自然语言处理**

自然语言处理是人工智能的一个分支，致力于赋予机器理解人类语言的能力。它使机器有可能阅读文本，理解语法结构，并解释句子中使用的单词的上下文含义。它在机器翻译中用于从一种语言翻译到另一种语言，常用的 NLP 翻译器是 *Google Translate* 。谷歌翻译可用于将文档和网站从一种语言翻译成另一种语言，支持 133 种不同的语言。OpenAI GPT-3 是创建的最先进的自然语言处理模型之一，它执行各种各样的语言任务，如文本生成，问答和文本摘要。情感分析是自然语言处理的一个重要分支，组织使用它来分析产品评论，以区分正面和负面评论。文本生成是自然语言处理的一个有趣的领域，它被用在手机的自动完成功能中，用于适当的单词建议，以及完成我们的句子。

NLP 有不同的分支，我将解释其中的一些。

*   **情感分析**:对文本进行分析，将文本的情感分为正面或负面。
*   **文本生成**:是文本的生成，在文本生成中我们提供文字提示，NLP 模型自动补全句子。
*   **文本摘要**:是利用 NLP 技术将长句归纳为短句。
*   **语言翻译**:使用自然语言模型将文本从一种语言翻译成另一种语言，例如将英语句子翻译成法语句子。
*   **屏蔽语言建模**:利用 NLP 模型对句子中的屏蔽词进行预测。

## 什么是变压器网络

Transformer 是一种神经网络架构，旨在解决自然语言处理任务。变压器网络使用一种称为注意力机制的机制来研究、理解句子中使用的单词的上下文，并从中提取有用的信息。《变形金刚》在流行论文 [*中被介绍*由Ashish Vaswani 等人](https://arxiv.org/abs/1706.03762)。

## **变压器网络的类型**

我们有三种主要类型的变压器网络，即编码器、解码器和序列 2 序列变压器网络。

**编码器变压器网络**:这是一个双向变压器网络，它接收文本，为句子中的每个单词生成一个特征向量表示。编码器使用自我注意机制来理解句子中使用的单词的上下文，并从单词中提取有用的信息。

编码器如何能够理解这个简单的句子“*编码是惊人的*”的图示。

![](img/3a8f945785ba91ace802e411c32818c7.png)

作者图片

*   **图分解:**编码器利用自我注意机制为句子中的每个单词生成一个特征向量或数字表示。单词“*”被分配一个特征向量 1，单词“ ***被分配一个特征向量 2，单词“*”被分配一个特征向量 3。单词" ***是*** "的特征向量既代表了单词" ***是*** "的上下文，又代表了其两侧单词的特征向量信息，即" ***编码*** "和" ***惊人的*** "，因此得名双向网络，因为它研究的是左右两侧单词的上下文。特征向量用于研究单词之间存在的关系，以理解所用单词的上下文，并解释句子的意思。想象一下，这是一个情感分析任务，我们要对这个句子的情感是积极的还是消极的进行分类。通过研究每个单词的上下文以及它与每个单词的关系，网络已经能够理解这个句子，因此将这个句子分类为肯定的。它是正面的，因为我们描述的是“ ***编码*** ”而我们用来描述它的形容词是“ ***惊艳*** ”。*****

*编码器网络用于解决分类问题，如掩蔽语言建模以预测句子中的掩蔽词，以及情感分析以预测句子中的正面和负面情感。常见的编码器变压器网络有 BERT、ALBERT 和 DistilBERT。*

***解码器变压器网络或自回归模型**。它使用掩蔽注意机制来理解句子中的上下文以生成单词。*

*![](img/d480af9b63d21caed48b61e10d066959.png)*

*作者图片*

*   *想象一个简单的场景，我们在包含各种电影信息的文本语料库上训练解码器网络，以生成关于电影的句子或自动完成句子。我们将这个不完整的句子“ ***【漫威复仇者联盟残局】*** 传入解码器模型，我们希望模型预测合适的单词来完成这个句子。解码器是一个单向网络，它为每个单词生成特征向量表示。它与编码器网络的区别在于它是单向的，而不是编码器的双向特性。解码器从单个上下文(右或左上下文)研究单词表示。在这种情况下，它会研究左边单词的上下文来生成下一个单词。它生成下一个单词“ ***是*** ”，基于前面的单词，在“ ***之后是*** ”它生成下一个单词“ ***超级英雄*** ”，在“ ***之后是*** ”它生成下一个单词***【a】，*** 最后它生成下一个单词 ***【电影*** 因此完整的句子是“ ***《漫威复仇者联盟》残局是一部超级英雄电影”*** 。我们可以观察它如何根据前面的单词生成单词，因此单词是自回归的，它必须向后看以研究单词的上下文，并从前面的单词中提取信息以生成后面的单词。自回归网络的例子有 GPT-3 和 CTLR。*

***编码器-解码器或 sequence 2 序列变换器网络:**它是编码器和解码器变换器网络的组合。它用于更复杂的自然语言任务，如翻译和文本摘要。*

***编码器-解码器网络翻译图解:**利用编码器网络对句子中的单词进行编码，生成单词的特征向量，理解上下文，从单词中提取有用的信息。来自编码器网络的输出被传递到解码器网络。解码器网络处理编码器生成的输出，并生成目标语言的适当单词，例如，我们将一个英语句子传递给编码器，它从英语上下文中提取有用的信息，并将其传递给解码器，解码器解码编码器输出，并生成法语句子。*

## *标记化概念*

***单词标记化:**就是把一个句子转换成单个单词**。**它通常会生成很大的词汇量，这对于训练 NLP 模型来说并不理想。*

*词汇量:是指一篇课文的字数。*

```
*text = "Python is my favourite programming language"print(text.split())##Output
['Python', 'is', 'my', 'favourite', 'programming', 'language']*
```

*这是一个示例代码，展示了单词标记化是如何完成的，我们将一个句子分割成单独的单词。*

***基于字符的标记化:**是将句子中的单词转换成字符。例如，像“*你好，大家好*”这样的句子将被拆分成如下单个字符:*

*![](img/278493465f3e435d4ca363fdce2b42fb.png)*

*作者图片*

*与单词标记化相比，它生成的词汇量较小，但还不够好，因为将单词拆分成单个字符与单个单词本身的含义不同。*

***子词标记化:**这是大多数自然语言处理任务中使用的最好的标记化形式。单词标记化通过将句子拆分成单个单词来处理标记化，这种方法并不适合所有情况。两个单词" ***鸟*** *"* "和" ***鸟*** *"* 在一个句子中，一个是单数，另一个是复数，单词标记化会将它们视为不同的单词，这就是子单词标记化的由来。子词标记化将合成词和生僻字划分为子词，它考虑了像 *bird* 和 *birds、*这样的词的相似性，而不是将单词 ***birds*** 拆分成两个不同的词，它表示单词***【s】***在单词*的末尾是单词 ***的子词像“ ***有意义的*** ”这样的词会被拆分成“ ***有意义的*** ”和子词“ ***ful*** ”，通常在这种情况下可以在子词中加上一个特殊的字符像“ ***##ful*** ”，来表示它不是一个句子的开头，它是另一个词的子词。子字记号化算法被用在像伯特、GPT 这样的变压器网络中。Bert 使用单词片段标记器作为其子单词标记器。*****

## *用变形金刚训练一个蒙面语言模型*

*我们的主要目标是使用编码器变压器网络来训练屏蔽语言模型，该模型可以为句子中的屏蔽词预测合适的词。在本教程中，我们将使用[拥抱脸变形金刚](https://github.com/huggingface/transformers)，这是一个非常好的库，可以很容易地用变形金刚训练一个模型。*

> *这部分教程需要具备 Python 编程语言和 Pytorch 深度学习库的基础知识。*

***安装 Pytorch***

*[](https://pytorch.org/)  

**安装其他软件包**

```
pip3 install transformerspip3 install datasetspip3 install accelerate
```

**微调电影评论的预训练屏蔽语言模型**

我们将利用一个预训练的 DistilBERT transformer 模型，一个在 IMDb 数据集(一个包含成千上万对不同电影的评论的数据集)上训练的 BERT 的简化版本，来预测句子中的屏蔽词。

## 加载和标记数据集

加载 IMDB 数据

**第 1–10 行:**我们导入了用于加载 IMDb 数据集的模块，并打印出数据集信息以确认其已加载。我们还从数据集中随机打印出两篇评论。如果数据集加载正确，输出应该是:

```
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['text', 'label'],
        num_rows: 50000
    })
})
'>> Review: "Against All Flags" is every bit the classic swashbuckler. It has all the elements the adventure fan could hope for and more for in this one, the damsel in distress is, well, not really in distress. As Spitfire Stevens, Maureen O'Hara is at her athletic best, running her foes through in defiance of the social norms of the period. Anthony Quinn rounds out the top three billed actors as the ruthless Captain Roc Brasiliano and proves to be a wily and capable nemesis for Brian Hawke (Flynn). For the classic adventure fan, "Against All Flags" is a must-see. While it may not be in quite the same league as some of Errol Flynn's earlier work (Captain Blood and The Sea Hawk, for instance), it is still a greatly entertaining romp.''>> Review: Deathtrap gives you a twist at every turn, every single turn, in fact its biggest problem is that there are so many twists that you never really get oriented in the film, and it often doesn't make any sense, although they do usually catch you by surprise. The story is very good, except for the fact that it has so many twists. The screenplay is very good with great dialogue and characters, but you can't catch all the development 
because of the twists. The performances particularly by Caine are amazing. The direction is very good, Sidney Lumet can direct. The visual effects are fair, but than again most are actually in a play and are fake. Twists way to much, but still works and is worth watching.'
```

它打印出 IMDb 数据集的轮廓、训练、测试和非监督部分，以及它们的行数。每行代表数据集中的一篇句子评论。训练和测试部分各有 25000 条评论，而非监督部分有 50000 条评论。最后两篇评论是从 IMDB 数据集中随机打印出来的。

## 标记化数据集

**第 1–11 行:**导入了 ***Autokenizer 包*，**我们从 DistilBERT 模型中加载了 tokenizer，这是一个 ***词块子词块 tokenizer*** 。我们创建了一个函数来表征 IMDb 数据集。

第**13–15**行:最后，我们调用了 tokenizer 函数，并将其应用于加载的数据集。当我们调用 tokenizer 函数时，我们从标记化的数据集中移除了文本和标签，因为不再需要它们了。我们打印了标记化的数据集，它显示了以下输出:

```
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'attention_mask'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['input_ids', 'attention_mask'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['input_ids', 'attention_mask'],
        num_rows: 50000
    })
})
```

在数据集的每一部分我们都有两个特征，*和 ***注意 _ 屏蔽*** 。***input _ id***是为分词生成的 id。***attention _ mask***是 tokenizer 模型生成的值，用来标识有用的单词的*，以及要忽略的单词的输入 id，attention 值以 1 和 0 生成，1 代表有用的单词，0 代表要忽略的单词。**

## **串联和区块数据集**

**在自然语言处理中，我们需要为要训练的文本序列长度设置一个基准，要使用的 DistilBERT 预训练模型的最大长度是 512。**

****第 2–6 行:**我们将块大小设置为 128。因为 GPU 的利用，我们用了一个块大小为 *128* 而不是 *512* 。我们将数据集中的所有文本序列连接成一个单独的连接数据集。**

****第 8–13 行:**我们获得了串联数据集的总长度，创建了一个字典理解来循环遍历串联长度，并根据 128 的块大小将该串联文本划分为块。如果有非常强大的 GPU 可用，则应该使用块大小 512。连接的数据集被分成许多大小相等的块，但最后一个块通常较小，我们将删除最后一个块。**

****第 18–22 行:**带有组块的字典被赋予一个新的列标签，以包含组块样本的输入 id。最后，我们在标记化的数据集上应用了 concat chunk 函数。**

## **用于评估的屏蔽测试数据集**

****第 1–9 行:**我们从 transformers 导入了***DataCollatorForLanguageModeling***，这是一个用于在数据集中创建屏蔽列的默认包。数据集被向下采样到 10000，分割 10%的样本，即 1000 个用于评估的测试数据集。**

****第 13–24 行:**我们定义了一个数据收集器，并定义了一个在数据集中随机插入掩码的函数。对测试数据集应用插入随机掩码函数，用掩码列替换未掩码列。屏蔽的测试数据集将作为训练期间测试模型的基础事实标签。**

## **培训程序**

****第 9–23 行:**我们将批量大小设置为 32，使用 pytorch 内置的数据加载器加载训练和测试数据集。我们加载了预训练的 DistilBERT 模型，并使用了 Adam Optimizer。**

****第 26–28 行:**我们调用 transformers accelerator 库进行训练，它接收预训练的模型、优化器、训练和评估数据集，为训练做准备。**

****第 31–37 行:**我们设置了训练时期数，获得了训练数据加载器的长度，并计算了训练步数。最后，我们设置学习率调度功能，接受优化，热身步骤和训练步骤的训练。**

## **列车代码**

****第 4–7 行:**我们使用 python 内置的 ***tqdm*** 定义了一个进度条用于训练进度监控，然后为输出训练好的模型设置一个目录。**

****第 9–19 行:**定义了一个 for 循环来循环通过多个历元，对于每个历元，我们开始数据集的训练，循环通过训练数据加载器，计算模型的输出，计算输出上的损失，使用变压器 ***加速器*** 包导入来对模型执行反向传播，使用优化器来优化模型以最小化损失。我们应用了学习率调度器，使用优化器将累积梯度设置为零，并更新了进度条。我们这样做，直到我们完成一个时期的整个数据集的训练。**

****第 22–38 行:**我们在测试数据集上评估了一个时期的训练模型，计算了测试数据集上的损失，类似于在训练期间所做的。我们计算了模型的交叉熵损失，然后计算了损失的指数，得到了模型的困惑度。**

> **困惑是一种用于评估语言模型的度量。它是交叉熵损失的指数。**

****第 41–45 行:**我们使用 ***加速器*** 保存预训练的模型，并使用标记器保存关于模型的重要文件，如标记器和词汇信息。训练好的模型和配置文件保存在输出目录文件夹***MLP _ 训练好的模型*** 。输出目录将包含以下文件。我训练了 30 个纪元，得到了一个令人费解的值 ***9.19*** 。输出模型文件夹目录将如下所示:**

```
**--MLP_TrainedModels
    --config.json
    --pytorch_model.bin
    --special_tokens_map.json
    --tokenizer_config.json
    --tokenizer.json
    --vocab.txt**
```

****全训码****

****测试训练好的模型****

**训练好的模型存储在***MLP _ 训练模型、*** 中，我们粘贴目录来设置模型值。我们打印出从模型中生成的句子列表，并为句子中的屏蔽词提供适当的值。**

*   **输出**

```
**>>> this is an excellent movie.
>>> this is an amazing movie.
>>> this is an awesome movie.
>>> this is an entertaining movie.**
```

**我们可以从模型中看到对假面词的预测，分别是****惊艳******牛逼*** 和 ***娱乐*** 。这些预测与完成句子完全吻合。***

**我们已经成功地训练了一个带有编码器转换器网络的屏蔽语言模型，该网络可以找到正确的单词来替换句子中的屏蔽单词。**

> **我已经将我训练的屏蔽语言模型推送到 huggingface hub，它可供测试。**检查拥抱人脸库上的蒙版语言模型****

**[](https://huggingface.co/ayoolaolafenwa/Masked-Language-Model)  

## **测试屏蔽语言模型的 Rest API 代码**

这是直接从拥抱脸测试屏蔽语言模型的推理 API python 代码。

输出

```
washington dc is the capital of usa.
```

它产生正确的输出， ***华盛顿特区是美国的首都。***

## 用转换器加载屏蔽语言模型

使用这段代码，您可以轻松地用转换器加载语言模型。

输出

```
is
```

它打印出预测的掩码字“ ***是*** ”。

## Colab 培训

我创建了一个 google colab 笔记本，上面有创建拥抱脸账户、训练蒙面语言模型以及将模型上传到拥抱脸库的步骤。检查笔记本。

[](https://colab.research.google.com/drive/1BymoZgVU0q02zYv1SdivK-wChG-ooMXL?usp=sharing)  

> 查看 github 资源库以获取本教程

[](https://github.com/ayoolaolafenwa/TrainNLP)  

## 结论

我们在本文中详细讨论了自然语言处理的基础知识、转换器的工作原理、不同类型的转换器网络、使用转换器训练屏蔽语言模型的过程，并且我们成功训练了一个可以预测句子中屏蔽单词的转换器模型。

## **参考文献**

[https://huggingface.co/course/chapter1/2?fw=pt](https://huggingface.co/course/chapter1/2?fw=pt)

[https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)

**通过以下方式联系我:**

电子邮件:[olafenwaayoola@gmail.com](https://mail.google.com/mail/u/0/#inbox)

领英:[https://www.linkedin.com/in/ayoola-olafenwa-003b901a9/](https://www.linkedin.com/in/ayoola-olafenwa-003b901a9/)

推特: [@AyoolaOlafenwa](https://twitter.com/AyoolaOlafenwa)***