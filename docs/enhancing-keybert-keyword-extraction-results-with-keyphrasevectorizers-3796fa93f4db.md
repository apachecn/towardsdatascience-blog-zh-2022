# 用 BERT 变换器和名词短语提取关键短语

> 原文：<https://towardsdatascience.com/enhancing-keybert-keyword-extraction-results-with-keyphrasevectorizers-3796fa93f4db>

## 利用名词短语预处理增强基于 BERT 的关键词抽取

![](img/b3ce34e064c854042ca1e1acf04f1ff6.png)

图片由[阿玛多·洛雷罗](https://unsplash.com/@amadorloureiro)在 [Unsplash](https://unsplash.com) 上拍摄

*这篇文章基于我们的论文* [*“模式排序:利用预训练的语言模型和词性进行无监督的关键短语提取(2022)”*](https://arxiv.org/abs/2210.05245)*。你可以在那里或者在我们的* [*PatternRank 博客文章*](/unsupervised-keyphrase-extraction-with-patternrank-28ec3ca737f0) *中阅读关于我们方法的更多细节。*

要快速浏览文本内容，提取能简明反映其语义上下文的关键词会很有帮助。虽然常用的术语是关键字，但我们通常实际上想要**关键短语**来实现这个目的。

> 关键词或关键短语都应该描述文章的本质。两者的区别在于，关键词是单个单词，而关键短语是由几个单词组成的。例如“小狗”vs“小狗服从训练”。——[**艾里斯·盖伦**](https://yoast.com/difference-between-keyword-and-keyphrase/)

关键短语比简单的关键字提供了更准确的描述，因此通常是首选。幸运的是，许多开源解决方案允许我们从文本中自动提取关键短语。最近非常流行的解决方案之一是 [**KeyBERT**](https://github.com/MaartenGr/KeyBERT) 。这是一个易于使用的 Python 包，通过 [BERT 语言模型](/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)提取关键短语。简单地说，KeyBERT 首先创建文档文本的 BERT 嵌入。然后，创建具有预定义长度的单词 n 元文法的 BERT 关键短语嵌入。最后，计算文档和关键短语嵌入之间的余弦相似度，以提取最佳描述整个文档的关键短语。关于 KeyBERT 更详细的介绍可以在[这里](/how-to-extract-relevant-keywords-with-keybert-6e7b3cf889ae)找到。

# 为什么需要增强 KeyBERT 结果

尽管 KeyBERT 能够自己提取好的关键短语，但实际上仍然存在两个问题。这是由 KeyBERT 在嵌入步骤之前从文档中提取关键短语的方式造成的。用户需要预定义一个*单词 n 元语法范围*来指定提取的关键短语的长度。然后，KeyBERT 从文档中提取定义长度的简单单词 n-grams，并将它们用作嵌入创建和相似性计算的候选关键短语。

> 单词 n 元语法范围让用户决定应该从给定文本中提取的连续单词序列的长度。假设我们定义了一个`word n-gram range = (1,3)` *。然后，我们将选择从文本中提取一元词(只有一个单词)、二元词(两个连续单词的组合)和三元词(三个连续单词的组合)。将单词 n-gram range 应用于`"an apple a day keeps the doctor away"`将导致`["an", "apple", "a","day", "keeps", "the", "doctor", "away", "an apple", "apple a", "a day", "day keeps", "keeps the", "the doctor", "doctor away", "an apple", "apple a day", "a day keeps", "day keeps the", "keeps the doctor", "the doctor away"]`。- [**德维什·帕尔马**](https://www.quora.com/What-is-the-n-gram-range)*

然而，用户通常不知道最佳的 n-gram 范围，因此必须花费一些时间进行试验，直到他们找到合适的 n-gram 范围。此外，这意味着根本不考虑语法句子结构。这导致这样的效果，即使在找到一个好的 n 元语法范围之后，返回的关键短语有时仍然在语法上不太正确或者稍微跑调。继续上面的例子，如果 KeyBERT 从候选关键短语集合中识别出最重要的关键短语*、*或*、*。

# 如何用关键短语向量增强 KeyBERT 结果

为了解决上面提到的问题，我们可以将[**keyphrasvectors**](https://github.com/TimSchopf/KeyphraseVectorizers)包与 KeyBERT 一起使用。KeyphraseVectorizers 包从一组文本文档中提取具有词性模式的关键短语，并将它们转换成文档-关键短语矩阵。文档关键短语矩阵是描述关键短语在文档集合中出现的频率的数学矩阵。

## KeyphraseVectorizer 包是如何工作的？

首先，文档文本用[空间](https://spacy.io/)词性标签标注。其次，从词性标签与预定义的正则表达式模式匹配的文档文本中提取关键短语。默认情况下，矢量器提取包含零个或多个形容词的关键短语，后跟一个或多个使用英语空间词性标签的名词。最后，矢量器计算文档关键短语矩阵。除了矩阵之外，这个软件包还可以为我们提供通过词性提取的关键短语。

**举例:**

我们可以用下面的命令来安装 KeyphraseVectorizers 包:`pip install keyphrase-vectorizers`。

```
{'binary': False, 'dtype': <class 'numpy.int64'>, 'lowercase': True, 'max_df': None, 'min_df': None, 'pos_pattern': '<J.*>*<N.*>+', 'spacy_pipeline': 'en_core_web_sm', 'stop_words': None, 'workers': 1}
```

默认情况下，矢量器针对英语进行初始化。这意味着，指定了一个英语`spacy_pipeline`，没有删除任何`stop_words`，并且`pos_pattern`提取具有零个或多个形容词的关键字，后跟一个或多个使用英语 spaCy 词性标签的名词。

```
 [[0 0 0 0 1 3 2 1 1 0 1 1 3 1 0 0 0 0 1 0 1 1 1 0 1 0 2 0 1 1 1 0 1 1 0 0 0 1 1 3 3 0 1 3 3]
 [1 1 5 1 0 0 0 0 0 1 0 0 0 0 1 1 1 1 0 1 0 0 0 2 0 1 0 1 0 0 0 2 0 0 1 1 1 0 0 0 0 5 0 0 0]]
```

```
['users' 'main topics' 'learning algorithm' 'overlap' 'documents' 'output' 'keywords' 'precise summary' 'new examples' 'training data' 'input' 'document content' 'training examples' 'unseen instances' 'optimal scenario' 'document' 'task' 'supervised learning algorithm' 'example' 'interest' 'function' 'example input' 'various applications' 'unseen situations' 'phrases' 'indication' 'inductive bias' 'supervisory signal' 'document relevance' 'information retrieval' 'set' 'input object' 'groups' 'output value' 'list' 'learning' 'output pairs' 'pair' 'class labels' 'supervised learning' 'machine' 'information retrieval environment' 'algorithm' 'vector' 'way']
```

矢量器的输出显示，与简单的 n 元语法不同，提取的单词语法正确，有意义。这是矢量器提取名词短语和扩展名词短语的结果。

> 名词短语是围绕一个名词构建的简单短语。它包含一个限定词和一个名词。例如:一棵树，一些糖果，城堡。一个**扩展名词短语**通过添加一个或多个**形容词来为名词添加更多细节。形容词是描述名词的词。例如:一棵*巨大的*树，一些*五彩缤纷的*糖果，那些*巨大的、皇家的*城堡。英国广播公司**

## KeyBERT 如何使用 KeyphraseVectorizers？

关键短语矢量器可以与 KeyBERT 一起使用，以提取与文档最相似的语法正确的关键短语。因此，矢量器首先从文本文档中提取候选关键短语，随后由 KeyBERT 基于它们的文档相似性对其进行排序。然后，前 n 个最相似的关键短语可以被认为是文档关键词。

> 除了 KeyBERT 之外，使用 KeyphraseVectorizers 的优点是，它允许用户获得语法正确的关键短语，而不是简单的预定义长度的 n 元语法。

关键短语分类器首先提取由零个或多个形容词组成的候选关键短语，然后在预处理步骤中提取一个或多个名词，而不是简单的 n 元语法。 [TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) 、 [SingleRank](https://aclanthology.org/C08-1122.pdf) 和[embe beed](https://aclanthology.org/K18-1022.pdf)已经成功使用这种名词短语方法进行关键短语提取。提取的候选关键短语随后被传递给 KeyBERT 用于嵌入生成和相似性计算。为了使用这两个包来提取关键短语，我们需要传递给 KeyBERT 一个带有`vectorizer`参数的关键短语矢量器。由于关键短语的长度现在取决于词性标签，因此不再需要定义 n 元语法长度。

## 示例:

KeyBERT 可以通过`pip install keybert`安装。

不是决定合适的 n 元语法范围，例如(1，2)…

```
[[('labeled training', 0.6013),
  ('examples supervised', 0.6112),
  ('signal supervised', 0.6152),
  ('supervised', 0.6676),
  ('supervised learning', 0.6779)],
 [('keywords assigned', 0.6354),
  ('keywords used', 0.6373),
  ('list keywords', 0.6375),
  ('keywords quickly', 0.6376),
  ('keywords defined', 0.6997)]]
```

我们现在可以让关键短语向量器决定合适的关键短语，而没有最大或最小 n 元语法范围的限制。我们只需将一个关键短语矢量器作为参数传递给 KeyBERT:

```
[[('learning', 0.4813), 
  ('training data', 0.5271), 
  ('learning algorithm', 0.5632), 
  ('supervised learning', 0.6779), 
  ('supervised learning algorithm', 0.6992)], 
 [('document content', 0.3988), 
  ('information retrieval environment', 0.5166), 
  ('information retrieval', 0.5792), 
  ('keywords', 0.6046), 
  ('document relevance', 0.633)]]
```

这使我们能够确保我们不会因为将 n 元语法范围定义得太短而删除重要的单词。例如，我们可能找不到带有`keyphrase_ngram_range=(1,2)`的关键词*“监督学习算法”*。此外，我们避免获得稍微跑调的关键短语，如*、*、*、【信号监控】、*或*、【快速关键词】、*。

**提取英语以外的语言中的关键短语:**

此外，我们还可以将这种方法应用于其他语言，如德语。这只需要对[键相器](https://github.com/TimSchopf/KeyphraseVectorizers)和[键齿](https://github.com/MaartenGr/KeyBERT)的一些参数进行修改。

对于关键相分离器，`spacy_pipeline`和`stop_words`参数需要修改为`spacy_pipeline=’de_core_new_sm’`和`stop_words=’german’`。因为德语 spaCy 词性标签与英语不同，`pos_pattern`参数也需要修改。regex 模式`<ADJ.*>*<N.*>+`提取包含零个或多个形容词的关键字，后跟一个或多个使用德语 spaCy 词性标签的名词。

对于 KeyBERT，需要通过`pip install flair`安装 Flair 包，并且必须选择德国 BERT 型号。

```
[[('schwester cornelia', 0.2491),
  ('neigung', 0.2996),
  ('angesehenen bürgerlichen familie', 0.3131),
  ('ausbildung', 0.3651),
  ('straßburg', 0.4022)],
 [('tochter', 0.0821),
  ('friedrich schiller', 0.0912),
  ('ehefrau elisabetha dorothea schiller', 0.0919),
  ('neckar johann kaspar schiller', 0.092),
  ('wundarztes', 0.1334)]]
```

# 摘要

[keyphrasvectors](https://github.com/TimSchopf/KeyphraseVectorizers)是最近发布的一个包，除了 [KeyBERT](https://github.com/MaartenGr/KeyBERT) 之外，它还可以用来从文本文档中提取增强的关键短语。这种方法消除了对用户定义的单词 n 元语法范围的需要，并提取语法正确的关键短语。此外，该方法可以应用于许多不同的语言。这两个开源包都易于使用，只需几行代码就可以精确提取关键短语。

也非常感谢 [Maarten Grootendorst](https://www.maartengrootendorst.com) ，他在我编写[keyphrasevectors](https://github.com/TimSchopf/KeyphraseVectorizers)包时给了我输入和灵感。

# 来源

[](https://github.com/TimSchopf/KeyphraseVectorizers) [## GitHub-TimSchopf/keyphrasevectors:一组向量器，用…

### 一组向量器，从一组文本文档中提取带有词性模式的关键短语，并转换…

github.com](https://github.com/TimSchopf/KeyphraseVectorizers) [](https://github.com/MaartenGr/KeyBERT) [## GitHub - MaartenGr/KeyBERT:用 BERT 提取最少的关键字

### KeyBERT 是一种简单易用的关键字提取技术，它利用 BERT 嵌入来创建关键字和…

github.com](https://github.com/MaartenGr/KeyBERT) [](https://arxiv.org/abs/2210.05245) [## PatternRank:利用预训练的语言模型和无监督关键短语的词性…

### 关键短语提取是从给定文本中自动选择一小组最相关短语的过程…

arxiv.org](https://arxiv.org/abs/2210.05245) 

米哈尔恰和塔劳(2004 年)。 [TextRank:将 or-
der 带入文本。](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)2004 年自然语言处理经验方法会议论文集-
ing，第 404–411 页，西班牙巴塞罗那。计算语言学协会。

万，谢，肖(2008)。 [CollabRank:实现一种
协作方法来提取单个文档的关键短语
。](https://aclanthology.org/C08-1122.pdf)《第 22 届国际计算语言学会议(T4)论文集》(2008 年出版)，第 969-976 页，英国曼彻斯特。科林 2008 组委会。

本纳尼-斯米尔，k .，穆萨特，c .，霍斯曼，a .，贝里斯维尔，
M .，贾吉，M. (2018)。[使用句子嵌入的简单无监督
关键短语提取。](https://aclanthology.org/K18-1022.pdf)第 22 届计算机自然语言学习会议论文集，第 221-229 页，
比利时布鲁塞尔。计算语言学协会。