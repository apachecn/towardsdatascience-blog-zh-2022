# 用 Python 总结文本—续

> 原文：<https://towardsdatascience.com/summarize-a-text-with-python-continued-bbbbb5d37adb>

## 如何用 Python 和 NLTK 有效地总结文本，并检测文本的语言

![](img/a959113477706dd9f2465df588fe1499.png)

梅尔·普尔在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

在上个月的文章'[用 Python 总结一个文本中，我展示了如何为一个给定的文本创建一个摘要。从那以后，我一直在频繁地使用这个代码，并发现了这个代码的一些用法上的缺陷。summary 方法被一个执行这个功能的类所取代，例如，使得使用相同的语言和摘要长度变得更加容易。之前的文章非常受欢迎，所以我很乐意与你分享更新！](https://medium.com/p/b3b260c60e72)

所做的改进包括:

*   引入了一个 Summarizer 类，在属性中存储一般数据
*   使用 NLTK 语料库内置的停用词表，保留使用您自己的列表的可能性
*   自动检测文本语言以加载该语言的停用词表
*   用一个字符串或字符串列表调用 summary 函数
*   可选句子长度权重
*   合并了文本文件的摘要方法

结果可以在 [my Github](https://github.com/lmeulen/SummarizeText) 上找到。可以随意使用，也可以根据自己的意愿进行改编。

## Summarizer 类的基础

所以让我们从`Summarizer`的基础开始。该类存储要生成的摘要的语言名称、停用词集和默认长度:

基本思想是使用 NLTK 中的停用词表。NLTK 支持 24 种语言，包括英语、荷兰语、德语、土耳其语和希腊语。可以添加一个准备好的停用词表(`set_stop_words`，第 32 行)或显示一个带有单词的文件(`read_stopwords_from_file`，第 42 行)。通过指定语言的名称，可以使用`set_language`方法从 NLTK 加载单词集。可以通过调用`set_summary_length`来更改摘要的默认长度。

使用类的最简单方法是使用构造函数:

```
summ = Summarizer(language='dutch', summary_length=3)
#or
summ = Summarizer('dutch', 3)
```

语言标识符、停用词列表和摘要长度存储在属性中，将由摘要方法使用。

## 总结一篇文章

该类的核心是`summarize`方法。该方法遵循与上一篇文章中[的`summarize`函数相同的逻辑](https://medium.com/towards-data-science/summarize-a-text-with-python-b3b260c60e72)

```
1\. Count occurrences per word in the text (stop words excluded)
2\. Calculate weight per used word
3\. Calculate sentence weight by summarizing weight per word
4\. Find sentences with the highest weight
5\. Place these sentences in their original order
```

这个算法的工作原理在[之前的文章](https://medium.com/towards-data-science/summarize-a-text-with-python-b3b260c60e72)中有解释。这里只描述不同之处。

以前的实现只接受一个字符串，而新的实现接受单个字符串和字符串列表。

第 19 到 26 行将输入转换成字符串列表。如果输入是单个字符串，则通过对输入进行标记化，将其转换为句子列表。如果输入是一个句子，那么创建一个包含一个句子的数组..在第 31 行，现在可以独立于输入迭代这个列表。

另一个变化是实现了使用加权句子权重的选项(第 50-51 行)。单个单词的重量除以句子的长度。上一篇文章的一些优秀反馈指出了在之前的实现中较短句子被低估的问题。包含重要单词的短句可能比包含更多不重要单词的长句得到的值低。启用或禁用此选项取决于输入的文本。可能需要进行一些实验来确定最适合您使用的设置。

## 汇总文本文件

前面已经提到了对大型文本文件进行汇总的挑战。通过这次重写，增加了这一点。方法`summarize_file`总结了一个文件的内容。这一大段文字总结如下

```
1\. Split the text in chunks of *n sentences*
2\. Summarize each chunk of sentences
3\. Concatenate these summaries
```

首先，文件的内容被读入一个字符串(第 20–22 行)。文本被清除了换行符和多余的空格。然后使用 NLTK 标记器将文本分割成句子(第 26–28 行)，并创建长度为`split_at`的句子块(第 29 行)。

对于每个组块，使用前面的方法来确定摘要。这些单独的摘要连接在一起，形成文件的最终完整摘要(第 32–36 行)。

## 自动检测语言

最后增加的是检测语言的方法。有几个库可以执行这个功能，比如 [Spacy LanguageDetector](https://pypi.org/project/spacy-langdetect/) 、 [Pycld](https://pypi.org/project/pycld2/) 、 [TextBlob](https://textblob.readthedocs.io/en/dev/) 和 [GoogleTrans](https://pypi.org/project/googletrans/) 。但是自己动手做总是更有趣，也更有教育意义。

这里，我们将使用 NLTK 中的停用词表来构建一个语言检测器，从而将它限制在 NLTK 中有停用词表的语言。这个想法是，我们可以计算文本中停用词的出现次数。如果我们对每种语言都这样做，那么计数最高的语言就是编写文本的语言。简单，不是最好的，但足够和有趣:

确定文本中每种语言的停用词出现的数量在第 16 到 22 行执行。`(nltk.corpus.)stopwords.fileids()`返回 NLTK 语料库中所有可用语言的列表。对于这些语言中的每一种，获得停用词，并确定它们在给定文本中出现的频率。结果存储在一个字典中，以语言作为关键字，以出现次数作为值。

通过采用具有最高频率的语言(第 26 行),我们获得了估计的语言。根据该语言初始化该类(第 27 行),并返回语言名称。

## 最后的话

自上一版本以来，代码经历了相同的重大变化，使其更易于使用。语言检测是一个很好的补充，尽管老实说，更好的实现已经可用。添加了一个如何构建此功能的示例。

尽管方法相对简单，但摘要的质量仍然让我吃惊。最大的优势是该算法适用于所有语言，而 NLP 实现通常适用于非常有限的几种语言，尤其是英语。

Github 上有完整的代码，您可以随意使用它，并在它的基础上构建自己的实现。

我希望你喜欢这篇文章。要获得更多灵感，请查看我的其他文章:

*   [用 Python 总结一段文字](https://medium.com/p/b3b260c60e72)
*   [F1 分析和 Python 入门](https://medium.com/p/5112279d743a)
*   [太阳能电池板发电分析](/solar-panel-power-generation-analysis-7011cc078900)
*   [对 CSV 文件中的列执行功能](https://towardsdev.com/perform-a-function-on-columns-in-a-csv-file-a889ef02ca03)
*   [根据你的活动跟踪器的日志创建热图](/create-a-heatmap-from-the-logs-of-your-activity-tracker-c9fc7ace1657)
*   [使用 Python 的并行 web 请求](/parallel-web-requests-in-python-4d30cc7b8989)

如果你喜欢这个故事，请点击关注按钮！

*免责声明:本文包含的观点和看法仅归作者所有。*