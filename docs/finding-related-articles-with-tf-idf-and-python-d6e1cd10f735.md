# 使用 TF-IDF 和 Python 查找相关文章

> 原文：<https://towardsdatascience.com/finding-related-articles-with-tf-idf-and-python-d6e1cd10f735>

## 如何找到与 TF-IDF 相关的文章？用 Python 实现 TF-IDF 算法。

![](img/ff86591dbc1ed62b9a797a32b344e61a.png)

马丁·范·登·霍维尔在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

正如我在关于 [*总结文字*](/summarize-a-text-with-python-continued-bbbbb5d37adb) 的文章中提到的，我收集了过去十年的新闻文章。总结完这些文章后，我面临的下一个挑战是找到相关的文章。如果我有一篇文章，我如何找到与这篇文章相关的文章？立即导致问题*什么是相关文章*？

经过初步研究，选择 TF-IDF 作为算法。这是一个复杂性有限的旧算法，因此可以创建一个实现，并从中学习如何找到类似的文章。通常，有更好的实现可用，例如在 [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) 中，但是我更喜欢为了学习而构建自己的实现。请注意，完全实现需要一些时间。

## TF-IDF 算法

那么，什么是 TF-IDF 呢？我们分解缩写的时候，说的是*词频——逆文档频，*一口*。术语频率*是指特定术语(词)在文本中出现的相对次数；

![](img/a8388f630228e18c9cb3fc5b81a3160e.png)

词频(图片由作者提供)

文档 *d* 中的 *t* 项的 *TF* 等于文档 *d* 中出现的 *t* 项的频率除以文档 *d* 中所有*t’*项的频率之和。示例:

```
'The green man is walking around the green house'
TF(man) = 1 / 9 = 0.11
TF(green) = 2 / 9 = 0.22
```

这是对一个单词在文档中的重要性的度量，根据文档的长度进行校正。

IDF 是对单词在整个语料库(分析的所有文档的集合)中的重要性的度量。这个想法是，如果一个单词出现在很多文档中，它不会增加很多信息。标准功能是

![](img/2ed67f9599fb6792a2b8acfbd9596af5.png)

反向文档频率(作者图片)

IDF 的计算方法是将文档数( *N* )除以一个术语出现的文档总数，并取其对数值。如果一个术语在 10.000 个文档中出现 10 次，则 IDF 等于 3。在同一组文档中出现 100 次的术语的 IDF 值为 2。当一个术语出现在所有文档中时，IDF 值等于 0.0。对数值用于减少 IDF 可能具有的大范围值。

最后，一项的 TF-IDF 值等于 TF 乘以 IDF:

![](img/4ec916a6f2146af62fd3c11f99ff1cd3.png)

TF-IDF 公式(图片由作者提供)

上面的公式是 TF 和 IDF 的标准公式。更多变种可以在 TF-IDF 的[维基百科页面找到。](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

为所有文档的语料库中的每个单词计算 TF-IDF 值，包括术语出现次数为 0 的文档。作为例子，下面两句话

```
0: 'the man walked around the green house',
1: 'the children sat around the fire'
```

产生以下 TF-IDF 值:

![](img/6922a10806b2cb2ba653debbc6f86b2c.png)

计算值(图片由作者提供)

对于所有唯一的单词，为每个句子计算 TF-IDF 值。在以下情况下，该值为 0.0

*   由于 TF 为 0.0，特定单词不会出现在句子中(例如，第二句中的“green”)
*   一个特定的词出现在所有的句子中，例如“大约”,因为 1 的对数值等于 0.0。在这种情况下，所有句子的 TF-IDF 值都是 0.0

## 计算距离

既然计算了 TF-IDF 值，那么可以通过计算这些值之间的[余弦相似度](https://en.wikipedia.org/wiki/Cosine_similarity)来确定两个文档之间的距离。这是通过使用每个单词的 TF-IDF 值(上面数据帧中的一行)作为向量来完成的。

余弦距离是 N 维空间中从原点到两点的直线之间的角度差。在二维空间中形象化是相对简单的。上面的例子是 9 维的，超出了我的想象。在二维空间中:

![](img/76636559a85c0e4234ff352d123bb106.png)

余弦差的定义(图片由作者提供)

余弦差可以使用[欧几里得点积](https://en.wikipedia.org/wiki/Euclidean_vector#Dot_product)公式计算:

![](img/09be6006b8b7569f77e6d1c37911fc1a.png)

计算余弦差(图片由作者提供)

它是两个向量的点积除以向量长度的乘积。当两个向量平行时，余弦差为 1.0，当它们正交时，余弦差为 0.0。

该功能需要应用于语料库中的所有文档组合。从 A 到 B 的差等于从 B 到 A 的差，从 A 到 A 的差是 1.0。下表显示了距离矩阵在对角线上的镜像:

![](img/6f2ca25d6fad832cb4bdda2db67dfddc.png)

差 A-B 等于 B-A(图片由作者提供)

完成所有这些计算后，是时候开始实现了！

## 实施 TF-IDF

干得好，你挺过了理论背景！但是现在的问题是如何实现这个功能。目标是计算几个文档之间的距离。反过来，我们需要 TF-IDF 来计算数据，我们需要 TF 和 IDF 来计算 TF-IDF。

计算 TF 和 IDF 需要对文档进行一些处理。我们需要 TF 的每个文档中每个单词的出现次数和文档中单词的总数，以及 IDF 的每个单词出现在文档中的数量和文档的数量。这些都需要相同的通用文档解析，因此增加了一个预处理步骤:

```
class DocumentDistance:
    distancespre = {}

    def add_documents(self, docs: typing.List[str],
                      ) -> None:
        """
        Calculate the distance between the documents
        :param documents: list of documents
        :return: None
        """
        word_occ_per_doc, doc_lens, doc_per_word = self.pre_proces_data(docs)

        # Calculate TF values
        tfs = []
        for i in range(len(word_occ_per_doc)):
            tfs.append(self.compute_tf(word_occ_per_doc[i], doc_lens[i]))

        # Calculate IDF values
        idfs = self.compute_idf(doc_per_word, len(docs))

        # Calculate TF-IDF values
        tfidfs = []
        for i in range(len(tfs)):
            tfidfs.append(self.compute_tfidf(tfs[i], idfs))

        # Calculate distances
        self.calculate_distances(tfidfs)
```

定义了一个类，它将所有文档之间的距离存储为一个二维数组。这个数组将由`calculate_distances`方法填充。但是首先，我们将创建 pre_process_data 方法，该方法将解析列表中的所有文档，并返回每个文档的单词出现次数、文档长度以及每个单词在文档中出现的次数:

```
 from nltk import word_tokenize

  class DocumentDistance:

      ...

      def pre_proces_data(self,
                        documents: typing.List[str]
                        ) -> (typing.Dict[int, typing.Dict[str, int]], 
                             typing.Dict[int, int], 
                             typing.Dict[str, int]):
        """
        Pre proces the documents
        Translate a dictionary of ID's and sentences to two dictionaries:
        - bag_of_words: dictionary with IDs and a list with the words in the text
        - word_occurences: dictionary with IDs and per document word counts
        :param documents:
        :return: dictionary with word count per document, dictionary with sentence lengths
        """
        # 1\. Tokenize sentences and determine the complete set of unique words
        bag_of_words = []
        unique_words = set()
        for doc in documents:
            doc = self.clean_text(doc)
            words = word_tokenize(doc)
            words = self.clean_word_list(words)
            bag_of_words.append(words)
            unique_words = unique_words.union(set(words))
        # 2\. Determine word occurences in each sentence for all words
        word_occurences = []
        sentence_lengths = []
        for words in bag_of_words:
            now = dict.fromkeys(unique_words, 0)
            for word in words:
                now[word] += 1
            word_occurences.append(now)
            sentence_lengths.append(len(words))

        # 3\. Count documents per word
        doc_count_per_word = dict.fromkeys(word_occurences[0], 0)
        # Travese all documents and words
        # If a word is present in a document, the doc_count_per_word value of
        # the word is increased
        for document in word_occurences():
            for word, val in document.items():
                if val > 0:
                    doc_count_per_word[word] += 1

        return word_occurences, sentence_lengths, doc_count_per_word
```

第一部分通过在 NLTK 包的`word_tokenize`、[部分的帮助下对文档进行标记，将文档分割成单词。在标记之前，会进行一些文档清理，比如将文档转换成小写(稍后讨论)。在`clean_word_list method`中清理词表，目前为空。每个文档的令牌列表存储在`bag_of_words`变量中，这是一个每个文档有一个条目的列表，包含一个令牌列表。在这个循环中，创建了一组所有唯一的单词。这个集合`unique_words`包含了语料库中出现的所有唯一单词。](https://www.nltk.org/api/nltk.tokenize.html)

步骤 2 为所有文档确定所有唯一单词的出现次数。对于每个文档(在`bag of words`上的循环)，为所有唯一的单词(`dict.fromkeys(…)`)创建一个值为 0(零)的字典。然后，代码遍历文档中的所有单词，并将字典值增加 1(一)。该词典被添加到`word_occurences`中，为所有文档创建一个词典列表，其字数和文档长度存储在`sentence_lengths`中。

最后一步，步骤 3，统计每个单词在文档中出现的次数。首先，创建一个列表`doc_count_per word`，其中每个唯一单词出现 0 次。当文档中特定单词的单词计数大于零时，该单词存在于文档中。

预处理的结果是:

```
 Documents:
'the man walked around the green house'
'the children sat around the fire'

word_occurences:
{
 0: {'green': 1, 'fire': 0, 'house': 1, 'sat': 0, 'around': 1, 'man': 1, 
     'the': 2, 'children': 0, 'walked': 1}, 
 1: {'green': 0, 'fire': 1, 'house': 0, 'sat': 1, 'around': 1, 'man': 0, 
     'the': 2, 'children': 1, 'walked': 0}
}

sentence_lengths:
[7, 6]

doc_count_per_word:
{'around': 2, 'green': 1, 'house': 1, 'walked': 1, 'sat': 1, 
 'children': 1, 'man': 1, 'fire': 1, 'the': 2}
```

有了这些数据集，可以相对直接地计算 TF 和 IDF:

```
 def compute_tf(self,
                   wordcount: typing.Dict[str, int],
                   words: typing.List[str]
                   ) -> typing.Dict[str, float]:
        """
        Calculates the Term Frequency (TF)
        :param wordcount: dictionary with mapping from word to count
        :param words: list of words in the sentence
        :return: dictionary mapping word to its frequency
        """
        tf_dict = {}
        sentencelength = len(wordcount)
        for word, count in wordcount.items():
            tf_dict[word] = float(count) / sentencelength
        return tf_dict

    def compute_idf(self,
                    doc_count_per_word: typing.List[typing.Dict[str, int]],
                    no_documents: int
                    ) -> typing.Dict[str, int]:
        """
        Calculates the inverse data frequency (IDF)
        :param doc_count_per_word: dictionary with all documents. A document is a dictionary of TF
        :param no_documents: number of documents
        :return: IDF value for all words
        """
        idf_dict = {}
        for word, val in doc_count_per_word.items():
            idf_dict[word] = math.log(float(no_documents) / val)
        return idf_dict

     def compute_tfidf(self,
                       tfs: typing.Dict[str, float],
                       idfs: typing.Dict[str, float]
                       ) -> typing.Dict[str, float]:
        """
        Calculte the TF-IDF score for all words for a document
        :param tfs: TFS value per word
        :param idfs: Dictionary with the IDF value for all words
        :return: TF-IDF values for all words
        """
        tfidf = {}
        for word, val in tfs.items():
            tfidf[word] = val * idfs[word]
        return tfidf
```

对每个文档调用`compute_tf`方法。每个单词的 TF 值是通过将出现次数除以句子长度来计算的(计算结果被强制转换为浮点类型)。

使用每个单词的文档计数和文档总数来调用`compute_idf`。讨论的公式适用于这些值。

最后，通过将 TF 值乘以相应的 IDF 值来计算每个句子每个单词的 TF-IDF 值。

```
tfs:
[{'sat': 0.00, 'green': 0.14, 'walked': 0.14, 'the': 0.28, 
  'around': 0.14, 'children': 0.00, 'fire': 0.00, 'man': 0.14, 
  'house': 0.14}, 
 {'sat': 0.16, 'green': 0.00, 'walked': 0.00, 'the': 0.33, 
  'around': 0.16, 'children': 0.16, 'fire': 0.16, 'man': 0.00, 
  'house': 0.00}]

idfs:
{'sat': 0.69, 'green': 0.69, 'walked': 0.69, 'the': 0.00, 
 'around': 0.00, 'children': 0.69, 'fire': 0.69, 'man': 0.69, 
 'house': 0.69}

tfidfs:
[{'sat': 0.00, 'green': 0.09, 'walked': 0.09, 'the': 0.00,
  'around': 0.00, 'children': 0.00, 'fire': 0.00, 'man': 0.09,
  'house': 0.09}, 
 {'sat': 0.11, 'green': 0.00, 'walked': 0.00, 'the': 0.00, 
  'around': 0.00, 'children': 0.11, 'fire': 0.11, 'man': 0.00,
  'house': 0.0}
]
```

现在我们有了 TF-IDFS 值，就可以计算不同文档之间的距离了:

```
 def normalize(self,
                  vector: typing.Dict[str, float]
                  ) -> float:
        """
        Normalize the dictionary entries (first level)
        :param tfidfs: dictiory of dictionarys
        :return: dictionary with normalized values
        """
        sumsq = 0
        for i in range(len(vector)):
            sumsq += pow(vector[i], 2)
        return math.sqrt(sumsq)

def calculate_distances(self,
                        tfidfs: typing.List[typing.Dict[str, float]]
                        ) -> None:
    """
    Calculate the distances between all elements in tfidfs
    :param tfidfs: The dictionary of dictionaries
    :return: None
    """
    vectors = []
    # Extract arrays of numbers
    for tfidf in tfidfs:
        vectors.append(list(tfidf.values()))

    self.distances = [[0.0] * len(vectors) for _ in range(len(vectors))]
    for key_1 in range(len(vectors)):
        for key_2 in range(len(vectors)):
            distance = np.dot(vectors[key_1], vectors[key_2]) / \
                       (self.normalize(vectors[key_1])* self.normalize(vectors[key_2]))
            self.distances[key_1][key_2] = distance
```

如本文第一部分所述，两个向量之间的余弦距离是通过将向量的点积除以归一化向量的积来计算的。

为了计算点积，使用了来自 [numpy 库](https://numpy.org/)的`dot`方法。点积通过对成对相乘的值求和来计算:

![](img/e73742d2fd0cc92fd36fb8c395aff1e2.png)

计算矩阵的点积(图片由作者提供)

向量的标准化值等于原点和向量所标识的点之间的直线长度。这等于每个维度的平方和的平方根:

![](img/f7834b862867e9d44ab131937ac2e441.png)

标准化矢量(图片由作者提供)

矢量的归一化是通过`normalize()`方法实现的。所有矢量组合之间的距离用`calculate_distances()`方法计算。所有距离都存储在该类的`distances`属性中。在计算距离之前，该变量被初始化为 0.0 值的 *N* x *N* 矩阵。

一个例子:

```
Sentences:
'the man walked around the green house'
'the children sat around the fire'
'a man set a green house on fire'

Distances:
[1.00, 0.28, 0.11] 
[0.28, 1.00, 0.03] 
[0.11, 0.03, 1.00]
```

请注意，值为 1.0 时距离最小，值为 0.0 时距离最大。由于两个句子没有共同的单词，所以它们的距离是 0.0。一句话和它本身的距离是 1.0。

## 性能改进

有了由三个文档组成的语料库，每个文档由一个小句子组成，距离可以快速计算出来。当文档变成文章并且数量增加到数百时，这个时间增加得很快。下图显示了独特词的数量与新闻文章数量的函数关系。

![](img/4c2492071efad7a1d5f518b3ba26efac.png)

独特单词的数量(图片由作者提供)

当处理 1000 篇文章时，唯一单词的数量增加到几乎 15000 个。这意味着描述文档的每个向量有 15000 个条目。一个点乘需要 2.25 亿次乘法和 2.25 亿次加法。每矢量。归一化一个向量也是 2.25 亿次乘法(计算平方)，2.25 亿次加法和一个平方根。因此，两个向量之间的距离计算是 7.75 亿次乘法，7.75 亿次加法，1 次平方根，1 次除法和 1 次加法。所有这 100 万次来填充整个距离数组。可以想象这需要一些时间…

那么如何才能减少工作量呢？黄金法则是尽可能少花时间。所以让我们看看我们能做些什么来优化。

## 距离计算

在第一部分中已经提到，距离矩阵包含重复值。A 到 B 的距离等于 B 到 A 的距离，我们只需要计算一次，分两个地方加到矩阵里。

其次，矩阵的对角线是 1.0，因为 A 和 A 之间的距离总是 1.0。这里不需要计算。这两步将使计算的距离减半。

第三，对于每次计算，所有向量都被归一化。这是多余的。我们可以预先计算所有的归一化，并将其存储在数组中以备再次使用。这将把距离计算减少 2/3，对于每个矢量组合，只需要计算点。

这些改进将所需时间减少了 85%!

```
def normalize(self,
              tfidfs: typing.List[typing.Dict[str, float]]
              ) -> typing.Dict[int, float]:
    """
    Normalize the dictionary entries (first level)
    :param tfidfs: dictiory of dictionarys
    :return: dictionary with normalized values
    """
    norms = []
    for i in range(len(tfidfs)):
        vals = list(tfidfs[i].values())
        sumsq = 0
        for i in range(len(vals)):
            sumsq += pow(vals[i], 2)
        norms.append(math.sqrt(sumsq))
    return norms

def calculate_distances(self,
                        tfidfs: typing.List[typing.Dict[str, float]]
                        ) -> None:
    """
    Calculate the distances between all elements in tfidfs
    :param tfidfs: The dictionary of dictionaries
    :return: None
    """
    norms = self.normalize(tfidfs)
    vectors = []
    # Extract arrays of numbers
    for tfidf in tfidfs:
        vectors.append(list(tfidf.values()))

    self.distances = [[1.0] * len(vectors) for _ in range(len(vectors))]
    for key_1 in range(len(vectors)):
        for key_2 in range(key_1 + 1, len(vectors)):
            distance = np.dot(vectors[key_1], vectors[key_2]) / (norms[key_1] * norms[key_2])
            self.distances[key_1][key_2] = distance
            self.distances[key_2][key_1] = distance
```

对代码的一些小的改变导致计算时间减少了 85%。但是有更多的可能性

## 减少向量大小

距离计算是代码中最昂贵的部分。上述更改大大减少了计算时间。但是还有一个方面会极大地影响性能，那就是文档向量的大小。对于 1.000 篇文章，每个文档向量是 15.000 个元素。如果我们能够移除这个向量大小，计算时间将会受益。

那么我们能做些什么来减少向量的大小呢？我们如何找到可以去除的没有影响的元素？查看 IDF 函数，它显示停止字对距离计算没有影响。它们出现在每个文档中，因此所有向量中的 IDF 值和 TF-IDF 将为 0.0。将所有这些零相乘仍然需要时间，所以第一步是从唯一单词列表中删除停用词。

NLTK 工具包中提供了每种语言的停用词。工具包提供了停用词列表，可以从唯一的词中过滤掉这些停用词。

第二，只在一个文档中出现的单词对于所有其他文档总是乘以零。我们可以安全地从向量中移除这个单词。

最后，最后一步是使用*词干*。单词列表将包含“house”和“houses”等单词。通过将这些合并到 house，同样的意思，我们可以进一步减少单词列表并提高整个算法的质量。它将不再把“house”和“houses”视为不同的词，而是一个词。NLTK 提供了几个词干分析器，支持多种语言，其中使用了 [*雪球词干分析器*](https://www.nltk.org/api/nltk.stem.snowball.html) 。

词干化也减少了所有动词的基数。像‘like’，‘liking’，‘liked’等词都将被简化为‘like’这个词干。

方法`reduce_word_list`已经被引入(但为空),所以现在我们可以用它来应用这些规则:

```
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("dutch"))
stemmer = SnowballStemmer("dutch")

def clean_word_list(self, words: typing.List[str]) -> typing.List[str]:
    """
    Clean a worlist
    :param words: Original wordlist
    :return: Cleaned wordlist
    """
    words = [x for x in words if x not in self.stop_words and 
                                 len(x) > 2 and not x.isnumeric()]
    words = [self.stemmer.stem(plural) for plural in words]
    return list(set(words))
```

第二个规则是去除只存在于一个文档中的单词，通过在第三个步骤之后增加第四个步骤，该规则被应用于预处理功能的修改版本中:

```
 # 4\. Drop words appearing in one or all document
        words_to_drop = []
        no_docs = int(len(documents) * .90)
        for word, cnt in doc_count_per_word.items():
            if cnt == 1 or cnt > no_docs:
                words_to_drop.append(word)
        for word in words_to_drop:
            doc_count_per_word.pop(word)
            for sent in word_occurences:
                sent.pop(word)
```

第一个循环收集出现一次或总是被收集的单词。在这个循环中不可能修改单词列表，因为它会改变用于迭代的集合。第二步需要从文档计数器和每个文档中删除( *pop* )多余的文字。这是一个相对昂贵的操作，但向量大小的减少是值得的。

对所有这些向量缩减步骤的效果感到好奇吗？让我们看看:

![](img/4ced17c68a8d66ac748cd272967a78db.png)

独特单词的数量(图片由作者提供)

蓝线是唯一单词的原始数量。灰线是减少的字数。因此，在 1.000 篇文章中，字数从 15.000 减少到 5.000。每个向量乘法减少 66%。

代码用 cProfile 进行分析，结果用 [snake_viz](https://jiffyclub.github.io/snakeviz/) 可视化。原始代码的执行时间:

![](img/ab4004ad501d75f810ac93965037a026.png)

剖析原始代码(作者截图)

添加文档需要 31.2 秒，其中 29.6 秒用于计算距离，0.4 秒用于预处理数据。

```
29.6 calculate_distance(...)
 0.6 compute_tf(...)
 0.4 pre_proces_data(...)
 0.4 compute_tfidf(...)
 0.1 compute_idf(...)
```

优化版本的执行时间:

![](img/2b8f79b8e8dcbd5c31c89aa0f7c95fea.png)

剖析优化版本(作者截图)

添加文档所需的时间现在为 5.39 秒(was 31.2)，其中 4.32 秒(was 29.6)用于计算距离，0.9 秒用于处理数据(was 0.4)。

所以预处理需要双倍的时间，增加了 0.5 秒的执行时间，但是距离计算只需要原来时间的 14%。额外的预处理时间很容易收回。

```
 4.2 calculate_distance(...)
 0.9 pre_proces_data(...)
 0.1 compute_tf(...)
 0.0 compute_tfidf(...)
 0.0 compute_idf(...)
```

## 开心吗？

对结果满意吗？老实说，没有。我真的很喜欢实现所有计算文档距离的逻辑。看到可能的性能改进是令人满意的。所以从算法开发的角度来看，我很高兴。

但是应用该算法并没有产生很好的效果。例如，关于寒冷天气的新闻文章似乎与一篇关于伊朗妇女权利的文章更相关，而不是与一篇关于有足够的天然冰可以滑冰的文章更相关。我以为会是另一种情况…

这并不像是浪费了几个小时的快乐编程时间，但是一定会进行一些额外的研究来找到更好的实现。尽管如此，我还是希望您学到了一些关于用 Python 实现算法和寻找提高性能的方法的知识。

## 最后的话

一如既往，完整的代码可以在我的 [Github](https://github.com/lmeulen/LanguageTools) 上找到。在这里你可以找到本文中讨论的实现，以及一个扩展版本，支持多种语言*和*TF 和 IDF 的多种实现，如 TF-IDF 的[维基百科页面上所讨论的。](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

我希望你喜欢这篇文章。要获得更多灵感，请查看我的其他文章:

*   [用 Python 对文本进行总结—续](https://medium.com/towards-data-science/summarize-a-text-with-python-continued-bbbbb5d37adb)
*   [F1 分析和 Python 入门](https://medium.com/p/5112279d743a)
*   [太阳能电池板发电分析](/solar-panel-power-generation-analysis-7011cc078900)
*   [对 CSV 文件中的列执行功能](https://towardsdev.com/perform-a-function-on-columns-in-a-csv-file-a889ef02ca03)
*   [根据活动跟踪器的日志创建热图](/create-a-heatmap-from-the-logs-of-your-activity-tracker-c9fc7ace1657)
*   [使用 Python 的并行 web 请求](/parallel-web-requests-in-python-4d30cc7b8989)

如果你喜欢这个故事，请点击关注按钮！

*免责声明:本文包含的观点和看法仅归作者所有。*