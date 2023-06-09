# 让我们从文本数据中提取一些主题——第三部分:非负矩阵分解(NMF)

> 原文：<https://towardsdatascience.com/let-us-extract-some-topics-from-text-data-part-iii-non-negative-matrix-factorization-nmf-8eba8c8edada>

## 了解更多有关线性代数派生的无监督算法的信息，该算法使用直观的方法进行主题建模

![](img/20a8849315a951183cee13e3afa88c79.png)

[pexel](https://www.pexels.com/ko-kr/photo/3184642/)免费使用照片

# 介绍

**主题建模**是一种自然语言处理(NLP)任务，它利用无监督的学习方法来提取我们处理的一些文本数据的主要主题。这里的“无监督”一词意味着没有具有相关主题标签的训练数据。相反，算法试图直接从数据本身中发现底层模式，在这种情况下是主题。

有各种各样的算法被广泛用于主题建模。在我之前的两篇文章中，我向您介绍了用于主题建模的两种算法，LDA 和 GSM。

</let-us-extract-some-topics-from-text-data-part-i-latent-dirichlet-allocation-lda-e335ee3e5fa4>  <https://medium.com/geekculture/let-us-extract-some-topics-from-text-data-part-ii-gibbs-sampling-dirichlet-multinomial-mixture-9e82d51b0fab>  

在本文中，我们将研究非负矩阵分解(NMF)，一种源于线性代数的无监督算法。Python 中的代码实现也将被引入。

# 什么是 NMF？

关于 NMF 的一个有趣的事情是，它被用于许多其他应用，包括推荐系统，不像其他专门用于主题建模的主题建模算法。NMF 是在线性代数领域发展起来的，能够识别数据中的潜在或隐藏结构。简而言之，NMF 将高维向量分解(或因式分解，因此得名因式分解)成低维表示。这些低维向量的所有系数都是非负的，这可以从算法的名称“非负”矩阵分解中得到暗示。

假设因式分解的原始矩阵是 A，NMF 将把 A 分解成两个矩阵 W 和 H。W 被称为包含 NMF 发现的主题的特征矩阵，H 被称为包含这些主题的系数(权重)的分量矩阵。如果矩阵 A 的形状为 M x N，主题数为 k，则特征矩阵和成分矩阵的形状分别为(M x k)和(k x N)。

# 一些数学

由于 NMF 是一个无人监督的算法，它需要一些措施，以便能够找出不同文档之间的相似性。有几种数学度量用来定义这种相似性或距离感，但我在本节中介绍了两种最流行的度量。

第一个是广义的 kull back-lei bler 散度，定义如下。

![](img/8d42e4b4e50634a2729135a20143ce0b.png)

来源:来自作者使用乳胶

kull back-lei bler 散度可以实现如下。

```
from numpy import sum

def kullback_leibler_divergence(p, q):
  return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

# There is built-in function in the SciPy package that 
# you can directly use as well

from scipy.special import kl_div
```

你可以在这里阅读更多关于这种分歧的信息。

另一个流行的度量是 Frobenius 范数，它是其元素的绝对平方和的平方根。它也被称为欧几里德范数。数学公式如下。

![](img/a7a80ac09210c7da25a4d495c36293ae.png)

来源:来自作者使用乳胶

您可以使用 NumPy 包来实现这一点。

```
import numpy as np

# Assuming some array a has been already defined
frobenius_norm = numpy.linalg.norm(a)
```

对于优化，通常使用这两种算法中的任何一种。它们是坐标下降解算器和乘法更新解算器。哪种优化算法更好的问题取决于您正在处理的数据类型及其特征(例如稀疏度)，因此尝试这两种算法以查看哪种返回的主题更连贯且质量更高可能是最佳方法。在评估主题质量方面，更多的内容将在本文的后面部分进行解释。参考这篇[文章](https://www.analyticsvidhya.com/blog/2021/06/part-15-step-by-step-guide-to-master-nlp-topic-modelling-using-nmf/)对这两种优化算法有更深入的了解。

# 履行

现在，让我们深入研究 NMF 的代码实现。和往常一样，我们从安装我们需要的包开始，并将它们导入到我们的环境中。

```
!pip install gensim

import nltk
from nltk.stem import *

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# Prepare list of stopwords for removal
stopwords = set(nltk.corpus.stopwords.words('english'))

import pandas as pd

# NMF implementation in sklearn
from sklearn.decomposition import NMF 

# TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# Dataset we use for this tutorial
from sklearn.datasets import fetch_20newsgroups 

from scipy import linalg # For linear algebra operations
import matplotlib.pyplot as plt
%matplotlib inline
np.set_printoptions(suppress=True)
```

然后，我们读入本教程的数据集。我们将使用 20 个新闻组数据，每个人都可以通过 sklearn 包获得这些数据。它使用[Apache 2.0 版许可](https://www.apache.org/licenses/LICENSE-2.0)。只是提醒一下，为了不同教程之间的一致性，我在关于主题建模的前两篇文章中使用了这个数据集。

```
# Categories
cats = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space'] 
remove = ('headers', 'footers', 'quotes')

# Read in 20 NewsGroup Data
fetch20newsgroups = \
fetch_20newsgroups(subset='train', categories=cats, remove=remove)

# Transform fetch20newsgroups object to pandas dataframe
df = pd.DataFrame(fetch20newsgroups.data, columns=['text'])
```

我们通过一些传统的文本预处理步骤来确保建模主题的质量不会变差。

```
# Remove Punctuations 

import string
PUNCT_TO_REMOVE = string.punctuation

df.text = \
df.text.apply(lambda x: x.translate(str.maketrans('', '', PUNCT_TO_REMOVE)))

# Remove URLs
import re
def remove_url(text):
  return re.sub(r'https?:\S*','',text)

df.text = df.text.apply(remove_url)

# Remove mentions and hashtags
import re
def remove_mentions_and_tags(text):
    text = re.sub(r'@\S*','',text)
    return re.sub(r'#\S*','',text)

df.text = df.text.apply(remove_mentions_and_tags)

# Make all the text lower case
df.text = df.text.str.lower()

# Remove words that exist across almost all documents (almost like removing
# stopwords) after tokenization
extraneous_words = \
['article', 'organization', 'subject','line','author','from', 
'lines', 'people', 're', 'writes', 'distribution', 
'x', 'nntppostinghost','think','university']

docs = []

for news in df.text:

   words = \
[w for w in nltk.tokenize.word_tokenize(news) \
if (w not in stopwords) & (w not in extraneous_words) & \
(nltk.pos_tag([w])[0][1] in ['NN','NNS'])]

    docs.append(words)
```

我们应用 [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) 矢量化。

```
# Initialize TFIDF Vectorizer
vect= TfidfVectorizer(
    min_df=10,
    max_df=0.85,
    max_features=5000,
    ngram_range=(1, 3), # Include unigrams, bi-grams and tri-grams
    stop_words='english' # Remove Stopwords
)

# Apply Transformation
X = vect.fit_transform(df.text)

# Create an NMF instance with 4 components
model = NMF(n_components=4, init='nndsvd', random_state=42)

# Fit the model
model.fit(X)
```

我们分别存储特征和组件矩阵。

```
# Features Matrix
nmf_features = model.transform(X)

# Components Matrix
components_df = pd.DataFrame(model.components_, 
                            columns=vect.get_feature_names_out())

terms =  list(vect.get_feature_names_out())
```

## 获取每个主题的最高系数值的单词(从组件矩阵中)

从组件矩阵中，我们可以通过检索与最高系数值相关联的标记来查看每个主题中最有意义的单词。

```
# Top 20 words of importance in each of the topics

for topic in range(components_df.shape[0]):
    topic_df = components_df.iloc[topic]
    print(f'For topic {topic+1} the words with the highest value are:')
    print(topic_df.nlargest(20))
    print('\n')
```

让我们看看第一个主题的输出作为例子。

![](img/2df6cafa3a2fd8b9e7ea3bb4ed1737a5.png)

来源:来自作者

与第一个主题相关的重要词汇是上帝、耶稣、圣经和信仰，这些词汇我们会认为会出现在与无神论(这是第一个主题的实际主题)相关的文献中。

## 主题描述符

基于以上信息，我们可以打印出每个主题的前 n 个关键词。

```
def get_descriptor( terms, H, topic_index, top ):
  # reverse sort the values to sort the indices
  top_indices = np.argsort( H.iloc[topic_index,:] )[::-1]

  # get the terms corresponding to the top-ranked indices
  top_terms = [ ]
  for term_index in top_indices[0:top]:
      top_terms.append( terms[term_index] )
  return top_terms

k = 4 # four topics
descriptors = [ ]
for topic_index in range(k):
  descriptors.append(get_descriptor(terms, components_df, topic_index, 10 ))
  str_descriptor = ", ".join( descriptors[topic_index] )
  print("Topic %02d: %s" % ( topic_index+1, str_descriptor ) )
```

然后我们得到:

```
Topic 01: god, people, dont, think, just, jesus, say, bible, believe, does 
Topic 02: thanks, files, graphics, image, file, know, program, format, help, need 
Topic 03: space, nasa, launch, shuttle, lunar, orbit, moon, station, data, earth 
Topic 04: sank manhattan sea, stay blew, away sank, away sank manhattan, sank manhattan, said queens stay, said queens, bob beauchaine, bronx away sank, bronx away
```

## 第 j 个文档的主题

我们还可以从特征矩阵中找出第 j 个文档属于哪个主题。

```
# Topic of the jth document

j = 100 

print(pd.DataFrame(nmf_features).loc[100])

# Add 1 to index to get the topic number since index starts from 0 not 1
print("\n {}th Document belongs to Topic {} ".\
format(j, np.argmax(pd.DataFrame(nmf_features).loc[100])+1))

>>
0    0.113558
1    0.016734
2    0.015008
3    0.000000
Name: 100, dtype: float64

100th Document belongs to Topic 1 
```

## 预测新文档的主题

对于一个以前没有被算法发现的新文档，我们也可以预测这个文档属于哪个主题。

```
# New Document to predict
new_doc = """Jesus is the light of thie world"""

# Transform the TF-IDF
X_new = vect.transform([new_doc])

# Transform the TF-IDF: nmf_features
nmf_features_new = model.transform(X_new)

# The idxmax function returns the index of the maximum element
# in the specified axis
print("This new document belongs to Topic {}".\
format(pd.DataFrame(nmf_features_new).idxmax(axis=1).iloc[0] + 1))

>>
This new document belongs to Topic 1
```

## 最佳主题数量的评估

在上面的代码中，我们选择了四个主题进行建模，因为我们已经预先知道我们正在处理的数据中有多少主题。然而，在大多数情况下，我们没有这样的知识。随机猜测不是最理想的方法。有没有一种编程的方式来做这件事？是啊！

为了自动确定主题的最佳数量，我们使用与前面介绍的两个算法 LDA 和 GSDMM 相同的方法。我们使用 gensim 的一致性分数来做到这一点。不幸的是，我们无法将 sklearn 的 NMF 实现与 gensim 的 coherence score 模型结合使用，因此，我们需要依赖 gensim 的 NMF 实现来实现这一目的。本文[的后半部分](/topic-modeling-articles-with-nmf-8c6b2a227a45)使用 NMF 的 gensim 实现来确定一致性分数的最佳数量，然后使用 NMF 的 sklearn 实现来进行实际的训练和主题提取。

## 复习题目质量

由于 NMF 是一个线性代数算法，我们有一个很好的方法来评估每个主题的主题质量，通过查看每个主题组中文档的平均残差。

假设我们使用 Frobenius 范数作为最小化的距离度量，我们可以如下计算每个文档的残差。

```
A = vect.transform(df.text) # Original Matrix (after tf-idf vectorization is applied)
W = model.components_ # Component matrix
H = model.transform(A) # Features matrix

# Get the residuals for each document
r = np.zeros(A.shape[0])
for row in range(A.shape[0]):
    # 'fro' here means we are using the Frobenium norm as the distance metric
    r[row] = np.linalg.norm(A[row, :] - H[row, :].dot(W), 'fro') 
```

然后，我们计算每个主题组中这些残差的平均值，看看哪个主题具有最低的平均残差将是质量最好的主题！看看这篇[文章](/topic-modeling-articles-with-nmf-8c6b2a227a45)，了解更多关于这个过程的信息。

# 奖金

正如我前面提到的，gensim 也有自己的 NMF 实现。虽然基础数学和原理与 sklearn 实现相同，但它可以与 gensim 中的 coherence 模型联合使用，以确定要建模的主题的最佳数量，而 sklearn 目前没有提供任何 coherence score 模型。要了解摩尔关于 gensim 的 NMF 实现，请参考此[文档](https://radimrehurek.com/gensim/models/nmf.html)。

迷你批处理 NMF 是一个 NMF 版本，是专为更大的数据集。它有一个 partial_fit 方法，在数据集的不同块上连续调用多次，以便实现小批量的核外或在线学习。要了解有关该算法的更多信息，请参考其[文档](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchNMF.html#sklearn.decomposition.MiniBatchNMF)。

# 结论

在本文中，我向您介绍了 NMF 算法，它是线性代数算法家族的一部分，不仅用于主题建模，还用于从推荐系统到维度缩减的广泛应用。只要用户对线性代数有所了解，NMF 的优势就在于其直观和简单的数学。很难判断哪种主题建模算法最适合您正在处理的特定项目和数据集，因此我建议使用几种最流行的主题建模算法，并在进行任何后续步骤之前目测每个主题中的关键词。

如果你觉得这篇文章有帮助，请考虑通过以下链接注册 medium 来支持我: )

joshnjuny.medium.com

你不仅可以看到我，还可以看到其他作者写的这么多有用和有趣的文章和帖子！

# 关于作者

*数据科学家。加州大学欧文分校信息学专业一年级博士生。主要研究兴趣是将 SOTA ML/DL/NLP 方法应用于健康和医疗相关的大数据，以提取有趣的见解，为患者、医生和决策者提供信息。*

*密歇根大学刑事司法行政记录系统(CJARS)经济学实验室的前研究领域专家，致力于统计报告生成、自动化数据质量审查、构建数据管道和数据标准化&协调。Spotify 前数据科学实习生。Inc .(纽约市)。*

他喜欢运动、健身、烹饪美味的亚洲食物、看 kdramas 和制作/表演音乐，最重要的是崇拜我们的主耶稣基督。结账他的 [*网站*](http://seungjun-data-science.github.io) *！*