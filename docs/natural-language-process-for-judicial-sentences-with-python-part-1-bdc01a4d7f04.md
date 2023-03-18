# 用 Python 实现司法判决的自然语言处理

> 原文：<https://towardsdatascience.com/natural-language-process-for-judicial-sentences-with-python-part-1-bdc01a4d7f04>

![](img/6dc8df6d7a0e828bd94a1ee879ce02fb.png)

[https://pixabay.com/](https://pixabay.com/)

## 第 1 部分:数据预处理

自然语言处理(NPL)是人工智能的一个领域，其目的是找到计算方法来解释人类语言的口语或书面语。NLP 的思想超越了可以由 ML 算法或深度学习 NNs 进行的简单分类任务。事实上，NLP 是关于*解释*:你想训练你的模型不仅仅是检测频繁出现的单词，统计它们或者消除一些嘈杂的标点符号；你希望它告诉你谈话的情绪是积极的还是消极的，电子邮件的内容是纯粹的宣传还是重要的事情，过去几年关于惊悚小说的评论是好是坏。

在这一系列文章中，我决定对与美国司法判决相关的新闻稿进行分析。这是一个历史数据集，包含了来自 https://www.justice.gov/news 司法部(DOJ)网站的 13087 篇新闻稿，我在 Kaggle 上找到了 json 格式的([https://www . ka ggle . com/jben Cina/Department-of-Justice-2009 2018-press-releases](https://www.kaggle.com/jbencina/department-of-justice-20092018-press-releases))。在这些行中，有 4688 行标有新闻稿的类别。

本研究的目标有两个主要研究问题:

*   如果您需要检索详细信息，拥有一个标记良好的知识库是至关重要的。新闻档案每天都在更新，手动标注每篇文章可能会非常耗时。我的问题是:有没有可能实现一个预测算法，自动将新文章分类到现有类别中？
*   能否推断出文章的情绪，并将其与所属的类别联系起来？为此，我将对我的文章进行无监督的情感分析，对于每个类别，显示有多少文章被归类为正面或负面。

为了回答这些问题，我将我的研究分为 9 个部分:

*   第 1 部分:数据预处理
*   第 2 部分:描述性统计
*   第 3 部分:TF-IDF 分析
*   第四部分:矩阵分解的潜在主题
*   第 5 部分:用 LDA 进行主题建模
*   第 6 部分:文档嵌入
*   第 7 部分:聚类分析
*   第 8 部分:命名实体和网络表示
*   第 9 部分:无监督情感分析
*   第 10 部分:用逻辑回归、SVC 和 Keras 神经网络进行预测分析

在本系列的第一部分中，我将介绍数据预处理活动，其中我们还将执行标记化、词干化和词汇化。

*   记号化→将给定文本分解成称为记号的单元的过程。在这个项目中，令牌将是单个单词，标点符号将被丢弃
*   词干提取→提取单词的词根或词干的过程。
*   词汇化→根据上下文将单词转换成有意义的基本形式或词根形式的过程。

**注意**:词汇化在自然语言处理和 NLU(自然语言理解)中非常重要，因为它比词干提取更准确。现实世界应用的一些例子是聊天机器人和情感分析。这种技术的缺点是词汇化算法比词干化算法慢得多。

为什么我们的数据框中需要这些进一步的元素？因为这些将是我们训练高效 NLP 模型的输入。例如，想象一个情感分析模型:你如何告诉模型“微笑”和“微笑”指的是同一个概念？

因此，让我们从导入必要的库开始(这些库对下一章也很有用):

```
#text preprocessing
import json
import nltk
import en_core_web_sm
import spacy
nlp = spacy.load("en_core_web_sm")
from nltk import SnowballStemmer
stemmer = SnowballStemmer("english")
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.corpus import stopwords
from gensim.models.doc2vec import TaggedDocument
from gensim.models import LdaMulticore, TfidfModel, Doc2Vec, CoherenceModel
from gensim.corpora import Dictionary#machine learning, models and statistics
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy.sparse import random#utilities
import numpy as np
import pandas as pd
import pickle
import unidecode
import datetime
import os
import warnings
from math import sqrt
from collections import defaultdict, Counter
import json
import re
import sys
import random
import time #to know how long training took
import multiprocessing 
from IPython.display import Markdown, display#keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import InputLayer, Input
from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten, Activation, Dropout
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Embedding
from keras.layers import Bidirectional, LSTM
from keras.utils import to_categorical#visualizationfrom matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
```

对于文本 NLP 包，我将使用 Python 库 [Spacy](https://spacy.io/) 和 [NLTK](https://www.nltk.org/) 。

现在，我将下载 json 数据并将其转换成 pandas 数据帧，并执行上述预处理任务。

```
#reading json files=[]
with open('combined.json') as f:
    for i in f.readlines():
        s.append(i)

res = [json.loads(s[i]) for i in range(len(s))]
contents = []
titles = []
date = []
topic = []
components = []
for i in range(len(res)):
    titles.append(res[i]['title'])
    contents.append(res[i]['contents'])
    date.append(res[i]['date'])
    topic.append(res[i]['topics'])
    components.append(res[i]['components'])#storing result into a pandas dfimport pandas as pd
df = pd.DataFrame(list(zip(titles, date, contents, topic, components)), 
               columns =['titles', 'date', 'text', 'topic', 'component'])
```

我注意到文本中充满了\xa0 符号，这是表示空格的 Unicode。更具体地说，这个符号代表“不间断空格”。幸运的是，Python 中有一个很好的包，可以通过 pip 安装，就是“unidecode”。一旦导入，函数`unidecode()`将获取 Unicode 数据，并尝试用 ASCII 字符表示它。

```
import unidecode
for i in range(len(df)):
    df['text'][i] = unidecode.unidecode(df['text'][i])#now let's convert the timestamp format of the "date" column to pandas datetime64.
import datetimefor i in range(len(df)):
    res = datetime.datetime.strptime(df['date'][i][0:10],"%Y-%m-%d")
    df['date'][i] = pd.to_datetime(res)
```

我还想知道有多少记录有标签，因为它将是我们分类预测分析的目标变量。

```
#counting how many articles are missing the category
counter=0
for i in range(len(df)):
    if len(df['category'][i])==0:
        counter+=1

print('Number of non-labeled articles: {}'.format(counter))
print('Number of labeled articles: {}'.format(len(df)-counter))
if len(df)-counter>=2000 and len(df)>=5000:
    print('Dataset convalidated:-)')Number of non-labeled articles: 8399
Number of labeled articles: 4688
Dataset convalidated:-)
```

所以我们有超过 2000 个文档被贴上标签，这正是我们要找的。现在，我们不需要它们，因为预测分析将是最后一部分。

现在是时候对我们的文本进行预处理了。我将使用上面提到的记号、词条和词干创建另外三列。当我们需要对单词和文档进行矢量化时，这些将非常有用。

```
df["Tokens"] = [" ".join([token.text.replace("'ll", "will").replace("'ve", "have").replace("'s", "is").replace("'m", "am") for sentence in nlp(speech).sents for token in sentence]) 
         for speech in df.text]df["Lemmas"] = [" ".join([token.lemma_ if token.lemma_ != "-PRON-" else token.text.lower() 
            for sentence in nlp(speech).sents for token in sentence if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "X"} 
                               and token.is_stop == False]) for speech in df.text]punctuation = set("""!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~""")
stemmer = SnowballStemmer("english")

df["Stems"] = [" ".join([stemmer.stem(token.text) for tokenized_sentence in nlp(tokenized_speech).sents 
                         for token in tokenized_sentence if token.text not in punctuation and token.is_stop == False])
                   for tokenized_speech in df.text]#saving the df into pickle format
df.to_pickle('data/df.pkl')
```

在预处理的最后，熊猫数据帧的最终外观将如下所示:

```
#reading pickle dataset
df = pd.read_pickle('data/df.pkl')#to avoid terminology confusion, I'm renaming the topic column to "category": indeed, I'll refer to topic while performing 
#topic modeling like LDA, hence topic!=category.df=df.rename(columns={'topic':'category'})
df.head()
```

![](img/572600b5495888174a56bdf8ee961246.png)

在下一篇文章中，我们将更深入地研究这个数据集的描述性统计，并开始检索相关信息，请继续关注下一章！

## 参考

*   自然语言工具包
*   [Python 中的 spaCy 工业级自然语言处理](https://spacy.io/)
*   [司法新闻| DOJ |司法部](https://www.justice.gov/news)
*   [司法部 2009-2018 年新闻发布| Kaggle](https://www.kaggle.com/datasets/jbencina/department-of-justice-20092018-press-releases)
*   【https://creativecommons.org/publicdomain/zero/1.0/ 号