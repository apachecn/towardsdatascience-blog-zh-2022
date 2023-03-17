# 单词袋和 TF-IDF 的快速介绍

> 原文：<https://towardsdatascience.com/a-quick-introduction-to-bag-of-words-and-tf-idf-fbd3ab84ecbf>

# 单词袋和 TF-IDF 的快速介绍

## 机器学习和自然语言处理模型如何处理文本

![](img/85a534c358493213966672643b91bb11.png)

克里斯蒂安·卢在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

# 什么是一袋单词

你有没有想过，当机器学习(ML)是基于数学和统计的时候，ML 是如何处理文本的？我的意思是，文本毕竟不是一串数字…对不对？

我给你介绍一下**字袋(BoW)** 模型。除了它听起来有趣的名字，BoW 是自然语言处理(NLP)的重要组成部分，也是对文本进行机器学习的基础之一。

BoW 仅仅是一个无序的单词及其频率(计数)的集合。例如，让我们看看下面这段文字:

```
"I sat on a plane and sat on a chair."

and  chair  on  plane  sat
  1      1   2      1    2
```

**注意**:令牌(单词)的长度必须为`2`或更多字符。

就这么简单。让我们看看这是如何计算的，如果再多加几个句子，或者我们通常所说的**文档**会是什么样子。首先，我们将导入必要的库。

```
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
```

我们将使用来自 [Scikit-Learn](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) 的文本矢量器。我们已经导入了其中的两个，一个是创建弓的`CountVectorizer`，另一个是`TfidfVectorizer`，我们稍后会谈到。让我们以字符串列表*的形式处理几个文档。*

```
corpus = [
    "Tune a hyperparameter.",
    "You can tune a piano but you can't tune a fish.",
    "Fish who eat fish, catch fish.",
    "People can tune a fish or a hyperparameter.",
    "It is hard to catch fish and tune it.",
]

vectorizer = CountVectorizer(stop_words='english') 
X = vectorizer.fit_transform(corpus) 
pd.DataFrame(X.A, columns=vectorizer.get_feature_names_out())
```

```
catch  eat  fish  hard  hyperparameter  like  people  piano  tune
0      0    0     0     0               1     0       0      0     1
1      0    0     1     0               0     1       0      1     1
2      1    1     3     0               0     0       0      0     0
3      0    0     1     0               1     0       1      0     1
4      1    0     1     1               0     0       0      0     1
```

我们可以更清楚地看到矩阵是什么样子的。**行是文档**,**列是专有字**。`CountVectorizer`带有各种内置的文本预处理，比如我们在这里做的删除停用词。如果一个句子包含一个单词，它会计算出现的次数，如果没有，它会使用一个`0`。BoW 方法会将更多的权重放在出现频率更高的单词上，因此您必须删除停用的单词。

# 什么是 TF-IDF？

我们看到，BoW 模型会计算出现的次数，并对大多数单词赋予更多的权重。另一种叫做 **TF-IDF** 的方法正好相反。TF-IDF 代表**Term Frequency-Inverse Document Frequency**，很好地说明了它使用的方法。它不是给更频繁出现的单词更多的权重，而是给不太频繁出现**的单词更高的权重(在整个语料库中*)。在你的文本中有更多的领域特定语言的用例中，这个模型通过给这些不经常出现的单词加权来表现得更好。让我们像以前一样在相同的文件上运行它。***

*在我们做逆文档频率之前，让我们用**术语频率**，它将像一把弓一样工作，但是给我们每个术语的值，其中矢量(文档)`= 1`的*平方和*。这与 BoW 模型相同，但是是归一化的。*

```
*vectorizer = TfidfVectorizer(stop_words='english', use_idf=False) 
X = vectorizer.fit_transform(corpus) 
df = pd.DataFrame(np.round(X.A,3), columns=vectorizer.get_feature_names_out())
df*
```

```
*catch    eat   fish  hard  hyperparameter  people  piano   tune
0  0.000  0.000  0.000   0.0           0.707     0.0  0.000  0.707
1  0.000  0.000  0.408   0.0           0.000     0.0  0.408  0.816
2  0.302  0.302  0.905   0.0           0.000     0.0  0.000  0.000
3  0.000  0.000  0.500   0.0           0.500     0.5  0.000  0.500
4  0.500  0.000  0.500   0.5           0.000     0.0  0.000  0.500*
```

*在第一个文档(`0`)中，我们看到单词`hyperparameter`，我们可以认为它是一个非常特定于领域的单词，与在整个语料库中更频繁出现的`tune,`具有相同的权重。*

*对于文档`2`，我们可以看到单词`fish`的值很大，因为它经常出现。现在我们有了自己的值，让我们看看应用**逆文档频率**时会发生什么。*

```
*vectorizer = TfidfVectorizer(stop_words='english') 
X = vectorizer.fit_transform(corpus) 
df = pd.DataFrame(np.round(X.A,3), columns=vectorizer.get_feature_names_out())
df*
```

```
*catch    eat   fish   hard  hyperparameter  people  piano   tune
0  0.000  0.000  0.000  0.000           0.820   0.000  0.000  0.573
1  0.000  0.000  0.350  0.000           0.000   0.000  0.622  0.701
2  0.380  0.471  0.796  0.000           0.000   0.000  0.000  0.000
3  0.000  0.000  0.373  0.000           0.534   0.661  0.000  0.373
4  0.534  0.000  0.373  0.661           0.000   0.000  0.000  0.373*
```

*我们来对比一下这两款。在第一个文档中，`hyperparameter`比 tune 具有更高的权重，因为它比单词 tune 少出现`50%`。但是，请注意，权重仍然依赖于文档；`tune`根据不同的上下文，在不同的文档中具有不同的权重。*

*对于文档`2`，我们可以看到，由于术语`fish`出现的频率，它的权重稍低。*

# *结论*

*希望这个快速概述有助于您理解 BOW 和 TF-IDF。虽然使用像 **Scikit-Learn** 这样的库来构建它们确实很容易，但是理解概念以及一个库何时可能比另一个库执行得更好还是很重要的。如果你想在实践中看到 TF-IDF，请查看关于数据科学的[聚类文本和**的 k-Means**](/clustering-text-with-k-means-c2953c8a9772) 的帖子。*

*在 [GitHub](https://github.com/broepke/BoW_TF-IDF) 上查看这篇文章的完整代码*

*如果你喜欢阅读这样的故事，并想支持我成为一名作家，可以考虑报名成为一名媒体成员。一个月 5 美元，让你可以无限制地访问成千上万篇文章。如果你使用[我的链接](https://medium.com/@broepke/membership)注册，我会赚一小笔佣金，不需要你额外付费。*

# *参考*

1.  *[对单词袋模型的温和介绍](https://machinelearningmastery.com/gentle-introduction-bag-words-model/)*
2.  *[乔纳森·索玛的 TF-IDF](https://jonathansoma.com/lede/foundations/classes/text%20processing/tf-idf/)*