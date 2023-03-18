# FuzzyTM:一个用于模糊主题模型的 Python 包

> 原文：<https://towardsdatascience.com/fuzzytm-a-python-package-for-fuzzy-topic-models-fd3c3f0ae060>

![](img/d3e438a8d2d5ee6d7de018412f27710c.png)

莎伦·麦卡琴在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

在我以前的文章中，我展示了如何开始使用 Python 包 OCTIS，这是一个比较和优化最先进的主题建模算法的包。[第一篇](/a-beginners-guide-to-octis-optimizing-and-comparing-topic-models-is-simple-590554ec9ba6)展示如何入门 OCTIS，[第二篇](/a-beginners-guide-to-octis-vol-2-optimizing-topic-models-1214e58be1e5)重点介绍模型优化。我和我的团队开始与 OCTIS 合作的原因是，我们开发了一种新的主题建模算法，称为 FLSA-W [1]，并希望看到它与现有的最先进的技术相比表现如何。基于对各种开放数据集的比较，我们发现它在大多数情况下在一致性(c_v)、多样性和可解释性方面优于其他模型(如 LDA、ProdLDA、NeuralLDA、NMF 和 LSI)。我还不能分享这些结果，因为我们已经将这项工作提交给一个会议，正在等待接受。与此同时，我们还开发了一个 Python 包， [FuzzyTM](https://pypi.org/project/FuzzyTM/) ，其中包含 FLSA-W 和另外两个基于模糊逻辑的主题建模算法(FLSA 和 FLSA-V)。这篇文章将简要描述模糊主题模型和 FLSA-W 的基本原理，然后演示如何开始使用 FuzzyTM(如果你想开始训练一个模型，只需转到“开始使用 FuzzyTM”)。在以后的帖子中，我将更详细地解释各种算法是如何工作的，并使用 OCTIS 将它们与现有算法进行比较。

# 模糊主题模型

注意，虽然有 50 种深浅不同的主题建模算法，但它们都分别返回两个矩阵 **P(W_i|T_k)** 和 **P(T_k|D_j)** 、给定主题的单词概率和给定文档的主题概率。

2018 年，模糊潜在语义分析被提出[2]，并在一致性方面优于 LDA。FLSA 利用贝叶斯定理、奇异值分解(SVD)和矩阵乘法求 **P(W_i|T_k)** 和 **P(T_k|D_j)** 。

对于主题建模，我们从文本语料库开始。在 Python 中，这被存储为字符串列表的列表，其中每个列表代表一个文档，每个字符串是文档中的一个单词。

让我们从定义下列量开始:

M —数据集中唯一单词的数量

N —数据集中文档的数量

C —主题的数量

S —奇异值分解维数

*i* —单词索引， *i* ∈ {1，2，3，…，M}

*j —* 文档索引， *j* ∈ {1，2，3，…，N}

*k* —主题索引， *k* ∈ {1，2，3，…，C}

那么，在 FLSA 获得 **P(W_i|T_k)** 和 **P(T_k|D_j)** 的步骤如下:

1.  **获取本地术语权重** ( *MxN* ) —指示单词 *i* 在文档 *j.* 中出现的频率的文档术语矩阵
2.  **获取全局术语权重** ( *MxN* ) —在该步骤中，一个文档中的单词的出现与其他文档中的单词的出现相关。
3.  在全局术语权重 ( *NxS* )上从 **SVD 获得**U**—SVD 用于降维，见[本帖](https://medium.com/towards-data-science/svd-8c2f72e264f)对 SVD 的直观解释。**
4.  对 **U^T** 使用模糊聚类得到
    **p(t|d)^t**(*mxc*)—最常用的方法是模糊 c 均值聚类，但 FuzzyTM 中有各种算法。
5.  使用基于贝叶斯定理的矩阵乘法，使用 **P(T|D)^T** 和 P(D_j)得到 **P(W_i|T_k)** 。

在 FLSA，SVD 的 **U** 矩阵被用作聚类的输入，这意味着文档正在被聚类。由于主题模型经常被用于寻找对应于主题的单词，所以采用奇异值分解的 **V^T** 进行聚类似乎更有意义，因为现在单词正在被聚类。在 FLSA-W(现在“W”有意义了，希望)中，单词而不是文档被聚集在一起。

# FuzzyTM 入门

FuzzyTM 是模块化构建的，因此每个算法的步骤都是父类中不同的方法，每个算法都是调用父类中方法的子类。新手可以用最少的工作训练各种主题模型，而研究人员可以修改每个步骤并添加更多的功能。

让我们开始使用 OCTIS 包中的数据集训练模型。首先，我们使用以下代码安装 FuzzyTM:

```
pip install FuzzyTM
```

其次，我们导入数据集:

```
from octis.dataset.dataset import Dataset
dataset = Dataset()
dataset.fetch_dataset('DBLP')
data = dataset._Dataset__corpus
```

让我们看看这个数据集是什么样子的:

```
print(data[0:5])
>>> [['fast', 'cut', 'protocol', 'agent', 'coordination'],
 ['retrieval', 'base', 'class', 'svm'],
 ['semantic', 'annotation', 'personal', 'video', 'content', 'image'],
 ['semantic', 'repository', 'modeling', 'image', 'database'],
 ['global', 'local', 'scheme', 'imbalanced', 'point', 'matching']]
```

现在，我们准备导入 FuzzyTM:

```
from FuzzyTM import FLSA_W
```

我们将模型初始化如下(` num_words '的缺省值是 20，但为了清楚起见，这里我只给出了 10 个单词):

```
flsaW = FLSA_W(input_file = data, num_topics=10, num_words=10)
```

然后，我们得到 **P(W_i|T_k)** 和 **P(T_k|D_j)** 如下:

```
pwgt, ptgd = flsaW.get_matrices()
```

现在，我们可以开始看主题了:

```
topics = flsaW.show_topics(representation='words')print(topics)
>>> [['machine', 'decision', 'set', 'evaluation', 'tree', 'performance', 'constraint', 'stream', 'process', 'pattern'], ['face', 'robust', 'tracking', 'error', 'code', 'filter', 'shape', 'detection', 'recognition', 'color'], ['generalization', 'neighbor', 'predict', 'sensitive', 'computation', 'topic', 'link', 'recursive', 'virtual', 'construction'], ['language', 'logic', 'data', 'web', 'mining', 'rule', 'processing', 'discovery', 'query', 'datum'], ['factorization', 'regularization', 'people', 'measurement', 'parametric', 'progressive', 'dimensionality', 'histogram', 'selective', 'correct'], ['active', 'spatial', 'optimal', 'view', 'level', 'modeling', 'combine', 'hierarchical', 'dimensional', 'space'], ['correspondence', 'calibration', 'compress', 'curve', 'geometry', 'track', 'background', 'appearance', 'deformable', 'light'], ['heuristic', 'computational', 'update','preference', 'qualitative', 'mechanism', 'engine', 'functional', 'join', 'relation'], ['graphic', 'configuration', 'hypothesis', 'walk', 'relaxation', 'family', 'composite', 'factor', 'string', 'pass'], ['theorem', 'independence', 'discourse', 'electronic', 'auction', 'composition', 'diagram', 'version', 'hard', 'create']]
```

从这个输出中我们可以识别出一些主题:第一个主题似乎是关于通用机器学习的，第二个主题是关于图像识别的，第四个主题是关于自然语言处理的。

现在，让我们来看看分数评估指标:

```
#Get coherence value
flsaW.get_coherence_value(input_file = data, topics = topics)
>>> 0.34180921613509696#Get diversity score
flsaW.get_diversity_score(topics = topics)
>>> 1.0#Get interpretability score
flsaW.get_interpretability_score(input_file = data, topics = topics)
>>> 0.34180921613509696
```

由于主题模型的输出由各种主题组成，其中每个主题都是单词的集合，因此主题模型的质量应该关注每个主题内的单词质量(主题内质量)和不同主题之间的差异(主题间质量)。连贯性分数捕捉每个主题内的单词相互支持的程度，多样性分数显示每个主题的多样性(主题之间是否有单词重叠)。然后，可解释性得分结合了这两个指标，并被计算为一致性和多样性之间的乘积。

从结果可以看出，FLSA-W 具有完美的多样性。这并不奇怪，因为它明确地将单词聚集在一起。不过，与大多数现有算法相比，这是一个很大的改进。

虽然 0.3418 的一致性分数看起来相当低，但是比较实验结果将显示，在大多数设置中，FLSA-W 具有比其他算法更高的一致性分数。

# 结论

在这篇文章中，我简要解释了主题模型 FLSA 和 FLSA-W 是如何工作的。然后，我演示了只用两行代码就可以训练 FLSA-W，以及如何分析主题。除了训练主题模型之外，FuzzyTM 还包含一种方法，用于基于训练的主题模型获得新文档的主题嵌入。这对于下游任务(如文本分类)的文档嵌入非常有用。在以后的文章中，我将更详细地描述这些算法，并将 FLSA-W 与现有算法进行比较。更多详情请看我的 Github 页面:[https://github.com/ERijck/FuzzyTM](https://github.com/ERijck/FuzzyTM)。

[1] Rijcken，e .，Scheepers，f .，Mosteiro，p .，Zervanou，k .，Spruit，m .，& Kaymak，U. (2021 年 12 月)。模糊话题模型和 LDA 在可解释性方面的比较研究。在 *2021 IEEE 计算智能系列研讨会(SSCI)* 。IEEE。

[2]a .卡拉米，a .甘戈帕迪亚，a .周，b .，&哈拉齐，H. (2018)。健康和医学语料库中的模糊方法主题发现。*国际模糊系统杂志*， *20* (4)，1334–1345。