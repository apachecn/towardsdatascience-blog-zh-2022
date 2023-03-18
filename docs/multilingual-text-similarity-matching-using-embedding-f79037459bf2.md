# 使用嵌入的多语言文本相似性匹配

> 原文：<https://towardsdatascience.com/multilingual-text-similarity-matching-using-embedding-f79037459bf2>

## 使用句子转换器进行对称的语义搜索

![](img/e8d9ef8d4266eee48327d6dc26ae73c4.png)

拉奎尔·马丁内斯在 [Unsplash](https://unsplash.com/s/photos/compare?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

今天的主题是计算相同或不同语言的两个句子之间的相似度。我们将利用`sentence-transformer`框架，它自带预先训练好的多语言 transformer 模型。

我们可以利用这些模型来计算 50 多种语言的文本嵌入。然后，输出嵌入可以用于对称语义搜索。

## 对称和非对称语义搜索的区别

> **对称语义搜索**专注于基于输入查询从语料库中寻找相似问题。例如，鉴于“如何在线学习人工智能？”作为输入查询，预期的输出应该类似于“如何在 web 上学习人工智能？”大多数时候，您可能会翻转查询和语料库中的数据，最终仍然得到与输出相同的配对。对称语义搜索主要用于文本挖掘或意图分类任务。
> 
> 另一方面，**不对称语义搜索**围绕基于输入查询从语料库中寻找答案。例如，鉴于“什么是 AI？”作为输入查询，您可能会期望输出类似“AI 是一种模仿人类智能来执行任务的技术。他们可以根据获得的信息学习和提高自己的知识。”输入查询不仅限于问题。可以是关键词，也可以是短语。非对称语义搜索适用于搜索引擎相关的任务。

在撰写本文时，`sentence-transformer`框架提供了以下用于多语言对称语义搜索的[预训练模型](https://www.sbert.net/docs/pretrained_models.html#model-overview):

*   `distiluse-base-multilingual-cased-v1`—15 种语言通用语句编码器的多语言模型。
*   `distiluse-base-multilingual-cased-v2`—50 种语言通用语句编码器的多语言模型。
*   `paraphrase-multilingual-MiniLM-L12-v2` —paraphrase-multilingual-MiniLM-L12-v2 的多语言模型，扩展到 50+种语言。
*   `paraphrase-multilingual-mpnet-base-v2` —paraphrase-mpnet-base-v2 的多语言模型，扩展到 50+种语言。

实际上，我们可以利用这些模型来计算英语句子和西班牙语句子之间的相似度。例如，给定我们语料库中的以下句子:

```
What are you doing?
I am a boy
Can you help me?
A woman is playing violin.
The quick brown fox jumps over the lazy dog
```

并输入如下查询:

```
Qué estás haciendo
```

相似度最高的句子应该是:

```
What are you doing?
```

为简单起见，我们的对称语义搜索的工作流程如下:

1.  计算查询和语料库文本的嵌入
2.  计算两个嵌入之间的余弦相似度
3.  查找具有最高相似性得分的前 5 个索引

# 设置

在此之前，让我们创建一个新的虚拟环境，并安装所有必需的包。

## 用 pip 安装

您可以轻松安装`sentence-transformer`包:

```
pip install -U sentence-transformers
```

## 用康达安装

对于 Anaconda 用户，您可以直接安装该软件包，如下所示:

```
conda install -c conda-forge sentence-transformers
```

继续下一节的实施。

# 履行

在您的工作目录中，创建一个名为`main.py`的新 Python 文件。

## 导入

在文件顶部添加以下导入语句:

```
from sentence_transformers import SentenceTransformer, util
import torch
```

## 模型初始化

然后，通过调用`SentenceTransformer`类初始化模型，并传入所需模型的名称:

```
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

在初始运行期间，模块将下载预训练的模型文件，作为缓存在以下目录中:

```
# linux
~/.cache/huggingface/transformers# windows (replace username with your username)
C:\Users\<username>\.cache\huggingface\transformers
```

您可以将缓存文件夹修改为当前工作目录，如下所示:

```
# save model in current directory
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder='./')# save model in models folder (you need to first create the folder)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder='./models/')
```

对于生产，您应该将模型移动到工作目录并在本地加载。例如，假设模型文件位于`models`文件夹，您可以如下初始化您的模型:

```
model = SentenceTransformer('models/sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2')
```

如果您在一台只有 CPU 的机器上进行测试，只需将`device`参数设置为`cpu`:

```
model = SentenceTransformer('models/sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
```

## 语料库和查询

接下来，初始化语料库和查询的数据。在这种情况下，我有一个 7 个字符串的列表作为`corpus`数据，而`queries`包含一个不同语言的 3 个字符串的列表。

```
corpus = [
'I am a boy',
'What are you doing?',
'Can you help me?',
'A man is riding a horse.',
'A woman is playing violin.',
'A monkey is chasing after a goat',
'The quick brown fox jumps over the lazy dog'
]queries = ['I am in need of assistance', '我是男孩子', 'Qué estás haciendo']
```

## 将数据编码到嵌入中

调用`encode`函数将语料库转换为嵌入。将`convert_to_tensor`参数设置为`True`以获取 Python 张量作为输出。同样，初始化一个名为`top_k`的新变量，并将其赋值为最小值 5 和语料库的总长度。我们稍后将使用这个变量来获得具有最高相似性得分的索引。

```
corpus_embedding = model.encode(corpus, convert_to_tensor=True)top_k = min(5, len(corpus))
```

> `encode`函数接受字符串列表或单个字符串作为输入。

## 计算余弦相似度

最后一步是遍历查询中的所有项目，并执行以下操作:

*   计算单个查询的嵌入。每个嵌入具有以下形状:`torch.Size([384])`
*   调用 `util.cos_sim`函数来获得输入查询和语料库之间的相似性得分
*   调用`torch.topk`函数获得 topk 结果
*   打印输出作为参考

```
for query in queries:
    query_embedding = model.encode(query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, corpus_embedding)[0]
    top_results = torch.topk(cos_scores, k=top_k) print("Query:", query)
    print("---------------------------")
    for score, idx in zip(top_results[0], top_results[1]):
        print(f'{round(score.item(), 3)} | {corpus[idx]}')
```

`top_results`变量是一个元组，包含:

*   表示输入查询和语料库之间的相似性得分的张量阵列

```
tensor([ 0.3326,  0.2809,  0.2258, -0.0133, -0.0333])
```

*   表示输入查询索引的张量数组

```
tensor([2, 0, 1, 4, 3])
```

您可以在以下要点的[中找到完整的代码:](https://gist.github.com/wfng92/448b80bd8ad94ed255788c089c18f5f0)

## 输出

运行该脚本时，您应该在终端上得到以下输出:

```
Query: I am in need of assistance
---------------------------
0.333 | Can you help me?
0.281 | I am a boy
0.226 | What are you doing?
-0.013 | A woman is playing violin.
-0.033 | A man is riding a horse.Query: 我是男孩子
---------------------------
0.919 | I am a boy
0.343 | What are you doing?
0.192 | Can you help me?
0.058 | A monkey is chasing after a goat
-0.001 | The quick brown fox jumps over the lazy dogQuery: Qué estás haciendo
---------------------------
0.952 | What are you doing?
0.396 | I am a boy
0.209 | Can you help me?
0.037 | A woman is playing violin.
0.032 | The quick brown fox jumps over the lazy dog
```

# 最佳化

上面的实现对于小型语料库(低于 100 万个条目)非常有用。对于大型语料库，执行会相对较慢。因此，我们需要优化实现，以便它可以无缝地工作。一些最流行的优化技术包括:

*   归一化嵌入并使用点积作为得分函数
*   使用近似最近邻将语料库划分成相似嵌入的较小部分

为了保持简单和简短，本教程将只涵盖第一种技术。当您规范化嵌入时，输出向量的长度将为 1。因此，我们可以使用点积而不是余弦相似度来计算相似度得分。点积是一个更快的损失，你会得到相同的相似性分数。

## 标准化嵌入

有两种方法可以标准化嵌入。第一种方法是在调用`encode`函数时将`normalize_embeddings`参数设置为`True`。

```
corpus_embedding = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)
```

或者，您可以利用`util.normalize_embeddings`函数来规范化一个现有的嵌入:

```
corpus_embedding = model.encode(corpus, convert_to_tensor=True)
corpus_embedding = util.normalize_embeddings(corpus_embedding)
```

## 计算点积

调用`util.semantic_search`函数并传入`util.dot_score`作为`score_function`的输入参数。它将返回带有关键字`corpus_id`和`score`的字典列表。此外，该列表按余弦相似性得分递减排序。

```
hits = util.semantic_search(query_embedding, corpus_embedding, score_function=util.dot_score)
```

修改后，[新的执行代码](https://gist.github.com/wfng92/27f99e027cdbf8f14a6d18b99312897d)应该如下:

当您运行脚本时，您应该得到与第一个实现相同的输出:

```
Query: I am in need of assistance
---------------------------
0.333 | Can you help me?
0.281 | I am a boy
0.226 | What are you doing?
-0.013 | A woman is playing violin.
-0.033 | A man is riding a horse.Query: 我是男孩子
---------------------------
0.919 | I am a boy
0.343 | What are you doing?
0.192 | Can you help me?
0.058 | A monkey is chasing after a goat
-0.001 | The quick brown fox jumps over the lazy dogQuery: Qué estás haciendo
---------------------------
0.952 | What are you doing?
0.396 | I am a boy
0.209 | Can you help me?
0.037 | A woman is playing violin.
0.032 | The quick brown fox jumps over the lazy dog
```

# 结论

让我们回顾一下你今天所学的内容。

本文首先简要介绍了`sentence-transformer` 模块。然后，比较了对称和非对称语义搜索的区别。

随后，它介绍了设置和安装。`sentence-transformer`可以安装`pip`或`conda`。

在实现部分，本文重点介绍了将语料库编码到嵌入中的步骤，以及使用余弦相似度计算相似度得分的步骤。

最后一节讨论优化技术。一种优化技术是将嵌入归一化为长度 1，然后使用点积计算相似性得分。

感谢你阅读这篇文章。祝你有美好的一天！

# 参考

1.  [句子变压器—安装](https://www.sbert.net/docs/installation.html)
2.  [句子变压器—预训练模型](https://www.sbert.net/docs/pretrained_models.html)
3.  [SentenceTransformer —语义搜索](https://www.sbert.net/examples/applications/semantic-search/README.html)