# 更多情感:文本分析包

> 原文：<https://towardsdatascience.com/morethansentiments-a-python-library-for-text-quantification-e57ff9d51cd5>

## 帮助研究人员计算样本、冗余、特异性、相对患病率等的函数集合。，在 python 中

![](img/9db8cf60a036fb6530db733b4916c10a.png)

帕特里克·托马索在 [Unsplash](https://unsplash.com/s/photos/text?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

# 介绍

【morethanworthies(Jiang 和 Srinivasan，2022)】是一个 python 库，用于帮助研究人员计算样本(Lang 和 Stice-Lawrence，2015)、冗余度(Cazier 和 Pfeiffer，2017)、特异性(Hope 等人，2016)、相对患病率(Blankespoor，2019)等。如今，人们经常谈论文本嵌入、语义相似性、意图检测和情感分析……然而，MoreThanSentiments 受到这样一种想法的启发，即适当量化文本结构也将有助于研究人员提取大量有意义的信息。并且这个独立于领域的包易于在各种文本量化任务的项目中实现。

# 支持的测量

## 样板文件

在文本分析中，*样板文件*是可以从句子中删除的单词组合，不会显著改变原来的意思。换句话说，它是信息含量的一种度量。它是根据包含样板文件的句子占总字数的比例来计算的。

![](img/df3340b17f3e9376a051359c67958cc4.png)

作者图片

## 裁员

冗余是对文本有用性的一种衡量。它被定义为在每个文档中出现一次以上的超长句子/短语(例如 10 克)的百分比。直觉上，如果一个超长的句子/短语被重复使用，这意味着作者试图重复前面提到的信息。

## 特征

特异性是对与特定主题唯一相关的质量的度量。它被定义为特定实体名称、数量值和时间/日期的数量，所有这些都与文档中的总字数成比例。目前，特异性的功能建立在 spaCy 的命名实体识别器上。

## 相对患病率

相对患病率是对硬信息的一种衡量。它是相对于整个文本长度的数值数量。它有助于评估给定文本中的定量信息部分。

# 装置

安装工具箱最简单的方法是通过 pip(某些发行版中的 pip3):

# 使用

## 导入包

## 从 txt 文件中读取数据

这是一个内置的功能，可以帮助你读取一个文件夹的分隔。python 中的 txt 文件。如果你已经把所有的数据存储在一个. csv 文件中，你可以像平常一样用熊猫来读这个文件。

## 句子标记

如果您想要计算样板和冗余，有必要对句子进行标记，因为 n 元语法是在句子级别上生成的。

## 干净的数据

如果你想在句子层面上清理:

如果要在文档级别进行清洗:

对于数据清理功能，我们提供以下选项:

*   lower:使所有单词小写
*   标点符号:删除语料库中的所有标点符号
*   数字:删除语料库中的所有数字
*   unicode:删除语料库中的所有 Unicode
*   停用词:删除语料库中的停用词

## 样板文件

参数:

*   input_data:这个函数需要标记化的文档。
*   n:要使用的 ngrams 的数量。默认值为 4。
*   min_doc:在构建 ngram 列表时，忽略文档频率严格低于给定阈值的 ngram。默认为 5 个文档。建议文件数量的 30%。min_doc 还可以将 0 到 1 之间的数字读取为百分比。(例如，0.3 将被读作 30%)
*   get_ngram:如果该参数设置为“True”，将返回一个包含所有 ngram 和相应频率的数据帧，“min_doc”参数将失效。

## 裁员

参数:

*   input_data:这个函数需要标记化的文档。
*   n:要使用的 n 元语法的数量。默认值为 10。

## 特征

参数:

*   input_data:该函数需要没有标记化的文档

## 相对患病率

参数:

*   input_data:该函数需要没有标记化的文档

# 结论

morthan opportunities 仍是一个发展中的项目。然而，它已经显示出在不同领域帮助研究人员的潜力。这个软件包简化了量化文本结构的过程，并为他们的 NLP 项目提供了各种文本分数。

以下是完整示例的链接:

*   [Python 脚本](https://github.com/jinhangjiang/morethansentiments/blob/main/tests/test_code.py)
*   [巨蟒 Jupyter 笔记本](https://github.com/jinhangjiang/morethansentiments/blob/main/Boilerplate.ipynb)

# 引用

> **如果这个包对你的工作有帮助，请随意引用它为**
> 
> 《情感:一个文本分析包》。软件影响，100456 (2022)。[https://doi.org/10.1016/J.SIMPA.2022.100456](https://doi.org/10.1016/J.SIMPA.2022.100456)

# 相关阅读

[](/use-r-to-calculate-boilerplate-for-accounting-analysis-f4a5b64e9b0d) [## 使用 R 计算用于会计分析的样板文件

### 用 30 家电信公司 CSR 计算样本的实证。

towardsdatascience.co](/use-r-to-calculate-boilerplate-for-accounting-analysis-f4a5b64e9b0d) 

[使用 R 计算用于会计分析的样板文件](/use-r-to-calculate-boilerplate-for-accounting-analysis-f4a5b64e9b0d)

# 参考

BLANKESPOOR，E. (2019)，信息处理成本对企业披露选择的影响:来自 XBRL 要求的证据。会计研究杂志，57:919–967。https://doi.org/10.1111/1475-679X.12268

希望，好的。胡，d .和陆，H. (2016)，特定风险因素披露的好处。*启帐螺柱* 21 **，**1005–1045。[https://doi.org/10.1007/s11142-016-9371-1](https://doi.org/10.1007/s11142-016-9371-1)

理查德·a·卡齐尔，雷·j·普发。(2017)，10-K 披露重复和管理报告激励。*财务报告杂志*；2 (1): 107–131.[https://doi.org/10.2308/jfir-51912](https://doi.org/10.2308/jfir-51912)

马克·朗，洛里安·斯泰斯·劳伦斯。(2015)，文本分析和国际财务报告:大样本证据，会计和经济学杂志，第 60 卷，第 2-3 期，第 110-135 页，ISSN 0165-4101。https://doi.org/10.1016/j.jacceco.2015.09.002。