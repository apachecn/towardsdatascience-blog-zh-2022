# 自然语言处理任务中模糊匹配的原始文本校正

> 原文：<https://towardsdatascience.com/raw-text-correction-with-fuzzy-matching-for-nlp-tasks-828547742ef7>

## 了解如何修复拼写错误的单词，以便更好地识别重要的文本表达式

![](img/013d48202172113b5d4f13300cb28407.png)

Diomari Madulara 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

今天，自然语言处理(NLP)被用于医疗保健、金融、营销等领域的许多 ML 任务和项目中。数据科学家经常努力清理和分析文本数据，以便获得洞察力。对于大多数自然语言处理任务，通常使用诸如标记化、词干化、词汇化等技术。

但是，在某些情况下，需要保持原始文本的完整性，而不是将其拆分成标记。例如，在作为命名实体识别(NER)的私人情况的数据去标识中，一种用于识别文档中不同实体的方法，该方法的输出显示原始文本，其中标签替换期望的实体。

在这些情况下，纠正拼写错误或错误的术语可能会很有挑战性。这篇文章将解释如何结合使用正则表达式和模糊字符串匹配来完成这项任务。

**模糊字符串匹配**

模糊字符串匹配是一种查找与给定字符串模式近似匹配的字符串的技术。模糊字符串匹配背后的算法使用距离度量(如 Levenshtein 距离)来计算两个字符串之间的差异，方法是确定将第一个字符串转换为第二个字符串所需的最少更改次数。我们将使用 python 库 *Fuzzywuzzy* 来执行这项任务。

安装和示例:

```
pip install fuzzywuzzyfrom fuzzywuzzy import fuzzfuzz.ratio("appls","apples")#91
```

在这个例子中，我们得到了 91 分的相似度，所以单词非常相似。现在，我们可以考虑使用什么阈值来决定是否“纠正”原始单词。

**正则表达式**

RegEx 是正则表达式的缩写，是一种特殊的文本字符串，用于指定文本中的搜索模式。这种模式基本上是一种语言，它确切地定义了在文本字符串中要寻找什么。例如，如果我们想提取除数字以外的所有字符，正则表达式模式将是:

```
[^0-9]+
```

如果我们想要提取所有的电子邮件地址，正则表达式模式将是:

```
[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}
```

我们将使用 python 库 *re* 来执行这项任务。

安装和示例:

```
pip install reimport restring = "my e-mail is example@email.com"pattern=r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}'print(re.search(pattern,string).group())# example@email.com
```

**原始文本校正**

回到最初的问题，当我们需要修复错误的单词或短语，但保持原始文本完整，而不是将其拆分为标记时，该怎么办。保持文本完整也意味着保持完全相同的间距、制表符、换行符、标点符号等。

假设在一个 NER 任务中，我们想要标记医院里给病人的药物。该信息可在电子健康记录(EMR)系统的医生笔记部分获得。例如，这里有一个条目:

> 病人 XXX 上周住院了。
> 
> 他被赋予了勇气。
> 
> 他是男性，65 岁，有心脏病史。

该药物的正确名称是“莫昔普利”。为了纠正这一点，我们需要:

1.  准备一个我们想要搜索的关键字列表，在这种情况下只有一个关键字
2.  决定相似性阈值(默认为 85)
3.  将文本拆分成标记
4.  在关键字和每个标记之间运行模糊匹配
5.  如果相似性得分超过预定阈值，则用关键字替换标记
6.  把它们放回原处

我们可以使用如下函数来实现这一点:

```
from fuzzywuzzy import fuzz
import redef fuzzy_replace(keyword_str, text_str, threshold=85):
    l = len(keyword_str.split())
    splitted = re.split(r'(\W+)',text_str) #split, keep linebreaks
    for i in range(len(splitted)-l+1):
        temp = "".join(splitted[i:i+l])
        if fuzz.ratio(keyword_str, temp) >= threshold:
            before = "".join(splitted[:i])
            after = "".join(splitted[i+l:])
            text_str= before + keyword_str + after
            splitted = re.split(r'(\W+)',text_str)    
    return text_str
```

运行此函数后，文本输出现已得到纠正，文本原始结构得以保留:

> 病人 XXX 上周住院了。
> 
> 他被给予莫昔普利。
> 
> 他是男性，65 岁，有心脏病史。

现在让我们看一个更复杂的例子。这一次，医疗记录中包含了一些我们想要纠正的不同的药物，而且有些在文本中出现了不止一次。为了解决这个问题，我们定义了一个包含所有正确药物名称的列表，并简单地遍历文本以找到需要纠正的内容。以下代码片段显示了如何实现这一点:

```
meds = ["moexipril", "vasotec", "candesartan"] text = """The patient XXX was hospitalized last week.He was given moxiperil and vasotek.He is male, 65 years old, with a history of heart disease.Patient has been taking vasotek for several years.In the past was given candasarta.""" for med in meds: text = fuzzy_replace(med, text)
```

结果是所有药物名称均已更正的相同文本。

> 病人 XXX 上周住院了。
> 
> 他接受了莫昔普利和 vasotec 治疗。
> 
> 他是男性，65 岁，有心脏病史。
> 
> 患者服用 vasotec 已有数年。
> 
> 曾被给予坎地沙坦。

我已经决定不包括强制转换成小写的文本，因为有些时候人们希望保持原来的大小写，例如识别缩写。然而，这可以通过将参数的小写形式输入到函数中很容易地完成，就像这样— `fuzzy_replace(med.lower(),text.lower())`

**结论**

我们可以使用模糊字符串匹配和正则表达式的组合来纠正错误的单词或短语，并保持原始文本不变。当处理不同的 NLP 任务(如 NER)时，这种操作是可取的。