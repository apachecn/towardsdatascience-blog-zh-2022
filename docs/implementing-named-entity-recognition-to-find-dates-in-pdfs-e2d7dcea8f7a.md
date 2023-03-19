# 实现命名实体识别以在 pdf 中查找日期

> 原文：<https://towardsdatascience.com/implementing-named-entity-recognition-to-find-dates-in-pdfs-e2d7dcea8f7a>

## 命名实体识别是一种自然语言处理技术，可以帮助从 PDF 文档中提取感兴趣的日期

![](img/62941f4643dc0c3d6496305f1f5e1c28.png)

[伊森·m .](https://unsplash.com/@itsethan?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

数据很重要。句号。虽然人们可以找到专门为某个数据科学任务定制的数据集，但总有一天这些数据集会变得缺乏原创性并被过度使用。最近我一直在用 Python 处理 pdf，老实说，这一开始并不是最容易的任务。您希望能够使用 PDF 文档的原因是，它们在互联网上非常容易访问，并且可以为您的数据集提供大量新数据。你可以看看我以前做过的 pdf 文件。此后，我清理并更改了我使用的一些代码，展示了您的代码是如何不断发展变化的！

[](/natural-language-processing-pdf-processing-function-for-obtaining-a-general-overview-6fa63e81fbf1)  

# →为什么日期在自然语言处理中很重要

如果您目前正在进行 NLP 项目，查找感兴趣的日期可能有助于在数据中找到趋势或模式。具体事件是什么时候发生的？事件发生的时间有规律吗？文本中有哪些重要的日期？围绕语料库中的日期，你可以也应该问自己很多问题。

# **→让我们深入研究代码！**

这段代码需要的主要库是 [**PyPDF2**](https://pypi.org/project/PyPDF2/) ， [**Spacy**](https://spacy.io/) ，以及[**re**](https://docs.python.org/3/library/re.html)**(Python 固有的)**。**首先，我们将创建我们的 PDF 解析类。我们想要创建的第一个函数是 pdf 阅读器，它允许我们将 PDF 读入 Python。**

**PDF 阅读器(来自作者)**

**我创建了一个快速示例 pdf 来展示这个解析类的强大功能。pdf 是:**

> **我叫 Ben，从 2020 年春天开始从事数据科学的工作。今天是 2022 年 8 月 31 日。这个类可以帮助找到感兴趣的日期。例如，也许你想了解 2014 年初发生的一件事的更多信息。这有助于找到这些日期**

**要将 pdf 读入 Python，只需将 pdf 的文件路径放入 pdf 解析对象类中。**

```
pdf = PdfParser('your file path here')
text = pdf.pdf_reader()
```

**现在我们的 PDF 已经处理完毕，我们可以创建一个函数来提取日期。**

**日期查找功能(来自作者)**

**这个函数使用命名实体识别在标注为 DATE 的语料库中查找任何标签。为此，您需要确保从 Spacy 下载“en_core_web_lg”。了解如何[在这里](https://spacy.io/models/en)！在该函数中，创建了一个日期集，以防有任何重复(可能我们会两次找到同一年)。如果您关心文本正文中的日期频率，您可以将它更改为列表。接下来，让我们在 pdf 的文本上运行这个函数。**

```
text.date_extractor()
```

**运行此代码会产生以下输出:**

```
['early 2014', 'Today', 'August 31, 2022', '2020']
```

**嘣！有用！就像这样，我们可以将 pdf 转换成 python 可读的格式，并找到感兴趣的日期！**

# **结论**

**今天，我们不仅学习了如何将 PDF 读入 Python，还学习了如何从 PDF 中提取感兴趣的日期。虽然在自然语言处理中找到文本的情感很重要，但知道一个事件何时发生，或者人们何时对一家公司做出负面评论，可能是有见地的。例如，在分析消费者评论时，可以在业务分析中使用该函数。如果一家公司收到的所有负面评价都发生在夏季，那该怎么办？也许这是由于公司由于预期消费者需求减少而减少了员工数量(许多人在夏天度假)。该公司可以将这些评论归因于该方面，而不是一个有缺陷的产品(在进一步调查后)，因为创建了这个功能，我一直在我的分析中寻找时间线模式，它无疑帮助我发现了潜在的模式。让我知道这如何为你工作！**

****如果你喜欢今天的阅读，请关注我，让我知道你是否还有其他想让我探讨的话题(这对我的帮助超乎你的想象)！如果没有中等账号，在这里** **报名** [**！另外，在**](https://medium.com/@ben.mccloskey20/membership)[**LinkedIn**](https://www.linkedin.com/in/benjamin-mccloskey-169975a8/)**上加我，或者随时联系！感谢阅读！****

**这里是完整的代码！希望这能对你以后的项目有所帮助！**

**完整代码(来自作者)**