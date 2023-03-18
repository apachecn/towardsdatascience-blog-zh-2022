# 拥有 GPT-3 和 GPT-J 的先进 NER

> 原文：<https://towardsdatascience.com/advanced-ner-with-gpt-3-and-gpt-j-ce43dc6cdb9c>

几年前出现了基于变形金刚的生成式深度学习模型。GPT-3 和 GPT-J 是当今最先进的文本生成模型，它们非常强大，几乎彻底改变了许多传统的 NLP 用例。实体抽取(NER)就是其中之一。

在本文中，我们将看到如何使用 GPT 模型来执行高级实体提取，而无需注释任何数据。注释和训练自己的 NER 模型一直是一个挑战，但幸运的是，这不再是必要的了！

由于 GPT-3 可能相当昂贵，我们所有的例子都将基于 GPT-J 通过 [NLP 云 API](https://nlpcloud.io) 进行快速原型制作。

![](img/f7ffa26bbed007e921c604ff069c9dd1.png)

法国阿尔卑斯山(图片由作者提供)

## 用老方法做 NER

实体提取是最古老的 NLP 用例之一，也可能是当今生产中最常见的 NLP 用例之一。传统上，像 [spaCy](https://spacy.io) 和 [NLTK](https://www.nltk.org/) 这样的框架曾经是最好的选择。

SpaCy 非常适合生产，因为它易于使用，并且在生产中运行速度极快。SpaCy 提供了许多预先训练的模型，人们可以很容易地下载并在生产中立即开始使用。空间模型支持一些现成的本地实体，如地址、日期、货币…

NLTK 不适合生产，但它是研究项目和 NLP 升级的一个很好的工具。NLTK 也提出了一些预训练的模型，但是比 spaCy 少了一些本地实体。

这些预训练模型的主要问题是，它们只支持它们被训练的原生实体…而这些实体在现实生活项目中很少有用。大多数公司希望使用 NER 提取自定义实体，如职位、产品名称、电影名称、餐馆等。唯一的解决方案是通过漫长而乏味的注释过程为这些新实体创建一个庞大的数据集，然后训练一个新的模型。每当人们想要支持一个新的实体时，唯一的解决方案就是重新注释和训练。像 [Prodigy](https://prodi.gy/) 这样伟大的注释工具确实有所帮助，但它仍然需要一个或几个人力资源在潜在的长时间内做大量的工作。

希望 GPT 大型语言模型现在能解决这个问题。

## 什么是 GPT-3 和 GPT-J？

2020 年 5 月，OpenAI 发布了一个巨大的 NLP 模型:GPT-3。

GPT-3 是一个基于变形金刚的大型语言模型，它开始革新自然语言处理领域。该模型在 175B 参数上进行训练。它是如此之大，以至于它可以理解许多人类的查询，而不必为此进行明确的训练。它几乎可以做传统 NLP 模型所做的一切:NER、摘要、翻译、分类…等等。

GPT 3 号有几个弱点:

*   它非常昂贵，只能通过 [OpenAI API](https://openai.com/api/) 使用
*   它非常慢(与 spaCy 这样的框架相比)
*   它不一定容易使用

好消息是，EleutherAI(一个人工智能研究人员的集体)在 2021 年发布了开源替代品:GPT-J 和 GPT-NeoX 20B。任何人都可以在任何服务器上部署这些 NLP 模型，但在实践中，这需要一些 MLOps 知识，而且成本可能相当高，这就是为什么我们要使用既经济又易于使用的 NLP 云 API。

## 设置 NLP 云

首先，在 NLP 云上注册。

然后切换到现收现付计划，因为该计划允许您尽可能多地使用 GPT-J，并且前 100k 令牌是免费的，这对于测试本文中的代码来说绰绰有余。

检索您的 API 令牌。

最后，下载他们的 Python 客户端，因为我们将在以下示例中使用它:

```
pip install nlpcloud
```

## 快速工程

假设我们想从一段文本中提取职位名称。

让我们向 GPT·J 提出第一个天真的请求:

```
import nlpcloudclient = nlpcloud.Client("gpt-j", "<your_token>", gpu=True)
generation = client.generation("""Extract job titles from the following sentence: Maxime is a data scientist at Auto Dataset, and he's been working there for 1 year.""")print(generation["generated_text"])
```

输出:

```
Extract job titles from the following sentence: Maxime is a data scientist at Auto Dataset, and he's been working there for 1 year.1\. Maxime works in the data science team at Auto Dataset.
```

如你所见，我们悲惨地失败了！原因是，像 GPT-3 和 GPT-J 这样的生成模型需要在提示中提供几个例子，以便理解你想要什么(也称为“少量学习”)。提示基本上是一段文本，您将在实际请求之前添加它。

让我们用提示中的 3 个例子再试一次:

```
import nlpcloudclient = nlpcloud.Client("gpt-j", "<your_token>", gpu=True)generation = client.generation("""[Text]: Helena Smith founded Core.ai 2 years ago. She is now the CEO and CTO of the company and is building a team of highly skilled developers in machine learning and natural language processing.
        [Position]: CEO and CTO
        ###
        [Text]: Tech Robotics is a robot automation company specialized in AI driven robotization. Its Chief Technology Officer, Max Smith, says a new wave of improvements should be expected for next year.
        [Position]: Chief Technology Officer
        ###
        [Text]: François is a Go developer. He mostly works as a freelancer but is open to any kind of job offering!
        [Position]: Go developer
        ###
        [Text]: Maxime is a data scientist at Auto Dataset, and he's been working there for 1 year.
        [Position]:""",
    max_length=500,
    end_sequence="\n###",
    remove_end_sequence=True,
    remove_input=True)print(generation["generated_text"])
```

输出:

```
Data scientist
```

成功了！这里发生了什么？

首先，在传递我们的实际句子之前，我们在提示中给出了 3 个问题和 3 个回答。3 个例子通常足以教会这些模型你想要达到的目标。

我们还使用了几个参数:

*   `max_length`告诉 API 最多生成 500 个令牌
*   `end_sequence="\n###"`告诉模型一旦到达`\n###`就停止生成文本。这使得文本生成速度更快，因为我们不必等到生成 500 个令牌(也更便宜)。我们知道模型会在结果后面加上`\n###` token 是因为我们在提示里每个结果后面都加了，而 GPT-J 非常擅长学习格式化。
*   `remove_end_sequence`简单地删除响应末尾的`\n###`。
*   `remove_input`从响应中删除输入

用 spaCy 做同样的事情需要一个专门的数据集+大量例子的训练…换句话说，我们只是节省了几周，甚至几个月的人工工作！

## 丰富

我们刚刚用 GPT J 做了第一个实体抽取的概念证明，但是我们当然可以改进它。

我们应该做的第一件事是降低顶部 P 值。Top P 和温度是 GPT 模型的两个常用参数。高值往往会产生更多的原始结果，这不是我们在这里想要的，因为我们希望模型以确定的方式一致地提取相同的实体，而不需要发明任何东西。最好的方法是将 Top P 降低到 0.1。

我们应该关心的第二件事是处理空响应的能力。您的文本不包含任何职位，这种情况确实可能发生，在这种情况下，您希望模型返回类似“none”的内容(例如)。所以你需要明确地告诉模型。

让我们实施这两项改进:

```
import nlpcloudclient = nlpcloud.Client("gpt-j", "<your_token>", gpu=True)
generation = client.generation("""[Text]: Helena Smith founded Core.ai 2 years ago. She is now the CEO and CTO of the company and is building a team of highly skilled developers in machine learning and natural language processing.
        [Position]: CEO and CTO
        ###
        [Text]: Tech Robotics is a robot automation company specialized in AI driven robotization. Its Chief Technology Officer, Max Smith, says a new wave of improvements should be expected for next year.
        [Position]: Chief Technology Officer
        ###
        [Text]: François is a Go developer. He mostly works as a freelancer but is open to any kind of job offering!
        [Position]: Go developer
        ###
        [Text]: The second thing we should care about is the ability to handle empty responses.
        [Position]: none
        ###
        [Text]: Maxime is a data scientist at Auto Dataset, and he's been working there for 1 year.
        [Position]:""",
    max_length=500,
    top_p=0.1,    
    end_sequence="\n###",
    remove_end_sequence=True,
    remove_input=True)print(generation["generated_text"])
```

例如，你可以通过处理多个结果来进一步改进你的提示(例如，从同一段文本中提取几个职位名称)。

## 生产部署

这些 GPT 模型的主要缺点是它们在生产中高效运行的成本很高，并且需要一些高级 DevOps 知识。

例如，你应该忘记在你自己的本地机器上或者甚至在 CPU 服务器上运行 GPT-J。如果你想让它在一个体面的时间(1 秒以下)提取你的实体，你需要使用一个好的 GPU，如英伟达 RTX A6000 或英伟达 A40。

## 结论

一旦你理解了如何处理提示，用 GPT-3 和 GPT-J 进行实体抽取会给出令人印象深刻的结果。

由于不需要更多的注释和培训，我认为这将极大地改变 NER 项目的组织方式。

虽然仍然存在一些基础设施的挑战，但我认为与数据标签的人力成本相比，这些挑战可以忽略不计。

我希望这篇文章有用，它会给你的下一个 NER 项目带来新的想法！