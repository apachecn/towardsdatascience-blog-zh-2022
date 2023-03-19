# 使用 PyTorch 构建问答应用程序

> 原文：<https://towardsdatascience.com/build-a-q-a-app-with-pytorch-cb599480e29>

## 如何使用 Docker 和 FastAPI 轻松部署 QA HuggingFace 模型

![](img/195f116ecba7a340e2609eff5ba770ee.png)

照片由 Pexels 的 [pixabay](https://www.pexels.com/@pixabay) 拍摄。

# 目录

1.  介绍
2.  定义搜索上下文数据集
3.  构建质量保证嵌入模型
4.  使用 Docker 和 FastAPI 部署模型
5.  结论
6.  参考

# 介绍

在过去的几年里，从计算机视觉到自然语言处理，大量的预训练模型已经可用，其中一些最知名的聚合器是 [Model Zoo](https://modelzoo.co/) 、 [Tensorflow Hub](https://tfhub.dev/) 和 [HuggingFace](https://huggingface.co/models) 。

如此大的一组预训练模型的可用性允许开发人员重用这些模型，而无需花费大量的时间和金钱来训练它们。例如，使用 Tesla V100 实例训练一个 GPT-3 模型将花费超过 460 万美元。

在本帖中，我们将讨论:

1.  如何使用 HuggingFace 上提供的预训练 PyTorch 模型创建问答(QA)模型；
2.  如何使用 Docker 和 FastAPI 部署我们的定制模型？

# 定义搜索上下文数据集

有两种主要类型的 QA 模型。第一种方法将领域特定知识的大型语料库编码到模型中，并基于学习到的知识生成答案。第二种方法利用给定的上下文，从上下文中提取最佳段落/答案。

第二种方法更容易推广到不同的领域，无需重新训练或微调原始模型。因此，在这篇文章中，我们将重点讨论这种方法。

要使用基于上下文的 QA 模型，我们首先需要定义我们的“上下文”。这里，我们将使用*斯坦福问答数据集* 2.0。要下载该数据集，请点击[此处](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)。

下载这个文件后，用 Python 打开它，检查它的结构。

观察到的结构应该类似于下面提供的结构。根据这些数据，我们将关注主题为`Premier League`的`question`和`answers`字段。这将为我们提供具体问题的确切答案。如果您想从上下文段落中提取答案，请查看`context`字段。

```
'data': [
  'topic1': {
    'title': str,
    'paragraphs': [
      'paragraph1':{
        'qas': [
          'qa1':{
            'id': str,
            'is_impossible': bool,
            '**question**': str,
            '**answers**': [
              'answer1':{
                'text': str,
                'answer_start': int
              }
              ...
            ],
          },
          ...        
        ]
      },
      ...
    ],
  'context': str
}
```

为了获得`questions`和`answers`，定义并运行以下函数`get_qa`。这将返回一组 357 对问题和答案。

# 构建质量保证嵌入模型

简而言之，我们的模型将通过比较来自用户的新问题和我们的上下文集中的问题集，然后提取相应的答案来工作。

由于我们无法比较原始格式(文本)的问题，因此在执行任何相似性评估之前，我们需要将上下文问题和来自用户的未知问题转换到一个公共空间。

为此，我们将定义一个新的文本嵌入类，用于将上下文和用户的未知问题从文本转换为数字向量。

## 1.下载预训练的嵌入模型

如“简介”部分所述，从头开始训练一个模型既费时又费钱。因此，让我们使用一个已经在 HuggingFace 上训练好的模型。

将下面的脚本保存到一个名为`download_model.sh`的文件中，在带有`bash download_model.sh`的终端中运行，下载所需的模型文件。

## 2.在本地测试嵌入模型

如果你没有`transformers`包，用 pip 安装它。

```
pip install transformers[torch]
```

然后，在新的笔记本单元格中定义并运行下面的`get_model`函数。如果所有文件都已正确下载，并且所有依赖项都已满足，那么运行起来应该没有问题。

现在让我们在一个上下文问题的样本上运行我们的嵌入模型。为此，请运行以下指令。

上面的脚本应该打印出我们新的`embeddings`向量的形状。

```
Embeddings shape: torch.Size([3, 384]
```

## 3.测试上下文与新问题的相似性

让我们从检查之前的样题开始:

```
[
  'How many club members are there?',
  'How many matches does each team play?',
  'What days are most games played?'
]
```

然后，将最后一条解释为:

```
'Which days have the most events played at?'
```

最后，嵌入我们的新问题，计算`new_embedding`和`embeddings`之间的欧氏距离。

上面的脚本应该输出以下距离，表明我们样本中的最后一个问题确实是距离我们的新问题最近的(最小的距离)。

```
tensor([71.4029, 59.8726, 23.9430])
```

# 使用 Docker 和 FastAPI 部署模型

上一节介绍了定义 QA 搜索模型的所有构件。要使其在生产环境中可用，我们需要:

1.  将前面的函数包装在一个或多个易于使用的类中；
2.  定义一个 app，通过 HTTP 调用需要的类方法；
3.  将整个应用程序和依赖项包装在一个容器中，以便于扩展。

## 1.定义 QA 搜索模型

让我们将前面介绍的概念包装成两个新的类:`QAEmbedder`和`QASearcher`。

`QAEmbedder`将定义如何从磁盘加载模型(`get_model`)并返回给定一组问题(`get_embeddings`)的一组嵌入。注意为了效率`get_embeddings`会一次嵌入一批问题。

`QASearcher`将设置相应问题和答案的上下文(`set_context_qa`)，并将我们上下文中最相似问题的答案返回给用户新的未看到的问题(`get_answers`)。

## 2.定义 FastAPI 应用程序

我们的应用程序应该包含 2 个 POST 端点，一个用于设置上下文(`set_context`)，一个用于获取给定的未知问题的答案(`get_answer`)。

`set_context`端点将接收包含 2 个字段(`questions`和`answers`)的字典，并更新`QASearcher`。

`get_answer`端点将接收具有 1 个字段(`questions`)的字典，并返回具有原始问题(`orig_q`)、上下文中最相似的问题(`best_q`)和相关答案(`best_a`)的字典。

## 3.构建 Docker 容器

最后一步是将我们的应用程序包装在 Docker 容器中，以便更容易地分发和扩展。在我们的 docker 文件中，我们需要:

1.  安装`wget`和所需的 Python 库`transformers`、`uvicorn`和`fastapi`；
2.  从 HuggingFace 下载预先训练好的 QA 模型；
3.  将所有应用程序文件(此处可用[)复制到 Docker 镜像并运行 uvicorn 应用程序。](https://github.com/andreRibeiro1989/medium/tree/main/qa_model/app)

要测试我们的新应用程序，使用以下命令构建并运行 Docker 映像:

```
docker build . -t qamodel &&\
  docker run -p 8000:8000 qamodel
```

如果一切顺利，您应该会收到以下消息:

```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 4.测试正在运行的应用程序

为了测试我们的应用程序，我们可以简单地使用`requests`来设置上下文，然后检索给定的新的看不见的问题的最佳答案。

要测试`set_context`端点，请运行以下脚本:

这应该会返回以下消息:

```
{'message': 'Search context set'}
```

要测试`get_answer`端点，请运行以下脚本:

这将返回以下消息:

```
orig_q : How many teams compete in the Premier League ?
best_q : How many clubs are currently in the Premier League?
best_a : 20

orig_q : When does the Premier League starts and finishes ?
best_q : When does the Premier League have its playing season?
best_a : During the course of a season (from August to May)

orig_q : Who has the highest number of goals in the Premier League ?
best_q : Who has the record for most goals in the Premier League?
best_a : Newcastle United striker Alan Shearer holds the record for most Premier League goals with 260
```

## 完成脚本

如需完整的脚本，请点击以下链接进入我的 GitHub 页面:

<https://github.com/andreRibeiro1989/medium/tree/main/qa_model>  

# 结论

通过利用免费提供的预训练模型，构建强大的计算机视觉和自然语言处理模型正变得越来越容易。

在本文中，我介绍了使用 HuggingFace、Docker 和 FastAPI 构建自己的问答应用程序的基本构件。请注意，这一系列步骤并不特定于问答，但确实可以用于大多数计算机视觉和自然语言处理解决方案。

如果你对以无服务器方式部署这个应用感兴趣，可以看看我以前的文章“[用 Amazon Lambda 和 API Gateway](/build-a-serverless-api-with-amazon-lambda-and-api-gateway-dfd688510436) 构建无服务器 API”⁴.

如果您刚刚开始了解 PyTorch，并希望快速了解它，那么这篇文章可能会让您感兴趣"[py torch 入门](/getting-started-with-pytorch-2819d7aeb87c) "⁵.

[**加入我的邮件列表，我一发布新内容，你就能收到新内容！**](https://andrefsr.medium.com/subscribe)

如果你喜欢阅读这样的故事，并想支持我成为一名作家，可以考虑报名成为一名媒体成员。每月 5 美元，让你可以无限制地访问 Python、机器学习和数据科学文章。如果你使用[我的链接](https://andrefsr.medium.com/membership)注册，我会赚一小笔佣金，不需要你额外付费。

<https://andrefsr.medium.com/membership>  

# 参考

[1]https://lambdalabs.com/blog/demystifying-gpt-3/*李川【OpenAI 的 GPT-3 语言模型:技术概述】*

[2] Pranav Rajpurkar 等著《*小队:机器理解文本的 10 万+问题*》(2016)，arXiv
https://arxiv.org/abs/1606.05250v3

[3] Stanford 问答数据集(SQUAD)——在 [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/legalcode) 许可下发布——可在 https://rajpurkar.github.io/SQuAD-explorer/获得

[4]里贝罗。"*用亚马逊 Lambda 和 API 网关*构建无服务器 API "
https://towardsdatascience . com/Build-a-server less-API-with-Amazon-Lambda-and-API-Gateway-DFD 688510436

[5]里贝罗。"*py torch 入门"* https://towardsdatascience . com/Getting-started-with-py torch-2819 D7 aeb 87 c