# 你需要知道的关于艾伯特，罗伯塔和迪沃伯特的一切

> 原文：<https://towardsdatascience.com/everything-you-need-to-know-about-albert-roberta-and-distilbert-11a74334b2da>

## 回顾不同 BERT 变形金刚的异同，以及如何从拥抱脸变形金刚库中使用它们

![](img/ad3a83b17f4ecaf1b635da90d0e55e24.png)

内特·雷菲尔德在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

在这篇文章中，我将解释你需要知道的关于 Albert、Roberta 和 Distilbert 的一切。如果你不能从名字上看出来，这些模型都是原始最先进的变压器 BERT 的修改版本。这三个型号，连同伯特，是目前最受欢迎的变形金刚。我将回顾这些模型与 BERT 的不同(和相似)之处，对于每个模型，我将包含代码片段，演示如何使用拥抱面部变形库中的每个模型。注意，这篇文章写于 2022 年 7 月，所以拥抱脸的早期/未来版本可能不起作用。还要注意的是，本文假设您对 BERT transformer 有所了解，所以在阅读本文之前要先了解这一点。然而，我将在本文中快速回顾 BERT，作为一个简短的概述。

## 伯特

BERT——或来自 Transformers 的双向编码器表示——是第一个建立在原始编码器-解码器转换器基础上的转换器，它使用掩蔽语言建模和下一句预测任务的自我监督训练来学习/产生单词的上下文表示。BERT 的核心架构由 12 个编码器模块堆叠而成(来自原始编码器-解码器转换纸)。为了对其他任务进行微调，例如(但不限于)问题回答、摘要和序列分类，BERT 在堆叠编码器的顶部添加了额外的线性层。这些额外的层用于根据 BERT 正在解决的任务生成特定的输出。然而，重要的是要记住，BERT 的原始的、核心的、不可改变的部分是来自堆叠的双向编码器的输出。这些模块使得 BERT 如此强大:通过定制/添加任何特定的层组合，您几乎可以配置 BERT 来解决任何任务。在本文中，我将向您展示这样配置 BERT 的代码。你可以从拥抱脸变形库[这里](/how-to-use-bert-from-the-hugging-face-transformer-library-d373a22b0209)找到如何使用 BERT 的代码。

为了方便起见，下面是如何使用 BERT 完成任何通用任务的代码片段:

```
from transformers import BertModel
class Bert_Model(nn.Module):
   def __init__(self, class):
       super(Bert_Model, self).__init__()
       self.bert = BertModel.from_pretrained('bert-base-uncased')
       self.out = nn.Linear(self.bert.config.hidden_size, classes)
   def forward(self, input):
       _, output = self.bert(**input)
       out = self.out(output)
       return out
```

上面的代码可以用来构建一个通用的 Pytorch BERT 模型，该模型可以在任何其他未指定的任务上进行微调。正如你所看到的，我没有下载一个已经为特定任务设计的特定的 BERT 模型，比如 BERTForQuestionAnswering 或 BERTForMaksedLM，而是下载了一个未经训练的 BERT 模型，它没有附带任何“头”。相反，我在上面添加了我自己的线性层，然后可以配置为其他任务，这些任务没有列在 HuggingFace transformer 库已经完成的任务中，你可以在这里找到。虽然上面的代码不一定是您想要的，但是您可以浏览 hugging face 提供的模型列表并使用它们的 API。例如，下面是如何为 BERT 建立一个屏蔽语言模型。

```
from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased',    return_dict = True)
text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."
input = tokenizer.encode_plus(text, return_tensors = "pt")
mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
output = model(**input)
logits = output.logits
softmax = F.softmax(logits, dim = -1)
mask_word = softmax[0, mask_index, :]
top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
for token in top_10:
   word = tokenizer.decode([token])
   new_sentence = text.replace(tokenizer.mask_token, word)
   print(new_sentence)
```

屏蔽语言建模本质上是一个“填空任务”，其中模型屏蔽一个标记，并训练自己使用屏蔽标记周围的上下文来准确预测屏蔽标记是什么。在上面的示例中，代码使用 BERT 列出屏蔽令牌的前 10 个候选令牌。你可以在这里阅读[发生的事情的更详细的描述。该代码片段的输出是:](/how-to-use-bert-from-the-hugging-face-transformer-library-d373a22b0209)

```
The capital of France, paris, contains the Eiffel Tower. 
The capital of France, lyon, contains the Eiffel Tower. 
The capital of France, lille, contains the Eiffel Tower. 
The capital of France, toulouse, contains the Eiffel Tower. 
The capital of France, marseille, contains the Eiffel Tower. 
The capital of France, orleans, contains the Eiffel Tower. 
The capital of France, strasbourg, contains the Eiffel Tower. 
The capital of France, nice, contains the Eiffel Tower. 
The capital of France, cannes, contains the Eiffel Tower. 
The capital of France, versailles, contains the Eiffel Tower.
```

## 罗伯塔

罗伯塔是伯特的一个简单但非常受欢迎的替代者/继承者。它主要通过仔细和智能地优化 BERT 的训练超参数来改进 BERT。几个简单明了的变化一起增强了 Roberta 的性能，使它在 BERT 设计解决的几乎所有任务上都优于 BERT。值得注意的一个有趣事实是，在 Roberta 出版的时候，另一个流行的新变形金刚， [XLNet](/how-to-use-xlnet-from-the-hugging-face-transformer-library-ddd0b7c8d0b9) ，也在一篇研究论文中发表/介绍。然而，与 XLNet 不同的是，XLNet 引入的变化比 Roberta 引入的变化更难实现，这只会增加 Roberta 在 AI/NLP 社区中的受欢迎程度。

正如我之前提到的，罗伯塔实际上使用了与伯特相同的架构。然而，与 BERT 不同的是，在预训练期间，它只通过掩蔽语言建模进行预训练(BERT 也通过下一句预测进行预训练)。下面是 Roberta 用来获得更好性能的一些超参数变化。

*   更长的训练时间和更大的训练数据(从 16GB 到 160GB 增加 10 倍)
*   从 256 到 8000 的更大批量和从 30k 到 50k 的更大词汇量
*   使用更长的序列作为输入，但是 Roberta 仍然像 BERT 一样有 512 个标记的最大标记限制
*   动态掩蔽允许每次将序列输入模型时掩蔽模式不同，这与使用相同掩蔽模式的 BERT 相反。

知道如何使用拥抱脸变形库中的 BERT 确实有助于理解 Roberta(以及本文中描述的所有模型)是如何编码的。你可以从拥抱脸变形库[这里](/how-to-use-bert-from-the-hugging-face-transformer-library-d373a22b0209)学习如何使用 BERT。按照那篇文章中的代码，使用拥抱脸中的 Roberta 非常简单。

```
from transformers import RobertaModel
import torch
import torch.nn as nnclass RoBERTa_Model(nn.Module):
  def __init__(self, classes):
    super(RoBERTa_Model, self).__init__()
    self.roberta = RobertaModel.from_pretrained('roberta-base')
    self.out = nn.Linear(self.roberta.config.hidden_size, classes)
    self.sigmoid = nn.Sigmoid()
  def forward(self, input, attention_mask):
    _, output = self.roberta(input, attention_mask = attention_mask)
    out = self.sigmoid(self.out(output))
    return out
```

上面的代码展示了如何构建一个通用的 Roberta Pytorch 模型。如果您将其与基于 BERT 的模型的代码进行比较，您确实可以看到我们实际上只是用 Roberta 替换了 BERT！这确实有道理——毕竟，罗伯塔更像伯特，但受过更好的训练。但是你很快就会发现，阿尔伯特和迪翁伯特也是如此。因为这些模型都是 BERT 的修改版本，所以拥抱脸代码的工作方式是，使用任何模型时，您只需从上面获取 BERT 代码，然后用 Roberta 替换所有 BERT 术语(即，改为导入 Roberta 模型，使用正确的模型 id“Roberta-base”，并导入正确的 Roberta tokenizer)。因此，如果你想用 Roberta 做屏蔽语言建模、抽取式问题回答或其他任何事情，你可以使用上面的 BERT 代码或这里的并直接用 Roberta、Distilbert 或 Albert(你想用哪个就用哪个)替换 BERT 术语。

## 蒸馏啤酒

Distilbert 的目标是通过减小 bert 的大小和提高 BERT 的速度来优化训练，同时尽可能地保持最佳性能。具体来说，Distilbert 比最初的 BERT-base 模型小 40%，比它快 60%，并且保留了 97%的功能。

Distilbert 是如何做到这一点的？它使用与 BERT 大致相同的通用架构，但只有 6 个编码器模块(回想一下，BERT base 有 12 个)。这些编码器块也通过仅从每 2 个预训练的 BERT 编码器块中取出 1 个来初始化。此外，BERT 的令牌类型嵌入和池功能也从 Distilbert 中移除。

与 bert 不同，Distilbert 仅使用掩蔽语言建模进行预训练(回想一下，BERT 是使用 MLM 和下一句预测进行训练的)。使用三重损失/三重损失函数训练 Distilbert:

*   伯特使用的相同的语言模型损失
*   蒸馏损失衡量蒸馏器和 bert 之间输出的相似性。
*   余弦距离损失衡量蒸馏伯特和伯特的隐藏状态有多相似。

这些损失函数的组合模拟了 Distilbert 和 bert 之间的学生-教师学习关系。Distilbert 还使用了几个与 Roberta 相同的超参数，比如更大的批量，动态屏蔽，以及我之前提到的，没有对下一句预测进行预训练。从上面看 Roberta(和 BERT)的代码，使用拥抱脸的 Distilbert 非常容易。

```
from transformers import DistilBertModel
import torch
import torch.nn as nnclass DistilBERT_Model(nn.Module):
 def __init__(self, classes):
   super(DistilBERT_Model, self).__init__()
   self.distilbert = DistilBertModel.from_pretrained('distilbert
                                                     base-uncased')
   self.out = nn.Linear(self.distilbert.config.hidden_size, classes)
   self.sigmoid = nn.Sigmoid()
 def forward(self, input, attention_mask):
   _, output = self.distilbert(input, attention_mask 
                                      = attention_mask)
   out = self.sigmoid(self.out(output))
   return out
```

## 艾伯特

Albert 与 Distilbert 大约在同一时间出版/推出，也有一些与论文中介绍的相同的动机。就像 Distilbert 一样，Albert 减少了 bert 的模型大小(参数减少了 18 倍)，训练速度也提高了 1.7 倍。然而，与 Distilbert 不同的是，Albert 在性能上没有折衷(Distilbert 在性能上确实有轻微的折衷)。这来自于 Distilbert 和 Albert 实验构造方式的核心差异。Distilbert 的训练方式是将 bert 作为其训练/蒸馏过程的老师。另一方面，艾伯特和伯特一样是从零开始训练的。更好的是，Albert 优于所有以前的模型，包括 bert、Roberta、Distilbert 和 XLNet。

Albert 能够通过这些参数缩减技术获得较小模型架构的结果:

*   因式分解的嵌入参数化:为了确保隐藏层的大小和嵌入维度是不同的，Alberta 将嵌入矩阵解构为 2 块。这允许它实质上增加隐藏层的大小，而不真正修改实际的嵌入尺寸。在分解嵌入矩阵之后，在嵌入阶段完成之后，Alberta 将线性层/全连接层添加到嵌入矩阵上，并且这映射/确保嵌入维度的维度是相同正确的。你可以在这里阅读更多关于这个[的内容。](https://tungmphung.com/a-review-of-pre-trained-language-models-from-bert-roberta-to-electra-deberta-bigbird-and-more/#albert)
*   跨层参数共享:回想一下，BERT 和 Alberta 各有 12 个编码器模块。在阿尔伯塔省，这些编码器块共享所有参数。这将参数大小减少了 12 倍，并且还增加了模型的正则化(正则化是在校准用于防止过拟合/欠拟合的 ML 模型时的一种技术)
*   Alberta 删除了辍学层:辍学层是一种技术，其中随机选择的神经元在训练过程中被忽略。这意味着他们不再被训练，基本上暂时无用。

```
from transformers import AlbertModel
import torch
import torch.nn as nnclass ALBERT_Model(nn.Module):
 def __init__(self, classes):
   super(ALBERT_Model, self).__init__()
   self.albert = AlbertModel.from_pretrained('albert-base-v2')
   self.out = nn.Linear(self.albert.config.hidden_size, classes)
   self.sigmoid = nn.Sigmoid()
 def forward(self, input, attention_mask): 
   _, output = self.albert(input, attention_mask = attention_mask)
   out = self.sigmoid(self.out(output))
   return out
```

## 其他类似变压器

虽然 Albert、Roberta 和 Distilbert 可能是最受欢迎的三种变形金刚(bert 的所有修改/版本/改进),但其他几种受欢迎的变形金刚也实现了类似的一流性能。这些包括但不限于 XLNet、BART 和 Mobile-BERT。 [XLNet](/how-to-use-xlnet-from-the-hugging-face-transformer-library-ddd0b7c8d0b9) 是一个自回归语言模型，建立在 Transformer-XL 模型的基础上，使用[置换语言建模](/permutative-language-modeling-explained-9a7743d979b4)来实现与 Roberta 类似的最先进的结果。Mobile-BERT 类似于 DistilBERT:它主要是为速度和效率而设计的。与 BERT-base 相比，它的体积小 4.3 倍，速度快 5.5 倍，但性能相当/相似。BART 是另一个预训练的模型，在 NLU(自然语言理解)任务上取得了与 Roberta 相似的性能。除此之外，BART 还可以在 NLG(自然语言生成)任务上表现出色，如抽象摘要，这就是它的独特之处。你可以在这里阅读更多关于他们[的信息。](https://tungmphung.com/a-review-of-pre-trained-language-models-from-bert-roberta-to-electra-deberta-bigbird-and-more/#albert)

我希望您觉得这些内容很容易理解。如果你认为我需要进一步阐述或澄清什么，请在下面留言。

## 参考

预训练语言模型回顾:从 BERT、RoBERTa 到 ELECTRA、DeBERTa、BigBird 等等:[https://tungmphung . com/a-review-of-pre-trained-language-models-from-BERT-RoBERTa-to-ELECTRA-DeBERTa-big bird-and-more/# distill BERT](https://tungmphung.com/a-review-of-pre-trained-language-models-from-bert-roberta-to-electra-deberta-bigbird-and-more/#distilbert)

伯特:用于语言理解的深度双向转换器的预训练:[https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

ALBERT:一个用于语言表征自我监督学习的 Lite BERT:[https://ai . Google blog . com/2019/12/ALBERT-Lite-BERT-for-Self-Supervised . html](https://ai.googleblog.com/2019/12/albert-lite-bert-for-self-supervised.html)

蒸馏伯特，伯特的蒸馏版本:更小、更快、更便宜、更轻:[https://arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)

抱抱脸变形金刚库:[https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)