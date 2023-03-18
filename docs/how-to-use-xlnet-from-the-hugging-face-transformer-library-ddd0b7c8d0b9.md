# 如何使用来自拥抱脸变压器库的 XLNET

> 原文：<https://towardsdatascience.com/how-to-use-xlnet-from-the-hugging-face-transformer-library-ddd0b7c8d0b9>

## 如何使用 XLNET 从拥抱脸变压器库的三个重要任务

![](img/972e842f48116d5dac765f8b1a144df1.png)

照片由 [Ahmed Rizkhaan](https://unsplash.com/@ahmed_rizkhaan?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

在本文中，我将演示如何使用 XLNET 通过拥抱脸变压器库完成三项重要任务。我还将展示如何配置 XLNET，这样，除了它被设计用来解决的标准任务之外，您还可以将它用于您想要的任何任务。

请注意，本文写于 2022 年 4 月，因此拥抱脸库的早期/未来版本可能会有所不同，本文中的代码可能无法工作。

## XLNet 快速回顾

XLNET 是一个通用的自回归模型，它使用置换语言建模来创建单词的双向上下文化表示。值得注意的是，它建立在 BERT transformer 的弱点之上，并在许多任务上优于 BERT，如问题回答、情感分析等。虽然 BERT 是一个非常强大和通用的转换器，但它的架构固有地具有两个弱点。首先，因为它使用屏蔽语言建模来生成单词的上下文化表示，所以它扭曲了输入，所以 BERT 真正使用屏蔽单词的方式是未知的。第二，当 BERT 屏蔽一个句子中的多个标记时，它不能捕获两个被屏蔽的标记之间的依赖关系，这两个标记可能拥有彼此的重要信息。XLNET 相对于 BERT 的另一个主要的、强大的优势是，与具有 512 个令牌输入限制的 BERT 不同，XLNET 是少数几个没有序列长度限制的模型之一。

XLNET 通过置换语言建模捕获单词周围的双向上下文来克服这些问题。在不屏蔽任何单词或改变输入的情况下，置换语言建模通过对句子中所有可能的单词置换训练自回归模型来捕获上下文。它最大化了一个句子所有排列的对数似然，因此，文本中的每个标记都学会了利用句子中所有其他标记的上下文信息，从而创建了强大、丰富的单词表示。

XLNet 可以解决许多任务，但是我将在本文中讨论的是多项选择问题回答、抽取问题回答和语言建模。我还将演示如何配置 XLNET 来完成除上述任务和拥抱脸提供的任务之外的任何任务。

请注意，对于我在本文中展示的所有代码/模型，我都是直接从 Hugging Face transformer 库中获取的，没有任何微调/培训。与许多其他多功能变压器一样，XLNET 在核心自回归模型的基础上增加了一个线性层，以针对特定任务进行自我微调。虽然拥抱脸确实为核心模型提供了预训练的权重，但它不为顶部的线性层提供权重。为了实现每个特定任务的最佳性能，必须针对正在解决的任务训练该线性层，因此本文中代码的结果可能不会很好。

## 多项选择问题回答

选择题答题简单来说就是顾名思义。我没有把这个题目叫做“问题回答”的唯一原因是因为问题回答的另一个版本:摘录问题回答。在抽取式问题回答中，该模型试图在上下文段落/文本中找到答案，而不是像选择题一样在几个答案选项中进行选择。

在运行下面的代码之前，您必须确保运行该代码，以导入代码编译所需的库。

```
pip install transformers 
pip install sentencepiece
pip install torch
## All of these lines can vary depending on what version of 
## each library you use
```

下面是我做选择题回答的代码:

```
from transformers import XLNetTokenizer, XLNetForMultipleChoice
from torch.nn import functional as F
import torch
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
model = XLNetForMultipleChoice.from_pretrained("xlnet-base-cased", return_dict = True)
prompt = "What is the capital of France?"
answers = ["Paris", "London", "Lyon", "Berlin"]
encoding = tokenizer([prompt, prompt, prompt, prompt], answers, return_tensors="pt", padding = True)
outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}) 
logits = outputs.logits
softmax = F.softmax(logits, dim = -1)
index = torch.argmax(softmax, dim = -1)
print("The correct answer is", answers[index])
```

拥抱脸是这样设置的，对于它有预训练模型的任务，你必须下载/导入那个特定的模型。在这种情况下，我们必须下载用于多项选择问题回答模型的 XLNET，而标记器对于所有不同的 XLNET 模型都是相同的。

我们首先对问题和 4 个答案选项进行编码。多项选择的工作方式非常简单:该模型计算每个答案选项的得分，对这些得分进行软最大化以获得概率分布，并只取最高值(张量中最高值的索引使用 torch.argmax 找到)。在 softmax 函数应用于 XLNET 的输出之前，logits 是 XLNET 模型的输出。通过将 softmax 应用于输出逻辑，我们可以获得每个答案选项的概率分布:具有较高概率的答案选项意味着它们是问题的更好/最佳答案。我们可以使用 torch.argmax 检索具有最高概率值的答案的索引。如果您想知道每个答案选项的每个概率值是多少(即模型如何对每个选项进行评级)，您可以简单地打印出 softmax 值的张量。在我的例子中，这是它打印的内容(记住，这个模型顶部的线性层没有经过训练，所以值不好)。

```
tensor([[0.2661, 0.2346, 0.2468, 0.2525]])
```

在这种情况下，模型正确地预测答案是巴黎。但是，您可以看到 softmax 值非常接近。HuggingFace 提供了能够处理问题和答案的基础、预训练的架构，以及顶部的未训练的线性分类器来创建适当的输出。从这些值来看，很明显模型需要训练才能达到好的结果。

## 抽取式问题回答

抽取式问题回答是在给定一些上下文文本的情况下，通过输出答案在上下文中所处位置的开始和结束索引来回答问题的任务。以下是我使用 XLNET 回答问题的代码:

```
from transformers import XLNetTokenizer 
from transformers import XLNetForQuestionAnsweringSimple
from torch.nn import functional as F
import torch
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
model = XLNetForQuestionAnsweringSimple.from_pretrained("xlnet-base-cased",return_dict = True)
question = "How many continents are there in the world?"
text = "There are 7 continents in the world."
inputs = tokenizer.encode_plus(question, text, return_tensors='pt')
output = model(**inputs)
start_max = torch.argmax(F.softmax(output.start_logits, dim = -1))
end_max = torch.argmax(F.softmax(output.end_logits, dim=-1)) + 1 
## add one because of python list indexing
answer = tokenizer.decode(inputs["input_ids"][0][start_max : end_max])
print(answer)
```

像多项选择问题回答一样，我们首先下载用于问题回答的特定 XLNET 模型，并标记我们的两个输入:问题和上下文。HuggingFace 提供了两个 XLNET 模型用于抽取式问题回答:用于简单问题回答的 XLNET 和用于问题回答的普通 XLNET。你可以在官方的 HuggingFace transformer 库页面上了解更多关于这两者的信息。提取性问题回答的过程与多项选择略有不同。抽取式问题回答的工作方式是通过计算答案在上下文中所处位置的最佳开始和结束索引。该模型返回上下文/输入中所有单词的分数，该分数对应于它们对于给定问题的起始值和结束值有多好；换句话说，输入中的每个单词接收表示它们是答案的好的开始单词还是答案的好的结束单词的开始和结束索引分数/值。然后，我们计算这些分数的 softmax 以找到值的概率分布，使用 torch.argmax()检索开始和结束张量的最高值，并在输入中找到对应于这个 start : end 范围的实际标记，解码它们并打印出来。

## 语言建模

语言建模的任务是在给定句子中所有单词的情况下，预测跟随/继续句子的最佳单词。

```
from transformers import XLNetTokenizer, XLNetLMHeadModel
from torch.nn import functional as F
import torch
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased', return_dict = True)
text = "The sky is very clear at " + tokenizer.mask_token
input = tokenizer.encode_plus(text, return_tensors = "pt")
output = model(**input).logits[:, -1, :]
softmax = F.softmax(output, dim = -1)
index = torch.argmax(softmax, dim = -1)
x = tokenizer.decode(index)
print(x)
new_sentence = text.replace(tokenizer.mask_token, x)
print(new_sentence)
```

我们首先下载用于语言建模的特定 XLNET 模型，并标记我们的输入:不完整的句子(该句子必须像我上面所做的那样将掩码标记连接到句子的末尾)。代码相对简单:我们必须检索模型的逻辑，使用-1 索引取最后一个隐藏状态的逻辑(因为这对应于句子中的最后一个单词)，计算这些逻辑的 softmax(在这种情况下，softmax 创建 XLNET 词汇表中所有单词的概率分布；具有较高概率值的单词将是掩码标记的更好的候选替换单词)，找到词汇表中的最大概率值，并解码和打印该标记。在上面的代码中，我正在检索具有最高概率值的单词(即最佳候选单词)，但是如果您想知道前 10 个候选单词是什么(可以是前 10 个或您喜欢的任何数字)，那么您可以这样做。通过使用 torch.topk()函数而不是 torch.argmax()，可以检索给定张量中的前 k 个值，并且该函数返回包含这些前 k 个值的张量。在此之后，过程与之前相同:迭代张量，解码每个候选单词，并用候选单词替换句子中的掩码标记。下面是执行此操作的代码:

```
from transformers import XLNetTokenizer, XLNetLMHeadModel
from torch.nn import functional as F
import torch
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased', return_dict = True)
text = "The sky is very clear at " + tokenizer.mask_token
mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
input = tokenizer.encode_plus(text, return_tensors = "pt")
output = model(**input).logits
softmax = F.softmax(output, dim = -1)
mask_word = softmax[0, mask_index, :]
top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
for token in top_10:
  word = tokenizer.decode([token])
  new_sentence = text.replace(tokenizer.mask_token, word)
  print(new_sentence)
```

## 使用 XLNET 完成任何任务

尽管问题回答、语言建模和 XLNET 可以解决的其他任务在 NLP 中非常重要，但人们通常希望使用 XLNET 这样的转换器来完成其他独特的任务，尤其是在研究中。他们这样做的方式是通过采用核心，基础 XLNET 模型，然后将他们自己的特定神经网络附加到它(通常是线性层)。然后，他们针对特定的任务，在特定的数据集上对这种架构进行微调。在 Pytorch 中，最好将其设置为 Pytorch 深度学习模型，如下所示:

```
from transformers import XLNetModel
import torch.nn as nn
class XLNet_Model(nn.Module):
  def __init__(self, classes):
    super(XLNet_Model, self).__init__()
    self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
    self.out = nn.Linear(self.xlnet.config.hidden_size, classes)
  def forward(self, input):
    outputs = self.xlnet(**input)
    out = self.out(outputs.last_hidden_state)
    return out
```

我没有下载已经为特定任务(如问答)设计的特定 XLNET 模型，而是下载了基本的、预训练的 XLNET 模型，并为其添加了一个线性层。要获取 XLNET 模型的原始核心输出，请使用 xlnet.config.hidden_size(实际值为 768)并将其附加到您希望线性图层输出的类的数量。

我希望您觉得这些内容很容易理解。如果你认为我需要进一步阐述或澄清什么，请在下面留言。

# 参考

[拥抱变脸库](https://huggingface.co/transformers/index.html)