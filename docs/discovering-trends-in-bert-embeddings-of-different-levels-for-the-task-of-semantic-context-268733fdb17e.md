# 为语义语境确定任务发现不同层次的 BERT 嵌入趋势

> 原文：<https://towardsdatascience.com/discovering-trends-in-bert-embeddings-of-different-levels-for-the-task-of-semantic-context-268733fdb17e>

![](img/9f695bcd5a6ab0a13edfa5966106c5a0.png)

伊戈尔·沙巴林的照片，经许可

## 如何从 BERT 模型输出中提取句子的上下文信息

谈到 BERT 中的上下文嵌入，我们指的是预训练模型的隐藏状态。然而，首先，BERT 使用从查找表中提取的非上下文的、预训练的(静态)嵌入。它发生在编码器层之前的嵌入层，然后编码器层为正在处理的序列生成隐藏状态。

简而言之，语境嵌入来自非语境，从某种意义上说，它是相邻标记嵌入扩散的产物。

这里讨论的方法可以用于广泛的分类 NLP 任务，如 NSP(下一句预测)，NER(命名实体识别)等。

# 假设

假设令牌的初始非上下文嵌入和随后使用编码器为该相同令牌生成的上下文嵌入之间的差异可以包含关于上下文本身的信息，因此，例如，当您需要在诸如句子蕴涵的任务中为整个句子生成摘要嵌入时，这可能是有用的。

# 直觉

一个记号的嵌入只不过是一个在 n 维空间中表示该记号的数字向量。每个维度都包含该标记的一些语义信息。这个概念可以通过下面的简化例子很容易理解。假设特征是银行、邮政、游戏。然后，单词 Card 可以用下面的三维向量来表示:(0.85，0.6，0.45)。正如您可能猜到的，这是单词卡的非上下文表示，它只显示该单词在特定上下文中出现的概率。现在假设单词 Card 出现在下面的短语中:

```
I sent her a postal card.
```

因此，Card 的上下文表示现在可能如下所示:(0.2，1，0.1)，增加与邮政相关的值，减少与其他类别相关的值。要按特征查看原始向量和新向量之间的变化，您可以用后者除以前者(按元素),使用对数确保平滑近似:

```
c = log(b/a)
```

```
where a — non-cotextual (initial) vector,
b — cotextual vector
```

在这个特定的例子中，上述计算的结果向量将如下所示:

```
(-1.4, 0.5, -1.5)
```

这种新的表现形式让你对每个特征的变化有一个清晰的概念，无论是积极的还是消极的。你可能想知道，为什么我需要这种表示——为什么不用上下文向量来代替呢？其思想是，当绝对值变化不大时，知道特征值(而不是其绝对值)的变化对模型更有帮助。当不清楚在当前环境中哪个特性占主导地位，哪个应该首先被关注时，就会发生这种情况。这可以通过例子得到最好的理解。继续我们的卡片例子，考虑下面的两句话:

```
We’re going to send her a card.
Where is the nearest mailbox?
```

第一句没有明确告诉你指的是哪种卡。然而，这里使用的及物动词 Send 允许您的模型谨慎地猜测是指一张明信片。因此，根据第一句话的上下文，卡的初始嵌入到上下文嵌入的过渡可能如下所示，邮政特征的值增加一点，而所有其他特征的值减少:

```
(0.85, 0.6, 0.45) -> (0.8, 0.75, 0.4)
```

但是，正如您所看到的，Banking 特性的值(首先出现)仍然大于其他值，这将使您的模型在决定上下文时将其视为优先特性。相比之下，考虑特征值的变化会提供更多信息，因为它明确显示了哪些权重增加了，哪些权重减少了:

```
log((0.8, 0.75, 0.4)/(0.85, 0.6, 0.45)) -> (-0.1, 0.2, -0.1)
```

在这个特定的例子中，第二个句子的上下文与邮资明确相关。因此，在句子蕴涵的任务中，所提出的方法将帮助你的模型做出正确的预测。

## 试用 BERT 嵌入

与上面的玩具例子不同，真实模型通常使用几百个维度来嵌入。例如，基本的 BERT 模型使用 768 维空间进行嵌入，其中每个维度都不与明确命名的语义类别相关联。但主要思想是不变的:如果两个嵌入在同一个维度上有很高的值，说明它们对应的词与某一个、同一个语义范畴有联系，比如银行、游戏等。

让我们用简单明了的例子来试验 BERT 嵌入。考虑以下三句话:

```
I want an apple.
I want an orange.
I want an adventure.
```

在上面的每个句子中，我们使用同一个及物动词 Want。但是在前两个句子中，我们使用了直接宾语:苹果和桔子，它们属于同一个语义范畴:水果。第三句用了直接宾语:显然属于另一个范畴的冒险。

```
The purpose of our experiment: 
```

检查直接宾语的语义差异如何影响及物动词的嵌入。
如果有这样的效果，使它更清楚地表达出来。
实现
Google Colab:https://Colab . research . Google . com/drive/1k _ R1 QoS 79 au ws 2 jej 7d 1 mymxhxa d 29 FD？usp =共享

为了获得预训练的 BERT 模型，我们将使用拥抱脸中的变形金刚:

```
pip install transformers
```

我们需要 Bert 记号赋予器和裸露的 BERT 模型转换器:

```
import torch
from transformers import BertTokenizer, BertModel
```

然后，我们可以加载预训练的模型标记器(词汇)和模型:

```
tokenizer = BertTokenizer.from_pretrained(‘bert-base-uncased’)
model = BertModel.from_pretrained(‘bert-base-uncased’,
 output_hidden_states = True, # Making the model return all hidden-states.
 )
```

我们将模型置于“评估”模式。

```
model.eval() 
```

下面我们定义在我们的例子中使用的例句:

```
sents = []
sents.append(‘I want an apple.’)
sents.append(‘I want an orange.’)
sents.append(‘I want an adventure.’)
```

接下来，我们需要标记这些句子:

```
tokenized_text = []
segments_ids = []
tokens_tensor = []
segments_tensors = []
```

```
for i, sent in enumerate(sents):
 tokenized_text.append(tokenizer.encode(sent, add_special_tokens=True))
 segments_ids.append([1] * len(tokenized_text[i]))
 tokens_tensor.append(torch.tensor([tokenized_text[i]]))
 segments_tensors.append(torch.tensor([segments_ids[i]]))
```

之后，我们可以通过 BERT 运行我们的句子，并收集其层中产生的所有隐藏状态:

```
outputs = []
hidden_states = []
with torch.no_grad():
 for i in range(len(tokens_tensor)):
 outputs.append(model(tokens_tensor[i], segments_tensors[i]))
 # we set `output_hidden_states = True`, the third item will be the 
 # hidden states from all layers. See the documentation for more details:
 # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
 hidden_states.append(outputs[i][2])
```

因此，对于每个句子，我们有嵌入层的输出+所有 12 个编码器层的输出。

```
len(hidden_states[0])
13 
```

每层有 768 个维度。

```
len(hidden_states[0][12][0][2])
768
```

为了简单起见，我们将只考虑嵌入的前 10 个值:

```
hidden_states[0][12][0][2][:10].numpy()
array([ 0.44462553, 0.21318859, 1.1400639 , -0.05000957, 0.43685108, 0.91370475, -0.6992555 , 0.13507934, -0.42180806, -0.66882026], dtype=float32)
```

在这个例子的每个句子中，我们只对第二个词(Want)感兴趣。下面我们获得在第 12 编码器层中为 Want 生成的嵌入，并将它们转换成 numpy:

```
l12_1 = hidden_states[0][12][0][2][:10].numpy()
l12_2 = hidden_states[1][12][0][2][:10].numpy()
l12_3 = hidden_states[2][12][0][2][:10].numpy() 
```

将获得的向量相互比较，以找出它们在语义上如何相互接近，这将是有趣的:

```
from scipy import spatial
```

```
1 — spatial.distance.cosine(l12_1, l12_2)
0.9869935512542725
```

```
1 — spatial.distance.cosine(l12_1, l12_3)
0.8980972170829773
```

```
1 — spatial.distance.cosine(l12_2, l12_3)
0.8874450922012329 
```

正如你所看到的(也可能是你所期望的)，前两个句子中 Want 的嵌入比第三个句子中 Want 的嵌入更接近。

现在让我们检查一下，我们是否能更清楚地了解嵌入中的差异，反映出由上下文引起的语义差异。

首先，我们需要获得单词 Want 在每个句子中的初始嵌入:

```
l0_1 = hidden_states[0][0][0][2][:10].numpy()
l0_2 = hidden_states[1][0][0][2][:10].numpy()
l0_3 = hidden_states[2][0][0][2][:10].numpy() 
```

我们现在可以将第 12 编码器层中生成的 Want 嵌入除以相应的初始嵌入:

```
import numpy as np
```

```
l0_12_1 = np.log(l12_1/l0_1)
l0_12_2 = np.log(l12_2/l0_2)
l0_12_3 = np.log(l12_3/l0_3)
Before proceeding we need to replace NaNs to 0s:
```

```
l0_12_1 = np.where(np.isnan(l0_12_1), 0, l0_12_1)
l0_12_2 = np.where(np.isnan(l0_12_2), 0, l0_12_2)
l0_12_3 = np.where(np.isnan(l0_12_3), 0, l0_12_3) 
```

现在让我们计算结果向量之间的距离，以了解这些新的表示是否可以更好地指示基础单词之间的语义差异:

```
1 — spatial.distance.cosine(l0_12_1, l0_12_2)
0.9640171527862549
```

```
1 — spatial.distance.cosine(l0_12_1, l0_12_3)
0.4167512357234955
```

```
1 — spatial.distance.cosine(l0_12_2, l0_12_3)
0.3458264470100403 
```

与先前从第 12 层生成的嵌入词获得的相似性结果相比较，我们可以得出结论，这些新的表示使我们能够更清楚地理解潜在的词如何根据它们所处的上下文而彼此不同。

## 注意力权重选择上下文中最重要的单词

毫无疑问，您已经意识到，一般的想法是，将一个标记的上下文嵌入除以这个相同标记的静态嵌入所得到的向量包括关于整个句子的上下文的信息。前面的例子说明了所提出的方法在应用于句子的及物动词时是如何工作的。问题出现了:它必须总是一个及物动词吗？对于这种分析来说，每句话一个词就足够了吗？

嗯，直观上很清楚，就上下文而言，它应该是一个重要的词。要在特定的句子中选择一个，您可以利用编码器层中生成的注意力权重。代码如下:

在继续之前，让我们先来看看我们要获取注意力权重矩阵的令牌:

```
tokenizer.convert_ids_to_tokens(tokenized_text[0])
[‘[CLS]’, ‘i’, ‘want’, ‘an’, ‘apple’, ‘.’, ‘[SEP]’]
```

这是句子“我想要一个青苹果”的第 12 层注意力权重。

```
outputs[0].attentions[0][0][11].numpy().round(2)
```

```
array([[0.93, 0.02, 0\. , 0.01, 0\. , 0\. , 0.03],
 [0.3 , 0.05, 0.24, 0.07, 0.14, 0.06, 0.15],
 [0.38, 0.41, 0.04, 0.02, 0.06, 0.02, 0.07],
 [0.48, 0.11, 0.16, 0.02, 0.02, 0.04, 0.17],
 [0.07, 0.07, 0.26, 0.27, 0.06, 0.05, 0.23],
 [0.52, 0.05, 0.06, 0.04, 0.07, 0\. , 0.26],
 [0.71, 0.06, 0.03, 0.03, 0.01, 0\. , 0.15]], dtype=float32) 
```

我们按列求和，不包括特殊符号:

```
np.sum(outputs[0].attentions[0][0][11].numpy(), axis=0)[1:-1]
```

```
array([0.7708196 , 0.7982767 , 0.45694995, 0.36948416, 0.17060593],
 dtype=float32) 
```

根据上面所说，第二个单词(Want)是句子中最重要的一个。

# 结论

本文中讨论的方法说明了如何在不同的层次上利用 BERT 模型生成的嵌入。特别是，您已经看到了如何获取并使用一个向量表示，该向量表示包含关于上下文嵌入值相对于同一标记的静态嵌入的变化的信息。这种表示携带了关于该标记所处的上下文的信息，从而为诸如句子蕴涵这样的分类任务打开了大门。