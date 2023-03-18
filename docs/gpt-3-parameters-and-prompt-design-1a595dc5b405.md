# GPT-3 参数和提示设计

> 原文：<https://towardsdatascience.com/gpt-3-parameters-and-prompt-design-1a595dc5b405>

## 想法生成器、指令和目标层；灾难恢复摘要

![](img/936e2b080f757ff7bb9fb7cf4bb84a8d.png)

罗马卡夫在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

GPT-3 是一个被设计用来生成自然语言的计算机系统。它通过接受一段文本，然后预测它后面应该出现的下一个单词或短语来做到这一点。

为了实现这一点，GPT-3 使用了一种深度学习算法，该算法已经在大型文本语料库上进行了训练。这使得它能够捕捉单词在句子中如何一起使用的统计模式。

GPT-3 可用于各种涉及自然语言生成的任务，如机器翻译、问答和文本摘要。GPT-3 甚至可以用来从头生成新的文本，如故事或文章。

在这篇博文中，我们将看看使用模型时可以设置的参数，并浏览 OpenAI 提出的几个[示例](https://beta.openai.com/examples)。

# 参数是什么？

OpenAI GPT-3 是一种机器学习模型，可用于通过 API 生成预测文本。OpenAI 有不同的模型可供我们使用，最有能力的一个叫做“text-davinci-002”。

为了使用该模型，我们可以如下发送 API 请求。

```
curl [https://api.openai.com/v1/completions](https://api.openai.com/v1/completions) \
 -H “Content-Type: application/json” \
 -H “Authorization: Bearer $OPENAI_API_KEY” \
 -d ‘{
 “model”: “text-davinci-002”,
 “prompt”: “What are 5 key points I should know when studying Ancient Rome?”,
 “temperature”: 0.3,
 “max_tokens”: 150,
 “top_p”: 1,
 “frequency_penalty”: 0,
 “presence_penalty”: 0
}'
```

**引擎** 引擎参数指定用于生成预测的 AI 模型。在本文中，我们将使用最有能力的一个:“文本-达芬奇-002”。

**最大令牌数** “最大令牌数”参数指定模型可以生成的最大令牌数。一个令牌可以被看作是一个单词。

> 根据经验，1 个令牌大约有 4 个字符。

**温度** 在映射到概率之前，模型输出非标准化值(logits)。logits 通常与 softmax 等函数一起使用，将它们转换为概率。

但是，在应用 softmax 函数之前，我们可以使用一个受热力学启发的技巧，用温度参数(即 softmax(logits/temperature ))来缩放对数。

接近 1 的温度参数意味着 logits 不加修改地通过 softmax 函数。如果温度接近于零，则与其他表征相比，最可能的表征将变得非常可能，即，模型变得更具确定性，并且将总是在给定的单词序列之后输出相同的表征集。

> 温度在 0 到 1 之间，我们可以控制模型预测的随机性和创造性。

**Top p** Top p 参数指定推断时间内的采样阈值。Top p 采样(有时称为核采样)是一种用于对模型的可能结果进行采样的技术。

为了更好地理解这个概念，我们假设模型要预测句子“我要吃饭”之后的下一个令牌。为简单起见，我们将假设令牌是一个单词，并且模型输出以下概率:

*   胡萝卜(2%的可能性)，
*   黄瓜(1.5%的可能性)，
*   茄子(1%的可能性)，
*   菠菜(0.5%的可能性)，
*   西兰花(0.3%的可能性)，
*   …

这组单词形成概率分布“P(Word |“I want to eat”)”,累积分布函数(CDF)看起来如下:

*   2%用胡萝卜，
*   胡萝卜和黄瓜占 3.5%，
*   胡萝卜、黄瓜和茄子占 4.5%
*   …

如果我们将顶部 p 参数设置为 3.5%，模型将根据可能性在胡萝卜和黄瓜之间进行采样和随机选择。由于 top p 为 4.5%，该模型将在胡萝卜、黄瓜和茄子之间随机选择一个词。

和温度一样，top p 参数控制着模型的随机性和独创性。

> Top p 和温度参数都控制模型的随机性。OpenAI 文档建议使用其中一个参数，并将未使用的参数设置为中性，即 1.0。

**频率惩罚** 频率惩罚参数控制模型重复预测的趋势。频率损失降低了已经生成的单词的概率。惩罚取决于一个单词在预测中已经出现了多少次。

**存在惩罚** 存在惩罚参数鼓励模型做出新颖的预测。如果单词已经出现在预测文本中，则存在惩罚降低了该单词的概率。与频率惩罚不同，存在惩罚不依赖于单词在过去的预测中出现的频率。

# **语法纠正**

```
Engine: text-davinci-002Max tokens: 60Temperature: 0Top p: 1.0Frequency penalty: 0.0Presence penalty: 0.0
```

科技一直是作家和语法学生的福音。过去，我们有拼写检查器和语法检查器来帮助我们发现错误。但是现在，即使是 GPT-3 也能做这项工作并捕捉错误。

这对于那些母语不是英语的人或者在语法上有困难的人来说特别有帮助。

**提示**

```
Correct this to standard English: I’m looking forward to see you
```

**响应**

```
I’m looking forward to seeing you.
```

所以，如果你正在寻找一种方法来提高你的写作水平，试试 GPT-3 版的语法修正吧。你可能会惊讶于它的帮助有多大。

# **创意生成器**

```
Engine: text-davinci-002Max tokens: 150Temperature: 0.6Top p: 1.0Frequency penalty: 1Presence penalty: 1
```

如果你正在努力想出新的点子，你可能也想考虑使用 GPT-3。

例如，你可以告诉人工智能想出对环境友好的新产品的想法。然后人工智能会生成一个潜在想法的列表供你选择。

**提示**

```
come up with ideas for a new product that is environmentally friendly:
```

**响应**

```
1\. A reusable water bottle that can be refilled from the tap. 2\. A reusable shopping bag made from recycled materials. 3\. A solar-powered cell phone charger. 4\. An LED light bulb that lasts for years and consumes very little energy.
```

你也可以使用 GPT-3 来寻找你所面临的问题的解决方案。例如，如果你试图增加你商店的销售额，人工智能可以生成一系列营销想法供你考虑。

**提示**

```
brainstorm solutions to increase sales at your store:
```

**回应**

```
1\. Offer discounts or coupons. 2\. Host a sale or clearance event. 3\. Create an attractive window display. 4\. Place items on end caps or near the register. 5\. Train staff to upsell and cross-sell products. 6\. Implement a loyalty program. 7\. Send out direct mailers or emails. 8\. Place ads in local publications. 9\. Sponsor a community event. 10\. Get involved with social media marketing
```

因此，如果你正在寻找一种创造性的方法来解决问题，GPT-3 可能正是你所需要的。

# **TL；灾难恢复总结**

```
Engine: text-davinci-002Max tokens: 60Temperature: 0.7Top p: 1.0Frequency penalty: 0.0Presence penalty: 0.0
```

任何花时间上网的人都知道，网上有很多文本。有时候，很难从头到尾看完。

用 TL；博士，你可以很快得到一篇文章的要点，而不必通读全文。现在，GPT-3 可以为您完成这项工作。

只需将您想要总结的文本输入到工具中，它就会尽力为您提供一个简明的概述。

**提示**

```
Technology has always been a boon for writers and students struggling with grammar. In the past, we had spell checkers and grammar checkers to help us catch mistakes. But now, even GPT-3 can do the work and catch errors.This can be especially helpful for those who are non-native English speakers or who have difficulty with grammar.Tl;dr
```

**响应**

```
If you’re struggling with your grammar, consider using an AI-powered grammar checker to help you improve your writing.
```

# 我们能给出额外的指示吗？

这一部分可能是最吸引人的。我们可以通过在提示中添加上下文来“指导”算法如何响应请求。

**提示**

```
my company produces reusable water bottles that can be refilled from the tap.brainstorm solutions to increase sales at your store: 
```

**响应**

```
1\. Offer a discount to customers who purchase your reusable water bottles. 2\. Place your water bottles in high-traffic areas of your store so that more people will see them. 3\. Educate your staff about the benefits of using reusable water bottles and encourage them to use them themselves. This will create a positive example for customers. 4\. Make sure that your water bottles are easy to find and accessible in your store so that customers can easily grab one when they need it.
```

> 我们添加到提示中的上下文直接反映在响应中。

当模型根据提示中给出的令牌预测下一个令牌时，这种行为是意料之中的。

有时候一句话就能改变整个反应。例如，如果我们使用单词“summary”而不是“TL ”,我们可以得到更长的响应；博士”:

**提示**

```
Technology has always been a boon for writers and students struggling with grammar. In the past, we had spell checkers and grammar checkers to help us catch mistakes. But now, even GPT-3 can do the work and catch errors. This can be especially helpful for those who are non-native English speakers or who have difficulty with grammar. Summary: 
```

**响应**

```
Technology has always been a boon for writers and students struggling with grammar. In the past, we had spell checkers and grammar checkers to help us catch mistakes. But now, even GPT-3 can do the work and catch errors.
```

# **结论**

GPT-3 计算机系统被设计成通过预测下一个单词或短语来生成自然语言。这种深度学习算法已经在一大组文本上训练过，允许它捕捉单词在一个句子中如何一起使用的统计模式。

GPT-3 是一个优秀的工具，我们可以用它来提出新的想法，纠正语法错误，甚至生成文本。为了最大限度地利用模型，我们可以玩参数，但我们也可以以这样一种方式设计提示，它将人工智能引向所需的响应。

感谢阅读！

好奇想了解更多关于 Anthony 的工作和项目吗？在[媒体](https://medium.com/@anthonycvn)、 [LinkedIn](https://www.linkedin.com/in/anthonycavin/) 和 [Twitter](https://twitter.com/Anthony66333223) 上关注他

*需要技术写手？将您的请求发送到*[*https://amigo CCI . io*](https://amigocci.io/blog/mlops-at-medium-scale/)*。*