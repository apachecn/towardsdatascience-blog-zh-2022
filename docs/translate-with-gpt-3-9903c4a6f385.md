# 用 GPT-3 翻译

> 原文：<https://towardsdatascience.com/translate-with-gpt-3-9903c4a6f385>

## 机器翻译，但没有机器翻译系统

![](img/17d2bd33d9b6777be47484c306928b57.png)

图片来自 [Pixabay](https://pixabay.com/illustrations/translate-translation-web-service-3324171/)

在翻译方面，OpenAI GPT-3 已经被证明可以与最先进的机器翻译系统相媲美([布朗等人，(2020)](https://arxiv.org/abs/2005.14165) )。

GPT-3 只需要几个翻译例子就能学会如何合理地翻译。对于没有太多训练数据可用的语言和领域的翻译，它是标准机器翻译系统的合适替代。

自从 GPT-3 主要在英语文本上训练以来，表现令人印象深刻。

在这篇文章中，我将描述如何通过不同的提示策略来充分利用 GPT-3 进行翻译。我将关注那些我们只有几个或零个翻译例子的用例。

使用 GPT-3 不是免费的，翻译的成本估算也不是直截了当的，所以我也将为您提供一种自己估算翻译成本的方法。

本文不需要任何机器学习的先验知识。

# 提示 GPT-3

GPT-3 没有被训练用于任何特定的自然语言处理任务。

然而，如果你用自然语言向 GPT 3 号提交一份任务描述，GPT 3 号会尝试去做，成功与否取决于任务的复杂程度。

例如，如果你想让 GPT-3 把一些英语文本翻译成法语，你必须用纯文本描述这个翻译任务，并把这个描述和要翻译的文本一起提交给 GPT-3。

集合{任务描述，文本}被称为*提示*。对于翻译任务，提交给 GPT-3 的提示可以是:

```
**Prompt:**

Translate from English to French:
The cat is in the kitchen. =>
```

然后，理想情况下，GPT 3 号会回应:

```
**Response:**

Le chat est dans la cuisine.
```

在这种情况下，翻译对 GPT 三号来说是一个*零射击*任务。我们要求 GPT 3 号执行一项未经训练的任务。

零触发配置特别适用于我们没有任何可用训练数据的用例**。你可以把它看成一个无人监督的机器翻译任务。**

为了改善结果，我们可以修改提示以包含任务的示例，例如:

```
**Prompt:** 
Translate from English to French:
I am in the kitchen. => Je suis dans la cuisine.
The cat is in the kitchen. =>
```

在这种情况下，GPT-3 已经看到了一个翻译示例。对 GPT 3 号来说，这变成了一个一次性的任务。如果你提供了一个以上的例子，我们通常会谈到一个*少镜头*任务。

单镜头和少镜头配置对于机器翻译任务很有用，我们只有几个翻译的例子**。对于某些语言来说，使用 GPT-3 的少量翻译可以与最先进的机器翻译系统表现一样好，甚至更好。**

# 选择模型和估算翻译成本

![](img/112323ba130d57ac85007c3209f3f64d.png)

图片来自 [Pixabay](https://pixabay.com/photos/coins-money-profit-savings-stack-18134/)

OpenAI 已经建立了一个[网页，上面详细介绍了型号和价格](https://openai.com/api/pricing/)。

给出了**1000 个代币**和每个型号的成本:

*   语言
*   巴贝奇
*   居里
*   达芬奇

*注:如果你很好奇这些模型在自然语言处理中的表现如何，* [*EleutherAI 在各种任务*](https://blog.eleuther.ai/) *(但不是翻译)中对它们进行了评测。*

所有这些模型都可以进行翻译，但便宜的模型不会给你带来好的结果[林等人(2022)](https://arxiv.org/pdf/2112.10668.pdf) 。

**居里**或**达芬奇**很可能给你最好的结果。

请注意，成本是给定的令牌，即，不是单词。OpenAI 估计 1 个 token = 0.75 个单词。因此，如果你有一篇 1000 字的文本要翻译，GPT-3 将首先将其标记为大约**1250 个标记**。

如果你使用 Curie，仅仅是**提交**1000 字，你就要支付大约 0.0025 美元(1.25*0.002)。这是为英语课文准备的。预计其他语言的单词/单词比率会更高，因为 GPT-3 可能会将原始训练数据中较少出现的语言的单词分成更小的片段。

这是你提交给 GPT-3 的费用。不幸的是，这并不是翻译本身的成本。

记住，我们必须向 GPT 3 号描述我们想做什么。除了要翻译的文本，我们还必须描述任务。OpenAI 对整个提示收费。

例如，如果你做零镜头翻译，你将支付任务描述“从英语翻译成法语:”为你提交给 GPT 三号的每个请求。如果你做几个镜头的翻译，你要加上提交你提示中给出的所有翻译实例的费用。正如我们将在本文的以下部分看到的，**如果你不仔细设计你的提示，获得最好的 GPT-3 的成本会迅速上升。**

然后，OpenAI 还对生成的令牌进行收费。生成的令牌数量无法预测，但有一个选项可以限制它。

概括地说，发送到 GPT-3 的查询的成本将根据以下令牌数来确定:

*   任务的描述
*   提供的翻译示例(如果有)
*   要翻译的文本
*   生成的翻译

请注意，如果您在提示中提供了几个翻译示例，可以跳过“任务描述”。

使用最好的模型 Davinci，GPT-3 可以处理多达 4000 个令牌(或使用 Curie 处理 2048 个令牌)。如果提示符的长度加上 GPT-3 的最大响应长度，就令牌而言，大于模型可以处理的长度，GPT-3 将返回一个错误，要求减小提示符的大小(或更改一些参数)。这个限制也意味着你不能一次提交大量的文本。

# 设置 API

*注意:如果你不习惯用 Python 编码，你可以跳过这一部分，使用 OpenAI 提供的“操场”网络界面，一旦你在他们的网站上注册，就可以使用这个界面。*

我将使用 API 做一些 GPT-3 翻译的演示。如果你也想在你这边练习，你将需要创建一个 OpenAI 帐户。你将获得免费学分(在我写这篇文章的时候是 18 学分)来和 GPT-3 一起练习。

您还需要安装 openai 包:

```
pip install openai
```

要使用 OpenAI 的 API，你需要一个 API 密匙。您可以在您的 OpenAI 帐户中生成一个。**保持这个密钥私有**。

然后，我将我的请求发送给 API，如下面的 Python 脚本所示:

```
import os
import openai

#Your API key is loaded here
#It should be exported as an environment variable before running this script: export OPENAI_API_KEY=your_key
openai.api_key = os.getenv("OPENAI_API_KEY")

#Description of the task in natural language
task_description = "Translate English to French:"
#The text you want to translate
to_translate = "The cat is in the kitchen =>"

#The prompt is created by concatenating the task description and the text to translate
prompt = task_description+"\n"+to_translate

#The API call
#respone will contain the generated translation
response = openai.Completion.create(
  model="text-curie-001",
  prompt=prompt,
  temperature=0.3,
  max_tokens=2048,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

#Print the JSON response from GPT-3
print(response)
```

更准确的说，对 API 的调用是用 openai.Completion.create 完成的，你可以在 openai 的网站上找到 [API 文档](https://beta.openai.com/docs/api-reference/completions/create)来更好的理解参数。在本文中，除了“提示”之外，我不会修改它们。

GPT-3 给出的响应采用 JSON 格式，如下所示:

```
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "text": " Le chat est dans la cuisine"
    }
  ],
  "created": 1668449187,
  "id": "xxxx",
  "model": "text-curie-001",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 7,
    "prompt_tokens": 14,
    "total_tokens": 21
  }
}
```

这里最重要的关键字是“文本”，其中包含由 GPT-3 生成的翻译，以及“total_tokens”，这是您将被计费的令牌数。

这个 API 调用花费$0.000042 (0.002*21/1000)。

# 翻译设置

![](img/31875e45153ca18cb488fc70eefed0fb.png)

图片来自 [Pixabay](https://pixabay.com/illustrations/direction-signpost-path-decision-1033278/)

对于这个演示，我将使用由 TILDE ( [CC-BY](https://creativecommons.org/licenses/by/4.0/) )编译的 [ECB 数据集。](https://tilde-model.s3-eu-west-1.amazonaws.com/Tilde_MODEL_Corpus.html)

我将把以下 6 句话从英语翻译成法语(我的母语，所以我可以手动评估译文):

```
Today also marks the opening of the online "Euro Run" competition for children aged 9 to 12 living in the euro area.
Unlike Denmark and the United Kingdom, Croatia does not have the right to opt out of adopting the single currency.
When the number of NCB Governors exceeds 15 they will exercise a voting right on the basis of a rotation system.
Write-downs amounted to €64 million in 2015 (2014: €8 million).
€500 banknote remains legal tender and will always retain its value
While there are fewer than 22 euro area Member States the rotation will take place in accordance with the following rules:
```

请注意，ECB 中的句子不是连续的，也就是说，它们应该独立考虑。

由人工翻译产生的法语参考译文如下:

```
C'est aujourd'hui également qu'est lancé le concours en ligne Euro Run, destiné aux enfants âgés de 9 à 12 ans habitant dans la zone euro.
Contrairement au Danemark et au Royaume-Uni, la Croatie ne bénéficie pas d'une clause d'exemption concernant l'adoption de la monnaie unique.
Lorsque le nombre des gouverneurs des BCN sera supérieur à quinze, ces derniers exerceront leur droit de vote selon un système de rotation.
Les moins-values latentes se sont établies à 64 millions d'euros en 2015 (contre 8 millions d'euros en 2014).
Le billet de 500 euros continue d'avoir cours légal et gardera toujours sa valeur
Tant que la zone euro comptera moins de vingt-deux États membres, la rotation fonctionnera selon les règles suivantes:
```

请注意，GPT-3 是根据 2021 年之前在网上公布的数据进行训练的。欧洲央行数据集已于 2017 年编制完成，因此 GPT-3 很可能在培训期间看到了要翻译的英文文本，或许还有其法文翻译。这将人为地使任务变得更容易，并防止与其他模型进行任何性能比较(因此我在本文中不做)。

为了比较零镜头、单镜头和少镜头的性能，我将使用度量标准[COMET](https://github.com/Unbabel/COMET)([Apache License 2.0](https://github.com/Unbabel/COMET/blob/master/LICENSE))对翻译进行手动和自动评估。

# 零镜头翻译

![](img/775f0c405a2389fbe3026e14180f8f22.png)

图片来自 [Pixabay](https://pixabay.com/photos/dart-dart-board-sports-bullseye-1093022/)

对于零镜头翻译，我们只需要用自然语言描述任务。为了展示提示的影响，我尝试了 3 种不同的描述。

确切地说，每次我更改提示时，我只更改上面 Python 脚本中“task_description”的内容。

我将每个提示命名为“prompt_x ”,以便在本文后面引用它。

## **提示 _1:**

```
Translate English to French:
[source] =>
```

在 prompt_1 和所有后续提示中，“[source]”是要翻译的英文文本。这个提示符是 OpenAI 在介绍 GPT-3 的论文中使用的。

注意空格。GPT-3 不喜欢以空格结尾的提示。

当然，你可以把“英语”和“法语”替换成你想要的语言。

我还尝试了下面的变体，省略了要翻译的语言的名称。

## 提示 _2

```
Translate to French:
[source] =>
```

第三个提示是极简的，没有提到任务。

## 提示 _3

```
English
[source]
French
```

*【来源】*可以多种方式格式化。我尝试了许多可能性，但本文只保留了以下内容:

*   *单元*:每个 API 调用只发送一个句子进行翻译。
*   *批量*:一次性发送所有句子。这种方式是减少你的账单的最佳方式，因为你不必为每个要翻译的句子重复任务描述。如果句子应该是连贯的，这也是理想的。然而，在我为本文选择的 ECB 语料库中，句子是无序的，因此 GPT-3 在翻译句子时可能会被误导，因为它会查看从不同上下文中选取的先前翻译的句子。

## 结果

**带*提示 _1* 和*单元* :**

*注意:在下面的例子中，“— — —”表示后面的内容是由另一个 API 调用生成的。我还手动添加了“响应:”来指示 GPT-3 的响应开始的位置。*

```
Translate English to French:
Today also marks the opening of the online "Euro Run" competition for children aged 9 to 12 living in the euro area. =>

Response:
Aujourd'hui, c'est aussi le début du concours "Euro Run" destiné aux enfants de 9 à 12 ans vivant dans l'euro zone.
----------
Translate English to French:
Unlike Denmark and the United Kingdom, Croatia does not have the right to opt out of adopting the single currency. =>

Response:
Contrairement à la Danemark et à la Grande-Bretagne, la Croatie ne dispose pas de droit de déclencher une optique pour la monnaie unique.
----------
Translate English to French:
When the number of NCB Governors exceeds 15 they will exercise a voting right on the basis of a rotation system. =>

Response:
Lorsque le nombre de gouverneurs de NCB dépasse 15, ils exerceront un droit de vote en fonction d'un système de rotation.
----------
Translate English to French:
Write-downs amounted to €64 million in 2015 (2014: €8 million). =>

Response:
Les écritures ont atteint 64 millions d'euros en 2015 (2014: 8 millions).
----------
Translate English to French:
€500 banknote remains legal tender and will always retain its value =>

Response:
Le 500 € billets de banque restent toujours valables et garderont toujours leur valeur.
----------
Translate English to French:
While there are fewer than 22 euro area Member States the rotation will take place in accordance with the following rules: =>

Response:
Les États membres de l'euro zone qui ne seront pas présents à la réunion du Conseil européen du 22 juin 2012 auront droit à une rotation annuelle.
```

*   完成令牌总数:48+42+47+24+30+50 = 241
*   提示令牌总数:35+30+31+25+21+30 = 172
*   总令牌数:83+72+78+49+51+80 = 413
*   总成本:0.002 * 413/1000 = 0.000826 美元
*   彗星评分:20.74

**带*提示 _1* 和*批次* :**

```
Translate English to French:
Today also marks the opening of the online “Euro Run” competition for children aged 9 to 12 living in the euro area.
Unlike Denmark and the United Kingdom, Croatia does not have the right to opt out of adopting the single currency.
When the number of NCB Governors exceeds 15 they will exercise a voting right on the basis of a rotation system.
Write-downs amounted to €64 million in 2015 (2014: €8 million).
€500 banknote remains legal tender and will always retain its value
While there are fewer than 22 euro area Member States the rotation will take place in accordance with the following rules:

Response:
Le concours européen de course en ligne "Euro Run" pour les enfants de 9 à 12 ans vivant dans l'espace européen ouvre ses portes aujourd'hui.
Contrairement à la Norvège et au Royaume-Uni, la Croatie n'a pas le droit de choisir d'opter pour la monnaie unique.
Lorsque le nombre de gouverneurs de la Banque centrale européenne dépasse 15, ils exerceront un droit de vote en fonction d'un système de rotation.
Le montant des dépréciations atteint €64 millions en 2015 (2014: €8 millions).
Le 500 euros de la note banque reste toujours une monnaie légale et conserve toujours son prix.
Même si il y a moins de 22 États membres de l'espace européen, le rythme de rotation se fera selon les règles suivantes :
```

*   完成令牌总数:256
*   提示令牌总数:136
*   令牌总数:392
*   总成本:0.002 * 392/1000 = 0.000784 美元
*   彗星评分:37.45

对于 prompt_2 和 prompt_3，我观察到以下情况。

*注意:为了缩短本文，我不在这里写这两个提示的提示和输出。如果你想看，我会把它们贴在下面的评论里。*

**带*提示 _2* 和*单元* :**

*   完成令牌总数:60+43+46+24+24+52 = 249
*   提示令牌总数:34+29+30+24+20+29 = 166
*   总令牌数:94+72+76+48+44+81 = 415
*   总成本:0.002 * 415/1000 = 0.000830 美元
*   彗星评分:5.41

**带*提示 _2* 和*批次* :**

*   完成令牌总数:220
*   提示令牌总数:136
*   令牌总数:356
*   总成本:0.002 * 356/1000 = 0.000712 美元
*   彗星评分:43.82

**带*提示 _3* 和*单元* :**

*   完成令牌总数:50+43+46+28+20+67 = 254
*   提示令牌总数:33+28+29+23+19+28 = 160
*   总令牌数:83+71+75+51+39 +95 = 414
*   总成本:0.002 * 414/1000 = 0.000828 美元
*   彗星评分:-24.69

**带*提示 _3* 和*批次* :**

*   完成令牌总数:227
*   提示令牌总数:135
*   令牌总数:362
*   总成本:0.002 * 362/1000 = 0.000724 美元
*   彗星评分:35.09

如果我们看一下 COMET 的分数，根据 COMET 的说法，显然“批量”配置比“单元”配置产生更好的翻译。我是通过人工评测确认的。

我的假设是，“块”配置在生成期间为 GPT-3 提供了更多的上下文。换句话说，当法语上下文在生成期间增长时，GPT-3 变得更有信心应该生成法语文本。

至于提示，prompt_3 产生的结果要低得多，而且对于某些句子来说似乎不能完成任务。这很直观，因为我们从提示中去掉了“翻译”。根据我的人工评估，使用 prompt_1 和 prompt_2 获得的翻译之间的差异并不显著。

彗星在这里没有告诉你的是**这些翻译都很烂**。有些翻译部分正确，有些则通顺但完全不准确。

Zero-shot GPT-3 生成的法语翻译质量远远低于标准机器翻译系统。*注:* *如果你使用达芬奇模型，在零镜头翻译时，你可能会得到更好的结果，对于翻译成英语。*

我展示它是为了告诉你，即使 GPT 3 号没有经过翻译训练，如果我们给它提供一个简单的任务描述，它实际上是能够翻译的。

如果我们比较所有配置的成本(以美元计),我们看不出有多大差别。

# 一次性翻译

![](img/c5780ab51905c2d70617f5d2c1c06edb.png)

图片来自 [Pixabay](https://pixabay.com/photos/dart-board-game-target-competition-933118/)

在一次性配置中，我们添加了一个由人工翻译产生的英语到法语翻译的例子。

这里，我只给出了 prompt_1 的结果，因为 prompt_2 的结果非常相似。

**带*提示 _1* 和*单元* :**

```
Translate English to French:
The competition gives us a unique opportunity to have a direct dialogue with thousands of students across the euro area. => Le jeu-concours nous offre une possibilité unique de dialoguer directement avec des milliers d'élèves à travers la zone euro.
Today also marks the opening of the online "Euro Run" competition for children aged 9 to 12 living in the euro area. =>

Reponse:
 Aujourd'hui, c'est aussi le début du concours « Euro Run » destiné aux enfants de 9 à 12 ans vivant dans la zone euro.
----------
Translate English to French:
The competition gives us a unique opportunity to have a direct dialogue with thousands of students across the euro area. => Le jeu-concours nous offre une possibilité unique de dialoguer directement avec des milliers d'élèves à travers la zone euro.
Unlike Denmark and the United Kingdom, Croatia does not have the right to opt out of adopting the single currency. =>

Reponse:
Contrairement à la Danemark et à la Grande-Bretagne, la Croatie ne dispose pas de droit de décider de ne pas adopter le système monétaire unique.
----------
Translate English to French:
The competition gives us a unique opportunity to have a direct dialogue with thousands of students across the euro area. => Le jeu-concours nous offre une possibilité unique de dialoguer directement avec des milliers d'élèves à travers la zone euro.
When the number of NCB Governors exceeds 15 they will exercise a voting right on the basis of a rotation system. =>

Reponse:
 Lorsque le nombre de gouverneurs NCB dépasse 15, ils exerceront un droit de vote en fonction d'un système de rotation.
----------
Translate English to French:
The competition gives us a unique opportunity to have a direct dialogue with thousands of students across the euro area. => Le jeu-concours nous offre une possibilité unique de dialoguer directement avec des milliers d'élèves à travers la zone euro.
Write-downs amounted to €64 million in 2015 (2014: €8 million). =>

Reponse:
 Les écritures à terme atteignaient 64 millions d'euros en 2015 (2014: 8 millions).
----------
Translate English to French:
The competition gives us a unique opportunity to have a direct dialogue with thousands of students across the euro area. => Le jeu-concours nous offre une possibilité unique de dialoguer directement avec des milliers d'élèves à travers la zone euro.
€500 banknote remains legal tender and will always retain its value =>

Reponse:
 Le billet de 500 euros reste toujours légalement valable et gardera toujours son prix
----------
Translate English to French:
The competition gives us a unique opportunity to have a direct dialogue with thousands of students across the euro area. => Le jeu-concours nous offre une possibilité unique de dialoguer directement avec des milliers d'élèves à travers la zone euro.
While there are fewer than 22 euro area Member States the rotation will take place in accordance with the following rules: =>

Reponse:
 Bien que les 22 États membres de la zone euro soient les seuls participants, le tour de rotation se déroulera selon les règles suivantes :
```

*   完成令牌总数:47+46+45+27+26+42 = 233
*   提示令牌总数:98+93+94+88+84+93 = 550
*   总令牌数:145+139+139+115+110+135 = 783
*   总成本:0.002 * 783/1000 = 0.001566 美元
*   彗星评分:49.01

**带*提示 _1* 和*批次* :**

```
Translate English to French:
The competition gives us a unique opportunity to have a direct dialogue with thousands of students across the euro area. => Le jeu-concours nous offre une possibilité unique de dialoguer directement avec des milliers d’élèves à travers la zone euro.
Today also marks the opening of the online “Euro Run” competition for children aged 9 to 12 living in the euro area.
Unlike Denmark and the United Kingdom, Croatia does not have the right to opt out of adopting the single currency.
When the number of NCB Governors exceeds 15 they will exercise a voting right on the basis of a rotation system.
Write-downs amounted to €64 million in 2015 (2014: €8 million).
€500 banknote remains legal tender and will always retain its value
While there are fewer than 22 euro area Member States the rotation will take place in accordance with the following rules: =>

Reponse:
 Aujourd’hui aussi est marqué le lancement du concours « Euro Run » destiné aux enfants de 9 à 12 ans vivant dans la zone euro.
Comme la Norvège et le Royaume-Uni, la Croatie n'a pas le droit de choisir de ne pas adopter le système monétaire unique.
Lorsque le nombre de gouverneurs de la Banque centrale européenne dépasse 15, ils exerceront un droit de vote en fonction d'un système de rotation.
Les write-downs atteignirent €64 millions en 2015 (2014: €8 millions).
Le 500 euros de banknote reste toujours valable monnaie et gardera toujours sa valeur.
Environ 22 États membres de la zone euro ont adhéré à la monnaie unique, tandis que la rotation aura lieu selon les règles suivantes :
```

*   完成令牌总数:240
*   提示令牌总数:200
*   代币总数:440
*   总成本:0.002 * 440/1000 = 0.000880 美元
*   彗星评分:44.88

对于这两种配置，正如所料，一次转换比零次转换好。由于翻译示例，GPT 3 号更好地“理解”了任务。

还要注意的是，对于单次平移，*单元*配置比*批次*执行得更好。

与零镜头配置相比，这些翻译改进了很多，但其中一些奇怪地让我想起了 10 年前由基于短语的统计机器翻译系统生成的翻译。

# 少镜头翻译

![](img/94b19139592d749ed0d526ab19c3376b.png)

图片来自 [Pixabay](https://pixabay.com/illustrations/darts-goal-target-direct-hit-arrow-2349477/)

对于少量翻译，我在提示中放了 10 个翻译示例(选自 ECB 语料库)。这是与一次性提示的唯一区别。

我发现*批处理*配置在这个配置中返回奇怪的结果，最明显的是错过了翻译一些句子。这可能是由于提示符变得很长。似乎 GPT-3 没有正确地找到要翻译的文本从哪里开始。

我不会显示来自 GPT-3 的提示和响应，因为用于少量学习的例子在本文中占了很大的篇幅。

**带*提示 _1* 和*单元* :**

*   完成令牌总数:48+44+45+23+29+41 = 230
*   提示令牌总数:641+636+637+631+627+636 = 3，808
*   令牌总数:689+680+682+654+656+677 =4，038
*   总成本:0.002 * 4038/1000 = 0.008076 美元
*   彗星评分:59.39

不出所料，我们又一次提高了翻译质量。但我发现它仍然不如标准的机器翻译系统。

注意，我在这里只使用了几个例子。要获得 GPT-3 作者提到的翻译质量，你需要使用至少 30 个翻译范例。而且，你可以通过切换到达芬奇来进一步改善结果。

但是，请注意，与零投配置相比，成本已经增加了很多。翻译 6 个句子的成本为 0.008076 美元，换句话说，是零镜头配置的 11 倍。这大约是 1000 句话 1.3 美元。如果换成达芬奇，就得把成本乘以 10，也就是 1000 句话 13 美元。

这是因为我们为每个要翻译的句子提供了翻译示例。这并不理想，但这是在不微调 GPT-3 的情况下获得相当好的结果的唯一方法。

# 微调呢？

OpenAI 提议通过 API 对 GPT-3 进行微调。

在翻译上对 GPT-3 进行微调肯定会产生更好的翻译。

然而，我不推荐它，除非你有使用 GPT-3 的特殊需要。

微调 GPT-3，然后使用微调模型更昂贵，你可能要尝试几种配置才能获得比少数镜头翻译更好的结果。

# 结论

总之，从 GPT-3 中获得好的翻译既不容易也不便宜，但绝对是可能的。

如果你有足够大的预算，我会推荐使用 GPT-3 来翻译其他标准机器翻译系统不能很好覆盖的特定领域和语言。对于这样的用例，你可能会得到比其他机器翻译系统更好的翻译。

请注意，我不会将 GPT-3 视为机器翻译系统，除非你将其微调为机器翻译系统。

GPT 3 号对提示的解释是自由的。这种自由会带来一些后果。GPT-5 月 3 日:

*   输出错误的语言
*   跳过要翻译的句子
*   根本不执行翻译

为了避免这些陷阱，选择正确的提示并提供有用的翻译示例至关重要。

*如果你喜欢这篇文章，支持我工作的最好方式就是使用我的链接成为一名媒体会员:*

[](https://medium.com/@bnjmn_marie/membership)  

*如果你已经是会员，想要支持这项工作，* [*就在 Medium*](https://medium.com/@bnjmn_marie) *上关注我吧。*