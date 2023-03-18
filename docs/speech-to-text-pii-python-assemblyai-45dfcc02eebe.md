# 如何使用 Python 执行语音转文本和删除 PII

> 原文：<https://towardsdatascience.com/speech-to-text-pii-python-assemblyai-45dfcc02eebe>

## 使用 AssemblyAI 和 Python 从文本转录中移除个人身份信息(PII)

![](img/37b59032a6491817cc795679e70fea8c.png)

本·斯威特在 [Unsplash](https://unsplash.com/s/photos/identity?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

## 介绍

在我最近的两篇文章中，我讨论了关于音频转录的 [*脏话过滤*](/profanity-filtering-audio-files-python-3dd2335bc567) 和[敏感内容检测](/content-moderation-python-assemblyai-4a09ca61d346)。数据处理和存储的另一个重要方面是如何处理敏感数据。

用来描述敏感数据的一个常用术语是**个人身份信息** (PII)。一般来说，任何可以用来(或结合其他信息)识别、**联系人**或**定位**个人的信息都被认为是 PII。

根据美国[国家标准与技术研究院](https://www.nist.gov/publications/guide-protecting-confidentiality-personally-identifiable-information-pii) (NIST)的规定，以下数据元素被视为个人身份信息:全名、面部数据、地址、电子邮件地址、身份证或护照号、指纹、信用卡号、出生日期、出生地、基因信息、电话号码、登录名、昵称、驾照或车牌号。

例如，考虑客户和代理之间的电话呼叫，其中前者需要提供信用卡信息。同时，该公司可能会出于监控或培训目的对通话进行录音。因此，我们需要编辑信用卡信息，以确保没有 PII 出现。另一个例子是医疗记录中的健康信息编辑。

在今天的教程中，我们将使用 AssemblyAI API 和 Python 来对音频文件执行语音到文本转换，同时确保消除个人身份信息。

## 执行语音到文本转换并消除 PII

在本节中，我们将逐步介绍如何对输入音频文件执行文本转录，同时删除任何个人身份信息。

为此，我们将使用 AssemblyAI API。如果你想跟随这个教程，你首先需要从他们的网站[发布一个访问令牌](https://app.assemblyai.com/signup)(这是完全免费的)，我们将在发送到 API 端点的请求的头中使用这个令牌。

现在让我们开始导入一些今天将使用的库，并为请求准备头部，包括我们的访问令牌。

准备请求的标题—来源:[作者](https://gmyrianthous.medium.com/)

现在，我们需要做的第二件事是读取输入音频文件，然后将其上传到 AssemblyAI 的托管服务。API 的端点将返回一个链接，指向我们将在后续请求中使用的上传文件。

读入输入音频文件并将其上传到 AssemblyAI 主机端点—来源:[作者](https://gmyrianthous.medium.com/)

既然我们已经在托管服务上上传了音频文件，我们就可以继续进行语音到文本的转换了。同时，我们将使用 AssemblyAI 的 [PII 修订](https://docs.assemblyai.com/audio-intelligence#pii-redaction)功能，以便在转录文本返回给我们之前，从转录文本中删除个人身份信息，如电话号码和社会保险号。

为了执行语音到文本转换，我们只需要在向 AssemblyAI 的相应端点发送我们的`POST`请求时提供`audio_url`参数。对于 PII 修订，我们只需要提供一个名为`redact_pii`的附加参数，并将其设置为`True`。

默认情况下，音频文件中说出的任何 PII 都将被转录为一个哈希码(`#`)。比如提到名字`John`，就会转录成`####`。该默认行为可通过`redact_pii_sub`参数控制，该参数用于定制如何替换 PII。该参数可以有两个值，或者是`hash`(这是我们已经提到的默认值)或者是`entity_name`。如果选择后一个值，则检测到的 PII 将替换为关联的策略名称。比如用`[PERSON_NAME]`代替`John`。**这是为了可读性**而推荐的。

今后，我们甚至可以从执行 PII 修订的检测算法中指定应该应用什么类型的策略。这种策略的列表可以通过接受各种有效选项的`redact_pii_policies`参数来确定。其中包括(但不限于)`email_address`、`blood_type`、`medical_process`、`date_of_birth`、`phone_number`。要更全面地了解可用政策选项，请务必参考官方文件的[相关章节。](https://docs.assemblyai.com/audio-intelligence#pii-redaction)

现在，让我们使用其中的一些参数来执行语音到文本转换和编辑 PII，如下所示:

PII 修订版语音转文本—来源:[作者](https://gmyrianthous.medium.com/)

在上面分享的要点的最后，我们可以看到一个来自端点的示例响应。您还可以观察所有参数的值(包括`redact_pii`和`redact_pii_sub`)。

所做的转录是[异步](https://docs.assemblyai.com/#processing-times)，这意味着当一个音频文件被提交用于文本转录时，它通常会在音频文件持续时间的大约 15–30%内完成。因此，我们需要不断地发出一些`GET`请求，直到响应的状态为`completed`或`error`。

从异步转录中获取结果—来源:[作者](https://gmyrianthous.medium.com/)

最后，我们现在可以将文本转录到输出文件中，这将使我们在检查结果时更加容易。

将文本转录写入输出文件—来源:[作者](https://gmyrianthous.medium.com/)

## 解读回应

现在，输出文件应该包含从相关 API 端点返回的文本转录。输入音频文件的(口述)内容如下。

```
John was born in 12/02/1993\. 
```

生成的文本转录如下所示:

```
[PERSON_NAME] was born in [DATE_OF_BIRTH]
```

下面给出了返回响应的更完整形式。

来自汇编 AI 转录端点的示例响应—来源:[作者](https://gmyrianthous.medium.com/)

文本转录位于响应的`text`参数下。转录算法检测到的单词将与观察到所说单词的时间戳和`confidence`分数一起包含在`words`字段中。关于响应中包含的各个字段的更完整描述，请参考[官方文件](https://docs.assemblyai.com/reference#transcript)中的相关章节。

## 完整代码

本教程中使用的完整代码可以在下面分享的 GitHub Gist 中找到。

文本转录和个人身份信息编辑完整代码—来源:[作者](https://gmyrianthous.medium.com/)

## 最后的想法

在今天的文章中，我们讨论了个人身份信息的重要性以及哪些数据被视为个人身份信息。此外，我们展示了如何使用 AssemblyAI API 执行语音到文本的转换，并从生成的文本转录中消除 PII。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership) [## 通过我的推荐链接加入 Medium-Giorgos Myrianthous

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

gmyrianthous.medium.com](https://gmyrianthous.medium.com/membership) 

**你可能也会喜欢**

[](/sentiment-analysis-assemblyai-python-a4686967e0fc) [## 如何用 Python 对音频文件进行情感分析

### 探索如何使用 AssemblyAI API 提取语音中的情感

towardsdatascience.com](/sentiment-analysis-assemblyai-python-a4686967e0fc) [](/real-time-speech-recognition-python-assemblyai-13d35eeed226) [## 如何用 Python 执行实时语音识别

### 使用 Python 中的 AssemblyAI API 执行实时语音转文本

towardsdatascience.com](/real-time-speech-recognition-python-assemblyai-13d35eeed226)