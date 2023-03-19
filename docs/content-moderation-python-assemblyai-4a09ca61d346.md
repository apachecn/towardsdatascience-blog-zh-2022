# 如何使用 Python 对音频文件进行内容审核

> 原文：<https://towardsdatascience.com/content-moderation-python-assemblyai-4a09ca61d346>

## 用 Python 和 AssemblyAI API 检测音频文件中的敏感内容

![](img/8072e098f61b16b49898d0b68748b322.png)

格伦·卡丽在 [Unsplash](https://unsplash.com/s/photos/scissor?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄的照片

## 介绍

在我以前的一篇文章中，我讨论了语音识别和如何用 Python 执行语音到文本转换。当处理音频或视频文件时，有时能够检测到任何敏感内容以便采取某些操作是非常重要的。

例如，您可能想要在播客上执行语音到文本转换，同时您想要检测是否有任何参与者或发言者讨论毒品、色情或酒精(或任何其他可能被视为敏感的内容)。

在今天的教程中，我们将讨论内容审核，以及如何使用 Python 和 AssemblyAI API 检测音频甚至视频文件中的敏感内容。

## 使用 Python 对音频和视频文件进行内容审核

AssemblyAI API 提供了一个[内容审核功能](https://docs.assemblyai.com/audio-intelligence#content-moderation\)，允许你执行**内容安全检测。**在接下来的几节中，将在我们的分步指南中使用它来检测输入文件中是否提到了敏感话题，如果是，则是在何时以及说了什么。API 可以捕捉的一些主题包括事故、灾难、公司财务、赌博和仇恨言论(还有更多的主题，因此如果您正在寻找其他主题，请确保查看包含所涵盖主题详细列表的[文档](https://docs.assemblyai.com/audio-intelligence#content-moderation)。

现在，让我们从一个音频文件开始，我们将对其执行语音到文本转换，同时我们还将启用内容审核功能，以便检测音频文件是否涉及 API 支持的任何敏感话题。请注意，如果您想要遵循这个逐步指南，您将需要来自 AssemblyAI 的一个绝对免费的 [API 访问令牌](https://app.assemblyai.com/signup)。

现在我们有了 API 的访问令牌，让我们做一些必需的导入，并为将发送到 AssemblyAI 端点的请求准备头部。

导入请求库并准备 POST 请求的标题—来源:[作者](https://gmyrianthous.medium.com/)

然后，我们需要读入我们想要执行语音识别和内容审核的音频文件，以便将其上传到 AssemblyAI 的相应托管服务，该服务将返回我们将在后续请求中使用的 URL。

将输入文件上传到 AssemblyAI 的托管服务—来源:[作者](https://gmyrianthous.medium.com/)

既然我们已经将音频文件上传到托管服务，并从端点接收回了`upload_url`,我们可以继续执行实际的语音到文本任务，将它与敏感内容检测结合起来。注意，为了启用内容审核特性，我们还需要将参数`content_safety`传递给`True`。

执行语音转文本和敏感内容检测—来源:[作者](https://gmyrianthous.medium.com/)

响应(显示在上面的注释中)将包括已经进行的转录的转录 ID。最后一步涉及使用`id`的`GET`请求。注意，我们一直发送`GET`请求，直到响应的状态被标记为`completed`(或`error`):

最后，我们将来自转录端点的响应写入一个文件:

将来自转录端点的响应写入文件—来源:[作者](https://gmyrianthous.medium.com/)

## 解释内容审核输出

我们在上一节中发送给转录端点的`POST`请求的响应将类似于下面共享的响应。

转录端点返回的响应包括语音转文本和内容安全标签—来源:[作者](https://gmyrianthous.medium.com/)

音频文件的文本转录(即语音转文本的结果)包含在外部`text`字段中。现在**内容安全检测结果**将包含在`content_safety_labels`键中，如下所述:

*   `**results**`:是包含内容安全检测模型标记的音频转录的所有部分的列表
*   `**results.text**`:包含触发检测模型的文本转录的字段。
*   `**results.labels**`:是包含内容安全检测模型为特定转录文本段预测的所有标签(例如`disaster`、`drugs`、`alcohol`)的列表。这个列表中的每个 JSONObject 还带有`confidence`和`severity`指标(关于这两个术语的更多内容将在下一节讨论)。
*   `**results.timestamp**`:表示包含被标记内容的音频文件的开始和结束时间(以毫秒为单位)的字段。
*   `**summary**`:该键包含`results`中预测的每个标签相对于整个音频文件的置信度。例如，如果我们以某个特定标签的单一结果结束，比如说`0.99`置信度，但同时音频文件长达 5 小时，那么 summary 将显示该标签的置信度低得多，因为它在整个音频中没有被广泛使用。
*   `**severity_score_summary**`:提供`results`中包含的每个预测标签相对于整个音频文件的整体严重性。

## 了解严重性和置信度得分

在解释内容安全检测模型给出的结果时，我们遇到了两个评分术语。

事实上，每一个预测的标签都会有其严重性分数**和置信度分数**。**严重性**分数表示**被标记的内容有多严重**，分数越低表示内容的严重性越低。另一方面，**置信度**分数表明**模型在预测输出标签时**的置信度。注意，它也是以`0-1`的刻度来测量的。****

例如，考虑我们在 API 响应中观察到的预测的`disasters`标签的严重性和置信度得分。

```
"labels": [
    {
        "confidence": 0.9986903071403503,
        "severity": 0.11403293907642365,
        "label": "disasters"
    }
],
```

`0.9986903071403503`的置信度指示 AssemblyAI 内容安全检测模型 99.87%确信在指定的`timestamp`的音频文件的口述内容是关于低严重性的自然灾害(`0.1140`)。

## 完整代码

今天教程中使用的完整代码可以在下面分享的 GitHub Gist 中找到。

用于音频转录和敏感内容检测的完整代码—来源:[作者](https://gmyrianthous.medium.com/)

## 最后的想法

在今天的文章中，我们展示了如何使用 Python 和 AssemblyAI API 对音频或视频文件执行敏感内容审核。相关的 API 端点提供了能够检测敏感内容的功能。

此外，我们讨论了如何解释返回的响应，并检查内容安全检测模型是否标记了输入音频的任何部分。最后，我们解释了如何解释信心和严重性分数以及它们的实际含义。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

<https://gmyrianthous.medium.com/membership>  

**你可能也会喜欢**

</speech-recognition-python-assemblyai-bb5024d322d8>  </real-time-speech-recognition-python-assemblyai-13d35eeed226>  </sentiment-analysis-assemblyai-python-a4686967e0fc> 