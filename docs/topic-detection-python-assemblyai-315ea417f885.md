# 如何使用 Python 执行语音转文本和主题检测

> 原文：<https://towardsdatascience.com/topic-detection-python-assemblyai-315ea417f885>

## 使用 Python 和 AssemblyAI 对音频文件执行文本转录和主题检测

![](img/29ed0cf6cc1cb32f0b4c1e94fa2a8dcc.png)

沃洛季米尔·赫里先科在 [Unsplash](https://unsplash.com/s/photos/speech?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

## 介绍

在我最近的一篇文章中，我讨论了关于[语音识别以及如何在 Python](/speech-recognition-python-assemblyai-bb5024d322d8) 中实现它。在今天的文章中，我们将进一步探讨如何对音频和视频文件进行主题检测。

作为一个例子，让我们考虑一下随着时间的推移变得越来越流行的播客。想象一下每天有多少播客被创建；不同平台(如 Spotify、YouTube 或 Apple Podcasts)上的推荐引擎根据讨论的内容对所有这些播客进行分类难道没有用吗？

## 使用 Python 执行语音转文本和主题检测

在本教程中，我们将使用 AssemblyAI API 来标记音频和视频文件中的主题。因此，如果您想继续，您首先需要获得一个 [AssemblyAI 访问令牌](https://app.assemblyai.com/signup)(这是绝对免费的)，我们将在调用 API 时使用它。

现在我们有了一个访问令牌，让我们开始准备在向 AssemblyAI 的各个端点发送请求时将使用的头。

为我们将发送到 API 端点的请求准备头部——来源:作者

接下来，我们需要将音频(或视频)文件上传到 AssemblyAI 的托管服务。然后，端点将返回上传文件的 URL，我们将在后续请求中使用它。

上传音频/视频文件到 AssemblyAI 主机服务-来源:作者

现在，下一步是最有趣的部分，我们将对上传的音频文件执行语音到文本的转换。在`POST`请求中，我们需要传递的只是从上一步接收到的`audio_url`以及需要设置为`True`的`iab_categories`参数。后者将触发文本转录上的主题检测。来自`TRANSCRIPT_ENDPOINT`的示例响应也显示在以下要点的末尾。

使用话题检测执行语音转文本—来源:作者

现在，为了获得转录结果(以及主题检测结果)，我们需要再发出一个请求。这是因为转录是[异步的](https://docs.assemblyai.com/#processing-times) —当提交文件进行转录时，我们需要一段时间才能访问结果(通常大约是整个音频文件持续时间的 15–30%)。

因此，我们需要发出一些`GET`请求，直到得到成功(或失败)的响应，如下图所示。

获取转录和主题检测结果—来源:作者

最后，让我们将接收到的结果写入一个文本文件，以便我们更容易检查输出并解释从转录端点接收到的响应:

将转录输出写入文件—来源:作者

## 解读回应

转录终点的响应示例如下所示:

启用话题检测的转录结果示例—来源:作者

外部的`text`键包含输入音频文件的文本转录结果。但是让我们更关注包含与主题检测结果相关的信息的`categories_iab_result`的内容。

*   `status`:包含话题检测的状态。正常情况下，这将是`success`。如果由于任何原因，主题检测模型已经失败，则该值将是`unavailable`。
*   `results`:该键将包括在输入音频文件中检测到的主题列表，包括影响预测并触发预测模型做出该决定的精确文本。此外，它还包括一些关于相关性和时间戳的元数据。我们将在下面讨论这两个问题。
*   `results.text`:该键包括已用特定主题标签分类的音频部分的精确转录文本。
*   `results.timestamp`:该键表示输入音频文件中说出`results.text`的开始和结束时间(以毫秒为单位)。
*   `results.labels`:这是包含由主题检测模型为`results.text`中的文本部分预测的所有标签的列表。相关性关键字对应于可以取`0`和`1.0`之间的任何值的分数，并且指示每个预测标签相对于`results.text`的相关程度。
*   `summary`:对于由主题检测模型在`results`阵列中检测到的每个唯一标签，`summary`键将包括该标签在输入音频文件的整个长度上的相关性。例如，如果在 60 分钟长的音频文件中仅检测到一次`Science>Environment`标签，则摘要关键字将包括该标签的相对较低的相关性分数，因为没有发现整个转录与该主题标签一致相关。

为了查看主题检测模型能够预测的主题标签的完整列表，请确保查看官方文档中的[相关章节。](https://docs.assemblyai.com/audio-intelligence#topic-detection-iab-classification)

## 完整代码

下面的 GitHub Gist 分享了作为本教程一部分的完整代码:

作为本教程一部分的完整代码——来源:作者

## 最后的想法

在今天的文章中，我们探讨了如何使用 Python 和 AssemblyAI API 对生成的文本转录执行语音转文本和主题检测。我们通过一个分步指南详细解释了如何使用各种 API 端点来对音频和视频文件执行主题检测。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

<https://gmyrianthous.medium.com/membership>  

**你可能也会喜欢**

</real-time-speech-recognition-python-assemblyai-13d35eeed226>  </summarize-audio-video-files-assemblyai-c9126918870c> 