# 用 Python 过滤音频文件中的脏话

> 原文：<https://towardsdatascience.com/profanity-filtering-audio-files-python-3dd2335bc567>

## 用 Python 和 AssemblyAI 过滤音视频文件中的脏话

![](img/39e74394969123cd610ce7e1d8f73f51.png)

沃洛季米尔·赫里先科在 [Unsplash](https://unsplash.com/s/photos/dont-speak?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

## 介绍

在我最近的一篇文章中，我们探讨了针对音频和视频文件的</content-moderation-python-assemblyai-4a09ca61d346>****敏感内容检测和调节。另一个密切相关的主题是亵渎检测和过滤。换句话说，*我们如何用星号替换亵渎的话？*****

****在今天的教程中，我们将探索如何使用 Python 和 AssemblyAI API 执行语音到文本转换。此外，我们将确保文本转录是免费的亵渎。****

## ****使用 Python 过滤脏话的语音转文本功能****

****现在，让我们从如何在 Python 中实现脏话过滤的分步指南开始。对于本教程，我们将使用 AssemblyAI API，因此如果您想继续学习，请确保从他们的网站上获得一个访问令牌(绝对免费)。****

****下一步是准备将发送到 AssemblyAI 端点的 POST 请求的头。这应该如下所示(确保使用您自己的访问令牌):****

****导入请求库和定义请求头—来源:[作者](https://gmyrianthous.medium.com/)****

****现在我们需要读取将要执行语音到文本转换的音频文件，然后将它上传到 AssemblyAI 提供的托管服务。端点将在托管服务上返回一个 URL，然后用于后续请求，以便获得输入文件的副本。****

****读取输入的音频文件并上传到 AssemblyAI 主机服务上—来源:[作者](https://gmyrianthous.medium.com/)****

****现在，我们打算转录的音频文件已经上传到托管服务上。在下一步中，我们将使用作为响应的一部分返回的`upload_url`,以便获得文本转录。****

> ****默认情况下，API 将返回音频的逐字记录，这意味着如果在音频中说脏话，将会出现在记录中。****

****为了**对输入执行脏话过滤，**我们只需要在`POST`请求的主体中提供一个附加参数，这个参数叫做`filter_profanity`，应该设置为`true`。****

****向转录端点发送 POST 请求—来源:[作者](https://gmyrianthous.medium.com/)****

****现在转录过程应该启动并运行了，我们需要发出一个`GET`请求来访问结果。注意，我们需要发出几个请求，直到端点返回的状态是`error`或`completed`。****

****这是因为 API 提供了 [**异步转录**](https://docs.assemblyai.com/#processing-times) 。提交音频文件进行转录时，通常会在音频文件持续时间的 15–30%内完成。请注意，AssemblyAI 还支持实时转录——要了解更多信息，请务必阅读我的另一篇关于用 Python 进行实时语音识别的文章。****

****从 AssemblyAI 端点检索转录结果—来源:[作者](https://gmyrianthous.medium.com/)****

****最后，我们可以将转录结果写入输出文件，这将使我们在检查输出时更加轻松。****

****将文本转录写入文本文件—来源:[作者](https://gmyrianthous.medium.com/)****

## ****解读回应****

****现在，如果我们检查写入输出文件的结果，我们应该看到文本转录。如果检测到亵渎，则会用星号替换，如下面的输出示例所示:****

```
**It was some tough s*** that they had to go through. But they did it. I mean, it blows my f****** mind every time I hear the story.**
```

****下面给出了返回响应的更完整形式****

****来自文本转录端点的示例响应—来源:[作者](https://gmyrianthous.medium.com/)****

****实际的文本转录在响应的`text`字段中提供。****

****`words`字段将包含在音频(或视频)文件中检测到的每个单词，以及开始和结束时间戳，以及`confidence`。后者对应于模型对检测到的单词的置信度(在`0-1`之间测量)。显然，`confidence`越高，该模型对口语单词越确定。****

****请注意，除了每个检测到的单词的`confidence`之外，响应还包括整个转录的`confidence`(外部字段)。****

****最后，响应包括在文本转录期间使用的配置和参数。例如，您可以验证是否启用了`profanity_filtering`选项，因为它在转录端点给出的响应中是`true`。****

****关于响应中包含的各个字段的更完整描述，请参考[官方文件](https://docs.assemblyai.com/reference#transcript)中的相关章节。****

## ****完整代码****

****作为本教程一部分的完整代码可以在下面的 GitHub Gist 中找到。****

****用 Python 和 AssemblyAI 过滤脏话的完整代码—来源:[作者](https://gmyrianthous.medium.com/)****

## ****最后的想法****

****在今天的文章中，我们展示了如何使用 Python 和 AssemblyAI API 对音频和视频文件进行脏话过滤。当需要过滤脏话时，您可能通常希望(或需要)执行敏感内容检测。关于如何执行内容审核的分步指南，请阅读下面的文章****

****</content-moderation-python-assemblyai-4a09ca61d346> **** 

****[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。******

****<https://gmyrianthous.medium.com/membership> **** 

******你可能也会喜欢******

****</speech-recognition-python-assemblyai-bb5024d322d8> **** ****</sentiment-analysis-assemblyai-python-a4686967e0fc> ****