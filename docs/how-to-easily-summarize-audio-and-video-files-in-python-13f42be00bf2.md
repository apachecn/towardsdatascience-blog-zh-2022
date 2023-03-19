# 如何用 Python 轻松汇总音视频文件

> 原文：<https://towardsdatascience.com/how-to-easily-summarize-audio-and-video-files-in-python-13f42be00bf2>

## 使用自动章节来总结 Python 中的音频和视频文件

![](img/a9a1d793882053a3b619d036f77756ff.png)

凯利·西克玛在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

Auto chapters 是一个强大的功能，它允许我们将音频/视频文件分成“章节”,然后为创建的每个章节自动生成摘要。

这种自动章节功能有不同的应用。像 YouTube 这样的视频平台有帮助用户跳转到他们正在寻找的内容的章节，播客播放器有使播客剧集更容易搜索的章节。

在本文中，我将向您展示如何使用 AssemblyAI API 在 Python 中使用这一特性。该 API 将执行语音到文本转换，然后自动生成包含摘要、标题和要点等数据的章节。

# 如何用 Python 对音视频文件进行摘要

在本指南中，我们将为史蒂夫·乔布斯 2005 年斯坦福毕业典礼演讲视频创建自动章节，你可以在 [YouTube](https://youtu.be/UF8uR6Z6KLc) 上找到。视频大约 15 分钟，我将只使用一个语音文件作为输入。

您可以随意使用自己的音频文件，但是，为了充分利用该功能，请确保您使用的音频超过 5 分钟。

除此之外，我们需要创建一个[免费的 AssemblyAI 帐户](https://www.assemblyai.com/)来获得一个 API 密匙。一旦你创建了你的帐号，进入首页标签，寻找一个名为“集成 API”的框来复制你的 API 密匙。

## 步骤 1:上传你的音频文件并获取网址

为了将语音转换成文本，然后对其进行总结，我们首先需要将音频文件上传到 AssemblyAI API。为此我们需要 2 个输入——音频文件的路径(`filename`)和 API 键。

之后，我们创建一个函数来读取我们的音频文件。我们将这个函数命名为`read_file`。为了上传我们的音频文件，我们将在 post 请求中调用这个函数。这个 post 请求将被发送到 AssemblyAI 的上载端点。

最后，我们使用请求中的`response`来获取上传音频文件的 URL。我们将所有这些存储在一个名为`audio_url`的变量中，我们将在下一步中使用它。

## 步骤 2:提交成绩单请求

在这一步中，我们向 AssemblyAI 的转录端点发送一个 post 请求。我们使用`audio_url`提交上传进行转录。

确保在`json` 参数中添加密钥/值对`‘auto_chapters’: True`。这将有助于我们不仅得到转录，而且总结。

最后，我们创建一个名为`transcript_id`的变量，表示我们提交的 id，我们将在第 3 步中使用它。

## 第三步:保存文字记录和总结

为了获得我们的脚本和摘要，首先，我们使用`transcript_endpoint`和`transcript_id`来创建一个名为`polling_endpoint`的变量。我们将定期向该端点发送 get 请求，以检查我们在步骤 2 中发送的请求的状态。

只有当状态设置为`completed`时，我们才在文本文件中保存抄本，在 JSON 文件中保存摘要。我们使用变量`transcript_id`作为这两个文件的名称。

就是这样！现在检查您的工作目录，您会发现一个包含脚本的`.txt`和一个包含摘要的`.json`文件。

以下是视频其中一部分的摘要、标题和要点:

```
{
"**summary**": "You have to trust that the dots will somehow connect in your future. You have to trust in something your gut, destiny, life, karma, whatever. Because believing that the dots will connect down the road will give you the confidence to follow your heart, even when it leads you off the well worn path.",
"**headline**": "Because believing that the dots will connect down the road will give you the confidence to follow your heart, even when it leads you off the well worn path.",
"**start**": 312538,
"**end**": 342070,
"**gist**": "the dots will somehow connect"
    },
```

现在是你自己尝试的时候了。你可以在我的 [Github](https://github.com/ifrankandrade/api.git) 上找到本文使用的代码。

[**加入我的电子邮件列表，与 1 万多人一起获取我在所有教程中使用的 Python for Data Science 备忘单(免费 PDF)**](https://frankandrade.ck.page/bd063ff2d3)

如果你喜欢阅读这样的故事，并想支持我成为一名作家，可以考虑报名成为一名媒体成员。每月 5 美元，让您可以无限制地访问数以千计的 Python 指南和数据科学文章。如果你用[我的链接](https://frank-andrade.medium.com/membership)注册，我会赚一小笔佣金，不需要你额外付费。

<https://frank-andrade.medium.com/membership> 