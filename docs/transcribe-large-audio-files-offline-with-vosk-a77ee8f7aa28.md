# 使用 Vosk 离线转录大型音频文件

> 原文：<https://towardsdatascience.com/transcribe-large-audio-files-offline-with-vosk-a77ee8f7aa28>

## 为您的下一个 NLP 项目转录大型音频文件

![](img/c82e3e6998783d386ead99fba4a8bf69.png)

乔纳森·贝拉斯克斯在 [Unsplash](https://unsplash.com/) 上拍摄的照片。

我受分析 reddit 数据的自然语言处理(NLP)项目的启发，我想到了使用播客数据的想法。然而，由于播客是(大)音频文件，人们需要先将它们转录成文本。这个过程也被称为 **A** 自动**S**peech**R**ecognition(ASR)或**S**peech-**t**o-**t**ext(STT)。

Google、Azure 或 AWS 等提供商提供了出色的 API 来完成这项任务。**但是，如果您想要离线转录，或者出于某种原因，您不被允许使用云解决方案，该怎么办？**

## TL；博士；医生

*   Vosk 是一个允许你离线转录音频文件的工具包
*   它支持 20 多种语言和方言
*   音频必须首先转换为波形格式(单声道，16Hz)
*   大型音频文件的转录可以通过使用缓冲来完成
*   Colab 笔记本可以在[这里找到](https://github.com/darinkist/medium_article_vosk/blob/main/Transcribe_large_audio_files_offline_with_Vosk.ipynb)

# 目标

这就是为什么我写这篇文章来给你一个备选方案以及如何使用它们的概述。

这个想法是使用提供**预训练模型**的包或工具包，这样我们就不必先自己训练模型。

在这篇文章中，我将重点介绍 Vosk。还有很多类似于 Moziall 的 DeepSpeech 或 SpeechRecognition 的软件包。然而，DeepSpeech 的未来是不确定的，SpeechRecognition 除了在线 API 之外，还包括使用 Vosk 的 CMUSphinx。

我假设我们要转录的数据在 youtube 上**不可用。如果有，我强烈推荐去看看`[youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/)`套餐。它允许你为一个给定的视频获得生成的脚本，这比我们在下面要做的要少得多。**

# 先决条件:以正确的格式输入数据

在我们进入转录部分之前，我们必须首先将数据转换成正确的格式。播客或其他(长)音频文件通常是 mp3 格式。然而，这不是包或工具包可以使用的格式。

更具体地说，我们需要将我们的(mp3)音频转换为:

*   波形格式(。wav)
*   单声道的
*   16，000 赫兹采样速率

转换非常简单。首先我们必须安装`ffmpeg`，它可以在[https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)下找到。

Mac 用户可以使用 brew 下载并安装它:

```
brew install ffmpeg
```

接下来我们安装`pydub`包:

```
pip install pydub
```

下面的代码片段将 mp3 转换成所需的 wav 格式。它将输出存储在与给定 mp3 输入文件相同的目录中，并返回其路径。如果我们想跳过几秒钟(例如介绍)，我们可以通过设置我们想跳过的秒数来使用 skip 参数。如果我们想先尝试一下，我们可以将*摘录*参数设置为*真*以仅获取音频文件的前 30 秒。

使用这个功能，我们现在可以将我们的播客文件转换成所需的 *wav* 格式。

为了有一个(互动)的例子，我选择转录以下播客插曲:

> 请注意:播客是随机选择的。我与创造者没有任何联系，也没有因为说出他们的名字而获得报酬。

由于前 37 秒是介绍，我们可以使用 skip 参数跳过它们。

对于第一个例子，我们也将参数摘录设置为*真*:

```
mp3_to_wav('opto_sessions_ep_69.mp3', 37, True)
```

我们的新文件*opto _ sessions _ EP _ 69 _ extract . wav*现在有 30 秒长，从 0:37 到 1:07 开始。

现在我们可以开始转录了！

# 沃斯克

Vosk 是一个支持 20 多种语言(如英语、德语、印度语等)的语音识别工具包。)和方言。它可以离线工作，甚至可以在树莓 Pi 这样的轻量级设备上工作。

它的便携式型号每个只有 50Mb。然而，还有更大的型号。所有可用型号的列表可以在这里找到:[https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)

拨打以下电话即可轻松安装 Vosk:

```
pip install vosk
```

安装了 Vosk 之后，我们必须下载一个预先训练好的模型。我决定选择最大的一个:`vosk-model-en-us-0.22`

现在我们已经拥有了我们需要的一切，让我们打开我们的 wave 文件并加载我们的模型。

在我们深入研究转录过程之前，我们必须熟悉一下 VOSKs 的输出。

VOSK 返回 JSON 格式的转录，如下所示:

```
{
    "text" : "cats are dangerous"
}
```

如果我们也想知道 VOSK 对每个单词有多自信，并且想知道每个单词的时间，我们可以使用`SetWords(True)`。例如，一个单词的结果如下所示:

```
{
   "result":[
      {
         "conf":0.953349,
         "end":6.090000,
         "start":5.700000,
         "word":"cats"
      },
      {
         etc.
      },
      etc.
   ],
   "text":"cats are dangerous"
}
```

因为我们想要转录大的音频文件，所以通过逐块转录 wave 文件来使用缓冲方法是有意义的。以下代码显示了转录方法:

我们读入前 4000 帧(第 7 行)并将它们交给我们加载的模型(第 12 行)。该模型返回(以 JSON 格式)结果，该结果作为 dict 存储在 *result_dict* 中。然后，我们只提取文本值，并将其附加到我们的转录列表中(第 14 行)。

如果没有更多的帧要读取(第 8 行)，循环停止，我们通过调用 *FinalResult()* 方法来捕捉最终结果。此方法还会刷新整个管道。

结果应该是这样的:

```
to success on today show i'm delighted to introduce beth kinda like a technology analyst with over a decade of experience in the private markets she's now the cofounder of io fund which specializes in helping individuals gain a competitive advantage when investing in tech growth stocks how does beth do this well she's gained hands on experience over the years was i were working for or analyzing a huge amount of relevant tech companies in silicon valley the involved in the market
```

> 注意:如果你对更“时尚”的解决方案感兴趣(使用进度条)，你可以在这里找到我的代码。

# 用于离线转录的其他包或工具包

正如在简介中提到的，还有更多可用的包或工具包。然而，它们的实现不像 Vosk 那么容易。不过如果你有兴趣，我可以推荐**英伟达的 NeMo** 。

## 英伟达尼莫

NeMo 是为从事自动语音识别、自然语言处理和文本到语音合成的研究人员构建的工具包。和 VOSK 一样，我们也可以从一堆预先训练好的模型中进行选择，这些模型可以在[这里](https://catalog.ngc.nvidia.com/orgs/nvidia/models/nemospeechmodels)找到。

实现需要更多的时间和代码。基于 [Somshubra Majumdar 的](https://github.com/titu1994) [笔记本](https://github.com/NVIDIA/NeMo/blob/c9d04851e8a9c1382326862126788fadd77663ac/tutorials/asr/Streaming_ASR.ipynb)我创建了一个**精简版**，可以在这里[找到](https://github.com/darinkist/medium_article_vosk/blob/main/NeMo_ASR_example.ipynb)。

# 结论

Vosk 是一个很棒的离线转录工具包。与我测试的其他离线解决方案相比，Vosk 是最容易实现的。唯一缺少的东西是标点符号。到目前为止，还没有整合 it 的计划。但是，与此同时，如果需要，可以使用外部工具。