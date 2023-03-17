# AI 应用的 Azure 资源概述

> 原文：<https://towardsdatascience.com/overview-of-azure-resources-for-ai-applications-f9f8d68b6426>

# AI 应用的 Azure 资源概述

## 了解主要的 Azure 服务，以构建语言、视觉和搜索人工智能应用程序

![](img/bfbb9f4fc060d223a5a4afe82e0b2ac5.png)

照片由[西格蒙德](https://unsplash.com/@sigmund?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

## ⚠️在我的博客⚠️中读到了[的原帖](https://anebz.eu/azure-ai-resources)

导航云提供商找到如何在云中构建人工智能应用程序可能是一项艰巨的任务。这篇文章介绍了不同人工智能应用的主要 Azure 人工智能资源。

# 视力

## 1.[计算机视觉](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/)

计算机视觉是图像应用的主要 Azure 资源。在其最基本的形式中，它支持图像分类和多类对象检测。

它的第一个有趣的功能是图像分析，它可以从图像中提取几个特征:它可以检测品牌，名人和地点。它可以检测图像中人的性别和年龄。它可以向图像添加描述文本，检测配色方案，使用智能裁剪生成缩略图等等。

此外，它可以通过 OCR 技术读取图像中的文本。这可以通过 NLP 工具、内容审核等用于进一步的文本分析。

另一个非常有用的功能是空间分析，计算机视觉可以实时分析人们如何在一个空间中移动，以进行占用计数、社交距离和面具检测。

它还集成了 Face API，如下所述。

计算机视觉有许多集成，可以部署在容器中。

## 2.[定制视觉](https://azure.microsoft.com/en-us/services/cognitive-services/custom-vision-service/)

Custom Vision 是 Azure 最简单的图像分类和对象检测工具。您可以导入您的图像，给它们贴上标签(Custom vision 至少需要 50 张)，并根据您的数据快速训练一个最先进的模型。

您可以探索模型的精确度和召回结果，并将模型发布到可以快速访问它并获得预测的端点。

## 3. [Face API](https://azure.microsoft.com/en-us/services/cognitive-services/face/)

这是 Azure 的高级面部识别工具。它支持多种面部功能:

*   人脸检测:检测图像中的人脸
*   人脸识别:搜索和识别人脸
*   人脸验证:检查两张人脸是否属于同一个人
*   面孔相似性:给定一张面孔/一个人，找出相似的面孔/人
*   人脸分组:根据相似性将未识别的人脸分组

分析图像时，Face API 可以提取以下特征:

*   面部位置:显示面部位置的边界框
*   [面部标志](https://docs.microsoft.com/en-us/azure/cognitive-services/face/concepts/face-detection#face-landmarks):面部细节点的集合，包括眼睛位置
*   人脸属性:年龄、性别、头发颜色、面具检测、配饰、情感、面部毛发、眼镜、头部姿势、化妆、微笑

## 4.[视频分析仪](https://azure.microsoft.com/en-us/products/video-analyzer/)

视频分析器是 Azure 的主要视频分析工具。它可以从视频中提取可操作的见解。

集成的人工智能模型提取准确和有意义的数据。他们利用空间分析来实时了解人们在物理空间中的运动。有了元数据，您可以创建基于时间轴的可视化、热图和异常检测。

视频分析仪主要用于工作场所安全、数字资产管理和流程优化。

该工具可以从视频中提取以下图像特征:

*   深度搜索:允许搜索整个视频库。例如，对说出的单词和面部进行索引可以实现在视频中查找某人说出某些单词或两个人被看到在一起的时刻的搜索体验
*   人脸检测
*   名人识别
*   视觉文本识别
*   视觉内容审核
*   场景分割
*   滚动积分
*   等等。

关于音频洞察，视频分析器可以生成以下洞察:

*   音频转录
*   自动语言检测
*   多语言语音识别和转录
*   隐藏字幕
*   噪声降低
*   说话者统计
*   情感检测
*   等等。

# 语言

Azure 有很多 NLP 应用的资源。基本的包括:

*   语言检测
*   关键短语提取
*   情感分析
*   命名实体识别
*   实体链接
*   文本翻译
*   问题回答
*   内容审核

## [路易斯](https://azure.microsoft.com/en-us/services/cognitive-services/conversational-language-understanding/)

这是 Azure 的旗舰 NLP 应用程序，用于语言理解。它实现了用户和对话式人工智能工具(如聊天机器人)之间的语言交互。LUIS 可以解释用户目标，并从会话短语中提取关键信息

借助 LUIS，您可以构建企业级对话机器人、商务聊天机器人或使用语音助手控制物联网设备。

在你的 LUIS 应用程序中，你可以创建触发该动作的意图(BookFlight)和话语。您可以添加实体以获得更好的洞察力，例如检测星期几、目的地、机场名称等。

定义意图和示例话语后，训练应用程序。您可以添加语音和拼写检查的集成。你可以用不同的话语来测试这个应用程序，看看它的表现如何。借助 LUIS 的预测洞察力，您可以实现主动学习并提高应用程序的性能。

当模型足够好的时候，你可以发布它，并将其与 Azure 的 Bot 框架或 QnA Maker 集成。

## 2. [Bot 框架](https://dev.botframework.com/)

这是创建机器人的服务，例如在后台使用 LUIS。

您可以使用开源 SDK 和工具轻松地将您的 bot 连接到流行的频道和设备。

## 3. [QnA 制造者](https://www.qnamaker.ai/)

基于现有的 FAQ URLs、结构化文档和产品手册发布一个简单的问答机器人。

无需任何经验，您可以使用 QnA Maker 的自动提取工具从半结构化内容中提取问答对，包括常见问题解答页面、支持网站、Excel 文件、SharePoint 文档等。

QnA Maker 允许您通过 QnA Maker 门户或使用 REST APIs 轻松设计复杂的多回合对话。它支持主动学习，支持 50 多种语言

# 演讲

## 1.[语音转文字](https://azure.microsoft.com/en-us/services/cognitive-services/speech-to-text/)

使用 Azure 语音到文本，您可以通过最先进的语音识别获得准确的音频到文本转录。您可以将特定的单词添加到您的基本词汇中，或者构建您自己的语音到文本模型。语音转文本可以在云中运行，也可以在容器的边缘运行。

## 2.[文本转语音](https://azure.microsoft.com/en-us/services/cognitive-services/text-to-speech/)

文本到语音允许你用 119 种语言和变体合成超过 270 种神经声音的文本。您可以调整语调、声音类型和许多其他功能。

# 结论

这只是对 Azure AI 资源的一个大概的概述，范围从视觉、语言、语音。

感谢您的阅读，并在 Twitter 上关注我[🚀](https://twitter.com/anebzt)