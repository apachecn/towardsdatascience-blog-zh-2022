# 探索创建数字艺术的中途 V4

> 原文：<https://towardsdatascience.com/exploring-midjourney-v4-for-creating-digital-art-4d20980a96f7>

## 深入探讨流行的文本到图像创建系统的功能和选项

![](img/4c5436c04a606f01a786f7fc84280c1b.png)

上排:**达芬奇的《蒙娜丽莎》**(左)、**被米杜尼程式化为“表现主义绘画”**作者(中)、以及**“立体派绘画”**作者(右)、下排:**克洛德·莫内的《郁金香与里恩斯堡风车的田野》**(左)、被米杜尼程式化为“表现主义绘画”作者(中)、以及**“立体派绘画”**作者(右)

在我上个月的[文章](https://medium.com/towards-data-science/digital-art-showdown-stable-diffusion-dall-e-and-midjourney-db96d83d17cd)中，我测试了三个文本到图像的生成系统，Stable Diffusion、DALL-E 和 Midjourney，看看哪一个最适合创建数字艺术。在大多数情况下，我只使用了三个系统的默认选项，尽管许多功能可以用来制作不同的、通常更好的图像。

在本文中，我将带领您深入了解 Midjourney 中用于创建数字艺术的功能和选项，包括目前作为公共测试版提供的版本 4。v4 太棒了！

以下是我将在文章中涉及的内容列表:

1.  中途基础知识
2.  探索版本
3.  创作变奏和混音
4.  带有文本提示的样式转换
5.  中途混搭
6.  渲染选项
7.  升迁选项

# 中途基础知识

Midjourney 是一种文本到图像的创建服务，它使用 Discord 服务器作为主要用户界面，Discord 服务器是一个主要由游戏玩家使用的群组消息平台[1]。[注册](https://www.midjourney.com/)是免费的，可以免费创作 200 张左右的图像。之后，您可以点击查看快速入门指南[中列出的定价选项。](https://midjourney.gitbook.io/docs/)

**基础设置**

一旦我设置好并前往一个“新手”房间，我通过键入**/设置**并按两次回车键看到了主要选项。出现了以下界面。

![](img/808bb5b0cac27fbe180a99e54e96cafe.png)

**中途设置**，图片作者

这些只是一些基本的设置。完整的列表可以在命令行选项[这里](https://github.com/midjourney/docs/blob/main/user-manual.md)找到。

**创建图像**

使用默认设置，我通过键入 **/imagine** 创建了四个缩略图，按 enter 键调出 UI，并输入下面的提示，“用彩色圆圈画抽象画。”

![](img/e37fd34d4dcde3bc0d9d668b886f986c.png)

**中途创作图片**，作者图片

系统考虑了一下，然后开始创建缩略图。大约 20 秒后，我得到了结果。

![](img/31852cc06f7aae48c440ad89eb681a6f.png)

**“彩色圆圈抽象画”中途缩略图，**作者图片

根据我的提示，我得到了四个 256x256 像素的缩略图。他们看起来很不错。下面的“U”按钮用于将选定的图像放大到 1024x1024，“V”按钮创建选定图像的变体，蓝色 spinny 图标将重新运行该过程，创建四个新的缩略图。

我喜欢右上角的图片(数字 2)，所以我按下 **U2** 按钮来放大那张图片。大约 10 秒钟后，我得到了结果。

![](img/2bab2e97942e086ef84fc98288fe27a4.png)![](img/ae4eaae0f8f30083378aceaaed68bdad.png)

**“彩色圆圈抽象画”中途升级结果**(左)、**升级图像**(右)，图片作者

这是我在升迁过程完成时看到的，有各种进一步处理的选项。当我检查图像时，它出现在我的中途账户中，如右图所示。

# 探索版本

Midjourney 在 2022 年 3 月发布了第一个图像生成服务版本。版本 2 在 4 月份发布，版本 3 在 7 月份发布。版本 4 的测试版于 2022 年 11 月 10 日发布。

如上所述，您可以使用**/设置**命令来选择版本 1 至 4。我注意到版本 4 创建了 512x512 的初始渲染，而以前的版本创建了 256x256 的初始图像。为了测试所有四个版本，我发送了提示“画一个戴着帽子的法国人在露天咖啡馆喝酒”,看看会得到什么。

![](img/aa3e4a2d28b095e62dc1e39289eb5fc7.png)![](img/b1bdc9d6e2aabdc20693e83865152acc.png)![](img/c10e65f76fc52674b5cde091874c8213.png)![](img/c5e53cdc64dd6e8b009d34198b903267.png)

**《一个戴着帽子的法国人在露天咖啡馆喝葡萄酒的画像》渲染于中途的 V1** (左上) **V2** (右上) **V3** (左下) **V4** (右下)，图片由作者提供

左上角的四个缩略图是使用版本 1 渲染的，右上角是版本 2，左下角是版本 3，右下角是版本 4。哇！质量差别好大啊！随着每个版本的推出，图像都在稳步改善。第 4 版的图像看起来非同寻常！在下一节中，我将展示一些定量的结果。

## 对比相似性测试

上个月我在[发表了一篇文章，我研究了各种衡量艺术作品美感的标准。我发现了一种使用 OpenAI [2]的 CLIP 模型的技术，可以用来评估美学质量。当我将图像嵌入与正面和负面文本提示的嵌入进行比较时，如“坏艺术”和“好艺术”，我发现利用相似性的差异可以产生合理的结果。我还计算了一个提示相似性度量，以查看使用相似逻辑的图像与提示的匹配程度。下图显示了上述 16 张图片的两个指标。](https://medium.com/towards-data-science/digital-art-showdown-stable-diffusion-dall-e-and-midjourney-db96d83d17cd)

![](img/94c3c819c083f87b029aaaeea343470d.png)

**即时相似性与审美质量，**作者图表

您可以看到使用较新 Midjourney 的渲染在审美质量指标(垂直轴)和即时相似性(水平轴)上比旧版本排名更高，V4 比其他版本高出一截。两个值得注意的异常值是 v2–1(非常好的提示相似性)和 v2–3(两个指标都非常差)。再次向上滚动查看图像，这些指标似乎保持得相当好。这个测试的数学和 Python 在我的 GitHub 库[这里](https://github.com/robgon-art/contrastive-similarity-test)。

# 创作变奏和混音

在创建了带有文本提示的四个缩略图后，系统允许您使用 V1-V4 按钮进行其他更改。例如，对于提示“一个男人和一个女人在雨中的城市街道上撑着伞”，系统生成四个缩略图。在右上角选择 V2 后，它会生成四个变量。请注意，我在这个实验和接下来的所有实验中使用了版本 4。

![](img/ae5a8cf53f7aeb1b83fcd3335505d682.png)![](img/a6c24bfe494ee8ad1f53f83535f5ad36.png)

**《一个男人和一个女人撑着雨伞走在雨中的城市街道》初始渲染**(左)和**右上**(右)的变化，图片由作者提供

这些变化看起来都很不错。从 Midjourney 的版本 3 开始，您可以在创建变体时修改文本提示。该功能被称为混音模式，可在设置中使用。例如，我使用混音模式在提示中添加了以下文本:“1880 年代”、“1950 年代”和“未来”这是结果。

![](img/1f8debd04ca2305ee9dcebb6a71db09f.png)![](img/91d8f242a14db22490d70c58d316e503.png)![](img/75b69cdd3ef674c8549930ff8f278def.png)

**《一个男人和一个女人撑着伞走在雨中的城市街道上》，图中有“1880 年代”**(左)、**【50 年代】**(中)、**【未来】**(右)，图片作者

果然，系统以与指定时间段相关联的独特视觉外观来呈现图像。

# 带有文本提示的样式转换

类似于使用混音模式来创建变体，您可以从网络上可用的图像开始，并添加文本来使图像风格化。我使用了 **/imagine** 提示并粘贴到基础绘画的 URL 中，并使用文本指定了样式。

例如，从蒙娜丽莎、莫奈的风景画和塞尚的静物画开始，我使用 Midjourney 将这些作品风格化为“表现主义绘画”和“立体主义绘画”这是结果。

![](img/9a7e2b5096d829fcce962bb399d24a54.png)![](img/c0e41e0c5b31296748e7830169fddc39.png)![](img/963bf96505e0476d8ff0d454659a7c82.png)

**达芬奇的《蒙娜丽莎》**(左)**米杜尼的《表现主义绘画》**作者(中)**中途的《立体派绘画》**作者(右)

![](img/1251b75afc29cf23353ef09b862ae599.png)![](img/305b33a8bc623aace03bfe527775b4e4.png)![](img/a1ea92e4cb3868347217210db308933c.png)

**克洛德·莫内(左)的《郁金香与里恩斯堡风车的田野》**；作者(中)的**米杜尼“表现主义绘画”**；作者(右)的**中途“立体派绘画”**

![](img/d0e392685c84a476311aa5ff3a98d831.png)![](img/d1afb8e73894f228cad5efebf67a1bdc.png)![](img/d0ee955b2360a3c22572287a1ce4a0bf.png)

**窗帘、水壶和水果**保罗·塞尚(左)**米杜尔尼《表现主义绘画》**作者(中)**中途《立体派绘画》**作者(右)

原作在左边，表现主义和立体主义风格在右边。结果似乎相当不错。我喜欢这个系统除了改变风格之外，还可以自由地改变一点构图。请注意蒙娜丽莎的形象是如何被放大和缩小的，以及一些额外的物品如花和酒瓶是如何被添加到静物画中的。

# 中途混搭

该系统有一个很好的特性，除了文本提示之外，允许用户指定两到三个图像作为输入。我用这个特性创建了我称之为中途混搭的东西。我在没有提示的情况下粘贴了两个公共领域绘画的 URL，下面是 Midjourney 创建的内容。新图像在中间。

![](img/8bed1ac2224f7c1c1aed973b1ad4992c.png)![](img/fdf154a5de37794fd49522551636a38d.png)![](img/1d1675ecbb05c359b58c105e65ce697c.png)

**戴珍珠耳环的女孩**约翰内斯·维米尔(左)**中途混搭**作者(中)**柔琳夫人摇摇篮**文森特·梵高(右)

![](img/5ae73e84fdb2875229a4fdbbeb43926f.png)![](img/5b666ea89bf383650d68662b5d7c591d.png)![](img/b3257528a794d724558f568bd1233c7d.png)

**风景**皮埃尔·奥古斯特·雷诺阿(左)**中途混搭**作者(中)**埃加列尔斯附近的高德加林麦田**
文森特梵高(右)

![](img/839ad4d663df512951913e2877fe8e20.png)![](img/43ae9dfc262d367bce3c5defd808c40a.png)![](img/1a424180fcc3ec70e965ad25b06ade22.png)

**Pieter Claesz 的《鲱鱼、葡萄酒和面包的静物》**(左)，作者(中)的**中途混搭**，保罗·塞尚的**静物、水罐和水果**(右)

![](img/3963cea14de60318c34d3fecaabe4390.png)![](img/3ae48db576eae3c9872e99501aa5166d.png)![](img/68d120fde46369d61c237e6847499d0f.png)

**黄-红-蓝**作者瓦西里·康丁斯基(左)**中期画作**作者(中)**红色、黄色和蓝色的构图 C(第三幅)** 作者皮耶·蒙德里安(右)

好的，这些看起来很酷。你可以看到系统是如何从两幅源图像中提取关键的构图元素，并找到一种新的方式来表达它们的。在对这种技术进行试验后，我发现这种方法最适用于主题相似的来源。当我用完全不同的源进行尝试时，结果是不可预测的，并且通常在视觉上不连贯。

此外，中途 V4 模型仅支持 1:1 的图像纵横比。希望他们会发布一个版本来创建具有纵向和横向纵横比的图像，就像他们对早期版本所做的那样。

## 当代作品的混搭

为了看看这在当代作品中会是什么样子，我联系了波士顿地区的四位艺术家，他们允许我使用他们最近的一些作品。 [Katie Cornog](https://www.katiecornog.com/) 是水彩艺术家， [Noah Sussman](https://noahsussman.weebly.com/) 在画布上作画， [Anna Kristina Goransson](https://www.annakristinagoransson.com/) 创作毛毡雕塑， [Georgina Lewis](http://www.birdfur.com/) 是在混合媒介中工作的装置艺术家。中间一栏显示了我制作的四个 Midjorney 混搭，混合了每个艺术家的两个作品。

![](img/eb0d3bb7a651c855ee7a309f2f17ef9c.png)![](img/e322547440f2cf06d54e7b55b3b676b4.png)![](img/57c7bb61fb3c8af40d86751d0f8a8b1d.png)

***桑迪点秋*** 作者凯蒂·科尔诺格(左)**作者中途混搭**(中) ***作者凯蒂·科尔诺格***

![](img/c12d1ec0c599c95a2d64f4015d067a91.png)![](img/538ded4bdbab620dd36f093f2fa07d1c.png)![](img/c504b2b951c37cefd86a6ec70d3a18bf.png)

**工作室场景**作者诺亚·苏斯曼(左)**中途混搭**作者(中) **S *超越月食*** 作者诺亚·苏斯曼(右)

![](img/50e70ac229a0451cca10bbda48b18c55.png)![](img/cce30c36bbd52bb3b4035453db1d4e1b.png)![](img/707c81fbbb57dc32399bb9677f5129c0.png)

**忧郁之美**安娜·克里斯蒂娜·戈兰松(左)**中途混搭**作者(中)**寻找**安娜·克里斯蒂娜·戈兰松(右)

![](img/388048e3df78e65996de9c6dfd085f8e.png)![](img/47ef609dc7025439d0a8b12e33530b22.png)![](img/f092248e30d82646d9180617b2e6d93a.png)

**最近的共同祖先**作者乔治娜·刘易斯(左)**中途混搭**作者(中)**拉帕奇尼的后代**作者乔治娜·刘易斯(右)

结果再一次令人惊叹。这四个混搭看起来都像是原创艺术家创作的。两位艺术家告诉我，他们想这样做。

# 渲染选项

正如我上面提到的，中途有很多渲染选项。然而，为了进行比较测试，我学会了如何检索和使用用于生成图像的随机种子。这有点隐蔽，但在我生成一个图像后，我对我的图像创建了一个“反应”，然后选择了“信封”表情符号。

![](img/d3a1c60b4652d8d2f0632b33c9f6e349.png)

**用不和谐的信封表情符号发送反应**，图片作者

这触发了机器人向我发送一条包含运行种子的消息，就像这样。

![](img/544a91133fe9f588b1043126a574a976.png)

**图片种子直接留言，**图片作者

然后我使用 **- seed 6534** 参数来生成具有不同选项的相同图像。例如，下面是使用提示“春天波士顿公共花园的绘画”的渲染，质量设置为 0.5，1.0(默认)与 2.0。

![](img/e61ad0d3911a3e27be3e732491454402.png)![](img/3653ea129ad43506e0629857b17d9279.png)![](img/66e782cb3c0efd9f2abf1f3dc900f92f.png)

**“春天的波士顿公园绘画”，质量 0.5** (左)、 **1.0** (中)、 **2.0** (右)，图片作者

有趣的是，我们可以看到不同质量选择下的图像变化有多大。中间的那棵黄树大致保持不变，但花、建筑和人都在四处移动。这很微妙，但右边质量为 2.0 的图像似乎有最少的“问题”区域。

# 升迁选项

Midjourney 还提供了用于放大的选项，可以改善最终图像的细节。对于这个测试，我使用了提示符“蒸汽朋克旋转电话”，质量设置为 2.0。我不得不使用“种子”技巧来测试三个升迁选项，lite、regular 和 beta。(声明一下，这是 24863 号种子。)下面是结果，以及旋转拨号器的特写。

![](img/33aa715d188903bb87c54ae699961551.png)![](img/f09bdf6e6b794215faa0cbe94a74e93a.png)![](img/a7db064151c96bcc36f7a10cd9af3eb8.png)![](img/c1841eed0e5ba00c32a54f8f9ab7d951.png)![](img/5f5f5e21c70631776aa4d18b962eee48.png)![](img/9918ac805c5b381afc45178bec61aeef.png)

**“蒸汽朋克旋转式手机”，尺寸设置为 Lite** (左)**、Regular** (中)**和 Beta** (右)**，细节在第二行**，图片由作者提供

这三张图片都显示了一些细节，但是右边的 beta 放大看起来更有序一些。比如拨号器上的内圈，看起来形成的更真实。

# 结论

在过去的两年半时间里，我一直在研究使用人工智能来生成数字艺术。我可以告诉你，Midjourney V4 是迄今为止我见过的最好的系统。希望，他们将发布这个版本的全套功能(长宽比，风格数量，等等。)此外，一些似乎缺失的功能正在修复和着色，就像 DALL-E 中一样。

# 源代码和 Colabs

运行图像对比相似性测试的源代码可在 [GitHub](https://github.com/robgon-art/contrastive-similarity-test) 上获得。我在[麻省理工学院开源许可](https://raw.githubusercontent.com/robgon-art/contrastive-similarity-test/main/LICENSE)下发布了源代码。

![](img/9e4901377b9bab81387105d5b424327f.png)

麻省理工学院开源许可证

# 感谢

我要感谢詹尼弗·林和奥利弗·斯特瑞普对这个项目的帮助。

# 参考

[1]中途 https://midjourney.gitbook.io/docs/[的](https://github.com/midjourney/docs)

[2]a .拉德福德等人的剪辑，[从自然语言监督中学习可转移的视觉模型](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf) (2021)

为了无限制地访问 Medium 上的所有文章，[成为](https://robgon.medium.com/membership)的会员，每月支付 5 美元。非会员每月只能看三个锁定的故事。