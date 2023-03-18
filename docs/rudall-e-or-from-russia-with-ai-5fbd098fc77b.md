# ruDALL-e，或来自俄罗斯的 AI

> 原文：<https://towardsdatascience.com/rudall-e-or-from-russia-with-ai-5fbd098fc77b>

## 用于文本 2 图像生成的多模态方法，具有示例和实验

![](img/5b8813aaf2c0637d3af4f78a8530caff.png)

“всянашажизнь—театр”//“整个世界是一个舞台”(ruDALLe) //作者截图

> **TL；博士:** ruDALL-e 与 DALL E 毫无关系，却承载着许多惊喜和激动人心的现象。

![I](img/b5c14894f3920127e09775ba3e711ed3.png)I2021 年开始，OpenAI 以另一组里程碑事件再次震惊世界:多模态 [CLIP](https://openai.com/blog/clip/) 和 [DALL E](https://openai.com/blog/dall-e/) ，能够通过提示将文本直接转换为图像(我写了[关于这种模式](/dall-e-by-openai-creating-images-from-text-e9f37a8fe016?sk=8d54f081ad0d5f4394ffa09ed7b62349)及其[文化影响](https://medium.com/merzazine/ai-dada-and-surrealism-dreaming-about-human-culture-75aecbffcac3?sk=5d08a5a923ee446a25a841beea1e9c51))。即使 DALL E 仍然不能公开使用(*剧透:敬请关注*)，CLIP 还是以开源形式发布——全世界的聪明人都可以将其应用到独特的生成艺术形式中(我的分析即将到来)。

正如 [OpenAI 演示](https://openai.com/blog/dall-e/)中所描述的，DALL E 的创造能力以敬畏和灵感影响了创造性的 AI 社区。

> 这不是创造力的昏睡，而是它的催化剂。

事实上，如果你要求创建一个"*带有蓝色草莓图像的彩色玻璃窗*，你会立即得到一组图像:

![](img/71488605a8fc865bd57d87fc3588a852.png)

[OpenAI 博客](https://openai.com/blog/dall-e/) //作者截图

你会看到各种各样的设计。他们中的一些是照片般真实的，一些是模糊的，风格和构图各不相同。但它给你带来了更接近视觉的想法。

或者，“一个甜甜圈形状的钟”怎么样？

![](img/7a92d5dbb8baf73ea00b1613f3e7ecee.png)

[OpenAI 博客](https://openai.com/blog/dall-e/) //作者截图

OpenAI 仍在为 DALL E 进行优化和用例工作，然后才会向全世界发布这一模型。他们肯定会的——他们发布了 GPT 3，现在没有等待名单。一切皆有可能(问题是“什么时候”)。

# 俄罗斯模特

但是突然间，DALL E 被释放了。

> 好吧，不是突然。而不是真正的 DALL E。

2020 年，由于 GPT-3 是最理想的 NLP 模型，俄罗斯研究人员公布了由俄罗斯储蓄银行 : [**ruGPT-3**](https://sbercloud.ru/ru/warp/gpt-3) 的[人工智能研究实验室创立的项目。它在](https://github.com/sberbank-ai) [600 GB 俄语文本](https://habr.com/ru/company/sberbank/blog/524522/)上进行训练，并以 **760 Mio (ruGPT-3 Large)** 和**13 亿(ruGPT-3 XL)** 参数发布，[比较他们项目](https://habr.com/ru/company/sberbank/blog/524522/)描述中的图表。

如果你将这些数据与开放的 GPT 模型进行比较，你会发现这一点。最大的 **GPT-2** 型号与[**15 亿**参数](https://openai.com/blog/gpt-2-1-5b-release/)一起工作。真正的 **GPT-3** (带达芬奇发动机)运行**1750 亿**参数。

结论是:俄语 **ruGPT-3 XL** 处于 **GPT-2** 的水平上(但是在大量的俄语文本、文章、书籍数据集上进行训练)。显著差异:俄语模型可以生成比 OpenAI(在英语文本上训练)的 GPT-2 质量更好的俄语文本。**数据集事关**。然而，它不能用俄语用烤面包机写情书。它不具备抽象认知或文体转换的能力。

所以，去年年底，一个由 Sberbank 资助的新项目发布了:

> [**ruDALL-e**](https://rudalle.ru/en/)**。**

它与 ruGPT-3 在同一个 *Christofari 集群*上接受训练，获得 Apache 2.0 许可，并且“*堪比 OpenAI* 的英文 DALL-e”(引自 [ruDALL-e 网站](https://rudalle.ru/en/))。

让我们看看，在哪些方面具有可比性。

# 俄罗斯 DALL-e

ruDALL-e 有两种型号:

**Malevich (XL)** —运行在 13 亿个参数上，具有图像编码器(定制 [VQGAN](https://compvis.github.io/taming-transformers/) 型号)，可作为[储存库](https://github.com/sberbank-ai/ru-dalle)和 [Colab 笔记本](https://colab.research.google.com/drive/1wGE-046et27oHvNlBNPH07qrEQNE04PQ?usp=sharing)使用，具有实现的升级 RealERSGAN。

**康定斯基(XXL)** —更好的模型，运行在 120 亿个参数上(像 OpenAI 的 DALL-e)，目前还不能用于测试。

由于 DALL-e 不可公开访问，研究人员试图通过与清华大学的 [CogView](https://github.com/THUDM/CogView) 项目合作，与原始论文( [PDF](https://arxiv.org/pdf/2102.12092.pdf) )一起重新创建模型，该项目也试图重新创建 DALL-e。

因此，由于最初的变压器不可用，他们试图重建它的架构——并将其命名为 **ruDALL-e** 。

# 但是真的管用吗？

幸运的是，有一个公开的 ruDALL-e 的 Colab 笔记本([使用较小的型号，Malevich](https://colab.research.google.com/drive/1AoolDYePUpPkRCKIu0cP9zV7lX5QGD3Z?usp=sharing) )。

关于**潜在空间不和谐**(由剪辑研究员和艺术家 [Advadnoun](https://twitter.com/advadnoun) 创造)的聪明头脑甚至可以优化笔记本——并发现一些奇怪的现象。

## 明显的例子

因此，让我们看看 ruDALL-e 能否生成 DALL-e 原始文章中引用的示例。

**著名的鳄梨椅**

![](img/9c1fa9ec17d5a863e3140b0937a751a1.png)

DALL-e by OpenAI ( [来源](https://openai.com/blog/dall-e/) ) //作者截图

![](img/176a4a6727c5212b422ac99a0231af31.png)

ruDALL-e(马列维奇)//作者截图

在这两种方法的情况下，我们看到模型如何“理解”将现象“*椅子*”与现象“*鳄梨*”在“*形状*”的功能中结合的任务。

就 OpenAI DALL-e 而言，结果看起来比 ruDALL-e Malevich (XL) 更有机。后一种型号选择了特定的特征，如*颜色*或*形状*，但是它的生成不太有机(尽管如此有趣)。即使是更大的型号**康定斯基(XXL)** 也无法再现 DALL-e 的椅子设计质量:

![](img/80271282a25811add5e8e8cae45a56ed.png)

来源:[https://rudalle.ru/en/](https://rudalle.ru/en/)(康定斯基(XXL)) //作者截图

让我们看一些更具体的例子，比如…

**桌子上放着一批钟表。**

![](img/eb709e3438c7c823a4e2b5eb395b3d46.png)

[OpenAI 博客](https://openai.com/blog/dall-e/) //作者截图

在 OpenAI 的 DALL-e 的例子中，我们可以看到它已经知道了更多关于时钟和它们的设计，相比之下**是一个很好的老 BigGAN** 。[你还记得](https://medium.com/merzazine/biggan-as-a-creative-engine-2d18c61e82b8?source=friends_link&sk=498b6f23dfa37347ac952e9d209710c4)，回到 2018 年:

![](img/8db8735404dfbe489c38015991631175.png)

比根时钟(2018) //作者创作

BigGAN 可以重新生成视觉元素，如“圆形”、“带箭头”等。在钟表上，通过 DALL-e 你可以看到数字、12 小时设计等。

![](img/5f60a94606e73eb3a939d424ae7bcd66.png)

ruDALL-e //由作者创建

ruDALL-e(上图)很好地传达了视觉效果——图案的多样性很好。然而，它仍然远离 DALL-e 的现实主义。它有 DALL-e 的视觉容量，但比根的理解等级，IMHO。

毕竟，ruDALL-e 的质量仍然令人惊叹——通过集成的 ERSGAN 放大过滤器，你甚至可以以更大的格式调整图像的大小(你也可以直接使用 ERSGAN 完成[，然后再使用](https://www.tensorflow.org/hub/tutorials/image_enhancing))。

# 俄罗斯风格(数据集很重要)

ruDALL-e 最令人兴奋的方面可能是**训练数据集**。你不会在西方实现中找到这样的内容，因为它源于俄语内容。

如果你用“закатвгороде”(城市的日落)来提示，你会看到一些俄罗斯城镇的美丽景色，主要是莫斯科和圣彼得堡(而不是纽约或伦敦)。在许多照片中，你可以认出莫斯科河、克里姆林宫以及俄罗斯/苏联建筑的古典主义和折衷主义风格。

![](img/abbeb8a031545cd52da7eb6064f51a7f.png)

ruDALL-e 完成，由作者创建

再比如，随着提示“портретгения”(天才的肖像)，你会得到流行的俄语教科书的插图，而不是刻板的好莱坞“疯狂科学家”或背景中黑板上有复杂公式的老师的股票图像。(我可以重新构建这一分类，因为“天才”这个可怜的绰号在俄罗斯教科书中伴随着重要的科学家、作家或艺术家时非常明显)。

![](img/712ef8d593a1ef7f84ce98751bfcba24.png)

ruDALL-e 完成，由作者创建

我们甚至可以在这里找到几个被创造性扭曲的名人。

![](img/5c9d7ce663f3c7866e6a341a5acaceeb.png)![](img/d6b7608a786c21601f4f99c6a7878d0d.png)

查尔斯-奥古斯丁·德·库仑？// ruDALL-e 完成，由作者创建

这幅肖像在物理教科书中关于库仑定律的描述广为人知，而这幅肖像也在俄罗斯教科书中使用。

## 数据集问题？

通过使用 ruDALL-e 进行实验，您将会发现并一点一点地重建模型被训练的数据集的内容。

一年前，[作为 DALL-e 和 CLIP](/dall-e-by-openai-creating-images-from-text-e9f37a8fe016?sk=8d54f081ad0d5f4394ffa09ed7b62349) 由 OpenAI 推出，一位被称为[**advad noun**](https://twitter.com/advadnoun/)**的传奇艺术家和研究者创作了几本 **CLIP Colab 笔记本**，成为 2021 年 Text2Image Art 的*复兴的基础。在这一年中，世界各地不同的艺术家和研究人员用不同的方法对 100 多台笔记本进行了微调(我会写一篇关于它的评论，我保证)。Advadnoun 用他的开创性实验和[不和谐论坛](https://twitter.com/advadnoun/status/1440383057857376257)开了一个 Patreon。聚集在那里的有创造力的人们正在探索更多不同的艺术人工智能模型，这些模型几乎每周都会在世界范围内出现，而 **ruDALLe** 成为了这个社区的焦点。***

**我们可以从 ruDALL-e 接受训练的数据集上部分识别出一些初始图像。**

**SberAI 写了[关于他们的训练数据集](https://habr.com/en/company/sberbank/blog/589673/):**

> **我们的第一步是捕捉 OpenAI 在其出版物中提供的数据(超过 2.5 亿对)以及 Cogview 使用的数据(3000 万对)。这包括:概念说明、YFCC100m、维基百科数据和 ImageNet。然后我们添加了 OpenImages、LAION-400m、WIT、Web2M 和 HowTo 数据集作为人类活动的数据源。我们还包括了我们感兴趣的领域的其他数据集。关键领域包括人、动物、名人、室内、地标和景观、各种类型的技术、人类活动和情感。([来源](https://habr.com/en/company/sberbank/blog/589673/))**

**另一方面，似乎有从网络上刮下来的图像(与 OpenAI 安全数据集方法相反)。通过创建一系列图像，您已经可以看到它了:**

**![](img/16d1ef1da94a0520be21f5b8adad5428.png)**

**ruDALL-e 完成，由作者创建**

**是的，没错。 *iStock 水印*。似乎数据集突出地包含了股票图像公司的预览图像。**

**还有无处不在的迷因，它们似乎给社交网络记忆(和图像数据集)打上了烙印，比如电视频道标识的旧电视屏幕烙印。**

**我们看到著名的迷因有[马龙·白兰度](https://www.meme-arsenal.com/en/create/template/215503)(维托·柯里昂)和[小罗伯特·唐尼迷因](https://imgflip.com/memegenerator/Face-You-Make-Robert-Downey-Jr)(翻白眼)。但是用的是准俄语文本。**

**![](img/e41f9496444d343b9f5275eb011b174c.png)**

**由丹尼尔·鲁斯发现**

**看起来像俄罗斯网站的图像被用于训练数据集。Elle / MichaelFriese 和 DanielRussRuss 发现了另一个奇怪的一致之处:**

**![](img/eb705acd65b764da34cacc77dadd5c31.png)**

**MichaelFriese10 截图**

**第三个图像看起来像一本书的封面，甚至更多，它几乎有 1/1 是从训练数据集转移到完成的。**

**使用图像反向搜索([，这可能是非常鼓舞人心的](/generative-ai-visual-search-as-a-bridge-between-fiction-and-reality-46d2d78ee15?sk=46d879e08f4ec2c2d07f655cefae8b4e))，特别是——在这种情况下——使用 [Yandex 图像搜索](https://yandex.com/images/)，你可以找到原始图像，图像的根源。**

**例如，这张图片:**

**![](img/f5fb8e27da335d9f088933299e83c7b0.png)**

**ruDALL-e 完成，由作者创建**

**![](img/181b1bf22f3f0312a57ab723b91287bb.png)**

**[Yandex 图片搜索](https://yandex.com/images/)，作者截图**

**在这种情况下，视觉完成(“上传的图像”)似乎源于莫斯科地铁的照片(在这种情况下，【Mayakovskaya 站)。**

**我们必须时刻注意人工智能有意或无意的抄袭(就像我们对莎士比亚和亚历山大波普的抄袭一样)。**

**类似的现象我也经历过，GPT-2 对非英语文本的过度训练。在我的例子中，我使用了一个单独的文件:歌德的《浮士德》(原文)。**

**在培训开始时，我们可以观察新的文本(用非常怪异的德语，但已经有了结构，与原戏剧相对应)。经过 7000 步后，补全接近原始文本；他们甚至重复了数据集的内容。**

**有限数据集上的过度训练重新构建了这个数据集。**

**![](img/e7e668a36e89d94fd819a2f3e38050b7.png)****![](img/6af913dc461fda9ecb17a6f6f2157e89.png)**

**在左边:训练步骤 600。右边:训练步骤 7400。//作者截图**

**在 ruDALL-e 的情况下，我想知道，如果训练数据集应该是非常多样化的，那么完整性如何变得与原始图像如此接近？**

## **抽象思维？**

**像上面这样简单的任务，ruDALLe 都可以重新想象。但是如果你尝试抽象的话题，一个令人兴奋的现象出现了。ruDALLe 没有“幻想”这样的问题，而是重新制作了书籍、CD 和网站模型的封面(全部模仿俄语)。**

****《怀旧》****

**![](img/250bd0259d4adf4945c282c3d1c9b34e.png)**

**ruDALL-e //作者截图**

****《前世记忆》:****

**![](img/5834d72c4f179c5cefd37e5a505028eb.png)**

**ruDALL-e //作者截图**

**尽管如此，一些完成的作品还是令人印象深刻。**

## **文化参考**

**如前所述，ruDALLe 接受的是基于俄罗斯的主流图像的训练。**

**如果你深入挖掘俄罗斯的内容/背景，你会发现对历史和文化有趣的重新诠释:**

## **六层(“шестидесятники”)**

**斯大林死后(1954 年)，在[赫鲁晓夫解冻](https://en.wikipedia.org/wiki/Khrushchev_Thaw)(20 世纪 50-60 年代)，苏联出现了一个迷人的文化景观:[六层](https://en.wikipedia.org/wiki/Sixtiers)。部分反映了美国和欧洲的抗议运动，部分是为了将自己从斯大林的创伤中解放出来，苏联知识分子——艺术家、作家、音乐家——开始发展另类的第二文化，与社会主义现实主义和(后)斯大林时代的爱国主义紧张对立。**

**不幸的是，一场独特的运动在苏联/俄罗斯之外鲜为人知，但它带来了文化多样性、新鲜感和开放思想，影响了世界六分之一人口的几代创作者。**

**这是一部由真正的诗人和艺术家参与的电影中的一小段片段，你可以感受一下这种氛围:**

**1964 年在莫斯科理工博物馆的诗歌朗诵会**

**在我用[自动点唱机](/jukebox-by-openai-2f73638b3b73?sk=003ba0e0d6416a4456c7a890fddf9461)进行的音乐实验中，当 [AI 重新构建六阶](https://medium.com/merzazine/latentvoices-95fc47559dbc?sk=7a1ad262edc028d9ffe1c0b6db8d81fc)的声音时，我偶然发现了一个诡异的现象。**

**正如你所看到的，这种背景很难表达，特别是如果它将由人工智能重新创建。ruDALLe(由于其预先培训)创建了带有提示“шестидесятники”(六层)的各种图像:**

**![](img/0b4e61f4a8ab37511493488e92ccd4ab.png)**

**ruDALL-e //作者截图**

**我可能会过度解读它，但我看到了一些纪实照片，带有类似表演场景的团体照片。第一个图像类似于著名的六层图像(也用于他们的先锋杂志“青年”(“юность")):**

**![](img/dad3f199360e3362c70a4488b2541403.png)**

**Dmitri Bykov 关于“第六层”的选集封面，使用了青年标志(Stasys Krasauskas)**

**也许我过度解释了我的观察，但是我越深入兔子洞，我就有越多的考古发现。**

## **ruDALLe 有创意吗？**

**虽然人们仍然在讨论机器的创造力，但我自己有一个明确的答案(自 2015 年以来与人工智能共同创造):是的。**

**接下来，我想展示一些 ruDALLe 关于创造力的有趣发现(所有的提示都是用俄语输入的，我为这篇文章翻译了它们)。**

**提示:**村里的蒙娜丽莎****

**![](img/ac636fb2aea65e9e9de5c00b26d0a760.png)****![](img/1569e335395b01a04a90f22dc65f868d.png)****![](img/4f4ee843fbf3c8e3059b942f22cc1cf9.png)**

**ruDALL-e //作者截图**

**即使不是每一部作品都展示了乡村的乔康达，其衍生作品也很有趣。即使是达芬奇的“sfumato”(烟熏色渐变)在大多数图像中都起作用。此外，她把脸转向不同方向的方式也很棒。**

**提示:**克里姆林宫的神奇宝贝****

**![](img/8a4f2ffd1beb4a92c1f09cb9a18ef32a.png)****![](img/4bf1a5c6fca8e0606f53217f34072af4.png)****![](img/2efc4db3f104efa2d75662ff2a5eb716.png)**

**ruDALL-e //作者截图**

**如果说第一张图片说明了叙事“皮卡丘在莫斯科参观观光”，那么第二张图片则需要更多的内容知识:在这里你看到了对[鲍里斯·叶利钦](https://en.wikipedia.org/wiki/Boris_Yeltsin)发型的完美刻画。在第三幅图中，你看到一个类似普京的人，对一些可爱的东西感到高兴，可能是——神奇宝贝在克里姆林宫游荡。**

**提示:**橱窗里的女孩**。**

**![](img/fa147c590942f335247163cfb21a83d9.png)**

**ruDALL-e //作者截图**

**这个提示会生成一系列扭曲的人类图像(ruDALLe 仍然无法像 StyleGAN3 一样生成逼真的人脸)。但有些作品蕴含着艺术力量，比如这幅:**

**![](img/cfa33bb316ea034b0a76880daa7b762e.png)**

**ruDALL-e //作者截图**

**我猜想 ruDALLe 训练的数据集包含艺术摄影——或者 Instagram 图像。**

**提示:**《自然》****

**![](img/82c478959b05ddcb0a672d1227bec76e.png)****![](img/c159a5fd4ce21276b931fa39059052c9.png)****![](img/0587df8aee57da0436a4f83343b83936.png)**

**ruDALL-e //作者截图**

**自从 BigGAN 以来，我们已经知道人工智能能够逼真地再现自然。**首先是**，自然特征**包含在数据集内的大多数** **图像**中(即使只是作为背景主题)。其次，是我们的大脑在购买自然图像，即使看了第二眼后这些图像并不令人信服。当我们看到人脸图像时，我们会立即发现 AI 所做的所有扭曲，因为我们的大脑被训练来检测面部特征。历史上，如果树或云看起来很奇怪，对我们的大脑来说并不重要。**

**提示:“**街头搞笑怪兽。**”**

**![](img/e5c2eee82fe535fcab1ba72e51d00d65.png)**

**ruDALL-e //作者截图**

**我是说，很迷人，不是吗？**

**提示:“**建构主义生日派对**”**

**![](img/0b5856147ccde528729e60721ec5c50b.png)****![](img/f19751bda9157feb5c5abd4671978438.png)****![](img/ee9bb984e862ae1f0157c7d67c98c479.png)**

**ruDALL-e //作者截图**

**提示:“**空间里的恋人**”**

**这些作品传达了苏联艺术家的科幻风格(其中还有宇航员阿列克谢·列昂诺夫)，蓝色的太空，地球大气层的朦胧烟雾，未来的理想主义和乌托邦氛围。**

**![](img/0b44974e7efbe8e9a11096fb260db302.png)****![](img/78ab279b87b51eb8c4c2ce27ef6c191f.png)****![](img/5bc9c30dd4c87996308ea226ce3e05c6.png)**

**ruDALL-e //作者截图**

**提示:“**云中列宁**”**

**毫无疑问，ruDALLe 数据集应该包含许多俄罗斯/苏联历史的镜头。探索这种特定的预训练模型来重新解释视觉叙事是很有趣的。**

**这个提示给了我几个非常有趣的补充，在这里你可以认出列宁(即使中间的图像描绘了不同苏联政治家的组合)。**

**![](img/5439d788534279794db1d7e3a9132e2e.png)****![](img/e2b7195e0e8bf45e1688a440b6ee7ea4.png)****![](img/7c2ae5c3bfac06af03373cb1c86faa45.png)**

**ruDALL-e //作者截图**

**但是下面的结果让我起了鸡皮疙瘩。不仅仅是重新想象列宁。这是一个独特的视角，是苏联末期概念主义者(如[科马尔&梅拉米德](https://de.rbth.com/kultur/81898-komar-melamid-soz-art))和其他艺术家异议运动的特征。**

**![](img/745fd1bf8b929c8109f3cd990abb0ae0.png)**

**ruDALL-e //作者截图**

**这幅从整个历史视角直到苏联解体的戏剧性肖像是独一无二的。**

**提示:“**复活**”**

**在这里，ruDALLe 把提示分配给了它占优势的内容领域:宗教，基督教背景。可视化是预先确定的，由于附属:希腊东正教的图标。**

**![](img/43dc66fe098258d307c4840b8df11d9e.png)****![](img/3b8f15c35f43f03f83eb698108481f88.png)****![](img/e7cc4ddf1a295e2c52e882a36740bbfe.png)**

**ruDALL-e //作者截图**

**但是在规范的视觉特征之外的重新解释也在这里重新提出:**

**![](img/a8632c6bdc8514588b643b676aff79cb.png)**

**ruDALL-e //作者截图**

# **摘要**

**ruDALLe 不是 DALLe，它不包含符号转换、隐喻能力和 OpenAI 多模态方法的灵活性。**

**尽管如此，这是一个体面和自力更生的模型，它产生鼓舞人心的和独特的视觉效果，适合灵感迸发，头脑风暴和跳出框框思考。**

**![](img/37c1c992eb6780bc4eca093f481d688b.png)**

**ruDALL-e //作者截图**