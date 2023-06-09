# 我用人工智能画出了 10 个最美的英文单词

> 原文：<https://towardsdatascience.com/i-used-ai-to-paint-the-10-most-beautiful-english-words-a687bfc33cff>

## 你想看看他们长什么样吗？

![](img/35041d9b292a2b7b57696d10a801f340.png)

在 Shutterstock 上由 [Golubovy](https://www.shutterstock.com/es/g/Golubovy) 拍摄的[照片](https://www.shutterstock.com/es/image-photo/creative-hobby-contemporary-art-course-talented-1379582222)

我正在参与新兴的人工智能驱动的艺术场景。现在你也可以成为艺术家了。

我最近发现了人工智能生成的艺术世界。研究人员和艺术家正在利用人工智能来推动他们的创作。在生成神经网络的帮助下，他们可以创建与数据集中的图像完全不同的新图像。

起初，艺术家不能直接参与他们的创作。GANs 可以生成新图像，但不能接受指令。当 OpenAI 发布 CLIP 的权重时，这种情况发生了变化。 [CLIP](https://openai.com/blog/clip/) 是作为一个代表性的神经网络创建的，用于寻找文本描述和图像之间的对应关系。

研究人员利用它的能力来选择与给定图像最匹配的句子。瑞安·默多克第一个，凯瑟琳·克劳森第二个找到了将 CLIP 的技巧与甘的创造力结合起来的方法。CLIP 允许艺术家引导生成(创作)过程。他们可以在文本-图像表示的巨大潜在空间中“搜索”，以找到哪个图像最符合给定的文本提示。

默多克创作了[大睡眠 colab 笔记本](https://colab.research.google.com/github/levindabhi/CLIP-Notebooks/blob/main/The_Big_Sleep_BigGANxCLIP.ipynb) (BigGAN+CLIP)，克劳森随后创作了更受欢迎的 [VQGAN+CLIP 笔记本](https://colab.research.google.com/github/justinjohn0306/VQGAN-CLIP/blob/main/VQGAN%2BCLIP_(z%2Bquantize_method_with_augmentations%2C_user_friendly_interface).ipynb)。仅仅通过提示这些模型——就像我们对 GPT 3 号所做的那样——我们就能创造出触及想象力极限的图片，挑战我们对艺术的理解

用查理·斯内尔的话来说，这是一个“新兴的艺术场景”，他就这个主题写了一篇很棒的文章

![](img/b5a2920b689f4e52afa99d352feb599b.png)![](img/d2cf45cb44a85fff1a2df37d80d3efaf.png)

提示音:[比根+夹子](https://colab.research.google.com/github/levindabhi/CLIP-Notebooks/blob/main/The_Big_Sleep_BigGANxCLIP.ipynb)上的[当风吹](http://When the wind blows)(鸣谢:[赖安·默多克](http://Ryan Murdock))和[VQGAN+夹子](https://colab.research.google.com/github/justinjohn0306/VQGAN-CLIP/blob/main/VQGAN%2BCLIP_%28z%2Bquantize_method_with_augmentations%2C_user_friendly_interface%29.ipynb)上的[一棵树枝变弱的树](https://twitter.com/advadnoun/status/1399896134420615170?s=20)(鸣谢:[凯瑟琳·克劳森](https://twitter.com/RiversHaveWings))

# 重新定义艺术性

我搜索了测试这些“promptive”生成模型的方法，并找到了 Wombo 的 web 应用程序 [Dream](https://app.wombo.art/) (没有透露他们使用的是哪种架构，但很可能是 CLIP +某种类型的 GAN)。它允许用户生成多达 20 种不同风格的无限图像。生成一幅图像大约需要 10-20 秒，与此同时，你可以看到人工智能如何将绘画从最初的模糊笔触发展到最终的抛光作品。

梦的意象有一种独特的美。与 DALL E 的作品相比，这些图像并没有描绘出定义明确、具体的物体或人物。相反，梦创造了一种具体的感觉，从未离开抽象美的安全。如果你仔细看这些图像，你会失去你认为你从远处看的物体的边界。

![](img/f59bc3e581e7e11ce5713553385a1196.png)

提示:村庄(风格玫瑰金)。如果你仔细看，一些以山景为背景的房子变成了一种抽象的混合形式，涂上了暖色调和柔和的颜色。作者在梦里创造的。

在被这个人工智能的绘画的独特性迷住之后，我有了一个想法:我要测试它在语言上的美感。我会让它画出最美的英语单词，而不是提示随机的句子。我搜索了前 10 名，但在意识到美丽不能被一致认可后，我决定把视觉和语言的美丽都考虑进去。(如果你想查其他我没画的漂亮字，这里有出处:[语法上](https://www.grammarly.com/blog/most-beautiful-english-words/)和 [BuzzFeed](https://www.buzzfeed.com/danieldalton/bob-ombinate) 。)

我直接向模型提示了一些单词，没有修饰的提示——那些可以自我解释的。对于其他人，人工智能更难理解，我决定使用他们的定义。CLIP 在数据集中看到的单词越多，它就能更好地表示文本-图像对。像“pluviophile”这样的词很少见，所以 CLIP 还没有形成一个精确的视觉概念。我还根据最符合意思的样式来选择样式。

# 来自语言美的视觉美

我对 Dream 将单词与非常准确的视觉图像联系起来的能力印象深刻，尽管大多数单词在本质上非常抽象。调整提示和样式让我觉得自己有点像艺术家。我理解为什么现在这么多的数字艺术家使用这些工具——助手——来增强他们的艺术。

人工智能不会取代艺术家，但它肯定会重新定义我们对艺术家这个词的理解。

下面，我为每个单词展示了三幅画，每幅都有不同的风格。字义和形式都很美——这些画很好地诠释了它们。

## **迷恋:迷恋另一个人的状态**

![](img/6f31c317177e231ee1b066af271a8ce4.png)![](img/09ecc65dbef601452552347104489df6.png)![](img/a9a8d194eac98212bdebbb17c97da7bb.png)

Limerence(风格:活力/神秘/黑暗幻想)。作者在梦里创造的。

摸头。身体锁在一起，跳着性感和裸体的舞蹈。两个人互相看着对方，周围是一个无关紧要的世界。梦捕捉到了理智和爱情之间的微妙区别。在所有三张图片中，一个人在另一个人之上，这可能反映了体验 limerence 的人和它的欲望对象之间的不平等关系。

## **机缘巧合:事件以有利的方式偶然发生**

![](img/cf51d212a3c5dc11f36c9a9f6bd404d8.png)![](img/0cd97b450cbcef96a5baf5a39300037c.png)![](img/5d2dcdd6bc9139191df8834d579698ab.png)

意外之喜(风格:活力/巴洛克/出处)。作者在梦里创造的。

梦在这里以颜色和形式的混合传达了意外之喜的内在积极性。这是非常有趣的出处风格的图片如何描绘福克，乐观的福龙在永无止境的故事。如果一个虚构的人物能体现出意外之喜，福克肯定是一个很好的候选人。

## **Petrichor:雨后宜人的泥土气息**

![](img/f93859e86c80dda33eae6a5e0450def1.png)![](img/ee3c230c8cfab47cdb3e517db89c694e.png)![](img/41089164d89fe14b7c64b0c3ba523dcb.png)

彼得里奇(风格:黑暗幻想/通灵/出处)。作者在梦里创造的。

在写这篇文章之前，我从未听说过彼得里奇。然而，我非常清楚它所代表的感觉。中间的图片对我来说是这种感觉的视觉定义。很晚了，你正穿过附近的森林回家。一直在下雨。你可以看到最绿的叶子和尚未过滤到地面的水晶般的水滴。空气中弥漫着一股明显的香味。那是彼得里奇。

## 孤独:一种隔离或隔绝的状态

![](img/93e85cd8df78128ea147b55a1d8ea381.png)![](img/1901b56e8ac444c5fb7389c34c49fc27.png)![](img/b9415a50a9869b20de5057d6104fb1a2.png)

孤独(风格:巴洛克/合成波/黑暗幻想)。作者在梦里创造的。

孤独让我想到自己是无限宇宙中的一粒微尘。这是一种艰难但强大的感觉，有一种独特的美。梦在合成波绘画中很好地捕捉了这种感觉。一个人走在似乎是悬崖和大海的附近。她独自一人在深不可测的广阔世界中，但并不孤独:远处，群山、云彩和夕阳都在注视着。

## **极光:清晨的黎明**

![](img/6f9fa8e2ec2f4dcd48480e6c49a60b8b.png)![](img/32d86f100690bc3a66d74c5c7b5c05de.png)![](img/8127d3a320e1f0a241a70190192988c6.png)

极光(风格:synth wave/wuthercuhler/黑暗幻想)。作者在梦里创造的。

Aurora 是其中一个不言自明的词。我们对这个词都有一个非常清晰的视觉表达——如果可能的话，会更漂亮。梦知道极光是什么，它在三幅画中表现出来。星星、夜晚和绿-蓝-紫的组合占据了主导地位。

## 田园诗般的:极其快乐、和平或如画的

![](img/03dfd1b481530356ed4fac81fd89789b.png)![](img/91461c7620e90d793222761aa4687dc9.png)![](img/9fe68d1e3671df724123a45fb7b2464f.png)

田园风格(风格:奇幻艺术/金玫瑰/wuthercuhler)。作者在梦里创造的。

田园诗般的是一个有着明确含义的词，对我们每个人来说可以有非常不同的形式。在第一张照片中，Dream 描绘了一个宁静的村庄，周围是绿色的田野、远处的群山和一个清澈的湖泊。在第二张照片中，似乎是一艘船在落日的背景下扬帆远航。第三个，一个被鲜花和棕榈树环绕的小房子，可能是一个遥远的天堂。

## 欣快:极度快乐的感觉或状态

![](img/0d944d3e4777ba91b555d8c531c96e1c.png)![](img/03807a5970d5b7184d5f66cb6955ad0b.png)![](img/10e15d846cb958b2b842def22d23ac76.png)

欣快感(风格:黑暗幻想/充满活力/出处)。作者在梦里创造的。

梦代表了强烈融合暖色的欣快感。即使在黑暗幻想风格中，这种风格通常使绘画充满模糊和低能的意象，欣快感战胜了黑暗，给了我们一个非常微妙的描绘，似乎是一个女人包裹在粉红色，紫色和橙色的柔软衣服中。

## 红杉:一种红杉树

![](img/2842651e8672c14e0526ab48d903957b.png)![](img/e897893e98aad667facb7ddbc3eb423a.png)![](img/c7b1378a80fe78db2bddda99a383adc6.png)

红杉(风格:神秘/合成波/黑暗幻想)。作者在梦里创造的。

用[劳伦·比奇](https://www.grammarly.com/blog/most-beautiful-english-words/)的话说，红杉的魅力在于它是“一个七个字母的单词，有字母 Q 和所有五个元音。”作为这个列表中唯一一个具体的单词，Dream 更看重这个单词具体的视觉意义，而不是风格。每幅画仍然反映了风格的特质，但没有失去画面的主角:一个非常高的红色树干，树枝让人想起圣诞树。

## 缥缈的:极其精致、轻盈，不属于这个世界

![](img/c686080bb6050adb697f81e99e828de6.png)![](img/6af2a9703eb1ed38a5875eae4661e1b1.png)![](img/66c59b9093a8a9f211a11822f4e39f4e.png)

缥缈(风格:黑暗幻想/神秘/通灵)。作者在梦里创造的。

我一直认为缥缈是半透明的，有灵性的。现在在这个世界上的东西，但即将离开，因为它不属于这里。Dream 通过透明的服装和柔和的颜色捕捉到了这一概念。画中没有人，更像是半人的幽灵，质地如此之薄，他们似乎在我们眼前消失了。

## 嗜雨者:雨的爱好者

![](img/e9ff3793f2084117e409b414c9e4d080.png)![](img/ac65395468def4873154adceff364a19.png)![](img/8561ef0ae501ce835b22d2526548bea8.png)

Pluviophile(风格:黑暗幻想/通灵/充满活力)。作者在梦里创造的。

一个女人正坐在潮湿的地板上。一条粉红色的裙子遮住了她的身体，而整个城市在后面看着她。但是如果你仔细观察，图像中并没有真正的雨。梦把房子的墙壁和窗户变形为细长的形状，就像落下的水滴。尽管如此，我们还是能感觉到这个女人正在享受这场雨，让它充满这一刻。

# 最终想法

人工智能创造的艺术已经开始存在。正如编码和编写人工智能一样，人们可以利用这些模型的可能性来推动他们的创作。AI 艺术的特殊性在于，绘画本来就是无边无际的。有一些规则，但是一旦你掌握了这些规则，你就可以找到无限的方法去打破它们。

人工智能生成器提供了一波新的工具，将帮助艺术家揭示想象创作的未知形式——比如给最美丽的英语单词赋予视觉意义。

如果你已经读到这里，可以考虑订阅我的免费双周刊 [***【明天的思想】***](https://mindsoftomorrow.ck.page/) *！每两周一次关于人工智能和技术的新闻、研究和见解！*

*您也可以直接支持我的工作，使用我的推荐链接* [***这里***](https://albertoromgar.medium.com/membership) *成为中级会员，获得无限权限！:)*