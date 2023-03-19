# 如何在 NLP 项目中执行数据扩充

> 原文：<https://towardsdatascience.com/how-to-perform-data-augmentation-in-nlp-projects-5c4b86c25035>

## 利用文本攻击库进行数据扩充的简单方法

![](img/1e43e0e3b43de239b56495f18cc3f938.png)

图片由 [Gerd Altmann](https://pixabay.com/users/geralt-9301/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=1989152) 从 [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=1989152) 拍摄

在机器学习中，为了实现强大的模型性能，拥有大量数据是至关重要的。使用一种称为数据扩充的方法，您可以为您的机器学习项目创建更多数据。数据扩充是一组技术，用于管理在现有数据基础上自动生成高质量数据的过程。

在计算机视觉应用中，增强方法非常普遍。例如，如果你正在做一个计算机视觉项目(例如图像分类)，你可以对每张图像应用许多技术:移动、修改颜色强度、缩放、旋转、裁剪等等。

如果您的 ML 项目有一个很小的数据集，或者希望减少机器学习模型中的过度拟合，建议您可以应用数据扩充方法。

> “我们没有更好的算法。我们只是有更多的数据。”-彼得[诺威格](https://research.google/people/author205/?ref=hackernoon.com)

在自然语言处理(NLP)领域，语言所具有的高度复杂性使得扩充文本变得困难。扩充文本数据的过程更具挑战性，不像有些人预期的那样简单。

在本文中，您将学习如何使用名为 [TextAttack](https://github.com/QData/TextAttack?ref=hackernoon.com) 的库来改进自然语言处理的数据。

# 什么是 TextAttack？

TextAttack 是一个 Python 框架，由 [QData 团队](https://qdata.github.io/qdata-page/?ref=hackernoon.com)构建，目的是在自然语言处理中进行对抗性攻击、对抗性训练和数据扩充。TextAttack 具有可独立用于各种基本自然语言处理任务的组件，包括句子编码、语法检查和单词替换。

TextAttack 擅长执行以下三个功能:

1.  对抗性攻击(Python: `**textattack.Attack**`，Bash: `**textattack attack**`)。
2.  数据增强(Python: `**textattack.augmentation.Augmenter**`，Bash: `**textattack augment**`)。
3.  模型训练(Python: `**textattack.Trainer**`，Bash: `**textattack train**`)。

**注意:**对于本文，我们将关注如何使用 TextAttack 库进行数据扩充。

# 如何安装 TexAttack

要使用这个库，请确保您的环境中有 python 3.6 或更高版本。

运行以下命令安装 textAttack。

```
pip install textattack
```

**注意:**一旦安装了 TexAttack，就可以通过 python 模块或命令行运行它。

# 文本数据的数据扩充技术

TextAttack 库有各种增强技术，您可以在 NLP 项目中使用这些技术来添加更多的文本数据。以下是您可以应用的一些技巧:

**1。它通过将字符替换成其他字符来扩充单词。**

```
from textattack.augmentation import CharSwapAugmentertext = "I have enjoyed watching that movie, it was amazing."charswap_aug = CharSwapAugmenter()charswap_aug.augment(text)
```

[“我很喜欢看那部电影，太棒了。”]

增强器将单词**“电影”**换成了**“om vie”**。

**2。DeletionAugmenter** 它通过删除文本的某些部分来扩充文本，以生成新的文本。

```
from textattack.augmentation import DeletionAugmentertext = "I have enjoyed watching that movie, it was amazing."deletion_aug = DeletionAugmenter()deletion_aug.augment(text)
```

[“我看过了，太棒了。”]

这种方法删除了单词**“享受”**来创建新的扩充文本。

**3。EasyDataAugmenter** 这是用不同方法的组合来扩充文本，例如

*   随机交换单词在句子中的位置。
*   从句子中随机删除单词。
*   在随机位置随机插入随机单词的随机同义词。
*   随机用同义词替换单词。

```
from textattack.augmentation import EasyDataAugmentertext = "I was billed twice for the service and this is the second time it has happened"eda_aug = EasyDataAugmenter()eda_aug.augment(text)
```

['我为这项服务开了两次账单，这是第二次'，T25'我为一项服务开了两次账单，这是第二次'，T26'我为这项服务开了两次账单，这是第二次'，T27'我为这项服务开了两次账单，这是第二次']

正如您从增强文本中看到的，它根据应用的方法显示了不同的结果。例如，在第一个扩充文本中，最后一个单词已经从**“发生”**修改为**“发生”**。

**4。它可以通过用 WordNet 词库中的同义词替换文本来扩充文本。**

```
from textattack.augmentation import WordNetAugmentertext = "I was billed twice for the service and this is the second time it has happened"wordnet_aug = WordNetAugmenter()wordnet_aug.augment(text)
```

['我为这项服务开了两次账单，这是第二次了']

该方法将单词**“发生”**改为**“经过”**，以创建新的增强文本。

**5。创建自己的增强器** 从`textattack.transformations` 和`textattack.constraints`导入变换和约束允许你从头开始构建自己的增强器。以下是使用`WordSwapRandomCharacterDeletion`算法生成字符串扩充的示例:

```
from textattack.transformations import WordSwapRandomCharacterDeletion
from textattack.transformations import CompositeTransformation
from textattack.augmentation import Augmentermy_transformation = CompositeTransformation([WordSwapRandomCharacterDeletion()])
augmenter = Augmenter(transformation=my_transformation, transformations_per_example=3)text = 'Siri became confused when we reused to follow her directions.'augmenter.augment(text)
```

[“当我们再次听从她的指示时，Siri 变得困惑起来。”当 e 重复使用她的指示时，Siri 变得很困惑，
“当我们重复使用 Siri 来遵循 hr 的指示时，它变得很困惑。”]

实施`WordSwapRandomCharacterDeletion`方法后，输出显示不同的增强文本。例如，在第一个扩充文本中，该方法随机移除单词“ **confused”中的字符“**o”**。**

# 结论

在本文中，您已经了解了数据扩充对于您的机器学习项目的重要性。此外，您还学习了如何使用 TextAttack 库对文本数据执行数据扩充。

据我所知，这些技术是完成 NLP 项目任务的最有效的方法。希望它们能对你的工作有所帮助。

您还可以尝试使用 TextAttack 库中其他可用的增强技术，例如:

*   嵌入增强器
*   检查表增强器
*   克莱尔增强器

如果你学到了新的东西或者喜欢阅读这篇文章，请分享给其他人看。在那之前，下期帖子再见！

你也可以在 Twitter [@Davis_McDavid](https://twitter.com/Davis_McDavid?ref=hackernoon.com) 上找到我。

*最后一件事:在以下链接中阅读更多类似的文章*

[](https://medium.com/geekculture/how-to-speed-up-model-training-with-snapml-b2e24b546fe5)  [](https://medium.com/geekculture/top-5-cloud-migration-strategies-you-need-to-know-fb1d92ed3c8a)  [](https://medium.com/geekculture/top-5-reasons-why-companies-are-moving-to-the-cloud-c3a609332125)  

*本文首发于* [*此处*](https://hackernoon.com/how-to-perform-data-augmentation-in-nlp-projects) *。*