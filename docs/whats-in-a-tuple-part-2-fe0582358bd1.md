# 元组里有什么？—第二部分

> 原文：<https://towardsdatascience.com/whats-in-a-tuple-part-2-fe0582358bd1>

## 既然您已经学习了 Python 中的元组，我将通过一个例子来展示它们为什么有用。

![](img/db816d35937399c08b0e8b5be379bdd1.png)

克里斯里德在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

这是我之前的文章[的后续文章，什么是元组？](/whats-in-a-tuple-5d4b2668b9a1)在最初的文章中，我简单地提到了使用元组可能有用的编程环境——但没有深入细节。在本文中，我将通过一个详细、具体的例子向您展示我的意思。这个会很短很甜。

具体来说，我之前暗示过元组在使用字典时是多么有用。你可能会问，为什么要使用字典？嗯，这完全是一个独立的问题——长话短说，字典是一种数据结构，在熊猫中有一系列的用途(非常双关)。因此，如果您从事数据处理和分析，了解如何使用它们是很好的。

继续，让我回到我在上一篇文章中建议的例子。考虑以下情况:你在教一个班，期末作业——占课程成绩的大部分——是一个小组项目。当您评分时，您意识到您想要存储每个组的分数，但是您也不想丢失成员的个人信息。换句话说，您将通过其中的两个人来唯一地识别每个组。

使用字典解决上述问题的第一次尝试可能如下所示:

```
proj_grades = {['Aaron', 'Ella']: 'A', ['Jackie', 'Elaine']: 'B', ['Arif', 'Julie']: 'F'}# Arif and Julie have really been slacking off
# or maybe they're just not interested in your class
# that's valid.
```

不幸的是，这导致 Python 用下面的不愉快消息向您问候:

```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unhashable type: 'list'
```

换句话说，因为列表在 Python 中是可变类型，所以它们不能被散列(这是一个在表中随机放置一个键的时髦词)。这对于我们的目的来说并不重要，但是如果你好奇的话，哈希表是 Python 字典的底层数据结构。

无论如何，这带来了一点问题，因为您仍然需要存储数据。你将如何解决这个问题？元组！因为它们的功能几乎与 list *相同，除了*它们不是可变的，所以它们在 Python 字典中是完全有效的键:

```
>>> proj_grades = {('Aaron', 'Ella'): 'A', ('Jackie', 'Elaine'): 'B', ('Arif', 'Julie'): 'A++'}
>>># Through the power of tuples, Arif and Julie are now
# acing the class.
```

现在您已经很好地组织了数据，您可以非常方便地将其转换为数据帧:

```
>>> pd.DataFrame(proj_grades, index=[0]) Aaron Jackie  Arif
   Ella Elaine Julie
0     A      B   A++
```

诚然，这不是 pandakind 所知的最漂亮的数据帧，但这是一个开始。重要的是，您已经以一种可读的方式收集了数据，现在可以按照您希望的方式进行清理、处理和分析。使用元组的目的是让您简洁明了地编写 Python 代码，并且仍然能够实现您的最初目标。当用 Python 编程时，这是一个你应该始终致力于体现的原则——因此我在我的文章中不断重申它。

目前就这些。下次见，伙计们！

**想擅长 Python？** [**获取独家，免费获取我简单易懂的攻略**](https://witty-speaker-6901.ck.page/0977670a91) **。**