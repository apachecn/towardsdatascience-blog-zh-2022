# 元组里有什么？

> 原文：<https://towardsdatascience.com/whats-in-a-tuple-5d4b2668b9a1>

## 概述 Python 对列表的替代以及为什么应该使用它们

![](img/042a8d697f56d0bde6961be10fd3f049.png)

克里斯里德在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

这是我讨论 Python 独特特性系列的第四篇文章；一定要检查一下 [lambdas](/whats-in-a-lambda-c8cdc67ff107) 、 [list comprehensions](/whats-in-a-list-comprehension-c5d36b62f5) 和[dictionary](/whats-in-a-dictionary-87f9b139cc03)上的前三个。

啊，臭名昭著的元组。这是什么？我们真的需要它吗？它实际上和列表有什么不同吗？这些只是我当计算机科学入门助教时，我的学生问我的问题的一个子集。

而且理由很充分。乍一看，元组似乎是没有实际用途的冗余数据结构。然而，这只是一种误解。一旦你深入理解了元组，你就能够将它们与列表区分开来，并确定在什么情况下使用它们。

让我们开始吧。

## 到底什么是元组？

元组是 Python 中的一种数据结构，可用于存储对象集合。它们可以包含任何其他内容:数字、字符串、列表，甚至其他元组！要创建元组，只需使用常规括号并用逗号分隔元组。正如我们在下面的例子中看到的，它们的功能与列表非常相似:

```
>>> my_tuple = ('words', 76, ['list'], ('another', 'tuple'))>>> my_tuple
('words', 76, ['list'], ('another', 'tuple'))>>> my_tuple[0]
'words'>>> my_tuple[2]
['list']>>> my_tuple[3]
('another', 'tuple')>>> len(my_tuple)
4
```

然而，元组和列表之间有一个非常重要的区别:元组是**不可变的**，而列表是可变的。

可变性是一个复杂的主题，值得单独写一篇文章，但是基本思想如下:如果一个对象的值在定义了之后能够改变*，那么这个对象就是**可变的**。更简单地说，您可以更改可变对象内部的值，但不能更改不可变对象内部的值。下面是一个使用列表和元组的具体例子:*

```
>>> my_lst = [1, 2, 3, 4, 5]>>> my_tuple = (1, 2, 3, 4, 5)>>> my_lst[2] = 77>>> my_lst
[1, 2, 77, 4, 5]>>> my_tuple[2] = 77
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
```

正如您所看到的，试图在事后更改元组会让 Python 不高兴，因此它会向您显示一个错误。

## 元组为什么有用？

那么，你究竟为什么要使用元组呢？毕竟，它只是看起来像一个有更多限制的列表，所以坚持使用列表并就此收工不是更好吗？

这可能很诱人，但是元组比列表有一个巨大的优势——这个优势在某些应用程序中可以产生很大的不同。如果你读了我上一篇关于字典的文章，你会记得我曾简单地说过字典中的键不能是可变对象。

然而，在很多情况下，当构建字典时，您可能需要将*组*与特定值相关联。例如，考虑这样一种情况，您正在跟踪班级中合作项目的成绩。您希望将合作伙伴收集到一个键中，而不丢失每个合作伙伴的身份信息，但是您不能使用列表，因为 Python 会抛出一个错误。对于这种情况，元组是完美的解决方案。

此外，如果您想减少 bug 并编写可维护的代码，元组也是一个不错的选择。如果你知道一个项目集合应该保持固定，那么使用一个元组可以防止一些其他的程序员来破坏那些应该保持不变的东西。如果不加检查，将可变列表留在代码中会导致问题。

我将把这个留给你:仅仅因为元组没有被充分利用并不意味着它们没有价值——许多 Python 程序员只是不明白如何正确地使用它们。现在你知道了，你可以把你的编程带到下一个层次。

下次见，伙计们！

**更新** : [在这里阅读本文的第二部分！](/whats-in-a-tuple-part-2-fe0582358bd1)

**想擅长 Python？** [**在这里**](https://witty-speaker-6901.ck.page/0977670a91) **获得独家、免费获取我简单易懂的指南。想在介质上无限阅读故事？用我下面的推荐链接注册！**

<https://murtaza5152-ali.medium.com/?source=entity_driven_subscription-607fa603b7ce--------------------------------------->  

*我叫 Murtaza Ali，是华盛顿大学研究人机交互的博士生。我喜欢写关于教育、编程、生活以及偶尔的随想。*