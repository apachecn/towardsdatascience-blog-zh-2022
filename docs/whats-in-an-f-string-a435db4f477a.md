# F 弦里有什么？

> 原文：<https://towardsdatascience.com/whats-in-an-f-string-a435db4f477a>

## 概述 Python 组合字符串和变量的方法，以及为什么应该使用它。

![](img/0d11d9182df68ad76caa883659a8db69.png)

穆罕默德·拉赫马尼在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

这是我讨论 Python 独特特性系列的第五篇文章；一定要检查一下 [lambdas](/whats-in-a-lambda-c8cdc67ff107) 、 [list comprehensions](/whats-in-a-list-comprehension-c5d36b62f5) 、[dictionary](/whats-in-a-dictionary-87f9b139cc03)和 [tuples](/whats-in-a-tuple-5d4b2668b9a1) 上的前四个。

如果您是一个相对较新的 Python 用户，您可能遇到过需要将一些您定义的变量添加到字符串中的情况。不知道如何方便地做到这一点，您可能会得到一些不太漂亮的代码，如下所示:

```
>>> fruit = 'persimmon'
>>> sport = 'lacrosse'
>>> print('I eat ' + fruit + ' and I play ' + sport)
I eat persimmon and I play lacrosse.
```

它起作用了，但这不是一个值得一看的景象。更别说打字都是一种痛苦。幸运的是，有一个更好的方法。

## 到底什么是 f 弦？

![](img/1bcc2a7f88bb2c38a91fae5ef1abf135.png)

照片由[湄木](https://unsplash.com/@picoftasty?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

f-string 是 Python 简化的字符串格式化方法。这方面的一点背景:

*   通常，您可以将字符串格式化视为一种构造字符串并在字符串中的指定位置插入用户定义的变量的方法(上面的代码显示了一个这样的示例)。
*   f 字符串并不是 Python 中完成这项任务的唯一方式。它还提供了带有“%”操作符和一个有用的`.format()`函数的 C 风格格式化，但是我在这里不详细讨论这些。我是 f 弦的拥护者，因为它们在形式和功能上是最具 Pythonic 风格的。

要定义一个 f 字符串，只需在字符串的第一个引号前加上一个`f`，并使用`{}`在字符串中的任意位置插入变量。让我们看一个例子。下面，我们定义与上面相同的字符串，但是这次使用 f 字符串:

```
>>> fruit = 'persimmon'
>>> sport = 'lacrosse'
>>> print(f"I eat {fruit} and play {sport}.")
I eat persimmon and play lacrosse.
```

有点酷，对吧？然而，他们的能力不止于此。从技术上来说，当我说 f-strings 可以将*变量*插入字符串时，我撒了一点谎。他们实际上可以插入整个*表达式*，比如函数调用:

```
>>> def my_func():
...     return "We are never ever getting back together."
...
>>> print(f"Taylor sings {my_func()}")
Taylor sings We are never ever getting back together.
```

就我个人而言，我觉得这非常方便。

## f 弦为什么有用？

那么，为什么要用 f 弦呢？这个问题的答案很简单:它们简单、易读、优雅。换句话说，它们是蟒蛇。当您可以一次写出整个字符串，表达式用花括号清楚地分隔时，为什么要将一堆引号和加号挤在一起呢？这对代码编写人员和读者来说都更容易。

让我说得更具体一点。假设你正在处理熊猫的一些数据，如果你正在读这篇文章，你可能至少偶尔会这样做。但是，现实世界的数据往往是杂乱的。您的行到处都是，您的列有一半是未命名的，并且有大量的空值分布在各处。

经过一番探索，您意识到您的字符串分散在不同的列中:

```
 col1          col2             col3
0           This is  an example of  a broken string
1  A second example           of a     a broken one
2         I promise        this is     the last one
```

您不需要修复数据帧；您只需要一种快速的方法将组合字符串打印到终端中，这样您就可以分析一些东西。回车，f 弦:

```
>>> print(f"My combined string: {df.loc[0, 'col1']} {df.loc[0, 'col2']} {df.loc[0, 'col3']}")
My combined string: This is an example of a broken string
```

任务完成。

现在，我并不是说这是解决这个问题的最佳或唯一的方法。您可以通过使用`.apply()`函数来修复数据帧，或者提出一些其他新颖的解决方案。然而，上面举例说明了一个用例，如果您希望轻松快速地完成一项任务，它可能会有所帮助。

一般来说，我建议在两种情况下使用 f 弦:

*   如果您只是浏览一些数据，并需要执行简单的字符串操作来收集一些信息
*   如果您正在编写涉及格式化字符串的代码——这似乎是显而易见的，但我想强调的是，这种方法比上面提到的其他方法更具可读性和简洁。因此，你应该使用它。

一如既往，目标是让您的生活更简单，让您的代码更好。

下次见，伙计们！

**想擅长 Python？** [**获取独家，免费获取我简单易懂的攻略**](https://witty-speaker-6901.ck.page/0977670a91) **。想在介质上无限阅读故事？用我下面的推荐链接注册！**

[](https://murtaza5152-ali.medium.com/?source=entity_driven_subscription-607fa603b7ce---------------------------------------) [## 穆尔塔扎阿里培养基

### 阅读媒介上穆尔塔扎·阿里的作品。华盛顿大学的博士生。对人机感兴趣…

murtaza5152-ali.medium.com](https://murtaza5152-ali.medium.com/?source=entity_driven_subscription-607fa603b7ce---------------------------------------) 

*我叫穆尔塔扎·阿里，是华盛顿大学研究人机交互的博士生。我喜欢写关于教育、编程、生活以及偶尔的随想。*