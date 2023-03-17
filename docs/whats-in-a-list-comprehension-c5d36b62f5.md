# 什么是列表理解？

> 原文：<https://towardsdatascience.com/whats-in-a-list-comprehension-c5d36b62f5>

# 什么是列表理解？

## 概述 Python 对循环的替代以及为什么应该使用它们。

![](img/6306467d91b773ef7d0dee72401526a6.png)

托尔比约恩·赫尔格森在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

*更新:一定要看看本文* *第二部分* [！](https://levelup.gitconnected.com/whats-in-a-list-comprehension-part-2-49d34bada3f5)

这是我的系列文章中的第二篇，讨论 Python 中的独特特性以及使用它们的好处。请务必在此查看第一部分:[Lambda 中有什么？](/whats-in-a-lambda-c8cdc67ff107)

Python 作为一种编程语言之所以受欢迎，很大程度上是因为它能够以简单紧凑的方式完成复杂的任务。大多数语言中存在的常见编程思想通常在 Python 中有独特的实现，这有助于编写干净、可读的代码。列表理解是我最喜欢的 Python 特有的特性之一，因为它们提供了传统 for 循环的单行替代方案。

## **列表理解到底是什么？**

列表理解是 Python 中的一种语法形式，允许程序员在一行中循环并转换一个 iterable。术语*列表理解*来源于两个特征:1)用于编写这种理解的类似列表的语法，以及 2)列表理解的输出总是 Python 列表的形式。

理解列表理解的最简单方法是将它们与传统的 for 循环进行对比。让我们来看看下面这个具体的例子，在这个例子中，我们要对列表中的每个数字求平方:

```
lst = [1, 2, 3, 4, 5]
squared_lst = []
for number in lst:
    squared_lst.append(number * number)
```

和...相对

```
lst = [1, 2, 3, 4, 5]
squared_lst = [number * number for number in lst]
```

关于上面的两个代码片段，需要注意一些要点:

*   与 for 循环不同，list comprehensions 自动将其输出放入一个列表中。在第一个代码片段中，我们必须显式地创建一个新的列表，然后在 for 循环中向其追加内容。相比之下，在第二个代码片段中，我们简单地将 list comprehension 的输出赋值给我们的新变量。
*   一开始阅读列表理解可能会令人困惑，因为如果你习惯于 for 循环，顺序就会颠倒。但是，它使用相同的基本概念。在这个列表理解中，语句`for number in lst`可以被认为对应于传统的 for 循环。它遍历列表中的每个元素，每次都将临时变量`number`绑定到元素。在列表理解的第一部分，`number * number`告诉 Python，在将元素添加到我们的新列表之前，我们要对它求平方。
*   与 for 循环一样，原始列表`lst`保持不变。list comprehension 将转换后的元素放入一个新的列表中，然后该列表被赋值给我们的变量`squared_lst`。

## 列表理解还能做什么？

上面的例子很酷，但是如果列表理解不能做任何更复杂的事情，那就太可悲了。这里我想强调两个特别的特性:1)过滤列表的能力和 2)模拟嵌套 for 循环的能力。

对于 Python 程序员来说，根据某种条件从列表中删除元素是一个相当常见的用例。例如，假设您希望过滤一个数字列表，只保留那些小于 100 的数字。您可以这样做:

```
>>> lst = [1, 324, 44, 500, 7]
>>> [number for number in lst if number < 100]
[1, 44, 7]
```

更好的是，您甚至可以对剩余的数字进行运算！列表理解允许您在一条语句中过滤和转换所有内容:

```
>>> lst = [1, 324, 44, 500, 7]
>>> [number * number for number in lst if number < 100]
[1, 1936, 49]
```

一个重要的注意事项:如果您使用上面的语法在单个列表理解中过滤和转换，请记住 Python **首先过滤，然后转换**。

Python 程序员的第二个常见用例是将一个 for 循环嵌套在另一个循环中。这里有一个这样的例子:

```
>>> nested_lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
>>> for inner_lst in nested_lst:
        for number in inner_lst:
            print(number)
1
2
3
4
5
6
7
8
9
```

通过列表理解，您可以将整个嵌套的 for 循环缩减为一行:

```
>>> nested_lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
>>> **[print(number) for inner_lst in nested_lst for number in inner_lst]**
1
2
3
4
5
6
7
8
9
[None, None, None, None, None, None, None, None, None]
```

我知道，这看起来有点混乱。一开始我也很困惑。但是如果你仔细阅读这一行，它遵循与嵌套的 for 循环相同的逻辑:首先，我们编写`for inner_lst in nested_lst`来访问外部列表中的每个单独的列表，然后我们编写`for number in inner_lst`来访问各自内部列表中的每个数字。和嵌套的 for 循环完全一样；我们只是删除了冒号和换行符。

一个小提示:出现 9 次`None`的列表只是列表理解的输出(因为 Python 中的`print`函数返回`None`)。

## 为什么列表理解有用？

好吧，所以列表理解很酷，但是拥有它们有什么意义吗？我们不能一直使用 for 循环吗？

从技术上讲，是的——但这并不意味着列表理解毫无用处。正如我在上一篇关于 lambdas 的文章中提到的，Pythonic 编程的原则之一是编写简单、简洁、可读的代码。列表理解是产生这种高质量代码的极好方法。

以下是使用列表理解的几种情况:

*   **使用**[**MapReduce**](https://en.wikipedia.org/wiki/MapReduce)**模型**快速处理数据。假设您正在处理一些数据，并且希望在编写利用专业 MapReduce 框架的密集型代码之前快速获得一些见解。列表理解提供了一种简单的方法来模拟这个模型，以便运行小型测试。您可以快速过滤您的数据，并在理解的范围内映射一些转换函数，然后您可以将 Python 的`reduce`函数应用于结果列表。
*   **在 Jupyter 中模拟随机实验**。如果你是一名数据科学家(或者正在接受培训成为一名数据科学家)，你可能已经使用`sample`和`random`以及 Jupyter 笔记本中的所有好东西运行了一些测试。一般来说，在评估某个假设时，使用 for 循环是为了模拟实验的数千次运行。下一次，你可以使用列表理解，让你的代码简单一点。

当然，这只是两个例子。总的来说，我建议明智地使用列表理解，以使你的代码更简洁，可读性更好。如果你是一名 Python 程序员，这是你应该非常熟悉的思维方式——如果你是这门语言的新手，这是你应该学好的思维方式。

下次见，伙计们！

**想擅长 Python？** [**获取独家、免费获取我简单易懂的指南点击**](https://witty-speaker-6901.ck.page/0977670a91) **。想在介质上无限阅读故事？用我下面的推荐链接注册！**

[](https://murtaza5152-ali.medium.com/?source=entity_driven_subscription-607fa603b7ce---------------------------------------) [## 穆尔塔扎阿里培养基

### 阅读媒介上穆尔塔扎·阿里的作品。华盛顿大学的博士生。对人机感兴趣…

murtaza5152-ali.medium.com](https://murtaza5152-ali.medium.com/?source=entity_driven_subscription-607fa603b7ce---------------------------------------) 

*我叫穆尔塔扎·阿里，是华盛顿大学的一名博士生，研究人机交互。我喜欢写关于教育、编程、生活以及偶尔的随想。*