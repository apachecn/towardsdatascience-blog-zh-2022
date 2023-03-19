# 来自 Python 的 Inventor 的 4 个改进代码的简单技巧

> 原文：<https://towardsdatascience.com/4-simple-tips-from-pythons-inventor-to-improve-your-code-5429297505a9>

## 利用该语言最博学的专家之一提供的易于实施的课程

![](img/bd4146afe375d5b5983585d62a664ecc.png)

[活动发起人](https://unsplash.com/@campaign_creators?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

您可能不知道，Python 安装附带了一首美妙而有用的诗，而且是免费的。要访问它，你需要做的就是在终端中键入`import this`。然后，您会看到以下内容:

> “Python 的禅”，作者蒂姆·彼得斯
> 
> 漂亮总比难看好。
> 显性比隐性好。简单比复杂好。
> 复杂总比复杂好。
> 扁平比嵌套好。
> 稀不如密。
> 可读性很重要。特例不足以特殊到违反规则。
> 虽然实用性胜过纯粹性。错误永远不会悄无声息地过去。
> 除非明确消音。
> 面对暧昧，拒绝猜测的诱惑。应该有一种——最好只有一种——显而易见的方法来做这件事。除非你是荷兰人，否则这种方式一开始可能并不明显。
> 现在总比没有好。
> 虽然永远也不会比现在*好。如果实现很难解释，这是个坏主意。
> 如果实现起来容易解释，这也许是个好主意。名称空间是一个非常棒的想法——让我们多做一些吧！"

这首诗的作者是 Tim Peters，他是一位著名的软件工程师，也是 Python 编程语言发展的主要贡献者之一。尽管这是一种既有趣又幽默的语言怪癖，但这首诗也有一些很好的建议。

我们来看看能学到什么。

## **请让你的代码易于阅读**

> 可读性很重要。

以下是 Python 中完全有效的代码块；它实现了它的目标，运行时没有任何错误:

```
x =   'All Too Well'
if x == 'All Too Well': print("Taylor Swift rocks")
else:
               print("You need to re-evaluate your life choices.")
```

事实证明，只要您与空格和制表符保持一致，Python 的解释器并不真正关心您将代码间隔开多宽或多均匀。

但你知道谁在乎吗？三个非常重要的人，如果你选择编写风格不正确的代码，他们都会讨厌你:

1.  你的老师/导师:这些人必须要么给你的代码打分，要么审查你的代码，他们不会因为不得不多盯着看几分钟来找出从哪里开始而感到兴奋。
2.  你的同事:他们需要构建你的代码。如果你不必要地让他们的生活变得更加艰难，他们不会喜欢你。
3.  **你未来的自己**:你很可能会发现自己在未来几天拿出自己的代码，重新开始工作。在那一点上，你真的会痛恨过去的自己写了一堆乱七八糟的东西，而不是清晰、干净和可维护的代码。

这么多人讨厌你的生活不是一种有趣的生活，所以编写符合 [Python 详细且写得很好的风格指南](https://peps.python.org/pep-0008/)【1】的代码确实对你更有利。确保你的缩进匹配，你的变量名合理，并添加一些适当的注释来引导读者:

```
my_favorite_song =   'All Too Well'# This block of code ensures the my_favorite_song variable has
# the correct value.
if my_favorite_song == 'All Too Well':
    print("Taylor Swift rocks")
else:
    print("You need to re-evaluate your life choices.")
```

你的导师、同事和未来的自己都会感谢你。

## 当你这么做的时候，把它变漂亮

> 漂亮总比难看好。

这里有一个重要的问题需要回答:什么是美丽的代码？

这要看上下文。Python 是一种为可读性和简单性而设计的语言，因此最简单的代码(仍然容易理解)也是最漂亮和优雅的，这是显而易见的。

有许多相关的例子，但我最喜欢的一个是新程序员中常见的冗余构造。它包括返回`True`或`False`,这取决于我们感兴趣的变量的值。请考虑以下情况:

```
# This code returns whether or not my code is beautifuldef my_function(is_code_beautiful):
    if is_code_beautiful == True:
        return True
    else:
        return False
```

原则上，这段代码没有任何问题。事实上，对于新的程序员来说，它可以说是一个有效的教学工具，因为它提供了思考条件语句实际上要做什么的方法。

然而，它可以用一种更简单(也更漂亮)的方式来写。由于变量本身已经是一个布尔值(真或假)，我们可以用下面的代码达到同样的效果:

```
# This code returns whether or not my code is beautifuldef my_function(is_code_beautiful):
    return is_code_beautiful
```

简单。优雅。蟒蛇皮。

太美了。

## 复杂不就是复杂吗？

号码

> 复杂总比复杂好。

很多人把*复杂*和*复杂*这两个词互换使用，那么我们就从两者的区别说起吧。它们在口语中可能有相似的含义，但在编程上下文中却不相同。

*复杂的*代码可能很难理解，但通常是出于必要，因为底层算法本身很复杂(没有人编写过比从头开始编写算法更简单的代码)。*复杂的*代码是不必要的复杂，特别是因为有一种更简单的方法来实现相同的解决方案。

让我们看一个例子。假设我们有以下深度嵌套的列表:

```
data = [[[0, 255, 255], [45, 0, 0]], [[0, 0, 0], [100, 100, 100]]]
```

顺便提一句，您可能认为嵌套如此之深的列表没有实际用途，但事实上这就是图像在 Python 中的存储方式——作为像素行，其中每个像素是由红、绿、蓝(RGB)值组成的 3 元素列表。

在我目前担任助教的课程中的一个项目中，学生们必须编写一个操作程序，通过颠倒所有行中像素的顺序来水平翻转照片。

当我为这个项目做准备时，这是我最初想到的解决方案:

```
for row in data:
    for i in range(len(row) // 2):
        temp = row[i]
        row[i] = row[len(row) - 1 - i]
        row[len(row) - 1 - i] = temp
```

起初，我认为这是一个非常漂亮的反转列表的解决方案。也就是说，直到我的教授指出我可以这样做:

```
for row in new_pixel_data:
    row.reverse()
```

这种差异令人震惊。

有人可能会说第二种解决方案在原理上仍然很复杂，因为反转列表是一项相当复杂的任务。然而，非常明显的是，与我的第一个手动解决方案相比，它要简单得多，因此也更容易理解。

请注意，您的代码是否复杂很大程度上取决于上下文。在上面的例子中，我的初始代码很复杂，因为有更干净的代码可以达到同样的目的。如果没有反转 Python 列表的简单方法，那么我的代码会很好(如果仍然复杂的话),仅仅是因为它是最好的选择。

一个很好的经验法则是:在你编写超级复杂的代码之前，检查一下是否有人已经做了一些工作来使你的生活变得更容易。

## 检查错误！

> 错误永远不会无声无息地过去。
> 除非明确消音。

有趣的是，这是一个在编程入门课上经常被忽略的话题。我自己的第一门编程课程只是简单地提到了捕捉错误，从未明确地教授如何去做。这是一项重要的技能，主要是因为它能让你的程序在出错时给用户更多的细节(或者甚至保持程序平稳运行)。

在 Python 中处理错误最简单的方法是通过 *try-except* 块。您像平常一样在`try`中编写代码，如果有错误，代码开始运行`except`块:

```
>>> try:
...     print(x)
... except:
...     print("You never defined your variable.")
...
You never defined your variable.
```

您还可以捕捉特定的错误:

```
>>> try:
...     1/0
... except ZeroDivisionError:
...     print("Can't divide by 0; sorry m8")
...
Can't divide by 0; sorry m8
```

捕捉错误使您能够在程序中断时继续运行。例如，下面的代码因`ZeroDivisionError`而失败，因此我们也不会计算循环中后面的表达式:

```
>>> for i in range(3):
...     print(1/i)
...
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
ZeroDivisionError: division by zero
```

但是如果您添加一个 *try-except* 块，您会得到更好的结果:

```
>>> for i in range(3):
...     try:
...             print(1/i)
...     except ZeroDivisionError:
...             print("Can't divide by 0; sorry m8")
...
Can't divide by 0; sorry m8
1.0
0.5
```

感觉很好，不是吗？

## 总结和最终想法

这是给你的 Pythonic 提示表:

1.  **可读性计数**。清晰地格式化和组织你的代码。
2.  **漂亮总比难看好**。避免多余的解决方案。
3.  **复杂比复杂好**。如果你的代码必须是复杂的，那没问题，但是如果它是不必要的复杂，那就不好了。
4.  **错误永远不会悄无声息地过去**。学会处理错误——这值得你花时间！

就这样，我祝你在 Pythonic 的冒险中好运。

**想擅长 Python？** [**获取独家，免费获取我简单易懂的攻略**](https://witty-speaker-6901.ck.page/0977670a91) **。想在介质上无限阅读故事？用我下面的推荐链接注册！**

[](https://murtaza5152-ali.medium.com/?source=entity_driven_subscription-607fa603b7ce---------------------------------------)  

*我叫穆尔塔扎·阿里，是华盛顿大学研究人机交互的博士生。我喜欢写关于教育、编程、生活以及偶尔的随想。*

## 参考

[1][https://peps.python.org/pep-0008/](https://peps.python.org/pep-0008/)