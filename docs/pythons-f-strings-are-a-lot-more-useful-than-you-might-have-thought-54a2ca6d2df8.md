# Python 的 F 字符串比你想象的要有用得多

> 原文：<https://towardsdatascience.com/pythons-f-strings-are-a-lot-more-useful-than-you-might-have-thought-54a2ca6d2df8>

## 大多数人都没有意识到 f 字符串在 Python 中可以做的一些很酷的事情

![](img/e35965d101b8f3a94e4f05237b6e6c45.png)

(图片由 [Pixabay](http://pixabay.com) 上的 [StarzySpringer](https://pixabay.com/images/id-2667529/) 提供)

# 介绍

我们很可能都熟悉弦乐；自计算历史开始以来，大多数编程语言中都有这种核心数据类型。字符串很棒，因为它们能够用 ASCII 或 Unicode 表示一系列对人类来说非常明显的字符。作为人类，我们可能不会说组成 unicode 字符的 0 和 1 的语言，但字符串是完全可以理解的。

如果您用来编程的编程语言本质上至少是高级的，那么您可能也熟悉字符串插值的概念。字符串插值是编程语言通过定义字符串来读取和修改字符串的能力。在 Python 中，像这样的字符串插值是通过一个叫做 F 字符串的特性来完成的。F 字符串允许你用字符串做很多事情，有些人甚至没有意识到 Python 中 F 字符串的全部功能。幸运的是，互联网上有一些随机的博客作者可以为你的大脑深入探究 F 弦，这就是我今天要做的。如果你想与接下来的代码进行交互，我在这个小项目中使用的笔记本也可以在 Github 上找到:

<https://github.com/emmettgb/Emmetts-DS-NoteBooks/blob/master/Python3/f%20strings.ipynb>  

# 什么是 F 弦？

在我们讨论 F 弦能做什么之前，我们先简单了解一下 F 弦到底是什么，这可能很重要。在 Python 中，普通字符串是不插值的。这当然可以帮助语言减少一些延迟，因为它是被解释的。我认为这也是一件好事，因为没有隐式插值的理由，它可能会导致比它解决的问题更多的问题。例如，在 Julia 中，字符串总是内插的，所以有可能你试图创建一个字符串

```
x = 5
mystr = "and I said for $x dollars in my wallet, I will buy one icecream"
```

当你想要 x 美元的时候却意外地得到了 5 美元。不确定所有这些会有多频繁地同时发生，但这肯定是**可能**发生的事情。无论如何，Python 中的 F 字符串是通过在字符串前提供一个 F 来表示的，就像这样:

```
f""
```

F 字符串最明显的用法是在 throws，output，等等里面插入简单的值。这是通过使用{}语法并提供值名来完成的。当然，如果可能的话，该值也将被隐式转换。

```
x = 5
f"One icecream is worth {x} dollars"
```

这会产生一个实际上看起来更像这样的字符串:

```
'One icecream is worth 5 dollars' 
```

# 更多关于 F 弦的信息

我们已经确定了 F 字符串可以用来插入 Python 代码，但是它能做什么呢？我们可以对 F 字符串做的一件值得注意的事情是提供一种“this = that”类型的插值，这可能对调试和错误处理有用。我们只需添加一个等号:

```
f"If you are wondering what x is, {x=}"'If you are wondering what x is, x=5'
```

我们可以利用的另一件事是所谓的转换。转换允许我们在插值之前对其应用函数，例如

*   ！r —显示字符串分隔符，调用 repr()方法。
*   ！a-显示字符的 Ascii。
*   ！s —将值转换为字符串。

这些都很有价值，只要打个电话就能发挥作用！r 在最后，经常！s 是不需要的，因为通常我们打印的数据类型，正如我提到的，是隐式转换的，不是典型的结构，但在某些情况下，你可能需要提供这个。以下是 repr 版本的使用示例:

```
food = "cabbage"
food2brand = "Mcdonalds"
food2 = "French fries"f"I like eating {food} and {food2brand} {food2!r}""I like eating cabbage and Mcdonalds 'French fries'"
```

我们也可以使用对象格式化来改变插值的格式。Python 中的每个对象都可以决定它们的格式在字符串中的表示方式。为了表示这就是我们想要做的，我们使用了一个冒号，后跟我们想要添加的格式，例如在这里我们这样做是为了日期-时间。

```
import datetime
date = datetime.datetime.utcnow()
f"The date is {time:%m-%Y %d}"'The date is 02-2022 15'
```

这将调用 __format__ 方法，该方法至少提供给大多数类。

# 结论

所以你可以用 F 弦做一些很棒的事情。一般来说，大多数人认为 F 字符串的功能非常有限，但是它们确实可以做很多不同的事情。字符串在 Python 中是一个非常重要的概念，虽然 Python 可能不是元编程和将字符串解析为代码的最佳语言，但在 Python 的这个领域肯定也有用例。非常感谢您的阅读，并快乐插值！