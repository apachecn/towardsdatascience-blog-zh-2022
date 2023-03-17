# Python 中单下划线和双下划线是什么意思？

> 原文：<https://towardsdatascience.com/whats-the-meaning-of-single-and-double-underscores-in-python-3d27d57d6bd1>

## 我从来没有注意过这些特殊的字符，直到我知道它们是什么意思

![](img/d99f9dbc1b3f77ac2e662380e7c53bf4.png)

Clark Van Der Beken 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

你有没有好奇过**单双下划线**在 Python 中给变量和函数命名时的不同含义？

我最近研究了这个主题来温习一些基础知识，并决定把它写下来，因为我确信有一天我会再次需要它。

这篇文章对下划线的不同用法进行了分类。它将涵盖:

1.  **单前导下划线:** `**_foo**`
2.  **单尾随下划线:**
3.  **单下划线:** `**_**`
4.  **前后双下划线(用于定义 *dunder 方法* ):** `**__bar__**`
5.  **双前导下划线:** `**__bar**`

> ***了解每种语法的含义以及何时应该使用将有助于你在可读性和效率方面提高代码质量。***
> 
> 这也有助于你符合标准的命名惯例。

事不宜迟，让我们来看一些实际的例子🔍

> 新到中？你可以每月订阅 5 美元，解锁我写的不限数量的关于编程、MLOps 和系统设计的文章，以帮助数据科学家(或 ML 工程师)编写更好的代码。

[](https://medium.com/membership/@ahmedbesbes) [## 通过我的推荐链接加入 Medium—Ahmed bes bes

### 阅读 Ahmed Besbes 的每一个故事(以及媒体上成千上万的其他作家)。您的会员费直接支持…

medium.com](https://medium.com/membership/@ahmedbesbes) 

# **1 —单前导下划线:_foo**

变量、函数或方法名前面的单个前导下划线表示这些对象在内部使用。

这对程序员来说更多的是一种语法提示，而不是由 Python 解释器强制执行的，这意味着这些对象仍然可以从另一个脚本以一种方式访问。

但是，有一种情况，带有前导下划线的变量不能被访问。

考虑下面的文件`module.py`，它定义了两个变量。

如果您从解释器或其他地方使用通配符导入(尽管这是强烈不推荐的)，您会注意到带有前导下划线的变量在名称空间中不可用。

但是，正如我们前面所说，如果导入模块并直接调用变量名，`**_private_variable**`仍然可以被访问。

单前导下划线一般在类中用来定义 ***【内部】*** 属性。我用双引号将*内部*括起来，因为 Python 中没有内部属性这种东西。前导下划线仍然是一个符号，应该如此对待。

如果您有一些不打算从类外部访问的属性，仅仅因为它们只在内部用于中间计算，您可以考虑给它们添加一个前导下划线，这将作为对您和您的 ide 的一个提示。

让我们来看一个例子:考虑一个定义了一个**_ residency**属性的 Employee 类，这个属性是计算薪酬包所需要的。

如您所见，`_seniority`属性可以从类的外部访问。

# **2 —单尾随下划线:foo_**

有些情况下你想使用的变量名实际上是 Python 中的保留关键字比如`class`、`def`、`type`、`object`等。

为了避免这种冲突，您可以添加一个尾部下划线作为命名约定。

```
**class_ = "A"**
```

# **3 —单下划线:_**

在某些情况下，您会看到 python 开发人员使用单下划线。

简而言之就是这些。

**→定义临时或未使用的变量。**

**例#1** 如果不使用 for-loop 的运行索引，可以很容易地用单下划线代替。

如果你的函数返回一个五个元素的元组，但是你只需要使用其中的两个(比如第一个和第四个)，你可以用下划线来命名剩下的三个。

**→python 交互式解释器中最后一次评估的结果存储在“_”中。**

**→用作数字分组的可视分隔符**

根据 [PEP 515](https://www.python.org/dev/peps/pep-0515/) ，下划线现在可以添加到数字文字中，以提高长数字的可读性。

这里有一个例子，你可以把十进制数按千分组。

# **4 —双前导和尾随下划线:__foo__**

双前导和尾随下划线用于定义特殊的通用类方法，称为 **dunder methods** (在 score methods 下**D**double**的简称)。**

Dunder 方法是保留的方法，您仍然可以覆盖它们。它们有特殊的行为，称呼也不一样。例如:

*   `**__init__**`被用作类的构造函数
*   `**__call__**`用于使对象可调用
*   `**__str__**`用于定义当我们将对象传递给`print`函数时，屏幕上打印的内容。

正如您所看到的，Python 引入了这个命名约定来区分模块的核心方法和用户定义的方法。

如果你想了解更多关于 dunder 方法的知识，你可以查看这个[链接](https://www.section.io/engineering-education/dunder-methods-python/)。

# **5 —双前导下划线:_ _ 条**

双前导下划线通常用于名称混淆。

名称管理是解释器改变属性名称的过程，以避免子类中的命名冲突。

让我们看下面的类来说明:

现在让我们通过使用内置的`dir`方法来检查`car`对象的属性。

我们注意到`color`和`_speed`可用，而`__brand`不可用。

然而，这里有一个`**_Car__brand**`属性。这就是名称管理:解释器在属性前面加上一个“_”和类名。这样做是为了避免`__brand`属性的值在子类中被覆盖。

让我们创建一个子类，并尝试覆盖前面的属性:

现在让我们检查一下`extended_car`对象的属性:

正如我们所看到的，添加了一个新属性:`_ExtendedCar_brand`，而来自父类的`_Car_brand`仍然存在。

如果我们尝试访问这两个属性:

我们注意到`__brand`属性的值(来自父类)没有被覆盖，尽管`ExtendedCar`类的定义建议这样做。

# 资源:

我从来不知道单下划线和双下划线在许多不同的上下文中会有同样多的含义。

学习它们很有趣，一如既往，这里是我的资源列表，你可以更深入地了解这个主题。

*   [https://towards data science . com/5-python-3 fa 6 CD 0379 中下划线的不同含义](/5-different-meanings-of-underscore-in-python-3fafa6cd0379)
*   【https://www.python.org/dev/peps/pep-0515/ 号
*   【https://www.python.org/dev/peps/pep-0008/ 
*   一篇关于邓德斯的法国博客文章:[https://he-arc.github.io/livre-python/dunders/index.html](https://he-arc.github.io/livre-python/dunders/index.html)
*   [https://www . section . io/engineering-education/dunder-methods-python/](https://www.section.io/engineering-education/dunder-methods-python/)

# 感谢阅读🙏

同样，如果您已经做到了这一步，我要感谢您的宝贵时间，并希望您已经了解了 Python 编程语言非常特殊的方面的一些有用的东西。

如果你有兴趣学习更多关于 Python 技巧的知识，可以看看我以前的文章，或者至少是上一篇。

[](/how-to-use-variable-number-of-arguments-in-python-functions-d3a49a9b7db6) [## 如何在 Python 函数中使用可变数量的参数

### 一个关于*args 和**kwargs 的小故事

towardsdatascience.com](/how-to-use-variable-number-of-arguments-in-python-functions-d3a49a9b7db6) 

今天就这些了。下次见！👋

## ***新到中？您可以每月订阅 5 美元，并解锁各种主题的无限文章(技术、设计、创业……)您可以通过点击我的推荐*** [***链接***](https://ahmedbesbes.medium.com/membership) 来支持我

[](https://ahmedbesbes.medium.com/membership) [## 加入我的介绍链接媒体-艾哈迈德贝斯

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

ahmedbesbes.medium.com](https://ahmedbesbes.medium.com/membership) ![](img/719933aeb1e21185674dd0e657d46c59.png)

照片由[卡斯滕·怀恩吉尔特](https://unsplash.com/@karsten116?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄