# Java 中的变量和数据类型

> 原文：<https://towardsdatascience.com/variables-and-data-types-in-java-76f7de62d1b>

## 用于数据科学的 Java 第 1 部分

![](img/90365d510408a2dbcde6cca6099e48b2.png)

卡斯帕·卡米尔·鲁宾在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

Java 作为一种语言，通常不会出现在数据科学的学习语言列表中。这种争论经常留给那些认为 R 或 Python 是最好的语言的人。但是，由于数据科学涵盖了广泛的领域和应用，这意味着一种语言不能总是做所有的事情，或者一些语言在其他领域可能更有用。因此，Java 是在数据科学的一些子领域中工作得最好的语言之一，由于与 Hadoop 的集成，它特别关注大数据的应用，由于它的可移植性，它还关注物联网基础设施和应用。事实上，除了数据科学之外，它还存在于许多不同的领域。如果您想要一种用于大数据或物联网应用程序的语言，或者一种超越数据科学的语言，这将使您很容易学习！因此，如果你有兴趣学习它，请继续下面的内容，我们将介绍 Java 中变量和数据类型的基础知识。

## 印刷

在学习如何创建和使用变量之前，首先要学习的是如何输出结果。这清楚地表明了我们在做什么，语言是如何表现的，这在开始学习一门新语言时是很重要的。虽然在 Python 中这是一个相对简单的命令，但在 Java 中，要将一些内容打印到控制台，我们必须使用:

```
System.out.println();
```

它是计算机在屏幕上显示某些东西的指令。

分解一下，这是一个系统命令(`System`)，它通过在屏幕上打印一个新行(`println()`)来将文本作为输出(`out`)，在 Java 中，要结束任何一行代码，我们需要添加一个`;`。因此，打印出我们想要的任何东西！

例如:

```
System.out.println(2+3);
```

会打印出 5！

## 创建和使用变量

既然我们知道了如何打印出我们所做的任何事情的结果，那么我们就可以讨论如何创建和使用变量。Java 中的每个变量都有一个名字和值，就像其他语言一样。虽然变量名不会改变，但变量值可以改变。

与 Python 等其他语言相比，Java 的主要区别在于 Java 是静态类型的。这意味着您必须定义变量将存储什么类型的信息，如整数、字符串或其他数据类型。请小心，因为这一点以后无法更改！

为了在 Java 中编写一个变量，我们可以声明它，声明它的名字和它包含的数据类型。例如，我们可以声明一个名为 passengers 的变量，它只包含整数值。

```
int passengers;
```

这告诉计算机创建一个名为 passengers 的变量，以后我们只给它赋值整数。

然后当我们给变量赋值时，我们*初始化变量*。首先，我们可以指定 0 个乘客值:

```
int passengers;passengers = 0;
```

所以我们现在已经用*定义了*一个名为 passengers 的变量，告诉计算机我们将只给它分配整数值，并且*通过给变量赋值 0 来初始化*。

一旦变量被初始化，它就可以被操作或编辑成我们想要的样子。在这种情况下，因为我们已经定义了一个实例化的整数变量，所以我们可以执行在其他语言中可以执行的所有标准数学操作。例如:

```
int passengers;passengers = 0;passengers = passengers + 5;passengers = passengers - 3;passengers = passengers - 1 + 5;
```

当打印到控制台时，将显示:

```
System.out.println(passengers);#out:
6
```

## 用线串

与我们*声明*一个整数变量并初始化它的方式相同，我们也可以*声明*一个字符串变量。要声明一个字符串，我们使用:

```
String driver
```

和一个整数一样，它告诉计算机我们想要*声明*一个名为 driver 的变量来保存一个字符串。

然后，通过为其赋值，可以用同样的方式将*实例化为*，如下所示:

```
String driverdriver = "Hamish";
```

与整数一样，这使我们能够像在其他语言中一样操作字符串，并从中提取信息。例如:

```
// define and initialise the string
String driver;
driver = "Hamish";// We can count the number of items in the string and
// assign the value to another variableint letter = driver.length();// we can capitalise all the lettersdriverUpper = driver.toUpperCase();// and we can add strings togetherString driverFirstName = "Hamish";String driverLastName = "Blake"String driverFullName = driverFirstName + " " + driverLastNameSystem.out.print(driverFullName);#out:
Hamish Blake
```

我们在这里对`driverFirstName`和`driverLastName`变量唯一不同的地方是*在同一行声明了*和*实例化了*。

## 数据类型

我们已经看到了如何创建一个字符串和一个整数，那么 Java 中还有什么其他的数据类型呢？

*   Byte ( `byte`):存储-128 和 128 之间的整数
*   Short ( `short`):存储-32，768 到 32，767 之间的整数
*   整数(`int`):一个整数，最多可有 10 个数字，但有利于加速已定义的程序
*   Long ( `long`):一个整数，但可以比 10 位数长得多，但代价是占用更多空间，并可能降低程序速度
*   Double ( `double`):可以用来存储任何东西，包括整数、小数、正数或负数，但是会降低程序的速度
*   String ( `String`):对，一串！
*   字符(`char`):存储单个字母，例如`char answer = "b";`
*   Boolean ( `boolean`):用于存储真或假(真！)

这些中的每一个都必须被*声明*，然后*初始化*，就像上面的整数和字符串一样。

## 命名规格

在命名这些变量时，遵守规则和惯例是很重要的。

第一条规则是名字应该以字母开头。虽然它们可以以`-`或`$`开头，但不建议这样做，因为这是不好的约定。此外，名称是区分大小写的，所以无论使用什么大小写，都要确保一致。

第二条规则是，为了一致性和易读性，名称应该用小写字母来定义。这确保了单词被清楚地分开和定义。变量名中不能有空格！

第三条也是最后一条规则是，名字应该总是使用完整的单词来描述它的用途！如果可能的话，尽量限制缩写，除非它们被一致地使用并且有明确的定义，因为你永远不知道谁会阅读你的代码！

最后，虽然不一定是命名约定，但是我们可以使用`//`注释掉一行代码，或者注释掉符号后面的一行代码的一部分，或者我们可以在`/* multi line comment*/`之间添加一个多行注释。

## 结论

Java 是数据科学之外广泛使用的语言，但在数据科学内部，它通常用于大数据应用程序或物联网。虽然它是 R 或 Python 的大热门，但它仍然有一些用处！有了这个实用的工具，你应该能够开始学习 Java 的旅程，不管是为了好玩还是你正在做的一个新项目。祝你好运！

如果你喜欢你所读的，并且还不是 medium 会员，请使用下面我的推荐链接注册 Medium，来支持我和这个平台上其他了不起的作家！提前感谢。

[](https://philip-wilkinson.medium.com/membership) [## 通过我的推荐链接加入 Medium—Philip Wilkinson

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

philip-wilkinson.medium.com](https://philip-wilkinson.medium.com/membership) 

或者随意查看我在 Medium 上的其他文章:

[](/eight-data-structures-every-data-scientist-should-know-d178159df252) [## 每个数据科学家都应该知道的八种数据结构

### 从 Python 中的基本数据结构到抽象数据类型

towardsdatascience.com](/eight-data-structures-every-data-scientist-should-know-d178159df252) [](/a-complete-data-science-curriculum-for-beginners-825a39915b54) [## 面向初学者的完整数据科学课程

### UCL 数据科学协会:Python 介绍，数据科学家工具包，使用 Python 的数据科学

towardsdatascience.com](/a-complete-data-science-curriculum-for-beginners-825a39915b54) [](/a-complete-guide-to-git-for-beginners-a31cb1bf7cfc) [## 初学者 git 完全指南

### 键盘命令、功能和用法

towardsdatascience.com](/a-complete-guide-to-git-for-beginners-a31cb1bf7cfc)