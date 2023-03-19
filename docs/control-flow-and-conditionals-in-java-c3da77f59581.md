# Java 中的控制流和条件

> 原文：<https://towardsdatascience.com/control-flow-and-conditionals-in-java-c3da77f59581>

## 用于数据科学的 Java 第 2 部分

![](img/6ea9308ad726a268a8a39c974de2a685.png)

克劳迪奥·施瓦茨在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

这是介绍用于数据科学的 Java 基础知识的系列文章的第二篇。在之前的帖子中，Java 被介绍为大数据和物联网应用中常用的语言，以及如何在 Java 中创建变量和使用不同的数据类型。在这篇文章中，我们将在这个基础上讨论 Java 中的控制流和条件，这是构建更复杂代码的基础。

[](/variables-and-data-types-in-java-76f7de62d1b)  

## If 条件

当考虑控制您的代码流时，您首先要做的是能够在条件为真时运行代码。例如，如果你知道外面很冷，那么你会希望你的电脑潜在地告诉你外套放在哪里。这可以在 Java 中使用`if`语句来实现。

Java 的工作方式是声明`if (condition) {perform some code}`。这意味着您首先需要声明`if`，然后在`if`之后，您需要将条件放在括号中。如果条件满足，那么在`{}`括号中指定的任何代码都将运行。简单！

如前所述，这方面的一个例子是，如果天气冷，你的电脑会告诉你穿上外套。这可以通过以下方式实现:

```
boolean isCold = true;

if (isCold) {
    // executes only if isCold is true
    System.out.println("It's cold, wear a coat!");
}
```

这里，`isCold`是必须解析为`true`或`false`的测试条件，它将决定是否运行包含在`{}`中的代码。

为此，花括号决定了变量的范围。这意味着如果我们在花括号中定义一个变量，那么它只能在花括号中使用，不能在其他地方使用。相反，在花括号外定义的任何变量(不是在不同的花括号中)都可以在花括号内使用！

## 其他声明

当满足`if`条件时，您很少想要运行代码。在某些情况下你想运行一些代码`if`条件被满足，否则你想运行一些其他代码。这可以使用与`if`语句一起使用的`else`语句来实现。

这方面的一个例子是在交通灯系统中。如果你开车时看到绿灯，那么你想继续，但如果你看到红灯，那么你需要停下来。这可以编码为:

```
boolean isLightGreen = true;

if (isLightGreen) {
    System.out.println("Drive!");
} else {
    system.out.println("Stop!");
}
```

## Else if 语句

如果第一个条件失败，然后您希望在运行任何其他代码之前检查另一个条件，那么这个过程会变得更加复杂。这可以使用`else if`语句来实现。这基本上是当`if`的第一个条件失败时，运行这段代码`if`的另一个条件被满足。

建立在交通灯系统上，第二个条件可以是是否有黄灯。这是因为如果有黄灯，那么这通常意味着你应该减速。在这种情况下，你不会想马上停下来，而是先慢下来。

```
boolean isLightGreen = false;
boolean isLightYello = true;

if (isLightGreen) {
    System.out.println("Drive!");
} else if (isLightYellow) {
    System.out.println("Slow Down!");
} else {
    System.out.println("Stop!");
}
```

在这种情况下，将首先检查`isLightGreen`条件。如果是`false`，则`isLightYellow`条件将被检查。然后，也只有在那时，如果`isLightYellow`是`false`，那么`else`语句中的代码将运行。

## 布尔评估

当运行这些`if`、`else`和`else if`语句时，我们只想在一条语句评估为`true`或`false`时进入代码块。在这种情况下，我们可以使用比较运算符来检查一个语句是真还是假。为此，我们可以使用:

*   `<`:小于
*   `>`:大于
*   `<=`:小于或等于
*   `> =`:大于或等于
*   `==`:相等

例如，如果我们想检查我们的银行账户中是否有足够的钱来购买一件商品:

```
double money = 12.3;
double price = 4.10;

if (money >= price){
  System.out.println("Item bought!");
} else {
  System.out.println("Not enough money!");
}
```

我们可以使用`>=`来比较运算符将我们拥有的钱是否大于价格的陈述简化为`true`或`false`。

## 逻辑运算符

当然，在某些情况下，我们希望将条件组合在一起，组成更复杂的`true`或`false`条件。这可以使用逻辑运算符来完成:

*   `&&`:还有
*   `||`:或者
*   没有

例如，如果想要确保我们在银行中有足够的资金，并且价格低于我们的最高价格:

```
double money = 12.3;
double price = 4.10;
double maxPrice = 4;

if (money >= price && price < maxPrice){
  System.out.println("Item bought!");
} else {
  System.out.println("Not enough money!");
}
```

在这种情况下，商品没有被购买！

了解这些逻辑运算符的求值顺序很重要，这样可以确保得到预期的结果。为此，评估的顺序是:

1.  `()`
2.  `!`
3.  `&&`
4.  `||`

## 交换语句

最后，Java 中还有 switch 语句。这使您可以检查变量的值，并根据它可能取值的列表测试它是否相等。这些值(或条件)中的每一个都被称为一个案例，我们可以为其创建不同的行为。

这方面的一个例子是检查自动售货机中的某个号码分配了什么项目和值。除了创建 if 和 else 语句，还可以使用 switch 语句，如下所示:

```
String item;
double cost;
int passcode;

switch(passcode){
    case 123: 
        item = "Snickers";
        cost = 1.80;
        break;
    case 123: 
        item = "Walkers";
        cost = 2.10;
        break;
    case 123: 
        item = "Lucozade";
        cost = 2.50;
        break;
    default: 
        item = "Unknown";
        cost = 0;
        break;
}
```

在每一种情况下，都会分配一个项目和成本。每个案例末尾的`break`意味着代码将转到花括号的末尾，而不是检查其他条件。这确保了一次只执行一个案例。

结尾的最后一个 case 是默认 case，它与 else 语句一样，在没有其他 case 匹配时触发。

## 结论

Java 是一种广泛使用的编程语言，经常应用在大数据或物联网应用中。这对于任何数据科学家来说都非常有用。通过这种实践，您现在应该能够使用控制流和条件，包括 if、else、else if、布尔求值和 switch 语句，在 Java 中构建更复杂的代码！

如果你喜欢你所读的，并且还不是 medium 会员，请使用下面我的推荐链接注册 Medium，来支持我和这个平台上其他了不起的作家！提前感谢。

[](https://philip-wilkinson.medium.com/membership)  

或者随意查看我在 Medium 上的其他文章:

[](/eight-data-structures-every-data-scientist-should-know-d178159df252)  [](/a-complete-data-science-curriculum-for-beginners-825a39915b54)  [](/a-complete-guide-to-git-for-beginners-a31cb1bf7cfc) 