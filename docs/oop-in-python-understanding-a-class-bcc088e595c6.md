# Python 中的 OOP 理解一个类

> 原文：<https://towardsdatascience.com/oop-in-python-understanding-a-class-bcc088e595c6>

![](img/005da2cb84554046cf70bd92214d40b8.png)

照片由[哈维·加西亚·查维斯](https://unsplash.com/@javchz?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

## 理解 Python 类的基本组件

# 背景

这是关于 Python 中面向对象编程(OOP)的博客文章系列的第二部分。

在我的上一篇文章中，我提到了什么是 OOP，以及为什么你可能想学习 OOP，即使你可能看不到它的明显用途。万一你没看过，好奇的话，可以看看:[Python 中的面向对象编程——什么和为什么？。](/object-oriented-programming-in-python-what-and-why-d966e9e0fd03)

在这篇文章中，我们将开始深入一些 OOP 概念的内部，并尝试加深我们对它们的理解。

# 一个类的构造块

为了保持连续性，也为了有一个具体的例子，我将从我在上一篇文章中创建的名为`NumList`的*类*开始。我们将剖析`NumList`来看看它的元素。然后我们会再回来对一个*类*的元素进行更一般的讨论。

首先，让我们注意以下事项:

*   定义一个类以关键字:`class`开始，然后跟一个类的名字，以一个类似于函数的冒号`(:)`结束。
*   一个类通常配备有这三个组件:
    –一个**名称**:在我们的例子中，`NumList`用来标识这个类。
    –**属性**:关于类的一组特性。在`NumList`中，`__list`是一个属性。一个类也可能没有属性。
    –**方法**:类执行某些任务的能力。在`NumList`、`add_value()`、`remove_value`和`get_list()`中是方法。与属性类似，类也可以不包含方法。

> *一个简单的 Python 类配备了三样东西:名称、属性和方法*

# 进入一个班级的内部

## Python 构造函数

现在让我们来谈谈`NumList`中定义`__init__()`的第一块代码。我们将讨论这个方法是什么，解释使用的符号和关键字。

*   这是一个特殊的方法，叫做*构造器*。
    –每次从类中创建对象时，构造函数都会自动运行。它加载了一个类的所有必要元素，使它变得可用。
    –在`NumList`的情况下，一创建`NumList`类的对象，空列表`__list`就被构造函数初始化。
    –构造函数只在类中使用。
    –构造函数必须至少有一个参数。在`NumList`中，`__init__()`有一个参数- `self`。
*   `self`:该参数用作自参考。
    –使用该参数使属于该类的变量可用。
    –它可以被称为任何其他名称，但称为`self`只是一种习惯。
*   虚线符号:我们创建了空列表`self.__list`。
    –该约定用于访问或创建对象的属性:`<ClassName><dot(.)><property name>`。
    –在`NumList`中，我们使用`self.__list = []`创建了一个名为`__list`的属性，属于`NumList`，用`self`表示，并作为一个空列表初始化。
*   `__`:以两个下划线开头的组件名称使组件成为私有的。
    –表示组件只能在类内访问。是 [*封装*](https://en.wikipedia.org/wiki/Encapsulation_(computer_programming) 的一个实现。
    –试图访问私有组件将导致错误。
    –试试:`list01 = NumList()`然后`len(list01.__list)`。这将导致一个`AttributeError`。

## 方法

## `add_value()`和`remove_value()`

接下来的三个代码块实现了三个方法或函数。

前两个方法基本上是我们创建的两个函数的再现，用于在类上下文中演示过程化编程，因此在关键字和符号方面有以下变化:

*   使用`self`作为参数:对于任何方法，`self`都是一个强制参数，如前所述，

> `self` *使得所有来自类的属性和方法，在我们的例子中* `*NumList*` *，可用到方法。*

*   方法中的参数:根据需要，方法可以有多个参数，作为类上下文之外的常规 Python 函数。例如，`add_value()`方法除了强制参数`self`外，还有一个名为`val`的参数。
*   点符号:根据前面讨论的约定，`self`指的是类本身。
    –如此，`self.__list`进入空单。为了使用一种方法，我们再次使用点符号。
    –为了构造方法:`add_value()`，我们使用了`List`类中的`append()`方法。因为我们的`__list`本身是一个列表，所以它继承了`append()`方法。所以要使用这个方法我们用:`self.__list.append()`。

## `get_list()`

因为我们将`__list`初始化为`NumList`中的隐藏组件，所以直接访问列表是不可能的。所以我们需要一个方法或函数来访问它。

> `*get_list()*` *为我们提供了进入* `*__list*` *的隐藏参数。*

# 超越单一的阶级

## 亚纲

现在让我们假设在将我们的产品`NumList`交付给我们的客户后，我们从推荐中获得了一个新客户。该客户想要相同的列表和功能，但还需要:

*   获取列表中数字总和的能力

那么，我们该怎么做呢？

一个显而易见的方法是复制`NumList`的代码，并为其添加另一个方法，姑且称之为`get_total()`。但是，如果我们不断获得新的客户或订单，并有不同的附加功能需求，该怎么办呢？很快，这种复制和修改的过程将不再是一个有效的解决方案。

OOP 有一个有效解决这个问题的方法:创建*子类*。创建子类有两个明显的优点:

*   **代码的重用**:一个*子类*继承了超类的所有方法，因此我们不必重新创建这些方法。
*   **定制**:一个*子类*可以包含新的方法和属性。这使得定制解决方案变得容易，同时保持超类的基础不变。

## 例子

现在让我们创建一个名为`NumListExt01`的子类来扩展`NumList`类的功能。

```
Initial list of values of cust02: []
Updated list after adding values to it: [2, 20, 44, 12]
Updated list after removing value 12 is:  [2, 20, 44]
Sum of all the elements of the current list is: 66
```

注意`NumList`是如何被子类利用和定制而不影响超类的:

> *♻️的子类* `*NumListExt01*` *继承了超类*`*NumList*`**`*remove_value()*`*`*get_list()*`*的方法。****
> 
> **➕子类* `*NumListExt01*` *有一个额外的方法* `*get_total()*` *，它只存在于这个类中，因此只对我们的新客户可用。**

# *下一步是什么？*

*在这篇文章中，我们看到*

*   *我们如何构造一个 Python 类*
*   *我们如何通过添加*子类*来扩展现有的*类*。*

*在下一篇文章中，我们将讨论更多关于*子类*的内容，深入*子类*的内部，并结合实例讨论 Python 中的*继承*。*

***更新**:查看下一篇文章[Python 中的面向对象编程——继承和子类](/object-oriented-programming-in-python-inheritance-and-subclass-9c62ad027278)。*