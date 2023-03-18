# Python 中的面向对象编程—理解变量

> 原文：<https://towardsdatascience.com/object-oriented-programming-in-python-understanding-variable-e451cf581368>

## 理解 Python 类中不同类型的变量以及如何使用它们。

![](img/3326f5840042afd12b40498421e5ba2d.png)

照片由 [Pankaj Patel](https://unsplash.com/@pankajpatel?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

在之前的文章中，我们介绍了作为*类*的一个基本组件的*变量*，但是我们并没有深入探讨。在这篇文章中，我们将深入探讨类中不同类型的变量*以及它们如何用于不同的目的。*

变量让我们在程序中存储特定的值。在*类*中，*变量*的一些常见用法是初始化一个空变量供*方法*使用，声明一个名称或默认值。

> 一个 Python 类*可以有两种类型的*变量*:实例和类变量。*

# 安装

我们将使用我们在上一篇文章中创建的*类* — `NumList`的稍微修改版本来继续我们的讨论。

```
Instance name of nlA1 =  Number List A1
```

# 实例变量

> *这些是与*对象*或*类实例*紧密相关的*变量*，而不是*类*。*

例如，如果你有一辆玩具车，我们认为它是一个`Toy`类类型，那么实例变量将是那些附加到**你的**玩具车的变量，而不是附加到`Toy`类的任何其他玩具车的变量。

## 声明实例变量

> *一个*实例变量*可以在*类*内部声明，也可以在*类实例*创建后声明。*

举个例子，

*   `__list`和`instName`是在`NumListA`类中声明的两个*实例变量*。另外，请注意如何在初始化*类*并在以后使用时传递参数，例如`nlA1.insName` = "Number List A1 "。
*   `outOfClassVar`是直接为`nlA1`对象*创建的*。

## 实例变量是独立

*实例变量*是特定于实例的，因此与其他实例相隔离，即使它们可能都属于相同的*类*类型。检查实例变量`instName`在被两个不同名称参数创建的*实例*(`nl`和`nl2`)调用时如何产生不同的结果。

```
nlA2 = NumListA(name = "Number List A2")
print('Name of nlA1 instance = ', nlA1.instName)
print('Name of nlA2 instance = ', nlA2.instName)Name of nlA1 instance =  Number List A1
Name of nlA2 instance =  Number List A2
```

# 类别变量

*类变量*是在*类*中声明的，并且它们坚持使用*类*。这意味着，与*实例变量*不同，即使没有*对象*被创建*类变量*仍将存在，并且它们的值在*类*级别更新，而不是在*实例*级别更新。

## 声明和调用类变量

*类变量*在*类*内声明，但在任何方法外声明。看下面的例子，在`NumListB`中创建了*类变量* `counter`。

*类变量*的调用遵循与*实例变量*相同的约定有:<`instance name`<`.`><`class variable name`>。

## 类变量是粘性的！

*实例变量*只被限制在它们的*实例*中，而*类变量*是粘性的。它们继承并更新从*类*创建的所有*对象*。

在`NumListB`中，我们在`__init__`方法中添加了`NumListB.counter += 1`，它基本上告诉 Python 在每次实例化`NumListB`时将*类变量* `counter`递增 1。

🛑注意到下面从`nlB1`和`nlB2`调用`counter`产生了相同的值。

```
nlB1 = NumListB(name = "Number List B1")
nlB2 = NumListB(name = "Number List B2")# printing out class variable
print("Number of NumList class instance created = ", nlB1.counter)
print("Number of NumList class instance created = ", nlB2.counter)Number of NumList class instance created =  2
Number of NumList class instance created =  2
```

## 检查属性是否存在

一旦你创建了一堆*类*和*对象*，你不太可能知道它们包含哪些属性或变量。或者想想当你不得不使用其他人创建的*类*的时候。在这种情况下，为了检查哪些属性包含在一个*类*或*对象*中，我们可以使用两个功能:`__dict__`和`hasattr`。

**列出所有属性**

`__dict__`是一个内置的功能，无论何时创建它们，都会自动带有一个*对象*或*类*。请看下面我们如何调用它来获得一个*对象*的所有属性。

```
# printing out the instance variables
print('Instance variables of %s are: \n' %(nlB1.instName), nlB1.__dict__)Instance variables of Number List B1 are: 
 {'instName': 'Number List B1', '_NumListB__list': []}
```

🛑注意到，`nlB1.__dict__`的输出没有显示*类变量* - `counter`。

> *从实例调用* `*__dict__*` *不会显示*类变量*，因为它们不是*实例*的属性。*

但是我们可以使用*类*中的`__dict__`来查看类属性。它将打印出一堆其他的东西，其中一些我们稍后会回来看，但是现在，检查下面代码的输出并寻找两个*类变量* : `counter`和`__hidden_code`。

```
# printing out the class variables
print('Properties of NumListB class:\n', NumListB.__dict__)Properties of NumListB class:
 {'__module__': '__main__', 'counter': 2, '_NumListB__hidden_code': 999, '__init__': <function NumListB.__init__ at 0x0000019447733490>, 'add_value': <function NumListB.add_value at 0x0000019447733520>, 'remove_value': <function NumListB.remove_value at 0x0000019447733010>, 'get_list': <function NumListB.get_list at 0x0000019447253C70>, '__dict__': <attribute '__dict__' of 'NumListB' objects>, '__weakref__': <attribute '__weakref__' of 'NumListB' objects>, '__doc__': None}
```

**姓名莽撞**

🛑你有没有注意到私有*变量*的名字是如何被`__dict__`变量打印出来的？

由于这些私有变量不应该在对象外部可用， *Python* 破坏了使它们可用的操作，

*   将*类*名称放在*变量*名称之前
*   在开头加上一个额外的下划线(`_`)。

因此，在输出中，您应该看到*实例变量* `__list`打印为`_NumListA__list`，而*类变量* `__hidden_code`打印为`_NumListB__hidden_code`。

> *这些*损坏的*名称可以用来直接访问这些私有*变量*。这显示了 Python 类的*私有*特征是如何受到限制的。*

```
# printing out private instance variable using mangled name
print('Private instance variable __list from instace: nlA1 =', nlA1._NumListA__list)# printing out privatge class variable using mangled name
print('Private class variable from class NumListB = ', NumListB._NumListB__hidden_code)Private instance variable __list from instace: nlA1 = [2]
Private class variable from class NumListB =  999
```

**检查特定属性**

使用`__dict__`大概是探索的好办法。但是如果你需要检查某个*属性*或者*属性*是否存在于*类*或者*对象*中呢？

> *Python 函数* `*hasattr()*` *可以用来检查一个特定的*属性*。*

`hasattr()`接受两个参数:被检查的对象，以及要作为字符串值搜索的属性的名称。如果属性存在，则返回`True`，否则返回`False`。

```
# n1B1 instance properties
print(hasattr(nlB1, '__list'))
print(hasattr(nlB1, 'counter')) # class variable# NumListB class properties
print(hasattr(NumListB, 'counter'))
print(hasattr(NumListB, '__hidden_code'))# checking mangled names for the private attributes
print(hasattr(nlB1, '_NumListB__list'))
print(hasattr(NumListB, '_NumListB__hidden_code'))False
True
True
False
True
True
```

注意:

与`__dict__`不同，`hasattr()`可以从*对象*中检查*类变量*并返回`True`。

🛑 *私有属性*可以使用它们的错位名称进行搜索，否则它们将返回`False`。

# 下一步是什么

在本文中，我们详细介绍了不同类型的*变量*及其在 Python *类*上下文中的属性。我们了解到，

*   什么实例和类变量？
*   如何检查对象和类的属性？
*   如何访问公共和私有变量？

在下一篇文章中，我们将深入探讨一个*类*的*方法*。

如果你喜欢这篇文章，请查看 Python 中面向对象系列的前几篇:

[](/object-oriented-programming-in-python-what-and-why-d966e9e0fd03) [## Python 中的面向对象编程——什么和为什么？

### 学习 Python 中的面向对象编程。

towardsdatascience.com](/object-oriented-programming-in-python-what-and-why-d966e9e0fd03) [](/oop-in-python-understanding-a-class-bcc088e595c6) [## Python 中的 OOP 理解一个类

### 理解 Python 类的基本组件。

towardsdatascience.com](/oop-in-python-understanding-a-class-bcc088e595c6) [](https://curious-joe.medium.com/object-oriented-programming-in-python-inheritance-and-subclass-9c62ad027278) [## Python 中的面向对象编程——继承和子类

### 理解继承的基本概念，并通过创建子类来应用它们。

curious-joe.medium.com](https://curious-joe.medium.com/object-oriented-programming-in-python-inheritance-and-subclass-9c62ad027278)