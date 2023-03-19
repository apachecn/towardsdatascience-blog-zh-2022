# Python 中的鸭式打字是什么？

> 原文：<https://towardsdatascience.com/duck-typing-python-7aeac97e11f8>

## 理解动态类型编程语言(如 Python)中鸭类型的概念

![](img/df910aff389492d89efd750aa2af8a97.png)

杰森·理查德在 [Unsplash](https://unsplash.com/s/photos/duck?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

## 介绍

Duck Typing 是一个通常与**动态类型化**编程语言和**多态性**相关的术语。这一原则背后的想法是，代码本身并不关心一个对象是否是一只**鸭子**，而是只关心它是否**嘎嘎叫**。

> 如果它走路像鸭子，叫起来像鸭子，那它一定是鸭子。

## 理解 Python 中的 Duck 类型

让我们考虑一下`+` Python 操作符；如果我们对两个整数使用它，那么结果将是两个数的和。

```
>>> a = 10 + 15
>>> a
25
```

现在让我们考虑同样的字符串对象类型的操作符。结果将是两个对象相加在一起的连接。

```
>>> a = 'A' + 'B'
>>> a
'AB'
```

这种多态行为是 Python 背后的核心思想，Python 也是一种动态类型化的语言。这意味着它在运行时执行类型检查，而静态类型语言(如 Java)在编译时执行类型检查。

此外，在静态类型语言中，我们还必须在源代码引用变量之前声明变量的数据类型。

```
**// Python** a = 1**// Java**
int a;
a = 1
```

Python 本身会自动重载一些操作符，以便它们根据正在处理的内置对象的类型执行不同的操作。

```
def add_two(a, b):
    print(a + b)
```

这意味着任何支持`+`操作符的对象都可以工作。事实上，语言本身重载了一些操作符，因此所采取的操作依赖于所涉及的对象的特定数据类型。

> **Duck Typing 指的是不将代码约束或绑定到特定数据类型的原则**。

让我们考虑下面的两个示例类。

```
class Duck: 

    def __init__(self, name):
        self.name = name def quack(self):
        print('Quack!')class Car: 

    def __init__(self, model):
        self.model = model

    def quack(self):
        print('I can quack, too!')
```

由于 Python 是一种动态类型语言，我们不必指定函数中输入参数的数据类型。

```
def quacks(obj):
    obj.quack()
```

现在，如果我们用不同的对象调用同一个函数两次，所采取的动作将取决于输入对象的数据类型。

```
>>> donald = Duck('Donald Duck')
>>> car = Car('Tesla')
>>>
>>> quacks(donald)
'Quack!'
>>>
>>> quacks(car)
'I can quack, too!'
```

如果对象不支持指定的操作，它将自动引发异常。

```
>>> a = 10
>>> quacks(a)
AttributeError: 'int' object has no attribute 'quack'
```

因此，在 Python 中，我们通常避免强制进行这种手动错误检查，因为这会限制可能涉及的对象类型。在一些特殊情况下，你可能仍然想检查一个对象的类型(例如使用`type()`内置函数)。

但是请注意，您应该避免根据特定的数据类型来区分所采取的操作，因为这将降低代码的灵活性。但是在其他一些情况下，这可能表明问题出在设计层面，而不是代码本身。

## 最后的想法

在今天的文章中，我们讨论了 Duck 类型以及它与多态和方法重载的关系。此外，我们讨论了静态类型语言和动态类型语言之间的两个主要区别，以及这如何使 Python 代码能够应用于对象而无需关心它们的实际数据类型。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**你可能也会喜欢**

[](/mastering-indexing-and-slicing-in-python-443e23457125)  [](/dynamic-typing-in-python-307f7c22b24e) 