# Python 的 Dunder 方法指南

> 原文：<https://towardsdatascience.com/a-guide-to-pythons-dunder-methods-3b8104fce335>

## Python 背后的魔力

![](img/8b3f9072e62cf761aef8b1bbce83d430.png)

克里斯·里德在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

Python 有几个神奇的方法——你通常会听到从业者称之为 *dunder 方法(我将它们互换使用)*。这些方法执行一个称为*操作符重载的过程:* **向操作符** *提供超出预定义含义的扩展含义。我们使用操作重载将自定义行为添加到我们的类中，这样我们就可以将它们与 Python 的操作符和内置函数一起使用。*

想到一个更笨的方法的最简单的方法是作为你的实现和 Python 解释器之间的契约。合同的一个条款涉及 Python 在某些给定的情况下在幕后执行一些动作(例如，试图向自定义类添加一个整数)。

Dunder 方法以两个下划线开始和结束:您可能遇到的最流行的方法是`__init__()`方法。我们在一个类中创建了`__init__()`方法，这样我们就可以用该类的特定属性来初始化该类— *在*[*Python 3*](https://medium.com/geekculture/getting-started-with-object-oriented-programming-in-python-3-e0a87d38acfc)*面向对象编程入门中了解更多关于面向对象编程的信息。*

但是`__init__()`只是几种魔法方法中的一种。在本文中，我们将介绍您可能遇到的不同类型的 dunder 方法以及它们的作用。

# 字符串表示方法

每当我们在 Python 中创建新对象时，我们都会隐式地创建一个相关的对象，因为所有的类都继承自`Object`。在`Object`中定义的方法被我们新创建的类继承，并在各种情况下使用，比如打印一个对象。

```
**class** Car: 
    passcar = Car()**print**(car)"""
<__main__.Car object at 0x7f53e19a8d90>
"""
```

上面的代码是怎么知道要打印什么的？简单。Python 中所有类的父类`Object`，有一个名为`__repr__()`(发音为 dunder repper)的 dunder 方法；当我们调用`print()`语句时，它从我们的`Car`对象调用`__repr__()`方法，该对象是从父类`Object`继承的，并将一个值返回给主程序。

但是子类可以覆盖父类。我们所要做的就是在子类中创建一个同名的方法——在 [*继承:在 Python*](https://medium.com/geekculture/inheritance-getting-to-grips-with-oop-in-python-2ec35b52570) 中了解更多关于继承的知识。

```
**class** Car: 
    **def** __repr__(self):
        **return** f"{self.__class__.__qualname__}"

car = Car()**print**(car)"""
Car
"""
```

我们还可以使用`__str__()`(发音为 stir) dunder 方法来创建一个字符串表示。`__str__()`方法返回一个人类可读的字符串，该字符串提供了关于对象的更多有见地的信息。

***注意*** *:我们的对象没有太多的变化，所以我们将使用与* `*__repr__()*` *中相同的信息。*

```
**class** Car: 
    **def** __str__(self):
        **return** f"{self.__class__.__qualname__}"

car = Car()**print**(car)"""
Car
"""
```

如果`__str__()`丢失，`__repr__()`方法将作为备份行为。因此，当您调用`print()`时，它首先查找`__str__()`以查看它是否已被定义，否则它调用`__repr__()`。

# 数学方法

当我们创建表达式时，我们使用称为*操作符*的特殊符号。如果我们希望在一个表达式中使用操作数，那么它必须有一个数学方法，操作符可以用它来计算表达式。如果不创建 math dunder 方法，Python 会引发类型错误。

```
**class** RandomNumbers: 
    **def** __init__(self, a, b): 
        self.a = a 
        self.b = b

set_a = RandomNumbers(2, 4)
set_b = RandomNumbers(3, 5)**print**(set_a + set_b)"""
Traceback (most recent call last):
  File "<string>", line 9, in <module>
TypeError: unsupported operand type(s) for +: 'RandomNumbers' and 'RandomNumbers'
"""
```

当然，我们可以简单地在我们的类中创建一个`add_random_numbers()`方法，但是解决这个问题的更好的方法是使用`__add__()` dunder 方法——这样，我们可以对我们的`RandomNumbers`对象使用`+`操作符。

```
**class** RandomNumbers: 
    **def** __init__(self, a, b): 
        self.a = a 
        self.b = b

    **def** __add__(self, other):
        # Only permit RandomNumber objects to be added
        **if not** isinstance(other, RandomNumbers): 
            **return** NotImplemented

        **return** RandomNumbers(other.a + self.a, other.b + self.b)

    **def** __repr__(self):
        **return** f"{self.__class__.__qualname__}({self.a}, {self.b})"

set_a = RandomNumbers(2, 4)
set_b = RandomNumbers(3, 5)**print**(set_a + set_b)"""
RandomNumbers(5, 9)
"""
```

当一个`RandomNumbers`对象位于`+`操作符的左侧时，Python 将调用`__add__()`方法:`+`操作符右侧的方法作为`other`参数传递给`__add__()`方法。

在我们的例子中，我们防止我们的对象添加不是`RandomNumber`实例的对象。让我们来看另一种方法，它允许我们将数字与整数相乘:

```
**class** RandomNumbers: 
    **def** __init__(self, a, b): 
        self.a = a 
        self.b = b

    **def** __add__(self, other):
        **if not** **isinstance**(other, RandomNumbers):
            **return NotImplemented**

        **return** RandomNumbers(other.a + self.a, other.b + self.b)

    **def** __mul__(self, other): 
        if not isinstance(other, int): 
            **return NotImplemented** 

        **return** RandomNumbers(self.a * other, self.b * other)

    **def** __repr__(self):
        return f"{self.__class__.__qualname__}({self.a}, {self.b})"

set_a = RandomNumbers(2, 4)**print**(set_a * 3)"""
RandomNumbers(6, 12)
"""
```

同样，请注意`RandomNumbers`实例位于`*`操作符的左侧。如果我们把它移到右边会发生什么？

```
-- snip -- set_a = RandomNumbers(2, 4)**print**(3 * set_a)"""
Traceback (most recent call last):
  File "<string>", line 23, in <module>
TypeError: unsupported operand type(s) for *: 'int' and 'RandomNumbers'
"""
```

Python 引发了一个类型错误。

出现这种情况的原因是，当对象位于数学运算符的左侧时，会调用 math dunder 方法。如果你想扩展这个功能，那么你也可以调用右边的对象，那么你必须定义 *reverse dunder 方法*。

让我们用乘法的例子来证明这一点:

```
**class** RandomNumbers:
    **def** __init__(self, a, b): 
        self.a = a 
        self.b = b

    **def** __add__(self, other):
        **if not** **isinstance**(other, RandomNumbers):
            **return** NotImplemented

        **return** RandomNumbers(other.a + self.a, other.b + self.b)

    **def** __mul__(self, other): 
        **i**f **not isinstance**(other, int): 
            **return** NotImplemented 

        **return** RandomNumbers(self.a * other, self.b * other)

    **def** __rmul__(self, other): 
        **return** self.__mul__(other)

    **def** __repr__(self):
        **return** f"{self.__class__.__qualname__}({self.a}, {self.b})"

set_a = RandomNumbers(2, 4)**print**(3 * set_a)"""
RandomNumbers(6, 12)
"""
```

问题解决了。

这只是两种类型的邓德方法，但是有几种——你可以在这里找到它们的列表。如果你对提升你的 Python 技能感兴趣，我建议你查看一下[数据营](https://www.datacamp.com/learn/python)提供的 Python 技能和职业轨迹。

*感谢阅读。*

**联系我:**
[LinkedIn](https://www.linkedin.com/in/kurtispykes/)
[Twitter](https://twitter.com/KurtisPykes)
[insta gram](https://www.instagram.com/kurtispykes/)

如果你喜欢阅读这样的故事，并希望支持我的写作，可以考虑[成为灵媒成员](https://kurtispykes.medium.com/membership)。每月支付 5 美元，你就可以无限制地阅读媒体上的故事。如果你使用[我的注册链接](https://kurtispykes.medium.com/membership)，我会收到一小笔佣金。

已经是会员了？订阅在我发布时得到通知。

[](https://kurtispykes.medium.com/subscribe) [## 每当 Kurtis Pykes 发表文章时都收到一封电子邮件。

### 每当 Kurtis Pykes 发表文章时都收到一封电子邮件。通过注册，您将创建一个中型帐户，如果您还没有…

kurtispykes.medium.com](https://kurtispykes.medium.com/subscribe)