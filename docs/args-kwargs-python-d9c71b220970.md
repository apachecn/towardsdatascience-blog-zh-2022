# * Python 中的 args 和**kwargs

> 原文：<https://towardsdatascience.com/args-kwargs-python-d9c71b220970>

## 讨论位置参数和关键字参数之间的区别，以及如何在 Python 中使用*args 和**kwargs

![](img/62f6235dbe68ff042af8ca420eb00c9b.png)

[潘云波](https://unsplash.com/@panyunbo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/t/3d-renders?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍照

## 介绍

在 Python 中，函数调用中提供的参数通过赋值传递(即通过对象引用)。这意味着每当调用一个函数时，每个参数都将变成一个指向指定值的变量。显然，这些变量的范围将限于被调用的函数。

这意味着函数和调用者通过引用共享对象，没有名称别名。因此，通过改变函数中的参数名，我们不应该期望调用者的名字也会改变。然而，只要我们的可变对象在一个函数中发生变化，我们就应该预料到这会对调用者产生影响，因为它们共享相同的对象引用。

在今天的文章中，我们将讨论在用 Python 编写(或调用)函数时，位置参数和关键字参数之间的区别。此外，我们将通过几个例子来演示如何在您的代码中使用`*args`和`**kwargs`,以使代码更加有效和 Pythonic 化。

## Python 中的位置参数与关键字参数

Python 函数(或方法)的参数可以通过关键字名称或位置传递，该语言能够在一次调用中收集位置和/或关键字参数。

**位置参数**指的是当感兴趣的函数或方法被调用时，基于它们被提供的位置索引被解析的参数类型。因此，提供位置参数的顺序很重要，因为它们**从左到右**匹配。

作为一个例子，让我们考虑内置的 Python 函数`[round()](https://docs.python.org/3/library/functions.html#round)`，它用于将输入数字四舍五入到小数点后指定的精度。函数的定义是`**round**(*number*[, *ndigits*])`，这意味着输入`number`是强制的，而`ndigits`参数是可选的，对应于期望的精度。

现在让我们假设我们有一个浮点变量`a = 10.2254`,我们想把它的精度限制在小数点后两位。我们可以通过调用`round()`函数来这样做，并提供两个参数作为位置参数。

```
>>> a = 10.2254
>>> round(a, 2)
10.23
```

由于我们提供了位置参数(即，我们没有为调用者中提供的每个值指定关键字参数)，根据函数定义，第一个位置参数将对应于`number`，第二个位置参数将对应于选项`ndigits`参数。

另一方面，**关键字参数**通过以`name=value`的形式指定关键字被传递给一个函数。因此，这些参数在调用者中传递的顺序并不重要，因为它们是由参数名匹配的**。**

回到使用`round()`函数的例子，我们可以通过传递关键字参数来调用它。

```
>>> a = 10.2254
>>> round(number=a, ndigits=2)
10.23
```

如前所述，我们提供关键字参数的顺序并不重要:

```
>>> a = 10.2254
>>> round(ndigits=2, number=a)
10.23
```

注意，我们甚至可以将位置参数和关键字参数结合起来，后者应该在前者之后指定——

```
>>> a = 10.2254
>>> round(a, ndigits=2)
10.23
```

但是请注意，如果在位置参数之前提供关键字参数，将会引发一个`SyntaxError`。

```
SyntaxError: positional argument follows keyword argument
```

## 位置参数和*参数

现在让我们假设我们想用 Python 写一个函数，它接受任意数量的参数。一种选择是在一个集合中传递参数——比如一个列表——但是在大多数情况下这并不方便。此外，这个想法不太符合 Pythonic 但这正是任意参数列表发挥作用的地方。

我们可以在定义函数时利用`*args`习语来指定它实际上可以接受任意数量的参数。实际上，正确处理所提供的参数取决于实现。

星号`*`被称为**解包操作符**，并将**返回一个包含调用者提供的所有参数的元组**。

例如，让我们考虑一个相当简单的函数，它接受任意数量的整数并返回它们的和

```
def sum_nums(*args):
    sum = 0
    for n in args:
        sum += n
    return sum
```

现在，我们可以使用任意数量的参数来调用上面的函数:

```
>>> sum_nums(10, 20)
30
>>> sum_nums(10, 20, 30)
60
>>> sum_nums(5)
5
```

请注意，您可以在函数定义中组合普通参数和任意参数:

```
def my_func(param, *args):
    ...
```

## 关键字参数和* *关键字

同样，我们有时可能想要编写能够接受任意数量关键字参数的函数。

当解包来自`**`的关键字参数时，结果将是一个 Python 字典，其中键对应于关键字名称，值对应于所提供的实际参数值。

```
def my_func(**kwargs):
    for key, val in kwargs.items():
        print(key, val)
```

现在我们可以用任意多的关键字参数来调用函数:

```
>>> my_func(a='hello', b=10)
a hello
b 10
>>> my_func(param1=True, param2=10.5)
param1 True
param2 10.5
```

再次提醒一下，任意关键字参数习语可以与普通参数和任意位置参数结合使用:

```
def my_func(param, *args, **kwargs):
    ...
```

## 何时使用*args 和**kwargs

装饰者是一个很好的实际例子，其中`*args`和/或`**kwargs`通常很有用。在 Python 中，decorator 是一个函数，它接受另一个函数作为参数，修饰它(即丰富它的功能)并最终返回它。

假设我们想要创建一个装饰器，负责向标准输出报告函数的执行时间。

```
import functools
import time

def execution_time(func): @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print(f'{func.__name__} took {end - start}s to run.') return wrapper
```

包装函数将简单地接受任意的位置和关键字参数，然后将这些参数传递给被修饰的函数。

## 最后的想法

在今天的文章中，我们讨论了位置参数和关键字参数，以及它们在用 Python 编写或调用函数时的主要区别。

此外，我们还讨论了`*args`和`**kwargs`之间的主要区别，以及如何根据您想要实现的目标，在您自己的职能中利用它们。此外，我们展示了如何在实践中使用来自`*args`和`**kwargs`的位置和关键字参数。

最后，我们通过一个实例展示了`*args`和`**kwargs`在函数包装器(即装饰器)环境中的应用。

值得一提的是，**实际符号使用星形符号** ( `*`和`**`)，并且都对应于**任意参数列表**。`***args**`**`****kwargs**`**这两个名字无非是一个约定**(在社区中相当流行，通常用来描述任意的参数列表)。因此，您不必这样引用它们——例如，您甚至可以在函数定义中将它们命名为`*hello`和`**hey`,尽管我不建议您使用这样的命名约定:)。**

**[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。****

**<https://gmyrianthous.medium.com/membership> ** 

****你可能也会喜欢****

**</pycache-python-991424aabad8> ** **</python-gil-e63f18a08c65> ** **</duck-typing-python-7aeac97e11f8> **