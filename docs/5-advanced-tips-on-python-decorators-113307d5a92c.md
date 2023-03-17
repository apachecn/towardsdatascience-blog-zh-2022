# Python Decorators 的 5 个高级技巧

> 原文：<https://towardsdatascience.com/5-advanced-tips-on-python-decorators-113307d5a92c>

# Python Decorators 的 5 个高级技巧

## 你想写出简洁、易读、高效的代码吗？嗯，python decorators 可能会在您的旅程中帮助您。

![](img/0eb9b0eb19d4eb7c1cd7d08eff538fd5.png)

毛里西奥·穆尼奥斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

在 Fluent Python 的第 7 章中，卢西亚诺·拉马尔霍讨论了装饰器和闭包。它们在基本的 DS 工作中并不常见，但是当您开始编写异步代码来构建生产模型时，它们就成为了一个无价的工具。

事不宜迟，我们开始吧。

# 1 —什么是室内设计师？

在我们进入技巧之前，让我们先了解一下装饰者是如何工作的。

装饰器是简单的接受一个函数作为输入的函数。从语法上来说，它们通常被描述为“修饰”函数上方的一行中的`@my_decorator`,例如…

```
temp = 0def decorator_1(func):
  print('Running our function')
  return func@decorator_1
def temperature():
  return tempprint(temperature())
```

然而，真正发生的是当我们调用`temperature()`时，我们只是运行`decorator_1(temperature())`，如下所示…

```
temp = 0def decorator_1(func):
  print('Running our function')
  return funcdef temperature():
  return tempdecorator_1(temperature())
```

好的，**所以 decorators 是以另一个函数作为参数的函数。但是我们为什么要这么做呢？**

嗯，装修工真的是多才多艺，厉害。它们通常用于异步回调和函数式编程。还可以利用它们在函数中构建类似类的功能，从而减少开发时间和内存消耗。

让我们进入一些提示…

# 2 —属性装饰者

**提示:使用内置的** `**@property**` **来增强 setter/getter 功能。**

最常见的内置装饰器之一是`@property`。许多 OOP 语言，比如 Java 和 C++，建议使用 getter/setter 范例。这些函数用于确保我们的变量不会返回/被赋予不正确的值。一个例子是要求我们的`temp`变量大于绝对零度…

我们可以使用`@property`方法增加很多这种功能，使我们的代码更具可读性和动态性…

注意，为了简洁起见，我们删除了`my_setter()`中的所有条件逻辑，但是概念是相同的。`c.temperature`不是比`c.my_getter()`可读性强很多吗？我当然这么认为。

但在我们继续之前，有一个重要的挑剔之处。**在 python 中，没有私有变量这种东西。**前缀`_`表示变量是受保护的，不应该在类外引用。但是，您仍然可以…

```
c = my_vars(500)
print(c._temp)      # 500c._temp = -10000
print(c._temp)      # -1000
```

python 中不存在真正的私有变量，这是一个有趣的设计选择。论点是 OOP 中的私有变量实际上并不是私有的——如果有人想访问它们，他们可以改变源类的代码，使变量成为公共的。

Python 鼓励“负责任的开发”，并允许你从外部访问类中的任何东西。

# 3 —类方法和静态方法

**提示:使用内置的** `**@classmethod**` **和** `**@staticmethod**` **来扩充类功能。**

这两个装饰者经常被混淆，但是他们的区别非常明显。

*   `@classmethod`将类作为参数。它绑定到**类本身，**而不是类实例。因此，它**可以**跨所有实例访问或修改该类。
*   `@staticmethod`不把类作为参数。它绑定到**类实例，**而不是类本身。因此，它**根本不能**访问或修改这个类。

让我们看一个例子…

classmethods 最大的用例是它们作为我们类的可选构造函数的能力，这对于[多态性](https://stackoverflow.com/questions/5738470/whats-an-example-use-case-for-a-python-classmethod)非常有用。即使你没有在继承上做什么疯狂的事情，不用 if/else 语句就能实例化类的不同版本也是不错的。

另一方面，静态方法通常用作完全独立于类状态的实用函数。注意，我们的`isAdult(age)`函数不需要通常的`self`参数，所以即使它想引用，也不能引用这个类。

# 4 —快速提示

**提示:使用** `**@functools.wraps**` **保存函数信息。**

记住，装饰器只是接受另一个函数作为参数的函数。所以，当我们调用装饰函数时，我们实际上首先调用了装饰器。这个流程覆盖了关于修饰函数的信息，比如`__name__`和 __doc__ 字段。

为了克服这个问题，我们可以利用另一个装饰者…

没有`@wraps`装饰器，我们的打印语句的输出如下。

```
print(f(5))        # 30
print(f.__name__)  # 'call_func'
print(f.__doc__)   # ''
```

**为了避免覆盖重要的函数信息，一定要使用** `**@functools.wraps**` **装饰器。**

# 5-创建自定义装饰器

提示:构建你自己的装饰器来扩充你的工作流程，但是要小心。

decorators 中的变量作用域有点奇怪。我们没有时间深入细节，但如果你真的如此专注，这里有一篇 29 分钟的文章。请注意，如果出现以下错误，请仔细阅读 decorator scope:

![](img/e9c907e34cd8fe2414591afab1b2c5a1.png)

有了这个免责声明，让我们继续看一些有用的自定义装饰器…

## 5.1 —基于装饰器的商店功能

下面的代码在函数被调用时将它们附加到一个列表中。

一个潜在的用例是单元测试，就像 [pytest](https://docs.pytest.org/en/6.2.x/) 一样。假设我们有快速测试和慢速测试。我们可以给每个函数添加一个`@slow`或`@fast`装饰器，然后调用相应列表中的每个值，而不是手动将每个函数分配给一个单独的列表。

## 5.2 —时间数据查询或模型训练

下面的代码打印了你的函数的运行时间。

如果您正在运行任何类型的数据查询或使用错误日志训练模型，那么估计运行时间真的很有用。仅仅通过一个`@time_it`装饰器，您就可以获得任何函数的运行时估计。

## 5.3-对功能输入执行流量控制

以下代码在执行函数之前对函数参数执行条件检查。

这个装饰器对我们所有函数的参数`x`应用条件逻辑。如果没有装饰器，我们必须将`if is not None`流控制写入每个函数。

这些只是几个例子。装修工真的很有用！

*感谢阅读！我会再写 18 篇文章，把学术研究带到 DS 行业。查看我的评论，链接到这篇文章的主要来源和一些有用的资源。*