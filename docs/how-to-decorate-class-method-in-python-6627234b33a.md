# 如何在 Python 中修饰类方法

> 原文：<https://towardsdatascience.com/how-to-decorate-class-method-in-python-6627234b33a>

![](img/8c0385f7f7d025d8d1d84a20a86b16e8.png)

劳拉·阿岱在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

在 Python 中，函数是一级对象。这意味着它可以作为参数传递给其他函数。使用 decorator，无需重新编写函数代码就可以改变 Python 函数的行为。

Python decorator 的一个流行例子是`@property`,它允许以安全的方式访问函数的值。您可以使用`@property`的 setter 属性安全地修改值。更多细节可以在这里找到[。](https://www.tutorialsteacher.com/python/property-decorator)

Python 的装饰器特性天生就令人困惑(至少对我来说是这样)。此外，我对给定代码中的信息流非常挑剔。我发现使用`class`作为一种尽可能保持代码简单和抽象的方式很方便。为了演示一种简洁的方式来编写`decorate`代码，我编写了一个示例代码。

我在这个例子中定义了两个`classes`。

*   `GenericClass`:带有变量`initial_value`和方法`operation`的演示类。`operation`的主要目标是返回作为参数传入的值的总和。
*   `DecoratorClass`:包含一个`wrapper`函数，用于修饰`GenericClass`对象的`operation`方法。`wrapper`函数的主要目的是使用`operator`函数和`value`变量来修饰`operation`方法。该操作符函数可以是 Python 的内置函数，如`max`或`min`,`value`可以是任意的`float`数。本质上，`wrapper`确保当`<GenericClass_instance>.operation(*args)`被调用时`operator(value, <GenericClass_instance>.operation(*args))`被返回。

让我们深入一下这个例子。让我们构造一个`<GenericClass>`的实例。

```
example = GenericClass(initial_value = -10)
```

这创建了一个`<GenericClass>`的`example`实例和 10 的实例变量`initial_value`。

让我们用实例变量`operator`和`value`构造一个`<DecoratorClass>`的实例。

```
always_positive = DecoratorClass(operator=max, value=0.0)
```

现在让我们来看几个更小的`always_positive`实例用例的例子。

*代码*:

获得 5、6 和-7 之间的最小值

```
min(5, 6, -7)
```

*输出*:

```
-7
```

但是现在使用`always_positive`实例，我们可以修改输出，使其返回值大于 0。注意在`<DecoratorClass>`实例中，操作符是`max`，值是 0.0。这意味着包装函数的输出是这个修改的函数，使得`<modified_function> = max(0.0, <func>)`

*代码:*

```
modified_function = always_positive(min)print(modified_function)
```

*输出:*

```
<function __main__.DecoratorClass.__call__.<locals>.wrapper at #ID>
```

输出返回修改函数的`type`和`id()`。

现在让我们在这个`modified_function`中传递几个参数并观察输出。

*代码*:

```
modified_function(5, 6, 7)
```

*输出*:

`5`

这是显而易见的，因为`min(5, 6, 7)`是 5，并且即使函数被修改为总是正的，输出仍然保持为 5，因为显而易见 5 大于 0。

*代码:*

```
modified_function(5, 6, -7)
```

*输出:*

`0.0`

由于作为`modified_function`参数传递的所有数字中最小的是-7，并且-7 明显小于 0，所以现在结果被修改为`always_positive.value`(在本例中为 0.0)。

现在让我们进一步扩展这个概念，使用 python decorator 修改类方法`<GenericClass>.operation`。装饰器`<DecoratorClass>`可以通过如下传递参数来实例化。

```
Class GenericClass:.
.
. @DecoratorClass(operator=max, value=0.0)
   def operation(self, *args) --> float:
       ...
```

这用装饰器修改了类方法`<GenericClass>.operation()`,方式与我之前的例子`modified_function`相似。在这种情况下，代码更加抽象和复杂，可以处理任何帮助函数(如`min`或`max`等)。).

这个概念可以进一步扩展到修饰任何类方法来改变它的行为。像这样使用装饰函数的一个用例是“防止”更复杂函数的错误输出。

**举例:**

这个演示代码可以使用 Google Colab 打开。

# 如果你喜欢它…

如果你喜欢这篇文章，你可能也会喜欢我其他类似主题的文章。

[](https://sidbannet.medium.com/membership)  

和 [*关注我*](https://medium.com/@sidbannet) 即将发布的关于随机建模、数据科学和 python 技巧的每周文章。

## 我以前的故事:

[](/how-to-analyze-time-series-data-with-pandas-4dea936fe012)  [](/how-to-build-plotly-choropleth-map-with-covid-data-using-pandas-in-google-colab-45951040b8e4)  

A 关于我——我开发**高性能计算模型**来理解*湍流*、*多相流、*和*燃烧火焰*。我应用**数据科学**来加速*推进*装置的设计创新。我于 2011 年获得了威斯康辛大学麦迪逊分校的博士学位，主修*机械和化学工程*，辅修*数学*、*统计学*和*计算机科学*。请随意查看我的 GitHub 资源库，并在[*Linkedin*](https://www.linkedin.com/in/sidban)*上关注我。*