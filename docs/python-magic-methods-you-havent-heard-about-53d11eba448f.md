# 你可能没听说过的 Python 魔术方法

> 原文：<https://towardsdatascience.com/python-magic-methods-you-havent-heard-about-53d11eba448f>

## 还有许多鲜为人知的 Python 魔术方法——让我们来看看它们的作用以及如何在我们的代码中使用它们

![](img/5393c114bc6cc07173cb9e11b0757830.png)

[Aaron Huber](https://unsplash.com/@aahubs?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

Python 的神奇方法——也称为 *dunder* (双下划线)方法——可以用来实现很多很酷的东西。大多数时候我们用它们来做简单的事情，比如构造函数(`__init__`)、字符串表示(`__str__`、`__repr__`)或者算术运算符(`__add__` / `__mul__`)。然而，还有很多你可能没有听说过的神奇方法，在这篇文章中，我们将探索所有这些方法(甚至是隐藏的和未记录的)！

# 迭代器长度

我们都知道可以用来在容器类上实现`len()`功能的`__len__`方法。但是，如果你想得到一个实现迭代器的类对象的长度呢？

你所需要做的就是实现`__length_hint__`方法，正如你在上面看到的，它也存在于内置迭代器(而不是生成器)中。此外，如您所见，它还支持动态长度变化。尽管如此——顾名思义——这实际上只是一个*提示*,而且可能完全不准确:对于列表迭代器，你会得到精确的结果，而对于其他迭代器则不一定。然而，即使它不准确，它对优化也很有帮助，正如不久前介绍它的 [PEP 424](https://peps.python.org/pep-0424/) 中所解释的。

# 元编程

您很少看到的大部分神奇方法都与元编程有关，虽然元编程可能不是您每天都必须使用的，但是有一些方便的技巧您可以使用它。

其中一个技巧是使用`__init_subclass__`作为扩展基类功能的快捷方式，而不必处理元类:

在这里，我们使用它向基类添加关键字参数，可以在定义子类时设置。在真实的用例中，您可能会在想要处理所提供的参数的情况下使用它，而不仅仅是分配给一个属性。

虽然这可能看起来非常晦涩，很少有用，但您可能已经遇到过很多次了，因为它可以在构建 API 时使用，用户可以像在 [SQLAlchemy models](https://docs.sqlalchemy.org/en/14/orm/inheritance.html) 或 [Flask Views](https://github.com/pallets/flask/blob/9b44bf2818d8e3cde422ad7f43fb33dfc6737289/src/flask/views.py#L162) 中那样子类化您的父类。

您可能会发现使用的另一个元类魔术方法是`__call__`。这个方法允许你定制当你*调用*一个类实例时会发生什么:

有趣的是，您可以用它来创建不能被调用的类:

如果你有一个只有静态方法的类，因此没有很好的理由去创建这个类的实例，这是很有用的。

我想到的另一个类似的用例是单例模式——一个最多只能有一个实例的类:

在这里，我们通过实现一个只能有一个实例的全局记录器类来演示这一点。这个概念可能看起来有点复杂，但是这个实现非常简单——`Singleton`类拥有一个私有的`__instance`——如果没有，它就被创建并赋给属性，如果已经存在，它就被返回。

现在，假设您有一个类，并且您想在不调用`__init__`的情况下创建它的一个实例。这个`__new__`魔法方法可以帮上忙:

有些情况下，您可能需要绕过创建实例的通常过程，上面的代码显示了您如何做到这一点。我们不调用`Document(...)`，而是调用`Document.__new__(Document)`，这将创建一个裸实例，而不调用`__init__`。因此，实例属性——在本例中是`text`——没有初始化，为了解决这个问题，我们可以使用`setattr`函数(顺便说一下，这也是一个神奇的方法——`__setattr__`)。

你可能想知道为什么你会想这么做。一个例子是实现可选的构造函数，如下所示:

这里我们定义`from_file`方法，它作为一个构造器，首先用`__new__`创建实例，然后在不调用`__init__`的情况下配置它。

接下来元编程相关的魔术方法我们将在这里看一看是`__getattr__`。当普通属性访问失败时，调用此方法。这可以用来将对缺失方法的访问/调用委托给另一个类:

假设我们想用一些额外的函数定义 string 的自定义实现，比如上面的`custom_operation`。然而，我们不想重复实现每一个单独的字符串方法，例如`split`、`join`、`capitalize`等等。因此，我们使用`__getattr__`来调用这些现有的字符串方法，以防在我们的类中找不到它们。

虽然这对于普通方法来说效果很好，但是请注意，在上面的例子中，魔法方法`__add__`提供的连接等操作并没有被委托。因此，如果我们希望它们也能工作，那么我们就必须重新实现它们。

# 反省

我们将试用的最终元编程相关魔术方法是`__getattribute__`。这张看起来和之前的`__getattr__`很像。然而有一点小小的不同——正如已经提到的,`__getattr__`仅在属性查找失败时被调用，而`__getattribute__`在属性查找尝试之前被调用*。*

因此，您可以使用`__getattribute__`来控制对属性的访问，或者您可以创建一个 decorator 来记录每次访问实例属性的尝试:

`logger` decorator 函数从记录它所修饰的类的原始`__getattribute__`方法开始。然后用自定义方法替换它，在调用原始的`__getattribute__`方法之前，首先记录被访问属性的名称。

# 魔法属性

到目前为止，我们只讨论了魔术方法，但是 Python 中也有相当多的魔术变量/属性。其中一个是`__all__`:

这个神奇的属性可以用来定义从一个模块中导出哪些变量和函数。在这个例子中，我们用单个文件(`__init__.py`)在`.../some_module/`中创建了一个 Python 模块。在这个文件中，我们定义了 2 个变量和一个函数，我们只导出其中的 2 个变量(`func`和`some_var`)。如果我们试图在其他 Python 程序中导入`some_module`的内容，我们只能得到 2 个导出的内容。

不过要注意的是，`__all__`变量只影响上面显示的`*`导入，你仍然可以通过`import some_other_var from some_module`这样的导入来导入未导出的函数和变量。

您可能见过的另一个双下划线变量(模块属性)是`__file__`。这个变量只是标识了访问它的文件的路径:

结合`__all__`和`__file__`，你可以加载一个文件夹中的所有模块:

最后一个我们要尝试的是`__debug__`属性。显然，这可以用于调试，但更具体地说，它可以用于更好地控制断言:

如果我们使用`python example.py`正常运行这段代码，我们会看到`"debugging logs"`被打印出来，但是如果我们使用`python3 -O example.py`，优化标志(`-O`)会将`__debug__`设置为假，并去掉调试消息。因此，如果您在生产环境中使用`-O`运行您的代码，您将不必担心调试遗留下来的被遗忘的`print`调用，因为它们将被全部去除。

# 隐藏和未记录

以上所有的方法和属性可能有些陌生，但是它们都在 Python 文档中。然而，有几个没有明确记录和/或有些隐藏。

例如，您可以运行以下代码来发现几个新代码:

除了这些，还有很多在 Python bug tracker[BPO 23639](https://github.com/python/cpython/issues/67827)中列出。正如在那里指出的，它们中的大多数是不应该被访问的实现细节或私有名称。所以他们不被记录可能是最好的。

# 自己做？

现在，有了这么多神奇的方法和属性，你真的能自己创造吗？你可以，但你不应该。

双下划线名称是为 Python 语言的未来扩展保留的，不应用于您自己的代码。如果您决定在您的代码中使用这样的名称，那么您将冒着在将来将它们添加到 Python 解释器中的风险，这很可能会破坏您的代码。

# 结束语

在这篇文章中，我们看了一些我认为有用或有趣的鲜为人知的神奇方法和属性，然而在文档中列出了更多可能对你有用的方法和属性。大多数可以在 [Python 数据模型文档](https://docs.python.org/3/reference/datamodel.html#special-method-names)中找到。然而，如果你想更深入地挖掘，你可以尝试在 Python 文档中搜索`"__"`，这将会出现[更多的方法和属性](https://docs.python.org/3/search.html?q=__&check_keywords=yes&area=default)来探索和使用。

*本文原帖*[*martinheinz . dev*](https://martinheinz.dev/blog/87)

[成为会员](https://medium.com/@martin.heinz/membership)阅读媒体上的每一个故事。**你的会员费直接支持我和你看的其他作家。**你还可以在媒体上看到所有的故事。

<https://medium.com/@martin.heinz/membership>  

你可能也喜欢…

<https://medium.com/@martin.heinz/python-cli-tricks-that-dont-require-any-code-whatsoever-e7bdb9409aeb>  <https://betterprogramming.pub/all-the-ways-to-introspect-python-objects-at-runtime-80e6991b4cc6> 