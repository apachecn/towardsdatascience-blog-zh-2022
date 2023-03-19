# 多态性:理解在 Scikit-Learn 中创建自定义类(如自定义转换器)背后的概念

> 原文：<https://towardsdatascience.com/polymorphism-understand-the-concept-behind-creating-custom-classes-like-custom-transformers-in-eb9d8b4f5e30>

## 曾经想知道为什么一些 Python 类从无到有的地方调用方法吗？还是实现一些方法只是为了过关？

如果你曾经遇到过 Scikit-Learn 定制转换器，你很可能对上面的现象很熟悉。如果是这样的话，这篇文章是给你的。我们将深入研究支持这种行为的叫做*多态性*的概念，并且我们将构建一些定制类来获得一些实践经验和更深入的理解。

![](img/05f516a5a8425504d34d8e8b8f1b9579.png)

照片由[Meagan car science](https://unsplash.com/@mcarsience_photography?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

Scikit-learn transformers 是为生产中的数据准备建立管道的一套很棒的工具。虽然内置的转换器列表非常详尽，但是构建您的定制转换器是自动化定制特性转换和实验的一个很好的方式。如果您曾经使用过 scikit-learn transformer，您很可能会遇到以下常用模式:

```
# defining a custom transformer
    class CustomTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self  
        def transform(self, X):
            ...... # calling fit_transform() 
  customTransformer = CustomTransformer()
  data = customTransformer.fit_transform(data)
```

但是如果你来自非编程背景，那么`fit_transform()`方法没有在`CustomTransformer`类中定义，却可以从该类中调用，这似乎有点令人困惑。此外，如果您已经知道该方法来自上面的某个类，但是想到它如何能够使用没有在它所属的同一个类中定义的方法，您可能会感到困惑。例如，在官方 GitHub 库[这里](https://github.com/scikit-learn/scikit-learn/blob/801cca8e73215d4946f05379319d97156be659d6/sklearn/base.py#L531)中检查`TransformerMixin`类的脚本，你不会在`TransformerMixin`类中找到任何为`fit()`或`transform()`定义的方法。

在本文中，我们将尝试理解这些概念:多态和鸭类型，这些概念支持这些行为。我们还会做一些动手练习来加深理解。

# 什么是多态性？

一般来说，*多态性*意味着一个对象采取不同形式的能力。一个具体的例子是，不同类的对象包含相同的方法，但表现出不同的行为。

例如，在 Python 中，我们可以运行类似于`1+2`或`'a' + 'b'`的操作，并分别得到`3`和`ab`的结果。Python 在幕后调用了一个已经在 string 和 integer 类中实现的名为`__add__()`的*魔法方法*。关于 Python 如何将这些核心语法转换成特殊方法的详细信息，请查看我上一篇关于 Python 核心语法的文章。

<https://betterprogramming.pub/python-core-syntax-and-the-magic-behind-them-3c912985b87c>  

这个神奇的方法——`__add__()`是多态性的一个例子——它是同一个方法，但是根据调用它的类对象，它调整它的行为，从累加数字到连接字符串。

> *在 Python 类上下文中，我们可以通过两种方式实现多态性:继承和 Duck 类型化。*

# 遗传多态性

*继承*在面向对象的编程环境中，意味着我们从另一个类继承或接收类属性。我们继承的类叫做*超类*，我们继承的属性叫做*子类*。因为这篇文章的重点不是继承，我们将跳到一个例子中，希望这个概念在我们进行的过程中有意义。

但是如果他们不知道或者你需要快速复习，请随意阅读我以前关于理解继承和子类的文章。

</object-oriented-programming-in-python-inheritance-and-subclass-9c62ad027278>  

对于我们的例子，我们将创建一个名为`InheritList`的超类和三个子类:`DefaultList`、`EvenList`和`OddList`来运行继承和多态的例子。

示例超类 01

示例子类

## 遗产

在上面的代码块中，注意我们没有在`DefaultList`类中实现任何方法。请注意，在下面的代码块中，我们还可以从该类创建的实例中调用方法(例如`add_value()`、`get_list()`)。因为`DefaultList`子类从它的超类- `InheritList`继承了这些方法。这是继承在起作用。nums = [1，2，3，4，5]

```
defaultNumList = DefaultList()[defaultNumList.add_value(i) for i in nums]print(f"List with all added values: {defaultNumList.get_list()}")​# removes the last item from the listdefaultNumList.remove_value()print(f"List after removing the last item: {defaultNumList.get_list()}")>>List with all added values: [1, 2, 3, 4, 5]
>>List after removing the last item: [1, 2, 3, 4]
```

上面的例子展示了基本的继承——我们从超类中获得所有的属性，并按原样使用它们。但是我们可以改变或更新子类中继承的方法，就像我们在其他两个子类中所做的一样— `EvenList`和`OddList`。

## 方法覆盖

在`EvenList`和`OddList`类中，我们修改了`remove_value()`方法，这样`EvenList`类将从构建的列表中移除所有奇数值，而`OddList`将移除所有偶数值。通过这样做，我们将引入多态性——其中`remove_value()`在两种情况下会有不同的表现。

演示:方法重写

```
>>evenNumList with all the values: [1, 2, 3, 4, 5]
>>evenNumList after applying remove_value(): [2, 4]

>>oddNumList with all the values: [1, 2, 3, 4, 5]
>>oddNumList after applying remove_value(): [1, 3, 5]
```

# 鸭分型多态性

在详细介绍 Duck 类型之前，让我们先谈谈在超类`InheritList`中实现的另一个方法`do_all()`。它接受一个值作为输入，将其添加到列表中，从列表中删除不需要的值，并返回最终的列表。要完成所有这些任务，需要依靠其他内部方法:`add_value()`、`remove_value()`和`get_list()`。看看下面的演示。

```
print(f"evenNumList after calling do_call(58): {evenNumList.do_all(58)}")print(f"oddNumList after calling do_call(58): {oddNumList.do_all(55)}")>>evenNumList after calling do_call(58): [2, 4, 58, 58]
>>oddNumList after calling do_call(58): [1, 3, 5, 55, 55]
```

但是 Python 允许我们更灵活地实现这一点。例如，我们可以从超类中完全移除`remove_value()`方法，创建一个单独的只有`combine_all()`方法的类，并且仍然能够毫无问题地使用它。全拜鸭子打字所赐！

基本上，我们不关心依赖属性是否来自同一个类。只要依赖属性可用，我们就很好。这基本上反映了代表鸭子类型的广泛使用的引用:

> “如果它像鸭子一样走路，像鸭子一样游泳，像鸭子一样嘎嘎叫，那么它很可能就是一只鸭子。”

为了演示，让我们创建一个名为`ComboFunc`的新类，它只有一个方法`combine_all()`，这个方法将执行与`do_all()`方法相同的功能。同样，让我们创建一个新的子类，它将拥有一个先前创建的子类— `EvenList`和这个新类作为超类。

注意，我们没有在这两个类中定义任何依赖方法(`add_value()`、`remove_value()`和`get_list()`)。然而，我们将能够成功地从`GenDuckList`类的实例中调用`combine_all()`方法。因为依赖方法将从`EvenList`类继承而来，而`combine_all()`方法并不关心它们来自哪里，只要它们存在。

```
>>Initial list: [1, 2, 3, 4, 5]
>>Final list: [2, 4, 40]
```

请注意，我们也可以通过其他方式完成上述任务，

1.  如果我们需要定制的东西，我们也可以完全避免从`EvenList`类继承任何东西，并在类内部实现依赖方法。或者，
2.  我们可以把它作为一个超类，但覆盖任何特定的依赖方法，使它更加定制化。总的来说，多态性让我们变得更加灵活，可以轻松地重用已经实现的方法。或者，
3.  我们可以从超类中移除`remove_value()`并在我们的`GenDuckList`类中实现它，但仍然能够执行相同的任务。

因此，为了完成这个循环，当我们使用`BaseEstimator`和`TransformerMixin`类作为超类在 scikit-learn 中构建自定义转换器时，我们基本上应用 *duck typing* 来实现*多态*。与之相关的是，您可以将`GenDuckList`视为一个虚拟的定制 transformer 类，`ComboFunc`视为一个虚拟的`TransformerMixin`类，`EvenList`视为一个虚拟的`BaseEstimator`类。上面的 duck 类型化示例和开始的 transformer 示例之间的实现级别差异是，我们从一个超类继承了`remove_value()`方法，而在 custom transformer 中，我们在 custom 类中定义它——上面提到的第三种替代方式。

感谢您阅读这篇文章。希望它能帮助您理解 Python 类上下文中多态性的概念。如果你喜欢这篇文章，请考虑关注我的个人资料，以获得关于我未来文章的通知。

<https://levelup.gitconnected.com/use-modules-to-better-organize-your-python-code-75690ba6b6e>  <https://levelup.gitconnected.com/use-modules-to-better-organize-your-python-code-75690ba6b6e> 