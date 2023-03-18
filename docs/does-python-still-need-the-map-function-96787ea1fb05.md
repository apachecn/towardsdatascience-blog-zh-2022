# Python 还需要 map()函数吗？

> 原文：<https://towardsdatascience.com/does-python-still-need-the-map-function-96787ea1fb05>

## 有了各种备选方案，Python 的 map()似乎就显得多余了。那么，Python 到底需不需要呢？

![](img/7622f49d5594235527e9e08bb7075852.png)

Python 需要 map()函数吗？Muhammad Haikal Sjukri 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

别担心，这不会是第一百万篇关于如何在 Python 中使用`map()`的文章。我不会告诉你这比列表理解或循环更好或更差。我不打算将它与相应的生成器或列表理解进行对比。我不会说使用`map()`会让你看起来像一个高级 Python 开发者…

你可以在 Medium 上发表的其他文章中读到所有这些内容。尽管内置的`map()`函数在 Python 开发人员中并不太受欢迎(你很少会在产品代码中发现它)，但它在 Python 作者中却很受欢迎(例如，参见[这里的](https://medium.com/swlh/higher-order-functions-in-python-map-filter-and-reduce-34299fee1b21)、这里的[这里的](/pythons-map-filter-and-reduce-functions-explained-2b3817a94639)和[这里的](https://medium.com/towards-data-science/understanding-the-use-of-lambda-expressions-map-and-filter-in-python-5e03e4b18d09))。这是为什么呢？也许因为这是一个有趣的函数。类似于函数式编程；或者可以用它的替代品作为基准，而基准通常会引起注意？

大多数关于 Python 的`map()`的文章仅仅展示了如何使用，而没有展示为什么:在展示如何使用它的同时，他们通常没有解释为什么应该使用它。难怪，尽管它在 Python 作者中很受欢迎，但在中级开发人员中似乎没有得到应有的重视。

如果你想了解更多关于`map()`的知识，这篇文章就是为你准备的。我们将讨论`map()`函数在 Python 代码路线图中的位置，以及为什么不管你是否会使用它，了解这个函数都是值得的。

# 几句关于`map()`的话

做一些大多数 Python 开发人员经常做的事情:为 iterable 的每个元素调用一个函数(实际上是一个 callable)。

它接受一个可调用函数作为参数，所以它是一个高阶函数。因为这是函数式编程语言的典型特征，`map()`符合函数式编程风格。比如，你会在洛特的《T21 函数式 Python 编程》一书中发现很多`map()`的应用。我认为`map()`使用了类似的 API，而不是真正的函数式编程。这是因为我们可以把`map()`用于不纯的函数，也就是有副作用的函数；这在真正的函数式编程中是不可接受的。

是时候看看`map()`的行动了。

```
>>> numbers = [1, 4, 5, 10, 17]
>>> def double(x):
...     return x*2
```

所以，我们有一个数字列表，我们有一个函数可以将一个数字加倍。`double()`适用于单个号码:

```
>>> double(10)
20
```

如果我们用`double()`代替`numbers`会怎么样？

```
>>> double(numbers)[1, 4, 5, 10, 17, 1, 4, 5, 10, 17]
```

如果你知道 Python 中列表相乘的工作原理，这不会让你吃惊。虽然这是正常的行为，但这绝对不是我们想要达到的目标。上图，`double(numbers)`将函数`double()`应用到`numbers` *整体*(作为对象)。这不是我们想要的；我们要将`double()`应用到*中*的每一个元素`numbers`。这有很大的不同，这就是`map()`的用武之地:当你想对 iterable 的每个元素应用 callable 时，你可以使用它。

*警告*:一些语言使用名称`map`作为哈希映射；在 Python 中，字典是哈希映射。所以，请注意，当你在另一种语言中看到术语“地图”时，首先检查它代表什么。比如 R 中的`map()`相当于 Python 的`map()`；但是在 Go 中，`map()`创建一个哈希映射，工作方式类似于 Python 的`dict()`。当我第一次开始学习围棋时，这让我很困惑，但过一段时间后，你就会明白了。

你应该这样使用`map()`:

```
>>> doubled_numbers = map(double, numbers)
```

如您所见，您向`map()`提供了一个 callable 作为第一个参数，一个 iterable 作为第二个参数。它返回一个 map 对象(在 Python 3 中，但是在 Python 2 中，您将获得一个列表):

```
>>> doubled_numbers #doctest: +ELLIPSIS
<map object at ...>
```

(请注意，我使用了`#doctest: +ELLIPSIS`指令，因为本文档包含在`doctest` s 中。它帮助我确保所有示例都是正确的。你可以在文档中读到更多关于 T2 的内容。)

一个`map`对象像一个发电机一样工作。所以，即使我们在`map()`中使用了一个列表，我们得到的不是一个列表，而是一个生成器。按需评估生成器(延迟)。如果你想把一个`map`对象转换成一个列表，使用`list()`函数，它将评估所有的元素:

```
>>> list(doubled_numbers)
[2, 8, 10, 20, 34]
```

或者，您可以用任何其他方式评估`map`的元素，比如在`for`循环中。为了避免不愉快的头痛，请记住，一旦这样的对象被评估，它是空的，因此不能再被评估:

```
>>> list(doubled_numbers)
[]
```

上面，我们为单个 iterable 应用了`map()`，但是我们可以使用多个 iterable。该函数将根据它们的索引来使用它们，也就是说，首先，它将为 iterables 的第一个元素调用 callable(在索引 0 处)；然后进行第二次；诸如此类。

一个简单的例子:

```
>>> def sum_of_squares(x, y, z):
...     return x**2 + y**2 + z**2>>> x = range(5)
>>> y = [1, 1, 1, 2, 2]
>>> z = (10, 10, 5, 5, 5)
>>> SoS = map(sum_of_squares, x, y, z)
>>> list(SoS)
[101, 102, 30, 38, 45]
>>> list(map(sum_of_squares, x, x, x))
[0, 3, 12, 27, 48]
```

# `map()`的替代品

代替`map()`，你可以使用一个生成器，例如，通过一个生成器表达式:

```
>>> doubled_numbers_gen = (double(x) for x in numbers)
```

这提供了一个发电机，就像`map()`一样。当你需要一个清单时，你会更好地理解相应的清单:

```
>>> doubled_numbers_list = [double(x) for x in numbers]
```

哪个可读性更强:`map()`版本还是生成器表达式(或者列表理解)？对我来说，没有一秒钟的犹豫，生成器表达式和列表理解更清晰，即使我理解`map()`版本没有问题。但是我知道有些人会选择`map()`版本，尤其是那些最近从另一种使用类似`map()`功能的语言迁移到 Python 的人。

人们经常将`map()`与`lambda`函数结合使用，当您不想在其他地方重用该函数时，这是一个很好的解决方案。我认为对`map()` 的部分负面看法来自于这种用法，因为`lambda`函数经常会降低代码的可读性。在这种情况下，通常情况下，生成器表达式的可读性会更好。下面比较两个版本:一个是`map()`结合`lambda`，另一个是对应生成器表达式。这一次，我们将不使用我们的`double()`函数，但是我们将在调用中直接定义它:

```
# map-lambda version
map(lambda x: x*2, numbers)# generator version
(x*2 for x in numbers)
```

这两行导致相同的结果，唯一的区别是返回对象的类型:返回一个`map`对象，而后者返回一个`generator`对象。

让我们暂时回到`map()`的多次使用:

```
>>> SoS = map(sum_of_squares, x, y, z)
```

我们可以按照以下方式使用生成器表达式重写它:

```
>>> SoS_gen = (
...     sum_of_squares(x_i, y_i, z_i)
...     for x_i, y_i, z_i in zip(x, y, z)
... )>>> list(SoS_gen)
[101, 102, 30, 38, 45]
```

这次我投票给`map()`版本！除了更简洁之外，在我看来，它更清晰。发电机版本利用`zip()`功能；即使这是一个简单的函数，它也增加了命令的难度。

# 所以，我们不需要 map()，不是吗？

根据以上讨论，不存在我们必须使用`map()`功能的情况；相反，我们可以使用生成器表达式、循环或其他东西。

知道了这一点，我们还需要什么吗？

思考这个问题，我得出了我们需要 Python 中的`map()`的三个主要原因。

## ***原因一:性能***

如前所述，`map()`被懒洋洋地评估。然而，在许多情况下，评估`map()`比评估相应的生成器表达式更快。尽管相应的列表理解并不一定是这种情况，但是在优化 Python 应用程序时，我们应该记住这一点。

然而，请记住，这不是一个普遍的规则，所以你不应该假设这一点。每次都需要检查`map()`在你的代码片段中是否会更快。

只有在性能上的细微差异也很重要时，才考虑这个原因。否则，以牺牲可读性为代价使用`map()`将会收获甚微，所以你应该三思而后行。通常，节省一分钟没有任何意义。其他时候，节省一秒钟意味着很多。

## ***原因二:平行度和穿线***

当您并行化您的代码或使用线程池时，您通常会使用类似于`map()`的函数。这可以包括诸如`multiprocessing.Pool.map()`、`pathos.multiprocessing.ProcessingPool.map()`或`concurrent.futures.ThreadPoolExecutor.map()`之类的方法。所以，学会使用`map()`会帮助你理解如何使用这些功能。通常，您会希望在并行和非并行版本之间切换。由于这些函数之间的相似性，您可以非常容易地做到这一点。看:

当然，在这个简单的例子中，并行化没有意义，而且会更慢，但是我想向您展示如何做到这一点。

## ***原因 3:对于来自其他语言的 Python 新人来说简单***

这个原因是非典型的，并不涉及语言本身，但它有时很重要。对我来说，生成器表达式几乎总是更容易编写，可读性更好。然而，当我刚接触 Python 时，理解对我来说并不容易，无论是写还是理解。但是自从我在 16 年的 R 编程后来到 Python，我非常熟悉 R 的`map()`函数，它的工作方式与 Python 的`map()`完全一样。然后，对我来说，使用`map()`比使用相应的生成器表达式或列表理解要容易得多。

更重要的是，对`map()`的熟悉帮助我理解。我也能够编写 Pythonic 代码；没错，用`map()`就是 Pythonic。我们知道，第三种选择是`for`循环，但这很少是更好的(甚至是好的)选择。因此，如果有人使用 Python 并且知道这些函数是如何工作的，那么他们编写 Python 代码就容易多了。例如，从 C 语言转向 Python 的人可能会使用一个`for`循环，这在这种情况下被认为是非 Python 的。

这意味着`map()`是 Python 和其他语言之间的桥梁——一座可以帮助其他人理解语言和编写 Python 代码的桥梁。

## ***原因 4:多个可迭代的情况下的可读性***

如上图所示，当你想同时为多个 iterables 使用一个 callable 时，`map()`可以比对应的生成器表达式更易读、更简洁。因此，即使在简单的情况下,`map()`可读性较差，但在更复杂的情况下，可读性成为了它的优势。

# **结论**

有人说 Python 不需要`map()`。[回到 2005 年](https://www.artima.com/weblogs/viewpost.jsp?thread=98196)，Guido 自己想把它从 Python 中移除，还有`filter()`和`reduce()`。但是 17 年后的今天，我们仍然可以使用它，我认为——并真诚地希望——这一点不会改变。这给我们带来了本文所要讨论的两个关键问题:

*`***map()***`***函数是 Python 中必须的吗？*** 不，不是。你可以用其他方法达到同样的效果。*

****既然如此，Python 还需要*** `***map()***` ***函数吗？*** 是的，确实如此。不是因为它是必须的，而是因为它仍然被使用，它服务于各种有价值的目的。*

*我认为 Python 开发者应该知道如何使用`map()`，即使他们不经常使用它。在某些情况下，它可以帮助提高性能，并行化代码，只需很小的改动，或者提高代码的可读性。它有助于理解。它还可以帮助来自其他语言的开发人员使用地道的 Python——因为，是的，`map()`仍然是 Python。*

*正是因为这些原因，我认为`map()`值得在 Python 代码库中占有一席之地，即使它并不经常被使用。*

## ***资源***

*   *[https://medium . com/swlh/higher-order-functions-in-python-map-filter-and-reduce-34299 fee1b 21](https://medium.com/swlh/higher-order-functions-in-python-map-filter-and-reduce-34299fee1b21)*
*   *[https://towards data science . com/python-map-filter-and-reduce-functions-explained-2b 3817 a 94639](/pythons-map-filter-and-reduce-functions-explained-2b3817a94639)*
*   *[https://medium . com/forward-data-science/understanding-the-use-of-lambda-expressions-map-and-filter-in-python-5e 03 E4 b 18d 09](https://medium.com/towards-data-science/understanding-the-use-of-lambda-expressions-map-and-filter-in-python-5e03e4b18d09)*
*   *https://docs.python.org/3/library/doctest.html*
*   *【https://www.artima.com/weblogs/viewpost.jsp?thread=98196 *
*   *洛特，S.F. (2018)。*函数式 Python 编程*。第二版。包装出版公司。*