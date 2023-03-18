# 优化 Python 应用程序中的内存使用

> 原文：<https://towardsdatascience.com/optimizing-memory-usage-in-python-applications-f591fc914df5>

## 通过这些简单的技巧和高效的数据结构，找出您的 Python 应用程序使用过多内存的原因，并减少它们的 RAM 使用

![](img/d3a3bb4c8dcf2ef172462fa202c21da5.png)

由[在](https://unsplash.com/@bayc7739?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 无聊的冒险家贝可拍摄的照片

说到性能优化，人们通常只关注速度和 CPU 使用率。很少有人关心内存消耗，直到他们耗尽内存。尝试限制内存使用有很多原因，不仅仅是为了避免应用程序因为内存不足错误而崩溃。

在这篇文章中，我们将探索发现 Python 应用程序中哪些部分消耗了过多内存的技术，分析其原因，并最终使用简单的技巧和内存高效的数据结构来减少内存消耗和内存占用。

# 为什么要这么麻烦呢？

但是首先，你为什么要费心保存内存呢？除了避免前面提到的内存不足错误/崩溃之外，还有其他节省内存的理由吗？

一个简单的原因就是钱。资源 CPU 和 RAM——都是要花钱的，如果有减少内存占用的方法，为什么还要通过运行低效的应用程序来浪费内存呢？

另一个原因是*“数据有质量”*的概念，如果有大量数据，那么它会慢慢移动。如果数据必须存储在磁盘上，而不是 RAM 或快速缓存中，那么加载和处理将需要一段时间，从而影响整体性能。因此，优化内存使用可能会有一个很好的副作用，那就是加快应用程序的运行时间。

最后，在某些情况下，可以通过添加更多的内存来提高性能(如果应用程序的性能受内存限制)，但是如果机器上没有任何内存，就不能这样做。

# 找到瓶颈

很明显，我们有充分的理由减少 Python 应用程序的内存使用，但是在我们这样做之前，我们首先需要找到占用所有内存的瓶颈或代码部分。

我们将介绍的第一个工具是`memory_profiler`。该工具逐行测量特定函数的内存使用情况:

为了开始使用它，我们将它与`pip`和`psutil`包一起安装，这显著提高了 profiler 的性能。除此之外，我们还需要用`@profile`装饰器标记我们想要进行基准测试的函数。最后，我们使用`python -m memory_profiler`对我们的代码运行分析器。这逐行显示了修饰函数的内存使用/分配情况——在本例中是`memory_intensive`——它有意创建和删除大型列表。

既然我们知道了如何缩小我们的关注范围，并找到增加内存消耗的特定行，我们可能想要更深入地挖掘一下，看看每个变量使用了多少。你可能以前见过用来测量这个的`sys.getsizeof`。然而，对于某些类型的数据结构，这个函数会给你一些有问题的信息。对于整数或 bytearrays，您将获得以字节为单位的实际大小，但是对于 list 这样的容器，您将仅获得容器本身的大小，而不是其内容的大小:

我们可以看到，对于普通整数，每当我们越过一个阈值时，大小就会增加 4 个字节。类似地，对于普通字符串，每次我们添加另一个字符，就会增加一个额外的字节。然而对于列表来说，这并不成立——在本例中，`sys.getsizeof`不*遍历数据结构，只返回父对象的大小。*

*更好的方法是使用专门设计的工具来分析记忆行为。一个这样的工具是，它可以帮助您获得关于 Python 对象大小的更现实的想法:*

*Pympler 为`asizeof`模块提供了相同名称的函数，它可以正确地报告列表的大小以及它包含的所有值。此外，这个模块还具有`asized`功能，可以给我们进一步的尺寸细分对象的单个组件。*

*Pympler 有更多的特性，包括[跟踪类实例](https://pympler.readthedocs.io/en/latest/classtracker.html#classtracker)或[识别内存泄漏](https://pympler.readthedocs.io/en/latest/muppy.html#muppy)。如果您的应用程序可能需要这些东西，那么我建议查看[文档](https://pympler.readthedocs.io/en/latest/tutorials/tutorials.html)中的教程。*

# *节省一些内存*

*既然我们知道如何寻找各种潜在的内存问题，我们需要找到一种方法来修复它们。潜在的、最快和最容易的解决方案可能是切换到更节省内存的数据结构。*

*在存储值数组时，Python `lists`是更需要内存的选项之一:*

*上面的简单函数(`allocate`)使用指定的`size`创建一个数字的 Python `list`。为了测量它占用了多少内存，我们可以使用前面显示的`memory_profiler`,它给出了函数执行过程中 0.2 秒间隔内使用的内存量。我们可以看到，生成 1000 万个数字的`list`需要 350MiB 以上的内存。对一堆数字来说，这似乎太多了。我们能做得更好吗？*

*在这个例子中，我们使用 Python 的`array`模块，它可以存储原语，比如整数或字符。我们可以看到，在这种情况下，内存使用的峰值刚刚超过 100MiB。与`list`相比，这是一个巨大的差异。通过选择适当的精度，可以进一步减少内存使用:*

*使用`array`作为数据容器的一个主要缺点是它不支持那么多类型。*

*如果您计划对数据执行大量的数学运算，那么您最好使用 NumPy 数组:*

*我们可以看到，NumPy 数组在内存使用方面表现也很好，峰值数组大小约为 123MiB。这比`array`多一点，但是使用 NumPy，您可以利用快速数学函数以及`array`不支持的类型，比如复数。*

*上述优化有助于值数组的整体大小，但我们也可以对 Python 类定义的单个对象的大小进行一些改进。这可以通过使用`__slots__` class 属性来完成，该属性用于显式声明类属性。在一个类上声明`__slots__`还有一个很好的副作用，就是拒绝创建`__dict__`和`__weakref__`属性:*

*这里我们可以看到`Smaller`类实例实际上要小得多。没有`__dict__`会从每个实例中删除整整 104 个字节，这在实例化数百万个值时可以节省大量内存。*

*上面的提示和技巧应该有助于处理数值和`class`对象。但是，字符串呢？你应该如何储存它们通常取决于你打算用它们做什么。如果你要搜索大量的字符串值，那么——正如我们已经看到的——使用`list`是非常糟糕的主意。如果执行速度很重要的话，`set`可能会更合适一些，但是可能会消耗更多的内存。最好的选择可能是使用优化的数据结构，如 *trie* ，尤其是用于查询等静态数据集。Python 中常见的是，已经有一个这样的库，以及许多其他类似树的数据结构，其中一些你可以在 https://github.com/pytries 找到。*

# *根本不使用内存*

*节省 RAM 的最简单方法是首先不要使用它。显然，您无法完全避免使用 RAM，但是您可以避免一次加载全部数据集，而是尽可能地增量处理数据。实现这一点最简单的方法是使用返回一个[惰性](https://en.wikipedia.org/wiki/Lazy_evaluation#Python)迭代器的生成器，它按需计算元素，而不是一次全部计算。*

*您可以利用的更强大的工具是*内存映射*文件，它允许我们只从一个文件中加载部分数据。Python 的标准库为此提供了`mmap`模块，可用于创建内存映射文件，其行为类似于文件和字节数组。您可以将它们用于文件操作，如`read`、`seek`或`write`以及字符串操作:*

*加载/读取内存映射文件非常简单。我们首先像往常一样打开文件进行阅读。然后我们使用文件的文件描述符(`file.fileno()`)来创建内存映射文件。从那里，我们可以通过文件操作(如`read`或字符串操作(如*切片*)来访问它的数据。*

*大多数情况下，您可能更有兴趣阅读如上所示的文件，但也有可能写入内存映射文件:*

*您将注意到的代码中的第一个不同之处是访问模式变成了`r+`，它表示读和写。为了展示我们确实可以执行读写操作，我们首先从文件中读取，然后使用 RegEx 搜索所有以大写字母开头的单词。之后，我们演示从文件中删除数据。这不像阅读和搜索那样简单，因为当我们删除一些内容时，我们需要调整文件的大小。为此，我们使用`mmap`模块的`move(dest, src, count)`方法，该方法将数据的`size - end`字节从索引`end`复制到索引`start`，在这种情况下，这意味着删除前 10 个字节。*

*如果您在 NumPy 中进行计算，那么您可能更喜欢它的`memmap`特性 [(docs)](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) ，它适合存储在二进制文件中的 NumPy 数组。*

# *结束语*

*优化应用程序通常是个难题。它还严重依赖于手头的任务以及数据本身的类型。在本文中，我们研究了查找内存使用问题的常用方法以及修复这些问题的一些选项。然而，还有许多其他方法可以减少应用程序的内存占用。这包括通过使用概率数据结构，如 [bloom filters](https://en.wikipedia.org/wiki/Bloom_filter) 或 [HyperLogLog](https://en.wikipedia.org/wiki/HyperLogLog) 来换取存储空间的准确性。另一种选择是使用树状数据结构，如 [DAWG](https://github.com/pytries/DAWG) 或 [Marissa trie](https://github.com/pytries/marisa-trie) ，它们在存储字符串数据方面非常有效。*

**本文原帖*[*martinheinz . dev*](https://martinheinz.dev/blog/68?utm_source=medium&utm_medium=referral&utm_campaign=blog_post_68)*

*[](/profiling-and-analyzing-performance-of-python-programs-3bf3b41acd16) [## 剖析和分析 Python 程序的性能

### 快速找到 Python 程序中的所有瓶颈并修复它们的工具和技术

towardsdatascience.com](/profiling-and-analyzing-performance-of-python-programs-3bf3b41acd16) [](/exploring-google-analytics-realtime-data-with-python-8625849c7d7a) [## 用 Python 探索 Google Analytics 实时数据

### 使用 REST API 和 Python 充分利用所有 Google Analytics 特性和数据

towardsdatascience.com](/exploring-google-analytics-realtime-data-with-python-8625849c7d7a) [](/all-the-ways-to-compress-and-archive-files-in-python-e8076ccedb4b) [## Python 中压缩和归档文件的所有方法

### 用 Python 压缩、解压缩和管理你可能需要的所有格式的档案和文件

towardsdatascience.com](/all-the-ways-to-compress-and-archive-files-in-python-e8076ccedb4b)*