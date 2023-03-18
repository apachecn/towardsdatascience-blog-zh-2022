# NumPy 内部:简介

> 原文：<https://towardsdatascience.com/numpy-internals-an-introduction-bcaafa1a68a2>

## 被子下的世界

![](img/b2780b1f1fa4c5fd6a4d1e90cb67f105.png)

图片由 [Elias](https://pixabay.com/users/schäferle-3372715/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=2043464) 来自 [Pixabay](https://pixabay.com//?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=2043464)

# 介绍

这是一篇关于 [NumPy](https://numpy.org) 如何在内部处理数组的短文。它被认为是一个高级主题，对于许多 NumPy 临时用户来说，并不严格要求深入理解。然而，我认为了解基本的 NumPy 概念是有用的，不管你是直接还是间接使用 NumPy，例如通过 [pandas](https://pandas.pydata.org) 。这不仅是为了满足个人好奇心，更是为了在数组填满内存时微调性能。在这种情况下，创建另一个副本，即使是短暂的，也是一个问题。这篇文章并不深入，但希望它能激起好奇读者的兴趣。NumPy [文档](https://numpy.org/doc/stable/dev/internals.html)和特拉维斯·奥列芬特的书[NumPy 指南](http://web.mit.edu/dvp/Public/numpybook.pdf)提供了更多的信息。对于我的需求，也许还有典型的数据分析师的需求，这篇文章可以为日常任务提供足够的基础。

NumPy 中的数组本质上有两个部分:包含实际(原始)数据的数据缓冲区，以及关于如何访问和使用原始数组数据的信息，这些信息可以被认为是数组元数据。在像 C 或 Fortran 这样的语言中，原始数据就是全部:一个包含固定大小数据项的[连续](https://numpy.org/doc/stable/glossary.html#term-contiguous)(固定)内存块。NumPy 的灵活性很大程度上是因为数组元数据。数据缓冲区可以用许多不同的方式解释，而无需重新创建。对于小数组来说，这并不重要。对于较大的数据量，重复复制数组缓冲区可能会成为瓶颈。例如，当对 NumPy 数组进行切片时，元数据会发生变化，但数据缓冲区保持不变，即数组[基址](https://numpy.org/doc/stable/glossary.html#term-.base)指向获取切片的原始数组。本文将列举一组例子，这些例子帮助我理解底层机制，并在使用 NumPy 和 pandas 时变得更加自信，但并不声称我已经完全掌握了所有细节。这样做需要研究源代码，并为不同的目标读者写一篇文章。我的好奇心是通过查看 NumPy 中的入口点(即数组创建)等 NumPy 文档字符串触发的

![](img/24d8080e26f3a8879459d27fc76fd14c.png)

图 1:在 NumPy 中创建数组时指定内存布局(图片由作者创建)

`order`参数的作用是什么，我们为什么需要它？不可否认，很少有人需要指定`order`参数，但同时我也看到 NumPy 抱怨这个形状与非连续数组不兼容。当我看到这个的时候，我的第一个问题是“一个数组怎么可能是不连续的呢？”紧接着是“我如何测试一个数组是否是不连续的？”。我不是唯一有同样担心的人。如果你也对 NumPy array 的内部组织感到好奇，请继续阅读！

# NumPy 数组是如何处理的？

在我们准备好试验不同的内存布局、步长和数组变换之前，让我们先建立一个效用函数

该函数使用了最常用的 f 字符串来简洁地打印一些数组属性，这些属性在以后会很有帮助。该代码还定义了一个 4x3 整数数组(int32 ),并使用实用函数打印属性

我们来看看这些都是什么意思。实用函数使用`[np.array_str](https://numpy.org/doc/stable/reference/generated/numpy.array_str.html)`函数创建了一个简洁的数组字符串表示，该函数被进一步修改为用逗号替换新行，考虑到 f 字符串不能包含反斜杠的约束，因此在括号中不能包含`\n`。第二条信息来自 array [flags](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html?highlight=ndarray%20flags#numpy.ndarray.flags) 对象，它提供了关于数组内存布局的信息。我们只保留了它的两个属性，这两个属性显示数据缓冲区是在单个 C 风格(或基于行)还是 Fortran 风格(基于列)的连续段中。下一条信息是包含多达八个条目的整个`[np.ndarray.__array_interface__](https://numpy.org/doc/stable/reference/arrays.interface.html)`字典。我们不会一一介绍。本文的重要关键词是:

*   `shape`每个维度中数组大小的元组，也可以通过`[np.ndarray.shape](https://numpy.org/doc/stable/reference/generated/numpy.shape.html)`获得
*   `typestr`具有基本类型(例如，I 代表整数)和数据的字节顺序的字符串，后跟提供该类型使用的字节数的整数；除了 byteorder 之外，该类型也可以通过`[np.ndarray.dtype](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dtype.html)`获得，它也是由实用函数返回的
*   `data`元组的第一个条目是指向数据缓冲区第一个元素的指针
*   `strides`为 None，表示 C 风格的连续数组，或者为 tuple of strides，提供跳转到相应维中下一个数组元素所需的字节数；效用函数还提供了总是被填充的`np.ndarray.strides`。

如果内存来自其他对象，实用函数也返回 [base](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.base.html) 对象，将其格式化为一行，就像数组本身一样。它还返回基本的内存布局。有了这些信息，我们可以进行基本的自省，以了解在引擎盖下发生了什么。

回到例子，传递给效用函数的数组`a`的 shape (4，3)和它的 dtype 是 int32，所以每个元素消耗 4 个字节。因为起点是一个 Python 列表，所以创建的 NumPy 数组是 C 顺序的，如[文档](https://numpy.org/doc/stable/reference/generated/numpy.array.html)中所解释的。为了在同一行(第二维)中向右移动一个元素，我们需要跳过 4 个字节。为了在同一列(第一维)中向下移动一个元素，我们需要跳转 3x4 =12 字节，这与(12，4) `strides`元组一致。

让我们看看如果我们使用 F 顺序内存布局创建数组会发生什么

那会打印

`strides`元组已经改变。现在向下移动一个元素会跳过 4 个字节，因为数组是按列优先顺序存储的。另一方面，数组本身是相同的，具有相同的形状，并且可以以相同的方式索引。这解释了为什么大多数 NumPy 的普通用户可能不会查看内存布局。

## 内存布局对性能有影响吗？

考虑以下对具有 1 亿个浮点 64 元素的数组的行和列的求和

这给了

![](img/92a4ee06cfa2b164e50f8182ab80c06b.png)

图 2:内存布局的效果总结(图片由作者创建)

我们可以看到，当数组以行为主(C-顺序)存储时，行的求和速度是以列为主(F-顺序)存储时的两倍。对列求和则正好相反。有人可能会说这是一个很小的差异，我也同意，尽管它表明内存布局也不是无关紧要的。

## 重塑

要探索的第一种情况是数组整形

那会打印

首先要注意的是，两个整形操作不会产生相同的结果。这是因为`np.ndarray.reshape`的 order 参数与底层数组的内存布局无关，而是定义了如何读取元素并将其放入整形后的数组中。“c”表示**和**以行优先顺序排列元素，即最后一个轴索引变化最快。f 的作用正好相反。在两种整形操作中，整形后的阵列是一个视图。然而:

*   在 C-order shape 中，基数是原始数组(`b.base is a`返回`True`)。考虑到基数和它的 C 顺序布局，我们可以推断出，如果我们想在整形后的数组中向下移动一个元素，我们需要进行 6×4 = 24 字节的跳跃，并将一个元素向右移动 4 字节的跳跃。这与效用函数报告的步幅信息一致。
*   在 F 阶整形中，基数不是原来的数组(`b.base is a`返回`False`，指向第一个元素的指针已经改变，`np.shares_momeory(a, b)`返回`False`)。换句话说，元素在内存中被重新定位。我的理解是，这在技术上构成了复制，虽然重塑后的数组的基数不是没有(如果你的理解不同，请评论！).知道了基数和它的 F 阶存储器布局，再一次很容易推断步幅。为了从整形后的数组中的元素 1 到 4，我们需要跳 4 个字节，并且从元素 1 到 7 跳 2×4 = 8 个字节。

在前面的示例中，原始阵列具有 C 顺序内存布局。如果我们从 F 阶初始数组开始，我们会看到 C 阶整形导致一个副本，而 F 阶整形导致一个视图

那会打印

这个示例对于演示另一种判断何时整形导致复制的方法很有用。根据 NumPy 文档，设置数组形状直接导致就地 C-order 整形。如果像上次 C-order 整形一样需要制作副本，则就地整形注定会失败

这给了

```
Traceback (most recent call last):
  File "D:\myLearningBooks\2022_10_30_PythonForDataAnalytics3rdEdition\venv\lib\site-packages\IPython\core\interactiveshell.py", line 3378, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-1405-1144ef469292>", line 3, in <module>
    a.shape = (2, -1)
AttributeError: Incompatible shape for in-place modification. Use `.reshape()` to make a copy with the desired shape.
```

文档[不鼓励](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html#numpy.ndarray.shape)直接设置形状元组而支持`np.ndarray.reshape`函数，声明它可能会被弃用。文档中还提到[如果您希望在复制数据时出现错误，您应该将新的形状分配给数组的 shape 属性。在我看来，这有些不一致，我希望将来文档会简化。本文中的 utility 函数提供了一种替代的、有希望是全面的内省，它包括关于何时制作副本的信息。](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy.reshape)

## 置换

这可能听起来比重塑更容易，但有一些事情要发现。我们将从与之前相同的数组开始，使用默认的 C 顺序内存布局

那会打印

转置数组的基础是保持其 C 顺序内存布局的原始数组。要在转置数组中向下移动一个元素，即从元素 1 移动到元素 2，我们需要跳跃 4 个字节。为了在转置的数组中向右移动一个元素，即从元素 1 移动到元素 4，我们需要跳跃 3x4 = 12 个字节。这些与通过效用函数获得的步幅一致。转置数组的内存布局是 F 顺序的。本质上，这意味着转置操作只需要对数组的内部表示进行很小的改变。以列为主而不是行为主的顺序读取数据缓冲区就足够了。我发现这非常聪明，是底层 NumPy 引擎的一个很好的演示。没有复制任何内容，但数组仍被转置。

## 限幅

最后一个例子是关于切片的

那会打印

对于所有三个片段，基底保持原始的 C 顺序数组，也由指向第一个元素的存储器地址的指针指示。步幅也保持不变。当选择行时，切片数组保持 C 连续。然而，一旦我们选择了列，切片数组既不是 C 连续的也不是 F 连续的。图 3 展示了推理过程。

![](img/9ecc2072638c11bf76329b926fbe1687.png)

图 3:切片后的非连续数组(图片由作者创建)

尽管切片数组是不连续的，但不需要创建副本。我们仍然可以使用大步。为了在整形后的数组中向下移动一个元素，我们需要进行 3×4 = 12 字节的跳跃，向右移动一个元素需要 4 字节的跳跃。相当整洁。NumPy 提供了用`np.ascontiguousarray()`函数从一个不连续的创建一个连续数组的可能性

但是正如所料，这需要创建一个副本

除非你想执行重复的行或列宽操作，我不太确定这为什么有用，因为它有自己的内存和处理时间成本，但 NumPy 提供了可能性。

# 结论

NumPy 并不新鲜。尽管如此，每当我想起它，我只能对它的独创性印象深刻。在原始数据之上添加数组元数据允许非常灵活地使用数组。通过改变元数据，可以改变形状，转置或切片数组，而无需重新排列原始数据。数据缓冲器被一次又一次地使用，并且通过被不同地解释，有可能以可忽略的计算成本创建新的视图。这真是太聪明了，这篇文章并没有公平对待 NumPy 的才华。尽管如此，我仍然希望它有助于理解诸如内存布局和跨度之类的概念。除了允许用户以一种高性能的方式使用 NumPy，获得这些知识是使用更深奥的 NumPy 函数的第一步，比如用`np.stride_tricks.as_strided`创建给定形状和步幅的视图。这听起来很有趣，但需要小心，因为它可能指向无效的内存，使程序崩溃，在更不幸的情况下破坏结果。如果你想走这么远，你可以参考这篇[文章](/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20)及其参考资料。