# Python 中的视图和副本

> 原文：<https://towardsdatascience.com/views-and-copies-in-python-1e46b5af4728>

## “我学得越多，就越意识到自己有多少不知道。”

本文解释了在使用切片、花式索引和布尔索引等操作时，如何复制或引用 Python 列表、NumPy 数组和 pandas 数据框。这些操作在数据分析中非常常见，不能掉以轻心，因为错误的假设可能会导致性能损失甚至意外的结果。Python 看起来很简单，但是每次回到基础，总会有新的东西需要探索和学习。副标题中阿尔伯特·爱因斯坦的名言适用于一切，Python 也不例外。

![](img/70cfdb572def954fedb74cca0d362a94.png)

图片由来自 [Pixabay](https://pixabay.com//?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=229335) 的 [Frank](https://pixabay.com/users/karla31-72895/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=229335) 提供

# 介绍

我经常怀疑自己什么时候真正了解一门学科。完成了博士学位，并在过去做了一段时间的研究员，我可以自信地说，答案是绝对不会。我使用 Python 已经有一段时间了，它是一个非常棒的数据分析工具。我为现实生活中的问题提供了非常需要的解决方案，并产生了影响。尽管如此，每次我回到基础，都有新的东西要学，或者有新的角度来看待事物。当阅读一本书的介绍部分时，这种想法经常被触发，这些部分在书的本质真正开始之前被认为是容易阅读的。阿尔伯特·爱因斯坦的这句话在我的脑海中回响，不久之后，我使用 Python 解释器来尝试一些东西，想知道我到底是如何发现如此基础的东西的。我也想知道如果有这么多问题我什么时候能看完这本书，但是时间管理是另一个痛苦的故事…

这篇文章就跟随了这样一个时刻。它旨在深入解释 Python 列表、NumPy 数组和 pandas 数据框在使用切片、花式索引和布尔索引等操作时如何创建视图或副本。有一些混淆，因为像浅拷贝和深拷贝这样的术语并不总是指同一件事，同时也不清楚什么时候像 NumPy 数组元数据和 Pandas 索引这样的辅助数据被拷贝或引用。这篇文章可能不会提供所有的答案，但是希望它除了提供文档参考之外，还提供了一个在有疑问时运行快速计算实验的框架。所有示例都是用 Python v3.8.10、pandas v1.5.1 和 NumPy v1.23.4 编写的。

# Python 列表

在这一节中，我们将运行一些计算实验来理解如何创建 Python 列表的副本。如果您运行类似的实验，请记住 Python 在内存中缓存小整数和字符串，以便它可以引用预先制作的对象，而不是创建新的对象。这个所谓的 [interning](/optimization-in-python-interning-805be5e9fd3e) 是本文中使用的标准 Python 实现 CPython 的优化之一。使用不同的字符串和整数是明智的，这样可以避免在查找对象地址时产生混淆。

Python 列表是人们可能认为容易的部分。让我们创建一个包含整数、Python 列表和嵌套 Python 列表作为元素的 Python 列表。我们还创建了一个实用函数来打印各种 Python 列表元素的地址，为了简洁起见，只显示了地址的最后四位数字。

上面的代码打印出来

```
a                   : 4160 | 7728 |  9888 3376 | 3232 0848 2480
```

注意，地址在每次执行中当然是不同的。因此，我们将确保数组`a`从现在开始不会被改变。让我们尝试不同的方法来复制数组`a`，从简单的(重新)绑定到另一个变量到深度复制

那会打印

```
new binding         : 4160 | 7728 |  9888 3376 | 3232 0848 2480
shallow copy I      : 7072 | 7728 |  9888 3376 | 3232 0848 2480
shallow copy II     : 9312 | 7728 |  9888 3376 | 3232 0848 2480
shallow copy III    : 1488 | 7728 |  9888 3376 | 3232 0848 2480
shallow copy IV     : 8128 | 7728 |  9888 3376 | 3232 0848 2480
deep copy           : 0528 | 7728 |  6848 3376 | 0816 2960 2480
```

首先要观察的是，除了新绑定(第一行)之外，列表的地址(第一列有地址)在所有其他尝试中都发生了变化。这意味着进行了复制。代码提供了四种不同的方法来创建浅层副本，这意味着列表的元素是相同的对象，尽管列表本身不是。如果我们试图改变列表的浅层副本的不可变元素，原始列表将不会被修改，但是改变可变元素会改变原始列表。举个例子，

印刷品

```
a_demo (before)     -> ['d1', ['d2', 'd3']]
a_demo (after)      -> ['d1', ['**D2**', 'd3']]
a_demo_shallow_copy -> ['**D1**', ['**D2**', 'd3']]
```

这意味着在嵌套列表和使用其他可变列表元素的情况下，浅拷贝会导致副作用。在深层拷贝的情况下，我们是安全的，如下面的代码所示

那会打印

```
a_demo (before)  -> ['d1', ['d2', 'd3']]
a_demo (after)   -> ['d1', ['d2', 'd3']]
a_demo_deep_copy -> ['**D1**', ['**D2**', 'd3']]
```

以上是相当简单的概括。任何类型的 Python 列表切片，如`a[:]`、`a[1:4]`、`a[:5]`或`a[::-1]`，都会创建列表保留部分的浅层副本。但是当我们连接或相乘列表时会发生什么呢？你能预测下面的操作会发生什么吗？

上面的版画

```
a                   : 4160 | 7728 |  9888 3376 | 3232 0848 2480
b (first part)      : 5712 | 7728 |  9888 3376 | 3232 0848 2480
b (second part)     : 5712 | 7728 |  9888 3376 | 3232 0848 2480a                   : 4160 | 7728 |  9888 3376 | 3232 0848 2480
b (first part)      : 5648 | 7728 |  9888 3376 | 3232 0848 2480
b (second part)     : 5648 | 7728 |  9888 3376 | 3232 0848 2480
b (third part)      : 5648 | 7728 |  9888 3376 | 3232 0848 2480
```

这意味着我们创建了列表元素的更多引用(绑定),也就是说，这就像创建了一个浅层拷贝。这可能会导致意想不到的副作用，如下面的实验所示

那会打印

```
a_demo (before) -> ['d1', ['d2', 'd3']]
b               -> ['**D1**', ['**D2**', 'd3'], 'd1', ['**D2**', 'd3']]
a_demo (after)  -> ['d1', ['**D2**', 'd3']]
```

再说一次，请随意进行如上所述的简单计算实验。Python 是一种很好的实验语言，因为它的语法简单、简洁。

# NumPy 数组

与 Python 列表类似，NumPy 数组也可以通过视图复制或公开。为了说明该功能，我们将通过绘制 0 到 9 范围内的随机整数来创建一个数组

我们还定义了一个实用函数来显示数组内容、数组元素消耗的总字节数、数组在内存中的总大小[、一个布尔值来显示数组](https://docs.python.org/3/library/sys.html#sys.getsizeof)[是拥有它使用的内存](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html)还是从另一个对象借用的内存，如果内存来自其他对象，则显示基本对象。上面的版画

```
[[8 2 8 8 1]
 [7 4 2 8 8]
 [3 3 2 3 3]
 [0 0 7 6 8]
 [2 7 3 4 6]]datatype is int64
number of bytes is 200 bytes (25 x 8 bytes)
size is 328 bytes
owndata is True
base is None
```

该数组将其数据类型显式设置为`int64`，因此每个数组元素消耗 8 个字节。总共 25 个数组元素消耗 200 个字节，但是由于数组[元数据](https://numpy.org/doc/stable/dev/internals.html#numpy-internals)，例如数据类型、步长和其他有助于轻松操作数组的重要信息，内存大小为 328。我们可以看到数组保存自己的数据，因此它的基数是`None`。

让我们看看如果我们创建一个视图会发生什么

那会打印

```
[[8 2 8 8 1]
 [7 4 2 8 8]
 [3 3 2 3 3]
 [0 0 7 6 8]
 [2 7 3 4 6]]datatype is int64
number of bytes is 200 bytes (25 x 8 bytes)
size is 128 bytes
owndata is False
base is [[8 2 8 8 1]
 [7 4 2 8 8]
 [3 3 2 3 3]
 [0 0 7 6 8]
 [2 7 3 4 6]]
```

数组的内容保持不变。类似地，数组元素的数据类型和字节数保持不变。其余的数组属性现在不同了。大小已经减少到 128 字节(即 238–200 字节)，因为数组视图为 NumPy 数组属性消耗内存。数组元素没有被复制而是被引用。这从不再是`None`的基本属性中可以明显看出。在 NumPy 术语中，视图有相同的数据缓冲区(实际数据),但有自己的元数据。修改视图的元素也会修改原始数组。

让我们看看创建拷贝时会发生什么

那会打印

```
[[8 2 8 8 1]
 [7 4 2 8 8]
 [3 3 2 3 3]
 [0 0 7 6 8]
 [2 7 3 4 6]]datatype is int64
number of bytes is 200 bytes (25 x 8 bytes)
size is 328 bytes
owndata is True
base is None
```

输出看起来与原始数组相同。修改副本的元素不会修改原始数组。

我们可以很容易地试验各种整形、切片和索引功能，以检查是否创建了视图或副本

那会打印

```
reshape produces a view
transpose/reshape produces a view
ravel produces a view
transpose/ravel produces a copy
transpose/ravel (F-order) produces a view
flatten produces a copy
transpose/flatten produces a copy
slicing produces a view
advanced indexing produces a copy
combined indexing and slicing produces a copy
Boolean indexing produces a copy
```

对于某些功能，行为并不总是相同的。例如，`[numpy.ravel](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html)`返回一个连续的扁平数组，该数组仅在需要时才是副本。另一方面，`[numpy.ndarray.flatten](http://numpy.ndarray.flatten)`总是返回折叠成一维的数组的副本。`[numpy.reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)`的行为有点复杂，感兴趣的读者可以参考官方文档。

需要说明的是，NumPy 在原始数组中用偏移量和跨距寻址元素时，即在使用基本索引和切片时，会创建视图。这与 Python 列表的行为相矛盾！另一方面，[高级索引](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing)总是创建副本。整形操作更复杂，是否返回副本或视图取决于上下文。

使用高级索引创建的副本，以及使用`[numpy.copy](https://numpy.org/doc/stable/reference/generated/numpy.copy.html)`创建的副本不会深入复制数组中的可变元素。与 Python 列表的浅层副本一样，NumPy 数组副本包含相同的对象，如果该对象可以被修改(可变)，这可能会导致意外:

那会打印

```
Numpy shallow copy
a_np_demo (before) ->  [1 2 list([3, 4])]
b                  ->  [**-1** 2 list([**-3**, 4])]
a_np_demo (after)  ->  [1 2 list([**-3**, 4])]Python deep copy
a_np_demo (before) ->  [1 2 list([3, 4])]
b2                 ->  [**-1** 2 list([**-3**, 4])]
a_np_demo (after)  ->  [1 2 list([3, 4])]
```

这可能是理论上的兴趣，因为 NumPy 数组通常不用于存储可变对象。不过，很高兴知道`copy.deepcopy()`起作用了。

# 熊猫数据框

现在让我们把注意力转向熊猫数据框。按照通常的方式，我们将定义一个熊猫数据框和一个效用函数来显示它的细节

该数据与早期的 NumPy 数组具有相同的数据结构，即它具有 5x5 int64 个元素，但是我们另外定义了索引和列名。效用函数已被修改。pandas 数据帧的不同列可以有不同的数据类型，因此我们用`a_df.dtypes.unique()`返回唯一的数据类型。为了查看底层数据何时被复制或引用，我们首先使用`a_df.to_numpy()`获取底层 NumPy 数组，然后使用[数组接口](https://numpy.org/doc/stable/reference/arrays.interface.html)获取指向数据第一个元素的指针。上面的版画

```
dataframe is
     c0  c1  c2  c3  c4
r0   5   2   8   6   6
r1   1   9   1   1   1
r2   0   7   6   3   7
r3   7   4   9   5   2
r4   5   8   3   7   1datatypes are [dtype('int64')]
number of bytes is 200 bytes (25 x 8 bytes)
size is 511 bytes
pointer to data area 2893487649296
```

我们现在有足够的设备来试验拷贝和视图。

查看 pandas API 参考，我们可以找到一个数据帧[复制](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.copy.html)函数，它接受一个`deep`布尔参数:当 True(默认)时，一个新对象被创建，带有调用对象的数据和索引的副本(这不是标准库的`copy.deepcopy()`意义上的深度副本；见下文！).我们可以修改数据和索引，而原始数据框将保持不变。如果为 False，将创建一个新对象，而不复制调用对象的数据或索引，也就是说，我们只创建对数据和索引的引用。这意味着对原始数据框数据的任何更改都将反映在副本中。

让我们用一种观点做实验

那会打印

```
dataframe is
     c0  c1  c2  c3  c4
r0   5   2   8   6   6
r1   1   9   1   1   1
r2   0   7   6   3   7
r3   7   4   9   5   2
r4   5   8   3   7   1datatypes are [dtype('int64')]
number of bytes is 200 bytes (25 x 8 bytes)
size is 511 bytes
pointer to data area 2893487649296
Same base: True
Same row index: True
Same column index: True
```

我们可以看到数据区指向同一个内存地址，NumPy 数组基是同一个对象，两个索引也是同一个对象。

让我们创建一个副本(`deep=True`是默认的，但为了清楚起见，我们包括它)

那会打印

```
dataframe is
     c0  c1  c2  c3  c4
r0   5   2   8   6   6
r1   1   9   1   1   1
r2   0   7   6   3   7
r3   7   4   9   5   2
r4   5   8   3   7   1datatypes are [dtype('int64')]
number of bytes is 200 bytes (25 x 8 bytes)
size is 511 bytes
pointer to data area 2893487655536
Same base: False
Same row index: False
Same column index: False
```

我们可以看到，副本与原始数据帧具有不同的基础，这也反映在不同的数据区指针中。我们还为这两个索引创建了新的对象。同样，类似于 NumPy 数组的情况，如果数据帧包含可变元素，那么改变副本的这些可变对象会修改原始数据帧，如下面的数值实验所示

那会打印

```
a_df_demo (before) ->  c1 c2 0 1 3 1 2 {'key1': '✓', 'key2': '✓'}
b                  ->  c1 c2 0 1 3 1 2 {'key1': '✓'}
a_df_demo (after)  ->  c1 c2 0 1 3 1 2 {'key1': '✓'}
```

对于熊猫数据框来说，这并不是一个非常常见的用例，但是记住这一点还是很有用的。对熊猫来说不幸的是，似乎不可能通过使用 Python 的`copy.deepcopy()`函数从标准库中获得真正的深度副本，因为熊猫开发者已经将[实现为](https://github.com/pandas-dev/pandas/issues/17406) `pd.DataFrame.__deepcopy__()`为`pd.DataFrame.copy(deep=True)`。不确定这在将来是否会改变，但无论如何它都被认为是一个反模式。熊猫在这方面与 NumPy 不同。

我们现在可以看看用熊猫选择行和列的各种方法

那会打印

```
select one column uses the same base
select one column using [] does not use the same base
select one column with loc uses the same base
select columns with loc and slicing uses the same base
select columns with loc and fancy indexing does not use the same base
select rows using loc and a Boolean mask does not use the same base
select rows with loc and slicing uses the same base
chained indexing uses the same base
```

基本的索引和切片，比如使用(单个)方括号或`.loc[]`访问器的简单列索引使用相同的基，而所有其他操作不使用。当有疑问时，上述计算实验框架可以给出快速答案。不幸的是，检查基数是否保持不变并不总是足以预测使用链式索引时会发生什么(见下文)，但它提供了一个起点。在最后一次尝试中，基数保持不变，但是如果我们使用这种链式索引来设置值，原始数据帧仍然保持不变。然而，反过来似乎是正确的:如果基数改变了，那么我们就在拷贝上操作。我将欢迎对此的评论，因为我开始如履薄冰。请继续阅读。

现在让我们转向与熊猫有关的最后一个话题，著名的链式索引和相关的`SettingWithCopyWarning`。使用先前定义的`a_df`数据框架，我们可以通过使用布尔索引来尝试改变列中某些元素的值。假设我们使用链式索引，有两种方法可以想到

那会打印

```
attempt 1
    c0  c1  c2  c3  c4
r0   5   2   8   6   6
r1   1   9   1   1   1
r2   0   7   6   3   7
r3   7   4   9   5   2
r4   5   8   3   7   1attempt 2
    c0  c1  c2  c3  c4
r0   5   2   8   6   6
r1   1   9   1  -1   1
r2   0   7   6  -1   7
r3   7   4   9   5   2
r4   5   8   3  -1   1<ipython-input-789-06440868e65b>:5: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead
See the caveats in the documentation: [https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy)
  a_df_demo.loc[msk]['c3'] = -1
```

[评估顺序事项](https://pandas.pydata.org/docs/user_guide/indexing.html#evaluation-order-matters)。第一次尝试给出了一个`SettingWithCopyWarning`，这并不奇怪。使用带有布尔掩码的`.loc[]`访问器创建一个副本。为副本的元素分配新值不会更改原始数据框。这是预期的行为，但是 pandas 更进一步，通知用户。NumPy 就没那么善良了。但是即使是熊猫，也不要总是依赖警告，因为它可能不会发布。例如

不给出任何警告，数据框也不会像上面打印的那样被修改

```
 c0  c1  c2  c3  c4
r0   5   2   8   6   6
r1   1   9   1   1   1
r2   0   7   6   3   7
r3   7   4   9   5   2
r4   5   8   3   7   1
```

这些我们都要记住吗？答案是否定的，不仅因为事实上不可能列举所有不同的链式索引的可能性，还因为当涉及到发出`SettingWithCopyWarning`时，不能确定不同的 pandas 版本中的行为是否保持相同。更糟糕的是，数据框可能会随着一个熊猫版本而改变，而不是另一个版本(我个人对此没有证据，但这是我的担心)。使用一个虚拟环境并建立一个[需求](https://learnpython.com/blog/python-requirements-file/)文件不仅可以防止依赖性地狱，还可以防止这样的问题，尽管最佳实践是知道哪些熊猫任务是有风险的并避免它们。当数据帧具有[层次索引](https://pandas.pydata.org/docs/user_guide/indexing.html#returning-a-view-versus-a-copy)和不同的数据类型时，情况甚至更加复杂。预测链式索引的结果是不安全的。

避免链式索引的正确方法是使用单个访问器来设置值，例如

打印修改后的数据帧，并且不发出警告。

## 这一切都是关于使用单个访问器吗？

使用单个访问器和避免使用赋值的链式索引绝对是可靠的建议，但是还有更多令人惊讶的地方。让我们创建一个数据帧的切片，并检查原始数据帧被修改后会发生什么情况。

第一次尝试做了三个这样的实验

实验的不同之处仅在于原始数据帧的修改是如何发生的。第一个实验只修改了列`a`中的一个元素，第二个实验使用`df.loc[0:,'a']`修改了整个列`a`，第三个实验也修改了整个列`a`，但是如果我们看看我们得到的结果，这次使用的是`df.loc[:,'a'].`

```
experiment 1
data buffer pointer (before) -> 2893435341184
data types (before) -> [dtype('int32')]
my_slice (before) -> [**1, 2, 3, 1, 2, 3]**
data buffer pointer (after)  -> 2893435341184
data types (after)  -> [dtype('int32')]
my_slice (after)  -> **[-10, 2, 3, 1, 2, 3]**experiment 2
data buffer pointer (before) -> 2893490708496
data types (before) -> [dtype('int32')]
my_slice (before) -> **[1, 2, 3, 1, 2, 3]**
data buffer pointer (after)  -> 2893490708496
data types (after)  -> [dtype('int32')]
my_slice (after)  -> **[-10, -10, -10, 1, 2, 3]**experiment 3
data buffer pointer (before) -> 2893435341184
data types (before) -> [dtype('int32')]
my_slice (before) -> **[1, 2, 3, 1, 2, 3]**
data buffer pointer (after)  -> 2893491528672
data types (after)  -> [dtype('int64'), dtype('int32')]
my_slice (after)  -> **[1, 2, 3, 1, 2, 3]**
```

该切片以粗体显示，可以很容易地看到发生了什么。在成功修改数据帧之后，切片在前两个实验中被修改，而在第三个实验中没有被修改。如果您仔细观察，就会发现数据框的一个列类型被更改为`int64`，并且它的数据缓冲区在内存中被重新定位。我假设这是因为我们改变了整个列`a`的值，使其成为`int64`。如果我们在创建数据框时显式设置数据类型，这似乎可以得到证实

那会打印

```
experiment 1
data buffer pointer (before) -> 2893491528672
data types (before) -> [dtype('int64')]
my_slice (before) -> [**1, 2, 3, 1, 2, 3]**
data buffer pointer (after)  -> 2893491528672
data types (after)  -> [dtype('int64')]
my_slice (after)  -> **[-10, 2, 3, 1, 2, 3]**experiment 2
data buffer pointer (before) -> 2893486517968
data types (before) -> [dtype('int64')]
my_slice (before) -> **[1, 2, 3, 1, 2, 3]**
data buffer pointer (after)  -> 2893486517968
data types (after)  -> [dtype('int64')]
my_slice (after)  -> **[-10, -10, -10, 1, 2, 3]**experiment 3
data buffer pointer (before) -> 2893491528672
data types (before) -> [dtype('int64')]
my_slice (before) -> **[1, 2, 3, 1, 2, 3]**
data buffer pointer (after)  -> 2893491528672
data types (after)  -> [dtype('int64')]
my_slice (after)  -> **[-10, -10, -10, 1, 2, 3]**
```

我想留下一个挥之不去的片段是没有意义的，除非它是一个带有`df.loc[1:3].copy()`的显式副本。否则，人们总是可以在需要时对数据帧进行切片，并且总是有新的数据。但是，这仍然是一个有效的计算实验，可以学习更多关于视图和副本的知识。

# 结论

理解 Python 何时创建副本和视图需要一些实践。Python 列表、NumPy 数组和 pandas 数据框提供了创建副本和视图的功能，如下表所示([作为 GitHub gist 为中等作者创建](https://levelup.gitconnected.com/3-tips-to-sharing-beautiful-tables-on-medium-post-25dab18670e))

然而，最重要的带回家的信息与使用 NumPy 数组和 Pandas 数据帧时基于索引的赋值行为有关:

*   NumPy 链式索引通常很容易理解:基本索引产生视图，而高级索引返回副本，防止在赋值时修改原始数组；当使用整形操作时，行为更加复杂
*   应该避免熊猫链式索引，而应该为所有赋值使用单个访问器；即使人们认为链式索引的行为是可以预测的，这也是事实

理解视图和副本至关重要，尤其是在处理大型数组和数据框时。希望这篇文章能为进一步阅读提供一个起点。我确信我错过了一些方面，很可能我误解了一些细微的差别。在你的评论中指出这些将再次证明阿尔伯特·爱因斯坦的引用是多么正确！

## 推荐进一步阅读

*   优秀的[文章](https://realpython.com/pandas-settingwithcopywarning)讲述了大熊猫无处不在的复制警告
*   关于[副本和视图的官方数字文档](https://numpy.org/devdocs/user/basics.copies.html#copies-and-views)
*   关于[副本和视图](https://pandas.pydata.org/docs/user_guide/indexing.html#returning-a-view-versus-a-copy)的官方熊猫文档