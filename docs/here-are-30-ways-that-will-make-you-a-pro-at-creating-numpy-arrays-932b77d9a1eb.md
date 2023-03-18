# 这里有 30 种方法可以让你成为创建 NumPy 数组的专家

> 原文：<https://towardsdatascience.com/here-are-30-ways-that-will-make-you-a-pro-at-creating-numpy-arrays-932b77d9a1eb>

## 创建 NumPy 数组的综合指南

![](img/eb257dbe86de276693feebc4a3dad094.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上 [Vardan Papikyan](https://unsplash.com/@varpap?utm_source=medium&utm_medium=referral) 拍摄的照片

Python 中的 [NumPy](https://numpy.org/doc/1.23/index.html) 库构成了当今几乎所有数据科学和机器学习项目的基础构建模块。

由于它在支持矢量化运算和高效处理大量数值数据方面的巨大潜力，它已经成为 Python 中最重要的库之一。

此外，整个数据驱动的生态系统在某种程度上依赖于 NumPy 及其核心功能。

Numpy 库的核心是它的数组对象，也称为 [**ndarray**](https://numpy.org/doc/1.23/reference/arrays.ndarray.html) 。这些用于在 Python 中执行大量的数据操作，如逻辑、算术、数学、统计操作等。

然而，这些操作只有在您首先创建 NumPy 数组时才有可能。因此，这篇文章旨在展示 30 种创建 NumPy 数组的不同方法。

文章的亮点如下:

[**#1 — #6 从 Python List/Tuple 中创建一个 NumPy 数组**](#ab7b)[**# 7—# 10 创建一个特定数值范围的 NumPy 数组**](#cfb8)[**# 11—# 16 创建一个特定值的 NumPy 数组**](#0b54)[**【17—# 17**](#3c3e)[**【26—# 30】其他流行方法**](#bf2b)[**结论**](#a5c4)

我们开始吧🚀！

# 导入依赖项

# #1 — #6 从 Python 列表/元组创建 NumPy 数组

## 方法 1:来自 Python 列表

要从给定的 Python 列表创建 NumPy 数组，使用`[np.array()](https://numpy.org/doc/stable/reference/generated/numpy.array.html)`方法:

我们可以分别使用`type()`方法和`dtype`属性来验证所创建对象的类型和数据类型，如下所示:

## 方法 2:来自 Python 元组

输入不一定必须是 Python 列表。您还可以使用`[np.array()](https://numpy.org/doc/stable/reference/generated/numpy.array.html)`方法从 Python 元组创建 NumPy 数组:

## 方法 3:来自具有特定数据类型的 Python 列表

要创建具有特定数据类型的 NumPy 数组，请将`dtype`参数传递给`[np.array()](https://numpy.org/doc/stable/reference/generated/numpy.array.html)`方法，如下所示:

## 方法 4:来自 Python 列表列表

为了从一系列列表中创建一个二维 NumPy 数组，您可以再次使用上面的`[np.array()](https://numpy.org/doc/stable/reference/generated/numpy.array.html)`方法。

## 方法 5:三维数字阵列

按照我们在**(方法 4)** 中所做的，您还可以扩展到更高维的 NumPy 数组。例如，下面的代码块演示了如何从列表的列表中创建一个三维 NumPy 数组。

## 方法 6:从字符串列表

您还可以使用`[np.array()](https://numpy.org/doc/stable/reference/generated/numpy.array.html)`方法从字符串列表中创建一个 NumPy 数组:

# #7 — #10 创建特定数值范围的 NumPy 数组

## 方法 7:定期增加数字

要创建一个值有规律递增的数组，使用`[np.arange()](https://numpy.org/doc/stable/reference/generated/numpy.arange.html)`方法。

可以用不同数量的位置参数调用`[np.arange()](https://numpy.org/doc/stable/reference/generated/numpy.arange.html)`:

*   `np.arange(stop)`:用`step=1`生成`[0, stop)`范围内的值。
*   `np.arange(start, stop)`:用`step=1`生成`[start, stop)`范围内的值。
*   `np.arange(start, stop, step)`:生成`[start, stop)`范围内的数值，数值间距由`step`给定。

## 方法 8:等距数字

如果您想要创建一个 NumPy 数组，其中包含特定数量的元素，这些元素在给定的值范围内等距分布，请使用`[np.linspace()](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)`方法:

## 需要注意的事项:

*   您应该总是在`[np.linspace()](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)`方法中指定`start`和`stop`参数。
*   `num`参数的默认值为`50`。
*   如果不想包含`stop`值，请指定`endpoint=False`。

## 方法 9:数字形成几何级数(GP)

上面讨论的`[np.linspace()](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)`方法在线性空间上生成数字。因此，它返回的数字构成了一个算术级数(AP)。如果您想生成形成几何级数(GP)的数字，请使用`[np.geomspace()](https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html)`方法。

## 方法 10:在日志空间中等距分布

最后，如果您想要生成在对数空间中等距的数字，请使用`[np.logspace()](https://numpy.org/doc/stable/reference/generated/numpy.logspace.html)`方法:

这相当于获取`[np.linspace()](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)`的输出，并将单个数字提升到`base`的幂。

# #11 — #16 创建特定值的 NumPy 数组

## 方法 11:标识数组

要创建单位数矩阵，使用`[np.identity()](https://numpy.org/doc/stable/reference/generated/numpy.identity.html)`方法:

它返回一个`n x n`数组，主对角线上的所有元素都设置为`1`，所有其他元素都设置为`0`。

## 方法 12:主对角线值为一的数组

要创建一个对角线上为 1，其他地方为 0 的二维 NumPy 数组(任何形状)，请使用`[np.eye()](https://numpy.org/doc/stable/reference/generated/numpy.eye.html)`方法。

## 需要注意的事项:

*   `[np.eye()](https://numpy.org/doc/stable/reference/generated/numpy.eye.html)`方法不同于`[np.identity()](https://numpy.org/doc/stable/reference/generated/numpy.identity.html)`方法。
*   `[np.eye()](https://numpy.org/doc/stable/reference/generated/numpy.eye.html)`是一个通用的方法，可以生成任意形状的 NumPy 数组，而`[np.identity()](https://numpy.org/doc/stable/reference/generated/numpy.identity.html)`总是返回一个正方形的 NumPy 数组。

## 方法 13:全 1 数组

要创建一个填充 1 的 NumPy 数组，使用如下所示的`[np.ones()](https://numpy.org/doc/stable/reference/generated/numpy.ones.html)`方法:

上面的演示创建了一个二维 NumPy 数组。但是，如果您希望创建 1 的一维 NumPy 数组，请将数组的长度传递给`[np.ones()](https://numpy.org/doc/stable/reference/generated/numpy.ones.html)`方法，如下所示:

## 方法 14:全零数组

类似于上面的方法，如果你想创建一个 NumPy 零数组，使用`[np.zeros()](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html)`方法。下面，我演示了一个形状为`(2, 2, 4)`的三维 NumPy 数组的创建:

## 方法 15:用特定值填充的数组

要创建一个用 0 和 1 以外的值填充的 NumPy 数组，请使用`[np.full()](https://numpy.org/doc/stable/reference/generated/numpy.full.html)`方法。

第一个参数是数组的形状(本例中为`(2, 4)`)，第二个参数是填充值(`5`)。

## 方法 16:空数组

最后，如果您想创建一个给定形状和类型的 NumPy 数组而不初始化条目，使用`[np.empty()](https://numpy.org/doc/stable/reference/generated/numpy.empty.html)`方法:

在这种方法中，Numpy 抓取一块内存并返回存储在那里的值，而不对它们做任何事情——返回看起来随机的值。

# #17 — #20 创建特定形状的 NumPy 数组

在本节中，我们将讨论创建一个形状与另一个 NumPy 数组相似的 NumPy 的方法。

## 方法 17:1 和另一个数组形状的数组

如果你想创建一个与另一个 NumPy 数组形状相似的 NumPy 数组，使用`[np.ones_like()](https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html)`方法:

## 方法 18:带有零和另一个数组形状的数组

与上面的方法类似，如果您想要创建一个具有另一个 NumPy 数组形状的 NumPy 零数组，请使用`[np.zeros_like()](https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html)`方法

## 方法 19:空数组和另一个数组的形状

要创建一个采用另一个 NumPy 数组形状的空 NumPy 数组，使用`[np.empty_like()](https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html)`方法:

## 方法 20:用特定值和另一个数组的形状填充的数组

最后，如果您想要创建一个具有特定填充值的 NumPy 数组，它与另一个 NumPy 数组具有相同的形状，那么使用`[np.full_like()](https://numpy.org/doc/stable/reference/generated/numpy.full_like.html)`方法。

# #21 — #25 创建随机数字数组

在这一节中，我将演示用随机值生成 NumPy 数组的方法。

## 方法 21:随机整数数组

要生成整数的随机 NumPy 数组，使用`[np.random.randint()](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html)`方法:

## 方法 22:浮点值的随机数组

要生成浮点值的随机 NumPy 数组，请使用`[np.random.random()](https://numpy.org/doc/stable/reference/random/generated/numpy.random.random.html)`方法。

## 方法 23:来自均匀分布的随机阵列

要从`[0, 1)`上的均匀分布生成随机 NumPy 数组，使用`[np.random.rand()](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html)`方法。

## 方法 24:来自正态分布的随机阵列

要使用`µ=0`和`σ = 1`从正态分布中生成随机 NumPy 数组，请使用`[np.random.randn()](https://numpy.org/doc/stable/reference/random/generated/numpy.random.RandomState.randn.html)`方法。

## 方法 25:正态分布的随机排列和特定的平均值和标准偏差

相反，如果您想从`*X* ~ *N(µ, σ^2)*`的正态分布中生成一个 NumPy 数组，请执行以下操作:

# #26— #30 其他流行方法

## 方法 26:来自熊猫系列

如果您想将 Pandas 系列转换为 numpy 数组，您可以使用`[np.array()](https://numpy.org/doc/1.23/reference/generated/numpy.array.html#numpy.array)`来完成:

## 方法 27:来自熊猫数据框架

您还可以使用`[np.array()](https://numpy.org/doc/stable/reference/generated/numpy.array.html)`方法将 Pandas 数据帧转换为 NumPy 数组。但是，它将创建一个二维数组:

## 方法 28:从字符串中的值

要从字符串中的文本数据创建一维 NumPy 数组，请使用`[np.fromstring()](https://numpy.org/doc/stable/reference/generated/numpy.fromstring.html)`方法:

如果输入字符串有逗号分隔的值，可以按如下方式更改分隔符:

## 方法 29:定义 NumPy 数组的对角线

您可以使用`[np.diag()](https://numpy.org/doc/stable/reference/generated/numpy.diag.html?highlight=diag#numpy.diag)`方法创建一个沿对角线具有特定值的方形二维 NumPy 数组:

## 方法 30:从 CSV 文件

最后，如果您想要加载一个 CSV 文件并将其解释为一个 NumPy 数组，请使用`[np.loadtxt()](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html?highlight=loadtxt#numpy.loadtxt)`方法。

CSV 文件的内容如下所示:

要创建一个 NumPy 数组，使用如下的`[np.loadtxt()](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html?highlight=loadtxt#numpy.loadtxt)`方法:

由于第一行是标题行，我们跳过读取，从第二行开始。

# 结论

总之，在这篇文章中，我展示了 30 种创建 NumPy 数组的流行方法。

如果你想深入了解这个话题，请点击查看 NumPy 创建例程的官方指南[。](https://numpy.org/doc/1.23/reference/routines.array-creation.html#routines-array-creation)

**感谢阅读！**

[🧑‍💻**成为数据科学专家！获取包含 450 多个熊猫、NumPy 和 SQL 问题的免费数据科学掌握工具包。**](https://subscribepage.io/450q)

✉️ [**注册我的电子邮件列表**](https://medium.com/subscribe/@avi_chawla) 不要错过另一篇关于数据科学指南、技巧和提示、机器学习、SQL、Python 等的文章。Medium 会将我的下一篇文章直接发送到你的收件箱。