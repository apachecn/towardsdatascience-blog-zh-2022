# 你的 Python For-loop 慢吗？请改用 NumPy

> 原文：<https://towardsdatascience.com/numpy-vectorization-speed-ffdab5deb402>

## 当速度很重要时，列表不是最好的。

![](img/d91f52b78be73f9486d926e64e0c489a.png)

来自 [Pexels](https://www.pexels.com/photo/close-up-photography-of-gray-stainless-steel-fan-turned-on-surrounded-by-dark-background-1374448/) 的 Alireza Kaviani 拍摄的照片。

速度一直是开发人员关心的问题——尤其是对数据敏感的工作。

迭代能力是所有自动化和扩展的基础。我们所有人的首要选择是 for 循环。它优秀、简单、灵活。然而，它们不是为扩展到大规模数据集而构建的。

这就是矢量化的用武之地。当您在 for 循环中进行大量数据处理时，请考虑矢量化。Numpy 在这里派上了用场。

这篇文章解释了 NumPy 操作与 for 循环相比有多快。

[](/how-to-speed-up-python-data-pipelines-up-to-91x-80d7accfe7ec) [## 如何将 Python 数据管道加速到 91X？

### 一个 5 分钟的教程可以为您的大数据项目节省数月时间。

towardsdatascience.com](/how-to-speed-up-python-data-pipelines-up-to-91x-80d7accfe7ec) 

# 用 NumPy 比较 For 循环

我们来做一个简单的求和运算。我们必须总结一个列表中的所有元素。

sum 是 Python 中的一个内置操作，可以在一系列数字上使用。但是让我们假设没有，您需要实现它。

任何程序员都会选择遍历列表并将数字添加到变量中。但是有经验的开发人员知道它的局限性，会选择优化的版本。

[](/how-to-detect-memory-leakage-in-your-python-application-f83ae1ad897d) [## 如何检测 Python 应用程序中的内存泄漏

### 标准的 Python 库，可以显示每一行的内存使用和执行时间

towardsdatascience.com](/how-to-detect-memory-leakage-in-your-python-application-f83ae1ad897d) 

这是我们总结的列表和数字版本。在这个例子中，我们创建了一个包含 100 万个 0 到 100 之间的随机数的数组。然后我们使用这两种方法并记录执行时间。

我在比较 NumPy sum 和 list 迭代的速度。—作者摘录。

让我们运行这个程序，看看我们会得到什么。输出可能如下所示。

```
$ python main.py 
Summation time with for-loop:  14.793345853999199
Summation time with np.sum:  0.1294808290003857
```

NumPy 版本更快。这大约是循环所用时间的百分之一。

# 使用 Numpy 加速计算的更多示例

NumPy 大量用于数值计算。也就是说，如果您正在处理庞大的数据集矢量化，NumPy 的使用是不可避免的。

大多数机器学习库使用 NumPy 来优化算法。如果你曾经创建过 scikit learn to model，你应该已经使用过 NumPy 了。

这里还有一些处理大量数字数据时经常用到的例子。

## NumPy 与列表中乘积的总和

这是一种流行的数值计算，你甚至可以在 Excel 上使用。让我们来衡量一下 lists 和 NumPy 版本的性能。

下面的代码将一个数组中的每个元素与另一个数组中的相应元素相乘。最后，我们总结所有的单个产品。

下面是上面代码的输出:

```
$ python main.py 
Sum of products with for loop:  26.099454337999987
Sum of products with np.sum:  0.28206900699990456
```

同样，NumPy 版本比遍历列表快 100 倍。

## NumPy 和链表的矩阵乘法性能。

矩阵乘法是和积的扩展版本。它涉及的不是单个数组，而是数组的数组。

在实现涉及大量数据的算法时，矩阵乘法也非常常见。这里是基准。

```
$ python main.py
Matrix multiplication with for loop:  1597.9121425140002
Matrix multiplication with numpy:  2.8506258010002057 
```

使用 NumPy 的结果是深远的。我们的矢量化版本运行速度快了 500 多倍。

随着数组的大小和维度的增长，NumPy 的优势更加突出。

# 为什么 NumPy 比 lists 快？

简单；它们被设计用于不同的目的。

NumPy 的角色是为数值计算提供一个优化的接口。然而，Python 列表只是对象的集合。

NumPy 数组只允许**同类数据类型**。因此，NumPy 操作在算法的每一步之前都不必担心类型。这就是我们提高速度的地方——快速取胜。

同样，在 NumPy 中，整个数组，而不是单个元素，是一个被称为**密集打包的**对象。因此，它需要更少的内存。

此外，NumPy 操作(主要是)**是用 C** 实现的，而不是用 Python 本身。

[](/challenging-cython-the-python-module-for-high-performance-computing-2e0f874311c0) [## 挑战 cy thon——高性能计算的 Python 模块。

### 现代的替代方案看起来很有希望，Python 可以以闪电般的速度运行。

towardsdatascience.com](/challenging-cython-the-python-module-for-high-performance-computing-2e0f874311c0) 

Python 中的列表只不过是一个对象存储。单个对象占用空间，你很快就需要更多的内存来处理它们。此外，列表可以容纳不同类型的对象。但是不利的一面是，您必须对每个操作进行元素类型检查。这使得成本很高。

# 最后的想法

这篇文章鼓励你将列表转换成 NumPy 数组，并使用向量化操作来加速执行。

人们在列表中使用 for 循环是很自然的，因为这很简单。但如果涉及到很多数字，就不是最优的方式。为了更好地理解它，我们比较了一些简单运算的性能，比如求和、和积和矩阵乘法。在所有情况下，NumPy 的表现都远远好于 lists。

For 循环在编程中也有它们的位置。经验法则是当你的数据结构更复杂，需要迭代的项目更少时使用它们。

没有 NumPy 的几百个数相加可能更好。此外，如果在每次迭代中你必须做比数值计算更多的工作，NumPy 不是你的选择。

[](/how-to-serve-massive-computations-using-python-web-apps-590e51624bc6) [## 如何使用 Python Web 应用服务于大规模计算？

### 克服 Python 的局限性，并通过 web 请求将其用于繁重的数据分析和机器学习。

towardsdatascience.com](/how-to-serve-massive-computations-using-python-web-apps-590e51624bc6) 

> 感谢阅读，朋友！在[**LinkedIn**](https://www.linkedin.com/in/thuwarakesh/)[**Twitter**](https://twitter.com/Thuwarakesh)[**Medium**](https://thuwarakesh.medium.com/)上跟我打招呼。
> 
> 还不是中等会员？请使用此链接 [**成为会员**](https://thuwarakesh.medium.com/membership) 因为，在没有额外费用的情况下，我赚取了一点佣金。