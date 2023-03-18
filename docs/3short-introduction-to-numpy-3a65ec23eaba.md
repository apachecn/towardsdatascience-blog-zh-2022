# NumPy 简介

> 原文：<https://towardsdatascience.com/3short-introduction-to-numpy-3a65ec23eaba>

## 数字图书馆和 ufuncs 的一些基本知识

![](img/547c0d82e9595d0afe60a1f9792ee01b.png)

埃里克·麦克林在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

NumPy 代表数值 Python，是一个用于处理数组的 [Python](https://databasecamp.de/en/python-coding) 库。在这些数组的帮助下，来自线性代数的元素，比如向量和矩阵，可以用 [Python](https://databasecamp.de/en/python-coding) 来表示。由于该库的大部分是用 C 语言编写的，所以即使使用大型矩阵，它也能执行特别高效和快速的计算。

# 什么是 NumPy？

[Python](https://databasecamp.de/en/python-coding) 提供了多种数据结构，可用于存储数据，无需额外的库。然而，这些结构，比如 [Python 列表](https://databasecamp.de/en/python-coding/python-lists)，非常不适合数学运算。在处理大量数据时，逐个元素添加两个数字列表[会很快对性能产生不利影响。](https://databasecamp.de/en/python-coding/python-lists)

由于这个原因，NumPy 被开发出来，因为它提供了快速有效地执行数值运算的可能性。尤其重要的是来自线性代数领域的计算，例如矩阵乘法。

# 如何安装 NumPy？

像许多其他库一样，NumPy 可以使用 pip 从笔记本上直接安装。为此，使用命令“pip install”和模块名称。这一行前面必须有一个感叹号，以便笔记本识别出这是一个终端命令:

如果安装成功，模块可以简单地导入并在笔记本中使用。这里经常使用缩写“np ”,以便在编程过程中节省一点时间，并且不必每次都输入 NumPy:

# 什么是 NumPy 数组？

NumPy 数组是传统 [Python 列表](https://databasecamp.de/en/python-coding/python-lists)的有效替代。它们提供了存储多维数据集的可能性。在大多数情况下，数字被存储，数组被用作向量或矩阵。例如，一维向量可能是这样的:

除了 NumPy 数组的不同功能(我们将在另一篇文章中介绍)之外，可能的维度对于区分仍然很重要:

区分了以下维度:

*   **0D —数组**:这只是一个标量，即单个数字或值。
*   **1D —数组**:这是一个向量，作为一维的一串数字或值。
*   **2D 阵列**:这种阵列是一个矩阵，也就是几个 1D 阵列的集合。
*   **3D — Array** :几个矩阵组成一个所谓的张量。我们已经在关于[张量流](https://databasecamp.de/en/python-coding/tensorflow-en)的文章中对此进行了更详细的解释。

# NumPy 数组和 Python 列表有什么区别？

根据来源的不同，NumPy 数组和 [Python 列表](https://databasecamp.de/en/python-coding/python-lists)之间有几个基本的区别。最常提到的有:

1.  **内存消耗**:数组的编程方式是它们占据内存的某一部分。然后，数组的所有元素都位于那里。另一方面，[列表](https://databasecamp.de/en/python-coding/python-lists)的元素在内存中可能相距很远。因此，一个[列表](https://databasecamp.de/en/python-coding/python-lists)比一个相同的数组消耗更多的内存。
2.  **速度**:数组的处理速度也比[列表](https://databasecamp.de/en/python-coding/python-lists)快得多，因为它们的内存消耗更低。这对于有几百万个元素的对象来说有很大的不同。
3.  **功能**:数组提供了更多的功能，例如，它们允许逐个元素的操作，而列表则不允许。

# 什么是 Numpy ufuncs？

所谓的“通用函数”(简称:ufuncs)用于不必逐个元素地执行某些操作，而是直接对整个数组执行。在计算机编程中，当命令直接对整个向量执行时，我们称之为矢量化。

这不仅在编程上快得多，而且导致更快的计算。在 NumPy 中，提供了几个这样的通用函数，可以用于各种操作。其中最著名的有:

*   使用“add()”可以逐个元素地对多个数组求和。
*   “subtract()”正好相反，它逐个元素地减去数组。
*   “multiply()”将两个数组逐个元素相乘。
*   " matmul()"形成两个数组的矩阵乘积。请注意，在大多数情况下，这不会给出与“multiply()”相同的结果。

# 这是你应该带走的东西

*   NumPy 代表 Numerical Python，是一个用于处理数组的 Python 库。
*   在这些数组的帮助下，线性代数中的元素，比如向量和矩阵，可以用 Python 来表示。
*   因为这个库的大部分是用 C 语言编写的，所以它可以执行特别高效和快速的计算，即使是大型矩阵。
*   NumPy 数组与 Python 列表相当，但在内存需求和处理速度方面明显优于 Python 列表。

*如果你喜欢我的作品，请在这里订阅*[](https://medium.com/subscribe/@niklas_lang)**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，medium 允许你每月免费阅读* ***3 篇*** *。如果你希望有****无限制的*** *访问我的文章和数以千计的精彩文章，请不要犹豫，点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格**

*[](/4-basic-commands-when-working-with-python-dictionaries-1152e0331604) [## 使用 Python 字典时的 4 个基本命令

### 让您了解 Python 字典的特征以及如何处理它们

towardsdatascience.com](/4-basic-commands-when-working-with-python-dictionaries-1152e0331604) [](/exception-handling-in-python-8cc8f69f16ad) [## Python 中的异常处理

### 了解如何使用 Python Try Except

towardsdatascience.com](/exception-handling-in-python-8cc8f69f16ad) [](/6-fundamental-questions-when-working-with-a-pandas-series-1d142b5fba4e) [## 使用熊猫系列时的 6 个基本问题

### 了解熊猫系列的特点

towardsdatascience.com](/6-fundamental-questions-when-working-with-a-pandas-series-1d142b5fba4e)*