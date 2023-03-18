# NumPy 数组简介

> 原文：<https://towardsdatascience.com/introduction-to-numpy-arrays-c0a1082afadd>

## 了解 NumPy 数组的基础知识

![](img/a548171b21bbede28fffbd455c3223cb.png)

Pierre Bamin 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

NumPy 数组是来自 Python 库 [NumPy](https://databasecamp.de/en/python-coding/numpy-en) 的一个数据对象，用于存储一个数据类型的对象。由于它的编程比同类的 [Python](https://databasecamp.de/en/python-coding) 数据对象更接近内存，它可以更有效地存储数据集，因此处理速度也更快。

# 什么是 NumPy？

[Python](https://databasecamp.de/en/python-coding) 提供了多种数据结构，可用于存储数据，无需额外的库。然而，这些结构，比如 [Python 列表](https://databasecamp.de/en/python-coding/python-lists)，非常不适合数学运算。在处理大量数据时，逐个元素地添加两个数字列表会很快对性能产生不利影响。

由于这个原因， [NumPy](https://databasecamp.de/en/python-coding/numpy-en) 被开发出来，因为它提供了快速有效地执行数值运算的可能性。尤其重要的是来自线性代数领域的计算，例如矩阵乘法。

# 如何定义 NumPy 数组

顾名思义，数组是 [NumPy](https://databasecamp.de/en/python-coding/numpy-en) 库的一部分。所以必须先导入才能使用数组。然后简单地通过在方括号中插入元素来创建。这里元素的顺序起了作用:

# 数组的属性是什么？

数组是元素的集合，这些元素必须都是相同的数据类型[。大多数数字存储在其中，但字符串也可以存储。只有在一个数组中不同数据类型的混合是不可能的。](https://databasecamp.de/en/data/data-types)

为了描述数组的结构，有三个基本的属性，这三个属性经常被混淆:

*   **维数:**数组中的维数表示查询一个数组的特定元素需要多少个索引。每个维度都可以用来存储相关信息。例如，如果您想按时间分析公司的销售数字，您可以使用一个维度来分析不同日期的销售额，而使用另一个维度来分析一个月的销售额。
*   **Shape:**Shape 指定数组包含的所有维度的大小。例如，如果你有一个三维数组，你将得到一个长度为 3 的[元组](https://databasecamp.de/en/python-coding/python-tuples)。[元组](https://databasecamp.de/en/python-coding/python-tuples)的第一个元素表示第一维度的元素数量，第二个元素表示第二维度的元素数量，依此类推。
*   **大小:**最后，数组的大小表示数组中总共存储了多少个数字或元素。具体来说，这是形状返回的单个元素的乘积。

我们现在定义一个二维 NumPy 数组，它在两个维度中都有三个元素:

# 多维数组是如何构造的？

在应用程序中，一个维度通常不足以完整地写出事实。NumPy 数组也适用于存储多维对象。表格就是这样一个例子，它由两个维度组成，即行和列。这也可以通过将行指定为列表的列表来相对容易地定义:

类似地，可以添加其他维度。例如，我们可以创建一个对象，该对象只包含前一个示例中的表两次。这样我们就得到一个三维物体:

# 如何从数组中获取单个元素？

由于 NumPy 数组在结构上与 [Python 列表](https://databasecamp.de/en/python-coding/python-lists)非常相似，所以数组也使用所谓的索引来访问元素就不足为奇了。这意味着您可以根据位置查询单个元素。计数从 0 开始向上。

同样，您也可以使用负索引，从而从后面遍历数组。然而，与正索引相反，它从-1 开始，代表数组的最后一个元素。-2 则相应地是数组倒数第二个元素的索引:

在多维数组中，单个索引不足以查询单个元素。通常，为了能够查询一个元素而不获取元素列表，必须为每个维度指定一个索引。在我们的多维数组中，我们查询第一维的第二个元素:

# Python 列表和 NumPy 数组有什么区别？

在本文的这一点上，您可能会认为 NumPy 数组只是对 [Python 列表](https://databasecamp.de/en/python-coding/python-lists)的一种替代，后者甚至有一个缺点，即只能存储单一[数据类型](https://databasecamp.de/en/data/data-types)的数据，而列表也存储字符串和数字的混合。然而， [NumPy](https://databasecamp.de/en/python-coding/numpy-en) 的开发者决定在数组中引入一个新的数据元素肯定是有原因的。

NumPy 数组相对于 [Python 列表](https://databasecamp.de/en/python-coding/python-lists)的主要优势是内存效率和相关的读写速度。在许多应用程序中，这可能并不重要，但是，当处理数百万甚至数十亿个元素时，它可以节省大量时间。因此，阵列通常用于开发高性能系统的复杂应用中。

# 这是你应该带走的东西

*   NumPy 数组是 NumPy 库中的一个数据元素，可以存储多维数组的信息。
*   与 Python 列表相比，它只能存储相同数据类型的元素。然而，反过来，它的内存效率更高，因此功能也更强大。
*   NumPy 数组的基本属性是维数、形状和大小。