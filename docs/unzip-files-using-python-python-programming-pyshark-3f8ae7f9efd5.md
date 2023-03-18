# 使用 Python-Python 编程 PyShark 解压缩文件

> 原文：<https://towardsdatascience.com/unzip-files-using-python-python-programming-pyshark-3f8ae7f9efd5>

## 在本教程中，我们将探索如何使用 Python 解压文件

![](img/a6067a4adc836001d1028333f57580e1.png)

托马斯·索贝克在 [Unsplash](https://unsplash.com/s/photos/zip?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

**目录**

*   介绍
*   创建一个示例 ZIP 文件
*   使用 Python 从一个 ZIP 文件中提取所有文件
*   使用 Python 从 ZIP 文件中提取单个文件
*   使用 Python 基于条件从 ZIP 文件中提取文件
*   结论

# 介绍

ZIP 文件是我们经常看到的东西。它只是一个包含多个压缩文件的文件(存档)。

这对于高效的数据传输以及通过减小文件大小来存储较大的文件非常有用。

一次处理许多 ZIP 文件可能是一项非常手动的任务，但是 Python 允许我们高效地处理多个 ZIP 文件，并非常快速地从中提取数据。

为了继续学习本教程，我们需要以下 Python 库:**zip file**(Python 中内置的)。

# 创建一个示例 ZIP 文件

为了继续学习本教程，我们需要一个 ZIP 文件。

如果你已经有了，那就太好了。如果你没有，那么请随意下载一个我创建并上传到 [Google Drive](https://drive.google.com/file/d/1MvTVGvMqMKdQ9Ab5KKVIi_yx4Aoq1XHN/view?usp=sharing) 的示例 ZIP 文件( *my_files.zip* )。

这个 ZIP 文件包含三个文件:

*   *customers.csv*
*   *products.csv*
*   *code_snippet.png*

下载完成后，将它放在 Python 代码文件所在的目录中。

# 使用 Python 从一个 ZIP 文件中提取所有文件

我们手动对 ZIP 文件执行的最常见的任务之一是从其中提取所有文件。

使用 Python 中的 zipfile 库，我们可以用几行代码来实现:

我们需要做的就是创建一个 ZipFile 类的实例，并将 ZIP 文件的位置和“读取”模式作为参数传递给它，然后使用**提取所有文件。**extract all()方法。

下面是编写相同代码的另一种方法:

在这两种情况下，这三个文件都将从 ZIP 文件中提取出来。

# 使用 Python 从 ZIP 文件中提取单个文件

我们可能有的另一个任务是使用 Python 从 ZIP 文件中提取特定的单个文件。

首先，让我们找到在 ZIP 文件中归档的文件列表:

您应该得到:

```
['code_snippet.png', 'customers.csv', 'products.csv']
```

假设我们只想提取 **customers.csv** 和 **products.csv** 文件。

因为我们知道文件名，所以在使用 Python 从 ZIP 文件中提取文件时，我们可以使用它们作为标识符:

你应该看看这两个。csv 文件被提取到 Python 代码所在的文件夹中。

# 使用 Python 基于条件从 ZIP 文件中提取文件

在上面的例子中，我们提取了两个。使用 Python 将 ZIP 文件转换为 csv 文件。

一个接一个地提取单个文件只在我们处理几个文件时有效。

假设我们现在有一个包含数百个文件的大型 ZIP 文件，我们只想提取 CSV 文件。

我们可以根据文件名中的某些条件提取这些文件。

对于 CSV 文件，它们的名称以“.”结尾。csv”，我们可以在使用 Python 从 ZIP 文件中提取文件时使用它作为过滤条件:

您应该会看到从 ZIP 文件中提取的两个 CSV 文件，这与上一节中的结果相同。

# 结论

在本文中，我们探讨了如何使用 Python 从 ZIP 文件中提取文件。

如果你有任何问题或对一些编辑有建议，请随时在下面留下评论，并查看我的更多 [Python 编程](https://pyshark.com/category/python-programming/)教程。

*原载于 2022 年 5 月 9 日*[*【https://pyshark.com】*](https://pyshark.com/unzip-files-using-python/)*。*