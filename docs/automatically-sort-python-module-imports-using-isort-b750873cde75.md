# 使用 isort 自动排序 Python 模块导入

> 原文：<https://towardsdatascience.com/automatically-sort-python-module-imports-using-isort-b750873cde75>

## 在本教程中，我们将探索如何使用 isort 库自动排序 Python 模块导入

![](img/fa484f4de5f8dbd5c3e82728cf98db93.png)

Jonah Pettrich 在 [Unsplash](https://unsplash.com/s/photos/sorted?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

**目录**

*   介绍
*   什么是 isort
*   示例代码文件
*   如何对模块导入进行排序
*   结论

# 介绍

随着 Python 项目的增长，您开始拥有越来越多的文件，每个文件都有更多的代码行，执行更多的操作，并且导入更多的依赖项。

在 research step 中，我们通常在需要时一个接一个地导入库，这使得整个导入部分变得杂乱无章，并且通常难以快速编辑。

此外，当在一个工程师团队中工作时，大多数人都有他们自己偏好的构造和排序导入的方式，这导致不同的文件版本在同一个存储库中相互覆盖。

使用 [isort](https://pycqa.github.io/isort/index.html) 可以很容易地解决这个问题，它提供了一种在 Python 项目中对导入进行排序的系统方法。

为了继续学习本教程，我们需要以下 Python 库:isort。

如果您没有安装它，请打开“命令提示符”(在 Windows 上)并使用以下代码安装它:

```
pip install isort
```

# 什么是 isort

isort 是一个 Python 实用程序和库，它自动按字母顺序对 Python 模块导入进行排序，同时按类型将它分成不同的部分。

除了 CLI 实用程序和 Python 库之外，它还有许多代码编辑器插件，如 VS Code、Sublime 等等。

# 示例代码文件

为了测试 isort 库的功能，我们需要一个示例 Python 文件。

在这个示例文件中，我们将混合导入的顺序，并添加一些间距来说明未排序文件和排序文件之间的区别。

下面是一个未排序的 Python 代码示例( **main.py** ):

# 如何对模块导入进行排序

一旦我们在目录中有了 Python 文件，就很容易用 isort 对模块导入进行排序。

打开命令行或终端，导航到包含 Python 文件的目录。

## 单个 Python 文件中的排序模块导入

如果只有一个文件需要对模块导入进行排序(在我们的例子中是 **main.py** ，只需运行:

```
isort main.py
```

重新格式化的示例 Python 文件应该如下所示:

看起来好多了，所有的模块导入都是有序的！

## 多个 Python 文件中的排序模块导入

如果您想对多个 Python 文件或整个 Python 项目中的模块导入进行排序，只需运行:

```
isort .
```

isort 将自动查找所有的 Python 文件，并在目录中的所有 Python 文件中对模块导入进行排序。

# 结论

在本文中，我们探索了如何使用 [isort](https://pycqa.github.io/isort/index.html) 库对 Python 文件中的模块导入进行自动排序。

如果你有任何问题或对编辑有任何建议，请随时在下面留下评论，并查看我的更多 [Python 编程](https://pyshark.com/category/python-programming/)教程。

*原载于 2022 年 9 月 23 日 https://pyshark.com*[](https://pyshark.com/automatically-sort-python-module-imports-using-isort/)**。**