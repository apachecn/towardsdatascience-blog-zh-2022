# 你应该避免的 5 个 Python 错误

> 原文：<https://towardsdatascience.com/5-python-mistakes-you-should-avoid-12104e350030>

## 一些最常见的错误很难找到

![](img/2bdc0de5e698e83baacba72b05ce6366.png)

Javier Allegue Barros 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

Python 非常直观，是编程初学者的最爱。虽然语法简单明了，脚本也很简短，但还是需要注意一些细节。忽略它们可能会导致您的代码被破坏，让您头疼。

在本文中，我们将看看五个常见的 Python 编程新手错误以及如何防止它们。

# 1.一遍又一遍地修改列表

在迭代集合或列表时对其进行更改并不是一个好主意。在迭代过程中，许多程序员不小心从列表中删除了项目。这里有一个恰当的例子:

```
odd = lambda x : bool(x % 2)
numbers = [i for i in range(10)]
for i in range(len(numbers)):
    if odd(numbers[i]):
        del numbers[i]
```

具体来说，错误如下:

> 索引错误:列表索引超出范围

解决方案:使用列表理解可以帮助我们。

```
odd = lambda x : bool(x % 2)
nums = [i for i in range(10)]
nums[:] = [i for i in nums if not odd(I)]
print(nums)
```

# 2.模块名称冲突

可用 Python 包和库的数量和种类令人印象深刻。如果您为 Python 模块指定的名称与 Python 标准库中已有的名称相同，则可能会出现名称冲突。

你应该注意代码中的模块和标准库中的模块之间的名称冲突，比如 **math.py** 和 **email.py** 。

如果您导入一个库，并且该库试图从 Python 标准库中导入模块，您可能会遇到一些棘手的问题。因此，软件包可能会尝试从 Python 的标准库中导入您的复制模块，而不是正式模块。

出于这个原因，您不应该使用与 Python 标准库中相同的模块名称。

# 3.不关闭已经打开的文件

Python 建议在完成最后一个操作并且文件不再使用后关闭打开的文件。

请务必记住，您打开的文件可能会占用系统资源，如果您在使用完这些文件后不关闭它们，它们会被锁定。

在读取文件时，始终使用**和**将有助于防止这些问题。它会自动保存您的更改，并在您完成后关闭文件。

# 示例:

而不是:

```
file_1 = open(‘filename_demo.txt’, ‘w’)
file_1.write(‘new_data’)
file_1.close()
```

方法如下:

```
with open(‘filename_demo.txt’, ‘w’) as file_1:
    file_1.write(‘new_data’)
```

# 4.不了解 Python 函数是如何工作的

Python 预装了几个有用的工具。他们中的一些人可能做类似的任务；然而，它们的做法可能不同。如果我们，作为程序员，没有完全掌握某个函数是如何操作的，那么如果我们使用它，我们就冒着得到意想不到的后果的风险。

在 Python 中，我们有两个不同的函数——**sort()**和 sorted——用于以某种 **sorted()** 排列集合中的项目。它们都服务于相同的目的——以一定的顺序排列一个集合。但是这两个功能是如何运作的是截然不同的。

# 示例:

```
list1 = [6, 5, 7, 2, 9, 3]
print(list1.sort())list2 = [6, 2, 8, 5, 3, 11]
print(sorted(list2))
```

# 输出:

```
None[2, 3, 5, 6, 8, 11]
```

到底发生了什么事？尽管 sort()和 sorted()都很有用，但排序后的列表是由后者打印的，而 sort()返回 None。

在这种情况下， **sort()** 在排序(就地排序)时修改原始序列，不返回任何内容。另外， **sorted()** 函数总是在不改变输入顺序的情况下生成一个排序列表。

# 5.误用 _init_ 方法

**_init_** 函数是一个特殊的、保留的 Python 方法，用于创建对象。每当 Python 创建一个类的实例时都会调用它，让该实例为该类的属性和方法设置值。

当一个类的对象被创建时，这个方法的工作是填充类的数据成员的值。然而，程序员经常因为让函数返回值而偏离了这个 **_init_** 函数的预期用途。

# 结论

我们已经讨论了每个 python 开发人员最常犯的五个错误以及如何解决它们。

如果您采用这些改进技巧，您可以使您的 Python 代码更加优化和无 bug。这些技巧在某些时候对竞争性编程也很有帮助。

我希望你喜欢这篇文章。更多精彩文章敬请期待。

感谢阅读！

> *在你走之前……*

如果你喜欢这篇文章，并且想继续关注更多关于 **Python &数据科学**的**精彩文章**——请考虑使用我的推荐链接[https://pranjalai.medium.com/membership](https://pranjalai.medium.com/membership)成为中级会员。

还有，可以随时订阅我的免费简讯: [**Pranjal 的简讯**](https://pranjalai.medium.com/subscribe) 。