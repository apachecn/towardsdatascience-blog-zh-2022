# 跑得比以往任何时候都快

> 原文：<https://towardsdatascience.com/running-faster-than-ever-before-afd1d213a3ae>

## Python 3.11 与 Python 3.10 和 Julia 1.7 相比表现如何？

![](img/c3671128cc599ebca29b772732be6fda.png)

肯尼·埃利亚松在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

性能并不总是数据科学家优先考虑的事情。然而，随着我们手上的数据量不断增加，更快地完成计算不是很好吗？即将到来的 Python 3.11 版本被寄予厚望，因为与当前稳定的版本 3.10 相比，预计[性能会提高 10-60%](https://docs.python.org/3.11/whatsnew/3.11.html)。我真的对这一声明很感兴趣，我想尝试一下我的应用程序的新版本，这些应用程序主要集中在蛋白质组学和质谱学上。

在这篇文章中，我关注了截至 2022 年 6 月的当前 Python 3.10 和 3.11 测试版的性能。与 MS 相关的函数包括从磁盘读取文本文件，对字符串和字典的操作，以及对 *numpy* 数组的循环和操作。这些操作非常通用，因此这些函数的性能应该反映一般的性能趋势。

既然我们在这里谈论速度，我在想，我们为什么不也比较一下 Python 3.11 和 [Julia](https://julialang.org/) 的性能。它是一种正在积极发展的编程语言，是专门为科学计算设计的，考虑到了可伸缩性和性能。Julia 的语法与 Python 并没有太大的不同，这使得它对于希望进一步扩展其计算工具箱的数据科学家来说是一种非常有吸引力的语言。

如果您希望直接查看基准测试的结果，请随意跳到文章末尾的总结部分。

## 设置

这些测试已经在 Kubuntu 22.04 中用 Python 3.10.4、Python 3.11 beta3 和 Julia 1.7.3 运行过。最初，我在 Jupyter 笔记本上运行 Python 3.10 和 Julia 测试，然而，我在安装 3.11 测试版的所有必要依赖项时遇到了一些困难。因此，我通过运行脚本完成了对两个 Python 版本的测试。

代码、笔记本和示例数据可以在 [GitHub 库](https://github.com/dev-ev/julia-vs-python-ms-scripting)中找到。

## 读取和解析文本文件

第一个函数从光盘中读取一个文本文件，将 MS 数据解析到一个字典中，该字典包括文本字符串、数值和 *pandas* dataframes。函数 *load_mgf* 的完整代码相当长，你可以在[的脚本](https://github.com/dev-ev/julia-vs-python-ms-scripting/blob/main/mgf_read_match.py)中找到。该函数转换一个纯文本文件，如下所示:

一个对象列表(字典)，每个代表一个质谱，包含文本、数字数据和一个*pandas*data frame*‘ms _ data’*，如下所示:

Github 存储库中的示例文件包含 1000 个项目(光谱):

```
1000
```

在库 [*timeit*](https://docs.python.org/3/library/timeit.html) 的帮助下，我们在 Python 的两个版本中重复运行这个函数:

```
Running version:
3.10.4 (main, Apr  2 2022, 09:04:19) [GCC 11.2.0]
Timing the load_mgf function:
Mean time 251.7 ms, standard deviation 7.5 ms for 200 repeatsRunning version:
3.11.0b3 (main, Jun  1 2022, 23:49:29) [GCC 11.2.0]
Timing the load_mgf function:
Mean time 217.8 ms, standard deviation 8.6 ms for 200 repeats
```

我们观察到 Python 3.11 的处理时间减少了 10%以上！这非常令人兴奋，因为我们不需要以任何方式更改代码就可以获得这种性能提升！

让我们[在 Julia](https://github.com/dev-ev/julia-vs-python-ms-scripting/blob/main/mgf_read_match.jl) 中实现相同的功能，尽可能地模仿 Python 函数的逻辑。使用 *@benchmark* ，看看有多快:

Julia 的性能证明在这个基准测试中得到证实，平均运行时间不到 Python 的一半！看起来很有希望！

## 循环和字典操作

下一个函数从列表中读取文本字符串，使用字典匹配字符，并计算相应值的总和。对于质谱分析人员来说，这个函数从列表中计算肽的质量。

让我们准备一个 10，000 个肽(字符串)的列表，每个肽由 20 个氨基酸残基(字符)组成，指定为单独的大写字母:

```
10000
'FFKGSQDTGYTYFNFMSHFY'
```

我们现在可以使用普通循环来计算列表中每个序列的质量。每个可能的残基对应一个数字质量值，我们可以使用字典来指定:

另一种更简洁的方法是使用 [*functools*](https://docs.python.org/3/library/functools.html) 包中的 *reduce* 函数:

让我们用两个版本的 Python 对函数进行基准测试:

```
Running version:
3.10.4 (main, Apr  2 2022, 09:04:19) [GCC 11.2.0]
Timing the calculate_masses_loop function:
Mean time 11.83 ms, standard deviation 0.97 ms for 200 repeats
Timing the calculate_masses_reduce function:
Mean time 24.15 ms, standard deviation 1.95 ms for 200 repeatsRunning version:
3.11.0b3 (main, Jun  1 2022, 23:49:29) [GCC 11.2.0]
Timing the calculate_masses_loop function:
Mean time 7.90 ms, standard deviation 0.88 ms for 200 repeats
Timing the calculate_masses_reduce function:
Mean time 19.65 ms, standard deviation 1.72 ms for 200 repeats
```

Python 3.11 在基于循环的函数上快了 49%！不幸的是，聪明的*calculate _ mass _ reduce*实现通常要慢得多，但是 Python 3.11 在这个函数上仍然比 3.10 版本快 18%。

这些计算应该在 Julia 中变得更快吗？让我们创建一个类似的循环函数，它将计算每个序列的质量，并在一个 10，000 长的序列列表上对其进行基准测试:

令人惊讶的是，这个实现比 Python 中的函数慢多了！我们可以通过预先分配 masses 数组来优化代码，而不是在每次迭代中添加一个新的单元素数组。让我们看看这是否有所不同:

这两种实现在 Julia 中略有不同，但 Python 3.11 是这一轮的明显赢家！

## 数值数组上的运算

谈到 Python 中的数字运算，我们当然希望避免多级 for 循环和重复的 if-else 语句。因此，在实现一个用于处理大量实验质谱数据的函数时，我们将使用 *numpy* 数组对逻辑进行矢量化。完整的功能 *find_matches* 可以在脚本中找到[。该算法的核心部分采用数值，作为 *pandas* 系列 *s['ms_data']['m/z']* 引入，并对相应的 *numpy* 数组执行一系列操作:](https://github.com/dev-ev/julia-vs-python-ms-scripting/blob/main/mgf_read_match.py)

让我们使用我们之前通过 *load_mgf* 加载到内存中的 1000 MS 光谱对函数 *find_matches* 进行基准测试。重复运行该函数，我们得到:

```
Running version:
3.10.4 (main, Apr  2 2022, 09:04:19) [GCC 11.2.0]
Timing the find_matches function:
Mean time 2.08 s, standard deviation 89.2 ms for 50 repeatsRunning version:
3.11.0b3 (main, Jun  1 2022, 23:49:29) [GCC 11.2.0]
Timing the find_matches function:
Mean time 2.07 s, standard deviation 26.0 ms for 50 repeats
```

两个版本的平均时间惊人地相似。也许，在 3.11 的测试版中，numpy 代码没有太大的变化。这将是很有趣的，看看它是否会被修改。

在 Julia 中实现一个类似的函数，我们可以在数组*s[' ms _ data '][' m/z ']【T21]上重复上面 Python 代码中相同的操作。在 Julia 代码中，矢量化运算以点为前缀:*

上面的基准测试结果显示，Julia 中的函数比 *numpy* 中快一倍，但这并不是终点。Julia 允许将数组操作融合到一个循环中，避免分配临时的中间数组。融合操作以“at-dot”宏为前缀:

做了这一点小小的改变后，让我们对结果函数 *matchesfuse* 进行基准测试:

融合进一步削减了 60%的运行时间！由此，Julia 完全主导了数组操作基准测试，优化后的函数运行速度比 Python/*numpy*analog 快 5 倍！这个令人印象深刻的表演真的激励我，至少对我来说，投入更多的时间去研究朱莉娅。

## 摘要

我们在基本操作(循环、字符串操作、算术运算)和数值数组的计算中测试了 Python 3.11 beta，所有这些都是在质谱数据的上下文中进行的。我们将 3.11 与当前的 Python 版本 3.10.4 以及最新的 Julia 1.7.3 进行了比较。测试结果汇总如下图所示:

![](img/5c0f46857ef9841d060891f4d275903e.png)

质谱相关基准功能的运行时间总结。作者图片

与 Python 3.10 相比，新的 3.11 在循环、字符串和字典操作等一般操作上提供了显著的性能提升，完成基准测试的速度提高了 10–49%。与此同时，使用 *numpy* 的计算在两个 Python 版本中花费的时间完全相同，而 Julia 在同一组数值计算的速度上提供了 2 到 5 倍的显著提升。然而，在 Python 中，使用循环和字典匹配的函数工作得更快，尤其是在新版本 3.11 中。