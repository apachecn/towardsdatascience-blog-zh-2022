# 这个装饰器将使 Python 速度提高 30 倍

> 原文：<https://towardsdatascience.com/this-decorator-will-make-python-30-times-faster-715ca5a66d5f>

## 以熊猫为例

撰写人:[阿玛尔·哈斯尼](https://medium.com/u/d38873cbc5aa?source=post_page-----40d1ab7243c2--------------------------------) & [迪亚·赫米拉](https://medium.com/u/7f47bdb8b8c0?source=post_page-----40d1ab7243c2--------------------------------)

![](img/f6011cac2adc152b6aa606213ae793cc.png)

人工智能生成的图像(稳定扩散)

您可能知道，python 是一种解释型语言。这意味着 python 代码不是直接编译成机器代码，而是由另一个名为**解释器**(大多数情况下为 `cpython`)的程序实时解释。

这也是为什么与编译语言相比，python 有如此大的灵活性(动态类型，到处运行等等)的原因之一。然而，这也是 python 非常慢的原因。

# 缓慢 python 的解决方案

python 速度慢实际上有多种解决方案:

*   使用`cython`:python 的超集编程语言
*   使用 C/C++语言结合 `ctypes`、 `pybind11`或 `CFFI`编写 python 绑定
*   [用 C/C++扩展 Python](https://docs.python.org/3/extending/extending.html)
*   使用其他编译语言，如 [rust](https://github.com/PyO3/pyo3)

如您所见，所有这些方法都需要使用 python 之外的另一种语言，并编译代码以便与 python 一起工作。

尽管这些都是有效的选择，但它们并不是让 python 变得更快的对初学者最友好的方法，并且不一定容易设置。

## Numba 和 JIT 编译

来认识一下 `numba`一个 python 包，它将使您的代码运行速度大大加快，而无需放弃 python 的便利性:

> Numba 是一个开源的 JIT 编译器，它将 Python 和 NumPy 代码的子集翻译成快速的机器代码。

`numba`使用**实时(JIT)编译**(这意味着它是在 python 代码执行期间的运行时编译的，而不是之前),在你问之前，不，你甚至不需要安装 C/C++编译器。你所需要做的就是用 pip/conda 安装它:

```
pip install numba
```

我们已经说得够多了，让我们试一个例子(来自 `numba`的文档)。我们希望使用蒙特卡罗模拟来计算 **π** 的估计值:

注意如何使用`numba`，只需要我们导入一个装饰器( `njit`)，它将完成所有的工作。

运行这段代码，对两个版本进行计时，显示出 **numba 比普通 python** 快 30 倍:

## 一些警告

如此呈现，听起来好得令人难以置信。但是它确实有它的缺点:

*   有一个开销，第一次运行一个`numba`修饰函数。这是因为`numba`会在函数第一次执行的时候试图找出参数的类型并编译它。所以第一次运行时会慢一点。
*   并不是所有的 python 代码都会用 `numba`编译，例如，如果你对同一个变量或者列表元素使用混合类型，你会得到一个错误。

# 立体公园里的熊猫

`numba`是专为 `numpy`设计的，对 numpy 数组非常友好。你知道 `numpy`上还建了什么吗？
你猜对了: `pandas`。当使用用户定义的函数或者甚至执行不同的数据帧操作时，这会导致疯狂的优化。

让我们看一些例子，从这个数据框架开始:

## 用户定义的函数

numba 的另一个方法是 `vectorize`，这使得创建 numpy 通用函数( [ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html) )变得轻而易举。

一个简单的例子是计算数据集中高度列的平方:

## 基本操作

另一个例子(使用 `njit`)是使用以下代码计算身体质量指数(身体质量指数):

你可以看到，即使是基本操作，numba 仍然比原始熊猫花费更少的时间(6.77 毫秒对 8.63 毫秒)

# 最后的想法

使用 `numba`是一种非常简单的方法，可以让你的代码运行得更快。有时，在你的代码被成功编译之前，可能需要一些尝试，但是一般来说，它可以开箱即用。

如果你对让熊猫跑得更快的更多方法感兴趣，看看我们关于 `eval` & `query`的文章:

[](https://python.plainenglish.io/these-methods-will-change-how-you-use-pandas-921e4669271f)  

谢谢你坚持到现在。注意安全，在接下来的故事中再见😊！

# 更多文章阅读

[](/rapidly-explore-jupyter-notebooks-right-in-your-terminal-67598d2265c2)  [](/equivalents-between-pandas-and-pyspark-c8b5ba57dc1d)  [](/8-tips-to-write-cleaner-code-376f7232652c)  [](/how-to-easily-merge-multiple-jupyter-notebooks-into-one-e464a22d2dc4) 