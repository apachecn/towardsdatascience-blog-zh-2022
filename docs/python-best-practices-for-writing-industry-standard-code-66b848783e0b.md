# 编写行业标准 Python 代码的最佳实践

> 原文：<https://towardsdatascience.com/python-best-practices-for-writing-industry-standard-code-66b848783e0b>

## 停止在 Python 代码中犯这些错误

![](img/20771f9efb56427358c4958d17697a78.png)

图片来源:Unsplash

# 为什么是 Python 编码最佳实践？

获得良好分析结果的旅程充满了探索、实验和多种方法的测试。在这个过程中，我们编写了许多代码，我们和其他人在团队工作时会参考这些代码。如果您的代码不够干净，那么理解、重用或修改您的代码对其他人(您也一样)来说将是一种痛苦。

> *如果你写了一个整洁干净的代码，你会感谢自己，别人也会感谢你。*

这篇博客将向您介绍编写整洁干净代码的 Python 编码最佳实践。

python 编程有 PEP8 等行业标准。PEP 代表 Python 增强提案。让我们详细看看这些标准:

# Python 编码最佳实践

## 1.变量命名约定

设置直观的变量名对任何人理解我们试图做什么都非常有帮助。

通常，我们使用单个字符创建变量，如 I、j、k——这些不是自明的变量，应该避免使用。

> **注意**:切勿使用 l、o 或 I 单字母名称，因为根据字体不同，这些名称可能会被误认为“1”和“0”。
> 
> `o = 2 # This may look like you're trying to reassign 2 to zero`
> 
> ***不好练***

我们可以使用下划线“_”来创建多单词变量名，如下所示:

> ***好的做法***

> 注意:停止使用 df 来定义数据框，而是使用更直观的名称。

## **2。函数中的注释**

注释是使我们的代码可读和易于理解的最重要的方面之一。

有两种类型的注释:

1.  行内注释:与代码在同一行的注释。# '用于设置行内注释。

```
a = 10 # Single line comment 
```

2.块注释:多行注释称为块注释。

```
""" Example of Multi-line comment 
    Line 2 of a multiline comment
"""
```

> ***不好的做法***

> ***好做法***

## **3。缩进**

缩进是一个强大的特性，可以使代码易于阅读和理解。每个缩进级别使用 4 个空格/制表符来设置缩进。

一些语法要求代码默认具有缩进。例如，函数定义体中的内容:

```
**def** function**(**var_one**):**
    print**(**var_one**)**
```

我们可以使用缩进来创建清晰易读的多行函数成分(如自变量、变量等)。).

> ***不好的做法***

> ***好的做法***

> 缩进用制表符还是空格？*空格是首选的缩进方法。制表符应该只用于保持与已经用制表符缩进的代码一致。Python 不允许混合使用制表符和空格来缩进。*

## **4。最大线路长度**

根据 PEP8，最佳实践是将所有行限制为最多 79 个字符。一般来说，80-100 个字符也可以。

对于圆括号或方括号内的代码，Python 默认假设行连续，而对于其他情况，我们可以使用“\”。

> ***不好的做法***

> ***好的做法***

## 5.单独行上的导入

建议在单独的行上导入新包。

> ***不好练***

```
**import** sys**,** os, pandas
```

> ***好的做法***

```
**import** os
**import** sys
**import** pandas
```

## 6.间距建议

间距还会影响用户的整体可读性体验。以下是关于间距的最佳实践:

始终用一个空格将以下二元运算符括起来:赋值(`=`)、增广赋值(`+=`、`-=,`等。)、比较(`==`、`<`、`>`、`!=`、`<>`、`<=`、`>=`、`in`、`not in`、`is`、`is not`)、布尔(`and`、`or`、`not`)。

> ***不好的做法***

```
i**=**i**+1**
submitted **+=1**
x **=** x ***** **2** **-** **1**
hypot2 **=** x ***** x **+** y ***** y
c **=** **(**a **+** b**)** ***** **(**a **-** b**)**
```

> ***好做法***

```
i **=** i **+** **1**
submitted **+=** **1**
x **=** x***2** **-** **1**
hypot2 **=** x*****x **+** y*****y
c **=** **(**a**+**b**)** ***** **(**a**-**b**)**
```

参考资料:

1.  [https://gist.github.com/sloria/7001839](https://gist.github.com/sloria/7001839)
2.  [https://sphinxcontrib-Napoleon . readthe docs . io/en/latest/example _ Google . html # example-Google](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google)

**谢谢！**

你可能也会喜欢下面的故事:

[](https://medium.com/codex/top-10-python-operations-that-every-aspiring-data-scientist-should-know-92b6f9a98ff9)  [](https://medium.com/codex/every-data-analysis-in-10-steps-960dc7e7f00b) 