# 用 Python 编写一个拼写检查程序

> 原文：<https://towardsdatascience.com/build-a-spelling-checker-program-in-python-a2c852330d70>

## 在这篇文章中，我们将探索如何使用 Python 来检查单词和句子的拼写

![](img/6bb2e12c640bb2051f56cf8688b65ad1.png)

[真诚媒体](https://unsplash.com/@sincerelymedia?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/spelling?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

**目录**

*   介绍
*   使用 Python 检查单词的拼写
*   使用 Python 检查句子的拼写
*   结论

# 介绍

在编程中，我们经常会用到大量的文本对象。

这些包括我们添加到项目存储库中的简单自述文件，或者用于发送电子邮件的[自动化](https://pyshark.com/sending-email-using-python/)，或者其他东西。

我们在这些对象中使用的文本容易出错、拼写错误等等。

使用 Python 我们可以快速有效地检查不同单词和句子的拼写。

使用本文中的模块，您还可以用 Python 构建一个[拼写纠正程序。](https://pyshark.com/spelling-corrector-program-in-python/)

要继续学习本教程，我们需要以下 Python 库:textblob。

如果您没有安装它，请打开“命令提示符”(在 Windows 上)并使用以下代码安装它:

```
pip install textblob
```

# 使用 Python 检查单词的拼写

在这一节中，我们将探索如何使用 Python 来检查单词的拼写。

## 步骤 1:导入所需的依赖项

**Word()** 是一个简单的单词表示，它有许多有用的方法，尤其是检查拼写。

## 第二步:定义一个单词来检查拼写

让我们使用一些有拼写错误的单词，如“appple ”,并对其进行拼写检查。

## 第三步:检查单词的拼写

您应该得到:

```
[('apple', 1.0)]
```

**。spellcheck()** 方法返回一个元组，其中第一个元素是正确的拼写建议，第二个元素是置信度。

在我们的例子中，拼写检查器 100%确信正确的拼写是“apple”。

**注:**

您可能会收到多个元组(正确拼写的多个建议)。

例如，如果您对单词“aple”运行相同的代码，您将得到:

```
[('able', 0.5140664961636828),
 ('pale', 0.4219948849104859),
 ('apple', 0.028132992327365727),
 ('ample', 0.023017902813299233),
 ('ape', 0.010230179028132993),
 ('ale', 0.0025575447570332483)]
```

因为上述拼写错误可能是许多拼写错误的单词的一部分，所以您可以获得多个正确拼写的选项，按信心排序。

## 使用 Python 检查单词拼写的程序

结合上述所有步骤并添加一些功能，我们可以使用 Python 创建一个检查单词拼写的程序:

使用示例单词“appple”运行此程序应返回:

```
Spelling of "appple" is not correct!
Correct spelling of "appple": "apple" (with 1.0 confidence).
```

# 使用 Python 检查句子的拼写

为了使用 Python 检查一个句子的拼写，我们将在上一节构建的程序的基础上进行构建。

不幸的是，我们不能将整个句子传递到 checked 中，这意味着我们将把句子分成单个单词，并执行拼写检查。

## 步骤 1:导入所需的依赖项

## 第二步:定义一个句子来检查拼写

让我们用一个有两个拼写错误的简单句子:“sentence”和“checkk”。

## 第三步:把句子分成单词

您应该得到:

```
['This', 'is', 'a', 'sentencee', 'to', 'checkk!']
```

## 第四步:将每个单词转换成小写

您应该得到:

```
['this', 'is', 'a', 'sentencee', 'to', 'checkk!']
```

我们转换成小写的原因是因为它会影响拼写检查的性能。

## 第五步:去掉标点符号

您应该得到:

```
['this', 'is', 'a', 'sentencee', 'to', 'checkk']
```

我们删除标点符号的原因是因为它会影响拼写检查的性能，因为它将标点符号视为单词的一部分。

## 第六步:检查句子中每个单词的拼写

使用我们[早先创建的](https://pyshark.com/spelling-checker-program-in-python/#program-to-check-spelling-of-a-word-using-python)函数 **check_spelling()** 。

您应该得到:

```
Spelling of "this" is correct!
Spelling of "is" is correct!
Spelling of "a" is correct!
Spelling of "sentencee" is not correct!
Correct spelling of "sentencee": "sentence" (with 0.7027027027027027 confidence).
Spelling of "to" is correct!
Spelling of "checkk" is not correct!
Correct spelling of "checkk": "check" (with 0.8636363636363636 confidence).
```

我们看到代码正确地识别了拼写错误的单词，并提供了正确拼写的建议以及置信度。

## 使用 Python 检查句子拼写的程序

结合以上所有步骤，使用我们之前[创建的](https://pyshark.com/spelling-checker-program-in-python/#program-to-check-spelling-of-a-word-using-python)函数 **check_spelling()** ，并添加一些功能，我们可以使用 Python 创建一个纠正单词拼写的程序:

用例句“这是一个要检查的句子！”应该返回:

```
Spelling of "this" is correct!
Spelling of "is" is correct!
Spelling of "a" is correct!
Spelling of "sentencee" is not correct!
Correct spelling of "sentencee": "sentence" (with 0.7027027027027027 confidence).
Spelling of "to" is correct!
Spelling of "checkk" is not correct!
Correct spelling of "checkk": "check" (with 0.8636363636363636 confidence).
```

# 结论

在本文中，我们探讨了如何使用 Python 检查单词和句子的拼写。

如果你有任何问题或对编辑有任何建议，请随时在下面留下评论，并查看我的更多 [Python 编程](https://pyshark.com/category/python-programming/)教程。

*原载于 2022 年 2 月 25 日 https://pyshark.com*<https://pyshark.com/spelling-checker-program-in-python/>**。**