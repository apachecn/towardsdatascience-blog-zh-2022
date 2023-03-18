# 如何在 Python 中执行 JSON 转换、序列化和比较

> 原文：<https://towardsdatascience.com/how-to-work-with-json-in-python-bdfaa3074ee4>

## 通过简单的例子学习基本的 JSON 操作

![](img/801e152ba866c066e826a2ece5ef864a.png)

[图片由 kreatikar 在 Pixabay 拍摄](https://pixabay.com/illustrations/web-design-user-interface-website-3411373/)

**JSON**(JavaScript Object Notation)是一种 ***text*** 格式，它与语言无关，通常用于不同应用程序之间的数据交换。一个很好的例子是，来自 API 的响应通常是 JSON 格式的，因此后端和前端可以自由地交换数据，而不需要知道彼此的技术细节。在这篇文章中，我们将介绍 Python 中 JSON 的常见用例，Python 是一种用于后端开发和数据工程/分析的流行语言。

## JSON 和字典

首先，我们应该知道 JSON 是一种字符串格式。因此，它不同于 Python 中的字典数据类型。JSON 字符串可以用任何现代编程语言解析成相应的数据。通常，JSON 字符串可以被解析成两种数据类型，即 object 和 array。对象是一组无序的键/值对，对应于 Python 中的字典数据类型，而数组是值的有序集合，对应于 Python 中的列表数据类型

## JSON 字符串和数据之间的转换

如上所述，在所有现代编程语言中，JSON 字符串可以被解析为对象或数组，反之亦然。在 Python 中，`json`库可以用于这种类型的转换。我们使用`loads`函数将 JSON 字符串转换成对象或数组，并使用`dumps`函数执行相反的转换。注意`loads`中的`s`和`dumps`代表 **s** tring，这意味着它们在 JSON 字符串上工作。如果没有指定`s`，那么这些函数将会处理 JSON 文件，这将在后面介绍。

下面这段代码演示了 JSON 字符串和对象/数组之间的常见转换。

有趣的是，当我们将数组转储回 JSON 时，结果与原始结果不同。如果你仔细检查，你会发现细微的差别。当我们不指定分隔符时，将在项目分隔符后面添加一个空格，默认情况下是逗号。我们可以指定一个自定义分隔符来使结果相同。请注意，我们需要指定项目分隔符和键分隔符，即使我们只想更改其中之一:

实际上，`separators`参数更常用于定制 JSON 对象的表示。我们可以使用不同的分隔符使转储的字符串更紧凑或更易读:

`indent`参数用于在每个键之前插入一些空格，以提高可读性。而`sort_keys`参数用于按字母顺序对键进行排序。

## 为无法序列化的值添加自定义序列化程序

在上面的例子中，目标字典(`dict_from_json`)的所有值都可以序列化。实际上，有些值是无法序列化的，尤其是`Decimal`和`date` / `datetime`类型:

在这种情况下，我们需要创建一个定制的序列化函数，并将其设置为`default`参数:

请注意，在自定义序列化程序中，我们在 f 字符串中使用`[!r](https://lynn-kwong.medium.com/special-python-string-formatting-in-logging-and-pint-unit-conversion-fddb51f3d03a)` [来显示值的表示，这对于调试来说非常方便。如果取消对其中一个`if/elif`条件的注释，并再次运行`json.dumps`命令，您将看到相应的错误:](https://lynn-kwong.medium.com/special-python-string-formatting-in-logging-and-pint-unit-conversion-fddb51f3d03a)

## 比较两个 JSONs 的区别

有时候我们需要比较两个 JSON 对象的区别。例如，我们可以检查和比较一些可以导出为 JSON 的表的模式，如果一些重要表的模式发生了变化，就会发出一些警报。

`[jsondiff](https://github.com/xlwings/jsondiff)` [库](https://github.com/xlwings/jsondiff)可用于比较 Python 中两个 JSON 对象之间的差异:

如果我们想要控制结果应该如何显示，我们可以使用`syntax`、`marshal`和`dump`参数来自定义结果。

我们可以使用`syntax`字段来指定值和动作应该如何显示。

我们可以使用`load`参数从 JSON 字符串中加载数据，并类似地使用`dump`参数将结果转储到一个 JSON 字符串中，该字符串可以直接写入一个文件，这将很快介绍。

## 读写 JSON

我们可以用`json.dump`函数将一个 JSON 字符串写到一个文件中。注意，函数名中没有`s`。带有`s` ( `json.dumps`)的那个是用来处理字符串的，不是文件。JSON 文件只是一个纯文本文件，默认扩展名是`.json`。让我们将由`jsondiff`返回的两个模式之间的差异写到一个名为`schema_diff.json`的文件中:

将创建一个名为`schema_diff.json`的文件，该文件将包含变量`result`中的 JSON 字符串。如果字典/列表的值包含不可序列化的数据，我们需要用序列化函数指定`default`参数，如本文开头所示。

最后，我们可以使用`json.load`函数从 JSON 文件中加载数据:

在这篇文章中，通过简单的例子介绍了 JSON 的基础知识以及如何在 Python 中使用它。我们已经学习了如何从字符串或文件中读写 JSON 对象。此外，我们现在知道如何为 JSON 对象编写定制的序列化程序，这些对象包含默认序列化程序无法序列化的数据。最后，我们可以使用`jsondiff`库来比较两个 JSON 对象之间的差异，这对于数据监控来说非常方便。

相关文章:

*   [如何使用 JSON 模式在 Python 中验证 JSON 文档](https://lynn-kwong.medium.com/how-to-use-json-schema-to-validate-json-documents-ae9d8d1db344)
*   [如何在 MySQL 中使用 JSON 数据](/how-to-work-with-json-data-in-mysql-11672e4da7e9)