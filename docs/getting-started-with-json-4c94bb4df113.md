# JSON 入门

> 原文：<https://towardsdatascience.com/getting-started-with-json-4c94bb4df113>

## 介绍用于数据交换的文件格式

![](img/3a602165c609b9ae19f86ad693e96bd0.png)

费伦茨·阿尔马西在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

JSON 是**J**ava**S**script**O**object**N**rotation 文件格式的缩写。它描述了存储数据的标准化数据格式。它是文本文件格式之一，因为数据是以人类可读的自然语言存储的。

# 这些文件是如何组织的？

简单地说，JSON 文件的结构是一个无序的键值对集合。对于 [Python](https://databasecamp.de/en/python-coding) 程序员来说，这种结构比 [Python 字典](https://databasecamp.de/en/python-coding/python-dictionarys)更好。

键“city”、“country”和“population”的顺序不是预定义的，所以与这里显示的顺序不同的 JSON 文件仍然与显示的文件相同。

在 JavaScript 中，对象只能有表示为字符串的键。由于 JSON 文件格式源自 JavaScript，因此只有字符串可以用作键。

一对中的值可以采用不同的数据类型。允许以下类型:

*   字符串，例如“城市”:“柏林”。
*   数字，例如“人口”:3645000
*   对象，例如另一个 JSON 对象
*   数组，例如“区”:[“Kreuzberg”、“Pankow”、“Reinickendorf”]
*   布尔值，例如“大写”:真
*   空

这种多种多样的数据类型使得 JSON 成为一种非常流行的文件格式。只有少数例外，它们不能作为值存储在 JavaScript 对象符号中。

在 JSON 中，不能存储任何函数。同时，日期格式不能以本机方式存储。但是，这不是主要问题，因为您可以将日期存储为字符串，然后在读取文件时将它们转换回日期。

# JavaScript 对象符号的优点和缺点

由于结构简单，这种文件的使用非常普遍。对数据结构的要求相对较低，并且文件对于许多用户来说是快速和容易理解的。

这种广泛使用也可以解释为，现在所有常见的编程语言都有自己的解析器，这使得 JSON 的使用更加容易。

这种文件格式的主要缺点是模糊的数字定义和缺乏解释不支持的数据类型的标准。JavaScript 对象符号不区分常见的数字格式，如整数或小数。这使得数字的解释依赖于实现。

如前所述，有些数据类型在默认情况下不受支持，比如日期。这可以通过转换为字符串来避免。然而，有不同的可能性和库，可用于这一点，它没有商定任何统一的标准。

# 用 JSON 可以实现哪些应用？

JavaScript 对象符号在许多不同的应用程序和编程语言中使用。NoSQL 和各种关系数据库都有连接器来存储这些文件类型。

此外，它还适用于以下用例:

*   **数据传输**:当通过 API 请求信息时，JavaScript 对象符号在很多情况下被用作响应格式。
*   **清晰的数据存储**:由于数据格式要求不高，灵活的数据结构可以轻松存储。
*   **临时数据存储**:JavaScript 对象符号也常用于在程序中短期存储信息。由于与其他编程语言的高度兼容性，这些文件也可以由不同的应用程序使用，而无需更多的麻烦。

# 如何用 Python 编辑 JSON 文件？

[Python](https://databasecamp.de/en/python-coding) 有自己的库来处理 JavaScript 对象符号。必须先用“打开”打开相应的文件。然后可以用“load”将存储的 JSON 对象转换成一个由 [Python 字典](https://databasecamp.de/python/python-dictionary)组成的数组。

从现在开始，您可以像使用 Python 字典一样使用数据。

# 这是你应该带走的东西

*   JSON 是**J**ava**S**script**O**object**N**旋转文件格式的缩写。
*   它描述了存储数据的标准化数据格式。
*   该文件格式用于各种应用程序，因为它非常容易阅读和理解。
*   此外，所有常见的编程语言都提供了用 JavaScript 对象符号简化工作的模块。
*   模糊的数字定义是使用这种文件格式的最大缺点。

*如果你喜欢我的作品，请在这里订阅*<https://medium.com/subscribe/@niklas_lang>**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，medium 允许你每月免费阅读* ***3 篇*** *。如果你希望有****无限制的*** *访问我的文章和数以千计的精彩文章，不要犹豫，点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格**

*</what-is-odbc-c27f95164dec>  </redis-in-memory-data-store-easily-explained-3b92457be424>  </understanding-mapreduce-with-the-help-of-harry-potter-5b0ae89cc88> *