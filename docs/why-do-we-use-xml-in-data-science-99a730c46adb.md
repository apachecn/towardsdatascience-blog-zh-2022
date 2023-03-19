# 为什么我们在数据科学中使用 XML？

> 原文：<https://towardsdatascience.com/why-do-we-use-xml-in-data-science-99a730c46adb>

## 学习 XML 的基础知识以及如何在 Python 中处理它

![](img/8cfd62ba6de7c4be3f1be06619f6a0f4.png)

瓦列里·塞索耶夫在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

XML 代表可扩展标记语言，现在被用作基于文本的数据格式，用于交换结构化的[数据](https://databasecamp.de/en/data)。最初开发它是为了取代 HTML，因为它在数据技术方面已经达到了极限。

# XML 文件是如何构造的？

所谓的标记语言，如 HTML 或 XML，使用标记来构造文本信息。在 HTML 中有一组固定的标记，它们被定义在尖括号(<>)中。例如，这些包括标题(

# 、

## 、

### 、…)或段落(

)，它们用于构建网站的结构:

```
<!DOCTYPE html> 
<html> 
<head> 
<title>Title Text</title> 
</head> <body>  
<h1>Main Header</h1> 
<p>The first paragraph</p>  
</body> </html>
```

例如，这是一个非常简单的网站的结构。标记定义了页面的结构，目前页面只包含一个标题和一个简短的段落。

XML 使用相同的结构，不同之处在于标记的数量不受限制，它们的命名可以自由选择。这使得模拟许多类型的数据结构变得相对容易。

可扩展标记语言给出的唯一规则是标记必须总是以开始标签<markup_1>开始，以结束标签</markup_1>结束。唯一的例外是带有此表单的标签<markup_1>。</markup_1>

为了显示嵌套信息，还可以在一个开放标签中定义几个新标签。这里有一个例子:

附加功能是所谓的参数。这些可以为每个标记定义，并且必须始终包含名称和值。该值必须用引号定义，即使它是一个数字。

在我们的示例中，当我们想要描述集合中的多辆汽车时，参数的引入非常有用:

# 可扩展标记语言的优点和缺点

使用可扩展标记语言文件具有以下优点:

*   广泛分布，因此与现有应用程序高度兼容
*   文件的高度安全性
*   由于文本文件的可读性，信息易于恢复
*   易于人机理解
*   简单的结构和布局，使它能被许多用户迅速理解
*   “方言”形式的扩展性

这个长长的优点列表唯一真正的缺点来自于可扩展标记语言使用的文本格式。文本信息只能用相对较多的内存来存储，因此会导致处理性能降低。二进制文件格式，如 BSON，对于相同的信息需要少得多的存储空间，但是不可读，因为信息是以 0 和 1 存储的。

# 哪些应用程序使用可扩展标记语言？

由于可扩展标记语言的基于文本的存储，这种格式相对容易阅读和理解。这就是它被广泛应用的原因。最常见的用例之一是数据交换，即在应用程序中导入和导出数据。这就是它对数据科学家如此有价值的原因。此外，导入 Python 并像处理 Python 字典一样处理它也非常容易。

此外，可扩展标记语言的一般用途很少，因为大多数用例已经创建了特定于其应用程序的 XML 变体。例如，有一种数学标记语言(MathML)，它是 XML 的一种方言，用于正确表示数学等式和术语。

# 如何用 Python 编辑 XML 文件？

在 [Python](https://databasecamp.de/en/python-coding) 中有几种打开 XML 文件的方式和模块。我们将尝试使用一个由我们之前的例子组成的字符串。我们可以试着保留原来的结构:

另一方面，我们也可以尝试将可扩展标记语言的结构转换成一个 [Python 字典](https://databasecamp.de/en/python-coding/python-dictionarys)。对于许多经常使用 Python 的开发人员来说，这要容易得多:

# 这是你应该带走的东西

*   XML 代表可扩展标记语言，现在被用作交换结构化数据的基于文本的数据格式。
*   由于可扩展标记语言的基于文本的存储，这种格式相对容易阅读和理解。
*   此外，这种格式的优势在于可以适应不同方言的不同用例。

*如果你喜欢我的作品，请在这里订阅*[](https://medium.com/subscribe/@niklas_lang)**或者查看我的网站* [*数据大本营*](http://www.databasecamp.de/en/homepage) *！还有，medium 允许你每月免费阅读* ***3 篇*** *。如果你希望有****无限制的*** *访问我的文章和数以千计的精彩文章，不要犹豫，点击我的推荐链接:*[【https://medium.com/@niklas_lang/membership】](https://medium.com/@niklas_lang/membership)每月花$***5****获得会员资格**

*[](/getting-started-with-json-4c94bb4df113)  [](/what-is-an-api-easily-explained-d153a736a55f)  [](/the-difference-between-correlation-and-causation-51d44c102789) *