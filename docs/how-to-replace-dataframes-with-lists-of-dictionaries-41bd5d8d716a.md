# 如何用字典列表替换数据框

> 原文：<https://towardsdatascience.com/how-to-replace-dataframes-with-lists-of-dictionaries-41bd5d8d716a>

![](img/5a81cabd7bb0f0fee1f2c1d145522c4b.png)

由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的 [Element5 数码](https://unsplash.com/@element5digital?utm_source=medium&utm_medium=referral)拍摄

Pandas DataFrames 是一种非常方便的读取、编辑和可视化小表格数据的方式。但是它们也有一些主要的缺点:

*   它们不能很好地处理 Python 原生数据类型
*   它们对嵌套数据不方便
*   有些操作可能会非常慢
*   它们占据了很多空间

当从 API 获取数据时，API 可能会返回一个 JSON 字符串。虽然有一种方法可以将 JSON 字符串转换成 DataFrame，但是一个好的旧字典列表会更好。

在本文中，我将比较数据帧和字典列表，并编写一个自定义类来重新创建字典列表中数据帧的主要功能。

# 记忆

假设你有一个卖水果的商店，想存储你的订单。对于这个例子，我们将生成一个 1000 个水果订单的随机列表。

现在，让我们比较一下这个列表和包含这个数据的熊猫数据帧之间的大小差异。我们可以使用 **sys** 库来这样做。

这给了我们以下结果:

```
List: 9032 bytes, DataFrame size : 111584 bytes
```

所以就内存而言，链表比数据帧表现得好得多。

# 打印对象

打印数据帧的头部会给我们带来很好的可读性(如果你在笔记本上阅读的话会更好):

```
id                                             basket
0   0  [{'fruit': 'raspberry', 'price': 68, 'weight':...
1   1  [{'fruit': 'peach', 'price': 5, 'weight': 107,...
2   2  [{'fruit': 'banana', 'price': 99, 'weight': 20...
3   3  [{'fruit': 'peach', 'price': 67, 'weight': 158...
4   4  [{'fruit': 'pear', 'price': 1, 'weight': 90, '...
```

然而，有了字典列表，结果就不那么美好了。

```
[{'id': 0, 'basket': [{'fruit': 'apple', 'price': 89, 'weight': 68, 'color': 'blue'}, {'fruit': 'pear', 'price': 94, 'weight': 89, 'color': 'yellow'}]}, {'id': 1, 'basket': [{'fruit': 'strawberry', 'price': 26, 'weight': 86, 'color': 'red'}, {'fruit': 'banana', 'price': 76, 'weight': 244, 'color': 'pink'}, {'fruit': 'pear', 'price': 90, 'weight': 123, 'color': 'pink'}, {'fruit': 'peach', 'price': 83, 'weight': 65, 'color': 'pink'}, {'fruit': 'banana', 'price': 83, 'weight': 229, 'color': 'yellow'}]}, {'id': 2, 'basket': [{'fruit': 'raspberry', 'price': 8, 'weight': 258, 'color': 'orange'}, {'fruit': 'banana', 'price': 86, 'weight': 31, 'color': 'green'}, {'fruit': 'apple', 'price': 39, 'weight': 208, 'color': 'green'}, {'fruit': 'peach', 'price': 73, 'weight': 116, 'color': 'orange'}, {'fruit': 'raspberry', 'price': 82, 'weight': 57, 'color': 'purple'}, {'fruit': 'strawberry', 'price': 72, 'weight': 31, 'color': 'pink'}]}, {'id': 3, 'basket': [{'fruit': 'strawberry', 'price': 47, 'weight': 99, 'color': 'orange'}, {'fruit': 'pear', 'price': 40, 'weight': 38, 'color': 'pink'}, {'fruit': 'pear', 'price': 19, 'weight': 182, 'color': 'orange'}, {'fruit': 'watermelon', 'price': 27, 'weight': 249, 'color': 'yellow'}, {'fruit': 'watermelon', 'price': 93, 'weight': 202, 'color': 'red'}, {'fruit': 'apple', 'price': 65, 'weight': 112, 'color': 'purple'}, {'fruit': 'banana', 'price': 24, 'weight': 150, 'color': 'pink'}]}, {'id': 4, 'basket': [{'fruit': 'banana', 'price': 19, 'weight': 271, 'color': 'purple'}]}]
```

别担心！我们可以从 list 类继承并重写格式化打印列表的神奇方法。

我们所要做的就是使用 **json.dumps** 函数来获得一个“经典的”json 视图。我们也可以尝试每行打印一行，但这不允许我们看到完整的数据。

```
[
 {
  "id": 0,
  "basket": [
   {
    "fruit": "peach",
    "price": 89,
    "weight": 121,
    "color": "red"
   },
   {
    "fruit": "peach",
    "price": 17,
    "weight": 163,
    "color": "red"
   },
   {
    "fruit": "watermelon",
    "price": 19,
    "weight": 36,
    "color": "yellow"
   },
   {
    "fruit": "watermelon",
    "price": 88,
    "weight": 39,
    "color": "blue"
   },
   {
    "fruit": "banana",
    "price": 14,
    "weight": 131,
    "color": "purple"
   },
   {
    "fruit": "peach",
    "price": 72,
    "weight": 94,
    "color": "yellow"
   },
   {
    "fruit": "raspberry",
    "price": 74,
    "weight": 285,
    "color": "pink"
   },
   {
    "fruit": "strawberry",
    "price": 88,
    "weight": 203,
    "color": "orange"
   }
  ]
 },
 {
  "id": 1,
  "basket": [
   {
    "fruit": "pear",
    "price": 77,
    "weight": 136,
    "color": "blue"
   },
   {
    "fruit": "peach",
    "price": 7,
    "weight": 98,
    "color": "green"
   },
   {
    "fruit": "peach",
    "price": 94,
    "weight": 292,
    "color": "green"
   },
   {
    "fruit": "raspberry",
    "price": 23,
    "weight": 69,
    "color": "pink"
   },
   {
    "fruit": "watermelon",
    "price": 35,
    "weight": 120,
    "color": "pink"
   }
  ]
 },
 {
  "id": 2,
  "basket": [
   {
    "fruit": "strawberry",
    "price": 52,
    "weight": 36,
    "color": "blue"
   },
   {
    "fruit": "pear",
    "price": 42,
    "weight": 241,
    "color": "purple"
   },
   {
    "fruit": "pear",
    "price": 66,
    "weight": 165,
    "color": "orange"
   }
  ]
 },
 {
  "id": 3,
  "basket": [
   {
    "fruit": "peach",
    "price": 88,
    "weight": 219,
    "color": "blue"
   },
   {
    "fruit": "raspberry",
    "price": 80,
    "weight": 87,
    "color": "green"
   },
   {
    "fruit": "apple",
    "price": 42,
    "weight": 247,
    "color": "purple"
   },
   {
    "fruit": "banana",
    "price": 97,
    "weight": 154,
    "color": "orange"
   },
   {
    "fruit": "banana",
    "price": 37,
    "weight": 177,
    "color": "green"
   }
  ]
 },
 {
  "id": 4,
  "basket": [
   {
    "fruit": "strawberry",
    "price": 48,
    "weight": 64,
    "color": "green"
   }
  ]
 }
]
```

# 访问值

## 现有方式

假设我们想要访问订单 5 和订单 6 的篮子。

对于数据帧，这非常简单，我们只需使用**。锁定**方法。

但是对于列表，你必须使用列表理解:

## 新方法

这有点烦人，它返回一个列表，而不是我们刚刚创建的类。因此，就像我们对 **__str__** 方法所做的一样，我们可以重写 **__getitem__** 魔法方法，它允许我们使用索引和切片来访问元素。

要对行进行过滤，我们只需使用切片父方法。但是要过滤列，我们必须更有创造性，使用字典理解来过滤关键字。

## 比较性能

现在让我们看看哪种方法更快。

```
List : 0.025424789999999975 seconds for 10000 iterations
DataFrame : 1.14112228 seconds for 10000 iterations
```

实际上，用我们的字典列表访问一个值要快得多！

# 设置值

假设我们想要设置值 **id_store** 。目前，我们只有一家商店，所以我们将把 **id_store** 设置为 1。

## 现有方式

有了数据框架，这是非常方便的。

对于列表，我们可以这样做:

虽然它不是一段很长的代码，但它仍然是两行而不是两行。

## 新方法

让我们定义另一个神奇的方法，使用括号为每个字典设置一个值。

我们现在可以非常容易地将一个键分配给一个值或一组值:

## **对比表演**

我们的新方法能像熊猫的数据帧一样好吗？让我们再次比较 timeit 模块的性能。

```
List : 0.852329793 seconds for 10000 iterations
DataFrame : 0.996496461 seconds for 10000 iterations
```

这一次，列表和数据帧的性能非常相似。

# 映射列

我们想计算每个订单的总购物篮价格。我们如何做到这一点？

## 现有方式

数据帧有一个映射功能，对于这种类型的计算非常方便。

对于一个列表，我们再次需要迭代:

**新方法**

让我们为字典列表创建一个映射函数，让它像熊猫一样简单！这其实很简单，因为我们可以在不同的函数中处理赋值和返回值( **map** 和 **setitem** )。

现在，它变得与 DataFrame 映射方法相同。

## 比较性能

又到了衡量业绩的时候了！

```
List : 11.182705553 seconds for 10000 iterations
DataFrame : 16.0493823 seconds for 10000 iterations
```

太棒了，我们的榜单再次成为冠军！

# 后续步骤

虽然 Pandas 在快速浏览 CSV 文件方面非常出色，但我肯定会更频繁地使用这个新的 DictList 类来对大文件或嵌套文件进行基本转换。

我可能会把这变成一个项目，并创建一个具有更多功能的库。

你会使用这个图书馆吗？要让字典列表成为你最喜欢的结构，你需要什么样的特性？

# 资源

下面是一些链接，可以探索文章中讨论的一些概念:

*   [熊猫文档](https://pandas.pydata.org/docs/user_guide/index.html)
*   [系统文档](https://docs.python.org/3/library/sys.html)
*   [Tech with Tim 魔术方法视频教程](https://www.youtube.com/watch?v=z11P9sojHuM)