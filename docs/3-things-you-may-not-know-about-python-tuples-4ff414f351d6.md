# 关于 Python 元组你可能不知道的 3 件事

> 原文：<https://towardsdatascience.com/3-things-you-may-not-know-about-python-tuples-4ff414f351d6>

## 更好地使用元组

![](img/5fc8022d132dbf1f7dc003bd355bb756.png)

奥斯卡·尼尔森在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄的照片

元组是 Python 中一种重要的内置数据类型。像列表一样，我们经常使用元组将多个对象保存为一个数据容器。然而，使它们不同于列表的是它们的不变性——不可变的数据序列。下面的代码片段向您展示了元组的一些常见用法。

```
response = (404, "Can't access website")response_code = response[0]
response_data = response[1]**assert** response_code == 404
**assert** response_data == "Can't access website"
```

上面的用法对你来说应该很直观。我们使用一对括号创建一个 tuple 对象，将元素括起来。当我们需要使用单个项目时，我们可以使用索引。

除了这些基本用法，还有其他一些不太为人所知的用法。让我们在本文中回顾一下它们。

## 1.创建仅包含一个项目的元组

我们提到，我们使用一对括号来创建一个元组对象。通常，一个 tuple 对象包含两个或更多项，我们使用逗号来分隔这些项。如果我们想创建一个单项式元组，应该怎么做？难道是`(item)`？让我们试试:

```
>>> math_score = (95)
>>> math_score[0]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'int' object is not subscriptable
>>> type(math_score)
<class 'int'>
```

正如你所看到的，`(95)`并没有像你们中的一些人想的那样创建一个`tuple`对象。相反，它创建了一个整数。解决方法是您需要在该项后面附加一个逗号:

```
>>> math_score = (95,)
>>> math_score[0]
95
>>> type(math_score)
<class 'tuple'>
```

您可能想知道何时需要使用单项式元组。这里有一个关于熊猫的例子。当我们使用 pandas 时，我们经常使用`apply`从现有列创建新数据。下面给你展示一个使用`apply`的简单案例。

```
>>> import pandas as pd
>>> df = pd.DataFrame({"a": range(1, 4), "b": range(4, 7)})
>>> df
   a  b
0  1  4
1  2  5
2  3  6
>>> def creating_exponents(x):
...     return pow(x["b"], 2)
... 
>>> df.apply(creating_exponents, axis=1)
0    16
1    25
2    36
dtype: int64
```

在大多数情况下，我们在`apply`中使用的映射函数只接受一个参数，即 DataFrame 的行(当`axis`参数为 1 时，如果`axis=0`为列)。我们可能有一个接受附加参数的映射函数，这样映射函数就更加灵活，并且可以通过不同地设置第二个参数来用于不同的 DataFrame 对象。在这种情况下，我们可以使用一个一项元组来传递第二个参数，如下所示。

```
>>> def creating_exponents(x, exp):
...     return pow(x["b"], exp)
... 
>>> df.apply(creating_exponents, axis=1, args=(3,))
0     64
1    125
2    216
dtype: int64
>>> df.apply(creating_exponents, axis=1, args=(4,))
0     256
1     625
2    1296
dtype: int64
```

如您所见，我们对`args`参数使用了一个单项式元组，它与行数据一起传递给映射函数。

## 2.使用下划线和*解包元组

虽然我们可以通过使用索引来访问一个元组的各个项，但更常见的是使用解包技术，如下所示:

```
response = (404, "Can't access website")
response_code, response_data = response
```

对于解包，您定义了一些变量，这些变量的数量与元组的计数相匹配。元组的每个项被分配给各自的变量。

如果您不需要使用所有创建的变量，建议您使用下划线来表示它们没有被使用。例如，我们可能只对使用响应数据感兴趣，而对代码不感兴趣，我们可以这样做:

```
_, response_data = response
```

这样，您就告诉了代码的读者，我们只对访问 tuple 对象的第二项感兴趣。

当一个元组对象中有多个项目时，您可能希望访问多个连续的项目。您可以使用带星号的表达式:

```
scores = (98, 95, 95, 92, 91)algebra, *others, zoology = scores**assert** others == [95, 95, 92]
```

如上所示，我们有一个 tuple 对象来保存按字母顺序排序的课程分数，我们知道第一门课程是代数，最后一门是动物学。在上面的例子中，我们得到了中间的三个分数。

这个特性(tuple 解包中的星号表达式)在您不知道到底有多少到表达式时特别有用。所有的学生都学代数和动物学，但他们在其他课程上可能会有所不同。我们可以使用带星号的表达式来获得其他课程的分数，而不需要知道课程的数量。

```
def extract_scores(scores):
    algebra, *others, zoology = scores
    return {"algebra": algebra, "zoology": zoology, "others": others}scores1 = (98, 95, 92, 93, 91)
scores2 = (97, 95, 95, 90)
scores3 = (90, 90)>>> extract_scores(scores1)
{'algebra': 98, 'zoology': 91, 'others': [95, 92, 93]}
>>> extract_scores(scores2)
{'algebra': 97, 'zoology': 90, 'others': [95, 95]}
>>> extract_scores(scores3)
{'algebra': 90, 'zoology': 90, 'others': []}
```

正如您所看到的，这个带星号的表达式可以处理中间任意数量的元素，包括零个元素(在`scores3`的情况下)。

## 3.命名元组

您可以通过使用索引或解包元组来访问元组的各个项，以将这些项分配给单独的变量。有时候，这样做可能会很乏味。请参见以下示例:

```
location1 = (27.2, 7.5)
location2 = (30.1, 8.4)
location3 = (29.9, 7.7)latitude1 = location1[0]
longitude2 = location2[1]
latitude3, longitude3 = location3
```

我们有三个位置，显示了它们各自的坐标。当我们访问这些坐标的单个项目时，代码看起来不太清晰。虽然我们可以使用定制类来实现坐标的数据模型，但是定制类对于这个简单的数据模型来说可能是“沉重”的。相反，我们可以使用命名元组作为轻量级数据模型:

```
from collections import namedtupleLocation = namedtuple("Location", ["latitude", "longitude"])location1 = Location(27.2, 7.5)
location2 = Location(30.1, 8.4)
location3 = Location(29.9, 7.7)latitude1 = location1.latitude
longitude2 = location2.longitude
location3.latitude, location3.longitude
```

如上所示，我们调用`namedtuple`通过指定类名及其属性来创建一个命名元组类。现在，我们可以调用该类的构造函数来创建命名元组类的实例。对于这些实例，我们可以使用点符号来访问它们的属性，这是常规元组对象所不具备的特性。

如前所述，命名元组是一个简单而方便的类，它们也在熊猫中使用。当我们迭代 DataFrame 对象的行时，我们可以使用`itertuples`方法，如下所示:

```
>>> df
   a  b
0  1  4
1  2  5
2  3  6>>> for row in df.itertuples():
...     print(row)
... 
Pandas(Index=0, a=1, b=4)
Pandas(Index=1, a=2, b=5)
Pandas(Index=2, a=3, b=6)
```

每一行都是一个有三个属性的命名元组:索引和两列的值。有了命名元组，您可以方便地访问列的数据，比如`row.a`和`row.b`。

## 结论

在本文中，我们回顾了 Python 中元组的三个特性。我们还使用了一些与使用 pandas 进行数据处理相关的例子。希望这篇文章对你有用。

感谢阅读这篇文章。通过[注册我的简讯](https://medium.com/subscribe/@yong.cui01)保持联系。还不是中等会员？通过[使用我的会员链接](https://medium.com/@yong.cui01/membership)支持我的写作(对你没有额外的费用，但是你的一部分会费作为奖励由 Medium 重新分配给我)。