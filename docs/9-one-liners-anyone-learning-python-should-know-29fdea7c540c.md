# 任何学习 Python 的人都应该知道的 9 个一行程序

> 原文：<https://towardsdatascience.com/9-one-liners-anyone-learning-python-should-know-29fdea7c540c>

## 使用这些一行程序使您的 Python 代码简洁

![](img/e8fa25fd854006af3f056cc2d01c847d.png)

图片来自 Shutterstock，授权给 Frank Andrade

当我们开始学习 Python 时，我们通常会编写完成工作的代码，而不会关注我们代码的可读性以及它有多简洁和高效。

这很好，但是有一些方法可以在不忽略可读性的情况下使我们的 Python 代码更短。这就是一行程序的全部内容(如果您正确使用它们的话)。

这里是任何学习 Python 的人都应该知道的 9 个一行程序。

*如果你不想看，可以看我的 YouTube 视频。*

# 1.If — Else 语句

if-else 语句是我们在 Python 中学习的第一批语句之一。它用于执行给定条件的真和假部分。

我们经常使用这种说法，但是你知道它可以简化成一行代码吗？在这种情况下，if 和 else 语句将位于同一行。

```
age = 18valid = "You're an adult"
invalid = "You're NOT an adult"**print**(valid) **if** age >= 18 **else** **print**(invalid)
```

# 2.基于现有列表创建列表

列表是存储数据的一种常见方式，但是您知道吗？您可以使用一行代码基于现有列表创建一个新列表。

确实如此！它被称为 list comprehension，它提供了一个简短的语法来基于现有列表的值创建一个列表。列表理解比制作列表的函数和循环更紧凑。

下面是语法，我们将使用。

```
[expression **for** item **in** list]
```

这里有一个例子:

```
words = ['united states', 'brazil', 'united kingdom']

capitalized = [word.title() **for** word **in** words]>>> capitalized
['United States', 'Brazil', 'United Kingdom']
```

看起来好多了，不是吗？记住，我们应该让代码对用户友好，所以不要在一行中写很长的理解列表。

# 3.词典理解

与列表理解类似，Python 中也有字典理解。想法是一样的。Dictionary comprehension 提供了一个简短的语法，可以在一行代码中创建一个字典。

下面是语法，我们将使用:

```
{key: value **for** key, value **in** iterable}
```

这里有一个例子:

```
dict_numbers = {x:x*x **for** x **in** range(1,6) }>>> dict_numbers
{1: 1, 2: 4, 3: 9, 4: 16, 5:25}
```

# 4.加入词典

加入字典有不同的方法。可以使用`update()`方法、`merge()`运算符，甚至字典理解。

也就是说，在 Python 中有一种更简单的连接字典的方法。这是通过使用解包操作符`**`实现的。我们只需要在我们想要组合的每个字典前面添加`**`，并使用一个额外的字典来存储输出。

```
dict_1 = {'a': 1, 'b': 2}
dict_2 = {'c': 3, 'd': 4}merged_dict = {******dict_1, ******dict_2}>>> merged_dict
{'a': 1, 'b': 2, 'c': 3, 'd': 4}
```

在我们将`**`操作符应用到字典之后，两者都将扩展它们的内容并组合起来创建一个新字典。

# 5.删除列表中的重复项

有时我们需要确保列表中没有任何重复的值。虽然没有一种方法可以轻松地处理它，但是您可以使用集合来消除重复。

集合是项目的无序集合，其中每个元素都是唯一的。这意味着，如果我们把我们的列表变成一个集合，我们可以删除重复的。然后我们只需要再次将集合转换成列表。

让我们看一个基本的例子来掌握它的窍门。

```
numbers = [1,1,1,2,2,3,4,5,6,7,7,8,9,9,9]

>>> **list**(**set**(numbers))
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

# 6.用 Python 在一行中分配多个变量

每当你需要给多个变量赋值时，不用一行一行的赋值，你可以在 Python 中一行的赋值(甚至是不同类型的变量)。

```
a, b, c = 1, "abc",  True>>> a
1>>> b
'abc'>>> c
True
```

更紧凑，不是吗？但是要小心！你分配的变量越多，给它们分配错误值的机会就越大。

# 7.从列表中筛选值

假设我们想从列表中过滤一些值。您可以使用许多方法来做到这一点，但是一个简单的方法是使用`filter()`函数。

下面是`filter`函数的语法:

```
**filter**(function, iterable)
```

如果您在`filter`函数中添加一个`lambda`函数，事情会变得更好！

让我们通过从列表中过滤偶数来掌握它的窍门。

```
my_list = [10, 11, 12, 13, 14, 15]>>> **list**(**filter**(**lambda** x: x%2 == 0, my_list ))
[10, 12, 14]
```

# 8.按关键字对字典排序

对字典进行排序不像对列表进行排序那么简单——我们不能像对列表那样使用`sort()` 或`sorted()`对字典进行排序。

好消息是，我们可以将字典理解与`sorted()`函数结合起来，按关键字对字典进行排序。

让我们看一看。在下面的例子中，我们将按产品名称对字典进行排序。

```
product_prices = {'Z': 9.99, 'Y': 9.99, 'X': 9.99}>>{key:product_prices[key] **for** key **in** **sorted**(product_prices.keys())}
{'X': 9.99, 'Y': 9.99, 'Z': 9.99}
```

# 9.按值对字典排序

类似于按键排序字典，我们需要使用`sorted()`函数和 list comprehension 来按值排序字典。但是，除此之外，我们还需要添加一个`lambda`函数。

首先，让我们看看`sorted()`函数的所有参数。

```
sorted(iterable, key=None, reverse=False)
```

为了按值对字典进行排序，我们需要使用*键*参数。此参数接受作为排序比较的键的函数。这里我们可以使用一个`lambda`函数来简化事情。

假设我们有一个包含人口值的字典，我们想按值对它进行排序。

```
population = {'USA':329.5, 'Brazil': 212.6, 'UK': 67.2}

>>> **sorted**(population.items(), **key**=**lambda** x:x[1])
[('UK', 67.2), ('Brazil', 212.6), ('USA', 329.5)]
```

现在唯一剩下的就是加字典理解了。

```
population = {'USA':329.5, 'Brazil': 212.6, 'UK': 67.2}

>>> {k:v **for** k, v **in** **sorted**(population.items(), **key**=**lambda** x:x[1])}
{'UK': 67.2, 'Brazil': 212.6, 'USA': 329.5}
```

用 Python 学习数据科学？ [**通过加入我的 10k+人电子邮件列表，获取我的免费 Python for Data Science 备忘单。**](https://frankandrade.ck.page/26b76e9130)

如果你喜欢阅读这样的故事，并想支持我成为一名作家，可以考虑报名成为一名媒体成员。每月 5 美元，让您可以无限制地访问数以千计的 Python 指南和数据科学文章。如果你使用[我的链接](https://frank-andrade.medium.com/membership)注册，我会赚一小笔佣金，不需要你额外付费。

[](https://frank-andrade.medium.com/membership) [## 通过我的推荐链接加入媒体——弗兰克·安德拉德

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

frank-andrade.medium.com](https://frank-andrade.medium.com/membership)