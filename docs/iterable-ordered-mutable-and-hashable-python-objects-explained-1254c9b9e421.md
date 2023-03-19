# 解释了可迭代、有序、可变和可散列的 Python 对象

> 原文：<https://towardsdatascience.com/iterable-ordered-mutable-and-hashable-python-objects-explained-1254c9b9e421>

## 讨论这些术语的真正含义和暗示，它们的主要细微差别，以及一些有用的变通方法

![](img/bbcb202df71080a5182b41317ce9766b.png)

来自 [Pixabay](https://pixabay.com/photos/archive-boxes-shelf-folders-1850170/)

可迭代、有序、可变和可散列(以及它们的反义词)是描述 Python 对象或数据类型的特征。尽管经常使用，这些术语经常被混淆或误解。在本文中，我们将讨论每个属性的真正含义和暗示，它们与什么数据类型相关，这些属性的主要细微差别，以及一些有用的解决方法。

# 可迭代的

Python 中的 iterable 对象是这样一种对象，它可以通过循环来逐个提取其项目，或者对每个项目应用特定的操作并返回结果。可迭代对象大多是表示一组项目(列表、元组、集、冷冻集、字典、范围和迭代器)的复合对象，但字符串也是可迭代的。

对于所有可迭代的数据类型，我们可以使用 For 循环来迭代对象:

```
for i in (1, 2, 3):
    print(i)**Output:**
1
2
3
```

对于 Python 字典，默认情况下对字典键执行迭代:

```
dct = {'a': 1, 'b': 2}
print('Dictionary:', dct)print('Iterating over the dictionary keys:')
for i in dct:
    print(i)**Output:**
Dictionary: {'a': 1, 'b': 2}
Iterating over the dictionary keys:
a
b
```

如果我们想要迭代字典值，我们必须在字典上使用`values()`方法:

```
print('Iterating over the dictionary values:')
for i in dct.values():
    print(i)**Output:** Iterating over the dictionary values:
1
2
```

相反，如果我们想要迭代字典键和值，我们应该使用`items()`方法:

```
print('Iterating over the dictionary keys and values:')
for k, v in dct.items():
    print(k, v)**Output:** Iterating over the dictionary keys and values:
a 1
b 2
```

所有可迭代的 Python 对象都有`__iter__`属性。因此，检查 Python 对象是否可迭代的最简单方法是对其使用`hasattr()`方法，检查`__iter__`属性是否可用:

```
print(hasattr(3.14, '__iter__'))
print(hasattr('pi', '__iter__'))**Output:** False
True
```

我们可以使用`iter()`函数从任何可迭代的 Python 对象中获得迭代器对象:

```
lst = [1, 2, 3]
print(type(lst))iter_obj = iter(lst)
print(type(iter_obj))**Output:** <class 'list'>
<class 'list_iterator'>
```

完全符合预期(有点赘述)，迭代器对象是可迭代的，所以我们可以对它进行迭代。**然而**与所有其他可迭代对象不同，迭代器对象会在迭代过程中一个接一个地丢失元素:

```
lst = [1, 2, 3]
iter_obj = iter(lst)
print('The iterator object length before iteration:')
print(len(list(iter_obj)))for i in iter_obj:
    pass
print('The iterator object length after iteration:')
print(len(list(iter_obj)))**Output:** The iterator object length before iteration:
3
The iterator object length after iteration:
0
```

注意，没有像其他可迭代数据类型的内置函数`len()`那样直接检查迭代器对象长度的方法。因此，为了检查迭代器中的项数，我们首先必须将它转换成另一个可迭代对象，比如一个列表或元组，然后才在其上应用`len()`函数:`len(list(iter_obj))`。

# 有序与无序

Python 中的有序对象是那些 *iterable* 对象，其中的项目保持确定的顺序，除非我们有意更新这些对象(插入新项目、删除项目、排序项目)。有序对象是字符串、列表、元组、范围和字典(从 Python 3.7+开始)、无序集、冷冻集和字典(在 Python 3.7 之前的版本中)。

```
lst = ['a', 'b', 'c', 'd']
s = {'a', 'b', 'c', 'd'}
print(lst)
print(s)**Output:** ['a', 'b', 'c', 'd']
{'b', 'c', 'a', 'd'}
```

由于有序对象中的顺序被保留，我们可以通过索引或切片来访问和修改对象的项目:

```
lst = ['a', 'b', 'c', 'd']# Access the 1st item of the list.
print(lst[0])# Access the 2nd and 3rd items of the list.
print(lst[1:3])# Modify the 1st item.
lst[0] = 'A'
print(lst[0])**Output:** a
['b', 'c']
A
```

为了从 Python 字典中提取特定的值，我们通常使用相应的键名称，而不是字典中的键索引:

```
dct = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
print(dct['b'])**Output:** 2
```

然而，如果出于某种原因，我们需要提取字典中已知位置的键的值(假设我们只知道*的位置，而不知道键本身)，或者字典中定义了位置范围的一部分键的一组值，我们在技术上仍然可以做到:*

```
dct = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}# Access the value of the 1st key in the dictionary.
print(list(dct.values())[0])# Access the values of the 2nd to 4th keys in the dictionary.
print(list(dct.values())[1:4])# Access the 2nd to 4th keys in the dictionary.
print(list(dct.keys())[1:4])**Output:** 1
[2, 3, 4]
['b', 'c', 'd']
```

上面的解决方法不是很直接。但是，它有助于通过键或值来索引 Python 字典。

在无序对象中，我们不能访问单个项目:

```
s = {'a', 'b', 'c', 'd'}
print(s[0])**Output:
---------------------------------------------------------------------------**
**TypeError**                                 Traceback (most recent call last)
**~\AppData\Local\Temp/ipykernel_9980/849534030.py** in <module>
      1 s **=** **{'a',** **'b',** **'c',** **'d'}**
**----> 2** print**(**s**[0])**

**TypeError**: 'set' object is not subscriptable
```

如果我们有一个包含其他复合对象的复合有序对象，我们可以更深入地挖掘并访问该对象项目的内部项目，如果它们也是有序的。例如，如果一个 Python 列表包含另一个 Python 列表，我们可以访问内部列表的项目:

```
lst_2 = [[1, 2, 3], {1, 2, 3}, 10]
print(lst_2[0][2])**Output:** 3
```

然而，我们不能访问列表中集合的项目，因为 Python 集合是无序的:

```
print(lst_2[1][2])**Output:
---------------------------------------------------------------------------**
**TypeError**                                 Traceback (most recent call last)
**~\AppData\Local\Temp/ipykernel_9980/895995558.py** in <module>
**----> 1** print**(**lst_2**[1][2])**

**TypeError**: 'set' object is not subscriptable
```

# 可变与不可变

Python 中的可变对象是那些可以被修改的对象。**可变性并不一定意味着能够通过索引或切片来访问复合对象的单个项目。**例如，一个 Python 集合是无序的和无索引的，然而，它是一个可变的数据类型，因为我们可以通过添加新的条目或从中删除条目来修改它。

另一方面，Python tuple 是一种不可变的数据类型，但是我们可以通过索引和切片轻松地访问它的各个项(但是不能修改它们)。此外，还可以对 range 对象进行索引和切片，从中提取整数或更小的范围。

一般来说，可变数据类型是列表、字典、集合和字节数组，而不可变数据类型是所有原始数据类型(字符串、整数、浮点、复杂、布尔、字节)、范围、元组和冻结集。

让我们探讨一个关于元组的有趣警告。作为不可变的数据类型，元组可以包含可变数据类型的项目，例如列表:

```
tpl = ([1, 2], 'a', 'b')
print(tpl)**Output:** ([1, 2], 'a', 'b')
```

上面的元组包含一个列表作为它的第一项。我们可以访问它，但不能为此项目重新分配其他值:

```
# Access the 1st item of the tuple.
print(tpl[0])# Try to re-assign a new value to the 1st item of the tuple.
tpl[0] = 1**Output:** [1, 2]**---------------------------------------------------------------------------**
**TypeError**                                 Traceback (most recent call last)
**~\AppData\Local\Temp/ipykernel_9980/4141878083.py** in <module>
      3 
      4 **# Trying to modify the first item of the tuple**
**----> 5** tpl**[0]** **=** **1**

**TypeError**: 'tuple' object does not support item assignment
```

然而，由于 Python 列表是可变的有序数据类型，我们既可以访问它的任何项，也可以修改它们:

```
# Access the 1st item of the list.
print(tpl[0][0])# Modify the 1st item of the list.
tpl[0][0] = 10**Output:** 1
```

结果，我们的元组中的一个项目被改变了，并且元组本身看起来不同于最初的那个:

```
print(tpl)**Output:** ([10, 2], 'a', 'b')
```

# 可散列与不可散列

可散列 Python 对象是任何具有散列值的对象，散列值是该对象的一个整数标识符，在其生命周期中不会改变。为了检查一个对象是否是可散列的，并找出它的散列值(如果它是可散列的)，我们在这个对象上使用了`hash()`函数:

```
print(hash(3.14))**Output:** 322818021289917443
```

如果对象不可修复，将抛出一个`TypeError`:

```
print(hash([1, 2]))**Output:
---------------------------------------------------------------------------**
**TypeError**                                 Traceback (most recent call last)
**~\AppData\Local\Temp/ipykernel_9980/3864786969.py** in <module>
**----> 1** hash**([1,** **2])**

**TypeError**: unhashable type: 'list'
```

*几乎*所有不可变的对象都是可散列的(我们很快会看到一个特殊的例外)，而不是所有可散列的对象都是不可变的。特别是，所有的原始数据类型(字符串、整数、浮点、复数、布尔、字节)、范围、冷冻集、函数和类，无论是内置的还是用户定义的，都是可散列的，而列表、字典、集合和字节数组是不可散列的。

一个奇怪的例子是 Python 元组。因为它是不可变的数据类型，所以应该是可散列的。事实上，似乎是这样的:

```
print(hash((1, 2, 3)))**Output:** 529344067295497451
```

还要注意，如果我们将这个元组赋给一个变量，然后对该变量运行`hash()`函数，我们将获得相同的哈希值:

```
a_tuple = (1, 2, 3)
print(hash(a_tuple))**Output:** 529344067295497451
```

如果一个元组包含可变项，就像我们在上一节中看到的那样，会怎么样呢？

```
tpl = ([1, 2], 'a', 'b')
print(hash(tpl))**Output:
---------------------------------------------------------------------------**
**TypeError**                                 Traceback (most recent call last)
**~\AppData\Local\Temp/ipykernel_8088/3629296315.py** in <module>
      1 tpl **=** **([1,** **2],** **'a',** **'b')**
**----> 2** print**(**hash**(**tpl**))**

**TypeError**: unhashable type: 'list'
```

我们看到 Python 元组可以是可散列的，也可以是不可散列的。只有当它们包含至少一个可变项时，它们才是不可取消的。

有趣的是，即使是不可修复的元组，如上图所示，也有`__hash__`属性:

```
# Check if a hashable tuple has the '__hash__' attribute.
print(hasattr((1, 2, 3), '__hash__'))# Check if an unhashable tuple has the '__hash__' attribute.
print(hasattr(([1, 2], 'a', 'b'), '__hash__'))**Output:** True
True
```

我们前面提到过，并不是所有的可散列对象都是不可变的。这种情况的一个例子是可变但可散列的用户定义类。此外，类的所有实例都是可散列的，并且具有与类本身相同的散列值:

```
class MyClass:
    passx = MyClassprint(hash(MyClass))
print(hash(x))**Output:** 170740243488
170740243488
```

通常，当两个 Python 对象相等时，它们的哈希值也相等:

```
# Check the equal objects and their hash values.
print(True==1)
print(hash(True)==hash(1))# Check the unequal objects and their hash values.
print('god'=='dog')
print(hash('god')==hash('dog'))**Output:** True
True
False
False
```

# 结论

总之，我们从多个方面详细探讨了经常使用但经常被误解的 Python 对象和数据类型特征，如可迭代、有序、可变、可散列以及它们的对立面，包括一些特殊情况和例外。

感谢阅读！

**你会发现这些文章也很有趣:**

[](/16-underrated-pandas-series-methods-and-when-to-use-them-c696e17fbaa4)  [](https://levelup.gitconnected.com/when-a-python-gotcha-leads-to-wrong-results-2447f379fdfe)  [](https://medium.com/geekculture/creating-toyplots-in-python-49de0bb27ec1) 