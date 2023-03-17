# Python 中的 Iterables vs 迭代器

> 原文：<https://towardsdatascience.com/python-iterables-vs-iterators-688907fd755f>

## 理解 Python 中 Iterables 和迭代器的区别

![](img/d75823e57f14b4d42578978cea93c6c7.png)

亨利&公司在 [Unsplash](https://unsplash.com/s/photos/loop?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

## 介绍

术语 **iterable** 和 **iterator** 经常(错误地)互换使用，以描述支持迭代的对象，即允许迭代其元素的对象。事实上，Python 中的迭代器和可迭代对象是两个截然不同的概念，通常会引起混淆，尤其是对新手而言。

在今天的文章中，我们将讨论 Python 中的迭代协议，以及迭代器和可迭代对象如何参与其实现。此外，我们将探索 iterable 和 iterator 对象之间的主要区别，并提供一个示例来帮助您理解 iterable 和 iterator 对象是如何工作的。

## Python 中的迭代协议

Python 中的**所有迭代工具**使用迭代协议，并通过各种对象类型(如 for 循环、综合、映射等)实现。本质上，协议由两种对象类型组成，即 **iterable** 和 **iterator** 。

*   **iterable** 对象是您**迭代**其元素的对象
*   **迭代器**对象是**在迭代**过程中产生值的对象，也是由可迭代对象返回的

**在接下来的部分中，我们将详细了解这两种对象类型是如何工作的，以及它们需要实现什么来满足迭代协议。**

## **什么是可迭代的**

**在 Python 中，Iterable 是一个对象，它由**实现** `**__iter__()**`方法，由**返回一个 iterator 对象**或者实现`__getitem__()`方法的对象(并且应该在索引用尽时引发一个`IndexError`)。内置的可迭代对象包括**列表、集合和字符串**，因为这样的序列可以在一个 for 循环中迭代。**

**注意**在最近的 Python 版本中，实现 Iterables 的首选方式是通过实现** `**__iter__()**` **方法**。`__getitem__()`方法是一种在现代迭代器之前使用的遗留功能。但是 Python 仍然认为实现了`__getitem__()`方法的对象是可迭代的。这意味着如果没有定义`__iter__()`，Python 解释器将使用`__getitem__()`。更多详情可以参考 [PEP-234](https://www.python.org/dev/peps/pep-0234/) 。**

**总而言之，Python 中的 Iterable 是任何对象，只要它**

*   **可以迭代(例如，可以迭代一个字符串的字符或一个文件的行)**
*   **实现`__iter__()`方法(或`__getitem__`)，因此可以用返回迭代器的`iter()`调用它**
*   **可以出现在 for 循环的右边(`for i in myIterable:`)**

## **什么是迭代器**

**另一方面，Python 中的迭代器是**以某种方式实现** `**__next__()**` **方法**的对象**

*   **返回 iterable 对象的下一个值，更新迭代器的状态，使其指向下一个值**
*   **当 iterable 对象的元素用尽时引发一个`StopIteration`**

**此外，**迭代器本身也是可迭代的，因为它还必须实现** `**__iter__()**` **方法**，其中它只是返回`self`。**

> **每个迭代器也是可迭代的，但不是每个可迭代的都是迭代器**

## **Python 中的可迭代对象和迭代器**

**正如我们已经提到的，Python List 是可迭代的内置对象类型之一。现在让我们假设我们有下面的整数列表，如下所示:**

```
>>> my_lst = [5, 10, 15]
```

**由于`my_lst`是一个 iterable，我们可以运行`iter()`方法来从 iterable 中获得一个 iterator 对象:**

```
>>> my_iter = iter(my_lst)
```

**我们可以验证`my_iter`是否属于`list_iterator`类型:**

```
>>> type(my_iter)
list_iterator
```

**现在由于`my_iter`是一个迭代器，因此它实现了`__next__()`方法，该方法将返回列表 iterable 的下一个元素:**

```
>>> my_iter.__next__()
5
```

**一旦迭代器在`__next__()`调用后返回下一个值，它应该改变它的状态，以便它现在指向下一个元素:**

```
>>> my_iter.__next__()
10
```

**注意，`next(iter_name)`也是一个有效的语法，相当于`iter_name.__next__()`:**

```
>>> next(my_iter)
15
```

**现在我们到达了最后一个元素，对`__next__()`方法的下一次调用应该会产生一个`StopIteration`，这是实现`__next__()`方法的迭代器必须满足的一个要求:**

```
>>> my_iter.__next__()Traceback (most recent call last):
  File "/usr/local/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 3331, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-8-56d779eb7d74>", line 1, in <module>
    my_iter.__next__()
**StopIteration**
```

## **最后的想法**

**在今天的文章中，我们讨论了 Python 中的迭代协议，以及迭代中如何涉及迭代和迭代器。此外，我们讨论了可迭代和迭代器的主要特征，并介绍了它们的主要区别。最后，我们展示了 Iterable 和 Iterator 对象是如何工作的。**

**在我的下一篇文章中，我将讨论如何创建用户定义的迭代器，以使用户定义的类可迭代。**

**[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。****

**[](https://gmyrianthous.medium.com/membership) [## 通过我的推荐链接加入 Medium-Giorgos Myrianthous

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

gmyrianthous.medium.com](https://gmyrianthous.medium.com/membership)** 

****你可能也会喜欢****

**[](/python-linked-lists-c3622205da81) [## 如何在 Python 中实现链表

### 探索如何使用 Python 从头开始编写链表和节点对象

towardsdatascience.com](/python-linked-lists-c3622205da81)** **[](/switch-statements-python-e99ea364fde5) [## 如何用 Python 编写 Switch 语句

### 了解如何使用模式匹配或字典在 Python 中编写 switch 语句

towardsdatascience.com](/switch-statements-python-e99ea364fde5)** **[](/augmented-assignments-python-caa4990811a0) [## Python 中的扩充赋值

### 了解增强赋值表达式在 Python 中的工作方式，以及为什么在使用它们时要小心…

towardsdatascience.com](/augmented-assignments-python-caa4990811a0)**