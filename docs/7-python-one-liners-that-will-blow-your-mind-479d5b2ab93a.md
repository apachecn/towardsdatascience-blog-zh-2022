# 会让你大吃一惊的 7 个 Python 单句程序

> 原文：<https://towardsdatascience.com/7-python-one-liners-that-will-blow-your-mind-479d5b2ab93a>

## 少即是多？

![](img/40aefddf41e6fd39f0fe50b641589456.png)

[摄影](https://unsplash.com/@photos_by_lanty)Lanty 摄影 [Unsplash](https://unsplash.com/)

一句话这个术语来自喜剧，其中一个笑话用一句台词来表达。一个好的一行程序据说是有意义的和简洁的。这个概念也存在于编程中。Python 一行程序是可以执行强大操作的简短程序。这在 Java 等其他语言中几乎是不可能的，所以它被认为是 Pythonic 的一个特性。

## **关于 Python 一行程序的争论**

如果您是第一次看到一行程序，请记住 Python 一行程序是一把双刃剑。一方面，它让代码*看起来很酷*，这绝对是给你的同事或面试官留下深刻印象的一种方式。但另一方面，花哨的俏皮话可能会令人困惑，难以理解，并变成炫耀技能的一种方式。因此，在代码审查期间，一定会有一些争论。根据经验，一行程序在 Python 中非常受欢迎，但如果它走得太远，开始让人困惑，那么就该放弃它了。最终，您的代码需要具有可读性和可维护性。

无论如何，在本文中，我想给你一些可以提高代码质量的 Python 一行程序的实际例子。我还将向您展示一些有趣的俏皮话。无论如何，你必须知道用代码表达你的想法的所有可能的方式，这样你才能做出正确的决定。

## 交换两个变量

```
# normal
c = a
a = b
b = c# 1-liner
a,b = b,a
```

在正常情况下，当你交换两个变量时，你需要一个中间人。我在我的[如何编写 Pythonic 代码文章](/how-to-write-pythonic-code-208ec1513c49)中也提到了这个窍门。在 Python 中，一行就可以完成。右边是一个表达式，实际上是一个元组(b，a ),左边是表示元组中第一个和第二个变量的变量。

## 列表理解

```
result = []
for i in range(10):
    result.append(i**2)# use list comprehension
result = [i**2 for i in range(10)]
```

这是我在上一篇文章中提到的另一个 Pythonic 特性。它告诉 Python 如何处理列表中的每个元素。该列表包含`for`和`if`语句。例如:

```
#list comprehension with if
result = [i**2 for i in range(10) if i%2==0]#read file in one-line
[line.strip() for line in open(filename)]
```

列表/集合理解是一个如此强大的工具，每个人都必须知道它。它允许您编写优雅的代码，几乎和普通英语一样易读。每个列表理解包括 3 个要素:1)表达。任何有效的表达式，如`i**2`或`line.strip()`。2)成员。iterable 中的对象。3)可迭代。列表、集合、生成器或任何其他可迭代对象。

## λ和映射函数

列表理解的下一个层次是 lambda 函数。查看[从头开始学习 Python Lambda](/learn-python-lambda-from-scratch-f4a9c07e4b34)。

> 与其他语言中添加功能的 lambda 形式不同，如果你懒得定义函数，Python lambdas 只是一种速记符号。

```
# traditional function with def
def sum(a, b):    
  return a + b# lambda function with name
sum_lambda = lambda x, y: x + y
```

Lambda 函数通常与 Python 高阶函数结合使用。有几个内置的高阶函数，如`map`、`filter`、`reduce`、`sorted`、`sum`、`any`、`all`。

```
map(lambda x:x**2, [i for i in range(3)]) 
# <map object at 0x105558a90>filter(lambda x: x % 2 == 0, [1, 2, 3])
# <filter object at 0x1056093d0>from functools import reduce
reduce(lambda x, y: x + y, [1, 2, 3])
# 6
```

值得注意的是，`map`和`filter`的返回是对象，不是函数的结果。如果想得到实际的结果，需要转换成类似`list(map(lambda x:x**2, [i for i in range(3)])`的列表。

## 打印时不换行

`print`是每种编程语言中最基本的语句之一。但是你看过 Python doc 中`print`接口的定义吗？

```
**print**(**objects*, *sep=' '*, *end='\n'*, *file=sys.stdout*, *flush=False*)
```

该功能打印对象，由`sep`分隔，后跟`end`。让我们来看看这个一行程序。

```
#for-loop
for i in range(1,5):
    print(i, end=" ")#one-liner
print(*range(1,5)) #1 2 3 4
```

由于`print`接受一个 iterable，我们可以直接使用`*range(1,5)`作为输入，它的输出与 for 循环相同。

## 海象

Walrus 操作符是 Python 3.8 以后的一个特性。它为表达式中间的赋值变量提供了新的语法`:=` 。是为了避免两次调用同一个函数。

这是 Python 文档中的一个例子。walrus operator 允许您在飞行途中进行计算，而不是预先单独计算。

```
# without walrus
discount = 0.0
mo = re.search(r'(\d+)% discount', "10% discount")
if mo:
  discount = float(mo.group(1))/100.0# with walrus
discount = 0.0
if (mo := re.search(r'(\d+)% discount', "10% discount")):
  discount = float(mo.group(1)) / 100.0
```

但是我发现 walrus 在 for/while 循环中非常有用，因为代码看起来非常整洁，就像普通的英语一样。

```
# without walrus
f = open("source/a.txt")
line = f.readline()
while line != '':
    print(line)
    line = f.readline()# with walrus
f = open("f.txt")
while (line := f.readline()) != '':
    print(line)
```

## 斐波纳契

好了，现在让我们来看看一些奇特的例子。你知道你可以用一行代码编写斐波那契算法吗？挺神奇的！

```
#for-loop
def fib(x):
    if x <= 2:
        return 1
    return fib(x - 1) + fib(x - 2)#1-liner
fib = lambda x: x if x<=1 else fib(x-1) + fib(x-2)
```

## 快速排序

如果你认为你还能处理它，让我们看看下一个层次的一行程序。这是快速排序的一个单行版本，一个著名的排序算法。

```
q = lambda l: q([x for x in l[1:] if x <= l[0]]) + [l[0]] + q([x for x in l if x > l[0]]) if l else []
q([])
```

在我看来，看着它，了解 Python 有多强大很好玩，但不要在工作中使用它。当一行程序变得太长而无法容纳一行时，是时候放弃它了。

## 结论

像往常一样，我希望您发现这篇文章很有用，并且学到了一种提高代码质量的新方法。请随意与一行程序分享您的经验。干杯！

## 参考

[](https://www.amazon.com/Python-One-Liners-Concise-Eloquent-Professional/dp/1718500505) 