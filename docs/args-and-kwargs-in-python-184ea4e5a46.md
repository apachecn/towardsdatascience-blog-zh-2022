# * Python 中的 args 和**kwargs

> 原文：<https://towardsdatascience.com/args-and-kwargs-in-python-184ea4e5a46>

## 在本文中，我们将探索 Python 中的 ***args** 和 ****kwards** 以及它们在函数中的用法，并给出例子

![](img/3dd5acc24c7e9fcf8da1457dde4c819b.png)

照片由[沙哈达特·拉赫曼](https://unsplash.com/@hishahadat?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/photos/gnyA8vd3Otc?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄

**目录**

*   介绍
*   * Python 中的参数
*   * * Python 中的 kwargs
*   结论

# 介绍

在 Python 编程中，我们经常使用自己创建的函数，以使代码更易于重用和理解。

如果我们需要多次执行类似的操作，而不是复制粘贴相同的代码，我们创建一个函数，用新的参数调用多次来执行这个操作。

***args** 和 ****kwargs** 允许我们向函数传递可变数量的参数，因此扩展了我们创建的函数的可重用性。

请注意， ***args** 和 ****kwargs** 的用法主要取决于您的函数将执行什么操作，以及它们接受更多参数作为输入的灵活性。

# * Python 中的参数

我们先来了解一下 ***args** 到底是什么，以及如何使用！

***args** 用于传递可变数量的非关键字参数，使用以下语法:

```
def my_func(*args):
    #do something
```

当你不确定函数每次使用时会收到多少个参数时，你可以使用 ***args** 。

例如，假设您有一个将两个数字相加并返回其和的函数:

```
#Define a function
def add(x, y):

    #Add two numbers
    result = x+y

    #Print the result
    print(f'Result: {result}')
```

如果您要用样本值测试这个函数:

```
#Test with sample values
add(3, 5)
```

您应该得到:

```
8
```

到目前为止， **add()** 函数工作正常，因为它只接受两个参数并返回一个结果。

如果你想把三个数字加在一起呢？

您可以尝试使用 **add()** 功能:

```
#Test with 3 values
add(3, 5, 9)
```

但是你会得到一个类型错误:

```
TypeError: add() takes 2 positional arguments but 3 were given
```

发生这种情况是因为当我们定义 **add()** 函数时，我们专门用两个参数 **x** 和 **y** 定义了它，所以当传递第三个参数时，它应该会返回一个错误。

但是我们如何使这个函数更健壮，以便它接受更多的参数作为输入呢？

这就是 ***args** 派上用场的地方！

让我们重写 **add()** 函数，将 ***args** 作为参数:

```
#Define a function
def add(*args):

    #Initialize result at 0
    result = 0

    #Iterate over args tuple
    for arg in args:
        result += arg

    #Print the result
    print(f'Result: {result}')
```

***args** 将允许我们向 **add()** 函数传递可变数量的非关键字参数。它将作为一个[元组](https://pyshark.com/python-tuple-data-structure/)被传递，这意味着我们可以使用 **for** 循环来迭代它。

现在让我们用示例值来测试这个函数:

```
#Test with 3 arguments
add(3, 5, 9)

#Test with 5 arguments
add(3, 5, 9, 11, 15)
```

您应该得到:

```
Result: 17
Result: 43
```

现在，我们通过使用 ***args** 使 **add()** 函数更具可重用性，使其能够处理可变数量的非关键字参数。

# * * Python 中的 kwargs

让我们先来了解一下 ***args** 到底是什么以及如何使用它！

****kwargs** 用于使用以下语法传递可变数量的关键字参数:

```
def my_func(**kwargs):
    #do something
```

当您不确定函数每次使用时会收到多少个参数时，您可以使用 ****kwargs** 。

请注意， ***args** 和 ****kwargs** 的主要区别在于， ***args** 允许可变数量的*非关键字*参数，而 as ****kwargs** 允许可变数量的*关键字*参数。

例如，假设您有一个函数，它接受两个关键字参数并打印它们的值:

```
#Define function
def print_vals(name, age):

    #Print first argument
    print(f'name: {name}')
    #Print second argument
    print(f'age: {age}')
```

如果您要用样本值测试这个函数:

```
#Test with sample values
print_vals(name='Mike', age=20)
```

您应该得到:

```
name: Mike
age: 20
```

到目前为止， **print_vals()** 函数工作正常，因为它只接受 2 个参数并打印它们。

如果你想打印 3 个值呢？

您可以尝试使用 **print_vals()** 函数:

```
#Test with 3 values
print_vals(name='Mike', age=20, city='New York')
```

但是你会得到一个*类型错误*:

```
TypeError: print_vals() got an unexpected keyword argument 'city'
```

出现这种情况是因为我们在定义 **print_vals()** 函数时，专门用两个关键字参数 **name** 和 **age** 定义了它，所以当第三个关键字参数被传递时，它应该会返回一个错误。

但是我们如何使这个函数更加健壮，以便它可以接受更多的关键字参数作为输入呢？

这就是 ****kwargs** 派上用场的地方！

让我们重写 **add()** 函数，将 ***args** 作为参数:

```
#Define function
def print_vals(**kwargs):

    #Iterate over kwargs dictionary
    for key, value in kwargs.items():

        #Print key-value pairs
        print(f'{key}: {value}')
```

****kwargs** 将允许我们向 **print_vals()** 函数传递可变数量的关键字参数。它将作为[字典](https://pyshark.com/python-dictionary-data-structure/)被传递，这意味着我们可以使用 **for** 循环来迭代它。

现在让我们用示例值来测试这个函数:

```
#Test with 3 keyword arguments
print_vals(name='Mike', age=20, city='New York')

#Test with 5 keyword arguments
print_vals(name='Mike', age=20, city='New York', height=6.0, weight=180)
```

您应该得到:

```
name: Mike
age: 20
city: New York

name: Mike
age: 20
city: New York
height: 6.0
weight: 180
```

现在，我们通过使用 ****kwargs** 使 **print_vals()** 函数具有更高的可重用性，使其能够处理数量可变的关键字参数。

# 结论

在这篇文章中，我们将探索 [Python](https://docs.python.org/3/library/) 中的 ***args** 和 ****kwards** 以及它们在函数中的使用，并给出例子。

既然您已经知道了基本的功能，您可以通过在不同的项目中重写一些现有的代码来实践它，并尝试解决更复杂的用例。

如果你有任何问题或者对编辑有任何建议，请在下面留下评论，并查看我的更多 [Python 函数](https://pyshark.com/category/python-functions/)教程。

*原载于 2022 年 12 月 25 日 https://pyshark.com*<https://pyshark.com/args-and-kwargs-in-python/>**。**