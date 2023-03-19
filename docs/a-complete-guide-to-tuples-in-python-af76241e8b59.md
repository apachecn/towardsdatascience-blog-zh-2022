# Python 中元组的完全指南

> 原文：<https://towardsdatascience.com/a-complete-guide-to-tuples-in-python-af76241e8b59>

## 什么是元组、元组实现、数据类型、索引、不变性和扩展功能

![](img/c75782963c6b0c2870c41b725bfd116d.png)

照片由[Paico official](https://unsplash.com/@paicooficial?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

## 什么是元组？

元组是一种数据结构，类似于 Python 中的列表，但通常不太为人所知或与之交互。它们共享列表的相同特征，因为它们是有序的和可索引的，但是它们的不同之处在于它们是不可变的，并且它们是使用`()`符号而不是`[]`符号创建的。这意味着:

*   **不可变**:一旦创建就不能更改
*   **有序**:一旦创建，它们就保持它们的顺序
*   **可索引**:如果我们知道条目在元组中的位置，就可以访问信息
*   **可以包含重复记录:**可以包含相同值的项目，没有任何问题。

这很重要，因为这些特征会影响元组相对于列表的使用方式。不可变的主要区别在于，因为它们一旦被创建就不能被更改，所以它们可以在程序中使用，一旦你设置了一个值，你就不希望它们被意外地更改，同时仍然具有与列表相同的功能。这样的例子包括存储实验结果或设置在整个程序中不变的起始参数。

## 履行

要创建元组，我们可以使用两种主要方法:

*   使用`()`将我们想要包含的信息用逗号分隔的条目包含在元组中
*   使用`tuple()`函数，该函数可用于将其他数据结构转换为元组或以列表作为参数

这些可以简单地实现为:

```
#create the tuple
cars = ("Ford", "Hyundai", "Toyota", "Kia")#create a second tuple
fruits_tuple = tuple(("Strawberry", "peach", "tomato"))#create the third tuple
vegetable_tuple = tuple(["potato", "onion", "celery"])#print the result
print(cars)
print(type(cars))print(fruits_tuple)
print(type(fruits_tuple))print(vegetable_tuple)
print(type(vegetable_tuple))#out:
('Ford', 'Hyundai', 'Toyota', 'Kia')
<class 'tuple'>
('Strawberry', 'peach', 'tomato')
<class 'tuple'>
('potato', 'onion', 'celery')
<class 'tuple'>
```

这里需要注意的一点是，`tuple()`函数只接受一个参数，这意味着我们可以将它传递给一个更早实现的数据结构，或者像我们在这里所做的那样，我们可以在`()`中传递一个元组本身，或者在`[]`中传递一个列表。

我们还可以看到，我们已经能够检查我们通过使用`type()`功能创建的数据结构的类型，该功能告诉我们我们有一个`tuple`类。我们还可以看到，当打印出一个元组时，它由元组本身中的项目周围的`()`来表示，并且这些项目保留了它们在原始数据结构中的顺序。

## 元组中的数据类型

由于元组的行为类似于列表，这意味着我们也可以在元组中包含不同的数据类型。虽然我们在上面实现了内部只有字符串的元组，但是我们也可以创建内部有整数和浮点数的元组:

```
#create a list of just numbers
num_tuple = (1, 2, 3, 4)#create a list of just floats
float_tuple = (1.2, 2.3, 4.5, 6.8)#print the results
print(type(num_tuple))
print(num_tuple)print("\n")print(type(float_tuple))
print(float_tuple)#out:
<class 'tuple'>
(1, 2, 3, 4)

<class 'tuple'>
(1.2, 2.3, 4.5, 6.8)
```

我们还可以在一个元组中组合不同的数据类型，这样我们就不必拥有一种以上的数据类型。例如，我们可以在里面放一个列表，甚至另一个元组！

```
#different list
random_tuple = ("Hello", 3, "Cheese", 6.2, [1,2,3], (4,6,7))#print the result
print(type(random_tuple))
print(random_tuple)#out:
<class 'tuple'>
('Hello', 3, 'Cheese', 6.2, [1, 2, 3], (4, 6, 7))
```

## 索引

就像列表一样，元组的一个重要特征是它们是项目的有序集合。这意味着一旦创建了元组，它们就有了明确定义的顺序，并且因为它们是不可变的，所以顺序不能改变。

这种顺序允许我们访问元组中的值，我们知道这些值在该顺序中的给定位置。例如，如果我们根据汽车制造商的位置创建了一个我们想要访问的汽车制造商列表，如果我们忘记了我们计划访问的第一个制造商，我们可以通过使用列表的第一个索引来访问第一个制造商。当然，因为这是 Python，所以一切都以 0 索引开始，所以我们可以用以下方式访问元组中的第一项:

```
#get the first item from the tuple
print(cars[0])#out:
Ford
```

为此，方括号用于输入索引号，这样`tuple[0]`就是我们访问第一个索引的方式。

按照这个例子，可以使用`tuple[1]`访问元组中的第二项，使用`tuple[2]`访问第三项，依此类推。为此，只要索引属于您尝试访问的元组，任何解析为数字的内容都可以用来访问元组中的内容。如果您试图使用一个超出元组的索引，那么您将得到一个错误，告诉您该索引不在元组中。

这种索引的一个有用的优点是，我们不仅可以向前计数，就像我们对汽车元组所做的那样，我们还可以向后计数。这意味着我们可以检查最后添加到元组中的项。例如，如果我们想要检查我们计划最后访问的制造商，我们将使用:

```
#get the last item from the tuple
print(cars[-1])#out:
Kia
```

为此，重要的是要注意倒计数从-1 开始，并随着我们的进行而增加。这是因为如果我们从 0 开始，那么我们将会混淆我们想要列表中的第一项还是最后一项。

最后，我们还可以使用索引同时提取一个元组中的多个项，而不是单个项。我们可以使用与 list 相同的方式来实现这一点，使用`tuple[start_item: end_item]`的切片符号，这样最后一项就不会出现在返回的元组中。这方面一个例子包括:

```
#get the second and third from the tuple
print(cars[1:3])#get all from the first index
print(cars[1:])#get all until the fourth one
print(cars[:3])#out:
('Hyundai', 'Toyota')
('Hyundai', 'Toyota', 'Kia')
('Ford', 'Hyundai', 'Toyota')
```

几个不同的规则适用于此，因为:

*   当使用`[1:3]`打印第二个最低的项目时，由于最终索引不包含在结果中，所以只返回两个项目
*   当打印`[1:]`时，我们没有指定结束索引，这就是为什么在第四个索引之后，包括第四个索引，整个元组都被打印出来
*   当打印第四个索引之前的所有内容时，因为我们没有指定开头，所以打印了元组的开头

也像列表一样，每当获取一个切片时，切片的类型将与您获取切片的对象的类型相同。在这里，因为我们获取一个元组的一部分，所以返回一个元组。

## 查找项目

从上面我们可以看到，当我们知道项目的索引时，我们可以访问项目，但是如果我们只知道元组包含项目，而不知道它的位置，那该怎么办呢？例如，在我们的汽车列表中，我们知道我们必须访问丰田，但不知道我们必须访问制造商的顺序。然后，像列表一样，我们可以使用`index()`方法找到项目的位置，如下所示:

```
#get the index for Toyota
print(cars.index("Toyota"))#out:
2
```

尽管需要注意的是，当访问重复值的索引时，`index()`方法将只返回该值的第一个索引的索引。

唯一的问题是，如果您拼错了该项或者该项不在元组中，该方法将抛出一个错误，并将停止代码运行。解决这个问题的一个简单方法是使用 if/else 语句，该语句可以使用:

```
if "Toyota" in cars:
    print("Toyota is at index:", cars.index("Toyota"))
else:
    print("Toyota not in tuple")#out:
Toyota is at index: 2
```

## 不变

元组是不可变的意味着它们不能被改变。如果你试图使用索引和赋值来改变一个值，比如`cars[0] = "Tesla"`，那么你将得到一个类型错误，说明“**type error**:‘tuple’对象不支持项赋值”，这表明你不能改变一个元组。

也就是说，如果你愿意，有一些方法可以解决这个问题，尽管元组的目的是你首先不要这么做。第一种方法是将元组转换为列表，更新值，然后将其转换回元组，如下所示:

```
#print the tuple
print(cars)#change it to a list
tuple_list = list(cars)#change the value
tuple_list[0] = "Maserati"#reassign back to the tuple
cars = tuple(tuple_list)#print the result
print(cars)#out:
('Ford', 'Hyundai', 'Toyota', 'Kia')
('Maserati', 'Hyundai', 'Toyota', 'Kia')
```

当然，如果你想这样做，那么你可能应该首先创建一个列表。

改变元组的另一种方法是将两个元组连接在一起形成一个新的元组。这并不一定要改变原来的元组，而是创建一个新的元组，所以这也是一种变通方法。这当然意味着你对一个元组所能做的唯一改变是在末尾或开头添加内容，而不是改变元组本身。这可以通过以下方式实现:

```
#create new tuples
tuple1 = ("a", "b", "c")
tuple2 = (1,2,3)#add together using the +
tuple3 = tuple1 + tuple2
print(tuple3)#multiply an existing tuple together 
tuple4 = tuple1*2
print(tuple4)#out:
('a', 'b', 'c', 1, 2, 3)
('a', 'b', 'c', 'a', 'b', 'c')
```

## 内置功能

最后，我们有几个内置的元组功能，就像我们处理列表一样，可以包括查找元组的长度，打印特定值的实例数，以及查找元组的最小值或最大值。这可以通过以下方式实现:

```
#print the length of the tuple
print(len(tuple1))#print the count of values within a tuple
print(tuple4.count("a"))#print the maximum value from a tuple
print(max(tuple2))#print the minimum value from a tuple
print(min(tuple2))#out:
3
2
3
1
```

这就是 Python 中元组的完整指南！元组在本质上明显类似于列表，因为它是有序的和可索引的。但是，它与列表的主要区别在于它不能被更改。这意味着当您不希望任何信息在创建后被更改时，例如当您不希望实验结果被覆盖或出于安全原因时，它会很有用。

这是探索数据结构及其在 Python 中的使用和实现系列的第三篇文章。如果您错过了列表和集合中的前两个，您可以在以下链接中找到它:

[](/a-complete-guide-to-lists-in-python-d049cf3760d4)  [](/a-complete-guide-to-sets-in-python-99dc595b633d)  

未来的帖子将涵盖 Python 中的字典、链表、栈、队列和图形。为了确保您将来不会错过任何内容，请注册以便在发布时收到电子邮件通知:

[](https://philip-wilkinson.medium.com/subscribe)  

如果你喜欢你所阅读的内容，并且还不是一个媒体成员，考虑通过使用我下面的推荐代码注册来支持我自己和这个平台上其他了不起的作者:

[](https://philip-wilkinson.medium.com/membership)  [](/an-introduction-to-sql-for-data-scientists-e3bb539decdf)  [](/git-and-github-basics-for-data-scientists-b9fd96f8a02a)  [](/london-convenience-store-classification-using-k-means-clustering-70c82899c61f) 