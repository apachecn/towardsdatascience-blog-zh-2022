# Python 中列表的完整指南

> 原文：<https://towardsdatascience.com/a-complete-guide-to-lists-in-python-d049cf3760d4>

## 关键特性、实现、索引、切片、定位项目、可变性和其他有用的功能

![](img/a9226e78a16deb51ce29b0b57e57e8e0.png)

马库斯·温克勒在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

## Python 中的列表

Python 中经常遇到的第一种数据结构是列表结构。它们可以用来在单个变量中存储多个项目，并有效地像您所期望的那样，像您自己的购物清单一样工作。它们是 Python 中四种内置数据类型之一，可用于存储数据集合以及元组、集合和字典。

它们的主要特点是:

*   可变的:一旦它们被定义，就可以被改变
*   有序:除非明确改变，否则它们保持它们的顺序
*   可索引:如果我们知道它们在列表中的位置，它们保持它们的顺序的事实允许我们访问特定的项目

它们也可能包含重复的记录。这一点很重要，因为它们会影响列表在程序中的使用方式，因为列表可以更改的事实可能意味着如果您希望数据是固定的，您可能不想使用它们，或者您可以通过索引访问信息的事实可能对以后的信息检索有用。

## 创建列表

要创建列表，我们可以使用两种主要方法:

*   使用`[]`将我们想要包含在列表中的内容用逗号分隔
*   使用`list()`函数，该函数可用于将其他数据结构转换为列表或以列表作为参数

这些可以简单地实现为:

```
#create a list of Fruit using the [] notation
fruit_list = ["Apple", "Banana", "Peach"]#creating a list of vegetables using the list() nottation
vegetable_list = list(["Pepper", "Courgette", "Aubergine"])
```

在使用`list()`符号时，我们不得不使用`[]`，因为函数只能接受一个参数，当我们想将其他数据类型转换成列表时，这很有用。

然后，我们可以使用`type()`函数检查这些结果，以确保它们是列表，并打印出列表本身的结果:

```
#examine the fruit list
print(type(fruit_list))
print(fruit_list)#print a seperate line
print("\n")#print the vegetable list
print(type(vegetable_list))
print(vegetable_list)#out:
<class 'list'>
['Apple', 'Banana', 'Peach']

<class 'list'>
['Pepper', 'Courgette', 'Aubergine']
```

我们可以看到，这两者的`class`属性以列表的形式给出。我们还可以看到，当我们打印出列表时，它们被打印在方括号中，并且与我们输入它们的顺序相同，这表明它们确实是列表，并且它们是有序的。

## 列表中的数据类型

列表的一个关键属性是它们可以在一个实例中包含所有不同类型的数据类型。尽管我们在上面只使用了字符串，但我们也可以在列表中输入数字或浮点数，例如:

```
#create a list of just numbers
num_list = [1, 2, 3, 4]#create a list of just floats
float_list = [1.2, 2.3, 4.5, 6.8]#print the results
print(type(num_list))
print(num_list)print("\n")print(type(float_list))
print(float_list)#out:
<class 'list'>
[1, 2, 3, 4]

<class 'list'>
[1.2, 2.3, 4.5, 6.8]
```

我们还可以在同一个列表中输入不同的数据类型，包括列表中的列表！

```
#different list
random_list = ["Hello", 3, "Cheese", 6.2, [1,2,3]]#print the result
print(type(random_list))
print(random_list)#out:
<class 'list'>
['Hello', 3, 'Cheese', 6.2, [1, 2, 3]]
```

这样做的好处是，对于更复杂的工作流，我们可以创建 2D 甚至 3D 列表，其中列表中有列表，这可以让我们可视化复杂的关系或模式。

## 索引

列表的一个重要部分是它们是有序的数据集合。这意味着他们有一个明确定义的顺序，输入到列表中的数据保持不变，除非我们告诉改变顺序。

这种顺序允许我们访问列表中的值，我们知道这些值在该顺序中处于固定位置。例如，如果我们根据每周购物时经过的地方对水果列表进行排序，这样我们就知道列表中的第一个水果将首先出现，而不是它是什么，我们可以简单地使用列表的第一个索引来访问它。当然，因为这是 Python，所以一切都以 0 索引开始，所以我们可以用:

```
#access the first item from the list
print(fruit_list[0])#out:
Apple
```

为此，方括号用于放入索引号，这样`[0]`就是我们访问第一个索引的方式。

按照这个例子，列表中的第二项可以用`list[1]`访问，第三项用`list[2]`访问，依此类推。为此，任何解析为数字的内容都可以用来访问列表中的内容，只要该索引属于您试图访问的列表。如果您试图使用列表之外的索引，例如`fruit_list[3]`，那么您将得到一个索引错误，告诉您该索引不在列表中。

这种索引的一个有用的特点是，我们不仅可以向前计数，就像我们对水果列表所做的那样，我们还可以向后计数。这意味着我们可以检查最后添加到列表中的项目，或者如果它们是有序的，那么最高/最低值是多少。例如，如果您创建了一个分数上升的列表，但您对第二大分数感兴趣，您可以这样访问它:

```
#create a list of scores
scores = [12,42,62,65,73,84,89,91,94]#extract the second highest score
second_highest_score = scores[-2]#print the result
print(second_highest_score)#out:
91
```

当然，在这样做的时候，不是也从 0 开始(这样会造成混乱),而是希望从-1 开始访问第一个条目，然后从希望访问的结尾开始进一步增加。

## 列表切片

使用索引来访问列表中的项目的一个好处是，我们还可以使用一个片一次访问多个元素。这一点很重要，因为片允许你通过使用符号`list[start_index: end_index]`来访问列表中的一系列条目，需要注意的是，片本身并不会返回结束索引。这些例子包括:

```
#second lowest to fifth lowest
print(scores[1:5])#print second lowest
print(scores[1:2])#print the fifth lowest to the highest
print(scores[5:])#print the third highest to the highest
print(scores[-3:])#print from beginning to end
print(scores[:])#print every 2nd 
print(scores[::2])#print the list in reverse
print(scores[::-1])#out:
[42, 62, 65, 73]
[42]
[84, 89, 91, 94]
[89, 91, 94]
[12, 42, 62, 65, 73, 84, 89, 91, 94]
[12, 62, 73, 89, 94]
[94, 91, 89, 84, 73, 65, 62, 42, 12]
```

几个不同的规则适用于此，因为:

*   当使用`[1:2]`打印第二个最低的项目时，由于最终索引不包含在结果中，因此仅返回一个项目
*   当打印时，我们没有指定一个结束索引，这就是为什么在第五个索引之后，包括第五个索引，整个列表都被打印出来
*   当打印`[::2]`时，使用第二个`:`允许我们指定索引之间的跳转，这就是为什么每第二个项目被显示

同样重要的是要注意，一个切片将总是返回一个列表，即使它只包含一个项目。这一点很重要，这样您就可以知道产生了什么类型的输出，以及我们可以用它来做什么。

## 查找项目的位置

从上面我们可以看到，当我们知道项目的索引时，我们可以访问项目，但如果我们只知道列表包含项目，但不知道它在哪里，那该怎么办？例如，在我们的水果列表中，我们知道我们的列表中有一个香蕉，但我们忘记了它在哪里？？我们可以使用`index()`方法找到该项目的位置，如下所示:

```
#find the index of banan
print(fruit_list.index("Banana"))#find the index of peach
print(fruit_list.index("Peach"))#out:
1
2
```

如果您忘记了项目的位置，或者如果您的列表中一个列表的顺序与另一个列表的顺序相关，这将非常有用。例如，如果一个分数列表链接到一个姓名列表，您可以找到该姓名的索引，然后使用该索引从另一个列表中访问他们的分数。

唯一的问题是，如果您拼错了项目或项目不在列表中，该方法将抛出一个错误，并停止代码运行。解决这个问题的一个简单方法是使用 if/else 语句，该语句可以使用:

```
if "Banana" in fruit_list:
    print("Banana is at index:", fruit_list.index("Banana"))
else:
    print("Banana not in list")#out:
Banana is at index: 1
```

## 易变性

关于列表的另一个重要的事情是它们是“可变的”,这仅仅意味着列表中的项目和它们的顺序是可以改变的。这可以包括改变单个结果，在列表中间插入新内容，甚至对它们进行排序。

要更改单个项目，我们需要使用该项目的索引，方法如下:

```
scores = [12,42,62,65,73,84,89,91,94]
#we can change the score at the second index#print the second lowest score
print("Original score:", scores[1])#reassign the score
scores[1] = 52#check the reassignment
print("Changed score:", scores[1])#out:
Original score: 42
Changed score: 52
```

这里我们知道项目的位置，但是您可以使用上面显示的索引方法来查找特定项目的位置。

我们还可以使用`append()`将新的分数添加到列表的末尾，或者如果我们想要在特定位置添加一个值，我们可以使用如下的`insert()`方法:

```
#we can add a new score at the end using the append function
print("Original scores", scores)#add new score
scores.append(67)#print the new scores
print("New scores", scores)#or add new scores in a specific position
scores.insert(3, 48)#print the newer scores
print("Newer scores", scores)#out:
Original scores [12, 52, 62, 65, 73, 84, 89, 91, 94]
New scores [12, 52, 62, 65, 73, 84, 89, 91, 94, 67]
Newer scores [12, 52, 62, 48, 65, 73, 84, 89, 91, 94, 67]
```

我们还可以使用`remove()`方法根据值从列表中删除值，或者使用`pop()`方法指定它们的位置，或者在极端情况下，我们可以使用`clear()`方法完全清除列表:

```
#we can remove a score from the list
print("Original scores", scores)#remove the score of 89
scores.remove(89)#print the new score
print("New scores", scores)#alternative methods for removal include:# the pop() method removes the specified index
# scores.pop(1)# If you do not specify an index the pop() method removes the last item
# scores.pop()# we can also completely clear the list
# scores.clear()#out:
Original scores [12, 52, 62, 48, 65, 73, 84, 89, 91, 94, 67]
New scores [12, 52, 62, 48, 65, 73, 84, 91, 94, 67]
```

## 其他有用的功能

列表在 Python 编程中非常有用，它们也是许多其他数据类型的基础。它们的一个有用的方面是它们的嵌入式方法和可以与它们一起使用的函数。

其中一种方法是使用`len()`函数相对容易地找到列表的长度:

```
print(len(scores))#out:
10
```

因此我们知道我们的数据集中有 10 个分数，这可以与循环遍历项目或访问数据中的特定索引结合使用。

当然，到目前为止，除了最初的实现之外，我们已经对`scores`列表进行了显著的修改。原始列表是按从小到大的顺序排列的(升序)。我们可以使用列表的其他功能来纠正这个问题，或者使用`sort()`方法，或者使用`sorted()`函数，如下所示:

```
#print current scores
print(scores)#we can assign the new sorted list to a new list as follows:
new_sorted_scores = sorted(scores)
print(new_sorted_scores)#or we can sort the list itself
scores.sort()
print(scores)#we can even sort it in descending order
scores.sort(reverse = True)
print(scores)
#or by using scores.reverse()#out:
[12, 52, 62, 48, 65, 73, 84, 91, 94, 67]
[12, 48, 52, 62, 65, 67, 73, 84, 91, 94]
[12, 48, 52, 62, 65, 67, 73, 84, 91, 94]
[94, 91, 84, 73, 67, 65, 62, 52, 48, 12]
```

其中我们是想要改变原始列表还是创建一个新列表将决定我们最终使用哪个功能。

最后，我们可能希望将列表添加到一起，我们可以简单地使用 python 的`+`功能，或者我们可以使用`extend()`方法将列表添加到现有列表，如下所示:

```
Names1 = ["Peter", "Geneva", "John"]
Names2 = ["Katie", "Suzie", "Scott"]#add lists together using +
added_names = Names1 + Names2
print(added_names)#add lists together by extending one list
Names1.extend(Names2)
print(Names1)#add the same list together by multiplying it by itself
double_names = Names2 * 2
print(double_names)#out:
['Peter', 'Geneva', 'John', 'Katie', 'Suzie', 'Scott']
['Peter', 'Geneva', 'John', 'Katie', 'Suzie', 'Scott']
['Katie', 'Suzie', 'Scott', 'Katie', 'Suzie', 'Scott']
```

这就是 Python 列表的完整指南！

这是探索数据结构及其在 Python 中的使用和实现的系列文章的第一篇。即将发表的文章将涵盖 Python 中的集合、元组、字典、链表、栈、队列和图形。为了确保您不会错过任何在发布时接收电子邮件通知的注册:

<https://philip-wilkinson.medium.com/subscribe>  

如果您喜欢您所阅读的内容，并且还不是 medium 会员，请随时使用下面的我的推荐代码注册 medium，以支持我自己和这个平台上的其他作者:

<https://philip-wilkinson.medium.com/membership>  

或者考虑看看我的其他媒介文章:

</an-introduction-to-sql-for-data-scientists-e3bb539decdf>  </git-and-github-basics-for-data-scientists-b9fd96f8a02a>  </london-convenience-store-classification-using-k-means-clustering-70c82899c61f> 