# Python 循环:如何在 Python 中迭代的完整指南

> 原文：<https://towardsdatascience.com/python-loops-a-complete-guide-on-how-to-iterate-in-python-b29e0d12211d>

## 利用 Python 中循环的力量

![](img/9ccaffc64dd03d8ac156d1b25782a6fb.png)

Photo by [愚木混株 cdd20](https://unsplash.com/@cdd20?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/loop?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

几个月前，我给 T2 写了一篇关于 Python 中的循环和语句的文章。这篇文章是为了让你理解循环和语句。

相反，在本文中，我想通过一些练习更深入地讨论使用循环的一些不同的可能性。

# 1.列表理解

我想给你看的第一件事是列表理解，以及为什么和什么时候使用它。首先，我建议你阅读我的[上一篇关于循环和语句的文章](/loops-and-statements-in-python-a-deep-understanding-with-examples-2099fc6e37d7)，因为你需要深入理解这些主题。

当我们想要基于列表中已经存在的值创建新的列表时，我们使用列表理解。举个例子吧。

假设我们有一个包含 10 个数字的列表:

```
#creating alist of 10 numbers
a = list(range(10))
```

我们希望两个创建两个新列表:

*   值小于 3 的一个
*   另一个值大于 3

对于传统的方法，我们必须编写一个 for 循环，它遍历列表“a ”,并将小于 3 的值附加到一个新列表中，将大于 3 的值附加到另一个新列表中；下面的代码可以做到这一点:

```
#creating a list of 10 numbers
a = list(range(10))#creating two new and empty new lists
y_1, y_2 = [], []#creating the cycle
for y in a:
    if y>3:
        y_1.append(y)
    else:
        y_2.append(y)#printing the new lists        
print(y_1)
print(y_2)>>> [4, 5, 6, 7, 8, 9]
  [0, 1, 2, 3]
```

使用列表理解方法，我们可以简单地在一行代码中声明一个新列表；就这样:

```
#creating a list of 10 numbers
a = list(range(10))#creating a new list with a condition
c = [x for x in a if x>3]#creating another new list with a condition
d = [x for x in a if x<=3]#printing the new lists
print(c)
print(d)>>>

  [4, 5, 6, 7, 8, 9]
  [0, 1, 2, 3]
```

我相信这个例子很好地展示了当我们想要基于现有的列表创建新的列表时，使用列表理解是多么容易。现在，想一个你可能有两个以上条件的情况；例如，您可能想要基于现有的列表创建 5 个列表…您可以真正了解使用列表理解比使用带有“if”、“elif”、“else”的经典 for 循环节省了多少时间和代码。

抱歉，现在我意识到我还没有引入‘elif’结构。让我们快速看一下:

```
#creating a list in a range
a = list(range(30))#creating 3 empty lists
y_1, y_2, y_3 = [], [], []#creating the cycle
for y in a:
    if y>=0 and y<10:
        y_1.append(y)
    elif y>=10 and y<20:
         y_2.append(y)
    else:
        y_3.append(y)#printing the lists        
print(y_1)
print(y_2)
print(y_3)>>> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
  [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
```

因此,“elif”构造允许我们在第一个“if”构造之后获取数据。

# 2.对于字典中的循环

字典是一个非常有用的“数据容器”，因为每个元素都有一个“键”和一个“值”。例如，假设我们想要存储一些房屋的价格。我们可以创建一个这样的字典:

```
#my dictionary
house_prices = {'house_1':100000, 'house_2':250000, 'house_3':125000}
```

这里,“key”是字符串部分,“value”是数字。例如，在这些情况下，我们可以遍历键或值。例如，假设我们想打印所有的值，我们可以这样做:

```
#accessing ant printing the values
for values in house_prices.values():
    print(values)>>> 100000
  250000
  125000
```

请注意，在字典中,“值”甚至可以是字符串。比如，假设我们要存储一些人的名字和姓氏；然后，我们希望将姓名保存在一个列表中并打印该列表:

```
#creating the dictionary
people = {'Jhon':'Doe', 'Mick':'Brown', 'James':'White'}#creaitng an empty list
my_list = []#accessing the values and appending to the list
for keys in people:
    my_list.append(keys) #printing the list
print(f'people names are:{my_list}')>>>
  people names are:['Jhon', 'Mick', 'James']
```

你能猜到如何简化这个符号吗？哦，太好了！使用列表理解！

```
names = [name for name in people]print(names)>>> ['Jhon', 'Mick', 'James']
```

当然，字典的值也是如此:

```
surnames = [surname for surname in people.values()]print(surnames)>>> ['Doe', 'Brown', 'White']
```

现在，有了字典，我们可以创造许多美丽的记录。假设我们已经记录了一些房屋的数据，并且我们知道房屋的坐标、范围、价格以及它是否有一些土地。我们可以将所有这些数据存储在字典中，并根据它们创建一个列表，如下所示:

```
#creating an empty list
estates = []#storing the house data and appending to the list
e1 = {'coordinates':[12.3456, -56.7890], 'extension':2000,  'has_land':False, 'price':10}estates += [e1] #appending to the list#storing the house data and appending to the list
e2 = {'coordinates':[-8.9101,  60.1234], 'extension':12000, 'has_land':False, 'price':125}estates += [e2] #appending to the list#storing the house data and appending to the list
e3 = {'coordinates':[45.6789,  10.3456], 'extension':100,   'has_land':True,  'price':350}estates += [e3] #appending to the list
```

现在，在我们将字典添加到“estates”列表后，我们可以遍历它并打印如下信息:

```
for i, e in enumerate(estates):
    print(f'\nHouse #:{i}')
    print(f"Coordinates: {e['coordinates']}")
    print(f"Extension (mq): {e['extension']}")
    print(f"Has land?:{'Yes' if e['has_land']==True else 'No'}")
    print(f"Price: {e['price']}K $")>>> House #:0
  Coordinates: [12.3456, -56.789]
  Extension (mq): 2000
  Has land?:No
  Price: 10K $

  House #:1
  Coordinates: [-8.9101, 60.1234]
  Extension (mq): 12000
  Has land?:No
  Price: 125K $

  House #:2
  Coordinates: [45.6789, 10.3456]
  Extension (mq): 100
  Has land?:Yes
  Price: 350K $
```

请注意，你不能写`print(f'Coordinates:{e['coordinates']}')`，否则会产生一个`f-string:unmatched '['`错误。这是因为，在这种情况下，我在“f”后面和“坐标”中都使用了一个“”。如果像上面的代码一样在“f”后面使用双“，”就不会有任何问题(请参考[关于栈溢出的讨论](https://stackoverflow.com/questions/67540413/f-string-unmatched-in-line-with-function-call))。

# 3.while 循环

我发现虽然周期很容易理解，但可能很难看到它们的真正用途。

顾名思义，当某个特定的条件被满足时，一个 while 循环起作用。一个典型的例子是这样的:

> 打印`i`只要`i`小于 6

我们可以用下面的代码来做这件事:

```
i = 1
while i < 6:
    print(i)
    i += 1>> 1
 2
 3
 4
 5
 6
```

我认为这个例子可能有点令人困惑，因为我们将“I”设置为等于 1，所以人们可以说:“嗯，而“i <6” is obvious since we set i=6; so, what’s the point?”

While loops are very useful when we have a condition that dynamically changes, for example over time, and I believe that it can be better understood with a practical example.

Consider for example the website “ [worldometers](https://www.worldometers.info/) ”。这个网站实时收集世界统计数据。

让我们假设你想得到世界人口的数据；所以你写一个机器人来获取这些数据。分析家说，世界人口很快将达到 80 亿；所以，你的 bot 可以有一个`while`周期，门槛 80 亿；虽然世界人口比这个门槛要少，但我们可以越过它；当人口达到 80 亿时，我们可以`print(f'we are 8 billions in the world!!')`

# 结论

我希望我给了你一个很好的指导，让你在不同的情况下深刻理解循环。现在是你做一些循环练习和迷你项目的时候了！

**剧透提示**:如果你是一个新手或者你想学习数据科学，并且你喜欢这篇文章，那么考虑一下在接下来的几个月里**我将开始辅导像你一样有抱负的数据科学家**。我会在接下来的几周告诉你我什么时候开始辅导，如果你想预订座位……[*订阅我的邮件列表*](https://federicotrotta.medium.com/subscribe) *:我会通过它和在接下来的文章中传达我辅导之旅的开始。*

考虑成为会员:你可以免费支持我和其他像我一样的作家。点击 [*这里*](https://federicotrotta.medium.com/membership) *成为会员。*

*我们一起连线吧！*

[*中等*](https://federicotrotta.medium.com/)

[*LINKEDIN*](https://www.linkedin.com/in/federico-trotta/)*(给我发送连接请求)*

[推特](https://twitter.com/F_Trotta90)