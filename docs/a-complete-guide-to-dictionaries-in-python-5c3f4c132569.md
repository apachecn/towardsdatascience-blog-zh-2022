# Python 词典完全指南

> 原文：<https://towardsdatascience.com/a-complete-guide-to-dictionaries-in-python-5c3f4c132569>

## 什么是词典，创建它们，访问它们，更新它们，以及特殊的方法

![](img/94233099770dc2dbe90fb1aff8c43c0f.png)

Joshua Hoehne 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

## 什么是字典

继列表、集合和元组之后，字典是 Python 中的下一个内置数据结构。它们通常用于编程，为 Python 中许多不同库中更高级的结构和功能奠定了基础。它们的形式类似于实际的字典，其中的键(单词)有值(描述)，因此采用键:值结构。它们的主要特点是:

*   **可变**:一旦定义，就可以改变
*   **有序**:除非明确改变，否则它们保持它们的顺序
*   **可索引**:如果我们知道特定条目在字典中的位置(它们的键)，我们就可以访问它们
*   **无重复键:**它们不能在键值中包含重复项，尽管它们可以在值中包含。

字典的重要部分是它们的键-值结构，这意味着我们不是像在列表或元组中那样使用数字索引来访问条目，而是使用它们的键来访问它们的值。这样，当我们想要使用特定的关键字访问记录时，就可以使用字典，比如销售记录或登录。这比列表或元组占用更多的空间，但是如果您知道可以用来访问特定项的值的键，它可以允许更有效的搜索。

## 履行

由于键-值关系，实现字典比创建列表、集合或元组要复杂一些。在这方面有两种主要方式:

*   使用`{}`符号，但是当键和值被一个`:`分开时，比如`{key:value}`
*   使用`dict()`函数，但是只有当我们有一个包含两个条目的元组列表时

这里的关键是确保我们可以使用`key:value`结构创建一个字典，如下所示:

```
#create a dict using {}
new_dict = {"Name":"Peter Jones",
           "Age":28,
           "Occupation":"Data Scientist"}#create two lists
keys = ["Name", "Age", "Occupation"]
values = ["Sally Watson", 30, "Technical Director"]
#zip them together
zipped_lists = zip(keys, values)
#turn the zipped lists into a dictionary
new_dict2 = dict(zipped_lists)#print the results
print(new_dict)
print(type(new_dict))print("\n")print(new_dict2)
print(type(new_dict))#out:
{'Name': 'Peter Jones', 'Age': 28, 'Occupation': 'Data Scientist'}
<class 'dict'>

{'Name': 'Sally Watson', 'Age': 30, 'Occupation': 'Technical Director'}
<class 'dict'>
```

我们可以看到，我们已经能够使用这两种方法创建一个字典。

## 数据类型

我们在上面已经看到，像列表、元组和集合一样，字典可以包含多种不同的数据类型作为值，但是它们也可以将不同的数据类型作为键，只要它们不是可变的，这意味着您不能使用列表或另一个字典作为键，因为它们是可变的。我们可以将此视为:

```
mixed_dict = {"number":52,
             "float":3.49,
             "string":"Hello world",
             "list":[12, "Cheese", "Orange", 52],
             "Dictionary":{"Name":"Jemma",
                          "Age":23,
                           "Job":"Scientist"}}print(type(mixed_dict))#out:
<class 'dict'>
```

是有效的字典。

## 访问项目

访问字典中的条目不同于访问列表或元组中的条目，因为现在我们不是使用它们的数字索引，而是使用键来访问值。这意味着我们可以像以前一样使用`[]`符号，但主要有两种方式:

*   使用`[]`符号来指定我们想要访问其值的键
*   使用`get()`方法来指定我们想要访问的键的值

这两个方法的主要区别在于，如果键不在字典中，第一个方法会引发一个问题，而如果键不在字典中，第二个方法会简单地返回`None`。当然，方法的选择将取决于您想要对结果做什么，但是每种方法都可以按如下方式实现:

```
#the first way is as we would with a list
print(new_dict["Name"])#however we can also use .get()
print(new_dict.get("Name"))#the difference between the two is that for get if the key
#does not exist an error will not be triggered, while for 
#the first method an error will be
#try for yourself:
print(new_dict.get("colour"))#out:
Peter Jones
Peter Jones
None
```

以这种方式访问信息意味着我们不能有重复的键，否则，我们就不知道我们在访问什么。虽然在初始化字典时创建重复的键不会产生错误消息，但它会影响您访问信息的方式。例如:

```
second_dict = {"Name":"William",
              "Name":"Jessica"}print(second_dict["Name"])#out:
Jessica
```

我们可以在这里设置两个`"Name"`键，当试图访问信息时，它只打印第二个值，而不是第一个值。这是因为第二个密钥会覆盖第一个密钥值。

## 易变的

与列表和集合一样，与元组不同，字典是可变的，这意味着一旦创建了字典，我们就可以更改、添加或删除条目。我们可以用类似的方式来实现这一点，即访问单个项目并使用变量赋值(`=`)来改变实际值。我们也可以添加新的`key:value`对，只需指定一个新的键，然后给它赋值(或者不赋值)。最后，我们可以使用`update()`方法向现有字典添加一个新字典。这些都可以表现为:

```
#create the dictionary
car1 = {"Make":"Ford",
       "Model":"Focus",
       "year":2012}#print the original year
print(car1["year"])#change the year
car1["year"] = 2013#print the new car year
print(car1["year"])#add new information key
car1["Owner"] = "Jake Hargreave"#print updated car ifnormation
print(car1)#or we can add another dictionary 
#to the existing dictionary using the update function
#this will be added to the end of the existing dictionary
car1.update({"color":"yellow"})
#this can also be used to update an existing key:value pair#print updated versino
print(car1)#out:
2012
2013
{'Make': 'Ford', 'Model': 'Focus', 'year': 2013, 'Owner': 'Jake Hargreave'}
{'Make': 'Ford', 'Model': 'Focus', 'year': 2013, 'Owner': 'Jake Hargreave', 'color': 'yellow'}
```

因此，我们可以改变字典中包含的信息，尽管除了删除它之外，我们不能改变实际的键。因此，我们可以看到如何从字典中删除条目。

要从字典中删除条目，我们可以使用`del`方法，尽管我们必须小心，否则我们可能会删除整个字典！我们也可以使用`pop()`方法来指定我们想要从字典中删除的`key`。这种方法的一个扩展是`popitem()`方法，在 Python 3.7+中，该方法从字典中删除最后一项(在此之前，它是一个随机项)。最后，我们可以使用`clear()`方法从字典中删除所有值。这些可以通过以下方式实现:

```
scores = {"Steve":68,
         "Juliet":74,
         "William":52,
         "Jessica":48,
         "Peter":82,
         "Holly":90}#we can use the del method
del scores["Steve"]
#although be careful as if you don't specify 
#the key you can delete the whole dictionaryprint(scores)#we can also use the pop method
scores.pop("William")print(scores)#or popitem removes the last time 
#(although in versinos before Python 3.7 
#the removes a random item)
scores.popitem()print(scores)#or we could empty the entire dictionary
scores.clear()print(scores)#out:
{'Juliet': 74, 'William': 52, 'Jessica': 48, 'Peter': 82, 'Holly': 90}
{'Juliet': 74, 'Jessica': 48, 'Peter': 82, 'Holly': 90}
{'Juliet': 74, 'Jessica': 48, 'Peter': 82}
{}
```

## 附加功能

因为我们有一个不同于列表、元组或集合的结构，所以字典也有自己的功能来处理它。值得注意的是，您可以使用`keys()`方法从字典中提取所有键的列表，使用`values()`方法从字典中提取所有值的列表，使用`items()`方法提取`key:value`对的元组列表。最后，你可以使用`len()`函数来提取字典的长度。我们可以将此视为:

```
dictionary = {"Score1":12,
             "Score2":53,
             "Score3":74,
             "Score4":62,
             "Score5":88,
             "Score6":34}#access all the keys from the dictionary
print(dictionary.keys())#access all the values form the dictionary
print(dictionary.values())#access a tuple for each key value pair
print(dictionary.items())#get the length of the dictionary
print(len(dictionary))#out:
dict_keys(['Score1', 'Score2', 'Score3', 'Score4', 'Score5', 'Score6'])
<class 'dict_values'>
dict_items([('Score1', 12), ('Score2', 53), ('Score3', 74), ('Score4', 62), ('Score5', 88), ('Score6', 34)])
6
```

这使您能够检查一个条目是在键中，还是在值中，或者同时在两者中，同时还可以检查字典的列表。

因此，这是一本相当完整的字典指南。字典的独特性质是具有键:值结构，这增加了字典的存储容量，但是如果您知道与值相关联的键，它可以使检索更加准确，这使它成为存储分层信息或跨数据源的信息的有价值的结构。这样，它们为更复杂的数据存储方法奠定了基础，如 Pandas DataFrames、JSON 等。

这是探索数据结构及其在 Python 中的使用和实现系列的第四篇文章。如果您错过了列表、元组和集的前三个，您可以在以下链接中找到它们:

[](/a-complete-guide-to-lists-in-python-d049cf3760d4)  [](/a-complete-guide-to-sets-in-python-99dc595b633d)  [](/a-complete-guide-to-tuples-in-python-af76241e8b59)  

本系列的后续文章将涵盖 Python 中的链表、栈、队列和图形。为了确保您将来不会错过任何内容，请注册以便在发布时收到电子邮件通知:

[](https://philip-wilkinson.medium.com/subscribe)  

如果你喜欢你所阅读的内容，并且还不是一个媒体成员，考虑通过使用我下面的推荐代码注册来支持我自己和这个平台上其他了不起的作者:

[](https://philip-wilkinson.medium.com/membership)  [](/a-complete-data-science-curriculum-for-beginners-825a39915b54)  [](/an-introduction-to-sql-for-data-scientists-e3bb539decdf)  [](/git-and-github-basics-for-data-scientists-b9fd96f8a02a) 