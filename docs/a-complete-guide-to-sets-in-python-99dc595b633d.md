# Python 中集合的完整指南

> 原文：<https://towardsdatascience.com/a-complete-guide-to-sets-in-python-99dc595b633d>

## 集合的关键特性、实现集合、访问项目、可变性和附加功能

![](img/b93726c48508cc0aa0536c69a22cd358.png)

照片由[利比·彭纳](https://unsplash.com/@libby_penner?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

## Python 中的集合

继列表和元组之后，集合是 Python 中经常遇到的另一种常见数据结构，在工作流中有多种用途。它们可以用于在单个变量中存储多个项目，如列表或元组，但主要区别是它们是无序的，不能包含重复值。这意味着，当您只想存储唯一值，不关心维护数据结构中项的顺序，或者想要检查不同数据源之间的重叠和差异时，它们会很有用。这种数据结构的主要特征是:

*   **可变:**一旦创建了集合，就可以对它们进行更改
*   **无序**:器械包内物品的顺序没有被记录或保存
*   **未索引**:因为它们是无序的，所以我们没有一个可以用来访问特定项目的索引
*   **不能包含重复值**:集合不允许包含相同值的多个实例

这些特征会影响它们在程序中的使用。例如，当您希望保持向集合中添加项目的顺序时，您可能不希望使用它们，但是当您只关心添加唯一的项目并希望节省内存空间时，您可能希望使用它们。

## 履行

要创建集合，我们可以使用两种主要方法:

*   使用`{}`将我们想要放入集合中的一组项目用逗号分隔开
*   使用可用于将其他数据结构转换为集合的`set()`函数

这可以通过以下方式实现:

```
#create a set using curly brackets
fruits = {"apple", "banana", "cherry"}#create a set using the set constructor
vegetables = set(("courgette", "potato", "aubergine"))#print the results
print(fruits)
print(type(fruits))print("\n")print(vegetables)
print(type(vegetables))#out:
{'apple', 'banana', 'cherry'}
<class 'set'>

{'aubergine', 'courgette', 'potato'}
<class 'set'>
```

由此我们可以看到，虽然我们可以使用`{}`符号从逗号分隔的原始输入中创建一个集合，但是我们已经使用了`set()`函数将一个元组改变为一个集合，这表明我们也可以将其他数据结构改变为集合。

我们还可以看到，在打印器械包时，它不一定按照输入数据的顺序出现(特别是蔬菜器械包)。这与这样一个事实有关，即它是一个无序的数据结构，所以项目不会总是以相同的顺序出现，所以我们不能以与列表相同的方式访问项目。

## 访问集合中的项目

因为集合中的条目是无序的，我们根本没有条目的索引，所以我们不能像访问列表或元组那样访问它们。因此，有两种主要的方法来检查一个项目是否在一个集合中。

第一种方法是简单地循环一个集合中的所有项目，以打印所有项目，然后检查它们和/或制定一个编程解决方案，以便在识别出所需项目时停止。这可以通过以下方式完成:

```
#use a loop to iteratre over the set
for x in fruits:
    print(x)#out:
cherry
apple
banana
```

这可能计算量很大，如果你必须用眼睛检查所有的值，那么如果你有一个非常大的集合，这可能需要很长时间。

另一种方法是简单地使用 python 中的`in`关键字来检查您感兴趣的值是否确实在集合中。这可以通过以下方式实现:

```
#or check whether the fruit you want is in the set
print("apple" in fruits)
#which acts the same way as if it were in a list#out:
True
```

在这种情况下，返回`True`,因为“苹果”在集合中。这比遍历整个集合要简单得多，并且节省了计算资源，因此可以用 if、elif 或 else 语句来触发其他代码。

## 易变性

虽然集合可能是可变的，因为它们可以被改变，但是集合中的实际值不能。这意味着，虽然您可以添加或删除集合中的项目，但您不能更改特定的项目，因为我们不能通过它们的索引来访问项目。

因此，我们可以使用几种不同的方法来改变设置。我们可以首先关注向现有集合添加项目。第一种方法是使用`add()`方法，该方法可用于向集合中添加特定的项目。第二种方法是使用`update()`方法，该方法可用于向现有集合添加另一个集合。最后，我们还可以使用`update()`方法将任何可迭代对象添加到集合中，比如元组或列表，但这只会保留唯一的值。这些可以通过以下方式实现:

```
#we can add using the add method
fruits.add("cherry")#check the updated set
print(fruits)#we can add another set to the original set
tropical = {"pineapple", "mango", "papaya"}
fruits.update(tropical)#print the updated set
print(fruits)#we can also use the update method to add any 
#iterable object (tuples, lists, dictionaries etc.)
new_veg = ["onion", "celery", "aubergine"]
vegetables.update(new_veg)#print the updated set
print(vegetables)#out:
{'apple', 'banana', 'cherry'}
{'mango', 'papaya', 'banana', 'pineapple', 'apple', 'cherry'}
{'aubergine', 'courgette', 'onion', 'potato', 'celery'}
```

另一方面，我们也可以从集合中删除我们不再需要的项目。为此，我们可以使用`remove()`方法从集合中删除一个特定的值。然而，这样做的问题是，如果集合中不存在该项目，那么这将引发一个错误，并停止您的代码运行。因此，我们也可以使用`discard()`方法，它不会引发错误，从而允许代码继续运行。当然，这些选项的选择将取决于您是否希望在项目是否在集合中时发生特定的动作！最后，我们也可以使用`pop()`方法，但是由于集合是无序的，我们不知道哪个条目将被删除！

```
#we can use the remove method
fruits.remove("apple")print(fruits)
#the issue with this is if the item does not 
#exist remove() will raise an error#or the discard method
fruits.discard("mango")
#this does not raise an errorprint(fruits)#finally we can also use the pop method
#but since this is unordered it will remove the last item
#and we also don't know which item will be removed
fruit_removed = fruits.pop()print(fruit_removed)
print(fruits)#finally we can clear the set using teh cleaer method
fruits.clear()
print(fruits)#or delete the set completely
del fruits#out:
{'mango', 'papaya', 'banana', 'pineapple', 'cherry'}
{'papaya', 'banana', 'pineapple', 'cherry'}
papaya
{'banana', 'pineapple', 'cherry'}
set()
```

我们还提供了`clear()`方法，如果你愿意，它可以完全清除集合，或者`del`函数，它可以完全删除集合！

## 附加功能

集合和其他数据结构的一个重要区别是它们不能包含重复值。例如，当我们希望最小化信息占用的空间时，我们不想要重复的信息，或者我们希望找到信息中包含的独特价值时，这是非常有益的。例如，如果我们想向列表中添加重复项，然后只提取唯一值，我们可以使用一个集合，如下所示:

```
cars = {"Ford", "Chevrolet", "Toyota", "Hyundai", "Volvo", "Ford"}print(cars)#out:

{'Toyota', 'Ford', 'Chevrolet', 'Hyundai', 'Volvo'}
```

该集合将只包含传递的项目中的单个值。

这对于我们何时想要将两个集合连接在一起具有重要的意义，并且有多种方法可以这样做。例如，如果我们想要合并集合并保留每个集合的所有唯一值，我们可以使用`union()`方法创建一个新的集合，或者我们可以使用`update()`方法更改一个现有的集合，如下所示:

```
set1 = {1, 2, 3}
set2 = {"one", "two", "three"}#we can use union to return a new set
#with all items from both sets
set3 = set1.union(set2)
print(set3)#or we can use update to insert items in set2 into set 1
set1.update(set2)
print(set1)#out:
{1, 2, 3, 'one', 'two', 'three'}
{1, 2, 3, 'one', 'two', 'three'}
```

另一种方法是只保留出现在两个集合中的副本，如果我们想创建一个新的集合，可以使用`intersection()`方法；如果我们想更新一个现有的 et，可以使用`intersection_update()`方法，如下所示:

```
fruits = {"apple", "banana", "cherry"}
companies = {"google", "microsoft", "apple"}#y creating a new set that contains only the duplicates
both = fruits.intersection(companies)print(both)#or keep only items that are present in both sets
fruits.intersection_update(companies)print(fruits)#out:
{'apple'}
{'apple'}
```

或者最后，我们可以做相反的事情，提取除了重复以外的任何内容，这样每个集合的值都是唯一的。当我们想要创建一个新的集合时，可以使用`symmetric_difference()`方法，或者当我们想要更新一个现有的集合时，可以使用`symmetric_difference_update()`方法。

```
fruits = {"apple", "banana", "cherry"}
companies = {"google", "microsoft", "apple"}#y creating a new set that contains no duplicate
both = fruits.symmetric_difference(companies)print(both)#or keep only items that are present in both sets
fruits.symmetric_difference_update(companies)print(fruits)#out:
{'cherry', 'microsoft', 'banana', 'google'}
{'cherry', 'microsoft', 'banana', 'google'}
```

因此，这是一个完整的 Python 集合指南！由此我们可以看出，集合是无序的、无索引的，并且不允许重复值。后者是一个重要的特征，因为当我们只想从某样东西中提取唯一的项而不是它们的多个实例时，可以使用它们，但是当我们想在数据集中保持某种顺序时，就不能使用它们。

这是探索数据结构及其在 Python 中的使用和实现系列的第二篇文章。如果您错过了 Python 中的第一个列表，可以在以下链接中找到:

</a-complete-guide-to-lists-in-python-d049cf3760d4>  

未来的帖子将涵盖 Python 中的字典、链表、栈、队列和图形。为了确保您将来不会错过任何内容，请注册以便在发布时收到电子邮件通知:

<https://philip-wilkinson.medium.com/subscribe>  

如果你喜欢你所阅读的内容，并且还不是一个媒体成员，考虑通过使用我下面的推荐代码注册来支持我自己和这个平台上的了不起的作者们:

<https://philip-wilkinson.medium.com/membership>  </an-introduction-to-sql-for-data-scientists-e3bb539decdf>  </git-and-github-basics-for-data-scientists-b9fd96f8a02a>  </london-convenience-store-classification-using-k-means-clustering-70c82899c61f> 