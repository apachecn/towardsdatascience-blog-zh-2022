# 了解 Python 集合

> 原文：<https://towardsdatascience.com/understanding-python-sets-9df4dc71e1ca>

## Python 中一个未被充分利用的类，因为列表不能解决所有问题

![](img/2d8c77e6a7369422dd17bde00ce5c464.png)

由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的 [CHUTTERSNAP](https://unsplash.com/@chuttersnap?utm_source=medium&utm_medium=referral) 拍摄

Python 集合是大多数人在早期学习 Python 时学到的东西，但有时会忘记在某些地方它可能比列表更有用。列表得到了所有的关注，可能没有在适当的上下文中使用，但在这篇文章中，我们将强调什么是集合，用集合论分析数据集的方法，以及这将如何应用于数据分析。

## 什么是集合？

在 Python 上下文中，集合是一种容器类型，包含唯一且不可变的元素。它的存储也没有任何特定的顺序。知道集合和列表的区别的关键是提到的前两个属性，*和 ***不可变*** 。任何集合都不能包含具有相同值且类似于元组的多个元素。一旦创建了集合，就不能在其中修改项目。使用集合数据类型时的另一个关键方面是，与列表或数组不同，它们是无序的，或者每个元素不与集合中的唯一索引或位置相关联。了解这一点很重要，因为当创建集合时，每个项目的顺序永远不会成为集合的特征。set 数据类型的另一个主要特性是集合的参数可以是可迭代的，这意味着在创建集合时可以给它一个列表或一个项目数组。最后，与其中的对象必须是相同类型的一些数据类型不同，集合可以包含不同类型的项，如字符串和数字类型。*

## *集合操作(或者你能？？？)*

*请记住，当我们之前提到集合是不可变的时，这使得可以在集合上完成的操作和操纵的类型与列表之类的东西相比非常有限。可以在集合上完成的一些非常有用的事情依赖于集合论和可以应用的各种逻辑运算。这将包括应用不同集合之间的联合、交集和差异。*

*![](img/3c36514fe888dbf126d1f55d9c9bd77b.png)*

*[Tatiana Rodriguez](https://unsplash.com/@tata186?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片*

## *联合*

```
*#Animals that eat meat
meateaters = ('dogs', 'humans', 'lions', 'tigers', 'monkeys')#Animals that eat plants
planteaters = ('humans', 'sheep', 'cows', 'monkeys', 'birds')#Notices that we can use the | or the x1.union(x2) logic to apply union
eaters = meateaters|planteaters
eaters = ('dogs', 'humans', 'lions', 'tigers', 'monkeys''sheep', 'cows','birds')#OR
eaters2 = meateaters.union(planteaters)
eaters2 = ('dogs', 'humans', 'lions', 'tigers', 'monkeys''sheep', 'cows','birds')#Also note that although humans and monkeys were mentioned multiple times it only stores a single instance of each in the final union*
```

*在上面的联合示例中需要注意的一点是，对集合执行联合操作有两种不同的方式。两者之间的一个关键区别是|操作符要求两个项目都是集合类型。x.union 方法可以应用于任何 iterable 参数，然后在应用 union 操作之前将其转换为集合。当使用可能在列表或其他数据类型中的项目时，这提供了灵活性，并且仍然在对象之间应用联合逻辑。*

*![](img/81c37d777f3519ebbf11521c133ca24a.png)*

*西尔维·沙伦在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片*

***差异***

*与联合类似，差异运算将查看两个不同的集合，并找出集合之间的差异。在这个过程中，它将创建一组新的唯一项目。以上面肉食动物和食草动物之间的例子为例，差异运算将把人类和猴子排除在外，因为它们同时存在于两个集合中。*

```
*#Animals that eat meat
meateaters = ('dogs', 'humans', 'lions', 'tigers', 'monkeys')#Animals that eat plants
planteaters = ('humans', 'sheep', 'cows', 'monkeys', 'birds')#Animals that are not omnivores 
nonomnivores = meateaters.difference(planteaters)
nonomnivores = ('dogs', 'lions', 'tigers', 'sheep', 'cows', 'birds')#The (-) minus operator can also be used in the same way
nonomnivores2 = meateaters - planteaters
nonomnivores2 = ('dogs', 'lions', 'tigers', 'sheep', 'cows', 'birds')#Also more than two sets can be applied with similar logic
x = difference(y, z)
#Or
x - y - z*
```

*![](img/09317bee68e4e62503f302c874654650.png)*

*由[思远](https://unsplash.com/@jsycra?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄*

## *十字路口*

*可以对集合进行的最后一个常见操作是应用交集逻辑。这将查看两个不同的集合，然后识别交集，或者作为输出存在于两个集合中的事物。下面的代码是一个将交集运算应用到食草动物和食肉动物的例子中的例子。*

```
*#Animals that eat meat
meateaters = ('dogs', 'humans', 'lions', 'tigers', 'monkeys')#Animals that eat plants
planteaters = ('humans', 'sheep', 'cows', 'monkeys', 'birds')#Animals that eat both
omnivores = meateaters.intersection(planteaters)
omnivores = ('humans', 'monkeys')#Alternative way to execute is using the & character
omnivores2 = meateaters & planteaters
omnivores2 = ('humans', 'monkeys')*
```

*![](img/4fd7724f99b8819b5a9f0b257e85ffd3.png)*

*照片由[Firmbee.com](https://unsplash.com/@firmbee?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄*

## *为什么这很重要？*

*关于编程的终极问题是，那又怎样？通过了解器械包，可以分析物品的属性以及它们与其他物品的关系。一个很好的例子是查看对象的属性，这些对象可能被保存为一个集合，甚至是一个列表。应用上面的三个操作将允许一个简单的方法来评估是什么使不同的项目在几个上下文中的属性相似或不同。*

*通过示例进行讨论就像是对通过链接连接的数据页面构建一个页面等级类型的分析。如果项目与各种属性相关联，应用集合论中的逻辑(或利用属性集)，可以很容易地应用并集、差集和交集运算来更深入地了解数据集。理想情况下，您可以比较前 N 个页面，然后使用它们属性之间的联合进行识别，以了解它们之间的共同点。通过查看顶部 N 和底部 N，然后理解它们之间的差异和/或交集，可以应用相同的逻辑。这无疑有助于数据处理步骤，并为那些对数据科学模型非常重要的功能提供支持。*

*另一个例子是在数据科学中应用相似性模型，最常见的相似性方法是余弦相似性，但通过利用集合论，另一个强大的相似性度量是 Jaccard 相似性。它通常用于在文本挖掘和推荐服务中应用相似性模型。这主要是因为 Jaccard 相似性可以用集合符号写成:*

> ***J(A，B) =** |A∩B| / |A∪ B|*

```
*import numpy as np#Introduce two vectors (which could represent a variety of parameters)
a = [0, 1, 2, 5, 6, 8, 9]
b = [0, 2, 3, 4, 5, 7, 9]#Define Jaccard Similarity functiondef jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

#Find Jaccard Similarity between the two sets 
jaccard(a, b)#Result
0.4*
```

*玩弄集合论思想和利用集合数据将有助于从不同角度处理问题，并对数据分析超级强大的数学子集给予更广泛的欣赏。*