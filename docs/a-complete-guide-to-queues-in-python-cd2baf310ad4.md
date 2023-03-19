# Python 中队列的完整指南

> 原文：<https://towardsdatascience.com/a-complete-guide-to-queues-in-python-cd2baf310ad4>

## 什么是队列以及如何用 Python 实现队列

![](img/949be9ddb0b18ab29154ce92d2ec6b47.png)

Melanie Pongratz 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

在编程术语中，队列是一种抽象数据类型，它存储项目被添加到结构中的顺序，但它只允许添加到队列的末尾，同时只允许从队列的前面移除。这样做时，它遵循先进先出(FIFO)数据结构。本质上，这就像你所期望的队列在现实中的行为一样，例如当排队进入商店时，那些先到的人会先到。然而与现实生活不同的是，我们以一种确保没有插队的方式来构建它！

Giphy 的 GIF

当我们希望将输出存储在一个结构中并保持其顺序，然后按照给定的顺序执行操作时，可以使用这些方法。编程中常见的例子包括为计算机的 CPU 调度作业、将打印机的作业排队以确保首先发送的作业被执行、为企业处理订单或处理消息。

记住这一点，知道我们希望数据结构做什么以及我们希望如何与它交互，我们就可以开始考虑如何在 Python 的实际数据结构中实现这一点。通常与此相关的主要方法包括:

*   enqueue(item):将项目添加到队列的末尾
*   dequeue():移除并返回队列前面的项目
*   peek():返回队列前面的项目，而不删除它
*   is_empty():如果队列为空，则返回 True
*   size():返回队列中有多少项

这意味着我们需要使用 Python 数据类型来构建它，这种数据类型是可变的，可以用来存储有序的项目集合。我们可以问自己的第一件事是，Python 中是否已经实现了一种数据类型可以为我们做到这一点？

Giphy 的 GIF

**一份名单！我们可以用一个列表！**

我们知道列表是可变的，它可以被改变，我们可以简单地从开始和结束添加和删除项目，使用列表的内置功能来实现队列。

在使用列表时，我们必须定义一个使用列表的类，但它只具有允许用户以我们想要的方式与之交互的特定功能。为此，我们可以使用 Python 的**类**结构来定义数据结构将包含的信息以及我们想要使用的方法。

接下来要做的第一件事是将**类**命名为`Queue`，并为这个新类创建构造函数。我们想要的是，一旦创建了对象的新实例，就初始化一个空列表，可以使用`.items`属性访问该列表。这方面的代码如下:

```
class Queue: #create the constructor
    def __init__(self): #create an empty list as the items attribute
        self.items = []
```

这意味着当我们创建一个`Queue`类的实例时，`item`属性将代表一个空列表。

然后我们可以开始向`Queue`类添加主要功能。我们希望能够做的第一件事是向我们的`Queue`添加一个项目。在这一点上，重要的是能够识别哪一端是我们从中获取数据的队列的前端，哪一端是我们可以向其中添加项目的队列的末端。

这暗示了每个过程将花费多长时间，因为在列表的开始处改变事物导致 O(n)的时间复杂度，因为我们随后必须改变列表中后续项目的所有索引以将所有内容向右移动。然而，如果我们操作列表末尾的内容，那么时间复杂度为 O(1 ),因为我们只在删除或添加项目时改变列表末尾的索引。

Giphy 的 GIF

然而，这并没有太大的关系，因为无论你选择从另一个函数中添加或删除哪一方，都仍然具有另一个时间复杂度。因此，记住这一点，我们可以实现`enqueue(item)`方法来将一个项目添加到我们的队列中。

```
 def enqueue(self, item):
        """
        Add item to the left of the list, returns Nothing Runs in linear time O(n) as we change all indices
        as we add to the left of the list
        """ #use the insert method to insert at index 0
        self.items.insert(0, item)
```

这只会将一个项目添加到我们队列的末尾(左侧)。在这种情况下，我们并不太关心我们在队列中添加了什么，只是我们可以添加它。当然，您可以向其中添加更多的功能，但是现在这样就可以了。

这意味着当我们想从队列类的列表右端移除项目时。在这样做的时候，我们需要注意一个相当简单的边缘情况，即我们的队列是否为空。在这种情况下，如果队列为空，我们可以简单地返回`None`,这样我们仍然会返回一些东西，程序不会抛出错误，但是如果队列不为空，我们可以返回队列中的“第一个”项并返回它。因为我们使用 Python 的内置 list 类，所以我们可以简单地使用`.pop()`方法，在 Python 3.6+中，该方法从列表中移除 ist 项并返回它。这在常数时间 O(1)内运行，因为我们只移除列表项，不影响任何其他索引。因此，这可以实现为:

```
 def dequeue(self):
        """
        Removes the first item from the queue and removes it Runs in constant time O(1) because we are index to
        the end of the list.
        """ #check to see whether there are any items in the queue
        #if so then we can pop the final item
        if self.items:

            return self.items.pop() #else then we return None
        else: return None
```

因此，我们实现了队列的主要功能，确保我们只在队列的一端添加项目，从队列的另一端移除项目。这将保持队列中项目的顺序，并确保我们按照它们被添加到队列中的顺序来使用这些项目。

然后，我们可以考虑在实际程序中帮助使用队列数据结构的其他补充方法。我们可以添加的第一个附加功能是允许用户查看下一个要从队列中删除的项目，而不需要实际删除它。它的实现将遵循与`dequeue()`方法相似的结构，但是我们可以暗示用列表中的最后一项来访问该项，而不是使用`.pop()`。这意味着当我们使用索引来访问条目时，我们的时间复杂度为 O(1 ),这使得它变得简单明了。

```
 def peek(self):
        """
        Returns the final item in the Queue without removal

        Runs in constant time O(1) as we are only using the index
        """ #if there are items in the Queue 
        if self.items:
            #then return the first item
            return self.items

        #else then return none
        else:
            return Non
```

我们还可以提供一种检查队列是否为空的方法。如果队列实际上是空的，这将简单地返回布尔值`True`，如果不是空的，则返回`False`。这也是在恒定时间内运行的，因为它只是检查`Queue`中的列表是否存在

```
 def is_empty(self):
        """
        Returns boolean whether Queue is empty or not Runs in constant time O(1) as does not depend on size
        """ return not self.items
```

我们可以创建一个返回队列大小的方法。这可以告诉我们队列的大小，就像它对于列表一样，并且告诉我们已经添加了多少项或者队列中还剩下多少项。

```
 def size(self):
        """
        Returns the size of the stack        Runs in constant time O(1) as only checks size
        """ #len will return 0 if empty
        #so don't need to worry about empty condition
        return len(self.items)
```

最后，我们要确保我们试图打印出一个`Queue`类的实例，它对于个人来说是可读的，既可以看到它是队列，也可以看到队列包含的内容。为此，我们使用该类的特殊的`__str__` dunder 方法来告诉解释器我们想要如何打印出该类的一个实例。在这种情况下，我们只想返回包含在堆栈中的整个列表，可以实现为:

```
 def __str__(self):
        """Return a string representation of the Stack"""" return str(self.items)
```

唷！那么把这些放在一起怎么样？最终产品看起来像这样:

此时，您可能会想为什么我需要知道这些？这种数据结构在编程中的常见应用包括:

*   在 CPU 中调度作业
*   图遍历算法
*   订单处理
*   打印机队列

当你在构建这些程序时，了解这种数据结构在 Python 中是什么样子，并为你提供能够处理这些挑战的功能，这是很好的。

这也可能出现在软件工程师或数据科学面试中，如要求您构建一个使用队列的打印机，或创建图形遍历算法，如深度优先搜索。

现在，你知道了如何用 Python 实现队列，也知道了它的一些用途！

这是探索数据结构及其在 Python 中的使用和实现系列的第七篇文章。如果您错过了 Python 中的链表、栈和字典的前三部分，您可以在以下链接中找到它们:

</a-complete-guide-to-linked-lists-in-python-c52b6cb005>  </a-complete-guide-to-stacks-in-python-ee4e2045a704>  </a-complete-guide-to-dictionaries-in-python-5c3f4c132569>  

本系列的后续文章将涉及链表、队列和图形。为了确保您不会错过任何内容，请注册以便在发布时收到电子邮件通知:

<https://philip-wilkinson.medium.com/subscribe>  

如果你喜欢你所读的，但还不是一个中等会员，那么考虑使用我下面的推荐代码注册，在这个平台上支持我自己和其他了不起的作者:

<https://philip-wilkinson.medium.com/membership>  

感谢您的阅读！