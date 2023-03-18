# 使用 Python Map-Reduce 函数解决了 5 个编码问题

> 原文：<https://towardsdatascience.com/5-coding-problems-solved-with-python-map-reduce-functions-c744193193ff>

## [面试](https://towardsdatascience.com/tagged/interviewing) | [算法](https://towardsdatascience.com/tagged/algorithms) | [Python 编程](https://towardsdatascience.com/tagged/python-programming)

## 厌倦了使用显式循环？试试 Python 的 map 和 reduce 函数吧。

![](img/b5c032295642695bb15a559720835988.png)

[照片由 Ylanite Koppens 在 Pexels 上拍摄](https://www.pexels.com/photo/pumpkins-3036364/)

## 建议的点播课程

*你们很多人联系我要有价值的资源* ***钉 Python 编码面试*** *。下面我分享几个* ***课程/平台*** *我强烈推荐练习完本帖算法后继续锻炼:*

*   [**Python 数据工程纳米学位(uda city)**](https://imp.i115008.net/zaX10r)**→***优质课程如果你致力于从事数据工程的职业，*
*   [***Python 高级编码问题(StrataScratch)***](https://platform.stratascratch.com/coding?via=antonello)***→****我找到的准备 Python 的最佳平台& SQL 编码面试到此为止！比 LeetCode 更好更便宜。*
*   [**用 Python 练习编码面试题(60+题)**](https://datacamp.pxf.io/DV3mMd) **→** *列表、数组、集合、字典、map()、filter()、reduce()、iterable 对象。*

*还不是中等会员？考虑与我的* [***推荐链接***](https://anbento4.medium.com/membership) *签约，以获得 Medium 提供的所有内容，价格低至每月 5 美元***！**

# *为什么要用 map()和 reduce()？*

*你有没有发现自己在想:*“如果有一种不使用显式循环的方法来解决这个编码问题会怎么样？如果我可以用一个内衬代替循环，我的解决方案会有多优雅？”**

*我知道，对吧？这是我在某个时刻问自己的那种问题，在用一个蛮力解决了几十个算法之后… *“一定有更好的办法，”*我想。*

*然后一个顿悟，一个突破…一个真正的游戏改变者:我发现了 Python 的`map()`和`reduce()`函数。*

*我能听到有人说:*“你又来了……又一个把函数式编程当灵丹妙药卖的……*`map()`*`reduce()`*早就该撤了。还有更好的方法，更大蟒的！”。***

**有几分真实，但是我喜欢把`map()`和`reduce()`想象成 SQL:它很老了，不再流行了，但是它很管用，所以人们一直在使用它。**事实上，** **这些方法在处理可重复项时特别方便，无需编写循环。****

**所以请原谅我，在本文中，我将解释这两种方法的优缺点，并与您分享用 `**map()**` **或** `**reduce()**`解决的 **5 Python 编码问题。我还将提供一个*替代*解决方案，来说明这种方法对于消除循环是有用的，但不是必不可少的。****

**但首先，一点理论…**

# **如何使用 Python 的地图( )**

**引用[文档](https://docs.python.org/3/library/functions.html#map)，`map()` *“返回一个迭代器，该迭代器将函数应用于 iterable 的每一项，产生结果”*。该函数的语法如下:**

```
****map**(*function*, *iterable*, *...*)**
```

*   ***函数*参数可以是任何 Python 可调用的，如**内置函数**、**方法**、**自定义函数** (UDF)、 **lambda 函数**。**
*   **该函数对一个或多个 *iterables* 应用转换，如**字典**，**列表**，**元组**返回地图对象。**

**为了更好地理解`map()`，假设你希望**计算 10 个圆的面积**。因为公式是 **A = π r** ，π是已知常数，所以你需要的只是一个半径列表( **r** ):**

```
**# Compute the area of 10 circles starting from their radius expressed in meters.# Formula --> A = π r²**from math import pi****radius =** [3, 5, 2, 8, 18, 22, 12, 6, 9, 15]# Using lambda function:**list(map(lambda x: pi*(x**2), radius))****Output:** [28.27, 78.53, 12.56, 201.06, 1017.87,1520.53, 452.38, 113.09, 254.46, 706.85]# Alternative - Using UDFdef area_of_circle(r):
    return pi*(r**2)**list(map(area_of_circle, radius))****Output:** [28.27, 78.53, 12.56, 201.06, 1017.87,1520.53, 452.38, 113.09, 254.46, 706.85]**
```

**如您所见，调用`map()`将匿名函数或 UDF 应用于`radius`中的所有项目，并且(*来自 Python 3.x* )返回一个迭代器，该迭代器产生每个圆的面积。然而，为了显示输出，你需要在它上面调用`list()`。**

## ****地图的主要优势()****

**如果有人(*特别是在采访*时)问你为什么在解决编码问题时选择使用`map()`，这里有一些原因:**

*   **由于 `map()`是用 C 语言编写的，它可以被认为是高度优化的，特别是当它的内部循环与常规 Python `for`循环比较时。**
*   **使用`map()`，一次评估一个条目(*而不是像 for 循环*那样在内存中存储完整的 iterable)。这导致了**内存消耗的显著减少**。这就是为什么`map()`返回一个*迭代器对象*而不是另一个*可迭代*像 list 的主要原因。**
*   **`map()`与 lambda 函数和用户定义的函数配合得很好，这反过来意味着它可以用于对 *iterables* 应用复杂的转换。**

## **地图的主要缺点( )**

**同样，面试官可能想知道你是否能想到使用这种方法的缺点。例如:**

*   **事实上`map()`返回一个**地图对象**也是一个缺点，因为**你总是需要在上面调用** `**list()**` **才能使用函数**的输出，这并不是最性感的语法…**
*   **很快，Lambda 函数和 UDF 对你的同事来说会变得很难解释。出于这些原因，由`map()`提供的功能几乎总是可以被替换为**列表理解**或**生成器表达式**，它们被认为更具可读性和 Pythonic 性。**

**既然你已经理解了如何使用`map()`，下面的编码挑战将帮助你测试你的能力。**

# **用地图解决的问题**

## **# 1.排序数组的正方形**

```
****Output:**
[4, 9, 9, 49, 121]
[4, 9, 9, 49, 121]**
```

**一个预热编码问题，它准确地显示了何时使用`map()`是有意义的:一个简单的转换，需要应用于 iterable ( *整数列表* `nums`)和 t **的每一项，产生相同数量的转换项**。**

**注意，`solution_map()`不需要调用`list()`来评估地图对象。这是因为`sorted()`函数(默认为*)返回指定可迭代对象的排序列表。***

****备选*解决方案(`solution_lc`)显示了如何使用**列表理解**实现相同的结果。考虑到任务的简单性，可读性方面的改进相当有限。***

## **# 2.反转字符串中的单词**

```
****Output:**
uoY lliw teg doog ta gnivlos gnidoc smelborp
uoY lliw teg doog ta gnivlos gnidoc smelborp**
```

**一个有趣的编码问题，因为您需要找到一种方法来反转输入字符串，保留单词顺序。这意味着您需要首先用`s.split()`创建一个 iterable，它将输出一个单词列表，然后用`x[::-1]`反转它们，最后用`" ".join()`重新创建一个句子。**

**尽管很有效，但`solution_map()`看起来有点乱(*因为这里* `*list()*` *必须被调用来访问颠倒的单词*)，并且需要一段时间来理解`map()`内部的内容。**

***备选*解决方案(`solution_genx()`)，使用**生成器表达式** ( *的语法类似于列表综合，但使用* `*()*` *括号*)提供了对`map()`的自然替换。生成器表达式在内存消耗方面和`map()`一样高效，并且经常使你的代码更具可读性。**

## **# 3.人数最多的至少是其他人的两倍**

```
****Output:**
3
3**
```

**当我试图自己解决这个编码问题时，使用`filter()`和`map()`功能的组合对我来说比使用`enumerate()`更自然，就像在*替代*解决方案中一样。**

**尽管`filter()`不是本文的主题，但这是另一个非常受欢迎的方法，它是从*函数编程*中派生出来的，可用于根据条件过滤一个可迭代的对象(在本例中为*，目标是用* `num != max(nums)`从数组中排除最大的整数)，然后在过滤后的列表`filter_max()`中调用`map()`。**

**然而，我不得不承认使用`enumerate()`导致了一个更干净的解决方案，它允许使用索引，而不必调用三个函数(即`filter()`、`all()`和`map()`)，这也使得它非常优雅。**

# **如何使用 Python 的 reduce()**

**首先，值得一提的是，`map()`是*内置的*函数，`reduce()`需要从`functools`模块导入。**

**该函数的语法如下:**

```
****functools**.**reduce**(*function*, *iterable*[, *initializer*])# Note that you can pass a third parameter named **initializer**.
# When provided, reduce() will feed it to the first call of function # as its first argument.**
```

**与`map()`类似，`reduce()`接受一个*可迭代*并对其应用一个函数。然而，`reduce()`从左到右将函数应用于项目，以便**将 iterable 减少为单个值**。**

> **因此，`map( )`和`reduce( )`的主要区别在于，前者意在输出与 iterable 中相同数量的项目，而后者将 iterable 中的项目减少到一个值。**

**使用`reduce()`最直接的方式是执行聚合，如求和或计数。例如，假设你想计算偶数的数量**是一个列表:****

```
****# Count the number of even numbers present in the values list.values = [13, 8, 12, 57, 13, 81, 10]# Using lambda function:**reduce(lambda x, y: x if y % 2 else x + 1, values, 0)******
```

****上面的 lambda 函数检查每个数字是否能被 2 整除而没有余数，用这种方法测试它是偶数还是奇数。****

****当余数为零时[if 语句评估为假，因此]计数增加 1(*请注意，初始化器已被设置为零，因为变量* `*x*` *充当计数器*)。****

## ****reduce()的主要优势****

****`reduce()`功能带来了与`map()`功能相似的好处，例如:****

*   ****因为`reduce()`也是用 C 编写的，所以它的内部迭代器比标准 Python for 循环要快。****
*   ****对于需要多行代码的问题，`reduce()`提供更简洁的解决方案并不罕见。*正因如此，* `*reduce()*` *是面试时的有力工具。*****

## ****reduce()的主要缺点****

****然而，使用该功能的主要缺点是:****

*   ****`reduce()`会给代码增加额外的计算时间，因为这个函数直到处理完整个 *iterable，*才会返回输出，然后无法实现**短路评估**。****
*   ****`reduce()`在使用复杂的 **UDF** 或 **lambda 函数**时会损害代码的可读性，所以只要有可能，最好选择专用的内置函数。****

****既然你也知道如何使用`reduce()`，让我们来练习一下你所学的。****

# ****使用 Reduce 解决了问题****

## ****# 4.数组乘积的符号****

```
******Output:**
-1
-1****
```

****计算数组中所有值的乘积是用`reduce()`执行的完美任务，我鼓励你在面试时使用它，因为在这种特定情况下，你实际上需要遍历完整的 *iterable* 、*、*，这样*短路评估*就不适用。****

****然而，从 **Python 3.8** 中，您可以简单地从`math`包中导入`prod`函数，以获得相同的结果，并且语法更加易读，如备选解决方案所示。****

## ****# 5.两个数组的交集****

```
******Output:**
[3, 3, 5]
[3, 3, 5]****
```

****最后一个编码问题是如何在字典中使用`reduce()`的例子(特别是在`collections.Counter`中)。我发现`solution_red()`相当优雅，因为它避免了显式循环，并且利用了`elements()`方法返回最终数组。****

****另一方面，另一种解决方案(`solution_ext`)确实使用了 for 循环，感觉有点小众，除非你确切地知道你在用`extend()`做什么。****

# ****总结****

****编写更紧凑和优雅的代码的能力，是面试官和其他开发人员真正欣赏的品质。****

> ****编写更紧凑和优雅的代码的能力，是面试官和其他开发人员真正欣赏的品质。****

****一个优秀的 Python 开发人员是能够以简洁的方式提供智能解决方案的人。效率和速度，在业内确实算数。****

****为了朝这个方向迈出一步，在本文中，您已经学习了 Python 的`map()`和`recuce()`函数是如何工作的，以及如何使用它们来处理 *iterables* ，而无需编写显式循环。****

****然后，你也实践了你所学到的，解决了 5 个需要在*可迭代*上应用转换的编码问题。****

****最终，通过不同的解决方案，您明白了有多种(*通常甚至更高效的*)方法可以替代 for 循环。****

# ****来源****

*   ****[*Python 的 map():无循环处理 Iterables*](https://realpython.com/python-map-function/)****
*   ****[*Python 的 reduce():从函数式到 Python 式*](https://realpython.com/python-reduce-function/)****
*   ****[*结合实例理解 Python Reduce 函数*](https://melvinkoh.me/understanding-the-python-reduce-function-with-examples-ck7mzz8l200na8ss1ogdvw5c5)****
*   ****[*LeetCode 问题 977*](https://leetcode.com/problems/squares-of-a-sorted-array/#:~:text=Squares%20of%20a%20Sorted%20Array%20%2D%20LeetCode&text=Given%20an%20integer%20array%20nums,1%2C0%2C9%2C100%5D.)****
*   ****[*LeetCode 问题 557*](https://leetcode.com/problems/reverse-words-in-a-string-iii/)****
*   ****[*LeetCode 问题 747*](https://leetcode.com/problems/largest-number-at-least-twice-of-others/)****
*   ****[*LeetCode 问题 1822*](https://leetcode.com/problems/sign-of-the-product-of-an-array/)****
*   ****[*LeetCode 问题 350*](https://leetcode.com/problems/intersection-of-two-arrays-ii/)****