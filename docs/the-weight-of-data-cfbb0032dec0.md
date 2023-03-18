# 数据的权重

> 原文：<https://towardsdatascience.com/the-weight-of-data-cfbb0032dec0>

## 我们应该如何以及为什么评估我们数据的权重

![](img/e1faf2f2c660dc3caa5c5eef74896780.png)

马库斯·克里斯蒂亚在 [Unsplash](https://unsplash.com/s/photos/numbers?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

探索性数据分析是数据分析和数据科学的重要组成部分；也许，是最重要的部分，因为这是我们在进行真正的分析之前试图弄清楚我们的数据发生了什么的时候。
然后，我认为在数据科学和数据分析中需要理解的一件重要事情是，我们必须评估数据的权重。

数据有权重是什么意思？我想用一个例子来解释我的意思。

假设你是一名教练，你必须选择运动员参加奥运会的 100 米比赛；假设你不知道运动员，你只能根据数据选择他们；当然，你有一个记录每个运动员跑 100 米的时间的数据集。

为了简化这项研究(和本文)，我将创建一个数据集，其中只包含与四名运动员相关的数据(但在本文结束时，我们将概括为数千人)。让我们创建数据集:

```
import pandas as pd# initializing
data = {'Name':['Tom', 'Jack', 'Nick', 'John',
                'Tom', 'Jack', 'Nick', 'John',
                'Tom', 'Jack', 'Nick', 'John',],
        'Time':[20, 21, 19, 18,
                20, 100, 19, 18,
                21, 22, 21, 20]}#dataframing the data
df = pd.DataFrame(data)
```

现在，我们通常首先做的一件事是计算平均值。让我们开始吧:

```
#calculating mean values and sorting
b = df.sort_values('Name').groupby('Name').mean()b---------Name       Time
------------------
Jack       47.7                 
John       18.7                 
Nick       19.7                 
Tom        20.3
```

正如我们所看到的，杰克的平均值比其他运动员高得多。这意味着杰克似乎比其他人慢，不让他参加奥运会的诱惑可能很大。
但由于我们是“数据爱好者教练”,我们想更深入一点，不要只看平均时间就结束分析。例如，我们可以计算某个值在我们的数据集中“出现”了多少次。我们可以这样做:

```
#counting single values and sorting for index
c = df['Time'].value_counts().sort_index()
c------------18     2
19     2
20     3
21     3
22     1
100    1
Name: Time, dtype: int64
```

正如我们所看到的，我们有一个值 100，它只出现一次，如果我们深入一点，我们可以看到它与我们的朋友 Jack 相关联。现在，我们可能决定删除这个 100 值，它只出现一次，看看杰克是否还能被选中参加游戏。也许，如果我们可以获得更多的数据，我们可以发现杰克跑了一段腿受伤的路程，并有强烈的意愿结束这段路程(他结束了！)，因为如果他决定退出，他将失去一些资格分。

当然，除了散点图之外，还可以使用“value_counts()”代码来图形化地查看异常值，我认为这样做确实是个好主意。但是使用“value_counts()”确实有助于我们理解如何衡量我们的数据:例如，在这种情况下，这个 100 值(对我们的朋友 Jack 来说是唯一的值)可以被丢弃，我们可以在没有它的情况下执行即将到来的数据分析。

## 概括和结论

当然，这只是一个简单的例子——正如我在本文开头所解释的。但是让我们把数字增加一点。你有 100 个或者 1000 个运动员的数据怎么样？在这种情况下，仅仅用散点图来可视化数据可能不会帮助你理解 100 的值仅仅是与仅仅一个运动员相关联的唯一值；这意味着，在我看来，离群值必须用图形来处理，但也要用分析来处理。当然，最后，有时时间是唯一重要的东西:进行深入的分析研究需要时间；所以，尽量花时间分析数据。

*我们一起连线吧！*

[](https://federicotrotta.medium.com/)

*[***LINKEDIN***](https://www.linkedin.com/in/federico-trotta/)*(向我发送连接请求)**

**如果你愿意，你可以* [***订阅我的邮件列表***](https://federicotrotta.medium.com/subscribe)**这样你就可以一直保持更新了！***

**考虑成为会员:你可以免费支持我和其他像我一样的作家。点击 [***这里***](https://federicotrotta.medium.com/membership)**成为会员。****