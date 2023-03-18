# 如何处理数据科学中的缺失值

> 原文：<https://towardsdatascience.com/how-to-deal-with-missing-values-in-data-science-9e5a56fbe928>

## 处理 DS 项目中缺失值的三种实用方法

![](img/de98e1c4a4101ecfa228b6070398304e.png)

Pierre Bamin 在 [Unsplash](https://unsplash.com/s/photos/missing?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

W 在处理现实世界的数据时，您可能经常会发现数据框中缺少一些值。发生这种情况有几个原因，例如:

*   一些测量值可能会丢失
*   缺乏信息
*   抄本错误

那么问题来了:如何处理缺失数据？我们可以接受 0 作为缺失数据的值吗？我们可以删除丢失数据的行吗？

在本文中，我将向您展示处理缺失数据的三种方法，并回答这些问题。

# 1.提问

当你在数据集中发现缺失值时，你要做的第一件事就是**提问**，因为，通过提问，你将理解问题；理解问题是数据科学项目最重要的任务:如果我们不理解问题，我们就不能提供价值。

如果有人向你提供数据，问他们这样的问题:

*   你从哪里得到的数据？
*   为什么会有缺失值？
*   这些特征意味着什么？我可以接受 0 作为这些缺失数据的值吗？

如果你自己得到数据，问自己同样的问题。

还有， **Google 了很多**。搜索您拥有的缺失数据的参考值，并尝试了解您可能会为您的缺失数据赋予哪个值(请记住:即使 0 也是一个值！！).

此外，如果可以的话，**尝试联系该领域的专家**。例如，如果您正在处理一个医疗数据集，请联系医生(可能是您的医生！)并对你拥有的数据提出问题，尤其是对你缺失的数据。

# 2.删除行/列

在某些情况下，我们可以删除缺少值或 Nan(Nan =不是一个数字；甚至可以是类似“未测量”或“缺失”的字符串)。

如前所述，我们必须在一定程度上确保我们在删除这些列/行方面做得很好。例如，[在我创建的项目](https://medium.com/mlearning-ai/ive-analyzed-the-world-food-production-and-those-are-the-results-c043c04226bf)中，我发现一些丢失的值，并决定删除这些行。

让我们来分析这个案例:这些数据与世界上所有国家的粮食产量有关。假设我们的数据框架是“df”，这是我发现的:

```
df.isnull().sum()>>> Area Abbreviation      0
   Area Code              0
   Area                   0
   Item Code              0
   Item                   0
                       ... 
   Y2009                104
   Y2010                104
   Y2011                104
   Y2012                  0
   Y2013                  0
   Length: 63, dtype: int64
```

对于 2009、2010 和 2011 年，我们有 104 个空值；但是此处列出的列并不都是此数据框的空值；无论如何，如果我们看一下数据，我们可以看到，在一些年，在一些国家，有 0 吨的食物生产值。这意味着一件简单的事情:***的数据无法被记录，或者，在那个特定的国家，那个特定的年份，那个特定的食物无法被生产*** (或者，在他们的历史上，他们从来没有在那个特定的国家生产过那个特定的食物！).

通过这个简单的分析，我决定使用下面的代码删除具有空值的行:

```
df **=** df**.**loc[(df**.**loc[:, 'Y1961':'Y2013']**!=**0)**.**any(axis**=**1)]
```

# 3.用平均值代替缺失值

有时值 0 是不可接受的，也许因为我们只有少量的数据，删除行/列是不可行的。那么，我们能做什么呢？一种可能性是用同一行/列中的其他值的平均值替换空值。

例如，在这个[项目](/ive-trained-my-first-machine-learning-model-and-i-got-sad-8abdc0c5b72b)中，我分析了一个数据集，其中我使用机器学习来预测糖尿病。在这种情况下，很容易理解我们不能接受 0 作为身体质量指数(身体质量指数)或血压的值。因为数据很少，所以我不能删除任何一行。所以，我用同一列中其他值的平均值来填充零；我用下面的代码做到了:

```
#filling zeros with mean value
non_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for coloumn in non_zero:
    diab[coloumn] = diab[coloumn].replace(0,np.NaN)
    mean = int(diab[coloumn].mean(skipna = True))
    diab[coloumn] = diab[coloumn].replace(np.NaN, mean)
    print(diab[coloumn])
```

这样，我用计算出的其他身体质量指数值的平均值替换了身体质量指数列中的零，等等。

# 结论

处理丢失的值总是很难，因为我们必须做出决定，并且这些决定必须在编写实际代码之前经过深思熟虑。

所以，**先提问，大量 Google**，尽量深入理解问题和你要处理的数据。然后，决定如何处理丢失的数据(例如，接受空值、删除列/行、用平均值替换丢失的值或空值)。

*我们一起连线吧！*

[*中等*](https://federicotrotta.medium.com/)

[*LINKEDIN*](https://www.linkedin.com/in/federico-trotta/)*(向我发送连接请求)*

*如果你愿意，你可以* [*订阅我的邮件列表*](https://federicotrotta.medium.com/subscribe) *这样你就可以一直保持更新了！*

考虑成为会员:你可以免费支持我和其他像我一样的作家。点击 [*这里*](https://federicotrotta.medium.com/membership) *成为会员。*