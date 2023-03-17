# 时间序列数据的排序如何识别异常行为

> 原文：<https://towardsdatascience.com/how-ranking-timeseries-data-can-identify-abnormal-behaviour-475c7fd21fb5>

## [行业笔记](https://towardsdatascience.com/tagged/notes-from-industry)

# 时间序列数据的排序如何识别异常行为

## 通过简单的 python 脚本告知您的维护策略并防止故障

![](img/22a7e1de2896a0e48b461f5076ff0c3f.png)

美国公共电力协会在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

温度和压力等传感器读数有助于了解资产的“正常”行为。当读数偏离该标准时，它可能是损坏或低效运行的指示，如果不加以纠正，可能会导致代价高昂的设备故障。因此，这些数据有助于形成任何主动维护或检查活动。

例如，问题是温度与许多因素有关；资产的工作强度(例如生产水平)、环境温度和冷却效率。这使得寻找和识别*真正*关注的温度水平和变化变得棘手，可能需要昂贵的数据专家，有时我们只是没有收集足够的数据来获得准确的结果，或者没有兴趣在这方面投资。

然而，我们可以做一些更简单的事情。

如果我们有多种资产，例如发电机，我们可以将每个发电机的行为相对于其他发电机进行排序。通过随着时间的推移跟踪这一排名，我们可以寻找行为变化的*证据。*然后可以对此进行标记，以便进行深入调查。代码非常简单，不需要昂贵复杂的模型就可以获得结果。

本文侧重于工程应用，但很容易看出如何将其应用于任何受多种因素影响的基准测试。

# 一个例子

让我们以发电机为例，假设您有 20 台不同发电机的历史发电量和温度读数。只考虑发电机正常生产的时间。发电机 A 是 2016 年排名最高的生产商(最高产量)，也是最热门的——这并不奇怪，我们知道这两者是相关的。在同一年，发电机 B 是最冷的(第 19 位)之一，具有中等的包装生产水平(第 11 位)。在接下来的四年里，这些相同类型的排名持续稳定，所以我们有信心这是他们的“标准”。

然而，在 2021 年，发现发电机 A 是该年第二热的，但却是最低的生产者之一。好像有什么不对劲，为什么还是这么热？此外，发电机 B 现在是第七热的发电机，但其生产值没有改变。这也很奇怪，这种发电机在这些生产水平下不会运行得这么热，还会发生什么呢？

只看温度读数可能会忽略发电机 A 的异常温度，因为从表面上看，温度读数没有太大变化，是它们与发电的关系发生了变化。孤立地看待这两者可能没有突出问题。同样，B 可能已经被遗漏，因为发电机仅是第七热的，因此可能被认为不够热而不必担心。

这是一个首过指标，现在应该对这两项资产进行全面的数据调查。当时还有什么在影响这个系统？例如，资产是否长期离线？通讯中断了吗？传感器精度有问题吗？是否存在可能导致此类行为的已知问题？回答这些问题，了解检查或主动工作是否符合您的利益。

# 代码

这个例子使用 python 中的 pandas 库。一般来说，工作流程是:

1.  **源时间序列数据** —以:时间戳、资产、世代、温度的形式
2.  **按期望的时间间隔**分组——获得年或月平均值、总和、偏差等。
3.  **申请排名**
4.  **分析结果**

下面是更详细的工作流程。

# 准备要排序的数据

```
# import pandas library
import pandas as pd# read in timeseries data, may be via csv or queried direct
# from a database
df = pd.read_csv(r'filepath\filename.csv')# create a variable for your desired time interval
# i.e. year or month
df['year'] = pd.to_datetime(df['TimeStamp']).dt.to_period('Y')# group the data by generator and year and find the aggregate values
#e.g. average temperature over the year
df2 = df.groupby(['GeneratorId','year']).agg({'average_temp': ['std', 'mean'], 'generation': ['std', 'sum']})# write to excel
df2.to_csv(r'filepath\filename.csv', index = False)
```

虽然您可以在这一步应用排名，但我建议导出数据子集，并对照实际值进行检查。这将确保脚本像预期的那样工作，并可能使您以后免受代码中任何讨厌的错误的困扰。

# 对数据进行排序

这个函数非常简单，一旦读回 python apply:

```
df2['average_temp_rank'] = df_year.groupby('year')['average_temp_mean'].rank(pct=True)
```

这将把温度等级应用到名为“average_temp_rank”的新列中。对其他要排序的值重复此操作，如 generation sum，就这样！

# 分析一下

然后可以在 python 或 excel 中进行进一步的分析，以跟踪这些排名是如何变化的。一种方法可以是趋势代排名:温度排名比率，这是保持稳定还是变得不匹配？

上面没有提到的是一段时间内的**标准差**排名。温度波动是否超过正常值？特别是在发电机中，这些频繁的波动会导致材料因这些变化而膨胀和收缩，从而导致损坏。

# 深入挖掘

如前所述，这部分是关键。排名分析将为我们提供一些线索，表明有什么地方出了问题，从而引发更广泛的调查。

虽然这是一个简化的例子，但显然排名分析可以有一系列有用的应用。请在评论中告诉我你如何在你的行业中使用它们。

[](https://medium.com/@hollydalligan/membership) [## 通过我的推荐链接加入 Medium-Holly Dalligan

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

medium.com](https://medium.com/@hollydalligan/membership)