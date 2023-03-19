# Python 中的美国市场银行假日

> 原文：<https://towardsdatascience.com/us-market-bank-holidays-pandas-fbb15c693fcc>

## 使用熊猫用 Python 计算纽约证券交易所市场银行假日

![](img/28bf72b086ed2e5b692ae8420a4662c9.png)

照片由 [Briana Tozour](https://unsplash.com/@britozour?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/umbrella-sun?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

## 介绍

在过去的几天里，我一直在试图找到一个合适的方法来计算任何一年的纽约证券交易所市场银行假日。尽管我遇到了一些不同的开源包，但大多数都不可靠，包含一些 bug，不允许我实现我想要的东西。

在今天的文章中，我将介绍纽约证券交易所市场每年应该关闭的特定银行假日，并展示如何实现一个非常简单的类，您可以使用它来计算任何给定时间的银行假日，而结果将包含在 pandas 数据帧中。

## 纽约证券交易所银行假日

根据纽约证券交易所官方市场日历，在接下来的美国假期，市场将关闭:

*   元旦
*   马丁·路德·金纪念日
*   华盛顿的生日
*   受难日
*   阵亡将士纪念日
*   6 月 10 日国家独立日
*   美国独立日
*   劳动节
*   感恩节
*   圣诞日

## 计算熊猫的市场银行假日

Pandas 的模块`pandas.tseries.holiday`包含几个与美国特定节日相对应的类。已经实现并且我们需要作为我们将在本部分构建的逻辑的一部分的是:

*   `USMartinLutherKingJr`
*   `USPresidentsDay`
*   `GoodFriday`
*   `USMemorialDay`
*   `USLaborDay`
*   和`USThanksgivingDay`

因此，我们错过了一些假日，我们将通过创建也包含在`pandas.tseries.holiday`模块中的`Holiday`类的实例来手动创建这些假日。

第一个是发生在一月一日的新年。此外，我们还将指定`nearest_workday`作为遵守规则，以便每当该日期是星期六时，它将被移动到星期五，同样，每当它是星期天时，它将被移动到星期一。

```
from pandas.tseries.holiday import Holiday, nearest_workdaynye = Holiday('NYDay', month=1, day=1, observance=nearest_workday)
```

同样，我们也将创建几个`Holiday`实例来代表 6 月 10 日美国独立日、美国独立日和圣诞节。

```
from pandas.tseries.holiday import Holiday, nearest_workday juneteenth = Holiday(
    'Juneteenth National Independence Day',
    month=6,
    day=19,
    start_date='2021-06-18',
    observance=nearest_workday,
)independence = Holiday(
    'USIndependenceDay',
     month=7,
     day=4,
     observance=nearest_workday
)christmas = Holiday(
    'Christmas',
     month=12,
     day=25,
     observance=nearest_workday
)
```

现在，为了创建一个包含纽约证券交易所市场银行假日的日历，我们将创建一个实现`AbtractHolidayCalendar`的类。`AbstractHolidayCalendar`类提供了返回假日列表的所有必要方法，只有`rules`需要在特定的假日日历类中定义。

因此，我们需要在我们的类中指定一个`rules`列表，其中包含我们之前创建的所有`Holiday`实例，以及已经在`pandas.tseries.holiday`模块中实现的剩余银行假日。

```
from pandas.tseries.holiday import nearest_workday, \
    AbstractHolidayCalendar, Holiday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, \                  
    USMemorialDay, USLaborDay, USThanksgivingDayclass USTradingHolidaysCalendar(AbstractHolidayCalendar):rules = [
        Holiday(
            'NewYearsDay',
            month=1,
            day=1,
            observance=nearest_workday
        ),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            'Juneteenth National Independence Day',
            month=6,
            day=19,
            start_date='2021-06-18',
            observance=nearest_workday,
        ),
        Holiday(
            'USIndependenceDay',
            month=7,
            day=4,
            observance=nearest_workday
        ),
        USLaborDay,
        USThanksgivingDay,
        Holiday(
            'Christmas',
            month=12,
            day=25,
            observance=nearest_workday
        ),
    ]
```

现在，为了计算特定年份的纽约证券交易所银行假日，我们需要做的就是创建一个`USTradingHolidaysCalendar`的实例，然后通过指定我们希望推断银行假日的日期范围来调用`holidays()`方法。

```
cal = USTradingHolidaysCalendar()
holidays = cal.holidays(start='2022-01-01', end='2022-12-31')
```

让我们验证一下我们的实现给出的结果:

```
print(holidays)*DatetimeIndex(['2022-01-17', '2022-02-21', '2022-04-15', '2022-05-30', '2022-06-20', '2022-07-04', '2022-09-05', '2022-11-24','2022-12-26'],
dtype='datetime64[ns]', freq=None)*
```

输出结果与纽交所市场官方网站上[列出的银行节假日完美匹配。您应该已经注意到输出中不包括元旦，这仅仅是因为 2022 年 1 月 1 日是星期六，因此它是在前一年的 12 月 31 日庆祝的。](https://www.nyse.com/markets/hours-calendars)

## 最后的想法

在今天的文章中，我们浏览了美国的纽约证券交易所市场银行假日，并展示了如何实现一个继承自 pandas `AbstractHolidayCalendar`类的非常简单的类，以便创建一个包含任何请求年份的所有银行假日的数据框架。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**相关文章你可能也喜欢**

[](/how-to-merge-pandas-dataframes-221e49c41bec)  [](/how-to-efficiently-convert-a-pyspark-dataframe-to-pandas-8bda2c3875c3)  [](/loc-vs-iloc-in-pandas-92fc125ed8eb) [## 熊猫中的 loc 与 iloc

towardsdatascience.com](/loc-vs-iloc-in-pandas-92fc125ed8eb)