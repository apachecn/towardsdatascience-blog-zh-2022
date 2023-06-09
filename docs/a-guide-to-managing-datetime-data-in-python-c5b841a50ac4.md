# Python 中日期时间数据管理指南

> 原文：<https://towardsdatascience.com/a-guide-to-managing-datetime-data-in-python-c5b841a50ac4>

## 以下是如何解决你的日期时间，熊猫指数和夏令时噩梦

![](img/87d5f68ed48f62915dd66b71d4b29cbb.png)

图片来源:pixabay 上的 SplitShire

使用日期时间数据集可能是计算机编程中最令人沮丧的方面之一。您不仅需要跟踪日期，还需要学习如何用每种语言表示日期和时间，从这些数据点中创建索引，并确保您的所有数据集以相同的方式处理夏令时。幸运的是，这本关于日期时间数据的入门书将帮助您开始使用 Python。

# 日期时间

Python 中大多数日期和时间的表示都是以 datetime 对象的形式呈现的，这些对象是从 Datetime 包中创建的。这意味着了解 Datetime 包以及如何使用它是至关重要的！

从根本上说，datetime 对象是一个包含日期和时间信息的变量。它还可以包括时区信息，并且有根据需要更改时区的工具。

让我们看几个使用 datetime 包创建的 Datetime 对象的例子。首先，我们可以使用包的命令创建一个存储当前时间的变量，如下所示:

```
import datetimeimport pytznow = datetime.datetime.now(pytz.timezone(‘US/Pacific’))print(now)print(now.tzinfo)
```

前两行导入了这个任务的重要包。第一个是 datetime 包，它使我们能够创建和操作 Datetime 对象。第二个是 Pytz 包，它提供时区信息。

第三行调用 **datetime.datetime.now** 函数来创建一个 datetime 对象，表示我们运行代码的时间。此行还向 datetime 添加了一个时区，说明 datetime 表示美国太平洋时区的时间。

第四行和第五行都是打印输出，用于演示代码的结果。

该代码的输出是:

```
2022–05–11 09:19:01.859385–07:00US/Pacific
```

第一个输出显示了变量**现在**的完整信息。显示该变量创建于 2022 年 5 月 11 日 9 点 19 分 1.86 秒。因为我将时区设置为'**美国/太平洋**'，所以程序将适当的 **-7** 小时(相对于 UTC)附加到变量上。第二个输出通过打印变量的时区是美国太平洋时区来确认时区信息。

您会注意到，在上面的代码中，我通过调用**pytz . time zone(' US/Pacific ')**将时区设置为美国太平洋时区。如果您想使用不同的时区，您需要知道正确的代码(尽管它们都遵循相同的格式并引用已知的时区，所以它们是相当容易预测的)。如果您想找到您的 timez one，您可以使用以下命令打印所有选项的列表。

```
print(pytz.all_timezones)
```

我们还可以使用 **datetime.datetime** 函数来创建指定日期的日期时间。请注意前面示例中显示当前时间的 datetime 对象的格式，因为对 **datetime.datetime** 的输入是以相同的顺序提供的(年、月、日、小时、分钟、秒)。换句话说，如果我们想要创建一个表示美国太平洋时区 2022 年 5 月 11 日 12:11:03 的 datetime 对象，我们可以使用以下代码:

```
specified_datetime = datetime.datetime(2022, 5, 11, 12, 11, 3).astimezone(pytz.timezone(‘US/Pacific’))print(specified_datetime)print(specified_datetime.tzinfo)
```

注意上面的 **datetime.datetime** 的输入是如何出现的。然后是**。调用 astimezone** 方法将时区设置为美国太平洋时区。

该代码的输出是:

```
2022–05–11 12:11:03–07:00US/Pacific
```

结果是代码创建了我们想要的变量。根据需要， **specified_datetime** 现在返回美国太平洋时区 2022 年 5 月 11 日 12:11:03。

如果我们想在不同的时区表示相同的时间呢？我们可以计算新的时区并相应地创建一个新的 datetime 对象，但是这需要我们知道时差，进行计算并相应地创建新的对象。另一个选项是将时区转换为所需的时区，并将输出保存到一个新变量中。因此，如果我们想将**指定日期时间**转换为美国东部时区，我们可以使用下面的代码。

```
eastern_datetime = specified_datetime.astimezone(pytz.timezone(‘US/Eastern’))print(eastern_datetime)print(eastern_datetime.tzinfo)
```

这段代码调用了**。用代表美国东部时区的新时区对象指定日期时间的方法。打印输出为:**

```
2022–05–11 15:11:03–04:00US/Eastern
```

请注意**指定日期时间**和**大西洋日期时间**之间的变化。由于美国东部时间比美国太平洋时间早三个小时，时间从 12 点变成了 15 点。时区信息从 **-7** 变为 **-4** ，因为美国东部时间与 UTC 相差 4 小时，而不是相差 7 小时。最后，注意打印的时区信息现在是**美国/东部**而不是**美国/太平洋**。

# 熊猫指数

Pandas 数据帧通常使用 datetime 对象作为索引，因为这使数据集能够跟踪记录测量的日期和时间。因此熊猫提供了许多你可以使用的工具。

您第一次介绍带有日期时间索引的数据帧很可能是导入别人的数据集，而别人恰好使用了一个数据集。考虑以下示例和提供的输出:

```
data = pd.read_csv(r’C:\Users\Peter Grant\Desktop\Sample_Data.csv’, index_col = 0)print(data.index)print(type(data.index))print(type(data.index[0]))
```

示例数据集中的索引使用 datetime，所以您会认为 dataframe 的索引是 datetime 索引，对吗？很不幸，你错了。让我们来看看输出:

```
Index([‘10/1/2020 0:00’, ‘10/1/2020 0:00’, ‘10/1/2020 0:00’, ‘10/1/2020 0:01’,‘10/1/2020 0:01’, ‘10/1/2020 0:01’, ‘10/1/2020 0:01’, ‘10/1/2020 0:02’,‘10/1/2020 0:02’, ‘10/1/2020 0:02’,…‘4/1/2021 2:01’, ‘4/1/2021 2:01’, ‘4/1/2021 2:02’, ‘4/1/2021 2:02’,‘4/1/2021 2:02’, ‘4/1/2021 2:02’, ‘4/1/2021 2:03’, ‘4/1/2021 2:03’,‘4/1/2021 2:03’, ‘4/1/2021 2:03’],dtype=’object’, length=1048575)<class ‘pandas.core.indexes.base.Index’><class ‘str’>
```

指数看起来就像我们预期的那样。每个都表示日期和时间，同时遍历日期和时间，直到索引结束。每 15 秒记录一次样本，每分钟有 4 个，这很好，但是值没有显示秒数，这有点奇怪。

但是事情变得奇怪了。当我们希望索引是日期时间索引时，它的类型是泛型。最后，第一个条目的类型是字符串，而不是日期时间对象。

Pandas 将日期时间索引作为字符串列表读入，而不是日期时间索引。每次都会这样。好在熊猫有一个 **to_datetime()** 函数解决了这个问题！考虑以下代码:

```
data.index = pd.to_datetime(data.index)print(data.index)print(type(data.index))print(type(data.index[0]))
```

以及输出:

```
DatetimeIndex([‘2020–10–01 00:00:00’, ‘2020–10–01 00:00:00’,‘2020–10–01 00:00:00’, ‘2020–10–01 00:01:00’,‘2020–10–01 00:01:00’, ‘2020–10–01 00:01:00’,‘2020–10–01 00:01:00’, ‘2020–10–01 00:02:00’,‘2020–10–01 00:02:00’, ‘2020–10–01 00:02:00’,…‘2021–04–01 02:01:00’, ‘2021–04–01 02:01:00’,‘2021–04–01 02:02:00’, ‘2021–04–01 02:02:00’,‘2021–04–01 02:02:00’, ‘2021–04–01 02:02:00’,‘2021–04–01 02:03:00’, ‘2021–04–01 02:03:00’,‘2021–04–01 02:03:00’, ‘2021–04–01 02:03:00’],dtype=’datetime64[ns]’, length=1048575, freq=None)<class ‘pandas.core.indexes.datetimes.DatetimeIndex’><class ‘pandas._libs.tslibs.timestamps.Timestamp’>
```

啊哈。这个看起来好多了。数据帧的索引现在是日期时间索引，第一个条目的类型现在是熊猫时间戳(相当于日期时间对象)。这是我们想要的，也是我们可以努力的。

但是，如果您想创建自己的日期时间索引呢？如果您知道想要创建日期时间索引的日期范围和频率，那么可以使用 Pandas **date_range()** 函数。这里有一个例子:

```
index = pd.date_range(datetime.datetime(2022, 1, 1, 0, 0), datetime.datetime(2022, 12, 31, 23, 55), freq = ‘5min’)print(index)
```

此代码返回以下输出:

```
DatetimeIndex([‘2022–01–01 00:00:00’, ‘2022–01–01 00:05:00’,‘2022–01–01 00:10:00’, ‘2022–01–01 00:15:00’,‘2022–01–01 00:20:00’, ‘2022–01–01 00:25:00’,‘2022–01–01 00:30:00’, ‘2022–01–01 00:35:00’,‘2022–01–01 00:40:00’, ‘2022–01–01 00:45:00’,…‘2022–12–31 23:10:00’, ‘2022–12–31 23:15:00’,‘2022–12–31 23:20:00’, ‘2022–12–31 23:25:00’,‘2022–12–31 23:30:00’, ‘2022–12–31 23:35:00’,‘2022–12–31 23:40:00’, ‘2022–12–31 23:45:00’,‘2022–12–31 23:50:00’, ‘2022–12–31 23:55:00’],dtype=’datetime64[ns]’, length=105120, freq=’5T’)
```

将调用 date_range()的代码与函数的[文档进行比较，可以看到前两个条目设置了范围的开始和结束日期。开始日期设置为 2022 年 1 月 1 日午夜，结束范围设置为 2022 年 12 月 31 日 23:55:00。第三个条目将日期时间索引的频率设置为五分钟。注意五分钟的代码是**‘5min’**。为了获得您想要的频率，您需要使用正确的代码来设置频率。幸运的是，有一个熊猫代码列表](https://pandas.pydata.org/docs/reference/api/pandas.date_range.html)[可用。](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)

用熊猫日期时间索引也可能有点麻烦。乍一看，似乎需要创建复杂的 datetime 对象来引用 dataframe 的正确部分。考虑下面的例子，其中我使用上一个例子中的索引创建了一个新的 dataframe，在 dataframe 中设置一个值并打印该值以确保它正确更新。

```
df = pd.DataFrame(index = index, columns = [‘Example’])df.loc[datetime.datetime(2022, 1, 1, 0, 0, 0), ‘Example’] = 2print(df.loc[datetime.datetime(2022, 1, 1, 0, 0, 0), ‘Example’])
```

这段代码的输出正是我想要的。它打印出 **2** ，显示 dataframe 在 **[datetime.datetime(2022，1，1，0，0，0)，‘Example ']**的值为所需的 **2** 。但是一遍又一遍地指定日期时间变得很乏味。

幸运的是，您仍然可以通过位置引用日期时间索引。如果要编辑索引中的第一个条目，可以按如下方式操作。

```
df = pd.DataFrame(index = index, columns = [‘Example’])df.loc[df.index[0], ‘Example’] = 2print(df.loc[df.index[0], ‘Example’])
```

请注意这段代码是如何做完全相同的事情的，只是它通过调用索引的第一个值来提供所需的索引值。你甚至不需要知道有什么价值，你只需要知道你想用第一个——或者第二个，或者第三个，或者任何你想要的价值。您只需要相应地更新呼叫。

# 夏令时

使用夏令时是 Python 中最大的难题之一，并且会导致非常严重的数据分析错误。考虑将物理测量值与理论近似值进行比较的例子。现在你有两个数据集，你想确保他们说的是同一件事。如果一个数据集使用夏令时，而另一个数据集不使用夏令时，该怎么办？突然间，你开始比较相差一小时的数据集。

解决此问题的一种方法是，在夏令时发生的时间内，从具有夏令时的 dataframe 索引中删除一个小时。为了帮助解决这个问题，熊猫的时间戳有一个**。dst()** 方法，返回任意点的夏令时差。如果时间戳发生在夏令时，它将返回一个小时的 **datetime.timedelta** 值。如果时间戳在夏令时期间没有出现，它将返回一个零小时的 **datetime.timedelta** 值。这使我们能够识别夏令时期间出现的时间戳，并相应地从中删除一个小时。

假设您有一个包含夏令时偏移量的 datetime 索引的数据帧。要删除夏令时，可以遍历索引，使索引时区为 naive，并从索引中删除一个小时。不幸的是，索引是不可变的，所以您不能直接编辑它们。您可以做的是创建一个外部值列表，将更新后的索引值添加到该列表中，并在最后用该列表替换索引。这段代码应该是这样的。

```
temp = []for ix in range(len(df.index)):if df.index[ix].dst() == datetime.timedelta(hours = 1):temp.append(df.index[ix].tz_localize(None) — datetime.timedelta(hours = 1))else:temp.append(df.index[ix].tz_localize(None))df.index = temp
```

现在你知道了。现在您知道了如何使用 datetime 对象，使用它们来构成 Pandas 数据帧的索引，并从您的数据集中删除夏令时。