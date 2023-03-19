# 如何将文本文件中的数据加载到 Pandas

> 原文：<https://towardsdatascience.com/txt-to-pandas-df3aeaf92548>

## 用 Python 将文本文件中存储的数据加载到 pandas 数据帧中

![](img/91ac68bd20031b83022f7ac84c4ff32d.png)

[布鲁斯洪](https://unsplash.com/@hongqi?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/pandas?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

Pandas 是事实上的 Python 包，允许用户在内存中执行数据转换和分析。在许多情况下，这些数据最初驻留在外部源中，如文本文件。通过其强大的 API，pandas 允许用户通过各种方法从这些来源加载数据。

在今天的文章中，我们将演示如何使用这些方法将文本文件中的数据加载到 pandas 数据帧中。此外，我们将讨论如何处理分隔符和列名(也称为标题)。

首先，让我们创建一个名为`employees.txt`的示例文本文件，我们将在今天的简短教程中使用它来演示一些概念。请注意，字段由单个空格字符分隔，第一行对应于标题。

```
name surname dob department
George Brown 12/02/1993 Engineering
Andrew Black 15/04/1975 HR
Maria Green 12/02/1989 Engineering
Helen Fox 21/10/2000 Marketing
Joe Xiu 10/11/1998 Engineering
Ben Simsons 01/12/1987 Engineering
Jess Middleton 12/12/1997 Marketing
```

## 使用 read_csv()

a**c**omma**s**separated**f**file(CSV)实际上是一个文本文件，它使用逗号作为分隔符来分隔每个字段的记录值。因此，使用`[pandas.read_csv()](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)`方法从文本文件加载数据是有意义的，即使文件本身没有`.csv`扩展名。

为了读取我们的文本文件并将其加载到 pandas DataFrame 中，我们只需要向`read_csv()`方法提供文件名、分隔符/定界符(在我们的例子中是一个空格)和包含列名的行，这似乎是第一行。

```
import pandas as pddf = pd.read_csv('employees.txt', sep=' ', header=0)print(df)
 ***name    surname         dob   department*** *0  George      Brown  12/02/1993  Engineering
1  Andrew      Black  15/04/1975           HR
2   Maria      Green  12/02/1989  Engineering
3   Helen        Fox  21/10/2000    Marketing
4     Joe        Xiu  10/11/1998  Engineering
5     Ben    Simsons  01/12/1987  Engineering
6    Jess  Middleton  12/12/1997    Marketing*
```

请注意，如果您要加载为 pandas DataFrame 的文件有不同的分隔符，比如逗号`,`、冒号`:`或制表符`\t`，那么您需要做的就是在调用`read_csv()`时在`sep`或`delimiter`参数中指定该字符。

## 使用 read_table()

或者，您可以利用`[pandas.read_table()](https://pandas.pydata.org/docs/reference/api/pandas.read_table.html)`方法将通用分隔文件读入 pandas 数据帧。

```
import pandas as pd df = pd.read_table('employees.txt', sep=' ', header=0)print(df)
 ***name    surname         dob   department*** *0  George      Brown  12/02/1993  Engineering
1  Andrew      Black  15/04/1975           HR
2   Maria      Green  12/02/1989  Engineering
3   Helen        Fox  21/10/2000    Marketing
4     Joe        Xiu  10/11/1998  Engineering
5     Ben    Simsons  01/12/1987  Engineering
6    Jess  Middleton  12/12/1997    Marketing*
```

## 最后的想法

创建 pandas 数据框架的最常见方法之一是加载存储在外部源中的数据，如文本或 csv 文件。在今天的简短教程中，我们一步一步地完成了一个过程，通过从一个文本文件中加载数据，最终可以帮助您构建一个熊猫数据框架，其中每个字段都由一个特定的字符(制表符、空格或其他字符)分隔。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读媒体上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**相关文章你可能也喜欢**

[](/diagrams-as-code-python-d9cbaa959ed5)  [](/args-kwargs-python-d9c71b220970)  [](/python-poetry-83f184ac9ed1) 