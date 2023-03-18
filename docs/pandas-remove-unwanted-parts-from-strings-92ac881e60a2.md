# 如何去除熊猫身上不需要的部分

> 原文：<https://towardsdatascience.com/pandas-remove-unwanted-parts-from-strings-92ac881e60a2>

## 在 Pandas 中从列中删除不需要的子字符串

![](img/0ec327846fbd17c13805c8da2f5879d1.png)

照片由[波莱特·伍滕](https://unsplash.com/@paullywooten?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/lamps?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

## 介绍

当使用 pandas 时，我们通常需要执行一些预处理任务，以便将数据转换成所需的形式。作为这一步骤的一部分，通常需要执行的一项常见任务涉及到字符串列的转换，我们会消除一些不需要的部分。

在今天的简短教程中，我们将讨论几种可能的方法，您最终可以应用于 pandas 数据帧，以便从某些列的字符串中删除任何不需要的部分。

首先，让我们创建一个示例 DataFrame，我们将在本文中引用它来演示一些概念，并展示如何在使用字符串列时获得预期的结果。

```
import pandas as pd df  = pd.DataFrame(
    [
        (1, '+9A', 100),
        (2, '-1A', 121),
        (3, '5B', 312),
        (4, '+1D', 567),
        (5, '+1C', 123),
        (6, '-2E', 101),
        (7, '+3T', 231),
        (8, '5A', 769),
        (9, '+5B', 907),
        (10, '-1A', 15),
    ],
    columns=['colA', 'colB', 'colC']
) print(df)
 ***colA colB  colC***
*0     1  +9A   100
1     2  -1A   121
2     3   5B   312
3     4  +1D   567
4     5  +1C   123
5     6  -2E   101
6     7  +3T   231
7     8   5A   769
8     9  +5B   907
9    10  -1A    15*
```

让我们假设我们想要转换存储在列`colB`下的数据，以便移除前缀符号`-/+`和后缀字母。

## 用熊猫。Series.str.replace()方法

`[pandas.Series.str.replace()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.replace.html)`方法可以用来替换`Series` / `Index`中每一次出现的模式/正则表达式。

在我们的例子中，我们可以指定一个正则表达式将所有非数字值替换成一个空字符串。因此，下面的表达式可以解决这个问题:

```
df['colB'] = df['colB'].str.replace(r'\D', '')print(df)
 ***colA colB  colC*** *0     1    9   100
1     2    1   121
2     3    5   312
3     4    1   567
4     5    1   123
5     6    2   101
6     7    3   231
7     8    5   769
8     9    5   907
9    10    1    15*
```

在使用正则表达式时，这无疑是众多选择之一。我猜你可以变得很有创造性，但是这个正则表达式应该是我们特定用例中最直接的。

另一个适合我们用例的选项是删除任何非整数字符:

```
df['colB'].str.replace(r'[^0-9]', '')
```

## 用熊猫。Series.str.extract()方法

在 pandas 中，从字符串中删除不需要的部分的另一个选择是`[pandas.Series.str.extract()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.extract.html)`方法，该方法用于提取 regex pat 中的捕获组作为 DataFrame 中的列。

在我们的例子中，我们将简单地提取字符串中我们希望保留的部分:

```
df['colB'] = df['colB'].str.extract(r'(\d+)', expand=False)print(df)
 ***colA colB  colC*** *0     1    9   100
1     2    1   121
2     3    5   312
3     4    1   567
4     5    1   123
5     6    2   101
6     7    3   231
7     8    5   769
8     9    5   907
9    10    1    15*
```

## 用熊猫。Series.replace()方法

`[pandas.Series.replace()](https://pandas.pydata.org/docs/reference/api/pandas.Series.replace.html)`是另一种选择，与我们在本教程中讨论的第一种方法非常相似。该方法将把`to_replace`中给定的值替换为 value，并允许用户指定提供给`to_replace`的值是否应该解释为正则表达式。

```
df['colB'] = df['colB'].replace(r'\D', r'', regex=True)
```

## 使用 map()方法

另一种选择是利用`[map()](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html)`方法，该方法可用于根据输入映射或函数来映射熊猫`Series`的值。

在我们的具体示例中，我们可以使用`map()`来应用一个 lambda 函数，该函数从字符串的开头移除`+/-`，从字符串的结尾移除任何 ascii 字符。

```
from string import ascii_lettersdf['colB'] = \
    df['colB'].map(lambda x: x.lstrip('+-').rstrip(ascii_letters))print(df)
 ***colA colB  colC*** *0     1    9   100
1     2    1   121
2     3    5   312
3     4    1   567
4     5    1   123
5     6    2   101
6     7    3   231
7     8    5   769
8     9    5   907
9    10    1    15*
```

同样，在处理这种类型的操作时，您可以变得非常有创造性，因此可以随意尝试最适合您的特定用例的方法。

你也可以在我最近的一篇文章中读到更多关于`map()`、`apply()`和`applymap()` pandas 方法的内容。

[](/apply-vs-map-vs-applymap-pandas-529acdf6d744) [## 熊猫中的 apply() vs map() vs applymap()

### 讨论 Python 和 Pandas 中 apply()、map()和 applymap()的区别

towardsdatascience.com](/apply-vs-map-vs-applymap-pandas-529acdf6d744) 

## 使用列表理解

当处理大型熊猫数据帧时，您应该始终考虑矢量化，因为这将极大地提高执行操作的效率。

大多数时候，字符串函数可能很难向量化，因此它们可能以迭代的方式执行(这[不是最好的方法](/how-to-iterate-over-rows-in-a-pandas-dataframe-6aa173fc6c84))。

一个可能解决这个问题的好方法是列表理解。例如，代替`str.replace()`，我们可以使用`re.sub()`方法作为列表理解。

> `[*re.****sub****(pattern, repl, string, count=0, flags=0)*](https://docs.python.org/3/library/re.html#re.sub)`
> 
> 返回用替换`*repl*`替换`*string*`中*图案*最左边不重叠出现的`*string*`。如果没有找到`*pattern*`，则不变地返回`*string*`。`*repl*`可以是字符串，也可以是函数；如果它是一个字符串，其中的任何反斜杠转义都会被处理。

```
import redf['colB'] = [re.sub('[^0-9]', '', x) for x in df['colB']]
```

或者，`str.extract()`方法可以用`re.search()`表示为列表理解:

```
import redf['colB'] = [re.search('[0-9]', x)[0] for x in df['colB']]
```

## 最后的想法

在今天的简短教程中，我们讨论了如何处理 pandas 数据帧，以便从包含 string 对象的列中截断任何不需要的部分。

请注意，在对相当大的数据帧应用这种转换时，应该考虑每种方法的性能。性能通常因数据帧的大小而异，例如，一种方法在应用于小数据帧时可能最有效，但在应用于大数据帧时就不那么有效了。

因此，尝试不同的方法并为您的特定用例应用最有效的方法总是有益的。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership) [## 通过我的推荐链接加入 Medium-Giorgos Myrianthous

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

gmyrianthous.medium.com](https://gmyrianthous.medium.com/membership) 

**相关文章你可能也喜欢**

[](/how-to-efficiently-convert-a-pyspark-dataframe-to-pandas-8bda2c3875c3) [## 加快 PySpark 和 Pandas 数据帧之间的转换

### 将大火花数据帧转换为熊猫时节省时间

towardsdatascience.com](/how-to-efficiently-convert-a-pyspark-dataframe-to-pandas-8bda2c3875c3) [](/combine-two-string-columns-pandas-fde0287485d9) [## 如何在熊猫中组合两个字符串列

### 了解如何在 pandas 数据框架中更有效地将两个字符串列连接成一个新列

towardsdatascience.com](/combine-two-string-columns-pandas-fde0287485d9)