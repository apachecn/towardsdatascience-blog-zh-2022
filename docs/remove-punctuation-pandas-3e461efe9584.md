# 如何删除熊猫的标点符号

> 原文：<https://towardsdatascience.com/remove-punctuation-pandas-3e461efe9584>

## 展示了从 pandas DataFrame 列中消除标点符号的不同方法

![](img/a62bf94ab9980bfeedce5ad9042536ea.png)

[亚历杭德罗·巴尔巴](https://unsplash.com/@albrb?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/dot?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

## 介绍

当处理文本数据时，我们有时可能需要执行一些清理转换。其中之一通常是在标记化之前删除标点符号。

在今天的文章中，我们将展示几种不同的方法来删除 pandas 数据帧中字符串列的标点符号。更具体地说，我们将使用

*   `str.replace()`
*   `regex.sub()`
*   还有`str.translate()`

首先，让我们创建一个示例数据框架，我们将在本文中引用它来演示一些概念。

```
import pandas as pd df = pd.DataFrame(
    [
        (1, 10, True, 'Hello!!!'),
        (2, 15, False, 'Hey..'),
        (3, 11, True, 'What?!'),
        (4, 12, True, 'Exactly!'),
        (5, 17, True, 'Not bad'),
        (6, 10, False, 'Yeap.!'),
        (7, 12, False, 'Hi. How are you?'),
        (8, 19, True, 'Nope,'),
    ],
    columns=['colA', 'colB', 'colC', 'colD']
)print(df)
 ***colA  colB   colC              colD*** *0     1    10   True          Hello!!!
1     2    15  False             Hey..
2     3    11   True            What?!
3     4    12   True          Exactly!
4     5    17   True           Not bad
5     6    10  False            Yeap.!
6     7    12  False  Hi. How are you?
7     8    19   True             Nope,*
```

## 使用 str.replace()和正则表达式

这里的第一个选项是`[pandas.Series.str.replace()](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.replace.html)`方法，它可以用来替换 Series 对象中出现的所有正则表达式。

所以对我们有用的一个正则表达式是`[^\w\s]+`。为了理解这个正则表达式是如何工作的，考虑这样一个事实，即任何标点符号实际上都不是单词或句子。所以我们用否定(即`^`)来表示我们要用空字符串替换任何非词非句(即标点符号)。

```
df['colD'] = df['colD'].str.replace(r'[^\w\s]+', '')print(df)
 ***colA  colB   colC            colD*** *0     1    10   True           Hello
1     2    15  False             Hey
2     3    11   True            What
3     4    12   True         Exactly
4     5    17   True         Not bad
5     6    10  False            Yeap
6     7    12  False  Hi How are you
7     8    19   True            Nope*
```

## 使用 re.sub()方法

现在另一个选择是`re`包中的`sub()`方法，它提供正则表达式匹配操作。

> `[**re.sub(*pattern*, *repl*, *string*, *count=0*, *flags=0*)**](https://docs.python.org/3/library/re.html#re.sub)`
> 
> 返回用替换`*repl*` *替换`*string*`中`*pattern*`最左边不重叠出现的`string`。*

我们将使用与上一节中相同的正则表达式。

```
import redf['colD']=[re.sub('[^\w\s]+', '', s) for s in df['colD'].tolist()] print(df)
 ***colA  colB   colC            colD*** *0     1    10   True           Hello
1     2    15  False             Hey
2     3    11   True            What
3     4    12   True         Exactly
4     5    17   True         Not bad
5     6    10  False            Yeap
6     7    12  False  Hi How are you
7     8    19   True            Nope*
```

还要注意，有时在替换之前先**编译正则表达式可能更有效。举个例子，**

```
import re r = re.compile(r'[^\w\s]+')
df['colD'] = [r.sub('', s) for s in df['colD'].tolist()]
```

## 使用 str.translate()方法

最后，另一种方法是使用`[pandas.Series.str.translate()](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.translate.html)`方法，通过给定的映射表映射输入字符串中的所有字符。

我们方法背后的直觉是将每行上的所有字符串组合在一起，形成一个大字符串，其中每个单独的字符串将由我们选择的分隔符分隔。

对于我们的例子，我们将使用`|`字符作为分隔符。因此，我们首先需要创建一个字符串，其中包含我们希望从字符串中删除的标点符号。请注意，此处不得包含分隔符。显然，选定的分隔符不能出现在任何现有的字符串中，否则这种方法将不起作用。

```
sep = '|'
punctuation_chars = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{}~'
```

现在我们需要使用`str.maketrans()`方法来创建一个可用于`translate()`方法的翻译映射表。

```
mapping_table = str.maketrans(dict.fromkeys(punctuation_chars, ''))
```

现在我们需要做的是使用选定的分隔符`sep`连接`colD`列中的所有字符串。然后使用我们创建的映射表执行`str.translate()`,以删除`punctuation_chars`中指定的标点字符，最后`split()`删除分隔符`sep`上的结果字符串，并将结果赋回列。

```
df['colD'] = sep \
    .join(df['colD'].tolist()) \
    .translate(mapping_table) \
    .split(sep)
```

并且目标列应该没有任何标点字符。

```
print(df)
 ***colA  colB   colC            colD*** *0     1    10   True           Hello
1     2    15  False             Hey
2     3    11   True            What
3     4    12   True         Exactly
4     5    17   True         Not bad
5     6    10  False            Yeap
6     7    12  False  Hi How are you
7     8    19   True            Nope*
```

注意，这个方法是用 C 实现的，所以它**应该是相当高效和快速的。**

## 最后的想法

在今天的简短教程中，我们探索了几种不同的方法，可以用来从 pandas 数据帧的字符串列中删除标点符号。

更具体地说，我们展示了如何做到这一点，使用了三种不同的方法— `str.replace()`、`str.translate()`和`regex.sub()`。请注意，不同的方法在不同的数据集规模上可能会表现出明显不同的性能和效率。因此，选择最适合您的特定用例的最佳方式是将结果相互比较，并选择更合适的那个。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**相关文章你可能也喜欢**

[](/how-to-merge-pandas-dataframes-221e49c41bec)  [](/setuptools-python-571e7d5500f2)  [](/save-trained-models-python-22a11376d975) 