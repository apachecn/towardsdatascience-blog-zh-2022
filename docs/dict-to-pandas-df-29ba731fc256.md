# 如何将字典转换成熊猫数据框架

> 原文：<https://towardsdatascience.com/dict-to-pandas-df-29ba731fc256>

## 使用 Pandas 将 Python 字典转换为数据帧

![](img/304b32875e1864e756649dd6cdf26236.png)

照片由[大卫·舒尔茨](https://unsplash.com/@davidschultz?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/programming?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

`[pandas](https://pandas.pydata.org/)`是 Python 生态系统中最受欢迎的库之一，它通过提供直观而强大的 API 让开发人员与数据进行交互，以快速有效的方式用于数据分析和操作。

使用 Python 和 pandas 时，最常见的任务之一是将字典转换为数据帧。当您想要对当前存储在字典数据结构中的数据进行快速分析甚至可视化时，这将非常有用。

在本文中，我们将探讨如何以几种不同的方式将 Python 字典转换成 pandas 数据框架，这取决于数据最初在 dict 中是如何构造和存储的。

## 从 Python 字典创建熊猫数据框架

现在，为了从 Python 字典中创建一个 pandas 数据帧，我们可以使用`pandas.DataFrame.from_dict`方法，该方法用于从类似数组的 dict 或 dicts 对象中构造数据帧。

让我们用一些虚拟值创建一个示例 Python 字典，我们将在接下来的几节中使用这些虚拟值来演示一些将其转换成 pandas 数据帧的有趣方法。

```
users = {
  'fist_name': ['John', 'Andrew', 'Maria', 'Helen'],
  'last_name': ['Brown', 'Purple', 'White', 'Blue'],
  'is_enabled': [True, False, False, True],
  'age': [25, 48, 76, 19]
}
```

在这个示例字典中，键对应于数据帧的列，而列表中的每个元素对应于特定列的行值。因此，我们可以(可选地)指定`orient`等于`'columns'`

> 数据的“方向”。如果传递的字典的键应该是结果数据帧的列，则传递' columns '(默认)
> 
> — [熊猫文档](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_dict.html)

```
import pandas as pd 

users = {
  'fist_name': ['John', 'Andrew', 'Maria', 'Helen'],
  'last_name': ['Brown', 'Purple', 'White', 'Blue'],
  'is_enabled': [True, False, False, True],
  'age': [25, 48, 76, 19]
}

df = pd.DataFrame.from_dict(users)
```

我们刚刚使用 Python 字典创建了一个熊猫数据框架！

```
print(df)

  fist_name last_name  is_enabled  age
0      John     Brown        True   25
1    Andrew    Purple       False   48
2     Maria     White       False   76
3     Helen      Blue        True   19
```

这种方法仅适用于字典中的数据是以每个键对应于 DataFrame 列的方式构造的情况。如果我们有不同的结构会怎么样？

## 将字典关键字填充为行

现在让我们假设我们有一个字典，它的键对应于我们想要创建的数据帧的行。

```
users = {
  'row_1': ['John', 'Brown', True, 25],
  'row_2': ['Andrew', 'Purple', False, 48],
  'row_3': ['Maria', 'White', False, 76],
  'row_4': ['Helen', 'Blue', True, 19],
}
```

在这个场景中，我们必须使用下面代码片段中展示的`orient='index'`选项，这样字典中的每个键值对都被解析为一个 DataFrame 行。但是请注意，当使用`orient='index'`时，我们必须在调用`from_dict()`方法时显式指定列名:

```
import pandas as pd

users = {
  'row_1': ['John', 'Brown', True, 25],
  'row_2': ['Andrew', 'Purple', False, 48],
  'row_3': ['Maria', 'White', False, 76],
  'row_4': ['Helen', 'Blue', True, 19],
}

cols = ['first_name', 'last_name', 'is_enabled', 'age']
df = pd.DataFrame.from_dict(users, orient='index', columns=cols)
```

我们又一次设法用 Python 字典构造了一个 pandas 数据帧，这次是通过将每个键值对解析为一个数据帧行:

```
print(df)

      first_name last_name  is_enabled  age
row_1       John     Brown        True   25
row_2     Andrew    Purple       False   48
row_3      Maria     White       False   76
row_4      Helen      Blue        True   19
```

您可能已经注意到，每个键也成为新填充的数据帧的索引。如果您希望删除它，可以通过运行以下命令来实现:

```
df.reset_index(drop=True, inplace=True)
```

现在应该重置索引:

```
print(df)

  first_name last_name  is_enabled  age
0       John     Brown        True   25
1     Andrew    Purple       False   48
2      Maria     White       False   76
3      Helen      Blue        True   19
```

## 紧密定向选项

从 pandas v1.4.0 开始，当从 Python 字典构造 pandas 数据帧时，您也可以使用`orient='tight'`选项。该选项假设输入字典有以下按键:`'index'`、`'columns'`、`'data'`、`'index_names'`和`'column_names'`。

例如，以下字典符合此要求:

```
data = {
  'index': [('a', 'b'), ('a', 'c')],
  'columns': [('x', 1), ('y', 2)],
  'data': [[1, 3], [2, 4]],
  'index_names': ['n1', 'n2'],
  'column_names': ['z1', 'z2']
}

df = pd.DataFrame.from_dict(data, orient='tight')

print(df)
z1     x  y
z2     1  2
n1 n2      
a  b   1  3
   c   2  4
```

在构建多索引数据框架时，最后一种方法通常很有用。

## 最后的想法

将 Python 字典转换成 pandas 数据框架是一个简单明了的过程。通过根据原始字典的结构方式使用`pd.DataFrame.from_dict`方法和正确的`orient`选项，您可以很容易地将数据转换成 DataFrame，这样您现在就可以使用 pandas API 执行分析或转换。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

<https://gmyrianthous.medium.com/membership>  

**相关文章你可能也喜欢**

</how-to-merge-pandas-dataframes-221e49c41bec>  </make-class-iterable-python-4d9ec5db9b7a>  </visual-sql-joins-4e3899d9d46c> 