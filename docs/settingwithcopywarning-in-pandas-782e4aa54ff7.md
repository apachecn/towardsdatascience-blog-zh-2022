# 在熊猫中设置版权警告

> 原文：<https://towardsdatascience.com/settingwithcopywarning-in-pandas-782e4aa54ff7>

# 在熊猫中设置版权警告

## 它是什么，为什么突然出现，以及如何摆脱它

(TL；dr:如果你是来寻求答案的，直接跳到[消除带有复制警告的设置](#0e7a)

如果你是一个熊猫用户，当你给一个`pd.DataFrame`或`pd.Series`赋值的时候，你可能已经看到 SettingWithCopyWarning 弹出来了。

```
In [1]: import pandas as pd
     …:
     …: df = pd.DataFrame({
     …:     “A”: [1, 2, 3, 4, 5],
     …:     “B”: [6, 7, 8, 9, 10],
     …:     }, index=range(5)
     …: )
     …: dfa = df.loc[3:5]
     …: dfa[“C”] = dfa[“B”] * 50<ipython-input-2–63497d1da3d9>:9: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value insteadSee the caveats in the documentation: [https://pandas.pydata.org/pandas-docs/](https://pandas.pydata.org/pandas-docs/)
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
 dfa[“C”] = dfa[“B”] * 50
```

pandas 在这里给出了一个警告，因为赋值可能会也可能不会像预期的那样工作。

需要说明的是，赋值确实发生了；这里强调的是“预期”。

```
In [2]: dfa
Out[2]:
  A B C
3 4 9 450
4 5 10 500In [3]: df
Out[3]:
  A B
0 1 6
1 2 7
2 3 8
3 4 9
4 5 10
```

你认为`df`的内容会受到`dfa`中赋值的影响吗？对于这种情况是否会发生，熊猫有着内在一致的(尽管有些迟钝)规则。只是在这种情况下，用户期望的模糊性需要一个警告，这样当我们的代码出现问题时，像你我这样的最终用户就知道去哪里找了。

# 带有视图和副本的链式分配

![](img/ce4bd701796f3f486277deb6e5542541.png)

照片由[郭佳欣·阿维蒂西安](https://unsplash.com/@kar111?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

从数据帧或系列中选择要访问的行或列的行为称为*索引*。pandas 的灵活性允许链式索引，您可以重复索引前一次索引操作的结果。

```
# Select the 2nd to 4th row of data where col A > 3 and col B != 7
df[df[“A”] > 3 & df[“B”] != 7].iloc[3:5]
```

然后 pandas 将返回数据帧的视图或副本。视图(浅层拷贝)引用原始数据帧中的数据，而拷贝(深层拷贝)是同一数据的单独实例。

很难预测索引操作将返回哪个，因为它取决于底层数组的内存布局。索引的链接方式会导致不同的`__getitem__`和`__setitem__`调用被隐藏起来。再现下面的例子:

```
# Example borrowed from [1]
dfmi.loc[:, (‘one’, ‘second’)] = value
# becomes
dfmi.loc.__setitem__((slice(None), (‘one’, ‘second’)), value)dfmi[‘one’][‘second’] = value
# becomes
dfmi.__getitem__(‘one’).__setitem__(‘second’, value)
```

链式索引本身没有问题，但使用链式索引赋值，即`chained assignment`，可能会有问题。根据具体情况，链式赋值要么直接修改原始数据帧，要么返回原始数据帧的修改副本。当链式索引不明显时，这可能会导致潜在的错误。

链式索引可以跨几行代码进行:

```
# The following doesn’t look like chain indexing, does it?
dfa = df.loc[row1:row2, col1:col2]
…
…
dfa[row2] = dfa[row1].apply(fn)
```

如果 pandas 在这个场景中没有发出警告，那么`df`没有被第二个赋值修改就不明显了。这就是 SettingWithCopyWarning 存在的原因。

熊猫医生[ ]对此进行了更详细的研究。警告信息链接到它很有帮助，这很好，因为如果你在 Google 上搜索`pandas settingwithcopywarning`，文档页面很容易被遗漏！在撰写本文时，它是 Google 首页的第 7 个结果，被 blogposts 和 StackOverflow 问题挤掉了。

# 使用内部 API 在幕后窥视

链式索引是选择正确数据的天赐便利，但是链式赋值是赋值正确值的雷区。[ ]中的 TowardsDataScience 文章提供了一个很好的例子，其中仅颠倒链式索引的顺序就可以决定是否对原始数据帧进行赋值:

```
# Example borrowed from [2]# This updates `df`
df[“product_group”][df[“product_group”]==”PG4"] = “PG14”# This doesn’t!
df[df[“product_group”]==”PG4"][“product_group”] = “PG14”# pandas raises warnings for both
# the user needs to confirm the intended outcome
```

从[这个 StackOverflow post](https://stackoverflow.com/questions/26879073/checking-whether-data-frame-is-copy-or-view-in-pandas) 、`pd.DataFrame`和`pd.Series`对象拥有`_is_view`和`_is_copy`属性作为它们内部 API 的一部分。`_is_view`如果对象是视图，返回 True 如果对象不是视图，返回 False。`_is_copy` 存储一个[弱引用](https://docs.python.org/3/library/weakref.html)到它被复制的数据帧，或者`None`如果它不关联到一个现有的数据帧。

打印这些内部属性，同时使用链式赋值确实揭示了一些有趣的信息。一方面，pandas 使用`_is_copy`来决定是否需要提高 SettingWithCopyWarning。另一方面，用`_is_view = True`修改数据帧意味着它将影响原始的底层数据帧。

在我们开始之前，一个免责声明:内部 API 不是最终用户可以访问的，可能会发生变化，使用它们需要您自担风险。

```
In [4]: pd.__version__
Out[4]: ‘1.3.3’# Setting up convenience functions
In [5]: def make_clean_df():
     …:     df = pd.DataFrame({
     …:         “A”: [1, 2, 3, 4, 5],
     …:         “B”: [6, 7, 8, 9, 10],
     …:         “C”: [11, 12, 13, 14, 15],
     …:         }, index=range(5)
     …:     )
     …:     return dfIn [6]: def show_attrs(obj):
     …:     print(f”view: {obj._is_view}, copy: {obj._is_copy}”)
```

我们将首先展示几种常见索引方法的`_is_view`和`_is_copy`属性。

```
In [7]: df = make_clean_df()In [8]: show_attrs(df.loc[3:5])
     …: show_attrs(df.iloc[3:5])
     …: show_attrs(df.loc[3:5, [“A”, “B”]])
     …: show_attrs(df.iloc[3:5, [0, 1]])
     …: show_attrs(df[“A”])
     …: show_attrs(df.loc[:, “A”])view: True, copy: <weakref at 0x7f4d648b2590; to ‘DataFrame’ at 0x7f4d648b54c0>
view: True, copy: <weakref at 0x7f4d648b2590; to ‘DataFrame’ at 0x7f4d648b54c0>
view: False, copy: None
view: False, copy: <weakref at 0x7f4d648be770; dead>
view: True, copy: None
view: True, copy: None
```

让我们来分解一下:

*   `df.loc[3:5]`和`df.iloc[3:5]`都返回了视图并引用了原始数据帧。
*   对于`df.loc[3:5, [“A”, “B”]]`和`df.iloc[3:5, [0, 1]]`，当在行的顶部额外指定列时，返回`df`的副本。使用`.loc`索引没有对 OG 数据帧的引用，而使用`iloc`索引会导致对已被垃圾收集的临时数据帧的引用，这与`None`本身一样好。我们看看这是否有什么意义。
*   直接使用`df[“A”]`或`df.loc[:, “A”]`引用一个列会返回一个视图，而不引用原始数据帧。这可能与每个 dataframe 列实际上被存储为一个`pd.Series`有关。

如果我们手动创建这些索引数据帧/系列的副本，会发生什么情况？

```
In [9]: show_attrs(df.loc[3:5].copy())
     …: show_attrs(df.iloc[3:5].copy())
     …: show_attrs(df.loc[3:5, [“A”, “B”]].copy())
     …: show_attrs(df.iloc[3:5, [0, 1]].copy())
     …: show_attrs(df[“A”].copy())
     …: show_attrs(df.loc[:, “A”].copy())view: False, copy: None
view: False, copy: None
view: False, copy: None
view: False, copy: None
view: False, copy: None
view: False, copy: None
```

显式调用`.copy`会返回不引用原始数据帧/序列的数据副本。在这些副本上分配数据不会影响原始数据帧，因此不会触发 SettingwithCopyWarnings。假设上面的`df.loc[3:5, [“A”, “B”]]`和`df.iloc[3:5, [0, 1]]`具有相似的属性，我们可以预期它们在链式分配下的行为应该类似于显式创建的副本。

接下来，我们将尝试几个链式分配场景。

## 场景 1:使用 loc 索引的特定行

以下三个链接的赋值引发 SettingWithCopyWarnings:

```
In [10]: df = make_clean_df()
      …: dfa = df.loc[3:5]
      …: show_attrs(dfa)view: True, copy: <weakref at 0x7fba308565e0; to ‘DataFrame’ at 0x7fba3084eac0># (1a)
In [11]: dfa[dfa % 2 == 0] = 100/tmp/ipykernel_34555/3321004726.py:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value insteadSee the caveats in the documentation: [https://pandas.pydata.org/pandas-docs/](https://pandas.pydata.org/pandas-docs/)
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
 dfa[dfa % 2 == 0] = 100
/home/tnwei/miniconda3/envs/ml/lib/python3.9/site-packages/pandas/core/
frame.py:3718: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrameSee the caveats in the documentation: [https://pandas.pydata.org/pandas-docs/](https://pandas.pydata.org/pandas-docs/)
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
 self._where(-key, value, inplace=True)# (1b)
In [12]: dfa[“D”] = dfa[“B”] * 10/tmp/ipykernel_34555/447367411.py:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value insteadSee the caveats in the documentation: [https://pandas.pydata.org/pandas-docs/](https://pandas.pydata.org/pandas-docs/)
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
 dfa[“D”] = dfa[“B”] * 10 # 1b# (1c)
In [13]: dfa[“A”][3] = 10/tmp/ipykernel_34555/1338713145.py:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrameSee the caveats in the documentation: [https://pandas.pydata.org/pandas-docs/](https://pandas.pydata.org/pandas-docs/)
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
 dfa[“A”][3] = 10
```

所有赋值对`dfa`本身生效，但只有(1a)和(1c)影响原始数据帧。(1b)没有。

```
In [14]: print(dfa)

  A  B  C   D
3 10 9  100 90
4 5  100 15 1000In [15]: print(df) A  B  C
0 1  6  11
1 2  7  12
2 3  8  13
3 10 9  100
4 5  100 15
```

另外，`dfa`不再是视图，而是 dataframe 的副本！

```
In [16]: show_attrs(dfa) # view changed to Falseview: False, copy: <weakref at 0x7fba308565e0; to ‘DataFrame’ at 0x7fba3084eac0>
```

这告诉我们，熊猫会在必要时将视图转换为副本。这进一步说明了为什么找出链式分配本质上是棘手的，并且很难在库级别自动满足。

## 场景 2:使用 iloc 索引的特定行

这与场景 1 相同，但是使用`iloc`代替。

```
In [17]: df = make_clean_df()
      …: dfb = df.iloc[3:5]
      …: show_attrs(dfb)view: True, copy: <weakref at 0x7fba30862040; to ‘DataFrame’ at 0x7fba30868c10># (1a)
In [18]: dfb[dfb % 2 == 0] = 100/tmp/ipykernel_34555/734837801.py:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value insteadSee the caveats in the documentation: [https://pandas.pydata.org/pandas-docs/](https://pandas.pydata.org/pandas-docs/)
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
 dfb[dfb % 2 == 0] = 100
/home/tnwei/miniconda3/envs/ml/lib/python3.9/site-packages/pandas/core/
frame.py:3718: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrameSee the caveats in the documentation: [https://pandas.pydata.org/pandas-docs/](https://pandas.pydata.org/pandas-docs/)
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
 self._where(-key, value, inplace=True) # (1b)
In [19]: dfb[“D”] = dfb[“B”] * 10/tmp/ipykernel_34555/4288697762.py:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value insteadSee the caveats in the documentation: [https://pandas.pydata.org/pandas-docs/](https://pandas.pydata.org/pandas-docs/)
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
 dfb[“D”] = dfb[“B”] * 10# (1c)
In [20]: dfb[“A”][3] = 10/tmp/ipykernel_34555/2062795903.py:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrameSee the caveats in the documentation: [https://pandas.pydata.org/pandas-docs/](https://pandas.pydata.org/pandas-docs/)
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
 dfb[“A”][3] = 10
```

观察到的结果与场景 1 相同。

```
In [21]: print(dfb)
  A  B  C   D
3 10 9  100 90
4 5  100 15 1000In [22]: print(df) A  B   C
0 1  6   11
1 2  7   12
2 3  8   13
3 10 9   100
4 5  100 15In [23]: show_attrs(dfb)view: False, copy: <weakref at 0x7fba30862040; to ‘DataFrame’ at 0x7fba30868c10>
```

## 场景 3:使用 loc 索引特定的行和列

与场景 1 相同，但是也指定了列。

```
In [24]: df = make_clean_df()
      …: dfc = df.loc[3:5, [“A”, “B”]]
      …: show_attrs(dfc)view: False, copy: NoneIn [25]: dfc[dfc % 2 == 0] = 100 # No warnings raised
      …: dfc[“D”] = dfc[“B”] * 10
      …: dfc[“A”][3] = 10
```

没有发出警告。所有更改在“dfc”上生效，但不影响“df”。

```
In [26]: print(dfc)

  A  B  D
3 10 9  90
4 5  100 1000In [27]: print(df) A  B   C
0 1  6   11
1 2  7   12
2 3  8   13
3 10 9   100
4 5  100 15
```

链式分配结果是不同的，而索引的数据与场景 1 中的相同。我的猜测是，对索引操作更完整的描述促使 pandas 直接提前返回一个副本，而不是链接到原始数据帧的视图。

## 场景 4:使用 iloc 索引特定的行和列

这类似于场景 3，但使用的是 iloc。鉴于过去的几个场景，这个场景与场景 3 的结果相同也就不足为奇了。

```
In [28]: df = make_clean_df()
      …: dfd = df.iloc[3:5, [0, 1]]
      …: show_attrs(dfd)view: False, copy: <weakref at 0x7fba306f29f0; dead>In [29]: dfd[dfd % 2 == 0] = 100 # No warnings raised
      …: dfd[“D”] = dfd[“B”] * 10
      …: dfd[“A”][3] = 10In [30]: print(dfd) A  B   D
3 10 9   90
4 5  100 1000In [31]: print(df) A B  C
0 1 6  11
1 2 7  12
2 3 8  13
3 4 9  14
4 5 10 15
```

此外，` dfd '在这段代码的末尾放弃了对垃圾收集数据帧的引用。

```
In [32]: show_attrs(dfd)view: False, copy: None
```

## 场景 5:直接引用数据帧的一列

此方案测试系列的链式分配。

```
In [33]: df = make_clean_df()
      …: dfe = df[“A”]
      …: show_attrs(dfe)view: True, copy: NoneIn [34]: dfe[1] = 99999 # No warnings raised
      …: dfe.loc[2:4] = 88888
```

`dfe`保留了`df[“A”].` 的视图所有影响`dfe`的变化都反映在`df[“A”]`中，它仍然是`df`的一部分。对于单个系列的链式作业，似乎没有什么可担心的。

```
In [35]: print(dfe)0 1
1 99999
2 88888
3 88888
4 88888
Name: A, dtype: int64In [36]: print(df) A     B C
0 1     6 11
1 99999 7 12
2 88888 8 13
3 88888 9 14
4 88888 10 15In [37]: show_attrs(dfe)view: True, copy: None
```

# 正在删除 SettingWithCopyWarnings

当 pandas 不确定是否希望值赋值影响原始数据帧时，会弹出 SettingWithCopyWarnings。因此，消除这些警告需要避免赋值的模糊性。从上面的代码示例可以看出，让 pandas 返回不引用原始数据帧的副本是一种干净的方法，可以确保值不会被意外地写入原始数据帧。

我发现这是我在研究这个主题时遇到的解决方案中的一个统一线索。总结如下:

## 禁用警告

如果您知道自己在做什么，并且代码的行为符合预期，您可以选择通过禁用警告来取消警告[ ]:

```
# Example borrowed from [³]# Disables SettingWithCopyWarning globally
pd.set_option(‘mode.chained_assignment’, None)# Resets the warning option to default
pd.reset_option(‘mode.chained_assignment’)# Disables SettingWithCopyWarning locally within a context manager
with pd.option_context(‘mode.chained_assignment’, None):
    # YOUR CODE HERE
```

或者，您可以通过将 dataframe `_is_copy`属性设置为`None` [ ]，取消警告。

```
# Example modified from [3]
In [38]: df = pd.DataFrame({
      …:     “A”: [1, 2, 3, 4, 5],
      …:     “B”: [6, 7, 8, 9, 10],
      …:     “C”: [11, 12, 13, 14, 15],
      …: }, index=range(5))
      …: 
      …: dfa = df.loc[3:5]
      …: print(dfa._is_copy)<weakref at 0x7f4d64792810; to ‘DataFrame’ at 0x7f4d64784460>In [39]: dfa._is_copy = None
      …: dfa[“D”] = dfa[“B”] * 10 # No warning raised
```

请记住，让警告消失并不能解决不可靠的链式分配问题。链式作业是一个雷区，你可能会也可能不会踩到地雷。禁用警告就像移除雷区警告标志。精神食粮。

## 让警告不在第一时间出现

当遇到 SettingWithCopy 警告时，请花点时间跟踪链式赋值，并决定是要直接修改原始数据帧，还是将值赋给数据帧的副本。

## 处理原始数据帧

使用`.loc`索引直接给数据帧赋值。

```
# Modified from examples in [2]
In [40]: df = pd.DataFrame({
 …: “A”: [1, 2, 3, 4, 5],
 …: “B”: [6, 7, 8, 9, 10],
 …: “C”: [11, 12, 13, 14, 15],
 …: }, index=range(5))
 …:
 …: df.loc[df[“A”] % 2 != 0, “B”] = df.loc[df[“A”] % 2 != 0, “B”] + 0.5
 …: print(df)A B C
0 1 6.5 11
1 2 7.0 12
2 3 8.5 13
3 4 9.0 14
4 5 10.5 15
```

熊猫医生推荐这种方法有两个原因:

*   使用`.loc`肯定会引用它所调用的底层数据帧。`.iloc`不具备这个属性。
*   `.loc`步进取代了链式步进，成为单一步进步骤。如果您参考上面带有视图和副本的[链式分配下的示例，`.loc`索引将链式索引解析为单个`__setitem__`调用。](#5424)

如果使用条件选择数据，可以考虑返回一个掩码，而不是原始数据帧的副本。屏蔽是一个布尔序列或数据帧，可方便地用于`.loc`索引，如下例所示:

```
# Modified from examples in [5]
In [41]: df = pd.DataFrame({
      …:     “A”: [1, 2, 3, 4, 5],
      …:     “B”: [6, 7, 8, 9, 10],
      …:     “C”: [11, 12, 13, 14, 15],
      …: }, index=range(5))In [42]: dfa = (df[“A”] <= 3) & (df[“C”] == 12)In [43]: df.loc[dfa, “B”] = 99 # dfa can be fed into the loc index!In [44]: print(df) # changes took place in the original dataframe A B  C
0 1 6  11
1 2 99 12
2 3 8  13
3 4 9  14
4 5 10 15
```

如果现有的索引逻辑很复杂，直接在原始数据帧上工作可能会很棘手。在这种情况下，您可以使用下一节中的方法之一返回一个副本，然后将它赋回原始数据帧[⁴].

## 为数据帧的显式副本赋值

使用`assign`、`where`和`replace`:

```
In [45]: df = pd.DataFrame({
      …:     “A”: [1, 2, 3, 4, 5],
      …:     “B”: [6, 7, 8, 9, 10],
      …:     “C”: [11, 12, 13, 14, 15],
      …: }, index=range(5))# 1\. Use the `assign` method to add columns
In [46]: df = df.assign(D=df[“C”] * 10)
      …: df = df.assign(**{“D”: df[“C”] * 10}) # allows passing variables as names# 2\. Use the `where` method to select values using conditionals and replace them
# Modified from examples in [2]
In [47]: df[“B”] = df[“B”].where(
      …:     df[“A”] < 2, df[“B”] * 10
      …: )# 3\. Use the `replace` method to select and replace values in the dataframe
# Modified from examples in [2]
In [48]: df = df.replace({“A” : 1}, 100)In [49]: print(df)
  A   B  C  D
0 100 6  11 110
1 2  70  12 120
2 3  80  13 130
3 4  90  14 140
4 5  100 15 150
```

将连锁作业步骤分解成单个作业[⁵]:]

```
# Examples borrowed from [4]
# Not these
df[“z”][mask] = 0
df.loc[mask][“z”] = 0# But this
df.loc[mask, “z”] = 0
```

一种不太优雅但十分简单的方法是手动创建原始数据帧的副本，并对其进行处理[ ]。只要不引入额外的链式索引，就不会看到 SettingWithCopyWarning。

```
In [50]: df = pd.DataFrame({
      …:     “A”: [1, 2, 3, 4, 5],
      …:     “B”: [6, 7, 8, 9, 10],
      …:     “C”: [11, 12, 13, 14, 15],
      …: }, index=range(5))In [51]: dfa = df.loc[3:5].copy() # Added .copy() here
      …: dfa.loc[3, “A”] = 10 # causes this line to raise no warning
```

# 重复上面的一些示例，而不触发带有副本的设置警告

用`where`替换链接的赋值:

```
# (i)
df = make_clean_df()
dfa = df.loc[3:5]# Original that raises warning
# dfa[dfa % 2 == 0] = 100dfa = dfa.where(dfa % 2 != 0, 100) # df is not affected
```

将在索引数据帧上创建新列替换为`assign`:

```
# (ii) 
df = make_clean_df()# Original that raises warning
# dfa[“D”] = dfa[“B”] * 10dfa = dfa.assign(D=dfa[“B”]*10) # df is not affected
```

在使用`.loc`索引赋值之前，创建数据帧的副本:

```
# (iii)
df = make_clean_df()# Original that raises warnings
# dfa = df.loc[3:5]
# dfa[“A”][3] = 10# Create a copy then do loc indexing
dfa = df.loc[3:5].copy()
dfa.loc[3, “A”] = 10
```

注意，使用`.loc`索引直接给`dfa`赋值仍然会引发警告，因为对`dfa`的赋值是否也会影响`df`还不清楚。

# 真正根除带有复制警告的设置

就我个人而言，我喜欢将重要脚本的 SettingWithCopyWarnings 提升为 SettingWithCopyExceptions，使用以下代码:

```
pd.set_option(‘mode.chained_assignment’, “raise”)
```

这样做可以强制处理链式赋值，而不是允许警告累积。

根据我的经验，清理被 SettingWithCopyWarnings 阻塞的带有 stderr 的笔记本是一种特殊的禅。我真心推荐。

原创博文:[在熊猫身上设置版权警告](https://tnwei.github.io/posts/setting-with-copy-warning-pandas/)

[ ]:链式任务的官方“熊猫”文件。
[[https://pandas . pydata . org/docs/user _ guide/indexing . html # returning-a-view-vs-a-copy](https://pandas . pydata . org/docs/user _ guide/indexing . html # returning-a-view-vs-a-copy)](https://pandas.pydata.org/docs/user_guide/indexing.html#returning-a-view-versus-a-copy](https://pandas.pydata.org/docs/user_guide/indexing.html#returning-a-view-versus-a-copy))

[ ]:这是一篇关于数据科学的文章，简要介绍了几种处理带有拷贝警告的设置的方法。
[[https://scribe . rip/@ towards data science . com/3-solutions-for-the-setting-with-copy-warning-of-python-pandas-dfe 15d 62 de 08】(https://scribe . rip/@ towards data science . com/3-solutions-for-the-setting-with-copy-warning-of-python-pandas-dfe 15d 62 de 08)](https://scribe.rip/@towardsdatascience.com/3-solutions-for-the-setting-with-copy-warning-of-python-pandas-dfe15d62de08](https://scribe.rip/@towardsdatascience.com/3-solutions-for-the-setting-with-copy-warning-of-python-pandas-dfe15d62de08))

[]:data quest 关于此主题的深入文章。值得注意的是，在《熊猫》中有一节专门讲述了处理链式分配的历史。
[[https://www . data quest . io/blog/settingwithcopywarning/](https://www . data quest . io/blog/settingwithcopywarning/)](https://www.dataquest.io/blog/settingwithcopywarning/](https://www.dataquest.io/blog/settingwithcopywarning/))

[⁴]: StackOverflow 后，包含更多的连锁转让的例子。[[https://stack overflow . com/questions/48173980/pandas-known-when-a-operation-affects-the-original-data frame](https://stack overflow . com/questions/48173980/pandas-known-when-a-operation-affects-the-original-data frame)](https://stackoverflow.com/questions/48173980/pandas-knowing-when-an-operation-affects-the-original-dataframe](https://stackoverflow.com/questions/48173980/pandas-knowing-when-an-operation-affects-the-original-dataframe))

[⁵]: RealPython 的文章覆盖了这个主题。对我来说，RealPython 是仅次于官方图书馆文档的值得信赖的 goto 参考资料。本文进一步深入研究了 pandas 和 numpy 中的底层视图和复制机制，pandas 依赖于 numpy。
[[https://real python . com/pandas-settingwithcopywarning/](https://real python . com/pandas-settingwithcopywarning/)](https://realpython.com/pandas-settingwithcopywarning/](https://realpython.com/pandas-settingwithcopywarning/))