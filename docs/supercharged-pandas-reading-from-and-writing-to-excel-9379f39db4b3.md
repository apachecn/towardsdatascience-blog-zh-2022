# 超级熊猫:从 Excel 中读取和写入

> 原文：<https://towardsdatascience.com/supercharged-pandas-reading-from-and-writing-to-excel-9379f39db4b3>

## 增强。read_excel &。to_excel 方法，让您专注于数据探索

![](img/7066d19fab51c0b8b17c5d21214bef86.png)

读取和写入 Excel(图片由作者提供)

# 谁应该读这个？

如果您已经使用 python pandas 库读取 Excel 文件，分析/转换数据，并将 pandas 数据帧写入 Excel，那么我将要分享的内容可能会让您受益。您可能已经经历了以下一种或多种情况:

*   使用`pd.read_excel`，将数据作为数据帧读取，进行一些分析，然后意识到您使用了错误的工作表进行分析。(向您提出要求的人并没有指定您应该使用`Sheet 1`而不是`Sheet 2`。熊猫默认读取，不抛出任何警告。)
*   运行一个函数进行数据分析，然后在最后一步，当您调用`df.to_excel`方法时，您意识到您没有键入`.xlsx`作为文件的扩展名，这导致了一个错误。你必须从头到尾重新运行整个脚本，浪费宝贵的时间。
*   您更喜欢输出到 Excel 来创建图表或数据透视表，因为最终用户更喜欢在 Excel 中查看输出，但总是要双击才能打开 Excel 输出文件是一件痛苦的事情。(您希望有一种方法可以在 Excel 文件编写完成后自动打开它。)

# 我为什么要写这个？

我经历过上述情况，我理解这有多么令人沮丧。更重要的是，这些情况会打断一个人在解决问题时的思路。作为一名为项目从一家银行转到另一家银行的顾问，我完全具备消除这些干扰的工具是至关重要的。通过分享我如何改进了`read_excel`和`to_excel`方法，我希望你也能从中受益。如果你扩展并构建了这些方法，也请与我分享！:D

完整的代码(pypo.py)可以在页面的最后找到，这里是。

# 增强型“`read_excel`”方法

增强型`read_excel`方法有 2 个关键功能:

1.  如果要读取的文件包含多个 Excel 工作表，它将打印每个工作表的名称以及每个工作表的行数和列数。人们可以通过输入`inspect=True`作为关键字参数来开启这种行为。pythonista 还会认识到，可以通过将`kw.get('inspect', True)`中的第二个参数从`True`调整为`False`来配置默认行为。
2.  如果没有提供文件名，会出现一个`tkinter`文件对话框，提示您选择一个 Excel 文件。这在数据浏览中非常有用，因为您可能更喜欢通过用户界面导航到文件，而不是键入文件的完整路径。

要在现有脚本中包含这个函数，假设您有一个名为`helloworld.py`的 python 脚本，只需将`pypo.py`放在同一个文件夹中，然后添加`import pypo`并将`pd.read_excelp = pypo.read_excelp`包含在`helloworld.py`中

# 增强型“`to_excel`”方法

数据分析师可能更喜欢将数据输出到 Excel 进行简单分析，或者准备一些报告和/或数据透视表/图表。在编写脚本的过程中，通常会经常执行脚本，然后打开 Excel 文件检查输出文件。这里有两个干扰因素:

1.  手动打开 Excel 需要双击 Excel 文件，或者按`Alt + Tab`并按箭头键导航到该文件。

> 如果文件能自动打开，这样我们就不需要在键盘和鼠标之间切换了，这不是很好吗？

2.您需要始终使用一个唯一的名称，否则如果它们共享相同的文件名，新的输出文件将会覆盖以前的文件。

> 如果有一种优雅的方式来重命名文件，如果存在另一个同名的文件，这不是很好吗？

该函数将数据帧写入 Excel 并打开它。如果您提供的文件名已经存在于目标文件夹中，那么它会在文件名中添加一个时间字符串(“YYYYMMDD_HHMMSS”)。如果您输入的文件名不包含 Excel 扩展名，则默认情况下会添加`.xlsx`。

要将此功能包含到您现有的脚本中，只需`import pypo`即可。最后一行代码，即`PandasObject.to_excelp = to_excelp`将`to_excelp`方法添加到所有 pandas dataframe 对象中

# 结论

对原始 pandas 方法的这些改进是出于一种需要。在几家银行的多个项目中工作过之后，我创建了自己的脚本小工具箱来加速我的工作。

如果你发现这个脚本已经或将要帮助你，请喜欢并分享我的文章，因为这将极大地鼓励我分享更多我工具箱里的东西。

也请看看我的其他文章:

[](/how-to-compare-2-dataframes-easily-b8f6788d5f07) [## 如何轻松比较两个数据帧

### …使用全面的数据协调脚本。

towardsdatascience.com](/how-to-compare-2-dataframes-easily-b8f6788d5f07) [](/how-to-create-a-list-of-files-folders-and-subfolders-and-then-export-as-excel-6ce9eaa3867a) [## 如何创建文件、文件夹和子文件夹的列表，然后导出为 Excel

### 一个生产力工具玛丽近藤你的文件夹

towardsdatascience.com](/how-to-create-a-list-of-files-folders-and-subfolders-and-then-export-as-excel-6ce9eaa3867a) 

## 警告

我只用 Windows 对于 Mac 用户来说，真的很抱歉，你将需要找到替代品或者相当于`win32com.client`的东西。这里有一个堆栈溢出[答案](https://stackoverflow.com/a/29629721/8350440)，它可以为你指出正确的方向。

我不使用 IDE 我使用可靠的 Notepad++和 Powershell。对我来说，这是我被分配到的所有公司中最可靠、最稳定的。我不使用 Juypter 笔记本，我很确定`tkinter`和/或`win32com.client`在 Juypter 笔记本上不能很好地工作。

# 完整代码