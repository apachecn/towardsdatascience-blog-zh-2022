# 如何用 Python 和熊猫把 CSV 文件转换成 XLSX 文件

> 原文：<https://towardsdatascience.com/how-to-convert-a-csv-file-into-an-xlsx-one-with-python-and-pandas-27aabc279d69>

## 如何使用 Python 转换文件的示例

![](img/0d5e1fa621cf70c90f08f4091e81000e.png)

亚历克斯·丘马克在 [Unsplash](https://unsplash.com/s/photos/python?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

ython 是一种非常灵活的编程语言。有人会说这是“万事通，无所不能”；但是，在我看来，它在许多任务中提供帮助的能力——从“基本编程”到机器学习——是 Python 的真正力量。

几天前，我想分析一些可以从网站上下载的数据，以便用 Tableau 做一些练习。不幸的是，当我试图下载 excel 格式的数据时——这是 Tableau 可以接受的格式之一——我收到了一条错误消息。所以我试着下载 CSV 格式的数据，没有任何问题。

事实是，Tableau 甚至接受 CSV 文件，但我们必须做一些调整，我更喜欢使用 XLSX 文件。所以问题是:我如何将 CSV 文件转换成 XLSX？嗯，我会用 Python！因此，在本文中，我们将看到如何做到这一点。

# 数据的来源

我想分析欧洲的粮食产量，我发现粮农组织网站给了我们一些数据(链接[此处](https://www.fao.org/faostat/en/#data/QCL))可以下载使用。但是当我试图下载 XLXS 中的数据时，出现了错误:

![](img/712a12189059dc10da5f2e31e9d87398.png)

粮农组织网站的错误。图片作者。

当然，我已经尝试了我能做的任何事情:我试了几次重新下载数据，改变选择，改变电脑，等等…但是我仍然得到同样的错误。所以，我试着用 CSV 格式下载它们，然后…砰！第一时间下载数据。

所以，现在，我遇到了将 CSV 文件转换成 XLSX 文件以加载到 Tableau 中的问题。让我们看看我是怎么做的。

# 用熊猫将 CSV 转换为 XLSX

因为我们处理的是 CSV，所以我们可以像往常一样打开熊猫:

```
import pandas as pd#opening data
grain = pd.read_csv('grain.csv')#showing head
grain.head()
```

![](img/e21446c33c72cc2ec679afff1beca5ad.png)

下载的数据帧。图片作者。

如果我们稍微研究一下数据，我们会发现有些国家的产值为 0 吨。因此，在保存 Tableau 的数据之前，因为我们使用 Python，所以删除这些行可能是个好主意。我们可以用下面的代码来实现:

```
#selecting rows with 'Value'!=0
grain = grain.loc[grain['Value']!=0]
```

现在，我们终于准备好将 CVS 转换成 XLSX 我们可以通过下面的代码轻松做到这一点:

```
#saving to xlsx
grain.to_excel('grain_excel.xlsx')
```

这样，我们简单地将文件保存在 XLSX 中，将其命名为“grain_excel.xlsx”。

# 概括代码

假设我们只想转换文件，而不清理数据；我们可以创建这样一个简单的脚本:

```
import pandas as pd#opening data
open_data = pd.read_csv('input_file.csv')#saving to xlsx
open_data.to_excel('output_file.xlsx')
```

上面的代码只是打开一个你需要命名为“input_file.csv”的 CSV 文件，返回一个 Excel 文件，命名为“output_file.xlsx”。您可以将此代码另存为”。py”文件，并在需要时运行它。就这么简单。

# 结论

我们已经看到，如果我们需要，将 CSV 文件转换成 XLSX 文件是多么容易。熊猫是一个非常强大的库，可以帮助我们完成这个简单的任务。

然后，如果我们需要多次做这样的转换，我们可能会创建一个通用的”。py”转换器文件，这样我们就可以在任何需要的时候运行它，它可以非常快速轻松地完成工作。

*我们一起连线吧！*

[中型 ](https://federicotrotta.medium.com/)

[*LINKEDIN*](https://www.linkedin.com/in/federico-trotta/) *(向我发送连接请求)*

[推特](https://twitter.com/F_Trotta90)

*如果你愿意，你可以* [*订阅我的邮件列表*](https://federicotrotta.medium.com/subscribe) *这样你就可以一直保持更新了！*

考虑成为会员:你可以免费支持我和其他像我一样的作家。点击 [*这里的*](https://federicotrotta.medium.com/membership) *成为会员。*