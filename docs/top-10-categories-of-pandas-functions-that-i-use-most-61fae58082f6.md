# 我最常用的熊猫功能的 10 大类

> 原文：<https://towardsdatascience.com/top-10-categories-of-pandas-functions-that-i-use-most-61fae58082f6>

## 熟悉这些函数可以帮助您处理数据

![](img/92a3013d60a539695174a39fd881c19f.png)

照片由[Firmbee.com](https://unsplash.com/@firmbee?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

人们喜欢使用 Python，因为它有一个用于各种工作的第三方库的通用存储库。对于数据科学，最受欢迎的数据处理库之一是 Pandas。多年来，由于它的开源特性，许多开发人员为这个项目做出了贡献，使得 pandas 对于几乎任何数据处理工作都非常强大。

我没数过，但我觉得熊猫有上百种功能可以使用。虽然我经常使用 20 或 30 个函数，但把它们都谈一谈是不现实的。因此，在这篇文章中，我将只关注 10 个最有用的函数类别。一旦你和他们相处得很好，他们可能会满足你超过 70%的数据处理需求。

## 1.读取数据

我们通常从外部来源读取数据。根据源数据的格式，我们可以使用相应的`read_*`函数。

*   `read_csv`:当源数据为 CSV 格式时使用。一些值得注意的参数包括`header`(是否以及哪一行是标题)、`sep`(分隔符)和`usecols`(要使用的列的子集)。
*   `read_excel`:当您的源数据为 Excel 格式时使用。一些值得注意的参数包括`sheet_name`(哪个工作表)和 header。
*   `read_pickle`:当你的源数据是一个酸洗的`DataFrame`时使用它。Pickling 是一种存储数据帧的好机制，通常比 CSV 和 Excel 要好。
*   `read_sas`:我经常使用这个功能，因为我以前用 SAS 处理数据。

## 2.写入数据

处理完数据后，您可能希望将数据帧保存到文件中，以便长期存储或与同事进行数据交换。

*   `to_csv`:写入 CSV 文件。它不保留某些数据类型，比如日期。尺寸往往比别人大。我通常将参数索引设置为`False`，因为我不需要额外的列来显示数据文件中的索引。
*   `to_excel`:写入 Excel 文件。
*   `to_pickle`:写入 pickle 文件。正如我刚才提到的，我使用 pickled 文件，这样当我读取它们时，数据类型可以被正确地保留。

## 3.数据总结/概述

将数据读入数据帧后，获取数据集的一些描述符是一个好主意。

*   `head`:检查前几行，看数据是否读取正确。
*   `tail`:检查最后几行。这同样重要。当您处理一个大文件时，读取可能是不完整的。通过检查尾部，你会发现阅读是否已经完成。
*   `info`:对数据集有一个整体的总结。一些有用的信息包括列的数据类型和内存使用情况。
*   `describe`:提供数据集的描述性摘要。
*   `shape`:行数和列数(是属性，不是函数)。

## 4.分类数据

我通常在完成大部分其他处理步骤后对数据进行分类。特别是，如果我要将数据帧写入外部文件，比如 Excel，我几乎总是在导出之前对数据进行排序。这是因为排序后的数据更容易让其他人利用眼球定位到需要的信息。

*   `sort_values`:通过指定列名对数据进行排序。因为我主要处理行是观察值的文件，所以排序是按列进行的。

## 5.处理重复

当我们处理现实生活中的数据集时，很可能会有重复的数据集。例如，有些数据意外地在数据源中输入了两次。重要的是删除重复的内容。

*   `duplicated`:识别数据帧中是否有重复。您可以指定使用哪些列来标识重复项。
*   drop_duplicates:从数据帧中删除重复项。我不盲目使用这个功能。为了谨慎起见，我总是首先使用`duplicated`函数来检查重复项。

## 6.处理缺失值

数据集中存在缺失值几乎是不可避免的。检查数据集的缺失并决定如何处理缺失值是一个很好的做法。

*   `isnull`:检查您的数据帧是否丢失。
*   `dropna`:删除缺失数据的观测值。值得注意的参数包括`how`(如何确定是否删除一个观察值)和`thred`(符合删除条件的缺失值的数量)。
*   `fillna`:按照指定的方式填充缺失值，如向前填充(`ffill`)。

## 7.提取新数据

列可以包含多条信息。例如，我们的数据集可能有像 proj-0001 这样的数据，其中前四个字母是项目的缩写，而最后四个数字是主题的唯一 ID。为了提取这些数据，我经常使用以下函数。

*   `map`:使用单个列中的信息创建一个列。换句话说，你在一个`Series`对象上调用这个函数，比如`df[“sub_id”] = df[“temp_id”].map(lambda x: int(x[-4:]))`。
*   `apply`:使用多列数据创建一列或多列。在创建列时，您经常需要指定`axis=1`。

## 8.转换数据

通常有两种数据。一种是“宽”格式，指的是每行代表一个单独的主题或观察，列包括对该主题的重复测量。另一种是“长”格式。在这种格式中，一个主题有多行，每一行可以代表某个时间点的度量。通常，您可能需要在这两种格式之间转换数据。

*   `melt`:将宽数据集转换为长数据集。值得注意的参数包括`id_vars`(用于标识符)和`value_vars`(其值构成值列的列列表)。
*   `pivot`:将长数据集转换为宽数据集。值得注意的参数包括`index`(唯一标识符)、`columns`(成为值列的列)和`values`(具有值的列)。

## 9.合并数据集

当您有单独的数据源时，您可能希望合并它们，这样您就有了一个组合的数据集。

*   `merge`:将当前的与另一个合并。您可以指定一列或多列作为合并的标识符(参数`on`，或`left_on` & `right_on`)。其他值得注意的参数包括`how`(如 inner 或 left，或 outer)，以及`suffixes`(两个数据集使用什么后缀)。
*   `concat`:沿行或列连接 DataFrame 对象。当您有多个相同形状的 DataFrame 对象/存储相同的信息时，这很有用。

## 10.按组汇总

我们的数据集通常包括指示数据特征的分类变量，例如学生的学校、科目的项目以及门票的班级级别。

*   `groupby`:创建一个 GroupBy 对象，可以指定一列或多列。
*   `mean`:您可以在 GroupBy 对象上调用 mean 来找出平均值。你可以对其他属性做同样的事情，比如`std`。
*   `size`:群体频率
*   `agg`:可定制的聚合功能。在这个函数中，您可以请求对指定的列进行统计。

## 结论

在这篇文章中，我回顾了我在日常数据处理工作中经常使用的 10 大类函数。虽然这些评论很简短，但它们为你组织学习熊猫提供了指导方针。

希望这篇文章对你有用。感谢阅读。