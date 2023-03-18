# 用当前日期字符串重命名文件的 3 种自动方法

> 原文：<https://towardsdatascience.com/3-automated-ways-to-rename-a-file-with-a-current-date-string-3c787a5eb78e>

## 分别使用 VBA 宏、批处理文件命令和 Python

![](img/f6eb359c418a8a6982de69df68d497d7.png)

照片由 [Maddi Bazzocco](https://unsplash.com/@maddibazzocco?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

# 背景

可以在数据准备工作流的早期阶段使用数据科学技术来自动化手动任务。具体来说，在我职业生涯的许多实例中，我不得不使用带有时间戳的固定命名约定来重命名文件。这是为了使下游的工作流程能够区分数据或识别随时间变化的趋势。虽然只手动执行几个文件的重命名是可以接受的，但如果有数百个文件，就不可以了，这在商业环境中是常见的情况。

通过一个用例，本文介绍了三(3)种基于当前或预定义日期使用字符串重命名文件的自动化方法，分别使用 **Excel VBA 宏**和**批处理文件命令**和 **Python** 。

# 用例

在我职业生涯的早期，我有机会在一家人寿保险公司担任再保险分析师，该公司将再保险业务转让给多家再保险人。由于再保险人最终是直接保险公司运营的外部第三方，我的部分职责包括每月为再保险人准备各种报告。对于这个用例，我准备的一个报告是系统在一个月的最后一个工作日生成的所有有效策略的摘录。

这个摘录和系统生成的许多其他再保险报告一样，有一个固定的(陈旧的)名称 *inforce_extract.csv* 。当时准备这些报告的流程要求我通过添加再保险人标签和日期字符串(例如*inforce _ extract _ reinsurer 1 _ 201206 . CSV*)来手动重命名这些报告。对于多个再保险人的数百份报告，我遵循这一手动流程两个月，然后决定这是不可行的，因为它非常耗时且容易出现手动错误。

下面我将首先介绍我当时分别使用 **Excel VBA 宏**和**批处理文件命令**实现的两种方法，以取代上面概述的手动重命名过程(这两种方法最终使我获得了重大的过程改进)，然后用 **Python** 中的一种实现来总结这一点，以帮助那些可能不熟悉前两种方法下的编程语言的人。

# Excel VBA 宏

对于那些熟悉 Excel 的人来说， **Excel VBA 宏**以通过在其内置的 VBA 编辑器中创建定制函数来自动化手动 Excel 相关任务而闻名。在这种情况下，由于大多数报告是在。csv 格式，我想我可以写一个定制的功能，自动执行以下步骤:

1.  打开底层。csv 报告
2.  在函数中创建一个软编码变量，用于捕获引用日期的字符串(例如，当前报告日期的年和月，或“yyyymm”)
3.  保存。csv 报告为另一个文件，采用选择的格式，文件名中包含步骤 2 中的字符串

以 inforce_extract.csv 报告为例，这可以通过运行以下 VBA 宏来实现:

**第一步**

```
' Open the system generated report
' path needs to be defined as path to directory of the system
' generated reportWorkbooks.Open Filename:=path & "\inforce_extract.csv"
Set inforce_extract = Application.Workbooks("inforce_extract.csv") 
```

**第二步**

```
' DateAdd function allows one to soft-code a date referencing a 
' pre-defined date' The code below would return 202112 at time of writing, which is 3 ' months in arrears to March 2022 date_string = Format(DateAdd("m", -3, Now), "yyyymm")
```

**第三步**

```
' Save the extract as "ïnforce_extract_reinsurerX_202112.csv"
' SAVEpath needs to be defined as the path to the directory where
' we want to save the renamed report
' There is also an option to save the file in another format, xlCSV ' is chosen for this instanceinforce_extract.SaveAs _
Filename:=SAVEpath & "inforce_extract_reinsurerX" & date_string & ".csv", _
FileFormat:=xlCSV, CreateBackup:=False, 
inforce_extract.Saved = True
```

虽然作为每月重命名过程的一部分，我花了一些时间为所有需要重命名的文件编写代码，但这是值得的，因为我只需点击一下鼠标，就可以重命名数百份引用最新报告日期的报告。

值得注意的是，在**步骤 2** 中引入的 DateAdd 函数也可以用于以类似的方式在**步骤 1** 中打开一个已经带有日期字符串的文件。

# 批处理文件命令

没过多久，我就对 VBA 宏自动化的过程感到不满意，因为我突然想到，我应该能够更有效地重命名报告，而不需要打开它们。这类似于在 Windows 目录中的文件上按 F2 键并键入新名称。

由于这是 Excel 之外的操作，VBA 宏不再是合适的工具。当时我的一位同事给我指了指**批处理文件命令**。简单来说，这是一种用于在 Windows 操作系统中创建可执行脚本文件的编程语言。

脚本文件可以在文本文件中编译和编辑(例如，在 Windows 的记事本中)，并以. bat 扩展名保存。然后，只需双击。蝙蝠文件。

我需要遵循的步骤有效地减少到只有前面章节中的**步骤 2** 和**步骤 3** 。概括地说，这只是基于最近的报告日期捕获日期字符串，并使用捕获再保险人名称(在报告日期中不变)和日期字符串的命名约定来重命名系统生成的报告，而不需要打开底层文件。

要在脚本文件中基于当前日期(即今天)捕获日期字符串(注意“@REM”是在脚本文件中显示注释的方式):

```
@REM Get the current date, by capturing the first 10 substrings  @REM This captures a date of dd/mm/yyyy
set stamp=%date:~-10% @REM Get the year (e.g. 2022), by extracting the 7th to 10th        @REM substrings
set year=%stamp:~6,4% @REM Get the month (e.g. 03), by extracting the 4th to 5th       @REM substrings
set month=%stamp:~3,2%
```

如果我需要捕获基于非当前日期的日期字符串，需要一些额外的步骤。下面的脚本演示了如何根据比当前日期晚 3 个月的日期提取日期字符串“yyyymm”:

```
@REM subtract 3 months
set /a month=1%month%-103 @REM "Rollback" year and month if the above is negative           @REM e.g. 101 for January - 103 would be negative, then add 12 to @REM month and subtract 1 from year
if %month% LSS 1  (
  set /a month=%month%+12
  set /a year=%year%-1
) @REM Prepend with zero for single-digit month numbers (i.e. if month @REM number is 9 or less, add a leading 0)
if %month% GTR 9 (set month_final=%month%) 
else (set month_final=0%month%)
```

最后，要使用再保险人和日期字符串重命名报告:

```
@REM Define current directory, path needs to be defined as path to @REM directory of the system generated report
cd "path"@REM Rename the inforce_extract.csv report in the path
rename inforce_extract.csv inforce_extract_reinsurerX_%year%%month_final%.csv
```

# 计算机编程语言

与上面的两种方法相比，我认为使用以下步骤在 Python 中可以更容易地实现相同的重命名任务:

**第一步:导入库**

```
from datetime import datetime
import os
```

**第二步:定义要重命名的文件**

```
## Path to be defined as directory where the file is saved
inforce_extract = 'path/inforce_extract.csv'
```

**第三步:创建日期字符串并为文件定义新名称**

```
## The below returns yyyymm, or 220203 at time of writing
date_string = datetime.today().strftime('%Y%m')## Define new name
new_name = 'path/inforce_extract_reinsurerX_'+ date_string +'.csv'
```

**第四步:重命名**

```
os.rename(inforce_extract, new_name)
```

# 总结想法

我观察到，与我一起工作的大多数人鄙视手工过程，而一些人选择接受，另一些人采取额外的步骤来自动化它们。

我很感激在我职业生涯早期所处的环境中，我被迫使用编程来改进过程。在接下来的几年里，这种有益的经历继续激励我学习各种应用程序的编程，自动化无疑是其中之一，除此之外，数据科学和机器学习也是我日常工作的应用程序。

如果你喜欢你正在阅读的东西，一定要访问我的个人资料，在这里查看我写的其他文章。