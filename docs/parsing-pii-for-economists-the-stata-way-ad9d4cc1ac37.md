# 为经济学家解析 PII——STATA 方式

> 原文：<https://towardsdatascience.com/parsing-pii-for-economists-the-stata-way-ad9d4cc1ac37>

## 良好的编码实践和伪代码，用于在 STATA(经济学家的常用语言)中从个人层面数据解析 PII

![](img/249689b924d7c2fa15ef64ae669f5eac.png)

来自[像素](https://www.pexels.com/ko-kr/photo/ipad-187041/)的免费使用照片

# 介绍

在过去的两年里，我在一个经济学实验室工作，对社会科学研究领域的各种概念越来越熟悉。特别是，PII(代表个人身份信息(PII))包含在应用程序、调查和管理操作中收集的各种个人级别的数据中。当研究人员想要将不同的数据集链接在一起，并在不同的数据或时间跨度上跟踪个人的运动或活动时，PII 尤其有价值。这允许在研究中进行更动态和更深入的分析。然而，PII 本身是敏感的，因此需要谨慎对待。这就是为什么包含 PII 的数据通常会被严格的安全协议分开处理。无论如何，本文将考察一些**的好的编码实践和伪代码，以便使用 STATA——许多经济学的首选软件——解析出 PII。**

# 使用局部变量和全局变量

尽管 STATA 不是编程语言，但良好的编码实践仍然适用。使用局部变量和全局变量可以更方便地管理代码路径。假设您正在对包含您需要使用的文件的目录的文件路径进行硬编码。如果要修改路径，就必须修改使用该路径的每一行代码。这就是本地人和全球人发挥作用的地方。

局部变量可以是任何东西，从标量到变量，再到存储在局部内存中的变量列表和字符串。这里，本地内存意味着在脚本中定义的本地变量只在该脚本中有效。它们将无法从另一个脚本中引用。另一方面，全局变量可以被全局引用，不管它是从哪个脚本调用的。

我们假设有一个你要读入的数据集叫做“pii_data.dta”。它位于“O:/josh/data”中。您可以通过以下两种方式定义数据目录路径:

```
local data_dir = “O:/josh/data”global data_dir = “O:/josh/data”use "`data_dir'/pii_data.dta" # read in data using local pathuse "$data_dir/pii_data.dta" # read in data using global path
```

请注意局部变量和全局变量在语法上的不同。STATA 局部使用反勾号(`)和撇号(')，而全局使用美元符号($)。

# 定义模式

在从数据中解析出 PII 之前，应该先定义一个“模式”。这里定义一个模式意味着确定我们将提取什么类型的 PII，以及信息将存储在什么变量中。这取决于你从事的研究或项目试图从 PII 中得到什么。全名是否足够，还是必须分开成单独的列(例如，名、中间名、姓和后缀)?).您只对每个人的年龄感兴趣，还是对出生的具体年月日感兴趣？

在本例中，我们将假设数据包含全名、出生日期、地址和邮政编码信息。我们将模式定义为以下变量:

名 _ 名|名 _ 中间|名 _ 姓|名 _ 后缀| dob_yyyy | dob_mm | dob_dd |地址 _bldnum |地址 _ 字符串|地址 _ 城市|地址 _st |地址 _zip

# 解析名称

名字可以有许多不同的格式。在某些情况下，它只出现在一列中，每个单元格中有全名。在其他情况下，它出现在单独的列中，每个列代表全名的一部分(例如姓氏)。就我个人而言，我发现后缀很少被提取到单独的列中。不幸的是，为了增强实体解析的性能(即，如果那些观察值利用概率匹配算法共享相似的 PII，则链接不同数据集中的不同数据点并给予它们相同的唯一个体标识符的过程)，具有诸如后缀的附加信息有助于将一个数据点与另一个数据点区分为两个不同的个体或实体。

假设数据包含三个与姓名相关的列，即姓氏、名字和中间名。我们希望转换这三列，使它们符合我们的模式，该模式有四个与名称相关的列。这样做的 STATA 代码如下:

```
gen n_last = strtrim(stritrim(lower(LASTNAME)))gen n_first = strtrim(strtrim(lower(FIRSTNAME)))gen n_mid = strtrim(strtrim(lower(MIDDLENAME))) foreach var of varlist n_* { replace `var’ = subinstr(`var’, “-“, “ “, .) replace `var’ = strtrim(stritrim(subinstr(`var’, “.”, “”, .))) /* check for suffix */ gen `var’_suff = regexs(0) if regexm(`var’, “(1|11|111|[  ]i|ii|iii|[  ]iv|[  ]v|[  ]v[i]+|jr|sr|[0-9](nd|rd|th)*)$” replace `var’ = strtrim(stritrim(subinstr(`var’, `var’_suff, “”, 1)))}gen suffix = “”foreach var of varlist n_*_suff {replace suffix = `var’ if missing(suffix) & !missing(`var’)}drop n_*_suff n_suffixreplace name_raw = raw_namereplace name_first = n_firstreplace name_middle = n_midreplace name_last = n_lastReplace name_suffix = suffix
```

前三行很简单——它们只是清除每个 name 变量中的空格，并将它们复制到新生成的变量中进行处理。请注意 *strtrim* 和 *stritrim* 函数对于清理 STATA 中的字符串变量很有用。

接下来是 for 循环。它在我们刚刚创建的三个变量之间循环。

循环的前两行(对于每个变量…)删除了特殊字符“-”和“.”。通常建议在应用之前删除它们，因为它们会妨碍正则表达式的有效应用。

循环中的下一部分是利用正则表达式，这在其他编程语言中很常见。这里，我们试图获取混合到每个名称列中的任何后缀。如果后缀始终只混合在一列中就好了，但这种情况很少发生。这就是为什么我们必须遍历名字、中间名和姓氏列，以查看它们中是否存在任何后缀。欢迎来到真实世界数据的世界，这里的数据很少有一致性！正则表达式旨在获取典型的后缀，包括第二、第三、第四、jr、sr 等。如果作为第一个参数输入的变量满足第二个参数中的正则表达式模式，STATA 中的 *regexm* 函数返回 1，否则返回 0。

# DOB 解析

在 STATA 中解析出生日期(DOB)非常简单，因为有一些定制的函数。假设 DOB 列是年、月和日的字符串格式(按此顺序)，我们可以很容易地解析 DOB，如下所示:

```
replace dob_yyyy = year(date(DOB, “YMD”))replace dob_mm = month(date(DOB, “YMD”))replace dob_dd = day(date(DOB, “YMD”))
```

请注意，date 函数首先将日期字符串变量转换为 date 对象，year、month 和 day 函数允许您提取日期的相关部分。如果 DOB 变量是非字符串格式的，那么在继续上面的代码之前，您必须将格式改为字符串。另外，请注意 date 函数的第二个参数是日期的格式(例如 YMD，MDY…)，所以请确保您指定了正确的格式。然而，真实世界的数据通常在一列中混合了多种格式，在解析之前需要仔细检查。

# 解析地址

典型的地址格式如下:

```
2301 County Street, Ann Arbor, MI 40000 apt 220
```

它以某个街道地址开始，后面是城市、州、邮政编码和更细粒度的信息，如公寓号。如果数据包含这些不同的组件，那么将它们解析成我们之前定义的地址模式会更容易。但是假设数据在一列中带有完整的地址，那么用于解析地址信息的 STATA 代码如下:

```
gen raw_addr = ADDRESSreplace raw_addr = strtrim(stritrim(lower(raw_addr)))gen bldnum = strtrim(regexs(0) if regexm(raw_addr, “^([0–9]+) [ ]”)gen street = strtrim(strtrim(subinstr(raw_addr, bldnum, “”, 1)))replace ZIP = “” if ZIP == “0” # zipcodes that are just 0 are probably missing zipcodes that have been filled in with 0s by the agencyreplace addr_raw = raw_addrreplace addr_bldnum = bldnumReplace addr_str = streetReplace addr_str = substr(addr_str, 1,5)Replace addr_city = strtrim(stritrim(lower(CITY)))Replace addr_st = strtrim(stritrim(lower(STATE)))Replace addr_zip = ZIP
```

与名称解析类似，我们也利用正则表达式来解析地址。STATA 的正则表达式模式和语法几乎类似于任何其他编程语言中使用的正则表达式，因此请参考 Python [文档](https://docs.python.org/3/howto/regex.html)中的正则表达式以深入了解它。

希望这有所帮助！以后请考虑关注我更多的文章！

# 关于作者

*数据科学家。在密歇根大学刑事司法行政记录系统(CJARS)经济学实验室担任副研究员。Spotify 前数据科学实习生。Inc .(纽约市)。即将入学的信息学博士生。他喜欢运动，健身，烹饪美味的亚洲食物，看 kdramas 和制作/表演音乐，最重要的是崇拜耶稣基督。结账他的* [*网站*](http://seungjun-data-science.github.io) *！*