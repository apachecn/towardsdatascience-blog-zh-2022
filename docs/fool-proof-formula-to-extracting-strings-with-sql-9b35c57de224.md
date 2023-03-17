# 用 SQL 提取字符串的简单公式

> 原文：<https://towardsdatascience.com/fool-proof-formula-to-extracting-strings-with-sql-9b35c57de224>

# 用 SQL 提取字符串的简单公式

## 解决讨厌的字符串问题的分步示例

![](img/0fc83c64af6e983a456bb462ed6146f8.png)

在 [Unsplash](https://unsplash.com/s/photos/string?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上由[Bozhin karivanov](https://unsplash.com/@bkaraivanov?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)拍摄的照片

你有没有遇到过一个字符串提取问题，几个小时后却发现自己快疯了？人们很容易迷失在所有潜在的字符串函数中。

字符串是我最不喜欢使用 SQL 的数据类型。对于模式匹配，您需要了解许多不同的函数和字符。但是，当你正确使用它们时，我在本文中向你展示的方法是相当简单的。

我写这篇文章的目的是让字符串不那么烦人。从我的错误和许多小时的挫折中吸取教训！我将向您介绍我正在研究的问题、我的解决方案中的关键功能以及解决方案本身。到本文结束时，您将成为使用 SQL 提取字符串的专家。

# 问题是

我最近受命为营销团队重新创建一个数据模型。很大一部分逻辑是从活动链接中提取 utm 信息。当您在 Google 上做广告时，会使用 utm 信息创建特殊链接，如媒体、出版商、活动和来源。这些都组合在一起形成一个链接，看起来像这样:

```
[https://www.etsy.com/market/lamp_shades?utm_source=google&utm_medium=cpc&utm_term=etsy%20lamp%20shades_p&utm_campaign=Search_US](https://www.etsy.com/market/lamp_shades?utm_source=google&utm_medium=cpc&utm_term=etsy%20lamp%20shades_p&utm_campaign=Search_US_Brand_GGL_ENG_Home_General_All&utm_ag=Lamp+Shades&utm_custom1=_k_Cj0KCQiAt8WOBhDbARIsANQLp94WN__lDzNNnwS6yptN8pqbeU09mUzcKN9-5hHMFTWbS4msnQJqh4YaAtaOEALw_wcB_k_&utm_content=go_6518959416_125883546672_536666915699_aud-459688891835:kwd-308153672827_c_&utm_custom2=6518959416&gclid=Cj0KCQiAt8WOBhDbARIsANQLp94WN__lDzNNnwS6yptN8pqbeU09mUzcKN9-5hHMFTWbS4msnQJqh4YaAtaOEALw_wcB)
```

关键是识别模式。不只是一种模式，而是所有可能的模式。你可能认为你已经找到了一个有效的模式。但是，随后会出现一个特殊的用例，您也需要将它构建到您的逻辑中。找出您正在处理的字符串列中的所有模式。相信我，越早找到他们越好。将这些模式写在列表中，如下所示:

*   utm_source 位于“utm_source”之后和“&”之前
*   utm_medium 位于“utm_medium”之后和“&”之前
*   utm_campaign 位于“utm_campaign=”之后

一旦你找到了字符串中的模式，你就可以把它们翻译成代码。开始用文字而不是代码编写逻辑要容易得多。有了字符串提取，你的代码会变得非常混乱，非常快。几乎到了无法阅读的程度…

**提示:**边走边注释你的代码。这不仅有助于你在编写代码时更好地理解它，也有助于阅读你的代码的人。

# SQL 字符串函数

## CHARINDEX()

在处理字符串时，我喜欢使用两种不同的函数。第一个是 **CHARINDEX()** 函数。这会返回您提供给它的字符串的索引。它有两个输入，您正在搜索的字符串和您在中搜索的列*。因此，如果我在`campaign_link`列中寻找‘UTM _ medium ’,我会将其写成:*

```
charindex('utm_medium', campaign_link)
```

这段代码将返回“utm_medium”在该列值中所处位置的索引。如果它不在列值中，函数将返回 0。

## 子字符串()

我要使用的下一个函数是 **SUBSTRING()** 函数。这需要您想要定位字符串的列和两个索引——一个用于字符串的开头，一个用于结尾。它返回所提供的列中这两个索引之间的字符串。

如果我知道字符串“utm_medium”在索引 5 和 10 之间，我会编写如下所示的代码:

```
substring(campaign_link, 5, 10)
```

这将返回`campaign_link`列中索引 5 和 10 之间的字符串。

# 其他有用的 SQL 命令

## 选择语句

**CASE** 语句总是很方便，但在处理字符串时更是如此。它帮助我们处理我们在数据中发现的所有奇怪的一次性场景。 **CASE** 语句遵循一个简单的“*当这种情况发生时，则执行这个*模式。

```
CASE 
    WHEN utm_medium='Facebook' THEN 'Social'
    ELSE 
END AS marketing_category
```

您还可以在 **CASE** 语句中添加一个`ELSE`来处理其他可能破坏代码的场景。并且，用`END`结束你的案例陈述！

## LIKE 运算符

**LIKE** 操作符对于在字符串中搜索模式特别有用。虽然它不能像`CHARINDEX()`和`SUBSTRING()`那样帮助提取那些模式，但它在 **CASE** 语句中特别有用。它们通常由我写的案例陈述中的“ *when* 部分组成。

当像使用一样使用**时，你指定一个你正在搜索的字符串。根据您希望它在要搜索的列中的位置，您需要使用特殊的运算符。**

*   _ 代表单个字符
*   %代表一个、一个或多个字符

我通常在要查找的字符串的开头和结尾使用%。只要字符串出现在列值中的任何位置，这将被证明是正确的。

```
CASE 
    WHEN campaign_link LIKE '%facebook%' THEN 'Social'
    ELSE 
END AS marketing_category
```

请注意，我用单引号括起了要搜索的字符串，并用%号括起来。

# 解决方案

现在，让我们进入解决问题的解决方案，并利用我刚才提到的这些功能。请记住，我是在反复试验之后才得出这个结论的。重要的是，如果你发现有什么不对劲的地方，就要不断地检查你的代码，并反复强调。

当我写字符串提取代码的时候，我是分几部分来做的。首先，我使用了`CHARINDEX()`函数来查找我正在寻找的字符串部分的索引。这将返回 utm_medium 中“u”的索引，所以我们要将这个字符串的长度加到 index 函数中。这将返回=之后的字符串的索引。

```
charindex('utm_medium=', campaign_link) + 11
```

然后，我将它与`SUBSTRING()`函数结合使用，以获取字符串索引右侧的所有内容(我通过在函数 100 中创建第二个索引来实现这一点)。

```
substring(campaign_link, charindex('utm_medium=', campaign_link) + 11, 100)
```

之后，我添加了另一个`CHARINDEX()`函数来查找下一个字符左边的字符串。为此，从索引中减去 1，因为它是字符串的结尾，只有 1 个字符长。

```
substring(campaign_link, charindex('utm_medium', campaign_link) + 11, charindex('&', campaign_link) - 1)
```

最后，我在这个逻辑中添加了一个 CASE 函数。但是，这只是在我意识到并非每个活动链接都有 utm_source、utm_medium、utm_publisher 等时才添加的。我需要一个案例来处理这些字符串后面没有“&”的情况，就像我逻辑中的最后一个要点。

```
substring(campaign_link,           # starting with the index of utm_medium in the link
          charindex('utm_medium', campaign_link) + 11,           # if this string contains an & then return that index
          # if not, return 100 as the ending index
          CASE 
              WHEN substring(campaign_link, charindex('utm_medium', campaign_link) LIKE '%&%' THEN charindex('&', campaign_link) - 1
              ELSE 100 
          END
)
```

看到事情变得多混乱了吗？在逻辑中添加 CASE 语句会使代码变得难以阅读，这就是为什么注释代码很重要的原因。阅读它的人需要很长时间才能意识到它在做什么。

让事情变得更加复杂的是，我们需要将它添加到另一个 case 语句中，如果 utm_medium 出现在`campaign_link`中，这个语句将只提取它。这看起来像这样:

```
CASE
# check if the link contains a utm_medium
    WHEN campaign_link LIKE '%utm_medium%' THEN 
# return the string after utm_medium=
substring(campaign_link, charindex('utm_medium=', campaign_link) + 11,
# if this extract link contains an &, find the index of that
CASE 
    WHEN substring(campaign_link, charindex('utm_medium=', campaign_link) LIKE '%&%' THEN charindex('&', campaign_link) - 1
# if it doesn't, return 100 as the ending index
    ELSE 100 
END
)
# if it doesn't contain utm_medium, return null 
    ELSE NULL
END AS utm_medium
```

# 结论

我发现字符串提取的关键是要有耐心。不要试图一口气写完整段代码。编写各个部分，验证它们是否如您所期望的那样工作，然后将它们集成在一起。虽然这看起来像是更多的工作，但从长远来看，它最终会花费更少的时间。如果你一头扎进去，肯定会出错，你会花更多的时间来调试你的代码。

如果您的工具箱中有这些函数和操作符，您将很快用 SQL 提取字符串。记住，注释你的代码！下次再来看看“ [3 个 SQL 交换以写出更好的代码](/3-sql-swaps-to-write-better-code-f8d304699cde)”和“[如何使用这 6 个不寻常的 SQL 函数](/how-to-use-these-6-unusual-sql-functions-b912e454afb0)”。