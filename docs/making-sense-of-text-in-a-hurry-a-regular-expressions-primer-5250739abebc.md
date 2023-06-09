# 快速理解文本:正则表达式入门

> 原文：<https://towardsdatascience.com/making-sense-of-text-in-a-hurry-a-regular-expressions-primer-5250739abebc>

![](img/fb19ba947014ec22cb9873524b61fc8a.png)

照片由丁诺(Pexels.com)拍摄

# 介绍

无论您是第一次接触正则表达式，并且有想要理解的文本数据，还是您有研究堆栈溢出问题的经验，希望找到完全相同的用例，而不太理解您正在使用的正则表达式的混乱；当你使用它们来扩展你的经验时，这个介绍将证明是一个有用的基础。

今天我们将讨论三个关键功能和三种关键模式。

# 正则表达式的简要说明

正则表达式是匹配文本中模式的有用工具。

模式识别对于给定文本串的分类特别有用。例如，假设您有客户的网站数据，但是您想检查给定短语的出现情况；regex 来救援了！

# 关键功能

我们将使用`re`包，并运行以下三个函数:

*   重新搜索()
*   重新拆分()
*   re.findall()

够简单！

# 搜索

## 定义

搜索允许我们在字符串中搜索给定的模式。如果模式出现在字符串中，函数将返回“匹配对象”,否则返回空值。

## 例子

假设您想要搜索一个名为 company description 的字符串，以查找包含“tech”的内容。出于示例的目的，让我们假设您将在公司描述中使用单词“tech”来对每个记录进行相应的分类。

调用函数，传递模式和字符串。

```
company_description = "selling tech products to retailers"
re.search('tech', company_description)
```

就这么简单，您有一个直接的方法来检测模式的存在。匹配对象本身将包括模式第一次出现的字符索引。

# 裂开

## 定义

Split 允许我们将一个字符串“分割”成一个列表中不同的元素。在 split 中，我们使用正则表达式来指定函数拆分字符串的模式。以下是更多相关信息。

## 例子

假设您希望根据潜在客户当前拥有的技术产品向他们进行营销，并且您甚至拥有这些数据，但不幸的是，每个客户的数据是一个长的连续字符串，用逗号分隔每项技术。

一个简单的解决方案是用逗号(和空格)拆分字符串。

```
technologies = 'salesforce, gainsight, marketo, intercom'
re.split(', ', technologies)
```

现在，您已经将每项技术分解到列表中它自己的条目中。

# 芬达尔

## 定义

Findall 与 search & match 非常相似。关键区别在于“findall”中的“all”。findall 不只是返回第一次出现的位置，而是返回模式的每次出现。为了便于说明，我们用直接引用的模式来保持事情的简单，但是很快我们将回顾您也可以使用的不同模式。

## 例子

假设你正在向电子商务公司销售退货处理产品，你浏览了一些潜在客户的网站，希望看看他们是否提供免费评论；在这种情况下，假设“免费退货”的提及量越大，表明注册该产品的倾向越高。

```
website_text = 'free returns... yada yada yada... free returns... and guess what... free returns'

returns = re.findall('free returns', website_text)
```

# 关键模式

现在您的工具箱中有了一些关键函数，让我们通过讨论模式来扩展它们的有用性。

在上面的每个例子中，我们明确地定义了我们的模式；我们现在要做的是回顾如何在更复杂的条件下更快地达到目标。

我们将回顾以下内容:

*   数字
*   话
*   间隔

# 数字

与前面的例子类似，我们将使用 findall 但在这种情况下，我们将这样做，以找到一个数字的每一次出现。假设我们第一季度的月销售额记录在一个字符串中，我们希望提取这些数字。密切注意我们通过的模式。

```
string = 'Jan: 52000, Feb: 7000, Mar: 9100'
print(re.findall(r"\d+", string))
```

让我们将这个命令分成不同的部分:

*   向 python 表明我们将使用正则表达式，这有助于 python 不会对你要做的事情感到困惑。
*   我们使用反斜杠`(\)`告诉 python 按字面意思处理下一个字符。有些情况下，一个“特殊的”字符告诉 python 做一些事情；在这种情况下，python 知道不要做任何时髦的事情。
*   `d`是我们用来表示我们想要的数字。
*   不带+号运行同样的操作会将每个单独的数字视为列表中自己的项目。+表示我们希望将符合我们指定标准的完整单词作为一个单独的项目。

# 话

让我们再做一次，只是把数字换成单词。我们会说我们想要提取月份值。

```
print(re.findall(r"[A-z]\w+", string))
```

我们看到这里包含了很多相同的东西:`r`、反斜杠、+；但是我们现在看到，我们已经包括了`w`，而不是 d。`w`是任何字符的指示，从技术上讲，它也可以扩展到其他字符，所以我们指定它是谨慎的。

在`\w+`语句之前，我们可以修改我们想要允许的特定字符类型的模式。在这种情况下，我们通过`[A-z]`指定包含所有大写和小写字母。

# 间隔

让我们重温一下之前为`re.split`制作的例子。

假设我们不想对逗号进行拆分，而是希望根据空格进行拆分。

```
print(re.split(r"\s", technologies))
```

如您所见，考虑到逗号现在是如何被包含在各个项目中的，这真的没有多大用处。如果没有逗号，这将是一个更有用的方法。

# 结论

你有它！几乎在任何时候，我们已经涵盖了相当多的内容。

你学到了:

3 个关键的正则表达式函数:

*   重新搜索()
*   重新拆分()
*   re.findall()

3 种便捷模式:

*   数字
*   话
*   间隔

和一些应该可以帮助你理解 regex 世界的规则。

我希望这可以证明是有用的脚手架，您可以使用它来构建您的正则表达式知识和经验。

我希望你喜欢这篇文章，并希望它对你的工作有所帮助！请分享什么有用，什么没用！

请随意查看我在 datasciencelessons.com 的其他帖子

祝数据科学快乐！