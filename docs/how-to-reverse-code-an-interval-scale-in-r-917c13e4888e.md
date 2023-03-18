# 如何反向编码 R 中的音程音阶

> 原文：<https://towardsdatascience.com/how-to-reverse-code-an-interval-scale-in-r-917c13e4888e>

## 清除问卷和测量数据的简单方法

![](img/347b88dd2265746d5c8833a2ebe33f2e.png)

[来自 Pexels 的 Alex Green 拍摄的照片](https://www.pexels.com/photo/crop-ethnic-psychologist-writing-on-clipboard-during-session-5699456/)

如果您使用 R 进行数据分析，很可能会遇到处理区间数据的情况。通常，能够在一个区间尺度上反转数据是很有用的。下面是如何做的，有一步一步的例子。如果您希望本文中的所有代码都包含在一个脚本中，请查看最后的 GitHub 要点。

# 使用调查数据

假设你是一名心理学家，收集了关于人们焦虑体验的数据。你对人们进行了调查，询问他们现在有多焦虑，他们每天经历焦虑想法的频率，等等。你要分析这些数据，给每个参与者的焦虑评分。但是，您使用的调查问卷中有几个措辞相反的问题，而通常的回答量表是反过来的。当你把每个参与者的回答加起来时，你需要颠倒这些答案的编码来得到正确的分数。

幸运的是，在 r 中有一种简单的方法来进行这种分析。

假设我们调查中的每个项目都使用 7 分的李克特量表，如下所示:

“1 =强烈不同意；2 =不同意；3 =有些不同意；4 =既不同意也不反对；5 =有些同意；6 =同意；7 =非常同意”

我们可以通过从这个响应量表中随机取样值来创建一个数据集，如下所示。

```
library(tidyverse)

# Generate some simple data
survey_data <- tibble(
  participant = 1:7,
  worried_thoughts = sample(myscale, 7),
  anxiety_effects = sample(myscale, 7)
)
```

该数据集的第一列`worried_thoughts`，包含参与者对陈述“我经常有侵入性的担忧想法”的响应。第二列`anxiety_effects`，包含对陈述“焦虑很少影响我的日常生活”的响应。

由于第二种说法的措辞，高数字分数表明焦虑感低。这意味着对这一陈述的回答必须进行反向编码，以便与其他正常措辞的问题进行比较，其中高数字分数意味着高焦虑。

幸运的是，这有一个简单的公式。

# 如何反转区间标度上的值

要在区间标度上反转值，您需要取变量所在标度的最小值，减去要反转的值，然后加上标度的最大值。

这个公式适用于从任何数字(正数或负数)开始的反应等级。它也适用于具有任何大小间隔的标尺，只要该间隔是一致的。

以下是应用此公式反转数据集中数值区间值的几种方法的示例。

```
myscale <- 1:7

# The tidyverse way
survey_data <- survey_data %>%
  mutate(anxiety_effects_reversed = min(myscale) - anxiety_effects + max(myscale))

# The base R way
survey_data$anxiety_effects_reversed <- min(myscale) - survey_data$anxiety_effects + max(myscale)
```

在这段代码中，我们首先定义问题中使用的数值范围。这只是一个数字 1 到 7 的向量，存储在变量`myscale`中。

然后，我们可以在反向编码公式中使用`myscale`中的值。在 tidyverse 示例中，该公式被表示为`min(myscale) — response + max(myscale)`，其中`response`是要重新编码的列中的值。

这真的很简单，你可以用 Base-R 或 Tidyverse 来应用它，如图所示。在这两个例子中，反向编码操作的结果存储在新列`response_reversed`中。

# 反转几个区间标度变量

![](img/d65308dd834bfc9b5ef8f05dd33457a3.png)

照片由[威廉·沃比](https://unsplash.com/@wwarby?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

当您需要翻转数据集中几个问题的尺度时，该怎么办？你可以像以前一样用同样的公式做这件事，只需要一点点小技巧。

对于本例，让我们创建更多的示例数据，这次有多个问题。

```
# Generate more data
full_survey_data <- tibble(
  question = unlist(map(1:5, ~ rep(paste0("Question", .), 5))),
  participant = rep(1:5, 5),
  response = sample(myscale, 25, replace = T)
)
```

该数据是长格式的，这意味着在第一列中有多个重复的行。如果你有兴趣，你可以在这里阅读更多关于[长数据和宽数据的区别。如果您的问卷数据不是长格式，您可以使用 r 中的](https://www.statology.org/long-vs-wide-data/) [pivot_longer](https://tidyr.tidyverse.org/reference/pivot_longer.html) 函数进行转换。

要对数据集中的一些问题进行反向编码，同时保持其他问题不变，我们首先需要选择我们想要重新编码的问题。我们可以通过创建一个包含我们正在记录的问题名称的向量来做到这一点。在这种情况下，我们希望颠倒问题 2 和 3 的得分。

```
# Define vector of question names that you want to reverse
reverse_questions <- c("Question2", "Question3")
```

现在，我们可以使用这个向量来选择性地应用我们的重新编码操作。我们可以通过在 mutate 中使用`case_when`来做到这一点，这是我最喜欢的 tidyverse 函数之一。

```
full_survey_reversed <- full_survey_data %>%
  mutate(response = case_when(question %in% reverse_questions ~ min(myscale) - response + max(myscale),
                              TRUE ~ as.integer(response)))
```

一开始可能会有很多东西需要处理，所以让我们把它分解一下。

这段代码使用了`mutate`，这是一个操作现有列的值或创建新列的函数。我在上一个例子中也使用了它，但是在这个例子中，我已经设置了它来将我们操作的结果分配给已经在数据中的`response`列。这意味着它将改变该列中的一些现有值。

然后,`case_when`函数反转 response 列中的值，但是只针对我们想要重新编码的问题。它的工作原理是根据一个条件检查每一行数据。这里，它检查问题列的值是否在我们前面定义的问题名称向量中。

如果是，它会将比例反转公式应用于“响应”列中的值。您可以在`case_when`行中的波浪号(~)后看到这一点。如果给定的行不包含我们选择重新编码的问题，`case_when`将返回 response 列的原始值。它通过`TRUE ~ as.integer(response)`命令来实现这一点。

这段代码的结果是，对于我们选择的问题，`response`中的值被反转，对于我们没有选择的问题，值保持不变。

给问卷打分现在很简单。同样，我们使用 tidyverse 函数，这一次是对响应进行求和，以获得每个参与者的分数。

```
# Sum up each participant's survey score
full_survey_reversed %>%
  group_by(participant) %>%
  summarise(total_score = sum(response))
```

恭喜你！您现在可以轻松地反转 R 音程音阶。对于本文中使用的所有代码，请参见下面的要点。

数据清理快乐！

想阅读我所有关于 R 编程、数据科学等方面的文章吗？在[这个链接](https://medium.com/@roryspanton/membership)注册一个媒体会员，就可以完全访问我所有的作品和媒体上的所有其他故事。这也直接帮助了我，因为我从你的会员费中得到一小部分，而不需要你额外付费。

只要我在这里订阅[，你就可以把我所有的新文章直接发到你的收件箱里。感谢阅读！](https://roryspanton.medium.com/subscribe)