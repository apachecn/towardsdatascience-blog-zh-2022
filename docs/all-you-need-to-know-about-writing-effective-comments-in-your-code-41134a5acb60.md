# 关于在代码中编写有效的注释，您需要知道的是

> 原文：<https://towardsdatascience.com/all-you-need-to-know-about-writing-effective-comments-in-your-code-41134a5acb60>

## 这和写源代码一样重要

![](img/fc35acec8c652681865f58363f5551da.png)

托马斯·博曼斯在 Unsplash[上的照片](https://unsplash.com?utm_source=medium&utm_medium=referral)

注释是程序员可读的简短解释或注释，直接写在计算机程序的源代码中。尽管计算机在执行程序时会忽略它们，但在源代码中编写有效的注释可能与实际代码本身一样重要，因为软件总是不完整的。

总有一些事情可以做来改进软件产品或服务，这意味着代码库必须不时更新。对你不理解的代码进行修改或添加新功能几乎是不可能的，因此重要的是代码总是被构造成能被人类阅读。

> "程序必须是为人们阅读而写的，并且只是附带地为机器执行而写的."
> 
> ——[**哈罗德·艾贝尔森**](https://en.wikipedia.org/wiki/Hal_Abelson)

注释被用来通知、警告和提醒那些没有编写代码的人[和你未来的自己]代码正在做的重要事情。在这篇文章中，我们将关注用 Python 写注释。

```
**Table of Contents** 
- [**5 Unwritten Rules about comments in Python**](#bfad)
    - [Rule #1 Comments are complete sentences](#d842)
    - [Rule #2 Comments should obey line limits](#c779)
    - [Rule #3 Comments must be on the same indentation level as the code it comments](#792e)
    - [Rule #4 Put a space after the #](#4d4c)
    - [Rule #5 Links don’t replace explanations](#db99)
- [**The different types of comments**](#8f41)
    - [Inline comments](#389e)
    - [Explanation comments](#bf3a)
    - [Summarization comments](#8385) 
    - [Legal comments](#9085)
    - [Code tag comments](#3a69) 
```

# 关于 Python 中注释的 5 条不成文的规则

我们可以用 Python 写单行或多行注释。使用`#`符号定义单行注释，并在行尾结束。Python 不一定有专门的多行注释语法，所以 Pythonistas 可以决定是使用多个单行注释(完全称为块注释)，还是使用三重引号多行字符串。

```
# This is a single line comment. # This is an example 
# of multiple single-line
# comments that make up a 
# multi-line comment."""This is an example
of triple quotes multi-line
strings that make up a
multi-line comment."""
```

这三种情况都有效，但是您会注意到三重引号多行字符串比使用多个单行注释来定义多行注释可读性更好。因此，如果需要的话，最好使用三重引号多行字符串来定义多行注释——记住，“程序是为人类阅读而编写的。”

你可能会遇到这样的人，他们认为评论是事后的想法，或者告诉你“评论不重要”，忽略它们。去阅读任何流行的库的源代码，看看它们的重要性。如果你希望你的代码 1)更易读和 2)更专业，注释当然是必需的。

然而，有 5 条黄金法则可以确保你写的代码是有效的。

## 规则 1 注释是完整的句子

注释代码的全部目的是让其他程序员(以及您未来的自己)阅读它，以便更好地理解它所注释的代码中发生了什么。因此，评论应该遵循正确的语法规则，包括标点符号，以确保它向读者提供清晰的信息。

```
# data from train dir               **<-- Not good** 
data = load_data()# Load data from train directory.        **<-- Good**
data = load_data() 
```

## 规则 2 注释应该遵守行的限制

Python 的风格指南 PEP 8 是作为 Python 编程的一套最佳实践而创建的。其中一个指导方针建议行数应该限制在 79 个字符以内:这个指导方针同样适用于源代码和注释。

```
# This comment is so long that it doesn't even fit on one line in the code cells provided by medium.   **<-- Not good** # This comment is much shorter and within the limits.   **<-- Good**
```

程序员总是打破 PEP 8 建议的行限制，这没什么——这毕竟只是一个指南。但是你的评论仍然应该遵守你和你的团队(或者你自己，如果你是一个人的话)达成的任何行限制。

## 规则 3 注释必须和它所注释的代码在同一缩进层次上

缩进是指某些代码开头的空格数；Python 使用四个空格作为默认缩进来对代码进行分组。在不同的缩进层次上写注释不会导致你的程序崩溃，但是当它在同一层次上的时候就容易理解多了。

```
**def** example_function(): 
# Perform a random calculation.        **<-- Not good**
    random = 24 + 4
    **return** random**def** example_function(): 
    # Perform a random calculation.       ** <-- Good**
    random = 24 + 4
    **return** random
```

## 规则#4 在#后面加一个空格

在`#`后面加一个空格有助于可读性。

```
#This is valid but not easy to read.        **<-- Not good**
# This is valid and easy to read.           **<-- Good**
```

## 规则 5 链接不能代替解释

有时我们可能需要链接到外部页面来进一步解释我们的代码在做什么。仅仅留下一个到页面的链接不如在链接到外部资源之前写下为什么要实现代码有效。这样做的原因是，页面可以被删除，然后你会留下一个无法解释的链接，导航到任何地方。

**注** : *下面的示例代码摘自*[*Scikit-learn code base*](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_search_successive_halving.py)*。*

```
**# Not good**
**def** _check_input_parameters(**self**, X, y, groups): 

    --- snip ---
    # see [https://github.com/scikit-learn/scikit-learn/issues/15149](https://github.com/scikit-learn/scikit-learn/issues/15149)
 **if not** _yields_constant_splits(self._checked_cv_orig): 
        **raise ValueError**(
            "The cv parameter must yield consistent folds across "
            "calls to split(). Set its random_state to an int, or "
            " set shuffle=False."
        )
    --- snip ---**# Good
def** _check_input_parameters(**self**, X, y, groups): 

    --- snip ---     # We need to enforce that successive calls to cv.split() yield 
    # the same splits: 
    # see [https://github.com/scikit-learn/scikit-learn/issues/15149](https://github.com/scikit-learn/scikit-learn/issues/15149) **if not** _yields_constant_splits(self._checked_cv_orig): 
        **raise ValueError**(
            "The cv parameter must yield consistent folds across "
            "calls to split(). Set its random_state to an int, or "
            " set shuffle=False."
        )
    --- snip --- 
```

**对于不热衷于遵循规则的开发人员，你听到的第一个理由是“你想太多了。”在某种程度上，他们有一个观点:如果你决定违背上面提到的每一条规则，你的程序不会崩溃。**

**然而，编程的主要部分是协作。尽管你努力成为一名伟大的程序员，创造出非凡的东西，但你也应该记住，你的大部分工作将作为团队的一部分来完成(在大多数情况下)。因此，考虑如何让别人更容易理解你的代码是有帮助的，其中一个方面就是你如何写你的注释。**

**你的评论可读性越强，其他开发者(和你未来的自己)就越有可能阅读它们:你的评论只有在被阅读时才有益处。确保你的评论被阅读的一部分是知道如何有效地放置它们。**

# **不同类型的评论**

**注释是源代码文档的一种额外形式。因此，它们的一般目的是通知其他人[包括您未来的自己]为什么代码中的某些功能是以某种方式实现的，但是有几种不同的方式来部署注释以实现这一目标。**

**让我们来看看其中的一些方法:**

## **内嵌注释**

**行内注释出现在一行代码的末尾。使用行内注释有两个主要原因:**

****#1** 如果一个变量已经被定义了，但是使用一个特定对象的原因还不清楚，你可以使用一个行内注释来证明你的决定。**

```
AGE = 18 # The legal drinking age in the UK. 
```

**内联注释的另一个很好的用途是通过为定义的内容提供更多的上下文来减少歧义。**

```
day = 3 # Days in a week range from 0 (Mon) to 6 (Sun)
height = 1.75 # Height is given in meters 
```

**您可能还会看到一些代码库使用注释来指定变量的数据类型。**

```
"""Code taken from my fraud detection model project.
See: [https://github.com/kurtispykes/fraud-detection-project/blob/main/IEEE-CIS%20Fraud%20Detection/packages/fraud_detection_model/fraud_detection_model/train_pipeline.py](https://github.com/kurtispykes/fraud-detection-project/blob/main/IEEE-CIS%20Fraud%20Detection/packages/fraud_detection_model/fraud_detection_model/train_pipeline.py)"""**from** config.core **import** config  # type: ignore
**from** pipeline **import** fraud_detection_pipe  # type: ignore
**from** processing.data_manager **import** load_datasets, save_pipeline  # type: ignore
**from** sklearn.model_selection **import** train_test_split  # type: ignore
```

**我在我的项目中这样做的唯一原因是我使用了一个名为 [typechecks](https://pypi.org/project/typechecks/) 的自动化工具来验证数据类型——我假设这也是它可能出现在其他代码库中的原因。**

**通常情况下，您不需要指定数据类型，因为从赋值语句中可以明显看出这一点。**

## **解释注释**

**注释的主要目的是解释为什么一个特定的部分以某种方式被实现。如你所知，做事情有几种方法，所以一个解释能让你更深入地了解程序员的意图。**

**例如，看看这个评论:**

```
number_of_words *= 0.2 # Multiply the number of words by 0.2\. 
```

**上面的场景是一个无意义评论的完美例子。不需要火箭科学家就能算出你在用`0.2`乘以`number_of_words`，所以没有必要在注释中再次声明。**

**这是一个改进的版本:**

```
number_of_words *= 0.2 # Account for each word valued at $0.20\. 
```

**这个评论提供了对程序员意图的更多洞察:我们现在知道`0.2`是每个单词的价格。**

****注** : *如果你已经阅读了* [***7 种你应该知道并避免的代码气味***](/7-code-smells-you-should-know-about-and-avoid-b1edf066c3a5) *，你就会知道我们本可以通过给它分配一个常数(即* `*PRICE_PER_WORD = 0.2*` *)来省略这个幻数，从而更好地改进这个代码。***

## **总结评论**

**有时我们别无选择，只能使用几行代码来实现某些功能。给你的同事(和你未来的自己)一个你的代码正在做什么的概要是非常有益的，因为它允许他们快速浏览你的代码。**

```
"""This is a small piece of functionality extracted from 
the Scikit-learn codebase. See: [https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/impute/_knn.py#L262](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/impute/_knn.py#L262)"""# Removes columns where the training data is all nan
**if not** np.any(mask):
    **return** X[:, valid_mask]row_missing_idx = np.flatnonzero(mask.any(axis=1))non_missing_fix_X = np.logical_not(mask_fit_X)# Maps from indices from X to indices in dist matrix
dist_idx_map = np.zeros(X.shape[0], dtype=**int**)        dist_idx_map[row_missing_idx] = np.arange(row_missing_idx.shape[0])
```

**总结注释只是对代码中发生的事情的一个高级概述。将它们放在代码库中的不同位置，可以让队友(和你未来的自己)非常容易地快速浏览代码，以便更好地理解——这也表明你知道代码是如何工作的。**

## **法律意见**

**根据您的工作地点，您可能需要在脚本的顶部包含版权、软件许可和作者信息——这更像是一项内部政策，而不是所有程序员的必需品。**

**这里有一个来自 Scikit-learn 的`[_knn.py](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/impute/_knn.py#L262)`源文件的例子:**

```
# Authors: Ashim Bhattarai <ashimb9@gmail.com>
#          Thomas J Fan <thomasjpfan@gmail.com>
# License: BSD 3 clause
```

## **代码标签注释**

**散布在各种源文件中的简短提醒注释并不少见。开发人员这样做是为了提醒自己，他们还没有抽出时间去做，但打算在未来做的事情。**

**最常见的标记是 TODO 标记。**

```
"""*This is an example of a code tag comment in the
Scikit-learn codebase. The full code can be found here:* [*https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/pairwise.py*](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/pairwise.py)"""**def** pairwise_distances_argmin_min(
X, Y, *, axis=1, metric="euclidean", metric_kwargs=None
):
    --- snip ---
    **else**:
        # TODO: once PairwiseDistancesArgKmin supports sparse input 
        # matrices and 32 bit, we won't need to fallback to 
        # pairwise_distances_chunked anymore. Turn off check for 
        # finiteness because this is costly and because array
        # shave already been validated.
    --- snip ---
```

**注意上面的例子是如何清楚地描述了要做什么——如果你使用代码标签，这是必不可少的。**

**这些评论不应该用来代替一些正式的跟踪工具或错误报告工具，因为如果你没有阅读它们所在的部分，很容易忘记它们的存在。**

**需要注意的是，Python 并没有严格执行所有这些约定。团队可能有他们自己关于注释代码的约定，在这些情况下，最好遵循提供的结构。但是要记住，你的代码是要被阅读的，所以尽可能地让它可读会让这个过程变得更容易，不管这个人是谁。**

**我错过了什么类型的评论吗？留下评论。**

***感谢阅读。***

****联系我:**
[LinkedIn](https://www.linkedin.com/in/kurtispykes/)
[Twitter](https://twitter.com/KurtisPykes)
[insta gram](https://www.instagram.com/kurtispykes/)**

**如果你喜欢阅读这样的故事，并希望支持我的写作，可以考虑成为一名灵媒。每月支付 5 美元，你就可以无限制地阅读媒体上的故事。如果你使用[我的注册链接](https://kurtispykes.medium.com/membership)，我会收到一小笔佣金。**

**已经是会员了？[订阅](https://kurtispykes.medium.com/subscribe)在我发布时得到通知。**

**<https://kurtispykes.medium.com/subscribe> **