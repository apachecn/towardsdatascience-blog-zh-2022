# Git 部分提交

> 原文：<https://towardsdatascience.com/git-partial-commits-888b94d97f01>

## 如何只提交文件的一部分

![](img/cdb578245b65890bd18ad696c256caa4.png)

照片由 [Praveen Thirumurugan](https://unsplash.com/@praveentcom?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

Git 是所有数据科学家和软件工程师都应该知道如何使用的工具。无论您是独自从事一个项目，还是作为一个大型分布式团队的一部分，了解如何使用 Git 从长远来看都可以为您节省大量时间。git 中一个很棒的功能是能够进行部分提交，这意味着您可以从一个文件而不是整个文件提交单独的更改。当您对同一个文件进行增量更改，但这些更改彼此之间没有直接关系，并且在不同的文件中提交更改时，这就变得很容易了。如果没有这一点，您可能会提交应该单独提交的代码片段，如果代码中有 bug、错误或失误，就很难撤销！

## 进行部分提交

部分提交的主要目的是确保您只一起提交相关的更改。这可能是在代码中，也可能是补充文件的一部分，比如在同一个文件中处理多个组件，或者在 csv 文件中添加不同的行。那么我们该怎么做呢？

进行部分提交相对简单，只需要对单个文件的暂存进行少量修改。当您只想从单个文件中添加一小部分更改时，您可以简单地在代码中使用`--patch`或`-p`标志，如下所示:

```
git add --patch <filename>
```

或者

```
git add -p filename
```

运行这些命令之一将打开一个逐步提示，它将迭代指定文件中的“大块”代码。在这个一步一步的提示中，您将被询问是否要存放显示的代码块，有几个选项可供选择。这些将是:

```
y - stage this hunk
n - do not stage this hunk
q - quit; do not stage this hunk or any of the remaining ones
a - stage this hunk and all later hunks in the file
d - do not stage this hunk or any of the later hunks in this file
g - select a hunk to go to
/ - search for hunk matching the given regex
j - leave this hunk undecided, see next undecided hunk
k - leave this hunk undecided, see previous undecided hunk
s - split the current hunk into smaller hunks
e - manually edit the current hunk
? - print help
```

你对此的反应将决定接下来会发生什么。最重要的是，`y`将暂存当前的“大块”代码并转移到下一个(如果有的话)，`n`不会保存当前的“大块”并转移到下一个，`s`会将“大块”代码分割成更小的部分，如果你想进一步分解的话。该提示将继续处理文件中的所有块，直到没有剩余块，或者您指定已经使用`d`或`a`完成。

需要注意的一件事是，当你想用`s`或`e`打破一个大块头时，交互会变得相当复杂。虽然`s`会为你打破盘踞，`e`会让你选择你感兴趣的相关线路。对于后者，您可以将`+`或`-`替换为`#`，以表明您希望给定的行包含在较小的代码“块”中。

为了确保您已经暂存了相关的代码行，您可以使用以下命令:

*   `git status` 或`git diff -staged`检查您是否已经进行了正确的更改。
*   `git reset -p`取消登台错误添加的大块并重放登台。
*   `git commit -v`编辑提交消息时查看提交。

如果您不想在命令行中完成所有这些，我当然可以理解为什么，您也可以使用 git GUI 来存放文件中的特定行。在 GUI 中，您在相关文件中找到想要暂存的行，右键单击它，您应该会看到一条消息“暂存该行以提交”。这样做的结果将与命令行交互的方式相同，并确保只提交相关的代码行！

## 结论

虽然在 git 中暂存不同文件中的更改非常简单，但是当同一个文件中有多个更改时，这可能会非常复杂。为此，您可以利用 git 中的部分提交，它允许您迭代文件中的每一行，以选择是否要暂存这些更改。当添加新文件时，这可以通过使用`-p`或`--patch`标志简单地触发，这将打开一个迭代编辑器来选择相关的行。但是要小心，要确保你一起提交所有相关的变更，否则你就有引入错误或错误的风险，并且以后很难撤销那些变更！

如果你喜欢你所读的，并且还不是 medium 会员，请使用下面我的推荐链接注册 Medium，来支持我和这个平台上其他了不起的作家！提前感谢。

<https://philip-wilkinson.medium.com/membership>  

或者随意查看我在 Medium 上的其他文章:

</eight-data-structures-every-data-scientist-should-know-d178159df252>  </a-complete-data-science-curriculum-for-beginners-825a39915b54>  <https://python.plainenglish.io/a-practical-introduction-to-random-forest-classifiers-from-scikit-learn-536e305d8d87> 