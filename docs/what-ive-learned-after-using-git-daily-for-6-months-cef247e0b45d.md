# 在每天使用 Git 个月后，我学到了什么

> 原文：<https://towardsdatascience.com/what-ive-learned-after-using-git-daily-for-6-months-cef247e0b45d>

## 我想我明白了

![](img/e47dbce8ba2ee5a984c28d8ad14cd9b6.png)

克里斯·林内特在 [Unsplash](https://unsplash.com/s/photos/merge-cars?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

这篇文章是写给那些刚开始学习 Git，立志学习 Git，或者已经使用过 Git 但没有积极使用的人。我曾经属于“用过 Git 但不要积极使用它”的群体。

学习 Git 可能会让人望而生畏。希望我在过去 6 个月里学到的 Git 命令列表会让这个任务看起来更简单。您可以将这些看作是 Git 命令的一种最简单的列表，您可以学习这些命令，从而确信您已经掌握了足够的知识，可以为项目做出贡献。

为了查找我分享的 git 命令的用法，这里有一个我过去用过的 Git 参考指南:【https://git-scm.com/docs

这里也有一个来自 Github 的有用的备忘单，我在过去也提到过:[https://education.github.com/git-cheat-sheet-education.pdf](https://education.github.com/git-cheat-sheet-education.pdf)

## 跳槽前认识的饭桶

我的职业生涯始于一名数据分析师。在我的分析师角色中，我只是偶尔使用 Git 所以可能一个月只有几次。我所在的公司并没有使用 Git 来进行版本控制。相反，我们会有版本文件夹，在那里我们保存新版本的代码，每个月/sprint 我会为更新的代码创建一个新文件夹(听起来熟悉吗？).我个人认为，如果所有数据专业人员正在编写代码，他们都应该使用 Git。公平地说，在我离开之前，这家公司试图将 Git 作为最佳实践来实施，但当一家公司在没有 Git 版本控制的情况下发展其分析部门时，让每个人都改变并开始使用它似乎几乎是不可能的。

3 年后，我转行做了数据科学家。以下是我在从事数据分析师工作时知道的 Git 命令(按照我学习的顺序排列):

```
[git config](https://git-scm.com/docs/git-config)
```

*   当您开始使用 git 时，系统会提示您设置用户名和密码，您需要使用 git config 来设置它们

```
[git clone](https://git-scm.com/docs/git-clone) <link to remote repository>
```

*   将远程存储库的文件复制到本地存储库

```
[git add](https://git-scm.com/docs/git-add) <file name or just use . to add all files>
```

*   暂存更改，为提交做准备

```
[git commit](https://git-scm.com/docs/git-commit) -m <commit message here>
```

*   提交或“保存”您的所有更改，并准备将它们推送到远程存储库

```
[git push](https://git-scm.com/docs/git-push)
```

*   将您的更改推送或“保存”到远程存储库

```
[git pull](https://git-scm.com/docs/git-pull)
```

*   用远程分支上所做的更改更新您的本地存储库

```
[git pull origin](https://git-scm.com/docs/git-pull) <branch name (typically master)>
```

*   用远程分支的更改更新当前分支

基本上，我知道最基本的 Git 命令——足够开始使用了。我从未使用过`git init`,因为我通常会创建一个新的主分支或者使用`git clone`😅

我还在 VS 代码中使用 Git，这样我就不用一直使用上面提到的一些终端命令了。然而，使用 VS 代码“源代码控制”侧边栏扩展的缺点是缺乏灵活性。最终使用终端变得更加容易。当我在多个 Github 库之间切换时尤其如此。

## 我在过去 6 个月中学习的 Git 命令

好了，这里是我现在知道的 git 命令(除了上面的列表)。如果你知道这些，你应该知道足够有意义的贡献。我试着按照我学到的命令的顺序列出这些命令。

```
[git status](https://git-scm.com/docs/git-status)
```

*   这将显示所有已更改并准备登台的文件(`git add`)

```
[git clone](https://git-scm.com/docs/git-clone) --branch <branch name> <repo link>
```

*   使用它来克隆远程存储库中的特定分支

```
[git branch](https://git-scm.com/docs/git-branch)
```

*   显示本地存储库中的分支列表

```
[git branch -d](https://git-scm.com/docs/git-branch) <branch name>
```

*   从本地存储库中删除一个分支

```
[git checkout](https://git-scm.com/docs/git-checkout) <branch name>
```

*   从远程存储库中检出(切换到)一个分支

```
[git checkout -b](https://git-scm.com/docs/git-checkout) <new branch name>
```

*   创建一个新分支，它是您当前所在分支的克隆

```
[git push --set-upstream origin](https://git-scm.com/docs/git-push) <branch name>
```

*   每当我在本地创建一个不在远程上的分支时，我都使用这个命令在远程上创建分支。通常在使用`git checkout -b`后会用到这个。

```
[git reset](https://git-scm.com/docs/git-reset)
```

*   删除所有暂存的更改

```
[git reset --hard](https://git-scm.com/docs/git-reset)
```

*   将文件重置为上次提交时的状态(使用该选项时要非常小心，因为您正在删除 Git 历史记录)

```
[git restore](https://git-scm.com/docs/git-restore) <file name>
```

*   将您的文件恢复到最新提交的状态

```
[git diff](https://git-scm.com/docs/git-diff)
```

*   检查您的更改与上次提交之间的差异

```
[git log](https://git-scm.com/docs/git-log)
```

*   查看您在该分支上的提交历史

```
[git stash](https://git-scm.com/docs/git-stash)
```

*   将所有更改放在一个工作目录中，以避免在切换分支之前必须暂存和提交您的更改

```
[git stash list](https://git-scm.com/docs/git-stash)
```

*   显示所有隐藏更改的列表

```
[git stash pop](https://git-scm.com/docs/git-stash)
```

*   将您的最后一次储存应用到您当前的分支，并将其从储存列表中移除

你可能会惊讶于`git merge`或`git rebase`不在名单上。从技术上来说，我用过一两次，但是我没有把它们包括在内，因为我还没有信心使用它们。我通常会使用 Github 进行合并，因此学习这些还不太相关。

## 我学到的最佳实践

这些最佳实践是我根据经验学到的。我将快速列出我认为需要牢记的重要事项:

*   在您暂存(`git add`)您的更改之前，请保存所有文件，因为`git add`只会暂存您已保存的更改。我通常喜欢在登台之前关闭 VS 代码中的所有选项卡，以确保没有遗漏任何需要保存的文件。
*   您不必在每次提交后都进行推送。我通常会添加一些提交，然后在切换分支或注销之前一次全部提交。
*   在切换分支之前保存并提交所有更改
*   当我写提交时，添加一个#后跟 Git 问题号，这样您的提交就会出现在相关的 Git 问题中
*   在创建主程序的新分支之前，一定要做一个`git pull`来确保你的新分支是基于最新的代码
*   如果我使用`git stash`，我会尽可能快地弹出并清除列表
*   合并旧分支后，立即将其从本地存储库中删除，以保持本地存储库分支列表的整洁
*   不要让你的 git 分支比 sprint 更老，因为你离开它们越久，它们就越难合并。如果他们需要进行另一次冲刺，使用`git pull origin master`将主分支合并到你的分支中，并用最新的变化更新它。
*   在 git 中，通常可以引用由空格分隔的多个文件。例如，当您使用`git branch -d <branch name>`时，您可以一次删除多个分支，只要您用空格将它们分开。

## 下一步是什么？

我想我想学习`git rebase`以及如何粉碎承诺。我还听说`git revert`比`git reset`更安全，所以我也要调查一下。在接下来的 6 个月里，我们将看看我是否能学会这些以及我还能学到什么！

感谢阅读！如果你喜欢读这篇文章，请在 LinkedIn 上与我联系，并查看我的其他文章。

如果你是中级新手，可以考虑使用下面的我的推荐链接订阅👇

[](https://medium.com/@andreasmartinson/membership) [## 加入 Medium 并通过我的推荐链接支持我

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

medium.com](https://medium.com/@andreasmartinson/membership) 

## 参考

1.  南 git-scm.com，沙孔和 b .施特劳布
2.  github.com[Git 备忘单](https://education.github.com/git-cheat-sheet-education.pdf)