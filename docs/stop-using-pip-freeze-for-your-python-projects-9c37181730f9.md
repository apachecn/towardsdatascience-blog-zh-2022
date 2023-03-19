# 停止对你的 Python 项目使用“pip 冻结”

> 原文：<https://towardsdatascience.com/stop-using-pip-freeze-for-your-python-projects-9c37181730f9>

![](img/8a50dff526ceba20f0e14be7d819ee59.png)

由 [Dev Benjamin](https://unsplash.com/@dev_irl?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

## 讨论为什么“pip 冻结”在管理 Python 依赖项时不那么酷

# 介绍

几年前我遇到过 pip freeze 和 virtual env，我完全被震撼了。我是一个总是害怕虚拟环境的人(别问了)，但一旦我知道管理我的依赖关系是多么容易，我就停不下来了。我感觉自己就像一个高级开发人员，为我所有的项目制作虚拟环境。从那时起，我的开发技能成倍增长，我找到了管理我的 Python 项目的完美方法，对吗？不对！

这个童话故事发生几个月后，当我回到我的老项目时，我开始面临问题。它们会停止运行，或者可用的依赖项会抛出一些兼容性错误。我很困惑，因为我认为我做的每件事都是对的。我已经通过创建一个虚拟环境来分离项目及其依赖项，那么为什么我的旧项目没有运行呢？事实证明，让我成为更好的 Python 开发人员的那个东西正成为我的障碍——***pip 冻结。*** 这是因为使用 *pip 冻结处理子依赖的方式。*

# 问题是

早些时候，当我开始一个新项目时，只要我安装了库，我就会运行我最喜欢的命令。

```
pip freeze > requirements.txt
```

这就是它引起问题的原因。假设您在项目中安装了包 **A** ，它可能有一个子依赖项 B、C 和 d

```
**A==1.0
B==2.0
C==1.4
D==1.2**
```

现在说，库 **A** 的所有者发布了一个新版本，它使用了库 **B** 的不同版本，并删除了库 **C** 。由于 **B** 和 **C** 已经安装完毕 *pip freeze* 会自动将它们捡起来，并按照它们最初安装时的版本进行卸载。现在，在一个有 100 多个依赖项的项目中，当您更改库时，您的需求文件将会变得非常有问题。您必须识别所有的子依赖项，并相应地删除它们。在这个例子中，如果 A 现在从项目中删除，您仍然会被 **B** 、 **C** 和 **D** 所困扰，即使它们只是因为 **A** 而被安装。删除它们中的每一个都是一项庞大的任务，在大型项目中可能会非常烦人。

这个问题还会引发其他许多问题，这些问题可能会在未来的任何一天破坏您的项目。

# 解决方案

然而，我在这里不仅仅是有问题，我也有一个解决方案。我找到了一个名为 [***pipreqs***](https://pypi.org/project/pipreqs/) 的库，修复了以上所有问题，非常人性化。

## 为什么更好？

以下是为什么切换到 pipreqs 是比使用 pip freeze 更好的想法的原因。

1.  ***pip 冻结*只保存在虚拟环境【1】**中使用 *pip install* 安装的包

*pip 冻结*只安装那些使用 *pip 安装*命令安装的软件包。然而，pip 并不是唯一的 python 包管理器。我们也可以使用*巧克力*、*康达*、*套装工具*等。并且它们不受 *pip 冻结*的支持，所以我们必须在 *requirements.txt* 文件中手动编写它们。 *pipreqs，*则没有这样的限制。

2. ***pip 冻结*保存环境中的所有包和依赖项，包括那些您在当前项目中不使用的包[1]**

这是 *pip 冻结的最大弊端。*在一个项目中，依赖关系不断变化，必须添加、更新和删除。然而，使用 pip freeze 实现这一点是一项艰巨的任务，因为它会转储环境中已经安装的任何内容。 *pipreqs，*另一方面，只把那些已经通过导入在项目中使用过的库放到需求文件中。当您稍后试图更改需求文件时，这是非常强大的。

3. ***pipreqs* 极其容易使用**

要生成一个 *requirements.txt* 文件，您所要做的就是运行下面的命令。

```
**$ pipreqs**
```

如果 requirements.txt 文件已经存在，那么运行下面的命令。

```
**$ pipreqs --force**
```

这将在项目的主目录中生成一个 requirements.txt 文件。如果要将文件保存在特定位置，也可以提供路径。

```
$ pipreqs /home/project/location
```

要了解更多关于图书馆的信息，[点击这里](https://pypi.org/project/pipreqs/)。你也可以看看其他的选择，比如 we [pip-tools](https://github.com/jazzband/pip-tools) 和[poems](https://python-poetry.org/)。

# 结论

*pip 冻结*最初可能看起来非常有用，但它可能会因为以下原因搞乱您的项目:

1.  它转储项目中安装的所有库，包括 requirements.txt 文件中的依赖项和子依赖项。
2.  它仍然错过了没有使用 pip 安装的库。
3.  如果项目中没有使用某个库，它不会自动移除该库。

出于上述原因，建议使用 *pipreqs，*一个 Python 库，它修复了上述所有问题，并且更易于使用。

# 参考

1.  pipreqs 官方文件:【https://pypi.org/project/pipreqs/ 

我写关于 Python 和数据科学的文章。这是我写的其他一些标题。

[](https://levelup.gitconnected.com/5-new-features-in-python-3-11-that-makes-it-the-coolest-new-release-in-2022-c9df658ef813)  [](/7-must-read-books-for-data-scientists-in-2022-aa87c0f9bffb)  [](/creating-multipage-applications-using-streamlit-efficiently-b58a58134030) [## 使用 Streamlit 创建多页面应用程序(高效！)

towardsdatascience.com](/creating-multipage-applications-using-streamlit-efficiently-b58a58134030)