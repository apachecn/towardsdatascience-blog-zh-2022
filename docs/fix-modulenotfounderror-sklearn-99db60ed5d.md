# 如何修复 ModuleNotFoundError:没有名为“sklearn”的模块

> 原文：<https://towardsdatascience.com/fix-modulenotfounderror-sklearn-99db60ed5d>

## 了解如何正确安装和导入 scikit-学习 Python

![](img/e3c0fa55c0b1bd805539fbcca061d73a.png)

由 [Milad Fakurian](https://unsplash.com/@fakurian?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/box?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

## 介绍

刚接触 Python 的人通常会在安装`scikit-learn`包时遇到麻烦，这个包是事实上的机器学习库。在源代码中导入包时，一个非常常见的错误是`ModuleNotFoundError`

```
ModuleNotFoundError: No module named 'sklearn'
```

这个错误表明`scikit-learn`(又名`sklearn`)包没有安装，或者即使安装了也无法解决。

在今天的简短教程中，我将介绍一些关于在 Python 上安装包的基本概念，这些概念最终可以帮助您摆脱这个错误，并开始处理您的 ML 项目。更具体地说，我们将讨论

*   通过`pip`安装包的正确方法
*   如何将`scikit-learn`升级到最新版本
*   如何正确使用**虚拟环境**和管理包版本
*   如果你正面临这个关于 **anaconda** 的问题，该怎么办
*   如果您在 **Jupyter 笔记本**中遇到此错误，该怎么办

我们开始吧！

## 使用 pip 以正确的方式安装软件包

事实上，您可能在本地机器上安装了多个 Python 版本。每次你安装一个软件包，这个安装只与一个版本相关联。因此，有可能您已经为一个 Python 版本安装了`scikit-learn`，但是您正在使用一个不同的版本执行您的源代码，这可能就是为什么找不到`scikit-learn`的原因。

因此，确保使用正确的命令安装`sklearn`至`pip`。通常，许多用户试图使用命令安装软件包

```
$ pip install package_name
```

或者

```
$ pip3 install package_name
```

以上两个命令都将安装与 Python 相关联的指定包。例如，您可以通过运行

```
$ pip --version**pip 19.0.3 from /usr/lib/python3.7/site-packages/pip (python 3.7)**
```

相反，确保在通过`pip`安装 Python 包时使用以下符号

```
$ python3 -m pip install scikit-learn
```

这将确保将要安装的包对于您将用来运行源代码的 Python 版本是可用的。您可以通过执行以下命令找到特定 Python 可执行文件在本地机器上的位置

```
$ which python3
```

## 将软件包升级到最新版本

此外，确保您使用的是最新版本的`scikit-learn`而不是非常旧的版本可能会有所帮助。要更新到可用的最新版本，您可以运行以下命令:

```
$ python3 -m pip install -U scikit-learn
```

## 使用虚拟环境

Python 的`[venv](https://docs.python.org/3/library/venv.html#module-venv)`模块允许创建所谓的虚拟环境。每个虚拟环境都是完全隔离的，都有自己的 Python 二进制。此外，它还可能在自己的站点目录中有自己的一组安装包。

这意味着，如果某个软件包安装在特定的虚拟环境中，它对于系统范围内安装的软件包或任何其他虚拟环境都是不可见的。

如果您目前没有使用虚拟环境，我建议您开始这样做，因为它将极大地帮助您更轻松、更有效地管理包依赖关系。

现在来看我们的具体用例，如果你希望从事一个需要`scikit-learn`的项目，那么你必须遵循三个步骤。

首先，为您的项目创建一个虚拟环境，并将其放置在您想要的位置。让我们使用名称`my_project_venv`创建一个

```
$ python -m venv /path/to/your/venv/called/my_project_venv
```

既然已经创建了虚拟环境，您现在应该激活它。您可以使用以下命令来完成此操作:

```
$ source /path/to/your/venv/called/my_project_venv/bin/activate
```

如果虚拟环境已经成功激活，您应该能够在命令行中看到 venv 名称作为前缀(例如`(my_project_venv)`)。

现在，您终于可以使用我们之前讨论的命令安装`sklearn`(以及构建 Python 应用程序所需的任何其他依赖项)。

```
(my_project_venv) $ python3 -m pip install scikit-learn
```

并最终执行您的脚本

```
(my_project_venv) $ python3 my_script_using_sklearn.py
```

## 如果您正在使用 anaconda，该怎么办

如果您目前正在使用 conda，您可能必须小心您实际使用的环境。

如果您想在根目录下安装`scikit-learn`(可能不推荐，因为我提到了为每个项目使用隔离虚拟环境的重要性)，那么您可以使用

```
$ conda install scikit-learn
```

或者，如果您想要将`scikit-learn`包安装到特定的 anaconda 环境中，那么您可以使用`-n`标志来指定环境名称。例如，下面的命令将把`scikit-learn`安装到名为`my_environment`的 conda 环境中:

```
conda install -n my_environment scikit-learn
```

如果以上方法都不起作用，那么你仍然可以安装`scikit-learn`到`pip`，即使是在 conda 环境下工作。在 anaconda 提示符下，只需运行

```
$ pip install scikit-learn
```

## 如果你和 Jupyter 一起工作，该怎么做

最后，如果你在 Jupyter 笔记本上安装了这个`ModuleNotFoundError`，那么你需要确保 Jupyter 和`scikit-learn`安装在同一个环境中。

第一步是检查 jupyter 笔记本在本地机器上的安装路径。举个例子，

```
$ which jupyter
$ which jupyter-notebook
```

如前所述，如果你在孤立的环境中工作会好得多(而且肯定会节省你的时间和精力)。

```
$ conda install -c anaconda ipython
```

或者

```
conda install -c anaconda jupyter
```

如果您在特定的 conda 环境中工作，请确保在同一环境中安装 Jupyter 笔记本电脑和`scikit-learn`软件包:

```
$ conda install -n my_environment jupyter
$ conda install -n my_environment scikit-learn
```

如果你在 Python 虚拟环境(又名`venv`)中工作，那么:

```
$ python3 -m pip install jupyter
$ python3 -m pip install scikit-learn
```

最后从激活的环境中打开你的 Jupyter 笔记本，导入`scikit-learn`。你现在应该可以走了！

## 还在烦恼吗？

如果您在导入`scikit-learn`时仍然有问题，那么可能是其他地方出错了。你可能会在我的一篇文章中找到答案

</how-to-fix-modulenotfounderror-and-importerror-248ce5b69b1c>  

## 最后的想法

在今天的文章中，我们讨论了在 Python 源代码中导入`sklearn`的主要原因。此外，我们探讨了一些可能与此问题相关的不同主题，最终可以帮助您解决导入错误。更具体地说，我们讨论了如何使用正确的`pip`安装命令解决问题，如何在隔离的虚拟环境中管理您的项目，以及如何在使用 Anaconda 和/或 Jypyter 笔记本时克服`ModuleNotFoundError`。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

<https://gmyrianthous.medium.com/membership>  

**相关文章你可能也喜欢**

</scikit-learn-vs-sklearn-6944b9dc1736>  </predict-vs-predict-proba-scikit-learn-bdc45daa5972>  </14-must-know-pip-commands-for-data-scientists-and-engineers-a59ebbe0a439> 