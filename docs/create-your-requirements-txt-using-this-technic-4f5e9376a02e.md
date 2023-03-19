# 使用这种技术创建 requirements.txt

> 原文：<https://towardsdatascience.com/create-your-requirements-txt-using-this-technic-4f5e9376a02e>

## 停止使用没有附加过滤器的“pip 冻结”

![](img/a7baaa289e71a6febf35e1e661b10a4b.png)

照片由[Joel&Jasmin fr estbird](https://unsplash.com/@theforestbirds?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/snow-landscape?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄

**简介**

`requirements.txt`文件是 Python 数据科学或机器学习项目中非常重要的文档，因为它不仅列出了运行代码所必需的包，还注册了它们各自的版本。这些数据增加了项目的可重复性，例如，允许其他人在他们的机器上创建一个新的虚拟环境，激活它，并运行`pip install -r requirments.txt`。因此，用户将在本地安装相同的软件包，具有相同的版本，所有这一切都在几秒钟内完成。

**传统** `**requirements.txt**`的问题**文件**

创建一个`requirements.txt`文件最常用的技术是当所有的包都已经安装好的时候运行`pip freeze > requirements.txt`。然而，这种方法的问题是，它不仅将您通过`pip install <package_name>`实际安装的 Python 包保存到`requirements.txt`中，还保存了它们自己的依赖包。这就是我的意思。

让我们考虑下面的场景:在一个新的虚拟环境中，我将只安装 Pandas 和 Django 作为我的项目的 Python 额外包。所以，我只是跑:

```
pip install pandas django
```

然而，这两个重要的 Python 模块依赖于其他包才能正常工作。例如，Pandas 是建立在 Numpy 之上的，所以前者将在运行`pip install pandas`时自动安装，以便我们可以使用后者。

Django 也是如此:当执行`pip install django`时，其他包会同时自动安装，因为 Django 依赖它们来运行。如果您现在执行`pip freeze > requirements.txt`，您将不会有一个只有两行的新文件，而是有 9 行(一个用于 Pandas，一个用于 Django，七个不必要的用于它们的依赖项)。

这是纯`pip freeze`方法最让我恼火的地方:它用不必要的信息污染了你的`requirements.txt`(所有额外的依赖包)。拥有一个实际上只列出你使用`pip`安装的软件包的`requirements.txt`不是更好吗？如果你的答案是肯定的，请继续阅读，我将向你展示如何用`grep` Linux 命令做到这一点。

**我现在创建一个** `**requirements.txt**` **文件**

因为我只安装了 Django 和 Pandas，所以我想在我的`requirements.txt`中只列出这两个。以下命令正是这样做的:

```
pip freeze | grep -i pandas >> requirements.txtpip freeze | grep -i django >> requirements.txt
```

请注意，这两个命令之间唯一的区别是包名。

因此，新的命令结构有一个管道(符号`|`)。它允许`pip freeze`的输出被用作`grep`命令的输入，这将只保留出现单词`pandas`和`django`的行。添加`-i`标志使`grep`不区分大小写是必要的，因为有些封装在`pip freeze`中以首字母列出。然后我们使用`>>`符号将这个新的过滤列表附加到`requirements.txt`文件中。

**创建一个 BASH 函数来自动化这个过程**

进一步说，我认为有一个 bash 函数是很有趣的，当使用任意数量的 Python 包名作为参数调用时，可以用`pip`安装它们，并自动将它们的信息附加到一个`requirements.txt`文件中。因此，在网上做了一些研究后，我创建了 bash 函数，如下所示:

```
pip_requirements() {if test "$#" -eq 0
then 
  echo $'\nProvide at least one Python package name\n' 
else 
  for package in "$@"
  do
    pip install $package
    pip freeze | grep -i $package >> requirements.txt
  done
fi}
```

在您的终端会话中创建了这个函数之后，您将能够调用它来代替纯粹的`pip install`命令。这里有一个例子:

`pip_requirements django pandas seaborn streamlit`

因此，只需上面的命令，您就可以安装这四个 Python 包，并创建一个只有它们的名称和版本号的干净的`requirements.txt`。

**结束语**

现在我的`requirement.txt`文件摆脱了不必要的信息，我想我可以睡得更好，甚至生活得更幸福！

玩笑归玩笑，尽管这些过程对某些人来说似乎太麻烦了，但我确实认为一个更干净的`requirements.txt`文件结合了 Python 从我们的项目中删除不必要代码的理念。当我们想要快速检查项目所有者实际上安装了什么样的包来构建他们的代码时，这也会有所帮助。

亲爱的读者，非常感谢你花时间和精力阅读我的文章。

</build-a-django-crud-app-by-using-class-based-views-12bc69d36ab6>  </django-first-steps-for-the-total-beginners-a-quick-tutorial-5f1e5e7e9a8c>  <https://medium.com/analytics-vidhya/python-and-openpyxl-gather-thousands-of-excel-workbooks-into-a-single-file-eff4e8c9b514>  

我有很多关于 Python 和 Django 的文章。如果你喜欢这篇文章，[可以考虑在媒体](https://fabriciusbr.medium.com/)上关注我，并在我发表新文章后订阅接收媒体通知。

*编码快乐！*