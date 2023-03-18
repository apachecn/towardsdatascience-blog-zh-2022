# 如何创建可重现的数据科学项目

> 原文：<https://towardsdatascience.com/how-to-create-a-reproducible-data-science-project-682d3a1b98ed>

## Anaconda 项目初学者指南

由[索菲娅·杨](https://sophiamyang.medium.com/)和[艾伯特·德弗斯科](https://www.linkedin.com/in/albertdefusco/)拍摄

![](img/e8fb3c95584b16a9148ea9760c1824f8.png)

照片由[克里斯·巴尔巴利斯](https://unsplash.com/@cbarbalis?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

创建可重复的数据科学项目非常重要。数据科学家需要能够与其他人共享项目，并确保其他人能够产生相同的结果。在我们看来，可再生数据科学集中在三个领域— **可再生代码**(例如使用 [Git](/git-workflow-for-data-scientists-c75445f23f44?sk=579671f9fcc2000bff07bdcba4777bcd) )、**可再生代码行为**(例如[编写测试](/testing-for-data-scientists-1223fcad4ac2?sk=bde5487fe3ad11a06ae3a92ed80d451b))和**可再生环境**。

在创建数据科学可复制环境方面，大多数数据科学家都熟悉 [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) (针对 conda 用户)或 [venv](https://docs.python.org/3/library/venv.html) (针对 Pip 用户)。一个不太为人所知但强烈推荐的选择是 [Anaconda 项目](https://anaconda-project.readthedocs.io/en/latest/getting-started.html)。它是一个 YAML 文本文件，有助于创建一个可再现的数据科学项目，并且不限于可再现的环境。Anaconda 项目具有以下特性:

*   它指定了 Conda 和 Pip 包来创建可复制的环境。
*   它将数据下载到项目中。
*   它会跟踪您想要执行的命令。
*   它允许您设置环境变量。
*   它易于共享，并提供了上传到 Anaconda Cloud 的选项。
*   它用一个简单的 CLI 命令构建 docker 映像。

# **一个例子**

下面是 anaconda-project.yml 的样子。在本例中，我们已经定义了包/依赖项、可执行命令和环境变量。

# **如何安装 Anaconda 项目？**

如果您还没有安装 Anaconda 或 Miniconda，请先安装它们。那么您应该可以在您的终端中访问`conda`。然后在命令行中，键入`conda install anaconda-project`来安装 Anaconda 项目。

# **如何创建项目？**

*   要从头开始创建项目，请运行

```
anaconda-project init
```

然后，您将看到在您的目录中创建了一个“anaconda-project.yml”文件。

*   要从现有的 Conda 环境中创建一个项目，运行

```
conda env export –from-history environment-name > anaconda-project.yml
```

# **如何添加包？**

要添加包(例如，本例中的 pandas 和 numpy)，您可以运行`anaconda-project add-packages pandas numpy`来添加 Conda 包，或者运行`anaconda-project add-packages --pip pandas numpy`来添加 Pip 包。然后，您应该会看到这些包出现在您的 anaconda-project.yml 文件中。

# **如何添加环境变量？**

Anaconda 项目允许用户通过运行

```
anaconda-project add-variable --default=default_value VARIABLE
```

# **如何添加可执行命令？**

例如，如果您的目录中有一个 Python 文件“hello.py ”,那么您可能想要运行`python hello.py`来运行这个代码。使用 Anaconda Project，您可以通过运行`anaconda-project add-command hello “python hello.py"`在 anaconda-project.yml 文件中添加这个可执行命令。然后，我们可以简单地运行`anaconda-project run hello`，而不是激活您的环境然后运行`python hello.py`，这将在您在 anaconda-project.yml 文件中定义的环境中自动运行这段代码。多简单啊！

类似地，您可以添加一个命令来启动 Jupyter 笔记本或 Python 应用程序。例如，这里我们添加了一个名为“notebook”的命令来启动笔记本，另一个名为“app”的命令来启动应用程序

```
anaconda-project add-command notebook notebook.ipynbanaconda-project add-command app “panel serve notebook.ipynb”
```

然后我们可以运行`anaconda-project run notebook`和`anaconda-project run app`来启动笔记本和应用程序。

# **如何分享一个项目？**

共享项目有三种方式:

*   我们可以直接在您的中创建一个归档文件:

```
anaconda-project archive filename.zip
```

然后，您将看到您的项目文件保存为 filename.zip。

*   我们还可以将我们的项目上传到 Anaconda Cloud:

```
anaconda-project upload
```

*   我们还可以创建一个 Docker 图像:

```
anaconda-project dockerize
```

# 其他的

如果您对 yaml 文件进行了大量的修改，您可以使用`anaconda-project prepare --refresh`从头构建环境。

要锁定项目，请运行`anaconda-project lock`。您可以使用`anaconda-project add-packages`添加包。但是如果直接编辑`anaconda-project.yml`，可以运行`anaconda-project update`更新`anaconda-project-lock.yml`来匹配。

通过使用 Anaconda Project，您不仅可以创建可再现的环境，还可以在您的数据科学项目上创建可再现的可运行命令和配置。试一试，让我们知道你的想法。

# 参考资料:

[https://anaconda . cloud/webinars/making-reproducible-conda-based-projects？UTM _ medium = web&UTM _ source = Anaconda-com&UTM _ campaign = Webinar](https://anaconda.cloud/webinars/making-reproducible-conda-based-projects?utm_medium=web&utm_source=Anaconda-com&utm_campaign=Webinar)

[https://anaconda-project . readthedocs . io/en/latest/index . html](https://anaconda-project.readthedocs.io/en/latest/index.html)

[https://towards data science . com/testing-for-data-scientists-1223 fcad 4 AC 2？sk = bde 5487 Fe 3 ad 11 a 06 AE 3a 92 ed 80d 451 b](/testing-for-data-scientists-1223fcad4ac2?sk=bde5487fe3ad11a06ae3a92ed80d451b)

[https://towardsdatascience . com/git-workflow-for-data-scientists-c 75445 f 23 f 44？sk = 579671 f 9 FCC 2000 BFF 07 BDC ba 4777 BCD](/git-workflow-for-data-scientists-c75445f23f44?sk=579671f9fcc2000bff07bdcba4777bcd)

[https://docs . conda . io/projects/conda/en/latest/user-guide/tasks/manage-environments . html](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

由[索菲亚·杨](https://sophiamyang.medium.com/)和[阿尔伯特·德福斯科](https://www.linkedin.com/in/albertdefusco/)于 2022 年 3 月 15 日