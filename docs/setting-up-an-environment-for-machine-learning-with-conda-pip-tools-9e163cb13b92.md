# 使用 Conda & Pip-Tools 建立机器学习环境

> 原文：<https://towardsdatascience.com/setting-up-an-environment-for-machine-learning-with-conda-pip-tools-9e163cb13b92>

![](img/3b6df094375e05d64dcd4dd60c3079af.png)

照片由[拉明·哈蒂比](https://unsplash.com/@raminix?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

## 用 conda 和 pip 工具创建一个健壮和确定的机器学习环境

如果我们在开始时没有设置一些基本规则，为机器学习项目设置一致和确定的环境有时会有点混乱。

***在本文中，我想介绍一下如何为机器学习建立一个健壮的开发环境，以便于管理依赖关系，并保证项目整个生命周期中开发和生产阶段之间的兼容性。***

我们的想法是从头开始，以包含以下内容的文件夹结束:

*   一个指定我们的 Python 和`cuda/cudnn`版本的`environment.yml`文件
*   分别规定开发和生产包要求的`dev.in`和`prod.in`文件
*   一个包含命令的`Makefile`在我们每次修改`environment.yml`文件或者改变`.in`文件中的包时自动更新我们的环境

***免责声明:*** 本文内容使用以下主要资源编写:

*   [https://github.com/full-stack-deep-learning/conda-piptools](https://github.com/full-stack-deep-learning/conda-piptools)
*   [https://github . com/full-stack-deep-learning/fsdl-text-recognizer-project](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-project)
*   [https://github.com/jazzband/pip-tools](https://github.com/jazzband/pip-tools)

主要是我从[“全栈深度学习课程](https://fullstackdeeplearning.com/)教授的课程中学到了很多，它除了是学习这个设置的主要资源之外，还是我所有与实用机器学习相关的主题的参考指南，所以我强烈建议你去看看！

# 步伐

1.  **设置蟒蛇**
2.  **创建虚拟环境并安装依赖项**
3.  **将环境导出到一个** `**environment.yml**` **文件**
4.  **创建需求文件，并为开发和生产添加我们的依赖关系**
5.  **编写一个 MakeFile 文件**

## 1.设置 Anaconda

*   [设置蟒蛇](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda)
*   用`conda -V`确认 conda 版本，在我的情况下，我得到:`conda 4.10.3`
*   将 conda 更新至当前版本:`conda update conda`

在我的情况下我得到:`conda 4.11.0`

## 2.创建虚拟环境并安装依赖项

在这个项目中，我将使用一个`pytorch`示例，因此我将首先使用必要的`cudatoolkit`创建环境，如下所示:

```
conda create -n setup_ml_env python==3.7 cudatoolkit=10.2 cudnn=7.6
```

现在，我们通过运行以下命令来激活环境:

```
conda activate setup_ml_env
```

并且，我们通过运行以下命令来测试安装:

```
python -V
```

预期产出:

```
Python 3.7.11
```

## 3.将环境导出到一个`environment.yml`文件

```
conda env export --from-history > environment.yml
```

`--from-history`命令确保您只向`environment.yml`文件添加目前为止您实际安装的包(在本例中只有`cudatoolkit`和`cudnn`包)。

让我们将`pip`和`pip-tools`添加到这个文件中，以便稍后用于安装我们的 Python 包，然后我们可以打印出文件的内容以供检查:

```
cat environment.yml
```

预期产出:

```
name: setup_ml_env
channels:
  - defaults
dependencies:
  - python=3.7
  - cudatoolkit=10.2
  - cudnn=7.6
  - pip
  - pip:
    - pip-tools
prefix: path/to/setup_ml_env
```

## 4.创建需求文件，并为开发和生产添加依赖性

在 Linux 终端中:

```
mkdir requirements
touch requirements/dev.in
touch requirements/prod.in
```

在`dev.in`文件中我们写道:

```
-c prod.txt
mypy
black
```

这里`-c prod.txt`将开发包约束到生产需求文件`prod.txt`中指定的包，这些包将从`prod.in`文件生成。

在`prod.in`文件里面写着:

```
torch
numpy
```

这只是一个使用`torch`和`numpy`包的玩具项目的示例。

## 5.编写 MakeFile

我们项目的 makefile 将包含:

```
# Command to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'
```

`help`命令打印 makefile 的所有可用命令。

```
# Install exact Python and CUDA versions
conda-update:
	conda env update --prune -f environment.yml
	echo "Activate your environment with: conda activate setup_ml_env"
```

下面是 makefile 命令，用于在我们每次修改`environment.yml`文件时更新我们的环境。

```
# Compile and install exact pip packages
pip-tools:
	pip install pip-tools
	pip-compile requirements/prod.in && pip-compile requirements/dev.in
	pip-sync requirements/prod.txt requirements/dev.txt
```

`[pip-tools](https://github.com/jazzband/pip-tools)`命令编译和安装所有需求的相互兼容的版本。

如全栈深度学习课程的[报告中所述，使用 pip-tools 使我们能够:](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-project/blob/master/setup.md)

*   将开发从生产依赖中分离出来(`dev.in` vs `prod.in`)。
*   为所有依赖项(自动生成的`dev.txt`和`prod.txt`)准备一个精确版本的锁定文件。
*   允许我们轻松部署到可能不支持 conda 环境的目标。

如果您添加、删除或需要更新某些需求的版本，请编辑。在文件中，只需再次运行`make pip-tools`。

## 总结想法

在我的机器学习生涯之初，我只是安装软件包和运行代码，没有考虑依赖问题等事情的负面影响。

现在，即使可能仍然有我遗漏的东西，我觉得这种方法试图覆盖开发机器学习项目的天真方法中的漏洞。

**总之:**

*   `environment.yml`指定 Python 和可选的`cuda/cudnn`
*   `make conda-update`创建/更新 conda 环境
*   `requirements/prod.in`和`requirements/dev.in`指定 python 包需求
*   `make pip-tools`解析并安装所有 Python 包

如果你喜欢视频，可以看看我在 Youtube 上关于这个主题的视频:

如果你喜欢这篇文章，可以考虑加入我的 [Medium](https://lucas-soares.medium.com/membership) 。另外，[订阅我的 Youtube 频道](https://www.youtube.com/channel/UCu8WF59Scx9f3H1N_FgZUwQ)。谢谢，下次再见！:)

*来自 https://enkrateialucca.github.io/lucas_soares_learning/*的博客