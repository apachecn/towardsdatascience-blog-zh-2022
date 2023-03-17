# 诗意的道歉

> 原文：<https://towardsdatascience.com/a-poetic-apology-71a41db4ecce>

## 或者为什么要用诗歌来管理 Python 的依赖性

如果你曾经花时间尝试编写一个 Python 应用程序，你可能在某个时候体验过 Python 臭名昭著的依赖地狱。

![](img/d3a0d553eda348da69287aa270b2fa06.png)

图片由 [Unsplash](https://unsplash.com/) 上的 [Alina Grubnya](https://unsplash.com/@alinnnaaaa) k 拍摄

幸运的是，在你(和我们所有人)写作的时候，有一些很好的解决痛苦纠结的方法，上面的图片生动地描述了这些方法。事实上，您很可能已经知道，如果您想要针对多个 Python 版本进行开发，您可以很容易地使用 [pyenv](https://github.com/pyenv/pyenv) 来解开您扭曲的环境。您可能还知道，如果您碰巧同时处理多个具有冲突依赖关系的项目，那么您可以使用*虚拟环境*来隔离冲突的库。在本文中，我们将介绍另一个工具，[poems](https://github.com/python-poetry/poetry)，并讨论为什么您应该将它添加到您自己的工具包中。

# 问题是

想象一下，在一个孤独的夜晚，您决定启动一个简单的虚拟 Python 项目，准确地说是名为`foo`，其结构如下

```
├── foo
│   ├── foo
│   │   ├── bar
│   │   │   └── data.py
│   │   └── constants.py
│   └── README.md
```

由于这不是您的第一个 Python 项目，并且您希望避免花费更多的时间来修复您的系统和项目模块之间的不兼容性，您努力地从您的 shell 中用

```
$> python -m venv ~/Desktop/venv/foo-venv
```

并在新创建的项目中激活它

```
$> source ~/Desktop/venv/foo-venv/bin/activate
```

配备了一个隔离的环境，您成功地开始安装无处不在的熊猫数据库。为了实现这一点，你使用 Python 事实上的包管理器`pip`,并小心地固定库版本以确保可复制性

```
$> pip install pandas**==**0.25.3
```

由于您在进行探索性数据分析方面有点懒惰，您还安装了漂亮的`pandas-profiling`模块来帮助您完成这项繁琐的工作

```
$> pip install pandas-profiling**==**2.5.0
```

在所有这些调情之后，您最终开始编码(假设将下面几行添加到`data.py`文件中，事实上，可以这样称呼它)

```
**import** pandas **as** pd
**from** pandas_profiling **import** ProfileReport
df**=**pd.DataFrame([['a',1],['b',None]],columns**=**['category', 'value'])
df['category']**=**df['category'].astype('category')
**if** __name__**==**'__main__':ProfileReport(df).to_file('foo.html')
```

鉴于您滥用`print`语句进行调试的日子已经一去不复返了，您安装了漂亮而方便的 [pdbpp](https://github.com/pdbpp/pdbpp) 库来检查这些代码行是否如预期的那样工作

```
$> pip install pdbpp**==**0.10.2
```

并使用`python -m pdb -cc data.py`在*事后调试模式*下运行您的代码。

满意于干净的运行，您现在意识到为了发布您的华丽的应用程序而不落入“在我的机器上工作”的陷阱，您需要一种方法来收集所有的依赖项。快速的 Google 搜索会显示 pip 的`freeze`子命令允许通过下面的咒语将当前的环境包记录到一个`requirements.txt`文件中

```
$> pip freeze **>** requirements.txt
```

它允许任何人通过简单地安装所需的依赖项来使用您的项目

```
$> pip install -r requirements.txt
```

就在你即将向世界展示你的杰作项目时，你意识到改进的调试模块实际上只供你在开发时使用。有了将冻结的需求分割成单独的生产和开发文件的想法，您可以查看生成的文件，却发现您的应用程序的依赖项的每一个子依赖项都在其中列出，并锁定到一个特定的版本。预见到维护这个庞大列表的噩梦，你卸载了`pdbpp`库以确保一个干净的需求文件

```
$> pip uninstall -y pdbpp **&&** pip freeze **>** requirements.txt
```

然而，快速浏览一下修改后的需求文件，会发现事情并不像预期的那样:`pdbpp`确实被删除了，但是它的依赖项，比如`fancycompleter`，仍然被安装。因为这似乎是一个死胡同，所以您选择从零开始，手动创建一个只有生产依赖关系的`requirements.txt`文件

```
pandas==0.25.3
pandas-profiling==2.5.0
```

以及等效的开发文件`requirements_dev.txt`，仅包含

```
pdbpp==0.10.2
```

通过记录孤立的顶级包，似乎帮助您避开了可怕的 Python 依赖地狱，这种聪明给您留下了深刻的印象，您决定今天就到此为止，第二天再对您的应用程序进行最后一次测试。

早上醒来，新闻铺天盖地:熊猫 v1 终于出来了(才过了十二年！).用令人难以置信的长 changelog 拖延了几个小时，使你得出结论，你的复杂 foo-project 肯定会通过更新到全新的版本而获得显著的改进。现在，既然你已经锁定了熊猫的确切版本，你不能简单地运行

```
$> pip install -U -r requirements.txt
```

相反，你必须执行

```
$> pip install pandas**==**1.0.0
```

这导致了一个特别奇怪和混乱的情况:您的终端弹出一个错误

```
ERROR: pandas-profiling 2.5.0 has requirement pandas==0.25.3, but you'll have pandas 1.0.0 which is incompatible.
```

但是`pandas 1.0.0`的安装还是发生了。假设这是一个`pip`出错的警告，你相应地更新你的`requirements.txt`文件，然后愉快地最后一次运行你的`data.py`模块，却发现它抛出了一个神秘的`TypeError`。感觉现在被 pip 明显无法解决依赖性所出卖，您回滚您的更改并坚持使用 Pandas(现在)过时的版本。

此时，您似乎有一个工作项目，但是 I)您不确定恢复 Pandas 版本是否会破坏您的应用程序的可复制性，ii)代码肯定会看起来更好，iii)睡了一夜好觉之后，您承认您的应用程序的整体功能没有您前一天晚上想象的那么复杂和丰富。为了解决前两个问题，您首先将`black`格式化程序添加到您的`requirements_dev.txt`中

```
black==19.10b0
```

然后在您的项目目录中，您用

```
$> rm -rf ~/Desktop/venv/foo-venv
$> python -m venv ~/Desktop/venv/foo-venv
$> source ~/Desktop/venv/foo-venv/bin/activate
$> pip install -r requirements_dev.txt
$> pip install -r requirements.txt
```

现在你在你的项目根中运行`black`(用`black .`)，并且对它所做的美化工作非常满意，但是为了遵守 Mutt Data 的格式风格(这恰好符合你不喜欢把每个单引号都变成双引号的习惯)，你添加了一个`pyproject.toml`，告诉`black`跳过这种可怕的字符串规范化默认设置

```
[tool.black]
skip-string-normalization = **true**
```

代码现在看起来很棒，新的事后调试运行表明，在新的(可复制的)环境中，一切似乎都工作得很好。在将代码部署到服务器上或与外界共享之前，唯一要做的事情是避免在代码周围硬编码常量，如报告名称。因此，您决定将下面几行添加到您的`constants.py`空模块中

```
REPORT_FILE **=** 'foo.html'
```

并修改`data.py`从相关父文件中导入该常量

```
**from** ..constants **import** REPORT_FILE
```

然而不幸的是，现在新的`data.py`运行显示下一个错误

```
ImportError: attempted relative import with no known parent package
```

根据无所不知的说法，这是有道理的，因为 Python 相对导入只在一个包中工作，因此如果你想从一个父目录导入，你应该创建这样的包或者破解`sys.path`。作为一个真正的纯粹主义者，你选择了前一条道路，并用以下内容创建了一个`setup.py`

```
**from** setuptools **import** setup**with** open('requirements.txt') **as** f:
    install_requires **=** f.read().splitlines()
**with** open('requirements_dev.txt') **as** f:
    extras_dev_requires **=** f.read().splitlines()setup(
    name**=**'foo',
    version**=**'0.0.1',
    author**=**'Mutt',
    author_email**=**'info@muttdata.ai',
    install_requires**=**install_requires,
    extras_require**=**{'dev': extras_dev_requires},
    packages**=**['foo'],
)
```

现在在一个全新的虚拟环境中，你用`pip install -e .[dev]`在可编辑模式下安装你的包，改变`data.py`中的导入行来解释包的结构

```
**from** foo.constants **import** REPORT_FILE
```

祈祷一切最终都顺利…

一切都确实(很容易)工作，但不知何故，所有让它工作的跳跃让你感到不安。简短的反思揭示了恐惧浪潮的几个原因:

1.  因为您计划同时处理多个 Python 项目，所以隔离是工作流的一个基本部分。虚拟环境确实解决了这个问题，但是激活/停用过程很麻烦并且容易忘记。
2.  隔离项目之间的依赖关系并不能解决项目内部的依赖冲突。适当的依赖解析是任何值得尊敬的包管理器的首要要求，然而`pip`直到 2020 年 10 月才实现了这个特性。人工保证复杂项目中的依赖一致性是一个死锁。
3.  如果你想把你的应用程序/项目作为一个包来安装，你必须在已经有多个需求文件的基础上增加一个`setup.py`的开销。但是，您已经阅读了 PEPs 517-518，并且想要尝试其中提到的更简单、更安全的构建机制。
4.  您考虑在不同的机器上尝试您的应用程序，但是意识到它运行 Python 3.7，而您的本地机器运行 3.8。要将`pyenv`用于您的隔离虚拟 env，您需要一个额外的插件 [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) ，这使得管理 venvs 更加麻烦。
5.  您曾短暂地使用过 [Pipenv](https://github.com/pypa/pipenv) ，它承诺给 Python 带来其他语言的更成熟的包管理器(如 Javascript 的`yarn/npm`或 Rust 的`Cargo`)令人羡慕的特性，但很快就失望了。Pipenv 不仅误导性地宣称自己是 Python 官方推荐的打包工具(实际上它是为编写应用程序而不是包而设计的)，而且它也没有发布超过一年，并且在创建确保可重复/确定性构建的锁文件时仍然无休止地挂起。

在绝望无助的状态下，你开始疯狂地在网上搜索，看看是否已经有了解决所有这些问题的方法。在众多部分/不完整的候选人中，你终于遇到了一个令人难以置信地让他们全都崩溃的候选人:这就是诗歌。

# 解决方案

# 安装(带 Pipx)

诗歌是一个用 Python 编写的 CLI 应用程序，所以你可以简单地用`pip install --user poetry`安装它。然而，你可能已经安装了或者将要安装其他 Python CLI 应用程序(例如奇特的 PostgreSQL 客户端`pgcli`或者下载 YouTube 视频的`youtube-dl`)。如果你用你系统的软件包管理器(比如说`apt`、`yay`或`brew`)来安装它们，它们将被安装在一个全局级别，它们的依赖关系可能会发生冲突。你可以为每一个创建单独的 venv，但是为了使用它们，你必须首先经历激活环境的麻烦…

为了避免这种恼人的场景，你可以使用 [pipx](https://github.com/pipxproject/pipx) ，它将在一个隔离的虚拟环境中精确地安装软件包，同时让它们在你的 shell 中随时可用(也就是将可执行文件添加到你的二进制文件中`$PATH`)。除了为全球访问提供 CLI 应用程序之外，它还可以轻松列出、升级和卸载这些应用程序。要用`pipx`安装诗歌，首先要用

```
$> python -m pip install --user pipx
$> python -m pipx ensurepath
```

然后直接做

```
$> pipx install poetry
```

如果你喜欢生活在边缘(像我一样)，你可以选择安装一个带`pipx install --pip-args='--pre' poetry`的`pre-release`版本。

# 使用

现在，你已经准备好去尝试诗歌承诺的奇迹了。为此，您用上面的`.py`文件创建一个名为`foo-poetry`的新文件夹/项目，然后运行`poetry init`。一个交互式提示将开始要求您提供关于您的包的基本信息(名称、作者等)，这些信息将用于创建一个`pyproject.toml`文件。这基本上是您之前添加到`setup.py`中的相同元数据，只有一些微小的变化

```
This command will guide you through creating your pyproject.toml config.Package name **[**foo-poetry]:  foo
Version **[**0.1.0]:  0.0.1
Description **[]**:
Author **[**petobens <petobens@yahoo.com>, n to skip]:  Mutt <info@muttdata.ai>
License **[]**:
Compatible Python versions **[**^3.8]:  ~3.7Would you like to define your main dependencies interactively? **(**yes/no**)** **[**yes**]** no
Would you like to define your development dependencies interactively? **(**yes/no**)** **[**yes**]** no
Generated file**[**tool.poetry]
name **=** "foo"
version **=** "0.0.1"
description **=** ""
authors **=** **[**"Mutt <info@muttdata.ai>"**]****[**tool.poetry.dependencies]
python **=** "^3.7"**[**tool.poetry.dev-dependencies]**[**build-system]
requires **=** **[**"poetry-core>=1.0.0a5"**]**
build-backend **=** "poetry.core.masonry.api" Do you confirm generation? **(**yes/no**)** **[**yes**]** yes
```

需要强调的两个相关设置是构建系统和 Python 版本规范。关于第一种方法，你现在唯一需要知道的是，它使用 PEPs 517–518 中的标准来定义一种替代方法，在没有`setuptools`的情况下从源代码构建一个项目(因此消除了对`setup.py`文件的需要)。关于第二个设置，为了理解指定 Python 版本约束的语法，您应该阅读[诗歌版本文档](https://python-poetry.org/docs/versions/)，在那里您会发现脱字符号(^)要求意味着只允许较小的和补丁更新(即我们的应用程序将与 Python 3.7 和 3.8 一起工作，但不能与 4.0 一起工作)。

到目前为止，您只有一个`TOML`文件(您也可以用它来集中您的`black`配置)。如何指定依赖关系？简单地跑

```
$> poetry add pandas**==**0.25.3
```

这导致了

```
Creating virtualenv foo-KLaC03aC-py3.8 **in** /home/pedro/.cache/pypoetry/virtualenvsUpdating dependencies
Resolving dependencies... **(**0.6s**)**Writing lock file Package operations: 5 installs, 0 updates, 0 removals - Installing six **(**1.15.0**)**
  - Installing numpy **(**1.19.1**)**
  - Installing python-dateutil **(**2.8.1**)**
  - Installing pytz **(**2020.1**)**
  - Installing pandas **(**0.25.3**)**
```

换句话说，一个初始的`add`命令将会 I)创建一个虚拟环境，ii)安装所请求的包及其子依赖项，iii)将每个下载的依赖项的确切版本写入到`poetry.lock`文件中(您应该将它提交到您的 VCS 中以加强可复制性),以及 iv)将新添加的包添加到`pyproject.toml`文件的`tool.poetry.dependencies`部分。最后一项还表明，如果您想安装一个新的依赖项，您可以重用`add`命令，或者直接在您的`pyproject.toml`文件中添加这样一行。例如，如果您现在想要添加`pandas-profiling`库，那么您可以修改 pyproject，使其具有

```
pandas-profiling = "2.5.0"
```

由于在这个阶段已经存在一个`poetry.lock`文件，如果您现在运行`poetry install`，那么 poems 将使用这个锁文件中指定的版本来解析和安装依赖关系(以确保版本的一致性)。然而，由于您手动向`pyproject.toml`文件添加了一个新的依赖关系，`install`命令将会失败。所以，在这种情况下，你需要运行`poetry update`，本质上相当于删除锁文件，再次运行`poetry install`。

添加一个`development`依赖项以类似的方式工作，唯一的警告是在执行`add`命令时需要使用`--dev`标志

```
$> poetry add pdbpp**==**0.10.2 --dev
$> poetry add black**==**19.10b0 --dev
```

产生的包将被附加到`tool.poetry.dev-dependencies`部分。

既然依赖关系已经设置好了，你就可以运行你的代码`data.py`文件了

```
$> poetry run python data.py
```

它将在项目的 virtualenv 中执行命令。或者，您可以简单地通过运行以下命令在活动的 venv 中生成一个 shell

```
$> poetry shell
```

现在假设您想要更新 Pandas 版本，就像您之前在检查 pip 无法执行依赖关系解析时所做的那样。为此，您可以像下面这样更新约束

```
$> poetry add pandas**==**1.0.0
```

这一次正确地失败了，并出现以下错误

```
Updating dependencies
Resolving dependencies... **(**0.0s**)****[**SolverProblemError]
Because pandas-profiling **(**2.5.0**)** depends on pandas **(**0.25.3**)**
 and foo depends on pandas **(**1.0.0**)**, pandas-profiling is forbidden.
So, because foo depends on pandas-profiling **(**2.5.0**)**, version solving failed.
```

到目前为止，您注意到诗歌似乎解决了您在上一节中列出的最初两个要求(即简单的项目隔离和适当的自动依赖解析)。在你满怀希望之前，你要验证它是否能直接打包你的代码(特别是没有`setup.py`)。值得注意的是，这个简单的代码归结为包含了下面一行:`pyproject.toml`文件的`tool.poetry`部分

```
packages = [{include = "foo"}]
```

随后执行一个新的`poetry install`，默认情况下，它将在可编辑模式下安装项目。

被诗歌的简单易用所激动，你开始怀疑诗歌是否是你一直在寻找的终极工具。它能检查所有的盒子吗？为了最终回答这个问题，您想看看在不同 Python 版本之间切换是否容易。假设您的本地机器默认使用 Python 3.8，那么您随后安装了带有`pyenv install 3.7.7`的`3.7.7`(由于您将 3.7 设置为您的应用程序`pyproject.toml`的下限，因此安装先前的版本将不会起作用)。为了使这个版本在本地可用，您将一个`.python-version`文件添加到您的项目的根目录中，该文件包含一个带有`3.7.7`的单行，然后告诉 poem 创建一个带有该版本的 virtualenv 并使用它

```
$> poetry env use 3.7
```

一旦你用`poetry env list`检查它是否被正确激活，你就用`poetry install`安装所有的依赖项，并最终运行你的代码,(不出所料)没有问题地完成。

惊叹于其直观的质朴，你得出结论，诗歌正是你所需要的。事实上，您还不知道这一点，但是您得到的比您期望的要多得多，因为您只触及了特性的表面。你仍然需要发现它可以并行安装软件包，当一切失控时抛出漂亮的彩色异常，与你选择的 IDE/编辑器集成(如果那是 vim，你可以试试你卑微的仆人的[无耻地处理这件事](https://github.com/petobens/poet-v))，有一个命令可以直接发布一个软件包，除了其他无数的乐趣之外，还计划有一个插件系统来进一步扩展。

**有一点非常清楚:诗歌是明天的 Python 包管理器。你不妨今天就开始使用它。**

有兴趣阅读更多机器学习相关内容，请访问我们公司的[博客](https://muttdata.ai/blog/)。我希望你觉得这篇文章有用，至少有点娱乐性。如果你需要帮助找出用数据推动业务的最佳方式，你可以在这里找到我们的公司。

这篇文章也被[发表在](https://muttdata.ai/blog/2020/08/21/a-poetic-apology.html)Mutt Data 的公司博客上。