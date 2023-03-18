# 诗歌的依赖性管理

> 原文：<https://towardsdatascience.com/dependency-management-with-poetry-f1d598591161>

## 使用 pyenv 和诗歌组织 Python

![](img/d7e5ee9582f6461fcb0fff1288c1e447.png)

照片由 [Clément Hélardot](https://unsplash.com/@clemhlrdt?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/python?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

Python 的一个重要特点是管理它的版本、环境和依赖性——从而将项目交付给其他人。在本帖中，我们将看看打包和依赖管理器 [**诗歌**](https://python-poetry.org/)**——在作者看来，这是做这些事情的最佳方式。**

**在开始之前，让我们定义一些术语并澄清一些困惑，这也是我最初的受害者:除了诗歌，我们将使用[**pyenv**](https://github.com/pyenv/pyenv)**——一个 Python 安装管理器。pyenv 允许我们在我们的系统 s.t .上管理不同的 Python 版本。例如，我们可以在项目 A 中使用 Python 2.7，在项目 B 中使用 Python 3.10——此外，如果我们只是通过`pip install ...`将所有需要的包安装到我们的系统 Python 中，还可以防止我们的 Python 安装受到污染。****

****除此之外，还有不同的打包和依赖管理器——比如 poem，但也有 pipenv、venv 或 conda。在这篇文章中，我们将强调诗歌，正如我之前提到的，我个人认为这是最好的解决方案。****

# ****安装 Pyenv****

****我们首先安装 pyenv(假设您在 Linux 机器上，对于所有其他操作系统，请遵循[安装指南](https://github.com/pyenv/pyenv)):****

```
**curl [https://pyenv.run](https://pyenv.run) | bash**
```

****然后，在 pyenv 中，我们安装我们希望用于项目的 Python 版本，例如:****

```
**pyenv install -v 3.10.0**
```

# ****设置诗歌****

****按照描述安装 Python 后，我们将注意力转移到诗歌上。****

****通过以下方式安装:****

```
**curl -sSL [https://install.python-poetry.org](https://install.python-poetry.org) | python3 -**
```

****然后，切换到您的 Python 项目文件夹，将诗歌指向您想要使用的 Python 版本——在我们的示例中，这是通过(您可能需要一个`pyproject.toml`文件，见下文)完成的:****

```
**pyenv local 3.10.0poetry env use $(which python)**
```

# ****用诗歌工作****

****使用两个文件的诗歌功能:`pyproject.toml`和`poetry.lock`。****

****在`pyproject.toml`中，我们描述所需的依赖/包，例如`torch`。开发人员在添加了所有需要的依赖项之后，运行`poetry update`。****

****这触发了`poetry.lock`文件的创建，该文件“捕获”并描述了由来自`pyproject.toml`的所有数据包组成的确切环境。现在，当用户或其他开发人员下载 repo /代码时，他们只需运行`poetry install`，这将安装所有的依赖项——导致他们获得与第一个开发人员预期的完全相同的 Python 环境。****

****这是一个巨大的优势，也是必须的——与每个人只安装自己的依赖项相反——不这样做只会带来太多的麻烦。****

****要使用您的环境(运行脚本，…)，运行`poetry shell`。这将激活您的环境，在其中您可以照常工作——例如通过`python file.py`运行 Python 程序。****

# ****示例项目****

****最后，让我们将所有这些放在一个示例项目中——并且第一次公开一个示例`pyproject.toml`文件的内容。****

****我们的项目将由三个文件组成:`main.py`、`pyproject.toml`和自动生成的`poetry.lock`文件。****

****`main.py`有以下内容:****

```
**import matplotlib.pyplot as plt
import numpy as np

def plot():
    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    plt.plot(x, y)
    plt.savefig("plot.png")

if __name__ == "__main__":
    plot()**
```

****正如我们所看到的，我们使用`numpy`和`matplotlib`生成一条正弦曲线并绘制出来。因此需要安装这些模块，我们对诗歌就是这么做的。****

****让我们来看看`pyproject.toml`的内容:****

****第一部分包含一些关于项目的元数据:****

```
**[tool.poetry]
name = "myproject"
version = "0.1.0"
description = "…"
authors = ["hermanmichaels <hrmnmichaels@gmail.com>"]**
```

****在此之后，是时候定义所需的依赖关系了。您也可以(建议)为每个包定义一个特定的版本，使`poetry update`过程具有确定性。此外，这里我们还定义了预期的 Python 版本。下面的代码片段安装了`matplotlib`和`numpy`，以及其他一些我为了方便而喜欢使用的包(在以后的帖子中会有更多的介绍！):****

```
**[tool.poetry.dependencies]
python = "3.10"
matplotlib = "3.5.1"
mypy = "0.910"
numpy = "1.22.3"
black = "22.3.0"**
```

****综合起来看:****

```
**[tool.poetry]
name = "myproject"
version = "0.1.0"
description = "…"
authors = ["hermanmichaels <hrmnmichaels@gmail.com>"]

[tool.poetry.dependencies]
python = "3.10"
matplotlib = "3.5.1"
mypy = "0.910"
numpy = "1.22.3"
black = "22.3.0"**
```

****如上所述，为了生成`poetry.lock`文件，我们运行`poetry update`。然后，另一个使用这个项目的人，可以简单地运行`poetry install`来获得我们所有的依赖项。然后，您或他们可以通过运行以下命令来执行该程序:****

****`poetry shell`****

****`python main.py`****

# ****摘要****

****让我们快速总结一下:诗歌是一个(在我看来是最好的)打包和依赖管理器。它建立在一个工作的 Python 安装之上，我建议通过 Python 安装管理器 pyenv 来管理它。****

****按照上面的安装和设置步骤，创建一个新项目，并添加一个`pyproject.toml file`。在此，定义所有需要的依赖项。****

****然后，运行:****

*   ****`poetry update`生成锁文件(或在添加了新的依赖项时更新它)，其他人将使用该文件来获取相同的数据包集****
*   ****`poetry install`安装来自锁文件的所有包(例如，当你下载了一个新的诗歌项目并想安装依赖项时——或者你的同事推了一个包含新包的新版本)****
*   ****`poetry shell`进入环境并运行任何 Python 脚本****

****这就结束了对诗歌的介绍。希望这对您未来的工作有所帮助——欢迎随时回来获取更多信息！****