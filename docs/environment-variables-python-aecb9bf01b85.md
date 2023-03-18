# 在 Python 中与环境变量交互

> 原文：<https://towardsdatascience.com/environment-variables-python-aecb9bf01b85>

## 使用 Python 访问、导出和取消设置环境变量

![](img/8a556474ca5ba5df54a25bb485e2d9a1.png)

照片由 [Clarisse Meyer](https://unsplash.com/@clarissemeyer?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/access?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

## 介绍

某些应用程序可能不得不使用已经在程序本身之外初始化的变量，而是在源代码应该被执行的环境中。

环境变量被指定为执行流程(如 Python 应用程序)的环境的一部分。它由一个名称/值对组成，可以在任何给定时间被访问、覆盖和取消设置。这种变量通常直接在命令行界面上定义，或者用 bash 脚本定义(例如，在操作系统启动时)。然而，甚至软件程序本身也可以与它们进行交互。

在今天的简短教程中，我们将展示**如何以编程方式访问覆盖和取消设置现有的环境变量，以及如何导出新的环境变量**。最后，我们还将演示几种检查环境变量是否存在的方法。

## 访问环境变量

首先，我们将演示如何以编程方式访问已经作为执行 Python 应用程序的环境的一部分导出的环境变量。

假设系统管理员已经用值`dev`初始化了一个名为`ENV`的环境变量:

```
$ export ENV=dev
```

我们可以通过命令行回显该值来验证环境变量是否已经初始化:

```
$ echo ENV
dev
```

现在，如果我们想用 Python 编程访问环境变量，我们需要使用`**os.environ**`映射对象:

> 一个[映射](https://docs.python.org/3/glossary.html#term-mapping)对象，其中键和值是表示流程环境的字符串。例如，`environ['HOME']`是你的主目录的路径名(在某些平台上)，相当于 c 中的`getenv("HOME")`
> 
> 这个映射是在第一次导入`[os](https://docs.python.org/3/library/os.html#module-os)`模块时捕获的，通常是在 Python 启动时作为处理`site.py`的一部分。除了通过直接修改`os.environ`所做的更改外，在此时间之后对环境所做的更改不会反映在`os.environ`中。
> 
> - [Python 文档](https://docs.python.org/3/library/os.html#os.environ)

你可以使用环境变量名作为`os.environ`对象的键来推断值:

```
import os**env_var = os.environ['ENV']**
print(f'Currently working in {env_var} environment'.
```

上面表达式的问题是，如果环境变量`ENV`不在环境中，它将会以`KeyError`失败。因此，最好使用`os.environ.get()`来访问它。如果不存在，它会简单地返回`None`而不是抛出一个`KeyError`。

```
import os**env_var =** **os.environ.get('****ENV****')**
print(f'Currently working in {env_var} environment'.
```

接下来，如果环境变量尚未初始化，您甚至可能需要设置一个默认值:

```
import os**env_var =** **os.environ.get('****ENV****', 'DEFAULT_VALUE')**
print(f'Currently working in {env_var} environment'.
```

显然，另一种选择是捕捉`KeyError`，但是我认为这对于这种操作来说可能是一种过度的破坏:

```
import ostry:
    env_var = os.environ['ENV']
except KeyError:
    # Do something
    ...
```

## 导出或覆盖环境变量

如果您想要导出或者甚至覆盖一个现有的环境变量，那么您可以在`os.environ`对象上使用一个简单的赋值:

```
import osos.environ['ENV'] = 'dev'
```

现在使用`try-except`符号可能更有意义:

```
import ostry:
    env_var = os.environ['ENV']
except KeyError:
    os.environ['ENV'] = 'dev'
```

## 检查环境变量是否存在

我们已经部分介绍了这一部分，但是我还将演示一些检查环境变量是否存在的其他方法。

第一种方法是简单地使用变量名作为关键字访问`os.environ`对象，并捕获表示环境变量不存在的`KeyError`:

```
import ostry:
    env_var = os.environ['ENV']
    print('ENV environment variable exists')
except KeyError:
    print('ENV environment variable does not exist')
```

另一种方法是简单地检查环境变量是否是`os.environ`对象的成员:

```
import os env_var_exists = 'ENV' in os.environ
```

第三种方法是检查`os.environ.get()`方法是否返回`None`(没有指定默认值):

```
import osenv_var = os.environ.get('ENV')if not env_var:
    print('ENV environment variable does not exist')
```

最后，您甚至可以使用`has_key()`方法来检查环境变量名是否作为关键字包含在`os.environ`映射对象中:

```
import osenv_var_exists = os.environ.has_key('ENV')
```

## 取消设置环境变量

现在为了取消设置环境变量，您可以调用`del`操作:

```
del os.environ['ENV']
```

同样，如果`ENV`没有初始化，上面的表达式将失败，因此它从`os.environ`映射对象中丢失。

为了避免这种情况，您可以使用 if 语句

```
import osif 'ENV' in os.environ:
    del os.environ['ENV']
```

或者，一种更优雅的方式是从映射对象中调用`pop`环境变量，如下所示。

```
os.environ.pop('ENV', None)
```

## 最后的想法

在今天的文章中，我们讨论了环境变量的重要性，以及它们如何与正在运行的 Python 应用程序进行交互。更具体地说，我们展示了如何用 Python 编程访问、取消设置甚至导出环境变量。

同样重要的是要提到，环境变量通常不会只被您自己的 Python 应用程序使用，因此您要确保以一种不会使环境处于可能中断其他应用程序甚至更糟的状态的方式与它们交互，从而错误地影响其他进程的执行流程。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读媒介上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership) [## 通过我的推荐链接加入 Medium-Giorgos Myrianthous

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

gmyrianthous.medium.com](https://gmyrianthous.medium.com/membership) 

**相关文章你可能也喜欢**

[](/16-must-know-bash-commands-for-data-scientists-d8263e990e0e) [## 数据科学家必须知道的 16 个 Bash 命令

### 探索一些最常用的 bash 命令

towardsdatascience.com](/16-must-know-bash-commands-for-data-scientists-d8263e990e0e) [](/setuptools-python-571e7d5500f2) [## Python 中的 setup.py 与 setup.cfg

### 使用 setuptools 管理依赖项和分发 Python 包

towardsdatascience.com](/setuptools-python-571e7d5500f2) [](https://betterprogramming.pub/kafka-cli-commands-1a135a4ae1bd) [## 用于日常编程的 15 个 Kafka CLI 命令

### 演示最常用的 Kafka 命令行界面命令的使用

better 编程. pub](https://betterprogramming.pub/kafka-cli-commands-1a135a4ae1bd)