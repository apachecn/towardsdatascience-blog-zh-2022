# 理解 Python 装饰者:从初学者到专家的六个层次的装饰者

> 原文：<https://towardsdatascience.com/six-levels-of-python-decorators-1f12c9067b23>

## 装饰者如何工作，何时使用它们，以及 6 个越来越复杂的例子

![](img/61215e1ae77448241a28316697d7e3de.png)

这段代码已经装饰得很漂亮了(图片由[复杂的探险家](https://unsplash.com/@intricateexplorer)在 [Unsplash](https://unsplash.com/photos/H0-3xfbU8wk) 上提供)

装饰器是一个非常方便的工具，可以用来改变函数的行为，而不用修改函数本身。它们允许您在不更改现有代码的情况下轻松添加功能。有许多用例(其中一些我们将在今天讨论),如日志记录、性能检查、验证权限等。

读完这篇文章后，你会对装饰者如何工作、如何应用以及何时应用有一个清晰的了解。我们将从最简单、最基本的例子开始，然后慢慢地向更复杂的例子发展。最后我们会有一个函数，用不同类型的多个装饰器来装饰。我们来编码吧！

# 装修工如何工作

首先，我们将指定一个我们想要装饰的函数:

正如你所看到的，这个函数叫做‘say hello ’,我们可以通过运行`sayhello()`来执行它(注意函数名后面的()。这和`print(sayhello)`不一样；这将返回函数名和存储函数的内存地址:
`<function sayhello at 0x000001F64CA25A60>`

下一步是创建一个装饰器:

正如你所看到的，我们创建了一个期望另一个函数的函数。然后，我们定义一个名为 wrapper 的内部函数，在其中执行接收到的函数。然后我们返回内部函数。定义内部函数的原因是为了处理参数，我们将在下一章看到这一点。

我们可以像这样使用这个装饰器`our_decorator(sayhello())`或者添加一点语法糖，然后像这样传递它:

这两种方法是等价的:我们将`sayhello`包装在`our_decorator`中。

# 复杂 6 步中的装饰者

现在我们已经了解了 decorator 及其语法的工作方式，我们将通过 decorator 的 6 个步骤向您展示它们提供的所有可能性和灵活性。

![](img/c5dd338edd2d84ab4ff08738ea54e54d.png)

让我们开始装饰吧(图片由[张家瑜](https://unsplash.com/@danielkcheung)于 [Unsplash](https://unsplash.com/photos/ZqqlOZyGG7g) 拍摄)

## 1.最简单的装饰

首先，我们创建一个装饰器的最基本的例子。我们创建了一个名为`logging_decorator`的函数，它将用于修饰其他函数。首先，我们检查代码，然后浏览:

如你所见，我们正在重新使用旧的`sayhello`功能。我们用一个叫做`logging_decorator`的函数来修饰它。正如您在第 3–5 行中看到的，我们可以在实际执行`sayhello`之前和之后做一些事情，所以让我们打印出我们调用这个函数用于调试的事实。

还要注意，我们可以将函数作为对象来访问。这样我们可以在第 3 行和第 5 行调用函数的名字。执行下面的代码会打印出以下内容:

```
[LOG]   calling sayhello
 === Hello from the function function
[LOG] called sayhello
```

注意，我们的函数非常简单；它甚至不返回任何东西。在下一部分，我们将增加一点复杂性。

## 2.传递参数和返回值

在前一部分中，我们的装饰器只是执行包装的函数。我们将升级代码，以便函数接收参数并实际返回一些内容。这就是名为`wrapper`的内部函数的用武之地。

在上面的代码中，我们修饰了`multiply`函数。这个函数需要两个参数。我们在第 2 行用*args 和**kwargs 修改了 decorator 函数内部的包装器。然后我们将它们传递给第 4 行的修饰函数。在同一行中，我们还接收修饰函数的结果，然后在第 6 行返回。这些微小的变化使得我们可以从我们的目标函数中得到一个结果:

```
[LOG]   calling multiply
  === Inside the multiply function
 [LOG] called multiply
[result]   20
```

## 3.多个装饰者

也可以用多个装饰器来装饰一个函数。

上面的代码首先将`multiply`包装在`logging_decorator_2`中，然后将整个串包装在`logging_decorator_1`中。它从目标函数开始，通过所有装饰器向上移动。您可以在下面的输出中看到装饰器被调用的顺序:

```
[LOG1]   calling decorator 1
[LOG2]   calling decorator 2
 multiply function
[LOG2] called decorator 2
[LOG1] called decorator 1
[result]   42
```

用我们的两个装饰器调用目标函数非常类似于下面的代码行。两者是等价的，但是在下面一行中，类似包装器的属性非常清楚:
`logging_decorator_1(logging_decorator_2(multiply(14,3)))`

## 4.将参数传递给装饰者

在这一部分中，我们将稍微修改一下日志装饰器，以便它可以接收参数。我们希望收到一个指示调试模式是打开还是关闭的参数。如果它打开，我们就打印出测井曲线，否则就不打印。

变化在第 1 行；这里我们允许装饰者接受一个变量。然后在包装器中，我们可以访问这个变量，并决定是否打印出这一行。我们可以在第 13 行确定调试模式是开还是关。

## 5 个有状态的装饰类

在这一部分中，我们将创建一个装饰器来计算目标函数被调用的次数。为了做到这一点，装饰器需要保存一些状态:它必须记住函数被调用的次数:

正如你看到的，我们用目标函数初始化了这个类。然后，当目标函数被调用时，我们增加 call_count。这是如何使用这个装饰器:

执行此操作将产生以下结果:

```
Called multiply for the 1th time
Called multiply for the 2th time
Called multiply for the 3th time
Called multiply for the 4th time
res: (42, 70, 54, 8)
```

跟踪一个函数被调用的频率对于限制 API 调用非常方便

## 6.装饰参数和函数参数之间的交互

在最后一步中，我们将创建一个装饰器，它可以用密码保护一个函数。在装饰参数中，我们将定义密码。我们将使用通过目标函数提供的密码对此进行检查。类似的装饰器通常用于为 API 路由提供安全性，例如

请记住，下面的例子只是一个例子；这不是一个安全代码的好例子。查看 [**这篇文章**](https://mikehuls.medium.com/keep-your-code-secure-by-using-environment-variables-and-env-files-4688a70ea286) 了解更多关于如何在代码中安全存储机密信息的信息。

如您所见，我们在第 14 行的装饰器中提供了我们的有效密码。然后，在装饰器中，我们从第 5 行的 args 或 kwargs 中获取通过目标函数提供的密码。然后，在第 6 行，我们检查提供的密码是否正确。执行上面的代码会产生下面的输出:

```
You can only see this if you provided the right answer
Incorrect password
```

![](img/37eb0614d9d269425bfaebf6572a07bd.png)

装饰者允许你创造真正的杰作(图片由[伊戈尔·米斯克](https://unsplash.com/@igormiske)在 [Unsplash](https://unsplash.com/photos/oLhTLD-RBsc) 上提供)

# 结论

在这篇文章中，我试图解释为什么以及何时需要使用装饰器。然后我们看了 6 个越来越复杂的例子，我希望用它们来展示装饰者可以带来的灵活性。

如果你有建议/澄清，请评论，以便我可以改进这篇文章。同时，看看我的[关于各种编程相关话题的其他文章](https://mikehuls.medium.com/)，比如:

*   [Python 为什么慢，如何加速](https://mikehuls.medium.com/why-is-python-so-slow-and-how-to-speed-it-up-485b5a84154e)
*   [Python 中的高级多任务处理:应用线程池和进程池并进行基准测试](https://mikehuls.medium.com/advanced-multi-tasking-in-python-applying-and-benchmarking-threadpools-and-processpools-90452e0f7d40)
*   [编写自己的 C 扩展来加速 Python x100](https://mikehuls.medium.com/write-your-own-c-extension-to-speed-up-python-x100-626bb9d166e7)
*   【Cython 入门:如何在 Python 中执行>每秒 17 亿次计算
*   [用 FastAPI 用 5 行代码创建一个快速自动归档、可维护且易于使用的 Python API](https://mikehuls.medium.com/create-a-fast-auto-documented-maintainable-and-easy-to-use-python-api-in-5-lines-of-code-with-4e574c00f70e)
*   [创建并发布你自己的 Python 包](https://mikehuls.medium.com/create-and-publish-your-own-python-package-ea45bee41cdc)
*   [创建您的定制私有 Python 包，您可以从您的 Git 库 PIP 安装该包](https://mikehuls.medium.com/create-your-custom-python-package-that-you-can-pip-install-from-your-git-repository-f90465867893)
*   [完全初学者的虚拟环境——什么是虚拟环境，如何创建虚拟环境(+示例)](https://mikehuls.medium.com/virtual-environments-for-absolute-beginners-what-is-it-and-how-to-create-one-examples-a48da8982d4b)
*   [通过简单的升级大大提高您的数据库插入速度](https://mikehuls.medium.com/dramatically-improve-your-database-inserts-with-a-simple-upgrade-6dfa672f1424)

编码快乐！

—迈克

页（page 的缩写）学生:比如我正在做的事情？跟我来！

<https://mikehuls.medium.com/membership> 