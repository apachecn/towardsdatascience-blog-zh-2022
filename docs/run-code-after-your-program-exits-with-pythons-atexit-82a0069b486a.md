# 用 Python 的 AtExit 在程序退出后运行代码

> 原文：<https://towardsdatascience.com/run-code-after-your-program-exits-with-pythons-atexit-82a0069b486a>

## 注册脚本结束或出错后运行的清理函数

![](img/53134bedccbc90aa1647070e3df7e00f.png)

(图片由 [Andrew Teoh](https://unsplash.com/@theandrewteoh) 在 [Unsplash](https://unsplash.com/photos/SKrgZQgYy2g) 上拍摄)

在本文中，我将向您展示如何使用 python 的内置模块 atexit 注册程序退出时执行的函数。这意味着您可以在代码退出后运行代码*(由于完成时的错误)，为您提供各种必要的工具来执行清理功能，保存数据，并使调试更容易。我们来编码吧！*

# 1.用简单的例子理解 atexit

主要概念是注册一个在脚本退出时执行的函数。发生这种情况的原因有很多，我们将在后面讨论。让我们从最简单的例子开始，逐渐让它变得更有用。

## 最简单的例子——注册函数

下面的代码演示了注册出口函数的最基本方法。

在上面的代码中，我们简单地注册了`say_goodbye`函数，然后调用`simple_example`函数。当脚本在`simple_example()`完成后结束时，你会看到`say_goodbye`执行了。这也可以在输出中看到:

```
calling the function..
Function enter
Function exit
exiting now, goodbye
```

轻松点。让我们把它变得复杂一点。

[](/why-is-python-so-slow-and-how-to-speed-it-up-485b5a84154e)  

## 简单的例子——退出循环

我们稍微调整了一下`simple_example` 函数:它现在包含了一个**无限循环**，显示“仍在运行”每一秒钟。我们再次注册了`say_goodbye`函数，只是现在是在`simple_example`函数的开始。

现在，当我们运行代码时，我们会看到“仍在运行..”每秒发送一条消息。当我们通过按 ctrl-c 退出**时，我们看到以下输出:**

```
still running..
still running..
still running..
still running..
still running..
exiting now, goodbye
```

所以在进入无限循环之前，我们注册了`say_goodbye`函数。该函数将在解释器终止时执行。

[](https://medium.com/geekculture/applying-python-multiprocessing-in-2-lines-of-code-3ced521bac8f)  

# 2.更简单的多次注册

在前一部分中，我们已经看到了放置`atexit.register`位代码的位置非常重要。如果我们在 while 循环之后注册函数，那么这个函数将永远不会被注册。

在这一部分中，我们将探索一种注册函数的新方法，这样您就不必担心这个问题。我们还发现你可以注册多个函数

我们可以退出所有我们想退出的，在一个 exit 中注册多个函数是可能的。还有，注册可以简单一点。

## 更简单的注册装饰

通过用 [**装饰器**](https://mikehuls.medium.com/six-levels-of-python-decorators-1f12c9067b23) 注册函数，你不必担心*何时注册函数，因为一旦解释器到达你的函数定义，它们就会发生。它们也非常容易使用:*

另一个好处是你的代码更干净；`simpel_example()`函数没有被 atexit 注册“污染”。如果您想了解更多关于装饰器如何工作以及如何创建自己的装饰器，请查看本文。

[](/how-to-make-a-database-connection-in-python-for-absolute-beginners-e2bfd8e4e52)  

## 注册多个功能

很容易**注册多个函数**以在退出时执行。重要的是要记住它们是按照注册的相反顺序执行的。查看下面的代码:

您可以通过下面的输出看到订单:

```
still running..
still running..
# 1 says au revoir
# 2 says later
# 3 says goodbye
```

[](/understanding-python-context-managers-for-absolute-beginners-4873b6249f16)  

## 注销函数

这就像注册一个函数一样简单:

上面的代码将产生以下输出(注意“# 2”没有显示):

```
Function enter
Function exit
# 1
```

您也可以使用`atexit._clear()`清除所有已注册的功能。

[](/cython-for-absolute-beginners-30x-faster-code-in-two-simple-steps-bbb6c10d06ad)  

## 注册带有参数的函数

下面的代码允许您**向注册的出口函数**传递参数。这对于向您事先不知道的注册函数传递数据非常有用:

上面的代码产生以下输出:

```
what is your name? mike
Function enter
Function exit
 -mike says hoi after 0:00:01.755111
 -mike says hello after 0:00:01.755111
 -mike says bonjour after 0:00:01.755111
 -mike says gutentag after 0:00:01.755111
```

[](/docker-for-absolute-beginners-the-difference-between-an-image-and-a-container-7e07d4c0c01d)  

## 手动触发所有注册的出口功能

下面的代码演示了如何通过`atexit._run_exitfuncs()`功能**手动触发所有已注册的退出功能**:

输出如下所示:

```
start
exiting now, au revoir
exiting now, later
exiting now, goodbye
continue with code
```

[](/image-analysis-for-beginners-creating-a-motion-detector-with-opencv-4ca6faba4b42)  

# atexit 什么时候运行注册的函数？

好问题。**在正常解释器终止时执行退出功能**。这意味着它们在你的脚本正常结束时运行，当`sys.exit(0)` `quit()`或引发类似`raise ValueError("oops")`的异常时运行。

异常终止是指程序被非 Python 处理的信号**杀死**或检测到 Python **致命内部错误**的情况。最后一种情况是调用`os._exit(0)`时；这将在不调用清理处理程序、刷新 stdio 缓冲区等的情况下退出。举个例子:

将产生:

```
start
exiting now, goodbye
```

用`os._exit(0)`替换`sys.exit(0)`的方向盘不会运行`say_goodbye`功能。

[](/create-a-fast-auto-documented-maintainable-and-easy-to-use-python-api-in-5-lines-of-code-with-4e574c00f70e)  

# 结论

在本文中，我们探索了 Python 的 atexit 函数的内部工作方式，以及如何和何时使用它。

我希望这篇文章像我希望的那样清楚，但如果不是这样，请让我知道我能做些什么来进一步澄清。同时，看看我的其他关于各种编程相关主题的文章，比如:

*   [Git 绝对初学者:借助视频游戏理解 Git](https://mikehuls.medium.com/git-for-absolute-beginners-understanding-git-with-the-help-of-a-video-game-88826054459a)
*   [创建并发布自己的 Python 包](https://mikehuls.medium.com/create-and-publish-your-own-python-package-ea45bee41cdc)
*   [使用 Docker 和 Compose 环境变量和文件的完整指南](https://mikehuls.medium.com/a-complete-guide-to-using-environment-variables-and-files-with-docker-and-compose-4549c21dc6af)

编码快乐！

—迈克

*又及:喜欢我正在做的事吗？* [*跟我来！*](https://mikehuls.medium.com/membership)

[](https://mikehuls.medium.com/membership) 