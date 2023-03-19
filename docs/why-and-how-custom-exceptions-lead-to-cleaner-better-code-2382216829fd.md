# 自定义异常为什么以及如何产生更干净、更好的代码

> 原文：<https://towardsdatascience.com/why-and-how-custom-exceptions-lead-to-cleaner-better-code-2382216829fd>

## 通过创建您自己的自定义异常来清理您的代码

![](img/102c7103cfe904d3965ce6ad790a9422.png)

(图片由 [Unsplash](https://unsplash.com/photos/52jRtc2S_VE) 上的[莎拉·基利安](https://unsplash.com/@rojekilian)拍摄)

异常和错误之间有很大的区别；异常是故意的，而错误是意外的。这篇文章关注于创建你自己的异常，以及如何使用它们来清理你的代码，从而产生一个不容易出错的程序，一个更好的流程和更快的调试。我们来编码吧！

# 肮脏的方式

为了演示自定义异常是如何工作的，我们将想象我们正在构建一个网站。我们重点关注**改进**网站的**登录功能**，目前看起来是这样的:

*(userexists、redirect 和 show_popup 是在我们项目的其他地方定义的函数；他们做什么和如何工作与本文无关)*

这是一个相当简单的功能:

1.  我们传递电子邮件和密码
2.  检查是否存在与该电子邮件地址相关联的帐户。如果没有，显示弹出窗口并重定向到/注册页面
3.  检查凭据是否有效。有效:显示弹出窗口。无效:显示弹出窗口和重定向

您可以像这样调用函数:

```
login_dirty(email='mike@bmail.com', password='my_pa$$word')
```

[](/a-complete-guide-to-using-environment-variables-and-files-with-docker-and-compose-4549c21dc6af)  

## 现在的登录功能有什么问题？

登录功能的问题是**脏**。我跟这个的意思是，它对不该负责的事情负责。我通常把我的项目分成两部分:

1.  干净的代码
2.  肮脏的代码

对我来说，区别在于干净的代码不知道你的业务逻辑。你可以从一个项目中提取“干净”的功能，并在完全不同的项目中使用它们。然而，脏代码包含业务逻辑；例如，当您无法登录时会发生什么。你被重定向了吗？是否显示弹出窗口？**让我们通过删除带有自定义异常的所有业务逻辑，使我们的登录功能变得更好、更干净。**

[](/cython-for-data-science-6-steps-to-make-this-pandas-dataframe-operation-over-100x-faster-1dadd905a00b)  

# 干净的方式

我们想要一个没有任何业务逻辑负担的登录功能。如果它不能登录一个用户，我们只需要发出信号，这是不可能的；我们代码的其他部分(脏的部分)可以决定如何处理它。

当你看我们函数时，有两件事可能出错:

1.  电子邮件地址未知(即用户未注册)
2.  电子邮件-密码组合不正确

这两件事就是阻止成功登录的**异常**:首先我们将编写一些代码，允许**引发这些异常**。然后我们将清理我们的函数和函数被调用的方式。

[](/why-is-python-so-slow-and-how-to-speed-it-up-485b5a84154e)  

## 创建自定义例外

创建自定义异常比您想象的要容易:

正如你看到的，我们创建了一个新类，它继承了异常类。然后，为了给其他开发人员一个好消息，我们只重写 __str__ 方法。

我们也可以像在`UserNotFoundException`中一样将参数传递给异常。这里我们做的事情和以前完全一样，只是我们需要使用 __init__ 方法将`email`存储为一个属性。

[](/image-analysis-for-beginners-destroying-duck-hunt-with-opencv-e19a27fd8b6)  

## 2.清理登录功能

使用新的异常来看看这个函数看起来有多漂亮和干净:

除了看起来更好之外，功能更干净和纯粹；它只负责登录，如果不知道任何关于重定向和弹出窗口。这种逻辑应该被限制在项目中的几个地方，而不应该到处都是。自定义异常对此有很大帮助。

[](/docker-for-absolute-beginners-what-is-docker-and-how-to-use-it-examples-3d3b11efd830)  

## 3.调用我们的登录

在我们的 main.py 文件中，我们现在可以像这样调用登录函数:

正如你所看到的，当我们无法登录时会发生什么是非常清楚的。主要的好处是，如果您以后决定无效的凭据也应该被重定向；没有很多地方可以搜索，因为您的业务逻辑并没有遍布整个项目。

[](/virtual-environments-for-absolute-beginners-what-is-it-and-how-to-create-one-examples-a48da8982d4b)  

# 结论

在本文中，我希望展示为什么要使用定制异常:为了帮助保持对代码的业务逻辑的概述，它们有助于保持您的函数纯净/干净和可调试。您自己的异常很容易在整个项目中创建和使用。

我希望一切都像我希望的那样清楚，但如果不是这样，请让我知道我能做些什么来进一步澄清。同时，看看我的其他关于各种编程相关主题的文章，比如:

*   [Python 为什么这么慢**如何加速**](https://mikehuls.medium.com/why-is-python-so-slow-and-how-to-speed-it-up-485b5a84154e)
*   [找出**巨蟒装饰者**如何在 6 个关卡中工作](https://mikehuls.medium.com/six-levels-of-python-decorators-1f12c9067b23)
*   [创建并发布自己的 **Python 包**](https://mikehuls.medium.com/create-and-publish-your-own-python-package-ea45bee41cdc)
*   [**Docker** 适合绝对初学者——Docker 是什么，怎么用(+举例)](https://mikehuls.medium.com/docker-for-absolute-beginners-what-is-docker-and-how-to-use-it-examples-3d3b11efd830)

编码快乐！

—迈克

喜欢我正在做的事情吗？ [*跟我来！*](https://mikehuls.medium.com/membership)

[](https://mikehuls.medium.com/membership) 