# 绝对初学者理解 Python 上下文管理器

> 原文：<https://towardsdatascience.com/understanding-python-context-managers-for-absolute-beginners-4873b6249f16>

## 理解关于光剑的 WITH 语句

![](img/f75f346c6d8091a34ec226c1836b3cf8.png)

我们将编写自己的光剑(图片由 [Venti Views](https://unsplash.com/@ventiviews) 在 [Unsplash](https://unsplash.com/photos/35uZM_4wjYg) 上提供)

您肯定熟悉上下文管理器——他们使用`with`语句:

```
with open('somefile.text', 'r') as file:
   data = file.read()
```

在本文中，我们将关注调用上下文管理器时会发生什么:

*   什么是上下文管理器？
*   它是如何工作的？
*   它的优点是什么？
*   创建您自己的上下文管理器

我们来编码吧！

在我们使用上下文管理器进行一些真实的代码示例之前，我们将通过一个清晰的示例来理解其内部工作原理:想象我们正在编写一把光剑。

[](/six-levels-of-python-decorators-1f12c9067b23)  

## 设置:编写光剑代码

我敢肯定每个人都很熟悉这些:超级危险的，赤热的等离子刀片，它可以切开它碰到的一切。在我们用它们刺或砍之前，我们需要激活它，在我们用完后，我们应该总是关闭这个危险的设备。在代码中:

没什么特别的:我们跟踪两个属性:`color`和`active`。在这个例子中`color`是非常装饰性的，`active`是非常重要的，因为如果我们激活了军刀，我们只能使用`slash()`和`stab()`方法。我们有两个控制状态的函数:`turn_on()`和`turn_off()`。

您使用这段代码的方式如下:

这段代码的维护成本相当高；我们不能忘记打开我们的光剑，否则它会抛出一个异常。而且我们每次都需要调用`turn_off()`方法。我们怎样才能让这段代码更容易使用呢？

[](/multi-tasking-in-python-speed-up-your-program-10x-by-executing-things-simultaneously-4b4fc7ee71e)  

## 何时使用上下文管理器？

以我们的光剑为例，我们总是需要打开光剑，然后才能用它做任何事情。另外，我们 ***总是*** 需要关掉它，因为否则我们就不能用它，而且现在能源价格相当高。

记住，为了打开和关闭 saber，我们需要调用`turn_on()`和`turn_off()`方法。对于上下文管理器来说，光剑是一个理想的用例，因为我们需要在每次使用它之前和/或之后执行一个动作。

## 用上下文管理器更新光剑

我们将在我们的光剑类中增加两个特殊的“dunder-methods”。Dunder-methods(双下划线-methods)覆盖定制类的内置函数的功能。你可能已经知道一种方法:T2 方法！

我们将添加一个`__enter__`和`__exit__`方法，所以我们的光剑类现在看起来像这样:

正如你将看到的，打开和关闭 saber 的函数在这些 enter 和 exit 方法中被调用。我们现在可以这样使用这个类:

一旦我们点击`with`语句，我们就进入`__enter__`方法，这将打开我们的光剑。然后我们就可以尽情挥刀捅人了。一旦我们退出代码块，就会调用`__exit__`方法关闭我们的光剑。

优势显而易见:

*   我们的代码更加简洁。
*   我们的代码更容易阅读。
*   当我们想开始砍和刺的时候它会自动开启。
*   当我们完成谋杀时，它会自动关闭，防止许多人试图将它藏起来而受伤。

[](/simple-trick-to-work-with-relative-paths-in-python-c072cdc9acb9)  

## 野外的上下文管理器

光剑有点像玩具，但是在很多情况下会用到上下文管理器。下面的例子详细说明了 Python 读取文件的内置函数。默认方式如下所示:

```
file = open('somefile.txt')
content = file.read()
file.close()
```

如果我们在上面的例子中不调用`file.close()`，那么我们的文件将被我们的脚本占用。这意味着没有其他进程可以访问它。让我们使用上下文管理器:

```
with open('somefile.txt') as file:
    content = file.read()
```

上面的代码会自动关闭文件。它也更容易阅读。

另一个例子是数据库连接。在我们创建了到数据库的连接并使用它来读取/写入数据到数据库之后，我们不能忘记提交数据并关闭连接。常规实施:

```
cursor = conn.cursor()
cursor.execute("SELECT * FROM sometable")
cursor.close()
```

幸运的是，许多包都实现了这样的上下文管理器:

```
with dbconnection.connect() as con:
    result = con.execute("SELECT * FROM sometable")
```

上面的代码确保连接在我们完成后立即关闭。目前，我正在创建一个将 Python 连接到数据库的指南；有兴趣就跟着我吧！

[](/create-a-fast-auto-documented-maintainable-and-easy-to-use-python-api-in-5-lines-of-code-with-4e574c00f70e)  

# 结论

在本文中，我们已经介绍了 Python 中日志记录的基础知识，我希望向您展示日志记录相对于打印的所有优势。下一步是添加文件处理程序，这样我们就可以将日志保存到文件中，通过电子邮件或 API 发送。请查看这篇文章。

如果你有任何改进这篇文章的建议；请评论！与此同时，请查看我的其他关于各种编程相关主题的文章，比如:

*   [面向绝对初学者的 cyt hon——两步代码速度提高 30 倍](https://mikehuls.medium.com/cython-for-absolute-beginners-30x-faster-code-in-two-simple-steps-bbb6c10d06ad)
*   [Python 为什么这么慢，如何加速](https://mikehuls.medium.com/why-is-python-so-slow-and-how-to-speed-it-up-485b5a84154e)
*   绝对初学者的 Git:借助视频游戏理解 Git
*   [在一行代码中显著提高数据库插入速度](https://mikehuls.medium.com/dramatically-improve-your-database-inserts-with-a-simple-upgrade-6dfa672f1424)
*   [Docker:图像和容器的区别](https://mikehuls.medium.com/docker-for-absolute-beginners-the-difference-between-an-image-and-a-container-7e07d4c0c01d)
*   [Docker 对于绝对初学者——什么是 Docker 以及如何使用它(+示例)](https://mikehuls.medium.com/docker-for-absolute-beginners-what-is-docker-and-how-to-use-it-examples-3d3b11efd830)
*   [绝对初学者的虚拟环境——什么是虚拟环境，如何创建虚拟环境(+示例](https://mikehuls.medium.com/virtual-environments-for-absolute-beginners-what-is-it-and-how-to-create-one-examples-a48da8982d4b))
*   [创建并发布你自己的 Python 包](https://mikehuls.medium.com/create-and-publish-your-own-python-package-ea45bee41cdc)
*   [6 个步骤让熊猫数据帧运算速度提高 100 倍](https://mikehuls.medium.com/cython-for-data-science-6-steps-to-make-this-pandas-dataframe-operation-over-100x-faster-1dadd905a00b)

编码快乐！

—迈克

附注:喜欢我正在做的事吗？ [*跟我来！*](https://mikehuls.medium.com/membership)

[](https://mikehuls.medium.com/membership) 