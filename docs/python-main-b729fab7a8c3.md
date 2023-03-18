# Python 中 if __name__ == "__main__ "是做什么的？

> 原文：<https://towardsdatascience.com/python-main-b729fab7a8c3>

## Python 中主方法的执行时间和方式

![](img/e14e1431fbe3f189674f6411991043b0.png)

由[布莱克·康纳利](https://unsplash.com/@blakeconnally)在[unsplash.com](https://unsplash.com/photos/B3l0g6HLxr8)拍摄的照片

如果您是 Python 的新手，您可能已经注意到，可以使用或不使用 main 方法来运行 Python 脚本。Python 中定义 one 的符号(即`if __name__ == ‘__main__'`)显然不是一目了然的，尤其是对于新手来说。

在今天的教程中，我们将探索和讨论 main 方法的用途，以及在 Python 应用程序中定义一个 main 方法时会遇到什么情况。

## `__name__`的目的是什么？

在执行程序之前，python 解释器将 Python 模块的名称赋给一个名为`__name__`的特殊变量。根据您是通过命令行执行程序还是将模块导入另一个模块，`__name__`的分配会有所不同。

例如，如果您将模块作为脚本调用

```
python my_module.py
```

然后 Python 解释器会自动将字符串`'__main__'`赋给特殊变量`__name__`。另一方面，如果您的模块被导入到另一个模块中

```
# Assume that this is another_module.py
import my_module
```

那么字符串`'my_module'`将被分配给`__name__`。

## main 方法是如何工作的？

现在让我们假设我们有以下模块，它包含以下代码行:

```
# first_module.py
print('Hello from first_module.py')

if __name__ == '__main__':
    print('Hello from main method of first_module.py')
```

所以在上面的模块中，我们有一个 print 语句在 main 方法之外，还有一个 print 语句在 main 方法之内。main 方法下的代码只有在模块作为脚本从命令行调用时才会被执行，如下所示:

```
python first_module.py
Hello from first_module.py
Hello from main method of first_module.py
```

现在，假设我们不是将模块`first_module`作为脚本调用，而是想将其导入另一个模块:

```
# second_module.py
import first_module

print('Hello from second_module.py')

if __name__ == '__main__':
    print('Hello from main method of second_module.py')
```

最后，我们调用`second_module`作为脚本:

```
python second_module.py
Hello from first_module.py
Hello from second_module.py
Hello from main method of second_module.py
```

注意，第一个输出来自模块`first_module`，特别是来自 main 方法之外的 print 语句。因为我们没有将`first_module`作为脚本调用，而是将其导入到了`second_module`中，first_module 中的 main 方法将被忽略，因为`if __name__ == ‘__main__'`的计算结果是`False`。回想一下，在上面的调用中，`second_module`的`__name__`变量被赋予了字符串`'__main__'`，而`first_module`的`__name__`变量被赋予了模块名，即`’first_module’`。

虽然`if __name__ == ‘__main__'`下的所有东西都被认为是我们所说的“main 方法”，但是定义一个适当的 main 方法是一个好的实践，如果条件评估为 True，就调用这个方法。举个例子，

```
# my_module.py
def main():
    """The main function of my Python Application"""
    print('Hello World')

if __name__ == '__main__': 
    main()
```

*注意:我通常不鼓励你在一个 Python 应用程序中使用多个主函数。为了举例，我使用了两种不同的主要方法。*

## 最后的想法

在本文中，我描述了如何在 Python 中执行 main 方法，以及在什么条件下执行。当一个模块作为一个字符串被调用时，Python 解释器会将字符串`'__main__'`赋给一个名为`__name__`的特殊变量，然后在条件`if __name__ == ‘__main__'`下定义的代码会被执行。另一方面，当一个模块被导入到另一个模块中时，Python 解释器会将带有该模块名称的字符串赋给特殊变量`__name__`。这意味着在这种情况下,`if __name__ == ‘__main__'`将评估为`False`,这意味着一旦导入，只有该条件之外的代码才会被执行。

[**成为**](https://gmyrianthous.medium.com/membership) **会员，阅读媒体上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership) [## 通过我的推荐链接加入 Medium-Giorgos Myrianthous

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

gmyrianthous.medium.com](https://gmyrianthous.medium.com/membership) 

**相关文章你可能也喜欢**

[](/args-kwargs-python-d9c71b220970) [## * Python 中的 args 和**kwargs

### 讨论位置参数和关键字参数之间的区别，以及如何在 Python 中使用*args 和**kwargs

towardsdatascience.com](/args-kwargs-python-d9c71b220970) [](/python-poetry-83f184ac9ed1) [## 用诗歌管理 Python 依赖关系

### 依赖性管理和用诗歌打包

towardsdatascience.com](/python-poetry-83f184ac9ed1) [](/pycache-python-991424aabad8) [## Python 中 __pycache__ 是什么？

### 了解运行 Python 代码时创建的 __pycache__ 文件夹

towardsdatascience.com](/pycache-python-991424aabad8)