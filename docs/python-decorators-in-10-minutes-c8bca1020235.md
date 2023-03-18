# 10 分钟后 Python Decorators

> 原文：<https://towardsdatascience.com/python-decorators-in-10-minutes-c8bca1020235>

## 装饰速成课程，包含流行的现实生活中的使用示例

你可能已经注意到在函数的正上方添加了`@<something>`的代码，比如在类方法的正上方添加了`@staticmethod`、`@classmethod`，这些实际上是 Python 的装饰器。Decorators 允许在不修改源代码的情况下扩展现有的函数。

在本文中，我们将通过理解装饰器的结构、探索高级行为(如克服装饰器的缺点、多嵌套装饰器、堆叠装饰器)以及一些实际应用来编写我们自己的 Python 装饰器。

***更新*** *:本文是系列文章的一部分。查看其他“10 分钟内”话题* [*此处*](https://medium.com/@kayjanwong/list/in-10-minutes-eeaa9aa67055) *！*

# 目录

*   [装饰器的结构](https://medium.com/p/c8bca1020235/#90bf)
*   [装修工的先进行为](https://medium.com/p/c8bca1020235/#4807)
*   [用途:测量执行时间](https://medium.com/p/c8bca1020235/#be80)
*   [用途:带测井调试](https://medium.com/p/c8bca1020235/#1664)
*   [用法:创建单例类](https://medium.com/p/c8bca1020235/#583f)

# 装饰者的结构

Decorators 可以被认为是一个函数包装器，这意味着它接受一个函数作为参数，并返回该函数的修改版本——向其添加扩展或功能。

```
def sample_decorator(func):
    def wrapper(*args, **kwargs):
        # do something before function execution
        result = func(*args, **kwargs)
        # do something after function execution
        return result
    return wrapper
```

从上面显示的结构中，我们可以看到函数执行发生在第 4 行，但是我们可以修改函数执行之前、期间甚至之后发生的事情。装饰者有可能改变函数的输入、输出或行为，但是最好以不降低它所包装的函数的可理解性的方式来实现它。

> 装饰者最适合用来向多个函数添加公共行为，而不用手动修改每个函数

# 装饰者的高级行为

## 保留包装函数的元数据

使用 decorator 的一个警告是，函数的元数据会被 decorator 隐藏。在上一节的代码片段中，我们返回了一个`wrapper`函数，而不是原始函数，这意味着任何修饰过的函数都将把它们的`__name__`元数据覆盖到`wrapper`。

```
@sample_decorator
def func_add(a, b):
    return a + b

print(func_add.__name__)
# wrapper
```

从技术角度来说，这不会影响函数或装饰器的预期执行，但这仍然是避免使用装饰器的任何意外后果的最佳实践。这可以通过为`wrapper`函数添加一个`@wraps`装饰器来轻松完成，如下所示。装饰器可以以同样的方式使用，但是包装函数的元数据现在不会被覆盖。

```
from functools import wraps

def sample_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # do something before function execution
        result = func(*args, **kwargs)
        # do something after function execution
        return result
    return wrapper
```

## 接受参数的装饰者

装饰器本质上是一个函数(包装另一个函数),应该能够接受参数。我发现传入一个布尔变量很有用，这样我就可以切换装饰器的行为，比如当我想进入或退出调试模式时打开或关闭打印。这可以通过用另一个函数包装器包装装饰器来实现。

在下面的例子中，我有一个装饰器`debug_decorator`，它可以接受参数并返回一个`decorator`装饰器，该装饰器将原始函数包装在`wrapper`中。由于多个嵌套函数，这一开始看起来相当复杂。建议首先编写原始的装饰器，最后包装它以接受参数。

```
from functools import wraps

debug_mode = True

def debug_decorator(debug_mode):
    """Example: Passing arguments to a decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if debug_mode:
                print(f"Function called: {func.__name__}")
            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@debug_decorator(debug_mode)
def func_add(a, b):
    return a + b

func_add(1, 2)
# Function called: func_add
# 3
```

## 堆叠装饰者

如前所述，decorators 允许扩展现有的功能。可以在一个函数上堆叠多个 decorators 来添加更多的扩展。执行的顺序将遵循装饰器的堆叠顺序。

需要注意的一点是，时间敏感的 decorators 如果是堆叠的，应该最后添加。比如，衡量一个函数执行时间的 decorator 应该是最后执行的，这样才能准确反映执行时间，不受其他 decorator 的影响。

现在我们已经理解了装饰器的结构和它的高级行为，我们可以深入到它们的实际应用中去了！

# 用法:测量执行时间

`timer` decorator 可以通过记录函数执行的开始时间和结束时间并将结果打印到控制台来测量包装函数的执行时间。

在下面的代码片段中，我们测量了函数执行前后的`start_time`和`end_time`。

```
import time

from functools import wraps

def timer(func):
    """Example: Measure execution time of function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {round(end_time - start_time, 4)}")
        return result
    return wrapper

@timer
def func_add(a, b):
    time.sleep(2)
    return a + b

func_add(1, 2)
# Execution time: 2.0064
```

# 用法:使用日志调试

`logging`装饰器可用于将信息记录到控制台或日志文件中，对于调试非常有用。在下面的代码片段中，我们将使用`logging` python 包来执行日志记录。

```
import logging

from datetime import datetime
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def logging(func):
    """Example: Logging with decorator"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        log_time = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"{log_time}: {func.__name__} called")
        result = func(*args, **kwargs)
        return result
    return wrapper
```

# 用法:创建单例类

`singleton`装饰器可以用来创建一个单例类。Singleton 是一种创造性的设计模式，它限制一个类只能有一个实例。这在对共享资源的并发访问或资源的全局访问点有限制的情况下很有用，例如对数据库的并发访问或数据库的单点访问施加限制。

单例类可以专门编码以确保单个实例化。然而，如果有多个单独的类，使用 decorators 是为多个类重用代码的好方法。

```
from functools import wraps

def singleton(cls):
    """Example: Create singleton class with decorator"""
    instances = {}

    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper

@singleton
class SampleClass:
    def __init__(self):
        pass

singleton_class = SampleClass()
singleton_class2 = SampleClass()
print(singleton_class == singleton_class2)
# True
```

希望你已经学会了装修工的基础知识，有用的小技巧，以及装修工的实际例子。还有其他的用法，比如使用 decorators 进行超时操作、记忆和缓存。那些 decorator 更高级，最好使用内置的 Python decorators 或者 Python 包中的 decorator，而不是自己实现。

**感谢您的阅读！**如果你喜欢这篇文章，请随意分享。

# 相关链接

伐木文件:[https://docs.python.org/3/howto/logging.html](https://docs.python.org/3/howto/logging.html)

超时装饰:[https://pypi.org/project/timeout-decorator/](https://pypi.org/project/timeout-decorator/)

缓存文档:[https://docs.python.org/3/library/functools.html](https://docs.python.org/3/library/functools.html)