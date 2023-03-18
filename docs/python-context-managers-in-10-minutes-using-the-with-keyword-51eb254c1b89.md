# Python 上下文管理器 10 分钟—使用“with”关键字

> 原文：<https://towardsdatascience.com/python-context-managers-in-10-minutes-using-the-with-keyword-51eb254c1b89>

## 使用示例简化上下文管理器

上下文管理器本质上是产生单个值的生成器。这个主题现在可能对您来说很陌生，但是您以前很可能使用过上下文管理器。最常用的上下文管理器是 Python 内置的打开文件的`open`命令，通常与关键字`with`一起使用，

```
with open("file.txt", "r") as file:
    data = file.read()
```

类似的用法是使用`open`命令打开文件，读取数据，然后使用`close`命令关闭文件。为了防止用户忘记关闭文件并释放资源，使用`with`关键字将利用上下文管理器并自动处理文件关闭。这个例子被称为`open-close`模式。上下文管理器也可以用于其他模式，如`lock-release`、`change-reset`、`enter-exit`、`start-stop`、`setup-teardown`、`connect-disconnect`。

这篇文章将涉及编写你自己的上下文管理器，高级行为，例如将上下文管理器分配给一个变量，错误处理，接受参数，嵌套上下文管理器，最后是上下文管理器的一些示例用法。

***更新*** *:本文是系列文章的一部分。查看其他“10 分钟内”话题* [*此处*](https://medium.com/@kayjanwong/list/in-10-minutes-eeaa9aa67055) *！*

# 目录

*   [上下文管理器的结构](https://medium.com/p/51eb254c1b89/#104b)
*   [情境经理的高级行为](https://medium.com/p/51eb254c1b89/#08f4)
*   [用途:连接数据库](https://medium.com/p/51eb254c1b89/#d87e)
*   [用途:捕获打印报表](https://medium.com/p/51eb254c1b89/#9754)

# 上下文管理器的结构

上下文管理器可以定义为一个函数或一个类。下面是作为功能实现的上下文管理器的结构，

```
import contextlib

@contextlib.contextmanager
def sample_context():
    # Enter context manager
    # Performs <do-something-here>
    yield
    # Exit context manager

with sample_context():
    # <do-something-here>
    pass
```

上下文管理器函数使用`contextlib.contextmanager`装饰器，它将函数转换成上下文管理器。你可以在我的另一篇文章中找到更多关于装饰者的信息。还有一个`yield`关键字，它现在什么也不做，但是作为命令的分隔符，这些命令将在进入和退出上下文管理器时运行。

为了将上下文管理器实现为一个类，在进入和退出上下文管理器时运行的命令将分别在`__enter__`和`__exit__` dunder 方法中定义。这不会使用任何修饰或`yield`关键字，这对于面向对象编程爱好者来说可能更容易理解。

```
class SampleContext:
    def __enter__(self):
        # Enter context manager
        pass

    def __exit__(self, *args):
        # Exit context manager
        pass

with SampleContext():
    # <do-something>
    pass
```

在`open-close`模式的情况下，文件将在进入上下文管理器时打开，在退出上下文管理器时关闭。这样，步骤和复杂性从用户那里抽象出来，并且与文件相关的资源以由用户控制的方式被释放。

> 上下文管理器是一种函数或类，它为代码运行设置上下文，运行代码，然后移除上下文。上下文管理器最适合用于控制资源的分配和释放。

# 上下文经理的高级行为

## 将上下文管理器分配给变量

可以使用关键字`as`将上下文管理器分配给一个变量，比如`with sample_context() as variable`。我们可以使用`as`关键字，仍然运行上一节中的代码，只是变量将是`None`，因为它是未定义的。

为了定义变量，我们可以对上下文管理器函数使用`yield`关键字，或者对上下文管理器类使用`__enter__`方法中的`return`关键字。

```
# Context Manager Function
@contextlib.contextmanager
def sample_context_variable():
    # Enter context manager
    # Performs <do-something>
    yield "something"
    # Exit context manager

with sample_context_variable() as variable:
    # <do-something>
    print(variable)  # "something"

# Context Manager Class
class SampleContextVariable:
    def __enter__(self):
        # Enter context manager
        return "something"

    def __exit__(self, *args):
        # Exit context manager
        pass

with SampleContextVariable() as variable:
    # <do-something>
    print(variable)  # "something"
```

## 用于错误处理的上下文管理器

上下文管理器将遇到错误，例如编码错误、文件无法打开或锁没有释放。因此，最好使用`try-except-finally`块进行错误处理。这将在下一节用一个例子来说明。

## 接受参数的上下文管理器

上下文管理器本质上是一个函数或类，应该能够接受参数。我们将定义一个简单的`open_file`上下文管理器，它使用一个`open-close`模式来说明过去的三个部分——使用`yield`将上下文管理器分配给一个变量，使用`try-finally`块和接受参数的上下文管理器进行错误处理，

```
import contextlib

@contextlib.contextmanager
def open_file(file_name):
    try:
        file = open(file_name, "r")
        yield file
        file.close()
    except FileNotFoundError:
        raise FileNotFoundError("File cannot be found")

with open_file("file.txt") as file:
    data = file.read()
```

## 嵌套上下文管理器

最后，可以堆叠上下文管理器，并且可以同时设置多个上下文。唯一需要注意的是，父上下文中的变量可以被子上下文访问，但不能被子上下文访问(根据 [scope](https://realpython.com/python-scope-legb-rule/#nested-functions-the-enclosing-scope) 的概念)，所以要合理地计划嵌套。一个例子是这样的，

```
with open(path_input) as file_input:
    with open(path_output, "w") as file_output:
        for line in file_input:
            # Read and write lines simultaneously
            file_output.write(line)
```

# 用法:连接到数据库

可以在一个上下文中建立到数据库的连接，其中可以有一个在进入上下文管理器时连接到数据库的设置脚本和一个在退出上下文管理器时断开数据库连接的拆卸脚本。

在下面的代码片段中，我们定义了一个`database`上下文管理器。进入上下文管理器后建立数据库连接，然后连接`cur`返回，代码中的`yield`关键字作为`db_conn`接收。最后，使用 teardown 脚本关闭数据库连接。

```
import contextlib
import psycopg2

@contextlib.contextmanager
def database(params):
    print("Connecting to PostgreSQL database...")
    # Setup script
    conn = psycopg2.connect(**params)
    cur = conn.cursor()
    try:
        yield cur
    finally:
        # Teardown script
        cur.close()
        conn.close()
        print("Database connection closed.")

db_params = dict(
    host="localhost", database="sample_db", user="postgres", password="Abcd1234"
)
with database(db_params) as db_conn:
    data = db_conn.execute("SELECT * FROM table")
```

上面的代码片段可以推广到涉及资源共享的其他用途，比如在多线程场景中实现锁。

# 用法:捕获打印报表

我发现上下文管理器的另一个有用的用法是在一个列表中捕获所有打印到控制台的打印语句，在那里可以进一步处理该列表或将其保存为一个文件。这在使用外部 Python 包时很有用，在这种情况下，您不能修改源代码，但函数会将大量信息打印到控制台，您希望捕获这些信息或希望阻止这些信息显示出来。

```
import sys
from io import StringIO

def print_function():
    print("Hello")
    print("World")
    print("New\nLine")

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = StringIO()
        self._current_string = sys.stdout
        return self

    def __exit__(self, *args):
        self.extend(self._current_string.getvalue().splitlines())
        del self._current_string
        sys.stdout = self._stdout

with Capturing() as output:
    print_function()
    # Nothing will be printed to console since it is captured

print(output)
# ['Hello', 'World', 'New', 'Line']
```

***更新*** :从 Python 3.4 开始，可以使用`contextlib`库中的上下文管理器捕获打印语句。不同之处在于，该方法将打印语句捕获为字符串，而不是字符串列表。

```
import sys
from io import StringIO

def print_function():
    print("Hello")
    print("World")
    print("New\nLine")

f = StringIO()
with contextlib.redirect_stdout(f):
    print_function()

output = f.getvalue()
# output is 'Hello\nWorld\nNew\nLine\n'
```

希望这篇文章已经介绍了上下文管理器的基础知识，如何创建定制的上下文管理器以及一些示例用法。有许多 Python 内置的上下文管理器和来自 Python 包的上下文管理器，所以在自己实现它之前一定要四处看看，以节省一些时间和精力。

**感谢您的阅读！**如果你喜欢这篇文章，请随意分享。

# 相关链接

`contextlib`文献:[https://docs.python.org/3/library/contextlib.html](https://docs.python.org/3/library/contextlib.html)

PostgreSQL Python 文档:[https://www . PostgreSQL tutorial . com/PostgreSQL-Python/connect/](https://www.postgresqltutorial.com/postgresql-python/connect/)

捕获打印报表:[https://stackoverflow.com/questions/16571150](https://stackoverflow.com/questions/16571150)