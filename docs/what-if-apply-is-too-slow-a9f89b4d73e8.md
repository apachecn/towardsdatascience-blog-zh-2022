# 如果……apply()太慢？

> 原文：<https://towardsdatascience.com/what-if-apply-is-too-slow-a9f89b4d73e8>

## 有时我们需要在 Python 中通过使用 Pandas 数据帧的列作为函数的输入，对 Pandas 数据帧应用一些函数。不过，用的最多的方法 ***。对整个数据帧应用()*** 可能会比预期花费更长的时间。我们做什么呢

![](img/c68229af88974a66cebbcd5892ed6bbf.png)

照片由[克里斯·利维拉尼](https://unsplash.com/@chrisliverani?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

如果您正在使用 Python 处理数据，Pandas 一定是您使用最多的库之一，因为它具有方便和强大的数据处理特性。

如果我们想对 Pandas 数据框中一整列的值应用相同的函数，我们可以简单地使用。应用()。Pandas 数据框和 Pandas 系列(数据框中的一列)都可以与一起使用。应用()。

但是你有没有注意到？当我们有一个超级大的数据集时，apply()会非常慢？

在本文中，我将讨论当您想要对列应用一些函数时，加速数据操作的技巧。

## 对单个列应用函数

例如，这是我们的玩具数据集。

```
import pandas as pd
import numpy as np
import timeit

d = {'category': ['apple', 'pear', 'peach'], 'radius': [3, 4, 2], 'sweetness': [1, 2, 3]}
df = pd.DataFrame(data=d)
df
```

如果我们希望根据半径列中的值向数据框中再添加一列“直径”，其中基本上直径=半径* 2，我们可以继续使用。在此应用()。

```
df['diameter'] = df['radius'].apply(lambda x: x*2)
df
```

然后，我们计算执行命令行 10k 次的时间，

```
# Timing
setup_code = """
import pandas as pd
d = {'category': ['apple', 'pear', 'peach'], 'radius': [3, 4, 2], 'sweetness': [1, 2, 3]}
df = pd.DataFrame(data=d)
"""

mycode = '''
df['radius'].apply(lambda x: x*2)
'''

# timeit statement
t1 = timeit.timeit(setup=setup_code,
                     stmt = mycode,
                     number = 10000)
print(f"10000 runs of mycode is {t1}")
```

产生 0.55 秒。还不错，嗯..？但请记住，这只是一个 3 行的玩具数据。如果我们有几百万行呢？

你可能已经注意到我们不需要使用。apply()在这里，您可以简单地执行以下操作，

```
df['diameter'] = df['radius']*2
df
```

我们可以看到输出与使用。应用()。如果我们计算 10k 运行的执行时间，

```
# Timing
setup_code = """
import pandas as pd
d = {'category': ['apple', 'pear', 'peach'], 'radius': [3, 4, 2], 'sweetness': [1, 2, 3]}
df = pd.DataFrame(data=d)
"""

mycode = '''
df['radius']*2
'''

# timeit statement
t1 = timeit.timeit(setup=setup_code,
                     stmt = mycode,
                     number = 10000)
print(f"10000 runs of mycode is {t1}")
```

总共给了我们 0.32 秒，比。apply()函数。

注意，我们可以简单地避免使用。这里只使用 apply()，因为我们使用一个非常简单的函数来计算一个乘以 2 的值。但是在大多数情况下，我们需要对列应用一个更复杂的函数。

例如，我们希望为每个观测值在半径和常数(比如 3)之间添加一个较大值的列。如果你简单地做以下事情，

```
max(df['radius'],3)
```

它将生成下面的错误消息，

因此，我们需要在 apply()函数中编写比较代码。

```
df['radius_or_3'] = df['radius'].apply(lambda x: max(x,3))
df
```

让我们计算一下执行时间，

```
# Timing
setup_code = """
import pandas as pd
d = {'category': ['apple', 'pear', 'peach'], 'radius': [3, 4, 2], 'sweetness': [1, 2, 3]}
df = pd.DataFrame(data=d)
"""

mycode = '''
df['radius'].apply(lambda x: max(x,3))
'''

# timeit statement
t1 = timeit.timeit(setup=setup_code,
                     stmt = mycode,
                     number = 10000)
print(f"10000 runs of mycode is {t1}")
```

给了我们 0.56 秒。但是如果数据有几百万行，需要多长时间呢？这里没展示，不过要几十分钟。这么简单的操纵是不能接受的吧？

应该怎么加快速度？

下面是通过使用 NumPy 而不是。apply()函数。

```
df['radius_or_3'] = np.maximum(df['radius'],3)
```

这里的 NumPy 函数最大值是一个比。应用()。让我们计算一下时间。

```
# Timing
setup_code = """
import pandas as pd
import numpy as np
d = {'category': ['apple', 'pear', 'peach'], 'radius': [3, 4, 2], 'sweetness': [1, 2, 3]}
df = pd.DataFrame(data=d)
"""

mycode = '''
np.maximum(df['radius'],3)
'''

# timeit statement
t1 = timeit.timeit(setup=setup_code,
                     stmt = mycode,
                     number = 10000)
print(f"10000 runs of mycode is {t1}")
```

它产生 0.31 秒，比。apply()函数吧？

因此，要点是，在简单地使用。对所有内容应用()函数。

## 将函数应用于多列

有时我们需要使用数据中的多个列作为函数的输入。例如，我们希望创建一列列表，记录“半径 _ 或 _3”和“直径”之间的可能大小。

我们可以利用。将()应用于整个数据帧，

```
df['sizes'] = df.apply(lambda x: list(range(x.radius_or_3,x.diameter)), axis=1)
df
```

这一步实际上非常耗时，因为我们实际上在。apply()函数。执行时间是，

```
# Timing
setup_code = """
import pandas as pd
import numpy as np
d = {'category': ['apple', 'pear', 'peach'], 'radius': [3, 4, 2], 'sweetness': [1, 2, 3]}
df = pd.DataFrame(data=d)
df['diameter'] = df['radius']*2
df['radius_or_3'] = np.maximum(df['radius'],3)
"""

mycode = '''
df.apply(lambda x: list(range(x.radius_or_3,x.diameter)), axis=1)
'''

# timeit statement
t1 = timeit.timeit(setup=setup_code,
                     stmt = mycode,
                     number = 10000)
print(f"10000 runs of mycode is {t1}")
```

给了我们 1.84 秒。我会告诉你，对于一个数百万行的数据帧来说，这需要 20 多分钟。

我们能找到更有效的方法来完成这项任务吗？

答案是肯定的。我们唯一需要做的就是创建一个函数，根据需要接受尽可能多的 NumPy 数组(pandas 系列)作为输入。

```
def create_range(a,b):
    range_l = np.empty((len(a),1),object)
    for i,val in enumerate(a):
        range_l[i,0] = list(range(val,b[i]))
    return range_l

df['sizes'] = create_range(df['radius_or_3'].values,df['diameter'].values)
df
```

这段代码有一个函数， *create_range* ，它接受两个 Numpy 数组，并通过一个简单的 for 循环返回一个 Numpy 数组。并且返回的 Numpy 数组可以自动转换为 Pandas 系列。

让我们检查一下我们节省了多少时间。

```
# Timing
setup_code = """
import pandas as pd
import numpy as np
d = {'category': ['apple', 'pear', 'peach'], 'radius': [3, 4, 2], 'sweetness': [1, 2, 3]}
df = pd.DataFrame(data=d)
df['diameter'] = df['radius']*2
df['radius_or_3'] = np.maximum(df['radius'],3)
"""

mycode = '''
def create_range(a,b):
    range_l = np.empty((len(a),1),object)
    for i,val in enumerate(a):
        range_l[i,0] = list(range(val,b[i]))
    return range_l

create_range(df['radius_or_3'].values,df['diameter'].values)
'''

# timeit statement
t1 = timeit.timeit(setup=setup_code,
                     stmt = mycode,
                     number = 10000)
print(f"10000 runs of mycode is {t1}")
```

它给了我们 0.07 秒！！！！

看到了吗？它的速度比。对整个数据框应用()函数！！

## 外卖食品

1.  如果你想用。将()应用于 Pandas 数据框中的单个列，尝试找到更简单的执行，例如 df['radius']*2。或者尝试为任务找到现有的 NumPy 函数。
2.  如果你想用。将()应用于 Pandas 数据框中的多个列，尽量避免。应用(，轴=1)格式。并编写一个独立的函数，它可以将 Numpy 数组作为输入，然后直接在。熊猫系列的值(数据框的列)。

为了方便起见，这是本文中代码的完整 Jupyter 笔记本。

[](https://github.com/jiananlin/what_if_apply_too_slow/blob/main/apply_too_slow.ipynb) [## 主建安林的 what _ if _ apply _ too _ slow/apply _ too _ slow . ipynb/what _ if _ apply _ too _ slow

### 此时您不能执行该操作。您已使用另一个标签页或窗口登录。您已在另一个选项卡中注销，或者…

github.com](https://github.com/jiananlin/what_if_apply_too_slow/blob/main/apply_too_slow.ipynb) 

这就是我想分享的全部！干杯！

如果你喜欢我的文章，别忘了[订阅我的邮件列表](https://medium.com/subscribe/@jianan-lin?source=publishing_settings-------------------------------------)或者[成为 Medium 的推荐会员](https://medium.com/membership/@jianan-lin?source=publishing_settings-------------------------------------)！！