# Python 中一个简单的多重处理框架

> 原文：<https://towardsdatascience.com/a-simple-multiprocessing-framework-within-python-9497bdf9b42b>

## 利用多重处理并不复杂

Python 中的基本多重处理类非常有用。如果您曾经需要作业运行得更快，也许您已经尝试了矢量化，并且已经测试了多种方法来提高速度，但您仍然等待了太长时间，请继续阅读。多处理您的工作负载有许多优势。但是警告，多重处理并不总是更快——你需要正确的用例来提高速度。不要担心，我有一个简单的解决方案，使利用多处理更容易。

![](img/8fb65df77de88a5b5356975482352d1d.png)

由[杰里米·贝赞格](https://unsplash.com/@unarchive?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

Python 中的基本多处理模块很棒，但是我发现扩展多处理包(在多处理功能的基础上有所改进)更容易，更不容易出错。下面是如何将它安装到您的环境中(您还将看到 tqdm，这是我们将用来跟踪作业之间进度的包)。

```
pip install tqdm multiprocess
```

[tqdm 文档](https://pypi.org/project/tqdm/)和[多进程文档](https://pypi.org/project/multiprocess/)的链接。

因为我不喜欢让你等太久才看到代码，这里是我们将要讨论的代码。

```
from tqdm import tqdm
from multiprocess import Pooldef lite_multi_process(any_function, all_events: list, n_jobs=1):
    num_events, results = len(all_events), list()
    with Pool(n_jobs) as pool:
        with tqdm(total=num_events) as progress:
            for result in pool.imap_unordered(any_function, all_events):
                results.append(result); progress.update()
    pool.close(); pool.join()
    return results
```

## lite_multi_process 简介

在我看来，这个函数胜过所有其他的多重处理函数。它接收任何函数、所有事件(作为一个可以迭代的对象，比如一个列表)，以及您希望并发运行的作业数量。

假设您需要从 PC 或笔记本电脑上的数据库(例如 SQL Server)中查询一组特定的数据。您的函数查询数据，通过一系列操作处理数据，并将结果保存到另一个位置(顺便说一句:如果您想使用 AWS 存储，如 S3，这可能是使用 AWS SQS 和 AWS Lambda 过程的正确时机)。为了一个例子，我们将继续。

关于事件列表，我建议您的列表包含键值对字典，即每个作业的唯一信息，如下所示:

```
{
    'query_id': 1,
    'sql_statement': '''SELECT * FROM table_name'''
}
```

您的函数可能如下所示(当然，带有用于查询、处理和保存的代码):

```
def my_function(event):

    query_id = event.get('query_id')
    sql_statement = event.get('sql_statement')

    # query

    # process

    # save

    return status
```

8 个作业同时运行时使用 lite_multi_process 函数的方法如下:

```
results = lite_multi_process(my_function, my_events, n_jobs=8)
```

超级简单对吧？

现在，假设您的函数没有以返回状态结束(因为您不会对状态的结果集做任何事情)，而是返回了您需要连接并保存到 AWS S3 的 Pandas 数据帧。这是它看起来的样子:

```
results = lite_multi_process(my_function, my_events, n_jobs=8)
data = pd.concat(results)
```

再说一遍，小菜一碟。您的结果包含一个迷你数据框架列表。您需要执行的只是一个连接操作。

最后要提的一块，也是我这么喜欢这个流的另一个原因，这个函数很容易调试。简单设置 n_jobs=1，调试，继续。通常，用更复杂的多处理引擎进行调试是一件非常麻烦的事情。

## 最后的话

我们讨论了多重处理模块，在这种情况下，我们希望并发处理作业。lite_multi_process 函数获取所有必要的信息，并返回结果列表。对结果做你想做的，继续前进。

请在评论中告诉我你的想法。希望这能对你的工作流程有所帮助。