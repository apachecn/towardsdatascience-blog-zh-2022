# 如何缓解 Python 中的内存问题

> 原文：<https://towardsdatascience.com/how-to-mitigate-memory-issues-in-python-c791b2c5ce7e>

## 一个小窍门，可以节省你的时间、精力和成本。

![](img/cba63e6572914326fa1c3537b237c7a6.png)

由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的 [Flipsnack](https://unsplash.com/@flipsnack?utm_source=medium&utm_medium=referral) 拍摄的照片

最近，由于 Python 中管理内存的方式，我遇到了一个问题，在内存受限的环境中使用 Pandas 数据帧和 Numpy 数组可能会变得很严重。这个问题让我花了两天时间，而不是预计的两个小时来实现一个标准的数据处理任务。真扫兴！

因为我们所有的时间都是很昂贵的，所以我写下了我的学习，这样将来的我，也许你可以避免这个问题，节省时间，成本和精力。你准备好了吗？我的故事来了。

# 故事

我必须使用一个 [AWS 弹性容器服务](https://aws.amazon.com/ecs/) (ECS)任务在云中处理一堆拼花文件。我想:没有比这更简单的了，我将使用 [Pandas](https://pandas.pydata.org/) 依次加载每个文件，应用我的机器学习模型，保存结果，我就完成了。没什么特别的，只是一个基本的数据处理任务。我的代码大致如下:

```
import pandas as pd

class Processor:
  def run(self, file_names: list[str]) -> None: 
    for file_name in file_names:
      data = pd.read_parquet(file_name)
      result = self._process(data)
      self._save(result)
```

现在，要在 ECS 上做到这一点，您必须编写一个 docker 文件，并设置一个 ECS 任务，相应的容器将在其中运行。在那里，您必须指定想要使用多少 CPU 和多少内存。显然，你走得越高，它就变得越贵，你给的钱就越多。由于我的每个文件只有几百兆大，我想我可以选择一个小的大小，因此不必在 AWS 上花费太多。所以我选择了两个 CPU 和 8g 内存。对我来说，考虑到我的数据量，这听起来已经有点夸张了。但我想保险起见，并认为现在可能会出错吗？

## 问题是

因此，我满怀信心地部署了我的容器，并开始了 ECS 任务。第一个文件已成功加载和处理。是啊。第二个文件已成功加载和处理。是的。第三次、第四次和第五次也是如此，但之后…

```
Container exited with code 137\. Reason was OutOfMemoryError: 
Container killed due to memory usage
```

![](img/eb917b504fdee9a457cb9f13ea03937f.png)

斯蒂芬·拉德福德在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

什么？我没想到会发生这种事。8 Gb 的内存怎么可能不够处理一个只有几百兆大小的文件？更重要的是，由于每个文件的大小完全相同，为什么这个问题只发生在第六个文件之后，而不是直接发生在第一个文件上？关于 Python 中如何管理内存，肯定有什么地方出错了，或者说是我没有考虑到的地方。

所以，我谷歌了一下 Python 中的内存管理，学习了[引用计数和垃圾收集](/memory-management-and-garbage-collection-in-python-c1cb51d1612c)的基础知识。简单来说，引用计数就是计算 python 对象在代码中被主动引用的频率。如果一个对象在任何地方都不再被引用，也就是说，它的引用计数器为零，Python 垃圾收集器可以介入并释放这个对象占用的内存。很好，听起来很琐碎。

## 无效的解决方案

因此，我认为加强这种行为将解决我的问题，我终于可以完成我的任务。那你做了什么？为了将引用计数器减少到零，我在引用数据的变量上使用了`**del**` 。为了强制垃圾收集器在那之后立即运行，我显式地使用了来自 [gc](https://docs.python.org/3/library/gc.html) 模块的`**collect**`函数。有了这个，我的更新版本大致如下

```
import pandas as pd
import gc

class Processor:
  def run(self, file_names: list[str]) -> None: 
    for file_name in file_names:
      data = pd.read_parquet(file_name)
      result = self._process(data)
      self._save(result)
      del result
      del data
      gc.collect()
```

怀着非常高的希望，我部署了这个更新版本，启动了容器，等待处理第六个文件。猜猜发生了什么？*由于内存使用，容器被杀死。*

妈的，一点进步都没有*。很明显，在我最基本的代码中还隐藏着一些我无法影响的东西，它们一直占据着我的内存。*

那么我们现在应该做些什么来解决这样一个常见的数据工程问题呢？选择更多的内存并进一步增加杰夫·贝索斯的财富是唯一的出路吗？幸运的是，这个问题的答案是 ***不是*** 。有一种方法应该总是有效的，尽管感觉有点粗糙。

## 工作解决方案

为了确保在一个函数完成后释放内存或其他资源，您所要做的就是在不同的进程中执行该函数。当该进程完成时，*操作系统* (OS)释放该进程已经获得的所有资源。这总是独立于底层操作系统工作。需要注意的是，启动一个进程会降低整体性能。

在 Python 的进程中运行函数的一种方式是通过标准的[多处理模块](https://docs.python.org/3/library/multiprocessing.html)。然而，这要求函数及其所有参数都是可选择的。参见这个[网站](https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled)获取可供选择的类型列表。在我的例子中，情况并非如此，因为我想运行一个类的方法，使用默认的 pickle 库是不可选择的。该死的。

幸运的是，有一个名为 [loky](https://loky.readthedocs.io/en/stable/API.html) 的不错的第三方库，它使用了引擎盖下的 [cloudpickle](https://github.com/cloudpipe/cloudpickle) 模块，几乎可以腌制任何东西。太好了！

有了这些，我的最终解决方案看起来像

```
import pandas as pd
import gc
from loky import get_reusable_executor
# I only need one workere as I don't want to do multiprocessing
executor = get_reusable_executor(max_workers=1)

class Processor:
  def _execute(self, file_name:str) -> None:
    data = pd.read_parquet(file_name)
    result = self._process(data)
    self._save(result)

  def run(self, file_names: list[str]) -> None: 
    for file_name in file_names:
      executor.submit(self._execute, file_name).result() 
```

这里你可以看到我已经为 *循环*拉出了*的内部部分，在这里数据被加载、处理并存储到一个单独的函数中。这个函数我现在使用 loky executor 在一个专用的进程中运行这个函数，它确保了一旦完成就释放内存。这样，我的任务成功地处理了所有文件，我甚至能够进一步减少所需的内存量。最后，*

![](img/8b63d9e2715ea06c136484f830050045.png)

照片由[雷·轩尼诗](https://unsplash.com/@rayhennessy?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

# 包裹

在我的短文中，我谈到了 Python 应用程序中没有释放内存的问题。在像 AWS ECS 这样内存受限的环境中运行时，这个问题可能会很严重。通过将一个函数的执行放在一个专用的进程中，您可以确保在函数完成时释放内存等资源。希望，这对你以后有所帮助。

感谢您关注这篇文章。一如既往，如果有任何问题、意见或建议，请随时联系我或关注我，无论是在 Medium 还是通过 [LinkedIn](https://www.linkedin.com/in/simon-hawe-75832057) 。