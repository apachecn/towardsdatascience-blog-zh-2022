# 用雷在 5 分钟内爬上汉密尔顿

> 原文：<https://towardsdatascience.com/scaling-hamilton-with-ray-in-5-minutes-3beb1755fc09>

## 扩展数据转换中人的方面以及计算方面

![](img/d0a9af8f9c3a489f83054b78244d4bb1.png)

Hamilton + Ray —帮助您扩大产量。托马斯·凯利在 [Unsplash](https://unsplash.com/s/photos/efficiency?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片。

[Hamilton](https://github.com/DAGWorks-Inc/hamilton) 是一个用于 Python 的开源、声明性数据流微框架。在这篇文章中，我将解释通过利用汉密尔顿的光线集成来扩展用于汉密尔顿工作流的数据和 CPU 的要求。这篇文章假设读者事先熟悉什么是汉密尔顿。对于背景故事和更长的介绍，我们邀请你阅读这个 [TDS 帖子](/functions-dags-introducing-hamilton-a-microframework-for-dataframe-generation-more-8e34b84efc1d)，或者对于更短的五分钟介绍见这个 [TDS 帖子](/how-to-use-hamilton-with-pandas-in-5-minutes-89f63e5af8f5)，或者对于它如何帮助规模团队&保持他们的代码库有组织见[这个 TDS 帖子](/tidy-production-pandas-with-hamilton-3b759a2bf562)。

对于那些不熟悉 [Ray](https://ray.io/) 的人来说，它是一个开源框架，可以扩展来自[加州大学伯克利分校](https://rise.cs.berkeley.edu/projects/ray/)的 python 应用。它有一个不断增长的工具生态系统，可以帮助许多机器学习相关的工作流。例如，它将自己标榜为使您能够非常容易地从笔记本电脑扩展到集群，而无需更改太多代码。就现实世界的使用而言，我喜欢使用 Ray 作为在 python 中实现[多重处理的一种非常快速的方式，而不用担心细节！](https://machinelearningmastery.com/multiprocessing-in-python/)

# 射线底漆

这里有一个射线引物。好消息是，你不需要了解太多关于雷的知识就可以和汉密尔顿一起使用它。你只需要知道**它将轻松地在多个 CPU 内核上并行化你的工作流**，并允许你**扩展到你的笔记本电脑之外**，如果你设置了一个 Ray 集群的话。但是为了让你理解汉密尔顿是如何与它联系起来的，让我们快速回顾一下如何使用雷。

## 光线使用前提

使用 Ray 的基本前提是，您必须注释您希望通过 Ray 调度执行的函数。例如(从[他们的文档](https://docs.ray.io/en/latest/ray-core/tasks.html#ray-remote-functions)):

```
*# This is a regular Python function.*
def normal_function():
    return 1

*# By adding the `@ray.remote` decorator, a regular Python function*
*# becomes a Ray remote function.*
@ray.remote
def my_ray_function():
    return 1
```

然后要执行`my_ray_function`函数，您应该做:

```
my_ray_function.remote()
```

然后它会告诉 Ray 调度函数的执行。要在本地运行而不是在集群上运行，您所要做的就是在调用上面的代码之前以不同的方式实例化“Ray”。

```
import ray
ray.init() # local execution over multiple cores.
ray.init({... cluster details ...}) # connect to cluster.
```

现在，🤔，您可能会想，要让我现有的代码运行起来，似乎需要做很多工作，例如，我如何向函数传递参数？我应该如何更改我的应用程序以更好地使用 Ray？等等。好消息！和汉密尔顿在一起你根本不用考虑这些！您只需编写您的标准 Hamilton 函数，只需更改一些“ *driver* ”代码就可以让它在 Ray 上运行。下一节将详细介绍这一点。

# 汉密尔顿+雷

要将 Ray 与 Hamilton 一起使用，首先需要安装它。

```
pip install "sf-hamilton[ray]"
```

接下来，默认情况下使用 Hamilton，所有的逻辑都写成 python 函数。所以你可以像平常一样写哈密顿函数。这里没有变化。

在执行时，在 Hamilton 框架级别，Hamilton 可以很容易地为您的函数定义的有向无环图(DAG)中的每个函数注入`@ray.remote`。重申一下*，你不需要修改任何 Hamilton 代码来使用 Ray！*要让 Hamilton 在 Ray 上运行，你需要做的就是提供一个"`*GraphAdapter*`*对象给 Hamilton "`*Driver*`*类你实例化。**

**一个`GraphAdapter`，只是一个简单的类，它定义了一些函数，使您能够增加 DAG 的遍历和执行方式。更多信息见文件。**

**就要添加/更改的代码而言，以下是增加标准 Hamilton 驱动程序代码所需的内容—参见 **bold** font:**

```
**import ray
from hamilton import **base**, driver
**from hamilton.experimental import h_ray**...**ray.init() # instantiate Ray**
config = {...} # instantiate your config
modules = [...] # provide modules where your Hamilton functions live
**rga = h_ray.RayGraphAdapter( # object to tell Hamilton to run on Ray
     result_builder=base.PandasDataFrameResult())  **  
dr = driver.Driver(config, *modules, **adapter=rga**) **# tell Hamilton**
df = dr.execute([...]) # execute Hamilton as you normally would
**ray.shutdown()****
```

**对于那些不熟悉 Hamilton 的人来说，如果您要删除加粗的代码，这就是您运行开箱即用的普通 Hamilton 的方式。**

**具体来说，我们需要:**

1.  **实例化射线:`ray.init()`。**
2.  **实例化一个`RayGraphAdapter`，传入一个`ResultBuilder`对象，该对象设置当在`driver.Driver`对象上调用`.execute()`时返回什么对象类型。在这个例子中，我们指定它应该返回一个熊猫数据帧。**
3.  **然后在创建`driver.Driver`对象时，`RayGraphAdapter`作为关键字参数`adapter=rga`被传递，这将告诉 Hamilton 增加 DAG 的行走并使用 Ray。**
4.  **不需要其他的改变，所以在你完成之后，你只需要关闭你和 Ray `ray.shutdown()`的连接。**

## **缩放就是这么简单！**

**通过添加几行代码，您现在可以:**

1.  **并行计算您的哈密尔顿函数。**
2.  **扩展到在集群规模数据上运行。**

**概括一下，使用 Ray 和 Hamilton 的方法与使用普通的 Hamilton 没有太大的不同:**

1.  **装汉密尔顿+雷。`pip install "sf-hamilton[ray]"`。**
2.  **写哈密尔顿函数。**
3.  **编写你的驱动程序代码——如果你想让它在 Ray 上运行，调整这个部分。**
4.  **执行你的驱动脚本。**

**由于切换到使用 Ray 或不使用 Ray 是如此容易，我们很想知道切换到 Ray 会在多大程度上提高数据流操作的速度或规模！**

**要获得完整的“Ray Hello World”代码示例，我们会将您引向这里的[示例目录](https://github.com/stitchfix/hamilton/tree/main/examples/ray/hello_world)。**

# **最后**

**通过使用 Hamilton，您可以组织和扩展编写数据转换的人的方面(*不，我在这篇文章中没有谈到这一点，请参见简介中的链接或下面的说服自己的链接*😉).借助 Ray，您可以扩展您的数据工作流，以超越笔记本电脑的限制。一起，天空的极限！**

**如果您喜欢汉密尔顿，并希望继续关注该项目:**

1.  **我们想要 github 上的[⭐️](https://github.com/DAGWorks-Inc/hamilton)！**
2.  **📣加入我们在 slack 上的羽翼未丰的社区！**
3.  **查看我们的文档📚。**

# **警告**

**关于使用 Hamilton + Ray 的注意事项。**

1.  **我们希望从“*”实验“*”中获得射线支持，但为此我们需要您的反馈！该 API 一直非常稳定(自发布以来一直没有改变)，但为了让它永久化，我们很想知道你的想法。**
2.  **我们不会公开 Ray 的所有功能，但是我们可以。例如存储器感知调度或为特定功能指定资源。如果你想曝光什么，请告诉我们——请在 github 上发表一篇文章——[https://github.com/dagworks-inc/hamilton](https://github.com/DAGWorks-Inc/hamilton)🙏。**

# **相关职位**

**有关汉密尔顿的更多信息，我们邀请您查看:**

*   **[介绍汉密尔顿](/functions-dags-introducing-hamilton-a-microframework-for-dataframe-generation-more-8e34b84efc1d)**
*   **[5 分钟内汉密尔顿+熊猫](/how-to-use-hamilton-with-pandas-in-5-minutes-89f63e5af8f5)**
*   **[在笔记本上与汉密尔顿迭代](/how-to-iterate-with-hamilton-in-a-notebook-8ec0f85851ed)**
*   **[与汉密尔顿一起整理生产熊猫](/tidy-production-pandas-with-hamilton-3b759a2bf562)**
*   **【未来邮报】5 分钟内汉密尔顿+达斯克**