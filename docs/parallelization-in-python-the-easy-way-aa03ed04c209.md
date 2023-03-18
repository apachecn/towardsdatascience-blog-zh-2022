# Python 中的并行化:简单的方法

> 原文：<https://towardsdatascience.com/parallelization-in-python-the-easy-way-aa03ed04c209>

## 并行化不一定很难

![](img/070f78092433a86f743d5b4c63b77990.png)

Python 中的并行化并不一定很难。阿巴斯·特拉尼在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

许多初学者和中级 Python 开发人员害怕并行化。对他们来说，并行代码意味着困难的代码。进程、线程、greenlets、协程……并行化代码的工作不是以高性能代码而告终，而是经常以令人头疼和沮丧而告终。

在这篇文章中，我想告诉大家事实并非如此。在简单的场景中，代码很容易并行化。众所周知，Python 是一种易于理解的编程语言，并行代码也易于阅读和实现。

本文是*而不是*并行化的介绍。不全面。相反，我想向您展示在简单的情况下并行化代码是多么简单。这将为您提供将并行化应用于更复杂场景的必要背景。或者至少，从简单的开始总比从难的开始好。

简单的例子不一定意味着学术上的例子。作为 Python 开发人员或数据科学家，您会经常遇到需要加快应用程序速度的情况。您应该做的第一件事是分析代码。忽略这一点可能是试图优化它时最常见的错误。概要分析可以向您展示应用程序中的瓶颈，在这些地方您可以获得一些收益。如果您不分析您的代码，您可能会花一天的时间来优化应用程序中实际上最有效的部分。

例如，在进行分析后，您会发现应用程序 99%的时间都在运行执行某些计算的 iterable 上的一个函数。那么，你可能会想，如果你并行运行这个函数，应用程序应该会更快——你*可能是对的。这种情况就是我所说的简单情况。*

本文旨在展示如何在这种情况下简单地并行化 Python 代码。不过，你可能还会学到一件事:如何设计和原型化代码。我不会写一个实际的应用程序，因为这样的代码会让我们分心。相反，我将编写这样一个应用程序的简单原型。原型可以工作，但不会做任何有价值的事情；它只会做一些类似于这种应用程序在现实生活中所能做的事情。

# 示例应用程序

假设我们的目标是编写一个处理文本的应用程序。这种处理可以是你想对文本做的任何事情。例如，你可以在书中搜索指纹，正如本·布拉特在他关于纳博科夫的精彩著作中所做的那样(布拉特 2017)。

在我们的应用程序中，对于每个文本，将运行以下管道:

*   读取文本文件
*   预处理文件；这意味着清洁和检查
*   处理文件
*   返回结果

一种优雅的方法是创建一个生成器管道。我在我的[另一篇文章](/building-generator-pipelines-in-python-8931535792ff)中展示了如何做。我们将使用这种方法。

下面，您将找到创建上述管道的一个函数，以及管道使用的三个函数。管道函数将负责运行所有这些函数，然后返回结果。

我将使用这些函数的简单模拟，因为我不希望您关注与并行化无关的代码。我还将添加一些睡眠时间，以便函数花费的时间与它们在现实中可能花费的时间成比例。请注意，这些休眠旨在表示计算量大(而不是等待 HTTP 响应)的任务，对于这些任务，加速计算的唯一方法是使用并行化。

```
import time
import pathlib
from typing import Dict, Generator, Iterable

# type aliases
ResultsType = Dict[str, int]

def read_file(path: pathlib.Path) -> str:
    # a file is read to a string and returned
    # here, it's mocked by the file's name as a string
    # this step should be rather quick 
    time.sleep(.05)
    text = path.name
    return text

def preprocess(text: str) -> str:
    # preprocessing is done here
    # we make upper case of text
    # longer than reading but quite shorter than processing
    time.sleep(.25)
    text = text.upper()
    return text

def process(text: str) -> ResultsType:
    # the main process is run here
    # we return the number of "A" and "O" letters in text
    # this is the longest process among all
    time.sleep(1.)
    search = ("A", "B", )
    results = {letter: text.count(letter) for letter in search}
    return results

def pipeline(path: pathlib.Path) -> ResultsType:
    text = read_file(path)
    preprocessed = preprocess(text)
    processed = process(preprocessed)
    return processed
```

现在，我们需要一个可迭代的路径。真正的`read_file()`函数会读取文件，但是我们的模拟函数不会；相反，它根据路径名生成文本。我这样做是为了让事情尽可能简单。所以，这将是我们的迭代:

```
file_paths = (
    pathlib.Path(p)
    for p in (
        "book_about_python.txt",
        "book_about_java.txt",
        "book_about_c.txt",
        "science_fiction_book.txt",
        "lolita.txt",
        "go_there_and_return.txt",
        "statistics_for_dummies.txt",
        "data_science_part_1.txt",
        "data_science_part_2.txt",
        "data_science_part_3.txt",
    )
)
```

我们已经拥有了我们所需要的一切:一个产生路径的`file_paths`生成器，就像`pathlib.Path`一样，指向文件。我们现在可以使用这个生成器运行整个管道，这样我们将使用一个生成器管道，就像我承诺的那样。为此，我们可以迭代`file_path`生成器，在每次迭代中，每条路径都将通过`pipeline()`函数进行处理。这样，我们将像承诺的那样使用一个生成器管道。

让我们在 name-main 块中执行生成器的评估:

```
if __name__ == "__main__":
    start = time.perf_counter()
    results = {path: pipeline(path) for path in file_paths}
    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")
    print(results)
```

这个管用。我增加了测量时间。因为我们有 10 个文件要处理，而一个文件的处理时间大约为 1.3 秒，所以我们应该预计整个管道的运行时间大约为 13 秒，外加一些开销。

事实上，在我的机器上，整个流水线需要 13.02 秒来处理。

# Python 并行化:操作指南

上面，我们使用字典理解评估了管道。我们可以使用生成器表达式或`map()`函数:

```
if __name__ == "__main__":
    start = time.perf_counter()
    pipeline_gen = map(lambda p: (p, pipeline(p)), file_paths)
    # or we can use a generator expression: 
    # pipeline_gen = ((path, pipeline(path)) for path in file_paths)
    results = dict(pipeline_gen)
    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")
    print(results)
```

对`map()`的调用看起来比上面的 dict 理解更难。这是因为我们不仅想返回`pipeline()`的结果，还想返回一个正在处理的路径。当我们需要将结果与输入联系起来时，这是一种常见的情况。在这里，我们通过返回一个包含两个条目的元组来实现这个目的，`(path, pipeline(path))`，基于这些元组，我们可以创建一个包含`path-pipeline(path)`键-值对的字典。将`lambda` s 和`map()`一起使用是一种常见的方法——但不一定是最清晰的方法。

注意，如果我们不需要路径名，代码会简单得多，因为我们唯一需要的是`pipeline(path)`的结果:

```
pipeline_gen = map(pipeline, file_paths)
```

然而，由于我们需要添加路径名，所以我们使用了`lambda`函数。产生的代码，带有`lambda`的那个，看起来不太吸引人，不是吗？幸运的是，有一个很好的简单技巧来简化对使用`lambda` s 的`map()`的调用。我们可以使用包装函数来代替:

```
def evaluate_pipeline(path):
    return path, pipeline(path)

if __name__ == "__main__":
    start = time.perf_counter()
    pipeline_gen = map(evaluate_pipeline, file_paths)
    results = dict(pipeline_gen)
    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")
    print(results)
```

上面的`map`对象产生一个二元元组的生成器`(path, pipeline(path))`。我们可以使用`dict(pipeline_gen)`来

*   评估发电机，以及
*   将结果转换成由`path` - `pipeline(path)`键值对组成的字典。

难道你不同意这段代码比带有`lambda`的版本可读性更强，更容易理解吗？

对了，如果你想知道生成器管道在哪里，就在这里:`map(evaluate_pipeline, file_paths)`。我们使用`dict()`函数对其进行评估。

## 为什么我们谈论 map()这么多？！

你可能想知道为什么我如此关注`map()`函数。这是怎么回事？我们不应该讨论并行化吗？

我们应该这样做——我们确实这样做了。为了并行化你的代码，你需要首先理解`map()`是如何工作的。这是因为你经常会使用与`map()`非常相似的函数；事实上，这个功能经常被称为`map`。因此，每当您考虑并行化您的代码时，不要使用`for`循环、生成器表达式、列表理解、字典理解、集合理解等——考虑使用`map()`函数。

正如我在关于`map()`(科萨克 2022)的*数据科学*文章中所写的，这个函数被许多并行化模块和技术使用，它们的`map()`版本与内置的`map()`函数非常相似。因此，如果您计划并行化代码，从一开始就使用`map()`通常是好的。不是必须的；只是一个建议。这样，当并行化代码时，您可以节省一些时间。

所以，也许从一个`map()`版本开始更好，甚至不用考虑是否将代码并行化？另一方面，`map()`有时比相应的生成器表达式可读性差；但是你可以使用一个包装函数，基于`map()`的代码应该变得非常可读。

我不会告诉你哪个设计是最好的。有时，`map()`将是完美的解决方案——尤其是当您计划并行化代码时。重要的一点是，永远要小心。代码可读性很重要，但并不总是简单明了的。没有单一的配方。用你的技能和经验来决定。

但是，如果您决定用最流行的并行化包之一来并行化代码，您通常会使用类似于`map()`的函数。所以，不要忽略内置的`map()`功能。明天，它可能会成为你比你今天想象的更亲密的朋友。

# 行动的并行化

现在我们已经使用了`map()`，我们的代码已经准备好被并行化了。为此，让我们使用`[multiprocessing](https://docs.python.org/3/library/multiprocessing.html)`，用于并行化的标准库模块:

```
import multiprocessing
```

在大多数情况下，这可能是一个好主意。然而，我们需要记住，其他模块可以提供更好的性能。我希望在其他文章中写他们。

我们的名称-主块现在变成了:

```
if __name__ == "__main__":
    start = time.perf_counter()
    with mp.Pool(4) as p:
        pipeline_gen = dict(p.map(evaluate_pipeline, file_paths))
    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")
    print(results)
```

四核，仅 4.0 秒！(使用 8 名工人，耗时 2.65 秒。)好看；我们看到`multiprocessing`像预期的那样工作。

请注意，我们不需要对代码做太多修改，只需要做两处修改:

*   `with mp.Pool(4) as p:`:为`mp.Pool`使用上下文管理器是一个很好的规则，因为您不需要记住关闭池。我选择了 4 个工人，这意味着该流程将在四个流程中运行。如果您不想对工作人员的数量进行硬编码，可以用不同的方式来实现，如下所示。我使用了`mp.cpu_count()-1`,这样我们的应用程序就不会触及一个进程；`mp.cpu_count()`返回要使用的内核数量。我有 4 个物理核和 8 个逻辑核，`mp.cpu_count()`返回 8。

```
workers = mp.cpu_count() - 1
with mp.Pool(workers) as p:
```

*   `p.map(evaluate_pipeline, file_paths)`:我们所做的唯一改变是将`map()`改为`p.map()`。还记得我告诉过你使用`map()`会使代码并行化更容易吗？给你，这是这点零钱。然而，不要忘记`p.map()`并不像`map()`那样懒洋洋地求值，而是贪婪地(立即)求值。因此，它不返回生成器；相反，它返回一个列表。

# 一个更简单的例子

```
import multiprocessing as mp
import random
import time

def some_calculations(x: float) -> float:
    time.sleep(.5)
    # some calculations are done
    return x**2

if __name__ == "__main__":
    x_list = [random.random() for _ in range(20)]
    with mp.Pool(4) as p:
        x_recalculated = p.map(some_calculations, x_list)
```

就是这样—这个脚本使用并行计算为`x_list`运行`some_calcualations()`。这是一段非常简单的 Python 代码，不是吗？在这样一个简单的情况下，当您想要使用并行计算时，您可以随时返回到它。

一件好事是，如果您想运行更多的并行计算，您可以为`mp.Pool`使用另一个这样的上下文管理器。

# 结论

我承诺向您展示 Python 中的并行化可以很简单。第一个例子是一个应用程序的原型，它有点复杂，就像现实生活中的应用程序一样。但是请注意，代码的并行部分非常短，相当简单。第二个例子强化了这一点。完整的代码只有十几行——而且是简单的代码。

即使在简单的情况下，也不要害怕使用并行化。当然，只在需要的时候使用。当你不需要让你的代码更快时，使用它是没有意义的。请记住，当要并行调用的函数非常快时，您可能会使您的应用程序变慢，因为并行化不是没有代价的；这一次，成本是创建并行后端的开销。

我展示了 Python 中的并行化可以很简单，但是您可以用它做更多的事情。我将在下一篇文章中讨论这个问题。如果你不想将你的知识局限于这些基础知识，Micha Gorelick 和 Ian Ozsvald 的书(2020)是一个非常好的来源——虽然不是一个简单的来源，因为作者讨论了相当高级的主题。

并不总是最好的方法。有时使用其他工具会更好。有`[pathos](https://pathos.readthedocs.io/en/latest/pathos.html)`，有`[ray](https://docs.ray.io/en/latest/index.html)`，还有其他工具。有时，您的环境会促使您选择并行化后端。例如，在 AWS SageMaker 中，您将需要使用`ray`。这个计划比`multiprocessing`提供的要多得多，我也会试着写一篇关于这个的文章。

今天到此为止。感谢阅读。如果您对 Python 中的并行化感到害怕，我希望您不会再有这种感觉了。亲自尝试一下，注意基本并行化是多么容易。当然，还有更多，但是让我们一步一步地学习。今天，你知道了没有理由害怕它，最基本的是…基本而简单。

我没有涉及技术细节，比如并行化代码的其他方式而不是池、命名和其他类似的事情。我只是想告诉你如何去做，以及有多简单。我希望我成功了。

感谢您阅读本文，请在评论中分享您对并行化的想法。我很乐意阅读您的首次尝试故事:您对并行化的首次尝试是什么？你并行了什么？你有什么问题吗？它像你期望的那样工作吗？你对它的工作方式满意吗？

# 资源

*   科萨克米(2022)。用 Python 构建生成器管道。
*   布拉特 B. (2017)。纳博科夫最喜欢的词是紫红色:数字揭示的经典、畅销书和我们自己的作品。西蒙和舒斯特。
*   [科萨克米(2022)。Python 还需要 map()函数吗？](/does-python-still-need-the-map-function-96787ea1fb05)
*   `[multiprocessing](https://docs.python.org/3/library/multiprocessing.html)` [模块文档](https://docs.python.org/3/library/multiprocessing.html)。
*   Gorelick M .，Ozsvald I. (2020 年)。*高性能 Python:人类实用的高性能编程*。第二版。奥莱利媒体。
*   `[pathos](https://pathos.readthedocs.io/en/latest/pathos.html)` [模块文档](https://pathos.readthedocs.io/en/latest/pathos.html)。
*   `[ray](https://docs.ray.io/en/latest/index.html)` [模块文档](https://docs.ray.io/en/latest/index.html)。