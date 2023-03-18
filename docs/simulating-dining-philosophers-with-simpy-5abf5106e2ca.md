# 用 SimPy 模拟哲学家进餐

> 原文：<https://towardsdatascience.com/simulating-dining-philosophers-with-simpy-5abf5106e2ca>

## 探索 Python 中的竞争条件、死锁和共享资源

![](img/9b29c5984d544493a2d3c403fbbb7764.png)

照片由[玛丽莎·哈里斯](https://unsplash.com/@marisa_harris?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/chopstick-rice?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄

哲学家进餐问题是计算机科学中的一个问题，特别是在并发系统中。它最初是由埃德格·迪杰斯特拉发明的一种考试题型，很快就采用了现在的形式，并成为一种经典。这可以被视为一个玩具问题，但它有效地展示了**资源争夺**的基本难题。今天，我们将把它作为一个完美的借口来学习关于 [Simpy](https://simpy.readthedocs.io/en/latest/) 的教程，这是一个 Python 的离散事件模拟包，非常适合对问题进行建模。设置描述如下:

*圆桌旁坐着 K 个哲学家。他们每个人面前都有一碗米饭，还有一根筷子——只有一根！—在每对相邻的碗之间。每个哲学家思考的时间是不确定的，然后变得饥饿，不得不吃东西。吃饭时，他/她必须同时拿到左右筷子。请注意，相邻的哲学家，因此，必须交替使用筷子坐在他们之间。吃完后，哲学家放下筷子，开始新的思考。*

相当不切实际的是，哲学家们不会自发地交流，也不会担心共用筷子而不用时不洗，但这只是一个模型！我们可以开始使用 Simpy 编写一个 Python 函数来模拟`K` 用餐的哲学家，同时介绍主要概念。

在 Simpy 中，所有模拟的东西都存在于第 4 行声明的`Environment`(在我们的例子中是`table`)的实例中。当存在有人会争夺的资源时，环境变得有趣起来:在我们的例子中是`chopsticks`，它是`Resource`类的`K`实例的列表。一个`Resouce`必须引用一个环境，在我们的例子中是(不出所料)`table`，并且有一个`capacity`，这意味着资源可以同时满足多少个使用请求。一根筷子的容量是 1，因为它一次只能被一个哲学家使用(容量为> 1 的资源的一个例子是，例如，有多个隔间的浴室)。
在第 10–11 行，我们使用`table`的`process`方法将`K`进程添加到环境中，每个哲学家一个进程。流程是*事件发生器*；在我们的例子中，他们会发出一些事件，比如抓住一根筷子(一旦它变得可用)并释放它。当我们看下一段代码时，这一点可能会更清楚。我个人发现，没有过程，环境和资源是没有意义的……反之亦然:你需要考虑两者才能有完整的图景。根据我们所写的，这些进程是对`philosopher_process`的调用，而这又是我们函数的一个参数。我们必须传递一些可以用正确的[签名](https://developer.mozilla.org/en-US/docs/Glossary/Signature/Function)调用的东西。

既然我们已经解决了环境问题，我们可以研究哲学家了。归根结底，这个问题很有趣，因为如果放任自流，哲学家们可能会做出一些行为，将他们带入一种僵局状态，在这种状态下，不可能有任何进展，他们将会挨饿。例如，假设他们使用以下程序:

```
while True:
    # 1-think for a random amount of time
    # 2-grab left chopstick
    # 3-grab right chopstick
    # 4-eat
    # 5-release left chopstick
    # 6-release right chopstick
```

正如我们在上一节中看到的，哲学家由一个进程表示，事实上我们可以从上面的伪代码中看到，它将发出筷子的使用请求。因此，我们将编写一个函数——特别是用 Python 行话编写一个[生成器](https://wiki.python.org/moin/Generators)。让我们把它编码进去并仔细检查一下。

这里面肯定有相当多的陈述。Simpy 中的一个进程可以通过`yield`产生尽可能多的事件，在底层，Simpy 引擎将遍历所有声明的进程生成的事件，并使它们以正确的顺序发生。在哲学家的例子中，我们允许(并希望)它永远继续下去，但这不是必须的:一个过程也可以在产生了有限数量的事件后明确终止，如果这是建模对象的性质的话。
我们在这里发布三种类型的事件:

*   `Resource.request()`发布资源使用请求。在引擎满足该请求之前，发布流程不会继续执行。如果资源不可用(换句话说，由于先前的请求而满负荷)，模拟将继续进行，直到它变得可用(如果曾经可用的话！)之后再将资源分配给进程并继续执行。
*   `Resource.release(request)`向先前请求的资源返回 1 个单位的能力。它需要一个先前完成的请求作为参数。
*   `Environment.timeout(duration)`指示 Simpy 引擎继续模拟一段时间`duration`，然后返回并继续执行。在此期间，其他进程自然会发出自己的事件。在 Simpy 模拟中，超时是时间流逝的唯一方式！如果其间没有超时，所有其他事件都将在时间 0 生成。

注意，抓住或放下筷子不是一个[原子操作](https://www.ibm.com/docs/en/aix/7.1?topic=services-atomic-operations)，但是它有一个持续时间`handling_time`。代码的其余部分应该非常简单。我们建立了一个小的日志系统，将记录附加到一个列表中，这将允许我们更清楚地看到最终发生了什么。此外，`philosopher`的签名包含一些额外的配置参数，在将它传递给我们的`run_dining_philosophers`函数之前，我们必须修复这些参数(例如通过使用[部分](https://docs.python.org/3/library/functools.html#functools.partial))。让我们在一个主脚本中将所有这些联系在一起，并尝试一下:

首先，这段代码*终止*，即使进程内部有一个无限循环。为什么它们以某种方式停止生成事件，Simpy 认识到这一点并停止模拟。让我们看看打印的日志:

```
<T=210131.76>  Phil_#0 is hungry!
<T=210133.76>  Phil_#0 has taken L chopstick (#0)
<T=210140.80>  Phil_#1 is done, releases chopsticks #1/#2
<T=210144.80>  Phil_#0 has taken R chopstick (#1) and can eat
<T=210145.72>  Phil_#3 is done, releases chopsticks #3/#4
<T=210164.80>  Phil_#0 is done, releases chopsticks #0/#1
<T=210179.26>  Phil_#4 is hungry!
<T=210181.26>  Phil_#4 has taken L chopstick (#4)
<T=210183.26>  Phil_#4 has taken R chopstick (#0) and can eat
<T=210203.26>  Phil_#4 is done, releases chopsticks #4/#0
<T=210214.71>  Phil_#3 is hungry!
<T=210215.58>  Phil_#2 is hungry!
<T=210215.80>  Phil_#4 is hungry!
<T=210216.71>  Phil_#3 has taken L chopstick (#3)
<T=210217.14>  Phil_#0 is hungry!
<T=210217.58>  Phil_#2 has taken L chopstick (#2)
<T=210217.80>  Phil_#4 has taken L chopstick (#4)
<T=210217.82>  Phil_#1 is hungry!
<T=210219.14>  Phil_#0 has taken L chopstick (#0)
<T=210219.82>  Phil_#1 has taken L chopstick (#1)
```

在某一点上，所有思考的哲学家同时变得饥饿，在需要实际抓住他们各自左边筷子的短暂时间里，他们都发现他们的邻居抓住了他们右边的那根筷子。它们不能按照当前的算法进行。他们陷入了**僵局。**

让我们来看看这个问题的一个可能的解决方案:奇怪的哲学家从左边的筷子开始，甚至哲学家也从右边的筷子开始。

如果你使用这个版本的哲学家进程，而不是以前的，死锁消失，程序不会停止。我喜欢这种解决方案，因为它有些优雅，不需要额外的同步原语，只需要一点小小的修改就可以完全避免死锁，**但它在实践中有一个主要问题**:如果`K`是奇数，最后一个哲学家比其他人有优势，平均起来，他等待筷子的时间会更少。事实上，这两个哲学家 0-1，2-3…会希望他们之间的筷子是他们的第一根筷子，但最后一个哲学家没有这样的竞争对手来争夺他/她的第一根筷子，这是一个优势。考虑到公平性要求、不对称性和其他微妙的交互现象，该问题还有其他解决方案(以及该问题本身的其他变体),如果您对共享资源和同步感兴趣，您应该看看它们。然而，通过这个相对简单的设置，我希望已经给出了它的要点并展示了 Simpy 的基本原理！

![](img/846afedca437a48a1285c1e62dc30934.png)

作者迷因

[1]如果处理确实是原子性的，这个问题无论如何都会存在，但在实践中基本上不会发生，例如解决方案#2 [中所示，这里的](http://web.eecs.utk.edu/~mbeck/classes/cs560/560/notes/Dphil/lecture.html)为每根筷子包含一个互斥体 Simpy 资源确实是互斥体。事实上，如果将`handling_time`设置为 0，死锁的概率是天文数字。