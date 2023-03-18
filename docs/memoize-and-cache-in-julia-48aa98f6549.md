# Julia 中的内存和缓存

> 原文：<https://towardsdatascience.com/memoize-and-cache-in-julia-48aa98f6549>

## 使用动态编程让你的函数更快

表情化是编程中的一个简单技巧，通过记住一些间歇的结果来减少计算量。在这里，我将向你展示如何用最简单的方法在 Julia 中做到这一点。

> 🙏非常感谢在这里查看回购[的人们。](https://github.com/JuliaCollections/Memoize.jl)

![](img/032514fe6799bf37418572028161647d.png)

照片由 [Fredy Jacob](https://unsplash.com/@thefredyjacob?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/memory?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

# 天真些

我们将从一个简单的斐波那契函数开始，看看对于较大的数字它会变得多慢。斐波纳契数列由以下简单规则定义:

> F(1) = 1，F(2) = 1，F(n) = n — 2 + n — 1

其中`n`为正整数。

让我们用 Julia 来编码:

这是一个**递归函数**，它调用它自己，这本身并不是一个罪过，但是对于这个特殊的函数来说，代价非常高。如果你想了解为什么这很贵以及记忆化是如何工作的，看看这个由 **freeCodeCamp** 制作的精彩视频:

由[freecodecamp.org](https://www.freecodecamp.org)进行动态编程

😉我喜欢 Julia 的一点是**你可以简洁地定义简单的函数。上面和这个小俏皮话一样:**

让我们通过使用方便的`@show`宏来检查上面的内容是否有效:

```
julia> for i in 1:10
           [@show](http://twitter.com/show) i, fibonacci(i)
       end
(i, fibonacci(i)) = (1, 1)
(i, fibonacci(i)) = (2, 1)
(i, fibonacci(i)) = (3, 2)
(i, fibonacci(i)) = (4, 3)
(i, fibonacci(i)) = (5, 5)
(i, fibonacci(i)) = (6, 8)
(i, fibonacci(i)) = (7, 13)
(i, fibonacci(i)) = (8, 21)
(i, fibonacci(i)) = (9, 34)
(i, fibonacci(i)) = (10, 55)
```

我们还可以通过使用`BenchmarkTools`包来检查上述操作有多快:

```
julia> [@btime](http://twitter.com/btime) fibonacci(30)
  2.280 ms (0 allocations: 0 bytes)
832040
```

> 想看看朱莉娅的其他作品吗？别害羞，跟我来😅

[](/build-your-first-neural-network-with-flux-jl-in-julia-10ebdfcf2fa3) [## 在 Julia 中用 Flux.jl 构建你的第一个神经网络

### 没有任何外部数据的初学者教程

towardsdatascience.com](/build-your-first-neural-network-with-flux-jl-in-julia-10ebdfcf2fa3) [](https://blog.devgenius.io/make-a-command-line-game-with-julia-a408057adcfe) [## 和 Julia 一起做一个命令行游戏

### 没有图形用户界面的 2048 游戏

blog.devgenius.io](https://blog.devgenius.io/make-a-command-line-game-with-julia-a408057adcfe) [](/jupyter-notebooks-can-be-a-pain-but-i-️-pluto-f47913c5c16d) [## 朱庇特笔记本可能是一种痛苦，但我❤️布鲁托

### 朱莉娅的 Pluto.jl 将使教育变得更好

towardsdatascience.com](/jupyter-notebooks-can-be-a-pain-but-i-️-pluto-f47913c5c16d) 

# 保持天真

我们现在拥有的代码可以工作，但是它非常慢，尤其是对于较大的数字。所以本着保持天真的精神，让我们试着实现我们自己的这个函数的记忆版本。

记忆只不过是使用字典(或类似的缓存)来存储结果，而不是重新计算`F(3)`无数次，而是从字典中查找。这是如何工作的:

这些步骤是:

1.  检查我们的内存中是否已经有了`n`的结果，如果有，快乐的日子。
2.  如果没有，计算它并立即存储在内存中。
3.  返回计算结果。

结果更好:

```
julia> [@btime](http://twitter.com/btime) fibonacci_memory(30)
  1.630 μs (7 allocations: 1.97 KiB)
832040
```

这是 0.00163 ms vs 之前的 2.28ms。那就是 **1400x 加速**！😜

# 不要多此一举

![](img/d83f9b20d51fcfefd4d4f7f6b5657b69.png)

乔纳森·肯珀在 [Unsplash](https://unsplash.com/s/photos/stone-wheel?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

尽管编写这个版本并不太复杂，但是有更好的方法。也就是说，`Memoize.jl`包确实如其名所言。使用起来超级简单。用`Pkg.add("Memoize")`或`] add Memoize`以通常的方式安装它，然后你就可以用一个很棒的小**宏来帮你记忆你的功能**:

这和我们在第一部分中的函数是一样的，除了它现在被包装在`@memoize`宏中。这是多么容易。

让我们看看它有多快:

```
julia> [@btime](http://twitter.com/btime) fibonacci_easy(30)
  58.189 ns (0 allocations: 0 bytes)
83204
```

等一下！这比以前快了 30 倍。这怎么可能呢？🧐

实际上，这个记忆函数存储了所有以前的结果，所以现在它根本不做任何计算。从字面上看，这就是从字典中查找结果。您可以使用包装中的`memoize_cache`功能查看字典:

```
julia> memoize_cache(fibonacci_easy)
IdDict{Any, Any} with 30 entries:
  (7,)  => 13
  (29,) => 514229
  (25,) => 75025
  (21,) => 10946
  (15,) => 610
  (4,)  => 3
  (13,) => 233
  (26,) => 121393
  (18,) => 2584
  (10,) => 55
  (2,)  => 1
  (27,) => 196418
  (20,) => 6765
  (16,) => 987
  (5,)  => 5
  (8,)  => 21
  (28,) => 317811
  (24,) => 46368
  (12,) => 144
  ⋮     =>
```

如你所见，它包含 30 个条目，所以它已经学习了前 30 个斐波那契数列。

如果您的函数需要大量不同的输入，并且您担心内存可能是一个问题，您可能想要将缓存切换到**最近最少使用的(LRU)缓存**。Memoize.jl 库的[主自述文件中有一个这样的例子。](https://github.com/JuliaCollections/Memoize.jl)

# 为了完全不同的东西… 🥁

为了更好地衡量，这里有一个斐波那契生成器，没有任何记忆，也没有递归:

以下是一些证据，证明这也能实现所有其他功能:

```
julia> for i in 1:10
           [@show](http://twitter.com/show) i, fibonacci(i), fib(i)
       end
(i, fibonacci(i), fib(i)) = (1, 1, 1)
(i, fibonacci(i), fib(i)) = (2, 1, 1)
(i, fibonacci(i), fib(i)) = (3, 2, 2)
(i, fibonacci(i), fib(i)) = (4, 3, 3)
(i, fibonacci(i), fib(i)) = (5, 5, 5)
(i, fibonacci(i), fib(i)) = (6, 8, 8)
(i, fibonacci(i), fib(i)) = (7, 13, 13)
(i, fibonacci(i), fib(i)) = (8, 21, 21)
(i, fibonacci(i), fib(i)) = (9, 34, 34)
(i, fibonacci(i), fib(i)) = (10, 55, 55)
```

![](img/e628b3ceeb2667c1d3fa09e6fc928071.png)

夏洛特·科内比尔在 [Unsplash](https://unsplash.com/s/photos/fast?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

哦，天啊，太快了:

```
julia> [@btime](http://twitter.com/btime) fib(30)
  2.000 ns (0 allocations: 0 bytes)
832040
```

这比在内存中查找结果快 30 倍。哦，茱莉亚，你是头野兽！😆

想要更多的斐波纳契还是渴望更多的茱莉亚？看看这些:

[](/fibonacci-series-with-user-defined-functions-in-bigquery-f72e3e360ce6) [## BigQuery 中的斐波那契数列

### 在 BigQuery 中使用用户定义的 JavaScript 函数来计算 Fibonacci

towardsdatascience.com](/fibonacci-series-with-user-defined-functions-in-bigquery-f72e3e360ce6) [](/vectorize-everything-with-julia-ad04a1696944) [## 向量化朱莉娅的一切

### 告别 for loops，广播所有的东西

towardsdatascience.com](/vectorize-everything-with-julia-ad04a1696944) 

# 摘要

如果您有一个函数可以从记住自己计算的中间结果中受益，那么使用同一个包中的`@memoize`函数。它会让你的功能运行得更快，而且超级容易使用。

如果你想要一种有意义的、易于使用的、快速的编程语言——即使是 for 循环——就用 Julia 吧！

> *这里用到的所有代码都可以在 GitHub 上【https://github.com/niczky12/medium】[](https://github.com/niczky12/medium/blob/master/julia/memoize.jl)*下找到**

> *要获得所有媒体文章的完整访问权限，包括我的文章，请考虑在此订阅。*