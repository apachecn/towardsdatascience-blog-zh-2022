# 我在 Julia 中创造性地处理大数据的新大脑方法

> 原文：<https://towardsdatascience.com/my-new-big-brain-way-to-handle-big-data-creatively-in-julia-57be77fc6a04>

# 我在 Julia 中创造性地处理大数据的新大脑方法

## 我想出了为了应用表达式来压缩内存中的大量数据，让我们来看看吧

![](img/2e999858c232b6a981fd758ab360ffc6.png)

(图片由 [Pixabay](https://pixabay.com/images/id-3017071/) 上的 [DamianNiolet](https://pixabay.com/users/damianniolet-7379422/) 拍摄)

# 简介——记忆问题

随着更新的编程语言 Julia 速度的提高，似乎一切皆有可能。当然，有些事情是无法做到的，无论人们试图在 Julia 中采用什么方法，我的意思是很难想象这种语言有像 FORTRAN 这样的数字能力。然而，这种语言因为如此高级而变得非常非常接近，这真的令人印象深刻。

我现在发现我的硬件限制已经发生了巨大的变化。过去的情况是，虽然我有一个支持 CUDA 的显卡和一个相对较好的 8 核处理器，但像 Python 这样的语言确实没有利用这一点。通常，当处理相当大的数据集时，Python 的核心问题就开始出现了。首先，Python 并没有真正考虑到处理器/显卡并行计算。怎么可能呢？在最初设计这种语言的时候，最好的图形卡可能是由 VOODOO 制作的，像 OpenGL 这样的图形 API 还处于起步阶段。这是使用像 Julia 这样的现代语言的好处的一部分，他们能够在头脑中计划语言的设计。甚至仅仅使用处理器的并行计算对于 Python 语言来说也是相对较新的。

考虑到这一点，我现在发现，当涉及到数据和建模时，最大的瓶颈是内存，至少对于我的系统是这样，我想其他人也是这样。当然，其中一些肯定与我的计算机的规格有关，我也将与您分享我的硬件描述符:

![](img/2adc30f0c361e1f924db1b766073997b.png)

(图片由作者提供，我也非常确定我的处理器上的 8 个核心更像 AMD 推土机/piledriver 核心，它们是虚拟的，而有 4 个物理核心。)

在 2022 年，8g 内存是一个相当低的数量，但通常这不是一个很大的障碍，直到它可能成为其他人的一个大障碍。事实上，朱莉娅宠坏了我们。我知道我可以通过一些东西传递 5000 万个观察结果，没有问题，没有评论，也没有来自我的处理器 Julia 的顾虑，没问题。然而，我经常碰到的是记忆的极限。

也就是说，我想探索一些将整个特性的观察结果分解成某种“规范形式”的想法，并且我开始研究这些主题。我在保存记忆的方法方面的发现非常酷，所以我认为这可能是一个有趣的阅读，看看我学到了什么，另外还有一个我自己想出的非常好的主意。所有这些代码都是我的项目 OddFrames.jl 的一部分，odd frames . JL 只是 Dataframes.jl 的一个替代品，具有更多的功能，我几乎准备好发布这个包了。如果你想了解更多，或者浏览这个包，这里有一个 Github 页面的链接:

[](https://github.com/ChifiSource/OddFrames.jl) [## GitHub-chifi source/odd frames . JL:Julia 的独特数据管理平台

### OddFrames.jl 是一个用于在 Julia 中管理数据的新包。轻到足以成为任何项目的依赖，但丰富…

github.com](https://github.com/ChifiSource/OddFrames.jl) 

# 代数表达式

当我探索像这样的科学应用程序的数据压缩概念时，我遇到了惰性数组的概念。我当然知道懒惰类型的存在，但并没有真正意识到这种类型的含义和能力。我还写了一篇关于这些类型的数组的文章，以及为什么它们如此酷，但在涉及到通用数据科学应用时也有一些问题。以下是我最初文章的链接:

[](/lazy-arrays-and-their-potential-applications-in-data-science-d1f34e8657f6) [## 惰性数组及其在数据科学中的潜在应用

### 什么是惰性数组，如何在数据科学环境中使用它们？

towardsdatascience.com](/lazy-arrays-and-their-potential-applications-in-data-science-d1f34e8657f6) 

自从那篇文章之后，我进一步发展了惰性数组的实现。构造函数现在是基于函数参数的。Julia 有一个惰性数组包，但是构造函数语法并不是我想要的，这更像是个人的事情，但是我最终还是在我的实现中添加了一些新函数，使得这个选择更有意义。此外，我还将随后的所有代码复制到一个笔记本中，这样我的数据科学读者就可以更容易地接触和测试这些代码，如果您愿意的话，Github 上有这样一个笔记本:

[https://medium.com/r/?URL = https % 3A % 2F % 2f github . com % 2 femmettgb % 2 femmetts-DS-NoteBooks % 2f blob % 2f master % 2f Julia % 2f gebraic % 2520 arrays . ipynb](https://github.com/emmettgb/Emmetts-DS-NoteBooks/blob/master/Julia/Algebraic%20Arrays.ipynb)

现在，相对来说，这些阵列相当健壮。我还希望这些类型能够适合另一种类型，这一点我们将在后面讨论(这可能是这个实现真正酷的原因。)新函数包括迭代能力、添加到计算表达式的能力，以及通过 compute()方法按索引进行计算的能力——现在由 getindex 调用。这个项目的代码都可以在下面的 OddFrames.jl 库中找到。让我们看看我的这个类型的实现，现在称为[代数数组](https://github.com/ChifiSource/OddFrames.jl/commit/a6b5850546ff04beaed245f3abe441a3f18eddd7)，在代码中是什么样子的。

```
mutable struct AlgebraicArray
    **f**::Function
    **n**::Int64
    **calls**::Vector{Pair{Function, Tuple}}
    function AlgebraicArray(f::Function, n::Int64)
        new(f, n, [])
    end
end
```

构造函数非常简单，并不需要太多的基类型。在我看来，这更像是一个大纲——我们可以用它作为模板来构建一些非常独特的类型，我们将在后面看到。我之所以将其描述为模板，是因为在大多数情况下，测量值需要保持连续的线性，才能从中获益。在大多数情况下，对真实世界的数据做类似的事情将需要系数，这违背了整个目的-因为存储表达式和东西最终只会占用更多的内存，而系数占用相同的内存量。这是一个相当有趣的问题。

这个结构有三个字段，上面用粗体突出显示，第一个是函数 f。这个函数只是我们使用的任何类型的数据的生成器。每当调用另一个方法 compute()时，首先调用它。最后一个字段是 calls 字段，它是数组中的一系列对，表示函数调用和附加的位置参数。需要注意的重要一点是，这些函数是在迭代计算每个值时调用的，**不是整个数组。同样，这些应该是值可变的，所以值应该在所有表达式中放在第一位。下面是生成这些对的实际函数:**

```
function **add_algebra(aa::Symbol, args::Any)**
    aa = eval(aa)
    if length(args) > 1
        farguments = Tuple(eval(args[length(args)]))
        **push!(aa.calls, eval(args[1]) => farguments)**
    else
        **push!(aa.calls, eval(args[1]) => [])**
    end
end
```

记住不要强调这里的细节，只要记住我们的输入和输出。我们这里的输出是突变，通过推！aa 的，作为一个符号提供给我们。我还没有在一个模块的作用域而不仅仅是一个主作用域中测试这个特性，因为这个特性还在构建中，所以对 aa 的评估可能会导致该符号不存在，在这种情况下，我只需要使用一点内省技巧。顺便说一句，你可以在我写的关于这个主题的文章中读到如何用元编程来做这件事:

[](/metaprogramming-in-julia-a-full-overview-2b4e811f1f77) [## Julia 中的元编程:全面概述

### 在 Julia 中用几种不同的技术看元编程

towardsdatascience.com](/metaprogramming-in-julia-a-full-overview-2b4e811f1f77) 

该函数提供了我们的数组作为一个符号，和参数，这应该是另一个符号子集。这几乎就是为我们生成每个单独调用的方法调用。接下来，我们将看看与此绑定的宏，也就是@代数！宏观。

```
**macro algebraic!(exp::Expr)**
    args = copy(exp.args)
    aa = exp.args[2]
    deleteat!(args, 2)
    **add_algebra(aa, args)**
**end**
```

这里非常简单，宏只分离我们提供给方法的第一个参数，它应该是我们的代数数组，然后把它和其他参数一起传递给 add_algebra。下一个函数有点不相关，但非常简单，我们将看到所有这些是如何立即结束的。此方法只是基本生成器，它使用 generator 函数生成值，直到数组的长度:

```
function generate(aa::AlgebraicArray)
    **[aa.f(n) for n in 1:aa.n]**
end
```

这个函数也有接受整数、范围甚至 bitarray 的绑定——尽管我在让 bitarray 工作时遇到了一些麻烦，尽管我认为这很容易，因为在这个数组上有比一个索引类型更重要的东西要处理。

```
function generate(aa::AlgebraicArray, range::UnitRange)
    if range[2] > aa.n
        throw(BoundsError(string("Invalid algebraic index, ", string(range[2],
                        " on algebraic expression of length ", string(aa.n)))))
    end
    **[aa.f(n) for n in range]**
end**function generate(aa::AlgebraicArray, index::Integer)**
    if index > aa.n
        throw(BoundsError(string("Invalid algebraic index, ", string(range[2],
                        " on algebraic expression of length ", string(aa.n)))))
    end
    **aa.f(index)[1]**
end
```

由于生成器函数通常会返回一个值数组，这就是我们用 1 来索引它的原因。这些调用之间唯一真正改变的是函数调用的最后一部分。我们要看的最后一个函数是计算函数:

```
function compute(aa::AlgebraicArray)
    **gen = generate(aa)**
    for call in **aa.calls**
        **gen = [call[1](val, call[2]...) for val in gen]**
    end
    **return(gen)**
end
```

这个函数将所有其他函数包装成一个，首先它生成我们的基值，然后将 aa.calls 中的所有后续调用应用于其中的每个值。OddFrames.jl、bitarray、range 和 index 中的所有常规索引也有一个绑定。

```
function compute(aa::AlgebraicArray, **r::Any**) # <- range, bitarray
    gen = generate(aa, r)
    for call in aa.calls
        gen = [call[1](val, call[2]...) for val in gen]
    end
    return(gen)
end
```

由于 generate 能够为我们指明哪个是哪个，并且这两个函数都将返回一个具有多个元素的数组，所以我在这里传递 Any 以避免必须编写两次该函数，因为除了 generate 中的调用之外，所有调用都是相同的。这与整数略有不同，因为我们不需要遍历 gen 中的值，如果我们试图迭代一个整数，我们会得到一个 MethodError。我相信你可以假设这个函数和一个单一指数的函数之间的细微差别。最后，还有一个函数的调度调用，然后是带有 compute()的代数数组。有趣的是，一个随机的行内注释告诉你这个函数实际上是在什么上下文中使用的:

```
function compute(f::Function, aa::AlgebraicArray) **# compute(aa) do _**
    gen = generate(aa)
    for call in aa.calls
        gen = [call[1](gen, call[2]...)]
    end
    return(f(gen))
end
```

这允许我们加载数组而不将它保存在内存中的语法。例如，让我们看一个均值函数，它将计算这种类型的均值。考虑一下，将整个值加载到内存中，尤其是在全局范围内，可能会很成问题。例如，如果我们想使用来自车床. jl 的 mean()函数，我们当然可以这样做

```
mu = mean(compute(aa))
```

然而，这将所有这些加载到内存中，并且都是在 mean 的上下文中。结果可能会被缓存，我们并不真正控制内存中代数数组的状态，我们把这一切都交给了垃圾收集器。如果我们做了

```
mu = compute(aa) do array
   mean(array)
end
```

然后我们从一个全新的、临时的、匿名的函数作用域中获取这个值。此外，我们可以一次应用更多的操作，而不必为每个操作计算 aa，或者将 aa 加载到我们的全局环境中。在像这样的高级动态语言中，一旦数据被全局定义，唯一的处理方法就是隐式的。在朱莉娅时代，我们已经删除了！()，很容易看出为什么这样的函数现在被弃用了。请记住，在我们的环境中管理这些值的内存状态的最佳方式是，首先不要将它们放入环境中，而是只私下处理这些值。

当然，在这种类型上仍然存在 getindex()的绑定。同样，iterate()也有一个绑定，它只是将生成的数组的值除以 1。当它离成为常规的可迭代数组只有一两步之遥时，这是一种使简单迭代工作的简便方法。因为 compute 和 generate 函数已经可以通过 index 进行计算或生成，所以索引就像通过 compute()函数绑定任何传递的索引一样简单。虽然这造成了 MethodErrors 的缺点，有点令人困惑，但由于 OddFrames 中的索引在扩展基本 Julia 类型索引的程度上是通用的，这在整个包中是一致的，因此在这里假设参数很容易。

```
function iterate(aa::AlgebraicArray)
    ret = Iterators.partition(compute(aa), 1)
endgetindex(aa::AlgebraicArray, i::Any) = compute(aa, i)
```

# 履行

现在我们已经看完了我的代数数组，让我们看看它在类型中的实现。让我们来看看一种新的 OddFrame 类型，[代数编码帧](https://github.com/ChifiSource/OddFrames.jl/commit/ed7c32f8396814b5c255cb7bfa5c83a616fb132e):

```
mutable struct AlgebraicOddFrame <: AbstractAlgebraOddFrame
        labels::Array{Symbol}
        **columns::Vector{AlgebraicArray}**
        types::Array{Type}
        head::Function
        dtype::Function
        not::Function
        only::Function
        drop!::Function
        dtype!::Function
        merge!::Function
        only!::Function
        **compute::Function**
        # Super
        function AlgebraicOddFrame(labels::Vector{Symbol},
                columns::Vector{AlgebraicArray},
                types::AbstractArray)
                length_check(columns)
                name_check(labels)
                head, dtype, not, only = member_immutables(labels, columns,
                                                                types)
                drop!, dtype!, merge!, only! = member_mutables(labels,
                columns, types)
compute() = OddFrame([label[i] => compute(columns[i]) for i in enumerate(labels)])
 **compute(;at = 1) = [label[at] => compute(columns[at])]
                compute(r;at = 1) = [label[at] => compute(columns[at], r)]**
**compute(r = 1:columns[1].n) = OddFrame([label[i] => compute(columns[i], r) for i in enumerate(labels)])**
                new(labels, columns, types, head, dtype, not, only, drop!,
                dtype!, merge!, only!, compute)
        end
```

> 我知道代码很多。

这里我们真正需要注意的是附加函数 compute()和列的新数据类型。我想我可能会删除这里除了 compute 之外的所有成员函数。这是因为在这个上下文中，我可能会为 compute()函数绑定一个类似的函数。换句话说，由于 compute()返回一个常规的 OddFrame，我认为只使用

```
aod.compute() do od
   od.whatever()end
```

也就是说，head()等函数除外。我可能会将 compute()定义导出到一个类似的 algebra_members()函数，该函数也可能会为此类事情提供绑定。真的，当涉及到 head 这样的函数时，只需要做微小的改变，可能不是对函数本身，因为通常这些只调用索引。

虽然我们可以通过它的构造函数直接调用 OddFrame 的代数版本，但我认为真正的力量将来自于根本不这样做。相反，对于这种类型的一些有趣的构造函数，我有两个命题想法，第一个使用机器学习。你看，问题在于，像这样的大量数据很难浓缩成一个单一的数学表达式。正如我前面提到的，通常这方面的任何事情都需要使用某种系数，这完全消除了试图将值存储在更少数据中的目的。

## MLOddFrame

这时候我突然想到。整个数据科学领域都是关于预测事物的，虽然数学表达式(如用于代数编码框架类型的函数)是精确的，但在许多情况下，只预测重复的值可能是有意义的。这让我想到了我的第一个真正高级的实现，MLOddFrame。到目前为止，这已经写入 OddFrames.jl 包中，尽管它实际上并不做任何事情，而且我还为它写了一个构造函数——这是一个奇怪的选择。相反，我希望有另一个完整的包，一个扩展，这是唯一的 MLOddFrame。这使得管理预编译时间变得容易得多，因为不是每个人都想在预编译数据管理包的同时预编译机器学习包。

同样，我们真的不需要定义任何类型。我们可以使用闭包函数来维护任何类型的名称定义，在实际计算 OddFrame 时，我们可能会用到这些定义。有趣的是，我实际上写了一整篇关于这个概念的文章，其中我谈到了这个令人不安的细节。如果你对这样一篇文章感兴趣，这里也有一个链接:

[](/what-on-earth-are-closures-a4f9c7c652d2) [## 闭包到底是什么？

### 闭包函数及其使用方法概述

towardsdatascience.com](/what-on-earth-are-closures-a4f9c7c652d2) 

这是一个一般的想法，因为在 OddFrames 中已经有了对一个代数编码帧的绑定，我们将只在一个不同的别名下为一个代数编码帧创建一个构造函数。在这个构造函数中，我们允许传递一个参数来决定要训练多少数据，以及我们想要使用什么模型。如果我们使用 Lathe.jl，我们可以通过这些参数传递整个管道。

```
function MLOddFrame(catmodel::Any, conmodel::Any; split = .25)end
```

当然，需要有一个特征类型的模型，我也有可能在这种情况下使用框架组类型，因为我们可以说——评估准确性，并决定我们是否应该使用模型来预测数据。在某些情况下，我们可能不想这样做。可能应该有一个参数来决定是否这样做，或者可能有一个完全独立的构造函数，因为它会极大地改变这个方法的返回。这将数据的维度降低到预测它的权重，这肯定会大大降低。

那么第二部分就像知道数据总长度一样简单，这将取决于我们正在构建的上下文。最后，最后一部分只是一个函数，它的定义将在这个函数中进行，保留函数中所有已定义的名称，比如我们的模型。最后，将该函数与 n 一起返回到一个新的代数编码框架中。

## StreamOddFrame

StreamOddFrame 将遵循 MLOddFrame 中相同的原则，它是基本 OddFrame 的扩展，并使用闭包函数来保存名称。唯一的区别是，StreamOddFrame 将有一个直接从数据文件中读取的函数。例如，我们有一个包含超过 500，000，000 百万个观察值的 CSV 文件，我们的函数根据需要将每一行读入 OddFrame。此外，我确信类似这样的事情也可以用其他流来完成，比如 HTTP 流，这可以打开使用套接字创建 OddFrames 的远程客户端实例的可能性。

# 优点，缺点

这里最明显的优势就是我之前提到的，内存。这里的工作是有效地将大部分(如果不是全部)数据保留在内存之外，特别是在全局范围内，但仍然允许您访问所有数据，就像数据在内存中和全局范围内一样。这样做的明显缺点和问题是计算。然而，有了这个交换，我认为用 Julia 写的这个肯定有它的应用。

此外，我们可以序列化这样的数据，并引用其他地方的所有计算，这一事实非常有价值。有时，您可以计算没有终端超时的东西，这肯定会杀死您的 Jupyter 内核，但显然 Jupyter 内核的便利性很高，作为数据科学家有非常好的理由使用它。最后，一旦这个包发布，引用远程计算肯定是我将作为扩展写入的内容。也就是说，虽然这有一些缺点，但我认为能够在任何地方引用这种计算的能力通常可以一起减轻或消除这些缺点。

MLOddFrames 的另一个显著缺点是使用 ML 预测数据可能会产生不准确的结果。同样，可能应该有某种方式来查看、过滤和调整特性是否以物理方式表示，作为其自身的样本或整体，或 MLOddFrame。幸运的是，当我实际编写所有这些内容时，OddFrames 支持在单个类型中包含多个框架，并且有自己方便的方法来处理这些类型。

# 结论

因此，我有一套将基于代数/惰性表达式的数组实际应用于现实世界数据的概念。虽然这是我相信我将要追求的两个想法，但我确信这类事情还有更大的潜力。例如，我们甚至可以有一个框架，在评估时通过请求来填充。因为核心思想是一个函数，它可以是一个闭包函数，实际上可能性是无限的。

感谢您阅读本文，我希望您发现其中的一些概念和想法令人兴奋。目前，我正致力于使 OddFrames.jl 接口作为一个整体，包括代数编码框架，更加健壮。我很快就要发布这个包了，这非常令人兴奋！不用说，一旦我真的完成了，这将是非常有用的。我真的为这个项目的结果感到兴奋，但主要是我只是兴奋地在我自己的项目中使用它的能力，一旦它是稳定的。