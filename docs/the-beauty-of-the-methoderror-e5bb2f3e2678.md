# 方法的妙处错误

> 原文：<https://towardsdatascience.com/the-beauty-of-the-methoderror-e5bb2f3e2678>

## 方法错误的概述，以及为什么它在 Julia 中如此有用

![](img/1c1e48c1889a1b44048db75ea92a47b0.png)

(图片由 [Pixabay](http://pixabay.com) 上的 [markusspiske](https://pixabay.com/images/id-1689066/) 拍摄)

# 介绍

抛出、错误、异常、退出代码，不管你想怎么称呼它们，它们在计算中有着巨大的用途。尽管在理想世界中，软件可以很好地应用于每一个用例及场景，但这在计算中并不现实。函数不处理特定的数据或输入是很常见的，这会导致问题。抛出是适当的，这样我们就能确切地知道哪里出错了，并且栈跟踪可以让我们追踪到任何问题的根源。

不管一个人犯了什么样的错误，返回一个实际的一致的错误输出，或者代码，或者任何东西，可以给开发人员很多关于哪里出错的上下文。此外，只要某种回报仍然有价值，就更容易知道到底哪里出了问题。我认为汇编如此令人讨厌和难以编写的最大原因之一是因为没有任何真正的抛出——所以我没有它们，相信我，没有它们调试是非常困难的。假设 Julia 是一种通常由最终用户直接使用的语言，并且很可能是一名程序员，在某种交互式的动态环境中与代码进行交互，那么 Julia 使用大量 throws 是很有意义的。今天，我想向您介绍一个 throw，它是 Julia 编程语言主干的基础。这当然是 MethodError，它在很多方面是多分派范例的事实错误。今天我想谈谈如何利用这个错误来弄清楚动态环境中的代码是如何工作的，并谈谈为什么这类错误在其他语言中实现会很棒，因为它们非常有用。

> [笔记本](https://github.com/emmettgb/Emmetts-DS-NoteBooks/blob/master/Julia/Beauty%20of%20method%20errors.ipynb)

# 方法错误

方法错误是一个核心的 Julia throw，它是语言核心思想的一个非常重要的组成部分。毕竟，该语言范式的整体思想是提供某些类型作为某些方法的参数。每当我们用错误的类型作为参数调用方法时，就会抛出方法错误。因为方法是通过多态定义的，所以无论何时用不正确的参数调用这些方法，我们都必须获得某种程度的返回。那么这些方法在技术上是不存在的。很棒的是，方法错误只会在你用错方法的时候出现。例如，如果我试图使用一个不存在任何类型的方法，我们会得到一个未定义的错误:

```
hrthtrh()UndefVarError: hrthtrh not defined
```

然而，如果我们叫推！()举个例子，用错误的参数，我们会得到我们光荣的方法错误:

```
push!()
MethodError: no method matching push!()
Closest candidates are:
  push!(::AbstractChannel, ::Any) at /opt/julia-1.6.3/julia-1.6.3/share/julia/base/channels.jl:10
  push!(::Set, ::Any) at /opt/julia-1.6.3/julia-1.6.3/share/julia/base/set.jl:59
  push!(::Base.InvasiveLinkedListSynchronized{T}, ::T) where T at /opt/julia-1.6.3/julia-1.6.3/share/julia/base/task.jl:570
```

方法错误真正伟大的地方在于，它实际上为方法提供了很多上下文，以及方法的能力。也就是说，这是在我们很好地理解了该方法的作用之后——以及 Julia 中的类型层次结构。实际上，我有一整篇关于我们目前正在使用的方法的文章，这可能会提供这个例子的一些背景。如果你想更深入地了解这一推！()方法，这里是那篇文章！：

[](/everything-you-need-to-know-about-push-in-julia-1f01891f1c0a) [## 关于 Push 你需要知道的一切！在朱莉娅

### 用力！()方法是你需要了解的 Julia 基础的重要部分。让我们来看看它能做什么。

towardsdatascience.com](/everything-you-need-to-know-about-push-in-julia-1f01891f1c0a) 

无论如何，这个输出的伟大之处在于它告诉我们到底哪里做错了。我们为方法调用提供了这组类型，但是只存在以下绑定。这很有意义，这个快速返回告诉我们，我们需要了解这些方法。我们基本上可以提供任何 iterable 和任何其他参数，以便做什么推！()会，它会将该值放入 iterable 中。在我们继续深入讨论这个问题之前，让我们写一些更简单的函数，以便把它变成白纸黑字。

```
function printthisthang(x::Int64)
    println(x)
end
```

因为这个方法只有一个参数，必须是 Int64。例如，如果我们提供一个 float，我们将得到:

```
printthisthang(5.5)MethodError: no method matching printthisthang(::Float64) Closest candidates are:   printthisthang(::Int64) at In[6]:1
```

方法错误的伟大之处在于它告诉我们到底哪里做错了。一旦你仔细分析了更具体的输出，它实际上说的是，我们提供了一个 Float64，但该函数的定义需要和 Int64 的唯一现有方法。我们确切地知道我们做错了什么，并且还可以从这个输出中了解关于这个函数的上下文的更多信息，这使得它成为一个巨大的错误—我们确切地知道如何修复我们所做的，并且我们确切地知道我们做了什么。更好的是，这些可以为扩展某些方法调用提供一些上下文，这些方法调用可能会被提取一点。一个很好的例子就是索引。让我们考虑这种新类型:

```
mutable struct OurCollection
    x
    function OurCollection(x::Any ...)
        new(x)
    end
end
```

如果我们试图索引它，我们会得到一个方法错误。但是，方法错误告诉我们是哪个函数导致了这个问题:

```
z = OurCollection(5, 10, 15)
z[2]
MethodError: no method matching getindex(::OurCollection, ::Int64)
```

所以我们基本上可以把它插入到一个新的方法中，然后很快地解决这个问题，这都是因为方法错误准确地告诉了我们该做什么。

```
import Base: getindex
getindex(x::OurCollection, i::Int64) = x.x[i]
z[2]10
```

我认为方法错误非常有价值。他们不仅能告诉你你的代码到底出了什么问题，还能提出一些可能的修复方法。从高层次的角度来看，这是很棒的，因为即使我们使用了完全错误的东西，我们仍然得到了所需的工具来修复我们搞砸的东西。此外，它可以为我们提供许多函数的上下文，我们可能需要运行文档的唯一内容是返回什么，或者所述函数的实际用途是什么。总而言之，我真的觉得其他语言应该实现方法错误。

我认为尤其是在 Python 中，方法错误很有意义。我最近写了一篇关于为什么应该用 Python 输入参数的文章，结论很简单，它有助于记录你的软件。如果有人提供了错误的类型，进行某种全面的分析会很好，而不是仅仅试图运行函数，然后每当方法不适用于某人错误提供的类型时就产生一些错误。如果您想阅读更多关于在 Python 中键入参数的内容，可以查看那篇文章:

[](/why-type-your-arguments-in-python-5bf24d7201eb) [## 为什么用 Python 输入你的参数？

### 为什么在 Python 中提供参数类型是有意义的一些令人信服的理由

towardsdatascience.com](/why-type-your-arguments-in-python-5bf24d7201eb) 

对我来说，在编程时考虑不同值的类型似乎更自然。这似乎更自然，因为类型是数据的一个非常重要的维度！很高兴看到这个错误被移植到我喜欢的其他语言中，因为我相信它非常有价值。当然朱利安版本是以上和超越，我只是说，这将是很好的人知道他们提供了什么类型的错误，在某种程度上类似于一个论点错误。感谢您阅读本文，我希望它主要有助于开发您的方法错误的用法来修复您的代码！