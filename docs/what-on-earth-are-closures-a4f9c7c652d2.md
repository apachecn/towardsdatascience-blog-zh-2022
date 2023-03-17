# 闭包到底是什么？

> 原文：<https://towardsdatascience.com/what-on-earth-are-closures-a4f9c7c652d2>

# 闭包到底是什么？

## 闭包函数及其使用方法概述

![](img/a74b307f2fd6b4b9d8c7b7ba2e4e0985.png)

(图片由 Pixabay 上的 [jarmoluk 提供)](https://pixabay.com/images/id-428549/)

# 介绍

如果有一样东西对每一种编程语言都很重要，不管那种语言的范例是什么，那就是函数。函数是从一种语言传到另一种语言的普遍概念，这很有意义。甚至当我们用汇编语言进行低级编程时，我们仍然在使用本质上的函数。虽然我不确定它们是否一定被称为函数，因为我一直认为它们是子例程(或者是例程，如果它是 jmped 并且没有被调用的话)。如果你不知道我刚才到底在说什么，不要担心…

> 这与闭包无关。

我的观点是:函数在编程中是通用的。无论何时你开始学习一门新的编程语言，你首先应该了解的事情之一就是如何用这门语言编写函数。在某些情况下，函数也可以非常灵活，当然这因语言而异。在某些情况下，某些语法是可能的，而在其他情况下则不可能。计算机编程中一个惊人强大的概念是闭包函数的概念。今天我想回顾一下闭包函数到底是什么，以及为什么我认为它们是非常有用的技术。正如长期读者所料，今天我将使用 Julia，但是闭包函数在许多其他语言中也是完全可能的，包括(但不限于):

*   计算机编程语言
*   C
*   C++

哦，对了，还有 SML，一种我只是模糊地熟悉的语言；不过，这种实现可能有点过火，因为到处都有函数定义。此外，如果您想看我努力安装这种语言，然后在编程时努力完成每一步，甚至到了我不知道如何将两个整数相除的地步，我实际上已经写了一篇关于尝试这种语言的文章，您可以在这里阅读:

[](/getting-introduced-to-sml-a-weird-polymorphic-language-originated-in-1983-797ff9d6622e) [## SML 入门:一种起源于 1983 年的怪异多态语言

### 尝试用 SML 语言编写和编译一些基本代码

towardsdatascience.com](/getting-introduced-to-sml-a-weird-polymorphic-language-originated-in-1983-797ff9d6622e) 

> 请不要嘲笑我的绝望。

# 什么是闭包函数？

为了开始理解为什么闭包函数如此重要——特别是对于数据科学，我们可能应该从定义一个闭包函数开始。如果你在谷歌上搜索这个词，你可能会得到类似于

> "闭包是与第一类函数绑定的词汇范围的名称."

如果你不理解什么是词法范围，或者什么是一级函数，那么这种描述是没有用的。幸运的是，这些只是计算机编程中简单概念的大名字。好吧，也许这些概念在引擎盖下不那么容易理解，但是通过查看代码比通过描述更容易理解。词汇范围意味着函数能够访问传统上不可用的范围。考虑典型的功能范围，

```
function example_of_scope(x, y)
    mu = mean([x, y])
    x += 5end
```

函数定义有一个**私有**范围。它可以访问它上面的作用域，在这个例子中是全局作用域，在 Julia 中是“Main”这是因为首先它是该作用域的子作用域，其次全局模块 Main 的作用域总是公共的。也就是说，如果我们移动到这个函数下面，并试图调用这个函数的值，我们会得到一个错误:

```
function example_of_scope(x, y)
    mu = mean([x, y])
    x += 5end
```

范围从顶级范围(在本例中是主模块)一直限制到我们的函数，每个部分包含越来越多的定义。例如，假设我们在 Main 中有一个名为“Tacos”的模块来保存我们的函数，那么我们的作用域应该是这样的:

```
module Main
# |Main scope
  | module Tacos
   # Tacos scope
     |
      function example_of_scope(x,y)...end
```

这是词法作用域，所以基本上——你可能用过的大多数语言都使用词法作用域，另一种方法是动态作用域，这通常意味着所有的名称绑定都是全局的。我们可以从 example_of_scope 访问 Main 和 Tacos 中定义的定义，但是不能从 Tacos 或 Main 访问 example_of_scope(x，y)中定义的定义。请记住，这些术语通常特指函数，在大多数情况下，使用一种非词汇范围的语言可能会很烦人。

这是拼图的一部分，那么什么是一级函数呢？第一类函数只是一个可以被当作参数的函数。在 Julia 中，我们可以很容易地传递 type::函数，并将函数作为参数。

```
example(f::Function, x::Int64) = f(x)
```

闭包函数是词法范围和更像常规类型对待函数的组合。在 Julia 中，这是非常流畅地完成的，并且使用了一些非常容易识别的语法，所以它实际上是一种很好的语言来演示这一点。

```
function example2(n)
    h = function hello()
        blahblahblah
    endend
```

所以现在我们有了 h 的函数作用域，它是在 example2 的函数作用域内定义的。记住，闭包函数实际上被称为闭包函数的一个重要部分是，这个作用域需要是词法性的。这意味着 hello 可以访问 example2 的范围。来自所述范围的所有值都保存在函数的定义中。例如，我们可以从 hello 函数中访问 n。

```
function example2(n)
    h = function hello()
        println(n)
    end h
endourfunc = example2(5)
ourfunc()5
```

闭包函数也是我的包 OddFrames.jl 中面向对象语法的主干，如果您想查看在该包中使用它们的上下文，可以在这里查看构造函数的源代码:

[](https://github.com/ChifiSource/OddFrames.jl/blob/last-0.0.7-patch/src/type/frame.jl) [## OddFrames.jl/frame.jl 终于-0 . 0 . 7-补丁 ChifiSource/OddFrames.jl

### 此文件包含双向 Unicode 文本，其解释或编译可能与下面显示的不同…

github.com](https://github.com/ChifiSource/OddFrames.jl/blob/last-0.0.7-patch/src/type/frame.jl) 

# 不同语言的实现

现在我们已经了解了闭包函数的基本知识，以及它们的技术定义，让我们看一些用不同语言实现闭包函数的例子。我们列表中的第一个是 Python，这有点奇怪。与 Java 类似，Python 使用 lambda 语法来定义匿名函数，这非常方便。只要想想作为一名数据科学家，您在另一个函数中编写匿名函数的频率，就可以看出这些函数有多有用。另外，我将要展示的所有代码都可以在我的随机代码库中的文件中找到，这里有一个链接:

[](https://github.com/emmettgb/Random_Code/tree/main/closure_examples) [## 主 emmettgb/Random_Code 上的 Random_Code/closure_examples

### 只是一堆随机的斑点。在 GitHub 上创建一个帐户，为 emmettgb/Random_Code 开发做贡献。

github.com](https://github.com/emmettgb/Random_Code/tree/main/closure_examples) 

无论如何，这是我们之前用 Python 复制的例子:

```
def example2(n):
    h = lambda : print(n) return(h)ourfunc = example2(5)
ourfunc()5
```

这个例子很容易理解。我们用 lambda 创建函数 h，然后返回它。Python 让这变得非常简单，只需理解 lambda 就能理解这种语法。顺便说一句，如果你对 lambda 没有很好的理解，我确实写了一整篇关于它的文章！这里有个链接！：

[](/scientific-python-with-lambda-b207b1ddfcd1) [## 带 Lambda 的科学 Python

### Python Lambda 函数的正确用法:Python 科学编程的最佳语法。

towardsdatascience.com](/scientific-python-with-lambda-b207b1ddfcd1) 

基本上，lambda 为 Python 做的就是允许语言创建匿名函数。朱莉娅也有这种能力，可能有更多的方法来创造它们，但这里有两个例子:

```
f = () -> 5
f()5begin
     5end
```

begin 块将把函数返回给放在它前面的任何东西，这是一件需要注意的重要事情。我有一整篇文章是关于匿名函数的概念的，如果你想详细说明这一点，当在另一个函数的上下文中使用时，匿名函数本身通常是闭包:

[https://towards data science . com/what-on-earth-is-an-anonymous-function-f 8043 EB 845 f 3](/what-on-earth-is-an-anonymous-function-f8043eb845f3)

不幸的是，当谈到在 C 中做同样的事情时，我们会很快意识到，确定返回类型有些困难。然而，这可以通过使用 [GNU FFCALL](https://savannah.gnu.org/projects/libffcall/) 变得简单一些。然而，有一种方法不需要 FFCALL 就可以实现，而且也很简单，最大的问题是获取返回类型。这是我们的函数，这次是用 C 语言复制的:

```
#include <stdio.h>
typedef int (*fptr)();fptr example2 (int n)
{
  void h ()
    { printf("%d", n); }
    return h;
}int main()
{
    fptr h = example2(5);
    h();
    return 0;
}
```

请允许我在这里解释一下代码。我们首先创建一个函数指针，*fptr，然后我们使用它作为返回类型，所以我们返回一个指向函数 h 的指针。这样做很好，并且有效地在 C 中创建了一个闭包。有趣的是，很多人使用苹果的 C 编译器的“block”语法，并使用 FFCALL 甚至 GCC 扩展来实现这个指针定义可以实现的功能。我不确定是否有必要更进一步，因为其他语言中的许多例子都将与前两个非常相似，只是要么有更多的语法，要么有不同的调用结构(除了像 SML 这样的函数式语言，但是……**没有**)。)

# 真实世界的用例

既然我们已经不再将闭包视为一个概念，那么请允许我向您展示一个实例，我在 Julia 中使用闭包做了一些非常棒的事情。我将要展示的代码是我的 Toolips.jl 项目的一部分。这个项目现在还很早，因为我现在有太多事情要做，而且我在空闲时间还在做的其他开源包的数量**多得令人应接不暇。**该项目旨在成为一个模块化的网络开发框架，既作为后端工具又作为前端工具，但通过元编程功能 JavaScript 完成所有前端工作。无论如何，如果你有兴趣观看这个项目或者给它一颗星(我很感激！)下面是 Github 资源库的链接:

[](https://github.com/ChifiSource/Toolips.jl) [## GitHub - ChifiSource/Toolips.jl:一个基于 Julia 的 JavaScript 兼容性 Web 开发框架

### toolips.js 构建在一个功能性编程的反应式可观察库之上，该库通过 JavaScript 表达式进行评估…

github.com](https://github.com/ChifiSource/Toolips.jl) 

无论如何，我想用这个包来描绘两个核心思想，我认为它们都是网络开发的好主意。

*   数据和回报应该功能化，
*   应该存储发电机类型，并为每个服务单独调用。

根据我在这里提出的第一点，函数被大量用作数据应该是很明显的。这里有一个与我在 OddFrames.jl 中使用的技术类似的技术，这里使用了一些私有方法，这些方法也是闭包函数，这就是我展示它们的原因:

```
mutable struct ToolipServer
    ip::String
    port::Integer
    routes::AbstractVector
    remove::Function
    add::Function
    start::Function
    function ToolipServer(ip::String, port::Int64)
        routes = []
        **add, remove, start = funcdefs(routes, ip, port)**
        new(ip, port, routes, remove, add, start)
    endfunction ToolipServer()
        port = 8001
        ip = "127.0.0.1"
        ToolipServer(ip, port)
    end
end
```

这里有很多要看的，但是重要的部分是我加粗的部分。该部分通过调用 funcdefs()创建闭包函数，如下所示:

```
function funcdefs(routes::AbstractVector, ip::String, port::Integer)
    add(r::Route) = push!(routes, r)
    remove(i::Int64) = deleteat!(routes, i)
    **start() = _start(routes, ip, port)**
    return(add, remove, start)
end
```

因为这些是闭包函数，在它们各自的定义中，我们能够使用调用它的原始函数中定义的所有值。这就是 ip、端口和路由都作为参数提供的原因。代码中引用我们下一个函数的部分再次以粗体显示，下面是这个函数:

```
function _start(routes::AbstractVector, ip::String, port::Integer)
    server = Sockets.listen(Sockets.InetAddr(parse(IPAddr, ip), port))
    println("Starting server on port ", string(port))
    **routefunc = generate_router(routes, server)**
    [**@async**](http://twitter.com/async) **HTTP.listen(routefunc, ip, port; server = server)**
    println("Successfully started Toolips server on port ", port, "\n")
    println("You may visit it now at http://" * string(ip) * ":" * string(port))
    return(server)
end
```

目前，这个服务器还很简单，但是你已经可以看到它的结构是如何发展的了。这里最重要的是 generate_router()调用和 HTTP.listen()调用。在 HTTP.listen()中，我们提供 routefunc 作为参数，它是 generate_router 的返回。单单这个名字 routefunc 就可以帮助人们设想它实际上是做什么的。routefunc 是函数的返回，而不是函数。这是因为函数不是静态的，它需要根据提供给它的数据而变化。下面是这种情况的具体表现:

```
function generate_router(routes::AbstractVector, server)
    route_paths = Dict([route.path => route.page for route in routes])
 **   routeserver = function serve(http)**
     HTTP.setheader(http, "Content-Type" => "text/html")
     full_path = split(http.message.target, '?')
     args = ""
     if length(full_path) > 1
         args = full_path[2]
     end
     if fullpath[1] ! in keys(route_paths)
         write(http, generate(route_paths["404"]), args)
     else
         write(http, generate(route_paths[fullpath[1]]), args)
     end
 end # serve()
 **   return(routeserver)**
end
```

我将 routeserver 的返回和定义都用粗体显示，以便于查看。该函数在每次发出请求时运行，HTTP I/O 流由 HTTP 作为该函数的参数提供。之后，这个函数所做的就是生成与某人请求的 uri 相对应的页面。从表面上看，它很简单，但从外表上看，它变得非常强大。我认为这是闭包函数的一个很好的用法，此外，我认为这是解决管理路线和事物问题的一个很酷的方法。我知道有些人可能会好奇为什么我决定使用这种技术，而不是使用 HTTP。路由器，或者更进一步说，为什么我想首先开发一个网络服务器。

要回答第一个问题，用 HTTP.Router 处理许多独立的路由要困难得多。此外，流的处理不可能像在这个上下文中那样干净，在某些情况下，定制流甚至根本不可能。为了回答第二个问题，老实说，在 Julia 的 web 开发框架方面，我们真的很有限。对于 Julia 中的这个任务，我们可以选择 Mux.jl 和 Genie.jl。以下是 Mux 的文档:

 [## Mux.jl

### Mux.jl 为您的 Julia web 服务提供了一些闭包。Mux 允许您根据高度模块化和…

docs.juliahub.com](https://docs.juliahub.com/Mux/cs9xb/0.7.4/) 

> 对，没错，就是这样。

有一段时间，Mux.jl 也被弃用了，我想——或者可能是其他什么包。无论如何，当谈到 web 交互性时，虽然我们有一个很好的 WebIO/Interact.jl 连接，但是 WebIO mimes 评估起来有点慢，这完全可以理解——但是很烦人。他们也更倾向于科学互动，老实说，这是我认为 Interact.jl 的归属。Interact 真的真的很擅长**那个**。这是软件包应用程序。虽然已经有一些网络应用程序，比如 Pluto.jl，是建立在这个基础上的，但是我认为 Interact 的目标从来就不是成为一个全功能网络应用程序的后端(或者可能是，我不确定？)

事实上，我是 Genie.jl 的早期用户。Genie 可能是目前 Julia 网络开发的最佳选择，只是有些事情我希望它有所不同。其中最大的问题是，整个文件系统是一个很难做到轻量级的系统。这让我想起了 Django 与 Flask 的争论，说用 Flask 制作 API 比用 Django 要容易得多，但用 Django 制作 web 应用程序比用 Flask 容易得多。Mux.jl 非常注重成为一个反应式前端工具，我认为它在那个应用程序中做得相当好。也就是说，我们没有任何真正微观的微观框架。也就是说，我试图创造这一点——以及一定程度的模块化，这将允许它建立在附加功能的基础上。

# 结论

在讨论 Toolips.jl 时，我们可能有点跑题了，但重点仍然存在，闭包函数非常有用。我认为它们对我如此有用的部分原因是因为 Julia 有非常好的语法表达式和函数语法。这使得把这些定义拿出来并很好地使用它们变得容易多了。此外，你可以在任何地方使用函数作为参数，计算和表达式化字符串，Julia 代码，等等，所有这些加起来就是一个非常健壮的接口来编写这种性质的函数。我也喜欢 Python 的实现，我确信作为数据科学家，我们以前都使用过 lambda，它在处理数据时非常方便，如果你曾经在另一个函数中使用过它，那么你实际上是在编写一个闭包函数。

我希望这篇文章对阐述闭包函数这个主题有所帮助。它们真的很酷，在很多不同的情况下肯定会派上用场。它们对于函数式编程也非常有用，我想这是一个我在这里没有过多涉及的话题，但是我认为它的含义是非常清楚的。总的来说，这肯定是需要了解并加以利用的。因此，我很高兴我能与你详细地分享它！感谢您阅读本文，祝您在将所有代码转化为函数的过程中愉快！