# 闭包在起作用:闭包函数作为类型的巨大威力

> 原文：<https://towardsdatascience.com/closures-in-action-the-dramatic-power-of-closure-functions-as-types-bf52be2b632e>

## 在 Julia 中，将函数视为正常类型的各种用法产生了一些相当有趣的结果。

![](img/06ed546944d02ff7f0a3b63766329fe3.png)

(图片由 [Pixabay](http://pixabay.com) 上的 [roketpik](https://pixabay.com/photos/code-programming-javascript-5113374/) 提供)

# 介绍

Julia 编程语言及其功能的一个非常强大的地方是对闭包的强大支持。也就是说，在将函数视为数据方面，Julia 是一种出色的语言。使用 dispatch 函数是完全可能的，因此会产生一些非常棒的语法。当涉及到使类型和函数协同工作以实现给定目标时，这种语法也有许多应用。如果您不熟悉闭包函数，这可能是本文信息的一个先决条件。幸运的是，我正好有一篇文章更详细地介绍了这些是什么，你可以在这里阅读:

</what-on-earth-are-closures-a4f9c7c652d2>  

另一篇文章提供了更多相关信息，介绍了这类事情在 Julia 中通常是如何通过匿名函数定义来实现的，在接下来的文章中也会有更详细的讨论:

</what-on-earth-is-an-anonymous-function-f8043eb845f3>  

如果代码是你正在寻找的，我在这个项目中使用的代码可以在 Github 的这个资源库中以[笔记本格式获得。](https://github.com/emmettgb/Emmetts-DS-NoteBooks/blob/master/Julia/Functions%20as%20types%20in%20Julia.ipynb)

# 快速复习

在 Julia 中，闭包函数定义处理得非常漂亮。通常情况下，函数和类型之间的差异是无法衡量的，除非我们想通过提供()将该函数用作函数。它总是含蓄的，这是一件伟大的事情。同样，函数也很容易定义，任何东西都可以等同于函数，例如我们可以这样定义:

```
n = function hello(x, y)
   println("hi") 
end
```

然后我们可以像调用 hello()的定义一样调用 n。

```
n(5, 1)
```

我们当然也可以内联定义函数，使得函数和传统构造的数据结构之间的区别更加抽象——这是一件很棒的事情。特别重要的是，这种实现也不会妨碍其他事情，而这类事情经常会发生。这是闭包赋予的基本能力，仅仅利用这种思想，就可以实现一些非常激进和强大的解决方案。

# 原始闭包实现

使用函数作为参数的第一个实现实际上将是一个非常酷的用例。在这个例子中，我们需要为我们的服务器制作一个路由器，因为默认的 HTTP。HTTP 包中的路由器类型只能在非异步的服务器上工作，至少据我所知是这样。无论哪种方式，这都可以被认为是一种更加手动的方式，实际上，人们可以围绕这个实现制作他们自己的 mimes 所以这非常酷。为了清楚起见，mime 是 Julia 中的 display()函数到常规 HTT 协议之间的转换层。这只是一种奇特的方式来表达我们如何把这个 Julia 翻译成 HTML。所有这些都可以通过闭包有效地完成。想想这可能会有多强大是有点意思的。让我们看看这在 Julia 代码中是什么样子的。

```
using HTTP
```

我从定义一些构造函数开始编写代码，使一些类型实际上保存我们可能希望在服务器上保存的大量信息。这些信息包括不同的部分，比如路径、页面的组成部分，以及我们希望通过 HTTP 在前端显示的所有信息。我们将从路由构造器开始:

```
mutable struct Route
    path::String
    page::Any
    function Route(path::String = "", page::Any = "")
        new(path, page)
    end
end
```

这是我们自己的独特类型，最著名的构造函数是 Route(::String，:Any)。也就是说，每当我们要创建一条路线时，我们都要注意这一点。接下来，我将为我们的服务器创建一个外部构造函数。我将使用闭包方法定义，并将它们保存在一个常规闭包实现之上的结构中，使用一个函数来控制传入的路由。

```
mutable struct OurServer
    ip::String
    port::Integer
    routes::AbstractVector
    remove::Function
    add::Function
    start::Function
```

考虑到这一点，我们需要首先查看为我们创建这些方法的支持函数，

```
function funcdefs(routes::AbstractVector, ip::String, port::Integer)
    add(r::Route) = push!(routes, r)
    remove(i::Int64) = deleteat!(routes, i)
    start() = _start(routes, ip, port)
    return(add, remove, start)
end
```

这些新功能只是添加和删除路由以及启动服务器。记住，start 只是开始为 HTTP 服务器提供服务。

```
function _start(routes::AbstractVector, ip::String, port::Integer)
    server = Sockets.listen(Sockets.InetAddr(parse(IPAddr, ip), port))
    println("Starting server on port ", string(port))
    routefunc = generate_router(routes, server)
    [@async](http://twitter.com/async) HTTP.listen(routefunc, ip, port; server = server)
    println("Successfully started server on port ", port, "\n")
    println("You may visit it now at http://" * string(ip) * ":" * string(port))
    return(server)
end
```

虽然这个基本实现中没有日志记录，但这可能是处理整个事情的更好方法，因为现在我将只使用 println()。我们的 start 函数只是启动一个服务器，然后使用一个自动生成的 router 函数异步地为它服务。

```
function generate_router(routes::AbstractVector, server)
    route_paths = Dict([route.path => route.page for route in routes])
    routeserver = function serve(http)
     HTTP.setheader(http, "Content-Type" => "text/html")
        fullpath = http.message.target
    if contains(http.message.target, '?')
         fullpath = split(http.message.target, '?')
         args = ""
    end
     if length(fullpath) > 1
         args = fullpath[2]
     end
     if fullpath in keys(route_paths)
        write(http, route_paths[fullpath])
     else
         write(http, route_paths["404"])
     end
 end # serve()
    return(routeserver)
end
function stop!(x::Any)
    close(x)
endfunction stop!(x::Any)
    close(x)
end
```

还有一站！()函数杀死服务器。最终，这个 generate_router 函数就是神奇之处，它定义了一个到其内容的页面路由，然后使用 write()将正确的内容写出流。需要注意的重要部分是 routeserver 定义所在的位置。这使得该功能在关于路由的所有情况下都是可再现的，并且在没有找到路由的情况下，该路由请求 404 路由。现在我们完成了我们的内部构造函数，它依赖于我们在它下面创建的这个新的类型系统。

```
mutable struct OurServer
    ip::String
    port::Integer
    routes::AbstractVector
    remove::Function
    add::Function
    start::Function
    function OurServer(ip::String, port::Int64)
        routes = []
        add, remove, start = funcdefs(routes, ip, port)
        new(ip, port, routes, remove, add, start)
    endfunction OurServer()
        port = 8001
        ip = "127.0.0.1"
        OurServer(ip, port)
    end
end
```

使用这种技术，我们现在已经构建了一个服务器，它将根据请求本身的内容来引导或限制流量请求。非常棒，这个结果非常好合作。首先，我们创建路线:

```
home = Route("/", "<h1>HELLO WORLD</h1>")
four04 = Route("404", "<h1> Directory not found </h1>")
```

然后，我们创建一个服务器，并在其中添加两条新的 HTTP 路由。感谢我们的内部构造器，这绝对是轻而易举的事:

```
server = OurServer()
server.add(home)
server.add(four04)
```

现在我们可以像这样启动一个新的异步服务器:

```
serving = server.start()Starting server on port 8001
Successfully started server on port 8001

You may visit it now at [http://127.0.0.1:8001](http://127.0.0.1:8001/)Sockets.TCPServer(RawFD(45) active)
```

访问该链接会产生以下结果:

![](img/a33fa74abdb1f26483e057048c7b6da5.png)

(图片由作者提供)

现在我们可以关闭服务器，转到另一个更酷的闭包实现。

```
stop!(serving)
```

# 作为参数的函数

虽然使用这些闭包函数的非常规酷的方式可能是将它们用作 HTTP 服务器的路由函数，但更有可能的情况是，人们最终会看到这些函数被用作参数。这通常是在处理复杂数据结构的基本 Julia 方法中完成的，所以这绝对是你想了解 Julia base 及其生态系统的东西。像这样的函数的一个例子就是过滤器！()函数，它采用一个函数作为筛选依据。这通常是通过匿名函数完成的。

```
x = [5, 10, 15, 20, 25]
```

假设我们想从这个数组中得到的只是 15 以上的值。分离这些值的典型方法是制作一个掩码。掩码是一个 bitarray，意思是一个布尔值数组，它决定每个值是否满足某个条件。我们真正需要做的是设计一个函数，然后应用于每个单独的值，以创建这个掩码。我们使用按位逻辑右操作符`->`创建这些函数。下面是我们如何单独创建该函数的示例:

```
x -> x > 15
#11 (generic function with 1 method)
```

或者，我们可以在过滤器的直接调用中定义它！()方法:

```
filter!(x -> x > 15, x)
```

虽然我们可以简单地通过名称将函数作为参数传递，但是还有另一个非常好的语法您可能想知道。这种语法称为 do/end 语法，它允许您在给定调用后将函数定义为代码块，然后将该函数作为参数进行处理。作为一个例子，我引用了我的代数数组的实现。这里有两个链接，一个是我详细介绍这些阵列如何工作的文章，另一个是这个项目的笔记本资源:

</my-new-big-brain-way-to-handle-big-data-creatively-in-julia-57be77fc6a04>  <https://github.com/emmettgb/Emmetts-DS-NoteBooks/blob/master/Julia/Algebraic%20Arrays.ipynb>  

代数数组基本上是可以用某种规范形式表示的数据，所以计算机只计算它需要处理的部分。这个想法是将所有这些内存消耗和事物排除在环境之外，通过选择何时与什么数据进行交互，使科学在更大范围内变得更加可行。Julia 实际上使这个实现变得非常简单，这要归功于完成这类事情所需要的非常简单的多重调度绑定。无论如何，这里是看看什么类型，然后是使用 do/end 语法计算函数。

```
mutable struct AlgebraicArray
    f::Function
    n::Int64
    calls::Vector{Pair{Function, Tuple}}
    function AlgebraicArray(f::Function, n::Int64)
        new(f, n, [])
    end
end
```

在这之上是生成层，它允许我们为一个调用生成单独的索引。这存在于几种不同的调度形式中，如 range、bitarray 和没有附加参数的代数数组。

```
function generate(aa::AlgebraicArray, range::UnitRange)
    if range[2] > aa.n
        throw(BoundsError(string("Invalid algebraic index, ", string(range[2], 
                        " on algebraic expression of length ", string(aa.n)))))
    end
    [aa.f(n) for n in range]
end
```

最后一个前端是计算()。Compute 是通常用来从类似这样的代数运算中获取值的方法。使用 do 语法，我们可以将它作为一个流参数来使用，为了设置它，我们应该像这样构造我们的方法:

```
function compute(f::Function, aa::AlgebraicArray) # compute(aa)
    gen = generate(aa)
    for call in aa.calls
        gen = [call[1](gen, call[2]...)]
    end
    return(f(gen))
end
```

这一努力的最终结果变成了这样的语法:

```
mask = compute(z) do aa
    [if val > 12 true else false end for val in aa[1]]
end
20-element Vector{Bool}:
 0
 0
 0
 0
 0
 1
 1
 1
 1
 1
 1
 1
 1
 1
 1
 1
 1
 1
 1
 1
```

如果您想深入了解这背后的实际代码，并且想了解更多关于它的一般信息，那么我肯定会建议您阅读上面的文章，因为它可能会给出关于这些数组实际如何工作的更多解释。

# 结论

关闭的力量被大大低估了。该技术提供了许多好处和解决困难或复杂问题的新方法，这无疑使它有别于许多其他编程技术和方法。此外，Julian 实现确实允许以一些非常酷的方式使用函数。虽然在其他编程语言中肯定有这种元素，Python/Java 的 lambda 和 Javascript 的函数定义浮现在脑海中，但我必须说我真的更喜欢 Julia 处理这些事情的方式。

感谢您阅读我的文章，我希望对闭包功能的强调似乎是有道理的。也许在你的下一个项目中，你会用它来解决一些问题，因为它们经常会派上用场！