# 对《朱莉娅》中构造函数的痛苦的深入观察

> 原文：<https://towardsdatascience.com/a-painfully-in-depth-look-at-constructors-in-julia-2053a69bc8c6>

## 在 Julia 中构建构造函数的所有细节

![](img/19488737bdd8d6115a3443d9cbc1577b.png)

(图片由 [pixel2013](https://pixabay.com/images/id-1745753/) 在 [Pixabay](http://pixabay.com) 上拍摄)

# 介绍

类型是 Julia 编程语言中一个非常重要的概念。当然，它们在任何语言中都很重要——但有时在其他类似于 Julia 的语言中，类型要抽象得多，不需要过多考虑事物的实际类型。由于多次调度，类型还涉及到另一个方面，即方法。也就是说，这使得类型的概念比在 Python 这样的语言中需要更多的关注。

类型当然需要构造函数来创建，掌握构造函数与掌握构造函数概念同等重要，甚至更重要。Julia 也有一个相当健壮的类型系统，人们肯定会想利用它，有了它，构造函数也是健壮的，这是一件很棒的事情。今天，我想认真研究一下构造函数，并详细说明人们需要了解的关于它们的所有信息。Julia 中的构造函数可能与人们可能习惯的构造函数有很大不同，可以理解的是，它们的工作方式可能有点难以想象。幸运的是，我已经为您做好了准备，如果这篇文章还有一些不足之处，我的代码也可以在笔记本上找到，您可以在这里查看:

<https://github.com/emmettgb/Emmetts-DS-NoteBooks/blob/master/Julia/All%20about%20constructors.ipynb>  

此外，如果你更愿意以视频的形式来看这个(尽管细节不多)，这里有一个我去年制作的视频，它讲述了一些概念:

# 基本构造函数

Julia 中最基本的构造函数形式是通过使用 struct 关键字创建的。这将创建一个不可更改的常量定义，并且类型将遵循这个约定，也是不可变的。为了改变这一点，我们还可以在 struct 前添加 mutable，以表明我们不希望类型成为不可变的。

```
struct NotMutable
    x::Int64
endmutable struct IsMutable
    x::Int64
end
```

应该在类型内部的项，在这个例子中是 Int64 类型的值 x，在 Julia 中被称为字段。我们可以使用.或 getfield()方法来访问我们的字段。电话。此方法接受一个符号，并将为该符号提供适当的字段。我们还可以使用 fieldnames 方法列出给定类型的字段:

```
fieldnames(IsMutable)(:x,) w = IsMutable(5)w.x5
```

这属于 Julia 中运行时自省的范畴，如果您想了解更多，我有一整篇关于这个主题的文章，因为它非常有用:

</runtime-introspection-julias-most-powerful-best-kept-secret-bf845e282367>  

```
mutable struct IsMutable
    x::Int64
end
```

每当我们创建这个新的构造函数 IsMutable 时，我们实际上用 Int64 创建了一个到别名 IsMutable 的新绑定。然而，多重分派在这里并没有消失，我们可以通过简单地编写一个新的函数来构造它，从而将多重分派应用到我们的构造函数中。我们希望将它放在字段的末尾之前和之后，因为这会创建一个内部/外部构造函数关系。内部构造函数总是返回外部构造函数，唯一改变的是为该字段提供的参数。一个重要的注意事项是，我实际上一开始并没有意识到，一旦我们创建了第一个内部构造函数，外部构造函数就不能在外部直接调用了。

# 内部构造函数

内部构造函数允许我们用一致的字段创建一个一致的类型，这个一致的字段具有本质上无限数量的不同参数组合。例如，让我们说，我们想从前面构造我们的 IsMutable 类型，但我们不希望用户必须提供一个值。我们可以通过简单地创建一个方法来轻松地做到这一点。然而，首先，让我们创建一个顶级内部构造函数，它将被其他内部构造函数调用。当然，您不需要这样做，但是这样可以避免重复许多行代码——如果您的构造函数变得特别复杂，这可能是您想要的。记住，有无限多的构造函数和参数组合，我们也可以绑定它们，所以创建一个主内部构造函数肯定是个好主意。在这种情况下，这个构造函数将与外部构造函数是同一个调度，并接受一个整数。我们使用 new()方法为上面的外部构造函数提供字段:

```
mutable struct IsMutable
    x::Int64
    function IsMutable(x::Int64)
        new(x)
    end
end
```

现在我们将添加另一个内部构造函数，它将调用这个内部构造函数并提供一个默认值。

```
mutable struct IsMutable
    x::Int64
    function IsMutable(x::Int64)
        new(x)
    end
    function IsMutable()
        IsMutable(0)
    end
end
```

我们在代码中看到了这种努力的结果:

```
z = IsMutable(5)IsMutable(5)b = IsMutable()IsMutable(0)
```

# 动态字段类型

在某些情况下，我们可能不知道我们提供的字段的类型。在某些情况下，使用 Any 可能是有意义的，它是 Julia 类型层次结构的绝对类型。在其他情况下，实际改变从构造函数返回的类型来表示某个字段的类型可能更有意义。我们可以用{}语法做到这一点，Julia 中的其他语法已经停止使用了。我们通过在名称后添加这些括号，然后为该类型创建一些名称来实现这一点，如下所示:

```
mutable struct IsMutable{T}
    x::Int64
    param::T
```

现在我们将需要更新我们的内部构造函数，但是在我们这样做之前，我还想指出的是，函数的处理方式和所有其他的 Julian 类型是一样的，所以我们也可以在这里使用闭包。也就是说，我们可以通过为一个函数添加一个新字段来有效地将这种范式转变为面向对象编程，但是对于本文，我们将只关注这种类型和这种代码。然而，这两种方法都工作得很好，它们都有不同的缺点和优点，就像设计中的任何选择都有优点和缺点一样。无论如何，为了使这成为某种真正的例子，我将稍微改变这些名字来创建一个新的“学生”构造函数。在此之前，让我们深入了解一下内部构造函数。我们在 new()方法的参数之前提供类型，这在整个 Julia 中是通用的。每当类型 T 改变时，新的 IsMutable{T}将是不同的类型。举个例子，

```
IsMutable{Int64}
IsMutable{Float64}
```

会被视为不同的类型。它们将有不同的方法，但是我们仍然可以通过使用我们已经有的类型层次结构以抽象的方式引用 IsMutable，基本上创建两层类型层次结构，并且在这种类型的两个不同维度上创建潜在的无限抽象层——这非常棒，因为您可能希望根据字段内容的类型以不同的方式处理给定的类型。这是我们的新构造函数，我用它来为我的教室变戏法让学生们存在，而这个教室是不存在的。

```
mutable struct Student{T}
    age::Int64
    label::T
    function Student(x::Int64, label::Any)
        new{typeof(label)}(x)
    end
    function Student()
        Student(0, 0)
    end
end
```

我们可以看到第一个内部构造函数已经为我们节省了一些时间，因为我们只需要改变 new()方法的调用一次。然而，在我们将它用于 dispatch 之前，我们可能应该看一下子类型，这样这里的想法在大脑中是新鲜的。

# 子类型

Julia 有一个相当健壮的类型层次系统，这对于用一个方法为许多类型编写许多功能的程序非常有帮助。我们用抽象类型来实现这一点，抽象类型本身不是类型，但是可以被分派给。然后我们提供子类型操作符，<: after="" defining="" our="" constructor="" name="" like="" so:=""/>

```
abstract type SchoolMember endmutable struct Student{T} <: SchoolMember
    age::Int64
    label::T
    function Student(x::Int64, label::Any)
        new{typeof(label)}(x)
    end
    function Student()
        Student(0, 0)
    end
end
```

We can further this infinitely with more sub-types, as well.

```
abstract type OnCampus end
abstract type SchoolMember <: OnCampus end
mutable struct Student{T} <: SchoolMember
    age::Int64
    label::T
    function Student(x::Int64, label::Any)
        new{typeof(label)}(x, label)
    end
    function Student()
        Student(0, 0)
    end
end
```

Now we have several different dispatch calls we can make which get more specific down the line.

*   OnCampus
*   SchoolMember
*   Student
*   Student{T}

For example, we could write this function:

```
function details(s::Student{Int64})
    println("The student is " * string(s.age) * " years old.")
end
function details(s::Student{String})
    println(s.label * " is" * string(s.age) * " years old.")
end
```

And then we see the printout is different depending on the type of the student’s label:

```
details(Student())The student is 0 years old.steven = Student(5, "Steve")details(Student(5, "Steve"))Steve is5 years old.
```

# Anonymous Function -> Type

关于构造函数的最后一个有趣的事情是，我们也可以匿名创建它们。为此，我们使用逻辑右操作符-->，就像在匿名函数中一样。我们添加了括号，以便 Julia 知道我们提供的是一个没有参数的元组，然后只返回括号中的字段:

```
function tz()
    z = 5
    y = 2
    (T)->(z;y)
end
```

为了让它工作，它需要是一个函数的返回。我们可以通过它们的参数名来访问这些字段。

```
v = tz()
v.z5
```

# 结论

朱莉娅的字体系统绝对是令人敬畏的。在伴随分派的子类型的所有不同层之间，限制和创建不同子类型的能力，以及能够改变，更重要的是；取决于字段类型的分派类型有助于一些非常强大的应用程序。非常感谢您的阅读！