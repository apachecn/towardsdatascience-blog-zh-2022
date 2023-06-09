# Python 和 Julia 中继承的比较

> 原文：<https://towardsdatascience.com/a-comparison-of-inheritance-in-python-and-julia-fb7432cd4929>

## Python 和 Julia 范式的不同继承性概述

![](img/febbe088f212199df95f5bcc6ff7871c.png)

(图片由 [Pixabay](http://pixabay.com) 上的 [naobim](https://pixabay.com/images/id-1239554/) 拍摄)

# 介绍

继承已经成为现代编程语言的一个主要特征。这个概念最初是由一种叫做 Simula 的语言实现的，它很快激发了 C++语言的诞生。当然，C++取得了巨大的成功，直到今天仍然如此，所以 C++通过其令人惊叹的泛型编程概念，真正将这个概念以及许多其他概念带到了编程语言设计的前沿。继承的概念贯穿了多年的编程，并最终出现在今天的几种高级编程语言中。集成了这一概念的编程语言的两个例子是 Python 和 Julia，这两种编程语言通常用于当今的计算科学。

虽然这两种语言在 2021 年主要用于类似的应用，但这两种语言也有根本的不同。Julia 是一种考虑到数值计算而创建的编译语言，它处理数字更像 FORTRAN 而不是 C++，Python 是一种为通用脚本创建的解释语言，它处理数字(至少没有 NumPy)的方式更像 C 或 Javascript。除了刚刚列出的技术差异，这两种语言在编程的核心理念上也有所不同。

Python 是一种面向对象的编程语言，尽管它肯定是多范式的，并且对面向对象没有特别严格的要求。我认为这是一件好事。Julia 在这一点上是相似的，你不需要强迫每件事都按照某种方式来编程。然而，Julia 更独特一些，这种语言是用一种新的范式编写的，它的创造者称之为“多调度范式”。在 Julia 中，范式是由语言定义的，所以没有办法说多重分派本身是或不是纯粹的范式，因为这是唯一一个将整个范式建立在多重分派基础上的语言实例。

也就是说，由于 Julia 和 Python 语言之间的这些基本差异，实现语言多范例的泛型编程概念当然需要以不同的方式处理。毕竟，在朱莉娅的情况下；如果一开始就没有类，我们如何继承我们类的子方法呢？这两种语言在语义上非常不同，因此它们内部的概念处理方式也不同。今天我想讨论和比较这两种语言是如何处理继承的，以及每种继承的范例和方法与另一种相比所具有的优势。

# 差别

从 Julia 的世界开始，继承是用抽象类型来处理的。与 Python 不同，在 Julia 中，抽象类型实际上只是一个名称，可以作为其他名称的别名。我们可以使用子类型操作符创建子类型。这个操作符也是一个 bool 类型的操作符，可以用来确定一个给定的类型是否是一个子类型。

```
abstract type Pickle endstruct BreadNButter <: 
    flavor::Symbol
end
```

在这种情况下，类型通过它们的名字来继承。面包巴特是泡菜的一个分支，因此，

```
BreadNButter <: Pickle
```

将返回 true，因为面包屑现在是泡菜的一个子类。至于实际上继承了什么，范式现在开始发挥作用。泡菜对面包师的传承是通过多重派遣实现的。我们可以通过简单地将方法分派给我们的抽象类型，而不是我们的类型，来使方法同时分派给几个类型。

```
flavor(p::Pickle) = p.flavor
```

Python 与众不同，因为它有一种非常传统的方法，通过类本身来创建方法的子类继承。Python 中的 Pickle 示例如下所示:

```
class Pickle:
    def __init__(flavor : string):
        self.flavor = flavor
    def flavor():
        return(self.flavor)
class BreadNButter(Pickle):
    def __init__(flavor : string):
        self.flavor = flavor
```

如果你想学习更多关于 Python 中继承和类型的知识，我也有一整篇关于这个主题的文章，你可以在这里阅读:

</everything-you-need-to-know-about-type-inheritance-in-python-2e173277ff22>  

这两种实现实现了相似的目标。我们不必多次编写方法，因为它们被带入我们的新类或通过多次分派来分派。然而，这两种方法都有一些区别和优点。

# 差异的结果

假设继承的这两个实现做类似的事情，但是做的方式不同，那么使用这两个系统可能各有利弊。这些不一定是主观的，但应该注意的是，在大多数情况下，这些是优点还是缺点几乎完全取决于代码实际使用的场景。首先，也是最明显的相似之处是，这两者都继承了方法。然而，要指出的第一个也是最明显的区别是，只有 Pythonic 实现还将继承类型的属性。如果我们希望我们的类型在 Julia 中有相同的字段，那么就要由程序员来编写这些字段。在 Julia 中也有检查一致性的能力，但是不管怎样，每个类型的字段都需要单独编写。

另一个实质性的区别是 Julia 中的方法不是构造类型的子方法。因此，将方法添加到特定的定义中要容易得多，而且还要确保它能与可能被子类化的类型数组一起工作。这些方法完全在子类型之外，实际上是继承——而不是类型本身。至于这样做是否更好，这当然取决于场景，在有些情况下，我认为继承属性是非常重要的事情，例如视频游戏编程或用户界面，但是也有方法为王的情况，继承方法并添加继承的方法将比继承属性更有价值。

总的来说，我认为这是继承方面的一个很大的区别因素，它可以用来做两种语言之间的事情。Python 的例子使继承与类型的关系更加密切。另一方面，Julia 的版本为类型创建了一个简单得多的抽象方法，而是依靠方法中的多态性将这些方法应用于类型。考虑到我是一名数据科学家，继承的核心思想给了 Julia 一个明确的优势。数据科学通常在全球范围内进行，使用许多方法和简单的数据类型，而不是疯狂的结构。然而，对于其他项目，我肯定会看到 Python 占据优势。

总而言之，我认为就像编程语言世界中的任何其他东西一样，没有一种语言在继承方面比另一种语言更好。在某些情况下，不同的方法可能更好，但这类事情总会有所取舍。总而言之，就看你想做什么了。此外，这两种语言只是简单地提供了继承，这是其他不提供继承的语言的一个优势。感谢您的阅读，我希望这个小小的比较有助于确定什么可能是您的下一个项目的最佳选择，该项目以某种能力的继承为特色！