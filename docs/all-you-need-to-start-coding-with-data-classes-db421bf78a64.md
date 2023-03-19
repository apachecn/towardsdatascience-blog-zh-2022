# Python 中的下一件大事:数据类

> 原文：<https://towardsdatascience.com/all-you-need-to-start-coding-with-data-classes-db421bf78a64>

## 全面了解 Python 中的数据类基础知识

![](img/46e7260f84ceac8f66d8824ce53e22aa.png)

照片由 [J Gowell](https://unsplash.com/@tmwsiy?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

# 什么是数据类？

数据类是 Python 版内置的一项新功能。它们提供了装饰器和函数来创建更简单、更方便、更安全的类，这些类主要用于处理数据，因此得名。

数据类的一个主要好处是，它们会自动为你生成几个特殊的方法，比如`__init__`、`__repr__`和`__eq__`。当定义主要用于处理数据的类时，这可以为您节省大量时间和冗余代码。

数据类的另一个主要好处是使用了强类型，这可以确保定义实例的属性。这是通过使用类型注释来实现的，类型注释允许您在定义类时指定每个属性的类型。这可以防止由于类型不明确而导致的错误，也使代码更容易被其他开发人员理解。

# 如何使用数据类

要使用数据类，首先需要从`dataclasses`模块导入 dataclass 装饰器。这个装饰器是 Python 3.7 和更高版本中固有的。

使用数据类非常简单。只需用`@dataclass`装饰器装饰你的类定义来定义一个数据类。

下面是一个带有默认参数的简单数据类的示例:

```
from dataclasses import dataclass, field

@dataclass
class Point:
    x: float
    y: float

p = Point(1.0, 2.0)
print(p)  # Output: Point(x=1.0, y=2.0)
```

我们定义了一个带有两个字段`x`和`y`的`Point`数据类，这两个字段都是浮点数。当您创建一个`Point`类的实例时，您可以指定`x`和`y`的值作为构造函数的参数。

默认情况下，数据类将为您生成一个`__init__`方法，该方法将类的字段作为参数。它们还将生成一个返回对象的字符串表示的`__repr__`方法，这是在上面的例子中调用`print(p)`时输出的内容。

您可以通过传递额外的参数来自定义由装饰器生成的方法，例如 *repr=False* 来停用`__repr__`方法。

# 场函数

您还可以通过使用`field()`函数的`default`和`default_factory`参数来指定数据类字段的默认值。

例如:

```
@dataclass
class Point:
    x: float = 0.0
    y: float = field(default=0.0)

p1 = Point(x=1.0, y=2.0)
print(p1)  # Output: Point(x=1.0, y=2.0)

p2 = Point(1.0)
print(p2)  # Output: Point(x=1.0, y=0.0)
```

在这个例子中，我们定义了一个具有两个字段`x`和`y`的`Point`类。x 字段的默认值为 0.0，直接在字段定义中指定，而 y 字段的默认值为 0.0，使用`field()`函数指定。

当我们创建一个`Point`类的实例时，我们可以指定`x`和`y`的值作为构造函数的参数。如果我们没有为`y`指定一个值，它将使用默认值 0.0。

在数据类内部，指定默认值的两种用法对于**不可变的**属性是等效的。

然而，`field()`函数允许更多的灵活性和额外的选项来定义属性。

## 字段 repr 和 init 参数

事实上，`field()`函数将默认参数`__repr__`和`__init__`设置为*真*。我们可以将这些设置为*假*来修改它们的行为。

```
@dataclass
class Point:
    x: float = field(default=0.0, repr=False)
    y: float = field(default=0.0, init=False)

p1 = Point(3.0) 
print(p1) # Output: Point(y=0.0)
p2 = Point(3.0, 2.0) # TypeError: Point.__init__() takes from 1 to 2 positional arguments but 3 were given
```

在第一个例子中，实例`p1`没有在其字符串表示中显示`x`的值，这对于隐藏临时或敏感变量(至少从字符串表示中)是有用的。

在第二个例子中，实例`p2`不能被创建，因为 init 参数被设置为 False，我们试图初始化`y`变量。这对于只应由方法返回的变量很有用。让我们向我们的数据类添加一个方法，该方法仅使用`y`作为函数`compute_y_with_x`中计算的输出:

```
@dataclass
class Point:
    x: float = field(default=0.0, repr=False)
    y: float = field(default=0.0, init=False)
    def compute_y_with_x(self):
        self.y = self.x ** 2

p2 = Point(x=2.0)
p2.compute_y_with_x()
print(p2) # Output: Point(y=4.0)
```

这里，我们注意到`p2`使用`y`作为未初始化的变量( *init=False* )，这是我们的输入变量`x`转换的结果。我们不关心初始化后的`x`，所以我们移除了它的字符串表示 *(repr=False)。*

## 默认工厂字段

你还记得我们讨论过使用类的默认参数或者用默认参数设置一个字段对不可变的对象有同样的影响，但是对可变的对象没有影响吗？

让我们看一个例子，我们试图初始化一个**可变**对象的属性，比如一个列表。我们还将创建一个方法，允许我们将元素添加到名为`add_a_dimension`的列表中:

```
@dataclass
class Points:
    coord: list = field(default=[])
    def add_a_dimension(self, element):
        self.coord.append(element)
# Output: ValueError
```

这个类不能被构造，因为如上所述，“不允许对字段<coord>使用可变默认值(比如列表)。”这是由数据类添加的安全措施，对于防止我们在常规类中可能遇到的错误非常有用。</coord>

事实上，如果我们想使用一个常规类来定义`Points`数据类，它将完全等同于:

```
class Points:
    coord = []
    def __init__(self, coord=coord):
        self.coord = coord
    def add_a_dimension(self, element):
        self.coord.append(element)
```

而且我们在定义这个类的时候不会有任何错误！然而，如果我们使用一个常规类以这种方式定义该类，我们会看到意想不到的行为:

```
p1 = Points()
p2 = Points()
p1.coord, p2.coord # Output: ([],[])
p1.add_a_dimension(3)
```

好了，我们实例化了一个空列表，并将 3 添加到实例`p1`中。你认为现在`p1.coord`和`p2.coord`的价值会是多少？

```
p1.coord, p2.coord # Output: ([3], [3])
```

难以置信，实例`p2`也受到了列表附加 3 的影响！

这是因为在 Python 中，当你创建一个类的实例时，这个实例将共享类属性的同一个*副本*。因为列表**是可变的**，所以`p1`和`p2`将共享`coord`列表的相同副本。

为了正确地实现这个类以避免这种意外的结果，数据类为被称为`default_factory`的`field()`提供了参数。

该参数允许您指定一个函数，每次创建一个新的类实例时，都会调用该函数为字段*创建一个新的默认值。这确保了类的每个实例都有它自己唯一的字段副本*，而不是共享同一个*可变*对象。

```
@dataclass
class Points:
    coord: list = field(default_factory=lambda: [])
    def add_a_dimension(self, element):
        self.coord.append(element)
p1 = Points()
p2 = Points()
p1.coord, p2.coord # Output ([], [])
```

值得注意的是，即使我们将一个可变列表定义为`coord`的默认值，也没有 *ValueError* 。

让我们只在一个实例上调用`add_a_dimension`方法后检查输出:

```
p1.add_a_dimension(3)
p1.coord, p2.coord # Output ([3], [])
```

啊，我们终于有了想要的结果！实例`p2`未被实例`p1`调用的方法更改。正是因为每个实例都有其*字段`coord`的唯一副本*。

正如我们所见，数据类通过正确处理可变和不可变对象来提供安全性。

# 遗产

要考虑的最后一点是数据类如何处理继承。默认情况下，数据类没有一个`__init__`方法，所以需要一个允许您覆盖继承属性的方法。这个方法叫做`__post_init__`，在实例初始化之后调用。

这里有一个例子来帮助说明这个概念:

```
@dataclass
class Point:
    x: float = field(default=0.0)
    y: float = field(default=0.0)

    def __post_init__(self):
        self.x = self.x ** 2
        self.y = self.y ** 2

@dataclass
class ColoredPoint(Point):
    color: str = field(default='black')

    def __post_init__(self):
        self.color = self.color.upper()
```

在这个例子中，`Point`类有一个`__post_init__`方法，它对`x`和`y`的值求平方。`ColoredPoint`类继承自`Point`，也有自己的`__post_init__`方法，该方法大写`color`属性的值。

让我们创建一个`Point`的实例:

```
p0 = Point(2.0,2.0)
print(p0) # Output: Point(x=4.0, 4.0)
```

我们从输出中注意到，调用了`__post_init__`方法，并且`x`和`y`的值都是平方的。

现在让我们创建一个`ColoredPoint`的实例:

```
p1 = ColoredPoint(2.0, 2.0, 'red')
print(p1) # Output: ColoredPoint(x=2.0, y=2.0, color='RED')
```

创建`ColoredPoint`的实例时，调用了`__post_init__`方法，并且`color`的值是大写的，但是`x`和`y`的值不是平方的，知道为什么吗？

这是因为我们`ColoredPoint`的`__post_init__`方法中没有调用`Point`的`__post_init__`方法！

要调用基类的`__post_init__`方法，可以使用`super()`函数，该函数返回对基类的引用。以下是`ColoredPoint`的修正版:

```
@dataclass
class ColoredPoint(Point):
    color: str = field(default='red')

    def __post_init__(self):
        super().__post_init__()
        self.color = self.color.upper()
p2 = ColoredPoint(2.0, 2.0, 'red')
print(p2) # Output: ColoredPoint(x=4.0, y=4.0, color='RED')
```

随着这一改变，当创建`ColoredPoint`的实例时，将调用`Point`类的`__post_init__`方法，并且`x`和`y`的值将如预期的那样平方。

我很高兴能够帮助您了解数据类。如您所见，数据类提供了常规类的增强版本，更加安全并强调强类型。我希望您对自己在未来项目中使用数据类的能力更有信心。

无需额外费用，您可以通过我的推荐链接订阅 Medium。

<https://medium.com/@arli94/membership>  

*或者你可以在收件箱里收到我所有的帖子。**[***在这里做*** *！*](https://arli94.medium.com/subscribe)*