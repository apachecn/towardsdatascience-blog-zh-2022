# Python 中控制属性的特殊方法

> 原文：<https://towardsdatascience.com/unusual-ways-to-control-attributes-in-python-8b69d9efda57>

## Led 指南示例

![](img/b1d207af4f863240a5a9c3d0a468db2f.png)

图片由 [Unsplash](https://unsplash.com/photos/oyXis2kALVg) 提供

属性验证确保用户提交的数据具有正确的格式或类型。Python 中属性控制可以通过多种方式进行控制。在这里，我提出了两种不同寻常但有趣的控制属性的方法。

## 示例 1:使用 magic call 方法进行属性重新分配

为了开始这个例子，我首先创建一个名为 Employees 的类，它有三个属性:姓名、年龄和地点。创建该类的一个实例，为其分配变量名 emp，并为属性设置相应的值。

使用魔术调用方法可以实现重新分配/更新属性的另一种方法。为了举例说明一个用例，我定义了一个 call 方法，它将*args 和**kwargs 作为方法参数。

当 Employees 类的实例首次初始化时，三个属性值被分配给对象属性' _name '、' _age '和' _location '。但是，使用 call 方法可以很容易地将这些属性值重置为新值。call 方法允许像对待函数一样对待实例。

调用方法定义如下。下面描述调用 call 方法的两个示例用例。

call 方法中的条件逻辑首先检查我们创建的对象 emp 是否具有属性' _name '。该条件将评估为真。此属性是在 init 方法期间设置的。条件逻辑还检查元组的长度(**args)是否等于 1。当两个条件都满足时，可以像调用函数一样调用 call 方法，如下所示:

EMP(*示例新名称*’)

名称属性现已重置。我们可以通过遍历对象字典来确认这一点。在这里，我们把名字从“斯蒂芬”改成了“西蒙”。

虽然上述方法是更改属性值的一种便捷方式，但不一定清楚我们要更改哪个属性值。我们可以通过编写一个小的 docstring 或提供例子来帮助使这一点更清楚。

然而，更简单的方法可能是在 call 方法中使用**kwargs 参数。

如果我们再次查看[Example _ 1 _ call _ method _ Part b . py](https://gist.github.com/StephenFordham/a36157ab4ef8150482d43661bbc47195#file-example_1_call_method_part-c-py)中的 Python 脚本，第二个条件检查是否设置了 location 属性，我们命名为**employee_info 的**kwargs dict 的长度为 1。当我们像这样调用 call 方法时:

emp(location='newloc ')

该条件将评估为真，我们可以重新设置位置属性，如下所示。

注意:给出的例子是为了说明如何使用 call 方法。使用**employees_info 字典更改年龄和姓名的属性值需要对其他参数进行额外的验证。

最后，在 else 语句中，如果像这样调用 call 方法:

电磁脉冲()

将调用 init 方法，尽管省略它可能更容易。我已经包含了它，以显示不会对属性值进行任何更改。

## 示例 2:使用自定义类设置属性

控制属性的另一种不同寻常的方法是使用自定义类。在下面 GitHub gist 的中心，我创建了一个名为 Employees 的类。我实例化了两个类对象，名为 name 和 location，作为 StrAttrValidation 类的实例。

在 StrAttrValidation 类中，一个空字典被初始化。

在 Empolyees 类中，当在 init 方法中调用“object.attribute = value”时，将调用 StrAttrValidation 的 set 方法，其中字典将实例作为键，将值作为用户设置的值。

为了证明这一点，当我创建名为 emp1 的 employees 对象时，名称和位置都是小写的。在下面的实现中，属性现在已经被资本化，参见下面的终端输出。

我们可以在 set 方法中添加进一步的验证。名称和位置只能是有效的字符串。让我们将验证添加到 SetAttrValidation 类的 set 方法中。现在，当我们试图将位置作为一个 zipcode(表示为一个整数)添加时，会引发一个异常，并且不会设置属性。

这两种方法代表了 Python 中控制属性的不同方式。通常更简单的实现，比如在初始化对象时在 init 方法中进行验证，可能是最好的方法。

感谢阅读！