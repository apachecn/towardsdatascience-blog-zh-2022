# 关于 Python 函数要知道的六件最重要的事情

> 原文：<https://towardsdatascience.com/the-six-most-important-things-to-know-about-python-functions-e1eacaab412>

## 如何使用这些强大的工具来编写更好的 Python 代码

![](img/b2bb57ca2653b4364e22f84e433cc86a.png)

图片来源:pixabay 的 Pixl2013

许多初学编程的人被函数弄得不知所措，努力理解他人的代码或在自己的脚本中使用函数。然而，函数是一个非常强大的编程特性，利用它们将会改进您的代码。为了帮助您学习这些技能，并立即开始编写优秀的函数，以下是关于使用 Python 函数需要了解的六件最重要的事情。

# 1.为什么应该使用函数

函数有用有两个原因。

首先，函数允许你一次编写一个代码块，然后多次使用它。您只需要编写和调试一次代码。这在将来为你节省了很多时间。函数也使你的代码更容易维护，因为如果你需要在某个时候更新你的函数，你只需要在一个地方编辑它。

其次，编写函数可以让读者更容易理解你的代码。如果你的函数有一个描述性的名字(例如，一个为计算餐馆账单上的小费而设计的函数可以被称为 **calculate_tip** )，那么任何阅读你的代码的人都可以理解每一行是做什么的，而不需要理解这些计算。

假设我已经编写了前面提到的 **calculate_tip** 函数，我想为我自己和我的朋友计算小费。我朋友的账单是 21 美元，我的是 32 美元，我们每个人都想留下 20%的小费。我可以编写下面的代码，调用 **calculate_tip** 两次来获得每一次的提示。现在不要担心函数的语法，我稍后会谈到。

```
friends_bill = 21my_bill = 32tip_percent = 20friends_tip = calculate_tip(friends_bill, tip_percent)my_tip = calculate_tip(my_bill, tip_percent)print(friends_tip)print(my_tip)
```

这段代码的输出将是我和我的朋友应该留下的小费金额。请注意阅读和理解这些代码是多么容易。对于这两种计算，我们只看到一行说明小费金额是根据账单和期望的百分比计算的，这非常直观。

# 2.Python 函数的基本结构

所有 Python 函数都使用相同的基本结构。首先，您需要定义函数，为它提供一个名称和函数处理所需的参数。然后，您需要定义每次调用时它将处理的代码。最后，当您调用函数时，您需要指定要返回的值(如果有的话)。让我们再次使用我们的 **calculate_tip** 函数示例:

```
def calculate_tip(bill, percent): tip = bill * percent/100 return tip
```

第一行代码定义了函数。

*   **def** 告诉 Python 该行正在创建一个新函数。
*   **calculate_tip** 是函数的名称，所以 Python 现在知道无论何时键入 **calculate_tip** 都要使用这段代码。
*   **(bill，percent)** 声明无论何时调用该函数，这两个输入都是必需的。
*   结尾的 **:** 告诉程序在调用这个函数时运行下面的缩进代码。

第二行代表该函数使用的计算。每当有人想计算小费时，该函数将账单乘以所需的百分比，然后除以 100(将百分比转换为小数)来确定想要的小费金额。

最后，最后一行表示计算出的小费是函数的输出，可以根据需要传递给变量。

在之前计算朋友小费的例子中，我们使用了账单中的 **21** 美元和 **20** 百分比作为小费。调用 **calculate_tip** 的代码导致该代码以 bill = 21 和 percent = 20 运行。Tip 被计算为 **21 * 20/100** ，或 **4.2。****calculate _ tip**函数然后返回 **4.2** ，它被存储到变量 **friends_tip** *，*中，然后我们打印出来以便于查看。

# 3.返回值

上一节展示了从函数返回值的基础。您可以使用 **return** 命令，后跟一个值来声明您想要的输出。Python 函数的一个巧妙之处在于，只要将输出赋给足够多的变量，就可以返回任意多的输出。

例如，我们可以让 **calculate_tip** 更有用，让它返回包括小费在内的总账单。然后，用户可以看到留多少小费以及总共要付多少。为此，我们将修改 **calculate_tip** 如下。

```
def calculate_tip(bill, percent): tip = bill * percent/100 total = bill + tip return tip, total
```

该代码有两处补充。首先，我们在计算和返回小费之间添加了一行。这一行把小费加到账单上，算出总支付额。第二个变化是将 **total** 添加到返回行中，告诉 Python 无论何时我们调用 **calculate_tip** ，它都应该返回这两个值。

我们可以使用稍微更新的代码来存储和读取这些函数的输出，如下所示:

```
friends_tip, friends_total = calculate_tip(friends_bill, tip_percent)my_tip, my_total = calculate_tip(my_bill, tip_percent)print(friends_tip, friends_total)print(my_tip, my_total)
```

由于 **calculate_tip** 现在返回两个输出，我们需要更新调用它的代码行以接收两个输出。为此，两条线现在都有两个变量。一个存储算出的小费，另一个存储算出的总数。打印输出的语句也是如此。每行打印小费，然后打印总数。

# 4.使用参数

函数所需的输入通常称为参数。它们有时被称为参数，尽管实参是更常见的术语。

在处理参数时，您可以使用一些技巧。

首先，您可以直接在函数定义中为每个参数提供一个默认值。如果你知道你通常给 20%的小费，你可以把它作为默认值输入。如果你想用一个不同于 20%的数字，你只需要指定小费的百分比。例如，考虑下面的代码:

```
def calculate_tip(bill, percent = 20): tip = bill * percent/100 total = bill + tip return tip, totalfriends_bill = 21my_bill = 32tip_percent = 10friends_tip, friends_total = calculate_tip(friends_bill, tip_percent)my_tip, my_total = calculate_tip(my_bill)print(friends_tip, friends_total)print(my_tip, my_total)
```

在函数定义中，您可以看到 **tip_percent** 现在被设置为 **20** ，这表明如果您在调用函数时没有指定值，将使用 20%。调用 **calculate_tip** 返回 **my_tip** 和 **my_total** 的行只传递 **my_bill** 作为输入。由于代码不会覆盖默认值 20%，因此 **calculate_tip** 在执行计算时会使用 20%。

另一方面，变量 **tip_percent** 被设置为 10 %,并在调用 **calculate_tip** 来标识 **friends_tip** 和 **friends_total** 时使用。这将覆盖默认的 20 %,并使用 10%执行计算。

Python 参数也可以是位置参数或关键字参数，代表指定其值的两种不同方式。位置参数是根据它们在函数调用中的位置来引用的，关键字参数是通过引用函数调用中的参数名称来指定的。

让我们看几个简单的例子。

你们已经熟悉位置论点了，因为这是我们目前为止一直在用的形式。为了突出这一点，请考虑下面的代码:

```
def calculate_tip(bill, percent = 20): tip = bill * percent/100 total = bill + tip return tip, totaltip, total = calculate_tip(21, 15)
```

请注意，函数调用没有包含任何指定哪个数字属于哪个参数的代码。因为没有指定，所以这些值根据它们的位置分配给参数。因为 **calculate_tip** 首先寻找账单变量，所以它使用第一个传递的值( **21** )。百分比也是如此；因为 **calculate_bill** 预期百分比是第二个，所以它使用第二个传递的变量( **15** )。

关键字参数是通过在传入参数时引用特定关键字来指定的。这允许您按照自己喜欢的任何顺序指定参数。例如，您可以使用以下代码:

```
def calculate_tip(bill, percent = 20): tip = bill * percent/100 total = bill + tip return tip, totaltip, total = calculate_tip(percent = 22, bill = 110)
```

在该示例中，函数调用明确声明 percent 为 **22** ，bill 为 110 美元。事实上，它们的顺序与 **calculate_tip** 预期的顺序相反，这很好，因为 Python 使用了关键字而不是位置。

# 5.单独的名称空间

在 Python 中，名称空间是变量和关于这些变量的信息的集合。程序的不同方面会创建新的不同的名称空间，所以您需要注意如何传递变量。

函数能够从主外部名称空间读取值，但是除非明确声明，否则它们不会返回到外部名称空间。

记住这一点很重要，原因有二:

1.  在函数内部定义变量时，你需要小心。如果不是这样，你可能会使用与函数外部相同的变量名，并且可能会使用错误的值。
2.  您需要仔细考虑要从函数中返回的值。如果不将变量作为输出返回，外部名称空间就完全无法访问它。

这里有一个突出这些影响的例子:

```
def calculate_tip(bill, percent = 20): tip = bill * percent/100 total = bill + tip print(my_bill)my_bill = 32tip_percent = 10calculate_tip(friends_bill, tip_percent)print(total)
```

请注意这段代码的一些特殊之处。首先，**我的账单**既没有传入**计算提示**中也没有进行计算。它只存在于 **calculate_tip** 之外的名称空间中。其次， **calculate_tip** 没有返回值，因此函数中计算的 tip 和 total 变量在该名称空间内。

当我们运行该代码时，我们得到以下输出:

```
32Traceback (most recent call last): File “main.py”, line 18, in <module> print(total) NameError: name ‘total’ is not defined
```

您可以看到 **calculate_tip** 成功打印了 **my_bill** (32)，因为它从外部名称空间继承了该信息。但是在打印 **total** 时代码出错，因为 **total** 只存在于**calculate _ tip**的名称空间中，不存在于主名称空间中。

# 6.记录您的功能

记录你的代码是非常重要的。这是你告诉用户如何使用你的代码，以及你为什么做出选择的方式。

您可能认为文档在您的大部分代码中并不重要，因为您是唯一使用它的人。然而，您会惊讶地发现，您很快就会忘记代码是如何工作的。如果你曾经写过一个函数，两年后再来看，你会非常感激你留给自己的注释。

最基本的代码文档是在函数开始时提供一些介绍。该文档应介绍功能的目的和方法，然后描述所需的输入和输出。这种描述通常是在一句输入和输出描述之前的两三句话。下面是我们的 **calculate_tip** 函数的一个例子:

```
def calculate_tip(bill, percent = 20): ''' Calculates the tip and total for a given bill and tip percentage. Processes for a single bill at a time. inputs bill: Float. The price to be paid by the person. percent: Float. The desired tip to be paid, expressed as a percent of the bill. outputs tip: Float. The tip to be paid for the bill and tip percent. total: Float. The total price to be paid, including the tip. ''' tip = bill * percent/100 total = bill + tip return tip, total
```

阅读该文档可为用户提供正确使用该功能所需的关键信息。其他人现在可以理解函数的目的、限制以及输入和输出的结构和目的。

另一个需要记住的要点是，你应该记录你所做的任何假设，或者强调为什么代码必须是你编程的方式(我在这里没有这样做，因为这是不言自明的)。这些包含的内容是一种很好的方式，可以与以后可能需要理解计算方法的其他人进行交流。

就是这样！现在你知道了使用 Python 函数的六个最重要的技巧。