# 你应该知道和避免的 7 种代码气味

> 原文：<https://towardsdatascience.com/7-code-smells-you-should-know-about-and-avoid-b1edf066c3a5>

## #2 使用打印语句进行调试

![](img/2ac3e8fea74e935430564564c6296de2.png)

由 [charlesdeluvio](https://unsplash.com/@charlesdeluvio?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄的照片

程序不一定要崩溃才能对其中的问题发出警报:一些其他因素可能会作为一个问题即将出现的严重警告。例如，如果你在房子的某个地方闻到煤气味或烟味，这可能表明你有煤气泄漏或有东西在燃烧。这两种情况都需要在成为重大问题之前进行调查(例如，你的房子爆炸)。

代码气味可以被认为是在你家里闻到煤气或烟味。您的代码不会因为它的存在而停止执行，但是在它失控之前，值得研究一下。这是一个指示性的警告，表明您的代码需要一些关注。

> “气味是代码中的某些结构，它表明违反了基本设计原则，并对设计质量产生负面影响”。代码气味通常不是 bugs 它们在技术上并非不正确，也不会妨碍程序的运行。相反，它们指出了设计中的弱点，这些弱点可能会减慢开发速度，或者增加将来出现错误或失败的风险。糟糕的代码气味可能是导致技术债务的因素的指示器。”
> -**来源** : [维基百科](https://en.wikipedia.org/wiki/Code_smell)

仅仅是代码气味的存在并不等同于 bug，但是它的气味值得关注，值得研究。所有程序员都会同意，在我们遇到 bug 之前阻止它需要更少的努力和花费更少的时间——消除代码气味是确保这一点的一种方法。

为了减少代码气味的数量，了解它们的样子是很重要的。在本文中，我们将按非时间顺序介绍其中的七个。

# #1 使用打印语句进行调试

Print 语句可能是您在编程之旅中学习的第一个内置语句之一(即大多数人的第一个程序是`print("Hello World")`)。print 语句本身没有什么错误，只是开发人员经常过于依赖它们。

你如何知道自己是否过于依赖打印报表？如果你用它来调试你的代码。

Print 语句很容易实现，因此它在欺骗人们认为这是调试代码的最佳方式方面做得非常好。然而，使用 print 进行调试通常需要您在显示必要的信息以修复代码中的错误之前执行多次程序运行迭代——这需要更长的时间，尤其是当您返回并删除所有这些信息时。

有两种解决方案比使用打印调试更好:1)使用调试器一次运行一行程序，2)使用日志文件记录程序中的大量信息，这些信息可以与以前的运行进行比较。

我更喜欢使用日志文件，这可以通过内置的`logging`模块在 Python 中轻松完成。

```
**import** logging logging.basicConfig(
    filename = "log_age.txt", 
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s") logging.debug("This is a log message.") 
```

# #2 重复代码

程序中最常见的代码味道很可能是重复的代码。识别重复代码是如此容易:您所要做的就是考虑在程序的不同部分中，您可以简单地复制和粘贴代码。因此，重复代码可以定义为在多个位置重复的代码。

```
**print**("What would you like for breakfast?")
breakfast = input()
**print**(f"One {breakfast} coming up")**print**("What would you like for lunch?")
lunch = input()
**print**(f"One {lunch} coming up")**print**("What would you like for dinner?")
dinner = input()
**print**(f"One {dinner} coming up")
```

表面上，重复代码看起来无害。当必须对代码进行更新或更改时，它就成了一个棘手的问题。更改重复代码的一个副本意味着必须对代码的所有区域进行更改，忘记这样做可能会导致程序中代价高昂且难以检测的错误。

这个问题的解决方案非常简单:对代码进行重复数据删除。通过利用函数或循环的力量，我们可以很容易地让代码在程序中出现一次。

```
**def** ask_meal(meal_of_the_day:str) -> str: 
    **print**(f"What would you like to eat for {meal_of_the_day}")
    meal = input()
    return f"One {meal} coming up"

meals_of_the_day = ["breakfast", "lunch", "dinner"]**for** meal **in** meals_of_the_day: 
    ask_meal(meal)
```

有些人将复制发挥到了极致，试图在复制和粘贴代码后的任何时候消除重复。虽然可能有一些程序员支持它，但有时它可能是多余的。复制粘贴代码一次或两次可能不会有问题，但如果出现三次，就创建一个函数或循环来修复它。

# #3 神奇的数字

有时我们不得不在代码中使用数字；我们在源代码中使用的一些数字会给其他开发人员带来极大的困惑——如果您将来不得不重新访问这些代码，也会给自己带来困惑。这些数字被称为 ***幻数*** 。

> “幻数或幻常数这个术语指的是在源代码中直接使用数字的反模式。”
> - [ **来源** : [维基百科](https://en.wikipedia.org/wiki/Magic_number_(programming))

幻数被认为是一种代码味道，因为它们没有给出任何关于它们为什么存在的指示——它掩盖了开发人员选择那个特定数字的意图。因此，您的代码可读性更差，您和其他开发人员将来更难更新或更改，并且更容易出现像打字错误这样的细微错误。

考虑以下场景:

```
**from** sklearn.model_selection **import** train_test_splitX_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    0.3, 
    0.7, 
    25, 
    True, 
    None
)
```

在上面的代码中，我们从 Scikit-learn 导入了`train_test_split`函数，并用一些似乎没有明确含义的超参数实例化了它。

使代码可读性更好的一个解决方案是添加信息性的注释，告诉我们为什么选择这个特定的数字。

```
**from** sklearn.model_selection **import** train_test_splitX_train, X_test, y_train, y_test = train_test_split(
    X # features array, 
    y # labels, 
    0.3 # test size, 
    0.7 # train size, 
    25 # random state, 
    True # shuffle, 
    None # stratify
)
```

解决这种代码味道的一个更有用的方法是使用一个 ***常量*** 。常数是每次执行程序时保持不变的有价值的数据。[我不确定其他语言，但是]在 Python 中，我们通常用大写字母来写常量，以告知他人(并提醒自己)它们的值在初始赋值后不应该改变。

您经常会看到在配置中定义的常量，或者在脚本开始时作为全局变量定义的常量。

```
**from** sklearn.model_selection **import** train_test_splitTEST_SIZE = 0.3
TRAIN_SIZE = 0.7
RANDOM_STATE = 25
SHUFFLE = True
STRATIFY = NoneX_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    TEST_SIZE, 
    TRAIN_SIZE, 
    RANDOM_STATE, 
    SHUFFLE, 
    STRATIFY
)
```

这有多大的可读性？

**使用不同的常数而不是用一个常数来解决两个问题是很重要的。这样做的原因是，它们可以在将来独立更改，这通常会减少很多麻烦。**

# **#4 保留注释掉的代码**

**当代码中的注释提供信息时，它无疑被视为一种好的实践。有时，我们甚至会暂时注释掉代码，看看在没有我们删除的代码行的情况下，剩余的代码是如何运行的——可能是在我们调试的时候——这本身也没有什么错。**

**当程序员变得懒惰时，它就会成为一个问题。这种懒惰的一个例子是注释掉代码，但是保留注释掉的代码。**

**就地注释掉代码的原因是代码味道，因为它是不明确的。其他程序员会将注释掉的代码视为一个完全的谜，并且不知道在什么条件下应该将它重新放入程序中。**

```
walk()
# run()
sprint()
stop()
```

**为什么`run()`被注释掉了？什么时候可以取消对`run()`的注释？如果不需要，那么删除代码。**

# **#5 死代码**

**为了节省计算和内存，必须处理程序中的所有死代码。**

> **“死代码是一个程序的源代码中的一部分，它被执行，但其结果永远不会在任何其他计算中使用。”
> - [ **来源** : [维基百科](https://en.wikipedia.org/wiki/Dead_code#:~:text=In%20some%20areas%20of%20computer,wastes%20computation%20time%20and%20memory.)**

**在你的程序中有死代码是非常误导人的。其他程序员阅读您的代码时可能不会马上理解，并认为它是代码的一个工作部分，而实际上，它除了浪费空间之外什么也没做。**

```
# Code source: [https://twitter.com/python_engineer/status/1510165975253069824?s=20&t=VsOWz55ZILPXCz6NMgJtEg](https://twitter.com/python_engineer/status/1510165975253069824?s=20&t=VsOWz55ZILPXCz6NMgJtEg)**class** TodoItem: 
    **def** __init__(self, state=None):
        self.state = state if state else -1 

    **def** __str__(self): 
        if self.state == -1: 
            return "UNDEFINED"
        elif self.state == 0: 
            return "UNSET" 
        else: 
            return "SET"
```

**乍一看，这段代码看起来不错，但是其中有一个 bug:这段代码永远不能被设置为 0，因为在`self.state`变量中的求值会将 0 设置为`False`。因此，将状态设置为 0 将返回`UNDEFINED`而不是`UNSET`。**

```
class TodoItem: 
    def __init__(self, state=None):
        self.state = state if state is not None else -1 

    def __str__(self): 
        if self.state == -1: 
            return "UNDEFINED"
        elif self.state == 0: 
            return "UNSET" 
        else: 
            return "SET"
```

*****注*** *:见本* [*视频*](https://www.youtube.com/watch?v=_9yJdVl-K9M&t=59s) *由 Python 工程师得到完整解释。***

# **#6 存储带数字后缀的变量**

**我已经被这种代码气味困扰过几次——直到今天我还没有完全摆脱它；有时，我们可能需要跟踪同一类型数据的几个实例。在这种情况下，重用一个名字并给它添加一个后缀，使它存储在程序中的一个不同的名称空间中，这是非常诱人的。**

```
person_1 = "John" 
person_2 = "Doe"
person_3 = "Michael
```

**这种代码味道之所以是代码味道，是因为后缀不能很好地描述每个变量中包含的内容或变量之间的差异。它也没有给出任何关于程序中有多少变量的指示——你不想搜索 1000 多行代码来确保没有其他的数字。**

**更好的解决方案是:**

```
people = ["John", "Doe", "Michael"] 
```

**不要把这当作改变所有以数字结尾的变量的指令:一些变量应该以数字结尾，尤其是当数字是你存储的数据的独特名称的一部分时。**

# **#7 不必要的类(特定于 Python)**

**像 Java 这样的编程语言使用类来组织程序中的代码。Python 使用模块。因此，试图像在 Java 中一样使用 Python 中的类(来组织代码)是不会有效的。**

**Python 中的代码不需要存在于类中，有时，使用类可能是多余的。**

**以这门课为例:**

```
**class** Human:
    **def** __init__(self, name: str): 
        self.name = name **def** introduction(self): 
        return f"Hi, my name is {self.name}"person = Human("Kurtis")
print(person.introduction()) """
Hi, my name is Kurtis
"""
```

**为什么这个类不需要成为一个类的主要决定因素是它只有一个函数。根据经验，如果一个类只包含一个方法或者只包含静态方法，那么它不一定是 Python 中的类。不如写个函数来代替。**

**要了解这个概念的更多信息，请查看 Jack Diederich 在 PyCon 2012 上关于为什么我们应该"[停止编写类](https://www.youtube.com/watch?v=o9pEzgHorH0)"的演讲。**

***感谢阅读。***

****联系我:**
[LinkedIn](https://www.linkedin.com/in/kurtispykes/)
[Twitter](https://twitter.com/KurtisPykes)
[insta gram](https://www.instagram.com/kurtispykes/)**

**如果你喜欢阅读这样的故事，并希望支持我的写作，可以考虑成为一名灵媒。每月支付 5 美元，你就可以无限制地阅读媒体上的故事。如果你使用[我的注册链接](https://kurtispykes.medium.com/membership)，我会收到一小笔佣金。**

**已经是会员了？[订阅](https://kurtispykes.medium.com/subscribe)在我发布时得到通知。**

**[](https://kurtispykes.medium.com/subscribe) [## 每当 Kurtis Pykes 发表文章时都收到一封电子邮件。

### 每当 Kurtis Pykes 发表文章时都收到一封电子邮件。通过注册，您将创建一个中型帐户，如果您还没有…

kurtispykes.medium.com](https://kurtispykes.medium.com/subscribe)**