# Python 依赖注入模式的简短解释

> 原文：<https://towardsdatascience.com/a-fairly-short-explanation-of-the-dependency-injection-pattern-with-python-4f11e7ee27f6>

## 深入研究这种非常有用但经常被忽视的设计模式

![](img/9dc268097a0b23dfcaf614e1d19e40aa.png)

一张注射器的照片，因为这是关于依赖性注射，你明白吗？哈哈哈。好吧，我是想开个小玩笑，但恐怕这是一种尝试。你明白了吗？哈哈哈。(不好意思)。戴安娜·波列希纳在 Unsplash 上的照片

依赖注入作为一个概念既不性感也不酷，就像几乎所有的设计模式一样。尽管如此，如果利用得当，它还是非常有用的——同样，几乎和任何设计模式一样。希望在这篇文章结束时，你会有另一个工具来使用你的皮带。

让我们去争取吧。

# 依赖注入是什么意思？

依赖注入是一个简单概念的花哨术语:给对象它需要的东西，而不是让它自己创建它们。这是有用的，原因有几个:它使我们的代码更加模块化，更容易测试，它可以使我们的程序更加灵活，更容易扩展。

这里有一个例子来说明这个概念。假设我们有一个代表超级英雄的类。这个类有一个`name`属性和一个`power`属性，它有一个`use_power()`方法，该方法打印出一条消息，说明超级英雄如何使用他们的能力。这个类可能是这样的:

```
class Superhero:
    def __init__(self):
        self.name = 'Spider-Man'
        self.power = 'spider-sense'

    def use_power(self):
        print(f'{self.name} uses their {self.power} power!')

# create a superhero and use their power
spiderman = Superhero()
spiderman.use_power()  # prints "Spider-Man uses their spider-sense power!"
```

这很好，但是有一个问题:我们的`Superhero`类正在创建它自己的名字和能力属性。这意味着我们创造的每个超级英雄都将拥有相同的名字和力量，除非我们在创建对象时明确设置它们。这不是很灵活或模块化。

为了解决这个问题，我们可以使用依赖注入。我们可以将它们作为参数传递给`__init__()`方法，而不是让`Superhero`类创建自己的名称和能力属性。这可能是这样的:

```
class Superhero:
    def __init__(self, name, power):
        self.name = name
        self.power = power

    def use_power(self):
        print(f"{self.name} uses their {self.power} power!")

# create a superhero with the name "Superman" and the power "flight"
superman = Superhero("Superman", "flight")

# use the superhero's power
superman.use_power()  # prints "Superman uses their flight power!"

# create a superhero with the name "Batman" and the power "money"
batman = Superhero("Batman", "money")

# use the superhero's power
batman.use_power()  # prints "Batman uses their money power!"
```

如您所见，使用依赖注入使我们的`Superhero`类更加灵活和模块化。我们可以创造任何名字和权力的超级英雄，我们可以很容易地交换不同超级英雄的名字和权力。

# 这方面有哪些真实的用例？

## **测试**

依赖注入使得为我们的代码编写单元测试变得更加容易。因为我们可以将类所依赖的对象作为参数传入，所以我们可以很容易地将真实对象替换为我们可以在测试中控制的模拟或存根对象。这允许我们单独测试我们的类，并确保它的行为符合预期。

## **配置**

依赖注入可以使我们的代码更具可配置性。例如，我们可能有一个发送电子邮件的类。这个类可能依赖于实际发送电子邮件的`EmailClient`对象。我们可以使用依赖注入将其作为参数传入，而不是将`EmailClient`对象硬编码到我们的类中。这样，我们可以很容易地改变我们的类使用的`EmailClient`对象，而不用修改类本身。

这是没有依赖注入的代码的样子:

```
class EmailSender:
    def __init__(self):
        self.email_client = GoodMailClient()

    def send(self, email_text, recipient):
        return self.email_client.send(email_text, recipient)

    # other methods that use self.mail_client
```

这是依赖注入的样子:

```
class EmailSender:
    def __init__(self, email_client):
        self.email_client = email_client

    def send(self, email_text, recipient):
        return self.email_client.send(email_text, recipient)

    # other methods that use self.mail_client
```

第二种方法允许您非常容易地从这个:

```
# Create an instance of a good email client
good_email_client = GoodMailClient()

# Create an instance of EmailSender, injecting the dependency
sender = EmailSender(email_client=good_email_client)

# Send the mail
sender.send('Hey', 'you@mail.com')
```

对此:

```
# Create an instance of a better email client
better_email_client = BetterMailClient()

# Create an instance of EmailSender, injecting the dependency
sender = EmailSender(email_client=better_email_client)

# Send the mail
sender.send('Hey', 'you@mail.com')
```

…根本不修改`EmailSender`类。

## **扩展**

依赖注入可以使我们的代码更具可扩展性。例如，我们可能有一个处理数据的类。这个类可能依赖于一个执行实际数据处理的`DataProcessor`对象。我们可以使用依赖注入将它作为参数传入，而不是让我们的类创建自己的`DataProcessor`对象。这样，如果我们想扩展类的功能，我们可以很容易地用不同的对象替换掉`DataProcessor`对象。

这只是几个例子，但是在现实世界中还有很多其他的依赖注入用例。

# 听起来很棒，有什么好处？

嗯，没有捕捉到*本身*，但当然有一些问题。一个主要的缺点是它会使我们的代码更加冗长和难以阅读，特别是当我们注入大量的依赖项时。这会使我们的代码更难理解和维护。

另一个潜在的缺点是依赖注入会使我们的代码更难调试。由于依赖项是作为参数传入的，所以如果出现问题，就很难跟踪错误的来源。

此外，依赖注入会使理解类或模块的依赖关系变得更加困难。由于依赖关系是从外部传入的，所以很难一眼看出一个对象依赖于什么，这就很难理解它是如何工作的。

总的来说，虽然依赖注入有很多好处，但它也有一些缺点。与任何软件设计模式一样，重要的是权衡利弊，并决定它是否是给定情况下的正确方法。

# 最后的想法

从我的经验来看，我想强调的是，这只是我的拙见，从一开始就使用这种模式通常是个好主意，甚至是 MVP。这听起来可能更复杂(事实也确实如此)，但是当你得到这个钻头时，它是一个不需要动脑筋的东西，并且可以灵活地进行你以后可能想要添加的任何进一步的修改。然而，关注注入的依赖项的数量也是一个好主意——您不希望它增长太多。

我个人发现的最常见的缺点是代码变得难以理解，尤其是当团队中有新成员加入时。但是，通过适当的入职培训，这个问题很容易解决:收益大于成本。

# 参考资料:

[1] M. Seeman，[依赖注入是松散耦合的](https://blog.ploeh.dk/2010/04/07/DependencyInjectionisLooseCoupling/) (2010)，Ploeh 博客

[2]哈佛 S .[依赖性注入的弊端](https://stackoverflow.com/q/2407540) (2010)，StackOverflow

[3] A .卡尔普，[依赖注入设计模式](https://learn.microsoft.com/en-us/previous-versions/dotnet/netframework-4.0/hh323705(v=vs.100)?redirectedfrom=MSDN) (2011)，MSDN

*如果你喜欢阅读这样的故事，你可以直接支持我在 Medium 上的工作，并通过使用我的推荐链接* [*这里*](https://medium.com/@ruromgar/membership) *成为会员而获得无限的访问权限！:)*