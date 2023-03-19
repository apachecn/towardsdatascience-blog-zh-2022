# Python 类中属性的指示和捉迷藏隐私

> 原文：<https://towardsdatascience.com/indication-and-hide-and-seek-privacy-of-attributes-in-python-classes-3f64f9d6da88>

## PYTHON 编程

## 了解“公共”和“私有”在 Python 类及其属性的上下文中的含义。

![](img/65e9ba807a96e63136582dcae40b2078.png)

照片由[戴恩·托普金](https://unsplash.com/@dtopkin1?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

# 公共与私人——说与想

一般在编程中，当某个东西是公共的，你就可以访问它，使用它；当它是私人的时候，你不能。这就像想某事和说某事:当你想某事时，它仍然是你的；但是无论你大声说什么，都不再仅仅是你的，而是公开的。

Python 中的工作方式不同。你可能听说过在 Python 中没有什么是真正私有的。这是什么意思？Python 有私有属性和方法吗？

我们在 Python 类的方法和属性的上下文中使用这两个术语，*公共*和*私有*。当一个属性是私有的，你不应该使用它；当一个方法是私有的，你不应该调用它。你可能注意到我用了“应该”这个词。正如我已经提到的，这是因为 Python 中的工作方式不同:当某个东西是公共的时，你可以访问和使用它；当这是私人的事情时，你不应该做，但这并不意味着你不能做。因此，当你用 Python 思考某件事时，它应该属于你——但是任何人都可以通过简单的方法听到它。

如你所见，Python 在隐私方面并不严格。它建议你遵守一些规则，而不是让你去遵守它们。它建议一个类的用户*而不是*访问私有方法和属性——但是用户无论如何都可以这样做，更重要的是，他们不必为此付出太多努力。

在这篇文章中，我将用简单的词语和例子来解释这些事情。

> 当你想某事时，它仍然是你的；但是无论你大声说什么，都不再仅仅是你的，而是公开的。
> 
> 当你用 Python 思考某件事时，它应该属于你——但是任何人都可以通过简单的方法听到它。

# “私有”方法和属性

Python 中没有真正的隐私。Python 提供的是伪隐私，或者准隐私。它有两个层次，我称之为*指示隐私*和*捉迷藏隐私*。

## 指示隐私

您可以指示特定属性是私有的。要做到这一点，只需在其名称中添加一个前导下划线即可。这样做，你*指示*，或者*建议*，或者*建议*，该方法/属性应该被视为私有的，意味着它不应该在类之外使用。

因此，`instance.do_it()`是一个常规(公共)方法，而`instance._do_it()`是一个表示为私有的方法。因此，作为该类的用户，您被要求不要使用它。它在那里是因为它服务于一些实现的目的——而你与它无关。这不是秘密。你可以看一下，没有人对你隐瞒什么。但这不是给你的。接受别人给你的东西，不要碰别人给你的东西。

让我们考虑一个简单的例子:

```
# class_me.py
class Me:
    def __init__(self, name, smile=":-D"):
        self.name = name
        self.smile = smile
        self._thoughts = []

    def say(self, what):
        return str(what)

    def smile_to(self, whom):
        return f"{self.smile} → {whom}"

    def _think(self, what):
        self._thoughts += [what]

    def _smile_to_myself(self):
        return f"{self.smile} → {self.name}"
```

(如果你不知道为什么我写的是`self._thoughts += [what]`而不是`self._thoughts += what`，请访问附录 1。)

好的，我们有一个类`Me`，它代表你——至少在你创建它的时候。它具有以下属性:

*   `.name`，一个公共属性→你的名字肯定是公共的。
*   `.smile`，一个公共属性→你的笑容在外面是可见的，所以肯定是公共的。
*   `._thoughts`，一个私人属性→你的想法*肯定是*私人的，不是吗？

如您所见，两个公共属性的名称没有前导下划线，唯一的私有属性的名称有。

现在让我们来看看可用的方法:

*   `.say()`，一个公开的方法→当你说一件事的时候，别人能听见，所以你的话是公开的。
*   `.smile_to()`，一个公共方法→当你对某人微笑时，这个人和周围的人都能看到你在微笑。
*   `._smile_to_myself()`，一个私密的方法→这是一种别样的微笑；它是为类的作者保留的(在我们的例子中，是为您保留的)，并且是在没有人注意的时候完成的——这就是为什么它是一个私有方法。
*   `._think()`，私法→当你想某件事的时候，那是你的私以为；如果要大声说出来，就要用 public `.say()`的方法。

让我们和全班一起玩。我将为自己创建该类的实例，所以我将它命名为`marcin`。您可以为自己创建一个实例。

```
>>> from class_me import Me
>>> marcin = Me(name="Marcin")
>>> marcin # doctest: +ELLIPSIS
<__main__.Me object at 0x...>
>>> marcin.say("What a beautiful day!")
'What a beautiful day!'
>>> marcin.smile_to("Justyna")
':-D → Justyna'
```

![](img/aaf5925c14a2019d9966d47b3f34c388.png)

我使用了`doctest`来格式化上面块中的代码。它帮助我确保代码是正确的。您可以从下面的文章中了解关于这个文档测试框架的更多信息:

</python-documentation-testing-with-doctest-the-easy-way-c024556313ca>  

如果您想将代码复制并粘贴为 doctest，并自己以这种方式运行，请访问本文末尾的附录 2，其中包含以这种方式格式化的剩余代码(例如，`Me`类的代码)。

![](img/b8397a4f4b1c870e04ae5c6a57b955a6.png)

好的，一切看起来都很好。然而，到目前为止，我们还算客气，甚至还没有看私有方法和属性；我们只用过公共的。是时候淘气一点了:

```
>>> dir(marcin)  #doctest: +NORMALIZE_WHITESPACE
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', 
'__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', 
'__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', 
'__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', 
'__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 
'__weakref__', '_smile_to_myself', '_think', '_thoughts', 'name', 
'say', 'smile', 'smile_to']
```

我们看到了什么？实际上，一切。我们当然会看到公共属性`.name`和`.smile`以及公共方法`.say()`和`.smile_to()`。但是我们也看到私有属性`._thoughts`和私有方法`._think()`和`._smile_to_myself()`。此外，我们会看到更多不是我们创建的方法和属性。

请记住，使用`.__name__()`约定命名的方法是 dunder 方法，而不是私有方法。我们改天再谈这个。

既然我们能够看到私有属性，很可能我们也能够使用它们:

```
>>> marcin._think("My wife is so beautiful!")
>>> marcin._think("But let this be my secret!")
```

什么都没发生？那也许就没事了？也许我们可以使用私人方法，但无论他们在做什么都瞒着我们？

当然不是。只是`._think()`方法不返回任何东西(或者说返回`None`)，而是将想法保存到`._thoughts`属性，也是私有的。让我们看看你是否能看到我的私人想法:

```
>>> marcin._thoughts
['My wife is so beautiful!', 'But let this be my secret!']
```

是的，你可以。最后一个测试:让我们看看你是否能看到我对自己微笑:

```
>>> marcin._smile_to_myself()
':-D → Marcin'
```

你也可以。因此，您可以清楚地看到私有属性，并且可以使用私有方法——尽管我通过在这些属性和方法的名称前添加下划线明确指出它们是私有的，所以我不希望您使用它们。使用私有方法或属性有点像在淋浴时偷窥我——你可以看到我想对你隐藏的东西。

然而，有时出于这样或那样的原因，您可能想要修改一个现有的类；这可能意味着覆盖私有属性或方法。这就是 Python 方法的亮点。理论上，这些属性是私有的，所以你不应该使用它们；有时，使用它们甚至可以打破一个类。这也是一种保护措施。你知道这些属性是隐私，所以最好不要碰它们。

但是当你知道你在做什么，当你的目的要求你使用私有属性时——Python 使这成为可能。这为 Python 开发人员带来了许多额外的机会。

> 使用私有方法或属性有点像在淋浴时偷窥我——你可以看到我想对你隐藏的东西。
> 
> 这为 Python 开发人员带来了许多额外的机会。

有点夸张，在 Python 里你可以为所欲为。您可以覆盖内置函数、异常等。(如果你想了解更多关于重写异常的信息，请阅读[这篇*更好编程*文章](https://medium.com/better-programming/how-to-overwrite-asserterror-in-python-and-use-custom-exceptions-c0b252989977)。)并且可以使用私有属性。这很好，假设——就像任何代码的情况一样——您不想对用户的计算机造成任何损害。

我相信你会同意这种隐私是脆弱的，因为用户可以像使用公共属性和类一样使用私有属性和类。然而，Python 提供了一种更严格的隐私方法，我称之为*捉迷藏隐私*。

## 捉迷藏隐私

隐私的指示级别仅包括指示属性是私有的还是公共的，而捉迷藏级别则更进一步。你马上就会看到，在某种程度上，它帮助你保护私有属性。

这是否意味着这一次，私有属性和方法将真正被隐藏，用户将无法使用它们？不完全是。正如我所写的，捉迷藏隐私提供了某种程度的保护——但不是完全的保护。Python 之所以能做到这一点，要归功于一种叫做 [name mangling](https://en.wikipedia.org/wiki/Name_mangling#Python) 的方法。

当您想要使用名称篡改，因此需要隐藏隐私时，您需要向私有属性的名称添加两个前导下划线，而不是一个。在我们的`Me`类中，比如说，`.__thoughts`和`.__think()`。多亏了名称管理，私有属性或方法以一种特殊的方式被修改，使得从类外部访问它们变得更加困难。

让我们在工作中看到这一点。我们先修改我们的`Me`类；让我们把它的名字改成`PrivateMe`(关于`doctest` ing 格式的代码，见附件 2):

```
# class_me.py
class PrivateMe:
    def __init__(self, name, smile=":-D"):
        self.name = name
        self.smile = smile
        self.__thoughts = []

    def say(self, what):
        return str(what)

    def smile_to(self, whom):
        return f"{self.smile} → {whom}"

    def __think(self, what):
        self.__thoughts += [what]

    def __smile_to_myself(self):
        return f"{self.smile} → {self.name}"
```

首先，让我们创建一个实例—同样，这将是我的一个实例—并使用公共方法:

```
>>> marcin = PrivateMe(name="Marcin")
>>> marcin.say("What a beautiful day!")
'What a beautiful day!'
>>> marcin.smile_to("Justyna")
':-D → Justyna'
```

(如果你在疑惑 Justyna 是我老婆还是我在对另一个女生微笑，你可以放心；她是！)

到目前为止一切顺利，但这并不令人惊讶——毕竟，我们已经使用了公共方法。以前，我们成功地使用了私有方法，比如`._smile_to_myself()`。让我们试试这次是否能成功。为了验证这一点，我会试着用`.__smile_to_myself()`方法对自己微笑:

```
>>> marcin.__smile_to_myself()
Traceback (most recent call last):
    ...
AttributeError: 'PrivateMe' object has no attribute '__smile_to_myself'
```

哈！我们知道`PrivateMe`类有`__smile_to_myself()`方法，但是我们不能使用它。显然，它是受保护的，任何私有方法都应该如此。

尽管如此……看起来这个方法是完全受保护的，而不久前我还声称在 Python 中，私有属性没有受到完全保护。这是怎么回事？

我们刚刚经历了如何命名 mangling 工程。它隐藏了私有属性——或者说，不管听起来有多奇怪，它隐藏了私有属性的名称。换句话说，它以一种特殊的方式改变了他们的名字；新名称将遵循以下`_ClassName__attribute`符号:

```
class MyClass:
    __privacy = None     # this becomes ._MyClass__privacy    
    def __hide_me(self): # this becomes ._MyClass__hide_me()
        pass
```

这样，不能使用属性的原始名称访问属性，但可以使用通过名称管理更新的名称访问属性。在我们的`PrivateMe`类中，它是这样工作的:

```
>>> marcin._PrivateMe__smile_to_myself()
':-D → Marcin'
```

你可以看到这个属性就在那里，只是被重命名了。我们肯定会在`dir()`函数的输出中看到这一点:

```
>>> dir(marcin) # doctest: +NORMALIZE_WHITESPACE
['_PrivateMe__smile_to_myself', '_PrivateMe__think',
 '_PrivateMe__thoughts', '__class__', '__delattr__',
 '__dict__', '__dir__', '__doc__', '__eq__', '__format__',
 '__ge__', '__getattribute__', '__gt__', '__hash__',
 '__init__', '__init_subclass__', '__le__', '__lt__',
 '__module__', '__ne__', '__new__', '__reduce__',
 '__reduce_ex__', '__repr__', '__setattr__',
 '__sizeof__', '__str__', '__subclasshook__',
 '__weakref__', 'name', 'say', 'smile', 'smile_to']
```

我们的私有方法和属性可以使用新的名称:

*   `.__smile_to_myself()` → `._PrivateMe__smile_to_myself()`
*   `.__think()` → `._PrivateMe__think()`
*   `.__thoughts` → `._PrivateMe__thoughts`

名称篡改使我们能够实现隐私的捉迷藏水平。

还记得一件事。当您想要通过添加两个前导下划线来使属性成为私有属性时，不要在名称末尾添加两个额外的下划线。这样命名的方法就变成了所谓的 dunder(***d***double-*score)方法——而且这些绝对是*不是*私有的；实际上，它们是与私有相对的。我们改天再谈。要使用名称管理，记住这条命名规则就足够了:不要对私有方法使用`.__name__()`约定，因为这不起作用。*

*![](img/87840f875600b7bc9d14c9b5950072c9.png)*

*克里斯蒂安·沃克在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片*

# *结论*

*我们已经在 Python 面向对象编程的上下文中讨论了隐私的概念。编写类时，有时可能希望隐藏一些实现细节，通过将类的一些属性和方法设为私有，可以实现这一点。但它们从来都不是真正的隐私。*

*这种方法对我来说听起来不自然。当我想到一个私有属性时，我把它想象成一个不能在类外看到和使用的属性。同样，它是一个公共属性，可以以这种方式看到和使用。*

*如果您的想象力以类似的方式工作，您需要使用改变世界的眼镜，以便您可以在 Python 世界中移动，而不会不时摔倒。每次用 Python 都要戴上这种眼镜。迟早，它们会帮助您习惯 Python 的不同世界，在这个世界中，隐私的概念是如此不同。*

> *您需要使用改变世界的眼镜，这样您就可以在 Python 世界中移动，而不会不时摔倒。*
> 
> *迟早，它们会帮助您习惯 Python 的不同世界，在这个世界中，隐私的概念是如此不同。*

*总之，Python 不能让你完全保护一个类的属性。然而，它提供了两级保护，我称之为*指示*和*捉迷藏*隐私。*

**指示隐私*。您可以将一个属性指定为私有，并相信没有人会在类之外使用该属性。指示方法是基于信任的:我们相信类的用户不会使用它的私有属性。除此之外，该方法不使用任何保护措施。*

> *指示方法是基于信任的:我们相信类的用户不会使用它的私有属性。除此之外，该方法不使用任何保护措施。*

**捉迷藏隐私*。这是更高层次的隐私——就类属性的隐私而言，我们可以从 Python 中获得最多的隐私。在指示 privacy 的情况下，你可以像使用 public 属性一样使用 private 属性，但是在这里你不能。你的私人属性得到了一定程度的保护。它仍然不是完全的保护；私有属性由于名字的改变而被隐藏。你仍然可以找到、访问和使用它们——但至少它们受到了某种程度的保护。它们并没有真正被隐藏，因为`dir()`将向我们展示所有的类属性，包括公共的和私有的，但是后者将会改变名称。*

*感谢阅读这篇文章。我希望 Python 类环境中的隐私不再对您构成问题。虽然乍一看这个主题似乎很难，或者至少很奇怪，但是您会很快习惯 Python 隐私的奇怪世界。请放心，许多 Python 开发人员都很欣赏这些东西在 Python 中的工作方式。如果你不这样做，你迟早会加入他们的行列。*

*对我来说，我不仅不反对 Python 对待隐私的方式，我甚至欣赏它。我多次使用过这种方法，知道它就在那里是很好的，以防万一，等待我去窥探类的属性和方法。*

*如果你喜欢这篇文章，你也可以喜欢我写的其他文章；你会在这里找到它们。如果你想加入 Medium，请使用我下面的推荐链接:*

*<https://medium.com/@nyggus/membership>  

# 脚注

记住，在 Python 中，方法是类的属性。因此，每当我提到属性的私有性时，我指的是属性的私有性，包括方法。

命名有两个目的:

*   它增加了对类的私有属性和方法的保护级别。
*   它确保父类的私有属性不会被从它继承的类覆盖。因此，当您使用两个前导下划线时，您不必担心类中的这个属性会被继承类覆盖。

本文讲的是第一点。然而，第二个问题超出了本文的范围；我们改天再讨论它。

# 附录 1

这个附录解释了为什么在编写`Me`类时，我写道

```
self._thoughts += [what]
```

而不是

```
self._thoughts += what
```

就地串联`+=`的工作方式如下:

```
>>> x = [1, 2, 3]
>>> y = [4, 5, 6]
>>> x += y
>>> y
[4, 5, 6]
>>> x
[1, 2, 3, 4, 5, 6]
```

如您所见，该操作添加了两个列表；作为就地操作，它影响第一个，而第二个保持不变。然而，这不适用于不可迭代的对象，比如数字(这里，`int`):

```
>>> x += 5
Traceback (most recent call last):
    ...
TypeError: 'int' object is not iterable
```

因此，您可以使用就地串联向列表中添加另一个 iterable，如列表、元组、`range`对象和生成器:

```
>>> x += (10, 20)
>>> x
[1, 2, 3, 4, 5, 6, 10, 20]
>>> x += range(3)
>>> x
[1, 2, 3, 4, 5, 6, 10, 20, 0, 1, 2]
>>> x += (i**2 for i in range(3))
>>> x
[1, 2, 3, 4, 5, 6, 10, 20, 0, 1, 2, 0, 1, 4]
```

字符串也是可迭代的，所以您也可以将它们添加到列表中:

```
>>> x += "Cuma"
>>> x
[1, 2, 3, 4, 5, 6, 10, 20, 0, 1, 2, 0, 1, 4, 'C', 'u', 'm', 'a']
```

如您所见，`"Cuma"`字符串被视为其单个字符的可重复项，添加到`x`的是这些字符，而不是单词本身。

这就是为什么`self._thoughts += what`不起作用的原因。如果我们使用它，我们将会达到以下不良效果:

```
>>> marcin._think("I am tired.")
>>> marcin._thoughts
['I', ' ', 'a', 'm', ' ', 't', 'i', 'r', 'e', 'd', '.']
```

因此，我们需要将思想添加到`._thoughts`中作为列表的元素，即`[what]`。这个单元素列表是要添加到`._thoughts`的 iterable。

# 附录 2

为`doctest`格式化的类`Me`:

```
>>> class Me:
...     def __init__(self, name, smile=":-D"):
...         self.name = name
...         self.smile = smile
...         self._thoughts = []
...     def say(self, what):
...         return str(what)
...     def smile_to(self, whom):
...         return f"{self.smile} → {whom}"
...     def _think(self, what):
...         self._thoughts += [what]
...     def _smile_to_myself(self):
...         return f"{self.smile} → {self.name}"
```

为`doctest`格式化的类`PrivateMe`:

```
>>> class PrivateMe:
...     def __init__(self, name, smile=":-D"):
...         self.name = name
...         self.smile = smile
...         self.__thoughts = []
...     def say(self, what):
...         return str(what)
...     def smile_to(self, whom):
...         return f"{self.smile} → {whom}"
...     def __think(self, what):
...         self.__thoughts += [what]
...     def __smile_to_myself(self):
...         return f"{self.smile} → {self.name}"
```