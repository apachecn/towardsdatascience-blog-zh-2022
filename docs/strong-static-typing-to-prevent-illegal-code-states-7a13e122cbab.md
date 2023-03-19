# 防止非法代码状态的强大静态类型| Python 模式

> 原文：<https://towardsdatascience.com/strong-static-typing-to-prevent-illegal-code-states-7a13e122cbab>

## 使用 Python 静态类型专注于业务逻辑，而不是使用“使非法状态不可表示”模式来验证数据的正确性

![](img/e2ec7e05bd0ee0b4a43077be58e9c4b6.png)

由[韦斯利·廷吉](https://unsplash.com/@wesleyphotography?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

Python 的动态类型简化了生命周期很短的脚本的编写。然而，尽管 Python 经常被认为是一种“脚本语言”，但它在机器学习和数据科学领域事实上的垄断地位使得它成为大型项目的常见选择，这些项目将被使用多年。

当软件应该被长期使用时，动态类型的短期便利性可能与易于理解的需要相冲突，并且通常与有原则的软件工程相冲突，因此静态类型系统可能是受青睐的。静态类型系统在大型项目中有很多好处:它提供了在团队中协作所必需的类型文档，包括与未来的自己，但它也帮助你设计更好的软件。

我个人使用 Python 中的工具对所有预期寿命超过几个小时的软件进行静态类型化。我开始这样做是因为我真的讨厌那些试图理解我不熟悉的代码库时，我进入了带有如下签名的关键函数:

```
def process(data):
```

上面代码的问题是，不仅我们不知道它做了什么(`process`？？)，而且我们也不知道它期望接收什么样的数据以及它返回什么数据(如果有的话)。当然，我们可以阅读代码，但是当它有几十行长时，祝你好运。无论如何，这肯定比阅读一个有意义的签名要花更多的时间，比如:

```
def remove_punctuation(data: str) -> str:
```

这样的签名告诉我们，我们的函数将一个字符串作为输入，并返回一个新的字符串作为输出(它是新的，因为`str`在 python 中是不可变的)这是没有标点符号的原始字符串。我们在没有阅读实现甚至文档字符串的情况下获得了所有这些信息。

这是一个简单的例子，说明了类型系统如何使我们的代码更易读，但是当我们积极地使用类型系统而不仅仅是作为代码文档时，我们可以获得更好的结果。我们可以用较少的条件分支和正确性的静态证明来编写代码。

在继续之前，你们中的一些人可能会问，如果我们把一个不同于字符串的参数传递给上面的函数，会发生什么？显然，Python(意为 Python 解释器)不会在运行时阻止它，但是我们可以使用一个静态分析器，比如现在是官方 Python 工具的 [mypy](https://github.com/python/mypy) ，或者其他类似的第三方工具。静态分析器检查您的源代码(因此，没有运行时)来执行多种分析，通常会发现 bug。在特定情况下，mypy 检查类型是否受到尊重。编译语言在编译器中集成了这一特性，但是由于 Python 被认为是一种动态类型的语言，我们不得不依赖其他工具。

# 例如:一些复杂的代码

假设我们有运行翻译模型的代码，可以翻译文本或音频，并且一次只能给出一个输入。此外，该函数在将输入输入到模型中之前对输入运行预处理步骤。此外，文本预处理功能只能与文本输入一起传递，而音频预处理只能与音频输入一起传递。不可能混合:

```
def forward(
  model: Model, 
  text_input: Optional[str], 
  audio_input: Optional[np.ndarray],
  text_preprocessing: Optional[Callable[[str], np.ndarray]],
  audio_preprocessing: Optional[Callable[[np.ndarray], np.ndarray]],
) -> np.ndarray:
  if text_input and audio_input:
    raise ValueError("Only one between text_input and audio_input can be provided")
  if text_input:
    if not text_preprocessing:
      raise ValueError("text_preprocessing must be provided with text_input")
    if audio_preprocessing:
      raise ValueError("provided audio_preprocessing with text_input") processed_input = text_preprocessing(text_input)

    # symmetric checks on audio input
    .
    .
    .
  return model(processed_input)
```

尽管处理代码量很少，但确保数据一致性的脚手架很长、很复杂且容易出错。如果我们想改变什么，它也是脆弱的。

</machine-translation-evaluation-with-cometinho-c89880731409>  

我们现在想做的是让我们的静态分析工具，如 mypy，为我们做烦人的工作，同时我们专注于建模什么是允许的，什么是非法的，以及带来价值的代码。

这就是“让非法国家无代表性”背后的理念。

# 使非法国家没有代表性

这个想法最初是由 [Yaron Minsky](https://blog.janestreet.com/effective-ml-revisited/) 在使用像 OCaml 这样的强类型语言时推广的，但是为了我们的利益，我们可以在 Python 中应用相同的想法。

上述函数的问题是，它有 4 个单独的可选参数，我们无法从签名中推断出允许哪些参数组合。我们通过应用我们对问题的了解来进行改进。我们必须提供一个且只有一个输入，一个且只有一个预处理函数。因此，与其使用不描述彼此之间关系的可选值，不如让我们用 Python 类型和代码来表达这个想法。

```
Text_PP = Callable[[str], np.ndarray]
Audio_PP = Callable[[np.ndarray], np.ndarray]def forward(
  model: Model, 
  input: Union[str, np.ndarray], 
  preprocessing: Union[Text_PP, Audio_PP],
) -> np.ndarray:
  if (isinstance(input, str) and isinstance(preprocessing, Audio_PP)) or (isinstance(input, np.ndarray) and isinstance(preprocessing, Text_PP)):
    raise ValueError("Illegal combination of input and preprocessing type")
```

我们首先为预处理函数(Text_PP 和 Audio_PP)的函数类型定义类型别名，这更易于内联编写。

那么，我们在这个新定义中的输入可以是一个 str 或一个 np.ndarray。没有一个是不合法的，也不可能两者都有。预处理也是一样，可以是 Text_PP，也可以是 Audio_PP，不能两者都是，也可以都不是。

我们在这里所做的是让我们的签名接管一些验证工作，我们可以减少函数体中的检查。

然而，非法状态仍然是可表示的，因为我们可以有一个 str 类型的输入并预处理 Audio_TT，或者输入 np.ndarray 并预处理 Text_PP。

我们的第一个函数签名的问题是大量的参数，这些参数是可替换的。通过减少参数的数量，我们成倍地减少了可能的非法状态的数量。

然后，我们可以再做一次，接受一个包含输入序列和正确类型的预处理函数的单个参数。

```
from abc import ABC
from dataclasses import dataclassText_PP = Callable[[str], np.ndarray]
Audio_PP = Callable[[np.ndarray], np.ndarray]class DataWithFunc(ABC):
  def __init__(self, data, func):
    self.data = data
    self.func = func def apply(self) -> np.ndarray:
    return self.func(self.input)@dataclass
class TextWithPreprocess(DataWithFunc):
  data: str
  func: Text_PP@dataclass
class AudioWithPreprocess(DataWithFunc):
  data: np.ndarray
  func: Audio_PPdef forward(
  model: Model, 
  input: DataWithFunc, 
) -> np.ndarray:
  return input.apply()
```

我们需要做一些搭建工作，这在这个小例子中看起来像是额外的工作，但是对于那些类型可以并且应该被重用的大型项目来说非常有帮助。

我们需要定义一个新的类型 DataWithFunc，它包含数据和一个函数，并且有一个方法将函数应用于数据。然后，我们定义了 DataWithFunc 的两个子类型，即 TextWithPreprocess 和 audiowithprocessor，每个子类型定义了两个有效输入组合中的一个。最后，我们的 forward 函数现在只接受模型和一个 DataWithFunc 类型的新参数作为参数，由于它是一个抽象类(ABC ),我们需要使用一个有效的子类型:现在只有合法的表示是有效的。

</python-polymorphism-with-class-discovery-28908ac6456f>  

这种设计的一个好处是，我们可以通过再次为新的有效组合子类化 DataWithFunc 来添加输入和预处理类型的新组合。因为我们的输入是 DataWithFunc 类型，所以签名不需要改变。

# 结论

“使非法状态不可表示”是一个简单而强大的想法，旨在通过巧妙使用类型系统来设计更优雅的代码。我们没有编写验证代码来确保输入数据的正确性，而是显式地对允许的状态建模，并让类型检查器来验证代码使用的正确性。

这是一种强大的技术，需要实践和对语言类型系统的良好了解，但在代码可维护性和可读性方面有巨大的回报。

我看到的唯一缺点是，它有时会导致“过度杀伤”的解决方案，并在不必要时导致过于复杂的代码，或者它不会带来任何好处。然而，只有不断地练习，你才能知道什么时候该用，什么时候不该用。

# 中等会员

你喜欢我的文章吗？你是否正在考虑申请一个中级会员来无限制地阅读我的文章？

如果您决定通过此链接订阅，您将通过您的订阅支持我，无需为您支付额外费用[https://medium.com/@mattiadigangi/membership](https://medium.com/@mattiadigangi/membership)

# 进一步阅读

</dynamically-add-arguments-to-argparse-python-patterns-a439121abc39>  </tips-for-reading-and-writing-an-ml-research-paper-a505863055cf> 