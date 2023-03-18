# 代码验证和测试实践

> 原文：<https://towardsdatascience.com/code-validation-testing-practices-86a304fd3ca>

## 组合各种断言、if 语句、pytest 等

![](img/6384884ba9e6f2f77e141bfa3479b9fb.png)

来自[像素](https://www.pexels.com/ko-kr/photo/5473298/)的免费使用照片

# 介绍

我们编写的代码必须按照我们预期的方式运行。验证这一点的过程被称为“代码测试”或“代码验证”。包括 Git 在内的版本控制和协作工具促进了这一过程。Git 不在本文讨论的范围之内，所以我将在其他时间讨论它。但是我想向您介绍一些关键的代码验证实践，它们可以帮助您保持您编写的代码和您处理的数据的良好质量。

# 定义错误情况

我个人认为定义潜在的错误来源和场景对代码验证至关重要。因此，记录整个生产流程中出现的错误并定期更新该列表非常重要。

# If 语句和 Assert 语句

一旦这个错误列表被充实，创建一个单独的测试/验证脚本(例如 testing.py)。检查每个错误的 If-Else 语句在这种脚本中非常有用。

断言语句是涉及使用“断言”的语句，顾名思义，断言某个逻辑为真。如果该逻辑结果为假，它将停止正在运行的脚本或程序，并返回 AssertionError。这使得调试和验证代码变得非常方便。

看看下面这个简单的例子。

```
number = 2
assert number > 0

number = -2
assert number > 0
Traceback (most recent call last):
    ...
AssertionError
```

上面的代码验证名为“number”的变量是否是大于 0 的值。如果不是，则返回 AssertionError。这种断言验证称为“比较断言”。从身份断言到成员断言，还有其他种类的断言可用(例如，检查某个元素是否是一个更大的组的一部分，比如 list)。下面的代码是成员断言的一个示例。

```
numbers = [1, 2, 3, 4, 5]
assert 1 in numbers
assert 2 in numbersassert 3 in numbers

assert 6in numbers
Traceback (most recent call last):
    ...
AssertionError
```

# 使用 __debug__

使用调试模式是 Python 中的内置常量。它默认为 True。它是一个“常量”,因为一旦 Python 解释器开始运行，它的值就不能改变。[1]

我们如何利用这个变量呢？请看看下面这个例子。

```
if __debug__:
    if not <some expression>:
        raise AssertionError(assertion_message)## Source: [https://realpython.com/python-assert-statement/](https://realpython.com/python-assert-statement/)
```

上面的代码等效于下面使用 assert 的代码。

```
assert <some expression>, assertion_message
```

我们如何使用这个 __debug__ 变量类似于我们如何使用 assert 语句，但是一个主要的区别是 __debug__ 为我们提供了在调试或非调试模式下运行脚本的灵活性。Python 的 python -o 命令将 __debug__ 变量设置为 false，允许用户

# Pytest

Pytest 是一个用于代码测试的 Python 库。它是一个框架，允许你使用上面的所有东西和许多其他东西来系统地测试你写的各种功能或代码。我将向您介绍一些有用的案例和 Pytest 应用程序示例。

## 装置

```
pip install -U pytest
```

## 示例 1:验证一个函数

假设我们想使用一些测试用例来验证您编写的名为“func”的函数是否按照您想要的方式工作。这里，测试用例将是 func(3) == 5。

```
**def** func(x): **return** x + 1**def** test_answer( ): **assert** func(3) == 5
```

我们将该脚本命名为 test.py，并在命令 shell 中运行以下内容:

```
pytest test.py
```

这是结果的样子。

```
>>>> $ pytest**=========================== test session starts ============================**platform linux -- Python 3.x.y, pytest-6.x.y, py-1.x.y, pluggy-0.x.ycachedir: $PYTHON_PREFIX/.pytest_cacherootdir: $REGENDOC_TMPDIRcollected 1 itemtest_sample.py F                                                     [100%]================================= FAILURES =================================**_______________________________ test_answer ________________________________**def test_answer():>       assert func(3) == 5**E       assert 4 == 5****E        +  where 4 = func(3)****test.py**:6: AssertionError========================= short test summary info ==========================FAILED test_sample.py::test_answer - assert 4 == 5============================ **1 failed** in 0.12s =============================
```

引发 AssertionError 是因为 func(3)返回 4，但我们断言它是 5。这很好，因为我们知道 func 正在以我们想要的方式工作。

## 示例 2:验证执行更复杂任务的函数

Pytest 还可以用来验证执行更复杂任务的函数或部分脚本，比如从文本中提取情感。你只需要定义一个函数的预期行为和非预期行为。看看这个来自[教程](/pytest-for-data-scientists-2990319e55e6)的例子，使用 Textblob，一个流行的用于情感提取的 NLP 库。

```
**## sentiment.py
### From** [https://towardsdatascience.com/pytest-for-data-scientists-### 2990319e55e6](/pytest-for-data-scientists-2990319e55e6)**from** textblob **import** TextBlob**def** extract_sentiment(text: str): '''Extract sentiment using textblob. Polarity is within range [-1, 1]''' text = TextBlob(text) return text.sentiment.polarity **def** test_extract_sentiment( ): text = “I think today will be a great day” sentiment = extract_sentiment(text) assert sentiment > 0
```

在测试用例中，我们将`extract_sentiment`函数应用于一个示例文本:“我认为今天将是伟大的一天”。我们认为该语句包含积极情绪，因此我们将 assert 语句设计为`assert sentiment > 0`。

奔跑

```
pytest sentiment.py
```

我们得到了

```
>> ========================================= test session starts ==========================================

platform linux -- Python 3.8.3, pytest-5.4.2, py-1.8.1, pluggy-0.13.1collected 1 itemprocess.py .                                                                                     [100%]========================================== 1 passed in 0.68s ===========================================
```

测试通过了，没有出现任何错误！

## 将多个测试分组到一个类中

您还可以使用类在 Pytest 中执行多个测试。您可以在一个类中捆绑多个测试，如下所示:

```
**class** **TestClass**: **def** first_test(self): [ some test ] **def** second_test(self): [ some test ]
```

如果其中一个测试未能通过，您将在类似于以下内容的输出消息中得到通知:

```
.....................................
========================= short test summary info ==========================FAILED test_class.py::TestClass::second_test- AssertionError: assert False**1 failed**, 1 passed in 0.56s
```

# JSON 模式验证

JSON 代表**J**ava**S**script**O**object**N**rotation。它是一种基于文本的标准格式，使用 JavaScript 对象语法表示结构化数据，因此得名于 JavaScript。该结构看起来类似于 Python 中的嵌套字典。

JSON 文件中最常见的错误之一是模式或格式问题不一致。jsonschema 库允许我们轻松地验证 JSON 文件模式。

验证 JSON 模式可以简单到两行，如下所示。

```
**from** jsonschema **import** validatevalidate(instance=[some json], schema=[define schema])
```

让我们看一个更具体的例子。

```
**from** jsonschema **import** validate**# Define Schema**
schema = {
"type" : "object",
"properties" :{
"name" : {"type": "string"}
"height" : {"type" : "number"},
"gender" : {"type" : "string"}, 
"ethnicity": {"type" : "string"},
"age": {"type": "number"},
}
```

如上定义模式后，您可以在某个测试实例上运行 validate 函数。

```
validate(instance={"name" : "Josh", "gender" : "Male"}, schema=schema)
```

如果实例与模式定义一致，那么 validate 函数将运行，不会出现任何问题或错误消息。

对以下实例运行 validate 确实会返回一条错误消息，因为 age 字段被定义为包含一个数值，但该实例却包含一个字符串。

```
validate(
instance={"name" : "Josh", "gender" : "Male", "ethnicity": "Asian", "age": "idk"}, schema=schema,
)>>> 
Traceback (most recent call last):
 …
ValidationError: ‘idk’ is not of type ‘number’
```

# 关于作者

*数据科学家。在密歇根大学刑事司法行政记录系统(CJARS)经济学实验室担任副研究员。Spotify 前数据科学实习生。Inc .(纽约市)。即将入学的信息学博士生。他喜欢运动，健身，烹饪美味的亚洲食物，看 kdramas 和制作/表演音乐，最重要的是崇拜耶稣基督。结账他的* [*网站*](http://seungjun-data-science.github.io) *！*

# 参考

[1] Python-Assert-Statements，真正的 Python，[https://realpython.com/python-assert-statement/](https://realpython.com/python-assert-statement/)

[2] K. Tran，面向数据科学的数据科学家 Pytest，[https://towardsdatascience . com/Pytest-for-Data-Scientists-2990319 e55e 6](/pytest-for-data-scientists-2990319e55e6)

[3] S. Biham，How to Validate Your JSON Using JSON Schema，Towards Data Science，[https://Towards Data Science . com/How-to-Validate-Your-JSON-Using-JSON-Schema-f55 F4 b 162 DCE](/how-to-validate-your-json-using-json-schema-f55f4b162dce)