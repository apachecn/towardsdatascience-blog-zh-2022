# 如何在 Python 中测试函数是否抛出异常

> 原文：<https://towardsdatascience.com/python-exception-test-290366618e7d>

## 编写测试用例来验证函数是否会用预期的消息引发异常

![](img/29abbd481d2d77460cf173b23acf4075.png)

杰里米·珀金斯在 [Unsplash](https://unsplash.com/s/photos/error?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

软件工程中最基本的方面之一是错误处理。软件应用程序可能由于许多不同的原因而中断，因此能够以一种能够有效处理和报告异常的方式编写代码是非常重要的。

然而，在源代码中处理异常是不够的。测试您的代码是否按预期处理和报告异常更加重要。测试可确保在适当的时候引发异常，并给出正确的消息。

在今天的文章中，我们将演示如何编写测试用例来测试某个函数在某个事件发生时是否会引发预期的异常。更具体地说，我们将使用 Python 中最流行的两个测试包，即`unittest`和`pytest`，展示如何做到这一点。

首先，让我们创建一个在特定事件发生时引发异常的示例函数。然后，我们将通过一步一步的指导，最终帮助您测试异常处理。

```
def greet(greeting):
    if greeting not in ('hello', 'hi'):
        raise ValueError(f'{greeting} is not allowed')

    print(greeting + ' world!')
```

如果输入的`greeting`参数不等于`hello`或`hi`，我们的`greet`函数将引发一个`ValueError`。现在让我们在实践中看看如何正确地测试函数使用`unittest`和`pytest`引发了预期的异常。

## 用 unittest 测试异常处理

为了创建一个测试用例来测试当输入参数不包含预期值时`greet()`是否正在引发`ValueError`，我们可以利用下面概述的`assertRaises`方法:

```
import unittestclass MyTestCase(unittest.TestCase): def test_greet_raises(self):
        self.assertRaises(ValueError, greet, 'bye')if __name__ == '__main__':
    unittest.main()
```

或者，您甚至可以使用上下文管理器，如下例所示:

```
import unittestclass MyTestCase(unittest.TestCase):def test_greet_raises(self):
        with self.assertRaises(ValueError) as context:
            greet('bye')if __name__ == '__main__':
    unittest.main()
```

我们甚至可以使用下面的方法测试异常所报告的实际消息:

```
import unittestclass MyTestCase(unittest.TestCase):def test_greet_raises(self):
    with self.assertRaises(ValueError) as context :
        greet('bye')
    self.assertEqual('bye is not allowed', str(context.exception))if __name__ == '__main__':
    unittest.main()
```

## 用 pytest 测试异常处理

同样，我们可以使用`pytest`来断言一个函数引发了一个异常。

```
import pytestdef test_greet():
    with pytest.raises(ValueError):
        greet('bye')
```

同样，我们也可以测试我们预期会引发的异常所报告的错误消息:

```
import pytestdef test_greet():
    with pytest.raises(ValueError, match='bye is not allowed'):
        greet('bye')
```

## 最后的想法

测试覆盖率是最重要的指标之一，您可以用它来判断源代码测试的好坏。除了良好的设计和高效的代码，确保代码的每个方面都按预期运行也很重要，这样你就可以将错误减到最少。这只能通过彻底的测试来实现。

在今天的简短教程中，我们演示了如何测试引发异常的函数。除了测试流程的“良好路径”，确保在源代码中正确处理错误也很重要。

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership) [## 通过我的推荐链接加入 Medium-Giorgos Myrianthous

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

gmyrianthous.medium.com](https://gmyrianthous.medium.com/membership) 

**相关文章你可能也喜欢**

[](/diagrams-as-code-python-d9cbaa959ed5) [## Python 中作为代码的图

### 用 Python 创建云系统架构图

towardsdatascience.com](/diagrams-as-code-python-d9cbaa959ed5) [](/big-o-notation-32fb458e5260) [## 大 O 符号

### 用 Big-O 符号计算算法的时间和空间复杂度

towardsdatascience.com](/big-o-notation-32fb458e5260) [](/how-to-merge-pandas-dataframes-221e49c41bec) [## 如何合并熊猫数据帧

### 对熊猫数据帧执行左、右、内和反连接

towardsdatascience.com](/how-to-merge-pandas-dataframes-221e49c41bec)