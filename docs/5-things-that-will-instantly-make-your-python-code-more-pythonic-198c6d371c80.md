# 让你的 Python 代码更加 Python 化的 5 件事

> 原文：<https://towardsdatascience.com/5-things-that-will-instantly-make-your-python-code-more-pythonic-198c6d371c80>

## 理解语言的规则

![](img/90f2fe6c1d97f625f151164117efee04.png)

照片由[亚历克斯·丘马克](https://unsplash.com/@ralexnder?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

任何会说多种语言的人都明白，一种语言的规则不一定适用于另一种语言；我的母语是英语，但我也会说 Twi 语，这是加纳阿坎部落的一种方言。每当我在加纳的时候，我都强迫自己说 Twi 语——我犯了很多错误，但是当地人很友好，他们帮我解决了问题。

我犯的错误不一定会破坏我所说的意思，这通常只是对 Twi 语言的糟糕练习，他们在说英语时经常犯同样的错误。例如，加纳人用 Twi 语交流时，通常会在开始说“请”,所以他们在说英语时也会这样做。

虽然“请是”不一定是英语语法的坏用法，但你可能不会说它。但这个错误源于试图将英语的规则应用于 Twi。许多用各种语言编写代码的程序员也陷入了这个陷阱。

您的代码可能不会抛出错误，但是当其他程序员试图理解您所写的内容时，您会让他们的日子更难过。花时间去理解一门语言的最佳实践不仅对你自己有益，对那些必须与你合作的人也有益。因此，我们将讨论编写 Pythonic 代码的五种标准方法。

# #1 枚举()而不是范围()

您将看到的最常见的 Python 误用之一是程序员使用`range()`和`len()`函数循环遍历一个列表[或序列]并生成索引。

```
names = ["john", "doe", "jane", "plane"]
**for** idx **in** range(**len**(names)): 
    **print**(idx, names[idx])0 john
1 doe
2 jane
3 plane
```

您会发现上面的代码是有效的，并且运行良好，那么为什么这是一个问题呢？这违背了 Python 成为如此成功的编程语言的一个原因:**可读性**。

使用`range(len(names))`约定很容易做到，但是不太理想，因为它牺牲了可读性。执行相同功能的更好方法是将列表传递给内置的`enumerate()`函数，该函数将返回序列中索引和元素的整数。

```
names = ["john", "doe", "jane", "plane"]
**for** idx, name **in enumerate**(names): 
    **print**(idx, name)0 john
1 doe
2 jane
3 plane
```

如果您不需要索引，那么您仍然可以按如下方式遍历元素:

```
names = ["john", "doe", "jane", "plane"]
**for** name **in** names: 
    **print**(name)john 
doe
jane
plane
```

# #2 使用“with”语句

你可能正在编写一个要求你读写文件的程序。`open()`和`close()`内置函数允许开发者分别打开和关闭文件。

```
requirements = open("requirements.txt", "w")
requirements.write(
  "scikit-learn >= 0.24.2, < 0.25.0", 
  "numpy >= 1.21.2, < 1.22.0",
  "pandas >= 1.3.3, < 1.4.0"
)
requirements.close() 
```

在上面的代码中，我们打开了一个名为`requirements.txt`的文本文件，写了一些内容，然后在完成后关闭它。这段代码是完全有效的，但是 Python 程序员会称之为 ***不协调*** 。

这也很危险，因为很容易忘记关闭一个文件，有时这是我们无法控制的——比如当一个`try`子句发生错误，程序跳过对`except`子句的`close()`调用。

```
**try**: 
    requirements = open("requirements.txt", "w")
    requirements.write(
      "scikit-learn >= 0.24.2, < 0.25.0", 
      "numpy >= 1.21.2, < 1.22.0",
      "pandas >= 1.3.3, < 1.4.0"
    )
    random_error = 25 / 0 # raises a 0 divide exception
    requirements.close() # this is skipped 
**except**: 
   **print**("An error occurred.")
```

上面的场景是不太可能的:我无法想象你写依赖关系，然后除以一些数字的例子。但是灾难的威胁是真实存在的。

在上面的代码中，跳过了`close()`方法，因为在我们的代码中发现了一个错误，这使得它跳到了`except`块。这可能会导致非常难以跟踪的文件损坏错误。

比较好的方法是`open()`和`close()`文件如下:

```
**with** open("requirements.txt", "w") as requirements:
    requirements.write(
      "scikit-learn >= 0.24.2, < 0.25.0", 
      "numpy >= 1.21.2, < 1.22.0",
      "pandas >= 1.3.3, < 1.4.0"
    )
    requirements.close()
```

上面的代码更 Pythonic 化，也更安全，因为一旦执行离开了`with`语句块，文件总是关闭的。

# #3 将``None``值与``is``值进行比较

将`None`值与`is`恒等运算符进行比较优于等式运算符`==`。

> "应该总是用`is`或`is not`来比较像 None 这样的单元组，而不是相等运算符."
> -人教版 8

原因在于他们如何进行比较:

*   相等运算符`==`比较两个对象的值。
*   `is`标识运算符比较两个对象的标识。因此，它是对等式的引用，这意味着它决定了两个对象是否有相同的身份。

如果这听起来像是行话，简单的说就是在 Python 中有相同值的两个对象不一定是相同的。

```
**# Example of equal values and identities**
a = [1, 2, 3, 4, 5] 
b = a **id**(a) 
"""
140049469156864
"""**id**(b)
"""
140049469156864
"""a is b
"""
True
"""a == b
"""
True
"""**# Example of equal values but different identities** 
a = [1, 2, 3, 4, 5] 
b = a[:] **id**(a) 
"""
139699152256576
"""**id**(b)
"""
139699151636672
"""a is b
"""
False
"""a == b
"""
True
"""
```

当你比较一个值和`None`时，你应该总是使用`is`，因为相等运算符`==`仍然可以计算为`True`，即使对象实际上是`None`。

```
**class** Example: 
    **def** __eq__(self, other=None):
        **return** Trueexample = Example()
example == None
"""
True
"""example is None
"""
False
"""
```

这种可能性是由于`==`操作符过载造成的。执行`example is None`会检查`example`标识符中的值是否为`None`。基本上，如果一个变量被设置为`None`，那么比较它，看看它`is None`是否总是评估为`True`——因为行为是可预测的，所以它是首选的。

# 原始字符串有其用途

Python 中以`r`或`R`为前缀的字符串称为原始字符串。

```
**print**(r"This is a raw string") 
"""
This is a raw string
"""**print**(R"This is also a raw string")
"""
This is also a raw string
"""
```

原始字符串最常见的用法是当我们处理使用几个转义字符`\`的字符串时(即 windows 路径、正则表达式，或者如果我们想要将文本插入到字符串文字中，否则像`'`这样是不可能的)。

```
***# without raw strings*** **print**("This is Kurtis\' phone")
"""
This is Kurtis' phone
"""**print**("C:\\Users\\kurtis\\documents") 
"""
C:\Users\kurtis\documents
"""***# with raw strings*** **print**(r"This is Kurtis' phone")
"""
This is Kurtis' phone
"""**print**(r"C:\Users\kurtis\documents")"""
C:\Users\kurtis\documents
"""
```

原始字符串不应该被认为是不同类型的字符串数据类型——它不是。这只是键入包含几个反斜杠的字符串的一种便捷方式。

# f 字符串更适合格式化代码

Python 3.6 中添加的新特性之一是 f 字符串(格式字符串的简写)。它们提供了一种更简洁、更方便的方法来格式化字符串——也符合 Python 的可读性对象。

为了理解它的有用性，我们必须了解它的发展历程。最初`+`操作符用于连接字符串:

```
name = "John"
age = "30"
city = "London"**print**("Hi, my name is " + name + " and I'm " + age + " years old. I live in " + city )
```

尽管这种方法可行，但它包含了几个引号和`+`操作符，这会影响可读性。

Python 随后引入了转换说明符`%s`使其更加具体:

```
name = "John"
age = "30"
city = "London"**print**("Hi, my name is %s and I'm %s years old. I live in %s" % (name, age, city))
```

这也是可行的，但是对于可读性来说仍然不是最好的。

下面是我们如何用 f 弦做同样的事情:

```
name = "John"
age = "30"
city = "London"**print**(f"Hi, my name is {name} and I'm {age} years old. I live in {city}")
```

干净多了，也更像蟒蛇了。

*感谢您的阅读。*

**联系我:**
[LinkedIn](https://www.linkedin.com/in/kurtispykes/)
[Twitter](https://twitter.com/KurtisPykes)
[insta gram](https://www.instagram.com/kurtispykes/)

如果你喜欢阅读这样的故事，并希望支持我的写作，可以考虑成为一名灵媒。每月支付 5 美元，你就可以无限制地阅读媒体上的故事。如果你使用[我的注册链接](https://kurtispykes.medium.com/membership)，我会收到一小笔佣金。

已经是会员了？[订阅](https://kurtispykes.medium.com/subscribe)在我发布时得到通知。

[](https://kurtispykes.medium.com/subscribe) [## 每当 Kurtis Pykes 发表文章时都收到一封电子邮件。

### 每当 Kurtis Pykes 发表文章时都收到一封电子邮件。通过注册，您将创建一个中型帐户，如果您还没有…

kurtispykes.medium.com](https://kurtispykes.medium.com/subscribe)