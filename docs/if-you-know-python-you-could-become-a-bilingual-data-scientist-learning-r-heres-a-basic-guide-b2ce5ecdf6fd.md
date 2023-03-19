# 如果你懂 Python，你可以成为一名学习 r 的双语数据科学家。

> 原文：<https://towardsdatascience.com/if-you-know-python-you-could-become-a-bilingual-data-scientist-learning-r-heres-a-basic-guide-b2ce5ecdf6fd>

## 用代码解释数据科学的核心编程概念

![](img/161509fc1908ee6fb9e76cfa76007964.png)

图片来自 Shutterstock，授权给 Frank Andrade

如果你对学习数据科学感兴趣，你应该学习 R。R 是一种擅长统计分析和可视化的编程语言。

但这还不是全部！使用 R，我们可以进行数据分析，应用机器学习算法，并完成许多其他数据科学任务，这些任务我们会使用 Python 等其他数据科学编程语言来完成。

与其在 Python 和 R 之间选择数据科学，为什么不两全其美，成为双语数据科学家呢？到最后，像 R 这样的新编程语言会为你打开不同的工作机会(即使你已经懂 Python)。

在本指南中，我们将学习每个数据科学家都应该知道的一些 R 核心编程概念。您可以将本指南视为您学习 R for data science 的第一步！

*注意:在本文的最后，你会发现一个 PDF 版本的 R 备忘单(下面目录中的第 8 节)*

```
**Table of Contents** 1\. [R Variables](#195c)
2\. [R Data Types](#1a9a)
3\. [R Vectors](#3424)
4\. [R If Statement](#7968)
5\. [R While Loop](#37ce)
6\. [R For Loop](#d795)
7\. [R Functions](#475c)
8\. [R for Data Science Cheat Sheet](#4095) (Free PDF)
```

# r 变量

在 R 中，我们使用变量来存储数值，比如数字和字符。为了给变量赋值，我们在 r 中使用了`<-`。

让我们创建第一条消息“我喜欢 R ”,并将其存储在一个名为`message1`的变量中。

```
message1 <- "I like"
```

现在我们可以在 R 中创建任意多的变量，甚至可以对它们应用函数。例如，我们可以使用`paste`函数连接 R 中的两条消息。

让我们创建一个值为“R programming”的变量`message2`,并使用`paste`函数将其连接到`message1`。

```
> message1 <- "I like"
> message2 <- "R programming"

> **paste**(message1, message2)
[1] "I like R programming"
```

我们甚至可以将输出分配给一个新变量`message3`。

```
message3 <- **paste**(message1, message2)
```

# r 数据类型

在 R 中，我们使用变量来存储不同类型的值，比如数字、字符等等。R 中的每个变量都有一个数据类型。以下是 R 编程语言中最常见的数据类型。

1.  整数:没有小数点的实数值。后缀`L`用于指定整数数据。
2.  数字:所有实数的集合
3.  复数:r 中的纯虚数。后缀`i`用于指定虚部。
4.  Character:它用于指定变量中的字符或字符串值(一系列字符)。单引号`''`或双引号`""`用来表示字符串。
5.  逻辑:也称为布尔数据类型，可以有`TRUE`或`FALSE`值。

下面是 r 中每种数据类型的表示。

```
x <- 1L *# Integer*
y <- 1.5 *# Numeric*
z <- 2i *# Complex*
message <- "Hi!" *# character*
is_cold <- TRUE *# logical*
```

为了检查变量的数据类型，我们在 r 中使用了`class`。

```
> message <- "Hi!"
> **class**(message)
[1] "character"
```

我们可以看到，`message`的数据类型是字符。

# r 向量

在 R 中，向量是共享相同数据类型的元素序列。实际上，您可以在一个 vector 中包含不同数据类型的元素，但最终，它们将被转换为相同的数据类型。

为了创建一个矢量，我们使用了`c()`函数。让我们创建一个名为 countries 的向量，并获取它的数据类型。

```
> countries <- c(‘United States’, ‘India’, ‘China’, ‘Brazil’)
> class(countries)
[1] "character"
```

我们刚刚创建了一个字符向量，因为向量的所有元素都是字符串。

## 向量名称

我们还可以给当前的`countries` 向量的每个元素添加向量名。让我们创建一个新的`population`向量，然后添加`countries`作为它的向量名。

```
> population <- c(329.5, 1393.4, 1451.5, 212.6)
> names(population) <- countries> population
United States         India         China        Brazil 
        329.5        1393.4        1451.5         212.6
```

## 向量索引

我们可以通过索引获得向量中的特定元素。向量中的每一项都有一个索引(也称为位置)。与其他编程语言不同，R 中的索引从 1 开始。

在 R 中，通过索引访问一个元素，我们使用方括号`[]`。让我们看几个使用我们之前创建的`countries`向量的例子。

```
> countries[1]
[1] "United States"> countries[4]
[1] "Brazil"
```

我们还可以使用方括号中的`c()`函数索引多个元素，甚至可以使用向量名来代替索引位置。

```
> countries[c(1,4)]
[1] "United States" "Brazil"> population[c("United States", "China")]
United States         China 
        329.5        1451.5
```

## 矢量切片

向量切片意味着访问向量的一部分。切片是元素的子集。符号如下:

```
vector[start:stop]
```

其中“start”表示第一个元素的索引，stop 表示要停止的元素(包括在切片中)。

让我们看一些例子:

```
> population[1:3]
United States         India         China 
        329.5        1393.4        1451.5
```

太好了！我们已经获得了索引 1 和 3 之间的元素。

## 过滤

我们可以通过比较过滤掉向量中的一些元素。使用的语法类似于索引，但是我们没有在方括号`[]`中键入索引，而是使用逻辑运算符进行比较。

让我们过滤掉人口少于 3 亿的国家。

```
> population[population>300]
United States         India         China 
        329.5        1393.4        1451.5
```

正如我们所看到的，拥有 2.12 亿人口的国家巴西被从我们的向量中过滤掉了。

# R 中的 If 语句

就像在任何其他编程语言中一样，`if`语句在 r 中非常常见。我们用它来决定一个(或多个)语句是否将被执行。

下面是 R 中`if`语句的语法:

```
**if** (condition1) {
    statement1
} **else if** (condition2) {
    statement2
} **else** {
   statement3
}
```

让我们通过一个例子来看看这是如何工作的。下面的代码将基于`height`变量输出一条消息。

```
height <- 1.9**if** (height > 1.8){
  **print**("You're tall")
} **else** **if**(height > 1.7){
  **print**("You're average")
} **else** {
  **print**("You're short")
}
```

代码的意思是“如果你的身高超过 1.8 米，你就是高的；如果身高在 1.7 到 1.8 之间，你就是一般；如果身高在 1.7 或以下，你就是矮的”

# 在 R 中循环时

只要满足指定的条件，循环就允许我们重复特定的代码块。

下面是 R 中`while`循环的语法:

```
**while** (condition) {
   <code>
}
```

让我们通过下面的例子来看看它是如何工作的。

```
x <- 5**while**(x<10){
  **print**(x)
  x <- x+1
}
```

如果我们运行上面的代码，输出将如下:

```
[1] 5
[1] 6
[1] 7
[1] 8
[1] 9
```

我们得到的值一直到数字 9，因为在那之后，条件 x <10 was unsatisfied, so the loop broke.

We can also use the 【 keyword to break out of a loop. To do so, we usually combine it with the 【 statement. Let’s see an example.

```
x <- 5**while**(x<10){
  **print**(x)
  x <- x+1
  **if** (x==8){
    **print**("x is equal to 8\. Break loop!")
    **break**
  }
}
```

The output will be the following:

```
[1] 5
[1] 6
[1] 7
[1] "x is equal to 8\. Break loop!"
```

In this case, the 【 loop broke when x reached the value of 8 because of the new condition we created.

# For Loop in R

One of the most common loops in any programming language is the 【 loop. The for loop allows us to iterate over items of a sequence (like our vectors in R) and perform an action on each item.

Here’s the syntax of the 【 loop in R:

```
**for** (val in sequence)
{
    <code>
}
```

Let’s loop through the 【 vector that we created before and print each item.

```
> countries <- **c**('United States', 'India', 'China', 'Brazil')> **for** (i **in** countries){
+   **print**(i)
+ }[1] "United States"
[1] "India"
[1] "China"
[1] "Brazil"
```

We can also use the for loop and if statement together to perform an action to only certain elements.

As an example, let’s loop through the 【 vector but now only print the element “India”

```
> **for** (i **in** countries){
+   **if** (i=="India"){
+     **print**(i)    
+   }
+ }[1] "India"
```

# R Functions

R has different built-in functions that help us perform a specific action. So far we’ve seen the 【 and 【 function, but there are many others. To name a few:

```
**sum**() -> returns sum
**rnorm**() -> draw random samples from a distribution
**sqrt**() -> returns the square root of an input
**tolower**() -> converts the string into lower case
```

But that’s not all! We can also create our own function in R. We have to follow the syntax below.

```
function_name <- **function**(<params>){
    <code>
    **return**(<output>)
}
```

Let’s create a function that converts the temperature from Fahrenheit to Celsius.

```
fahrenheit_to_celsius <- **function**(temp_f){
    temp_c <- (temp_f - 32) * 5 / 9
    **return**(temp_c)
}
```

Now if we call the function 【 we will convert the temperature from Fahrenheit to Celsius.

```
> fahrenheit_to_celsius(100)
[1] 37.77778
```

Learning Data Science with R? [**通过加入我的 20k+人的电子邮件列表来获得我的免费 R for Data Science 备忘单。**](https://frankandrade.ck.page/7b621bc82c)

如果你喜欢阅读这样的故事，并想支持我成为一名作家，可以考虑报名成为一名媒体成员。每月 5 美元，让您可以无限制地访问数以千计的 Python 指南和数据科学文章。如果你用[我的链接](https://frank-andrade.medium.com/membership)注册，我会赚一小笔佣金，不需要你额外付费。

<https://frank-andrade.medium.com/membership> 