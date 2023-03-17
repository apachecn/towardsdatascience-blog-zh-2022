# 用围棋对两个和进行编码

> 原文：<https://towardsdatascience.com/leetcode-two-sum-with-go-67d24e5a53f3>

## Python 用户 Golang 入门

![](img/699737a63b6bdbd87a649da177667020.png)

卡拉·埃尔南德斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

我喜欢用 LeetCode 来入门新语言。这是在复习数据结构和算法的同时熟悉新语法的简单方法。我主要是一个 Python 用户，我想强调一些您需要理解的新概念，以帮助您将 Python 知识转化为 Go。

本教程不会详细介绍 go 中的每个特性或数据结构，因为它旨在为您提供解决 LeetCode Two Sum 问题的基本工具。

## 问题是

给定一个整数数组和一个目标，返回两个元素的索引，两者之和等于目标。每个数字只能使用一次，返回的索引顺序无关紧要。你可以假设只有一个解决方案。

示例:

```
arr: [1,2,3,4,5]
target: 3
answer: [0,1] or [1,0]
```

下面是 LeetCode 上问题的链接:

[](https://leetcode.com/problems/two-sum) [## 两个总和代码

### 给定一个整数 num 数组和一个整数 target，返回这两个数字的索引，使它们加起来等于…

leetcode.com](https://leetcode.com/problems/two-sum) 

## 该算法

这个问题的最佳解决方案是使用哈希表将数组的每个元素映射到它的索引。对于数组中的每个元素，您将从目标中减去它以获得补数。然后您将检查补码是否是哈希表中的一个键。如果它在哈希表中，您可以将解作为包含当前索引和补码索引的数组返回。否则，向哈希表添加一个新元素，其中键是当前数字，值是它的索引。循环继续，直到找到解决方案。

以下是一个 Python 解决方案:

```
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        complements = {}
        for idx, num in enumerate(nums):
            complement = target - num
            if complement in complements:
                return [complements.get(complement), idx]
            complements.update({num: idx}) 
```

## 开始熟悉围棋

使用 Python 算法作为模板，这里有一个快速检查表，列出了在 Go 中求解两个和需要学习的主要概念:

1.  变量
2.  部分
3.  条件式
4.  地图
5.  环
6.  功能

## 变量

Go 是一种静态类型的语言，在处理变量的时候是有区别的。

**声明和初始化:**

```
package main

import "fmt"

func main() {
    var a int
    fmt.Println(a)
}
```

在上面的代码中，第 6 行声明了一个整数变量`a`。由于代码没有显式初始化`a`，它被初始化为`0`。如果要在声明时初始化变量，可以这样做:

```
package main

import "fmt"

func main() {
    var a int = 12
    fmt.Println(a)
}
```

Go 也可以暗示变量类型:

```
package main

import (
    "fmt"
    "reflect"
)

func main() {
    var a = 12
    fmt.Println(reflect.TypeOf(a))
}
```

上面的代码片段打印了`int`，即使在声明中没有明确定义`a`的类型。

**简短的赋值语句**

Go 还有一个简短的赋值语句，去掉了`var`关键字，并隐含了 type。这里有一个例子:

```
package main

import "fmt"

func main() {
    a := 12
    fmt.Println(a)
}
```

与前面的例子不同，短赋值语句只能在函数内部使用。

## 部分

一个`slice`是一个`array`的特定元素的视图。与数组不同，它们是动态调整大小的。在 Go 中，你可能会在 Python 中使用`list`的地方使用`slice`。

这里有一个快速入门`slices`的例子:

```
package main

import "fmt"

func main() {
    a := 12
    fmt.Println(a)
}
```

第 6 行显示了如何声明一个`slice`。第 7 行向`slice`追加一个整数，该整数在第 9 行打印出来。这类似于下面的 Python 片段:

```
a = []
a.append(1)
print(a)
```

第 10 行展示了如何用一个`slice`文本初始化一个变量。第 12 行显示了如何在特定索引处更改`slice`中的值，这与您在 Python `list`中使用的语法相同。虽然本教程不涉及`array`数据结构，但是您应该知道，更改`slice`的值将会修改底层的`array`。

## 条件式

Python 用户应该对 Go 中的条件语句非常熟悉。Go 使用花括号而不是空格来分隔块，并且去掉了冒号。

**Python:**

```
def main():
    a = 12

    if a > 20:
        print('a is greater than 20')
    else:
        print('a is not greater than 20')
```

**去:**

```
package main

import "fmt"

func main() {
    a := 12

    if a > 20 {
        fmt.Println("a is greater than 20")
    } else {
        fmt.Println("a is not greater than 20")
    }
}
```

Go 还允许缩短条件语句:

```
package main

import "fmt"

func main() {
    a := 12

    if b := a + 2; b > 20 {
        fmt.Println("a + 2 is greater than 20")
    } else {
 fmt.Println("a + 2 is not greater than 20")
    }
}
```

在上面的代码片段中，`a + 2`在`if`语句中被赋值给`b`。您可以通过下面的代码片段在 Python 中实现类似的功能:

```
def main():
    a = 12

    if (b := a + 2) > 20:
        print('a + 2 is greater than 20')
    else:
        print('a + 2 is not greater than 20')
```

Go 和 Python 代码片段之间的主要区别在于作用域，下面的代码片段可以更好地解释这一点:

**Python:**

```
def main():
    a = 12
    b = 'hello'

    if (b := a + 2) > 20:
        print('a + 2 is greater than 20')
        print(f'b = {b}')
    else:
        print('a + 2 is not greater than 20')
        print(f'b = {b}')

    print(f'b = {b}')
```

Python 函数打印以下内容:

```
a + 2 is not greater than 20
b = 14
b = 14
```

变量`b`在第 3 行被初始化为“hello ”,然后在第 5 行的条件中被覆盖。也就是说，条件语句在函数范围内修改`b`，因此`b`在第 12 行打印为 14 而不是“hello”。

**去:**

```
package main

import (
    "fmt"
    "io"
    "os"
)

func main() {
    a := 12
    b := "hello"

    if b := a + 2; b > 20 {
      fmt.Println("a + 2 is greater than 20")
      out := fmt.Sprintf("b = %d \n", b)
      io.WriteString(os.Stdout, out)
    } else {
      fmt.Println("a + 2 is not greater than 20")
      out := fmt.Sprintf("b = %d \n", b)
      io.WriteString(os.Stdout, out)
    }

    out := fmt.Sprintf("b = %s", b)
    io.WriteString(os.Stdout, out)
}
```

Go 函数返回以下内容:

```
a + 2 is not greater than 20
b = 14 
b = hello
```

在 Go 中，条件语句中初始化的变量是条件块的局部变量。在上面的代码中，变量`b`在条件结束后超出范围，因此对`b`的第二次调用在外部范围中访问它的值。

## 地图

在 Go 中，你可以在 Python 中使用`dict`的地方使用`map`。要实现上述算法，您需要能够执行以下操作:

1.  声明/初始化一个`map`
2.  添加键值对
3.  检查特定的键是否存在

下面是一个入门示例:

```
package main

import "fmt"

var m map[string]int

func main() {
    m = make(map[string]int)
    m["a"] = 2
    fmt.Println(m)
}
```

第 3 行声明了一个变量`m`，它是一个`nil` `map`，其中键的类型是`string`，值的类型是`int`。此时，`map`没有键值数据，不能添加任何东西。

在第 6 行中，调用了`make`函数，该函数初始化一个`map`并将它分配给`m`。第 7 行向`m`添加了一个新元素，其中键是“a”，值是 2，这也是更新 a `dict`的有效 Python 语法。

这是练习短变量赋值的好地方，如下所示:

```
package main

import "fmt"

func main() {
    m := make(map[string]int)
    m["a"] = 2
    fmt.Println(m)
}This example deletes the variable declaration step, by using the := operator. This helps to make the code less verbose.
```

要实现两个和算法，您需要检查给定的补码是否出现在`map`中。下面是一个在围棋中如何做到这一点的例子:

```
package main

import "fmt"

func main() {
    m := map[string]int{"a": 1, "b": 2, "c": 3}
    if val, ok := m["a"]; ok {
        fmt.Println(val)
    } else {
        fmt.Println("not in m")
    }
}
```

上面的代码片段创建了一个`map`文本，并检查`a`是否是`map`中的一个键。如果键在`map`中，`val`将被设置为该值，`ok`将被设置为`true`。

## 环

Go 中的 For 循环接近 C 风格的 for 循环:

```
package main

import "fmt"

func main() {
    for i := 0; i < 10; i++ {
        fmt.Println(i)
    }
}
```

如果您需要遍历一个数组并访问索引和值，您可以在 Python 中尝试这样做:

```
def main():
    nums = [10, 22, 33, 44, 52]

    for i, v in enumerate(nums):
        print(i)
        print(v)
```

以下是您在 Go 中的操作方法:

```
package main

import "fmt"

func main() {
    nums := []int{10, 22, 33, 44, 52}
    for i, v := range nums {
        fmt.Println(i)
        fmt.Println(v)
    }
}
```

当你在 Go 中使用带有`range`的 for 循环时，它完成的事情类似于在 Python 中使用`enumerate`。迭代变量为`i`和`v`，其中`i`为索引，`v`为`nums`在索引`i`处的值。

与 Go 中的条件一样，for 循环中初始化的变量是循环块的局部变量。因此，如果你试图在 Go 循环后打印`i`或`v`，你会得到一个错误。在 Python 中，在循环中初始化的变量仍然存在于外部作用域中。

## 功能

如果你习惯用 Python 中的类型提示来定义函数，那么 Go 中的函数会感觉非常相似。下面的片段大概是 LeetCode 为你提供的开始使用 Two Sum 的内容。这两个函数都有两个参数，`nums`和`target`，它们返回一个整数数组。

**Python:**

```
def twoSum(nums: List[int], target: int) -> List[int]:
```

如果您不习惯在 Python 中使用类型提示，它们只是定义了参数和返回值应该是什么样子。在上面的函数定义中，`nums`应该是一个`list`，其中每个元素都是一个`int`，而`target`是一个`int`。返回值被指定为`->`右边的部分，是一个`List`，其中每个元素都是一个`int`。

**去:**

```
func twoSum(nums []int, target int) []int {

}
```

Go 功能的设置方式相同。`nums`参数和返回值都是包含`int`类型元素的`slice`，这就是`[]int`的含义。

与 Python 不同，为参数和返回值指定类型不是可选的。同样与 Python 不同的是，如果用户传递参数或生成不符合函数定义的返回值，将会出现错误。

## 把它们放在一起解两个和

下面是 Go 中两个和算法的实现:

```
func twoSum(nums []int, target int) []int {
    m := make(map[int]int)
    var ans []int

    for idx, num := range nums {
        complement := target - num

        if c, ok := m[complement]; ok {
            ans = []int{c, idx}
            break
        }
        m[num] = idx
    }
    return ans     
}
```

第 2 行用类型`int`的键和值初始化`map`。第 3 行声明了`ans`，它是一个`int`类型的`slice`。

循环从第 5 行开始，使用`range`以便索引和值都可用。循环的第一步计算`complement`，然后检查它是否是`m`中的一个键。如果`ok`为真，`ans`被设置为等于包含`nums`和`idx`中的`complement`的索引的`slice`文字。然后循环中断，函数返回`ans`。如果`ok`为假，则向`m`添加新元素，其中键为`num`，值为`idx`。

因为`ans`的值是在条件语句中确定的，并且需要被返回，所以`ans`在函数范围中被声明。如果您想避免这种情况，可以尝试以下实现方式:

```
func twoSum(nums []int, target int) []int {
    m := make(map[int]int)

    for idx, num := range nums {
        complement := target - num

        if c, ok := m[complement]; ok {
            return []int{c, idx}
        }
        m[num] = idx
    }
    return []int{}    
}
```

不用把要返回的值保存在变量中，可以直接返回`slice`。为了使用 LeetCode 提供的结构来实现这一点，您需要在底部返回一个空的`slice`。第 12 行只在无解的情况下执行，这超出了这个问题的范围。

## 一般提示

1.  在 Python 中我经常用单引号表示`str`值，但是 Go 需要双引号。
2.  Go 对空值使用`nil`，这类似于 Python 中的`None`。
3.  Go 有一个很棒的基于网络的 IDE，你可以在这里找到:[https://go.dev/play/](https://go.dev/play/)
4.  这可能是最好的入门教程:[https://go.dev/tour/list](https://go.dev/tour/list)

## 结论

你刚刚学习了如何用围棋解两个和。如果您是一名使用 Python 的用户，那么您应该已经注意到了这两种语言之间的一些关键差异。这只是皮毛，但我希望它对你有所帮助。