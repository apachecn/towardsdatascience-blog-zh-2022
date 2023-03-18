# 数据科学家的递归

> 原文：<https://towardsdatascience.com/recursion-for-data-scientists-b08b4f96e450>

## 要理解递归，首先你必须理解递归

![](img/1c7214f88c41d7ad7b078f7c9bed6798.png)

信用:Pexels.com

对于数据科学家来说，绕过计算机科学基础和软件工程最佳实践以追求工具精通(Pandas & SKLearn 举两个例子)并立即开始为您的组织增加价值的诱惑是真实的。

CS 中的许多基本概念真的值得争论，因为它们将使您在冗长地争论笨拙的数据时构建简洁的、可伸缩的逻辑。在这篇文章中，我们将定义一个递归函数，它可以在 python 字典中搜索目标值。事不宜迟，我们开始吧！

# 什么是递归？

递归通俗的说就是自引用函数。简单来说，函数可以自己调用！但是这样不会导致无限循环吗？诀窍是通过使用控制流(if/elif/else 逻辑)来定义终止功能的基本情况。)

## 伪代码

```
def recursive_function(argument):
   if base case:
      return terminal_action(argument)
   else:
      new_argument = intermediate_action(argument)
      recursive_function(new_argument)
```

关于上述内容的几点说明:

1.  我们需要指定一个“岔路口”要么基本情况得到满足，我们退出递归逻辑，要么我们递归地使用函数*到达*基本情况。
2.  基本情况下的终端功能是完全可选的。您可能只想按原样返回参数，在这种情况下，您的终端函数将归结为`lambda x: x`。
3.  为了不陷入无限循环，我们不能一遍又一遍地把同一个参数传递给递归函数；我们需要使用一个*中间动作*来修改参数。与终端动作不同，中间动作不能是被动的(归结为`lambda x: x`)。)
4.  可以根据需要添加副作用，或者对递归逻辑或基本情况不重要的动作。

## 说明性示例

这个例子，blastoff，是递归函数的 helloWorld。

```
def blastoff(x):
   # base case 
   if x == 0:
      print("blastoff!") # side effect
      return # terminal action: None/passive # recursive logic
   else:
      print(x) #side effect
      blastoff(x-1) # intermediate action: decrementing x by 1blastoff(5)>>>5
4
3
2
1
blastoff!
```

# 真实世界的例子

现在，让我们来处理一个真实世界的例子——在嵌套字典中搜索给定值。

为什么是嵌套字典？因为字典是类似于 JSON 的 python 原生目标。事实上，Python 很容易支持将 JSON 对象转换成字典。此外，API 响应通常是用 JSON 交付的(尽管有些仍然使用 XML，但是这种方式越来越不受欢迎了)。)

所以下一次你需要发现一个给定值是否在你的 API 请求结果中，你可以简单地(1)将结果从 JSON 转换成 Python 字典，然后(2)使用下面的递归函数！

```
def obj_dfs(obj, target, results=None):
   if not results:
      results = [] # recursive logic
   if isinstance(obj, dict):
      for key, val in obj.items():
         results.append(obj_dfs(key, target, results))
         results.append(obj_dfs(val, target, results)) elif isinstance(obj, (list, tuple)):
      for elem in obj:
         results.append(obj_dfs(elem, target, results)) # base case
   else:
      if obj == target:
         return True
      else:
         return False return any(results)
```

让我们把它分成几部分！

## 基础案例

基本情况是一个不是字典、列表或元组的对象。换句话说，*如果对象不能包含嵌套字典、元组或列表*，则基本情况有效。这些是我们实际搜索的值——字符串、整数、浮点数等。

当我们检测到一个基本案例时，我们简单地评估这个问题，“*我们找到目标了吗？*“返回真或假

## 递归逻辑

如果基本情况无效，使得对象可以包含嵌套字典、列表或元组，*我们递归地调用那些非常嵌套的字典、列表或元组上的函数*。

我使用了一个 **if** **块**和一个 **elif 块**以一种方式处理字典，以另一种方式处理列表和元组。这是必要的，因为字典由(键，值)元组组成，而列表和元组只包含对象。

“isinstance”是一个内置的 Python 函数，它确定对象的类型是等于 x 还是 in (x，y)。

## 结果

我们首先检查结果对象是否存在——如果它不存在，这是默认行为，我们定义一个空列表。在随后的递归调用中，相同的结果对象被更新。

将递归函数追加到列表*中似乎不常见，但是函数* ***本身*** *永远不会追加到列表*中。该函数遵循递归逻辑，仅在基本情况下返回 True 或 False。因此，结果只会被对象填充，每个对象要么等于真，要么等于假。

最后，该函数返回“any(results)”，在 Python 中，如果一个或多个嵌套元素为真，则返回真。因此，如果在至少一个检测到的基本案例中找到目标对象，则该函数全局返回 True。

## 扩展空间

假设您对精确匹配不感兴趣，而是对模糊匹配感兴趣——或者您有自己的特定于上下文的逻辑。这很好，我们可以将一个函数作为参数传递给我们的递归函数，这将在基本情况下使用。

```
obj = {'chicago': 
          [{'coffee shops':'hero'}, {'bars':'la vaca'}],
       'san francisco': 
          [{'restaurants':'el techo'}, 
          {'sight seeing':'golden gate bridge'}]}def base(x):
   return 'golden' in xdef obj_dfs(obj, f_base, results=None):
   if not results:
      results = [] # recursive logic
   if isinstance(obj, dict):
      for key, val in obj.items():
         results.append(obj_dfs(key, f_base, results))
         results.append(obj_dfs(val, f_base, results)) elif isinstance(obj, (list, tuple)):
      for elem in obj:
         results.append(obj_dfs(elem, f_base, results)) # base case
   else:
      return f_base(obj) return any(results)
```

# 解析 JSON 参数

最后，要在实际的 JSON 格式的 API 响应上使用这个函数，使用下面的代码将 API 响应转换成 dictionary。

```
import json
with open('data.json') as json_file:
   data **=** json.load(json_file)
```

您可以参考这个免费的模板 API 获得 JSON 响应来测试这段代码！[https://fakestoreapi.com/](https://fakestoreapi.com/)

```
import requests
import json
data = requests.get('https://fakestoreapi.com/products/1')
response = json.loads(data.text)def obj_dfs(obj, target, f_base, results=None):
   if not results:
      results = [] # recursive logic
   if isinstance(obj, dict):
      for key, val in obj.items():
         results.append(obj_dfs(key, target, f_base, results))
         results.append(obj_dfs(val, target, f_base, results))

   elif isinstance(obj, (list, tuple)):
      for elem in obj:
         results.append(obj_dfs(elem, target, f_base, results)) # base case
   else:
      return base_case_function(obj) if any(results):
      return list(filter(lambda x: x!=False, results))[0]
   else:
      return Falsedef find_url(x):
   x = str(x)
    if 'http' in x:
      return x
   else:
      return Falseobj_dfs(obj=response, target=None, f_base=find_url)
>>> 'https://fakestoreapi.com/img/81fPKd-2AYL._AC_SL1500_.jpg'
```

我对代码做了一点小小的改动；具体来说，我没有返回`any(results)`,而是返回了结果中的第一个非 False 元素，这是在 JSON API 响应中找到的图片链接 url。

我希望你能感受到递归的强大，这是你腰带上的新工具。您可以根据需要修改这段代码！

**我希望这有所帮助——如果有，请订阅我的博客！**