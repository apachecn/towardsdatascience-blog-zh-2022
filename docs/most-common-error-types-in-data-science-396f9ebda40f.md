# 数据科学中最常见的 Python 错误类型

> 原文：<https://towardsdatascience.com/most-common-error-types-in-data-science-396f9ebda40f>

## 当您将 Python 用于数据科学时，了解这些错误类型以及如何解决它们

![](img/0c9a97c5881946522cb39c17e763727f.png)

由[布雷特·乔丹](https://unsplash.com/@brett_jordan?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/error?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

## 错误

编码出现错误。他们不是*想要的——当然是*——但是他们应该是被期待的。我这么说是因为错误是数据科学家日常工作的一部分。有时原因会是一个简单的分心。其他时候将需要更深入的搜索和研究，以找出是什么在阻止我们的程序正常运行。

因此，如果我们知道 Python 中最常见的错误类型，这给了我们寻找答案的优势。

在数据科学中，一般来说，我们编写代码来探索和转换数据，以使它们符合确定的 ML 模型，或者只是为了获得对数据集的一些见解。

通常，错误出现在对象中或由于函数使用不当而弹出:

*   当你忘记创建或声明一个变量时。
*   当您试图在错误类型的对象中执行操作时，如 sum 文本。
*   如果您试图在错误类型的对象上使用属性。

此外，当我们使用数据框架或操作数据时，以及当我们拟合模型时，也会发生错误。

在我开始举例之前，让我先说明一下我将使用的数据集:我将使用 Seaborn 库中的数据集 *Tips* 。

让我们看一些例子。

## 名称错误

`NameError` *在局部或全局范围内找不到变量时发生。*

简单地说，就是当你忘记在运行的命令中声明一个变量的时候。如果您没有运行 Python 将值赋给变量的单元格，您可能会看到该错误。

```
# Printing an object I never declared.
print(my_object)-------------------------------------------------------------NameError             Traceback (most recent call last)[<ipython-input-1-1f04422500d2>](/<ipython-input-1-1f04422500d2>) in <module>()
----> 1 print(my_object)NameError: name 'my_object' is not defined
```

另一个很好的例子:如果我们想使用来自*提示*的数据拟合一个回归模型，我们必须将我们的模型分成独立(X)和非独立(y)变量。如果您试图让 *total_bill* 放入 X，但是您忘记了引号，您将看到同样的错误。

```
# Trying to assign total_bill to X
X = df[total_bill]NameError                   Traceback (most recent call last)[<ipython-input-83-cc074c7738f3>](/<ipython-input-83-cc074c7738f3>) in <module>()
----> 1 X = df[total_bill]NameError: name 'total_bill' is not defined
```

**如何解决:**大多数时候，只要声明 Python 说没有定义的变量或者对变量名使用合适的引号就可以了。

## 类型错误

`TypeError`弹出*当一个函数或操作被应用到一个不正确类型的对象时。*

是的，比如说，当你试图对一个文本进行数学运算时。你不能将两个字母相加，除非它们被赋予了数值。

```
# trying to multiply two strings
my_sum = 'a' * 'b'---------------------------------------------------------TypeError                   Traceback (most recent call last)[<ipython-input-3-b463ecefa547>](/<ipython-input-3-b463ecefa547>) in <module>()
----> 1 my_sum = 'a' * 'b'TypeError: can't multiply sequence by non-int of type 'str'
```

如果您给出的参数多于函数所需的数量，也会发生这种情况。

```
# Predict does not need the target variable
model.predict(X,y)--------------------------------------------------------------TypeError               Traceback (most recent call last)[<ipython-input-85-ec74152c0fb0>](/<ipython-input-85-ec74152c0fb0>) in <module>()
----> 1 model.predict(X,y)TypeError: predict() takes 2 positional arguments but 3 were given
```

**如何解决:**仔细检查你用于该操作的对象类型，或者检查你是否没有给一个函数提供太多的参数。

## 属性错误

`AttributeError` —如果您试图将属性用于错误的对象类型，将会发生这种情况，Python 告诉它不能这样做。

示例:尝试对列表使用 dataframe 属性。

```
# list
a = [1,2,3]# Trying to use an attribute from Pandas objects for a list
a.shape--------------------------------------------------------------AttributeError                Traceback (most recent call last)[<ipython-input-4-bb809aa8b209>](/<ipython-input-4-bb809aa8b209>) in <module>()
 **3** 
 **4** # Trying to use an attribute from Pandas objects for a list
----> 5 a.shapeAttributeError: 'list' object has no attribute 'shape'
```

**如何求解:**确保使用的属性适用于那个对象。知道这一点的一个好方法是使用`dir(object_name)`。它将列出可用的属性和方法。

## 索引错误

`IndexError` *序列的索引超出范围时发生。*

如果您有一个包含 3 个元素的列，但是您想要打印 4 个元素，您的循环将不会再次开始。它将升高**索引错误**。

```
# list
a = [1,2,3]# Loop out of range
for i in range(4):
print( a[i] )1
2
3----------------------------------------------------------------IndexError                   Traceback (most recent call last)[<ipython-input-95-cd66a0748b5d>](/<ipython-input-95-cd66a0748b5d>) in <module>()
 **4** # Loop out of range
 **5** for i in range(4):
----> 6   print( a[i] )IndexError: list index out of range
```

**如何解决:**确保你在物体的范围/长度内循环。

## 键盘错误

当在字典或数据帧中找不到关键字时，`KeyError`为*。*

这个错误有时让我有些头疼。假设您正在处理一个数据框，并且删除了一行。如果您忘记重置索引(或出于某种原因不想这么做),并且您稍后将执行使用索引的操作，如果没有找到给定的数字，则 **KeyError** 将弹出。

```
# Loop through Tips dataset and make an operation for every row.
# However, the dataset goes from 0-243 index# length is 3, but let's try a loop 4 times.
for i in range(245):
df.tip[i] - 0.1 -----------------------------------------------------------ValueError                      Traceback (most recent call last)[/usr/local/lib/python3.7/dist-packages/pandas/core/indexes/range.py](/usr/local/lib/python3.7/dist-packages/pandas/core/indexes/range.py) in get_loc(self, key, method, tolerance)
 **384**                 try:
--> 385                     return self._range.index(new_key)
 **386**                 except ValueError as err:ValueError: 244 is not in rangeThe above exception was the direct cause of the following exception:KeyError                                  Traceback (most recent call last)[/usr/local/lib/python3.7/dist-packages/pandas/core/indexes/range.py](/usr/local/lib/python3.7/dist-packages/pandas/core/indexes/range.py) in get_loc(self, key, method, tolerance)
 **385**                     return self._range.index(new_key)
 **386**                 except ValueError as err:
--> 387                     raise KeyError(key) from err
 **388**             raise KeyError(key)
 **389**         return super().get_loc(key, method=method, tolerance=tolerance)KeyError: 244
```

再比如。

```
# Create a Dataframe
df = pd.DataFrame( {'number': range(1,6), 'val': np.random.randn(5)})# Drop row 2
df.drop(2, axis=0, inplace=True)# Print Index 2
df.loc[2]---------------------------------------------------------------KeyError: 2The above exception was the direct cause of the following exception:KeyError                                  Traceback (most recent call last)[/usr/local/lib/python3.7/dist-packages/pandas/core/indexes/base.py](/usr/local/lib/python3.7/dist-packages/pandas/core/indexes/base.py) in get_loc(self, key, method, tolerance)
 **3361**                 return self._engine.get_loc(casted_key)
 **3362**             except KeyError as err:
-> 3363                 raise KeyError(key) from err
 **3364** 
 **3365**         if is_scalar(key) and isna(key) and not self.hasnans:KeyError: 2
```

**如何解决:**仔细检查指标没有任何遗漏数字。此外，您可以检查该“键”是否存在于您的数据集中。

## 值错误

`ValueError` *当一个函数得到一个类型正确但值不正确的参数时弹出。*

当你有正确的函数，但是你输入了错误的参数时，就会发生这种情况。例如，尝试将文本列指定为整数。

```
# Create a Dataframe
df = pd.DataFrame( {'Letter': ['a', 'b', 'c', 'd', 'e'], 'val': np.random.randn(5)})# Try to assign string to integer
df.Letter.astype('int')------------------------------------------------------------------ValueError                     Traceback (most recent call last)[<ipython-input-61-805ac8fc7e9b>](/<ipython-input-61-805ac8fc7e9b>) in <module>()
----> 1 df.Letter.astype('int')/usr/local/lib/python3.7/dist-packages/pandas/_libs/lib.pyx in pandas._libs.lib.astype_intsafe()ValueError: invalid literal for int() with base 10: 'a'
```

或者你可以看到，如果你试图运行一个非数字变量的线性回归。不同场景下也是一样的情况。

```
import seaborn as sns
from sklearn.linear_model import LinearRegression# Load dataset Tips
df = sns.load_dataset('tips')# Regression Model
model = LinearRegression()# Assign X and y
X = df[['sex']]
y= df.tip# Mode fit
model.fit(X, y)-------------------------------------------------------------------ValueError        Traceback (most recent call last)ValueError: could not convert string to float: 'Female'The above exception was the direct cause of the following exception:ValueError                Traceback (most recent call last)[/usr/local/lib/python3.7/dist-packages/sklearn/utils/validation.py](/usr/local/lib/python3.7/dist-packages/sklearn/utils/validation.py) in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)
 **790**           "Unable to convert array of bytes/strings "
 **791**          "into decimal numbers with dtype='numeric'"
--> 792                 ) from e
 **793**         if not allow_nd and array.ndim >= 3:
 **794**             raise ValueError(ValueError: Unable to convert array of bytes/strings into decimal numbers with dtype='numeric'
```

**如何求解:**理解你的变量，打印它们`type(variable)`以了解你正在处理的对象，阅读文档并检查函数期望接收的预期参数是什么。

![](img/81faadbc7a4c8aafbdb7b89172588609.png)

照片由[Olav Ahrens rtne](https://unsplash.com/@olav_ahrens?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/solution?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

## 在你走之前

编程是一个逻辑问题。你必须学会如何像计算机一样思考，因为它期望接收的信息中的微小差异都会给你带来错误。

学习如何阅读文档和在互联网上查找有用的信息会给你很大的帮助，特别是如果你知道你正在处理什么类型的错误。

当然，随着你越来越有经验，这些错误对你来说也越来越熟悉。

总而言之:

*   你忘了定义变量名。
*   `TypeError`:你试图用错误类型的对象做一些事情。不能取文字的中值。
*   `AttributeError`:属性不针对该对象。列表没有形状。使用`dir()`查看有什么可用的。
*   `IndexError`:你的循环超出范围。你试图将某些东西应用到物体不存在的部分。
*   `KeyError`:钥匙不存在。例如，您删除了第一行。索引 0 将不再作为键存在。
*   你正试图使用一个错误的值作为函数的参数。如果函数需要文本，则不能输入数字。解决这个问题的最好方法是阅读文档。

## 参考

[](https://www.tutorialsteacher.com/python/error-types-in-python)    [](https://github.com/gurezende/Studying/tree/master/Python/errors)  

如果这个内容有意思就关注我吧。

[](https://gustavorsantos.medium.com/)  

如果你想订阅 Medium，这里有[我的推荐链接](https://gustavorsantos.medium.com/membership)。