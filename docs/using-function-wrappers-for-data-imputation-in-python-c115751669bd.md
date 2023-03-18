# 在 Python 中使用函数包装器进行数据插补

> 原文：<https://towardsdatascience.com/using-function-wrappers-for-data-imputation-in-python-c115751669bd>

## 使用 Python Decorators 的自定义数据插补方法

![](img/49422addcc4248d100298be9c123a7cb.png)

图片由乔治·多尔吉克在[像素](https://www.pexels.com/photo/red-white-and-brown-gift-boxes-1303081/)上拍摄

在 Python 中，函数包装器(也称为装饰器)用于修改或扩展现有函数的行为。他们有各种各样的应用程序，包括调试、运行时监控、web 开发中的用户登录访问、插件等等。虽然通常应用于软件工程的环境中，但是函数包装器也可以用于数据科学和机器学习任务。例如，在开发数据处理管道和机器学习模型时，可以使用运行时监控和使用函数包装进行调试。

函数包装器在数据科学和机器学习领域的一个有趣应用是用于数据插补。数据插补是推断和替换数据中缺失值的任务。数据插补可以帮助减少偏差，提高数据分析的效率，甚至提高机器学习模型的性能。

有几种众所周知的技术用于输入数据集中的缺失值。最简单的方法是用零替换所有缺失的值。这是有限的，因为这种插补值可能无法准确反映现实，也不一定能减少偏差和提高数据分析的效率。在某些情况下，它实际上可能会引入大量的偏差，尤其是当一列中的大部分值缺失时。另一种方法是用平均值替换缺失的数值。虽然这比用零进行估算要好，但它仍然会导致偏差，尤其是当一列中有很大一部分数据丢失时。

另一种方法是建立一个机器学习模型，根据数据中其他列的值来预测缺失值。这种方法非常理想，因为即使在特定列中有很大一部分数据缺失的情况下，基于其他列的推断也应该有助于减少偏差。这种方法可以通过在类别级别应用机器学习模型来进一步改进。理论上，这可以用来相对较好地估算一整列缺失值。此外，类别和模型的粒度越细，这种方法应该工作得越好。

对于前两种方法，我们可以简单地使用 pandas fillna()方法用零、平均值和众数来填充缺失值。对于输入带有预测的缺失值，我们可以使用 [Scikit-learn](https://scikit-learn.org/stable/index.html) 包中的[迭代输入](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer)模块。在这里，我们将看看如何使用函数包装器来设计每种方法的数据插补方法。

在这里，我们将使用葡萄酒杂志数据集，可以在这里找到。在知识共享许可(CC0:公共领域)下，这些数据可以公开自由使用、修改和共享。

对于我的分析，我将在 [Deepnote](https://deepnote.com/) 中编写代码，这是一个协作数据科学笔记本，使运行可重复的实验变得非常容易。

## 用零输入缺失值

首先，让我们导航到 Deepnote 并创建一个新项目(如果您还没有帐户，可以免费注册)。

让我们创建一个名为“数据 _ 估算”的项目，并在这个项目中创建一个名为“估算”的笔记本:

![](img/b3f734fe60fe1c29f2f752902b763871.png)

作者截图

接下来，让我们导入将要使用的包:

作者创建的嵌入

现在，让我们定义我们的函数，我们将使用零来估算缺失值。我们称之为简单插补。它将采用一个名为 input_function 的参数作为自变量。我们还将把输入函数传递给 functools 包装器中的 wraps 方法，我们将把它放在实际插补函数之前，称为 simple _ attribute _ wrapper:

```
def simple_imputation(input_function):
    @functools.wraps(input_function)
    def simple_imputation_wrapper(*args, **kwargs):
```

接下来，在 simple _ attraction wrapper 的范围内，我们指定了在输入函数返回的数据帧中输入缺失值的逻辑。

```
def simple_imputation_wrapper(*args, **kwargs):
    return_value = input_function(*args, **kwargs)
    print(" — — — — — — — Before Imputation — — — — — — — ")
    print(return_value.isnull().sum(axis = 0)).  return_value.fillna(0, inplace = True)
    print(" — — — — — — — After Imputation — — — — — — — ")   
    print(return_value.isnull().sum(axis = 0))
    return return_value
```

我们的插补函数(simple _ attachment _ wrapper)是在我们的 simple _ attachment 函数的范围内定义的。完整的功能如下:

作者创建的嵌入

接下来，让我们定义一个函数，该函数读入我们的 Wines 数据集并返回包含我们的数据的 dataframe:

```
def read_data():
    df = pd.read_csv(“wines_data.csv”, sep = “;”)
    return df
```

现在，如果我们调用 read_data 函数，它将具有简单插补方法的附加行为:

作者创建的嵌入

## 使用均值和模式输入缺失值

接下来，我们将定义一种数据插补方法，用平均值替换缺失的数值，用模式替换缺失的分类值。

我们将称我们的新函数为均值模式 _ 插补。它还会将 input_function 作为参数。我们还将把输入函数传递给 functools 包装器中的 wraps 方法，我们将把它放在我们实际的均值/模式插补函数之前，称为 mean mode _ attribute _ wrapper:

```
def meanmode_imputation(input_function):
    @functools.wraps(input_function)
    def meanmode_imputation_wrapper(*args, **kwargs):
```

接下来，在 meanmode _ 插补包装器的范围内，我们指定了在由输入函数返回的数据帧中输入缺失值的逻辑。这里，我们将迭代列类型，如果列类型为“浮点型”，则估算平均值，如果列类型为“类别”，则估算模式:

```
def meanmode_imputation_wrapper(*args, **kwargs):
    return_value = input_function(*args, **kwargs)
    print("— — — — — — — Before Mean/Mode Imputation — — — — — — — ")
    print(return_value.isnull().sum(axis = 0))
    for col in list(return_value.columns):
        if return_value[col].dtype == float:       
            return_value[col].fillna(return_value[col].mean(), inplace = True).       
        elif return_value[col].dtype.name == 'category':
           return_value[col].fillna(return_value[col].mode()[0], inplace = True) print(" — — — — — — — After Mean/Mode Imputation — — — — — — — ")
    print(return_value.isnull().sum(axis = 0))
    return return_value
```

完整的功能如下:

作者创建的嵌入

我们还需要修改 read_data 函数，使其接受列名字典，并将分类列和数字列类型分别指定为 category 和 float。我们通过迭代列名并使用我们的列和数据类型字典为每个列转换列类型来实现这一点:

```
for col in list(df.columns):
        df[col] = df[col].astype(data_type_dict[col])
```

完整的功能如下:

作者创建的嵌入

接下来，我们需要定义我们的数据类型映射字典:

```
data_type_dict = {'country':'category', 'designation':'category','points':'float', 'price':'float', 'province':'category', 'region_1':'category','region_2':'category', 'variety':'category', 'winery':'category', 'last_year_points':'float'}
```

我们可以将字典传递给 read data 方法:

```
df = read_data(data_type_dict)
```

我们得到以下输出:

作者创建的嵌入

## 用迭代输入器输入缺失值

对于我们的最终函数包装器，我们将使用 Scikit-learn 插补模块中的迭代插补器。iterative input 使用一个估计器，通过使用所有其他列中的值来迭代地估算一列中的缺失值。默认估计量是贝叶斯岭回归估计量，但这是一个可以修改的参数值。让我们从导入 IterativeImputer 开始:

```
from sklearn.impute import IterativeImputer
```

接下来，类似于前面的函数包装器，我们定义一个称为迭代 _ 插补的函数，它采用一个输入函数，在插补包装器之前调用 wraps 方法，并将插补包装器定义为迭代 _ 插补 _ 包装器。我们还存储输入函数的返回值，并打印插补前缺失值的数量:

```
def iterative_imputation(input_function):
    @functools.wraps(input_function)
    def iterative_imputation_wrapper(*args, **kwargs):                           
        return_value = input_function(*args, **kwargs)
        print("--------------Before Bayesian Ridge Regression Imputation--------------")
        print(return_value.isnull().sum(axis = 0))
```

接下来，在迭代插补包装范围内，我们定义包含分类列和数字列的数据帧:

```
return_num = return_value[['price', 'points', 'last_year_points']]
return_cat = return_value.drop(columns=['price', 'points', 'last_year_points'])
```

然后我们可以定义我们的插补模型。我们用 10 次迭代和一个随机状态集来定义我们的模型对象。我们还将使用默认估计量，即贝叶斯回归估计量:

```
imp_bayesian = IterativeImputer(max_iter=10, random_state=0)
```

然后，我们可以拟合我们的模型并估算缺失的数值:

```
imp_bayesian.fit(np.array(return_num))return_num = pd.DataFrame(np.round(imp_bayesian.transform(np.array(return_num))), columns = ['price', 'points', 'last_year_points'])
```

我们还将继续用模式输入分类变量。值得注意的是，分类值也可以用分类模型估算(我将把这个任务留到以后的文章中):

```
for col in list(return_cat.columns):
    return_cat[col].fillna(return_cat[col].mode()[0], inplace = True)
return_value = pd.concat([return_cat, return_num], axis=1)
```

完整的功能如下:

作者创建的嵌入

现在，我们可以将迭代插补装饰器放在 read_data 方法之前:

作者创建的嵌入

像以前一样调用我们的方法:

作者创建的嵌入

改进这种插补方法的另一种方法是在类别一级建立插补模型。例如，为每个国家建立一个估算器来估算缺失的数值。我鼓励你试验一下代码，看看你是否能做出这样的修改。在以后的文章中，我将介绍如何建立这些类别级插补模型，并探索其他插补方法。

这篇文章中的代码可以在 [GitHub](https://github.com/spierre91/deepnote/blob/main/imputer.ipynb) 上找到

## 结论

有各种各样的技术可用于数据插补。我们介绍的最简单的方法是用零替换缺失值。这种方法并不理想，因为它会导致很大的偏差，尤其是在有大量缺失值的情况下。更好的方法是用平均值估算缺失的数值，用模式估算缺失的类别值。虽然这是对用零输入缺失值的改进，但它可以使用机器学习模型来进一步改进。此外，在类别一级建立插补模型可以提供进一步的改进。