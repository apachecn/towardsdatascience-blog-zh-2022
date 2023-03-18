# 在数据准备过程中必须使用的热门 Python 关键字

> 原文：<https://towardsdatascience.com/top-python-keywords-that-you-must-use-in-your-data-preparation-process-290536a46576>

## 为初学者提供易于理解的示例

![](img/01853aee62728c3134d1608af5be59c9.png)

Theo Crazzolara 在 Unsplash 上的照片

你有数据。

你需要洞察力。

不幸的是，在您从数据中获得任何洞察力之前，您需要处理**数据准备**过程。

此时，有常用的 **Python 关键字**帮助您完成基本的数据准备任务。

在本文中，我将通过简单的例子来解释那些**顶级 Python 关键字**以及它们在数据准备过程中的用途。

# 什么是关键字

**Python 关键字**是保留字，不能用作变量名、函数名或任何其他标识符，因为它们在 Python 语言中有特定的用途和含义。

**Python 3.8** 有 **35 个关键字**，列举如下:

```
False      await      else       import     pass
None       break      except     in         raise
True       class      finally    is         return
and        continue   for        lambda     try
as         def        from       nonlocal   while
assert     del        global     not        with
async      elif       if         or         yield
```

你不必**导入** **关键词**，因为它们总是可用的，但它们需要完全按照上面的写法拼写。

**Python 关键字**可分类如下:

*   导入关键词:`import`、`from`、`as`
*   结构关键词:`def`、`class`、`with`、`pass`、`lambda`
*   价值关键词:`True`、`False`、`None`
*   操作员关键词:`and`、`or`、`not`、`in`、`is`
*   控制流关键字:`if`、`elif`、`else`
*   迭代关键词:`for`、`while`、`break`、`continue`
*   返回关键词:`return`，`yield`
*   异常处理关键字:`try`、`except`、`raise`、`finally`、`assert`
*   异步编程关键词:`async`，`await`
*   变量处理关键字:`del`、`global`、`nonlocal`

# 什么是数据准备

**数据准备**过程包含一组预建模任务。这些任务可以分类如下:

*   **数据清理**:纠正或删除数据集中不正确、损坏、缺失、重复或不完整的数据
*   **特征选择**:定义与任务最相关的输入变量。
*   **数据转换**:改变数据的规模或分布。
*   **特征工程**:从可用数据中推导出新变量。
*   **降维**:减少数据集中输入变量的数量，同时保持尽可能多的变化。

使用哪种特定的数据准备任务取决于将用于建模的数据和算法。

# 数据准备任务中使用的热门 Python 关键字

## **‘导入’和‘as’**

在数据科学项目中执行特定任务时，不要重新发明轮子，您需要使用其他人的**模块和库**。要使用这些库，您需要使用**导入关键字**将它们导入到您的代码中，例如**‘import’**、**as’**，以及**from’**。

```
**import** pandas **as** pd
**import** numpy **as** np
```

在上面的代码中， **pandas 和 numpy，**库被导入。我们将在后面的代码中使用这些模块。**‘as’**关键字在这里帮助我们重命名模块。当使用长名称的模块时，或者当需要分隔**名称空间**时，这尤其有用。

## **‘def’**

**‘def’**用于定义一个 python 函数。函数在数据科学项目中大量使用。它们帮助我们将大的代码块转换成逻辑的和可管理的部分。

让我们创建**一个函数**来打印数据框列中缺失项目的计数。

```
**def** missing_item_count(**df**):
```

## for 和 in

一种常见的做法是循环遍历**数据帧、**或复杂的**数据对象**中的项目，如**字典**或**列表**。**中的**和**中的**对这样的任务来说是绝配。下面你可以看到，我们可以用关键字的**得到**‘Airbnb’**data frame 的列。**

我们的循环从关键字的**开始，然后我们添加变量**‘col’**来分配数据容器的每个元素，后面是关键字**中的**’。在'**关键字中的**之后，最后是**【df . columns】**，它是数据容器本身。**

```
**airbnb_url** = 'https://raw.githubusercontent.com/ManarOmar/New-York-Airbnb-2019/master/AB_NYC_2019.csv'
**airbnb** = pd.read_csv(**airbnb_url**)**def** missing_item_count(**df**):
  **for** col **in** df.columns:
    **missing_item_count** = df[**col**].isna().sum()
    print(f'Column {**col**} has {**missing_item_count**} missing items')missing_item_count(**airbnb**)**Output:** Column **id** has 0 missing items 
Column **name** has 16 missing items 
Column **host_id** has 0 missing items 
Column **host_name** has 21 missing items
```

现在，在 for 循环中，我们可以迭代 **df.column** 对象中的项，并在 **'col'** 变量中获取它们。

## “如果”和“否则”

**‘如果’**，**‘否则’**控制流关键字用于决策。代码块的执行取决于**测试表达式的值。**

```
**def** missing_item_count(**df**):
  **for** col **in** df.columns:
    **missing_item_count** = df[**col**].isna().sum()

    **if** **pct**:
      print(f'Column {**col**} has {**missing_item_count**} missing items')
    **else:**
      print(f'Column {**col**} has ZERO missing item')missing_item_count(**airbnb**)**Output:**
Column **id** has ZERO missing item 
Column **name** has 16 missing items 
Column **host_id** has ZERO missing item 
Column **host_name** has 21 missing items
```

在上面的代码中，如果 **missing_item_count** 变量是 **True** (如果在我们的例子中它不是一个**零整数**，它打印列名和 **missing_item_count** 值。

如果 **missing_item_count** 变量为 **False，(如果是零整数则为**)**，**则执行 **else 关键字**内的代码块。

这就是你如何用**‘if’和‘else’**关键字控制你的代码流。

# 关键要点和结论

*   **Python 关键字**是保留字，有特定的含义和用途。您不必**将** **关键字**导入到您的代码中，因为它们总是可用的
*   **‘def’**用于定义一个 python 函数。函数在数据科学项目中大量使用。它们帮助我们将大的代码块转换成逻辑的和可管理的部分。
*   **【if】****【else】**控制流关键字用于决策。代码块的执行取决于**测试表达式的值。**
*   一种常见的做法是循环遍历**数据帧、**或复杂的**数据对象**中的项目，如**字典**或**列表**。**中的**和**中的**对这样的任务来说是绝配。

我希望你已经发现这篇文章很有用，并且**你将开始在你自己的代码**中使用上述关键词。