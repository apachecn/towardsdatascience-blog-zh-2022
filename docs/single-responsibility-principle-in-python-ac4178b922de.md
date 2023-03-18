# Python 中的单一责任原则

> 原文：<https://towardsdatascience.com/single-responsibility-principle-in-python-ac4178b922de>

## 为初学者提供易于理解的解释

![](img/cecaf4ca367cd7c7c2d5e58e6232effa.png)

由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的 [Dmitriy](https://unsplash.com/@d_lipenchuk?utm_source=medium&utm_medium=referral) 拍摄的照片

> 让软件工作和让软件干净是两个非常不同的活动——**罗伯特·C·马丁(干净的代码)**

编码时，你很容易被手头的任务冲昏头脑，只专注于让你的代码工作。结果，你失去了添加到代码中的代码行的踪迹，并以**庞大、无组织、但仍能工作的函数**结束。

**单一责任原则**是一个软件设计指南，它规定您代码中的每个**模块**、**类**或**函数**都应该有**唯一的责任和唯一的更改理由**。

这个原则是关于**组织你的代码的复杂性，把因为同样原因而改变的东西收集在一起**，这样你就知道**在哪里寻找**来修改东西，而不用考虑**所有涉及的复杂性。**

继续阅读，了解更多关于单一责任原则的内容，为什么以及如何在 Python 中实现。

# 适用单一责任原则的好处

如果你遵循**单一责任原则**，你最终会得到**大量的小函数，或者类**而不是大函数。

你可能会认为这可能不是一个好主意。也许你更喜欢一些大的函数和类。

显然，如果你盲目地遵循这个原则，将你的代码分解成原子部分，这也会导致一些不希望的副作用。应该有一个**平衡考虑**，下面的引用定义了它是什么；

> *把* ***因为同样的原因而改变的东西*** *。把那些* ***因不同原因而改变的东西分开****——***罗伯特·C·马丁**

为什么遵循单一责任原则是一个好主意；

*   它有助于将一大块代码转换成定义明确、标记良好、高度内聚、干净和健壮的组件。
*   它要求你命名更多的代码块，并强迫你明确你的意图。随着时间的推移，这将使您的代码更具可读性。
*   当你的函数和类尽可能小的时候，很容易发现重复的代码。
*   定义良好的小代码块可以更好地混合和重用。
*   小功能容易测试，也容易被别人理解。
*   当一个函数不属于一个名称空间时，很容易发现。
*   花时间确定独特的职责有助于我们在代码中识别和创建更好的抽象。

![](img/05003830b6d20e05473d902711106183.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Austin Distel](https://unsplash.com/@austindistel?utm_source=medium&utm_medium=referral) 拍摄的照片

# 单一责任原则的 Python 实现

## 单一责任原则— Python 类:

你可以看到下面的**模型**类有很多职责。预处理数据、训练和评估模型、做出预测是不同的职责，都在**模型**类中处理。这违反了单一责任原则，由于上述原因，强烈建议不要这样做。

```
**# Before the single responsibility principle**class **Model**:

  def **pre_process(self)**:
    pass def **train(self)**:
    pass

  def **evaluate(self):**
    pass def **predict(self):**
    pass
```

如下所示，我们可以创建单独的类来处理每个职责，以使我们的类与单一职责原则兼容。

```
**# After the single responsibility principle applied**class **PreProcess**:
  passclass **Train**:
  passclass **Evaluate**:
  passclass **Predict**:
  pass
```

## 单一责任原则— Python 函数:

说到职能，遵循单一责任原则就更重要了。我总是发现自己在一个功能体中处理许多任务，这使得功能变得庞大而不杂乱。

```
**# Before the single responsibility principle applied**def **pre_processing_data()**:
  #importing data
  #converting data types
  #handling missing values
  #handling outliers
  #transforming data
```

现在，一旦我们在一个单独的函数中处理每个任务，我们就可以拥有比第一个更干净、更容易混合和重用的函数。

```
**# After the single responsibility principle applied**def **import_data()**: 
  passdef **convert_data_type():** passdef **handle_missing_values(): 
 ** passdef **handle_outliers():** passdef **transform_data():** pass 
```

# 关键要点和结论

关键要点是:

*   **单一责任原则**是一个软件设计指南，它规定您代码中的每个**模块**、**类**或**函数**都应该有**唯一的责任和唯一的更改理由**。
*   它有助于将一大块代码转换成**定义明确、标记良好、高度内聚、干净和健壮的组件**。
*   定义良好的小代码块可以更好地混合和重用。

我希望你发现这篇文章很有用，并且**将开始在你自己的代码**中应用单一责任原则。