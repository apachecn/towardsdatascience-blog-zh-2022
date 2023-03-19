# 如何将 NumPy 数组规范化为单位向量

> 原文：<https://towardsdatascience.com/normalise-numpy-unit-vector-f9f086134507>

## 用 Python 将 numpy 数组归一化为单位向量

![](img/476791d37b1ca32dca6e1f80ee195b50.png)

由 [Sebastian Svenson](https://unsplash.com/@sebastiansvenson?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/vector?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

## 介绍

标准化是一个数据转换过程，发生在模型开发的早期阶段，目的是**改变数据数值的分布形状**。例如，您可能希望对数字数据点进行规范化，使它们的总和为 1，并被描述为概率分布。

各种机器学习模型受益于数据标准化，**特别是那些利用欧几里德距离的模型**。

在今天的文章中，我们将展示如何**将一个 numpy 数组规格化为一个单位向量**。这意味着我们将改变向量的大小，使每个向量的长度为 1。

更具体地说，我们将使用以下工具来探索如何做到这一点:

*   `scikit-learn`图书馆
*   `numpy`的`linalg.norm()`方法，
*   `scipy`中的`linalg.norm()`

首先，让我们创建一个 NumPy 数组，我们将在示例中引用它来演示一些概念。

```
import numpy as np # Ensure values are reproducible
np.random.seed(1234)array_1d = np.random.rand(15) * 10print(array_1d)
***array([1.9151945 , 6.22108771, 4.37727739, 7.85358584, 7.79975808,
       2.72592605, 2.76464255, 8.01872178, 9.58139354, 8.75932635,
       3.5781727 , 5.00995126, 6.83462935, 7.12702027, 3.70250755])***
```

## 使用 scikit-learn normalize()方法

谈到归一化一个 numpy 数组，我们的第一个选择是`[sklearn.preprocessing.normalize()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html)`方法，该方法可用于*将输入向量单独缩放到单位范数(向量长度)*。下面分享的例子说明了这一点。

```
from sklearn.preprocessing import normalize**array_1d_norm = normalize(array_1d[:,np.newaxis], axis=0).ravel()**print(array_1d_norm)
***[0.07945112 0.25807949 0.18158971 0.32580306 0.32357003 0.11308402
 0.11469016 0.33265366 0.39748051 0.36337736 0.14843915 0.20783595
 0.28353203 0.29566176 0.15359713]***
```

注意，默认情况下，用于标准化输入的标准将被设置为`'l2'`。这意味着，如果我们对规范化数组的元素求和，我们不应该期望它等于 1。事实上，这里的情况是这样的:

```
print(sum(array_1d_norm))
***3.578845135327915***
```

如果你希望得到的向量的和等于 1(概率分布)，你应该将`'l1'`值传递给`norm`参数:

```
from sklearn.preprocessing import normalize**array_1d_norm = normalize(
    array_1d[:,np.newaxis], 
    axis=0, 
    norm='l1',
).ravel()**print(array_1d_norm)
***[0.02220021 0.0721125  0.05073975 0.09103581 0.09041186 0.03159791
 0.03204669 0.09295    0.1110639  0.10153481 0.04147683 0.05807347
 0.07922445 0.08261373 0.04291807]***
```

现在，如果我们对结果数组的值求和，我们应该期望它等于 1:

```
print(sum(array_1d_norm))
***1***
```

## 使用 numpy 和 linalg.norm()方法

我们的另一个选择是 numpy 的`[linalg.norm()](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)`方法，它将根据`ord`参数中指定的值(默认为`None`)返回八个不同矩阵范数中的一个。

```
**array_1d_norm = array_1d / np.linalg.norm(array_1d)**print(array_1d_norm)
***[0.07945112 0.25807949 0.18158971 0.32580306 0.32357003 0.11308402
 0.11469016 0.33265366 0.39748051 0.36337736 0.14843915 0.20783595
 0.28353203 0.29566176 0.15359713]***print(sum(array_1d_norm))
***3.578845135327915***
```

同样，如果你想把这些值加起来等于 1(例如概率分布)，你需要指定`ord=1`。

```
**array_1d_norm = array_1d / np.linalg.norm(array_1d, ord=1)**print(array_1d_norm)
***[0.02220021 0.0721125  0.05073975 0.09103581 0.09041186 0.03159791
 0.03204669 0.09295    0.1110639  0.10153481 0.04147683 0.05807347
 0.07922445 0.08261373 0.04291807]***print(sum(array_1d_norm))
***1***
```

## 使用 scipy linalg()方法

或者，你甚至可以使用`[scipy.linalg()](https://docs.scipy.org/doc/scipy/tutorial/linalg.html)`方法，它本质上包含了与`numpy`的`linalg()`方法相同的功能，加上一些后者不包含的额外的和更高级的功能。

这里使用的符号与我们在前面的例子中使用的符号完全相同。

```
from scipy import linalg**array_1d_norm = array_1d / linalg.norm(array_1d, ord=1)**print(array_1d_norm)
***[0.02220021 0.0721125  0.05073975 0.09103581 0.09041186 0.03159791
 0.03204669 0.09295    0.1110639  0.10153481 0.04147683 0.05807347
 0.07922445 0.08261373 0.04291807]***print(sum(array_1d_norm))
***1***
```

## 最后的想法

在今天的简短教程中，我们讨论了单位向量的 numpy 数组归一化。数据标准化是改变特定数据集中数值数据点分布的过程。这是在模型开发的早期阶段应用的一个常见的预处理转换步骤。

今天，我们展示了一些不同的方法来归一化 numpy 数组，最终将帮助您的机器学习模型执行得更好。

在我的另一篇关于 Medium 的文章中，您还可以了解到另一个关于数据预处理和转换的类似概念，称为**特性缩放**。

[](/feature-scaling-and-normalisation-in-a-nutshell-5319af86f89b)  

[**成为会员**](https://gmyrianthous.medium.com/membership) **阅读介质上的每一个故事。你的会员费直接支持我和你看的其他作家。你也可以在媒体上看到所有的故事。**

[](https://gmyrianthous.medium.com/membership)  

**你可能也会喜欢**

[](/random-seed-numpy-786cf7876a5f)  [](/scikit-learn-vs-sklearn-6944b9dc1736) 