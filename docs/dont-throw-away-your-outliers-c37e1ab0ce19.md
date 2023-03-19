# 不要扔掉你的离群值！

> 原文：<https://towardsdatascience.com/dont-throw-away-your-outliers-c37e1ab0ce19>

![](img/4ed7fc488fa6265a3a40a70fe53ec778.png)

由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的[absolute vision](https://unsplash.com/@freegraphictoday?utm_source=medium&utm_medium=referral)拍摄

## 首先，在数据集中寻找错误的原因。

**在** [**检测到异常值**](https://hennie-de-harder.medium.com/are-you-using-feature-distributions-to-detect-outliers-48e2ae3309) **或者异常后，你需要决定如何处理它们。这篇文章解释了处理异常值的技巧。**

# 第一步，调查

调查你的异常值。它们为什么会发生？它们真的是错误吗？现实生活中永远不会发生吗？它们在数据中，所以发现错误来自哪里是很重要的。由于实验或人为错误，可能会出现异常值。加载或处理数据时可能会出错。找到原因后，你就可以决定该怎么做了。

问题解决了吗？如果您决定离群值不应该出现在数据中并删除它们，请确保您可以定义原因。或者更好:记录下来。如果您发现新数据可能与异常值具有相同的值，您应该使用其他技术来处理它们，如下所述。

# 处理异常值

如果离群值与其他数据点相比只是不规则的，但是可能会发生，那么要小心它们！您可以尝试几种技术来改善机器学习模型的结果，而不删除异常值。

这些技术基于数据转换:

## 给数据封顶

您可以尝试的一种特殊转换是对数据进行 capping 或[**winsor izing**](https://www.statology.org/winsorize/)**。这是一种将特性的最大值设置为特定值的方法。您可以决定将最低的 2%设置为第二个百分位的值，最高的 2%设置为第 98 个百分位的值。这里是一个使用 python 代码的例子。**

## **其他数据转换**

**除了封顶，还有其他方法来转换数据。您可以使用缩放器、对数变换或宁滨。**

**缩放器是您应该尝试的一种技术，因为缩放会对模型的结果产生巨大的影响。在这里你可以找到 sci-kit learn 对不同 scaler 的比较。你可能想[阅读更多关于**鲁棒定标器**](https://machinelearningmastery.com/robust-scaler-transforms-for-machine-learning/) 的信息，它被设计用来处理异常值。**

****当变量以指数比例增长或衰减时，最好使用对数变换**，因为变换后变量更接近线性比例。尾部的影响减小了。**

****宁滨**，或称分桶，将一系列值替换为一个代表值。您可以以不同的方式使用宁滨:您可以设置箱的[距离](https://pandas.pydata.org/docs/reference/api/pandas.cut.html)，您可以使用[频率](https://pandas.pydata.org/docs/reference/api/pandas.qcut.html)(每个箱获得相同数量的观察值)，或者[采样](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html)。**

**![](img/96363f66e737ce1900d11493e05aaec0.png)**

**照片由[玛丽娜·赫拉波娃](https://unsplash.com/es/@mimiori?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄**

**基于模型的技术:**

## **使用正则化**

**正则化减少了方差，这使得它在处理异常值时很有用。两种最常见的正则化类型是套索正则化和岭正则化，也分别称为 L1 正则化和 L2 正则化。**

**正则化通过惩罚高值回归系数来工作。它简化了模型，使它更健壮，不容易过拟合。**

**正则项在所有著名的 ML 包中都有实现。例如，看看 scikit-learn [套索](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso)和[脊](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)或[添加正则化到 Keras 层](https://keras.io/api/layers/regularizers/)。**

## **型号选择**

**有些模型在处理异常值方面比其他模型更好。例如，如果你正在处理一个回归问题，你可以尝试随机样本一致性(RANSAC)回归或者[希尔森回归](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html#sklearn.linear_model.TheilSenRegressor)。 [RANSAC](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html) 使用来自数据的样本，建立一个模型，并使用残差来分离内部值和外部值。最终的模型只使用内联器。**

**如果您想要使用所有数据点，您可以使用基于树的模型。基于树的模型通常比其他模型更好地处理异常值。一个直观的解释是:树关心的是一个数据点位于拆分的哪一侧，而不是拆分和数据点之间的距离。**

## **更改误差度量**

**最后一个选项是改变误差度量。一个简单的例子:如果你使用均方差，你会更严厉地惩罚异常值，因为你平方了误差。您可能希望切换到平均绝对误差，因为您采用误差的绝对值，而不是平方它。**

# **摘要**

**在扔掉离群值之前，调查它们。寻找数据集中出现错误的原因。如果您无法解决这些问题，请使用变换、添加正则化或选择适当的模型或误差度量来处理数据。**

**感谢阅读！❤**

## **有关系的**

**[](https://hennie-de-harder.medium.com/are-you-using-feature-distributions-to-detect-outliers-48e2ae3309) **