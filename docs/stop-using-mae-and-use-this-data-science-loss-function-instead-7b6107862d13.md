# 停止使用 MAE，改用这个数据科学损失函数

> 原文：<https://towardsdatascience.com/stop-using-mae-and-use-this-data-science-loss-function-instead-7b6107862d13>

## 意见

## 研究分位数损失以及何时使用它——业务用例

![](img/78923c3a27cc170e32a01fdb03d25dbd.png)

在[Unsplash](https://unsplash.com/s/photos/%25?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【1】上由[叶祖尧](https://unsplash.com/@josephyip?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)拍摄的照片。

# 目录

1.  介绍
2.  回归和损失函数
3.  何时使用分位数，何时
4.  摘要
5.  参考

# 介绍

作为一名在网上学到了很多东西的数据科学家，我看到了关于损失函数的讨论很少，这些函数不是梅或 RMSE 的。出于这个原因，我将给出一个何时使用不同损失函数的快速总结，它是强大的分位数损失函数及其变体。除了那些想了解更多有关何时使用分位数损失函数的人之外，这个讨论对于没有听说过这个函数的数据科学家也很有用。话虽如此，让我们看看分位数的一些什么、为什么和何时，特别是业务用例。

## 回归和损失函数

首先，让我们在深入业务用例之前先了解一下我们的方向。我们可以使用分位数损失函数来解决回归问题，这一点我将在本文中讨论。回归是一种预测连续变量的算法。例如，如果我们想预测一个在 0 到 100 范围内的值。

> 以下是常用于回归算法的其他损失函数的示例:

*   **MAE** 针对中值(*平均绝对值*)进行优化，而不关注方向优化，因此出现了“*绝对值*”部分
*   **RMSE** 优化异常值(*均方根误差*)—惩罚较大的误差

因此，如果您的数据更符合正态分布并且没有异常值，您可以使用 MAE，而如果您的数据中有异常值，并且较大的错误对您的用例来说特别痛苦，您可以使用 RMSE。

现在我们知道了典型的损失函数是什么样子，我们可以看看分位数。

# 何时使用分位数，何时

![](img/0c9a8a7172cb8e43c4bd86491a78b1da.png)

马克西姆·霍普曼在[Unsplash](https://unsplash.com/s/photos/graph?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【2】上拍摄的照片。

术语分位数是百分位数的另一种说法，但以分数的形式出现。此外，例如，如果分位数值是 0.80，那么我们可以说，预测不足将被罚因子 0.80。反之，反过来也可以说是过度预测，它将被扣分 0.20。因此，过度预测比预测不足受到的惩罚要少。在这种情况下，我们会在 80%的时间里过度预测。

> 如果您的观察/实际计数更频繁地位于中值之上，这可能特别有用。

现在让我们开始有趣的部分——当这个函数实际上对您的业务或任何学术用例有用时。

假设我们有上述相同的 0-100 个实际观察范围的例子。如果中位数是 50，但更多的实际值落在 50 以上，比如说 60-80 比 20-40 多，那么我们应该使用更高的分位数 alpha 值。你可以测试不同的 alphas 值，但是你应该从高于 0.50 的值开始，否则你就违背了分位数损失的目的，并且在这种情况下实际上使用了 MAE。

> 为了进一步推动这一点，让我们总结两个简单的用例，它们几乎可以代表您将使用 quantile 做出的任何决策:

*   用例 1:

预测长途旅行的飞机价格。

如您所见，我们已经想要惩罚预测不足，因此我们将选择 0.50+的预测过度分位数，您可以从 0.55、0.60 等开始。仍然测试 0.50 作为基线比较可能是个好主意。很可能你的**数据向左**倾斜，你应该检查一下，最好是高估，因为过去的价格通常更接近最大范围而不是最小范围。例如，我们不会期望长途飞行通常是 10 美元(*，即使观察到的最低价格是 10 美元*)，而是期望它更接近于 200 美元。

*   用例 2:

预测干旱地区夏季的降雨量。

如果我们在一个更干燥的地区，例如，任何地方，现在是夏天，但我们想预测某一天的降雨量，我们可能会期望我们的实际值相对于我们的最大范围来说相当低，其中确实包含一些雷暴。在这种情况下，我们可能希望使用 0.45 或更低的 alpha，等等。，因为我们看到降雨量低的行的计数更频繁，所以我们想低估降雨量。

# 摘要

![](img/0686cdd8efbb25c52213fbf4ad219ffc.png)

爱德华·豪厄尔在[Unsplash](https://unsplash.com/s/photos/graph?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【3】上拍摄的照片。

正如你所看到的，损失函数没有一个放之四海而皆准的方法。这实际上取决于以下几点:

```
* Data* Distribution of that data* Business case* And how predictions will affect the business, is it better to overpredict or underpredict? Sometimes, it can even be more straightforward where you want one or the other regardless - without focusing on error itself, but focusing on tuning smaller or larger predictions for any reason
```

我希望你觉得我的文章既有趣又有用。如果您同意或不同意使用一个损失函数而不是另一个，请在下面随意评论。为什么或为什么不？你认为还有哪些损失函数需要进一步讨论？这些当然可以进一步澄清，但我希望我能够揭示一些数据科学损失函数及其应用。

***我不属于这些公司中的任何一家。***

*请随时查看我的个人资料、* [*Matt Przybyla*](https://medium.com/u/abe5272eafd9?source=post_page-----7b6107862d13--------------------------------) 、*和其他文章，并通过以下链接订阅接收我的博客的电子邮件通知，或通过点击屏幕上方的* ***图标关注图标*** *的订阅图标，如果您有任何问题或意见，请在 LinkedIn 上联系我。*

**订阅链接:**[https://datascience2.medium.com/subscribe](https://datascience2.medium.com/subscribe)

**引荐链接:**[https://datascience2.medium.com/membership](https://datascience2.medium.com/membership)

(*如果你在 Medium* 上注册会员，我会收到一笔佣金)

# 参考

[1]2021 年 [Unsplash](https://unsplash.com/s/photos/%25?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上[叶](https://unsplash.com/@josephyip?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)的照片

[2]马克西姆·霍普曼在 [Unsplash](https://unsplash.com/s/photos/graph?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片，(2021)

[3]爱德华·豪厄尔在 [Unsplash](https://unsplash.com/s/photos/graph?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片，(2020)