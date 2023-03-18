# 永远不要忘记。在 Python 代码中复制()

> 原文：<https://towardsdatascience.com/never-forget-copy-in-your-python-codes-9826bba31f78>

## 如果您记得使用，它将为您节省大量调试时间。始终在 python 代码中复制()

![](img/59f6296c506743e4a9029ad36c9d6bff.png)

布鲁斯·马斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

周六下午，我本应该开车带孩子去购物中心，但我却被困在了我的家庭办公室的办公桌前，调试着一段简单的代码，这段代码从上周五开始就一直困扰着我。

严格地说，这不是“调试”，而是为了解决我的代码的一些奇怪的性能。我正在以迭代的方式实现一个简单的无监督模型，并试图根据我设计的特定性能指标记录“最佳”模型。

让我惊讶的是，我保存的“最佳模型”并不是最好的，并且一直记录着我迭代中的最后一个模型。下面是我的实现的 sudo 代码的简化版本:

```
iter = 0
while iter < 10: iter += 1    
    # initiate the performance and empty data for recording
    best_model = None
    best_performance = 0
    best_data = None # modeling
    M = sklearn.unsupervised(random_seed= seed_dict[iter])
    M.fit(X)
    X2.predict = M.predict(X2)
    current_performance = sum(X2.predict == 1)
    print("my current performance is %i" %(current_performance)) # compare to the recorded best model
    if current_performance > best_performance:
        best_model = M
        best_data = X2
        best_performance = current_performance
        print("Ah ha!! My best model is updated with performance %i !" %(best_performance))
```

如果你是一个有经验的程序员，你可能已经发现了代码的问题。如果您还没有，请跟随我学习下面的示例输出代码，

```
# Python
print("My best performance is %i based on best_data!" %(sum(best_data.predict == 1)))
print("My best performance is %i based on best_performance!" %(best_performance))my current performance is 300
Ah ha!! My best model is updated with performance 300 !
my current performance is 400
Ah ha!! My best model is updated with performance 400 !
my current performance is 200
my current performance is 270
my current performance is 360
my current performance is 870
Ah ha!! My best model is updated with performance 870 !
my current performance is 1224
Ah ha!! My best model is updated with performance 1224 !
my current performance is 1687
Ah ha!! My best model is updated with performance 1687 !
my current performance is 350
my current performance is 128My best performance is 128 based on best_data!
My best performance is 1687 based on best_performance!
```

看到问题了吗？我根据记录的“最佳数据”报告的“最佳性能”是来自上一次迭代。然而，基于变量“最佳性能”，真正的“最佳性能”应该是 1，687。

我的原始代码在建模部分更复杂，所以我在建模部分花了很多时间，甚至没有考虑记录最佳模型和数据的代码。

经过几轮编辑(那花费了我一个周五晚上外加一个周六下午！)，我终于注意到下面这行代码，

```
best_data = X2
```

应该是，

```
best_data = X2.copy()
```

原因是 Python 中的赋值语句不复制对象，它们在目标和对象之间创建绑定。因此，当我键入 best_data = X2 时，我实际上在两个变量之间建立了一个链接，每次 X2 发生变化，best_data 也会发生变化！

根据 Python 的文档，“对于可变的或包含可变项的集合，有时需要一个副本，这样可以在不改变另一个副本的情况下改变一个副本。”

在编码中很容易忽略这一点，尤其是当你把主要精力放在建模部分的时候。

我不想让你像我一样在这个愚蠢的错误上浪费宝贵的周六下午，所以我把它写成了一个简短的帖子。希望有帮助！

我要回到我的家人身边！

![](img/29c0aa035a596662f5628eb862574768.png)

由[杰德·维尔霍](https://unsplash.com/@jmvillejo?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片