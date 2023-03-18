# 为机器学习和软件测试生成合成数据集

> 原文：<https://towardsdatascience.com/generating-a-synthetic-dataset-for-machine-learning-and-software-testing-7bebaed69386>

## 使用 Python 生成统计上相似的虚拟数据集，用于代码开发和测试健壮性

![](img/a81b03d2d703dbed8fadd8b340bfe77b.png)

照片由[亨利·佩克斯](https://unsplash.com/@hjkp?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

# 据此

为测试生成真实的数据集是一个突出的问题。这在两个领域之一经常遇到:机器学习或软件工程。

## 创建更可靠的模型

在机器学习中，过拟合往往是一个非常突出的问题。如果一个模型过拟合一个小的/有限的/不平衡的数据集，那么它对未知数据产生准确预测的可靠性就会大大降低。对此的一个解决方案是使用合成数据集来扩展“训练”数据集，合成数据集是自动生成的数据集，具有与原始数据集相同的形式、大小和分布。

## 看不见或不可用的数据

类似地，如果我们正在编写一个应用程序来读取、查询或可视化数据(这还不可用)，我们可能希望通过使用一个伪真实的虚拟数据集来测试它的健壮性。通过这样做，我们能够在数据的最新迭代被处理之前就开始开发全新的应用程序。

# 数据预测

虽然生成数据的方法很多，例如缩放、变换、标准化、添加噪声、混洗等。我们感兴趣的是保持形式，同时产生一个新的扰动样本集。为了做到这一点，我们依赖于 copulas——与传统的相关性计算相比，copulas 允许我们更好地捕捉诸如非线性尾部相关性等细微差别。

## 什么是系词？

联合分布解释了我们的`n`随机变量(向量)是如何相互关联的。它由两件事组成:

*   边际:对边际行为的描述(每个分布的边缘是什么)
*   每个变量的边际(尾部)概率分布均匀的多元累积分布函数，赢得区间`[0,1]`。这是系词。

copula 描述了数据集中所有随机变量或组件之间的相关性结构。

# 将它们全部编码成 Python

## 装置

为了生成我们的合成数据集，我们使用了 [Synthia](https://dmey.github.io/synthia/index.html) 包。这可以与以下设备一起安装:

```
pip install synthia
```

## 加载和清理数据

我们首先加载数据，并提取数值列的子集，用于数据生成器。在这里，我还将任何`nan (*not a number)*`项替换为零——这是因为我希望应用高斯连接函数，对于该函数，线性代数(sqrtm)计算需要对矩阵求逆。

*如果使用独立系词(参见“代码”部分的片段),则不需要这一步。*

```
*# Load the original data* 
data  = pd.read_csv(filename, index_col=0) *# Get file datatypes* 
dtypes = data.dtypes *# Get the names of the columns with numeric types   * 
numeric = data.columns[dtypes.apply(pd.api.types.is_numeric_dtype)] *# Extract numeric subset* 
subset = data.loc[:,numeric].replace(np.nan, 0)
```

## 构建数据生成器

使用 Synthia 包，我们创建了一个 copula 数据生成器。利用这一点，我们拟合高斯连接函数，并使用以下代码将数据参数化为分位数:

```
*# Create Generator  *  
generator = syn.CopulaDataGenerator() *# Define Coupla and Parameterizer *   
parameterizer = syn.QuantileParameterizer(n_quantiles=100) generator.fit(subset, copula=syn.GaussianCopula(), parameterize_by=parameterizer) 
```

## 创建新数据集

现在我们有了一个生成器，我们可以用它来创建一个与原始数据集大小相同的样本数据集。在 ML 任务的情况下，我们可能希望将样本量增加到一个任意大的数字。

```
samples = generator.generate(n_samples=len(subset), uniformization_ratio=0, stretch_factor=1) synthetic = pd.DataFrame(samples, columns = subset.columns, index = subset.index) 
```

最后，我们可以复制原始数据帧，并用新生成的值替换这些值:

```
update = data.loc[:]
update.loc[:,numeric]= synthetic.loc[:,numeric]    
update = update.astype(dtypes) 
```

# 结论

我们现在能够使用 pandas 和 Syntia 包创建人工数据集，并在旧数据集的位置注入它们。这有许多应用，并不局限于取代测试的机密信息，在数据集可用之前进行软件开发，或扩展训练数据集作为提高人工智能性能的手段。

# 密码

这个项目中使用的脚本附后。

# 资源

关于系词的更多信息可以在[这里](https://medium.com/kxytechnologies/a-primer-on-copulas-from-a-machine-learning-perspective-b9ea11c8681b)和[这里](https://www.youtube.com/watch?v=WFEzkoK7tsE)找到。