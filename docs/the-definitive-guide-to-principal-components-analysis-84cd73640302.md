# 主成分分析权威指南

> 原文：<https://towardsdatascience.com/the-definitive-guide-to-principal-components-analysis-84cd73640302>

## 这是一个教程，它剥离了底层代码，您可以在浏览器中编辑和运行这些代码，以便一劳永逸地理解 PCA 的细节。还可以找到一个完全成熟的 web 应用程序供您使用。

尽管有数百个网页致力于主成分分析(PCA)，但我没有找到一个完整和详细的关于它实际上是如何计算的。有些教程擅长解释如何获得协方差矩阵，有些则擅长描述所涉及的奇异值分解，还有一些只是简单地介绍现成的程序或代码片段，但是你不知道为什么和如何。另一方面，一些资源非常深入，但是使用的技术术语最终会让你更加困惑。还有一些人专注于主成分分析的输入和输出，但完全忽略了黑盒内部发生的事情。

这就是为什么我写了这个新教程，它剥离了我最近构建的在线运行 PCA 的 web 应用程序的底层代码:

[](/a-free-online-tool-for-principal-components-analysis-with-full-graphical-output-c9b3725b4f98)  

我所说的“低级”是指除了一个取自库的函数之外，其余的代码都是简单的操作和循环。

由于这是从 web 应用程序中获取的，所以涉及的代码是 JavaScript。但是它非常简单，正如您将看到的，因此您应该能够轻松地将其翻译成其他编程语言。JavaScript 的好的一面是，你可以很容易地编辑它，并在你的浏览器中测试它，自己探索 PCA 是如何工作的，并且一劳永逸地理解它的细节。

# 简单地说，主成分分析

PCA 是一种用于减少数据集中的维数，同时保留其中最重要的信息的技术。主成分分析通过将高维数据线性投影到其变异的主要成分(称为主成分(PC))上来实现这一点。

PCA 可用于识别数据集的底层结构或降低其维数。在实践中，您使用 PCA 来简化复杂数据的显示，方法是仅在 1、2 或 3 维(最常见的是在 2 维，以二维图的形式)上展开数据；找出哪些原始变量对数据集中的可变性贡献最大；还可以通过从其主 PC 重建数据集并忽略包含更多噪声的那些来去除噪声或压缩数据(例如图像);以及可能的一些其他相关应用。你可以在我介绍 web 应用的[文章中看到一些具体的例子。](/a-free-online-tool-for-principal-components-analysis-with-full-graphical-output-c9b3725b4f98)

# 保持简单但完整的数学

有些资源是完整的，但是太复杂了，难以理解，而有些资源很简单，但是那是因为它们没有展示出全貌。例如，一些教程更多地关注于计算协方差矩阵，但是对于如何从该矩阵到 PCA 的实际结果，你却被抛在了一边。其他教程更深入地研究了这个方面，但在这个过程中，他们认为协方差矩阵是理所当然的，并且经常使用不必要的复杂术语。

这里我将从起始数据集开始，它是一列由*变量*描述的*no objects*对象。例如，每个对象可以是在宽波长范围内为一个样本收集的光谱；每个波长的读数将是一个变量。哦，是的，在文本中，我会像在代码中一样调用所有变量，用斜体。

## 步骤 1:从起始数据集到协方差矩阵

现在输入已经很清楚了，是时候写代码了。起始数据集在下面的代码中，该代码包含在名为 *X* 的变量中，该变量有*个变量*行和*个对象*列。

在最常见的形式中，正如我们将在这里运行的，PCA 是协方差矩阵的线性分解。让我们来计算一下。我们首先需要通过减去每个变量的平均值来对所有输入进行均值居中。为此，我们首先计算所有变量的平均值(通过将所有对象相加，然后除以它们的数量)，然后减去平均值。注意，最后我们没有创建一个新的变量；相反，起始矩阵 *X* 的元素被以平均值为中心的值代替:

```
//This first block computes each variable's average
averages=[]
for (i=0;i<nvariables;i++) {
  var tmp2=0
  for (j=0;j<nobjects;j++) {
    tmp2=tmp2+X[i][j]
  }
  averages.push(tmp2/nobjects)
}//This block subtracts each reading by the average of the corresponding variable
for (i=0;i<nvariables;i++) {
  for (j=0;j<nobjects;j++) {
    X[i][j] = X[i][j] — averages[i]
  }
}
```

现在我们已经减去了平均值，我们可以计算协方差矩阵。到目前为止，包括协方差矩阵的计算，在这篇外部博客中已经很清楚了:

  

下面是底层代码:

```
//Here we just setup a covariance matrix full of 
var covar = []
for (i=0;i<nvariables;i++) {
  var tmp3 = []
  for (j=0;j<nvariables;j++) {
    tmp3.push(0)
  }
  covar.push(tmp3)
}//This block fills in the covariance matrix by increasingly summing the products
for (i=0;i<nvariables;i++) {
  for (j=0;j<nvariables;j++) {
    for (k=0;k<nobjects;k++) {
      covar[i][j] = covar[i][j] + X[i][k]*X[j][k]
    }
    covar[i][j] = covar[i][j] / (nobjects-1)   //Here can be -0
  }
}
```

请注意，我们需要三个索引，因为我们需要计算每个变量和所有其他变量之间的协方差，并且对于这些计算中的每一个，我们需要遍历数据集中的所有对象。

还要注意协方差计算过程中间的注释(“这里可以是-0”)。最正确的是将乘积之和除以*no objects*-1，如下图所示；然而，一些例子使用了常规方差，其中总和只除以*no objects。*

## 步骤 2:对角化协方差矩阵

我们现在需要通过奇异值分解(SVD)对角化协方差矩阵。

这是这个过程中唯一的一点，我宁愿使用一个健壮的、经过良好测试的、正确编码的、高效的矩阵代数库，而不是在底层编码所有的东西。在这种情况下，我使用了 Lalolib 包中用于浏览器内数学的一个过程，我在最近的一篇文章中介绍了这个过程:

[](/websites-for-statistics-and-data-analysis-on-every-device-ebf92bec3e53)  

请注意，要使用 Lalolib，您需要在 HTML 中找到它，如下所示:

```
<script type="application/x-javascript" src="lalolib.js"> </script>
```

一旦完成，Lalolib 的所有函数都暴露在您的 JavaScript 中。

现在，在我们对协方差矩阵运行 SVD 之前，我们需要将这个矩阵转换成 Lalolib 可以处理的对象类型。这是因为该库使用自己的数据类型，而不是像我们的 *covar* matrix 这样的原始 JavaScript 数组。

因此，在下一段代码中，您将看到 *covar* 矩阵被转换为 Lalolib 可以处理的矩阵，称为 *matrixforsvd* ，然后您将看到对该矩阵上的 svd 例程的实际调用。如此处所示，您可以将该过程的四个主要输出记录到控制台:

```
var matrixforsvd = array2mat(covar);

var svdcalc = svd(matrixforsvd, “full”);console.log(svdcalc.U)
console.log(svdcalc.S)
console.log(svdcalc.V)
console.log(svdcalc.s)
```

**这些输出是什么？**前三个是矩阵。当您对协方差矩阵运行 SVD 时，您会看到 *U* 和 *V* 是相同的，因为这是一个对称矩阵(您可以记录 *covar* 来自己检查这一点)。 *U* 和 *V* 具有负载，它们是定义如何构建主成分空间以最优地展开数据集变化的系数。

*s* (小写相对于大写的 *S* 矩阵)是从 S 矩阵中提取的对角线。原来 *S* 矩阵的所有非对角元素都是零，所以 *s* 向量更简单，包含的信息量也是一样的。该信息包括每个主成分对数据集中总变化的贡献。它们总是按降序排列，你可以通过在我的 web 应用程序上运行示例来验证这一点。如果你将这些数字标准化为它们的总和，你会得到由每个主成分解释的%变化。

## 步骤 3:将数据投影到主成分上

这是 PCA 过程的最后一部分，也是大多数教程经常很快通过的部分。本质上，我们需要做的是将包含起始数据的矩阵(以均值为中心，所以这是我们的 X)乘以通过协方差矩阵的 SVD 获得的 U 矩阵。

因为我在底层编写了这个乘法，所以我首先必须转换 *svdcalc。U* 放入一个 JavaScript 数组。你会看到这个用*完成。toArray()* 。紧接着，两个嵌套的 for 循环执行乘法。

```
var projection = []
var pc1 = 0, pc2 = 0
var U = svdcalc.U.toArray()

for (i=0; i<nobjects; i++) {
  pc1=0
  pc2=0
  for (j=0;j<nvariables;j++) {
    pc1 = pc1 + U[j][0] * X[j][i]
    pc2 = pc2 + U[j][1] * X[j][i] 
  }
  var tmp4 = []
  tmp4.push(pc1)
  tmp4.push(pc2)
  projection.push(tmp4)
}
```

现在，最后一个变量，*投影*，包含*对象*行和 2 列，这是两个第一主成分。PC 图是所有这些 PC1、PC2 对的散点图。

# 结束语

## 在线运行 PCA，通过示例了解它

对于在线运行 PCA 的实际 web 应用程序，包括图形输出，以及关于如何解释结果的解释，请查看另一个故事:

[](/a-free-online-tool-for-principal-components-analysis-with-full-graphical-output-c9b3725b4f98)  

## 尝试一下

将代码片段复制到一个 HTML 页面中，不要忘记获取 Lalolib 库，用一些数据创建一个 X 矩阵，并通过使用 console.log()或 document.write()显示中间结果和最终结果来测试代码。

我希望这篇教程能让你更清楚地了解 PCA 是如何运行的，以及它背后的数学原理。如果本文的某些部分仍然不清楚，那么通过突出显示和评论来表明这一点，我会看看我能做什么！

祝你快乐！

www.lucianoabriata.com*[***我写作并拍摄我广泛兴趣范围内的一切事物:自然、科学、技术、编程等等。***](https://www.lucianoabriata.com/) **[***成为媒介会员***](https://lucianosphere.medium.com/membership) *访问其所有故事(我为其获得小额收入的平台的附属链接无需您付费)和* [***订阅获取我的新故事***](https://lucianosphere.medium.com/subscribe) ***通过电子邮件*** *。到* ***咨询关于小职位*** *查看我的* [***服务页面这里***](https://lucianoabriata.altervista.org/services/index.html) *。你可以* [***这里联系我***](https://lucianoabriata.altervista.org/office/contact.html) ***。******