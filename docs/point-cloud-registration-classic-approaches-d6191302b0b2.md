# 点云配准:经典方法

> 原文：<https://towardsdatascience.com/point-cloud-registration-classic-approaches-d6191302b0b2>

## 迭代最近点(ICP)和相干点漂移(CPD)方法简介

![](img/ad83cb23fe84bd2cb3796e6ecbdd6da3.png)

[Ellen Qin](https://unsplash.com/@ellenqin?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

在我作为算法开发人员的工作中，我使用点云作为 3D 表示来解决几个问题。我的一些项目需要将两组点彼此对齐，这并不简单。

这个问题叫做**点云配准**。给定不同坐标系中的几组点，配准的目标是将它们彼此对齐。通过找到使它们之间的差异最小化的变换来完成对齐。

像许多计算机视觉任务一样，注册问题并不容易，并且具有许多挑战。例如，在某些真实场景中，点云具有不同的密度和有限的重叠。在其他情况下，点集可能是对称的或不完整的。这些例子影响了结果的准确性和效率。已经进行了广泛的研究来提高点云配准的准确性、效率和鲁棒性。

在这篇文章中，我将描述两种主要的经典和流行的建议方法。

# 迭代最近点(ICP)

给定两组点 P 和 Q，ICP 优化[刚性变换](https://en.wikipedia.org/wiki/Rigid_transformation)以将 P 与 Q 对齐:

![](img/761574a53ea38096230ebdbcfe5920f1.png)![](img/f2af071efba8ba1e72d3c8ac283f975a.png)

ICP 从初始对准开始，并在两个步骤之间迭代

*   **对应步骤** —为 p 中的每个点 *p_j* 寻找 Q 中最近的点 *q_i*
*   **对准步骤** —通过最小化对应点之间的 L2 距离来更新变换。

对于理解和实现来说，这是一个简单明了的算法。同时，它也有几个局限性。

## ICP 算法的局限性

*   结果的准确性强烈依赖于良好的初始对准。
*   倾向于收敛到局部最小值。
*   对异常值和部分重叠敏感。
*   仅适用于刚性变换。

在这一领域进行了大量的研究，并且针对 ICP 算法提出了许多变体来解决上述限制。

一种不同的注册方法是概率注册方法。这些方法使用对应关系的软分配，这意味着根据某种概率分配所有点之间的对应关系。这与使用二进制赋值的 ICP 算法相反。这种算法的一个例子是相干点漂移(CPD)。

# 相干点漂移

相干点漂移是一种用于配准问题的概率方法，它对刚性和非刚性变换都有效。正如作者在原始论文中解释的那样:

> 两个点集的对齐被认为是一个概率密度估计问题。一个点集代表[高斯混合模型](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model) (GMM)质心，另一个点集代表数据点。通过最大化似然性将 GMM 质心拟合到数据来完成配准

与 ICP 相比，CPD 对噪声、异常值和缺失点更具鲁棒性。

## CPD 算法的局限性

*   使用关于噪声和异常值数量的假设。不正确的假设将导致注册不良。
*   对于具有密度变化的数据，配准趋向于收敛到局部极值(局部最小值或局部最大值)

# 摘要

注册任务是计算机视觉领域中的一个热门任务。大多数方法依赖于寻找两点云之间的对应关系。提出了两种主要的方法——像 ICP 中使用的硬对应，以及像 CPD 中使用的对应的软分配。每种方法都有许多不同的变体，各有利弊。

哪个最适合你？这取决于您的数据、您是需要刚性方法还是非刚性方法、噪声和异常值的数量、密度变化等等。正如我之前提到的，这些方法有很多变体。我建议找到最适合您的任务和数据的一个。

## 参考

这是我在这篇文章中使用的论文列表。您可以阅读它们以了解更多信息。

[](https://ieeexplore.ieee.org/abstract/document/8490968) [## [1]迭代全局相似点:用于成对 3D 的稳健的粗到细整合解决方案…

### 在本文中，我们提出了一种由粗到细的积分方法，该方法受经典 ICP 算法的启发，用于两两三维物体的积分

ieeexplore.ieee.org](https://ieeexplore.ieee.org/abstract/document/8490968) [](https://ieeexplore.ieee.org/abstract/document/9336308) [## [2]快速和稳健的迭代最近点

### 迭代最近点(ICP)算法和它的变种是一个基本的技术之间的严格注册…

ieeexplore.ieee.org](https://ieeexplore.ieee.org/abstract/document/9336308) [](https://ieeexplore.ieee.org/abstract/document/5432191) [## [3]点集配准:相干点漂移

### 点集配准是许多计算机视觉任务中的关键组成部分。点集配准的目标是…

ieeexplore.ieee.org](https://ieeexplore.ieee.org/abstract/document/5432191) [](https://link.springer.com/article/10.1007/s11432-011-4465-7) [## [4]一种用于点集配准的精细相干点漂移(CPD)算法-科学中国…

### 相干点漂移(CPD)算法是一种强有力的点集配准方法。然而，它有一个缺点…

link.springer.com](https://link.springer.com/article/10.1007/s11432-011-4465-7) [](https://ieeexplore.ieee.org/abstract/document/8897021) [## [5]基于相干点漂移的 TLS 点云的密度自适应和几何感知配准

### 概率配准算法[例如，相干点漂移(CPD)]为点云提供了有效的解决方案

ieeexplore.ieee.org](https://ieeexplore.ieee.org/abstract/document/8897021)