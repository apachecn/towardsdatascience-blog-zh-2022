# 算法解释#5:动态编程

> 原文：<https://towardsdatascience.com/algorithms-explained-5-dynamic-programming-e5472a4ce464>

## 用 Python 中离散背包和最长公共子序列问题的示例解决方案解释动态编程

![](img/b1a96a2d28faced06d726c643f84e7a9.png)

图片由[穆罕默德·哈桑](https://pixabay.com/users/mohamed_hassan-5229782/)来自 [Pixabay](http://pixabay.com)

动态规划对于解决具有重叠子问题和最优子结构的问题是有用的。在上一篇关于贪婪算法的文章中，我们谈到了贪婪的选择或者在每个决策点选择最佳的下一个选择有时会产生局部最优的选择。在这些情况下，我们可以使用动态规划来克服这个问题，更有效地解决这些优化问题。

动态规划对于表现出以下两个特征的问题是有效的:

1.  *最优子结构:*组合子问题的最优解，产生全局最优解。
2.  *重叠问题*:多次求解同一个子问题可以找到最优解。

换句话说，动态规划是一种适用于问题的优化方法，通过将其分解为更小的子问题，找到这些子问题的最优解，并将最优解组合以产生全局最优解，可以最优地解决这些问题。为了提高算法的效率，通常使用**记忆**来重用和跟踪已经在一些数据结构中评估过的先前项目。

使用动态规划解决问题的步骤包括:I)定义子问题，ii)寻找递归，iii)求解基本案例。在接下来的几节中，我们将介绍一些可以用动态编程解决的例子。

在本文中，我们将介绍两个优化问题，以及如何使用 Python 中的动态编程来解决它们。

## 离散背包问题

在上一篇关于贪婪算法的文章中，我们使用贪婪算法实现了分数背包问题的解决方案。然而，对于离散背包问题，贪婪算法并不总是导致使用贪婪算法的全局最优解，因为项目是不可分的。相反，我们将探索如何使用动态规划来解决离散背包问题。

提醒一下，背包问题的前提是一个窃贼正在抢劫一家有 n 件物品的商店，每件物品的价值是美元，重量是磅。窃贼想最大化被盗物品的价值，但他/她的背包只能装最大重量的物品，他/她应该带什么物品？

我们可以把这个问题分解成:

1.  *子问题:*构造一个矩阵 M 来跟踪结果，并通过用重量限制为 j 的物品 *i[0…i]* 填充背包来定义 M[i，j]为最大值
2.  *递归:*递归可以分为以下两种情况之一:I)如果第 I 个项目的权重 *w[i]* 在第 j 个单元格小于或等于我们背包的剩余容量，那么我们可以选择添加第 I 个项目；ii)如果在第 j 个单元格中 *w[i]* 的重量超过了背包的剩余容量，那么我们不添加它，重量保持不变。
3.  *基本情况:*初始化 *M[i，0]* 和 *M[0，j]* 为 0，因为背包将从权重 0 开始。

离散背包问题的动态规划解的时间复杂度是 O(nW)，其中 *n* 是物品的数量，而 *W* 是背包的容量或重量限制。这是因为我们首先遍历物品的数量，然后遍历背包中允许的总重量，以找到这个问题的最优解。以下是离散背包问题的动态编程解决方案的 Python 实现:

```
def fill_knapsack_discrete(W, values, weights):
   """Function to find maximum value to fill knapsack 
      whereby items are not divisible.

   Parameters:
      W (int): maximum weight of knapsack.
      values (list): list of item values.
      weights (list): list of item weights.

   Returns: 
   int: value of filled knapsack.
   """
   # Initialize matrix
   num_items = len(values)
   M = [[None] * (W + 1) for i in range(num_items + 1)] for i in range(num_items + 1):
     for j in range(W + 1):
        # Base case
        if i == 0 or j == 0:
           M[i][j] = 0
        # Recurrence if weight of item is less than or 
        # equal to the remaining capacity of knapsack
        elif weights[i - 1] <= j:
           M[i][j] = max(
              values[i - 1] + M[i - 1][j - weights[i - 1]], 
              M[i - 1][j]
           )
        # Recurrence if weight of item is more than 
        # remaining capacity of knapsack
        else:
           M[i][j] = M[i - 1][j] return M[num_items][W]
```

## 最长公共子序列(LCS)问题

这个问题如下:给定两个字符串 *a* 和 *b* ，求两个字符串中存在的最长公共子序列(LCS)的长度。请注意，子序列是以相同的相对顺序出现的序列，但它可能不连续。

在这种情况下，我们可以将问题分解为:

1.  *子问题:*构造一个矩阵 L 来跟踪结果，并将 *L[i，j]* 定义为 *a[0…i]* 和 *b[0…j]的最长公共子序列的长度。*
2.  *递归:*我们可以把递归分成以下两种情况之一:I)如果 *a[i]* 等于 *b[j]* ，那么它们都对 LCS 有贡献，我们可以把结果加 1，所以 *L[i，j] = L[i-1，j-1]+1；* ii)如果 *a[i]* 不等于 *b[j]* ，那么没有匹配，可以丢弃一个所以 *L[i，j] = max(L[i-1，j]，L[i，j-1])。*
3.  *基本情况:*初始化 *L[i，0]* 和 *L[0 j]* 为 0。

这个问题的动态编程解决方案的时间复杂度是 O(nm)，因为我们首先遍历长度为 *n* 的第一个字符串 *a* ，然后遍历长度为 *m* 的第二个字符串 *b* 。下面是 Python 中的实现:

```
def find_lcs(a, b):
   # Find length of strings
   len_a = len(a)
   len_b = len(b) # Initialize matrix
   L = [[None] * (len_b + 1) for i in range(len_a + 1)] # Loop through strings and record the length of LCS at
   # each a[0...i] and b[0...j]
   for i in range(len_a + 1):
      for j in range(len_b + 1):
         # Base case
         if i == 0 or j == 0:
            L[i][j] = 0
         # Recurrence if characters match
         elif a[i - 1] == b[j - 1]:
            L[i][j] = L[i - 1][j - 1] + 1
         # Recurrence if characters do not match
         else:
            L[i][j] = max(L[i - 1][j], L[i][j - 1]) return L[len_a][len_b]
```

## 结论

动态规划是解决最优化问题的一个非常有用的工具。实现动态规划算法的步骤包括将问题分解成子问题，识别其重现和基本情况以及如何解决它们。

更多请看本算法讲解系列: [#1:递归](/algorithms-explained-1-recursion-f101500f9316)、 [#2:排序](/algorithms-explained-2-sorting-18d0875528fb)、 [#3:搜索](/algorithms-explained-3-searching-84604e465838)、 [#4:贪婪算法](/algorithms-explained-4-greedy-algorithms-f60792046d40)、 [#5:动态规划](/algorithms-explained-5-dynamic-programming-e5472a4ce464)(本期文章) [#6:树遍历](/algorithms-explained-6-tree-traversal-1a006ba00672)。