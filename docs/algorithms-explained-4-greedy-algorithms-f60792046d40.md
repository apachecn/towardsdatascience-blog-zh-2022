# 算法解释#4:贪婪算法

> 原文：<https://towardsdatascience.com/algorithms-explained-4-greedy-algorithms-f60792046d40>

## 对贪婪算法的解释以及用 Python 实现的分数背包和硬币兑换问题的解决方案

![](img/c4f808801b5f48844e9b2ea8675c596c.png)

图片来自 [Pixabay](http://pixabay.com) 的 [Clker-Free-Vector-Images](https://pixabay.com/users/clker-free-vector-images-3736/)

贪婪算法通常是一种实用的方式来找到一个体面和优雅的，虽然不总是最优的，优化问题的解决方案。它的工作原理是做出一系列选择，并在每个决策点总是选择下一个最佳选择。一类这样的问题是活动选择问题，我们试图在各种竞争活动中安排资源。

贪婪算法通常具有以下两个属性:

1.  *贪婪选择性质:*选择局部最优(贪婪)选择导致全局最优解，意味着每一步的贪婪选择都会产生全局最优解。
2.  *最优子结构:*整个问题的最优解也包括子问题的最优解。换句话说，最优解与第一步是一致的，这种解决子问题的一般策略也可以用来解决整个问题。

在接下来的几节中，我们将介绍两个可以用贪婪算法解决的经典例子。

**背包问题**

背包问题是优化问题的一个典型例子，其中的前提是一个窃贼正在抢劫一家有 n 件物品的商店，每件物品的价值为美元，重量为磅。窃贼想要最大化被盗物品的价值，但是他们的背包只能装最大重量的物品，他们应该拿什么物品呢？

这个问题有两个版本:

*   0–1 背包问题:这也叫做离散背包问题，因为物品要么被拿走，要么不被拿走。对于这个问题，贪婪解产生局部最优解，但不总是全局最优解，因为项目是不可分的。因此，我们将在下一篇文章中研究如何使用动态编程找到这个问题的最优解。
*   *分数背包问题:*这也叫连续背包问题，因为窃贼可以拿走一个物品的分数。对于背包问题的这种变体，贪婪算法将总是产生最优解。

要解决分数背包问题，首先计算每件物品的每磅价值( *v_i/ w_i* )。然后，在每个决策点，做出贪婪的选择，小偷尽可能多地拿走每磅价值最高的物品，直到他们达到最大重量 *W* ，背包满了。

分数背包问题的时间复杂度是 O(n log n)，因为我们必须根据每磅的价值对物品进行排序。下面是一个贪婪算法在 Python 中对此问题的实现:

```
def fill_knapsack_fractional(W, values, weights):
   """Function to find maximum value to fill knapsack.

   Parameters:
   W (int): maximum weight of knapsack .
   values (dict): dictionary with item name as key 
      and value of item as value.
   weights (dict): dictionary with item name as key 
      and weight of item as value.

   Returns: 
   int: value of filled knapsack.
   """
   knapsack_value = 0
   W_remaining = W
   items_taken = {}
   value_per_pound = {} # Calculate value per pound for each item
   for item, v_i in values.items():
     w_i = weights.get(item)
     value_per_pound[item] = v_i / w_i # Sort items by value per pound
   items_sorted = dict(sorted(value_per_pound.items(), key=lambda x: x[1],  reverse=True))

   # Add items to knapsack
   for item, value_per_pound in items_sorted.items():
     v_i = values.get(item)
     w_i = weights.get(item)
     if W_remaining - w_i >= 0:
        knapsack_value += v_i
        W_remaining -= w_i
     else:
        fraction = W_remaining / w_i
        knapsack_value += v_i * fraction
        W_remaining -= int(w_i * fraction) return knapsack_value
```

## 硬币兑换问题

硬币兑换问题的目标是在给定一组具有不同支配的有限硬币的情况下，找到总数达到期望数量的最少硬币。

为了使用贪婪算法解决这个问题，我们首先按降序排列硬币。然后，我们总是会找到最大的硬币支配，小于或等于我们剩余的数量，并重复这个过程，直到我们达到我们想要的数量。如果用给定的一组硬币无法达到期望的数量，则返回-1。

请注意，虽然您可以用贪婪算法解决硬币兑换问题，但您可能并不总能找到最佳解决方案。为此，建议使用动态编程——在[的下一篇文章](/algorithms-explained-5-dynamic-programming-e5472a4ce464)中有所涉及。(感谢[查德张](https://medium.com/u/8db76ec5f722?source=post_page-----f60792046d40--------------------------------)指出这一点！)

这个算法的时间复杂度是 O(n log n)，因为硬币列表是按支配从大到小排序的。下面是 Python 的实现:

```
def get_change(amount, coins):
   """Function to find the desired amount with the 
   smallest number of coins given a finite set of coins. Parameters:
   amount (int): desired amount to reach.
   coins (list): list of coin dominations. Returns:
   list: list of coins chosen
   """
   # Sort coins from largest to smallest domination
   coins_sorted = sorted(coins, reverse=True)
   change = [] for coin in coins_sorted:
       if coin <= amount: 
          change.append(coin)
          amount -= coin
          if amount == 0:
             return change

   return -1
```

## 结论

贪婪算法有许多应用，本文中我们介绍了两个例子——分数背包问题和硬币兑换问题。在贪婪算法失败的情况下，即局部最优解不会导致全局最优解，更好的方法可能是动态规划(接下来)。

更多请看本算法讲解系列: [#1:递归](/algorithms-explained-1-recursion-f101500f9316)、 [#2:排序](/algorithms-explained-2-sorting-18d0875528fb)、 [#3:搜索](/algorithms-explained-3-searching-84604e465838)、 [#4:贪婪算法](/algorithms-explained-4-greedy-algorithms-f60792046d40)(本期文章)、 [#5:动态编程](/algorithms-explained-5-dynamic-programming-e5472a4ce464)、 [#6:树遍历](/algorithms-explained-6-tree-traversal-1a006ba00672)。