# 算法解释#3:搜索

> 原文：<https://towardsdatascience.com/algorithms-explained-3-searching-84604e465838>

## 通过 Python 中的线性和二分搜索法示例了解搜索

![](img/8203ac1a48177252cf07b3c8700dff22.png)

图片来自 [Pixabay](http://pixabay.com) 的 [Clker-Free-Vector-Images](https://pixabay.com/users/clker-free-vector-images-3736/)

搜索算法对于检索存储在数据结构中的一个或多个元素是有用的。搜索算法适用的一些问题示例包括检索数据库中的特定记录或搜索文章中的关键字。

搜索算法通常分为两种类型:

1.  **顺序搜索:**顺序搜索算法可以在有序或无序的数据结构上执行。该算法遍历数据结构并按顺序检查每个元素，直到找到所需的元素。线性搜索就是一个例子。
2.  **区间搜索:**要执行这种类型的搜索，底层数据结构必须已经排序。因此，给定一个排序的数据结构，区间搜索算法递归地在中点将搜索空间分成两半，直到找到想要的元素，因此它比顺序搜索算法更快更有效。二分搜索法就是一个例子。

## 线性搜索

给定一个已排序或未排序的数组，线性搜索从索引 0 开始，遍历数组中的每个元素，直到找到目标项并返回其索引。如果没有找到目标元素，返回-1。

该算法的时间复杂度为 O(n)。这是因为在最坏的情况下，目标项是数组中的最后一个元素，我们将不得不遍历整个数组中的每一项，导致 *n* 次比较。

```
def linear_search(arr, target):
   for idx, num in enumerate(arr):
      if num == target:
         return idx
   return -1
```

## 二进位检索

给定一个已排序的数组，二分搜索法的工作方式是在数组的中点将数组一分为二，并将目标元素与数组中的中间元素进行比较。如果目标元素小于中间的元素，则继续搜索数组的左半部分。否则，如果元素大于中间的元素，则继续搜索数组的右半部分。重复这个过程，直到目标元素等于中间元素，并返回它的索引。如果没有找到目标元素，返回-1。

二分搜索法算法的时间复杂度是 O(log n ),因为我们每次迭代都将搜索空间分成两半。

```
def binary_search(arr, start, end, target):
   while start <= end:
      midpoint = start + (end - start) // 2
      mid_elem = arr[midpoint] # if target is greater, ignore the left half
      if target > mid_elem:
         binary_search(arr=arr, start=midpoint+1, end=end, target=target) # if target is smaller, ignore the right half
      elif target < mid_elem:
         binary_search(arr=arr, start=start, end=midpoint-1, target=target) # otherwise target is equal to midpoint
      else:
         return midpoint return -1
```

## 结论

在本文中，我们讨论了两种重要的搜索算法——顺序搜索和区间搜索。线性搜索是时间复杂度为 O(n)的顺序搜索的经典示例，而二分搜索法是时间复杂度为 O(log N)的更高效的区间搜索的常用示例。

更多请看本算法讲解系列: [#1:递归](/algorithms-explained-1-recursion-f101500f9316)、 [#2:排序](/algorithms-explained-2-sorting-18d0875528fb)、 [#3:搜索](/algorithms-explained-3-searching-84604e465838)(本期文章)、 [#4:贪婪算法](/algorithms-explained-4-greedy-algorithms-f60792046d40)、 [#5:动态规划](/algorithms-explained-5-dynamic-programming-e5472a4ce464)、 [#6:树遍历](/algorithms-explained-6-tree-traversal-1a006ba00672)。