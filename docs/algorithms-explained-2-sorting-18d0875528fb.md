# 算法解释#2:排序

> 原文：<https://towardsdatascience.com/algorithms-explained-2-sorting-18d0875528fb>

## 三种排序算法的解释及其在 Python 中的实现

![](img/716d3b85bcb9c3d44d3eee5559ab07d5.png)

图片由 [200 度](https://pixabay.com/users/200degrees-2051452/)来自 [Pixabay](http://pixabay.com)

在前一篇文章中，我介绍了递归，我们将在这篇文章的基础上构建排序算法。排序算法用于重新排列数组中的元素，使每个元素都大于或等于其前一个元素。有许多不同类型的排序算法，我将介绍三种最常见的值得熟悉的算法:选择排序、插入排序、合并排序。

作为演示如何应用每种排序算法的例子，假设您正试图按作者的姓氏对书架上的 *n* 本书进行排序。

# 选择排序

在选择排序中，首先在整个书架中查找作者姓氏出现在字母表中最早的那本书，并将该书放在书架的开头，从左数第 1 个位置。接下来，我们将从位置 2 的书开始，向右移动，在剩余的子数组中查找其作者的姓氏出现在字母表中最早的书，然后将该书放在位置 2。重复这个过程，直到槽 *n — 1* ，我们将使用选择排序完成整个书架的排序。

为了在 Python 中实现选择排序算法，我们需要跟踪两个子数组:已排序的子数组(当前索引 *i* 右侧原始数组中的元素)和剩余的未排序子数组(当前索引 *i* 左侧原始数组中的元素)。对于原始数组中的每个元素，我们需要遍历剩余的未排序子数组中的元素，以找到最小的元素，并将其与当前索引 *i* 中的元素交换。

选择排序算法的时间复杂度是 O(n ),因为该算法中有两个 for 循环。

```
def selection_sort(arr):
   n = len(arr)
   for i in range(n):
      # Find smallest element in the remaining indices of the array
      min_idx = i
      for j in range(i + 1, n):
         if arr[j] < arr[min_idx]:
            min_idx = j # Swap smallest minimum element with the element in position i
      arr[i], arr[min_idx] = arr[min_idx], arr[i]

   return arr
```

# 插入排序

在插入排序中，第一个 I 索引中的元素与最初在第一个 I 索引中的元素相同。类似于选择排序，我们从左到右遍历原始数组的每个元素。但是，这一次我们将比较当前索引 *i* (key)处的元素与当前索引右侧的已排序子数组中的每个元素，直到我们找到一个不超过当前元素的元素，并将它放在这个新位置。

插入排序算法的时间复杂度的上限是 O(n ),当数组中的元素顺序相反时会出现这种情况，因为内部 while 循环必须遍历已排序子数组中的每个元素。如果所有元素都已排序，并且内部 while 循环不必进行任何迭代，则插入排序算法的时间复杂度的下限是 O(n)。

```
def insertion_sort(arr):
   n = len(arr)
   for i in range(1, n):
      key = arr[i]
      j = i - 1 # Compare key to every element in the 
      # sorted subarray until the key is smaller 
      # than the current element 
      while j >= 0 and key < arr[j]:
         arr[j + 1] = arr[j]
         j -= 1 # Insert key in identified position
      arr[j + 1] = key return arr 
```

# 合并排序

合并排序是一种分治算法，我们将问题分解成子问题，递归地解决子问题，然后将子问题的解决方案组合起来解决原始问题。在我们的图书分类示例中，我们将如何应用分而治之:

1.  *Divide* :将数组分成两半；
2.  *征服*:递归排序上一步得到的两半书中的书籍。基本情况发生在子阵列中只剩下一本书的时候；
3.  *合并*:将排序后的两半合并在一起。

合并排序的运行时复杂度为 O(n log n ),因为将数组分成块需要 O(log n)时间，对每个块进行排序需要线性时间来合并两半。

```
def merge_sort(arr):
   # Terminating base case
   if len(arr) < 2:
      return arr
   else:
      # Divide array into two subarrays
      midpoint = len(arr) // 2
      left = arr[:midpoint]
      right = arr[midpoint:] # Sort subarrays
      sorted_left = merge_sort(left)
      sorted_right = merge_sort(right) # Merge sorted subarrays
      sorted_arr = []
      while len(sorted_left) > 0 and len(sorted_right) > 0:
         # Compare first elements of subarrays
         if sorted_left[0] < sorted_right[0]:
            sorted_arr.append(sorted_left[0])
            sorted_left.pop(0)
         else:
            sorted_arr.append(sorted_right[0])
            sorted_right.pop(0) # Insert remaining items in sorted subarrays
      sorted_arr.extend(sorted_left)
      sorted_arr.extend(sorted_right) return sorted_arr
```

# 结论

选择排序、插入排序和合并排序是需要了解的典型排序算法。Python 示例旨在帮助演示这些算法在实践中如何工作。在算法解释系列的下一部分，我将介绍搜索算法。

更多请看本算法讲解系列: [#1:递归](/algorithms-explained-1-recursion-f101500f9316)、 [#2:排序](/algorithms-explained-2-sorting-18d0875528fb)(本期文章)、 [#3:搜索](/algorithms-explained-3-searching-84604e465838)、 [#4:贪婪算法](/algorithms-explained-4-greedy-algorithms-f60792046d40)、 [#5:动态规划](/algorithms-explained-5-dynamic-programming-e5472a4ce464)、 [#6:树遍历](/algorithms-explained-6-tree-traversal-1a006ba00672)。