# 算法解释#6:树遍历

> 原文：<https://towardsdatascience.com/algorithms-explained-6-tree-traversal-1a006ba00672>

## 用 Python 中的例子解释树遍历算法

![](img/dd518d4192a252e9ca99d5f9f652483c.png)

图片由 [Clker-Free-Vector-Images](https://pixabay.com/users/clker-free-vector-images-3736/) 来自 [Pixabay](http://pixabay.com)

树由一组由边连接的节点表示。它们被认为是分层的非线性数据结构，因为树中的数据存储在多个层次上。树具有以下属性:

*   *根节点:*每棵树都有一个节点被指定为根节点，它是树的最顶端的节点，并且是没有任何父节点的节点。
*   *父节点:*父节点是它之前的节点，除了根节点之外的每个节点都有一个父节点。
*   *子节点:*子节点是其后的节点。根据树的类型，每个节点可以有不同数量的子节点。

# 树形数据结构的类型

有一些通用树具有上面列出的属性，并且对节点数量没有其他限制。还有各种类型的树数据结构，对于不同的用例有更多的限制，下面是一些常见的:

1.  *二叉树:*二叉树是一种节点最多可以有两个子节点的树，正如“二进制”标签所暗示的。
2.  *二叉查找树(BST)*:BST 是二叉树的一个特例，其中的节点是按照它们的值排序的。对于每个父节点，左边的子节点是一个较小的值，而右边的子节点是一个较大的值。这种结构使得查找很快，因此对于搜索和排序算法很有用。
3.  *堆:*堆是一种特殊的树数据结构，有两种类型的堆:I)最小堆是一种树结构，其中每个父节点小于或等于其子节点；ii)最大堆是一种树形结构，其中每个父节点大于或等于其子节点。这种类型的数据结构对于实现优先级队列很有用，在优先级队列中，项目按权重排列优先级。请注意，Python 有一个名为`heapq`的内置库，其中包含对堆数据结构执行不同操作的函数。

# 树遍历算法

在回顾了树数据结构的常见类型之后，下面是一些常见的树遍历算法及其在 Python 中的实现的例子。

## 广度优先搜索(BFS)

面包优先搜索(BFS)是一种常用算法，用于按系统顺序遍历二叉查找树(BST)或图。它从第 0 层的根节点开始，一次访问一个节点，从一侧横向移动到另一侧，直到找到所需的节点或访问完所有节点。该算法在深入搜索之前先进行大范围搜索，因此被称为面包优先搜索。

BFS 的时间复杂度是 O(n ),因为树的大小由项目搜索长度决定，并且每个节点被访问一次。下面是 Python 中的实现:

```
def bfs(graph, source):
   """Function to traverse graph/ tree using breadth-first search Parameters:
   graph (dict): dictionary with node as key and 
                 list of connected nodes as values.
   source (str): source node to start from, usually 
                 the root node of the tree. Returns:
   bfs_result (list): list of visited nodes in order.
   """
   # Define variables
   bfs_result = []
   queue = [] # Add source node to queue
   queue.append(source) while queue:
     # Visit node at front of queue
     node = queue.pop(0) # Check if we have visited this node before
     if node not in bfs_result:
        # Mark node as visited
        bfs_result.append(node)
        # Add all neighbor nodes to queue
        for neighbor in graph.get(node, []):
           queue.append(neighbor) return bfs_result
```

## 深度优先搜索

深度优先搜索(DFS)是树遍历算法的另一种变体，它也从根节点开始，但沿着一个分支向下移动，并尽可能沿着一个分支向下移动。如果所需的节点不在该分支中，它会返回并选择另一个分支。该算法一直这样做，直到找到期望的节点或者所有节点都被访问过。该算法在进入另一个分支之前首先向下探索一个分支(深度)，因此被称为深度优先搜索。

基于与 BFS 相同的原因，DFS 的时间复杂度也是 O(n)。下面是 Python 中的实现:

```
def dfs(graph, source):
   """Function to traverse graph/ tree using depth-first search Parameters:
   graph (dict): dictionary with node as key and 
                 list of child nodes as values.
   source (str): source node to start from, usually 
                 the root node of the tree. Returns:
   dfs_result (list): list of visited nodes in order.
   """
   # Define variables
   dfs_result = []
   queue = [] # Add source node to queue
   queue.append(source) while queue:
      # Get last item in queue
      node = queue.pop() # Check if we have visited node before
      if node not in dfs_result:
         dfs_result.append(node)
         # Add neighbors to queue
         for neighbor in graph.get(node, []):
            queue.append(neighbor) return dfs_result
```

## 结论

树是表示分层和非线性数据的有用数据结构。最常用的树数据结构类型是二叉树(树中的每个节点最多有两个子节点)、二叉查找树(二叉树，其中节点按值排序)和堆(树，其中父节点少于或多于它们的子节点，取决于它是最小堆还是最大堆)。

在算法方面，面包优先搜索和深度优先搜索是重要的树遍历算法，具有许多实际应用，从 GPS 导航系统中的路径寻找和路线规划到调度到网络拓扑。在下一篇文章中，我们将介绍一个用于信息编码的树遍历的特殊例子，称为霍夫曼码。

更多请看本算法讲解系列: [#1:递归](/algorithms-explained-1-recursion-f101500f9316)、 [#2:排序](/algorithms-explained-2-sorting-18d0875528fb)、 [#3:搜索](/algorithms-explained-3-searching-84604e465838)、 [#4:贪婪算法](/algorithms-explained-4-greedy-algorithms-f60792046d40)、 [#5:动态规划](/algorithms-explained-5-dynamic-programming-e5472a4ce464)、 [#6:树遍历](/algorithms-explained-6-tree-traversal-1a006ba00672)(本期文章)。