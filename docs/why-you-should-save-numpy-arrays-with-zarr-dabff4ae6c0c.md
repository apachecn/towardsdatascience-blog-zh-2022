# 为什么应该用 Zarr 保存 NumPy 数组

> 原文：<https://towardsdatascience.com/why-you-should-save-numpy-arrays-with-zarr-dabff4ae6c0c>

## 利用 Dask 更快地读写阵列

![](img/1e1619876fb12e73f8e095436e531c85.png)

图片由 Sebastian Kanczok 通过 [Unsplash](http://unsplash.com) 拍摄

# TL；博士；医生

这篇文章告诉你为什么以及如何使用 Zarr 格式来保存你的 NumPy 数组。它指导您使用 Zarr 和 Dask 并行地读取和写入大型 NumPy 数组。

如果你想直接进入，这是代码。如果您对代码有任何疑问，请在 Twitter 上联系我[。](https://twitter.com/richardpelgrim)

# 保存 NumPy 数组的常用方法

存储 NumPy 数组的三种常见方式是[存储为](https://crunchcrunchhuman.com/2021/12/25/numpy-save-csv-write/) `[.csv](https://crunchcrunchhuman.com/2021/12/25/numpy-save-csv-write/)` [文件](https://crunchcrunchhuman.com/2021/12/25/numpy-save-csv-write/)，[存储为](https://mungingdata.com/numpy/save-numpy-text-txt/) `[.txt](https://mungingdata.com/numpy/save-numpy-text-txt/)` [文件](https://mungingdata.com/numpy/save-numpy-text-txt/)或`.npy`文件。

这些方法都有重要的局限性:

*   CSV 和 TXT 文件是人类可读的格式，不能包含大于二维的 NumPy 数组。
*   原生 NPY 二进制文件格式**不支持并行读/写**操作。

让我们在下面看看这一点。

我们将从创建一个伪 NumPy 数组开始。我们将使用`np.random.rand`生成两个填充了随机数的数组，一个是二维数组，一个是三维数组:

```
import numpy as np array_XS = np.random.rand(3,2) 
array_L = np.random.rand(1000, 1000, 100)
```

将三维`array_L`存储为`.txt`或`.csv`会抛出一个值错误:

```
np.savetxt('array_L.txt', array_L, delimiter=" ") np.savetxt('array_L.csv', array_L, delimiter=",") ValueError: Expected 1D or 2D array, got 3D array instead
```

您可以将三维数组存储为. npy 文件。这是可行的，但不能扩展到大于内存的数据集或其他需要并行读取和/或写入的情况。

```
np.save('array_L.npy', array_L)
```

# 用 Zarr 保存 NumPy 数组

除了上面的三个选项，考虑将 NumPy 数组保存到 [Zarr](https://zarr.readthedocs.io/en/stable/index.html) 中，这是一种用于存储分块、压缩的 N 维数组的格式。

Zarr 的三个最重要的优势是:

1.内置**多种压缩选项**和级别

2.支持**多个后端数据存储** (zip，S3 等。)

3.**可以并行读写数据** *在 n 维压缩块中吗

Zarr 还被 Dask、TensorStore 和 x-array 等 PyData 库广泛采用，这意味着将这种文件格式与支持的库一起使用时，性能会有显著提高。

** Zarr 支持分别并发读取和并发写入，但不支持同时并发读取和写入。*

# 用 Zarr 压缩 NumPy 数组

让我们看看 Zarr 的压缩选项。下面，我们将小数组和大数组保存到`.zarr`并检查结果文件大小。

```
import zarr 
# save small NumPy array to zarr 
zarr.save('array_XS.zarr', array_XS) # get the size (in bytes) of the stored .zarr file 
! stat -f '%z' array_XS.zarr 
>> 128 # save large NumPy array to zarr 
zarr.save('array_L.zarr', array_L) # get the size of the stored .zarr directory 
! du -h array_L.zarr >> 693M	array_L.zarr
```

将`array_L`存储为 Zarr 会导致文件大小显著减少(`array_L`减少约 15%)，即使只有默认的开箱即用压缩设置。查看随附的笔记本以获得更多压缩选项，您可以调整这些选项来提高性能。

# 用 Zarr 加载 NumPy 数组

您可以使用`zarr.load()`将存储为`.zarr`的数组加载回 Python 会话中。

```
# load in array from zarr 
array_zarr = zarr.load('array_L.zarr')
```

它将作为一个常规的 NumPy 数组加载。

```
type(array_zarr) >>> numpy.ndarray
```

Zarr 支持多个后端数据存储。这意味着你也可以轻松地从基于云的数据商店加载`.zarr`文件，比如亚马逊 S3:

```
# load small zarr array from S3 
array_S = zarr.load(
"s3://coiled-datasets/synthetic-data/array-random-390KB.zarr"
)
```

# 利用 Zarr 和 Dask 并行读写 NumPy 数组

如果您正在处理存储在云中的数据，那么您的数据很可能比本地机器内存大。在这种情况下，您可以使用 Dask 并行读写大型 Zarr 数组。

下面我们试着加载一个 370GB。zarr 文件直接插入到我们的 Python 会话中:

```
array_XL = zarr.load(
"s3://coiled-datasets/synthetic-data/array-random-370GB.zarr"
)
```

这将失败，并出现以下错误:

```
MemoryError: Unable to allocate 373\. GiB for an array with shape (10000, 10000, 500) and data type float64
```

加载同样的 370GB。zarr 文件放入 Dask 数组工作正常:

```
dask_array = da.from_zarr(
"s3://coiled-datasets/synthetic-data/array-random-370GB.zarr"
) 
```

这是因为 Dask *评估懒惰*。在您明确指示 Dask 对数据集执行计算之前，不会将数组读入内存。在这里阅读更多关于 Dask [的基础知识。](https://coiled.io/blog/what-is-dask/)

这意味着您可以在本地对该数据集执行一些计算。但是将整个数组加载到本地内存仍然会失败，因为您的机器没有足够的内存。

*注意:即使你的机器在技术上有足够的存储资源将这个数据集溢出到磁盘，这也会大大降低性能。*

[](https://coiled.io/blog/common-dask-mistakes/) [## 使用 Dask 时要避免的常见错误

### 第一次使用 Dask 可能是一个陡峭的学习曲线。经过多年的建设 Dask 和引导人们通过…

coiled.io](https://coiled.io/blog/common-dask-mistakes/) 

# 使用线圈扩展到 Dask 集群

我们需要在云中的 Dask 集群上运行它，以访问额外的硬件资源。

为此:

1.  旋转盘绕的簇

```
cluster = coiled.Cluster(
    name="create-synth-array", 
    software="coiled-examples/numpy-zarr", 
    n_workers=50, worker_cpu=4, 
    worker_memory="24Gib", 
    backend_options={'spot':'True'}, 
)
```

2.将 Dask 连接到该群集

```
from distributed import Client 
client = Client(cluster)
```

3.然后轻松地在整个集群上运行计算。

```
# load data into array
da_1 = da.from_zarr(
"s3://coiled-datasets/synthetic-data/array-random-370GB.zarr"
) # run computation over entire array (transpose)
da_2 = da_1.T da_2%%time 
da_2.to_zarr(
"s3://coiled-datasets/synthetic-data/array-random-370GB-T.zarr"
) CPU times: user 2.26 s, sys: 233 ms, total: 2.49 s 
Wall time: 1min 32s
```

我们的 Coiled 集群有 50 个 Dask workers，每个都有 24GB RAM，都运行一个包含必要依赖项的预编译软件环境。这意味着我们有足够的资源轻松地转置数组，并将其写回 S3。

Dask 能够并行地为我们完成所有这些工作，而无需将数组加载到本地内存中。它在不到两分钟的时间内将一个 372GB 的阵列加载、转换并保存回 S3。

# 保存 NumPy 数组摘要

让我们回顾一下:

*   许多存储 NumPy 数组的常用方法都有很大的局限性。
*   Zarr 文件格式提供了强大的压缩选项，支持多个数据存储后端，并且可以并行读/写 NumPy 数组。
*   Dask 允许您充分利用这些并行读/写能力。
*   将 Dask 连接到按需盘绕的集群允许在大于内存的数据集上进行高效计算。

在 LinkedIn 上关注我[,了解定期的数据科学和机器学习更新和黑客攻击。](https://www.linkedin.com/in/richard-pelgrim/)

*原载于 2022 年 1 月 5 日*[*https://coiled . io*](https://coiled.io/blog/save-numpy-dask-array-to-zarr/)*。*