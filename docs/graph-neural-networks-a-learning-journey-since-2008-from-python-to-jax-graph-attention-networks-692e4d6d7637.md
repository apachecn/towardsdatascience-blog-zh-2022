# 图形神经网络:2008 年以来的学习之旅——从 Python 到 JAX:图形注意力网络

> 原文：<https://towardsdatascience.com/graph-neural-networks-a-learning-journey-since-2008-from-python-to-jax-graph-attention-networks-692e4d6d7637>

## 实践中的 GAT！探索 PyTorch、Torch Geometric 和 JAX #GPU 实现来加速 GAT 性能

![](img/6c2374b48bcb6b2af114263da50e749f.png)

图片由[本杰明·沃罗斯](https://unsplash.com/@vorosbenisop)在 [Unsplash](https://unsplash.com/photos/phIFdC6lA4E) 上拍摄

*通过我的推荐链接加入 Medium 来支持我的写作和项目:*

[](https://stefanobosisio1.medium.com/membership)  

我之前关于图形神经网络的文章:

*   [图形神经网络:2008 年以来的学习之旅——第一部分](/graph-neural-networks-a-learning-journey-since-2008-part-1-7df897834df9?source=your_stories_page----------------------------------------)
*   [图形神经网络:2008 年以来的学习之旅——第二部分](/graph-neural-networks-a-learning-journey-since-2008-part-2-22dbf7a3b0d?source=your_stories_page----------------------------------------)
*   [图形神经网络:2008 年以来的学习之旅——深度行走](/graph-neural-networks-a-learning-journey-since-2008-deep-walk-e424e716070a?source=your_stories_page----------------------------------------)
*   [图形神经网络:2008 年以来的学习之旅——Python&深度行走](/graph-neural-networks-a-learning-journey-since-2008-python-deep-walk-29c3e31432f?source=your_stories_page----------------------------------------)
*   [图形神经网络:2008 年以来的学习历程——图形卷积网络](/graph-neural-networks-a-learning-journey-since-2008-graph-convolution-network-aadd77e91606)
*   [图神经网络:2008 年以来的学习之旅——Python&图卷积网络](/graph-neural-networks-a-learning-journey-since-2008-python-graph-convolutional-network-5edfd99f8190)
*   [图形神经网络:2008 年以来的学习之旅——扩散卷积神经网络](/graph-neural-networks-a-learning-journey-since-2008-diffusion-convolutional-neural-networks-329d45471fd9)
*   [图形神经网络:2008 年以来的学习之旅——图形注意力网络](/graph-neural-networks-a-learning-journey-since-2008-graph-attention-networks-f8c39189e7fc)

上次我们学习了图形注意网络(GAT)的理论原理。GAT 试图克服以前的图形神经网络方法中的一些已知问题，强调注意力的力量。我们看到，自 2014 年 Cho 和 Bahdanau 开始解决序列对序列(Seq2Seq)问题(例如，从法语到英语的翻译)以来，注意力一直是一个热门话题。GAT 的核心部分是 Bahdanau 的注意力，它计算输入序列和预期输出序列之间的比对分数。也可以为图节点的特征计算这样的分数，将输入图转换成潜在表示。此外，这种新的特征表示可以通过*多头注意力*得到进一步增强，其中多个注意力层被并行使用，以检测和使用彼此远离的节点之间的长期依赖性和关系。

今天，所有这些理论都可以付诸实践。所有的实现都将在 Cora 数据集上进行测试(许可证:[https://paperswithcode.com/dataset/cora](https://paperswithcode.com/dataset/cora)；
CC0:公共领域)[1，4]。Cora 数据集由机器学习论文组成，分为 7 类:`case_based`、`genetic_algorithms`、`neural_networks`、`probabilistic_methods`、`reinforcement_learning`、`rule_learning`、`theory`。论文总数为 2708，这将是节点数。去除词干和停用词后，最终数据集只有 1433 个唯一词，这将是要素的数量。因此，该图可以用一个 2708 x 1433 的矩阵来表示，其中 *1* s 和*0*取决于特定单词的存在。从论文引文文件中可以获得边列表并创建一个 2708×2708 的邻接矩阵。最初将颁发一个 PyTorch。随后，JAX 又进一步改进了深记*的*包。最后，用 Torch 几何包进行测试。

！！！[所有这些测试都能在这个 Jupyter Colab 笔记本里找到](https://gist.github.com/Steboss89/b21d6abe548d106119666fec6b65965f)！！！

## 前奏:加载数据例程

对于所有的测试，数据都是用模板化的 Python 代码加载的，这里是好心提供的

图 1:加载 CORA 数据集程序。该函数返回邻接矩阵、节点特征、标签、用于训练、验证和测试的索引以及长传感器格式的边列表

CORA 数据集从[https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz)下载，然后节点的特征被编码成一个稀疏矩阵`features = sp.csr_matrix(idx_features_labels[:, 1:-1],dtype=np.float32`并被归一化，使得它们的值范围在[0，1]之间。从`Cora.cites`创建边缘列表，并从那里获得稀疏的`adjacency`矩阵。最终元素被转换为 PyTorch 张量:

```
features = torch.FloatTensor(np.array(features.todense()))    
labels = torch.LongTensor(np.where(labels)[1])    
adj = torch.LongTensor(adj.todense())   

idx_train = torch.LongTensor(idx_train)    
idx_val = torch.LongTensor(idx_val)    
idx_test = torch.LongTensor(idx_test) edge_tensor = torch.LongTensor(edges)
```

# 第一首赋格:PyTorch 实现 GAT

让我们从 PyTorch 开始我们的实现之旅。图 2 显示`layers.py`开始覆盖 GAT 的第一个关键点:注意力算法。

图 2: Bahdanau 在 PyTorch 对 GAT 的关注

代码是我们在理论中看到的一个总结。首先，我们需要指定一个大小为`in_features, out_features`的权重矩阵`W`，它是输入节点特征矩阵`h`的倍数。这个产品然后被传递到 attention，它是由两层和一维输出的神经网络`a`组成的。比对分数如下获得:

```
WH1 = torch.matmul(WH, self.a[:self.out_features, :]) 
WH2 = torch.matmul(WH, self.a[self.out_features:, :])
e = nn.LeakyReLU(WH1 + WH2.T)
attnt = torch.where(adjacency > 0, e, -1e9*torch.ones_likes(e) )         attnt = funct.softmax(attnt, dim=1)
```

最后，结果可以连接或平均，从这里返回新节点的表示`hfirst.` 非常容易实现具有多头关注的 GAT:

图 3: GAT 模型，多头注意力，向前迈步，准备接受训练

导入基本注意层`from layers import GraphAttentionLayer`后，构造器建立注意头和最终输出:

```
self.attheads = [GraphAttentionLayer(in_features, out_features, concat=True) for _ in range(nheads)]     

self.output = GraphAttentionLayer(out_features*nheads, nclass, concat=False)
```

`forward`步骤计算每个注意力头`attn(X, adjacency) for attn in self.attheads`的输出，并返回给定输入图`X`的最终输出节点预测

我们现在已经准备好根据给定的 CORA 输入开始训练 GAT，图 5:

图 5:基于 PyTorch 的 GAT 的训练路由

训练使用亚当优化器，100 个周期的学习率为`lr=0.005`和`weight_decay=5e-4`。该测试在单 CPU 处理器英特尔至强 2.20GHz 的 google Colab 上运行。100 个时代可以在 2 分 55 秒内运行，训练和验证负似然损失`nll`，从 1.945 提高到 1.931。好但不是最好的方法！我们可以用 PyTorch Geometric 放大我们的模型

# 更好的是:PyTorch 几何中的一首赋格曲

PyTorch Geometric 是一个建立在 PyTorch 之上的库，并针对图形神经网络的实现和计算进行了优化。PyTorch Geometric 的关键结构点是易于使用的小型批量加载器，它允许模型处理非常大的数据集，以及优化的图形管道支持，并在箭筒中进行分布式图形学习。简而言之，用 PyTorch Geometric 编写代码非常简单，只需几行代码就可以实现图形神经网络，还可以在 CPU 和 GPU 之间切换。

图 6 显示 PyTorch 几何安装和 CORA 数据集的加载。数据集可以很容易地通过`torch_geometric.datasets`导入。数据集对象可以很容易地与`torch_geometric.transforms`交互，因此特性可以在一行代码中规范化为`dataset.transform = T.NormalizeFeatures()`

图 6: PyTorch 几何安装和 Cora 数据集的加载

为了与之前的测试保持一致，CORA 数据集将通过`load_data()`函数加载。从这里我们可以创建我们的 GAT 网络对象:

图 GAT 对象的实现和训练。培训可以是

GAT 类遵循通常的 PyTorch 方案。在结构`__init__`中，我们定义了模型参数和模型层。这些层由两个`GATConv`单元组成。前者是多头注意层，输出级联，后者是多头注意层，平均输出。在`forward`功能中，定义了模型序列。初始丢弃后，输入节点的特征在`conv1`层进行处理，最后在`conv2`层进行平均。与之前一样，我们将在英特尔至强 2.20GHz 处理器上使用 8 个隐藏单元和 8 个注意力头执行计算，并通过 Adam optimiser 进行参数优化。正如之前一样，我们可以调用上面的`train`函数来执行训练，而神奇的命令`%%time`测量执行时间。

PyTorch Geometric 中引入的伟大优化仅在 9.58 秒内处理 100 个时代，训练损失从 1.946 到 1.107 不等，有效损失从 1.945 到 1.352 不等。这是一个了不起的结果，这应该会让我们想起在处理 GNN 问题时应该用什么方法。它还没有完成。我将向您展示一种处理 GAT 的更深入的方法。

# JAX +图形处理器

作为最后的测试，我检查了吉列姆·库库鲁[https://github.com/gcucurull/jax-gat](https://github.com/gcucurull/jax-gat)的实现，他出色地实现了 JAX 的 GAT，我对原始代码做了一些修改。什么是 JAX？JAX 是 DeepMind 中使用的主要编码框架，用于放大大模型，使它们运行得更快。JAX 的基础核心单元是`autograd`一个在 Python 中高效计算导数的 Python 包和`XLA`一个加速线性代数编译器，可以用几行代码加速 Tensorflow 模型，在 8 Volta V100 GPUs 上实现 BERT 7 倍的性能提升。正如你所看到的，如果我们要在 GPU 或 TPU 上加速我们的模型，JAX 工作得非常好。此外，DeepMind 最近发布了 https://github.com/deepmind/jraph/tree/master/jraph 的`jraph`一个新的 JAX 包，完全专注于图形神经网络的实现。遗憾的是，我还没有时间处理它——正如你看到的，我还在调试我的注意力实现——但是更多的更新即将到来:)

图 9 示出了在 JAX 中实现注意力算法的核心。首先，`GraphAttentionLayer`从构造器`init_fun`开始初始化神经网络和矩阵`W`权重。这是通过`jax`作为`jax.random.split(random.PRNGKey(NUMBER),4)`产生一些特定的随机数来实现的。`jax`需要随机数，例如，初始化数组或计算概率分布。特别是，输入参数需要是一个伪随机数生成器密钥或`PRNGKey`。`NUMBER`是一个随机数，可以通过`NumPy`轻松生成。随机生成器函数是`create_random`，它与`jit`装饰器一起存在。`jit`标记一个要优化的函数，它的意思是`just in time`，即该函数的编译将推迟到第一次函数执行时进行。由于我们只需要生成 4 个随机数，`jax`并不是非常高效，因此我们可以用`jit`来加速这个过程——记住`jax`在我们扩大规模时非常有用，因此在生成数百万个随机数时，它会比简单的`NumPy` [更高效。](https://github.com/google/jax/issues/968)

一旦输入权重被初始化，`apply_fun`计算输入特征矩阵`x`上的关注度。Guillem 已经实现了`Dropout`函数，允许将`is_training`作为一个参数。然后，代码与 PyTorch 和`NumPy`非常相似，例如`jax.numpy.dot(x,W)`用于点积乘法，`jax.nn.softmax(jax.nn.leaky_relu(...))`用于最终注意力的 softmax 计算

图 9:JAX GAT 关注层的实现

接下来，创建多头注意力层，如图 10 所示。首先，我们创建一个空列表`layer_funs`来存储注意力头，并创建一个空列表来初始化权重`layer_inits`。然后，`init_fun`在这种情况下初始化`layer_inits`列表中每个元素的权重。`apply_fun`为`layer_funs`中的每个元素设置关注的应用。如果我们在 GAT 的最后一层，我们对所有的注意力进行平均，否则我们连接结果`x = jax.numpy.concatenate()`

图 10:多头关注层的实现

图 11 将最后的`GAT`函数合并在一起。在这种情况下，`nhid`头被创建为`MultiHeadLayer(nheads[layer_i], nhid[layer_i]...)`，每个头的初始化函数被附加到`init_funs`列表，应用函数被附加到`attn_funs`列表。

图 11:最终的 GAT 实现，合并多标题层和关注层

在`GAT`中，`init_fun`通过`att_init_func`初始化每个头，同时`apply_fun`运行完整的 GAT 网络算法。

为了创建输入 CORA 数据，我对上面的 load data 函数做了一点修改，返回了`NumPy`数组而不是张量(图 12)

图 load _ data 函数的轻微修改。返回 NumPy 数组，而不是 PyTorch 张量

一旦一切就绪，我们就可以开始训练了。训练在特斯拉 K8 和特斯拉 T4 GPU 上运行——由谷歌❤实验室免费提供

图 13 显示了训练程序。在每一步中，`jax.experimental.optimizer.Adam`通过`autograd`功能更新所有参数。`update`函数需要在每个时期结束时运行，所以我们可以用`jit`装饰器来加速。然后，每个数据批次可以作为`jax.device_put(train_batch)`直接加载到计算设备上，在这种情况下是 GPU。这是 jax 的一个显著特性，因为整个数据都可以很容易地读取，从而提高了整个模型的性能。如果没有这个步骤，该算法在 Tesla K80 上运行大约需要 1/2 分钟，而在 GPU 上推送数据平均将计算时间减少到 30/40 秒。显然，特斯拉 T4 的表现更好，平均成绩为 15/18 秒。对于训练和验证批次，最终训练负对数似然损失的范围为 1.908 至 1.840。

图 13: JAX 加特训练部分。Update 函数和 device_put 是这个在 GPU 上实现的核心特性。

今天是一切:)

[您可以在这里找到包含上述测试的整个笔记本](https://gist.github.com/Steboss89/b21d6abe548d106119666fec6b65965f)

请继续关注下一个图表冒险！

如有任何问题或意见，请随时给我发电子邮件:stefanobosisio1@gmail.com 或直接在 Medium 这里。

# 文献学

1.  网络数据中的集体分类。艾杂志 29.3(2008):93–93。
2.  《推特上仇恨用户的描述和检测》*第十二届 AAAI 国际网络和社交媒体会议*。2018.
3.  彭宁顿、杰弗里、理查德·索彻和克里斯托弗·d·曼宁。"手套:单词表示的全局向量."*2014 年自然语言处理经验方法会议论文集*。2014.
4.  《用机器学习自动化互联网门户的构建》*信息检索*3.2(2000):127–163。