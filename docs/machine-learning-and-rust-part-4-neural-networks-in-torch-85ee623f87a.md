# 机器学习与 Rust(第四部分):Torch 中的神经网络

> 原文：<https://towardsdatascience.com/machine-learning-and-rust-part-4-neural-networks-in-torch-85ee623f87a>

## 我们可以在 Rust 中使用 PyTorch 吗？什么是锈绑定？tch-rs 是什么？Rust 中的神经网络研究

![](img/5f227f81592faeedc4780988cf7ae457.png)

乔尔·菲利普在 [Unsplash](https://unsplash.com/photos/RFDP7_80v5A) 上拍摄的图片

自从我们上次看到 Rust 及其在机器学习中的应用已经有一段时间了——请向下滚动到底部，查看以前关于 ML 和 Rust 的教程。今天我将向大家介绍 Rust 中的神经网络。存在一个铁锈火炬，它允许我们创建任何我们想要的神经网络。捆绑是焊炬落地的关键。绑定允许创建*外部函数接口*或 FFI，这在 Rust 和用语言编写的函数/代码之间建立了一座桥梁。[在 Rust nomicon](https://doc.rust-lang.org/nomicon/ffi.html#calling-rust-code-from-c) 中可以找到很好的例子

要用 C 和 C++创建绑定，我们可以使用 bindgen，一个由[自动生成 Rust FFI](https://github.com/rust-lang/rust-bindgen) 的库。从绑定到 PyTorch 的 C++ api， [Laurent Mazare](https://github.com/LaurentMazare/tch-rs) 已经帮助 Rust 社区拥有了一个 Rustacean 版本的 PyTorch。正如 GitHub 页面所说，tch 在 C++ libtorch 周围提供了薄薄的包装。最大的好处是，这个库和原来的严格相似，所以没有学习障碍需要克服。[核心代码相当易读。](https://github.com/LaurentMazare/tch-rs/blob/main/src/nn/linear.rs)

# 初始化和线性:让我们学习巨人肩膀上的铁锈

首先，我们来看一下代码。这是进一步了解 Rust 基础设施的最佳起点。

首先，为了对 Rust FFI 有所了解，我们可以查看这些文件。其中大部分是自动生成的，而 Laurent 和他的同事们已经编写了大量代码，将 c++ Torch API 与 Rust 连接起来。

下面，我们就可以开始阅读`src`中的核心代码了，具体来看一下`[init.rs](https://github.com/LaurentMazare/tch-rs/blob/main/src/nn/init.rs:)`。在定义了一个`enum Init` 之后，有一个公共函数`pub fn f_init` ，它匹配输入初始化方法并返回一个权重张量和一个偏差张量。我们可以学习 C 中反映`switch`的`match`和 Python 3.10 中的`match`的用法。权重和偏差张量通过随机、统一、明凯或正交方法初始化(图 1)。

图 Rust 中的匹配大小写，它反映了 C 中的 switch 和 Python 3.10 中的 match

然后，对于类型`enum Init`，我们有了[方法实现](https://github.com/LaurentMazare/tch-rs/blob/a022da9861efbe66a4920d318166341c3a60be9e/src/nn/init.rs#L82) `[impl Init](https://github.com/LaurentMazare/tch-rs/blob/a022da9861efbe66a4920d318166341c3a60be9e/src/nn/init.rs#L82)` [。](https://github.com/LaurentMazare/tch-rs/blob/a022da9861efbe66a4920d318166341c3a60be9e/src/nn/init.rs#L82)实现的方法是一个 setter `pub fn set(self, tensor: &mut Tensor)`，这是一个很好的例子来进一步理解 Rust 中所有权和借用的概念:

图 init 的实现。注意&mut 张量，这是解释 Rust 中借力的一个很好的例子。

[我们在第一个教程](https://levelup.gitconnected.com/machine-learning-and-rust-part-1-getting-started-745885771bc2)中谈到了借贷。现在是更好地理解这个概念的时候了。假设我们可以有一个类似的`set`函数:

```
pub fn set(self, tensor: Tensor){}
```

在主代码中，我们可以调用这个函数，传递一个张量`Tensor`。`Tensor`会被设定，我们会很开心。但是，如果我们再次在`Tensor`上呼叫`set`呢？嗯，我们会遇到错误`value used here after move`。这是什么意思？这个错误告诉你你把`Tensor`移到了`set`。 *A* `*move*` *表示您已经将所有权*转让给了`set`中的`self`，当您再次调用`set(self, tensor: Tensor)`时，您希望将所有权归还给`Tensor`以便再次设置。幸运的是，在 Rust 中这是不可能的，而在 C++中则不同。在 Rust 中，*一旦一个* `*move*` *已经完成，分配给该进程的内存将被释放*。因此，我们在这里要做的是*将`Tensor`的值借用给`set`，这样我们就可以保留所有权。为此，我们需要通过引用调用`Tensor`，因此`tensor: &Tensor`。因为我们预计`Tensor`会发生变异，所以我们必须添加`mut`以便:`tensor: &mut Tensor`*

接下来，我们可以看到另一个重要的元素，它很简单，使用了`Init`类:`[Linear](https://github.com/LaurentMazare/tch-rs/blob/main/src/nn/linear.rs)`，即一个完全连接的神经网络层:

图 3:定义线性结构并为其实现默认配置

图 3 显示了建立一个完全连接的层是多么容易，它由一个权重矩阵`ws_init`和偏置矩阵`bs_init`组成。重量的默认初始化是通过`super::Init::KaimingUniform`完成的，这是我们在上面看到的功能。

然后可以使用功能`linear`创建主全连接层。正如您在函数签名中看到的，也就是在`<...>`之间，有一些有趣的事情(图 4)。其一， [*一生注释*](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html) `['a](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html)`。如上所述，Rust 会自动识别变量何时超出范围并被释放。*我们可以注释一些变量，让它们有一个特定的生命周期*，这样我们就可以决定它们能活多久。标准注释是`'a`，其中`'`表示寿命参数。需要记住的一件重要事情是，这个签名不会修改函数中的任何内容，但是它告诉函数借用者识别所有那些其生存期可以满足我们所施加的约束的变量。

图 4:实现完全连接的神经网络层的功能。在函数签名中，您可以注意到一个生存期注释和一个通用变量 T，它从 nn::Path 借用了一个值

第二个参数是`T: Borrow<super::Path<'a>`这个注释的意思是:取 `[var_store.rs](https://github.com/LaurentMazare/tch-rs/blob/2c2b4545966be04e8377ffa7f34fe01b9a20acd0/src/nn/var_store.rs#L40)`中指定的`nn::Path` [，把这个类型借用到`T`。Rust 中的任何类型都可以自由借用为几种不同的类型。该类型将用于定义输入硬件(如 GPU)，如您在`vs:T`中所见。最后，网络的输入和输出维度与`LinearConfig`一起被指定为整数`in_dim: i64, out_dim: i64`，用于初始化权重和偏差`c: LinearConfig.`](https://github.com/LaurentMazare/tch-rs/blob/2c2b4545966be04e8377ffa7f34fe01b9a20acd0/src/nn/var_store.rs#L40)

# 让我们与巨人同行:你在 Rust 的第一个神经网络

是时候弄脏我们的手玩火炬锈了。让我们使用 MNIST 数据集建立一个简单的线性神经网络，然后是序列网络，最后是卷积神经网络。一如既往，你可以在我的 ML ❤生锈回购上找到所有的材料。 [Yann LeCun 和 Corinna Cortes 拥有 MNIST 数据集](https://keras.io/api/datasets/mnist/)的版权，并已根据[知识共享署名-类似共享 3.0 许可证的条款提供。](https://creativecommons.org/licenses/by-sa/3.0/)

## Rust 中的一个简单神经网络

和往常一样，新 Rust 项目的第一步是`cargo new NAME_OF_THE_PROJECT`，在这里是`simple_neural_networks`。然后，我们可以开始用我们需要的所有包设置`Cargo.toml`:我们将使用`mnist`、`ndarry`，显然还有`tch`——图 5。我决定使用`mnist`提取原始的 MNIST 数据，这样我们可以看到如何转换和处理数组和张量。请随意使用`tch.`中已经存在的`vision`资源

图 5: Cargo.toml 用于建立一个简单的线性神经网络。

我们将使用`mnist`下载 MNIST 数据集，使用`ndarray`对图像向量执行一些转换，并将它们转换成`tch::Tensor`。

让我们跳到`main.rs`代码。简而言之，我们需要:

1.  下载并提取 MNIST 图像，并返回用于训练、验证和测试数据的向量。
2.  从这些向量中，我们必须执行一些到`Tensor`的转换，这样我们就可以使用`tch`。
3.  最后，我们将实现一系列时段，在每个时段中，我们将输入数据乘以神经网络权重矩阵，并执行反向传播来更新权重值。

`mnist`自动从[下载输入文件到](http://yann.lecun.com/exdb/mnist/)这里。我们需要在`Cargo.toml`中添加`features = ['download']`来激活下载功能。下载文件后，提取原始数据`download_and_extract()`，并细分为训练集、验证集和测试集。注意，主函数不会返回任何东西，所以您需要在代码末尾指定`-> Results<(), Box<dyn, Error>>`和`Ok(())`(图 6)

图 6:从 mnist::MnistBuilder 下载、提取和创建训练、验证和测试集。

现在，代码的第一件事:将一个数组转换成`Tensor.`，`mnist`的输出数据是`Vec<u8>`。训练向量结构具有`TRAIN_SIZE`个图像，其尺寸是`HEIGHT`乘以`WIDTH`。这三个参数可以指定为`usize`类型，与输入数据向量一起，可以传递给`image_to_tensor`函数，如图 7 所示，返回`Tensor`

图 7: image_to_tensor 函数，给定输入数据向量、图像数量、高度和宽度，我们将返回 tch::Tensor

输入的`Vec<u8>`数据可以用`from_shape_vec`整形到`Array3`，数值被归一化并转换到`f32`，即`.map(|x| *x as f32/256.0)`。从一个数组很容易建立一个火炬张量，如第 14 行所示。对于我们的训练数据，输出张量大小为`dim1 x (dim2*dim3)`，设置`TRAIN_SIZE=50'000`、`HEIGHT=28`和`WIDTH=28`，输出训练张量大小为`50'000 x 784`。

类似地，我们将标签转换为张量，其大小将为`dim1` —因此对于训练标签，我们将有一个`50'000`长张量[https://github . com/ste boss/ML _ and _ Rust/blob/aa7d 495 C4 a2 C7 a 416d 0b 03 Fe 62 e 522 b 6225180 ab/tutorial _ 3/simple _ neural _ networks/src/main . RS # L42](https://github.com/Steboss/ML_and_Rust/blob/aa7d495c4a2c7a416d0b03fe62e522b6225180ab/tutorial_3/simple_neural_networks/src/main.rs#L42)

我们现在准备开始处理线性神经网络。在权重和偏差矩阵的零初始化之后:

```
let mut ws = Tensor::zeros(&[(HEIGHT*WIDTH) as i64, LABELS], kind::FLOAT_CPU).set_requires_grad(true);let mut bs = Tensor::zeros(&[LABELS], kind::FLOAT_CPU).set_requires_grad(true);
```

类似于 PyTorch 实现，我们可以开始计算神经网络权重。

图 8:主要训练功能。对于 N _ EPOCHS，我们在输入数据和权重及偏差之间执行 matmul。计算每个历元的精确度和损失。如果两个连续损失之间的差异小于三，我们停止学习迭代。

图 8 示出了运行线性神经网络训练的主例程。首先，我们可以用`'train`给最外层的 for 循环命名，在这种情况下，撇号不是生命期的指示器，而是循环名的指示器。我们正在监控每个时期的损失。如果两个连续的损失差小于`THRES`，当我们达到收敛时，我们可以停止最外面的循环——你可以不同意，但目前让我们保持它:)整个实现非常容易阅读，只是在从计算的`logits`中提取精度时需要注意一点，工作就完成了:)

当你准备好了，你可以直接在我的 2019 年 MacBook Pro 上用`cargo run`运行整个`main.rs`代码，2.6GHZ，6 核英特尔酷睿 i7，16GB RAM，计算时间不到一分钟，在 65 个周期后达到 90.45%的测试准确率

## 顺序神经网络

现在我们来看顺序神经网络实现[https://github . com/ste boss/ML _ and _ Rust/tree/master/tutorial _ 3/custom _ nnet](https://github.com/Steboss/ML_and_Rust/tree/master/tutorial_3/custom_nnet)

图 9 解释了顺序网络是如何建立的。首先，我们需要导入`tch::nn::Module`。然后我们可以为神经网络`fn net(vs: &nn::Path) -> impl Module`创建一个函数。该函数返回`Module`的实现，并接收作为输入的`nn::Path`，该输入是关于用于运行网络的硬件的结构信息(例如 CPU 或 GPU)。然后，时序网络被实现为输入大小为`IMAGE_DIM`和`HIDDEN_NODES`节点的线性层、`relu`和具有`HIDDEN_NODES`输入和`LABELS`输出的最终线性层的组合。

图 9:顺序神经网络的实现

因此，在主代码中，我们将神经网络创建称为:

```
// set up variable store to check if cuda is available
let vs = nn::VarStore::new(Device::cuda_if_available());// set up the seq net
let net = net(&vs.root());// set up optimizer
let mut opt = nn::Adam::default().build(&vs, 1e-4)?;
```

还有一个 Adam 优化器— [记住](https://www.google.com/search?client=safari&rls=en&q=question+mark+in+Rust&ie=UTF-8&oe=UTF-8) `[opt](https://www.google.com/search?client=safari&rls=en&q=question+mark+in+Rust&ie=UTF-8&oe=UTF-8)` <https://www.google.com/search?client=safari&rls=en&q=question+mark+in+Rust&ie=UTF-8&oe=UTF-8>末尾的 `[?](https://www.google.com/search?client=safari&rls=en&q=question+mark+in+Rust&ie=UTF-8&oe=UTF-8)` [，否则你会返回一个`Result<>`类型，它没有我们需要的功能。在这一点上，我们可以简单地按照 PyTorch 的过程来做，所以我们将设置一些 epochs，并用优化器的`backward_step`方法和给定的`loss`来执行反向传播](https://www.google.com/search?client=safari&rls=en&q=question+mark+in+Rust&ie=UTF-8&oe=UTF-8)

图 10:针对给定的历元数 N_EPOCHS 训练序列神经网络，并使用 opt.backward_step(&loss)设置反向推进；

## 卷积神经网络

我们今天的最后一步是处理卷积神经网络:[https://github . com/ste boss/ML _ and _ Rust/tree/master/tutorial _ 3/conv _ nnet/src](https://github.com/Steboss/ML_and_Rust/tree/master/tutorial_3/conv_nnet/src)

图 11:卷积神经网络结构

首先，你可以注意到我们现在使用的是`nn::ModuleT`。这个模块特征是附加训练参数。这通常用于区分训练和评估之间的网络行为。然后，我们可以开始定义网络`Net`的结构，它由两个 conv2d 层和两个线性层组成。`Net`的实现陈述了网络是如何构成的，两个卷积层的步幅分别为 1 和 32，填充为 32 和 64，膨胀分别为 5 和 5。线性层接收 1024 的输入，最后一层返回 10 个元素的输出。最后，我们需要为`Net`定义`ModuleT`实现。这里，前进步骤`forward_t`接收一个额外的布尔参数`train`，它将返回一个`Tensor`。前一步应用卷积层，以及`max_pool_2d`和`dropout`。dropout 步骤只是出于训练目的，所以它与布尔值`train`绑定在一起。

为了提高训练性能，我们将从输入张量中分批训练 conv 层。为此，您需要实现一个函数来将输入张量分成随机批次:

图 12:为从图像输入池创建批次生成随机索引

`generate_random_index`获取输入图像数组和我们想要分割的批次大小。它创建一个随机整数的输出张量`::randint`。

图 13:卷积神经网络的训练时期。对于每个时期，我们通过输入数据集进行批处理，并训练计算交叉熵的模型。

图 13 显示了训练步骤。输入数据集被分成`n_it`批，其中`let n_it = (TRAIN_SIZE as i64)/BATCH_SIZE;`。对于每一批，我们计算网络损耗并用`backward_step`反向传播误差。

在我的本地笔记本电脑上运行卷积网络需要几分钟，实现了 97.60%的验证准确率。

# 结论

你成功了！我为你骄傲！今天我们来了解一下`tch`以及如何设置一些计算机视觉实验。我们看到了初始化和线性层代码的内部结构。我们回顾了 Rust 中关于借用的一些重要概念，并了解了什么是终生注释。然后，我们开始实现一个简单的线性神经网络、一个顺序神经网络和一个卷积神经网络。在这里，我们学习了如何处理如何输入图像并将其转换为`tch::Tensor.`，我们看到了如何使用模块`nn:Module`作为一个简单的神经网络，来实现一个向前的步骤，我们还看到了它的扩展`nn:ModuleT`。对于所有这些实验，我们看到了两种执行反向传播的方法，要么使用`zero_grad`和`backward`，要么直接将`backward_step`应用于优化器。

希望你喜欢我的教程:)敬请期待下一集。

# 支持我的写作:

*通过我的推荐链接加入 Medium 来支持我的写作和项目:*

<https://stefanobosisio1.medium.com/membership>  

如果有任何问题或意见，请随时给我发电子邮件，地址是:stefanobosisio1@gmail.com，或者直接在 Medium 这里。

# 以前关于 Rust 和 ML 的教程

<https://levelup.gitconnected.com/machine-learning-and-rust-part-1-getting-started-745885771bc2>  <https://levelup.gitconnected.com/machine-learning-and-rust-part-2-linear-regression-d3b820ed28f9>  <https://levelup.gitconnected.com/machine-learning-and-rust-part-3-smartcore-dataframe-and-linear-regression-10451fdc2e60> 