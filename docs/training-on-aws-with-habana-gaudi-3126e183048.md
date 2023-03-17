# Habana Gaudi 的 AWS 培训

> 原文：<https://towardsdatascience.com/training-on-aws-with-habana-gaudi-3126e183048>

# Habana Gaudi 的 AWS 培训

## 利用专用 DNN 训练芯片的力量—第 2 部分

![](img/d26a25c7e658fd1fccc7c1bb870f5abf.png)

由[安东尼·高迪](https://en.wikipedia.org/wiki/Antoni_Gaud%C3%AD)设计的圣家族大教堂的中殿，由[维德·昆乔罗](https://unsplash.com/@wiwid?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

今年十月，AWS [宣布](https://aws.amazon.com/blogs/aws/new-ec2-instances-powered-by-gaudi-accelerators-for-training-deep-learning-models/)亚马逊[EC2 DL1 实例类型](https://aws.amazon.com/ec2/instance-types/dl1/)的到来。DL1 由 8 个 Habana Gaudi 加速器驱动，是第一个包含专用 AI 加速器的 AWS 实例类型，这些加速器是**而不是**GPU。Habana Gaudi 以著名的加泰罗尼亚建筑师[Antoni Gaudi](https://en.wikipedia.org/wiki/Antoni_Gaud%C3%AD)的名字命名，是一款全新的人工智能 ASIC，专门为深度学习工作负载而从头设计。这提供了提高资源利用率和降低培训成本的潜力。事实上，DL1 实例已经向全世界发布，并承诺“比当前一代基于 GPU 的实例的性价比高 40%”。

在这篇博文中，我们将评估 DL1 实例，并展示它的一些独特属性。这是上一篇[文章](/tpu-training-6eb84100d138)的续篇，在那篇文章中，我们讨论了使用专用人工智能芯片的潜力以及采用它们的一些潜在挑战。在那里，我们建议将您的训练应用程序迁移到新的人工智能芯片的任务分解为四个步骤:

1.  **高级兼容性分析**:尽早评估您的工作负载特性是否符合芯片规格和支持软件堆栈。
2.  **调整您的模型以在新芯片上运行**:您可能需要对您的模型进行一些调整，例如替换专用人工智能芯片不支持的操作。
3.  **优化新芯片的运行时性能**:为了充分利用芯片，您需要分析并最大化其利用率。
4.  **调整模型以在新芯片上收敛**:可能需要对模型超参数进行一些修改，以确保及时收敛。

这些步骤在我们之前的[帖子](/tpu-training-6eb84100d138)中有详细描述。在本帖中，我们将按照这些步骤评估 DL1 实例。

这篇博文和我们包含的代码片段是基于撰写本文时可用的最新软件栈，[版本 1.2.0](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html#release-versions) 。鉴于 Habana Gaudi 产品的相对新颖性，新版本可能会包括重要的增强和优化。您必须使用最新的可用软件堆栈，并确保相应地重新评估我们的一些陈述。
虽然我们将专注于[tensor flow](https://www.tensorflow.org/)2.7 版本，但我们写的大部分内容也同样适用于 Habana Gaudi 支持的其他机器学习框架。

# 1.高级兼容性评估

此步骤的目的是收集尽可能多的公开信息，以便评估 DL1 产品是否满足了您的培训需求。这包括以下在线资源:

*   **系统架构规格**:[DL1 硬件细节](https://aws.amazon.com/ec2/instance-types/dl1/)以及 [Habana Gaudi 架构指南](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Overview.html#gaudi-architecture)应该能让你对机器的训练能力有一个大致的了解。特别是，您可以验证内存、计算和其他硬件资源是否符合您的需求。
*   **软件文档** : Habana Gaudi 附带了一个名为 [SynapseAI 软件套件](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Overview.html#synapseai-software-suite)的综合软件堆栈。[开发者文档](https://docs.habana.ai/en/latest/#getting-started)包括支持的[框架和版本](https://docs.habana.ai/en/latest/#getting-started)、 [API 限制](https://docs.habana.ai/en/latest/Release_Notes/GAUDI_Release_Notes.html#known-issues-and-limitations-1-2-0)等细节。有几个[用户指南](https://docs.habana.ai/en/latest/#synapseai-user-guides)演示了如何使用 API 套件的许多特性。
    使用软件文档来验证支持的框架、版本和操作是否满足您的机器学习项目的需求。
*   **基准测试报告**:哈伯纳在各种流行的模型架构上分享了性能[基准测试结果](https://developer.habana.ai/resources/habana-training-models/)。你可以将这些与其他人工智能加速器厂商的[性能结果进行比较。您还可以查看](https://developer.nvidia.com/deep-learning-performance-training-inference) [MLPerf](https://mlcommons.org/en/) ，这是一个流行的人工智能培训基准套件，可以比较多个人工智能加速器(包括 Habana Gaudi)在各种工作负载上的性能。最新的 MLPerf 报告摘要(在撰写本文时)可在[这里](https://habana.ai/mlperf-ai-training-benchmark-habana-gaudi-performance-and-scale-results/)找到。
    基准测试结果可以让你了解高迪擅长的车型类型。然而，正如我们在[之前的帖子](/tpu-training-6eb84100d138)中所警告的，除非您正在训练的模型与基准报告中包含的模型之一相同，否则根据报告的基准来预测您自己的模型的性能可能不那么容易。这是因为模型中的小变化会对其运行时性能产生有意义的影响。

与任何其他新颖的硬件产品一样，要真正感受 DL1 的功能，除了开始使用它之外，没有其他更好的方法了。是的，您确实面临着投资进入潜在死胡同的风险，但我们相信，即使您最终没有在 DL1 上培训您当前的模型，您在此过程中积累的知识和技能也会很好地为您服务。

在下面的项目中，我们总结了与其他人工智能加速器和训练实例相比，基于高迪的 DL1 产品的一些主要亮点。接下来是一些潜在的担忧。这些都是基于我们自己的个人印象。我们的列表并不全面，也不应该被视为官方文件的替代品。

*   **异构架构**:单个 Gaudi 内核，有时被称为 HPU (Habana 处理单元)，由一群张量处理内核(TPC)和可配置矩阵数学引擎(GEMM)组成。GEMM 擅长矩阵乘法，而非线性和元素运算在 TPC 上运行。这种异构性使 Gaudi 能够在各种各样的工作负载上实现高效率。通过有效地平衡资源之间的负载，可以实现最大的利用率。
*   **高规模训练**:架构设计特别注重高迪处理器之间的数据吞吐速度。这使得高迪能够[展示](https://habana.ai/mlperf-ai-training-benchmark-habana-gaudi-performance-and-scale-results/)当将训练扩展到多核时，出色的、接近线性的结果。
*   **框架支持**:SynapseAI API 包括对 [TensorFlow](https://docs.habana.ai/en/latest/Tensorflow_User_Guide/Tensorflow_User_Guide.html) 和 [PyTorch](https://docs.habana.ai/en/latest/PyTorch_User_Guide/PyTorch_User_Guide.html) 的支持，这是目前使用的最流行的两个机器学习框架。它还支持 [Horovod](https://horovod.readthedocs.io/en/stable/) ，这是一个流行的分布式培训框架。这些产品使得现代机器学习开发人员非常容易为 Gaudi 创建模型。
*   **丰富的模型花园**:Habana SW 产品包括各种各样的[参考模型](https://github.com/HabanaAI/Model-References)——已经移植并优化用于在 Gaudi 上运行的流行模型的实现。
*   **定制内核创建**:与其他一些专用 AI ASICs 相反，SynapseAI SW 套件包括用于[实现定制 Gaudi (TPC)内核的工具](https://docs.habana.ai/en/latest/TPC_User_Guide/TPC_User_Guide.html#tpc-user-guide)。与用于 GPU 内核开发的 [CUDA 工具包](https://developer.nvidia.com/cuda-toolkit)类似，这一功能使用户能够设计、开发和优化专门针对其工作负载需求的低级操作。
*   **运行并行试验**:SW 套件支持在 DL1 实例上的八个底层 Gaudi 加速器的不相交子集上运行并行工作负载。一种方法是使用训练管弦乐队，比如 kubernetes。我们将演示如何利用这种能力来并行化超参数调优试验。
*   **CPU 与 Gaudi 计算比率**:在一个标准的机器学习项目中，计算将在 CPU 资源和 AI 加速器之间分配。通常，输入数据预处理管道将在 CPU 上运行，模型计算图(向前向后传递)将在 AI 加速器上运行。在理想情况下*所有的*培训资源都将被充分利用。但是最大化利用率对于 AI 加速器资源是最重要的，这些资源通常是系统中最强大和最昂贵的资源。在某些情况下，你可能会发现 CPU 跟不上人工智能加速器的速度，导致 CPU 瓶颈和人工智能加速器的利用不足。CPU 瓶颈的可能性由整体 CPU 计算能力和整体加速器计算能力之间的比率决定。DL1 实例具有相对较高的 CPU 与 Gaudi 计算比率，从而降低了 CPU 瓶颈和加速器利用不足的可能性。事实上，DL1 实例包含与 [p4d.24xlarge](https://aws.amazon.com/ec2/instance-types/p4/) EC2 实例相同的 CPU 计算能力(96 个第二代英特尔至强可扩展 CPU 内核)，尽管其 AI 加速器 A100 被认为比 Habana Gaudi 更强大。
*   **批量灵活性**:与其他人工智能加速器相反，Habana Gaudi 能够在很大的批量范围内实现高利用率，而其他人工智能加速器可能需要特别高的批量培训，以便充分利用硬件的价格优势。这使得高迪成为可能无法适应大批量生产的模型的可行选择。

以下是您应该考虑的几点:

*   **API 限制**:确保仔细阅读 SynapseAI 软件套件的限制。正如在撰写本文时在[文档](https://habana.ai/mlperf-ai-training-benchmark-habana-gaudi-performance-and-scale-results/)中明确指出的那样，“并非所有的模型在 Gaudi 上都得到支持”。如果您不能在 DL1 上编译您的模型，您可以尝试调整您的模型以符合 API 支持，或者探索[创建定制操作](https://docs.habana.ai/en/latest/TensorFlow_CustomOp_API/page_index.html#)的选项。或者，您可以简单地报告您的发现并跟踪[synapse ai 版本的未来版本](https://github.com/HabanaAI/synapseai-roadmap)。
*   **每个 DL1 实例八个加速器**:在撰写本文时，唯一基于 Gaudi 的 AWS 实例产品包括八个加速器。这意味着，为了充分利用这个系统，你需要或者运行并行实验，或者将你的训练分布在所有的加速器上。如果这两个选项都不适合您，那么 DL1 实例可能不是您的最佳选择。
*   **性价比**:你可能会发现你的型号*是*支持的，但是你最初的试用并没有达到你预期的性价比。在这种情况下，您可以使用 Habana [优化指南](https://docs.habana.ai/en/latest/Model_Performance_Optimization/Model_Performance_Optimization_in_Habana_Gaudi.html)、[性能分析指南](https://docs.habana.ai/en/latest/Profiler_Guide/Profiler_User_Guide.html)、[自定义 op 创建指南](https://docs.habana.ai/en/latest/TensorFlow_CustomOp_API/page_index.html#)和其他资源来提高性能。如果尽管你尽了一切努力，你还是不能达到足够的性价比，你最好的选择可能是报告你的发现，并等待 SynapseAI 套件的更新版本。

# 2.调整您的模型以在 DL1 上运行

在本节中，我们将讨论在 DL1 实例上启动和运行模型所需的一些步骤。这些都基于 Habana 官方文档，尤其是 [TensorFlow 迁移指南](https://docs.habana.ai/en/latest/Migration_Guide/Migration_Guide.html#porting-a-simple-tensorflow-model-to-gaudi)。更多详情请见此处。

## 系统设置

有多种方法可以启动一个 [Amazon EC2](https://aws.amazon.com/ec2/) 实例和[建立一个 DL1 运行时环境](https://aws.amazon.com/ec2/)。最适合您的选择将取决于您/您组织的整体云架构。

## 加载 Habana 模块

调整 TensorFlow 模型以在 Habana Gaudi 上运行只需要两行代码，如下面摘自 [Habana TensorFlow Hello World 示例](https://github.com/HabanaAI/Model-References/blob/1.2.0/TensorFlow/examples/hello_world/example.py)的代码片段所示。

```
import tensorflow as tf
**from habana_frameworks.tensorflow import load_habana_module
load_habana_module()**(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(10),
])
loss = tf.keras.losses.SparseCategoricalCrossentropy(
                                           from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])model.fit(x_train, y_train, epochs=5, batch_size=128)
```

运行脚本时，确保使用适当的 python 可执行文件。这取决于您选择的设置，如这里的所示。

## 检查设备放置

当运行你的脚本时，你要做的第一件事就是验证你的模型确实运行在 Gaudi 加速器上。高迪运行时环境包括 [*hl-smi* 工具](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html)，它报告八个高迪内核的资源利用情况。它的外观和感觉类似于用于 GPU 的 [*nvidia-smi* 工具](https://developer.nvidia.com/nvidia-system-management-interface)。您可以使用该工具来验证运行您的训练脚本是否增加了第一个 Gaudi 内核的内存和计算资源。
除了断言正在使用 Gaudi 内核之外，您还会希望确保您的训练计算图的所有操作都在 Gaudi 上运行，而不是在 CPU 上运行。Gaudi 不支持的操作被卸载到 CPU 上，这可能会导致很高的事务开销，并大大降低您的培训速度。
检查器件布局的一种方法是使用[*TF . debugging . set _ log _ device _ placement*](https://www.tensorflow.org/api_docs/python/tf/debugging/set_log_device_placement)函数。当设置为 *True* 时，该例程将生成一个日志，记录程序中所有 TensorFlow 操作的设备位置。高迪核心在 TensorFlow 中注册为“HPU”设备。如果你所有的训练任务都分配给“HPU ”,你的情况就很好。如果你的任何训练运算被分配给“CPU ”,你可能需要调整你的计算图，我们将在下一小节讨论。
此处记录了分析 op 布局的另一种方法[。](https://docs.habana.ai/en/latest/Model_Performance_Optimization/Model_Performance_Optimization_in_Habana_Gaudi.html#place-ops-in-hpu)

**示例—使用不支持的数据类型**:在下面的代码片段中，我们添加了对[*TF . math . arg max*](https://www.tensorflow.org/api_docs/python/tf/math/argmax)*的调用，后跟 [*tf.equal*](https://www.tensorflow.org/api_docs/python/tf/math/equal) 。*

```
*import tensorflow as tf
from habana_frameworks.tensorflow import load_habana_module
load_habana_module()
**tf.debugging.set_log_device_placement(True)** (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()x_train, x_test = x_train / 255.0, x_test / 255.0model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(10),
            tf.keras.layers.Lambda(lambda x: 
               tf.where(tf.expand_dims(**tf.equal**(
                  **tf.math.argmax**(x,axis=-1),2),-1),
                  x,
                  tf.math.square(x)))])
loss =  tf.keras.losses.SparseCategoricalCrossentropy(
                                          from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=128)*
```

*[*tf.math.argmax*](https://www.tensorflow.org/api_docs/python/tf/math/argmax) 的默认输出为 tf.int64 类型，然而，截至本文撰写之时，int64 并不在 HPU 支持的[数据类型列表中。结果是](https://www.google.com/search?q=%D7%94%D7%A8%D7%98%D7%95%D7%9D+13+%D7%94%D7%A8+%D7%97%D7%95%D7%A6%D7%91%D7%99%D7%9D&rlz=1C1GCEB_enIL904IL904&oq=%D7%94%D7%A8%D7%98%D7%95%D7%9D+13&aqs=chrome.2.69i57j0i19j0i19i22i30l3.8084j0j7&sourceid=chrome&ie=UTF-8)[*TF . equal*](https://www.tensorflow.org/api_docs/python/tf/math/equal)*操作将在 CPU 上运行。器件放置调试日志将包括以下几行:**

```
**sequential/lambda/ArgMax: (ArgMax): /job:localhost/replica:0/task:0/device:HPU:0
sequential/lambda/Equal: (Equal): /job:localhost/replica:0/task:0/**device:CPU**:0**
```

**在这个玩具示例中，修复方法是简单地将[*TF . math . arg max*](https://www.tensorflow.org/api_docs/python/tf/math/argmax)*的 *output_type* 设置为 tf.int32。***

## **模型调整**

**在 HPU 上训练模型可能需要对模型进行一些更改。所需的模型调整在复杂程度上会有所不同。在某些情况下，它们就像指定底层数据类型一样简单，如上例所示。在其他情况下，您可能需要修改数据流或替换操作序列。例如，SynapseAI 1.2.0 版本的[发行说明](https://docs.habana.ai/en/latest/Release_Notes/GAUDI_Release_Notes.html#tensorflow-known-issues)中包含了“Gaudi 目前不支持 [tf.cond](https://www.tensorflow.org/api_docs/python/tf/cond) 和 [tf.while_loop](https://www.tensorflow.org/api_docs/python/tf/while_loop) 等控制流操作”的限制。在撰写本文时，Gaudi 不支持的另一个操作示例是 [tf.numpy_function](https://www.tensorflow.org/api_docs/python/tf/numpy_function) ，这是一个允许在计算图中包含任意 python 代码(例如用于度量计算)的例程，有时用于绕过本机 TensorFlow API 施加的限制。如果您的模型包含这样的操作，您将需要设计一个替代流程，或者接受让它们在 CPU 上运行的性能损失。**

**您需要为高迪进行模型调整的三个重要资源是 HPU 支持的 TensorFlow 操作的[列表、](https://docs.habana.ai/en/latest/TensorFlow_Operators/TF_Operators.html#tensorflow-operators)[模型参考目录](https://github.com/HabanaAI/Model-References)和[定制内核创建指南](https://docs.habana.ai/en/latest/TPC_User_Guide/TPC_User_Guide.html#tpc-user-guide)。**

**[**支持的 TensorFlow 操作**](https://docs.habana.ai/en/latest/TensorFlow_Operators/TF_Operators.html#tensorflow-operators) :使用本文档在图形中查找高迪对操作的支持。**

**[**模型参考目录**](https://github.com/HabanaAI/Model-References) :哈瓦那高迪产品包括各种常见机器学习模型架构的参考实现。这些实现已经过修改和调整，可以在高迪上运行。如果您正在使用其中一种实现的架构，您应该认为自己很幸运。但是，即使您使用的是不同的模型架构，模型参考目录中也可能包含您认为有用的计算层或计算块。例如，如果您正在研究变压器架构，最好查看一下变压器块[和](https://github.com/HabanaAI/Model-References/blob/master/TensorFlow/nlp/transformer/layers/transformer_layers.py)的高迪特定实现，要么照原样使用，要么深入了解如何对您自己的变压器块进行适当的调整。**

**[**自定义内核创建**](https://docs.habana.ai/en/latest/TPC_User_Guide/TPC_User_Guide.html#tpc-user-guide):Habana Gaudi 区别于市场上其他一些专用 AI ASICs 的特性之一是其可由最终用户编程。Habana 提供了关于[创建定制 HPU 内核](https://docs.habana.ai/en/latest/TPC_User_Guide/TPC_User_Guide.html#tpc-user-guide)和[用 TensorFlow 操作符包装它们](https://docs.habana.ai/en/latest/TensorFlow_CustomOp_API/page_index.html)的全面指南。也可以看看[这个详细的例子](https://github.com/HabanaAI/Model-References/tree/master/TensorFlow/examples/custom_op)和这个[视频教程](https://vimeo.com/543215096)。**

****预处理流水线**:需要注意的是，虽然将你的模型移植到 Gaudi 可能需要改变运行在加速器上的训练计算图，但是*不*需要调整运行在 CPU 内核上的预处理流水线。这与其他一些人工智能加速器相反，正如我们在过去看到的。**

## **关于 DL1 的分布式培训**

**当然，要充分利用 DL1 资源，仅将其移植到单个 HPU 上运行是不够的；您将希望利用所有八个 hpu。一种方法是使用[数据分布式训练](https://docs.habana.ai/en/latest/Tensorflow_Scaling_Guide/TensorFlow_Gaudi_Scaling_Guide.html)在所有八个 hpu 上并行训练。Habana Gaudi 软件堆栈提供了两种实施分布式培训的机制。第一个使用了 Habana Gaudi 流行的 [Horovod](https://horovod.readthedocs.io/) 框架的具体实现。第二个使用了一个定制的[TF . distribute . strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy)API 实现。[分布式培训指南](https://docs.habana.ai/en/latest/Tensorflow_Scaling_Guide/TensorFlow_Gaudi_Scaling_Guide.html)包括两个选项的详细信息。
选择 Horovod 选项的优势在于，如果您已经使用 Horovod for GPUs 实施了分布式培训，则无需更改代码即可在 hpu 上运行。您需要做的只是验证 [habana-horovod 套件](https://github.com/HabanaAI/Setup_and_Install#check-tfhorovod-habana-packages)的正确安装。事实上，与其他定制 AI ASIC 产品相比，Horovod 支持是 Habana 产品的优势之一。**

**请注意，在撰写本文时，遵守导入命令的特定顺序很重要，如摘自 [Habana Gaudi 文档](https://docs.habana.ai/en/latest/Tensorflow_Scaling_Guide/TensorFlow_Gaudi_Scaling_Guide.html#example-scale-up-within-a-server)的这段代码所示。**

```
**import tensorflow as tf
from habana_frameworks.tensorflow import load_habana_module
***# ensure that load_habana_module() needs to be called before
# import horovod***
load_habana_module()
import horovod.tensorflow.keras as hvd
*#Initialization of Horovod.* 
hvd.init()**
```

**Horovod 框架也可用于在[多个 DL1 实例上并行训练](https://docs.habana.ai/en/latest/AWS_Distributed_Training_Multiple_DL1/AWS_Distributed_Training_Multiple_DL1.html)。**

# **3.在 DL1 上优化您的模型性能**

**此时，您应该能够在 DL1 上成功运行一个培训周期。接下来是性能分析和优化的关键步骤。正如我们在[上一篇文章](/tpu-training-6eb84100d138)中强调的，人工智能加速器的好坏取决于它为性能分析和优化提供的工具。如果你不能分析和优化性能，你就不能充分利用人工智能芯片。**

**Habana 为性能分析和优化提供了三个重要的资源:最佳实践列表(T0)、T2 性能优化指南(T3)和 T4 性能分析器(T5)。应该详细研究这些指南并经常参考。我们将对每一个提供一些简短的评论。**

## **[高迪培训的最佳实践](https://docs.habana.ai/en/latest/Best_Practices_for_Model_Training/Best_Practices_for_Model_Training_on_Gaudi.html)**

**这个页面包括关于高迪培训的一般(框架不可知)指南。虽然这个列表很紧凑(在撰写本文时只有 7 点)，但是每一项都可以对模型性能产生有意义的影响。有两点值得一提:**

1.  ****动态形状**:不鼓励使用返回未确定大小的形状的操作符。参见我们之前的[帖子](/tpu-training-6eb84100d138)，在其中我们演示了如何替换使用一个这样的函数，[*TF . boolean _ mask*](https://www.tensorflow.org/api_docs/python/tf/boolean_mask)。**
2.  ****张量形状**:有些项目建议张量形状的选择(如批量大小和特征/通道数量)遵循一定的公式。这对于一个专门的人工智能加速器(或者任何芯片，就此而言)来说并不罕见。正如我们在第 1 节中提到的，其他 AI 芯片需要使用大批量来最大化利用率。在这方面，高迪为用户提供了更大的自由/灵活性。**

## **[性能优化指南](https://docs.habana.ai/en/latest/Model_Performance_Optimization/Model_Performance_Optimization_in_Habana_Gaudi.html)**

**本页主要关注 TensorFlow 框架的优化指南。其中一个建议是利用高迪内置的对 [bfloat16](https://docs.habana.ai/en/latest/Tensorflow_User_Guide/Tensorflow_User_Guide.html#tf-mixed-precision-training) 的支持，使用 [TensorFlow 的混合精度 API](https://www.tensorflow.org/guide/mixed_precision)。在训练期间使用低精度浮点(16 位)可以潜在地减少存储器使用和训练步骤时间。有两种低精度浮点格式，float16 和 [bfloat16](https://cloud.google.com/tpu/docs/bfloat16) ，bfloat16 具有许多属性，使其成为机器学习的首选格式。
需要注意的是，虽然使用混合精度时内存利用率的降低几乎是可以保证的，但步长时间的降低以及模型的收敛能力仍需验证。**

## **[性能分析器](https://docs.habana.ai/en/latest/Profiler_Guide/Profiler_User_Guide.html)**

**[Profiler 用户指南](https://docs.habana.ai/en/latest/Profiler_Guide/Profiler_User_Guide.html)包含关于 SynapseAI Profiler 的大量文档，包括其设置、执行和分析工具。还有这个有用的[视频教程](https://vimeo.com/532336520)。
正如我们在之前的帖子中详细讨论的那样(例如这里的[和这里的](/tensorflow-performance-analysis-314b56dceb59)和[和](/overcoming-data-preprocessing-bottlenecks-with-tensorflow-data-service-nvidia-dali-and-other-d6321917f851))，描述您的培训绩效对于最大限度地利用您的培训资源、加速您的培训和降低培训成本至关重要。**

**Habana 性能分析器的主要工件是[分析图](https://docs.habana.ai/en/latest/Profiler_Guide/Profiler_User_Guide.html#viewing-instructions)。与 [TensorBoard 跟踪查看器](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras#debug_performance_bottlenecks)类似，该图显示了不同系统资源上发生的不同事件的时间线，特别是 DMA、MME 和 TPC。下面是一些您可能遇到的资源使用模式的例子，以及从中可以学到什么:**

1.  ****数据输入管道上的瓶颈**:HPU 上训练步骤之间的大间隔可能表示 HPU 在等待 CPU 传递训练数据时处于空闲状态。在这种情况下，您应该致力于优化您的数据输入管道。(此处见[。)](/overcoming-data-preprocessing-bottlenecks-with-tensorflow-data-service-nvidia-dali-and-other-d6321917f851)**
2.  ****将操作卸载到 CPU 上**:在训练步骤中间的 HPU 空闲与增加的 DMA 活动相结合，可能表明一些图形操作正在被卸载到 CPU 上。在这种情况下，您应该重新检查图形操作的设备位置。**
3.  ****MME 利用率和 HPC 利用率之间的不平衡**:您可能会在训练步骤中发现 MME 空闲而 HPU 非常繁忙的时段，反之亦然。在这种情况下，您可以通过改善资源之间的负载平衡来减少步骤时间。这可以通过编程/设计等效的设备特定内核来实现，如这里建议的。**

**在撰写本文时，Habana Gaudi 性能分析器的使用需要 Gaudi 特定的配置步骤和工具。我们的预期是，即将发布的版本将包括对分析器使用的改进，包括将其完全集成到 TensorFlow 分析 API 和 TensorBoard 中。**

# **4.调整您的模型以收敛于 DL1**

**此时，您的模型已经被调整到您满意的程度，您可以开始训练了。您可能需要对模型进行一些更改，这需要重新调整超参数以确保模型收敛。此类更改可能包括替换某些操作、更改控制流或更改底层数据类型。即使你没有对你的模型做任何改变，你也应该确保你的训练收敛在新的 AI ASIC 上。这是因为不同的硬件加速器以不同的方式实现，并且可能在它们的行为中表现出微小的数值差异。在一个 ASIC 上的收敛并不保证在另一个上的收敛。**

# **示例 DL1 上的超参数调谐**

**在本例中，我们展示了如何利用八个 HPU 内核在超参数调整环境下运行八个并行实验。超参数调整指的是为您的模型搜索最佳[超参数](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))的任务。 [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) 是一个流行的 python 库，用于自动化超参数调整，支持各种最先进的优化算法。虽然默认版本只将 CPU 和 GPU 视为可能的“培训资源”，但它也可以扩展为使用 HPU。在下面的代码块中，我们通过将 hpu 注册为 GPU 来演示一种相当简单的方法。在下面的代码片段中，它基于 Ray Tune 记录的 [mnist 示例](https://docs.ray.io/en/latest/tune/examples/tune_mnist_keras.html)，我们突出显示了两个必需的更改:**

1.  **通过[光线初始化](https://docs.ray.io/en/latest/package-ref.html#ray-init)命令显式注册八个 GPU。这是必需的，因为当前版本的 Ray 不能识别 HPU 加速器。**
2.  **进入列车功能后，根据 *CUDA_VISIBLE_DEVICES* 环境变量的值设置 *HABANA_VISIBLE_DEVICES* 环境变量。这将确保每个进程在单独的 HPU 上运行。**

```
**import os
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCallbackdef train_mnist(config):
 **os.environ['HABANA_VISIBLE_DEVICES'] = \
        os.environ['CUDA_VISIBLE_DEVICES']**    import tensorflow as tf
    from habana_frameworks.tensorflow import load_habana_module
    from tensorflow.keras.datasets import mnist
    from filelock import FileLock
    load_habana_module() with FileLock(os.path.expanduser("~/.data.lock")):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0 model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(10)])
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=config['lr'])
    model.compile(optimizer=optimizer,
                  loss=loss, 
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=128,
              verbose=0, validation_data=(x_test, y_test),
              callbacks=[TuneReportCallback({
                 "mean_accuracy": "accuracy"})])def tune_mnist(num_training_iterations):
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20)
 **# explicitly init ray with number of accelerators set to 8** 
 **ray.init(num_gpus=8)**    analysis = tune.run(
        train_mnist,
        name="exp",
        scheduler=sched,
        metric="mean_accuracy",
        mode="max",
        stop={
            "mean_accuracy": 0.9,
            "training_iteration": num_training_iterations
        },
        num_samples=8,
        resources_per_trial={
            "cpu": 12,
            "gpu": 1
        },
        config={
            "lr": tune.uniform(0.001, 0.1),
        })
    print("Best hyperparameters found were: ", analysis.best_config)if __name__ == "__main__":
    tune_mnist(num_training_iterations=1000)**
```

# **摘要**

**新的训练实例选项的可用性总是令人兴奋的消息，尤其是当它基于专用的 AI ASIC 时。为 DL1 实例提供动力的 Habana Gaudi 产品似乎具备了当今市场上其他人工智能加速器的所有有价值的替代物。特别是，其附带的软件堆栈在设计和优化机器学习工作负载方面为用户提供了极大的灵活性。与此同时，重要的是要记住，Habana Gaudi 相对较新，因此应该以适当的心态来对待。达到你的最佳结果可能需要耐心和韧性。但是它值得潜在的回报。在我们自己的车型上，性价比的增长达到甚至超过了公布的 40%大关。**

**这篇文章仅仅介绍了 DL1 实例培训的几个方面。请务必参考丰富的在线文档以了解更多详细信息。**

**在这篇博文的研究过程中，我发现“高迪”在德语中是“有趣”的意思。我想不出更好的方式来描述我迄今为止在 DL1 上的经历。我所能希望的是，你也有“高迪”和你的高迪。**