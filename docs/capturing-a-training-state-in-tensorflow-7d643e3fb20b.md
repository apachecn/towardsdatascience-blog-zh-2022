# 在 TensorFlow 中捕获训练状态

> 原文：<https://towardsdatascience.com/capturing-a-training-state-in-tensorflow-7d643e3fb20b>

## 如何调试培训项目中出现的 NaN 损失

![](img/28814aa1cd28ebd40c478400212cef9d.png)

艾伦·埃梅里在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

在[之前的帖子](/debugging-in-tensorflow-392b193d0b8)中，我们试图为 TensorFlow 中的调试任务提供一些支持，这项任务通常很困难，有时不可能，而且总是令人抓狂。这个博客包括了一个描述，我认为这是现代机器学习开发者潜在痛苦的终极例子——NaN 损失的突然出现。在我们所说的被诅咒的场景中，在一段时间(可能是延长的)成功的模型收敛之后，模型损失突然被报告为*NaN*——“不是一个数字”，并且模型状态(权重)变得灾难性的和不可修复的被摧毁。我们提出了一种创新的方法，用于在权重被破坏之前捕捉模型训练的精确状态。在假设训练步骤不包括随机性(例如，没有丢失层)的情况下，该方法允许以可靠和有效的方式再现和调查导致 *NaN* 的问题。(我们将在下面的训练步骤中回到随机性的可能性。)

从那时起，我们开发了一种额外的技术来捕获模型的训练状态，这种技术可以扩展到原始方法没有解决的场景。这篇文章的目的是分享这种技术，并与它的前辈进行比较。

我们将从简要回顾问题和以前的解决方案开始。为了简洁起见，我将不再详述调试 *NaN* 损失的独特挑战和我们提出的原始解决方案，而是建议您阅读[以前的帖子](/debugging-in-tensorflow-392b193d0b8)。接下来，我们将描述原始解决方案的一些局限性，并介绍一种解决这些局限性的新技术。最后，我们将简要演示一个使用这种技术的示例调试流程(基于一个真实的故事)。

这篇文章将包括几个代码块，演示如何对我们讨论的解决方案进行编程。代码基于 TensorFlow 版。和往常一样，你应该对照你读到这篇文章时的最新版本来验证代码和我们提出的一般论点。

# 调试 NaN 损失可能很困难

虽然调试通常很难，但有许多原因使得调试 TensorFlow 中出现的 *NaN* 损失尤其困难。

## 符号计算图的使用

TensorFlow 包括两种执行模式，[急切执行](https://www.tensorflow.org/api_docs/python/tf/config/run_functions_eagerly)和[图形执行](https://www.tensorflow.org/guide/intro_to_graphs)。虽然在急切执行中训练时更容易调试，但是图模式有许多[的好处](https://www.tensorflow.org/guide/intro_to_graphs#the_benefits_of_graphs)，使得训练明显更有效。图形模式的缺点是很难调试。具体来说，在图形模式中，您预先构建模型，并将其发送到训练设备(例如 GPU)。在训练过程中，您不能随意访问和测量计算图上的张量。显而易见的结论是，我们应该总是在图模式下训练，在急切执行模式下调试。然而，这种策略的先决条件是能够容易地捕获精确的训练状态以用于再现和调试。

## 重现训练状态的挑战

为了完全重现一个训练步骤，我们需要知道三件事:模型的状态、精确的数据输入批次，以及控制训练步骤中包含的任何随机性的种子。

**捕获模型状态** : TensorFlow 提供了将其模型权重和优化器变量的值保存到文件中的工具(例如[TF . keras . model . save _ weights](https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights))。在培训期间定期捕获这些内容是一个很好的做法(例如使用 [tf.keras ModelCheckpoint 回调](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint))。
理想情况下，我们希望能够捕获导致 *NaN* 损失的精确模型状态。这将简化和缩短复制。不幸的是，在典型的场景中，一旦报告了 NaN 的损失，模型状态就已经被破坏了。在我们之前的帖子中，我们展示了一种在权重被不可逆转地混淆之前发现 NaN 损失的方法。在这篇文章中，我们也将采用类似的技术。

**捕获数据输入:**由于[tensor flow TF . data . datasets](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)的顺序性质，一旦训练批次通过训练步骤，我们就不再能够访问它。在我们的上一篇文章中，我们演示了一种捕获训练批次以进行复制的机制。

**捕捉控制随机性的种子**:随机性是机器学习训练算法的固有属性，往往是它们成功的关键。与此同时，随机性可能是复制 bug 的噩梦。克服这个挑战的一个方法是显式地设置和捕获决定随机性的特定种子(例如[TF . random . generator . from _ seed](https://www.tensorflow.org/api_docs/python/tf/random/Generator#from_seed))。

## 我们先前提案的局限性

在我们之前的文章中，我们提出了一种方法，包括定制 [tf.keras.model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 类的 [train_step](https://www.tensorflow.org/api_docs/python/tf/keras/Model#train_step) 和 [make_train_function](https://www.tensorflow.org/api_docs/python/tf/keras/Model#make_train_function) 例程，以便:

1.  将输入批处理保存为一个渴望的(可访问的)张量，
2.  在将梯度应用于模型权重之前，测试 *NaN* 值的梯度，
3.  终止训练并保存上次输入批次和当前权重。

这种方法有两个明显的局限性:

**在分布式训练中捕获状态**:在其当前形式下，该方法不支持使用 [tf.distribute](https://www.tensorflow.org/api_docs/python/tf/distribute) 分布策略捕获在多个内核上运行的训练会话的状态。

**捕捉 TPUs 上的状态**:该方法要求部分训练步骤在 [tf.function](https://www.tensorflow.org/api_docs/python/tf/function) 范围之外运行，以捕捉输入数据。TPU 不支持此方案。

在下一节中，我们将描述一种克服这些限制的新技术。

# 捕获训练状态的新建议

我们的新提案包括四个部分:

1.一辆 [tf.keras.models.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 带定制 [train_step](https://www.tensorflow.org/api_docs/python/tf/keras/Model#train_step) 。

2.一个定制的 TensorFlow 输入数据捕获层，我们将其附加到输入数据管道的末端。

3.将数据集[预取](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch)值设置为 0(以确保最后一批数据不会被覆盖)。

4.标准的[TF . keras . callbacks . termin ate onnan](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TerminateOnNaN)回调。

## 自定义训练步骤

我们定制训练步骤来测试 *NaN* 梯度，然后将它们应用于模型权重。如果发现了一个 *NaN* 梯度，我们明确地将*损失*度量设置为 *NaN* 并且*而不是*更新权重。 *NaN* 损失将由[TF . keras . callbacks . termin ate onnan](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TerminateOnNaN)回调识别，培训将停止。与我们之前的解决方案相反，定制训练步骤完全在 [tf.function](https://www.tensorflow.org/api_docs/python/tf/function) 范围内。下面的代码块演示了自定义训练步骤。突出显示了[默认训练步骤](https://github.com/keras-team/keras/blob/v2.8.0/keras/engine/training.py#L831-L864)的主要变化。

```
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapterclass DebugModel(tf.keras.models.Model):
  def train_step(self, data):
    data=data_adapter.expand_1d(data)
    x, y, sample_weight=data_adapter.unpack_x_y_sample_weight(data)
    # Run forward pass.
    with tf.GradientTape() as tape:
      y_pred=self(x, training=True)
      loss=self.compiled_loss(
         y, y_pred, 
         sample_weight, regularization_losses=self.losses)
    grads_and_vars=self.optimizer._compute_gradients(
              loss, var_list=self.trainable_variables, tape=tape)
    grads_and_vars=self.optimizer._aggregate_gradients(
              grads_and_vars)
    self.compiled_metrics.update_state(y, y_pred, sample_weight)
    # Collect metrics to return
    return_metrics = {}
    for metric in self.metrics:
      result = metric.result()
      if isinstance(result, dict):
        return_metrics.update(result)
      else:
        return_metrics[metric.name] = result
 **def t_fn():
      # if any of the gradients are non, set loss metric to NaN       
      return tf.constant(float('NaN'))
    def f_fn():
      # if all gradients are valid apply them
      self.optimizer.apply_gradients(
               grads_and_vars,
               experimental_aggregate_gradients=False)
      return return_metrics['loss']
    grad_nan=[tf.reduce_any(tf.math.is_nan(g)) for g,_ in  
              grads_and_vars if g is not None]
    grad_nan=tf.reduce_any(grad_nan)
    return_metrics['loss'] = tf.cond(grad_nan,
                                     t_fn,
                                     f_fn)**    return return_metrics
```

## 数据输入捕获层

在[之前的帖子](/the-tensorflow-keras-summary-capture-layer-cdc436cb74ef)中，我们介绍了一种独特的方法，使用自定义的 [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) 来捕捉图张量的值，该方法将张量值分配给可访问的渴望张量。我们将这种张量捕捉层应用于数据输入管道的末端(在[预取](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch)之前)，以便在输入数据批次进入训练步骤之前捕捉它们。

```
class TensorCaptureLayer(tf.keras.layers.Layer):
  def __init__(self, shape, dtype, **kwargs):
    super(TensorCaptureLayer, self).__init__(dtype=dtype, **kwargs)
    self.record_tensor_var = tf.Variable(shape=shape,    
             initial_value=tf.zeros(shape=shape, dtype=dtype),
             validate_shape=False,
             dtype=dtype,
             trainable=False)
  def call(self, inputs, **kwargs):
     self.record_tensor_var.assign(inputs)
     return inputs# Here we demonstrate a case where the input consists of a 
# single batch of 32 frames and a single batch of 32 labels
class DataInputCaptureLayer(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(DataInputCaptureLayer, self).__init__(**kwargs)
    self.frame_input = TensorCaptureLayer([32,512,512,3],tf.float32)
    self.label_input = TensorCaptureLayer([32],tf.int32)
  def call(self, x, y, **kwargs):
    self.frame_input(x)
    self.label_input(y)
    return x,y
```

## 将数据集预取设置为零

[tf.data.Dataset.prefetch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch) 例程是一种常用的技术，用于简化训练流程和最大化资源利用率(参见[此处](https://www.tensorflow.org/guide/data_performance#prefetching))。但是，如果我们将预取设置为一个大于零的值，我们将冒着用即将到来的批处理而不是当前批处理填充 DataInputCaptureLayer 的风险，破坏我们捕获导致 *NaN* 丢失的输入批处理的能力。

## 使用[终端启动](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TerminateOnNaN)回调

我们使用标准的 [TerminateOnNaN](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TerminateOnNaN) 回调来监控损失，如果是 *NaN* 则终止训练。

下面的代码块演示了我们是如何把它们放在一起的。

```
dataset=...
capture_layer=DataInputCaptureLayer()
dataset=dataset.map(lambda x,y:capture_layer(x,y))
dataset=dataset.prefetch(0)strategy=... 
with strategy.scope():
  model = DebugModel(inputs=..., outputs=...)
model.compile(loss=..., optimizer=...)
model.fit(dataset, epochs=..., steps_per_epoch=...,
          callbacks=[tf.keras.callbacks.TerminateOnNaN()])if model.stop_training: # set by callback in case of Nan
  # capture the model state
  model.save_weights('model_weights.ckpt')
  # capture the last data batch
  np.save('frm.npz',capture_layer.frame_input.record_tensor.numpy())
  np.save('lbl.npz',capture_layer.label_input.record_tensor.numpy())
```

## 再现*楠*的失落

有了现在的训练状态，我们就可以开始复制 NaN 的失败。使用[TF . keras . model . load _ weights](https://www.tensorflow.org/api_docs/python/tf/keras/Model#load_weights)例程可轻松恢复模型状态，并使用自定义层将输入数据注入管道，如下所示:

```
class InputInjectionLayer(Layer):
  def __init__(self, **kwargs):
    super(InputInjectionLayer, self).__init__(**kwargs)
    self.frames=tf.convert_to_tensor(np.load(frm.npz))
    self.labels=tf.convert_to_tensor(np.load(lbl.npz))
  def call(self, x, y, **kwargs):
    return self.frames,self.labelsdataset=...
inject_layer=InputInjectionLayer()
dataset=dataset.map(lambda x,y:inject_layer(x,y))
dataset=dataset.prefetch(0)strategy=... 
with strategy.scope():
  model = DebugModel(inputs=..., outputs=...)
model.compile(loss=..., optimizer=...)
model.fit(dataset, epochs=1, steps_per_epoch=1)
```

## 与以前解决方案的比较

与我们之前提出的解决方案相反，该解决方案可用于调试在分布式培训设置中遇到的 *NaN* 丢失，也可用于 TPU。

值得指出的另一个区别是两种解决方案所需的资源。在第一种解决方案中，数据批次作为训练步骤的一部分存储在训练工作者的存储器中(例如 GPU)。在第二种解决方案中，数据批次被存储为在主机 CPU 上运行的数据输入管道的一部分。根据您的工作负载，数据批处理可能需要大量内存，存储数据的资源可能会影响您构建模型的方式(例如，最大批处理大小)。

最后，不同的解决方案可能会对您的训练步骤时间产生不同的影响。每个解决方案的性能损失在很大程度上取决于模型的属性。例如，如果您有一个 CPU 瓶颈，那么添加数据输入捕获层可能会比我们原来的解决方案有更高的代价。另一方面，如果您的 CPU 没有得到充分利用，损失可能会更低。对于我们的模型(通常相对较大)，我们没有注意到两种解决方案的惩罚有显著差异，两者都很低(<10%).

## Solution Extensions

A number of enhancements can be made to this solution to extract more debug information and/or extend its coverage to additional use cases. Here we will note a few of them.

**使用 tf.print** 提取更多信息当检测到 *NaN* 坡度时，我们上面实现的定制训练步骤只不过是将损失设置为 *NaN* 。但是，我们可以使用 [tf.print](https://www.tensorflow.org/api_docs/python/tf/print) 实用程序来增强收集和打印附加诊断信息的功能。例如，我们可以打印梯度张量的完整列表，以及它们是否包含 *NaN* 值。这可能会为问题的根源提供一些线索。值得注意的是(在撰写本文时), TPU 还不支持 tf.print，因此尽管捕捉更多信息仍然是可能的，但它需要更多的创造力(例如，使用自定义指标)。

**支持备选数据加载方案** 在我们分享的示例代码中，我们假设数据分发由 [tf 分发策略](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy)管理，并且使用默认的数据加载机制。但是，该解决方案可以很容易地扩展到其他数据加载方案。例如，如果 [Horovod](https://horovod.ai/) 用于数据分发，那么每个训练核心(例如 GPU)都有其自己的单独数据集，所需要的就是让每个 [Horovod](https://horovod.ai/) 进程将其输入数据保存(并注入)到唯一的文件路径。如果 [tf 分配策略](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy)与[distribute _ datasets _ from _ function](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy#distribute_datasets_from_function)一起使用，则单个数据集管道可以为多个训练核提供数据。在这种情况下，可以使用队列机制来增强 InputDataCaptureLayer，以捕获多个数据批次(根据单个数据集管道提供的训练核心的数量)。

**支持在进样期间修改批次大小** 有时，您可能希望尝试在可用内存少于训练设备的环境中进行重现和调试。为此，您可能需要使用较小的输入数据批量运行。我们上面描述的注入层可以很容易地修改，以支持将捕获的训练批次大小的数据分成更小的批次。当然，总有可能问题是由整批引起的，不会在任何较小的批次上重现。如果您采用这种技术，请确保使用原始模型重量测试每一批次，即不要在批次之间更新重量。

**再现随机操作** 一些模型包括由随机变量控制的层。一个经典的例子是[漏失](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)层，一种正则化技术，它将一部分输入单元随机设置为零。在某些情况下，你可能会怀疑你的损失是由随机变量的不幸组合造成的。在当前形式下，我们描述的解决方案不能解决这种情况。当然，您可以多次运行 train_step，并希望问题能够重现，但这并不是成功的秘诀，尤其是在您有大量随机变量的情况下。这里有两种方法可以解决这个问题。首先是捕捉每个随机层的随机变量。例如，对于下降图层，您将捕获设置为零的单位。第二种选择是为每个随机层和每个训练步骤明确设置 RNG 种子(例如，作为迭代的函数)。这些选项在理论上都很简单，但在 TensorFlow 中实现需要一点掌握和创造力，这超出了本文的范围。

# 调试流程示例

为了让您更好地了解如何使用这个解决方案，我将描述一个基于真实用例的典型调试流程。

我们最近的任务是在 TPUs 上训练一个相当大的计算机视觉模型。(该模型*没有*包括随机层，如 dropout。)训练*没有*按计划进行。早期，我们发现几乎一半的训练实验都以 *NaN* 的失败告终。此时，我们开始在上面描述的*捕获*模式下运行我们的实验，使用定制的 DebugModel 代替默认的 keras 模型，并使用 DataInputCaptureLayer。幸运的是，这使我们能够很容易地捕捉到导致 NaN 失败的训练状态。捕获数据的批量大小为 1024。因为我们对在 CPU 环境中调试感兴趣，所以我们致力于识别一个大小为 1 的子批次，该子批次再现了 *NaN* 损失。在 CPU 上，我们使用 [tf.gradients](https://www.tensorflow.org/api_docs/python/tf/gradients) 显式计算特定变量的梯度，使用 [tf.print](https://www.tensorflow.org/api_docs/python/tf/print) 打印诊断结果。通常这是在定制层内完成的。(有时我们会扩展一个标准层并覆盖它的调用函数，以便包含这些操作。)我们还利用[急切执行](https://www.tensorflow.org/api_docs/python/tf/config/run_functions_eagerly)模式，遍历图表并分析张量值。通过这种方式，我们能够识别出导致 *NaN* 梯度的精确操作。我们已经实现了一个定制的损失函数，其中包括对 [tf.norm](https://www.tensorflow.org/api_docs/python/tf/norm) 的调用。结果是，在特定的数据样本和捕获的模型状态上，范数运算的值为零。由于范数运算在零处不可微，我们非常恰当地得到了 *NaN* s。(解决这个问题的一种方法是分解范数 op，并在执行平方根运算之前添加一个小ε。)

在另一种情况下，我们使用相同的技术来调查我们在 GPU 上使用[混合精度](https://www.tensorflow.org/guide/mixed_precision)时的 *NaN* 损失。我们发现 tf.float16s 的使用导致了损失计算的溢出。我们迅速将损失函数设置为使用 tf.float32，问题得到了解决。

这种方法无论如何也不能保证成功。在上面的轶事中，我们可能会在几个步骤中运气不佳。我们可能无法在较小的批量上重现该问题，我们可能会发现该问题仅在 TPU 上重现(例如，由于使用了 bfloat16)或仅在图形模式下重现，或者该模型可能依赖于随机层。在任何一种情况下，我们都需要从头开始，看看如何改进我们的方法来应对我们面临的挑战。

# 摘要

这篇文章讨论了在 TensorFlow 中训练的更困难的挑战之一，即 *NaN* loss 的出现。我们提出了一种重现和调试这一问题的新技术。正如我们所展示的，这项技术可以很容易地实现。虽然它不能解决所有可能的情况，但当所有其他方法都失败时，还是值得一试。你不觉得吗？