# 如何在带有 Keras 的 TensorFlow 中使用带有自定义梯度的自定义损失

> 原文：<https://towardsdatascience.com/how-to-use-custom-losses-with-custom-gradients-in-tensorflow-with-keras-e87f19d13bd5>

## *解释文档没有告诉您的内容的指南*

![](img/bc8fa23db0e28ab3e614f0665f6fe8e5.png)

上传者:来自德语维基百科的 Eleassar。、CC BY-SA 3.0、via Wikimedia Commons[https://Commons . Wikimedia . org/wiki/File:Meerkatmascotfull _ BG _ F7 F8 ff . png](https://commons.wikimedia.org/wiki/File:Meerkatmascotfull_bg_F7F8FF.png)

# **简介**

Keras 在抽象神经网络创建的底层细节方面做得很好，因此您可以专注于完成工作。但是，如果你正在阅读这篇文章，你可能已经发现，Keras 现成的方法并不总是能够用来学习你的模型的参数。也许你的模型有一个无法通过 autodiff 的魔力计算的梯度，或者你的损失函数不符合 Keras 的文档中提到的签名`my_loss_fn(y_true, y_pred)`。如果您发现在线文档完全没有帮助，请继续阅读！我[希望]有你在其他地方找不到的所有答案。

# 放弃

我不保证把我所有的解释都钉死；我不认为自己是 TensorFlow/Keras guru。我在这里写的所有内容都参考了许多不同页面的 TensorFlow/Keras 文档和一些源代码检查。如果你有改正或改进的建议，我鼓励你在评论中留下它们，让每个人受益！最后，TensorFlow 是 Google LLC 的商标，本文既不被 Google LLC 认可，也不以任何方式隶属于 Google LLC。

# **学习目标**

阅读完本指南后，你将理解如何使用 Python 中的自定义子类化 Keras `Layer`来创建自定义子类化 Keras `Model`对象。您将能够编写自己的自定义损失函数，这些函数与 Keras 的[文档](https://keras.io/api/losses/#creating-custom-losses)中描述的[http://archive.ics.uci.edu/ml](http://archive.ics.uci.edu/ml)不一致。您还可以将自定义渐变与自动微分(autodiff)算法结合使用，以优化模型的可训练参数。

# **本指南将展示什么？**

本指南将最终展示你如何仍然可以使用自定义损失和自定义梯度，而不必放弃方便的`keras.Model.fit`方法来训练你的神经网络。

在本指南中，我们将创建具有单一密集层和逻辑回归输出层的神经网络模型。原始数据将被输入到密集层，密集层的输出将被输入到逻辑回归层进行二元分类。本指南不包括验证或模型测试；它将只涵盖在训练集上构建和拟合模型所涉及的步骤。

让我们开始吧。

# **加载(或创建)数据集**

我们将使用开放访问`[german_credit_numeric](https://www.tensorflow.org/datasets/catalog/german_credit_numeric)`数据集，它可以通过 tensorflow_datasets 库下载。它由 24 个特征的 1000 个例子组成，与“好”(1)或“坏”(0)的二元信用风险评估相关。

```
import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Optional

@tf.function
def sigmoid(x: tf.Tensor) -> tf.Tensor:
    return 1 / (1 + tf.exp(-x))

if __name__ == "__main__":

    ds = tfds.load("german_credit_numeric", split="train", as_supervised=True)
    ds = ds.shuffle(1000).batch(100).prefetch(tf.data.AUTOTUNE)
```

# **创建自定义 Keras 层**

在我们构建模型之前，我们应该构建它的组件。回想一下，该模型将由一个密集图层组成，该图层将我们数据集中观察到的要素转换为潜在表示，该潜在表示将作为逻辑回归输出图层的输入。为了演示如何混合和匹配定制和预构建的 Keras `Layer`，我们将使用 Keras 的内置`keras.layers.Dense`类来构建模型的第一层，并通过子类化`keras.layers.Layer`为逻辑回归构建我们自己的定制层类。

让我们分解一下客户物流层。

*`*get_config*`*方法**

*对于该层的工作来说，`get_config`方法并不是绝对必要的，但是如果你想创建一个所谓的“可序列化”的层，可以与 [Functional API](https://keras.io/guides/functional_api/) 一起使用，那么它是必要的。您可以使用`get_config`方法来获取重新创建图层所需的属性。参见[文档](https://www.tensorflow.org/guide/keras/custom_layers_and_models#you_can_optionally_enable_serialization_on_your_layers)。

*`*build*`*方法*

与`get_config`方法一样，`build`方法不是层工作所必需的，但它是定义层的可训练参数(即权重、偏差)的推荐方法。使用`build`方法，参数创建被推迟，直到用一些输入数据第一次调用该层。`build`方法通过`input_shape`参数获得输入的形状(可能事先不知道)，并根据`input_shape`创建图层的参数。例如，如果层的`call`方法的`inputs`参数是一个张量元组，Keras 会将一个`TensorShape`对象数组分配给`input_shape`(输入张量中的每一项对应一个对象)。建议使用 build 方法，因为它允许即时创建权重，即使您事先不知道输入形状。

*`*call*`*方法*

该方法通过层计算输入的正向传递。这里，它返回
逻辑回归的概率输出。

在我们定义模型类之前，让我们实例化代码主体中的输入层和密集层:***

```
 **features_spec, labels_spec = ds.element_spec
del labels_spec  # Not used

feature_inputs = tf.keras.Input(type_spec=features_spec, name="feature_inputs")
dense = tf.keras.layers.Dense(units=4, name="dense_layer")**
```

**请注意，我们创建了一个`Input`对象`feature_inputs`，它期望一个具有五个分量的向量值张量(即一个具有 24 个特征的训练示例)。`dense`图层将计算仅包含四个特征的输入的潜在表示。由于我们使用潜在表示作为逻辑回归的输入，`Logistic`层将有四个可训练的权重(每个输入特征一个)，以及一个单一偏差。还要注意，我们没有实例化一个`Logistic`层。我们将把它留给自定义的`Model`对象。**

# **创建模型**

**这是指南的症结所在。我们将创建一个`keras.Model`的子类，它有一个定制的训练循环、损失函数和渐变。**

**损失函数将是给定
相关特征的目标标签的负对数可能性。最小化负对数似然的权重和偏差是逻辑回归模型的最佳参数。换句话说，如果目标标签由我们的神经网络建模，我们将最大化在给定相关特征的情况下观察到目标标签的概率。这是概率机器学习中常见的损失函数。为了在神经网络模型下最大化数据集的对数似然性(等价地，最小化负对数似然性)，我们需要找到最优的权重和偏差。我们使用梯度下降的一些变体来寻找最优值(选择你最喜欢的方法)。当然，这需要计算负对数似然损失相对于模型可训练参数的梯度。**

**使用负对数似然作为我们的自定义损失函数在 Keras 中提出了一些挑战。首先，负对数似然损失不一定符合 Keras 文档为定制损失函数建议的签名`my_loss_fn(y_true, y_pred)`;在我们的例子中，它是输入特征和目标标签的函数。因此，我们不能简单地为负对数似然定义一个自定义损失函数，并在训练模型时将其作为参数传递给`Model.fit`的`loss`参数。其次，并不是所有基于可能性的损失函数都有可以被 autodiff 算法“神奇地”解决的梯度，在这种情况下，您需要告诉 TensorFlow 如何精确地计算梯度。当损失函数不符合 Keras 建议的特征时，事情变得有点复杂。最后，通常会有一些早期图层学习训练示例的观察特征的潜在表示，如果 autodiff 可用于求解这些图层的梯度，我们希望使用它，仅在必要时定义自定义梯度(这是一项大量工作！).**

**让我们定义一个`keras.Model`的子类来克服这些挑战。首先我会给出代码，然后我会解释各个部分。**

**keras 的一个亚纲。使用带有非标准签名的自定义损失函数以及自定义梯度和自动微分梯度的模型**

**让我们来分解一下这个模型。**

***`*__init__*`*方法*

该模型被设计成接受早期的层块`nn_block`，其学习原始输入数据的潜在表示。在本指南中，`nn_block` 将是`keras.layers.Dense`的一个实例，但它可以是任何`keras.layers.Layer`对象。如果您只想对原始输入数据执行逻辑回归，也可以完全省略它。

模型自动初始化一个`Logistic`层，该层输出一个单一值(`units=1`)，该值表示给定输入特征的标注概率。***

**最后，每个样本的平均损失由`loss_tracker`跟踪。

*损失函数*

这里的目标是定义一个损失函数，它允许我们使用 autodiff 来计算相对于`Dense`层的可训练参数的损失梯度，但是对`Logistic`层的参数使用自定义梯度计算。为此，我们必须将损失计算中涉及自定义坡度参数的部分与涉及自动挖掘参数的部分分开。隔离是通过将自定义组件的函数嵌套在更广泛的损失函数中实现的，该损失函数提供了从模型输入到模型输出的完整路径。通过这个路径，所有层的可训练参数都可以被优化。

我们先来关注一下内部函数，`logistic_loss(x, y)`，先。第一个参数，`x`，代表`Logistic`层的输入张量(即来自之前`Dense`层的输出)。第二个参数`y`表示训练示例的真实标签。该函数隔离了损失计算中涉及我们希望通过自定义梯度学习的参数的部分。`@tf.custom_gradient`装饰器向 TensorFlow 发出信号，使用自定义公式代替自动 diff 来计算相对于装饰器范围内可训练参数的损失梯度。因此，为装饰器范围内的所有可训练参数指定一个自定义渐变是很重要的。这就是为什么我们根据对`Logistic`层的输入而不是对更早层的输入来定义`logistic_loss`:我们有效地将装饰器的范围限制在我们希望使用自定义梯度来学习的权重和偏差，并将其余的梯度计算留给自动挖掘。

自定义渐变在`logistic_loss`内部定义，由
装饰器要求(详见 TensorFlow [文档](https://www.tensorflow.org/api_docs/python/tf/custom_gradient))。

外部函数`loss_fn`将训练数据(或验证/测试数据)中的原始特征和目标标签作为输入。注意，我们没有用`tf.custom_gradient`装饰器包装外部的`loss_fn`。这确保了自动识别用于计算不在`logistic_loss`范围内的剩余参数的损失梯度。外部函数返回由内部`logistic_loss`函数计算的负对数似然。

您可能已经注意到，只有`Logistic`层的输入才是计算输入批次的负对数似然损失所必需的。那么为什么要大费周章地将`logistic_loss`嵌套在外层`loss_fn`中呢？如果您去掉外部函数，转而使用`logistic_loss`作为总损失函数，TensorFlow 将警告您，相对于致密层参数的损失梯度不再定义。那是因为我们没有在`@tf.custom_gradient`装饰器下定义它们。逻辑层的参数将被训练，但是密集层的参数不会从它们的初始值改变。

*`*train_step*`*方法*

该方法由`Model.fit` 方法用来更新模型参数和计算模型指标。正是定制模型对象的组件让我们可以使用高级的`Model.fit`方法来定制损失和渐变。`GradientTape`上下文管理器跟踪`loss_fn`的所有梯度，在不使用自定义梯度计算的情况下使用 autodiff。我们使用`tape.gradient(loss, self.trainable_weights)`访问与可训练参数相关的梯度，并告诉优化器对象使用这些梯度来优化`self.optimizer.apply_gradients(zip(grads, self.trainable_weights))`中的参数。最后，训练步骤用训练批次的平均损失(由`loss_fn`计算)更新`loss_tracker`。***

# **训练模型**

**我们差不多准备好训练模型了。首先，让我们编写一个自定义回调函数，在每个时期后打印可训练权重，只是为了验证参数正在被调整:**

**回调将按照每层在神经网络中出现的顺序，打印每层的可训练参数数组。每个数组中的第一项是图层的权重，第二项是偏差。**

**现在，让我们回到代码的主体并训练模型:**

```
**model = CustomModel(nn_block=dense, name="nn_logistic_model")
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4))
model.fit(ds, epochs=5, callbacks=[ReportWeightsCallback()])**
```

**现在，您已经准备好进行端到端的培训了！完整的代码包含在本指南的末尾。**

# **最后的想法**

**你可能想知道为什么我们子类化`keras.Model`来实现我们的目标。对你来说可能很复杂。例如，为什么不在`Logistic`层中定义一个带有自定义渐变的损失函数，并完全去掉自定义模型对象呢？**

**一开始我也是这么尝试的，但是我意识到这样做效果不好。通过告诉自定义的`Layer`将要素和目标标注作为输入，用户可以在`Logistic`层内使用自定义梯度实现负对数似然损失函数。当您可以访问标签时，这在训练期间是有效的，但是当您使用模型进行推理时，这就不太有效了，因为对未知数据的预测不涉及作为输入的目标标签:用于推理的输入与用于训练的输入不具有相同的形状。**

**使用模型可以解决这个问题，因为它允许用户创建自定义的训练、评估和预测循环，而不会影响模型的各个层所预期的输入形状。它允许我们在训练期间访问计算自定义损失所需的特征和标签，并在推断期间仅使用特征进行预测。**

# **完整的代码**

**构建和训练 Keras 模型的端到端脚本，该模型使用带有自动微分和自定义梯度的自定义损失**

****来源****

**德国信贷数据。2017.UCI 机器学习知识库。可从:[http://archive.ics.uci.edu/ml](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))获得。**