# 迁移学习:提升你的 ML 模型的秘密武器

> 原文：<https://towardsdatascience.com/transfer-learning-the-secret-weapon-for-boosting-your-ml-models-2fc35c83709a>

## 释放预训练模型的力量，提高性能，缩短训练时间

![](img/ca38ae4885bdb4dde505f52f9bb33aa7.png)

稳定扩散生成的图像

机器学习模型的开发涉及在大型(已标记)数据集上训练算法，这可能是耗时和资源密集型的。因此，我们需要一些技术，比如分布式培训或迁移学习，让我们能够更快地迭代，减少从研究到上市的时间。

迁移学习是机器学习中一种强大的技术，它允许您利用从解决一个问题中获得的知识，并将它应用到另一个相关的问题中。换句话说，迁移学习使您能够将知识从以前训练的模型“迁移”到新的模型，从而节省您从头训练模型所需的时间和资源。

近年来，迁移学习在机器学习社区变得越来越流行，这是有原因的。事实证明，它可以显著提高模型的性能，尤其是在处理小型数据集或希望针对特定任务对模型进行微调时。它还被证明可以显著减少培训时间，使其成为及时部署机器学习模型的组织的一个有吸引力的选择。

在本文中，我们将更详细地探讨迁移学习的概念。我们将介绍它在计算机视觉和 NLP 领域的基本工作原理，并看看如何使用 PyTorch 或 Keras 在自己的项目中实现它。

到本文结束时，你将会很好地理解迁移学习如何使你的 ML 项目受益，以及你如何利用它来达到更好的结果。

> [Learning Rate](https://www.dimpo.me/newsletter?utm_source=medium&utm_medium=article&utm_campaign=learning-rate) 是为那些对 AI 和 MLOps 的世界感到好奇的人准备的时事通讯。你会在每个月的第一个星期六收到我关于最新人工智能新闻和文章的更新和想法。订阅[这里](https://www.dimpo.me/newsletter?utm_source=medium&utm_medium=article&utm_campaign=learning-rate)！

# 计算机视觉中的迁移学习

迁移学习是一种强大的技术，可用于计算机视觉中，以提高机器学习模型的性能并减少训练时间。正如我们之前看到的，它涉及到使用从先前训练的模型中获得的知识，并将其应用于新的相关问题。

![](img/3a42fd3c759582c20f7e57062c2c26cf.png)

稳定扩散生成的图像

在计算机视觉的背景下，迁移学习可以用于在新的数据集上微调预训练的模型，或者使用较小的数据集训练新的模型。这在处理小型或专门的数据集时特别有用，在这种情况下，由于缺少数据，很难从头开始训练模型。

例如，假设您想要训练一个模型将动物图像分类到特定类别。你可以从使用预先训练好的图像分类模型开始，比如在 ImageNet 上训练的卷积神经网络(CNN ),作为你的基础。然后，您必须更改模型的输出图层，以符合数据集中的类别或标注。这允许您利用从预训练模型中获得的知识，并将其应用于您的特定问题。

或者，您可以使用较小的数据集，通过迁移学习来训练新模型。在这种情况下，您可以从一个预先训练好的模型开始，并将其作为新模型的起点。这允许您使用更少的数据点来训练模型，从而可能减少训练时间并允许您更快地部署模型。

# 自然语言处理中的迁移学习

迁移学习也广泛用于自然语言处理(NLP)，这是一个专注于分析和解释人类语言的机器学习领域。在 NLP 中，迁移学习可以用来提高模型的性能并减少训练次数，类似于它在计算机视觉中的使用。

![](img/2e89994845d37e0e8b63e4170cf63d24.png)

稳定扩散生成的图像

迁移学习在自然语言处理中的一个常见应用是语言建模。给定前面单词的上下文，语言模型用于预测单词序列中的下一个单词。这些模型通常在大型文本数据集上训练，如书籍或文章。当你训练这样一个模型时，你会得到一个非常理解人类语言的系统。

下一步是使这种模型的任务更加具体；例如，针对语言翻译、文本生成和文本摘要等任务对其进行微调。杰瑞米·霍华德等人在 NLP 的开创性论文中推广了这一技术:[用于文本分类的通用语言模型微调](https://arxiv.org/abs/1801.06146)。

# 用 Pytorch 迁移学习

让我们看一个 PyTorch 使用迁移学习的例子。我们将在 CIFAR-10 数据集上训练一个简单的图像分类模型:

首先，我们将从安装 PyTorch 开始:

```
pip install torch
pip install torchvision
```

然后，下载 CIFAR-10 数据集:

```
import torch
import torchvision
import torchvision.transforms as transforms

train_dataset = torchvision.datasets.CIFAR10(
  root='.', train=True, download=True,
  transform=transforms.Compose([transforms.ToTensor()]))
```

接下来，我们将定义一个简单的卷积神经网络(CNN)作为我们的基础模型。我们将使用 torchvision 库中预训练的 VGG-16 模型作为基础模型，并在顶部添加几个额外的层用于分类:

```
import torch.nn as nn
import torch.optim as optim

class TransferLearningModel(nn.Module):
  def __init__(self, num_classes=10):
    super(TransferLearningModel, self).__init__()

    # Use a pre-trained VGG-16 model as the base
    self.base_model = torchvision.models.vgg16(
      weights=torchvision.models.VGG16_Weights.DEFAULT)

    # Replace the classifier layer with a new one
    self.base_model.classifier = nn.Sequential(
      nn.Linear(in_features=25088, out_features=4096, bias=True),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5, inplace=False),
      nn.Linear(in_features=4096, out_features=4096, bias=True),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5, inplace=False),
      nn.Linear(in_features=4096, out_features=num_classes, bias=True))

  def forward(self, x):
    return self.base_model(x)

model = TransferLearningModel()
```

然后，我们可以使用标准 PyTorch API 来定义用于训练和评估的损失函数、优化器和数据加载器:

```
# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Define a data loader for the training set
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True)
```

最后，我们可以使用标准 PyTorch 训练循环来训练模型:

```
# Train the model
for epoch in range(10):
  for inputs, labels in train_loader:
    # Clear the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass
    loss.backward()
    optimizer.step()
```

该训练循环训练 10 个时期的模型，其中时期是训练数据集的完整过程。在每个历元中，模型成批处理训练数据，使用优化器根据计算的梯度更新模型的权重。

# 使用 Keras 进行迁移学习

这是一个使用 Keras 和 TensorFlow 的迁移学习在 CIFAR-10 数据集上训练简单图像分类模型的示例。

首先，我们将从安装 TensorFlow 开始:

```
pip install tensorflow
```

然后，我们需要下载 CIFAR-10 数据集:

```
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
```

接下来，我们将定义一个简单的卷积神经网络(CNN)作为我们的基础模型。我们将使用来自 Keras 应用程序库的预训练 VGG-16 模型作为基础模型，并在顶部添加几个额外的层用于分类:

```
# Load the VGG-16 model
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
# Add a few layers on top of the base model
model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(10, activation='softmax')])
# Freeze the base model's layers
for layer in base_model.layers:
  layer.trainable = False
```

然后，我们可以使用标准的 Keras API 来编译模型，并定义损失函数和优化器:

```
# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy'])
```

最后，我们可以使用`fit`方法训练模型:

```
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

这为 10 个时期训练模型，其中一个时期是训练数据集的一次完整通过。在每个历元中，模型成批处理训练数据，使用优化器根据计算的梯度更新模型的权重。

# 结论

总之，迁移学习是机器学习中的一项强大技术，它允许您利用从先前训练的模型中获得的知识，并将其应用于新的相关问题。事实证明，它可以显著提高模型的性能并减少训练时间，对于希望及时部署机器学习模型的组织来说，这是一个有吸引力的选择。

迁移学习在计算机视觉和自然语言处理(NLP)中被广泛用于各种任务，包括图像分类、对象检测、语言建模和文本生成。

在本文中，我们看到了什么是迁移学习，以及如何使用 PyTorch 或 Keras 实现迁移学习的应用。

# 关于作者

我叫[迪米特里斯·波罗普洛斯](https://www.dimpo.me/?utm_source=medium&utm_medium=article&utm_campaign=learning-rate)，我是一名为[阿里克托](https://www.arrikto.com/)工作的机器学习工程师。我曾为欧洲委员会、欧盟统计局、国际货币基金组织、欧洲央行、经合组织和宜家等主要客户设计和实施过人工智能和软件解决方案。

如果你有兴趣阅读更多关于机器学习、深度学习、数据科学和数据运算的帖子，请在 Twitter 上关注我的 [Medium](https://towardsdatascience.com/medium.com/@dpoulopoulos/follow) 、 [LinkedIn](https://www.linkedin.com/in/dpoulopoulos/) 或 [@james2pl](https://twitter.com/james2pl) 。

所表达的观点仅代表我个人，并不代表我的雇主的观点或意见。