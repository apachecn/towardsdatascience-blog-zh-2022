# Python 机器学习生态系统

> 原文：<https://towardsdatascience.com/the-python-machine-learning-ecosystem-7c05be4ac48d>

## 6 Python 机器学习工具。它们是什么，什么时候应该使用它们？

![](img/b1e1015fa13b9a94277549b3bc3e6378.png)

本·沃恩在 [Unsplash](https://unsplash.com/s/photos/ecosystem?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

Python 作为数据科学家使用的头号编程语言继续发展。在 Anaconda 最近发布的《2021 年 T4 数据科学状况》报告中，63%接受调查的数据科学家表示他们总是或经常使用 Python。

Python 在数据科学社区中的流行部分是由于可用于机器学习和深度学习的良好支持的开源工具。目前，人工智能(AI)或其机器和深度学习子集没有一刀切的工具。因此，在开源 Python 生态系统中，有几个核心库，每个都服务于这个领域中的一组特定用例或问题。

理解使用哪些工具以及何时使用它们经常会令人困惑。在下面的文章中，我将简要介绍六个最广泛使用的机器学习包。涵盖了每个库的核心目的是什么以及何时应该使用它们。本帖中涉及的包有:

*   **Scikit-learn**
*   **Pycaret**
*   **PyTorch**
*   **张量流**
*   **Keras**
*   **FastAI**

# 1.sci kit-学习

[Scikit-learn](https://scikit-learn.org/stable/) 是用于实现机器学习算法的最广泛使用的 Python 包之一。它提供了一个干净、统一的 API，使您能够以标准化的方式与各种模型进行交互。

Scikit-learn 极大地简化了机器学习模型的开发，此外还提供了一系列用于数据预处理、模型选择、调整和评估的实用程序，所有这些都是通过一个转换、拟合和预测命令的通用界面实现的。

这个接口使得 Scikit-learn 非常容易上手，也有助于数据科学中的代码再现性。Scikit-learn 广泛应用于工业领域，目前是对表格数据进行机器学习的首选库。

# 2.Pycaret

Pycaret 是一个低代码的机器学习库，旨在让更多的用户能够使用机器学习。与 Sckit-learn 类似，它提供了一个一致且易于使用的界面来与机器学习算法进行交互。

然而，Pycaret 通过包含 AutoML 方面，如自动化数据预处理和模型选择，进一步简化了这个过程。Pycaret 还旨在通过与云提供商如 [AWS](https://aws.amazon.com) 和 MLOps 包如 [mlflow](https://mlflow.org) 集成，成为一个完整的端到端解决方案，包含机器学习部署工具和 MLOps 功能。

# 3.PyTorch

尽管 Scikit-learn 是一个基于表格数据的机器学习问题的伟大库，但它并不太适合处理自然语言或基于视觉的用例所需的大规模数据。对于这些应用，需要深度学习。

PyTorch 提供的功能主要集中在构建和训练神经网络——深度学习的主干。PyTorch 提供跨单个或多个 CPU 和 GPU 的可扩展分布式模型训练。它也有自己的生态系统，提供与 Scikit-learn (skorch)、模型服务(TorchServe)和调度(AdaptDL)的集成。

与 TensorFlow 相比，它相对较新，第一次发布是在 2016 年 9 月，但它很快被业界广泛采用，特斯拉自动驾驶仪和优步的 PyTorch 都是基于 py torch 开发的。

# 4.张量流

[TensorFlow](https://www.tensorflow.org) 最初由谷歌大脑团队开发，于 2015 年首次开源。这一点，对于一个深度学习库来说，使其相对成熟。

自发布以来，TensorFlow 通常被视为开发神经网络的首选工具。它可以用于 Python 之外的各种编程语言，包括 Javascript、C++和 Java。它还具有灵活的架构，这意味着它可以部署在各种平台上，从 CPU 和 GPU 到移动设备上的服务器。

正是这种灵活性使其适用于各种各样的行业和用例，也是为什么它仍然是深度学习最受欢迎的工具之一。

# 5.克拉斯

TensorFlow 虽然是一个高度可扩展和强大的深度学习库，但并不以拥有特别友好的用户界面而闻名。 [Keras](https://keras.io) 将自己标榜为“为人类而非机器设计的 API”，是与 TensorFlow 交互的高级包装器。

Keras API 有一个与 Scikit-learn 相似的公共接口。具有一致的编译、拟合和预测命令。它旨在允许对 TensorFlow 库进行快速实验，同时也使深度学习可用于更广泛的用户。

# 6.法斯泰

FastAI 是另一个旨在让实践者更容易获得深度学习的库，同时也为研究人员提供解决方案。

它有一个高级 API 接口，可以抽象出许多复杂性，并允许从业者从深度学习中快速获得最先进的结果。同时还提供研究人员可以用来发现新方法的低级组件。

在引擎盖下，FastAI 使用 PyTorch。它本质上提供了一个更简单的抽象层，同时也引入了许多新的功能，如数据可视化和拆分及加载数据的新方法。

在本文中，我介绍了六个最流行的用于机器学习和深度学习的 Python 包。在 Python 生态系统中还有许多其他可用的工具，包括 [Theano](https://theano-pymc.readthedocs.io/en/latest/) 、 [Chainer](https://chainer.org) 和 [Spark ML](https://spark.apache.org/docs/1.2.2/ml-guide.html) 。

正如本文前面所述，对于机器学习来说，真的没有放之四海而皆准的库，更多的是为正确的工作选择正确的工具。

本文中描述的每个工具都为一组特定的问题或用例提供了解决方案。我在下面的一行中总结了每个库的主要用途。

**Scikit-learn —机器学习的首选库，提供用户友好、一致的界面。**

**Pycaret —通过低代码、自动化和端到端解决方案降低机器学习的起点。**

**py torch——利用其高度灵活的架构构建和部署强大、可扩展的神经网络。**

**tensor flow——最成熟的深度学习库之一，高度灵活，适合广泛的应用。**

**Keras — TensorFlow 制作简单。**

**FastAI——通过构建在 PyTorch 之上的高级 API，使深度学习变得更容易访问。**

感谢阅读！