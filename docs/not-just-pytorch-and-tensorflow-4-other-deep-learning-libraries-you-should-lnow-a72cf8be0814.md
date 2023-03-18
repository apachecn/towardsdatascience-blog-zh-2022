# 不仅仅是 PyTorch 和 TensorFlow:你应该知道的另外 4 个深度学习库

> 原文：<https://towardsdatascience.com/not-just-pytorch-and-tensorflow-4-other-deep-learning-libraries-you-should-lnow-a72cf8be0814>

## JAX、MXNet、MATLAB 和 Flux 的快速介绍

![](img/a9a5254595c142bc459cb5dff94cabc0.png)

由[加布里埃尔·索尔曼](https://unsplash.com/@gabons?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

机器学习库加速深度学习革命。他们通过抽象出 GPU 加速、矩阵代数、自动微分等许多高难度的东西，降低了从业者的入门门槛。在行业和学术界，有两个深度学习库占据着至高无上的地位:PyTorch 和 TensorFlow。在本文中，我将向您介绍一些其他具有相当大使用量的深度学习库，或者是因为它们在某些方面实现了加速，或者是因为它们被非常特定的群体使用。我们开始吧！

# JAX

**是什么？**最初由 Google 开发的一个开源和正在开发的数值框架(想想 NumPy，但针对 GPU)。

**谁用？**谷歌内部的很多团队，比如 DeepMind。

**你为什么要知道这件事？** JAX 由谷歌开发，用于在 GPU 和谷歌自己的硬件 TPU 上加速数值计算。使用加速线性代数、实时编译(JIT)和自动矢量化等思想，JAX 实现了巨大的加速和扩展。尽管他们的语法相似，以尽量减少学习曲线，JAX 有不同于 NumPy 的设计哲学。JAX 通过`vmap`和`pmap`(向量化+并行化)等函数鼓励函数式编程。

目前，已经为 JAX 开发了许多高级 API。值得注意的是俳句和亚麻。

# 阿帕奇 MXNet

**什么事？**开源老牌机器学习框架，前端绑定多种语言，包括 Python、C++、R、Java、Perl。

**谁用？**亚马逊 AWS。

你为什么要知道这件事？ MXNet 最强大的特性是它对多种编程语言的支持以及它的可扩展性。英伟达的基准测试表明，MXNet 在一些深度学习任务上比 PyTorch 和 TensorFlow 更快。

MXNet 附带了 Gluon，这是一个用于构建神经网络的高级 API。它还拥有一个用于图像分类(GluonCV)和 NLP(gluonlp)的生态系统。

# MATLAB 深度学习工具箱

**什么事？**为 MATLAB 用户提供的附加工具箱，可以为各种任务创建和训练神经网络。

谁使用它？学术界和航空航天、机械工程等行业。例如，空客用它来检测飞机内部的缺陷。

你为什么要知道这件事？无论你对 MATLAB 有什么感觉，它仍然是学术界和工程师中流行的编程生态系统。它有很好的用户支持，在我看来，它是这个列表中所有深度学习库中最好的文档。深度学习工具箱面向那些希望使用最少编程来构建系统的人。MATLAB 中的图形编程界面 Simulink 提供了创建易于理解的深度学习管道的方法。

# 朱莉娅·弗勒斯

**什么事？**为 Julia 编程语言打造的开源机器学习库。

**谁使用它？**医药和金融等计算密集型领域。例如，阿斯利康用它来预测药物毒性。

你为什么要知道这件事？ Julia 编程语言多年来在数据科学家、定量分析师和生物信息学研究者中获得了发展势头。就速度而言，它与 C/C++不相上下，而且它被设计成像 Python 一样对初学者友好。Julia deep learning 在谷歌 TPU 上的一个实现显示，与 CPU 相比，速度提高了 200 倍。如果你已经在用 Julia 编程，Flux 是一个很好的库。

# 结论

我希望，通过这篇短文，您可以了解一些其他的深度学习库。它们都支持高效的加速、GPU 扩展和生产部署。互联网上有很好的学习资源。编码快乐！

# 来源

[1][https://www . deep mind . com/blog/using-jax-to-accelerate-our-research](https://www.deepmind.com/blog/using-jax-to-accelerate-our-research)

[https://github.com/aws/sagemaker-python-sdk](https://github.com/aws/sagemaker-python-sdk)

[3][https://developer . NVIDIA . com/deep-learning-performance-training-inference](https://developer.nvidia.com/deep-learning-performance-training-inference)

[4][https://www . mathworks . com/company/user _ stories/case-studies/airbus-uses-artificial-intelligence-and-deep-learning-for-automatic-defect-detection . html](https://www.mathworks.com/company/user_stories/case-studies/airbus-uses-artificial-intelligence-and-deep-learning-for-automatic-defect-detection.html)

[https://twitter.com/jeffdean/status/1054951415339192321?[5]lang=en](https://twitter.com/jeffdean/status/1054951415339192321?lang=en)

[6][https://julialang.org/blog/2012/02/why-we-created-julia/](https://julialang.org/blog/2012/02/why-we-created-julia/)

[7][https://juliacomputing.com/case-studies/astra-zeneca/](https://juliacomputing.com/case-studies/astra-zeneca/)