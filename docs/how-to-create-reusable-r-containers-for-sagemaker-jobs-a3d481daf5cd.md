# 如何为 SageMaker 作业创建可重用的 R 容器

> 原文：<https://towardsdatascience.com/how-to-create-reusable-r-containers-for-sagemaker-jobs-a3d481daf5cd>

## 为 R 开发人员在 SageMaker 上创建可重用容器的指南

![](img/c483d12c2b74d59db4ea0aecdf1280e8.png)

图片来源于 unsplash.com

SageMaker 非常棒，它给了您充分的灵活性，让您可以使用自己的运行时和语言来使用它的服务。如果没有可用的运行时或语言适合您的代码，您首先需要克服最初的障碍，创建一个兼容的 docker 容器(T2)，SageMaker 可以使用。

在这篇博客中，我们深入探讨了如何在 SageMaker 中创建这样的 *R-containers* ，并试图更深入地理解 SageMaker 的工作方式*。这使我们在容器构建阶段做出的一些决策更加清晰。要获得利用这些 R 容器的 ML 管道的端到端示例，请查看这个 [GitHub](https://github.com/aws-samples/rstudio-on-sagemaker-workshop/tree/main/03_SageMakerPipelinesAndDeploy) 示例。*

# 码头集装箱简而言之

您可能已经阅读了这篇文章，但是不知道 docker 容器是什么。我不会试图解释 docker 或 containers 是什么，因为已经有大约一百万篇这样的文章写得比我更好。

简而言之，容器是一个标准的软件单元，它将代码及其所有依赖项打包在一个“对象”中，可以跨不同的系统安全可靠地执行。

对于这个博客，你需要广泛地熟悉一些概念，即什么是 docker 文件、图像、容器注册表和容器。如果你对容器很好奇，想了解更多，可以从这里[开始了解更多。](https://www.docker.com/resources/what-container)

# 为什么是容器+ SageMaker？

SageMaker 是以模块化的方式构建的，允许我们使用自己的容器来提供服务。这给了我们使用我们选择的库、编程语言和/或运行时的灵活性，同时仍然利用使用其服务的全部好处。

# 用于 SageMaker 处理的 r 容器

为处理作业创建一个 R 容器可能是我们在 SageMaker 上可能需要的所有容器中最简单的。
docker 文件可以如下所示:

当容器被创建并注册到 [ECR，Amazon Elastic Container Registry](https://aws.amazon.com/ecr/)时，我们可以运行一个处理作业。这类似于我们通常运行处理作业的方式，我们只需将参数 *image_uri* 即新创建的图像的 uri 传递给作业。这种处理作业运行的例子(也作为流水线的一部分)可以在[流水线的第 33 行中找到。上面分享的例子中的 R](https://github.com/aws-samples/rstudio-on-sagemaker-workshop/blob/main/03_SageMakerPipelinesAndDeploy/pipeline.R#L33) 。当处理作业运行时，SageMaker 使用以下命令运行容器:

```
docker run [AppSpecification.ImageUri]
```

因此，将运行入口点命令，并且将运行传递到 *ScriptProcessor* 的*代码*参数中的脚本。在这种情况下，我们的入口点是命令 *Rscript* ，因此这个容器可以被所有需要执行任意代码的处理作业重用，当然假设必要的包依赖关系对它可用。

进一步的定制是可能的，如果你有兴趣更深入地了解 SageMaker 容器是如何具体处理作业的，请随意阅读相关文档页面。

# 用于 SageMaker 培训和部署的容器

与上面这个简单明了的例子相比，为训练作业创建一个 R 容器(也可以在部署模型时重用)需要更多的步骤。

模板 Dockerfile 文件可以如下所示:

您会注意到，一旦我们安装了模型/代码所需的必要包，我们还复制了一个 run.sh 和一个 entrypoint。r 文件。让我们看看这些文件是什么，为什么需要它们。

```
#!/bin/bash
echo "ready to execute"
Rscript /opt/ml/entrypoint.R $1
```

run.sh 脚本非常简单，它所做的只是运行*入口点。R* 脚本在$1 下传递命令行参数。我们这样做是因为 SageMaker 使用以下命令运行 docker 容器进行培训和服务:

```
docker run image train
```

或者

```
docker run image serve
```

这取决于我们称之为培训还是部署方法。基于参数$1，即“train”或“serve ”,我们想区分下一步。这里需要 bash 脚本将这个参数传递给 Rscript 执行，因为没有直接的方法从 R 代码中读取 docker run 参数。如果你知道更好/更简单的方法，请在评论中告诉我！

现在让我们看看入口点。r 脚本:

这现在变得更加 SageMaker 具体的方式，让我们打开它！SageMaker 有一个定义非常好的文件结构，它保存文件并期望在/opt/ml/下找到文件。具体来说，我们在这里使用的是:

```
/opt/ml/
    - input/config/hyperparameters.json
    - code/
    - model/
        - <model artifacts>
        - code/
```

*hyperparameters.json* 文件
当创建一个训练评估器时，我们将希望传入一些自定义代码来定义和训练我们的模型。通过之后，SageMaker 会将这些文件(可能是您需要通过培训的文件的整个目录)压缩到一个名为“*source dir . tar . gz”*的文件中，并将它上传到 S3 的一个位置。一旦我们开始一个训练作业，SageMaker 将在/opt/ml/input/config/位置创建 hyperparameters.json 文件，该文件包含任何传递的 hyper 参数，但也包含关键字"*sage maker _ submit _ directory "*，其值为" *sourcedir.tar.gz"* 文件上载到的 S3 位置。
在训练模式下，我们需要下载并解压缩我们的训练代码。这正是上面 if 语句的第一部分所做的。

*代码*目录
遵循 SageMaker 如何在内置算法和托管框架容器上下载和解包训练代码的约定，我们正在提取/opt/ml/code/目录中的训练代码。然而，这不是一个要求，而是遵循服务标准的一个好的实践。

*模型*目录
这是 SageMaker 自动下载模型工件和与推理相关的代码的目录。
上面代码片段中 if 语句的第二部分利用了这一点，来获取 *deploy。R* 脚本。这里需要注意的是，这个 Dockerfile &代码示例假设我们的推理代码将包含一个*部署。R* 文件，该文件将为部署而运行。如果您遵循不同的命名习惯，请随意重命名该文件。
在这个代码示例中，在训练过程中，一旦创建了模型，模型的工件就保存在/opt/ml/model 文件夹下。我们还将推理代码保存在同一目录下的子文件夹 code/中。这样，当 SageMaker 压缩文件以创建 model.tar.gz 文件时，这个文件也将包含部署代码所必需的内容。

以上是一个架构/设计决策，用来将推理代码与模型本身捆绑在一起。对于您的用例来说，想要分离这两者并保持推理代码独立于模型工件是完全合理的。这当然是可能的，由你来决定采用哪种方法。

还请注意，模型工件保存在 S3 上的一个单独的*model.tar.gz*文件中，然而，在部署期间，SageMaker 会自动下载并解压缩这个文件，所以我们不必在部署期间自己手动这么做。

*Pro 提示:您可能希望有不同的容器用于训练和部署，在这种情况下，可以简化上述步骤，跳过 run.sh 脚本的使用。*

进一步的定制是可能的，如果你有兴趣深入了解 SageMaker 容器如何专门用于训练和推理工作，请随意阅读[相关文档页面](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers-create.html)。

# 建造集装箱

如果你熟悉构建容器，你会意识到下面的过程没有什么本质上的特别。我们所需要做的就是根据提供的 docker 文件构建容器，并向 ECR 注册图像，SageMaker 作业将在运行时提取图像。如果你已经知道如何建立和注册一个图像到 ECR，请随意跳过这篇文章的这个部分。

对于 SageMaker 上 RStudio 的用户或者任何不能或不愿意让 docker 守护进程在他们的开发环境上运行的人，我建议将容器的实际构建外包给另一个 AWS 服务，即 [AWS CodeBuild](https://aws.amazon.com/codebuild/) 。幸运的是，我们不需要主动与该服务交互，这要感谢有用的实用程序 [SageMaker Docker Build](https://github.com/aws-samples/sagemaker-studio-image-build-cli) ，它对我们隐藏了所有这些复杂性。
使用如下命令安装该实用程序:

```
py_install("sagemaker-studio-image-build", pip=TRUE)
```

我们准备好了。构建容器只需要一个命令:

```
sm-docker build . --file {Dockerfile-Name} --repository {ECR-Repository-Name:Optional-Tag}
```

# 结论

SageMaker 的处理、培训和托管功能非常全面，通过自带容器，我们可以按照自己的方式构建模型和应用程序。

在这篇博客中，我们探索了如何创建我们自己的、可重用的、支持 R 的 docker 容器，我们可以用它来满足我们的处理、培训和部署需求。

本文中使用的完整代码示例可以在这个 [Github 资源库](https://github.com/aws-samples/rstudio-on-sagemaker-workshop/tree/main/03_SageMakerPipelinesAndDeploy)中找到。

如果您正在 SageMaker 上为 R 构建自己的容器，请在评论中联系我，或者在 LinkedIn 中与我联系，SageMaker 愿意就此进行讨论！