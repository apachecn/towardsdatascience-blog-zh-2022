# Amazon SageMaker 项目的灵活的端到端模板

> 原文：<https://towardsdatascience.com/a-flexible-end-to-end-template-for-amazon-sagemaker-projects-72d750b6933>

## 如何为 Amazon SageMaker 项目使用新的“ [shapemaker](https://github.com/smaakage85/shapemaker) ”模板

![](img/d84d148f491205a5bb3848613a3afca2.png)

shapemaker 的标志。资料来源:Lars Kjeldgaard。

在本文中，我将介绍如何使用新的“ [shapemaker](https://github.com/smaakage85/shapemaker) ”模板来创建具有最大灵活性的 Amazon SageMaker 项目。

本文的目标读者是对 python、Amazon SageMaker 以及 AWS、docker、shell 脚本和 web 应用程序开发具有中级知识的*全栈*数据科学家。

**您为什么要关心？**
Amazon SageMaker 管理 TensorFlow、PyTorch 和 HuggingFace 等流行框架的容器，您可以使用开箱即用来创建模型训练作业和用于推理的端点。这允许开发人员只关注提供训练和推理脚本(即在[脚本模式](https://aws.amazon.com/blogs/machine-learning/bring-your-own-model-with-amazon-sagemaker-script-mode/)下工作)。

然而，根据我在实际生产环境中的经验，通常情况是，这种方法没有提供足够的灵活性:可能(SageMaker 的现有容器不支持您正在使用的框架，(2)您需要定制培训作业容器，或者(3)您需要定制端点容器(或者如何提供服务)。

**解决方案:自带容器** 为了解决这个问题，SageMaker 提供了一个名为[自带容器](https://sagemaker-workshop.com/custom/containers.html) (BYOC)的功能，提供完全的开发者控制。顾名思义，BYOC 意味着您可以为训练作业和推断端点使用 SageMaker 自己的容器，然后您可以通过 SageMaker API 使用这些容器。

乍看之下，与亚马逊 Sagemaker BYOC 合作似乎有点拗口。尤其是如果你习惯于在[脚本模式](https://aws.amazon.com/blogs/machine-learning/bring-your-own-model-with-amazon-sagemaker-script-mode/)下工作。

为了让 BYOC 更容易理解，我为 Amazon SageMaker 项目打造了一个模板，实现了 BYOC。

**推出“shape maker”** `[shapemaker](https://github.com/smaakage85/shapemaker)`是亚马逊 SageMaker AWS 项目的完整端到端模板，旨在实现最大的灵活性。它建立在 [BYOC](/bring-your-own-container-with-amazon-sagemaker-37211d8412f4) SageMaker 功能的基础上，实现了开发者的完全控制。

该模板包括:

*   模型代码的极简模板
*   用于模型训练的 docker 图像模板
*   一种用于实时推理的端点标记图像模板
*   用于与模型/端点交互的命令行功能
*   用于交付和集成带有 SageMaker 的模型的命令行功能
*   持续集成/交付工作流。

[‘shape maker’](https://github.com/smaakage85/shapemaker)支持 Linux/macOS。

# **‘shape maker’一览**

您可以使用[cookiecutter](https://github.com/cookiecutter/cookiecutter)**:**从[‘shape maker’](https://github.com/smaakage85/shapemaker)模板创建一个新项目

```
cookiecutter gh:smaakage85/shapemaker
```

下面的视频快速介绍了[“shape maker”](https://github.com/smaakage85/shapemaker)模板的一些最重要的功能以及如何使用它:它介绍了如何从模板创建项目，以及如何使用其内置命令行功能来构建培训和端点图像，以及创建培训作业和端点。此外，我还展示了如何启用`[shapemaker](https://github.com/smaakage85/shapemaker)` CI/CD 工作流。

**注意:**对于那些刚接触 AWS 的人来说，如果你想继续下去，请确保在下面的 [**链接**](https://aws.amazon.com/console/) 中做一个帐户。部署过程中会产生成本，尤其是如果您让端点保持运行。

“塑造者”——绝技。资料来源:Lars Kjeldgaard。

# **“塑造者”深潜**

接下来，我将详细介绍一下[‘shape maker’](https://github.com/smaakage85/shapemaker)模板的一些细节。

关于如何使用[‘shape maker’，](https://github.com/smaakage85/shapemaker)的更多详细说明，请参考[自述文件](https://github.com/smaakage85/shapemaker/blob/main/README.md)。

1.  **型号代码**

模型代码被组织成一个 python 包:`modelpkg.`

模板附带了一个作为占位符的*虚拟*模型:一个估计薪水和年龄之间线性关系的模型。

该模型被实现为它自己的类，具有用于(1)模型训练，(2)预测观察，(3)性能评估和(4)加载/保存模型工件的方法。

模型代码

**2。模型训练图像** 训练脚本`train.py`借鉴了`modelpkg`中的模型代码。`train.py`运行一个参数化的训练任务，生成并保存一个模型工件。

培训脚本。建立模型培训码头工人形象。

`modelpkg`和上面的训练脚本被构建到能够运行模型训练工作的模型训练 docker 映像中。

**3。端点图像** 模型端点图像[“shape maker”](https://github.com/smaakage85/shapemaker)附带一个 Flask web 应用程序，它使用`modelpkg`来加载模型工件并计算预测，作为应用程序用户请求的答案。

SageMaker 模型端点的 Web 应用程序。

该应用内置于端点 docker 映像中。默认情况下，[‘shape maker’](https://github.com/smaakage85/shapemaker)实现了一个 [NGINX](https://www.nginx.com/) 前端。

端点 docker 图像。

**4。使用命令行函数** 构建、训练和部署模型所有与模型项目交互相关的任务都在`Makefile`中作为方便的命令行函数实现，这意味着使用`make [target]`调用函数，例如`make build_training_image`。

如果您想要即时构建、训练和部署模型，您可以通过调用一系列`make`目标来实现，即:

1.  `make init`
2.  `make build_training_image`
3.  `make push_training_image`
4.  `make create_training_job`
5.  `make build_endpoint_image`
6.  `make push_endpoint_image`
7.  `make create_endpoint`

之后，您可以通过调用`make delete_endpoint`来删除端点。

**注意:** `make` + space + tab + tab 列出所有可用的`make`目标。

**5。配置文件** 培训工作、端点等的配置。与代码分开，存放在`configs`文件夹中。这意味着，如果您想要更改端点的配置方式，您只需更改`configs/endpoint_config.json:`

SageMaker 端点的配置。

**6。CI/CD 工作流程** `[shapemaker](https://github.com/smaakage85/shapemaker)`附带了许多通过 Github 操作实现的自动化(CI/CD)工作流程。

要启用 CI/CD 工作流，请将您的项目上传到 Github，并通过提供您的 AWS 凭据作为`Github`机密，将 Github 存储库与您的 AWS 帐户连接起来。秘密应该有名字:

1.  *AWS_ACCESS_KEY_ID*
2.  *AWS_SECRET_ACCESS_KEY*

默认情况下，每次提交到`main`都会触发一个工作流`./github/workflows/deliver_images.yaml`，该工作流运行单元测试，构建并推送训练和端点映像:

持续交付培训/端点 docker 图像。

所有工作流程都可以通过[手动运行](https://docs.github.com/en/actions/managing-workflow-runs/manually-running-a-workflow)。

结论
BYOC 为配置和调整 SageMaker 项目提供了难以置信的灵活性，但代价是增加了复杂性。

为了让 BYOC 更容易理解，[“shape maker”](https://github.com/smaakage85/shapemaker)为亚马逊 Sagemaker 项目提供了一个完整的端到端模板，实现了 BYOC。

我希望，你会给[【塑造者】](https://github.com/smaakage85/shapemaker)一个旋转。如果你这样做了，我会喜欢你的反馈。

**资源** 这篇博客文章基于(并借用)以下由[玛利亚·维克斯拉德](https://github.com/m-romanenko)和[拉姆·维吉拉朱](https://github.com/RamVegiraju)写的博客文章——大声喊出来:

<https://www.sicara.fr/blog-technique/amazon-sagemaker-model-training>  </bring-your-own-container-with-amazon-sagemaker-37211d8412f4>  <https://github.com/smaakage85/shapemaker> 