# 如何打造一个 AI 服装设计师

> 原文：<https://towardsdatascience.com/how-to-build-an-ai-fashion-designer-575b5e67915e>

## 基于 StyleGAN 和 GANSpace 的服装语义编辑

![](img/ddbcbc36959f6f55a9cdb4d3fb61346e.png)

ClothingGAN 演示[图片由作者提供]

# 概观

这是我的老项目[clothing an](https://github.com/mfrashad/ClothingGAN)的一篇报道。该项目使用 StyleGAN 用 AI 生成服装设计，并使用袖子、尺码、连衣裙、夹克等属性对其进行语义编辑。您也可以如上图所示进行风格转换，首先生成 2 个不同的服装设计(输出 1)，使用不同的种子编号。然后，它将生成第三个设计(输出 2)，该设计混合了前两个设计。然后，您可以调整希望它从两个原始设计中继承多少风格或结构。

> 你可以试试这里的[演示](https://huggingface.co/spaces/mfrashad/ClothingGAN)，这里是[源代码](https://github.com/mfrashad/ClothingGAN)(随意开始回购)

# 概述

*   灵感
*   我是如何建造的
*   训练风格 GAN 模型
*   用 GANSpace 进行语义编辑
*   用 Gradio 构建 UI
*   部署到巨大的空间

# 灵感

GAN 或生成对抗网络是一种生成模型，它能够通过学习大型图像数据集的概率分布来生成图像。我总是觉得 GANs 很吸引人，因为它使我能够创作出高质量的艺术或设计，即使没有绘画方面的技术或艺术技巧。最近在 GAN 上看到很多人脸编辑演示，但在其他数据集上很少看到语义操纵。因此，我创建了 ClothingGAN 应用程序，你可以与人工智能合作设计衣服，而不需要很高的技术专业知识。

# 我是如何建造的

第一步，要有一个可以生成服装的生成模型。我没有设法找到一个公共模型，可以产生一个体面的质量图像，因此我决定用样式训练我自己的甘服装模型。然后我使用 GANSpace，一种基于潜在空间的语义编辑方法，来提供编辑功能。它在 GAN 潜在空间中找到重要的方向，这些方向可能代表某些视觉属性，然后我手动标记这些属性。最后，我使用 Gradio 库构建了演示界面，并将其部署到 HuggingFace 空间。

# 训练风格 GAN 模型

我使用了 StyleGAN2-ADA[2]模型，因为在项目进行的时候，最新的 StyleGAN 模型是 StyleGAN2-ADA 模型。但是，您可能希望使用当前最新版本 StyleGAN3。虽然我不确定 StyleGAN3 与我的方法或我正在使用的其他库有多兼容。

为了训练模型，我使用了由[dong eun Yoo](https://dgyoo.github.io/)在[PixelDTGAN【1】](https://github.com/fxia22/PixelDTGAN)论文中创建的服装数据集。该数据集具有 84，748 幅图像，包括 9，732 幅具有干净背景的上部服装图像，这些图像与剩余的 75，016 幅时装模特图像相关联。我只使用了背景干净的服装图片。因此，用于训练 StyleGAN 模型的总图像大约是分辨率为 512×512 的 9k 图像。这是作者网站上分享的数据集的[链接。PixelDTGAN 论文受麻省理工学院许可。](https://dgyoo.github.io/)

![](img/30a452592d46d84ef3ebc1aeea6b2ab6.png)

LookBook 数据集一览[图片由作者提供，数据集由 [PixelDTGAN[1]](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_31) 提供]

我不会讨论如何训练模型的具体步骤，因为我之前已经就这个主题写了一篇文章。只需对所选数据集执行相同的步骤。

[](/how-to-train-stylegan2-ada-with-custom-dataset-dc268ff70544)  

这是训练后的结果。

![](img/f7595d1e055d42096c00cc58823972d9.png)![](img/3793467fbb18853a6109fe8923fa740f.png)

由训练模型生成的设计的样本和插值[图片由作者提供]

# 用 GANSpace 进行语义编辑

语义图像编辑是在给定的源图像中修改语义属性(例如样式或结构)的任务。例如，修改一个人的头发颜色，同时保留这个人的身份。图像编辑的应用范围很广，从照片增强、用于艺术和设计目的的风格处理到数据扩充。语义图像编辑通常有两个目标:允许同时对多个属性进行连续操作，并在保持图像真实性的同时尽可能多地保留源图像的身份。

现有的使用 GANs 的语义图像编辑方法可以主要分为图像空间编辑或潜在空间编辑。图像空间编辑学习直接将源图像转换成目标域中的另一图像的网络。这些方法通常只允许二进制属性改变，而不允许连续改变。这些方法的例子有 pix2pix、StarGAN 和 DRIT++等。

相反，潜在空间编辑通过操纵跨越 GAN 模型的潜在空间的输入向量来间接操纵图像。这些方法主要集中于在潜在空间中寻找代表生成图像的语义属性的路径。在这些路径中导航输入向量允许连续编辑属性。

无监督、自我监督和监督的潜在空间编辑方法都已被提出。gan space[【3】](https://proceedings.neurips.cc/paper/2020/hash/6fe43269967adbb64ec6149852b5cc3e-Abstract.html)在潜在或特征空间中使用主成分分析(PCA)以无监督的方式寻找重要方向。使用闭型因子分解也可以类似地找到重要方向(来自 SeFa 论文)。自监督方法也能够在没有标签的情况下找到这些方向，因为它们生成自己的标签，但是通常受限于几何属性，例如旋转或缩放。另一方面，像 InterfaceGAN 这样的监督方法需要标签信息或方法的属性分类器。

[GANSpace[3]](https://proceedings.neurips.cc/paper/2020/hash/6fe43269967adbb64ec6149852b5cc3e-Abstract.html) 讨论了预训练的 GAN 模型在对生成的图像进行造型时的使用。GAN 模型学习将噪声分布 z 映射到图像分布的函数。因此，给定不同的噪声输入 z，产生的输出将会不同。然而，深度学习模型通常是一个黑箱，它并不明确知道噪声输入和生成的输出之间的关系，因此无法明确控制输出。然而，GAN 模型可以被调整为在给定类标签的情况下生成特定的类输出，如条件 GAN 中所研究的。然而，在训练期间，需要数据集的标签信息来调节 GAN 模型，这对于某些情况可能是不可行的。

另一方面，GANSpace 的论文[3]提出，可以在 z 潜在空间中找到某些重要的方向，z 潜在空间表示生成的输出中的已知语义概念，例如输出的风格。为了找到这个方向，对于几个样本观察中间层中的激活，并且从中间网络激活空间中的值计算 PCA 方向 v。然后，方向 v 将被转移以在 z 潜在空间中找到对应方向 u。整个过程如下图所示，取自 GANSpace 的论文。

![](img/8f576f8210780aeb0eaf67660bdc6a5a.png)

在 GAN 潜在空间中识别 PCA 方向的 2D 图解[来源: [GANSpace 论文[3]](https://proceedings.neurips.cc/paper/2020/hash/6fe43269967adbb64ec6149852b5cc3e-Abstract.html)

可以在不同层中计算重要方向 u，并且每层中的方向可以表示不同的语义概念。在早期层中找到的方向通常表示高级特征，如布料结构，而在最后几层中找到的方向通常表示低级特征，如光照或颜色。通过在这些已知的方向上操纵噪声输入 z，我们可以将生成的输出操纵到期望的特征。下图显示了在不同 GAN 模型中应用 GANSpace 方法时的操作结果。

![](img/c1cba4f690e22b8f74decc1bbc03eb0a.png)

GANSpace 的结果是不同的模型。[来源:[甘斯佩斯论文[3]](https://proceedings.neurips.cc/paper/2020/hash/6fe43269967adbb64ec6149852b5cc3e-Abstract.html)

## 在训练好的模型中寻找方向

这里将显示的代码是在 Google Colab 上测试的，你可以跟随[我的笔记本](https://colab.research.google.com/drive/1JRJ9GfnLKybQFbs7iE3PEz4DZrAN4BY8?usp=sharing)或者在你自己的环境中，但是如果你跟随 Colab 环境之外的环境，确保你的环境有预先安装在 Colab 中的依赖项。

> 如果你想跟随，这是教程笔记本

首先，我们需要安装 GANSpace 所需的依赖项。

```
!pip install ninja gradio fbpca boto3 requests==2.23.0 urllib3==1.25.11`
```

运行完代码后重启运行时，然后克隆 GANSpace repo。

```
!git clone https://github.com/mfrashad/ClothingGAN.git%cd ClothingGAN/
```

运行以下代码进行进一步设置。确保您位于 GANSpace 文件夹中。

```
!git submodule update --init --recursive!python -c "import nltk; nltk.download('wordnet')"
```

接下来，我们必须修改 GANSpace 代码来添加我们的定制模型。对于 StyleGAN2，我们需要模型文件的 PyTorch 版本。因为我们的 StyleGAN 模型文件在 Tensorflow 中。pkl 格式，我们需要用 rosinality 做的转换器把它改成 pytorch 格式。pt 文件。只要按照这个[笔记本](https://colab.research.google.com/github/dvschultz/stylegan2-ada-pytorch/blob/main/SG2_ADA_PT_to_Rosinality.ipynb)里的步骤就可以了。(该项目是在正式的 StyleGAN2 PyTorch 版本实现之前完成的，如果您的模型文件已经存在，您可以跳过这一部分。pt 或 Pytorch 格式)。

接下来，回到 GANspace 文件夹，修改`models/wrappers.py`来添加我们的模型文件。首先，转到 StyleGAN2 类，在`config`变量中添加我们的模型名称和输出分辨率。

![](img/1faef0ba0f789fa646d90fbf9d0e2437.png)

我在 models/wrappers . py[Image by Author]的第 117 行的配置变量中添加了分辨率为 512x512 的“lookbook”模型

接下来，再向下滚动一点，在`checkpoints`变量中添加到模型的链接。要生成到我们模型的链接，只需将模型文件上传到 Google drive，并使用[这个站点](https://sites.google.com/site/gdocs2direct/)生成到它的直接链接。

![](img/c614bfa0227299783b46d8185f774e16.png)

我在 models/wrappers.py 文件的第 149 行添加了一个新的生成器模型“lookbook”

将模型添加到文件中之后。运行`visualize.py` c 脚本进行 PCA，并在向计算出的主成分方向移动输入时可视化视觉变化。

![](img/8a70b543656b21883eff3d9e0258fdce.png)

用于 PCA 和可视化变化的命令[图片由作者提供]

`--use_w`选项意味着我们将操纵中间潜在代码`w`，而不是 StyleGAN 中的原始潜在代码`z`。`num_components`是指定你想保留多少个方向或主成分。最大组件将是 512 或输入`z`或`w`尺寸。`--video`选项是在主成分方向上移动时生成视觉变化的视频，而不仅仅是生成图像。该脚本可能需要大约 30 分钟才能完成。

一旦完成，它将在 out 文件夹中生成可视化的更改。对我来说，它在`out/StyleGAN2-lookbook`文件夹下。

![](img/9263ac9ab2593eb53bf441cb4fdefd34.png)

生成的可视化输出[图片由作者提供]

我们将看一下`style/ipca/summ/components_W.jpg`，因为它可视化了前 14 个主要组件。

![](img/48c1f9fbde7d996fa2b66787771b8edf.png)

前 14 个主成分的可视化[图片由作者提供]

从上图中，我们可以开始选择要放入演示中的主要组件，并给它们贴上标签。例如，在我看来，C0 可以标记为袖长，C1 为夹克，C2 和 C3 为外套，C4 和 C5 为服装的亮度，C6 为较短的服装。

您还可以在附加文件`sampX_real_W.jpg`中看到不同样本的可视化，以确保由主成分引起的变化在不同样本之间保持一致。还有 9 个额外的样本是由`visualize.py`脚本生成的。

这是另一个例子的可视化。

![](img/01e88e48b9705d02f937d54ad2e9b836.png)

不同样本的主要成分的可视化[图片由作者提供]

你可以看到，即使是不同的样品(袖长为 C0，夹克为 C1 等)，变化也大致一致。

此外，您还可以在`comp`或`inst`文件夹中看到每个组件的可视化视频。主成分本身以`.npz`的格式保存在`cache/components/`文件夹中

![](img/2e5b2bc0c1c1944f1e1ad772053870c8.png)

计算出的主成分文件的位置[图片由作者提供]

一旦我们有了组件，我们就可以开始构建演示 UI 了。

# 用 Gradio 构建 UI

Gradio 是一个 python 库，它使得用几行代码构建 ML 项目的 UI/demo 变得极其容易。下面是 Gradio 有多简单的一个例子:

![](img/dc76082cc2a806304cb6289d768706cd.png)![](img/1bfef46acb10ce615db9466228e2bfc3.png)

示例代码和使用 Gradio 生成的应用程序[图片由作者提供]

当你想把你的 ML 应用程序演示成一个单一的函数时，Gradio 是合适的。

首先，我们需要将发电机模型和主要组件加载到内存中。

然后，我们将定义一个效用函数来操作指定方向的`w`输入，并使用生成器生成图像。

最后，我们可以定义主函数`generate_image`并使用 Gradio 库为该函数构建 UI。

这就是结果！

![](img/91aeb2831bac915f6850cd759ff2622c.png)

带有 Gradio 的演示用户界面[图片由作者提供]

Gradio 还将提供一个链接，您可以在那里分享您的朋友或任何人都可以尝试演示。然而，演示的主机并不是永久的，Colab 服务器被限制在 12 或 24 小时后才会自行终止。对于永久托管，您可以简单地在云中或您自己的服务器上运行代码。但幸运的是，拥抱脸创造了 [Spaces](https://huggingface.co/spaces) ，一个你可以简单上传你的 ML 应用并永久免费托管的平台(如果你想要 GPU 需要付费)。此外，它与 Gradio 和 Streamlit 完美集成，开箱即用。

# 部署到巨大的空间

首先，前往[空间](https://huggingface.co/spaces)并登录/注册您的帐户。然后单击“创建新空间”。

![](img/d9ea9c38fafbe77b5b47144d21a452c3.png)

拥抱脸空间主页[图片由作者提供]

然后，选择您想要的名称和许可证，并选择 Gradio 作为 space SDK。

![](img/1ea56fa2509205d3ba80bae259f1ad8b.png)

创造新空间[作者图片]

接下来，克隆拥抱脸回购。

![](img/a79f25447828fcaae655bb92ac5b4791.png)

[图片由作者提供]

就我个人而言，在推送回购时，我遇到了一个身份验证问题，我不得不使用令牌作为身份验证方法，方法是将远程 URL 设置为:

```
https://HF_TOKEN@huggingface.co/spaces/mfrashad/Test
```

或者你也可以在 URL 中使用你的拥抱脸账户的用户名和密码进行认证。

```
https://HF_USERNAME:PASSWORD@huggingface.co/spaces/mfrashad/Test
```

一旦克隆完成，我们就可以开始创建演示所需的文件。在 Spaces repo 中需要 3 个重要的文件:`requirements.txt`指定要与`pip install`一起安装的所有 python 依赖项，`packages.txt`指定要与`apt install`一起安装的依赖项，以及包含 Gradio 演示代码的主 python 文件`app.py`。

此外，您需要使用 git-lfs 将任何二进制文件上传到 repo，例如图像。

所以我所做的只是将我们在 Colab 中的所有代码复制到 Spaces repo 中。删除图像、二进制文件和演示不需要的文件。把我们笔记本里所有的 python 代码放到一个 python 文件`app.py`里。然后创建`requirements.txt`和`packages.txt`。一旦完成，简单地`git push`和瞧！该演示将在拥抱脸空间提供给任何人尝试(假设你没有任何错误)。

> 关于代码的完整内容，你可以查看[clothingan Space repo](https://huggingface.co/spaces/mfrashad/ClothingGAN/tree/main)中的文件。

![](img/47693f89bce24ac9065de3a903a5d18b.png)

您的共享空间演示将显示在您的个人资料和共享空间主页上[图片由作者提供]

恭喜你！你设法一直读到这一点，并希望设法做好一切。对于更多的挑战，您可以尝试训练自己的 StyleGAN 模型，并应用语义编辑。例如，我也将相同的方法应用于角色和时装模型的生成。

[你可以在这里试试 CharacterGAN 的试玩](https://huggingface.co/spaces/mfrashad/CharacterGAN)。

![](img/398db2c244aaf9f8e10eb6409bb58ec7.png)

字符生成和语义编辑[图片由作者提供]

如果你喜欢阅读这样的教程，并希望支持我成为一名作家，可以考虑注册成为一名媒体会员。每月 5 美元，你可以无限制地阅读媒体上的故事。如果你注册使用我的链接，我会赚一小笔佣金。

[](https://medium.com/@mfrashad/membership)  

另外，看看我的其他故事。

[](/how-i-built-an-ai-text-to-art-generator-a0c0f6d6f59f)  [](/animating-yourself-as-a-disney-character-with-ai-78af337d4081)  

# 参考

[1] Yoo，d .，Kim，n .，Park，s .，Paek，A. S .，& Kweon，I. S. (2016 年 10 月)。像素级域转移。在*欧洲计算机视觉会议*(第 517–532 页)。斯普林格，查姆。

[2]t . Karras，m . Aittala，j . hells ten，Laine，s .，Lehtinen，j .，& Aila，T. (2020 年)。用有限数据训练生成性对抗网络

[3]哈尔科宁、赫茨曼、莱蒂宁和巴黎(2020 年)。Ganspace:发现可解释的 gan 控制。*神经信息处理系统的进展*， *33* ，9841–9850。