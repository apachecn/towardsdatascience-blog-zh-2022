# DVC 的数据版本控制:了解其他数据科学家忽略了什么

> 原文：<https://towardsdatascience.com/how-to-version-gigabyte-sized-datasets-just-like-code-with-dvc-in-python-5197662e85bd>

## Python 中 DVC 数据版本控制的完整教程

![](img/4e8b1384461fa6a665fd37c54ac1b104.png)

菲奥娜·阿特摄于 Pexels

## 数据科学中的大问题

当数据集很大时，它会造成更大的混乱。为什么？数据科学家和 ML 工程师在大规模数据集和模型上进行许多实验，它们的庞大规模给协作和软件工程最佳实践带来了巨大的麻烦。

传统上，软件工程师通过制作中央代码库的副本并通过拉请求提出修改建议来进行协作。然后，请求被审查、测试，如果被批准，就合并到主代码库中。这个过程可能在一天内发生多次。

像 Git 这样的工具已经成熟了近二十年，使得上述过程对程序员来说轻而易举。但是，Git 只是为轻量级代码脚本设计的，而不是我们用来训练昂贵的 CNN 的成千上万的图像。

是的，有像 GitLFS 这样的替代品，但是设置起来太麻烦了；它不允许对大文件进行安全的分支、提交和实验，而这些是必备的特性。

因此，现在有许多工具可以解决这些问题。其中之一是 DVC(数据版本控制)。

## 什么是数据版本控制和 DVC？

数据版本控制是对数据集和模型更改进行跟踪和版本控制。一个好的数据版本控制系统必须具备以下特征:

1.  像 Git 处理脚本一样跟踪数据/模型变化。
2.  易于安装和使用:您应该能够用一个命令安装它。
3.  与 Git 等现有系统的兼容性，所以它不应该重新发明轮子。
4.  支持分支和提交:必须支持单独创建分支、提交和试验。
5.  再现性:允许其他团队成员快速、轻松地再现 ML 实验。
6.  共享功能:与其他用户无缝共享数据和模型以进行协作。

具有上述所有特性的一个工具是 DVC，它模仿 Git 的大文件特性。

Git 在 GitHub 或 GitLab 等托管服务上存储代码库，而 DVC 使用远程存储来上传数据和模型。远程存储可以是任何云提供商，如 AWS、GCP、Azure，甚至是本地机器上的一个普通目录。一个遥控器将是整个项目的唯一真实来源，由所有团队成员使用，就像 GitHub 存储库一样。

当 DVC 跟踪一个文件时，它会将它添加到远程存储中。然后，创建一个轻量级的`.dvc`文件(点 DVC ),作为原始大文件的占位符。它将包含 DVC 如何从遥控器下载文件的说明。

## 你会在教程中学到什么？

通过完成本教程，您将拥有一个用于图像分类项目的 GitHub 存储库。其他人只需两个命令就可以获得您的所有代码、数据、模型和实验:

这篇文章将教你运行`dvc pull`命令所需的一切，并理解几乎所有的东西。让我们直接跳进来吧！

## 设置项目和环境

让我们从创建一个`conda`环境开始:

接下来，转到您的 GitHub 帐户，并分叉[这个库](https://github.com/BexTuychiev/dvc-tutorial.git)。这将在您的帐户下创建确切的回购版本。然后，在终端上克隆它，并更改到工作目录。

```
$ git clone https://github.com/YourUsername/dvc-tutorial.git
$ cd dvc-tutorial
```

现在，让我们创建带有一些依赖项的`requirements.txt`文件并安装它们。

> 如果你还没有安装支持 GPU 的 TensorFlow，我在这里为你准备了一个指南。

> 运行带有`-e`标签的`echo`命令可以让它检测特殊字符，比如换行符(`\n`)。

我们安装了几个标准数据库:`scikit-image`用于图像处理，而`tensorflow`用于构建模型。最后一个是`dvc`，这是文章的主要重点。

现在，让我们构建项目的树形结构:

```
$ mkdir data notebooks src data/raw data/prepared data/prepared/train
```

我们将把脚本存储在`src`中，而`data`和`notebooks`将保存我们以后可能创建的图像和分析笔记本。

## 下载并设置数据

现在，我们将为项目下载数据集。GTSRB 德国交通标志识别基准数据集包含 50k 多幅图像，分为 40 个道路标志类别。我们的任务是建立一个卷积神经网络，可以准确地对每个类别进行分类。

您可以进入[数据集页面](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)或使用[此链接](https://storage.googleapis.com/kaggle-data-sets/82373/191501/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20221210%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20221210T130850Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=65eeae3c577195c0b9185b9e37ab185a3e5cc8c990a501390621201196cfd2e5ecbb0952db6bc443a09d08e252744472705c7bc90caa2c82aaa699b7d24f5592075046a771f05e424bb0d7fc6e8f8bff4e04e25a5e4e2b2e816a966e25df023050344400b97e676d9d0ac0c93c9046a007d74db740d311822fd79ea6bbdfa4d6459de2b2b061ca5187d2bf83c284feef39b06296cf4f46c7bc6f95c6488d7ea78a4eaf28ea43e7f8ef0afd97805d0943782b99377fd35a9e8781f17419d2fff43d66822d56c11802f209822dd86ba4e64edd7800d3125a7cff88b5616fbd3ddc0f2f3dfea2f86325cd185fc88cb5e46d517a846d407d4b6637df713cd8a36c36)和以下命令直接下载:

```
$ curl "the_link_inside_quotes" -o data/traffic_signs.zip
```

下载完成后，将图像解压到`data/raw`目录。然后，我们可以删除不必要的文件和目录，比如图像和元数据的副本。这将只给我们留下`data/raw`中的`train`和`test`文件夹。

最后，我们也删除了原始的 zip 文件。

`train`文件夹有 43 个文件夹，每个类一个。请记住这个目录结构，因为我们将在训练模型时使用它。

## 正在初始化 DVC

本节将展示 Git 和 DVC 如何协同工作的基础知识。

要将 DVC 跟踪添加到您的项目中，我们需要调用`dvc init`。DVC 只在 Git 库上工作，所以如果您将它用于其他项目，请确保您已经运行了`git init`命令。我们已经从 GitHub 派生了 repo，所以它已经初始化了 Git。

`dvc init`命令将添加一个保存 DVC 配置的特殊的`.dvc`目录。我们将在后面的章节中仔细研究 DVC 内部。

```
$ git status -s
A  .dvc/.gitignore
A  .dvc/config
A  .dvcignore
```

该命令创建了`.dvcignore`文件，可以用来列出 DVC 应该忽略的目录。Git 存储库已经预先填充了`.gitignore`文件。

一旦 DVC 被初始化，它需要一个叫做远程存储的地方来上传数据和大文件，这样 Git 就不会跟踪它们。DVC 远程可以是任何云存储提供商，如 AWS、Azure、GCP，或者只是你机器上的任何其他目录。

为了简单起见，我们将这个项目的远程存储设置到主目录中一个名为`dvc_remote`的新目录。

```
$ mkdir ~/dvc_remote
$ dvc remote add -d remote ~/dvc_remote
```

`remote`命令用于控制远程存储。在这里，我们将我们的远程存储简单地命名为`remote`。标签告诉 DVC`dvc_remote`是你默认的远程存储路径。

运行这些命令后，您可以查看`.dvc`文件夹中的`config`文件:

如您所见，远程名称被列为`remote`，而`url`被设置为我的主目录中的一个路径。如果我们的遥控器是基于云的，它将是一个网址。

## 使用 DVC 添加要跟踪的文件

要开始用 DVC 跟踪文件和目录的变化，您可以使用`dvc add`命令。下面，我们将整个`data`文件夹添加到 DVC 中，因为它包含了成千上万的图片，如果添加到`git`中，无疑会导致崩溃:

```
$ dvc add data
```

当运行`add`命令时，下面是发生的情况:

1.  目录被置于 DVC 的控制之下。
2.  `data`目录被添加到`.gitignore`文件中，因此它永远不会被`git`跟踪。
3.  创建一个轻量级的`data.dvc`文件，作为原始`data`目录的占位符。

这些轻量级的`.dvc`(点 DVC)文件被 Git 持续跟踪。当用户克隆我们的 Git 存储库时，`.dvc`文件将包含关于原始大文件存储位置的指令。

> 请记住，在`.gitignore`文件中的新行上添加文件或文件夹会使它们对`git`命令不可见。

现在，由于`data`目录被添加到了`.gitignore`中，我们可以安全地用`git`存放所有其他文件并提交它们:

```
$ git add --all
$ git commit -m "Initialize DVC and add the raw images to DVC"
```

所以，下面是如何结合使用 Git 和 DVC 的总结:

1.  每当您对代码或其他轻量级文件进行更改时，使用`git add filename`或`git add --all`跟踪这些更改。
2.  每当用`dvc`跟踪的大文件有变化时，通过运行`dvc add file/or/dir`来跟踪它，这会更新相应的`.dvc`文件。因此，您用`git add filename.dvc`将`.dvc`文件中的更改添加到`git`中。

例如，运行`python src/preprocess.py`将调整`raw/train`中所有图像的大小和比例，并将它们保存到`data/prepared/train`:

> 你可以从[这里](https://github.com/BexTuychiev/traffic_signs_recognition/blob/main/src/preprocess.py)复制/粘贴以上脚本的完整版本。

`resize`函数获取一个图像路径，并使用`imread`函数作为 NumPy 数组读取它。它被调整到`target_size`并保存到`prepared`目录中的新路径。

在`__main__`上下文中，我们收集所有图像路径，并使用并行执行来同时调整和保存多个图像。

一旦脚本完成，您可以使用`dvc status`查看 DVC 跟踪的文件是否有变化。您应该会看到类似下面的输出:

因此，我们用`dvc add`跟踪新的变更，用`git add --all`暂存对`data.dvc`所做的变更，并提交变更。

```
$ dvc add data
$ git add --all
$ git commit -m "Save resized images"
```

## 上传文件

现在，让我们推送所有使用`git`和 DVC 跟踪的变更进行的提交。我们运行`git push`，然后是`dvc`推送。

`git push`会将代码和`.dvc`文件上传到 GitHub，而`dvc push`会将原始的和调整后的图像发送到`remote`，也就是你机器上的`~/dvc_remote`目录。

```
$ git push
$ dvc push
```

一旦大文件存储在遥控器中，您就可以删除它们:

```
$ rm -rf data/raw/train
```

如果你想重新下载这些文件，你可以调用`dvc pull`:

```
$ dvc pull
```

`dvc pull`将检测工作目录和远程存储器之间的任何差异并下载它们。

当一个新用户克隆您的 Git 存储库时，他们也将使用`dvc pull`命令用存储在您的遥控器中的文件填充工作目录。

## 构建图像分类模型

是时候建立一个基线模型并和 DVC 一起追踪它了。在`src/train.py`中，我们有下面的脚本，它使用`ImageDataGenerator`类训练一个基线 CNN。由于本文的重点不是 TensorFlow，你可以从文档中了解`[ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)`[如何工作](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)。

> 你可以在的资源库[找到完整的脚本。](https://github.com/BexTuychiev/traffic_signs_recognition/blob/main/src/train.py)

脚本的关键部分是`main`函数。在内部，我们使用`joblib.dump`在新创建的`models`和`metrics`目录中拟合和保存模型及其度量。

我们运行脚本:

```
$ python src/train.py
```

完成后，我们将`models`目录添加到 DVC:

```
$ dvc add models
$ git add --all
$ git commit -m "Baseline model with 0.2192 accuracy"
```

然后，我们再次运行`git add --all`来暂存`models.dvc`文件和`metrics.dvc`文件。用`git`标记每个实验也是一个很好的做法:

```
$ git tag -a baseline -m "Baseline model with 0.2192 accuracy"
```

最后，我们推送提交，DVC 更改，并用以下内容进行标记:

```
$ dvc push
$ git push
$ git push origin --tags
```

现在，如果我们想通过尝试不同的 CNN 架构来提高准确率，我们修改`train.py`脚本，运行它，并跟踪新的`model.joblib`和`history.joblib`文件。我们还创建了总结模型性能的提交和标记。最后，我们用 Git 和 DVC 推送更改和标记。

尽管这个实验工作流简单而有效，但在文章的下一部分，我们将看到一种更好的方法来跟踪我们的实验。使用 DVC 管道和 VSCode DVC 扩展，我们将能够在 ide 中可视化我们的度量和模型运行。

## DVC 内部

现在你知道如何跟踪和上传文件到 DVC 远程，是时候深入了解 DVC 内部。

我们已经讨论过 DVC 远程，它类似于 GitHub，在那里你可以存储你用`dvc push`上传的数据和模型的最新官方版本。

但是，就像 Git 在将文件提交到 GitHub 之前先将文件添加到暂存区一样，DVC 有一个名为 cache 的暂存区。

当调用`dvc init`时，`cache`目录被添加到`.dvc`文件夹中。每次调用`dvc add`，文件都会被复制到缓存中。

现在，您会问——这难道不会重复文件和浪费空间吗？是啊！但是就像您可以配置远程存储的位置一样，您也可以配置缓存。

在大型项目中，许多专业人员共用一台功能强大的机器，而不是笔记本电脑或个人电脑。因此，让每个团队成员在自己的工作目录中都有一个缓存是没有意义的。一种解决方案是将缓存指向共享位置。

如果您一直在关注，我们的项目缓存在`.dvc/cache`下。但是我们可以使用以下命令指向另一个目录:

```
$ dvc cache dir path/to/shared_cache
$ mv .dvc/cache/* path/to/shared_cache
```

`mv`命令将旧缓存中的文件移动到新的缓存位置。

共享一台开发机器时，确保所有团队成员都有读/写权限`path/to/shared_cache`。

> 如果是自己一个人在工作，就没必要按照这一步来。

## 结论

以下是与 DVC 合作的总结:

*   DVC 项目是在 Git repo 之上用`dvc init`初始化的
*   您应该使用`dvc remote add -d remote_name path/to/remote`为项目设置一个遥控器
*   要开始跟踪文件，请使用`dvc add`
*   `dvc add`将指定的目录或文件复制到`.dvc/cache`或`shared_cache/you/specified`，为每个被跟踪的文件夹或文件创建`.dvc`文件并添加到`.gitignore`
*   `.dvc`等文件用`git add --all`跟踪
*   要推送提交和 DVC 跟踪的文件更改，请同时使用`git push`和`dvc push`
*   `dvc push`将文件从缓存上传到远程存储器
*   用标签标记每个 ML 实验运行，并对每个更改的文件重复`dvc add` / `dvc push`和`git add` / `git push`。

这个循序渐进的教程已经足以解决您在数据科学项目中关于协作和可再现性的大多数问题。在文章的下一部分，我们将讨论用 DVC 简化机器学习实验(是的，它可以变得更简单)！

感谢您的阅读！

文章的第二部分:

[](https://pub.towardsai.net/how-to-track-ml-experiments-with-dvc-inside-vscode-to-boost-your-productivity-a654ace60bab)  

第三部分:

[](https://pub.towardsai.net/how-to-create-highly-organized-ml-projects-anyone-can-reproduce-with-dvc-pipelines-fc3ac7867d16)  [](https://ibexorigin.medium.com/membership)  [](https://ibexorigin.medium.com/subscribe)  

## 数据集的引用:

J.Stallkamp、M. Schlipsing、J. Salmen 和 C. Igel。德国交通标志识别基准:多类别分类竞赛。在*IEEE 神经网络国际联合会议记录*中，第 1453-1460 页。2011.

以下是我的一些故事:

[](/how-to-create-perfect-machine-learning-development-environment-with-wsl2-on-windows-10-11-2c80f8ea1f31)  [](/how-to-boost-pandas-speed-and-process-10m-row-datasets-in-milliseconds-48d5468e269)  [](/a-complete-shap-tutorial-how-to-explain-any-black-box-ml-model-in-python-7538d11fae94)  [](/3-step-feature-selection-guide-in-sklearn-to-superchage-your-models-e994aa50c6d2) 