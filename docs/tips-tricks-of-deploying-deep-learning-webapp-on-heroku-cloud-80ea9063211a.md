# 在 Heroku Cloud 上部署深度学习 Webapp 的技巧和诀窍

> 原文：<https://towardsdatascience.com/tips-tricks-of-deploying-deep-learning-webapp-on-heroku-cloud-80ea9063211a>

# 在 Heroku Cloud 上部署深度学习 Webapp 的技巧和诀窍

## 通过在 Heroku 服务器上部署基于 Tensorflow 的图像分类器 Streamlit 应用程序，我学到了一些东西

![](img/b2ee4cfa65c83f02de1420625c5b89e5.png)

作者图片

Heroku 是 web 开发人员和机器学习爱好者中的一个著名平台。该平台提供了一种部署和维护 web 应用程序的简单方法，但如果您不熟悉部署深度学习应用程序，您可能会遇到存储和依赖问题。

本指南将使您的部署过程更加紧凑，这样您就可以专注于创建令人惊叹的 web 应用程序。我们将学习 DVC 集成、基于 Git 和 CLI 的部署、错误代码 H10、使用 python 包以及优化存储。

# 基于 Git 和 CLI 的部署

Streamlit 应用程序可以通过基于 Git 的 GitHub 集成或使用 Docker 来部署。基于 Git 是在 Heroku 服务器上部署任何数据应用程序的更快更简单的方法。

## **简单的基于 Git 的**

可以使用以下方式部署 streamlit 应用程序:

```
git remote add heroku https://heroku:$HEROKU_API_KEY@git.heroku.com/<name of your heroku app>.git

git push -f heroku HEAD:master
```

**为此，您需要:**

*   Heroku API 密钥
*   Heroku App:通过 CLI 或使用网站。
*   基于 Git 的项目
*   Procfile
*   Requirements.txt

## 基于 CLI

基于 CLI 的部署是简单易学的。

![](img/bc4ad779a75a771aa0fd743f598b7184.png)

作者图片

1.  在这里创建一个**免费** Heroku 账号[。](https://signup.heroku.com/)
2.  使用此[链接](https://devcenter.heroku.com/articles/heroku-cli#download-and-install)安装 Heroku CLI。
3.  克隆远程存储库或使用`git init`
4.  类型`heroku login`和`heroku create dagshub-pc-app`。这将使您登录到服务器并在 web 服务器上创建一个应用程序。
5.  现在创建 **Procfile** ，其中包含运行应用程序的命令:`web: streamlit run --server.port $PORT streamlit_app.py`
6.  最后，将代码提交并推送到 heroku 服务器`git push heroku master`

# 港口

如果您使用`streamlit run app.py`运行应用程序，它将产生一个错误代码 **H10** ，这意味着服务器分配的 **$PORT** 未被 streamlit 应用程序使用。

**你需要:**

*   使用 Heroku CLI 设置端口

```
heroku config:set PORT=8080
```

*   在您的 **Procfile** 中进行更改，并在参数**中添加服务器端口。**

```
web: streamlit run --server.port $PORT app.py
```

# 调整 Python 包

这部分花了我两天时间调试，因为 Heroku cloud 有 **500MB** 的限制，而新的 [TensorFlow](https://pypi.org/project/tensorflow/#files) 包是 **489.6MB** 。为了避免依赖性和存储问题，我们需要对 **requirements.txt** 文件进行更改:

1.  添加 **tensorflow-cpu** 代替 tensorflow，这将使段塞大小从 765MB 减少到 400MB。
2.  添加 **opencv-python-headless** 代替 opencv-python，避免安装外部依赖。这将解决所有 cv2 错误。
3.  移除除**数字、枕头**和**流线型**之外的所有不必要的包装。

# DVC 一体化

![](img/a115e94ba55b956059e21025b2eaf571.png)

作者图片

从 DVC 服务器成功提取数据需要几个步骤。

1.  首先，我们将安装一个[构建包](https://elements.heroku.com/buildpacks/heroku/heroku-buildpack-apt)，它将允许使用 Heroku API `heroku buildpacks:**add** --**index** 1 heroku-community/apt`安装 apt 文件
2.  创建一个文件名 **Aptfile** 并添加最新的 DVC 版本[https://github . com/iterative/DVC/releases/download/2 . 8 . 3/DVC _ 2 . 8 . 3 _ amd64 . deb](https://github.com/iterative/dvc/releases/download/2.8.3/dvc_2.8.3_amd64.deb)
3.  在您的 **app.py** 文件中添加额外的代码行，以从远程 DVC 服务器获取数据:

```
**import** os

**if** "DYNO" **in** os.environ **and** os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    **if** os.system(f"dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")
```

之后，提交并推送你的代码到 Heroku 服务器。成功部署后，该应用程序将自动从 DVC 服务器拉数据。

# 优化存储

优化存储有多种方法，最常用的是使用 Docker。通过使用 docker，你可以绕过 500MB 的限制，你也可以自由地安装任何第三方集成或软件包。要了解更多关于如何使用 docker 的信息，请查看本指南。

**用于优化存储:**

*   仅在 requiremnets.txt 中添加模型推理 python 库
*   我们可以通过使用`dvc pull {model} {sample_data1} {sample_data2}..`从 DVC 提取选择性数据
*   我们只需要模型推理文件，所以将其余文件添加到`.slugignore`中，它的工作方式类似于`.gitignore`，以了解更多信息，请查看 [Slug 编译器](https://devcenter.heroku.com/articles/slug-compiler#ignoring-files-with-slugignore)。
*   从服务器成功拉取数据后，删除`.dvc`和`.apt/usr/lib/dvc` 目录。

# 结果

最初的块大小是 850MB，但随着存储和封装的优化，最终的块大小减少到 400MB。我们用一个简单的命令解决了错误代码 H10，并添加了 opencv-python-headless 包来解决依赖性问题。本指南旨在解决 Heroku 服务器初学者面临的一些常见问题。

基于 [Docker 的](https://devcenter.heroku.com/categories/deploying-with-docker)部署可以解决很多存储问题，但它带来了复杂性和缓慢的部署过程。您可以使用`heroku container:push web`,但在此之前，您需要构建 docker，测试它，并在本地解决所有问题，然后才能推送它。这种方法是高级 Heroku 用户的首选。

下一个挑战是用 webhook 部署您的 web 应用程序。这将允许您从任何平台自动化整个机器学习生态系统。自动化过程将包括创建一个运行 shell 命令的简单 Flask web 服务器。

# 额外资源

*   [创建一个 Streamlit web 应用程序，使用 Docker + GitHub 操作进行构建，并托管在 Heroku | R-bloggers 上](https://www.r-bloggers.com/2020/12/creating-a-streamlit-web-app-building-with-docker-github-actions-and-hosting-on-heroku/)
*   [使用 Python 部署影像分类 Web 应用| Ruben Winastwan |面向数据科学](/deploying-an-image-classification-web-app-with-python-3753c46bb79)
*   [GuilhermeBrejeiro/deploy _ ML _ model _ Heroku _ FastAPI](https://github.com/GuilhermeBrejeiro/deploy_ML_model_Heroku_FastAPI)
*   Heroku CLI | Heroku 开发中心

*原载于 2021 年 12 月 24 日*[*K*Dnuggets](https://www.kdnuggets.com/2021/12/tips-tricks-deploying-dl-webapps-heroku.html)*。*