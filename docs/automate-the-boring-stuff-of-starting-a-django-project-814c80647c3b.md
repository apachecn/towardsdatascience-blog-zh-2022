# 自动化启动 Django 项目的枯燥工作

> 原文：<https://towardsdatascience.com/automate-the-boring-stuff-of-starting-a-django-project-814c80647c3b>

## 运行一个脚本，按照您需要的方式设置项目

![](img/38690a39813203a978be299dd82ab571.png)

照片由 [Diego PH](https://unsplash.com/@jdiegoph?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/automation?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

**简介**

这篇文章的标题显然是对 Al Sweigart 的书 [*的致敬，这本书用 Python*](https://automatetheboringstuff.com/2e/) 自动化了枯燥的东西，我总是推荐给那些想以正确的方式开始学习 Python 的人。虽然 Al 的书没有涵盖 Django，但它让读者很好地理解了 Python 是如何成为自动化枯燥、重复的日常任务的伟大工具。

你们中的一些人，在阅读了我的一篇或多篇教程后，可能已经知道我是姜戈的忠实粉丝。但我们必须承认，当开始一个新的 Django 项目时，我们需要在终端上运行大量命令，手动创建一些文件夹和文件，并在其他一些文件夹和文件上进行周期性的更改。大多数初始任务都是重复性的，这为自动化提供了很好的机会。

因此，[我写了这个 Python 脚本](https://github.com/fabricius1/my-django-starter)，它允许我在开始工作之前，按照我需要的方式创建一个新的 Django 项目。与另一种方法相比，我更喜欢这种方法，即复制或克隆一个 starter Django 项目。事实上，我认为这个脚本给了我更多的灵活性，可以按照我需要的方式定制最初的项目结构，例如，如果我愿意，可以一次创建两个、五个、十个或一百个应用程序。

(这里有个小提示:请不要在你的 Django 项目上创建一百个应用。你能，不代表你应该)。

我不会一行一行地讨论我的代码，因为你可以参考`README.md`文档和关于`my_django_starter.py`的评论来了解细节。我将对我想通过编写这个自动化脚本来解决的问题做一些评论。

**我的完美 DJANGO 启动项目特色**

1.它已经包含了我工作中需要的所有新应用程序文件夹，具有以下特征:

*   一个`APP_NAME/templates/APP_NAME`文件夹；
*   一个`urls.py`文件，因为我喜欢让我的应用程序路径由应用程序文件夹中的一个特定的`urls.py`文件处理；
*   在`urls.py`和`views.py`中的代码足以让路线`http://localhost:8000/APP_NAME`工作。

2.已经执行了以下终端命令:

```
django-admin startproject PROJECT_NAME .python manage.py startapp APP_NAME # for each apppython manage.py migratepython manage.py createsuperuser
```

3.这些额外的文件夹被创建并准备好使用:

```
/media
/scripts
/templates
/templates/static
```

4.对`PROJECT_NAME/settings.py`文件的修改:

*   `django_extensions`包和我的所有应用名称已经自动包含在`INSTALLED_APPS`列表中；
*   设置`static`、`media`和`templates`目录的代码；
*   初始多行注释已经删除(与`urls.py`中的注释相同)

5.在`PROJECT_NAME/urls.py`文件中的改变，使得应用程序`urls.py`文件已经被包括并且它的初始路线可以被导航。

这都是通过遵循我在关于如何将 Python Jupyter 笔记本转换成 RMarkdown 文件的文章 [*中已经探索过的原则实现的:*](/how-to-convert-a-python-jupyter-notebook-into-an-rmarkdown-file-abf826bd36de) : `.py`和`.html`文件是纯文本文件，因此它们可以被我们的 Python 代码作为字符串读取、编辑和保存。

**结束语**

我们可以为一个初始的 Django 项目制作更进一步的自动化代码，比如:

*   创建带有`{% extends 'base.html' %}`和`{% block content %}`标签的模板文件；
*   从一开始就让其他一些文件和目录可用；
*   添加代码，从我们在`my_django_starter.py`的`config_info`部分设置的值自动改变`LANGUAGE`和`TIMEZONE`；
*   如果我们正在用基于类的视图制作 CRUD 应用程序，我们已经可以用由类视图要求的名字(`APP_NAME_list.html`、`APP_NAME_form.html`、`APP_NAME_detail.html`等等)构成的`.html`文件填充`templates/APP_NAME`文件夹，我们也可以给它们一些基本的内容；

现在轮到您了:创建您自己的自动化脚本，以您需要的方式启动您的新 Django 项目。只是不要在不必要的时候一遍又一遍地输入相同的命令。你是一个程序员:写一个程序让它为你工作。

如果你想了解更多关于 Django 的内容，请查看下面链接的文章。

亲爱的读者，我再次感谢你花时间和精力阅读我的文章。

*快乐编码*！

[](/build-a-django-crud-app-by-using-class-based-views-12bc69d36ab6)  [](/django-first-steps-for-the-total-beginners-a-quick-tutorial-5f1e5e7e9a8c)  [](/create-a-django-app-with-login-restricted-pages-31229cc48791)  [](/use-python-scripts-to-insert-csv-data-into-django-databases-72eee7c6a433) 