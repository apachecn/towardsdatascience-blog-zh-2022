# 使用基于类的视图构建 Django CRUD 应用程序

> 原文：<https://towardsdatascience.com/build-a-django-crud-app-by-using-class-based-views-12bc69d36ab6>

## 是的，对姜戈来说就是这么简单和快速

![](img/1099d84c220645bb60a253ee9ef37eff.png)

丹尼尔·科尔派在 [Unsplash](https://unsplash.com/s/photos/screen?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

**简介**

在本教程中，我将向您展示如何使用 Django 最强大的功能之一:基于类的视图(CBV)来构建一个完全功能化的 CRUD(创建-读取-更新-删除)应用程序。这种方法最大化了代码重用，并允许您更快、更有效地构建 Django 应用程序。

所以，没有任何延迟，让我们开始吧。我们将创建一个简单的应用程序来管理电影及其流派的信息。如果你愿意，可以访问[我的 GitHub 库上的完整代码。或者继续阅读并遵循以下步骤。我使用 GitBash for Windows 作为我的终端，所以这里的一些命令可能需要适应您的首选终端，特别是如果您使用 Power Shell 或 Windows 命令提示符。](https://github.com/fabricius1/DjangoFilmsCRUD)

**第 1 部分:建立项目**

1.创建您的项目文件夹，并在其中移动。我把我的叫做`films_project`。

```
mkdir films_project
cd films_project
```

2.用`venv`(我的名字叫`.myenv`)创建一个新的虚拟环境。

```
python -m venv .myenv
```

3.现在激活你的虚拟环境。下面的命令适用于 GitBash。参考这个 [Python 文档页面](https://docs.python.org/3/library/venv.html)来检查你的终端和操作系统的命令。

```
source .myenv/Scripts/activate
```

4.安装我们在这个项目中需要的 Python 包:

```
pip install django django-extensions
```

5.创建一个`requirements.txt`文件。您可以使用以下命令:

```
pip freeze | grep -i django >> requirements.txt
```

6.开始一个新的 Django 项目。我的就叫`project`。不要忘记在这个命令的末尾加上点。

```
django-admin startproject project .
```

7.启动一个叫`films`的 Django app。

```
python manage.py startapp films
```

8.打开`project/settings.py`并将`django_extensions`和`films`应用添加到`INSTALLED APPS`列表中。不要换其他线。

```
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles', # add these lines: 
    'django_extensions',
    'films.apps.FilmsConfig',
]
```

9.应用迁移在 sqlite3 数据库中创建表。

```
python manage.py migrate
```

10.通过运行以下命令创建超级用户:

```
python manage.py createsuperuser
```

告知用户名、电子邮件和密码。稍后，您将使用这些数据登录管理区域。

11.使用`python manage.py runserver`运行服务器，访问`http://localhost:8000`并检查应用程序是否正常工作。

12.去`http://localhost:8000/admin`，使用你的登录数据，检查你是否可以访问管理区。

**第二部分:定义模型**

13.在`films/models.py`中，输入以下代码:

我们正在创建两个模型类(`Film`和`Genre`)，它们之间有一个简化的关系，因为一部电影在这里只能有一个类型。将来，如果需要，您可以将这些表之间的关系改为多对多关系，或者您甚至可以向模型中添加其他实体，如主管或奖项。

在`Films`类中，注意重要的`get_fields()`方法。这将为`films_film`表中除 id 之外的每一列创建一个标签/值对列表，我们可以在未来的`film_detail.html`模板中的 for 循环中使用这个结构。如果我们在一个模型中有几十甚至几百个列，这样的结构可能会更有帮助，所以这段代码可能值得保存起来供将来的项目使用。

**第 3 部分:使用管理区创建新类别**

14.运行命令在数据库中创建新表:

```
python manage.py makemigrations
python manage.py migrate
```

15.打开`films/admin.py`，写下这几行。现在`Films`和`Genre`型号将出现在管理区:

```
from django.contrib import admin
from .models import Film, Genreadmin.site.register(Film)
admin.site.register(Genre)
```

16.接下来，登录到管理区域，选择`Genre`型号，然后创建并保存两到三个新流派。当我们使用 films CRUD 保存新电影时，我们将需要这些类型。

**第 4 部分:构建应用程序路线和基于类别的视图**

17.创建`films/urls.py`文件。

18.打开`project/urls.py`，导入`include`函数，并使用它创建一条新路线:

```
from django.contrib import admin
from django.urls import path, include # addedurlpatterns = [
    path('', include('films.urls')), # added
    path('admin/', admin.site.urls),
]
```

19.接下来，打开`films/urls.py`并添加以下代码。

CBV 尚未创建，因此此时运行服务器将会产生错误。

20.在`films/views.py`中，这段代码将创建我们需要的通用 CBV:

注意这段代码有多简洁:我们只创建了一个`FilmBaseView`，这样我们就不会重复`model`、`fields`和`success_url`信息行，然后我们让我们的电影视图类继承了`FilmBaseView`和 Django 已经拥有的用于 CRUD 操作的通用视图类。所有这些都不需要使用表单逻辑，也不需要编写代码来访问数据库，然后将这些信息保存在上下文字典中。Django 的 CBV 已经设计出了这些程序。这样做的好处是，我们可以随时修改它们的任何方法，并根据我们的特定需求向视图类添加新的行为。

现在我们只需要创建模板来显示和更改我们的电影类视图提供的信息。

**第 5 部分:创建模板**

21.打开`project/settings.py`，在变量`TEMPLATES`中，用以下信息填充`TEMPLATES['DIR']`键中的空列表(只改变这一行，保持另一行不变):

```
TEMPLATES = [
{(...)'DIRS': [ BASE_DIR.joinpath('templates') ],(...)}
]
```

22.确保您在根项目中(文件夹`films_project`，与`manage.py`在同一层)，然后创建这三个新文件夹。

```
mkdir templates films/templates films/templates/films
```

现在，是时候创建我们将用作电影 CRUD 模板的 HTML 文件了。然而，在我们这样做之前，一个实际的观察:事实上，我认为最好不要在这里复制模板代码，因为大部分代码都太长了。因此，我将提供一些链接，链接到保存在我的 GitHub 库中的每个模板文件的原始内容，以便您可以复制/粘贴或转录到您自己的代码中。

23.创建一个`templates/base.html`文件。这个文件不应该在`films`模板文件夹中，要小心。

*   `touch templates/base.html` ( [内容此处](https://raw.githubusercontent.com/fabricius1/DjangoFilmsCRUD/master/templates/base.html))；

24.在`films/templates/films`文件夹中，创建四个文件名如下的文件，因为这些是 Django 的通用视图使用的模板名称模式:

*   `film_confirm_delete.html`(此处[内容](https://raw.githubusercontent.com/fabricius1/DjangoFilmsCRUD/master/films/templates/films/film_confirm_delete.html))；
*   `film_detail.html` ( [内容此处](https://raw.githubusercontent.com/fabricius1/DjangoFilmsCRUD/master/films/templates/films/film_detail.html))；
*   `film_form.html`(此处[内容](https://raw.githubusercontent.com/fabricius1/DjangoFilmsCRUD/master/films/templates/films/film_form.html))；
*   `film_list.html` ( [内容此处](https://github.com/fabricius1/DjangoFilmsCRUD/blob/master/films/templates/films/film_list.html))；

您可以通过在 Linux 终端(或 Git Bash)中运行这一行代码来立即创建这些文件:

```
touch films/templates/films/film_{confirm_delete,detail,form,list}.html
```

注意这些 HTML 文件名是如何在前面加上`film_`的，这是我们的模型名，用小大写字母加上下划线。在未来的 Django 项目中，当使用 CBV 创建 CRUD 应用程序时，您应该使用这种模式。

25.运行`python manage.py check`看看有没有错误；然后用`python manage.py runserver`运行服务器。您的 CRUD films 应用程序已经完成，现在您可以使用它来列出、添加、更新和删除电影信息。

**第 6 部分:使用 PYTHON 脚本向你的数据库添加一些额外的电影**

探索您的新应用程序，并对其进行一些更改。如果你想要额外的数据，你可以把我准备的 20 多部皮克斯电影加载到 Django 中。你只需要遵循以下步骤:

26.在`manage.py`的同一层创建一个`scripts`文件夹

27.创建一个`films/pixar.csv`文件，然后[将该内容复制到其中](https://raw.githubusercontent.com/fabricius1/DjangoFilmsCRUD/master/films/pixar.csv)。

28.创建一个`scripts/load_pixar.py`文件，[将该内容复制到其中](https://raw.githubusercontent.com/fabricius1/DjangoFilmsCRUD/master/scripts/load_pixar.py)。

29.运行`python manage.py runscript load_pixar`。访问您的电影应用程序，并检查它是如何填充皮克斯电影信息的。

如果您想了解更多关于在 Django 中运行 Python 脚本的知识，我邀请您阅读我写的另一篇教程:

</use-python-scripts-to-insert-csv-data-into-django-databases-72eee7c6a433>  

**最后备注:制作你自己的 DJANGO CRUD 应用**

现在，您已经拥有了使用自己的模型构建更复杂的 Django CRUD 应用程序的所有必要工具。

例如，假设您有三个表:机构、项目和课程。一个机构可以提交多个项目，一个项目可以有一个或多个与之相关的课程。

要在这种情况下制作 Django CRUD，您只需要在一个新的 Django 项目中按照我们上面讨论的这三个实体的步骤:

*   在一个`models.py`文件中定义模型；
*   在一个`urls.py`文件中，按照我们展示的模式，为每个模型创建五条路线。因为我们有三个模型，所以总共有 15 个路由，可能还有一个主页。你可以在导航栏中设置链接，直接进入各自的*列表，所有*页面的机构、项目和课程。
*   将所有必要的代码放入`views.py`。我们为电影创建了六个基于类的视图，所以我们总共有 18 个 CBV:6 个用于机构，6 个用于项目，6 个用于课程，所有这些都在同一个文件中。
*   最后，只需为您的 CRUD 创建 12 个模板文件(每个模型 4 个)，遵循 HTML 名称模式并在模板 HTML 代码中进行必要的更改。

一旦你有了这个电影的启动代码，就可以很容易很快地为你自己的项目复制代码，并根据你的具体需要进行必要的调整。

我自己最近有一个非常好的经历，当我设法在仅仅几个小时的工作中构建了我上面向我妻子描述的几乎精确的 CRUD。而`models.py`变成了一个巨大的，例如，仅课程模型就有超过 50 列。所以，我花了大部分时间对数据建模；一旦完成，安装和运行 CRUD 应用程序就变得超级简单和快速。

在这段代码投入生产之前，我和我的妻子仍然会讨论一些调整和改进，但是这样的经历足以向我展示用 Django 构建 CRUD 应用程序是多么容易。如果你把你在这里学到的东西和一个更简洁、更精细的前端结合起来，你可能会用你新的 Django 编码技能给你的客户或潜在雇主带来惊喜。

亲爱的读者，非常感谢你花时间和精力阅读我的文章。

*编码快乐！*