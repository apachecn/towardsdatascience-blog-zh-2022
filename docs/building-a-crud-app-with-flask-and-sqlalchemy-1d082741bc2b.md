# 用 Flask 和 SQLAlchemy 构建 CRUD 应用程序

> 原文：<https://towardsdatascience.com/building-a-crud-app-with-flask-and-sqlalchemy-1d082741bc2b>

## 关于构建书店 CRUD 应用程序后端的深入教程

![](img/7e7683e423e11bebdd1c7076c745070b.png)

汤姆·赫曼斯在 [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

CRUD 应用程序是一个网络应用程序，允许你创建、读取、更新和删除东西。这是 web 开发中非常常见的任务，对于学习如何构建 web 应用程序非常有用。

在本教程中，你将学习如何在 [Flask](https://flask.palletsprojects.com/en/2.1.x/) 中构建一个 CRUD 应用程序，结果将是一个书店 web 应用程序的工作后端。我们将定义服务来处理 CRUD 操作；RESTful bookshop API 的获取、发布、上传和删除请求。

构建书店应用程序有助于学习，因为这是一个真实的例子，而不是一个玩具项目。

本教程基于 [Flask](https://flask.palletsprojects.com/en/2.1.x/) 和 [Flask-SQLAlchemy 扩展](https://flask-sqlalchemy.palletsprojects.com/en/2.x/)。

Flask 是一个使用 Python 构建 web 应用的微框架。这是一个非常轻量级的框架，易于学习和使用。

轻量级并不意味着 Flask 不强大。每当您想在应用程序中使用类似于 [ORM(对象关系映射)](https://en.wikipedia.org/wiki/Object%E2%80%93relational_mapping)的东西时，您可以使用 Flask 提供的扩展。在本教程中，我使用 Flask-SQLAlchemy 扩展创建了一个数据库和一个存储书籍的表。

SQLAlchemy 是一个 Python ORM(对象关系映射)库，它使得使用数据库变得容易。

目录:

*   [设计数据库](#3d4c)
*   [构建 API](#2f79)
*   [安装依赖关系](#e65d)
*   [创建数据库模型](#776c)
*   [配置你的烧瓶应用](#9838)
*   [设置获取请求](#c8fc)
*   [向 SQLAlchemy 添加数据](#9e9a)
*   [运行烧瓶应用程序](#62b2)
*   [删除一本书](#d480)
*   [添加新书](#d881)
*   [更新一本书](#fec5)
*   [结论](#d20e)

# 设计数据库

在创建数据库之前，我们需要定义数据库模式和表。模式是数据库的元数据结构，而表是我们想要存储的实际数据。

这个项目的设计很简单:我们有一个名为`books`的表来存储图书字段:isbn(图书的 ISBN 号)、标题、作者和价格。

这些字段将通过 SQLAlchemy ORM 存储在数据库中。Flask API 将使用这些字段作为 CRUD 操作的数据模型。

下面是一个 UML 图，显示了依赖于数据库模式的 API 中使用的函数:

![](img/c314556280a1ea6060f19da6e7b61825.png)

Flask 应用程序调用了依赖于 Book 表的函数。(作者在 Plantuml 上设计的)

正如我们所见，Flask API 有 5 个依赖于 Book 表的函数。您将看到这些函数如何从 SQLAlchemy 调用适当的方法。让我们首先看看我们是如何构造 API 函数的。这些函数将被 Flask API 调用，并用`@app.route`装饰器进行装饰。每个的映射如下所示:

*   `get_books()`使用 GET 请求列出所有映射到`/book/list` URL 的书籍。
*   `get_book(isbn)`获取由我们传递给函数的 URL 参数`isbn`定义的指定书籍。这个函数使用 GET 请求映射到`/book/<isbn>` URL。
*   `create_book()`向数据库添加新书。这个函数使用 POST 请求映射到`/book` URL。
*   `update_book(isbn)`使用 PUT 请求更新映射到`/book/<isbn>` URL 的指定图书。
*   `delete_book(isbn)`使用删除请求删除映射到`/book/<isbn>` URL 的指定图书。

# 构建 API

为了跟进这个项目，您可以创建一个文件并将代码转储到其中。您还可以创建多个文件，并从这些文件中导入代码来分离关注点。

我更喜欢将 API 代码组织在多个文件中。原因是它有助于你保持代码的组织性，也有助于你保持代码的整洁。

让我们创建一个名为`bookshop`的文件夹。在该文件夹中，该项目的结构如下所示:

```
**.**
**├──** app
**│  ** ├── __init__.py
**│  ** ├── models.py
**│  ** └── routes.py
**├──** bookshop.py
**├──** config.py
```

我试图在不使用[蓝图](https://flask.palletsprojects.com/en/2.1.x/blueprints/)的情况下将结构做得尽可能的小，我认为这对这个小应用来说太过了。以下是每个文件的分类:

*   `bookshop.py`是包含 Flask API 的主文件。
*   `config.py`包含 API 的配置。
*   `app/__init__.py`是包含数据库和应用程序实例的文件。
*   `app/models.py`包含数据库模式和 ORM。
*   `app/routes.py`包含将由 API 调用的 API 函数。

让我们从`app/models.py`文件开始创建数据库。

# 安装依赖项

在创建 db 模型之前，让我们安装 Flask 和 Flask-SQLAlchemy 扩展。

让我们也安装数据库引擎。在本教程中，您将使用 MySQL，但也可以随意使用任何其他数据库引擎。SQLAlchemy 支持 MySQL、SQLite、Postgres 等等。

对于 MySQL，安装 PyMySQL 库。

让我们全部安装:

```
**$** pip install flask flask-sqlalchemy PyMySQL
```

现在是时候创建数据库了。

# 创建数据库模型

让我们如下定义`app/models.py`文件中的数据库模型:

```
from . import db **class** Book(db.Model):
    __tablename__ = 'books'
    isbn = db.Column(db.Integer, primary_key=True)
    author = db.Column(db.String(100), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float) **def** to_json(self):
        **return** {
            'isbn': self.isbn,
            'author': self.author,
            'title': self.title,
            'price': self.price
        }
```

在该文件中，我们将表名定义为`books`，将字段定义为:

*   `isbn`:表格的主键。
*   `author`:图书作者，字符串字段，不可为空。它的长度限制在 100 个字符以内。
*   `title`:书名，为必填字段，长度为 100 个字符。
*   `price`:书的价格，是一个浮点字段，可以为空。

这里使用的`to_json()`函数将 Book 对象转换为 JSON 对象，该对象可以在浏览器上返回给客户机。我们将在接下来的章节中看到更好的方法。

注意，`Book`类是`db.Model`类的子类。该`db`实例在`app/__init__.py`文件中定义如下:

```
from flask_sqlalchemy import SQLAlchemydb = SQLAlchemy()
```

# 配置您的烧瓶应用程序

现在我们有了数据库模型，让我们为 SQLAlchemy 扩展配置 Flask 应用程序。

配置应用程序的最佳实践是创建并定义一个父配置类，该类将保存所有环境的通用配置。然后为您的环境创建子配置类的实例。在我们的例子中，我们将创建三个环境:开发、测试和生产。

让我们看看实际情况:

```
import os **class** Config:
    SQLALCHEMY_TRACK_MODIFICATIONS = False @staticmethod
    **def** init_app(app):
        **pass** **class** DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv("DEV_DATABASE_URL") **class** TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = os.getenv("TEST_DATABASE_URL") **class** ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL") config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}
```

因此`Config`类保存了应用程序的全局配置，它们是:

*   `SQLALCHEMY_TRACK_MODIFICATIONS`设置为 False，禁用修改跟踪系统。这是避免跟踪从 Flask-SQLAlchemy 到 SQLAlchemy 库的变化的开销的好方法。
*   `init_app()`是一个静态方法，用于初始化应用程序配置。

在这个`Config`父类之后，我们为每个环境创建了三个子类。每个环境都定义了适合该环境的配置。

最后，我们有了`config`字典，它将环境名映射到配置类。默认环境是我们将在本教程中使用的开发环境。

`DevelopmentConfig`类将`DEBUG`属性设置为 True，因为如果 API 中有任何错误，我们希望在浏览器中看到调试消息。

此外，它的`SQLALCHEMY_DATABASE_URI`属性设置了数据库 URL，这是我们为连接到数据库而定义的。

在我们的例子中，我们将数据库 URL 设置为环境变量`DEV_DATABASE_URL`，这是 MySQL 数据库的 URL。在终端上运行以下命令来定义 env 变量:

```
**$** export DEV_DATABASE_URL=mysql+pymysql://<username>:<password>@localhost:3306/flaskapp
```

其中`<username>`和`<password>`是 MySQL 数据库的凭证，`flaskapp`是数据库的名称。请随意用您自己的值替换任何值。

如果您正在使用另一个数据库引擎，您可以将`DEV_DATABASE_URL`更改为该数据库的适当 URL。例如，如果您正在使用 sqlite，您可以将其设置为`sqlite:///<path_to_db>`。

现在让我们导入配置字典并开始创建 Flask 应用程序。现在，`app/__init__.py`文件如下所示:

```
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import configdb = SQLAlchemy() **def** create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app) db.init_app(app)
    **return** app
```

`create_app()`函数用于创建基于环境的应用实例，该环境通过`config_name`参数作为变量传递给函数。

`app.config.from_object()`方法用于从`config`字典加载配置。然后，该配置用于初始化应用程序。

最后，用 app 实例初始化 SQLAlchemy 实例`db`。

让我们为 API 设置第一个端点，它们是 GET 请求。

# 设置获取请求

让我们将 GET 请求函数添加到`app/routes.py`文件中:

```
import os
from . import create_app
from .models import Book
from flask import jsonifyapp = create_app(os.getenv('FLASK_CONFIG') **or** 'default') @app.route("/book/list", methods=["GET"])
**def** get_books():
    books = Book.query.**all**()
    **return** jsonify([book.to_json() **for** book **in** books]) @app.route("/book/<int:isbn>", methods=["GET"])
**def** get_book(isbn):
    book = Book.query.get(isbn)
    **if** book **is** None:
        abort(404)
    **return** jsonify(book.to_json())
```

`create_app()`函数实例化应用程序实例，然后使用`app.route()`装饰器注册端点。在`get_books()`方法中，我们使用`Book.query.all()`查询所有书籍的数据库，然后使用`jsonify()`函数返回所有书籍的 JSON 表示；它是一个帮助器函数，将 Python 对象序列化为 JSON。

现在，让我们在运行应用程序之前通过 SQLAlchemy 添加一些数据。

# 向 SQLAlchemy 添加数据

向数据库添加数据的一种方法是打开一个 Flask 交互式 shell，然后创建一个新的`Book`模型实例。

让我们通过运行:

```
$ flask shell
```

此命令为您打开一个交互式会话来执行 Python 命令。这对调试和测试您的代码很有帮助。

我们现在在壳里面。让我们导入`db`实例和`Book`模型:

```
>>> from app import db
>>> db
<SQLAlchemy engine=mysql+pymysql://root:***@localhost:3306/flaskapp?charset=utf8>
>>> from app.models import Book
>>> Book
<**class** 'app.models.Book'>
```

假设已经在您的机器上创建了`flaskapp`数据库，让我们在数据库中创建`Book`表，然后定义一个新的 book:

```
>>> db.create_all()
>>> book = Book(author="Ezz", title="Cleaner Python", price=0.0)
>>> book
<app.models.Book **object** at 0x7f404a052e50>
>>> db.session.add(book)
>>> db.session.commit()
```

所以现在已经用`db.create_all()`创建了`Book`表，它创建了所有属于`db.Model`子类的表。

使用`db.session.add()`将`book`变量添加到数据库中。注意，将 book 对象添加到数据库中并不意味着您可以查询它。它还没有*提交*到数据库。这就是为什么我们需要运行`db.session.commit()`来提交我们对数据库所做的更改。

让我们再创作一本书:

```
>>> book2 = Book(author="Ahmed", title="Python", price=10.99)
>>> db.session.add(book2)
>>> db.session.commit()
```

所以现在我们书店里有两本书。这将足以演示如何列出和使用 API。

让我们用`Ctrl+C`(或`CMD+C`)关闭外壳，返回我们的终端运行 app。

# 运行烧瓶应用程序

您可以通过在终端上运行以下命令来运行该应用程序:

```
**$** export FLASK_APP=bookshop.py
**$** flask run
```

第一个命令定义了`FLASK_APP`环境变量指向`bookshop.py`文件。但是当你做`flask run`的时候，你会发现一个预期的错误。那个`bookshop.py`文件是空的。让我们解决这个问题，从`app/routes.py`文件导入`app`变量:

```
from app.routes import app
```

现在，您可以运行应用程序并在`[http://localhost:5000/book/list](http://localhost:5000/book/list.)` [公开 API 端点。](http://localhost:5000/book/list.)

在这种情况下，您会发现以下 JSON 响应:

```
[
  {
    "author": "Ezz", 
    "isbn": 1, 
    "price": 0.0, 
    "title": "Cleaner Python"
  }, 
  {
    "author": "Ahmed", 
    "isbn": 2, 
    "price": 10.99, 
    "title": "Python"
  }
]
```

当您调用这个端点`http://localhost:5000/book/1`时，您将得到第一本书:

```
{
  "author": "Ezz", 
  "isbn": 1, 
  "price": 0.0, 
  "title": "Cleaner Python"
}
```

将`1`替换为图书的 ISBN，您将得到与您查询的图书相关的响应。

# 删除图书

删除请求类似于我们对 GET 请求所做的。

让我们打开`app/routes.py`文件并添加以下代码片段:

```
from . import db
...@app.route("/book/<int:isbn>", methods=["DELETE"])
**def** delete_book(isbn):
    book = Book.query.get(isbn)
    **if** book **is** None:
        abort(404)
    db.session.delete(book)
    db.session.commit()
    **return** jsonify({'result': True})
```

这里，我们使用`db.session.delete(book)`从数据库中删除这本书，然后使用`db.session.commit()`提交修改。

你可能想知道如何测试，特别是因为`DELETE`路线与`GET`路线相同。要删除一本书，您需要使用`curl`并选择`DELETE`方法，因为默认情况下它会将请求视为`GET`请求。

例如，如果要删除第二本书，可以使用以下命令:

```
**$** curl http://localhost:5000/book/2 -X DELETE
**{**
  "result"**:** true
**}**
```

它成功地返回了一个 JSON 响应，其中的`result`键被设置为`True`。

您可以再次调用 GET 端点，通过查看 URL: `http://localhost:5000/book/list`或使用`curl`命令来检查第二本书是否已经完成:

```
**$** curl http://localhost:5000/book/list
**[**
  {
    "author": "Ezz", 
    "isbn": **1,** 
    "price"**:** 0.0, 
    "title"**:** "Cleaner Python"
  }
**]**
```

它只给出了一本书的列表。第一个。

# 添加新书

我们还可以通过使用`POST`方法调用一个函数来向数据库添加一本新书。

```
from flask import request
...@app.route('/book', methods=['POST'])
**def** create_book():
    **if** **not** request.json:
        abort(400)
    book = Book(
        title=request.json.get('title'),
        author=request.json.get('author'),
        price=request.json.get('price')
    )
    db.session.add(book)
    db.session.commit()
    **return** jsonify(book.to_json()), 201
```

为了测试添加一本新书，让我们使用`curl`程序:

```
**$** curl -H "Content-Type: application/json" -X POST -d '{"title": "Learning", "author": "Ibrahim", "price": "3.44"}' http://localhost:5000/book
**{**
  "author"**:** "Ibrahim", 
  "isbn"**:** 3, 
  "price"**:** 3.44, 
  "title"**:** "Learning"
**}**
```

当然，我们也可以使用`flask shell`添加新书。这种方法的问题是您必须导入`db`和`Book`实例。为了避免这种情况，现在的`bookshop.py`文件应该如下所示(在添加了一个代码片段之后):

```
from app import db
from app.routes import app
from app.models import Book @app.shell_context_processor
**def** make_shell_context():
    **return** **dict**(db=db, Book=Book)
```

`@app.shell_context_processor`装饰器用于注册一个函数，该函数将被调用来向 shell 会话注入变量。

`make_shell_context()`函数返回一个字典，其中包含我们需要添加到 shell 会话中的`db`和`Book`实例，这样我们就可以在 shell 中使用它们，而不必导入它们。

# 更新图书

更新一本书类似于增加一本新书，除了我们使用`PUT`方法而不是`POST`。

让我们将以下内容添加到`app/routes.py`文件中:

```
@app.route('/book/<int:isbn>', methods=['PUT'])
**def** update_book(isbn):
    **if** **not** request.json:
        abort(400)
    book = Book.query.get(isbn)
    **if** book **is** None:
        abort(404)
    book.title = request.json.get('title', book.title)
    book.author = request.json.get('author', book.author)
    book.price = request.json.get('price', book.price)
    db.session.commit()
    **return** jsonify(book.to_json())
```

为了测试图书更新，让我们再次使用`curl`:

```
**$** curl http://localhost:5000/book/3 -X PUT -H "Content-Type: application/json" -d '{"author": "Ahmed", "title": "Python for Beginners", "price": 12.99}'
**{**
  "author"**:** "Ahmed", 
  "isbn"**:** 3, 
  "price"**:** 12.99, 
  "title"**:** "Python for Beginners"
**}**
```

# 结论

本教程介绍了如何通过真实应用程序使用 Flask 创建 RESTful API 的基础知识。

我们已经看到了如何使用 SQLAlchemy 连接到数据库，如何创建模型，如何将模型映射到表，如何创建路由，如何使用`curl`程序调用和测试 API，以及如何使用`flask shell`调试应用程序。

我们还讲述了如何从 CRUD bookshop 应用程序中读取、创建、更新和删除一本书。

你可以从[这个链接](https://github.com/EzzEddin/tutorials/tree/master/backend_crud_flask_sqlalchemy)获得本教程的完整代码。

**想用 Python 写出干净的代码？** [**下载免费清理器 Python 电子书**](https://ezzeddin.gumroad.com/l/cleaner-python) **。它提供了 5 种方法来编写高效的 Python 代码。**