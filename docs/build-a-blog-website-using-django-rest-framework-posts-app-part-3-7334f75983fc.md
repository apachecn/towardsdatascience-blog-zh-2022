# 使用 Django Rest 框架——Posts 应用程序构建一个博客网站(第 3 部分)

> 原文：<https://towardsdatascience.com/build-a-blog-website-using-django-rest-framework-posts-app-part-3-7334f75983fc>

## 在第三部分中，我们处理应用程序的整个 posts 应用程序，从而完成了应用程序的后端

![](img/164ad83e3da68c2f18af44d6eccbc43e.png)

[真诚媒体](https://unsplash.com/@sincerelymedia?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

大家好，我希望你们都过得很好，并且喜欢这个 DRF 文章系列的前两部分。在本系列的第一部分中，我们处理了设置这个项目的基础知识，你们都已经对我们将要构建的项目有了一个大致的了解，而在第二部分中，我们处理了我们的应用程序的`users`应用程序，在那里我们为我们的`users`部分编写了序列化程序和视图

如果您还没有阅读这些部分，我强烈建议您先阅读前两部分，然后再回到本文。

[](/build-a-blog-website-using-django-rest-framework-overview-part-1-1f847d53753f) [## 使用 Django Rest 框架构建博客网站——概述(第 1 部分)

### 让我们使用 Django Rest 框架构建一个简单的博客网站，了解 DRF 和 REST APIs 如何工作，以及我们如何添加…

towardsdatascience.com](/build-a-blog-website-using-django-rest-framework-overview-part-1-1f847d53753f) [](/build-a-blog-website-using-django-rest-framework-part-2-be9bc353abf3) [## 使用 Django Rest 框架——用户应用程序构建一个博客网站(第 2 部分)

### 在第二部分中，我们将处理构建用户相关的模型和视图，并将测试用户相关的 API。

towardsdatascience.com](/build-a-blog-website-using-django-rest-framework-part-2-be9bc353abf3) 

在本系列的第三部分，我们将处理 Django 应用程序的完整的`posts`应用程序，从而完成我们应用程序的完整后端。我们还将使用可浏览 API 接口测试 API，就像我们测试`users`应用程序一样。所以，让我们直接进入`posts`应用程序，并开始构建它。

因此，当我们移动到`posts`文件夹时，我们会看到我们拥有的文件与我们移动到该文件夹时为`users`应用程序所拥有的文件相同。首先，我们将进入`models.py`文件来构建我们的模型。

## models.py

先从`models.py`文件开始吧。我们将在这里定义我们的数据库模型。因此，对于`posts`部分，我们将有三种不同的模型— `Post`、`Upvote`和`Comment`。

我们首先将所需的依赖项导入到我们的文件中，如下所示:

```
from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse
```

正如我们所看到的，我们已经导入了默认的`User`模型，还从 Django 导入了`models`和`reverse`。这些将有助于我们建立模型。

因此，我们将从`Posts`型号开始。让我们先看看代码，然后我们会明白这些行的意义。

```
class Post(models.Model):
    user = models.ForeignKey(User, on_delete = models.CASCADE)
    title = models.CharField(max_length = 100)
    body = models.TextField()
    created = models.DateTimeField(auto_now_add = True)
    updated = models.DateTimeField(auto_now = True)
    upvote_count = models.IntegerField(default = 0)

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('post-detail', kwargs = {'pk': self.pk})
```

上面的代码定义了一个从 Django 的`models.Model`类继承而来的`Post`类。这使得`Post`类成为一个 Django 模型，允许它保存在数据库中并从数据库中检索。

`Post`类有几个字段，包括`user`、`title`、`body`、`created`、`updated`和`upvote_count`。每个字段都使用 Django 的`models`模块中的一个类来定义，该类指定了每个字段将存储的数据类型。例如，`title`字段是一个`CharField`，它将存储博客文章的标题，`created`字段是一个`DateTimeField`，它存储文章创建的时间。

`user`字段被定义为`ForeignKey`字段，这意味着它是对默认 Django `User`模型的引用。`on_delete`参数指定如果删除了引用的`User`对象，那么`Post`对象会发生什么。这里，`on_delete`参数被设置为`models.CASCADE`，这意味着如果删除了`User`对象，那么`Post`对象也将被删除。

`ForeignKey`字段允许将`Post`链接到特定的`User`对象，可以使用`post.user`属性访问该对象。这可用于获取创建`Post`的用户的用户名，或仅允许创建`Post`的用户编辑或删除它。

`upvote_count`字段将存储博客文章从用户那里获得的支持票数。默认值已经被设置为零，这意味着当一个新的`Post`对象被创建时，`upvote_count`被设置为零。当用户点击博文上的 upvote 按钮时，`upvote_count`将会增加。

`__str__`方法是一个特殊的方法，它定义了一个`Post`对象应该如何被表示为一个字符串。在这种情况下，该方法返回`Post`的`title`。

`get_absolute_url`方法返回`Post`对象的详细页面的 URL。Django 用这个来确定对象保存到数据库时的 URL。

接下来，我们将研究将在我们的应用程序中存储所有 upvotes 的`Upvote`模型。让我们看看这个模型，然后我会解释这个代码的意义。

```
class Upvote(models.Model):
    user = models.ForeignKey(User, related_name = 'upvotes', on_delete = models.CASCADE)
    post = models.ForeignKey(Post, related_name = 'upvotes', on_delete = models.CASCADE)
```

这段代码定义了一个继承自 Django 的`models.Model`类的`Upvote`类，就像我们之前讨论的`Post`类一样。

`Upvote`类有两个字段:`user`和`post`。两个字段都被定义为`ForeignKey`字段，这表明它们分别是对`User`和`Post`对象的引用。`related_name`参数指定了用于访问反向关系的名称。例如，如果一个`User`对象`user`投了几个`Post`对象的赞成票，那么`user.upvotes`属性将返回一个`user`投了赞成票的`Post`对象的 queryset。

两个字段的`on_delete`参数被设置为`models.CASCADE`，这意味着如果删除了`Upvote`对象引用的`User`或`Post`对象，则`Upvote`对象也将被删除。

这个类用于跟踪哪些用户对哪些帖子投了赞成票。我们将使用它来防止一个用户不止一次地向上投票一个`Post`，或者我们也可以使用它来检索已经向上投票给一个给定的`Post`的用户列表，尽管我们不会在我们的应用程序中实现后者。但是，正如我们所看到的，这很容易做到，您肯定可以尝试实现它来改进应用程序。

接下来，我们移动到最后一个模型类`Comment`，它将在我们的应用程序中存储博客帖子上的所有评论。相同的代码如下。

```
class Comment(models.Model):
    user = models.ForeignKey(User, related_name = 'comments', on_delete = models.CASCADE)
    post = models.ForeignKey(Post, related_name = 'comments', on_delete = models.CASCADE)
    body = models.TextField()
    created = models.DateTimeField(auto_now_add = True)

    def __str__(self):
        return self.body
```

代码定义了一个继承自 Django 的`models.Model`类的`Comment`类，就像`Post`和`Upvote`类一样。

`Comment`类有四个字段:`user`、`post`、`body`和`created`。`user`和`post`字段被定义为`ForeignKey`字段，这意味着它们分别是对`User`和`Post`对象的引用。两个字段的`related_name`参数指定了用于访问反向关系的名称。例如，如果一个`User`对象`user`写了几个`Comment`对象，`user.comments`属性将返回一个`user`写的`Comment`对象的 queryset。

`body`字段是一个`TextField`，它存储了评论的主要内容。

`created`字段是一个`DateTimeField`字段，存储`Comment`对象创建的日期和时间。`auto_now_add`参数设置为`True`，这意味着`Comment`对象的`created`字段将自动设置为对象首次创建时的当前日期和时间。

`__str__`方法是一个特殊的方法，它定义了一个`Comment`对象应该如何被表示为一个字符串。在这种情况下，该方法返回`Comment`的`body`，这是注释的主要内容。

`Comment`类用于确定哪些用户评论了一个特定的`Post`对象。我们将使用它来显示博客帖子上的评论以及用户的姓名。此外，这可以用于仅允许创建评论的用户编辑或删除评论，尽管我们还没有在当前应用程序中实现这一部分。

`models.py`部分的完整代码如下:

```
from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse

class Post(models.Model):
    user = models.ForeignKey(User, on_delete = models.CASCADE)
    title = models.CharField(max_length = 100)
    body = models.TextField()
    created = models.DateTimeField(auto_now_add = True)
    updated = models.DateTimeField(auto_now = True)
    upvote_count = models.IntegerField(default = 0)

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('post-detail', kwargs = {'pk': self.pk})

class Upvote(models.Model):
    user = models.ForeignKey(User, related_name = 'upvotes', on_delete = models.CASCADE)
    post = models.ForeignKey(Post, related_name = 'upvotes', on_delete = models.CASCADE)

class Comment(models.Model):
    user = models.ForeignKey(User, related_name = 'comments', on_delete = models.CASCADE)
    post = models.ForeignKey(Post, related_name = 'comments', on_delete = models.CASCADE)
    body = models.TextField()
    created = models.DateTimeField(auto_now_add = True)

    def __str__(self):
        return self.body
```

接下来，我们将移动到`serializers.py`文件。因为这个文件在默认情况下是不存在的，所以我们需要像创建`users`应用程序一样创建它。

## serializers.py

在这个文件中，我们将序列化我们的模型，使它们的格式可以在 web 上共享，并且可以被任何前端框架访问。

首先，我们将把所需的依赖项导入到我们的文件中。我们将从`rest_framework`包中导入`serializers`，并且导入我们创建的三个模型——`Post`、`Upvote`和`Comment`。

接下来，我们将编写三个序列化器——三个模型各一个。下面给出了序列化程序类的代码。让我们先看看代码，然后我们可以稍后讨论它。

```
from rest_framework import serializers
from .models import Post, Upvote, Comment

class PostSerializer(serializers.ModelSerializer):
    user = serializers.ReadOnlyField(source = 'user.username')
    class Meta:
        model = Post
        fields = ('id', 'title', 'body', 'created', 'updated', 'user', 'upvote_count')

class UpvoteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Upvote
        fields = ('id', 'user', 'post')

class CommentSerializer(serializers.ModelSerializer):
    user = serializers.ReadOnlyField(source = 'user.username')
    class Meta:
        model = Comment
        fields = ('id', 'user', 'post', 'body', 'created')
```

该结构非常类似于我们在前一篇文章中为`Users`模型编写的序列化程序。

上面写的三个类中的每一个都为特定的模型定义了一个序列化器:`Post`、`Upvote`和`Comment`。每个序列化程序类的`Meta`内部类指定了序列化程序应该使用的模型，以及模型的哪些字段应该包含在序列化数据中。

正如我们所看到的，`PostSerializer`将序列化`Post`模型的实例，并将在序列化的数据中包含`id`、`title`、`body`、`created`、`updated`、`user`和`upvote_count`字段。

同样，`UpvoteSerializer`和`CommentSerializer`会序列化自己的字段。

接下来，我们转移到最重要的部分——视图，在这里我们将定义应用程序的所有逻辑，我们将处理文件中的`get`、`post`、`put`和`delete`请求。

## views.py

在本节中，我们将定义各种`views`来处理与我们已经创建的三个模型相关的不同请求。

我们将在应用程序中添加以下功能:

1.  看到所有的博客帖子
2.  创建新的博客文章
3.  编辑该特定用户撰写的博客文章
4.  删除该特定用户撰写的博客文章
5.  查看特定用户撰写的博客文章
6.  投票支持该职位
7.  在帖子上添加评论

因此，让我们从所需的导入开始。

```
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import permissions
from .models import Post, Upvote, Comment
from .serializers import PostSerializer, UpvoteSerializer, CommentSerializer
from django.contrib.auth.models import User
```

因此，在`views.py`文件中，我们导入了各种类和函数——有些来自`rest_framework`包，有些是`serilializers`和`models`。

`APIView`是一个类，它提供了一种定义视图的方法，这些视图以 RESTful 方式处理传入的 HTTP 请求。视图是一个可调用的对象，它接受一个 HTTP 请求作为它的参数，并返回一个 HTTP 响应。

`Response`是一个提供创建 HTTP 响应对象的方法的类。HTTP 响应通常包括状态代码、标头和包含响应数据的正文。

`status`是一个将多个 HTTP 状态码定义为常量的模块。这些代码表示 HTTP 请求的成功或失败。

`permissions`是一个模块，提供了在视图中实现权限检查的类。权限是确定用户是否有权访问特定视图的规则。

代码还从`.models`模块导入了几个模型类，以及从`.serializers`模块导入了序列化器类。最后，它从`django.contrib.auth.models`模块导入`User`类，该类代表 Django web 框架的用户，即默认的 Django `User`模型类。

此外，`views.py`文件中的所有类，我们将要求用户登录，所以我们将使用`permissions`模块在所有类中强制这样做。

接下来，我们将定义`PostListAPIView`类，它将提供获取所有可用博客文章的能力，并且还将具有创建新博客文章的能力。下面是相同的代码:

```
class PostListAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        posts = Post.objects.all()
        serializer = PostSerializer(posts, many = True)
        return Response(serializer.data, status = status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        data = {
            'user': request.user.id,
            'title': request.data.get('title'),
            'body': request.data.get('body')
        }
        serializer = PostSerializer(data = data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status = status.HTTP_201_CREATED)
        return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)
```

这是`APIView`的一个子类，提供了几种处理不同 HTTP 请求方法的方法，比如`GET`、`POST`、`PUT`、`DELETE`。

`PostListAPIView`类的`permission_classes`属性指定只允许经过身份验证的用户访问这个视图。这是通过使用`isAuthenticated`方法实现的。这意味着只有已经登录的用户才能看到帖子列表或创建新帖子。我们也将在所有其他课程中使用相同的内容。

当一个`GET`请求被发送到视图时，调用`PostListAPIView`类的`get`方法。该方法检索网站上的所有博客文章，并使用`PostSerializer`类创建数据的序列化表示。然后，它在 HTTP 响应中返回该数据，状态代码为 200 OK。

当一个`POST`请求被发送到视图时，调用`PostListAPIView`类的`post`方法。这个方法使用来自请求的数据创建一个新的`Post`对象，使用`PostSerializer`类序列化数据，并在 HTTP 响应中返回序列化的数据，创建状态代码 201。如果来自请求的数据无效，它将返回一条错误消息，状态代码为 400 BAD REQUEST。

接下来，我们定义了`PostDetailAPIView`,它将处理博客中的单个帖子的操作。它将处理获取特定的文章，编辑或删除它。它还拥有我们在上一课中拥有的相同权限。下面是它的代码:

```
class PostDetailAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self, pk):
        try:
            return Post.objects.get(pk = pk)
        except Post.DoesNotExist:
            return None

    def get(self, request, pk, *args, **kwargs):
        post = self.get_object(pk)
        if post is None:
            return Response({'error': 'Post not found'}, status = status.HTTP_404_NOT_FOUND)
        serializer = PostSerializer(post)
        return Response(serializer.data, status = status.HTTP_200_OK)

    def put(self, request, pk, *args, **kwargs):
        post = self.get_object(pk)
        if post is None:
            return Response({'error': 'Post not found'}, status = status.HTTP_404_NOT_FOUND)
        data = {
            'user': request.user.id,
            'title': request.data.get('title'),
            'body': request.data.get('body'),
            'upvote_count': post.upvote_count
        }
        serializer = PostSerializer(post, data = data, partial = True)
        if serializer.is_valid():
            if post.user.id == request.user.id:
                serializer.save()
                return Response(serializer.data, status = status.HTTP_200_OK)
            return Response({"error": "You are not authorized to edit this post"}, status = status.HTTP_401_UNAUTHORIZED)
        return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, *args, **kwargs):
        post = self.get_object(pk)
        if post is None:
            return Response({'error': 'Post not found'}, status = status.HTTP_404_NOT_FOUND)
        if post.user.id == request.user.id:
            post.delete()
            return Response({"res": "Object deleted!"}, status = status.HTTP_200_OK)
        return Response({"error": "You are not authorized to delete this post"}, status = status.HTTP_401_UNAUTHORIZED)
```

`get_object`方法是一个助手方法，用指定的主键(`pk`)检索文章。如果帖子不存在，则返回`None`。

当一个`GET`请求被发送到视图时，调用`PostDetailAPIView`类的`get`方法。它使用`get_object`方法检索具有指定`pk`的帖子，使用`PostSerializer`类序列化帖子，并在 HTTP 响应中返回序列化的数据，状态代码为 200 OK。如果具有指定的`pk`的 post 不存在，它返回一个错误消息，状态代码为 404 NOT FOUND。

当一个`PUT`请求被发送到视图时，调用`PostDetailAPIView`类的`put`方法。它使用`get_object`方法用指定的`pk`检索帖子，用来自请求的数据更新帖子，并保存更新后的帖子。如果当前用户不是帖子的所有者，它将返回一条错误消息，状态代码为 401 未授权。如果来自请求的数据无效，它将返回一条错误消息，状态代码为 400 BAD REQUEST。

当一个`DELETE`请求被发送到视图时，调用`PostDetailAPIView`类的`delete`方法。它使用`get_object`方法检索带有指定`pk`的帖子，然后如果当前用户是帖子的所有者，则删除它。如果当前用户不是帖子的所有者，它将返回一条错误消息，状态代码为 401 未授权。如果指定`pk`的帖子不存在，则返回一条错误消息，状态代码为 404 未找到。

接下来，我们将拥有处理查看来自特定用户的帖子的`UserPostAPIView`类。

```
class UserPostAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, username, *args, **kwargs):
        user = User.objects.filter(username = username).first()
        if user is None:
            return Response({'error': 'User not found'}, status = status.HTTP_404_NOT_FOUND)
        posts = Post.objects.filter(user = user)
        serializer = PostSerializer(posts, many = True)
        return Response(serializer.data, status = status.HTTP_200_OK)
```

`UserPostAPIView`类的`get`方法检索具有指定`username`的用户，检索该用户创建的所有帖子，然后使用`PostSerializer`类创建数据的序列化表示。然后，它在 HTTP 响应中返回该数据，状态代码为 200 OK。如果指定的用户`username`不存在，则返回错误信息，状态代码为 404 未找到。

接下来，我们将拥有`UpvoteAPIView`类，它将处理与 upvotes 部分相关的所有功能。

```
class UpvoteAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self, pk):
        try:
            return Post.objects.get(pk = pk)
        except Post.DoesNotExist:
            return None

    def post(self, request, pk, *args, **kwargs):
        post = self.get_object(pk)
        if post is None:
            return Response({'error': 'Post not found'}, status = status.HTTP_404_NOT_FOUND)

        upvoters = post.upvotes.all().values_list('user', flat = True)
        if request.user.id in upvoters:
            post.upvote_count -= 1
            post.upvotes.filter(user = request.user).delete()
        else:
            post.upvote_count += 1
            upvote = Upvote(user = request.user, post = post)
            upvote.save()
        post.save()
        serializer = PostSerializer(post)
        return Response(serializer.data, status = status.HTTP_200_OK)
```

`get_object`方法是一个助手方法，用指定的主键(`pk`)检索文章。如果帖子不存在，则返回`None`。和我们给`PostDetailAPIView`类写的差不多。

`UpvoteAPIView`类的`post`方法使用`get_object`方法检索带有指定`pk`的帖子，然后添加或删除当前用户对该帖子的投票。然后更新帖子的`upvote_count`并保存更新后的帖子。最后，它使用`PostSerializer`类创建帖子的序列化表示，并在 HTTP 响应中返回序列化数据，状态代码为 200 OK。如果具有指定的`pk`的 post 不存在，它返回一个错误消息，状态代码为 404 NOT FOUND。

最后，我们有`CommentAPIView`,它将处理特定帖子上的评论。它将具有获取特定帖子的所有评论并发布新评论的功能。

```
class CommentAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self, pk):
        try:
            return Post.objects.get(pk = pk)
        except Post.DoesNotExist:
            return None

    def get(self, request, pk, *args, **kwargs):
        post = self.get_object(pk)
        if post is None:
            return Response({'error': 'Post not found'}, status = status.HTTP_404_NOT_FOUND)
        comments = Comment.objects.filter(post = post)
        serializer = CommentSerializer(comments, many = True)
        return Response(serializer.data, status = status.HTTP_200_OK)

    def post(self, request, pk, *args, **kwargs):
        post = self.get_object(pk)
        if post is None:
            return Response({'error': 'Post not found'}, status = status.HTTP_404_NOT_FOUND)
        data = {
            'user': request.user.id,
            'post': post.id,
            'body': request.data.get('body')
        }
        serializer = CommentSerializer(data = data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status = status.HTTP_201_CREATED)
        return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)
```

我们可以看到，它有同样的`get_object` helper 函数，它接受主键`pk`并返回与之对应的帖子。

当一个`GET`请求被发送到视图时，调用`CommentAPIView`类的`get`方法。它使用`get_object`方法检索带有指定`pk`的帖子，检索该帖子的所有评论，然后使用`CommentSerializer`类创建数据的序列化表示。然后，它在 HTTP 响应中返回该数据，状态代码为 200 OK。如果带有指定`pk`的 post 不存在，它返回一个错误消息，状态代码为 404 NOT FOUND。

当一个`POST`请求被发送到视图时，调用`CommentAPIView`类的`post`方法。它使用`get_object`方法检索具有指定`pk`的帖子，使用来自请求的数据创建一个新的`Comment`对象，使用`CommentSerializer`类序列化数据，并在 HTTP 响应中返回序列化数据，创建状态代码 201。如果来自请求的数据无效，它将返回一条错误消息，状态代码为 400 BAD REQUEST。如果具有指定的`pk`的帖子不存在，它将返回一个错误消息，状态代码为 404 NOT FOUND。

下面可以找到`views.py`文件的完整代码。

```
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import permissions
from .models import Post, Upvote, Comment
from .serializers import PostSerializer, UpvoteSerializer, CommentSerializer
from django.contrib.auth.models import User

class PostListAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        posts = Post.objects.all()
        serializer = PostSerializer(posts, many = True)
        return Response(serializer.data, status = status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        data = {
            'user': request.user.id,
            'title': request.data.get('title'),
            'body': request.data.get('body')
        }
        serializer = PostSerializer(data = data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status = status.HTTP_201_CREATED)
        return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)

class PostDetailAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self, pk):
        try:
            return Post.objects.get(pk = pk)
        except Post.DoesNotExist:
            return None

    def get(self, request, pk, *args, **kwargs):
        post = self.get_object(pk)
        if post is None:
            return Response({'error': 'Post not found'}, status = status.HTTP_404_NOT_FOUND)
        serializer = PostSerializer(post)
        return Response(serializer.data, status = status.HTTP_200_OK)

    def put(self, request, pk, *args, **kwargs):
        post = self.get_object(pk)
        if post is None:
            return Response({'error': 'Post not found'}, status = status.HTTP_404_NOT_FOUND)
        data = {
            'user': request.user.id,
            'title': request.data.get('title'),
            'body': request.data.get('body'),
            'upvote_count': post.upvote_count
        }
        serializer = PostSerializer(post, data = data, partial = True)
        if serializer.is_valid():
            if post.user.id == request.user.id:
                serializer.save()
                return Response(serializer.data, status = status.HTTP_200_OK)
            return Response({"error": "You are not authorized to edit this post"}, status = status.HTTP_401_UNAUTHORIZED)
        return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, *args, **kwargs):
        post = self.get_object(pk)
        if post is None:
            return Response({'error': 'Post not found'}, status = status.HTTP_404_NOT_FOUND)
        if post.user.id == request.user.id:
            post.delete()
            return Response({"res": "Object deleted!"}, status = status.HTTP_200_OK)
        return Response({"error": "You are not authorized to delete this post"}, status = status.HTTP_401_UNAUTHORIZED)

class UserPostAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, username, *args, **kwargs):
        user = User.objects.filter(username = username).first()
        if user is None:
            return Response({'error': 'User not found'}, status = status.HTTP_404_NOT_FOUND)
        posts = Post.objects.filter(user = user)
        serializer = PostSerializer(posts, many = True)
        return Response(serializer.data, status = status.HTTP_200_OK)

class UpvoteAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self, pk):
        try:
            return Post.objects.get(pk = pk)
        except Post.DoesNotExist:
            return None

    def post(self, request, pk, *args, **kwargs):
        post = self.get_object(pk)
        if post is None:
            return Response({'error': 'Post not found'}, status = status.HTTP_404_NOT_FOUND)

        upvoters = post.upvotes.all().values_list('user', flat = True)
        if request.user.id in upvoters:
            post.upvote_count -= 1
            post.upvotes.filter(user = request.user).delete()
        else:
            post.upvote_count += 1
            upvote = Upvote(user = request.user, post = post)
            upvote.save()
        post.save()
        serializer = PostSerializer(post)
        return Response(serializer.data, status = status.HTTP_200_OK)

class CommentAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self, pk):
        try:
            return Post.objects.get(pk = pk)
        except Post.DoesNotExist:
            return None

    def get(self, request, pk, *args, **kwargs):
        post = self.get_object(pk)
        if post is None:
            return Response({'error': 'Post not found'}, status = status.HTTP_404_NOT_FOUND)
        comments = Comment.objects.filter(post = post)
        serializer = CommentSerializer(comments, many = True)
        return Response(serializer.data, status = status.HTTP_200_OK)

    def post(self, request, pk, *args, **kwargs):
        post = self.get_object(pk)
        if post is None:
            return Response({'error': 'Post not found'}, status = status.HTTP_404_NOT_FOUND)
        data = {
            'user': request.user.id,
            'post': post.id,
            'body': request.data.get('body')
        }
        serializer = CommentSerializer(data = data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status = status.HTTP_201_CREATED)
        return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)
```

接下来，我们移动到`admin.py`文件来注册我们创建的模型，以便管理面板可以访问这些模型。

## 管理. py

```
from django.contrib import admin
from .models import Post, Upvote, Comment

admin.site.register(Post)
admin.site.register(Upvote)
admin.site.register(Comment)
```

如上图所示，我们注册了三个模型— `Post`、`Upvote`和`Comment`。

接下来，我们将在我们的`posts`文件夹中创建一个名为`urls.py`的文件，我们将为我们为`posts`应用编写的所有视图编写 URL。

## urls.py

在`urls.py`文件中，我们将为我们的`posts`应用程序定义 URL 模式，然后我们必须将它们包含在`blog`应用程序中的`urls.py`文件中。

```
from django.urls import path
from .views import PostListAPIView, PostDetailAPIView, UserPostAPIView, UpvoteAPIView, CommentAPIView

urlpatterns = [
    path('', PostListAPIView.as_view()),
    path('<int:pk>/', PostDetailAPIView.as_view()),
    path('<int:pk>/upvote/', UpvoteAPIView.as_view()),
    path('<int:pk>/comment/', CommentAPIView.as_view()),
    path('<username>/', UserPostAPIView.as_view())
]
```

这里，`urlpatterns`是博客应用的 URL 模式列表。列表中的每个元素都是一个将 URL 路径映射到视图的`path`对象。

列表中的第一个元素将根 URL ( `/`)映射到`PostListAPIView`视图，因此当用户访问根 URL 时，`PostListAPIView`视图将被调用并处理请求。列表中的其他元素将表单`/<int:pk>/`、`/<int:pk>/upvote/`、`/<int:pk>/comment/`和`/<username>/`的 URL 映射到相应的视图。这些 URL 用于访问单个帖子、投票赞成/投票反对帖子、创建帖子评论以及查看特定用户的帖子。

`<int:pk>`是一个 URL path 参数，指定 URL 路径应该在那个位置包含一个整数值(post 的主键)。当访问 URL 时，这个整数值作为参数`pk`传递给视图，视图可以用它来检索带有指定主键的文章。这允许视图处理特定帖子而不是所有帖子的请求。

类似地，`/<username>/`用于访问特定用户的帖子。当 URL 被访问时，`username`被传递给视图。

现在，在`blog`应用程序的`urls.py`文件中，我们将像对待`users`应用程序一样包含`posts`应用程序的 URL。

```
from django.contrib import admin
from django.urls import path, include
from posts import urls as posts_urls
from users import urls as users_urls

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api-auth/", include("rest_framework.urls", namespace="rest_framework")),
    path("api/posts/", include(posts_urls)),
    path("api/users/", include(users_urls)),
]
```

在测试 API 之前，我们需要执行迁移，因为我们已经创建了新的模型。

因此，我们将运行以下命令来实现这一点:

```
python manage.py makemigrations
python manage.py migrate
```

当我们在 Django 中创建新模型时，我们需要运行一个迁移，因为数据库模式需要匹配将要存储在其中的数据的结构。如果不运行迁移，数据库将没有正确的表和字段来存储新模型中的数据，并且当我们试图从模型中保存或检索数据时将会出现错误。

`python manage.py makemigrations`命令分析我们对模型所做的更改，并创建一个新的迁移文件，其中包含更新数据库模式的指令。

`python manage.py migrate`命令将迁移文件中的更改应用到数据库。这将更新数据库模式，并确保它与我们的模型同步。

我们可以像测试`users`应用程序一样测试`posts`应用程序的 API。由于这篇文章已经太大了，所以我将把测试部分留给`posts` app 让你自己做。这与您为`users`应用程序所做的非常相似，首先运行服务器，然后转到`urls.py`文件中定义的各个 URL。

那么，这就把我们带到了本系列文章后端部分的结尾。我希望你们都喜欢后端部分，并了解了 DRF 是如何工作的，现在你应该尝试自己构建它，并添加功能来进一步增强应用程序。

我将很快开始我们网站的前端工作，并将很快在前端部分添加文章。当他们准备好的时候，我将在这儿把他们连接起来。在那之前，继续学习！

这里还有一些你想读的文章系列:

[](/build-a-social-media-website-using-django-setup-the-project-part-1-6e1932c9f221) [## 使用 Django 构建一个社交媒体网站——设置项目(第 1 部分)

### 在第一部分中，我们通过设置密码来集中设置我们的项目和安装所需的组件…

towardsdatascience.com](/build-a-social-media-website-using-django-setup-the-project-part-1-6e1932c9f221) [](https://javascript.plainenglish.io/build-an-e-commerce-website-with-mern-stack-part-1-setting-up-the-project-eecd710e2696) [## 让我们建立一个 MERN 堆栈电子商务网络应用程序

### 第 1 部分:设置项目

javascript.plainenglish.io](https://javascript.plainenglish.io/build-an-e-commerce-website-with-mern-stack-part-1-setting-up-the-project-eecd710e2696)